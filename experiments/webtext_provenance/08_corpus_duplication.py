#!/usr/bin/env python3
"""Source-agnostic duplication and templating detection across ALL 260,000
documents.

Why this exists. `04_near_duplicates.py` compared only the 578 documents that
the indicator list had already flagged — it could confirm duplication among
known actors and was structurally incapable of discovering it anywhere else.
That is a confirmatory test wearing the clothes of a discovery one. This script
inverts the logic: the corpus nominates its own anomalies, and identity is
checked afterwards.

Coordinated content leaves duplication traces regardless of who produced it and
regardless of whether anyone has publicly attributed them. So the question here
is not "is this actor's copy duplicated" but "what is duplicated at all, and
does any of it belong to anyone on the list".

Method:
  1. Rolling polynomial hash over 8-word shingles, vectorised per document.
  2. Bottom-k min-hash sketch (k=8) per document. Order-independent, so a
     document and its partial copy share sketch values with high probability.
  3. Inverted index over sketch values; candidate pairs are documents sharing
     >= 2 sketch values (cuts the O(n^2) comparison to the plausible pairs).
  4. Exact Jaccard over full shingle sets for candidates only.
  5. Single-link clustering at Jaccard >= 0.50, with >= 0.25 also recorded.
  6. Every cluster is then intersected with the indicator scan.

Output: output/corpus_duplication.json
"""

import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from scanlib import iter_corpus

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"
OUT = HERE / "output"

SHINGLE_W = 8
SKETCH_K = 8
MIN_SHARED = 2
BIG_BUCKET = 200
CLUSTER_T = 0.50
LOW_T = 0.25
BASE = np.int64(31)
WORD_MASK = np.int64((1 << 20) - 1)
POWERS = (BASE ** np.arange(SHINGLE_W, dtype=np.int64)).astype(np.int64)

# A document must have at least this many words to carry a meaningful sketch.
MIN_WORDS = SHINGLE_W + 4

# Python's hash() on str is salted per process (PYTHONHASHSEED), so using it
# here made cluster counts drift between runs — 726 one run, 722 the next.
# CRC32 is deterministic across processes and machines; the cache keeps it
# roughly as fast, since unique words are far fewer than word occurrences.
_WORD_HASH = {}


def _wh(word):
    h = _WORD_HASH.get(word)
    if h is None:
        h = zlib.crc32(word.encode("utf-8", "replace")) & int(WORD_MASK)
        _WORD_HASH[word] = h
    return h


def shingle_hashes(text):
    """Vectorised rolling polynomial hash over 8-word shingles.

    Word hashes are masked to 20 bits and BASE**7 ~ 2.7e10, so the maximum
    dot product is ~8 * 2^20 * 31^7 ~ 2.3e17, comfortably inside int64.
    """
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    wh = np.fromiter((_wh(w) for w in words), dtype=np.int64, count=len(words))
    win = np.lib.stride_tricks.sliding_window_view(wh, SHINGLE_W)
    return win @ POWERS


def main():
    t0 = time.time()

    # ---- pass 1: sketches for every document -------------------------------
    keys, sketches = [], []
    n_docs = n_short = 0
    for split, doc in iter_corpus(DATA):
        n_docs += 1
        h = shingle_hashes(doc["text"].lower())
        if h is None or h.size < SKETCH_K:
            n_short += 1
            continue
        k = min(SKETCH_K, h.size)
        sk = np.partition(h, k - 1)[:k]
        keys.append((split, doc["id"]))
        sketches.append(np.unique(sk))
    print(f"pass 1: {n_docs:,} documents, {n_short:,} too short to sketch "
          f"({time.time() - t0:.0f}s)")

    # ---- inverted index over sketch values ---------------------------------
    index = defaultdict(list)
    for i, sk in enumerate(sketches):
        for v in sk.tolist():
            index[v].append(i)

    pair_shared = defaultdict(int)
    # Buckets above BIG_BUCKET documents are excluded from pair generation:
    # each costs O(n^2) comparisons and a value shared by hundreds of documents
    # is usually site furniture. An earlier version asserted that without
    # checking, so the excluded set is now RECORDED — every document id, and a
    # text sample — and inspected in output/oversized_buckets.json. Exclusion
    # affects only which pairs are *generated*; the documents remain in the
    # sketch index and can still pair through their other sketch values.
    big_bucket_records = []
    for v, members in index.items():
        if len(members) < 2:
            continue
        if len(members) > BIG_BUCKET:
            big_bucket_records.append({"sketch_value": int(v),
                                       "n_docs": len(members),
                                       "doc_idx": members})
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair_shared[(members[a], members[b])] += 1
    big_buckets = len(big_bucket_records)
    candidates = [p for p, c in pair_shared.items() if c >= MIN_SHARED]
    print(f"index: {len(index):,} sketch values, {big_buckets:,} oversized buckets skipped, "
          f"{len(candidates):,} candidate pairs ({time.time() - t0:.0f}s)")

    # ---- pass 2: exact Jaccard on candidates, plus bucket samples ----------
    needed = sorted({i for p in candidates for i in p})
    need_keys = {keys[i] for i in needed}
    bucket_doc_idx = {i for r in big_bucket_records for i in r["doc_idx"]}
    bucket_sample_idx = {i for r in big_bucket_records for i in r["doc_idx"][:3]}
    bucket_keys = {keys[i] for i in bucket_sample_idx}
    full, bucket_text = {}, {}
    for split, doc in iter_corpus(DATA):
        key = (split, doc["id"])
        if key in need_keys:
            h = shingle_hashes(doc["text"].lower())
            if h is not None:
                full[key] = set(h.tolist())
        if key in bucket_keys:
            bucket_text[key] = doc["text"][:400]
    print(f"pass 2: rehashed {len(full):,} candidate documents "
          f"({time.time() - t0:.0f}s)")

    scored = []
    for a, b in candidates:
        sa, sb = full.get(keys[a]), full.get(keys[b])
        if not sa or not sb:
            continue
        inter = len(sa & sb)
        if not inter:
            continue
        j = inter / (len(sa) + len(sb) - inter)
        if j >= LOW_T:
            scored.append((keys[a], keys[b], j))
    scored.sort(key=lambda x: -x[2])

    # ---- clustering --------------------------------------------------------
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b, j in scored:
        if j >= CLUSTER_T:
            union(a, b)

    clusters = defaultdict(list)
    for a, b, j in scored:
        if j >= CLUSTER_T:
            clusters[find(a)].append(a)
            clusters[find(a)].append(b)

    # ---- intersect with the indicator scan ---------------------------------
    flagged = defaultdict(list)
    for line in open(DATA / "scan_hits_full.jsonl"):
        r = json.loads(line)
        flagged[(r["split"], r["doc_id"])].append(r["indicator"])

    cluster_out = []
    for root, members in clusters.items():
        uniq = sorted(set(members), key=lambda m: m[1])
        hits = sorted({i for m in uniq for i in flagged.get(m, [])})
        cluster_out.append({
            "size": len(uniq),
            "members": [f"{s}:{d}" for s, d in uniq],
            "indicator_hits": hits,
            "any_indicator": bool(hits),
        })
    cluster_out.sort(key=lambda c: -c["size"])

    n_docs_clustered = len({m for c in cluster_out for m in c["members"]})
    n_clusters_with_ind = sum(1 for c in cluster_out if c["any_indicator"])

    # The headline claim of RESULTS.md 5.1 is that no state-produced document
    # appears in any cluster. Computed here over ALL clusters rather than left
    # to be inferred from the truncated cluster_list written below.
    produced_docs = set()
    for c in json.load(open(OUT / "classified_hits.json")):
        if c["final_verdict"] in ("origin", "laundered_origin", "wire_carriage"):
            split, doc_id, _ = c["pair"].split(":")
            produced_docs.add(f"{split}:{doc_id}")
    clustered_docs = {m for c in cluster_out for m in c["members"]}
    produced_in_clusters = sorted(produced_docs & clustered_docs)

    result = {
        "generated_by": "08_corpus_duplication.py",
        "scope": "ALL documents in the sample, no indicator pre-filter",
        "documents_scanned": n_docs,
        "documents_too_short_to_sketch": n_short,
        "shingle_words": SHINGLE_W,
        "sketch_k": SKETCH_K,
        "min_shared_sketch_values": MIN_SHARED,
        "oversized_buckets_skipped": big_buckets,
        "candidate_pairs": len(candidates),
        "pairs_at_or_above_0.25": len(scored),
        "pairs_at_or_above_0.50": sum(1 for _, _, j in scored if j >= CLUSTER_T),
        "clusters": len(cluster_out),
        "documents_in_clusters": n_docs_clustered,
        "clusters_containing_an_indicator_hit": n_clusters_with_ind,
        "state_produced_documents": len(produced_docs),
        "state_produced_documents_in_clusters": produced_in_clusters,
        "n_state_produced_in_clusters": len(produced_in_clusters),
        "top_pairs": [
            {"a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}", "jaccard": round(j, 3),
             "indicators_a": flagged.get(a, []), "indicators_b": flagged.get(b, [])}
            for a, b, j in scored[:60]
        ],
        # Full membership, not a truncated view. Any test of cluster
        # membership against a null model needs every document; truncating to
        # the largest 200 clusters would silently bias it toward big clusters.
        "cluster_list": cluster_out,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(OUT / "corpus_duplication.json", "w") as f:
        json.dump(result, f, indent=1)

    # ---- the excluded set, written out so it can be judged ----------------
    bucket_out = []
    for r in sorted(big_bucket_records, key=lambda r: -r["n_docs"]):
        docs = [f"{keys[i][0]}:{keys[i][1]}" for i in r["doc_idx"]]
        hits = sorted({ind for i in r["doc_idx"]
                       for ind in flagged.get(keys[i], [])})
        bucket_out.append({
            "sketch_value": r["sketch_value"],
            "n_docs": r["n_docs"],
            "indicator_hits": hits,
            "state_produced_members": sorted(
                set(docs) & produced_docs),
            "samples": [
                {"doc": f"{keys[i][0]}:{keys[i][1]}",
                 "head": bucket_text.get(keys[i], "")}
                for i in r["doc_idx"][:3]
            ],
            "documents": docs,
        })
    with open(OUT / "oversized_buckets.json", "w") as f:
        json.dump({
            "generated_by": "08_corpus_duplication.py",
            "threshold": BIG_BUCKET,
            "n_buckets": len(bucket_out),
            "n_distinct_documents": len(bucket_doc_idx),
            "note": "Excluded from PAIR GENERATION only. These documents remain "
                    "in the corpus, in the sketch index, and in every other "
                    "script; they can still cluster via their other sketch "
                    "values. Recorded because an earlier version called them "
                    "boilerplate without inspecting them.",
            "buckets": bucket_out,
        }, f, indent=1)

    print()
    print(f"pairs >= {LOW_T}: {result['pairs_at_or_above_0.25']:,}   "
          f"pairs >= {CLUSTER_T}: {result['pairs_at_or_above_0.50']:,}")
    print(f"clusters: {result['clusters']:,}  "
          f"documents in clusters: {n_docs_clustered:,}  "
          f"clusters touching an indicator: {n_clusters_with_ind}")
    print(f"state-produced documents in ANY cluster: "
          f"{len(produced_in_clusters)} of {len(produced_docs)}")
    print(f"runtime {result['runtime_seconds']}s -> {OUT / 'corpus_duplication.json'}")


if __name__ == "__main__":
    main()
