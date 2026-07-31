"""The near-duplicate clustering pipeline, extracted so it can be run over
arbitrary document streams.

08_corpus_duplication.py runs it over the corpus. 11_injection_validation.py
runs it over the corpus plus synthetic coordinated documents, which is the only
way to learn what the detector's recall actually is. Sharing one implementation
means the validation characterises the detector that produced the findings,
rather than a lookalike written beside it.

`cluster_documents` reproduces 08's numbers exactly when given the plain
corpus; 11 asserts that before reporting anything.
"""

import zlib
from collections import defaultdict

import numpy as np

SHINGLE_W = 8
SKETCH_K = 8
MIN_SHARED = 2
CLUSTER_T = 0.50
LOW_T = 0.25
BIG_BUCKET = 200
MIN_WORDS = SHINGLE_W + 4

_BASE = np.int64(31)
WORD_MASK = np.int64((1 << 20) - 1)
_POWERS = (_BASE ** np.arange(SHINGLE_W, dtype=np.int64)).astype(np.int64)

# CRC32 rather than hash(): the latter is salted per process, which made
# cluster counts drift between runs.
_WORD_HASH = {}


def word_hash(word):
    h = _WORD_HASH.get(word)
    if h is None:
        h = zlib.crc32(word.encode("utf-8", "replace")) & int(WORD_MASK)
        _WORD_HASH[word] = h
    return h


def shingle_hashes(text, shingle_w=SHINGLE_W):
    words = text.split()
    if len(words) < shingle_w + 4:
        return None
    wh = np.fromiter((word_hash(w) for w in words), dtype=np.int64,
                     count=len(words))
    win = np.lib.stride_tricks.sliding_window_view(wh, shingle_w)
    powers = _POWERS if shingle_w == SHINGLE_W else (
        _BASE ** np.arange(shingle_w, dtype=np.int64)).astype(np.int64)
    return win @ powers


def cluster_documents(doc_iter, sketch_k=SKETCH_K, min_shared=MIN_SHARED,
                      cluster_t=CLUSTER_T, low_t=LOW_T,
                      big_bucket=BIG_BUCKET, shingle_w=SHINGLE_W):
    """Cluster documents by near-duplicate text.

    doc_iter yields (key, text). Returns a dict with the cluster list, the
    scored pairs, and the counts 08 reports.

    Two passes over doc_iter are needed, so it must be re-iterable: pass a
    callable returning a fresh iterator.
    """
    keys, sketches = [], []
    n_docs = n_short = 0
    for key, text in doc_iter():
        n_docs += 1
        h = shingle_hashes(text.lower(), shingle_w)
        if h is None or h.size < sketch_k:
            n_short += 1
            continue
        sk = np.partition(h, sketch_k - 1)[:sketch_k]
        keys.append(key)
        sketches.append(np.unique(sk))

    index = defaultdict(list)
    for i, sk in enumerate(sketches):
        for v in sk.tolist():
            index[v].append(i)

    pair_shared = defaultdict(int)
    big = []
    for v, members in index.items():
        if len(members) < 2:
            continue
        if len(members) > big_bucket:
            big.append({"sketch_value": int(v), "n_docs": len(members),
                        "doc_idx": members})
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair_shared[(members[a], members[b])] += 1
    candidates = [p for p, c in pair_shared.items() if c >= min_shared]

    need = {i for p in candidates for i in p}
    need_keys = {keys[i] for i in need}
    full = {}
    for key, text in doc_iter():
        if key in need_keys:
            h = shingle_hashes(text.lower(), shingle_w)
            if h is not None:
                full[key] = set(h.tolist())

    scored = []
    for a, b in candidates:
        sa, sb = full.get(keys[a]), full.get(keys[b])
        if not sa or not sb:
            continue
        inter = len(sa & sb)
        if not inter:
            continue
        j = inter / (len(sa) + len(sb) - inter)
        if j >= low_t:
            scored.append((keys[a], keys[b], j))
    scored.sort(key=lambda x: -x[2])

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, j in scored:
        if j >= cluster_t:
            parent[find(a)] = find(b)

    groups = defaultdict(set)
    for a, b, j in scored:
        if j >= cluster_t:
            groups[find(a)].add(a)
            groups[find(a)].add(b)
    clusters = [sorted(v) for v in groups.values()]
    clusters.sort(key=lambda c: -len(c))

    return {
        "n_docs": n_docs,
        "n_short": n_short,
        "n_sketch_values": len(index),
        "oversized_buckets": big,
        "n_candidate_pairs": len(candidates),
        "pairs": scored,
        "n_pairs_ge_low": len(scored),
        "n_pairs_ge_cluster": sum(1 for _, _, j in scored if j >= cluster_t),
        "clusters": clusters,
        "n_clusters": len(clusters),
        "n_docs_clustered": sum(len(c) for c in clusters),
    }
