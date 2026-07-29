#!/usr/bin/env python3
"""Scan the 260k-document WebText sample for every indicator in indicators.json.

Outputs:
  _DATA/scan_hits_full.jsonl  (uncommitted) — every hit, all tiers, with up to
                              3 context snippets per (doc, indicator)
  output/hits_nonbase.jsonl   (committed)   — tiers A/B/C/M only, 1 snippet,
                              so the evidence behind every reported number is
                              in the repository
  output/scan_summary.json    (committed)   — per-indicator and per-tier doc
                              and occurrence counts, corpus totals, provenance

A "hit" is a (document, indicator) pair; n_occurrences counts matches inside
that document. Counting is by document when rates are reported.
"""

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from scanlib import (
    DomainMatcher,
    PhraseMatcher,
    context_snippet,
    iter_corpus,
    load_indicators,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"
OUT = HERE / "output"


def main():
    t0 = time.time()
    spec = load_indicators(HERE / "indicators.json")
    indicators = spec["indicators"]
    dom = DomainMatcher(indicators)
    phr = PhraseMatcher(indicators)

    ind_meta = {i["id"]: i for i in indicators}
    doc_counts = defaultdict(int)
    occ_counts = defaultdict(int)
    corpus = {s: {"docs": 0, "chars": 0, "bpe_tokens": 0} for s in ("train", "valid", "test")}

    full_f = open(DATA / "scan_hits_full.jsonl", "w")
    slim_f = open(OUT / "hits_nonbase.jsonl", "w")

    n_hit_docs = 0
    for split, doc in iter_corpus(DATA):
        text = doc["text"]
        low = text.lower()
        corpus[split]["docs"] += 1
        corpus[split]["chars"] += len(text)
        corpus[split]["bpe_tokens"] += doc.get("length", 0)

        per_ind = defaultdict(list)  # indicator_id -> [(start, end)]
        for ind, s, e in dom.scan(low):
            per_ind[ind["id"]].append((s, e))
        for ind, s, e in phr.scan(text, low):
            per_ind[ind["id"]].append((s, e))
        if not per_ind:
            continue

        n_hit_docs += 1
        for iid, spans in per_ind.items():
            doc_counts[iid] += 1
            occ_counts[iid] += len(spans)
            tier = ind_meta[iid]["tier"]
            rec = {
                "doc_id": doc["id"],
                "split": split,
                "indicator": iid,
                "tier": tier,
                "n_occurrences": len(spans),
                "doc_chars": len(text),
                "contexts": [context_snippet(text, s, e) for s, e in spans[:3]],
            }
            full_f.write(json.dumps(rec) + "\n")
            if tier != "BASE":
                slim = dict(rec)
                slim["contexts"] = rec["contexts"][:1]
                slim_f.write(json.dumps(slim) + "\n")

    full_f.close()
    slim_f.close()

    with open(HERE / "indicators.json", "rb") as f:
        ind_sha = hashlib.sha256(f.read()).hexdigest()

    tier_docs = defaultdict(set)
    # doc-level tier rollups need doc ids; cheap second pass over the full hits
    with open(DATA / "scan_hits_full.jsonl") as f:
        for line in f:
            r = json.loads(line)
            tier_docs[r["tier"]].add((r["split"], r["doc_id"]))

    summary = {
        "generated_by": "02_scan_corpus.py",
        "indicators_sha256": ind_sha,
        "runtime_seconds": round(time.time() - t0, 1),
        "corpus": corpus,
        "total_docs": sum(c["docs"] for c in corpus.values()),
        "docs_with_any_hit": n_hit_docs,
        "per_tier_docs": {t: len(s) for t, s in sorted(tier_docs.items())},
        "per_indicator": {
            iid: {
                "tier": ind_meta[iid]["tier"],
                "actor": ind_meta[iid]["actor"],
                "pattern": ind_meta[iid]["pattern"],
                "docs": doc_counts[iid],
                "occurrences": occ_counts[iid],
            }
            for iid in sorted(doc_counts, key=lambda k: (-doc_counts[k], k))
        },
        "zero_hit_indicators": sorted(
            i["id"] for i in indicators if i["id"] not in doc_counts
        ),
    }
    with open(OUT / "scan_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("total_docs", "docs_with_any_hit", "per_tier_docs", "runtime_seconds")}, indent=1))
    print(f"hits -> {OUT / 'hits_nonbase.jsonl'}")


if __name__ == "__main__":
    main()
