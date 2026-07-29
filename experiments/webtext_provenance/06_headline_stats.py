#!/usr/bin/env python3
"""Aggregate the numbers RESULTS.md quotes, so every figure in the write-up
is regenerable by running one file rather than trusted from prose.

Output: output/headline_stats.json
"""

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def main():
    summ = json.load(open(OUT / "scan_summary.json"))
    clsum = json.load(open(OUT / "classification_summary.json"))
    cl = json.load(open(OUT / "classified_hits.json"))
    cen = json.load(open(OUT / "domain_census.json"))
    dup = json.load(open(OUT / "dup_clusters.json"))
    spec = json.load(open(HERE / "indicators.json"))

    def docs_where(pred):
        out = set()
        for c in cl:
            if pred(c):
                split, doc_id, _ = c["pair"].split(":")
                out.add((split, doc_id))
        return out

    PRODUCED = ("origin", "laundered_origin", "wire_carriage")
    produced = docs_where(lambda c: c["final_verdict"] in PRODUCED)
    laundered = docs_where(lambda c: c["final_verdict"] == "laundered_origin")
    amplified = docs_where(lambda c: c["final_verdict"] == "citation_amplification")

    fam = defaultdict(set)
    for c in cl:
        if c["final_verdict"] in PRODUCED:
            split, doc_id, _ = c["pair"].split(":")
            fam[c["actor"]].add((split, doc_id))

    tier_a = [i for i in spec["indicators"] if i["tier"] == "A"]
    tier_a_hit = sorted({c["indicator"] for c in cl if c["tier"] == "A"})
    tier_a_produced = sorted(
        c["pair"] for c in cl
        if c["tier"] == "A" and c["final_verdict"] in PRODUCED)

    total = summ["total_docs"]
    headline = {
        "generated_by": "06_headline_stats.py",
        "corpus": {
            "sample_docs": total,
            "sample_splits": {k: v["docs"] for k, v in summ["corpus"].items()},
            "sample_bpe_tokens": sum(v["bpe_tokens"] for v in summ["corpus"].values()),
            "sample_chars": sum(v["chars"] for v in summ["corpus"].values()),
        },
        "scan": {
            "indicators_total": len(spec["indicators"]),
            "indicators_with_zero_hits": len(summ["zero_hit_indicators"]),
            "zero_hit_indicator_ids": summ["zero_hit_indicators"],
            "docs_with_any_hit": summ["docs_with_any_hit"],
            "pairs_total": clsum["pairs_total"],
            "pairs_adjudicated": clsum["adjudicated_pairs"],
        },
        "tier_A": {
            "indicators_in_list": len(tier_a),
            "indicators_with_any_hit": tier_a_hit,
            "verdicts": clsum["per_tier_verdicts"].get("A", {}),
            "state_produced_pairs": tier_a_produced,
            "covert_asset_domains_with_zero_hits": sorted(
                i["id"] for i in tier_a
                if i["type"] == "domain" and i["id"] in summ["zero_hit_indicators"]),
        },
        "state_produced": {
            "docs_total": len(produced),
            "rate_per_10k_docs": round(1e4 * len(produced) / total, 2),
            "laundered_docs": len(laundered),
            "citation_amplification_docs": len(amplified),
            "per_actor": {a: len(s) for a, s in
                          sorted(fam.items(), key=lambda kv: -len(kv[1]))},
        },
        "meta_discussion": {
            "verdicts": clsum["per_tier_verdicts"].get("M", {}),
            "docs": len(docs_where(lambda c: c["tier"] == "M")),
        },
        "near_duplicate": {
            "docs_compared": dup["n_docs_compared"],
            "pairs_at_or_above_0.50": dup["n_pairs_over_cluster_threshold"],
            "pairs_at_or_above_0.25": dup["n_pairs_over_low_threshold"],
            "clusters": len(dup["clusters"]),
        },
        "census": {
            "top1000_total_links": cen["top1000_total_links"],
            "share_of_45M_links": cen["top1000_share_of_links_nominal"],
            "ranked_rows": {
                r["census_label"]: {
                    "tier": r["tier"], "rank": r["census_rank"],
                    "links": r["census_links"],
                    "sample_produced_docs": r["sample_produced_docs"],
                }
                for r in cen["rows"] if r["census_rank"] is not None
            },
        },
    }
    with open(OUT / "headline_stats.json", "w") as f:
        json.dump(headline, f, indent=1)
    print(json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()
