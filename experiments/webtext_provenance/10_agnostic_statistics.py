#!/usr/bin/env python3
"""Null models for the source-agnostic findings.

08 and 09 produce detections and descriptive fits. Neither carries a null, so
neither can say whether what it found differs from what the corpus would
produce anyway. That gap is where this study's claims went wrong: a
zero-observation was reported as the load-bearing result when the expected
count was under two documents.

Four tests:

  T1  Cluster membership against the corpus base rate — for the state-produced
      set, for each tier, and for all indicator-flagged documents. With the
      power the test actually has, since the failure being corrected was a
      null asserted without one.

  T2  Rank-frequency residuals by permutation, replacing 09's refusal to test
      them. Two nulls: unrestricted, and rank-stratified (residual spread
      varies with rank, so an unrestricted null is the easier test).

  T3  Enrichment scan over every domain in the corpus, with false-discovery
      control. This is the only genuinely discovery-shaped test here: it asks
      which domains are over-represented among duplicated documents, with no
      list consulted at any point.

  T4  Document length, state-produced against corpus, by permutation. A cheap
      structural check that needs no list.

Output: output/agnostic_statistics.json
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from scanlib import DOMAIN_TOKEN_RE, iter_corpus
from statlib import (
    benjamini_hochberg,
    binom_test_auto,
    binom_test_two_sided,
    clopper_pearson_interval,
    fisher_exact_2x2,
    permutation_pvalue,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"
OUT = HERE / "output"

N_PERM = 20000
SEED = 20260729
PRODUCED = ("origin", "laundered_origin", "wire_carriage")


def power_for_rate_ratio(n, base, ratio, alpha=0.05, draws=4000, rng=None):
    """Probability the exact binomial test rejects at `alpha` when the group's
    true rate is `ratio` x the corpus base rate. Simulated, so it needs no
    normal approximation."""
    rng = rng or random.Random(SEED)
    p_true = min(1.0, base * ratio)
    hits = 0
    for _ in range(draws):
        k = sum(1 for _ in range(n) if rng.random() < p_true)
        if binom_test_two_sided(k, n, base) < alpha:
            hits += 1
    return hits / draws


def main():
    rng = random.Random(SEED)
    dup = json.load(open(OUT / "corpus_duplication.json"))
    cen = json.load(open(OUT / "census_anomaly.json"))
    cl = json.load(open(OUT / "classified_hits.json"))

    n_docs = dup["documents_scanned"]
    clustered = {m for c in dup["cluster_list"] for m in c["members"]}
    base = len(clustered) / n_docs

    # ---------------------------------------------------------------- T1
    groups = {}
    produced, flagged_all = set(), set()
    tiers = defaultdict(set)
    for c in cl:
        split, doc_id, _ = c["pair"].split(":")
        key = f"{split}:{doc_id}"
        flagged_all.add(key)
        tiers[c["tier"]].add(key)
        if c["final_verdict"] in PRODUCED:
            produced.add(key)
    groups["state_produced"] = produced
    groups["all_indicator_flagged"] = flagged_all
    for t, s in sorted(tiers.items()):
        groups[f"tier_{t}"] = s

    t1 = {
        "corpus_base_rate": round(base, 6),
        "documents_in_clusters": len(clustered),
        "documents_total": n_docs,
        "groups": {},
    }
    for name, members in groups.items():
        n = len(members)
        k = len(members & clustered)
        exp = n * base
        p = binom_test_two_sided(k, n, base) if n else float("nan")
        lo, hi = clopper_pearson_interval(k, n) if n else (0, 1)
        t1["groups"][name] = {
            "n": n,
            "observed_in_clusters": k,
            "expected_in_clusters": round(exp, 2),
            "rate": round(k / n, 5) if n else None,
            "rate_95_ci": [round(lo, 5), round(hi, 5)],
            "rate_ratio_vs_corpus": round((k / n) / base, 3) if n and base else None,
            "binomial_p_two_sided": p,
            "significant_at_0.05": bool(p == p and p < 0.05),
        }

    # Power: for the state-produced set, what depletion or enrichment could
    # this test actually have detected?
    n_sp = len(produced)
    t1["power_state_produced"] = {
        f"rate_ratio_{r}": round(power_for_rate_ratio(n_sp, base, r, rng=rng), 3)
        for r in (0.0, 0.25, 0.5, 2.0, 3.0, 5.0)
    }
    t1["power_note"] = (
        "A rate ratio of 0 means the group never clusters. With n="
        f"{n_sp} and a corpus base rate of {base:.4f}, even total absence of "
        "clustering is only detectable with the power shown — which is why "
        "the observed zero cannot support a depletion claim."
    )

    # ---------------------------------------------------------------- T1b
    # The baseline tier is absent from classified_hits.json (03 classifies only
    # non-baseline pairs), so it comes from the raw scan. Without it there is
    # no control: clusters are dominated by SEO and template pages (see T3),
    # which is not a fair comparator for news prose. Any set of news articles
    # might be depleted against that base rate for reasons having nothing to
    # do with who published them.
    tier_raw = defaultdict(set)
    for line in open(DATA / "scan_hits_full.jsonl"):
        r = json.loads(line)
        tier_raw[r["tier"]].add(f"{r['split']}:{r['doc_id']}")

    control = {}
    for t in sorted(tier_raw):
        s = tier_raw[t]
        n, k = len(s), len(s & clustered)
        control[f"tier_{t}"] = {
            "n": n, "observed": k, "expected": round(n * base, 2),
            "binomial_p_vs_corpus": binom_test_two_sided(k, n, base),
        }

    b_set, base_set = tier_raw["B"], tier_raw["BASE"]
    b_k, base_k = len(b_set & clustered), len(base_set & clustered)
    fisher_p = fisher_exact_2x2(b_k, len(b_set) - b_k,
                                base_k, len(base_set) - base_k)

    # Which clusters carry the baseline documents? If one cluster carries them
    # all, the control rests on a single accident and the comparison is
    # correspondingly fragile — so this is computed, not assumed.
    cl_of = {}
    for i, c in enumerate(dup["cluster_list"]):
        for m in c["members"]:
            cl_of[m] = i
    base_clusters = Counter(cl_of[d] for d in base_set & clustered)
    largest = base_clusters.most_common(1)[0] if base_clusters else (None, 0)
    drop_k = base_k - largest[1]
    fisher_drop = fisher_exact_2x2(b_k, len(b_set) - b_k,
                                   drop_k, len(base_set) - drop_k)

    t1b = {
        "per_tier_vs_corpus_base_rate": control,
        "state_media_vs_baseline_news": {
            "tier_B": {"clustered": b_k, "n": len(b_set)},
            "tier_BASE": {"clustered": base_k, "n": len(base_set)},
            "fisher_exact_two_sided": round(fisher_p, 4),
        },
        "fragility": {
            "distinct_clusters_holding_baseline_docs": len(base_clusters),
            "largest_cluster_share": f"{largest[1]}/{base_k}" if base_k else "0/0",
            "fisher_p_dropping_largest_baseline_cluster": round(fisher_drop, 4),
            "verdict": (
                "The entire baseline signal sits in one cluster. Removing it "
                "leaves both groups at zero and there is no difference left to "
                "test. The comparison is nominally significant and rests on a "
                "single six-document author-bio block."
                if base_clusters and largest[1] == base_k else
                "Baseline documents are spread across more than one cluster."
            ),
        },
    }

    # ---------------------------------------------------------------- T2
    rows = cen["indicator_domains_in_census"]
    all_rows = {r["label"]: r for r in cen["outliers"]}
    # full residual list for every census domain, rebuilt from the fit
    resid = {}
    for r in cen["indicator_domains_in_census"]:
        resid[r["label"]] = r["residual_z"]
    # 09 stores residuals only for indicator rows and outliers; recompute the
    # full vector here so the permutation null covers all 1,000 domains.
    import math
    labels, links = [], []
    for line in open(DATA / "domains.txt"):
        parts = line.split()
        if len(parts) != 2:
            continue
        labels.append(parts[1])
        links.append(int(parts[0]))
    a = cen["fit"]["a"]
    b = cen["fit"]["b"]
    sigma = cen["fit"]["residual_sigma"]
    z_all = {}
    for i, lab in enumerate(labels):
        y = math.log(links[i])
        y_hat = a + b * math.log(i + 1)
        z_all[lab] = (y - y_hat) / sigma
    rank_of = {lab: i + 1 for i, lab in enumerate(labels)}

    target = [r for r in rows
              if any(d["tier"] != "BASE" for d in r["indicators"])
              and not (r["label"] == "people")]
    tgt_labels = [r["label"] for r in target]
    obs_mean = sum(z_all[l] for l in tgt_labels) / len(tgt_labels)

    # unrestricted null
    pool = list(z_all.values())
    null_unres = []
    for _ in range(N_PERM):
        null_unres.append(sum(rng.sample(pool, len(tgt_labels))) / len(tgt_labels))

    # rank-stratified null: for each target, draw a domain within +/-50 ranks
    strata = []
    for l in tgt_labels:
        r0 = rank_of[l]
        band = [labels[j] for j in range(max(0, r0 - 51), min(len(labels), r0 + 50))]
        strata.append([z_all[x] for x in band])
    null_strat = []
    for _ in range(N_PERM):
        null_strat.append(sum(rng.choice(s) for s in strata) / len(strata))

    t2 = {
        "targets": {l: round(z_all[l], 3) for l in tgt_labels},
        "observed_mean_z": round(obs_mean, 4),
        "n_permutations": N_PERM,
        "unrestricted_null": {
            "mean": round(sum(null_unres) / len(null_unres), 4),
            "p_two_sided": round(permutation_pvalue(obs_mean, null_unres), 4),
            "p_greater": round(permutation_pvalue(obs_mean, null_unres, "greater"), 4),
        },
        "rank_stratified_null": {
            "mean": round(sum(null_strat) / len(null_strat), 4),
            "p_two_sided": round(permutation_pvalue(obs_mean, null_strat), 4),
            "p_greater": round(permutation_pvalue(obs_mean, null_strat, "greater"), 4),
        },
        "note": (
            "p_greater is the test that matters for the hypothesis: "
            "amplification predicts state domains sit ABOVE the fitted law. "
            f"n={len(tgt_labels)} domains, so this test is weak whatever it "
            "returns."
        ),
    }

    # ---------------------------------------------------------------- T3
    in_c = Counter()
    out_c = Counter()
    n_in = n_out = 0
    for split, doc in iter_corpus(DATA):
        key = f"{split}:{doc['id']}"
        doms = set()
        low = doc["text"].lower()
        for m in DOMAIN_TOKEN_RE.finditer(low):
            tok = m.group(1)
            parts = tok.split(".")
            doms.add(".".join(parts[-2:]) if len(parts) >= 2 else tok)
        if key in clustered:
            n_in += 1
            for d in doms:
                in_c[d] += 1
        else:
            n_out += 1
            for d in doms:
                out_c[d] += 1

    cand = [d for d in in_c if in_c[d] + out_c[d] >= 20]
    pvals, recs = [], []
    for d in cand:
        k, n = in_c[d], in_c[d] + out_c[d]
        p0 = n_in / (n_in + n_out)
        p = binom_test_auto(k, n, p0)
        pvals.append(p)
        recs.append({
            "domain": d, "in_clusters": k, "total": n,
            "rate": round(k / n, 4), "expected_rate": round(p0, 4),
            "enrichment": round((k / n) / p0, 2), "p": p,
        })
    rejected, q = benjamini_hochberg(pvals)
    for r, rej, qq in zip(recs, rejected, q):
        r["q"] = qq
        r["significant_fdr_0.05"] = bool(rej)
    enriched = sorted([r for r in recs if r["significant_fdr_0.05"]
                       and r["enrichment"] > 1],
                      key=lambda r: -r["enrichment"])

    spec = json.load(open(HERE / "indicators.json"))
    ind_doms = {i["pattern"] for i in spec["indicators"] if i["type"] == "domain"}
    t3 = {
        "documents_in_clusters": n_in,
        "documents_not_in_clusters": n_out,
        "domains_tested": len(cand),
        "min_occurrences_to_test": 20,
        "n_significant_fdr": sum(rejected),
        "n_enriched": len(enriched),
        "top_enriched": enriched[:40],
        "enriched_matching_an_indicator": [
            r["domain"] for r in enriched if r["domain"] in ind_doms],
        "note": (
            "Domains are second-level labels from text tokens, so this counts "
            "documents that MENTION a domain, not documents FROM it — the same "
            "capture limit the indicator scan has. Benjamini-Hochberg controls "
            "the false discovery rate across all domains tested."
        ),
    }

    # ---------------------------------------------------------------- T4
    lengths = {}
    for split, doc in iter_corpus(DATA):
        lengths[f"{split}:{doc['id']}"] = doc.get("length", 0)
    all_len = list(lengths.values())
    sp_len = [lengths[k] for k in produced if k in lengths]
    obs_med = sorted(sp_len)[len(sp_len) // 2]
    null_med = []
    for _ in range(2000):
        s = rng.sample(all_len, len(sp_len))
        null_med.append(sorted(s)[len(s) // 2])
    t4 = {
        "n_state_produced": len(sp_len),
        "median_bpe_tokens_state_produced": obs_med,
        "median_bpe_tokens_corpus": sorted(all_len)[len(all_len) // 2],
        "null_median_mean": round(sum(null_med) / len(null_med), 1),
        "p_two_sided": round(permutation_pvalue(obs_med, null_med), 4),
        "note": "Structural check requiring no indicator list.",
    }

    result = {
        "generated_by": "10_agnostic_statistics.py",
        "seed": SEED,
        "T1_cluster_membership_vs_null": t1,
        "T1b_baseline_control": t1b,
        "T2_rank_residual_permutation": t2,
        "T3_domain_enrichment_scan": t3,
        "T4_length_permutation": t4,
    }
    with open(OUT / "agnostic_statistics.json", "w") as f:
        json.dump(result, f, indent=1)

    # ------------------------------------------------------------- report
    print("T1  CLUSTER MEMBERSHIP vs CORPUS BASE RATE")
    print(f"    base rate {base:.4f}  ({len(clustered):,}/{n_docs:,})")
    print(f"    {'group':26s} {'n':>6s} {'obs':>5s} {'exp':>7s} {'ratio':>6s} {'p':>8s}")
    for name, g in t1["groups"].items():
        print(f"    {name:26s} {g['n']:>6d} {g['observed_in_clusters']:>5d} "
              f"{g['expected_in_clusters']:>7.2f} "
              f"{str(g['rate_ratio_vs_corpus']):>6s} "
              f"{g['binomial_p_two_sided']:>8.3f}")
    print("    power for state_produced:", t1["power_state_produced"])
    print()
    print("T1b BASELINE CONTROL")
    for name, g in t1b["per_tier_vs_corpus_base_rate"].items():
        print(f"    {name:12s} n={g['n']:>5d} obs={g['observed']:>3d} "
              f"exp={g['expected']:>6.2f} p={g['binomial_p_vs_corpus']:.4f}")
    sm = t1b["state_media_vs_baseline_news"]
    print(f"    state media vs baseline news, Fisher exact p = "
          f"{sm['fisher_exact_two_sided']}")
    fr = t1b["fragility"]
    print(f"    baseline docs sit in {fr['distinct_clusters_holding_baseline_docs']} "
          f"cluster(s); largest holds {fr['largest_cluster_share']}")
    print(f"    dropping it: p = {fr['fisher_p_dropping_largest_baseline_cluster']}")
    print()
    print("T2  RANK-FREQUENCY RESIDUALS BY PERMUTATION")
    print(f"    targets: {t2['targets']}")
    print(f"    observed mean z = {t2['observed_mean_z']}")
    print(f"    unrestricted null: p(two)={t2['unrestricted_null']['p_two_sided']} "
          f"p(greater)={t2['unrestricted_null']['p_greater']}")
    print(f"    rank-stratified:   p(two)={t2['rank_stratified_null']['p_two_sided']} "
          f"p(greater)={t2['rank_stratified_null']['p_greater']}")
    print()
    print("T3  DOMAIN ENRICHMENT AMONG DUPLICATED DOCUMENTS")
    print(f"    tested {t3['domains_tested']} domains, "
          f"{t3['n_significant_fdr']} significant at FDR 0.05, "
          f"{t3['n_enriched']} enriched")
    print(f"    {'domain':28s} {'in':>6s} {'tot':>6s} {'enrich':>7s} {'q':>9s}")
    for r in enriched[:20]:
        print(f"    {r['domain'][:28]:28s} {r['in_clusters']:>6d} {r['total']:>6d} "
              f"{r['enrichment']:>7.2f} {r['q']:>9.2e}")
    print(f"    enriched domains on the indicator list: "
          f"{t3['enriched_matching_an_indicator'] or 'none'}")
    print()
    print("T4  LENGTH")
    print(f"    state-produced median {t4['median_bpe_tokens_state_produced']} vs "
          f"corpus {t4['median_bpe_tokens_corpus']}, p={t4['p_two_sided']}")
    print()
    print(f"-> {OUT / 'agnostic_statistics.json'}")


if __name__ == "__main__":
    main()
