#!/usr/bin/env python3
"""Inferential statistics for the WebText provenance audit.

This script exists because the first version of this study reported raw counts
and let prose carry the inference. Its central claim is a NULL — 18 of 19
attributed covert-asset domains return zero documents — and a null with no
bound on it is not a result. Everything below either bounds an unobserved
quantity, puts an interval on an observed one, or tests an assumption the
study leans on.

Sections:
  S1  Exact upper bounds on the zero-count indicators, and what they mean at
      corpus scale. This is the number the study was missing.
  S2  Detection power: what prevalence this scan could have found.
  S3  Exact and Wilson intervals on every observed rate.
  S4  Scanner recall estimated from the census, and a recall-corrected
      prevalence — turning "every count is a floor" into a number.
  S5  Classifier agreement (Cohen's kappa), with its in-sample caveat.
  S6  Family-wise bound over the 19-domain covert tier.
  S7  Where inference does NOT apply, stated explicitly.

Output: output/statistics.json
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from statlib import (
    binom_test_two_sided,
    clopper_pearson_interval,
    clopper_pearson_upper,
    cohen_kappa,
    detectable_prevalence,
    wilson_interval,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

N = 260_000            # documents scanned
CORPUS_DOCS = 8_000_000  # "slightly over 8 million" (Radford et al. 2019)
PRODUCED = ("origin", "laundered_origin", "wire_carriage")


def per_10k(p):
    return round(p * 1e4, 3)


def main():
    spec = json.load(open(HERE / "indicators.json"))
    summ = json.load(open(OUT / "scan_summary.json"))
    cl = json.load(open(OUT / "classified_hits.json"))
    cen = json.load(open(OUT / "domain_census.json"))

    ind_by_id = {i["id"]: i for i in spec["indicators"]}
    zero_ids = set(summ["zero_hit_indicators"])

    # ---------------------------------------------------------------- S1
    # Zero observations in N documents. Exact one-sided Clopper-Pearson upper
    # bound; for k=0 this is the closed form 1 - alpha^(1/N), the exact
    # statement behind the 'rule of three'.
    ub95 = clopper_pearson_upper(0, N, alpha=0.05)
    ub99 = clopper_pearson_upper(0, N, alpha=0.01)

    covert_zero = sorted(
        i for i in zero_ids
        if ind_by_id[i]["tier"] == "A" and ind_by_id[i]["type"] == "domain")

    s1 = {
        "observed_hits": 0,
        "documents_scanned": N,
        "upper_bound_95_per_document": ub95,
        "upper_bound_99_per_document": ub99,
        "upper_bound_95_per_10k_documents": per_10k(ub95),
        "implied_corpus_ceiling_95": round(ub95 * CORPUS_DOCS, 1),
        "implied_corpus_ceiling_99": round(ub99 * CORPUS_DOCS, 1),
        "covert_domains_at_zero": covert_zero,
        "n_covert_domains_at_zero": len(covert_zero),
        "reading": (
            f"For any single indicator observed zero times, the prevalence is "
            f"below {per_10k(ub95)} per 10,000 documents at 95% confidence. "
            f"Extrapolated to the ~8M-document corpus under the assumption "
            f"that the sample is a random draw from it, that ceilings each "
            f"absent covert asset at ~{round(ub95 * CORPUS_DOCS)} documents "
            f"corpus-wide — against ~8 million documents total."
        ),
    }

    # ---------------------------------------------------------------- S2
    # What could this scan have found? A null is only as strong as the power
    # behind it.
    s2 = {
        "prevalence_detectable_at_power": {
            f"{int(100 * pw)}%": {
                "per_document": detectable_prevalence(N, pw),
                "per_10k_documents": per_10k(detectable_prevalence(N, pw)),
                "equivalent_corpus_documents": round(
                    detectable_prevalence(N, pw) * CORPUS_DOCS, 1),
            }
            for pw in (0.5, 0.8, 0.95, 0.99)
        },
        "reading": (
            "A covert asset contributing as few as ~50 documents corpus-wide "
            "would have produced at least one hit in this sample with 80% "
            "probability. The null is therefore informative about assets of "
            "that size and larger, and uninformative about a handful of "
            "documents hiding in 8 million."
        ),
    }

    # ---------------------------------------------------------------- S3
    # Intervals on what WAS observed.
    produced_docs = set()
    fam_docs = defaultdict(set)
    for c in cl:
        if c["final_verdict"] in PRODUCED:
            split, doc_id, _ = c["pair"].split(":")
            produced_docs.add((split, doc_id))
            fam_docs[c["actor"]].add((split, doc_id))

    k = len(produced_docs)
    lo, hi = clopper_pearson_interval(k, N)
    wlo, whi = wilson_interval(k, N)
    s3 = {
        "state_produced": {
            "k": k, "n": N,
            "point_per_10k": per_10k(k / N),
            "exact_95_ci_per_10k": [per_10k(lo), per_10k(hi)],
            "wilson_95_ci_per_10k": [per_10k(wlo), per_10k(whi)],
            "implied_corpus_documents_95_ci": [
                round(lo * CORPUS_DOCS), round(hi * CORPUS_DOCS)],
        },
        "per_actor": {},
    }
    for actor, docs in sorted(fam_docs.items(), key=lambda kv: -len(kv[1])):
        ka = len(docs)
        alo, ahi = clopper_pearson_interval(ka, N)
        s3["per_actor"][actor] = {
            "k": ka,
            "point_per_10k": per_10k(ka / N),
            "exact_95_ci_per_10k": [per_10k(alo), per_10k(ahi)],
        }

    # ---------------------------------------------------------------- S4
    # Scanner recall, estimated by comparing what the census predicts against
    # what the text scan actually captured. This converts the qualitative
    # "every count is a floor" caveat into a measured correction factor.
    #
    # H0: uniform survival from link -> corpus document -> sample, AND perfect
    # text capture. Under H0, observed ~ Binomial(N, expected/N). Rejecting H0
    # measures the scanner's recall deficit, which is the quantity of interest.
    recall_rows = []
    for r in cen["rows"]:
        exp = r["naive_expected_in_sample"]
        obs = r["sample_produced_docs"]
        if r["census_rank"] is None or obs is None or exp <= 0:
            continue
        p0 = exp / N
        p_val = binom_test_two_sided(obs, N, p0)
        recall = obs / exp if exp else float("nan")
        rlo, rhi = clopper_pearson_interval(obs, N)
        recall_rows.append({
            "label": r["census_label"],
            "tier": r["tier"],
            "census_links": r["census_links"],
            "expected_docs_under_uniform_survival": exp,
            "observed_produced_docs": obs,
            "estimated_recall": round(recall, 3),
            "recall_95_ci": [round(rlo * N / exp, 3), round(rhi * N / exp, 3)],
            "binomial_p_two_sided": p_val,
            "label_collision": r["label_collision"],
        })
    recall_rows.sort(key=lambda x: -x["census_links"])

    clean = [r for r in recall_rows if not r["label_collision"]]
    pooled_obs = sum(r["observed_produced_docs"] for r in clean)
    pooled_exp = sum(r["expected_docs_under_uniform_survival"] for r in clean)
    pooled_recall = pooled_obs / pooled_exp if pooled_exp else float("nan")
    plo, phi = clopper_pearson_interval(pooled_obs, N)

    corrected = k / pooled_recall if pooled_recall else float("nan")
    s4 = {
        "rows": recall_rows,
        "pooled_excluding_collisions": {
            "observed": pooled_obs,
            "expected": round(pooled_exp, 1),
            "estimated_recall": round(pooled_recall, 3),
            "recall_95_ci": [round(plo * N / pooled_exp, 3),
                             round(phi * N / pooled_exp, 3)],
        },
        "recall_corrected_state_produced": {
            "observed_k": k,
            "correction_factor": round(1 / pooled_recall, 2) if pooled_recall else None,
            "corrected_docs_in_sample": round(corrected, 1),
            "corrected_per_10k": per_10k(corrected / N),
            "corrected_corpus_documents": round(corrected * CORPUS_DOCS / N),
        },
        "caveats": [
            "The expectation rests on uniform survival from link to sampled "
            "document, which is an assumption, not a measurement: the census "
            "counts links (~21.8M over the top 1000) while the corpus holds "
            "~8M deduplicated documents, and survival plausibly varies by "
            "domain (paywalls, robots.txt, dead links, dedup).",
            "Recall is estimated from four to six domains, so the pooled "
            "figure is dominated by the largest of them.",
            "The corrected figure is an order-of-magnitude estimate, not a "
            "measurement. The observed count remains the defensible floor.",
        ],
    }

    # ---------------------------------------------------------------- S5
    # How much should the 392 unadjudicated pairs be trusted? Measure how often
    # the heuristic agreed with a human on the 213 that were adjudicated.
    adjudicated = [(c["heuristic_verdict"], c["final_verdict"])
                   for c in cl if c["adjudicated"]]
    agree = sum(1 for a, b in adjudicated if a == b)
    kappa = cohen_kappa(adjudicated)
    disagreements = Counter(
        (a, b) for a, b in adjudicated if a != b)

    # "ambiguous" is the heuristic declining to decide, not a rival label. A
    # kappa that counts it as disagreement conflates abstention with error, so
    # both are reported: the decided-only figure measures how often the rules
    # were WRONG, the full figure how often they were wrong or silent.
    decided = [(a, b) for a, b in adjudicated if a != "ambiguous"]
    decided_agree = sum(1 for a, b in decided if a == b)
    s5 = {
        "n_adjudicated_pairs": len(adjudicated),
        "raw_agreement": round(agree / len(adjudicated), 3) if adjudicated else None,
        "cohen_kappa": round(kappa, 3),
        "abstention_rate": round(
            sum(1 for a, _ in adjudicated if a == "ambiguous") / len(adjudicated), 3),
        "decided_only": {
            "n": len(decided),
            "raw_agreement": round(decided_agree / len(decided), 3) if decided else None,
            "cohen_kappa": round(cohen_kappa(decided), 3) if decided else None,
            "note": "pairs where the heuristic committed to a verdict",
        },
        "top_disagreements": [
            {"heuristic": a, "human": b, "n": n}
            for (a, b), n in disagreements.most_common(8)
        ],
        "IN_SAMPLE_CAVEAT": (
            "The heuristics were REVISED after reading these documents — "
            "wire-dateline, photo-credit and repost-credit patterns were added "
            "once adjudication revealed them. This agreement is therefore an "
            "in-sample fit, an optimistic bound on out-of-sample accuracy, and "
            "NOT a validation. It is reported to show how far the rules had to "
            "be corrected, not to certify them."
        ),
    }

    # ---------------------------------------------------------------- S6
    # The study's claim is not about one indicator but about a family: 'no
    # attributed covert asset is present'. Bonferroni over the family.
    m = len(covert_zero)
    fam_alpha = 0.05 / m if m else 0.05
    fam_ub = clopper_pearson_upper(0, N, alpha=fam_alpha)
    s6 = {
        "family_size": m,
        "per_test_alpha_bonferroni": fam_alpha,
        "family_wise_upper_bound_per_document": fam_ub,
        "family_wise_upper_bound_per_10k": per_10k(fam_ub),
        "implied_corpus_ceiling_per_asset": round(fam_ub * CORPUS_DOCS, 1),
        "reading": (
            f"Holding the family-wise error rate at 5% across all {m} covert "
            f"domains simultaneously, each is still bounded below "
            f"{per_10k(fam_ub)} per 10,000 documents — about "
            f"{round(fam_ub * CORPUS_DOCS)} documents corpus-wide. The "
            f"correction weakens the per-asset bound by roughly a factor of "
            f"{round(fam_ub / ub95, 1)}, which does not change the conclusion."
        ),
    }

    # ---------------------------------------------------------------- S7
    s7 = {
        "census_is_a_population_not_a_sample": (
            "domains.txt reports counts over the WHOLE corpus. Confidence "
            "intervals and significance tests do not apply to it: RT ranking "
            "92nd is a fact about the corpus, not an estimate with sampling "
            "error. Section 4 uses the census only as a fixed expectation "
            "against which to measure the scanner, which is a different use."
        ),
        "mechanism_remains_untestable": (
            "No statistic here speaks to inauthentic amplification. Vote "
            "provenance was never released; no amount of analysis of the "
            "documents recovers it. The bounds above constrain CONTENT "
            "prevalence only."
        ),
        "near_duplicate_null_bound": {
            "pairs_compared": 578 * 577 // 2,
            "observed_at_or_above_0.25": 0,
            "upper_bound_95_on_pair_rate": clopper_pearson_upper(
                0, 578 * 577 // 2, alpha=0.05),
            "note": (
                "Pairs are not independent (they share documents), so this "
                "bound is indicative rather than exact. Reported because a "
                "bounded null beats an unbounded one."
            ),
        },
    }

    result = {
        "generated_by": "07_statistics.py",
        "S1_zero_count_upper_bounds": s1,
        "S2_detection_power": s2,
        "S3_observed_rate_intervals": s3,
        "S4_scanner_recall_and_correction": s4,
        "S5_classifier_agreement": s5,
        "S6_family_wise_bound": s6,
        "S7_scope_of_inference": s7,
    }
    with open(OUT / "statistics.json", "w") as f:
        json.dump(result, f, indent=1)

    # ------------------------------------------------------------- report
    print("S1  ZERO-COUNT BOUNDS")
    print(f"    0 hits in {N:,} docs -> p < {ub95:.3e} (95%), < {ub99:.3e} (99%)")
    print(f"    = under {per_10k(ub95)} per 10,000 documents")
    print(f"    = ceiling of ~{round(ub95 * CORPUS_DOCS)} documents corpus-wide, per absent asset")
    print(f"    {len(covert_zero)} covert-asset domains sit at this bound")
    print()
    print("S2  DETECTION POWER")
    for pw, v in s2["prevalence_detectable_at_power"].items():
        print(f"    {pw:>4s} power: detects prevalence >= {v['per_10k_documents']:.4f}/10k "
              f"(~{v['equivalent_corpus_documents']:.0f} docs corpus-wide)")
    print()
    print("S3  OBSERVED RATES (exact 95% CI)")
    sp = s3["state_produced"]
    print(f"    state-produced: {sp['k']}/{N:,} = {sp['point_per_10k']}/10k "
          f"[{sp['exact_95_ci_per_10k'][0]}, {sp['exact_95_ci_per_10k'][1]}]")
    print(f"    implied corpus-wide: {sp['implied_corpus_documents_95_ci'][0]:,} "
          f"to {sp['implied_corpus_documents_95_ci'][1]:,} documents")
    for actor, v in list(s3["per_actor"].items())[:4]:
        print(f"      {actor[:44]:44s} {v['k']:>3d}  {v['point_per_10k']:>5.3f}/10k "
              f"[{v['exact_95_ci_per_10k'][0]}, {v['exact_95_ci_per_10k'][1]}]")
    print()
    print("S4  SCANNER RECALL (census expectation vs text capture)")
    print(f"    {'label':16s} {'links':>7s} {'exp':>7s} {'obs':>5s} {'recall':>7s} {'p':>10s}")
    for r in recall_rows:
        flag = " *collision" if r["label_collision"] else ""
        print(f"    {r['label']:16s} {r['census_links']:>7d} "
              f"{r['expected_docs_under_uniform_survival']:>7.1f} "
              f"{r['observed_produced_docs']:>5d} {r['estimated_recall']:>7.3f} "
              f"{r['binomial_p_two_sided']:>10.2e}{flag}")
    pc = s4["pooled_excluding_collisions"]
    rc = s4["recall_corrected_state_produced"]
    print(f"    pooled recall = {pc['estimated_recall']} "
          f"-> correction x{rc['correction_factor']}")
    print(f"    recall-corrected: {rc['corrected_docs_in_sample']} docs in sample "
          f"({rc['corrected_per_10k']}/10k), ~{rc['corrected_corpus_documents']:,} corpus-wide")
    print()
    print("S5  CLASSIFIER AGREEMENT (in-sample)")
    print(f"    heuristic vs human on {s5['n_adjudicated_pairs']} pairs: "
          f"agreement {s5['raw_agreement']}, kappa {s5['cohen_kappa']} "
          f"(abstained on {s5['abstention_rate']:.0%})")
    d5 = s5["decided_only"]
    print(f"    where the heuristic committed ({d5['n']} pairs): "
          f"agreement {d5['raw_agreement']}, kappa {d5['cohen_kappa']}")
    for d in s5["top_disagreements"][:5]:
        print(f"      heuristic={d['heuristic']:22s} -> human={d['human']:22s} n={d['n']}")
    print()
    print("S6  FAMILY-WISE (Bonferroni over the covert tier)")
    print(f"    {s6['family_size']} domains, alpha={s6['per_test_alpha_bonferroni']:.5f} "
          f"-> p < {s6['family_wise_upper_bound_per_document']:.3e} "
          f"(~{round(s6['implied_corpus_ceiling_per_asset'])} docs corpus-wide)")
    print()
    print(f"statistics -> {OUT / 'statistics.json'}")


if __name__ == "__main__":
    main()
