#!/usr/bin/env python3
"""Source-agnostic anomaly detection over the domain census.

Why this exists. Reporting that RT ranks 92nd is a fact with no baseline
attached: it invites the reader to supply their own sense of whether 92nd is
surprising. It is not an analysis. The analysis is to ask what rank-frequency
structure the census has, and then which domains sit ABOVE what that structure
predicts — with no reference to who anybody is.

Link-sharing populations of this kind are approximately power-law (Zipf) in
rank. Fitting log(links) ~ a + b*log(rank) over all 1,000 domains gives an
expectation for every domain; the residual says whether a domain carries more
links than a corpus of this shape predicts for something at its rank. Positive
outliers are over-represented relative to the corpus's own internal law.

This is the only handle the released data offers on the amplification question,
and it is a weak one — stated plainly in `interpretation` below. It cannot see
votes. It can only see whether the link distribution is lumpy in a way the
fitted law does not explain, and lumpiness has many innocent causes.

The test is run BEFORE identity is consulted. Only afterwards is the outlier
set intersected with the indicator list, so the indicators cannot steer it.

Output: output/census_anomaly.json
"""

import json
import math
from pathlib import Path

import numpy as np

from scanlib import load_indicators

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"
OUT = HERE / "output"


def second_level_label(domain):
    parts = domain.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac", "gov"}:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else parts[0]


def main():
    labels, links = [], []
    for line in open(DATA / "domains.txt"):
        parts = line.split()
        if len(parts) != 2:
            continue
        labels.append(parts[1])
        links.append(int(parts[0]))
    links = np.array(links, dtype=float)
    ranks = np.arange(1, len(links) + 1, dtype=float)

    # ---- Zipf fit: log(links) = a + b*log(rank) ---------------------------
    x, y = np.log(ranks), np.log(links)
    b, a = np.polyfit(x, y, 1)
    y_hat = a + b * x
    resid = y - y_hat
    sigma = float(resid.std(ddof=2))
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    z = resid / sigma

    # Head-truncated refit: the top of a Zipf curve is routinely flattened
    # (here by aggregators — google, archive, blogspot), which drags the slope
    # and can manufacture apparent outliers further down. Refit without the
    # top 20 and report both, so a conclusion cannot rest on the fit choice.
    xt, yt = x[20:], y[20:]
    bt, at = np.polyfit(xt, yt, 1)
    resid_t = y - (at + bt * x)
    sigma_t = float(resid_t[20:].std(ddof=2))
    z_t = resid_t / sigma_t

    # ---- outliers, chosen with no reference to identity -------------------
    THRESH = 2.0
    over = [i for i in range(len(labels)) if z[i] >= THRESH]
    over_t = [i for i in range(len(labels)) if z_t[i] >= THRESH]
    over.sort(key=lambda i: -z[i])

    # ---- only now is identity consulted -----------------------------------
    spec = load_indicators(HERE / "indicators.json")
    ind_labels = {}
    for ind in spec["indicators"]:
        if ind["type"] == "domain":
            ind_labels.setdefault(second_level_label(ind["pattern"]), []).append(
                {"id": ind["id"], "tier": ind["tier"], "actor": ind["actor"]})

    def row(i):
        return {
            "rank": i + 1,
            "label": labels[i],
            "links": int(links[i]),
            "predicted_links": round(math.exp(y_hat[i])),
            "residual_z": round(float(z[i]), 2),
            "residual_z_head_truncated_fit": round(float(z_t[i]), 2),
            "indicators": ind_labels.get(labels[i], []),
        }

    outlier_rows = [row(i) for i in over]
    n_outliers_indicator = sum(1 for r in outlier_rows if r["indicators"])

    # Where does every indicator domain actually sit?
    indicator_rows = []
    for i, lab in enumerate(labels):
        if lab in ind_labels:
            indicator_rows.append(row(i))
    indicator_rows.sort(key=lambda r: r["rank"])

    # Is the indicator set as a whole shifted relative to the corpus? Compare
    # non-baseline indicator residuals against the full distribution by rank.
    nb_z = [r["residual_z"] for r in indicator_rows
            if any(d["tier"] != "BASE" for d in r["indicators"])]
    base_z = [r["residual_z"] for r in indicator_rows
              if all(d["tier"] == "BASE" for d in r["indicators"])]

    result = {
        "generated_by": "09_census_anomaly.py",
        "scope": "ALL 1,000 census domains; outliers selected before identity is consulted",
        "fit": {
            "form": "log(links) = a + b*log(rank)",
            "a": round(float(a), 4), "b": round(float(b), 4),
            "r_squared": round(r2, 4),
            "residual_sigma": round(sigma, 4),
            "head_truncated_fit": {
                "a": round(float(at), 4), "b": round(float(bt), 4),
                "residual_sigma": round(sigma_t, 4),
                "note": "refit excluding the top 20 ranks",
            },
        },
        "outlier_threshold_z": THRESH,
        "n_outliers": len(over),
        "n_outliers_head_truncated_fit": len(over_t),
        "n_outliers_matching_an_indicator": n_outliers_indicator,
        "outliers": outlier_rows,
        "indicator_domains_in_census": indicator_rows,
        "residual_summary": {
            "non_baseline_indicator_z": {
                "n": len(nb_z),
                "values": [round(v, 2) for v in nb_z],
                "mean": round(float(np.mean(nb_z)), 3) if nb_z else None,
            },
            "baseline_indicator_z": {
                "n": len(base_z),
                "values": [round(v, 2) for v in base_z],
                "mean": round(float(np.mean(base_z)), 3) if base_z else None,
            },
        },
        "interpretation": [
            "A positive residual means a domain carries more links than the "
            "corpus's own rank-frequency law predicts for its rank. It is a "
            "statement about the shape of the link distribution, NOT about "
            "vote authenticity.",
            "Innocent causes of a positive residual are numerous and likely "
            "dominant: a domain that publishes far more articles than its "
            "neighbours, a paywall-free source that gets linked in preference "
            "to paywalled rivals, an aggregator, a site with a long archive.",
            "Because rank and links are not independent (rank is assigned BY "
            "links), residuals from this fit are not test statistics with "
            "clean null distributions. The z-scores are descriptive distances "
            "from a fitted line, not p-values, and are deliberately not "
            "converted into one.",
        ],
    }
    with open(OUT / "census_anomaly.json", "w") as f:
        json.dump(result, f, indent=1)

    print(f"Zipf fit: log(links) = {a:.3f} + {b:.3f}*log(rank)   R^2 = {r2:.4f}   "
          f"sigma = {sigma:.3f}")
    print(f"head-truncated refit: b = {bt:.3f}, sigma = {sigma_t:.3f}")
    print()
    print(f"positive outliers at z >= {THRESH}: {len(over)}  "
          f"(head-truncated fit: {len(over_t)})")
    print(f"of those, matching an indicator domain: {n_outliers_indicator}")
    print()
    print(f"{'rank':>5s} {'label':22s} {'links':>8s} {'pred':>8s} {'z':>6s} {'zt':>6s}  indicator")
    for r in outlier_rows[:25]:
        tag = ",".join(d["id"] for d in r["indicators"]) or "-"
        print(f"{r['rank']:>5d} {r['label'][:22]:22s} {r['links']:>8d} "
              f"{r['predicted_links']:>8d} {r['residual_z']:>6.2f} "
              f"{r['residual_z_head_truncated_fit']:>6.2f}  {tag}")
    print()
    print("WHERE THE INDICATOR DOMAINS ACTUALLY SIT")
    print(f"{'rank':>5s} {'label':22s} {'links':>8s} {'pred':>8s} {'z':>6s} {'zt':>6s}  tier")
    for r in indicator_rows:
        tiers = ",".join(sorted({d["tier"] for d in r["indicators"]}))
        print(f"{r['rank']:>5d} {r['label'][:22]:22s} {r['links']:>8d} "
              f"{r['predicted_links']:>8d} {r['residual_z']:>6.2f} "
              f"{r['residual_z_head_truncated_fit']:>6.2f}  {tiers}")


if __name__ == "__main__":
    main()
