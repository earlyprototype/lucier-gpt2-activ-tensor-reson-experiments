#!/usr/bin/env python3
"""Cross-reference OpenAI's whole-corpus domain census against the indicator
list, and reconcile it with the text-scan sample.

What domains.txt is, learned the hard way: its 1,000 counts sum to ~21.8
million, far more than the corpus's ~8 million documents — so the census
counts LINKS (of the ~45 million scraped Reddit outbound links), not
deduplicated final documents. Every number below is a link frequency, and
the sample expectation column is computed under the explicitly naive
assumption that survival from link to final document to sample is uniform
across domains (expected = census_links * 260000 / 45000000).

Names in the census are bare second-level labels ('rt', 'nytimes'), so
matching maps each indicator domain to its label. Labels can collide across
registrable domains; rows carry an explicit collision flag, and the one
known-bad case — 'people', which is People magazine, not People's Daily —
is excluded from tier totals.

The 'sample_produced_docs' column joins, by actor family, the number of
sample documents whose text the classification pass verdicts as
state-produced (origin, laundered_origin, or wire_carriage) — including
phrase-indicator captures (a Sputnik wire story carries no sputniknews.com
URL in its text). Baseline rows are rate context only and never classified,
so the column is null there.

Output: output/domain_census.json
"""

import json
from collections import defaultdict
from pathlib import Path

from scanlib import load_indicators

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"
OUT = HERE / "output"

TOTAL_LINKS_NOMINAL = 45_000_000  # "45 million links" (Radford et al. 2019; model card)
SAMPLE_DOCS = 260_000

# indicator id -> actor family, for joining sample capture onto census rows
FAMILY = {
    "rus-state-rt": "rt", "rus-state-russia-today": "rt",
    "rus-state-sputniknews": "sputniknews", "rus-state-sputnik-phrase": "sputniknews",
    "irn-state-presstv-ir": "presstv", "irn-state-presstv-com": "presstv",
    "irn-state-presstv-phrase": "presstv",
    "chn-state-xinhuanet": "xinhua", "chn-state-xinhua-phrase": "xinhua",
    "chn-state-chinadaily": "chinadaily",
    "chn-state-peoples-daily": "people", "chn-state-peoples-daily-cn": "people",
    "chn-state-globaltimes": "globaltimes",
    "prk-state-kcna": "kcna", "prk-state-kcna-jp": "kcna",
    "ven-state-telesur": "telesurtv", "ven-state-telesur-en": "telesurenglish",
    "chn-state-cctv": "cctv", "chn-state-cgtn": "cgtn",
    "rus-intel-strategic-culture": "strategic-culture",
    "amp-globalresearch": "globalresearch",
    "amp-veteranstoday": "veteranstoday",
    "rus-state-tass": "tass", "rus-state-tass-ru": "tass",
    "rus-state-rbth": "rbth", "rus-state-ria": "ria",
}

# census labels known or strongly suspected to aggregate an unrelated site
LABEL_COLLISIONS = {
    "people": "almost certainly people.com (People magazine), not people.com.cn — excluded from totals",
    "rt": "assumed rt.com at this volume; minor collision possible (e.g. rt.de is also RT)",
}


def second_level_label(domain):
    parts = domain.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac", "gov"}:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else parts[0]


def main():
    census = {}
    anomalies = []
    for rank, line in enumerate(open(DATA / "domains.txt"), start=1):
        parts = line.split()
        if len(parts) != 2:
            # line 217 of the published file is a bare count with an empty
            # domain name: 20,068 links attributed to no domain at all
            anomalies.append({"rank": rank, "raw": line.strip()})
            continue
        count, name = parts
        census[name] = {"rank": rank, "links": int(count)}
    total_top1000 = sum(v["links"] for v in census.values()) + sum(
        int(a["raw"]) for a in anomalies if a["raw"].isdigit())

    spec = load_indicators(HERE / "indicators.json")
    ind_by_id = {i["id"]: i for i in spec["indicators"]}

    produced_by_family = defaultdict(set)  # family -> {(split, doc_id)}
    for c in json.load(open(OUT / "classified_hits.json")):
        if c["final_verdict"] in ("origin", "laundered_origin", "wire_carriage"):
            fam = FAMILY.get(c["indicator"])
            if fam:
                split, doc_id, _ = c["pair"].split(":")
                produced_by_family[fam].add((split, doc_id))

    # one row per census label, aggregating the indicator domains behind it
    by_label = {}
    for ind in spec["indicators"]:
        if ind["type"] != "domain":
            continue
        key = FAMILY.get(ind["id"]) or second_level_label(ind["pattern"])
        row = by_label.setdefault(key, {
            "census_label": key,
            "tier": ind["tier"],
            "actors": set(),
            "domains": [],
        })
        row["actors"].add(ind["actor"])
        row["domains"].append(ind["pattern"])

    rows = []
    for key, row in by_label.items():
        hit = census.get(key)
        fam_produced = produced_by_family.get(key)
        collision = LABEL_COLLISIONS.get(key)
        rows.append({
            "census_label": key,
            "tier": row["tier"],
            "actors": sorted(row["actors"]),
            "domains": sorted(set(row["domains"])),
            "census_rank": hit["rank"] if hit else None,
            "census_links": hit["links"] if hit else 0,
            "share_of_top1000_links": round(hit["links"] / total_top1000, 5) if hit else 0.0,
            "naive_expected_in_sample": round(
                (hit["links"] if hit else 0) * SAMPLE_DOCS / TOTAL_LINKS_NOMINAL, 1),
            "sample_produced_docs": (len(fam_produced) if fam_produced else 0)
            if row["tier"] != "BASE" else None,
            "label_collision": collision,
        })
    rows.sort(key=lambda r: (r["census_rank"] is None, r["census_rank"] or 0))

    per_tier = defaultdict(int)
    for r in rows:
        if r["label_collision"] and r["census_label"] == "people":
            continue
        per_tier[r["tier"]] += r["census_links"]

    result = {
        "generated_by": "05_domain_census.py",
        "census_provenance": "openai/gpt-2 domains.txt — 'the top 1,000 domains "
                             "present in WebText and their frequency' (model card)",
        "census_unit": "links among the ~45M scraped (counts sum to ~21.8M, "
                       "far above the ~8M deduplicated documents)",
        "total_links_nominal": TOTAL_LINKS_NOMINAL,
        "top1000_total_links": total_top1000,
        "top1000_share_of_links_nominal": round(total_top1000 / TOTAL_LINKS_NOMINAL, 4),
        "per_tier_census_links_excluding_collisions": dict(per_tier),
        "census_anomalies": anomalies,
        "rows": rows,
    }
    with open(OUT / "domain_census.json", "w") as f:
        json.dump(result, f, indent=1)

    print(f"top-1000 total links: {total_top1000}  "
          f"({100 * total_top1000 / TOTAL_LINKS_NOMINAL:.1f}% of nominal 45M)")
    print(f"{'label':20s} {'tier':4s} {'rank':>5s} {'links':>8s} {'expect':>7s} {'seen':>5s}")
    for r in rows:
        if r["census_rank"] is not None or (r["sample_produced_docs"] or 0) > 0:
            seen = "-" if r["sample_produced_docs"] is None else str(r["sample_produced_docs"])
            flag = " (collision)" if r["label_collision"] else ""
            print(f"{r['census_label']:20s} {r['tier']:4s} "
                  f"{str(r['census_rank'] or '-'):>5s} {r['census_links']:>8d} "
                  f"{r['naive_expected_in_sample']:>7.1f} {seen:>5s}{flag}")


if __name__ == "__main__":
    main()
