#!/usr/bin/env python3
"""Classify every non-baseline hit: is the document FROM the indicated source,
or ABOUT it?

The distinction carries the whole study. A corpus full of BBC articles about
the Internet Research Agency is a corpus that covered the war; a corpus
containing Strategic Culture Foundation essays is a corpus that ingested its
munitions. Naive keyword counting cannot tell these apart; this script plus
recorded human adjudication can.

Verdict vocabulary (one per document-indicator pair):
  origin                 — the document is content produced by the indicated
                           source (a Press TV article, an RT article)
  laundered_origin       — content produced by the source but carried into the
                           corpus via a repost/syndication elsewhere
                           ("originally posted at strategic-culture.org")
  citation_amplification — a third-party document approvingly built on the
                           source and citing it ("Sources: GlobalResearch.ca")
  wire_carriage          — a third-party document carrying the source's wire
                           copy with attribution ("BEIJING (Xinhua) —")
  mention                — the document talks about the source (coverage,
                           debunking, passing reference)
  false_positive         — the match is not the indicated entity at all
                           ("Deep in the Heart of Texas")
  ambiguous              — heuristics cannot decide and no adjudication exists

Precedence: a human adjudication in output/adjudications.json always
overrides the heuristic. All Tier A pairs MUST end adjudicated; the script
reports any that are not.

Outputs:
  _DATA/hit_docs.jsonl       (uncommitted) — full text of every hit document
  _DATA/review_queue.md      (uncommitted) — reading sheet for adjudication
  output/classified_hits.json    (committed) — final verdict per pair
  output/classification_summary.json (committed) — rollups used by RESULTS.md
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scanlib import iter_corpus, load_indicators

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"
OUT = HERE / "output"

MENTION_NEAR = re.compile(
    r"according to|reported|reports|state-run|state-owned|state media|"
    r"propaganda|kremlin-funded|state-funded|government-run|government-owned|"
    r"mouthpiece|-backed|funded by|fake news|disinformation|troll",
    re.IGNORECASE,
)

# "originally posted at <X>" / "Submitted by <name> via <X>" / "Source: <X>"
LAUNDER_FMT = r"(?:originally (?:posted|published) (?:at|on)|via|source[s]?\s*:)\s*(?:https?://)?(?:www\.)?{dom}"
CITE_FMT = r"(?:https?://|www\.){dom}"

# Wire-service carriage: "BEIJING, May 4 (Xinhua) --" style datelines
XINHUA_WIRE = re.compile(r"\((?:Xinhua|Xinhua News Agency)\)")

PRESSTV_ORIGIN = [
    re.compile(r"Press TV has conducted an interview"),
    re.compile(r"Press TV['’]s website"),
    re.compile(r"presstv\.(?:ir|com)", re.IGNORECASE),
    re.compile(r"PTV"),
]
RT_ORIGIN = [
    re.compile(r"READ MORE:"),
    re.compile(r"https?://(?:www\.)?rt\.com", re.IGNORECASE),
]
SPUTNIK_SATELLITE = re.compile(
    r"Sputnik (?:1|2|5|satellite|program|launch|era|moment|Planum|Planitia|"
    r"Monroe|Skull|Chandelier)|the Sputnik of|post-Sputnik|Sputnik-\d|"
    r"launch(?:ed)? (?:of )?Sputnik|Sputnik was launched",
)
# Sputnik's own copy: wire dateline ("MOSCOW (Sputnik) —") or photo furniture
SPUTNIK_STRONG = re.compile(r"\(Sputnik(?: News)?\)\s*[—–-]|©\s*Sputnik\s*/")
SPUTNIK_MEDIUM = re.compile(r"Radio Sputnik|Sputnik Persian|told Sputnik")
CHINADAILY_ORIGIN = re.compile(
    r"\(\s*China Daily(?: USA| Europe)?\s*\)|\(chinadaily\.com\.cn\)|"
    r"/chinadaily\.com\.cn\]|@chinadaily\.com\.cn"
)
CCTV_ORIGIN = re.compile(r"(?:^|丨|Editor:[^\n]{0,40})CCTV\.com", re.MULTILINE)
RT_LAUNDER = re.compile(
    r"ORIGINALLY PUBLISHED AT RT\.COM|Contributed by RT\.com|"
    r"R[Tt]\.com reports:", re.IGNORECASE
)
RT_AMERICA = re.compile(r"Find RT America in your area|\bBy RT\b")
PRESSTV_STRONG = re.compile(
    r"Press TV has conducted|(?:interview with|told|tells) Press TV"
)
PRESSTV_LAUNDER = re.compile(r"Press TV original title|[-–—]\s*Press TV\s*$|^\s*Press TV\s*[–—-]", re.MULTILINE)


def classify_pair(ind, text, low, n_occ):
    """Heuristic verdict + fired-feature list for one (doc, indicator) pair."""
    iid = ind["id"]
    pat = ind["pattern"].lower()
    feats = []

    if ind["type"] == "domain":
        dom = re.escape(pat)
        if re.search(LAUNDER_FMT.format(dom=dom), low):
            feats.append("launder-format")
            return "laundered_origin", feats
        if iid == "chn-state-chinadaily" and CHINADAILY_ORIGIN.search(text):
            feats.append("chinadaily-byline")
            return "origin", feats
        if iid == "chn-state-cctv" and CCTV_ORIGIN.search(text):
            feats.append("cctv-header")
            return "origin", feats
        n_url = len(re.findall(CITE_FMT.format(dom=dom), low))
        near_mention = bool(MENTION_NEAR.search(low))
        if iid == "rus-state-rt":
            if RT_LAUNDER.search(text):
                feats.append("rt-republication-credit")
                return "laundered_origin", feats
            if RT_AMERICA.search(text):
                feats.append("rt-america-boilerplate")
                return "origin", feats
            if any(rx.search(text) for rx in RT_ORIGIN) and n_occ >= 2:
                feats.append("rt-selflink-boilerplate")
                return "origin", feats
        if n_url and not near_mention:
            feats.append(f"cited-as-url x{n_url}")
            return "citation_amplification", feats
        if near_mention:
            feats.append("mention-vocabulary")
            return "mention", feats
        return "ambiguous", feats

    # phrases
    if iid == "chn-state-xinhua-phrase":
        if XINHUA_WIRE.search(text):
            feats.append("wire-dateline")
            return "wire_carriage", feats
        if MENTION_NEAR.search(low):
            feats.append("mention-vocabulary")
            return "mention", feats
        return "ambiguous", feats
    if iid == "irn-state-presstv-phrase":
        if PRESSTV_LAUNDER.search(text):
            feats.append("presstv-repost-credit")
            return "laundered_origin", feats
        if PRESSTV_STRONG.search(text) or text.strip().startswith("Press TV"):
            feats.append("presstv-boilerplate")
            return "origin", feats
        if any(rx.search(text) for rx in PRESSTV_ORIGIN):
            feats.append("presstv-weak-marker")
            return "ambiguous", feats
        if MENTION_NEAR.search(low):
            feats.append("mention-vocabulary")
            return "mention", feats
        return "ambiguous", feats
    if iid == "rus-state-sputnik-phrase":
        if SPUTNIK_STRONG.search(text):
            feats.append("sputnik-dateline-or-photo-credit")
            return "origin", feats
        if SPUTNIK_SATELLITE.search(text):
            feats.append("satellite-context")
            return "false_positive", feats
        if SPUTNIK_MEDIUM.search(text):
            feats.append("sputnik-weak-marker")
            return "ambiguous", feats
        if MENTION_NEAR.search(low):
            feats.append("mention-vocabulary")
            return "mention", feats
        return "ambiguous", feats
    if MENTION_NEAR.search(low):
        feats.append("mention-vocabulary")
        return "mention", feats
    return "ambiguous", feats


def main():
    spec = load_indicators(HERE / "indicators.json")
    ind_meta = {i["id"]: i for i in spec["indicators"]}

    hits = []
    want = defaultdict(list)  # (split, doc_id) -> [hit records]
    with open(DATA / "scan_hits_full.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["tier"] == "BASE":
                continue
            hits.append(r)
            want[(r["split"], r["doc_id"])].append(r)

    # pull full texts for every hit document
    texts = {}
    for split, doc in iter_corpus(DATA):
        key = (split, doc["id"])
        if key in want:
            texts[key] = doc["text"]
    with open(DATA / "hit_docs.jsonl", "w") as f:
        for (split, doc_id), text in sorted(texts.items(), key=lambda kv: kv[0][1]):
            f.write(json.dumps({"split": split, "id": doc_id, "text": text}) + "\n")

    adj_path = OUT / "adjudications.json"
    adjudications = json.load(open(adj_path)) if adj_path.exists() else {}

    classified = []
    for r in hits:
        key = (r["split"], r["doc_id"])
        text = texts[key]
        low = text.lower()
        ind = ind_meta[r["indicator"]]
        verdict, feats = classify_pair(ind, text, low, r["n_occurrences"])
        pair_key = f"{r['split']}:{r['doc_id']}:{r['indicator']}"
        adj = adjudications.get(pair_key)
        final = adj["verdict"] if adj else verdict
        classified.append({
            "pair": pair_key,
            "tier": r["tier"],
            "indicator": r["indicator"],
            "actor": ind["actor"],
            "n_occurrences": r["n_occurrences"],
            "heuristic_verdict": verdict,
            "heuristic_features": feats,
            "adjudicated": bool(adj),
            "final_verdict": final,
            "adjudication_note": (adj or {}).get("note"),
        })

    tier_rollup = defaultdict(Counter)
    actor_rollup = defaultdict(Counter)
    for c in classified:
        tier_rollup[c["tier"]][c["final_verdict"]] += 1
        actor_rollup[c["actor"]][c["final_verdict"]] += 1

    unreviewed_a = [c["pair"] for c in classified if c["tier"] == "A" and not c["adjudicated"]]

    with open(OUT / "classified_hits.json", "w") as f:
        json.dump(sorted(classified, key=lambda c: (c["tier"], c["indicator"], c["pair"])), f, indent=1)
    summary = {
        "generated_by": "03_classify_hits.py",
        "pairs_total": len(classified),
        "adjudicated_pairs": sum(c["adjudicated"] for c in classified),
        "per_tier_verdicts": {t: dict(v) for t, v in sorted(tier_rollup.items())},
        "per_actor_verdicts": {a: dict(v) for a, v in sorted(actor_rollup.items())},
        "unreviewed_tier_A_pairs": unreviewed_a,
    }
    with open(OUT / "classification_summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    # review sheet: all A and C; all B domain pairs; every ambiguous pair;
    # plus the first 25 of each large phrase family by doc id
    fam_cap = Counter()
    with open(DATA / "review_queue.md", "w") as f:
        for c in sorted(classified, key=lambda c: (c["tier"], c["indicator"], c["pair"])):
            iid = c["indicator"]
            is_domain = ind_meta[iid]["type"] == "domain"
            big_family = iid in ("chn-state-xinhua-phrase", "rus-state-sputnik-phrase",
                                 "rus-state-russia-today", "irn-state-presstv-phrase")
            take = (
                c["tier"] in ("A", "C")
                or (c["tier"] in ("B", "M") and is_domain)
                or c["heuristic_verdict"] == "ambiguous"
                or big_family
            )
            if big_family:
                fam_cap[iid] += 1
                if fam_cap[iid] > 25 and c["heuristic_verdict"] != "ambiguous":
                    take = False
            if not take:
                continue
            split, doc_id, _ = c["pair"].split(":")
            text = texts[(split, int(doc_id))]
            f.write(f"## {c['pair']}  [{c['tier']}] heuristic={c['heuristic_verdict']} "
                    f"({', '.join(c['heuristic_features']) or 'no features'})\n\n")
            f.write("```\n" + text[:1500].replace("```", "'''") + "\n```\n\n")

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
