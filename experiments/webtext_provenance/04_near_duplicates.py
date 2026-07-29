#!/usr/bin/env python3
"""Near-duplicate clustering across all non-baseline hit documents.

Duplicate or near-duplicate documents inside the corpus are a coordination
fingerprint worth checking: the same state-produced text arriving through
two different URLs means two separate Reddit submissions each cleared the
karma gate — republication doing exactly what laundering is for. (It can
also just be an ordinary syndication artefact; the cluster listing lets the
reader judge.)

Method: 8-word shingles, exact pairwise Jaccard similarity over every pair
of hit documents (N is ~1k, so brute force is fine and exact), single-link
clusters at Jaccard >= 0.5.

Output: output/dup_clusters.json
"""

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"
OUT = HERE / "output"

SHINGLE_W = 8
THRESHOLD = 0.5       # cluster threshold: near-duplicate documents
LOW_THRESHOLD = 0.25  # recorded but not clustered: substantial shared text


def shingles(text):
    words = text.lower().split()
    return {" ".join(words[i:i + SHINGLE_W]) for i in range(max(0, len(words) - SHINGLE_W + 1))}


def main():
    docs = {}
    for line in open(DATA / "hit_docs.jsonl"):
        d = json.loads(line)
        docs[(d["split"], d["id"])] = d["text"]

    verdicts = defaultdict(list)
    for c in json.load(open(OUT / "classified_hits.json")):
        split, doc_id, _ = c["pair"].split(":")
        verdicts[(split, int(doc_id))].append(
            (c["indicator"], c["final_verdict"]))

    keys = sorted(docs)
    sh = {k: shingles(docs[k]) for k in keys}

    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    pairs = []
    for i, a in enumerate(keys):
        sa = sh[a]
        if not sa:
            continue
        for b in keys[i + 1:]:
            sb = sh[b]
            if not sb:
                continue
            inter = len(sa & sb)
            if not inter:
                continue
            j = inter / (len(sa) + len(sb) - inter)
            if j >= LOW_THRESHOLD:
                pairs.append({"a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}",
                              "jaccard": round(j, 3)})
            if j >= THRESHOLD:
                union(a, b)

    clusters = defaultdict(list)
    for k in keys:
        clusters[find(k)].append(k)
    cluster_out = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        cluster_out.append({
            "size": len(members),
            "members": [
                {"doc": f"{m[0]}:{m[1]}",
                 "chars": len(docs[m]),
                 "hits": verdicts.get(m, [])}
                for m in sorted(members, key=lambda m: m[1])
            ],
        })
    cluster_out.sort(key=lambda c: -c["size"])

    result = {
        "generated_by": "04_near_duplicates.py",
        "n_docs_compared": len(keys),
        "shingle_words": SHINGLE_W,
        "jaccard_threshold": THRESHOLD,
        "n_pairs_over_cluster_threshold": sum(p["jaccard"] >= THRESHOLD for p in pairs),
        "n_pairs_over_low_threshold": len(pairs),
        "pairs": sorted(pairs, key=lambda p: -p["jaccard"]),
        "clusters": cluster_out,
    }
    with open(OUT / "dup_clusters.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in
                      ("n_docs_compared", "n_pairs_over_cluster_threshold", "n_pairs_over_low_threshold")}, indent=1))
    for c in cluster_out:
        print("cluster:", [m["doc"] for m in c["members"]])


if __name__ == "__main__":
    main()
