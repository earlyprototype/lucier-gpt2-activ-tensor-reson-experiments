#!/usr/bin/env python3
"""Fetch the public WebText sample and domain census, with checksum verification.

What this fetches, and what it is:

1. webtext.{train,valid,test}.jsonl — 250,000 / 5,000 / 5,000 documents of
   original WebText, released by OpenAI in the gpt-2-output-dataset repository
   ("250K documents from the WebText test set" — i.e. drawn from the WebText
   scrape's held-out portion, same collection pipeline as the training split:
   outbound Reddit links with at least 3 karma, scraped, deduplicated,
   Wikipedia removed; Radford et al. 2019, §2.1). This is the only sample of
   original WebText ever made public. The GPT-2 *model* was trained on the
   complementary split of the same scrape; for questions about what the
   collection pipeline admitted, the held-out sample is an unbiased draw.

2. domains.txt — OpenAI's official census: "the top 1,000 domains present in
   WebText and their frequency" (gpt-2 model card). Unlike the sample above,
   this covers the *whole* corpus, training split included. Domain names are
   given as bare second-level labels ("nytimes", "rt"), top-level domain
   stripped — a caveat every downstream number inherits.

Idempotent: files already present with matching SHA-256 are not re-downloaded.
Uses curl (present in the analysis environment, honours the proxy setup).
"""

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "_DATA"

AZURE = "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1"
RAW_GH = "https://raw.githubusercontent.com/openai/gpt-2/master"

# SHA-256 recorded at first fetch (2026-07-29); MD5s match the server's
# Content-MD5 headers, so these pin the same bytes OpenAI serves.
FILES = {
    "webtext.train.jsonl": {
        "url": f"{AZURE}/webtext.train.jsonl",
        "bytes": 679_129_270,
        "sha256": "cf6db7bb5c72b5ac683e52ed102203ae36ad88089344aafe0d64ca5f09f3c5e8",
    },
    "webtext.valid.jsonl": {
        "url": f"{AZURE}/webtext.valid.jsonl",
        "bytes": 13_622_302,
        "sha256": "e080dcf561d9ec19e9c4333ef1f48a48d1b677ff0b48bae2451f3aac5522033a",
    },
    "webtext.test.jsonl": {
        "url": f"{AZURE}/webtext.test.jsonl",
        "bytes": 13_478_245,
        "sha256": "934921a11915ffee16ecba791777dafa66cea5fe9cd2ecceb9776a41b1717f4c",
    },
    "domains.txt": {
        "url": f"{RAW_GH}/domains.txt",
        "bytes": 14_754,
        "sha256": "84b7eb3d611444b154e2b041254746e97e772bb52411fc3e9e190cdbdd000dc5",
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, spec: dict) -> dict:
    dest = DATA / name
    if dest.exists() and sha256_of(dest) == spec["sha256"]:
        print(f"  ok (cached)   {name}")
        return {"name": name, "verified": True, "cached": True}
    print(f"  downloading   {name}  ({spec['bytes'] / 1e6:.1f} MB)")
    subprocess.run(
        ["curl", "-sS", "--fail", "--retry", "4", "--retry-delay", "3",
         "-o", str(dest), spec["url"]],
        check=True,
    )
    got = sha256_of(dest)
    if got != spec["sha256"]:
        sys.exit(f"CHECKSUM MISMATCH for {name}: got {got}, pinned {spec['sha256']}")
    print(f"  ok (fetched)  {name}")
    return {"name": name, "verified": True, "cached": False}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    results = [fetch(name, spec) for name, spec in FILES.items()]
    manifest = {
        "fetched_on": str(date.today()),
        "files": {n: {**FILES[n]} for n in FILES},
        "results": results,
        "provenance": {
            "sample": "openai/gpt-2-output-dataset README: '250K documents from "
                      "the WebText test set' (train split of the release), plus "
                      "5K valid / 5K test.",
            "census": "openai/gpt-2 model_card.md: 'we've published a list of "
                      "the top 1,000 domains present in WebText and their "
                      "frequency' (domains.txt).",
            "corpus": "Radford et al. 2019 §2.1: outbound Reddit links with "
                      ">=3 karma, up to December 2017; ~8M documents, 40 GB.",
        },
    }
    with open(DATA / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest -> {DATA / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
