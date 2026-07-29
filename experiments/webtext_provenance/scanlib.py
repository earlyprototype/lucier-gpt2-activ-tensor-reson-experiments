"""Matching machinery for the WebText provenance scan.

Kept importable and free of I/O so the boundary rules are unit-testable
(tests/test_webtext_provenance_scanlib.py). Two matchers:

- DomainMatcher: extracts domain-shaped tokens and matches an indicator when
  the indicator's pattern is the *registrable tail* of the token. This is the
  rule that makes 'www.rt.com' and 'amp.rt.com' hits for rt.com while
  'support.com' and 'rt.com.au' are not.

- PhraseMatcher: whole-word phrase matching, honouring per-indicator case
  sensitivity, with alphanumeric/underscore boundaries on both sides.
"""

import json
import re
from pathlib import Path

# Domain-shaped token: dotted labels ending in a 2-24 letter top-level domain.
# The guards stop us starting mid-token or ending before a trailing label.
DOMAIN_TOKEN_RE = re.compile(
    r"(?<![a-z0-9.-])"
    r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})"
    r"(?![a-z0-9-])"
)


def load_indicators(path):
    with open(path) as f:
        spec = json.load(f)
    return spec


class DomainMatcher:
    def __init__(self, indicators):
        # pattern (lowercase) -> indicator dict
        self.by_pattern = {
            ind["pattern"].lower(): ind
            for ind in indicators
            if ind["type"] == "domain"
        }
        # cheap substring prefilter needles
        self.needles = list(self.by_pattern.keys())

    def _match_token(self, token):
        """Return the indicator whose pattern is the registrable tail of token."""
        if token in self.by_pattern:
            return self.by_pattern[token]
        # walk suffixes: a.b.c.d -> b.c.d -> c.d
        parts = token.split(".")
        for i in range(1, len(parts) - 1):
            tail = ".".join(parts[i:])
            if tail in self.by_pattern:
                return self.by_pattern[tail]
        return None

    def scan(self, text_lower):
        """Yield (indicator, start, end) for each matching domain token."""
        if not any(n in text_lower for n in self.needles):
            return
        for m in DOMAIN_TOKEN_RE.finditer(text_lower):
            ind = self._match_token(m.group(1))
            if ind is not None:
                yield ind, m.start(1), m.end(1)


class PhraseMatcher:
    def __init__(self, indicators):
        self.entries = []
        for ind in indicators:
            if ind["type"] != "phrase":
                continue
            flags = 0 if ind.get("case_sensitive", True) else re.IGNORECASE
            rx = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(ind["pattern"]) + r"(?![A-Za-z0-9_])",
                flags,
            )
            needle = ind["pattern"] if ind.get("case_sensitive", True) else ind["pattern"].lower()
            self.entries.append((ind, rx, needle, ind.get("case_sensitive", True)))

    def scan(self, text, text_lower):
        """Yield (indicator, start, end) for each phrase occurrence."""
        for ind, rx, needle, cased in self.entries:
            hay = text if cased else text_lower
            if needle not in hay:
                continue
            for m in rx.finditer(text):
                yield ind, m.start(), m.end()


def context_snippet(text, start, end, radius=140):
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    snip = text[lo:hi].replace("\n", " / ")
    return ("…" if lo > 0 else "") + snip + ("…" if hi < len(text) else "")


def iter_corpus(data_dir):
    """Yield (split, doc_dict) across the three sample files, in id order."""
    for split in ("train", "valid", "test"):
        path = Path(data_dir) / f"webtext.{split}.jsonl"
        with open(path) as f:
            for line in f:
                yield split, json.loads(line)
