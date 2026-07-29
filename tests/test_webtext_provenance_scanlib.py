"""Boundary rules for the WebText provenance scanner.

The whole study rests on the matcher not lying: a domain indicator must hit
subdomains of itself and nothing else, and phrase indicators must respect
word boundaries and case. These tests pin those rules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "webtext_provenance"))

from scanlib import DomainMatcher, PhraseMatcher  # noqa: E402


def _dom(patterns):
    inds = [
        {"id": f"i{k}", "type": "domain", "pattern": p, "tier": "B", "actor": "t"}
        for k, p in enumerate(patterns)
    ]
    return DomainMatcher(inds)


def _hits(matcher, text):
    low = text.lower()
    return [ind["pattern"] for ind, _, _ in matcher.scan(low)]


def test_domain_exact_and_subdomain_match():
    m = _dom(["rt.com"])
    assert _hits(m, "read https://www.rt.com/usa/article") == ["rt.com"]
    assert _hits(m, "cached at amp.rt.com today") == ["rt.com"]
    assert _hits(m, "plain rt.com works") == ["rt.com"]


def test_domain_no_substring_false_positive():
    m = _dom(["rt.com"])
    # 'rt.com' is a substring of both, but the registrable tail differs
    assert _hits(m, "visit support.com for help") == []
    assert _hits(m, "the site rt.com.au is different") == []


def test_domain_trailing_punctuation_and_urls():
    m = _dom(["presstv.ir"])
    assert _hits(m, "reported by presstv.ir.") == ["presstv.ir"]
    assert _hits(m, "(http://presstv.ir/Detail/2017)") == ["presstv.ir"]


def test_domain_multi_label_pattern():
    m = _dom(["chinadaily.com.cn", "people.cn"])
    assert _hits(m, "see www.chinadaily.com.cn/china") == ["chinadaily.com.cn"]
    assert _hits(m, "see en.people.cn/n3/article") == ["people.cn"]
    # people.com.cn must NOT satisfy the people.cn suffix rule
    m2 = _dom(["people.cn"])
    assert _hits(m2, "see people.com.cn today") == []


def _phr(pattern, cased=True):
    inds = [{
        "id": "p0", "type": "phrase", "pattern": pattern,
        "case_sensitive": cased, "tier": "M", "actor": "t",
    }]
    return PhraseMatcher(inds)


def _phits(matcher, text):
    return [ind["pattern"] for ind, _, _ in matcher.scan(text, text.lower())]


def test_phrase_word_boundaries():
    m = _phr("TEN_GOP")
    assert _phits(m, "the @TEN_GOP account") == ["TEN_GOP"]
    assert _phits(m, "brightened by OFTEN_GOPHER") == []


def test_phrase_case_sensitivity():
    m = _phr("Russia Today")
    assert _phits(m, "watched Russia Today at noon") == ["Russia Today"]
    assert _phits(m, "in russia today the weather") == []
    mi = _phr("troll farm", cased=False)
    assert _phits(mi, "a Troll Farm in St Petersburg") == ["troll farm"]


def test_phrase_boundary_blocks_prefix_suffix():
    m = _phr("Xinhua")
    assert _phits(m, "per Xinhua, the launch") == ["Xinhua"]
    assert _phits(m, "the Xinhuanet mirror") == []
