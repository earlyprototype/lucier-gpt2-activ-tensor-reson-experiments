"""Exactness checks for the provenance study's statistics.

statlib.py is pure Python so the repository gains no dependency. That is only
safe if the implementations are actually right, so each is checked against a
closed form, a known published value, or scipy where scipy is installed.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "webtext_provenance"))

from statlib import (  # noqa: E402
    binom_cdf,
    binom_sf,
    binom_test_two_sided,
    clopper_pearson_interval,
    clopper_pearson_lower,
    clopper_pearson_upper,
    cohen_kappa,
    detectable_prevalence,
    wilson_interval,
)

try:
    from scipy import stats as _sps
except ImportError:  # pragma: no cover
    _sps = None

needs_scipy = pytest.mark.skipif(_sps is None, reason="scipy not installed")


def test_binom_cdf_against_closed_form():
    # P(X <= 1) for n=3, p=0.5 is (1 + 3)/8
    assert binom_cdf(1, 3, 0.5) == pytest.approx(0.5)
    assert binom_cdf(3, 3, 0.5) == pytest.approx(1.0)
    assert binom_cdf(-1, 3, 0.5) == 0.0


def test_binom_sf_complements_cdf():
    for k in range(0, 6):
        assert binom_sf(k, 5, 0.3) + binom_cdf(k - 1, 5, 0.3) == pytest.approx(1.0)


def test_zero_count_upper_bound_is_rule_of_three():
    """For k=0 the exact bound is 1 - alpha^(1/n), which the rule of three
    approximates as 3/n. The study's headline bound depends on this."""
    n = 260_000
    exact = clopper_pearson_upper(0, n, alpha=0.05)
    assert exact == pytest.approx(1 - 0.05 ** (1 / n))
    assert exact == pytest.approx(3 / n, rel=0.03)


def test_upper_bound_is_monotone_in_k_and_alpha():
    n = 1000
    assert clopper_pearson_upper(0, n) < clopper_pearson_upper(5, n)
    assert clopper_pearson_upper(0, n, 0.01) > clopper_pearson_upper(0, n, 0.05)


def test_clopper_pearson_bounds_bracket_the_estimate():
    lo, hi = clopper_pearson_interval(155, 260_000)
    assert lo < 155 / 260_000 < hi


def test_clopper_pearson_lower_is_zero_at_zero_successes():
    assert clopper_pearson_lower(0, 100) == 0.0


@needs_scipy
def test_clopper_pearson_matches_scipy_beta():
    for k, n in [(0, 100), (1, 100), (5, 1000), (155, 260_000)]:
        hi = clopper_pearson_upper(k, n, alpha=0.025)
        expect = 1.0 if k >= n else _sps.beta.ppf(1 - 0.025, k + 1, n - k)
        assert hi == pytest.approx(expect, rel=1e-6, abs=1e-12)
        if k > 0:
            lo = clopper_pearson_lower(k, n, alpha=0.025)
            assert lo == pytest.approx(_sps.beta.ppf(0.025, k, n - k + 1),
                                       rel=1e-6, abs=1e-12)


@needs_scipy
def test_binom_test_matches_scipy():
    for k, n, p in [(9, 260_000, 222.9 / 260_000), (16, 260_000, 41.1 / 260_000),
                    (74, 260_000, 72.9 / 260_000), (3, 10, 0.5)]:
        got = binom_test_two_sided(k, n, p)
        expect = _sps.binomtest(k, n, p).pvalue
        assert got == pytest.approx(expect, rel=1e-6, abs=1e-12)


def test_wilson_interval_known_value():
    # Textbook check: 10 successes in 100 trials, 95% Wilson.
    lo, hi = wilson_interval(10, 100)
    assert lo == pytest.approx(0.0554, abs=5e-4)
    assert hi == pytest.approx(0.1739, abs=5e-4)


def test_wilson_degenerates_at_zero_but_exact_bound_does_not():
    """The reason the study does not use Wilson for its central null."""
    wlo, whi = wilson_interval(0, 260_000)
    assert wlo == 0.0
    assert whi < 1e-4
    assert clopper_pearson_upper(0, 260_000) > 0


def test_detectable_prevalence_inverts_the_power_equation():
    n = 260_000
    for power in (0.5, 0.8, 0.95):
        p = detectable_prevalence(n, power)
        assert 1 - (1 - p) ** n == pytest.approx(power)


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([("a", "a"), ("b", "b"), ("a", "a")]) == pytest.approx(1.0)
    # Complete disagreement between two equally-used labels gives kappa -1
    assert cohen_kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)


@needs_scipy
def test_cohen_kappa_matches_sklearn_if_available():
    pytest.importorskip("sklearn")
    from sklearn.metrics import cohen_kappa_score
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("c", "a"), ("b", "b"), ("c", "c")]
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    assert cohen_kappa(pairs) == pytest.approx(cohen_kappa_score(a, b))


def test_log_binom_pmf_sums_to_one():
    from statlib import log_binom_pmf
    total = sum(math.exp(log_binom_pmf(k, 20, 0.37)) for k in range(21))
    assert total == pytest.approx(1.0)
