"""Exact small-sample statistics, in pure Python.

No new dependency: the repository's requirements.txt does not carry scipy, and
every quantity needed here is either closed-form or a bisection over a binomial
tail computed with math.lgamma. `tests/test_webtext_provenance_statlib.py`
cross-verifies each function against scipy where scipy is installed, and skips
if it is not.

Normal approximations are deliberately avoided. The central results of this
study are zero-count observations, where the normal approximation is not merely
inaccurate but degenerate: it returns a zero-width interval around zero.
"""

import math


def log_binom_pmf(k, n, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.exp(log_binom_pmf(i, n, p)) for i in range(0, k + 1)))


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, 1.0 - binom_cdf(k - 1, n, p))


def clopper_pearson_upper(k, n, alpha=0.05):
    """One-sided exact (Clopper-Pearson) upper confidence bound on p.

    The largest p consistent with observing <= k successes at level alpha.
    For k = 0 this reduces to the closed form 1 - alpha**(1/n), the exact
    statement behind the 'rule of three' approximation 3/n.
    """
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_lower(k, n, alpha=0.05):
    """One-sided exact lower confidence bound on p."""
    if k <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if binom_sf(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_interval(k, n, alpha=0.05):
    """Two-sided exact interval, alpha split evenly between the tails."""
    return (clopper_pearson_lower(k, n, alpha / 2.0),
            clopper_pearson_upper(k, n, alpha / 2.0))


def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score interval. Reported alongside the exact interval because it
    is the standard choice for a proportion; it is NOT used for zero counts."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def detectable_prevalence(n, power=0.8):
    """Smallest per-document prevalence p at which a scan of n documents yields
    at least one hit with probability `power`.

    P(at least one) = 1 - (1-p)^n = power  =>  p = 1 - (1-power)^(1/n)
    """
    return 1.0 - (1.0 - power) ** (1.0 / n)


def binom_test_two_sided(k, n, p):
    """Exact two-sided binomial test by the method of small p-values: sum the
    probability of every outcome no more likely than the one observed."""
    if n == 0:
        return 1.0
    obs = log_binom_pmf(k, n, p)
    tol = 1e-7
    total = 0.0
    for i in range(0, n + 1):
        li = log_binom_pmf(i, n, p)
        if li <= obs + tol:
            total += math.exp(li)
        if i > k and li < obs - 50:  # far into the upper tail, terms vanish
            break
    return min(1.0, total)


def cohen_kappa(pairs):
    """Cohen's kappa for two raters over a list of (rater_a, rater_b) labels."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels = sorted({x for pair in pairs for x in pair})
    obs = sum(1 for a, b in pairs if a == b) / n
    ca = {lab: sum(1 for a, _ in pairs if a == lab) / n for lab in labels}
    cb = {lab: sum(1 for _, b in pairs if b == lab) / n for lab in labels}
    exp = sum(ca[lab] * cb[lab] for lab in labels)
    if exp >= 1.0:
        return float("nan")
    return (obs - exp) / (1 - exp)
