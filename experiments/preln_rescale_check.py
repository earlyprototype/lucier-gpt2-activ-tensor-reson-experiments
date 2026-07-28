"""Is a pre-LN residual stack invariant to a global rescale of its input?

SCALING_ARTEFACT_ANALYSIS.md §1.1 concludes the ATR rescale is "approximately
inert for the forward map", reasoning that layer-0 LayerNorm is invariant to a
positive global rescale. The LayerNorm premise is true. The question here is
only whether the conclusion follows for the *block*, which also has a residual
path around that LayerNorm.

Pure Python, no dependencies. Random weights, not GPT-2's — so this settles the
STRUCTURE of the argument, not the magnitude of the effect in GPT-2.
"""
import math

D, NBLOCKS, EPS = 64, 12, 1e-5


def lcg(seed):
    s = seed
    while True:
        s = (1103515245 * s + 12345) % (2 ** 31)
        yield s / (2 ** 31) - 0.5


def matrix(rng, n, m, scale):
    return [[next(rng) * scale for _ in range(m)] for _ in range(n)]


def matvec(W, v):
    return [sum(wij * vj for wij, vj in zip(row, v)) for row in W]


def layernorm(v):
    mu = sum(v) / len(v)
    var = sum((x - mu) ** 2 for x in v) / len(v)
    denom = math.sqrt(var + EPS)
    return [(x - mu) / denom for x in v]


def gelu(v):
    return [0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))) for x in v]


rng = lcg(20260728)
BLOCKS = [(matrix(rng, D, D, 0.3), matrix(rng, D, D, 0.3)) for _ in range(NBLOCKS)]


def block(x, W1, W2):
    """Pre-LN block: x + W2 @ gelu(W1 @ LN(x)). The residual bypasses the LN."""
    inner = matvec(W2, gelu(matvec(W1, layernorm(x))))
    return [xi + gi for xi, gi in zip(x, inner)]


def forward(x, n=NBLOCKS):
    for i in range(n):
        x = block(x, *BLOCKS[i])
    return x


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


rng2 = lcg(99)
x = [next(rng2) for _ in range(D)]

print("Does LayerNorm ignore a global rescale?  (the premise -- expected: yes)")
for c in (2.0, 100.0):
    ln_a, ln_b = layernorm(x), layernorm([c * xi for xi in x])
    print(f"   c={c:<7g} max|LN(cx) - LN(x)| = {max(abs(p - q) for p, q in zip(ln_a, ln_b)):.3e}")

print()
print("Does the BLOCK ignore it?  (the conclusion -- one block)")
print(f"   {'c':>8}  {'cos(F(cx), F(x))':>18}  {'cos(F(cx), cF(x))':>19}")
for c in (1.001, 1.1, 2.0, 10.0, 100.0):
    fc = block([c * xi for xi in x], *BLOCKS[0])
    f1 = block(x, *BLOCKS[0])
    print(f"   {c:>8g}  {cosine(fc, f1):>18.6f}  {cosine(fc, [c * v for v in f1]):>19.6f}")

print()
print(f"Full {NBLOCKS}-block stack -- does the discrepancy compound with depth?")
print(f"   {'c':>8}  {'cos(F(cx), F(x))':>18}")
for c in (1.001, 1.1, 2.0, 10.0, 100.0):
    print(f"   {c:>8g}  {cosine(forward([c * xi for xi in x]), forward(x)):>18.6f}")

print()
print("Depth profile at c=10 (cosine of the two streams after each block):")
a, b = [10.0 * xi for xi in x], list(x)
for i in range(NBLOCKS):
    a, b = block(a, *BLOCKS[i]), block(b, *BLOCKS[i])
    print(f"   after block {i:>2}: cos = {cosine(a, b):.6f}")
