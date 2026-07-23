# Permutation Test: the "All-Warm" Cross-Similarity Matrix

**Date:** 2026-07-11 · **Script:** `../02b_permutation_test.py` · **Seed:** 1969 · **N:** 10,000

## Question (pre-registered)

Session 02 observed that the 14×14 pairwise cosine matrix of the 5 basin + 9
waypoint tokens in GPT-2 Small's W_E is positive in all 91 off-diagonal pairs
(range 0.18–0.47), and read this as a compact "thematic-centre-of-mass"
subspace. The pre-registered test (RESEARCH_NOTE §6 / FINDINGS caveat 4): how
often do random 14-token sets reproduce this?

## Reproduction gate

Observed matrix recomputed from canonical token IDs (Session 01 §8):
**91/91 positive, min 0.181, mean 0.288, max 0.470**, matches the recorded
0.18–0.47. Measurement confirmed.

## Result

| Statistic | Observed | Random sets meeting/exceeding it | p |
|---|---|---|---|
| S1: all 91 pairs positive | true | **9,994 / 10,000** | ≈ 1.0 |
| S2: min pair ≥ 0.181 | 0.181 | 1,674 / 10,000 | 0.167 |
| S3: mean ≥ 0.288 | 0.288 | 989 / 10,000 | 0.099 |

Context: the **global mean pairwise cosine of GPT-2 Small's embedding space is
0.268** (200k random pairs): the space is strongly anisotropic (a shared mean
direction makes almost all token pairs "warm"; cf. Ethayarajh 2019 on
anisotropy in LM embedding spaces). The observed set's mean (0.288) sits just
above the global average; its minimum is near the random-set median (0.158).

## Verdict

**The all-warm property is an anisotropy artifact, not evidence of a special
compact subspace.** Virtually any 14 tokens in this embedding space are
all-warm. The "thematic-centre-of-mass subspace" interpretation of the
cross-similarity matrix is withdrawn.

**Scope:** this test addresses the all-warm matrix only. The semantic
neighbourhood observation (basin tokens' nearest neighbours: `prolet` →
bourgeoisie/capitalists, `Divine` → Sacred/God, etc.) is a separate, local,
qualitative claim and is neither tested nor withdrawn here. A quantitative
version of that claim would need labelled neighbourhoods and its own null.
