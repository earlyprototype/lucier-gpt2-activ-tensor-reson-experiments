# EXP: Readout Confidence Audit of Converged ATR States

*"Is the room singing, or is it rattling and the tuner is guessing?"*

**Date:** 2026-07-19. **Model:** GPT-2 Small (TransformerLens `from_pretrained` defaults). **Runner:** [`04_readout_confidence.py`](../04_readout_confidence.py). **Raw data:** [`confidence_results.json`](confidence_results.json), [`chordness.json`](chordness.json), converged tensors in [`converged_tensors.pt`](converged_tensors.pt).

## The Question

Every basin label ever assigned in this project came from an argmax: the single top token of the converged state's readout distribution. An argmax always names *something*, however unsure the distribution underneath it is. This experiment re-runs the original five-prompt piece (500 iterations, original schedule) plus 15 calibrated noise trials, and interrogates the **full softmax distribution** of each converged state instead of its winner: top-1 probability, top1-top2 logit margin, full-vocabulary entropy, effective support, and the semantic coherence of the top of the distribution.

## Result 0: Independent-Hardware Replication

Before the new measurements, a milestone in passing: this run was executed on entirely different hardware from all previous runs (a fresh cloud container, CPU, weights obtained from a legacy mirror). The terminal attractors are **identical** to the original: four prompts → `prolet`, the Syntactic prompt → `Divine`. The intermediate dissolution waypoints also reproduce (`Ag` at iteration 10, `Rousse` at iteration 50, `capit` en route). TECHNICAL.md listed cross-hardware reproduction as "not attempted"; it has now been attempted once, and it passed.

## Result 1: The Confidence Inversion

| State | Converges? (tensor) | Top-1 token | p(top-1) | Logit margin | Entropy (nats) | Effective support |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Lucier → | yes | `prolet` | 0.064 | 0.07 | 5.09 | ~163 tokens |
| Semantic → | yes | `prolet` | 0.086 | 0.27 | 5.07 | ~159 |
| Nonsense → | yes | `prolet` | 0.080 | 0.22 | 5.07 | ~160 |
| Imperative → | yes | `prolet` | 0.081 | 0.23 | 5.07 | ~159 |
| Syntactic → | **no** (never settles) | `Divine` | **0.505** | **2.07** | **3.05** | ~21 |

Reference: the model speaking normally (iteration-0 next-token distributions on these prompts) spans p(top-1) from 0.03 to 0.73 and entropy from 1.6 to 7.6 nats, so both regimes above are within the model's ordinary expressive range. Uniform entropy over the 50,257-token vocabulary would be 10.82 nats.

The inversion: **the tensors that settle read out diffusely; the tensor that never settles reads out confidently.** `prolet` wins its basins with 6-9% probability and a whisker-thin margin. `Divine` commands more than half the probability mass of a state that is still moving after 500 iterations. The `Divine` dissociation documented in FINDINGS is now double: a never-settling tensor with a readout that is not only *stable* but *loud*.

## Result 2: `prolet` Is Not a Note. It Is a Chord.

The full distribution under the `prolet` argmax (Semantic prompt shown; the other three are near-identical):

| Rank | Token | p | Rank | Token | p |
|:---:|:---|:---:|:---:|:---|:---:|
| 1 | `prolet` | .086 | 6 | `proletarian` | .036 |
| 2 | `bourgeois` | .066 | 7 | `socialist` | .021 |
| 3 | `Anarch` | .060 | 8 | `anarchist` | .020 |
| 4 | `comrade` | .044 | 9 | `congress` | .019 |
| 5 | `Marx` | .041 | 10 | `labour` | .018 |

Then: `anarchism`, `the`, `movement`, `Lenin`, `comrades`. The top of the distribution is **thematically saturated**: essentially every high-probability token belongs to one lexical field, revolutionary political vocabulary. The state is not weakly saying `prolet`; it is *strongly humming an entire chord*, of which `prolet` is merely the loudest partial.

Quantified as **chordness** (mean pairwise cosine similarity in embedding space, W_E, among the top-10 tokens; random-token baseline 0.266 ± computed over 50 draws):

| State | Chordness |
|:---|:---:|
| prolet basins (4 prompts) | **0.410 - 0.471** |
| Divine | 0.318 |
| Noise trials (n=15) | 0.267 - 0.313 (median ≈ 0.28), one outlier 0.511 |
| Random tokens | 0.266 |

Two further notes. First, `Anarch`, a *separate basin* in the 125-prompt sweep, is the #3 token inside the `prolet` distribution: the prolet and Anarch basins are two peaks of the same chord, consistent with their geometric proximity in the original convergence matrix. Second, the entropy trace shows the distribution itself is the fixed point: from iteration 100 to 500, entropy is pinned at 5.07-5.09 nats while the argmax never wavers. The chord, not the winner, is what converged.

## Result 3: `Divine` Is a Solo

The distribution under `Divine` (p = 0.505) is a different kind of object. Its runners-up: `【`, `Fairy`, `Falling`, `……`, `―`, `「`, `Elements`, `Darkness`, `Cancel`, `Yu`, `Holy`. No lexical field; fantasy-adjacent fragments and CJK typography debris, and chordness barely above random (0.318 vs 0.266). So the two attractor families are opposites on both axes:

| | p(top-1) | Coherence under the winner |
|:---|:---:|:---:|
| `prolet` basins | low (0.06-0.09) | high (a chord) |
| `Divine` | high (0.51) | low (a solo over noise) |

## Result 4: Noise Rattles

The 15 calibrated noise trials (Gaussian tensors, norm-matched, seed 42) reproduce the original null-model picture and add the confidence dimension. They converge into the familiar non-semantic attractors (`―` seven times, `ei`, `vertex`, `trader`, `instant`, `Hindu`) with confidence spanning the entire range (p from 0.02 to 0.73), but with chordness at random-baseline levels in 13 of 15 trials. Confidence alone does **not** separate language-driven attractors from noise attractors; **coherence does**, almost cleanly.

The honest exceptions, reported because they are interesting: trial 11 fell into a genuinely coherent chord (`Hindu`, `Bombay`, `Hindus`, `Shiv`, `Brah`, `Kashmir`, `Gujarat`; chordness 0.511), and trial 10 into a philosophy-flavored one (`instant`, `relat`, `Rousse`, `justified`, `judgment`, `calculus`), containing `Rousse`, a waypoint on the language prompts' dissolution path. Noise *can* stumble into semantic wells; it just rarely does, while language always did.

## Interpretation

The question this experiment was built to answer ("singing or rattling?") turns out to have a third answer the argmax could never have revealed:

1. **The `prolet` basin sings a chord.** The argmax under-sold the original finding. At the distribution level, the basin is *more* semantically coherent than the single-token story suggested: the whole top of the distribution is one theme. The claim "four of five basins are semantically coherent," previously established via the W_E neighbourhood of the winning token, now holds one level deeper, in the readout distribution itself.
2. **`Divine` is a genuinely different phenomenon,** not just a non-converging nuisance: a loud, pure, incoherent-residue tone over a never-settling tensor. Whatever the `Divine` state is, it is not a quieter version of the `prolet` state.
3. **Noise mostly rattles:** every confidence level, near-random coherence. The tuner names its attractors, but there is no chord under the name, with rare exceptions where noise finds a real semantic well.

For the J-space bridge (docs/JSPACE_PRIMER.md, Part 6): the chord structure sharpens margin question 1 considerably. A state whose readout distribution is a coherent lexical field is exactly what a workspace-like, verbalizable state should look like, whereas a loud incoherent solo is a good candidate for a state *outside* the verbalizable subspace whose readout is a projection artifact. The J-space membership test now has a concrete prediction to check: **prolet inside, Divine outside.**

## Caveats

Single run per condition on one machine (though that machine differs from all prior runs, giving the project its first cross-hardware replication). Noise n=15 versus the original 125. Chordness uses top-10 tokens and W_E cosine only; no permutation test has been run on these specific scores. The weights were loaded from a legacy mirror of the standard `gpt2` checkpoint; identical terminal attractors and waypoints argue they are the same weights.
