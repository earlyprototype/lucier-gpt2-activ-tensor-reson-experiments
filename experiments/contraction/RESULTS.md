# How fast does the state settle, and is the position collapse exact?

**Short answer: the collapse is exact to the limit of the stored precision, and cannot be pushed
further without a new run. The settling speed does not track model size — inside each family, the
two sizes disagree about which direction it goes.**

**Run:** 2026-07-28, analysis only — no forward passes, no model loaded.
**Status:** executed; scripts in this directory, all inputs already in the repository.
**Origin:** issue [#71](https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments/issues/71),
items **M1** and **M2**.

---

## In plain terms

Two questions were left open.

**M1.** ATR reports that all token positions end up holding the same internal state
(`position_similarity` → 1.0000). But every published figure is rounded to four decimal places, at
which 0.99994 and 1.000000 look identical. M1 asked for the unrounded number, because a hypothesis
in the project (**H-pos0**) assumes the collapse is *exact*, and 0.9999 would not be good enough.

**Answer: it is 1.000000000000 to twelve decimal places, in all eight committed runs.** The largest
shortfall is one part in 10¹³.

But there is a catch, and it is the real result. The tensors were saved in float32, a format that
can only distinguish numbers to about one part in 10⁷. The angle between positions comes out at
about 2×10⁻⁷ radians — which is float32's own resolution. So the measurement has hit the floor of the
format it was stored in. **The committed data says "exactly parallel, or parallel to one part in
10 million — cannot tell which."** That is enough for H-pos0's premise to survive, and not enough to
confirm it. Settling it needs a run recorded at higher precision, which is gated by
[`docs/ATR_PAUSE.md`](../../docs/ATR_PAUSE.md).

**M2.** How *fast* does the state settle? M2 asked for a decay curve across iterations.

**That exact curve cannot be drawn from committed data.** It needs the full state of every position
at every iteration, and no archive in this repository keeps that — the per-iteration files store the
*average* across positions, and the files that keep every position store only the final one. This is
a gap in what was saved, not a gap in what was run.

What *can* be measured is how fast the average state stops moving, and that is available for all four
models. It gives a clear and slightly awkward result:

| Model | Settles? | Half-life |
|---|---|---|
| Pythia-160M | yes, cleanly | ~0.8 iterations |
| GPT-2 Medium | yes, cleanly | ~1.3 iterations |
| GPT-2 Small | yes, but only after a delay | flat for ~30 iterations, then 1.9–3.0 |
| Pythia-410M | no, not within 60 iterations | — |

**Size does not predict this.** In the GPT-2 family the *larger* model settles faster and more
cleanly. In the Pythia family the *smaller* one does. Whatever governs settling speed, it is not
parameter count, and it is not shared across families — which is what the 2×2 was built to detect.

**A prediction that failed.** GPT-2 Small is the odd one out above, and there was an obvious
candidate explanation: [`experiments/sink_geometry/RESULTS.md`](../sink_geometry/RESULTS.md) already
found that a handful of oversized dimensions distort GPT-2 Small's similarity numbers specifically.
If those dimensions were also behind its ragged settling curve, deleting them should tidy it up.
**They are not.** Deleting the ten largest, or the fifty largest, barely moves it (goodness-of-fit
0.55 → 0.67, against 0.95+ for the models that settle cleanly). The two findings are unrelated. The
actual explanation is the delay: GPT-2 Small barely moves for the first thirty iterations and then
settles quickly, so no single rate describes it — see below.

---

## Terms used here

| Term | What it means |
|---|---|
| **position** | One token slot in the input. Each holds its own list of numbers. |
| **`position_similarity`** | How alike those lists are across positions, averaged over every pair. 1.0 means identical direction. Defined in `atr_engine.py`. |
| **float32 / float64** | Number formats. float32 stores about 7 significant digits, float64 about 16. The archives are float32. |
| **epsilon** | The smallest difference a format can represent. For float32, 1.19×10⁻⁷. |
| **contraction rate** | How fast successive states stop differing. Reported as the slope of log(difference) against iteration — a straight line means a constant rate. |
| **half-life** | Iterations needed to halve the remaining difference. |
| **R²** | Goodness of fit, 0 to 1. Near 1 means a single constant rate describes the process; low means it does not. |
| **H-pos0** | The hypothesis that position 0's forward map is autonomous, up to one scalar per position. Assumes exact position collapse. |

---

## M1 — the unrounded numbers

`experiments/contraction/01_position_collapse_precision.py`, reading
`output_confidence/converged_tensors.pt` and `output_divine_motion/state_*.pt`.

The engine's own metric is reproduced verbatim, then recomputed in float64 so that float32
accumulation order cannot be what produces or hides the answer.

| Run | Positions | Engine (float32) | float64 | 1 − similarity | Worst single pair |
|---|---|---|---|---|---|
| Lucier | 15 | 1.0000000000 | 1.000000000000 | 2.39e−14 | 1.000000000000 |
| Semantic | 12 | 1.0000000000 | 1.000000000000 | 1.60e−14 | 1.000000000000 |
| Syntactic | 10 | 1.0000000000 | 1.000000000000 | 7.88e−15 | 1.000000000000 |
| Nonsense | 11 | 1.0000000000 | 1.000000000000 | 1.79e−14 | 1.000000000000 |
| Imperative | 11 | 0.9999999404 | 1.000000000000 | 1.60e−14 | 1.000000000000 |
| Divine_Syntactic | 10 | 1.0000000000 | 1.000000000000 | 1.20e−14 | 1.000000000000 |
| Control_noise | 10 | 1.0000000000 | 1.000000000000 | 1.01e−13 | 1.000000000000 |
| Control_prolet_Semantic | 12 | 1.0000000000 | 1.000000000000 | 2.13e−14 | 1.000000000000 |

Two things worth noting in that table.

**The worst *single pair* is also 1.0, not just the average.** The metric is a mean over all pairs, so
a mean of 1.0 could in principle hide one dissenting position. It does not — the minimum over every
off-diagonal pair is 1.0 to twelve places in every run. There is no outlier position, and in
particular position 0 is not one.

**`Imperative` reads 0.9999999404 through the engine's float32 path and 1.000000000000 in float64.**
Same tensor, same formula. That single digit of apparent disagreement is accumulation order, not
physics — which is precisely why M1 needed doing rather than being read off the existing logs.

### The precision floor

| Run | Angle between positions | σ₂/σ₁ (rank-1 test) | Spread in the per-position scalar |
|---|---|---|---|
| Lucier | 2.18e−07 | 1.17e−07 | 1.75e−07 |
| Semantic | 1.79e−07 | 8.31e−08 | 7.03e−08 |
| Syntactic | 1.26e−07 | 3.92e−08 | 6.01e−08 |
| Nonsense | 1.89e−07 | 9.56e−08 | 1.41e−07 |
| Imperative | 1.79e−07 | 8.72e−08 | 9.92e−08 |
| Divine_Syntactic | 1.55e−07 | 6.09e−08 | 4.51e−08 |
| Control_noise | 4.49e−07 | 2.80e−07 | 5.50e−07 |
| Control_prolet_Semantic | 2.06e−07 | 1.11e−07 | 1.04e−07 |

float32 epsilon is **1.19e−07**. Every column above sits within a small factor of it. The converged
tensor is rank-1 to the same tolerance: σ₂/σ₁ ≈ 10⁻⁷ means the second singular direction carries
nothing float32 can represent.

**Consequence for H-pos0.** H-pos0 allows each position its own scalar *c_n*, so the prediction is a
tensor whose rows are one shared direction with differing lengths. What the data shows is stronger:
the lengths agree too, to 10⁻⁷ — the spread in *c_n* is at the same floor as everything else. At
convergence there is no scalar freedom left to observe; *c_n* = 1 for every *n*, as far as float32
can say. H-pos0's premise is not contradicted. It is also not confirmed, because a 10⁻⁷ effect and a
zero effect are the same number in this format.

**Recommendation.** Any future run intended to test H-pos0 should record `position_similarity` in
float64, and should archive the full per-position tensor at each snapshot rather than only its mean.
Neither is expensive: the divine_motion snapshots are 0.2 MB each because they discard exactly the
array that would have answered M2.

---

## M2 — what the archives can and cannot support

`experiments/contraction/02_contraction_rate.py`.

### Why the literal plot is not drawable

| Archive | Per iteration? | Per position? |
|---|---|---|
| `output_divine_motion/snapshots_*.pt` | yes, 25 snapshots to iteration 1000 | **no** — `last_vector` and `mean_vector` only, both 1-D |
| `output_divine_motion/state_*.pt` | no, final only | yes — `current_tensor` |
| `output_confidence/converged_tensors.pt` | no, converged only | yes |
| `sink_geometry/output/trajectories.pt` | yes, 61 iterations × 4 models | **no** — mean over positions |

`position_similarity` requires both columns to be yes. Nothing in the repository has that. The
measurement below is therefore of a related quantity — how fast the mean state stops moving — and is
labelled as such throughout.

### Two estimators, because the obvious one is biased

Measuring 1 − cos(vₜ, v_T) against the *last recorded state* is what M2 describes, but v_T is not a
proven fixed point; it is just where recording stopped. That forces the curve to zero at t = T by
construction, steepening the fitted slope near the end whatever the dynamics do. The primary number
here is instead the step-to-step difference 1 − cos(vₜ, vₜ₊₁), which references no endpoint at all.
Both are reported. Where they agree, the rate is real.

### Rates

From `sink_geometry/output/trajectories.pt`: 4 models × 5 prompts × 61 iterations.

| Model | Step-to-step slope | vs last iterate | R² (step) | Half-life |
|---|---|---|---|---|
| gpt2 | −0.12 to +0.008 | −0.06 to −0.03 | **0.03–0.55** | 5.6–11.4 (4 of 5 prompts) |
| gpt2-medium | −0.56 to −0.52 | −0.46 to −0.42 | 0.95–0.96 | 1.25–1.34 |
| pythia-160m | −0.94 to −0.50 | −0.86 to −0.47 | 0.99 | 0.74–1.39 |
| pythia-410m | −0.01 to +0.004 | −0.01 to −0.0005 | **0.00–0.11** | none |

Both estimators agree on the ordering and roughly on magnitude, so the ranking is not an artefact of
either. The two bolded R² columns are the models where a single rate does *not* describe the process,
and their slopes should not be quoted as rates.

One gpt2 prompt (index 4) is an outlier within its own model: its step-to-step slope is *positive*
(+0.008, R² 0.03) and its difference ends the run at 0.37 rather than 10⁻⁴, so it has not settled at
all in 60 iterations while the other four have. It is excluded from the half-life column above rather
than averaged into it. Whether it is a genuinely non-converging prompt or a slower instance of the
latency described below cannot be told from 60 iterations.

**pythia-410m does not settle within 60 iterations.** Three of its five prompts have a slope
indistinguishable from zero, and two are slightly positive. Its differences stay between 0.13 and
0.74 for the whole run, against 10⁻¹⁰ for the models that converge. This is not a slow version of the
same process; over this horizon it is not the same process.

### GPT-2 Small: a delay, not a slow rate

Splitting each run at its midpoint:

| Model | First half | Second half |
|---|---|---|
| gpt2 | −0.015 to −0.037, **R² 0.02–0.06** | −0.23 to −0.37, R² 0.77–0.91 (prompt 4: −0.007) |
| gpt2-medium | −0.77 to −0.84, R² 0.99 | −0.33 to −0.35, R² 1.00 |
| pythia-160m | −0.52 to −1.14, R² 0.95–0.99 | −0.54 to −0.85, R² 0.98–1.00 |
| pythia-410m | −0.010 to −0.031, R² 0.03–0.35 | −0.022 to +0.028, R² 0.00–0.26 |

GPT-2 Small is flat for the first thirty iterations — slope near zero, and an R² near zero saying
even that is not a trend — and then contracts at −0.3, which is the same order as GPT-2 Medium. It
is not slow. It has a **latency**, and the whole-run fit averages a plateau with a decay and
describes neither. GPT-2 Medium and Pythia-160M show ordinary mild deceleration: both halves fit
almost perfectly, the second a little shallower than the first.

### The failed prediction

`sink_geometry/RESULTS.md` established that ~10 oversized dimensions distort GPT-2 Small's cosine
measurements and not the other models'. GPT-2 Small is also the model with the ragged fit here.
Testable prediction: masking those dimensions should straighten it, and leave the others alone.

| Model | Unmasked R² | Mask 10 | Mask 50 |
|---|---|---|---|
| gpt2 | 0.03–0.55 | 0.03–0.65 | 0.09–0.67 |
| gpt2-medium | 0.95–0.96 | 0.96 | 0.96–0.97 |
| pythia-160m | 0.99 | 0.99 | 0.99 |
| pythia-410m | 0.00–0.11 | 0.00–0.15 | 0.00–0.16 |

**The prediction fails.** The second half of it holds — the other three models are unmoved by
masking, as `sink_geometry` found — but GPT-2 Small does not straighten. Removing its fifty largest
dimensions leaves it at 0.67, nowhere near the 0.95+ of models that fit a single rate. The oversized
dimensions are not the cause of the ragged fit. The latency above is.

### Cross-check on the long runs

The 1000-iteration GPT-2 Small runs, on a non-uniform schedule (0, 100, 250, 500, then every 10 from
800):

| Run | Snapshots | Step-to-step | vs last iterate |
|---|---|---|---|
| Divine_Syntactic | 25 | −0.035, R² 1.00 | −0.042, R² 0.99 |
| Control_noise | 25 | −0.009, R² 0.67 | −0.002, R² 0.11 |
| Control_prolet_Semantic | 25 | only 2 usable points — no fit | — |

These are consistent with GPT-2 Small being the slow-to-start model, but they are weak evidence: the
schedule puts only four points before iteration 800, so the phase that matters is barely sampled. A
note on this in `02_contraction_rate.py`: the x axis must be the recorded iteration, not the snapshot
index. Using the index treats 800 iterations as 4 steps and inflates the slope by two orders of
magnitude — an error made and corrected while preparing this analysis.

---

## What this does and does not settle

**Settled.** `position_similarity` at convergence is 1.0 to twelve decimal places in all eight
committed runs, including the worst individual pair, with no outlier position. The float32 discrepancy
in `Imperative` is accumulation order.

**Settled.** No committed archive can support M2 as worded. The per-position tensor was not saved per
iteration. Future runs should save it.

**Settled.** Contraction speed is not monotone in model size, and the two families order oppositely.
GPT-2 Small's ragged fit is a latency, not a slow rate, and is not caused by the massive-activation
effect documented in `sink_geometry`.

**Not settled.** Whether the position collapse is *exactly* exact. The archives are float32 and the
residual is at float32's floor. This is the one thing M1 was meant to decide and the data cannot
decide it.

**Not settled.** Whether pythia-410m converges at all given more iterations, or is a genuinely
different regime. 60 iterations is not enough to tell, and extending it is a new run.

**Gated.** Both of the above need forward passes and are blocked by
[`docs/ATR_PAUSE.md`](../../docs/ATR_PAUSE.md).
