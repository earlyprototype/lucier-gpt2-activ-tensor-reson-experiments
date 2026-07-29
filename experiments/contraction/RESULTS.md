# How fast does the state settle, and is the position collapse exact?

**Summary:** coordinate by coordinate, positions agree to about two units of float32 precision, the
scale of float32 arithmetic noise. This bounds any structured disagreement to below that scale and
does not establish its absence. Settling speed does not track model size: within each family the two
sizes order oppositely. The loop's per-step rescale factor, which H-pos0 states is 1 at a settled
state, is measured at 0.27–0.29 in two runs and 0.099 in a third.

**Run:** 2026-07-28, analysis only — no forward passes, no model loaded.
**Status:** executed; scripts in this directory, all inputs already in the repository.
**Origin:** issue [#71](https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments/issues/71),
items **M1**, **M2** and **M5**.

---

## In plain terms

**M1.** ATR reports that all token positions end up holding the same internal state
(`position_similarity` → 1.0000). Published figures are rounded to four decimal places, at which
0.99994 and 1.000000 are indistinguishable. M1 asked for the unrounded number, because **H-pos0**
assumes the collapse is exact.

**Unrounded, it is 1.000000000000 to twelve decimal places in all eight committed runs.** The largest
shortfall is one part in 10¹³.

The tensors were saved in float32, which distinguishes numbers to about one part in 10⁷, and the angle
between positions is about 2×10⁻⁷ radians. The raw figures alone therefore do not separate "exactly
parallel" from "parallel to one part in 10 million".

Coordinate by coordinate, the typical coordinate of one position agrees with the same coordinate of
another to about **two units of float32 precision**, and stays there over 1000 iterations. This is the
scale at which float32 arithmetic operates.

**Scope.** The measurement bounds a structured disagreement between positions to below the arithmetic
scale. It does not establish that none exists below that scale; a float32 measurement cannot. A
higher-precision run is required to determine it.

**M2.** M2 asked for a decay curve across iterations.

**That curve cannot be drawn from committed data.** It requires the full state of every position at
every iteration. The per-iteration files store the average across positions; the files that store
every position store only the final iteration. This is a gap in what was saved, not in what was run.

What is measurable is the rate at which the average state stops moving, for all four models:

| Model | Settles within 60 iterations? | Half-life |
|---|---|---|
| Pythia-160M | yes | ~0.8 iterations |
| GPT-2 Medium | yes | ~1.3 iterations |
| GPT-2 Small | after a delay | flat ~30 iterations, then 1.9–3.0 |
| Pythia-410M | no | — |

**Size does not predict this.** In the GPT-2 family the larger model settles faster; in the Pythia
family the smaller one does.

**A tested prediction, which failed.**
[`experiments/sink_geometry/RESULTS.md`](../sink_geometry/RESULTS.md) found that a handful of
oversized dimensions distort GPT-2 Small's similarity numbers specifically. GPT-2 Small is also the
model with the poorest single-rate fit here. Prediction: masking those dimensions straightens the fit.
**Result: it does not.** Masking the ten or fifty largest moves goodness-of-fit from 0.55 to 0.67,
against 0.95+ for the models that fit a single rate. The measured cause is the delay: GPT-2 Small is
flat for the first thirty iterations and then contracts, so no single rate describes the whole run.

**M5.** M5 asks for the per-step rescale factor and is filed as gated on the grounds that it is never
recorded. It is derivable from what is recorded. Measured, the loop rescales the state to **29%** of
its size per round in one run and **10%** in another, stable across iterations. H-pos0 states this
figure is 1 at a settled state. Because the map is not scale-invariant over this range (#69), the
scalar does not drop out of the fixed-point condition.

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
| ***c* / rescale factor** | The loop resizes the state before each step so its total size returns to the seed's. *c* is that resize: seed size ÷ current size. *c* = 1 would mean no resizing happens. |

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

Two features of that table.

**The worst *single pair* is also 1.0, not just the average.** The metric is a mean over all pairs, so
a mean of 1.0 could in principle hide one dissenting position. It does not — the minimum over every
off-diagonal pair is 1.0 to twelve places in every run. There is no outlier position, and in
particular position 0 is not one.

**`Imperative` reads 0.9999999404 through the engine's float32 path and 1.000000000000 in float64.**
Same tensor, same formula. The difference is float32 accumulation order, not a property of the
state. Recomputing was therefore necessary; the existing logs alone would not distinguish the two.

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

### Size of the residual

Two further measurements were taken to characterise the residual.

The first is a scale: **d = √(1 − position_similarity)**. The geometry is exact — 1 − cos = θ²/2, so
*d* is the angle up to a factor of √2 — and it is expressed in units of float32 epsilon below. **It is
an RMS summary.** Reading *d* as a per-component relative error requires the disagreement to be spread
evenly and independently across coordinates, which it is not (measured below). The table is an order
of magnitude, not a per-number statement:

| Run | 1 − similarity | *d* | ***d* / ε₃₂** |
|---|---|---|---|
| Lucier | 2.39e−14 | 1.54e−07 | **1.30** |
| Semantic | 1.60e−14 | 1.26e−07 | **1.06** |
| Syntactic | 7.88e−15 | 8.88e−08 | **0.74** |
| Nonsense | 1.79e−14 | 1.34e−07 | **1.12** |
| Imperative | 1.60e−14 | 1.26e−07 | **1.06** |
| Divine_Syntactic | 1.20e−14 | 1.10e−07 | **0.92** |
| Control_noise | 1.01e−13 | 3.17e−07 | **2.66** |
| Control_prolet_Semantic | 2.13e−14 | 1.46e−07 | **1.22** |

float32 ε = 1.192e−07.

**But *d* is an RMS summary**, and reading it as "every component agrees to ~1 ε" assumes the
disagreement is spread evenly across coordinates. It is not — one coordinate carries 4–55% of it, and
the effective number of participating coordinates is 3–23 out of 768. That assumption was flagged in
review and it does not hold, so the per-coordinate question has to be asked per coordinate.

### Per-coordinate measurement

Concentration on its own proves nothing: under *relative* rounding every coordinate carries
|Δu_k|/|u_k| ~ ε regardless of its size, and this state's energy is 91% in ten coordinates, so
rounding would look concentrated too. What separates rounding from structure is whether the *relative*
disagreement is flat at a few ε. Scale is divided out first — H-pos0 lets each position keep its own
scalar (M5), so the direction is what is at issue.

| Run | median | p90 | p99 | >100 ε | their \|u\| | their share of angle |
|---|---|---|---|---|---|---|
| Lucier | 1.90 | 10.03 | 63.67 | 0.45% | 0.026 | 0.35% |
| Semantic | 1.82 | 9.39 | 71.55 | 0.79% | 0.007 | 0.47% |
| Syntactic | 1.69 | 9.51 | 182.02 | 1.41% | 0.003 | 1.48% |
| Nonsense | 1.87 | 9.77 | 90.63 | 0.92% | 0.012 | 0.63% |
| Imperative | 1.80 | 9.40 | 83.86 | 0.88% | 0.016 | 0.78% |
| Divine_Syntactic | 1.85 | 10.24 | 190.03 | 1.56% | 0.003 | 1.32% |
| Control_noise | 3.38 | 15.17 | 209.01 | 1.81% | 0.010 | 0.43% |
| Control_prolet_Semantic | 1.81 | 9.61 | 77.40 | 0.79% | 0.007 | 0.36% |

**The typical coordinate agrees to about 2 ε.** The p99 column is driven by coordinates at 0.3–2.6%
of typical magnitude, where a negligible absolute difference produces a large relative one; those
coordinates carry ~1% of the angle. In the coordinates carrying the angle, agreement is at the few-ε
level.

**M1 conclusion.** The residual between positions is the size of float32 arithmetic noise. This bounds
a structured disagreement to below that scale, and it holds steady over 1000 iterations. It does not
establish that no structure exists below the arithmetic scale; a float32 measurement cannot. A
higher-precision run is required.

**`Control_noise` is the outlier on all three measures**, which are distinct quantities:

| Measure | `Control_noise` | Other seven |
|---|---|---|
| RMS scale *d* | 2.66 ε | 0.74–1.30 ε |
| Per-coordinate median | 3.38 ε | 1.69–1.90 ε |
| Raw deviation 1 − sim | 1.01e−13 | 7.9e−15 – 2.4e−14 |

Roughly a factor of two on either per-number measure — the 9× in the raw deviation is the same fact
seen through a square, not a separate finding. It is also the noise control, and the run with by far
the largest rescale factor (10.1× amplification against 3.5×, see M5). Whether those are connected is
not answerable from one run, but it is the one place here where something might sit above the floor.

<details>
<summary>Secondary cross-check: a synthetic sensitivity sweep, and what it is not</summary>

The sweep perturbs an exactly-collapsed tensor by relative Gaussian noise of *k* × ε and reports the
resulting deviation. **It is not a float32 round-to-nearest baseline**, for two reasons:

- Gaussian noise of standard deviation ε is not round-to-nearest, whose relative error is bounded by
  ε/2 and is roughly uniform — standard deviation about ε/(2√3), some 3.5× smaller.
- A single rounding is not the applicable comparison: the archived state is the output of a
  twelve-layer forward pass and carries accumulated error from many roundings.

The *k* column matching an observation therefore does not give a ULP count. The sweep is reported for
its slope: a factor of 4 in *k* moves the deviation ~15×, which bounds the sensitivity of the
conclusion to the assumed error scale. The per-coordinate measurement above is the primary evidence
and does not depend on it.

| Run | Observed | k = 1 | k = 4 | k = 16 | k = 64 |
|---|---|---|---|---|---|
| Lucier | 2.39e−14 | 1.41e−14 | 2.15e−13 | 3.16e−12 | 6.04e−11 |
| Semantic | 1.60e−14 | 1.44e−14 | 2.14e−13 | 3.12e−12 | 5.28e−11 |
| Syntactic | 7.88e−15 | 1.23e−14 | 2.36e−13 | 3.29e−12 | 5.32e−11 |
| Nonsense | 1.79e−14 | 1.39e−14 | 2.20e−13 | 3.09e−12 | 5.12e−11 |
| Imperative | 1.60e−14 | 1.40e−14 | 2.20e−13 | 3.09e−12 | 5.12e−11 |
| Divine_Syntactic | 1.20e−14 | 1.23e−14 | 2.36e−13 | 3.28e−12 | 5.32e−11 |
| Control_noise | 1.01e−13 | 1.12e−14 | 1.85e−13 | 3.08e−12 | 5.13e−11 |
| Control_prolet_Semantic | 2.13e−14 | 1.47e−14 | 2.14e−13 | 3.12e−12 | 5.28e−11 |

One further null was tested and rejected: constructing a rank-1 tensor and rounding it to float32.
Because the per-position norms agree to ~10⁻⁷, every row rounds to nearly the same float32 vector and
two runs come out bit-identical at deviation exactly 0. That measures storage rounding, which is not
the source of the observed deviation.

</details>

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

### Two estimators

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
800). **After the gap-width guard below, this cross-check supports two of six fits.**

| Run | Snaps | Gap used | Dropped | Step-to-step | vs last iterate |
|---|---|---|---|---|---|
| Divine_Syntactic | 25 | 100 | 3 | no fit | −0.042, R² 0.99 |
| Control_noise | 25 | 10 | 4 | −0.028, R² 0.76 | −0.002, R² 0.11 |
| Control_prolet_Semantic | 25 | 100 | 1 | no fit | no fit |

The schedule mixes gaps of 10, 100, 150, 250 and 300. That breaks the step estimator in a way the
iteration-vs-index fix does **not** repair: `1 − cos(vₜ, vₜ₊D)` carries a factor `(1 − e^(−λD))²`, so a
100-iteration gap and a 10-iteration gap are not the same quantity. Putting the recorded iteration on
the x axis fixes the *spacing* of the points but not *what is being plotted*. The effect is visible
directly in `Control_noise`: 1.97e−1 at a 300-gap, then 2.1e−2 at the first 10-gap — a drop that is
mostly the gap change, not the dynamics.

`step_points()` now keeps a single gap width and reports how many usable points that discards. Once
that guard is applied, two of the three runs have no usable step fit at all, and `Control_noise`
covers only its 10-step tail.

Superseded figures: an earlier revision reported step slopes of −0.035 and −0.009 here, fitted across
mixed gaps. The corrected `Control_noise` figure is −0.028, three times the biased value; the other
two have no usable step fit. The GPT-2 Small delay finding rests on the 61-iteration trajectories,
which are uniformly spaced at D = 1 and are unaffected by this correction.

Two defects in the step estimator were found and corrected during this analysis: regressing against
snapshot index rather than recorded iteration (inflating the slope by two orders of magnitude), and
fitting across unequal gaps (the bias above).

---

## M5 — the rescale factor

`experiments/contraction/03_rescale_factor.py`.

#71 files **M5** — record the per-iteration rescale factor *c_n* — as `GATED` `EXPERIMENT`, on the
grounds that the ratio is "currently never recorded" and needs one line adding to the loop. **It is
partly recoverable from committed data already**, because of the order of operations in the loop
(`atr_engine.py:211-216`):

```python
for i in 1..max_iter:
    current_norm = ||current_tensor||          # PRE-rescale
    current_tensor *= initial_norm / current_norm
    ... forward pass ...
    current_tensor = new state                 # NOT rescaled
    ... snapshot recorded here ...
```

Every recorded state is post-forward and **pre-rescale**. So a snapshot's norm is exactly the
denominator of *c* for the next iteration: *c*ₙ₊₁ = `initial_norm` / ‖xₙ‖. The archives dropped
`tensor_norm` but kept `last_norm`, and once positions have collapsed ‖x‖ = √(seq_len) · ‖any
position‖. That identity is verified against the state files (which record both) before anything
uses it — it holds to 2–6 × 10⁻⁷. `initial_norm` is recorded directly.

### The measurement

| Run | seq | initial_norm | settled ‖x‖ | ***c*** | amplification |
|---|---|---|---|---|---|
| Divine_Syntactic | 10 | 1468.49 | 5098.14 | **0.2880** | 3.47× |
| Control_noise | 10 | 397.18 | 4017.69 | **0.0989** | 10.12× |
| Control_prolet_Semantic | 12 | 1392.65 | 5230.65 | **0.2662** | 3.76× |

*c* settles fast and then holds: from *c*₁₀₁ to *c*₁₀₀₁ the spread is 2×10⁻⁴ to 6×10⁻³. It is a
**stable constant**. (Index convention, easy to get wrong: a snapshot recorded at iteration *n* holds
the state *after* *n* passes and *before* the rescale that precedes pass *n*+1, so it yields *c*ₙ₊₁.)

### Relation to H-pos0

#75 states the H-pos0 argument as:

> At a settled, position-uniform state ‖xⁿ‖ is constant, so ***c_n* = 1**, and the shared vector must
> satisfy ***x\* = F₀(x\*)***

‖xⁿ‖ is constant at settlement, as stated. The measured *c* at settlement is not 1: the rescale
target is the *initial* norm and the settled norm is 3.5–10× that, giving *c* = 0.288, 0.099, 0.266 in
the three committed runs.

The fixed-point condition is therefore not *x\** = *F₀*(*x\**) but

> *x\** = *c* · *x\** + *g₀*(LN(*x\**))   with *c* ≈ 0.29 measured

**The scalar does not drop out**, because the map is not scale-invariant over this range. #69 measured
cos(*F*(*c*·*x*), *F*(*x*)) at 0.936 for *c* = 2 and 0.505 for *c* = 10; the measured amplification here
is 3.5–10×.

**Unaffected:** H-pos0's structure. Position 0's trajectory remains autonomous up to one scalar, and
an *n* = 1 run implements the same rescaled map. The intermediate claim and the fixed-point form
require rewording.

**Consequence for Control A, untested.** #75 predicts the *n* = 1 run converges to the same terminal
basin, treating a differing *c* as a trajectory-level difference. If the fixed point depends on *c*,
and *c* is fixed by the seed's initial norm — which differs between an *n* = 1 run and the sweep — the
terminal states would also differ. This follows only under LN(*c*·*x*) ≈ LN(*x*), which #69 reports as
holding to ~10⁻⁵ at the LayerNorm and not around the residual path. Stated as a consequence to test,
not a result.

**Coverage:** three runs, one model, one prompt family. *c* differs between the two content prompts
(0.288, 0.266) and the noise control (0.099); the noise control also converges elsewhere. Whether *c*
tracks basin identity is not determinable from three runs.

### Correction: no engine change required; M5 gated in error

An earlier revision of this file stated the transient "still needs the one-line change" to the engine.
That statement is incorrect, as is M5's premise. `atr_engine.py` records all of the fields M5 requires,
at every snapshot:

```python
"tensor":              current_tensor.clone().cpu(),   # line 260
"tensor_norm":         current_tensor.norm().item(),   # line 265
"position_similarity": position_similarity,            # line 274
```

The iteration-0 snapshot carries the same fields, so `initial_norm` = `snapshots[0]["tensor_norm"]`
and *c*ₙ₊₁ = `snapshots[0].tensor_norm / snapshots[n].tensor_norm`, available from any engine run whose
snapshots were saved intact. M5's "currently never recorded" does not hold for the engine.

**The loss occurred at save time, in two scripts independently:**

| Script | What it did |
|---|---|
| `experiments/gpt2_small/05_divine_motion.py:118` | `make_snapshot()` — *"Slim snapshot with just the fields the analysis needs."* Reimplements the snapshot from scratch and keeps 7 of the engine's 20 fields. Drops `tensor`, `tensor_norm`, `position_similarity`. |
| `experiments/sink_geometry/02_masking_control.py:86-88` | Calls `run_atr_loop`, receives the full snapshots, keeps `means` and discards the rest before `torch.save`. |

Both selected fields at save time. The fields this analysis required are among those dropped, and
neither loss is recoverable without re-running.

The residual limits below are therefore limits of **these two archives**, not of the engine or method:

- Iterations 1–99, which `05_divine_motion.py`'s schedule does not sample — so the *approach* of *c_n*
  to its settled value is unobserved, and only the settled value is in hand.
- The pre-collapse regime, where reconstructing ‖x‖ from `last_norm` is not exact. The iteration-0
  values are printed marked with `*` and are not used.
- The cross-model runs, which archived no per-position data at all.

**M5 status:** not gated. It requires neither a forward pass nor an engine change, only archives
retaining the engine's snapshot fields.

---

## What this does and does not settle

**Determined.** `position_similarity` at convergence is 1.0 to twelve decimal places in all eight
committed runs, including the worst individual pair, with no outlier position. The float32 discrepancy
in `Imperative` is accumulation order.

**Determined.** No committed archive supports M2 as worded. The cause is not the engine: the engine
records the full per-position tensor, `tensor_norm` and `position_similarity` at every snapshot, and
two experiment scripts discarded them at save time. See the M5 section.

**Determined.** Contraction speed is not monotone in model size, and the two families order
oppositely. GPT-2 Small's poor single-rate fit is a latency rather than a slow rate, and is not caused
by the massive-activation effect documented in `sink_geometry`.

**Determined.** The rescale factor *c* settles to a stable constant that is not 1: 0.288, 0.099, 0.266
across three committed runs. H-pos0 (#75) states this figure is 1 at a settled state. Because the map
is not scale-invariant over the measured range (#69), the scalar does not drop out of the fixed-point
condition. M5 required neither a forward pass nor an engine change.

**Bounded, not determined.** Whether the position collapse is exactly exact. The typical coordinate
agrees to ~2 float32 ε, the scale of arithmetic noise, which bounds a structured disagreement to below
that scale. A float32 measurement cannot establish the absence of structure below float32's own noise.
A higher-precision run is required.

**Open.** `Control_noise` is the outlier on all three measures: RMS scale *d* = 2.66 ε against
0.74–1.30 for the other seven, per-coordinate median 3.38 ε against 1.69–1.90, raw deviation 1.01e−13
against 7.9e−15–2.4e−14. Roughly a factor of two on each per-number measure. It is also the noise
control and the run with the largest rescale factor. One run; not determinable here.

**Open.** Whether pythia-410m converges given more iterations, or is a different regime. 60 iterations
is insufficient; extending it requires a new run.

**Open.** How *c_n* reaches its constant. No snapshot samples iterations 1–99 in the one archive
retaining enough to compute it. The engine records the quantity; the saving script did not retain it.

**Gated.** Extending pythia-410m and any float64 confirmation require forward passes and are blocked
by [`docs/ATR_PAUSE.md`](../../docs/ATR_PAUSE.md). No other item here is.
