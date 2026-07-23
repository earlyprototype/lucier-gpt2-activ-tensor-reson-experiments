# EXP: Divine Motion Audit (Issue #7): Where Does the Never-Settling Tensor Go?

*"The tensor dances; the shadow stands still." Does it?*

**Date:** 2026-07-19. **Model:** GPT-2 Small (TransformerLens, weights loaded offline via `ATR_GPT2_LOCAL`). **Runner:** [`05_divine_motion.py`](../05_divine_motion.py). **Raw data:** [`divine_motion_results.json`](divine_motion_results.json) (lag-10 schedule), [`probe_lag1_results.json`](probe_lag1_results.json) (lag-1 probe), [`divine_motion_tables.md`](divine_motion_tables.md), snapshot checkpoints in `snapshots_*.pt`, loop states at iteration 1000 in `state_*.pt`.

## The Question

The readout confidence audit ([`../output_confidence/confidence_report.md`](../output_confidence/confidence_report.md)) left one sharp anomaly: the Syntactic prompt's `Divine` state never passes the convergence gate, yet decodes to one token at p = 0.505. Hypothesis H-D1 (issue #7): the late-stage motion lies mostly in directions the readout map (`ln_final -> W_U`) flattens away.

Design: run the Syntactic prompt ("The cat sat on the mat and then the") to 1000 iterations with snapshots at [0, 100, 250, 500] plus every 10 iterations from 800 to 1000. Controls: the Semantic prompt ("The Eiffel Tower is located in the city of", a `prolet` basin) and one calibrated noise tensor (Gaussian, seq_len 10, norm 397.18, `torch.manual_seed(42)`, the trial-0 configuration of the noise baseline). For each successive late snapshot pair, measure at the last token position: tensor motion (cosine, L2), full-readout motion (KL, total variation, p(top1), entropy), and an **invisibility ratio**: the norm of the delta's actual effect on the full logit vector, divided by the mean effect of 20 random directions of equal norm. A ratio well below 1 means the motion is preferentially readout-invisible.

## Result 1: At Lag 10, Everything Is Frozen (Including Divine)

The headline tables (full versions for all three runs in [`divine_motion_tables.md`](divine_motion_tables.md); all numbers are last-token position, KL is KL(p_new || p_old) in nats).

`Divine_Syntactic`, representative rows of 20:

| Pair | Tensor cos | Tensor L2 | Readout KL (nats) | Readout TV | p(top1) | Entropy (nats) | Invisibility ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 800-810 | 1.000000 | 0.0004 | 4.0e-08 | 8.0e-07 | 0.5046 | 3.0500 | 1.065 |
| 900-910 | 1.000000 | 0.0004 | 2.9e-08 | 9.9e-07 | 0.5046 | 3.0500 | 0.973 |
| 990-1000 | 1.000000 | 0.0004 | 2.4e-08 | 7.1e-07 | 0.5046 | 3.0500 | 0.971 |

Late-band summary (800 to 1000): path length 0.008, net displacement 0.002, total TV 1.8e-05. `Control_prolet_Semantic` is the same story at the same magnitudes (path length 0.008, p(top1) pinned at 0.0858, entropy at 5.0674). `Control_noise` is the only run that moves at lag 10: tensor L2 per pair 24 to 258, per-pair KL up to 0.55, TV up to 0.40, p(top1) climbing from 0.03 to 0.64 as it slowly falls toward the horizontal-bar attractor `―` (U+2015), with invisibility ratio 1.03 to 1.40 (mean 1.20).

Taken at face value this would say: by iteration 800 the Divine tensor is a fixed point to four decimal places, and H-D1 is moot. That reading is wrong, and the trace says why: **the consecutive-iteration cosine at every late snapshot sits at 0.6849**, unchanged from iteration 250, while snapshots 10 apart are identical. A tensor cannot move that much every iteration and be back where it started every 10 iterations unless it is periodic with a period dividing 10. The lag-10 schedule was sampling the orbit phase-locked.

## Result 2: Divine Is a Period-2 Limit Cycle

A lag-1 probe (20 extra iterations from the saved state at iteration 1000, snapshotting every iteration) settles it. L2 distance from the base state at iteration 1000:

| Lag | 1 | 2 | 3 | 4 | 5 | ... | 19 | 20 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| L2 from base | 1249.43 | 0.000 | 1249.43 | 0.001 | 1249.43 | ... | 1249.43 | 0.001 |
| cos with base | 0.6849 | 1.0000 | 0.6849 | 1.0000 | 0.6849 | ... | 0.6849 | 1.0000 |

The Divine state is not a wandering orbit and not a fixed point. It is an **exact period-2 limit cycle**: the tensor alternates between two states A and B separated by L2 1249 (against a last-vector norm of 1612; cosine 0.685 between them), reproduced to machine precision every two iterations. The mean-pooled vector oscillates with the same amplitude, so the whole tensor flips, not just the last position. "Never settles" was true at the resolution of consecutive-iteration cosine, but the object it named is a two-state oscillator, locked in since at least iteration 800 (and essentially since 250).

## Result 3: The Cycle's Motion Is 3.4x Readout-Suppressed, and the Readout Breathes

Lag-1 measurements over the 20 probe steps (each step is the full A to B or B to A swing):

| Transition | Tensor cos | Tensor L2 | Readout KL (nats) | Readout TV | p(top1) | Entropy (nats) | Invisibility ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A to B (e.g. 1000 to 1001) | 0.6849 | 1249.43 | 0.273 | 0.304 | 0.2252 | 4.6175 | 0.300 |
| B to A (e.g. 1001 to 1002) | 0.6849 | 1249.43 | 0.246 | 0.304 | 0.5046 | 3.0500 | 0.292 |

Mean invisibility ratio 0.295 (sd 0.003, n = 20). Concretely: the A-B delta moves the full logit vector by norm 198, while equal-norm random directions move it by 662 on average (sd 30). Per unit of tensor motion, the readout responds at **29.5 percent of the random-direction baseline**; in squared (energy) terms, about 91 percent of the expected logit response to a displacement of this size is absent. The motion is strongly, though not totally, concentrated in readout-flattened directions.

What the readout does see is a two-phase breathing, invisible to every previous schedule because, from lock-in onward, every recorded snapshot fell on an even iteration: phase A reads `Divine` at p = 0.5046, entropy 3.05; phase B reads `Divine` at p = 0.2252, entropy 4.62. KL between phases is about 0.25 nats and total variation 0.304, per half-cycle. **The argmax is `Divine` in both phases.** The p = 0.505 single token reported by the confidence audit is real but it is one phase of the pair; the other phase is a lower-probability, broader `Divine`.

## Result 4: Controls

| Run | Late tensor motion (lag 1) | Late readout motion | Invisibility ratio |
|:---|:---|:---|:---:|
| Divine (Syntactic) | period-2 cycle, L2 1249.4 per step | oscillates, KL 0.25, TV 0.30 per step, argmax fixed | **0.295** (sd 0.003) |
| prolet (Semantic) | fixed point, L2 about 3e-04 per step (numerical floor) | frozen, TV about 8e-07 | 0.75 (sd 0.15), not meaningful at this amplitude |
| Noise (seed 42) | slow drift, L2 3.5 per step, no cycle (distance from base grows linearly) | drifting, TV 3.8e-03 per step, p(top1) still rising at 1000 | 1.12 (sd 0.02) |

The prolet control behaves exactly as predicted: near-zero motion in both spaces (its deltas sit at the numerical noise floor, where the ratio measurement carries no signal, consistent with its lag-10 ratios scattering around 1). The noise control provides the sharpest contrast: its motion is slightly readout-amplified (ratio consistently above 1), the opposite of Divine. Preferential readout-invisibility is not a generic property of ATR trajectories; it is specific to the Divine cycle in this measurement set.

## Interpretation

Answers to the acceptance questions of issue #7:

1. **What fraction of Divine's late motion is readout-invisible?** Per unit tensor motion, the logit response is 29.5 percent of the equal-norm random baseline (about 91 percent suppression in squared terms). The controls bracket it: prolet has no motion to hide, noise is mildly amplified (1.12).
2. **Is the distribution essentially frozen while the tensor moves?** No, and this is the surprise. H-D1 imagined a dancing tensor under a still shadow. The truth is sharper and stranger: the tensor is a period-2 oscillator, and the shadow oscillates with it (TV 0.30 between phases), but the readout displacement is 3.4x smaller than the tensor displacement warrants, and the *argmax* is invariant across both phases. The stable-argmax story survives; the stable-distribution story does not. Every prior measurement of this state on the cycle sampled phase A only (schedules 0/2/3/5/10/20/50/100/250/500: iterations 3 and 5 predate lock-in, and from lock-in onward every recorded snapshot fell on an even iteration), which is why a p = 0.505 "single token" appeared frozen.
3. **Status of H-D1:** supported in a weakened, more precise form. The motion is preferentially readout-suppressed (ratio 0.295, far below both controls), but not null-space motion in the strict sense: a 0.28-probability swing in p(top1) crosses the readout every iteration. The correct geometric statement is: the Divine anomaly is a period-2 limit cycle whose oscillation direction is about 3.4x closer to the readout map's flattened subspace than a random direction, with both endpoints inside the `Divine` argmax cell.

The earlier finding "the tensor that never settles reads out confidently" should now be restated: the tensor settles into a two-state oscillation, and reads out `Divine` from both states, at high probability from one (p 0.50) and moderately from the other (p 0.23). The convergence gate (consecutive-iteration cosine) can never pass on a period-2 cycle by construction; the lag-2 gate since added to the engine (`gate_lag = 2`) does classify this audited trajectory as converged ([`../output_lagk/lagk_report.md`](../output_lagk/lagk_report.md)).

## Caveats

Single trajectory per condition, one noise seed (the fixed-norm 397.18 variant of noise trial 0, which was still converging at iteration 1000; its late numbers describe a trajectory in flight, not a terminal state). The invisibility ratio uses the direct nonlinear effect of the delta on logits versus 20 equal-norm Gaussian directions, not an SVD of a linearised Jacobian; it therefore measures readout response at the actual step size, which for Divine is enormous (78 percent of the vector norm), exactly the regime where a linearisation would be least trustworthy. Period-2 exactness was verified over 20 iterations at iteration 1000 and at machine precision (L2 residual under 1e-03 against amplitude 1249); longer horizons were not probed. The lag-10 invisibility ratios near 1 for the two settled runs are measurements on numerical residue and should not be interpreted.
