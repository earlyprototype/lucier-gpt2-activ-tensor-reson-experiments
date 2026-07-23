# SESSION 04 HANDOVER: The mechanism series, and a deliberate pause on the gate

*Date: 2026-07-23. Continuation context for the next session, human or AI. Read `SESSION_03_HANDOVER.md` first for the deep state, the environment notes, and the working agreements (no em dashes ever; plain language with the technical terms and no decorative metaphors; draft PRs; CodeRabbit for independent review). This document covers only what changed in this repo's experiments since, and the one condition that gates all further work.*

## The operative state: ATR is PAUSED

No ATR experiment runs, and no session resumes ATR work, until an understanding gate is passed. This is durable in `docs/ATR_PAUSE.md` (on main). It is deliberate and operator-set: the investigation outran a working grasp of its dynamical-systems basis, so work resumes only once that grasp is demonstrated, and direction is driven by understanding rather than momentum.

### The gate (re-entry condition)

Two parts, both done with the primers and findings documents neither open nor consulted:

1. **Cold writeup, one page, in Thom's own words:** what ATR does mechanically, step by step; why (the question it asks, and why iterating a model on its own activations is a way to ask it); and how we know what we know, with a clear line between what is **established** and what is **inferred or speculative**. That split is the load-bearing part.
2. **Prediction, cold:** three short "what would happen if" questions, reasoned live rather than recalled, posed at gate time so they cannot be pre-studied.

Examiner stance is **adversarial, not agreeable**: find the memorised phrase standing in for a concept, and the place the established or speculative line is drawn wrong. Passing is holding the whole account without a primer, with the split sound. **Sequencing:** the gate follows Thom's dynamical-systems fundamentals catch-up. When he says he is ready, pose the three questions cold; do not pre-share them.

## Experiment work this cycle: the mechanism series (07-11)

Run after SESSION_03 under issue #14, pushing the period-2 cycle and flip-axis findings (F9, F10) toward mechanism. Each has a full report in its `output_*` directory. **Now integrated into canonical `FINDINGS.md` as findings F13-F17**; the per-experiment reports still live beside their outputs in `output_*/`.

Note: "the hinge" was renamed **"the flip axis d"** in prose this cycle; script names, folder names, and JSON keys keep the old word.

- **07 glitch alignment** (`output_glitch/`): the flip axis's phase-B pole points into GPT-2's anomalous-token cluster (the SolidGoldMagikarp family): cos(d, u) = -0.596 against the geometric core, p < 0.001. Grounds F10's "glitch-token pole," which was previously by inspection. The lowest-norm rows, the high-frequency function words, are a separate set and are NOT aligned with d.
- **08 flip-axis eigenvalue** (`output_hinge_eigen/`): the linearised ATR map inverts the flip axis and only the flip axis, and one attention head, **L11.H8**, does about 99 percent of it. The pivot eigenvalue along the axis is **-4.3** (an overshooting flip, not the conjectured -1); the projected multiplier around the two-step cycle is **+0.10** (strongly contracting). A textbook period-doubling configuration. Measured with `torch.func.jvp` plus finite differences, agreeing to 3-4 significant figures.
- **09 lag-k re-gate** (`output_lagk/`): `atr_engine.run_atr_gated` gained a `gate_lag` parameter (default 1, verified bit-identical to the old consecutive-iteration gate) and a `lag_scan` helper. `Divine` passes cleanly at **lag 2** (cos 1.0000000). Confirms the SESSION_03 correction: the 34 holdouts were exactly the 34 Divine-basin prompts, cycling, not failing to converge.
- **10 J-lens phase probe** (`output_jlens_phase/`): re-ran the pilot membership probe on both cycle phases, the pivot M, and the flip axis (the pilot had probed only phase A). Inherits the pilot's confidence and limits in full, and reports the physical on-shell axis d_sym alongside the frame-mixed committed d. The flip axis is about 95 percent mute to the readout (logit response ratio 0.054).
- **11 suppression test** (`output_suppression/`): three tests on L11.H8. (1) its OV circuit inverts d_sym more strongly than any of the 144 heads (rank 1); (2) ablating it collapses the cycle to a fixed point in about 10 iterations while a same-layer control does not, so it is load-bearing; (3) the copy-suppression signature is **refuted with the opposite sign**: on ordinary text L11.H8 RAISES the attended token's logit (91.4 percent of positions, mean +5.97), where the documented L10.H7 suppressor lowers it (mean -3.62). So L11.H8 sustains the cycle by inverting the flip axis, but it is a copy PROMOTER, not a suppressor, and the "learned copy-suppression function" reading is unsupported.

## Next experiment, signposted (held until the gate passes)

**Issue #17, basin geometry.** Measure how deep each converged attractor's basin is and how steep its walls, by the reverse-ATR move: inject text into a settled loop and measure the dose needed to knock it out of its basin. This is the chosen next step. It does not start until the gate is passed.

## Where things live

- Pause and gate: `docs/ATR_PAUSE.md` (main).
- Canonical findings F1 to F17: `docs/FINDINGS.md` (the mechanism series 07-11 is F13-F17, with the detailed reports beside their outputs in `output_*/`).
- Engine: `atr_engine.py` (now with `gate_lag` and `lag_scan`).
- Prior context, environment, working agreements: `docs/sessions/SESSION_03_HANDOVER.md`.

## Not in this repo (so it is not hunted for here)

The forward J-space programme (Stage 2 planning) and the Stage 1 trajectory-data backup were moved this cycle into the private `atr_research` repo, out of the `fold` lab. This repo stays the public Stage-1 artifact and is not where Stage 2 grows.
