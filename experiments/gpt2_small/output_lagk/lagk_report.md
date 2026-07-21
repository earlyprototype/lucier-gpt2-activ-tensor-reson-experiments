# EXP: Lag-k Re-Gate (Issue #14, Thread 2): Gating the Bell at Its Own Period

*A lag-1 gate asks the tensor: are you where you were one step ago? A bell always answers no. Ask at its period and it answers yes, to machine precision.*

**Date:** 2026-07-19. **Model:** GPT-2 Small (TransformerLens, weights loaded offline via `ATR_GPT2_LOCAL`). **Runner:** [`09_lagk_gate.py`](../09_lagk_gate.py). **Raw data:** [`lagk_results.json`](lagk_results.json). **Engine change:** [`atr_engine.run_atr_gated`](../../../atr_engine.py) now takes `gate_lag` (default 1: the historical consecutive-iteration gate, verified identical old-vs-new on matched runs), and a new pure-tensor helper `atr_engine.lag_scan` returns mean cosine at every lag 1..max_lag over densely recorded iterates.

## The Question

The gated re-sweep ([`gated_report.md`](../output_gated/gated_report.md)) locked in 91 of 125 prompts and left 34 running to the 1000-iteration ceiling. The motion audit ([`divine_motion_report.md`](../output_divine_motion/divine_motion_report.md)) then showed why the holdouts cannot ever lock: the gate compares consecutive iterates (`cos_sim_mean` at lag 1 above 0.999), and the Divine state is an exact period-2 limit cycle whose consecutive iterates sit at cosine 0.6849 forever. A lag-1 gate can never pass a period-2 cycle by construction. The Session 03 handover recorded the standing correction: "34 prompts never converge" should become "34 prompts ring, pending re-gate", and the fix is a one-line engine change. (Arithmetic on the committed sweep report makes the identity concrete: the converged basins sum to 54 + 19 + 17 + 1 = 91, so the 34 holdouts are exactly the 34 Divine-basin prompts.)

This experiment makes the engine change and runs the first census. `gate_lag = k` compares iterate t with iterate t-k against the same threshold; `lag_scan` is the survey instrument that says which k, if any, a state would pass.

## Method

The three committed iteration-1000 loop states from the motion audit (`state_divine.pt`, `state_prolet.pt`, `state_noise.pt`) were each continued 24 further iterations with the exact ATR map, recording every iterate: 25 dense iterates per state, no schedule, no aliasing. Sanity gate before measuring anything: the Divine continuation must reproduce the committed bell numbers, and did, exactly: cos(A, f(A)) = 0.684912 (bell_anatomy.json: 0.684912) and cos(A, f(f(A))) = 1.000000 (committed: 1.000000). `lag_scan` then ran on each state's mean vectors (the gate's metric; last-vector tables agree to the seventh decimal and sit alongside in the JSON), with the pass threshold 0.999 read from the engine's own default, the same value the 125-prompt sweep ran at.

## The Lag Table

Mean cosine between iterates k apart over the 25 dense iterates (mean vector). Pass = above 0.999.

| Lag k | Divine | pass | prolet | pass | noise | pass |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.6849117 | no | 1.0000000 | yes | 0.9999962 | yes* |
| 2 | 1.0000000 | **yes** | 1.0000000 | yes | 0.9999852 | yes* |
| 3 | 0.6849117 | no | 1.0000000 | yes | 0.9999669 | yes* |
| 4 | 1.0000000 | yes | 1.0000000 | yes | 0.9999413 | yes* |
| 5 | 0.6849117 | no | 1.0000000 | yes | 0.9999088 | yes* |
| 6 | 1.0000000 | yes | 1.0000000 | yes | 0.9998696 | yes* |
| 7 | 0.6849117 | no | 1.0000000 | yes | 0.9998236 | yes* |
| 8 | 1.0000000 | yes | 1.0000000 | yes | 0.9997709 | yes* |

Smallest passing lag: prolet 1, Divine 2, noise 1 (nominal: see the caveat section for the asterisks). Three signatures, one instrument:

- **prolet**: flat at 1.0000000 at every lag (per-pair minimum 0.9999999, the float32 floor). A true fixed point passes everywhere. Its final lag-1 steps move L2 0.0002 to 0.0004: numerical residue, not motion.
- **Divine**: fails every odd lag at 0.6849117 and passes every even lag at 1.0000000 (per-pair minimum 0.9999999). The parity stripe of an exact period-2 cycle: each lag-1 step is the full A-to-B swing, L2 1249.43, cosine 0.6849, every iteration, unchanged.
- **noise**: decays monotonically with lag, 0.9999962 down to 0.9997709. The cosine deficit (1 minus cos) grows from 3.8e-06 at lag 1 to 2.3e-04 at lag 8 while prolet's stays pinned at the 1e-07 floor: the signature of drift, no period anywhere. Its lag-1 steps still move L2 3.2 to 3.4.

## Re-Gate Verdict: Divine Counts as Converged at gate_lag = 2

At the standard threshold (0.999, the engine default): the Divine state's lag-2 mean-vector cosine over the window is 1.0000000 (mean), 0.9999999 (minimum pair). Every possible lag-2 check clears the threshold, so any `patience` and `check_every` schedule locks in. Under `gate_lag = 1` the same state reads 0.6849117 at every check, 0.31 below threshold, forever. **Verdict: Divine is converged under `gate_lag = 2`; it is unconvergeable under `gate_lag = 1`.**

Both phases decode to the same token, as the bell anatomy requires: iterate 1023 (phase B) argmax ` Divine` (id 13009) at p = 0.2252, entropy 4.62 nats; iterate 1024 (phase A) argmax ` Divine` (id 13009) at p = 0.5046, entropy 3.05 nats. One timbre, two volumes, one gate verdict.

This is the first concrete instance of the canon correction "34 prompts ring, pending re-gate": one of the 34, the Syntactic prompt, is now re-gated as converged at its own period.

## Caveat: The Noise Row (Threshold Blindness Is a Different Axis)

The honest row: within this late 24-iteration window the noise control also clears 0.999 at every lag. Its drift has decelerated to L2 about 3.3 per step against a large-norm vector, which no one-step cosine at this threshold can see; the lag-1 gate as configured would lock this still-drifting state in as converged if applied to this window. Earlier in its own trajectory the same threshold rejected it (lag-10 cosine 0.9788 at iterations 800-810, 0.9996 at 990-1000), and its readout is still moving: p(top1) of the horizontal-bar token (U+2015, id 31857) read 0.6422 at iteration 1000 and 0.5971 at iteration 1024. The lag-k gate corrects lag aliasing of exact cycles; it does not fix threshold blindness to slow drift. What separates the three states is the pattern across lags, not any single number: flat at the floor (fixed point), parity-striped (period 2), monotone decay (drift).

## The Period-4-and-Longer Blind Spot

A period-p cycle passes a lag-k comparison exactly when p divides k. The Divine bell hid under every snapshot schedule previously used because the sampled lags were multiples of its period; a plain lag-2 re-gate inherits the same blindness one octave up. A period-4 ringer would fail lags 1, 2, 3, 5, 6, 7 and pass only 4 and 8: under a `gate_lag = 2` gate it would look exactly the way Divine looked under lag 1, never converging, invisible again. In this census no state shows a period above 2 (nothing passes at 4 that does not already pass at 2, and the odd/even stripe is complete). Periods above 8 or quasi-periodic orbits would need a longer scan (`lag_scan` takes `max_lag`) over a window a few cycle lengths deep. Recommendation for the eventual 34-prompt re-gate: run the full lag table on a short dense continuation, as here, and gate each state at its smallest passing lag; do not just swap one fixed lag for another.

## What Stays Blocked on Issue #9

The other 33 ringing prompts exist in the sweep records only as ids and terminal tokens; their texts live in `prompt_library.py`, which exists only on Thom's home machine (issue #9, his errand). Until it is restored they cannot be re-run, so "34 prompts ring, pending re-gate" resolves today to: 1 re-gated (the Syntactic prompt's Divine bell, converged at `gate_lag = 2`), 33 pending on #9. The machinery is ready for them: a 24-iteration dense continuation plus `lag_scan`, then `run_atr_gated(..., gate_lag=k)` at the smallest passing k, is the whole recipe this script demonstrates.

## Caveats

One window (iterations 1000 to 1024) per state; at lag 8 the mean is over 17 pairs. The verdict for Divine is a statement about the committed locked state, not about when a fresh gated run would first lock (that needs the early trajectory, out of scope for this light census). The noise nominal passes are a property of this decelerated late window, not of the trajectory. Periods above 8 were not scanned. The engine default `gate_lag = 1` was verified bit-identical to the pre-change engine on matched runs (default arguments, gate-check path, and lock-in path) before any census was run.
