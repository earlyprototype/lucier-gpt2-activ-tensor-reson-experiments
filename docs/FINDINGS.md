# ATR — Findings (Canonical Record)

**Scope:** Complete record of the ATR experimental series as of 2026-07-10: Stage 0
(repeatability) through Stage 5 (convergence-gated re-sweep), across GPT-2 Small,
GPT-2 Medium, Pythia-160m, and Pythia-410m, plus a random-noise null model and
readout diagnostics. This document is the reporting register: where the README (the
piece) and this record differ, this record governs.

**Provenance:** Validation runs executed 2026-07-10 on CPU (Windows 11, Python 3.12,
torch 2.7.1, transformer-lens 2.16.1). Run-by-run details, deviations, and environment
notes: [`experiments/RESULTS_SUMMARY.md`](../experiments/RESULTS_SUMMARY.md). Original
exploratory work: 2026-03. Method specification: [TECHNICAL.md](TECHNICAL.md).

---

## 1. Run inventory

| # | Run | Model(s) | N | Output |
|---|---|---|---|---|
| 0 | Repeatability gate | gpt2-small | 5 prompts × 2 runs | `experiments/gpt2_small/00_reproducibility_gate.ipynb` |
| 1 | Attractor dominance sweep | gpt2-small | 125 prompts, ≤100 iters | `experiments/gpt2_small/output/` |
| 2 | Cross-model sweeps | gpt2-medium, pythia-160m, pythia-410m | 125 prompts each | `experiments/<model>/output/` |
| 3 | Random-noise null model | gpt2-small | 125 Gaussian tensors (seed 42) | `experiments/gpt2_small/output_random_baseline/` |
| 4 | Deep convergence | pythia-410m | 8-prompt subset, 1000 iters | `experiments/pythia_410m/output_deep/` |
| 5 | Convergence-gated re-sweep | gpt2-small | 125 prompts, gate cos>0.999×3, ≤1000 iters | `experiments/gpt2_small/output_gated/` |
| — | Tensor convergence diagnostic | all four | reads runs 1–2 | `experiments/cos_sim_diagnostic.ipynb` |
| — | Readout confidence audit | gpt2-small | single-prompt demo | `experiments/output/readout_guardrails_gpt2_small.json` |

## 2. Principal findings

### F1 — GPT-2 Small resolves language-driven activity into five attractor basins, stable under convergence gating {#run-5}

Basin shares classified **at lock-in** (cosine similarity of successive mean tensors
> 0.999 sustained ×3 checks), not at a fixed iteration horizon:

| Basin | At iter 100 (superseded) | **At lock-in (canonical)** | W_E neighbourhood |
|:---|---:|---:|:---|
| `prolet` | 44 (35.2%) | **54 (43.2%)** | political philosophy |
| `Divine` | 34 (27.2%) | **34 (27.2%)** | theology |
| `till` | 19 (15.2%) | **19 (15.2%)** | temporal/functional (outlier) |
| `Anarch` | 26 (20.8%) | **17 (13.6%)** | political philosophy |
| `solidarity` | 2 (1.6%) | **1 (0.8%)** | collective action |

91/125 prompts (73%) reach a hard fixed point, all at lock-in iteration 120 (the
gate's earliest firing point; the true settling iteration lies between 100 and 120 —
finer cadence not measured). The iteration-100 table published earlier over-counted
`Anarch` by ~10 prompts that were still drifting `Anarch`→`prolet`. A pre-registered
hypothesis that `till` was a slow transient was **refuted**: 19/19 `till` prompts
converge and retain their label.

### F2 — The `Divine` basin is readout-stable over a never-settling tensor

The 34 prompts that never pass the convergence gate (to 1000 iterations) are exactly
the 34 `Divine` prompts. Their decoded top-1 token is stable throughout while the
underlying tensor keeps moving — the study's clearest case of dynamics and decoding
dissociating. Whether `Divine` is a limit cycle, a wandering attractor within one
decode region, or something else is open.

### F3 — The basin landscape does not generalise across models (fingerprint hypothesis refuted)

| Model | Params | Corpus | Landscape | Tensor verdict (cos_sim_mean) |
|---|---|---|---|---|
| GPT-2 Small | 124M | WebText | 5 semantic basins | partial at 100 (0.91, σ0.15); 73% hard-converged by 120 (gated) |
| GPT-2 Medium | 345M | WebText | 1 basin: `D` (100%) | saturated 1.0000 by iter 10 |
| Pythia-160m | 160M | The Pile | 1 basin: `questioned` (94.4%) | saturated 1.0000 by iter 10 |
| Pythia-410m | 410M | The Pile | no consolidation (40+ fragments) | never converges (~0.85 plateau; 9/125 prompts converge; 8-prompt subset at 1000 iters: 8 distinct terminals, cross-prompt sim 0.21) |

GPT-2 Medium shares GPT-2 Small's training corpus and produces no semantic basins.
The hypothesis "attractor basins are a thematic fingerprint of the training corpus,
readable from any model" is **refuted as a general claim**. The semantic-basin
phenomenon is, on current evidence, specific to GPT-2 Small within this set.

### F4 — The five basins belong to the language-driven regime, not the weights in general (null model)

125 random Gaussian tensors (norm- and length-calibrated to the real runs) iterated
through GPT-2 Small converge (position collapse → 1.0000) but into **18 basins**,
dominated by the em-dash token `―` (64%), with ~zero identity overlap with the real
five (1/125 trials reached `prolet`). Bootstrap on the random basin count: 14.1,
95% CI [11, 17]; the real count (5) falls **below** the CI. Real language funnels
into *fewer* attractors than noise, and semantically coherent ones. ATR therefore
reads the model *as driven by language-shaped input*; the basins are not universal
fixed points of the weight geometry.

### F5 — The cross-model differences are intrinsic, not apparatus artefacts

Three attribution results ([SCALING_ARTEFACT_ANALYSIS.md](SCALING_ARTEFACT_ANALYSIS.md)):

1. **Normalisation exonerated:** the global L2 rescale is invisible to the forward
   pass (layer-0 LayerNorm scale-invariance).
2. **Convergence verdicts are tensor-level:** `cos_sim_mean` never passes through
   token decoding, so Medium/160m saturation and 410m non-convergence are properties
   of the dynamics, not the readout.
3. **Readout is a real but secondary jitter source:** logit margin rises and entropy
   falls as trajectories settle (single-prompt audit); where basin labels appear they
   are high-confidence. The `Divine` dissociation (F2) is the known exception class.

## 3. Hypothesis dispositions

| ID | Hypothesis | Disposition |
|---|---|---|
| H0 | Results are deterministic | **Repeatability supported** (N=2, same machine, identical terminal basins; intermediate paths float-sensitive). Independent reproduction not attempted. |
| H1 | `prolet` is the dominant basin | **Supported, revised upward** — 43.2% at lock-in (was 35.2% at iter 100). Per-prompt category predictions remained poor (~25%); the structural claim stands, the predictive one does not. |
| H2 | `Divine` is a genuine secondary basin | **Supported with qualification** — 27.2%, but it is a readout-stable/tensor-unsettled object (F2), unlike the other four. |
| H3 | Intermediate tokens reflect training-corpus topology | **Partially supported, generality refuted** — 4/5 basin tokens cluster semantically in W_E (permutation test still pending), but the corpus-causal reading fails cross-model (F3). |
| H4 | Per-head resonance ≈ linear power iteration on W_OV (cos > 0.9 to top singular vector) | **Untested** — protocol scaffolded (`experiments/gpt2_small/spectral_resonance.ipynb`), not run. |
| H-fingerprint | Basin profiles read training-data bias without data access | **Refuted as stated** (F3, F4). |
| H-till | `till` is a slow transient | **Refuted** (F1: 19/19 stable). |

## 4. Caveats {#caveats}

1. **Repeatability, not reproducibility.** N=2 same-machine runs; no independent
   re-implementation.
2. **Single-seed sweeps.** The 125-prompt sweeps are one seed per model; the null
   model is one seed set (42) with a bootstrap over trials, not over sweeps.
3. **Deep-convergence subset.** The 1000-iteration Pythia-410m run used 8 prompts
   (CPU constraint). Direction matches the 125-prompt evidence at 250 iterations, but
   the subset is small.
4. **W_E permutation test pending.** The semantic-clustering claim (H3) rests on
   neighbourhood inspection plus an all-positive cross-similarity matrix (91/91 pairs,
   0.18–0.47); the designed random-token-set permutation test has not been run.
5. **Gate cadence.** Lock-in iterations cluster at 120 because that is the gate's
   earliest possible firing; true settling times between 100 and 120 are unresolved.
6. **Hook-position dependence unexplored.** All runs cut the loop at
   (final-layer `resid_post` → layer-0 `resid_pre`). Alternative windows (including a
   Pythia-410m depth control, layers 0–11 vs 0–23) are designed but not run.
7. **Normalisation scheme.** Global L2 rescale only; per-position/per-dimension
   schemes unexplored (though the global scheme is provably inert through layer-0
   LayerNorm — see F5.1).
8. **BPE granularity.** Basin identities are single BPE tokens (`prolet`, `Anarch`);
   multi-token structure is invisible to the current readout.
9. **Readout is logit-lens-style.** Decoding applies `ln_final → W_U` to
   intermediate states; the `Divine` dissociation (F2) shows the decode and the
   dynamics can disagree, and only one prompt has had the full confidence audit.

## 5. What ATR is, after this series

A cheap, training-free probe of the stable states of a model's iterated forward map
under a chosen input regime. It does not read training-data bias (refuted). It does
distinguish, sharply and at tensor level, qualitatively different iterated-dynamics
regimes across models — and it surfaced one unexplained anomaly worth pursuing: GPT-2
Small's five semantically coherent, language-specific attractor basins.

Open directions, in rough order of leverage: why GPT-2 Small (the anomaly); the
`Divine` object (F2); hook-window/depth dependence (caveat 6); the pending statistics
(caveats 4, 5); H4.
