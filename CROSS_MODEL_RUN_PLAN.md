# Cross-Model Run Plan — executing the April 2026 validation protocol

**For:** any operator session (human or agent) executing on this machine.
**Branch:** `cross-model` (this working tree). Do not touch `main`.
**Theory behind this plan:** `docs/SCALING_ARTEFACT_ANALYSIS.md` — read it first. The question is whether the cross-model results (GPT-2 Medium → single `D` basin; Pythia-410m → fragmentation) are readout artefacts or intrinsic model properties.

---

## Ground rules

1. Work only on the `cross-model` branch of this repo (`C:\Users\Fab2\Desktop\Work\lucier-repo`).
2. **Do not delete `experiments/*/output/*.pt`** — these are the April sweep trajectories, gitignored, local-only, and several diagnostics read them instead of re-running hours of sweeps.
3. Do not edit `README.md` on `main`, do not push to `main`, do not rewrite history.
4. Commit to `cross-model` and push after each notebook completes — results must not sit uncommitted again.
5. Runs are CPU (`device: cpu` in prior configs). Sweeps take on the order of an hour each; diagnostics that read existing `.pt` files are minutes.

## Execution order (decisiveness per unit cost)

### 1. `experiments/cos_sim_diagnostic.ipynb` — Control 1
Cross-model `cos_sim_mean` chart from the existing `stage1_results.pt` files. No model runs needed.
**Decides:** for each model, did the *tensor* converge even where the *token* flickered?
- Pythia-410m saturates while tokens flicker → readout artefact (fragmentation is illusory).
- Pythia-410m stays below convergence → genuine structural fragmentation.

### 2. `experiments/readout_guardrails.ipynb` — ATR-R1/R3
Adds the missing readout-confidence metrics (top-1 vs top-2 logit margin, entropy, ID-first traces) and classifies snapshots per the concordance audit in the analysis doc: readout ambiguity vs stable attractor vs true dynamics.
**Decides:** whether GPT-2 Medium's `D` and Pythia's fragments are high-confidence attractors or low-confidence boundary flicker.

### 3. `experiments/gpt2_small/03_random_baseline.ipynb` — the null model
Iterate from random tensors / random token IDs on GPT-2 Small (no real prompts). ~1 hr CPU.
**Decides:** whether the five semantic basins are properties of the weight geometry (noise reproduces them) or of the prompt regime (it doesn't). Interpretation guide for all five possible outcomes: `C:\Users\Fab2\Desktop\Skills\_portfolio\_outreach\Lucier\NULL_MODEL_SKETCH.py` (bottom comments).

### 4. `experiments/pythia_410m/01b_deep_convergence.ipynb` — Control 3
Long-horizon run to distinguish "not yet converged at 100–250" from "structurally fragmented." If the full 125-prompt sweep is too slow on CPU, run a 10–20 prompt subset to 1000 iterations and say so in the results.

### Not scaffolded (note only, don't build unless asked)
Control 2 — Pythia-410m depth control (loop layers 0–11 vs 0–23). Log as follow-on work in the results summary.

## How to run

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace <notebook> --ExecutePreprocessor.timeout=7200
```

If a notebook's plumbing is broken (stale paths from the restructure), fix minimally to run — no refactors.

## Deliverable

`experiments/RESULTS_SUMMARY.md`, committed to `cross-model`, containing per notebook:
- what ran (parameters, duration, any deviations),
- headline numbers/verdicts against the "Decides" lines above,
- one-paragraph interpretation,
- open questions.

End state: four executed notebooks with outputs, the summary, everything committed and pushed to `cross-model`. The synthesis pass (updating the public README's scientific claims) happens in a separate session afterwards — do not attempt it here.

---

## Addendum (added after diagnostics 1–2): GPT-2 Small convergence-gated re-sweep

**Motivation:** Control 1 showed gpt2-small mean cos_sim = 0.91 (σ 0.15) at iter 100 —
the published basin table was classified before convergence. Original 5-prompt run
showed token movement to ~iter 500.

### 5. GPT-2 Small, 125 prompts, convergence-gated
- Re-run the gpt2_small attractor sweep with `MAX_ITERATIONS = 1000` and an
  **early-stop gate**: stop a prompt when `cos_sim_mean > 0.999` for 3 consecutive
  checks (check every 10 iters past 100). Classify terminal basin **at lock-in**,
  not at a fixed horizon. Record lock-in iteration per prompt.
- Adapt `atr_engine.py` minimally (early-stop param); do not touch the saved April `.pt`s —
  write to `experiments/gpt2_small/output_gated/`.
- **Decides:** are the five published basins (prolet 35.2 / Divine 27.2 / Anarch 20.8 /
  till 15.2 / solidarity 1.6) stable under proper convergence, or stop-time artefacts?
  Specific hypothesis to test: `till` is a slow transient whose share collapses.
- Run only AFTER notebooks 3 & 4 complete (CPU contention). Append results to
  `RESULTS_SUMMARY.md` as section 5 and update the Synthesis.
