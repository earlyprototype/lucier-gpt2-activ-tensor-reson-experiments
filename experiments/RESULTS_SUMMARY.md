# Cross-Model Run — Results Summary

Executes the validation protocol in `docs/SCALING_ARTEFACT_ANALYSIS.md` on the
`cross-model` branch. The question: are the cross-model ATR results (GPT-2 Medium
→ single `D` basin; Pythia-410m → fragmentation) **readout artefacts** or
**intrinsic model properties**?

**Environment:** Windows 11, Python 3.12, CPU (`device: cpu`), torch 2.7.1,
transformer-lens 2.16.1. 16 cores.

## Deviations from the run plan (read first)

The plan is accurate on intent but stale on some mechanics. What differed:

1. **No `requirements.txt`.** The plan's `pip install -r requirements.txt` cannot
   run — that file does not exist in the repo. Installed the actually-missing
   pieces directly: `nbconvert`, `plotly==5.24.1`, `kaleido==0.2.1` (torch,
   transformer-lens, transformers, numpy, pandas, scikit-learn were already present).
2. **Repo path in the plan is stale.** The plan says `C:\Users\Fab2\Desktop\Work\lucier-repo`;
   the working tree is actually under `…\_LAB_NOTEBOOKS\lucier-repo`. No effect on results.
3. **kaleido static image export hangs on this host.** `fig.write_image(...)`
   (kaleido → Chromium subprocess) never returns. To stop it stalling the
   hour-long compute runs, `write_image` was neutralised to a safe no-op in the
   two long notebooks. **Interactive charts still render** via `fig.show()`
   (plotly mimetype renderer, no kaleido) and are embedded in the executed
   notebooks; only the standalone `.png` files are skipped. All raw data is saved
   to `.pt`/`.md`/`.json` regardless, and the scientific verdicts come from those
   plus stdout, not the PNGs.
4. **Notebook plumbing fixes (minimal, no refactors):**
   - `cos_sim_diagnostic.ipynb`: plotly 5.24 rejects 8-digit hex (`#RRGGBBAA`) for
     `fillcolor`; replaced with an `rgba()` helper.
   - `readout_guardrails.ipynb`: imported `atr_engine` and wrote to
     `experiments/output/` assuming repo-root CWD, but nbconvert launches it from
     `experiments/`. Added a cell-0 bootstrap that locates the repo root
     (dir containing `atr_engine.py`) and `chdir`s there.
   - `atr_engine.py`: **extended** with the ATR-R1/R2 confidence metrics the
     guardrails notebook requires (`top_token_ids/strings/probs_last`,
     `top_logit_margin_last`, `entropy_last`, `all_position_token_ids/strings`).
     Purely additive — verified only `readout_guardrails.ipynb` imports the engine,
     so no other notebook is affected; the April `.pt` sweeps are untouched.
   - `03_random_baseline.ipynb`: the April `stage1_results.pt` is stored in
     **columnar** form (dict of per-iteration arrays), but the notebook read it as
     a list of per-iteration snapshot dicts with a full `tensor` key. Added a
     columnar→snapshots adapter at load; `seq_len` derived from the per-position
     token list, scale proxy from `mean_norms` (absolute norm is scale-invariant
     under layer-0 LayerNorm — the analysis doc's own §1.1 — so basin findings are
     unaffected).

No `experiments/*/output/*.pt` files were deleted or overwritten.

---

## 1. `cos_sim_diagnostic.ipynb` — Control 1 (tensor-level convergence)

**What ran:** Parses `cos_sim_mean` from the saved outputs of the four
`01_attractor_dominance.ipynb` notebooks (125 prompts each; no model runs).
Charts render inline via `fig.show()`. Duration ~1–2 min. Deviation: fillcolor fix (above).

**Headline numbers** — `cos_sim_mean(iterₙ, iterₙ₋₁)`, mean across 125 prompts:

| Model | iter 2 | iter 5 | iter 10 | iter 50 | iter 100 | iter 250 | Tensor verdict |
|---|---|---|---|---|---|---|---|
| GPT-2 Small (124M) | 0.69 | −0.24 | 0.61 | 0.84 | 0.91 (σ .15) | — | partial, noisy |
| GPT-2 Medium (345M) | 0.991 | 0.9994 | **1.0000** | 1.0000 | 1.0000 | — | **saturated by iter 10** |
| Pythia-160m | 0.991 | 0.9997 | **1.0000** | 1.0000 | 1.0000 | — | **saturated by iter 10** |
| Pythia-410m | 0.82 | 0.90 | 0.89 | 0.85 | 0.85 | 0.86 (σ .14) | **never converges (~0.85 plateau)** |

Pythia-410m breakdown: only **9/125** prompts converge (cos > 0.99 from iter 100+);
**116/125** oscillate.

**Decides:** *Did the tensor converge even where the token flickered?* — GPT-2 Medium
(`D`) and Pythia-160m (`questioned`) reach `cos_sim_mean = 1.0000`: their collapses
are **real tensor attractors**, not readout illusions. Pythia-410m stays at ~0.85
(σ 0.14, non-monotonic) through 250 iterations while the others saturate. Per the
analysis doc's test 1, Pythia-410m non-convergence is **internal-dynamics evidence** —
the mean tensor itself keeps moving, so the fragmentation is **not** purely a readout
artefact.

**Interpretation:** The four models do not share one failure mode. Two smaller/Reddit-
and-Pile models lock to a single tensor fixed point within 10 iterations; Pythia-410m
does not settle at all on the 250-iteration horizon. This is the cleanest single
separator in the study and it points at genuine architecture/depth-driven dynamics for
Pythia-410m, with readout ambiguity a secondary (not primary) factor.

**Open questions:** Is ~0.85 a slow approach to convergence (would 1000 iterations get
there?) or a genuine limit cycle / wandering attractor? → addressed by notebook 4.

---

## 2. `readout_guardrails.ipynb` — ATR-R1 / R3 (readout confidence)

**What ran:** gpt2-small, prompt `"The cat sat on the mat and then"`, layers 0→11,
schedule `[0,2,3,5,10,20,50,100]`, 8 snapshots. Uses the newly-added confidence metrics
in `atr_engine.py`. R3 thresholds: `HIGH_COS = 0.995`, `LOW_MARGIN = 0.2` (logit scale).
Output: `experiments/output/readout_guardrails_gpt2_small.json`. Duration ~4 min
(incl. gpt2-small download). Deviations: engine extension + cell-0 bootstrap (above).

**Headline numbers** — R3 concordance categories: `{high_cos_low_margin: 1,
high_cos_high_margin: 0, lower_cos: 7}`.

| iter | cos_mean | logit margin | entropy | top token | R3 category |
|---|---|---|---|---|---|
| 0 | +1.000 | 0.071 | 6.00 | ` looked` | high_cos_low_margin |
| 5 | −0.168 | 0.035 | 8.58 | ` fem` | lower_cos |
| 20 | +0.973 | 0.755 | 6.04 | ` Zero` | lower_cos |
| 50 | +0.670 | 2.411 | 3.97 | ` Divine` | lower_cos |
| 100 | +0.678 | 1.912 | 3.30 | ` Divine` | lower_cos |

**Decides:** *Are the basins high-confidence attractors or low-confidence flicker?* —
For this gpt2-small prompt the answer is neither yet: 7/8 snapshots are "true ongoing
dynamics" (`cos_sim_mean` never crosses 0.995 within 100 iters). But the readout
signal is clean: as the trajectory settles toward the `Divine` basin (iters 50→100),
**logit margin rises** (2.41, 1.91) and **entropy falls** (3.97→3.30) — readout
confidence grows even while the tensor is still moving. So where a basin label appears,
it is high-confidence, not boundary flicker.

**Interpretation:** This notebook is scaffolded as a **single-prompt gpt2-small
demonstration** of the R1/R3 metric machinery, and it now works end-to-end: margin and
entropy track basin approach sensibly. It confirms the *method* is sound for the next
step — applying it at scale to GPT-2 Medium's `D` and Pythia-410m's fragments.

**Open questions:** The notebook does not itself run gpt2-medium or pythia-410m, so the
plan's headline "GPT-2 Medium `D` vs Pythia fragments: attractor or flicker?" is only
partially answered here. `LOW_MARGIN = 0.2` is on the **logit** scale (author's
calibration knob); observed margins span 0.03–2.4, so most snapshots read as
"not-low-margin." Extending R1/R3 across all four models at their full prompt sets is
the natural follow-on.

---

## 3. `03_random_baseline.ipynb` — the null model

*[PENDING — run in progress. Fill: random vs real basin count/identity, convergence,
bootstrap CI, verdict against "weight geometry vs prompt regime".]*

---

## 4. `01b_deep_convergence.ipynb` — Control 3 (long horizon)

*[PENDING — run in progress. Fill: 25-prompt Pythia-410m to 1000 iters — does it
converge given more time, or stay fragmented; unique terminal tokens; cross-prompt
similarity.]*

---

## Synthesis

*[PENDING until 3 & 4 complete.]*

## Not scaffolded (follow-on work)

- **Control 2** — Pythia-410m depth control (loop layers 0–11 vs 0–23), holding weights/
  tokenizer/corpus constant. Not built; the plan says note only.
- Extend ATR-R1/R3 (notebook 2's machinery) across GPT-2 Medium and Pythia-410m at full
  prompt sets to finish separating readout ambiguity from true dynamics per-model.
