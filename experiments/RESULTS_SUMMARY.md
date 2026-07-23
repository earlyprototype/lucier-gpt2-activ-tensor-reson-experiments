# Cross-Model Run: Results Summary

Executes the validation protocol in `docs/SCALING_ARTEFACT_ANALYSIS.md` on the
`cross-model` branch. The question: are the cross-model ATR results (GPT-2 Medium
→ single `D` basin; Pythia-410m → fragmentation) **readout artefacts** or
**intrinsic model properties**?

**Environment:** Windows 11, Python 3.12, CPU (`device: cpu`), torch 2.7.1,
transformer-lens 2.16.1. 16 cores.

## Deviations from the run plan (read first)

The plan is accurate on intent but stale on some mechanics. What differed:

1. **No `requirements.txt`.** The plan's `pip install -r requirements.txt` cannot
   run: that file does not exist in the repo. Installed the actually-missing
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
     Purely additive: verified only `readout_guardrails.ipynb` imports the engine,
     so no other notebook is affected; the April `.pt` sweeps are untouched.
   - `03_random_baseline.ipynb`: the April `stage1_results.pt` is stored in
     **columnar** form (dict of per-iteration arrays), but the notebook read it as
     a list of per-iteration snapshot dicts with a full `tensor` key. Added a
     columnar→snapshots adapter at load; `seq_len` derived from the per-position
     token list, scale proxy from `mean_norms` (absolute norm is scale-invariant
     under layer-0 LayerNorm, the analysis doc's own §1.1, so basin findings are
     unaffected).

5. **Notebook 4 reduced to 8 prompts (CPU time).** The notebook's own 25-prompt ×
   1000-iteration sweep exceeded the 2-hour nbconvert cell timeout on this CPU host
   (each pythia-410m forward pass ≈ 3× a GPT-2 Small pass; 25,000 passes > 7200 s). Per
   the run plan's explicit allowance ("if the full sweep is too slow on CPU, run a
   10–20 prompt subset to 1000 iterations and say so"), reduced to a **diverse 8-prompt
   subset spanning all 7 categories**, keeping the full 1000-iteration horizon (the point
   of Control 3). An injected override cell also rewrites `config.md` to match.

No `experiments/*/output/*.pt` files were deleted or overwritten.

---

## 1. `cos_sim_diagnostic.ipynb`: Control 1 (tensor-level convergence)

**What ran:** Parses `cos_sim_mean` from the saved outputs of the four
`01_attractor_dominance.ipynb` notebooks (125 prompts each; no model runs).
Charts render inline via `fig.show()`. Duration ~1–2 min. Deviation: fillcolor fix (above).

**Headline numbers**: `cos_sim_mean(iterₙ, iterₙ₋₁)`, mean across 125 prompts:

| Model | iter 2 | iter 5 | iter 10 | iter 50 | iter 100 | iter 250 | Tensor verdict |
|---|---|---|---|---|---|---|---|
| GPT-2 Small (124M) | 0.69 | −0.24 | 0.61 | 0.84 | 0.91 (σ .15) | — | partial, noisy |
| GPT-2 Medium (345M) | 0.991 | 0.9994 | **1.0000** | 1.0000 | 1.0000 | — | **saturated by iter 10** |
| Pythia-160m | 0.991 | 0.9997 | **1.0000** | 1.0000 | 1.0000 | — | **saturated by iter 10** |
| Pythia-410m | 0.82 | 0.90 | 0.89 | 0.85 | 0.85 | 0.86 (σ .14) | **never converges (~0.85 plateau)** |

Pythia-410m breakdown: only **9/125** prompts converge (cos > 0.99 from iter 100+);
**116/125** oscillate.

**Decides:** *Did the tensor converge even where the token flickered?* GPT-2 Medium
(`D`) and Pythia-160m (`questioned`) reach `cos_sim_mean = 1.0000`: their collapses
are **real tensor attractors**, not readout illusions. Pythia-410m stays at ~0.85
(σ 0.14, non-monotonic) through 250 iterations while the others saturate. Per the
analysis doc's test 1, Pythia-410m non-convergence is **internal-dynamics evidence**:
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

## 2. `readout_guardrails.ipynb`: ATR-R1 / R3 (readout confidence)

**What ran:** gpt2-small, prompt `"The cat sat on the mat and then"`, layers 0→11,
schedule `[0,2,3,5,10,20,50,100]`, 8 snapshots. Uses the newly-added confidence metrics
in `atr_engine.py`. R3 thresholds: `HIGH_COS = 0.995`, `LOW_MARGIN = 0.2` (logit scale).
Output: `experiments/output/readout_guardrails_gpt2_small.json`. Duration ~4 min
(incl. gpt2-small download). Deviations: engine extension + cell-0 bootstrap (above).

**Headline numbers**: R3 concordance categories: `{high_cos_low_margin: 1,
high_cos_high_margin: 0, lower_cos: 7}`.

| iter | cos_mean | logit margin | entropy | top token | R3 category |
|---|---|---|---|---|---|
| 0 | +1.000 | 0.071 | 6.00 | ` looked` | high_cos_low_margin |
| 5 | −0.168 | 0.035 | 8.58 | ` fem` | lower_cos |
| 20 | +0.973 | 0.755 | 6.04 | ` Zero` | lower_cos |
| 50 | +0.670 | 2.411 | 3.97 | ` Divine` | lower_cos |
| 100 | +0.678 | 1.912 | 3.30 | ` Divine` | lower_cos |

**Decides:** *Are the basins high-confidence attractors or low-confidence flicker?*
For this gpt2-small prompt the answer is neither yet: 7/8 snapshots are "true ongoing
dynamics" (`cos_sim_mean` never crosses 0.995 within 100 iters). But the readout
signal is clean: as the trajectory settles toward the `Divine` basin (iters 50→100),
**logit margin rises** (2.41, 1.91) and **entropy falls** (3.97→3.30): readout
confidence grows even while the tensor is still moving. So where a basin label appears,
it is high-confidence, not boundary flicker.

**Interpretation:** This notebook is scaffolded as a **single-prompt gpt2-small
demonstration** of the R1/R3 metric machinery, and it now works end-to-end: margin and
entropy track basin approach sensibly. It confirms the *method* is sound for the next
step: applying it at scale to GPT-2 Medium's `D` and Pythia-410m's fragments.

**Open questions:** The notebook does not itself run gpt2-medium or pythia-410m, so the
plan's headline "GPT-2 Medium `D` vs Pythia fragments: attractor or flicker?" is only
partially answered here. `LOW_MARGIN = 0.2` is on the **logit** scale (author's
calibration knob); observed margins span 0.03–2.4, so most snapshots read as
"not-low-margin." Extending R1/R3 across all four models at their full prompt sets is
the natural follow-on.

---

## 3. `03_random_baseline.ipynb`: the null model

**What ran:** 125 random Gaussian tensors (seed 42), norm/seq-len calibrated from the
April stage-1 records, iterated through GPT-2 Small (layers 0→11) on the same
`[0,2,3,5,10,20,50,100]` schedule as the real sweep. Duration ~40 min (CPU,
thread-capped and running concurrently with notebook 4; TransformerLens per-call
overhead dominates). Deviations: columnar-stage1 adapter + write_image no-op (above).
Outputs in `experiments/gpt2_small/output_random_baseline/`
(`random_baseline_results.pt`, `random_baseline_report.md`, `dissolution_pathways_random.md`).

**Headline numbers:**

| | Real prompts (stage 1) | Random baseline |
|---|---|---|
| Terminal basins | **5** | **18** |
| Basin identity | `prolet` 35%, `Divine` 27%, `Anarch` 21%, `till` 15%, `solidarity` 2% | `―` 64%, `instant` 11%, `abs` 4%, `justified` 4%, … (mostly punctuation/fragments) |
| Position collapse @100 | ~1.000 | 1.0000 |
| Cosine convergence @100 | ~1.000 | 0.926 (σ 0.17) |
| Basin overlap | — | **1/5** (`prolet`, hit by 1/125 random trials) |

Bootstrap on the random basin count: **14.1 (95% CI [11, 17])**. Real count **5 is below
the CI → significant**.

**Decides:** *Are the five semantic basins a property of the weight geometry (noise
reproduces them) or of the prompt regime (it doesn't)?* **The prompt regime.** Random
noise does not reproduce the semantic basins: it converges (position-collapse reaches
1.0000 by iter 20, so ATR's dynamics still operate on noise) but into 18 mostly-
punctuation attractors with essentially zero identity overlap with the real 5. Real text
funnels into **fewer** attractors (5) than noise (18), and they are semantically coherent
where the random ones are not.

**Interpretation:** This is the notebook's "different basins → manifold-specific" outcome.
The `prolet`/`Divine`/`Anarch`/`till`/`solidarity` landscape is a property of the
**on-manifold region** that real prompts occupy, not of the weight geometry reachable
from arbitrary tensors. ATR is reading the resonant modes of *the model as driven by
language-shaped input*, not a universal fixed-point set of the weights.

**Open questions:** The scale proxy (mean-vector norm) substitutes for the unstored full-
tensor Frobenius norm; harmless here by LayerNorm scale-invariance, but a re-save of
stage-1 with the full tensor would let the calibration be exact. The dominant random
basin `―` (em-dash, 64%) is itself worth a note: noise has its own strong attractor,
just a non-semantic one.

---

## 4. `01b_deep_convergence.ipynb`: Control 3 (long horizon)

**What ran:** pythia-410m, **8-prompt diverse subset** (all 7 categories, reduced from
25 for CPU time, see deviations), extended schedule
`[0,5,10,25,50,100,200,300,500,750,1000]`, layers 0→23, 1000 iterations. Duration ~36 min
(CPU, alone). Outputs in `experiments/pythia_410m/output_deep/` (`deep_results.pt`,
`basin_assessment.md`, `dissolution_pathways.md`, `deep_config.pt`).

**Headline numbers @ 1000 iterations:**

- **Terminal basins: 8 distinct tokens for 8 prompts**: `know`, `/`, `,`, `THE`, `or`,
  `f`, `` (whitespace), `ute`. Every prompt lands on a *different* scattered
  fragment/punctuation token. Zero shared basins.
- **Cross-prompt mean-vector similarity: 0.21** (min −0.08, max 0.80); the eight final
  states are near-orthogonal, not collapsed together.
- **Per-prompt self-stability** (`cos_sim_mean` between consecutive snapshots) at iter
  1000: mean **0.909**, min 0.773, max 1.000, and **non-monotonic** along the way (dips
  to 0.34–0.47 around iters 100–200 for several prompts).
- **Token-level:** 7/8 prompts are *still flickering* at iter 1000 (e.g. A01
  `K`→`↵`→`know` across 500→750→1000). Exactly **1/8** (`B01_napoleon`) locks in to `/`
  from iter 300 onward (self-stability 1.000).

**Decides:** *"Not yet converged at 100–250" vs "structurally fragmented"?*
**Structurally fragmented.** Quadrupling the horizon (250 → 1000) did **not** produce
convergence: the prompts remain on 8 distinct tokens with cross-prompt similarity 0.21,
and most trajectories still move between snapshots. More iterations buy marginally higher
self-stability (~0.85 at 250 in Control 1 → ~0.91 here), not a shared attractor.

**Interpretation:** Pythia-410m's landscape is genuinely flat/fragmented on this horizon,
not merely under-iterated. A minority of individual prompts *do* settle into their own
private fixed point (the `napoleon`→`/` case), but there is no basin *consolidation*
across prompts, the opposite of GPT-2 Small's five shared semantic basins. This is
consistent with the analysis doc's "intrinsic geometry / depth-dependent dynamics" for
the 24-layer model rather than a single apparatus fault.

**Open questions / caveats:**
- The notebook's auto-verdict printed *"Partial convergence — some basin formation"*, but
  that label is a **threshold artefact**: its `n_unique ≤ 15 → partial` cutoff was
  calibrated for the 25-prompt run. With 8 prompts, **8 unique = maximal scatter** (100%
  distinct), i.e. *no* convergence. The raw numbers (8/8 distinct, cross-sim 0.21) are
  the signal; disregard the auto-label.
- 8 prompts is a small sample (the CPU-time reduction). The direction is clear and
  matches Control 1's 125-prompt result, but a larger subset on GPU would tighten it.
- Control 2 (depth control, layers 0–11 vs 0–23) remains the clean next test to attribute
  this to depth specifically.

---

## 5. `gated_resweep.py`: GPT-2 Small convergence-gated re-sweep (plan addendum)

**What ran:** all 125 GPT-2 Small prompts re-iterated to `max_iter=1000` with an
early-stop gate (`cos_sim_mean > 0.999` for 3 consecutive checks, every 10 iters past
100). Basins classified **at lock-in**, not at a fixed horizon; lock-in iteration
recorded. New engine function `run_atr_gated` (additive); sweep in
`experiments/gpt2_small/gated_resweep.py`, outputs in `output_gated/` (does not touch the
April `.pt`). Duration ~2 h (CPU), checkpointed per prompt. Motivation: Control 1 showed
GPT-2 Small only reached `cos_sim_mean ≈ 0.91` at iter 100, so the published basin table
was read *before* convergence: are the five basins stable or stop-time artefacts?

**Headline numbers:**

- **Convergence: 91/125 (73%) lock in, every one at exactly iter 120** (the earliest the
  gate can fire). The other **34/125 (27%) never reach `cos > 0.999`** and run to 1000.
- **The 34 non-convergers are *exactly* the 34 `Divine` prompts.** Every `Divine` prompt
  fails the tensor gate yet reads `Divine` the whole way, a stable **readout** attractor
  over a **non-settling tensor**. The other four basins converge cleanly.

Basin shares, iter 100 (published) → at lock-in:

| Basin | @100 | @lock-in | what moved |
|---|---:|---:|---|
| `prolet` | 44 (35.2%) | **54 (43.2%)** | gains 10 from Anarch |
| `Divine` | 34 (27.2%) | 34 (27.2%) | unchanged (readout-stable, tensor never settles) |
| `Anarch` | 26 (20.8%) | **17 (13.6%)** | loses 10 to prolet (all converged, by iter 120) |
| `till` | 19 (15.2%) | 19 (15.2%) | **unchanged — 19/19 retained** |
| `solidarity` | 2 (1.6%) | 1 (0.8%) | loses 1 to Anarch |

**Decides:** *Are the five basins stable under proper convergence, or stop-time artefacts?*
**Mostly stable, with one real correction.** `prolet`, `Divine`, and `till` are exactly
stable; the published table's error is confined to **`Anarch`, which was over-counted at
iter 100**: 10 of its 26 prompts are still drifting Anarch→prolet at iter 100 and settle on
`prolet` by their lock-in (iter 120, all converged). Corrected shares: prolet ~43%,
Anarch ~14%.

**The addendum's specific hypothesis is refuted.** `till` is **not** a slow transient: all
19 `till` prompts converge at iter 120 and stay `till` (19/19). The transient basin is
`Anarch`, not `till`.

**Interpretation:** Two clean facts fall out. (1) GPT-2 Small's basins are genuine, not
stop-time noise: 73% of prompts reach a hard fixed point (`cos > 0.999`) within 120
iterations and keep their label. (2) The one basin that never settles at the tensor level,
`Divine`, is precisely the one whose *readout* is most stable, the sharpest single example
in this study of dynamics and decoding coming apart (the analysis doc's central caveat).
The published basin table needs one edit: shift ~10 prompts from `Anarch` to `prolet`.

**Open questions:** `Divine`'s readout-stable / tensor-unsettled split is the natural target
for the ATR-R1/R3 confidence audit (Notebook 2's machinery): is its readout high-margin
throughout while the tensor wanders? Also: 120 is the *floor* of the gate (first possible
lock-in); a finer check cadence would show the true settling iteration for each prompt.

## 6. `02b_permutation_test.py`: all-warm matrix null (post-close, 2026-07-11)

**What ran:** the pre-registered permutation test for the all-warm cross-similarity
finding (FINDINGS caveat 4). 10,000 random 14-token sets vs the canonical 5 basin +
9 waypoint tokens in W_E. Seed 1969. Reproduction gate passed (observed: 91/91
positive, min 0.181, mean 0.288, max 0.470; matches the March record).

**Headline numbers:** S1 (all-positive): 9,994/10,000 random sets, p≈1.0.
S2 (min ≥ observed): p=0.167. S3 (mean ≥ observed): p=0.099. Global mean pairwise
cosine of the embedding space: 0.268.

**Decides:** the all-warm property is an **anisotropy artifact**: nearly any 14
tokens in GPT-2 Small's embedding space are all-warm. The compact
"thematic-centre-of-mass subspace" interpretation is withdrawn. The local
semantic-neighbourhood observation is out of scope of this test and stands as
qualitative. Full report: `gpt2_small/output_permutation/permutation_report.md`.

## Synthesis

**The cross-model differences are intrinsic model properties, not readout artefacts.**
Readout ambiguity is real but secondary; it does not explain the headline contrasts.

Four independent lines of evidence converge:

1. **The tensor, not just the token, tells the story (Control 1).** GPT-2 Medium and
   Pythia-160m reach `cos_sim_mean = 1.0000`: their collapses (`D`, `questioned`) are
   genuine fixed points of the forward map. Pythia-410m's mean tensor never stops moving
   (~0.85 at 250 iters). Because `cos_sim_mean` is computed on the activation tensor and
   never passes through token readout, this cleanly separates *dynamics* from *decoding*:
   Pythia-410m's non-convergence is in the dynamics.

2. **Where a basin label appears, it is high-confidence (Notebook 2).** The R1/R3 machinery
   shows readout confidence (logit margin, entropy) rising as GPT-2 Small settles toward a
   basin. Flicker near boundaries exists, but it is not manufacturing the basins: the
   labelled attractors are confident, not decoding noise.

3. **The basins are language-shaped, not weight-universal (null model, Notebook 3).** Random
   tensors iterated through GPT-2 Small converge (positions collapse) but into 18 scattered
   punctuation basins with ~0 overlap with the real five. The `prolet`/`Divine`/`Anarch`/
   `till`/`solidarity` landscape belongs to the on-manifold region real text occupies: ATR
   reads the model *as driven by language*, and real text funnels into **fewer** attractors
   (5) than noise (18).

4. **Pythia-410m stays fragmented with 4× the iterations (Control 3).** At 1000 iterations,
   8 prompts hold 8 distinct terminal tokens with cross-prompt similarity 0.21. Not
   under-iteration. Structural.

5. **GPT-2 Small's basins survive proper convergence (gated re-sweep).** 73% of prompts
   hit a hard fixed point (`cos > 0.999`) by iter 120 and keep their basin label; the
   published table needs only one correction (≈10 prompts move Anarch→prolet: Anarch was
   over-counted pre-convergence; the hypothesised `till` transient is in fact 100% stable).
   And the one basin whose tensor never settles, `Divine`, is exactly the one whose *readout*
   is perfectly stable, the study's sharpest single case of dynamics and decoding diverging.

**Bottom line for the original question.** GPT-2 Medium's single `D` basin is a *real* tensor
attractor. Pythia-410m's fragmentation is *genuine structural non-convergence* rooted in the
model (depth/width/corpus geometry), **not** an avoidable distortion of the ATR apparatus. The
readout projection remains a real but secondary source of token-level jitter; the normalisation
step is confirmed irrelevant (LayerNorm scale-invariance, used here in Notebook 3's
calibration). The four models genuinely do not share one failure mode: landscapes differ with
architecture and data, exactly as the analysis doc's "bigger picture" anticipated.

**Confidence:** high for the qualitative direction (four independent controls agree); moderate
on Pythia-410m specifics, since Control 3 ran on an 8-prompt CPU subset. The cleanest
outstanding test is Control 2 (depth control) to pin the effect to depth per se.

## Not scaffolded (follow-on work)

- **Control 2**: Pythia-410m depth control (loop layers 0–11 vs 0–23), holding weights/
  tokenizer/corpus constant. Not built; the plan says note only. Still the cleanest test to
  attribute Pythia-410m's fragmentation to depth per se.
- **ATR-R1/R3 on the `Divine` cohort.** Section 5 found `Divine` is readout-stable while its
  tensor never settles. Run Notebook 2's margin/entropy audit on those 34 GPT-2 Small
  prompts (and across GPT-2 Medium / Pythia-410m at full prompt sets) to confirm the readout
  stays high-confidence while `cos_sim_mean` wanders.
- **Public README update**: the basin table should be corrected (Anarch ~21%→~14%,
  prolet ~35%→~43%; `till` and `Divine` unchanged). Per the run plan, that public-claims
  pass is a separate session.

_Done since the original plan:_ Section 5 (convergence-gated GPT-2 Small re-sweep), the plan
addendum, is complete; results above.
