# EXP_009 Regeneration Report — 2026-07-25

## What this is

Re-execution of two notebooks to regenerate the lost EXP_009 output artifacts
(issue #25, both halves):

1. `experiments/gpt2_small/head_resonance.ipynb` (EXP_009bFIX) — recovered
   verbatim from git history (`d91049e^:ActivationTensorResonance_Head/head_resonance.ipynb`)
   and executed end to end.
2. `experiments/gpt2_small/spectral_resonance.ipynb` (EXP_009c) — executed end
   to end against the regenerated 009b data; both the spectral-prediction half
   and the empirical-comparison half ran.

`experiments/gpt2_small/layer_resonance.ipynb` was also recovered from
`d91049e^:ActivationTensorResonance_Layer/layer_resonance.ipynb` for the record
but was not executed.

**These are 2026-07-25 regenerated artifacts, not the March originals.** The
original outputs were lost because they existed only locally under the
gitignored `_DATA/` directory. Package versions and hardware differ from the
March run; numbers here should be attributed to this re-execution only.

## Authorisation

`docs/ATR_PAUSE.md` pauses new ATR experiments. The operator (TC) ruled on
2026-07-25 that re-executing the March notebook `head_resonance.ipynb` to
regenerate its lost output artifacts is **artifact regeneration, not a new
experiment**, and is explicitly permitted under the pause.

## Environment

| Component | Version |
| :--- | :--- |
| Python | 3.11.15 |
| torch | 2.13.0+cpu |
| transformer_lens | 3.5.1 |
| transformers | 5.14.1 |
| plotly | 6.9.0 |
| nbformat | 5.10.4 |
| Hardware | CPU only (Linux 6.18.5) |
| Model | gpt2-small via `HookedTransformer.from_pretrained` (huggingface.co download) |

Execution command (both notebooks):
`jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=7200`

## Durations

| Run | Wall time |
| :--- | :--- |
| `head_resonance.ipynb` (all cells, incl. model download) | 1 min 42 s |
| `spectral_resonance.ipynb`, first attempt | failed at STEP 3 after 15 s (see Deviations) |
| `spectral_resonance.ipynb`, final run | 26 s |

## Artifacts

All under `experiments/_DATA/EXP_009/` (the notebooks' default
`../_DATA/EXP_009` save_dir, unchanged); committed via dated `.gitignore`
negation entries.

| File | Size | Producer |
| :--- | ---: | :--- |
| `009bFIX_head_loop_results.pt` | 4,734,891 B (4.6 MB) | head_resonance STEP 7 |
| `009bFIX_convergence_grid.pt` | 3,155 B | head_resonance STEP 7 |
| `009c_spectral_data.pt` | 1,124,241 B (1.1 MB) | spectral_resonance STEP 5 |
| `009c_validation_grid.pt` | 3,067 B | spectral_resonance STEP 5 |

Plus the two executed notebooks themselves (outputs embedded in-place).

## Deviations (complete list)

1. **`.gitignore` negation entries instead of moving save_dir.** Root
   `.gitignore` ignores both `_DATA/` and `*.pt`. Rather than editing the
   notebooks' save/load paths, dated negation entries were appended
   re-including `experiments/_DATA/EXP_009/` and the four named artifact
   files. Chosen as the smaller change: the spectral notebook's STEP 2 load
   path then needed no modification.
2. **`spectral_resonance.ipynb` STEP 3: `.detach()` added** to the `W_OV`
   extraction line. Under transformer_lens 3.5.1 the weight tensors carry
   `requires_grad=True`; `S.numpy()` raised
   `RuntimeError: Can't call numpy() on Tensor that requires grad` without it.
   No numerical effect.
3. **`spectral_resonance.ipynb` STEP 5: `.clone()` added** to
   `dominant_vector` before `torch.save`. The vector is a view of the full
   768×768 `Vh` matrix and `torch.save` serialises each view's entire backing
   storage, producing a 340 MB file — over GitHub's 100 MB per-file limit.
   Cloned, the artifact is 1.1 MB. Identical stored values.
4. **Junk file `experiments/gpt2_small/=4.2.0` deleted and gitignored.**
   Byproduct of head_resonance cell 0's unquoted
   `pip install nbformat>=4.2.0` (the shell redirects into a file named
   `=4.2.0`). Added to `.gitignore` with a dated comment.
5. **No other code changes.** `head_resonance.ipynb` executed unmodified.
   `layer_resonance.ipynb` was not executed.
6. **Report location:** artifacts landed at the notebooks' default
   `experiments/_DATA/EXP_009/`; this report is kept at the prescribed
   `experiments/gpt2_small/output_head_loop_regen/` (the repo's `output_*`
   convention, adjacent to the notebooks) rather than inside the data
   directory.

## Observations (numbers only)

### head_resonance (EXP_009b regeneration)

- 144/144 heads scanned (12 layers × 12 heads), probe prompt
  `"Am I sitting in a room different from the one you are in now"`,
  500 iterations, snapshot schedule [0, 2, 3, 5, 10, 20, 50, 100, 250, 500].
- Final cosine similarity to previous iterate (iter 500), raw values:
  min −1.000000, mean 0.345219, median 0.994028, max 1.000000.
- Raw final cos > 0.999: 61/144 heads. Raw final cos < −0.999: 34/144 heads.
  Heads ending at negative final cos (sign-alternating iterates): 47/144.
- Absolute final |cos|: min 0.100424, mean 0.965145, median 1.000000.
  |cos| > 0.999: 95/144. |cos| > 0.99: 115/144. |cos| > 0.9: 134/144.
  |cos| > 0.5: 140/144.
- L9.H9 cosine-to-previous trajectory over the schedule:
  1.0000, 0.1438, 0.3993, 0.7628, 0.9878, 0.9987, 1.0, 1.0, 1.0, 1.0;
  final top token `SF` (p = 0.4713).

### spectral_resonance (EXP_009c)

- SVD computed for all 144 W_OV matrices.
- Spectral gap σ₁/σ₂: min 1.0012 (L7.H9), mean 1.3511, median 1.1624,
  max 11.4872 (L11.H8).
- Top singular value σ₁ across heads: min 1.3913, max 334.8713.
- Predicted-vs-empirical validation grid, |cos(empirical final vector,
  dominant right singular vector)| over 144 heads:
  mean 0.2387, median 0.1614, min 0.0008, max 0.9997.
- Heads with |cos| > 0.9: 5/144. > 0.95: 2/144. > 0.7: 8/144. > 0.5: 18/144.
- The notebook's pre-registered threshold classifier printed:
  `NOT SUPPORTED: loop dynamics are dominated by nonlinear effects for most
  heads.` (mean |cos| 0.2387 against the pre-registered 0.7/0.9 thresholds).

Assessment of these numbers against H4 is the operator's call on review;
`docs/FINDINGS.md` and `docs/JOURNEY_MAP.md` were deliberately not edited.

## Post-review addendum (2026-07-25, PR #29 review)

Known issue, recorded not patched: `spectral_resonance.ipynb`'s
`get_top_tokens` decodes without applying `ln_final` before unembedding
(unlike the fixed sibling in `head_resonance.ipynb`). This affects ONLY the
qualitative token map (section 4d) and the `top_tokens` field persisted in
`009c_spectral_data.pt` — the near-uniform `,`/`the` map is consistent with
unnormalized logits collapsing onto high-frequency tokens. It does NOT
affect the cosine-similarity validation (section 4c) or the `NOT SUPPORTED`
classifier output, which compare raw vectors. Per the regeneration scope
(execution-blocking fixes only, no result patching), the corrected decode is
deferred to a tracked follow-up re-run (issue #25).

Housekeeping applied in the same review round: pip version specs quoted in
both recovered notebooks (removes the `=4.2.0` junk-file byproduct and its
gitignore workaround); `.gitignore` negation added for
`009a2_layer_scan_results.pt` so a future layer_resonance run cannot be
silently untracked.
