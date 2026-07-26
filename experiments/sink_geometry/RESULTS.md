# Sink geometry at the ATR read site, and what it does to `cos_sim_mean`

**Run:** 2026-07-26, CPU, `torch` 2.x + `transformer-lens`, four models.
**Status:** executed; outputs in `output/`, trajectories committed.
**Origin:** board thread [#60](https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments/discussions/60), opened as a hypothesis and **refuted by this run**.

---

## The hypothesis, and its fate

Transformers carry a few residual-stream coordinates — *massive activations*, tied to attention
sinks — that are far larger than the rest and roughly state-independent (Sun et al.,
[arXiv:2402.17762](https://arxiv.org/abs/2402.17762)). Writing consecutive iterates as `x = c + a`,
`y = c + b` with `|c| >> |a|,|b|`:

```text
cos(x, y) = (|c|² + c·(a+b) + a·b) / (|x||y|)  ≈  1 + small
```

**Predicted:** the shared bulk drags `cos_sim_mean` toward 1, so the models that saturate to 1.0000
(GPT-2 Medium, Pythia-160m) are inflated, and masking those coordinates would knock them down.

**Found:** the saturating models are completely unaffected, and the one model that *does* move goes
the **opposite way**. The prediction was wrong in its sign.

---

## 1. Sink structure at `blocks.{n_layers-1}.hook_resid_post`

12 natural prompts from `prompt_library.py`, no iteration. Coordinates ranked by mean |activation|
over content positions. (`01_sink_profile.py` → `output/sink_profile.json`.)

| model | max/median coord | top-10 coords shared by **all** prompts | energy in top-10 |
|---|---|---|---|
| gpt2 | 42.8× | **10/10** | **90.8%** |
| gpt2-medium | 71.4× | 8/10 | **93.4%** |
| pythia-160m | 1.5× | **0/10** | 2.3% |
| pythia-410m | 20.8× | 3/10 | 42.4% |

Position 0 is excluded from these profiles **only where it is BOS** — i.e. for the GPT-2 models
(see §2). An earlier version of `01_sink_profile.py` dropped it unconditionally, which silently
removed a real content token from every Pythia profile; that is fixed, and the Pythia-410m figures
above are the corrected ones (they were 18.7× and 40.4% before).

**Massive activations at the ATR read site are a GPT-2 phenomenon.** GPT-2 Small puts over 90% of
its final-layer residual energy into ten of 768 coordinates, and they are *the same ten for every
prompt*. Pythia-160m has essentially no such structure — 2.3% in its top ten, no coordinate shared
across prompts, max only 1.5× the median.

**Consequence for cross-model work:** any metric computed on the raw residual stream compares a
~90%-sink-dominated space against a ~2% one. That asymmetry is real regardless of what it does to
any particular metric.

**The sink is coordinate-structured, not position-structured.** Position-0 norm relative to the mean
over remaining positions is 1.003× (gpt2) and 0.369× (gpt2-medium) — the BOS position carries no
unusual norm in either. "The L2 renormalisation rescales whatever sits at position 0" is not the
sharp worry; "the L2 renormalisation is dominated by ten coordinates" is. (For the Pythia models the
same ratio is 0.65 and 0.60, but there position 0 is an ordinary content token, so it is not a
statement about sinks at all.)

---

## 2. BOS: the two arms of the 2×2 do not tokenise the same way

`atr_engine.py` passes a bare string to `run_with_cache`, so tokenisation is whatever
TransformerLens defaults to — and that default **differs by model family**:

| model | raw HF tokens | via `run_with_cache(str)` | BOS prepended? |
|---|---|---|---|
| gpt2 | 4 | 5 | **yes** — `['<\|endoftext\|>', 'The', ' quick']` |
| gpt2-medium | 4 | 5 | **yes** |
| pythia-160m | 4 | 4 | **no** — `['The', ' quick', ' brown']` |
| pythia-410m | 4 | 4 | **no** |

TransformerLens sets `default_prepend_bos=True` for GPT-2 and `False` for the NeoX family, because
Pythia was not trained on BOS-prefixed sequences.

**So in every ATR run: the GPT-2 arm has `<|endoftext|>` at position 0; the Pythia arm has an
ordinary content token there.** Nobody chose this and it is invisible at the call site. Every
position-indexed cross-model comparison — finding 2's position uniformity above all — is comparing
sequences whose position 0 means different things.

`experiments/gpt2_small/11_suppression_test.py:607` already assumed the GPT-2 half of this
(`# [1, L], BOS at 0`, and `n_tokens_no_bos: L - 1`) and was right.

---

## 3. The masking control

Trajectories regenerated with this repo's own `atr_engine.run_atr_loop` — the April
`stage1_results.pt` sweeps are gitignored and absent from a fresh clone. 5 prompts × 60 iterations,
layers 0 → n_layers-1. Metric is the engine's own `cos_sim_mean`
(`cosine_similarity(mean_vec_t, mean_vec_{t-1})`, `atr_engine.py:198`), averaged over the last 10
transitions. Mask fixed per trajectory as the coordinates with largest mean |value| across the run.
(`02_masking_control.py` → `output/masking_control.json`.)

| model | energy in top-10 | unmasked | mask top-1 | mask top-10 | mask top-50 |
|---|---|---|---|---|---|
| gpt2 | 70.8% | **0.9167** | **0.9933** | 0.9847 | 0.9939 |
| gpt2-medium | 92.3% | 0.999999926 | 1.000 | 1.000 | 1.000 |
| pythia-160m | 5.5% | 1.000000006 | 1.000 | 1.000 | 1.000 |
| pythia-410m | 11.4% | 0.7552 | 0.7537 | 0.7690 | 0.7586 |

**On the saturating figures.** They are reported here to twelve significant figures precisely because
five would hide the question. Neither is exactly 1: GPT-2 Medium's is `1 − cos = 7.4e-08`, and
Pythia-160m's reads `−6.0e-09`, i.e. marginally *above* 1, which is only possible as round-off. Both
sit at float32 epsilon, so the defensible claim is **convergence to numerical precision**, not exact
equality — and the distinction does not affect any conclusion below.

**GPT-2 Medium and Pythia-160m: convergence is genuine.** GPT-2 Medium holds 92.3% of its energy in
ten coordinates and still returns 1.000 to within round-off after those ten are deleted — and after
fifty. The remaining 8% of the energy is itself aligned step to step at the same precision. This is
the stronger of the two claims in `SCALING_ARTEFACT_ANALYSIS.md`'s closing judgement, and it passes a
test built to break it.

**Pythia-410m: unmoved.** ±0.014 across every masking. Non-convergence is not sink geometry.

**GPT-2 Small: the real finding, and it is the reverse.** Unmasked 0.9167; deleting **a single
coordinate** gives 0.9933. The dominant coordinates were *depressing* apparent convergence by ~0.077.

### Why the sign was wrong

The algebra assumed `c` is shared *and constant between consecutive iterates*. On a natural forward
pass that is an excellent assumption — GPT-2's top-10 coordinates are identical across all 12
prompts. But **under ATR iteration the dominant coordinates are the ones carrying the dynamics**, not
the ones sitting still. A large coordinate that *changes* between iterates contributes a large
disagreement term and pushes cosine down. Static-geometry intuition does not transfer to the
iterated regime.

---

## What this changes

1. **GPT-2 Small's convergence is under-reported by the raw metric**, and one coordinate accounts
   for most of the gap. A trajectory a lag-1 gate scores at 0.92 sits at 0.99 once that coordinate
   is removed. This bears on gating thresholds and basin-membership decisions for the model with the
   five semantic basins — plausibly tightening them.
2. **The BOS asymmetry is a live confound** in any position-indexed cross-model claim, and is one
   line to fix (`prepend_bos=` explicit at the call site) once someone decides which way it should go.
3. **The 1.0000 saturations are safe.** Recorded so nobody re-runs this control expecting otherwise.

## Caveats

- 5 prompts, 60 iterations, single seed. **Not** the 125-prompt / 250-iteration April sweeps; the
  Pythia-410m plateau here reads 0.755 against the record's ~0.85 — directionally consistent, not
  the same run.
- Masking is applied to the mean-pooled vector, matching `cos_sim_mean`. Per-position metrics
  (`position_similarity`, `cos_sim_last`) untested.
- Sink profile in §1 is measured on natural forward passes; §3 masks are computed from the ATR
  trajectories themselves. The two rankings need not agree, and the difference between them is
  precisely what §3's "why the sign was wrong" is about.
- No seed variation. Whether the single dominant GPT-2 Small coordinate is a property of the
  architecture or of this checkpoint is exactly the question the seed controls would answer.

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformer-lens
python3 experiments/sink_geometry/01_sink_profile.py      # ~10 min CPU, downloads 4 models
python3 experiments/sink_geometry/02_masking_control.py   # reuses output/trajectories.pt if present
```

`02` caches trajectories to `output/trajectories.pt` (committed, ~4.2 MB) so the masking analysis can
be re-run without another sweep. The cache carries a **manifest** — model list, `MAX_ITER`, prompt
count, a hash of the prompt text, and a hash of `atr_engine.py` — and regenerates itself on any
mismatch, printing which field changed. So editing the config or the engine cannot silently republish
stale measurements under new-looking parameters. Delete the file to force a rebuild regardless.

The cache is loaded with `weights_only=True, map_location="cpu"`. It is a committed artifact, so it
is a file a third party can modify, and an unrestricted `torch.load` unpickles — that is, executes.

**Note on `transformers` 5.x:** it renamed GPTNeoX's `embed_out` to `lm_head`, which the pinned
TransformerLens release still reaches for. Both scripts alias it before conversion; without that,
Pythia loading raises `AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out'`.
