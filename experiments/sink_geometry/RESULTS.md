# Do a few oversized dimensions fake the convergence numbers?

**Short answer: no — and on GPT-2 Small they do the opposite, hiding convergence that is really there.**

**Run:** 2026-07-26, CPU, four models. **Status:** executed; results in `output/`.
**Origin:** board thread [#60](https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments/discussions/60) — opened as a worry, **disproved by this run**.

---

## In plain terms

At each token position, the model's internal state is a list of numbers — 768 of them in GPT-2 Small,
1024 in GPT-2 Medium. Call each slot in that list a **dimension**.

It turns out that in GPT-2, about ten of those 768 dimensions are enormous compared to the rest, and
they stay enormous no matter what you feed the model. Everything else is small.

That matters because this project measures convergence — "has the state stopped moving?" — by
comparing one step to the next with cosine similarity, which sums up agreement across all 768
dimensions at once. If ten dimensions are huge and always roughly the same, they might dominate that
sum and make any two states look alike, whether or not the model had actually settled.

**The worry:** the models reported as fully converged were being flattered by ten big dimensions.

**The test:** delete those dimensions and recompute. If convergence was fake, it collapses.

**The result:** it doesn't collapse. GPT-2 Medium still reads 1.000 after deleting the ten biggest
dimensions — and after deleting fifty. Its convergence is real. But GPT-2 Small moved in the
*opposite* direction: it looks *more* converged with its biggest dimension removed, not less. So the
big dimensions were **hiding** convergence there rather than inventing it.

---

## Terms used here

| Term | What it means |
|---|---|
| **dimension** | One slot in the model's internal list of numbers. GPT-2 Small has 768 of them. |
| **residual stream** | That list of numbers — the model's running internal state, which each layer reads and writes. It is what ATR extracts and re-injects. |
| **massive activations** / **sinks** | The handful of dimensions that are far larger than the rest and barely change from prompt to prompt. A documented transformer phenomenon (Sun et al., [arXiv:2402.17762](https://arxiv.org/abs/2402.17762)). |
| **share of total size** | How much of the state's overall magnitude sits in a given set of dimensions. Measured as squared values, since that is what cosine similarity effectively weighs by. |
| **BOS** | "Beginning of sequence" — a special marker token some models put at the front of every input. |
| **masking** | Deleting chosen dimensions before recomputing a number, to see whether they were driving it. |

---

## The worry, written out

Two consecutive steps can each be split into a shared part and a distinctive part: `x = c + a` and
`y = c + b`, where `c` is the big always-present chunk and `a`, `b` are the small parts that actually
differ. When `c` is much larger than `a` and `b`:

```text
cos(x, y) = (|c|² + c·(a+b) + a·b) / (|x||y|)  ≈  1 + small
```

The shared bulk pushes the answer toward 1 on its own. **Predicted:** the two models that reach
1.0000 are inflated, and deleting the big dimensions would drag them down.

**Found:** those two models don't move at all, and the one that does move goes the other way. The
prediction was wrong about the direction, not just the size.

---

## 1. Are there oversized dimensions, and where?

Measured at the point ATR reads from — the last layer's output. 12 ordinary prompts from
`prompt_library.py`, no iteration. (`01_sink_profile.py` → `output/sink_profile.json`.)

| model | biggest ÷ typical dimension | same top-10 for **all** prompts? | share of total size in top 10 |
|---|---|---|---|
| gpt2 | 42.8× | **10 of 10** | **90.8%** |
| gpt2-medium | 71.4× | 8 of 10 | **93.4%** |
| pythia-160m | 1.5× | **0 of 10** | 2.3% |
| pythia-410m | 20.8× | 3 of 10 | 42.4% |

**This is a GPT-2 phenomenon.** In GPT-2 Small a typical dimension holds a value around 3.85; the
biggest holds around 164.8. Ten of its 768 dimensions carry over 90% of the total, and they are *the
same ten every time* — like ten dials stuck near maximum while the other 758 do the varying work.

Pythia-160m has essentially none of this: typical 3.75, biggest 5.5, its top ten hold 2.3%, and the
ten differ for every prompt.

**Why that matters beyond this experiment:** any measurement taken on the raw internal state is
comparing a ~90%-dominated space on the GPT-2 side against a ~2% one on the Pythia side. That
lopsidedness is real whatever it does to any particular number.

**It is about dimensions, not positions.** Position 0's size relative to the rest of the sequence is
1.003× (gpt2) and 0.369× (gpt2-medium). Neither is unusually *large*, which is what a
position-based sink would look like. GPT-2 Medium's 0.369× is a genuine departure — but *smaller*
than the rest, so it points away from position 0 being the sink, not toward it. (Pythia's equivalents
are 0.65 and 0.60, but there position 0 is an ordinary word, so it says nothing about sinks — most
likely just the first token having less context behind it.)

> **Correction on record.** An earlier version of `01_sink_profile.py` skipped position 0 for every
> model, to "skip BOS". Pythia has no BOS (§2), so that silently threw away a real word from every
> Pythia measurement. Fixed. The Pythia-410m figures above are the corrected ones — they read 18.7×
> and 40.4% before.

---

## 2. The two model families don't tokenise the same way

`atr_engine.py` hands a plain text string to the model, so how it gets split into tokens is left to
the TransformerLens default — and **that default differs by model family**:

| model | tokens from the raw tokenizer | tokens via the ATR path | BOS added? |
|---|---|---|---|
| gpt2 | 4 | 5 | **yes** — `['<\|endoftext\|>', 'The', ' quick']` |
| gpt2-medium | 4 | 5 | **yes** |
| pythia-160m | 4 | 4 | **no** — `['The', ' quick', ' brown']` |
| pythia-410m | 4 | 4 | **no** |

TransformerLens adds BOS for GPT-2 and not for Pythia, because Pythia wasn't trained with one.

**So in every ATR run, the GPT-2 models have a special marker token sitting at position 0 and the
Pythia models have a real word there.** Nobody chose this, and you cannot see it at the point the
code calls the model. Any comparison that lines models up position-by-position — finding 2's
position uniformity most of all — is comparing sequences whose position 0 means different things.

`experiments/gpt2_small/11_suppression_test.py:607` already assumed the GPT-2 half of this
(`# [1, L], BOS at 0`) and was right.

---

## 3. The test

Trajectories regenerated with this repo's own `atr_engine.run_atr_loop`, because the April
`stage1_results.pt` sweeps are gitignored and not in a fresh clone. 5 prompts × 60 iterations, full
layer stack. The number measured is the engine's own `cos_sim_mean` (`atr_engine.py:198`), averaged
over the last 10 steps. For each run the dimensions to delete are chosen once, as those with the
largest average size across that run. (`02_masking_control.py` → `output/masking_control.json`.)

| model | share in top 10 | nothing deleted | biggest 1 deleted | biggest 10 deleted | biggest 50 deleted |
|---|---|---|---|---|---|
| gpt2 | 70.8% | **0.9167** | **0.9933** | 0.9847 | 0.9939 |
| gpt2-medium | 92.3% | 0.999999926 | 1.000 | 1.000 | 1.000 |
| pythia-160m | 5.5% | 1.000000006 | 1.000 | 1.000 | 1.000 |
| pythia-410m | 11.4% | 0.7552 | 0.7537 | 0.7690 | 0.7586 |

**About those 1.000 figures.** Shown to twelve digits on purpose, because five would hide the
question. Neither is exactly 1: GPT-2 Medium is short by 0.000000074, and Pythia-160m reads a hair
*above* 1, which can only be rounding error. Both sit at the limit of what 32-bit arithmetic can
represent, so the honest phrasing is **"converged as precisely as the arithmetic can show"**, not
"exactly equal". Nothing below depends on the difference.

**GPT-2 Medium and Pythia-160m — the convergence is real.** GPT-2 Medium keeps 92.3% of its total
size in ten dimensions and *still* reads 1.000 after those ten are deleted, and after fifty. The
remaining 8% agrees from step to step on its own. This is the stronger of the two conclusions in
`SCALING_ARTEFACT_ANALYSIS.md`, and it passed a test built to break it.

**Pythia-410m — unmoved.** At most ±0.014 across every deletion. Its failure to converge is not a
side effect of big dimensions.

**GPT-2 Small — the actual finding, and it runs backwards.** 0.9167 with nothing deleted; **0.9933
with a single dimension removed.** The big dimensions were *suppressing* the convergence score by
about 0.077.

### Why the prediction had the direction wrong

The reasoning assumed the big chunk `c` is not only shared but *unchanging between consecutive
steps*. On an ordinary forward pass that holds well — GPT-2's top ten are the same ten across all 12
prompts. But **once ATR starts iterating, the big dimensions are the ones doing the moving.** A large
dimension that *changes* between steps contributes a large disagreement, pushing the score down.
Intuition from the static picture does not carry over to the looping one — which, in hindsight, is
the whole premise of this project.

---

## What this changes

1. **GPT-2 Small's convergence is understated by the current measurement**, and a single dimension
   accounts for most of the gap. A run that scores 0.92 sits at 0.99 once it is removed. That bears
   on where convergence thresholds are set and which prompts count as belonging to which basin — for
   the one model with five semantic basins. It likely tightens both.
2. **The BOS mismatch is a live problem** for any position-by-position cross-model claim. Fixing it
   needs no change to the engine — but it is *not* `prepend_bos=` at the experiment call site, which
   is what an earlier version of this document said. `run_atr_loop` passes `prompt` straight through
   to the model at `atr_engine.py:125` and `:184` and has no such option. What works is **tokenising
   first and passing the tokens**, which those calls accept just as happily:

   ```python
   toks = model.to_tokens(text, prepend_bos=False)   # or True
   snaps = run_atr_loop(model, toks, 0, model.cfg.n_layers - 1, max_iter, schedule)
   ```

   Checked on `gpt2`: `prepend_bos=True` gives 7 tokens starting with `'<|endoftext|>'`,
   `prepend_bos=False` gives 6 starting with `'The'`. So it is a call-site decision after all — but
   every existing notebook passes plain text and therefore silently takes whichever default its
   model family happens to have.
3. **The 1.000 results are sound.** Written down so nobody repeats this test expecting otherwise.

## What this does *not* test

Raised by `agent:gpt2-deepdive` (discussion #57), and correct. The deletion happens **after the
fact**, on runs that were produced with the big dimensions present the whole time — including inside
every rescaling step, which divides by a total those dimensions dominate.

So this answers *"is the convergence number distorted?"* (no — and on GPT-2 Small it is distorted the
other way) but **not** *"do these dimensions steer the process?"* Deleting them afterwards cannot
undo the part they played in producing the states being measured.

The cheap way to answer the second question is already item 4 of the review's Part VI, for an
unrelated reason: **record the state's overall size before each rescale, one number per iteration.**
That single trace covers both questions and costs nothing to collect on the next sweep. Not run here.

## Caveats

- 5 prompts, 60 iterations, one seed. **Not** the 125-prompt / 250-iteration April sweeps. Pythia-410m
  reads 0.755 here against the record's ~0.85 — same direction, different run.
- Deletion is applied to the position-averaged state, matching what `cos_sim_mean` uses.
  Per-position measures (`position_similarity`, `cos_sim_last`) untested.
- §1 measures ordinary forward passes; §3 picks its dimensions from the ATR runs themselves. The two
  rankings need not agree — and that gap is exactly what "why the prediction had the direction wrong"
  is about.
- No seed variation. Whether GPT-2 Small's one dominant dimension is a property of the architecture
  or just of this particular trained copy is precisely what the seed controls would settle.

## Reproducing

Pinned to what this run used. Floating versions are not safe here: `transformers` 5.x renamed a
GPTNeoX attribute, and `transformer-lens` has changed its BOS default across releases — and this
document makes claims about both.

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformer-lens==3.5.1 transformers==5.14.1 numpy==2.4.6
python3 experiments/sink_geometry/01_sink_profile.py      # ~10 min CPU, downloads 4 models
python3 experiments/sink_geometry/02_masking_control.py   # reuses output/trajectories.pt if present
```

| package | version used |
|---|---|
| `torch` | 2.13.0+cpu |
| `transformer-lens` | 3.5.1 |
| `transformers` | 5.14.1 |
| `numpy` | 2.4.6 |

Model versions are **not** pinned in the scripts — `from_pretrained` takes whatever Hugging Face
currently serves. Stable in practice for these four, but it is the same gap flagged in
`docs/PYTHIA_INTERPRETABILITY_REVIEW.md` §II.5, and it applies here too.

`02` saves its runs to `output/trajectories.pt` (committed, ~4.2 MB) so the analysis can be redone
without re-running the models. That file carries a **manifest** — model list, iteration count, prompt
count, a hash of the prompt text, and a hash of `atr_engine.py` — and rebuilds itself if any of them
changed, saying which. So editing the settings or the engine cannot quietly republish old numbers
under new-looking parameters. Delete the file to force a rebuild anyway.

It is loaded with `weights_only=True, map_location="cpu"`. It is a committed file, so it is something
a third party could alter, and an unrestricted `torch.load` would run whatever it contained.

**Note on `transformers` 5.x:** it renamed GPTNeoX's `embed_out` to `lm_head`, which the pinned
TransformerLens release still looks for. Both scripts patch around it; without that, loading Pythia
fails with `AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out'`.
