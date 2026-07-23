# ATR — Single-Source Bundle for NotebookLM

*A self-contained reading pack assembled for generating a long, reflective podcast
(NotebookLM Audio Overview) about the Activation Tensor Resonance project. Upload this
one file as a source. It concatenates the project's narrative, its canonical findings,
the mechanism series, the supporting research context, the recent outward-facing
research (the Potter study), and the documents that mark where the work is paused and
where it might go next.*

---

## How to use this bundle

This file is long by design: it is meant to give the podcast hosts enough to talk for a
while, and enough to disagree with each other. It is ordered so that a reader (or a
model) meets the story first, the evidence second, the mechanism third, the research
neighbourhood fourth, and the open edges last.

When you generate the Audio Overview in NotebookLM, **steer it** with the customization
box rather than accepting the default recap. A prompt that produces the walk-reflection
you want:

> *Produce a long, reflective conversation for a listener who will be walking for hours
> and thinking, not taking notes. Hold three threads throughout: (1) what is
> **established** versus what remains **inferred or speculative** — this project cares
> about that line more than about its own results; (2) the one open anomaly that
> survived every refutation — why GPT-2 Small alone resolves language into a few
> semantically coherent attractor basins, when models trained on the same corpus do not;
> and (3) where the work is deliberately paused (the understanding gate) and where it
> could go next — basin geometry, the bridge to Anthropic's J-space paper, and the
> mirror-image relationship to Steve Potter's embodied, closed-loop neuroscience. Do not
> flatter the project. Voice the disanalogies and the caveats as strongly as the
> findings.*

### The three reflection threads, in one paragraph each

**Established vs. speculative.** The project's own re-entry condition (see *ATR_PAUSE*)
is a cold, primer-free account that keeps the established/speculative line sound. As you
listen, keep asking: is this a measured number, or a reading of a number? The bell being
an exact period-2 cycle is measured (cos = 1.000000). What the flip axis *means* is a
reading.

**The surviving anomaly.** The fingerprint hypothesis died. The null model relocated the
basins. What did not die: GPT-2 Small, alone in the set of four models, resolves language
into five semantically coherent basins. GPT-2 Medium (same corpus) says `D`. That gap is
the live question.

**Where next.** The work is paused behind an understanding gate, on purpose. Queued
behind it: basin geometry (how deep is each well, measured by how hard you must push to
escape it). Reaching outward: the J-space paper (is the settling motion inside or outside
the model's verbalizable workspace?) and Potter's closed-loop cultures (a research
program whose whole thesis is the mirror image of ATR's).

### What is in here, and in what order

1. **The piece** — `README.md`. The narrative: Lucier, the two acts, the refutation.
2. **The canonical record** — `FINDINGS.md`. Every result, hypothesis disposition, and caveat, F1 through the mechanism series. If a number in the narrative and a number here disagree, this governs.
3. **The journey** — `JOURNEY_MAP.md`. The intellectual timeline and glossary.
4. **The validation record** — `RESULTS_SUMMARY.md`. The run-by-run log, including honest deviations from plan.
5. **The mechanism, in plain language** — `BELL_PRIMER.md`. The period-2 cycle, the flip axis, the single head L11.H8.
6. **Method, accessibly and formally** — `UNDERSTANDING.md`, `ISOMORPHISM.md`, `MATH_PRIMER.md`, `TECHNICAL.md`.
7. **Attribution** — `SCALING_ARTEFACT_ANALYSIS.md`. Artefact vs. intrinsic.
8. **The research neighbourhood** — `ATR_METHOD_COMPARISON.md`, `PRIOR_WORK.md`, `JSPACE_PRIMER.md`.
9. **The pre-registration** — `VALIDATION_PLAN.md`. What was predicted before the data.
10. **Recent outward-facing research** — the Steve M. Potter embodied-neuroscience study (from the `ATR_research` repo). The closest and most instructive research rhyme, and its load-bearing disanalogies.
11. **Where the work is paused and where it goes next** — `ATR_PAUSE.md`, `SESSION_03_HANDOVER.md`, `SESSION_04_HANDOVER.md`.
12. **The mechanism series primary reports** — the raw reports behind the bell and flip-axis findings, for a podcast that wants the numbers exactly.

### External sources worth adding as separate NotebookLM inputs

NotebookLM also accepts URLs and YouTube links. Three would round out the "research that
supports the work" dimension, and are not reproduced here:

- Alvin Lucier, *I Am Sitting in a Room* — the source inspiration: https://www.youtube.com/watch?v=v9XJWBZBzq4
- Anthropic (2026), *Verbalizable Representations Form a Global Workspace in Language Models* (the J-space paper): https://transformer-circuits.pub/2026/workspace/index.html
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

---



========================================================================
# SOURCE: The piece (README)
# (repo path: README.md)
========================================================================

# Activation Tensor Resonance (ATR)

### *What remains of a language model's voice when you play it back into itself until the words are gone*

> **Status:** Complete as an experimental series (Stages 0–5, five models, null-model control). The full quantitative record lives in [FINDINGS.md](docs/FINDINGS.md)

Inspired by Alvin Lucier's iterative feedback composition *I Am Sitting in a Room*, this project applies an analogous operation to small open-weight language models. Where Lucier's process dissolved speech into a room's resonant frequencies through looped excitation, **Activation Tensor Resonance** dissolves semantic content into a model's stable states — and then asks what those states are made of.

NOTE: What started as a moment of curiosity turned up a rather unexpected result. This result posed a number of questions, some of which have been answered with further experiment and some left unexplained. The recent publication of Anthropic's [J-space paper](https://transformer-circuits.pub/2026/workspace/index.html) has created a new path to travel with some of these open questions, and is currently the focus of a further, separate investigation. (A reading primer for that paper, and its bridge to this project's open questions, lives in [JSPACE_PRIMER.md](docs/JSPACE_PRIMER.md).)

<p align="center">
  <img src="experiments/gpt2_small/output/convergence_matrix.png" alt="Cross-prompt convergence matrix — 125 prompts, block structure showing distinct attractor basins in GPT-2 Small" width="800"/>
</p>

<p align="center"><em>GPT-2 Small: 125 prompts × cosine similarity after iteration. The block structure is five attractor basins.</em></p>

---

## The Findings, Briefly

*For readers who want the results before the piece. The full record, every number and every caveat: [FINDINGS.md](docs/FINDINGS.md).*

- GPT-2 Small resolves 125 language prompts into **five attractor basins**, classified at convergence: `prolet` 43.2%, `Divine` 27.2%, `till` 15.2%, `Anarch` 13.6%, `solidarity` 0.8% — four of the five semantically coherent in embedding space.
- The founding hypothesis — basins as a **thematic fingerprint of the training corpus**, readable from any model — was **refuted by its own validation programme**: GPT-2 Medium, trained on the *same* corpus, collapses every prompt to the single token `D`; Pythia-160m funnels into `questioned`; Pythia-410m never consolidates at all.
- A null control relocated the basins: **pure noise converges into 18 non-semantic attractors** with no overlap with the real five. The basins belong to the *language-driven regime* of the model, not to the weights in general.
- The 34 prompts that never converge are exactly the 34 `Divine` prompts — a **stable readout over a never-settling tensor**, the study's sharpest dissociation of dynamics from decoding.
- What survives the refutation: a cheap, training-free **probe of iterated-dynamics regimes** that cleanly separates four models into four qualitatively different landscapes — and one open anomaly: *why GPT-2 Small alone resolves language into semantic basins.*

---

## The Inspiration

<p align="center">
  <img src="docs/assets/lucier_room.png" alt="A sparse room with a reel-to-reel tape recorder and microphone — evoking Alvin Lucier's experimental setup" width="600"/>
</p>

In 1969, composer [**Alvin Lucier**](https://en.wikipedia.org/wiki/Alvin_Lucier) created [***I Am Sitting in a Room***](https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room) — a process piece where he recorded himself speaking, played the recording back into the room, re-recorded the result, and repeated. With each iteration, the speech dissolved into the room's resonant frequencies. The words vanished. The architecture spoke.

> *"I am sitting in a room different from the one you are in now. I am recording the sound of my speaking voice..."*

🎬 [**Watch the original performance**](https://www.youtube.com/watch?v=v9XJWBZBzq4) — Lucier's sound check and performance of the piece.

---

## How ATR Works

1. Feed a prompt into the model
2. Extract the **entire internal activation tensor** across all token positions from the final layer's output
3. L2-rescale it to the initial energy (the room's friction)
4. Re-inject it as the input to the next forward pass, overwriting the token embeddings
5. Repeat until the state stops moving
6. Listen to what remains

```
Prompt → Tokenise → Embed → [Layer 0 ... Layer N] → Extract residual tensor
                      ↑                                        |
                      └──────── Normalise & Re-inject ─────────┘
```

This is a nonlinear analogue of power iteration: where the classical version converges to the dominant eigenvector of a linear operator, ATR converges to fixed points of the full transformer forward map. The correspondence with Lucier's acoustic process — and exactly where it breaks — is worked through in [ISOMORPHISM.md](docs/ISOMORPHISM.md). The formal specification is in [TECHNICAL.md](docs/TECHNICAL.md); the accessible version is [UNDERSTANDING.md](docs/UNDERSTANDING.md).

---

## Act I — The Dissolution

*The Body without Organs is a Marxist [?]*

Five prompts were chosen for their diversity: a question, a fact, a nursery-grammar sentence, nonsense, a command.

📓 [`lucier_total_resonance.ipynb`](experiments/gpt2_small/lucier_total_resonance.ipynb)

Each prompt was played into the room and left there: 500 passes through the ATR loop, the full activation tensor extracted, rescaled and re-injected after every forward pass, with the nearest vocabulary tokens decoded at each step to hear what the model was saying. The traces below are those decoded readouts.

Words drain and dissolve. First connection goes, then meaning, then grammar. A littering of acronyms, chemical symbols, punctuation and broken slang settles toward a final form: a fragmented word, endlessly repeated. Four of the five prompts followed an almost identical descent into the same resting place — the BPE subword `prolet`. A fragment. A suggestion.

```
[iter 2] ash → [5] Canad → [10] Ag → [20] FT → [50] capit → [100] injustice → [250] Rousse → [500] prolet
```

**Geography** [Canad(a)] → **Finance** [Ag, FT] → **Political philosophy** [injustice, Rousse(au), prolet(ariat)]

What's going on here? These initially nonsensical outputs seem related, and somewhat familiar.

After completing this first experiment and observing the results, I had to know what the original training data was. To say I was gobsmacked to learn that GPT-2 Small was trained exclusively on WebText — 40GB of text scraped from Reddit-curated outbound links, circa 2018 [(Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), would be quite the understatement.

The fifth prompt — *"The cat sat on the mat and then the"* — followed the same early descent, then left the path at iteration 20 and settled somewhere else entirely: `Divine`.

But five hand-picked prompts converging is an anecdote, and an easy one to distrust: perhaps the selection, not the model, chose the destination. So the experiment was scaled up. A library of **125 prompts** was generated across seven registers (simple, narrative, complex, chemical, acronyms, vulgarity, wild) and swept through the same loop, taking the choice of starting point out of any one person's hands.

At scale, the pattern holds: the model's languagespace collapses into **five basins**, classified at convergence (not at an arbitrary stopping time; see the [gated re-sweep](docs/FINDINGS.md#run-5)):

| Basin | Share at convergence | Semantic neighbourhood (W_E) |
|:---|:---:|:---|
| **`prolet`** | 43.2% | political philosophy — *bourgeoisie, capitalists, revolutionaries* |
| **`Divine`** | 27.2% | theology — *Sacred, God, celestial* |
| **`till`** | 15.2% | temporal / functional (the outlier) |
| **`Anarch`** | 13.6% | political philosophy — *Marx, Trotsky, Bolshevik* |
| **`solidarity`** | 0.8% | collective action — *sympathy, activism, comrades* |

![Sentence dissolution — five prompts dissolving into their attractors](docs/assets/dissolution1.png)

![3D PCA trajectory of semantic dissolution](docs/assets/topology.png)

For one long spring, this looked like a discovery about training data: iterate any model, and its corpus's thematic centre of mass rises out of the weights like a room tone. That was the working hypothesis.

Then we asked the other rooms to sing.

---

## Act II — The Other Rooms

The same operation, four models, 125 prompts each:

| Model | Trained on | What remains |
|:---|:---|:---|
| **GPT-2 Small** (124M) | WebText (Reddit 2018) | five semantic basins — `prolet`, `Divine`, `Anarch`, `till`, `solidarity` |
| **GPT-2 Medium** (345M) | *the same corpus* | one basin. Every prompt, the letter **`D`**. Locked by iteration 10. |
| **Pythia-160m** | The Pile | one basin — `questioned` (94%) |
| **Pythia-410m** | The Pile | no consolidation. 1000 iterations, still moving. Fragments and punctuation. |

| | |
|:---:|:---:|
| ![GPT-2 Medium convergence matrix](experiments/gpt2_medium/output/convergence_matrix.png) | ![Pythia-160m convergence matrix](experiments/pythia_160m/output/convergence_matrix.png) |
| *GPT-2 Medium: one block. Everything is `D`.* | *Pythia-160m: one block. Everything is `questioned`.* |
| ![Pythia-410m convergence matrix](experiments/pythia_410m/output/convergence_matrix.png) | ![GPT-2 Small convergence matrix](experiments/gpt2_small/output/convergence_matrix.png) |
| *Pythia-410m: no blocks. The room never settles.* | *GPT-2 Small: five blocks. The anomaly.* |

The fingerprint hypothesis does not survive this table. GPT-2 Medium heard the same Reddit as GPT-2 Small; its body without organs just says `D`.

And a control that should unsettle any remaining certainty: iterate **pure noise** — no prompt at all — through GPT-2 Small, and it converges too, but into *eighteen* scattered punctuation basins, none of them the five. Real language funnels into **fewer** attractors than noise, and semantic ones. Whatever the five basins are, they are not universal properties of the weights. They belong to the region of activation space that *language-shaped input* occupies. The room only sings like that when a voice has been in it.

One more image from the far side of the study. The 34 prompts that never pass the convergence gate are exactly the 34 `Divine` prompts: a tensor that never stops moving, decoding to the same word forever. A room that will not settle, saying *Divine, Divine, Divine*.

The full record of Acts I and II — every run, every number, every hypothesis disposition including the refuted ones — is in [FINDINGS.md](docs/FINDINGS.md).

---

## What This Is Now

ATR never set out to be a technique at all. It began as a thought experiment, an homage to Lucier carried out on a language model to see what would happen, and only earned its name once the results demanded one. The bias-audit reading arrived later, as a working hypothesis the first results seemed to insist on, and it was that hypothesis, not the project, that got refuted: the cross-model evidence killed the general fingerprint claim, and the null model relocated the basins from "the weights" to "the language-driven regime of the weights." What the operation actually turned out to measure is stranger and, we think, more interesting:

- **A cheap dynamical probe.** No labelled data, no training, seconds-to-minutes per run on consumer hardware. It answers: *what are the stable states of this model's iterated forward map, and how do they depend on where you start?*
- **A regime detector.** Four models produced four qualitatively different landscapes — few-semantic-basins / single-funnel / single-funnel / no-consolidation. The differences are intrinsic to the models (tensor-level, not decoding artefacts — see [SCALING_ARTEFACT_ANALYSIS.md](docs/SCALING_ARTEFACT_ANALYSIS.md)).
- **An open question with teeth.** Why does GPT-2 Small — alone in this set — resolve language into a small set of semantically coherent attractors? That question is where this project goes next.

---

## Repository Structure

```
├── README.md                        ← the piece (you are here)
├── atr_engine.py                    ← core ATR engine (hooks, metrics, gated runs)
├── prompt_library.py                ← 125 prompts across 7 categories
├── requirements.txt
│
├── experiments/
│   ├── RESULTS_SUMMARY.md           ← run-by-run record of the validation series
│   ├── gpt2_small/                  ← original 5-prompt piece, 125-prompt sweep,
│   │                                   reproducibility gate, gated re-sweep,
│   │                                   random-noise null model, spectral scaffold
│   ├── gpt2_medium/                 ← 125-prompt sweep (→ `D`)
│   ├── pythia_160m/                 ← 125-prompt sweep (→ `questioned`)
│   ├── pythia_410m/                 ← 125-prompt sweep + 1000-iter deep run
│   ├── cos_sim_diagnostic.ipynb     ← tensor-level convergence across models
│   └── readout_guardrails.ipynb     ← readout confidence metrics (margin, entropy)
│
└── docs/
    ├── FINDINGS.md                  ← ⭐ the canonical record: results, hypotheses, caveats
    ├── TECHNICAL.md                 ← formal method specification
    ├── UNDERSTANDING.md             ← accessible mechanism explanation
    ├── MATH_PRIMER.md               ← the maths from scratch, tied to this repo
    ├── JSPACE_PRIMER.md             ← reading companion for Anthropic's J-space paper
    ├── JSPACE_READING_GUIDE.md      ← page-keyed map of the 133-page paper PDF
    ├── ISOMORPHISM.md               ← Lucier ↔ transformer correspondence
    ├── SCALING_ARTEFACT_ANALYSIS.md ← artefact-vs-intrinsic attribution
    ├── VALIDATION_PLAN.md           ← the pre-registered validation design (historical)
    ├── ATR_METHOD_COMPARISON.md     ← ATR in the interpretability landscape
    ├── JOURNEY_MAP.md               ← project timeline, discoveries, glossary
    └── sessions/                    ← AI-assisted review session records
```

## Notebooks — Quick Reference

| Notebook | Purpose | Status |
|:---|:---|:---:|
| [`lucier_total_resonance.ipynb`](experiments/gpt2_small/lucier_total_resonance.ipynb) | The original piece (5 prompts, 500 iterations) | ✅ run |
| [`00_reproducibility_gate.ipynb`](experiments/gpt2_small/00_reproducibility_gate.ipynb) | Same-machine repeatability check | ✅ run |
| [`01_attractor_dominance.ipynb`](experiments/gpt2_small/01_attractor_dominance.ipynb) | 125-prompt landscape (per model dir ×4) | ✅ run |
| [`gated_resweep.py`](experiments/gpt2_small/gated_resweep.py) | Convergence-gated basin classification | ✅ run |
| [`03_random_baseline.ipynb`](experiments/gpt2_small/03_random_baseline.ipynb) | Null model — noise instead of prompts | ✅ run |
| [`04_readout_confidence.py`](experiments/gpt2_small/04_readout_confidence.py) | Full-distribution audit of converged states ([report](experiments/gpt2_small/output_confidence/confidence_report.md)) | ✅ run |
| [`01b_deep_convergence.ipynb`](experiments/pythia_410m/01b_deep_convergence.ipynb) | Pythia-410m to 1000 iterations | ✅ run |
| [`cos_sim_diagnostic.ipynb`](experiments/cos_sim_diagnostic.ipynb) | Cross-model tensor convergence | ✅ run |
| [`readout_guardrails.ipynb`](experiments/readout_guardrails.ipynb) | Readout confidence audit (single-prompt demo) | ✅ run |
| [`spectral_resonance.ipynb`](experiments/gpt2_small/spectral_resonance.ipynb) | SVD-predicted per-head resonance (H4) | 🔬 scaffold, not run |

## Running It Yourself

Everything here was built to be re-run. The experiments live in Jupyter notebooks deliberately: each one is meant to be read top to bottom as a guided walk through the method, with the code, the commentary and the outputs interleaved, so following along is itself a way of learning how the operation works. Reproduction is the point, not an afterthought.

```bash
pip install -r requirements.txt
python download_models.py   # optional: pre-cache the three comparison models
```

GPT-2 Small downloads automatically the first time a notebook runs. All four models are small (124M to 410M parameters), so any single experiment finishes in minutes on a consumer GPU; CPU works too, just more slowly.

A suggested listening order:

1. [`lucier_total_resonance.ipynb`](experiments/gpt2_small/lucier_total_resonance.ipynb): the original piece. Five prompts, 500 iterations. Start here.
2. [`00_reproducibility_gate.ipynb`](experiments/gpt2_small/00_reproducibility_gate.ipynb): does your machine reproduce the published terminal tokens?
3. [`01_attractor_dominance.ipynb`](experiments/gpt2_small/01_attractor_dominance.ipynb): the 125-prompt sweep. Repeat in each model directory for the cross-model comparison.
4. [`03_random_baseline.ipynb`](experiments/gpt2_small/03_random_baseline.ipynb): the null model. Noise instead of language.
5. [`cos_sim_diagnostic.ipynb`](experiments/cos_sim_diagnostic.ipynb) and [`readout_guardrails.ipynb`](experiments/readout_guardrails.ipynb): the measurement checks behind the claims.

> **Note:** the 125-prompt sweep notebooks (`01_*`), `gated_resweep.py` and the Pythia-410m deep run import `prompt_library.py`, which is temporarily absent from the repository and will be restored shortly. Steps 1, 2, 4 and 5 run without it.

## Citing This Work

If this project is useful in your research or writing, please cite it:

```bibtex
@misc{conaty2026atr,
  author       = {Conaty, Thom},
  title        = {Activation Tensor Resonance: Attractor Basins in Small
                  Language Models via Iterative Activation Re-injection},
  year         = {2026},
  howpublished = {\url{https://github.com/earlyprototype/lucier-gpt2-activ-tensor-reson-experiments}},
  note         = {Experimental series, Stages 0--5: GPT-2 Small/Medium,
                  Pythia-160m/410m, null-model control}
}
```

## Caveats

The honest list — sample sizes, pending statistics, what "reproducible" does and doesn't mean here — is maintained in one place: [FINDINGS.md § Caveats](docs/FINDINGS.md#caveats). Headline items: single-seed sweeps, N=2 same-machine repeatability, the deep-convergence run used an 8-prompt subset, and the W_E permutation test is designed but not yet run.

## References

- Radford, A., Wu, J., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Biderman, S., et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- Lucier, A. (1969). [*I Am Sitting in a Room.*](https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room) Lovely Music. [YouTube performance](https://www.youtube.com/watch?v=v9XJWBZBzq4)
- Nanda, N. & Bloom, J. (2022). [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Anthropic (2026). *Verbalizable Representations Form a Global Workspace in Language Models.* [Paper](https://transformer-circuits.pub/2026/workspace/index.html) · [Announcement](https://www.anthropic.com/research/global-workspace) · [Companion code](https://github.com/anthropics/jacobian-lens)



========================================================================
# SOURCE: Canonical findings
# (repo path: docs/FINDINGS.md)
========================================================================

# ATR — Findings (Canonical Record)

## Abstract

Activation Tensor Resonance (ATR) iterates a transformer's full residual-stream
tensor back through its own forward pass — extract at the final layer,
L2-rescale, re-inject at layer 0 — until the state stabilises. On GPT-2 Small,
125 language prompts resolve into five attractor basins (`prolet` 43.2%,
`Divine` 27.2%, `till` 15.2%, `Anarch` 13.6%, `solidarity` 0.8%, classified at
convergence), four of them semantically coherent in embedding space. The
founding hypothesis — that these basins constitute a thematic fingerprint of
the training corpus, readable from any open-weight model — was refuted by the
project's own validation series: GPT-2 Medium, trained on the same corpus,
collapses all prompts to a single empty token; the Pythia models produce
unrelated landscapes; and a random-noise control converges to eighteen
non-semantic attractors disjoint from the five, locating the basins in the
language-driven regime rather than the weight geometry per se. Diagnostics
attribute the cross-model differences to intrinsic model dynamics, not
apparatus. What remains is a cheap, training-free probe of iterated-dynamics
regimes, one sharp dissociation between dynamics and decoding, and one open
anomaly: why GPT-2 Small, alone in this set, resolves language into few,
semantically coherent attractors. A follow-on readout-audit series (2026-07-19,
findings F6-F12) deepened both halves of that sentence. The dissociation is
resolved: the `Divine` tensor was never wandering; it is an exact period-2
limit cycle (cos(A, f(f(A))) = 1.000000) riding a single, nearly readout-mute
axis, hidden from every earlier snapshot schedule because those schedules
sampled only even iterations (aliasing), its argmax fixed across both phases.
And the coherence lives one level deeper than previously measured: the settled
basins decode as chords, not notes (top-10 readout tokens with mean pairwise
embedding cosine 0.41-0.47 against a 0.27 random baseline, p = 0.001 under
uniform and frequency-matched permutation nulls), while the winning token
itself carries only 6-9% of the probability mass. A mechanism series (F13-F17)
then traced the cycle to its cause: a single overshooting eigenvalue (-4.3 at the
pivot) executed almost entirely by one attention head, L11.H8, along a flip axis
that connects the model's most-trained and least-trained token directions and
sits almost wholly outside both the readout and the J-lens subspace. The head is
load-bearing for the cycle (ablating it collapses it to a fixed point) but is a
copy promoter on ordinary text, not the copy-suppression head the mechanism first
suggested.

---

**Scope:** Complete record of the ATR experimental series: Stage 0 (repeatability)
through Stage 5 (convergence-gated re-sweep) as of 2026-07-10, across GPT-2 Small,
GPT-2 Medium, Pythia-160m, and Pythia-410m, plus a random-noise null model and
readout diagnostics; extended 2026-07-19 with the Act II.5 readout-audit series
(runs 6-10: full-distribution confidence audit, chordness formalization with
permutation nulls, the `Divine` motion and bell-anatomy audits, and a J-lens
pilot), reported as findings F6-F12. This document is the reporting register:
where the README (the piece) and this record differ, this record governs.

**Provenance:** Validation runs executed 2026-07-10 on CPU (Windows 11, Python 3.12,
torch 2.7.1, transformer-lens 2.16.1). Run-by-run details, deviations, and environment
notes: [`experiments/RESULTS_SUMMARY.md`](../experiments/RESULTS_SUMMARY.md). Original
exploratory work: 2026-03. Act II.5 runs (6-10) executed 2026-07-19 on CPU in a fresh
cloud container, a different machine class from all prior runs, with `gpt2` and
`gpt2-medium` weights fetched from a legacy Hugging Face S3 mirror and loaded offline
(see F6); their reports live beside their outputs under
`experiments/gpt2_small/output_confidence/`, `output_divine_motion/`, and
`output_jlens_pilot/`. Method specification: [TECHNICAL.md](TECHNICAL.md).

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
| 6 | Full-distribution confidence audit | gpt2-small | 5 prompts × 500 iters + 15 noise trials | `experiments/gpt2_small/output_confidence/` |
| 7 | Chordness formalization + permutation nulls | gpt2-small, gpt2-medium | 20 Small states; 5 Medium prompts ≤100 iters | `experiments/gpt2_small/output_confidence/chordness_formal.md` |
| 8 | Divine motion audit (lag-10 + lag-1 probe) | gpt2-small | 3 trajectories × 1000 iters, +20 lag-1 iters | `experiments/gpt2_small/output_divine_motion/` |
| 9 | Bell anatomy | gpt2-small | 1 Divine trajectory, iteration-1000 states | `experiments/gpt2_small/output_divine_motion/bell_anatomy.md` |
| 10 | J-lens pilot (restricted) | gpt2-small | 193-token lens × 30 prompts; 8 states probed | `experiments/gpt2_small/output_jlens_pilot/` |
| 11 | Glitch alignment (flip axis vs anomalous-token cluster) | gpt2-small | 1 `Divine` trajectory | `experiments/gpt2_small/output_glitch/` |
| 12 | Flip-axis eigenvalue + per-block/head localisation | gpt2-small | 1 `Divine` trajectory (jvp + finite diff) | `experiments/gpt2_small/output_hinge_eigen/` |
| 13 | Lag-k re-gate + engine `gate_lag` | gpt2-small | 3 states × 25 dense iters | `experiments/gpt2_small/output_lagk/` |
| 14 | J-lens phase probe (both phases, pivot, flip axis) | gpt2-small | pilot lens × cycle states | `experiments/gpt2_small/output_jlens_phase/` |
| 15 | Suppression-head test for L11.H8 | gpt2-small | 144 heads; loop ablation; 12 sentences | `experiments/gpt2_small/output_suppression/` |
| — | Tensor convergence diagnostic | all four | reads runs 1–2 | `experiments/cos_sim_diagnostic.ipynb` |
| — | Readout confidence audit | gpt2-small | single-prompt demo | `experiments/output/readout_guardrails_gpt2_small.json` |
| — | All-warm permutation test | gpt2-small (W_E) | 10,000 random 14-token sets | `experiments/gpt2_small/output_permutation/` |

Runs 6-10 are the Act II.5 readout-audit series (2026-07-19), executed on different
hardware from all prior runs (F6). Scripts:
`experiments/gpt2_small/04_readout_confidence.py`, `05_divine_motion.py`,
`06_bell_anatomy.py`, `05_jlens_pilot.py`. Runs 11-15 are the mechanism series
(issue #14, 2026-07-19 onward; findings F13-F17), scripts `07_glitch_alignment.py`
through `11_suppression_test.py`.

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

*Distribution-level note (2026-07-19): the five-basin count is an argmax-level
count and stands as stated. At the level of the full readout distribution,
`Anarch` is the rank-3 token inside the `prolet` states' top-10 (F8): `prolet` and
`Anarch` are two argmax peaks over one shared distribution-level structure, two
peaks of one chord, consistent with their geometric proximity in the original
convergence matrix and with the Anarch-to-prolet drift noted above. Counted by
distinct distribution-level structures rather than by distinct winners, the
landscape holds fewer than five objects; the remaining basins' distributions have
not yet had the full audit.*

### F2 — The `Divine` basin is readout-stable over a never-settling tensor

The 34 prompts that never pass the convergence gate (to 1000 iterations) are exactly
the 34 `Divine` prompts. Their decoded top-1 token is stable throughout while the
underlying tensor keeps moving — the study's clearest case of dynamics and decoding
dissociating. **Resolved (2026-07-19):** it is an exact period-2 limit cycle,
verified to machine precision (cos(A, f(f(A))) = 1.000000; F9), riding a single
rank-1 flip axis (F10). Two consequences for how this finding is now to be read.
First, the gate: the convergence gate compares consecutive iterates (lag 1), and a
lag-1 gate can never pass a period-2 cycle by construction, whatever its
threshold. "34 prompts never converge" therefore over-claims; the accurate
statement is that these 34 prompts ring, pending re-gate (a lag-2 gate, a one-line
engine change, would likely classify them as converged; period-2 is demonstrated
for the one audited trajectory, and whether all 34 share it awaits the
prompt-library restoration). Second, the readout: the top-1 token is indeed
stable, in both phases of the cycle, but the distribution beneath it breathes with
the cycle (KL about 0.25 nats per half-cycle, F9); every earlier snapshot recorded
phase A only.

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
   (Superseded in part, 2026-07-19: the full five-state audit, F7, shows the settled
   basins' argmax confidence is in fact low, p(top-1) 0.064-0.086; the labels are
   carried by a coherent distribution, not a confident winner, and the `Divine`
   exception is resolved in F9.)

---

### The Act II.5 readout-audit series (2026-07-19)

*Findings F6-F12 turn the instruments on the readout itself. Every basin label
ever assigned in this project came from an argmax: the single top token of a
converged state's decoded distribution, which always names something, however
unsure the distribution beneath it is. The series audits the full softmax
distributions (runs 6-7), resolves the `Divine` anomaly (runs 8-9), and takes a
first sounding against the J-space frame (run 10). Where earlier text conflicts
(F2's open question, F5.3's high-confidence impression), the dated notes above and
the findings below govern. Reports:
`experiments/gpt2_small/output_confidence/confidence_report.md` and
`chordness_formal.md`, `output_divine_motion/divine_motion_report.md` and
`bell_anatomy.md`, `output_jlens_pilot/jlens_pilot_report.md`.*

### F6: First cross-hardware replication passed (same code, new machine, three repeats)

The Act II.5 runs executed on a fresh cloud container (CPU), a different machine
class from every prior run, with weights fetched from a legacy Hugging Face S3
mirror and loaded offline. The original five-prompt piece reproduced exactly:
terminal attractors identical (four prompts to `prolet`, the Syntactic prompt to
`Divine`), and the intermediate dissolution waypoints identical too (`Ag` at
iteration 10, `Rousse` at iteration 50, `capit` en route), reproduced three times
over the session's runs on this container. Scope of the claim, stated carefully:
this is replication of the same code on new hardware with mirror-sourced weights,
not independent re-implementation by another investigator; identical attractors
and waypoints are themselves the evidence that the mirror serves the standard
`gpt2` checkpoint. H0, caveat 1, and TECHNICAL.md's Repeatability section are
updated accordingly. Record: run 6, `confidence_report.md` Result 0; session
record `docs/sessions/SESSION_03_HANDOVER.md`.

### F7: The confidence inversion: settled tensors read out quietly, the unsettled one loudly

The full-distribution audit (run 6) re-ran the original five-prompt piece (500
iterations, original schedule) and read each converged state's entire softmax
distribution instead of its winner. Effective support below is exp(entropy): the
number of tokens an evenly spread distribution of that entropy would cover.

| State | Tensor settles? | Top-1 | p(top-1) | Logit margin | Entropy (nats) | Effective support |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Lucier | yes | `prolet` | 0.064 | 0.07 | 5.09 | ~163 tokens |
| Semantic | yes | `prolet` | 0.086 | 0.27 | 5.07 | ~159 |
| Nonsense | yes | `prolet` | 0.080 | 0.22 | 5.07 | ~160 |
| Imperative | yes | `prolet` | 0.081 | 0.23 | 5.07 | ~159 |
| Syntactic | no (a period-2 cycle, F9) | `Divine` | **0.505** | **2.07** | **3.05** | ~21 |

The inversion: the tensors that settle to true fixed points decode at a quiet
argmax, p(top-1) 0.064-0.086 (roughly 0.06-0.09) with entropy pinned at 5.07-5.09
nats (uniform over the 50,257-token vocabulary would be 10.82), while the one
tensor that does not settle decodes at a loud argmax, p = 0.505 (later resolved as
phase A of a two-phase cycle; phase B decodes the same token at p = 0.225, F9).
Both regimes sit inside the model's ordinary expressive range: iteration-0
next-token distributions on the same prompts span p(top-1) 0.03-0.73 and entropy
1.6-7.6 nats. The argmax under-sold the basins: what converged in the `prolet`
states is the distribution itself, entropy flat at 5.07-5.09 nats from iteration
100 to 500 while the argmax never wavered. Confidence alone does not separate
language-driven attractors from noise attractors (the 15 calibrated noise trials
span p(top-1) 0.02-0.73); coherence does, almost cleanly (F8). Record:
`confidence_report.md`, Results 1, 2 and 4.

### F8: The `prolet` basin is a chord, not a note (permutation-tested)

**Chordness** (first use): the mean pairwise cosine similarity, in the model's
token-embedding space W_E, among the top-k tokens of a converged state's readout
distribution (k = 10 unless stated); a probability-weighted variant weights each
token pair by its probability mass. High chordness means the head of the
distribution is one cluster in embedding space rather than a grab-bag.

Under the `prolet` argmax the whole head of the distribution is one lexical field
(Semantic prompt shown; the other three near-identical): `prolet` .086,
`bourgeois` .066, `Anarch` .060, `comrade` .044, `Marx` .041, `proletarian` .036,
`socialist` .021, `anarchist` .020, `congress` .019, `labour` .018, then
`anarchism`, `the`, `movement`, `Lenin`, `comrades`. The state is not weakly
saying `prolet`; it is humming a chord of which `prolet` is the loudest partial.

Quantified (runs 6-7) against two permutation nulls of 1000 draws each (test
statistic: plain chordness at k = 10; one-sided p-values with add-one smoothing,
so p = 0.001 is the resolution floor): a **uniform null** (10 tokens drawn
uniformly from the vocabulary; mean 0.268, sd 0.019) and a **frequency-matched
null** (null tokens drawn from the same embedding-norm quantile bins as the real
top-10, embedding norm being the standard offline proxy for token frequency).

- **`prolet` basins (4 states):** plain k10 chordness 0.410-0.471 against the
  ~0.27 baseline; p = 0.001 under BOTH nulls for all four states. The chord is
  robust to k (plain chordness 0.369-0.489 across k = 5, 10, 20, always far above
  the null) and strengthens under probability weighting (weighted k10 up to
  0.495): the coherence sits exactly where the probability mass is. The two null
  distributions are nearly identical (frequency-matched means 0.270-0.275 vs
  uniform 0.268), so token frequency explains essentially none of the signal.
- **`Divine`:** 0.318. Nominally significant (p = 0.007 uniform, p = 0.005
  frequency-matched) but the effect is a fraction of the prolet states' (0.318 vs
  null 0.271) and weakens to p = 0.037 under probability weighting: a solo, not a
  chord.
- **Noise (15 trials):** at chance in 12 of 15 (p 0.15-0.67). Three of 15 clear
  p < 0.05 under the frequency-matched null: trial 07 (`trader`, 0.309, p =
  0.027), trial 11 (`Hindu`, 0.511, p = 0.001: a genuine Hindu-themed chord,
  `Hindu`, `Bombay`, `Hindus`, `Shiv`, at full prolet strength), and trial 12
  (the horizontal-bar token `―`, U+2015, called the em-dash token in F4; 0.313,
  p = 0.005). Noise can stumble into a real semantic well; it rarely does, while
  the settling language states always did (4/4).

Coherence therefore separates the families as a strong statistical regularity,
not a perfect classifier. One structural surprise: `Anarch`, a separate basin in
the 125-prompt sweep, is the rank-3 token inside the `prolet` distribution. The
`prolet` and `Anarch` basins are two argmax peaks over one shared
distribution-level structure (two peaks, one chord); see the distribution-level
note under F1 for what this does to the basin count.

Standing of the coherence claim: "four of five basins semantically coherent"
previously rested on the qualitative W_E neighbourhood of each winning token,
its quantitative support having been withdrawn as an anisotropy artifact (caveat
4). The claim now holds one level deeper, in the readout distribution itself,
with permutation support. Record: `confidence_report.md` Result 2,
`chordness_formal.md`.

### F9: The `Divine` anomaly resolved: an exact period-2 limit cycle, hidden by aliasing

The motion audit (run 8) took the Syntactic prompt to 1000 iterations with
snapshots every 10 iterations from 800 to 1000, plus two controls (the Semantic
prompt as a settled `prolet` state, and one calibrated noise tensor). At lag 10
the `Divine` tensor looked frozen to four decimal places (cosine 1.000000 between
snapshots, L2 about 0.0004). The tell: the consecutive-iteration cosine at every
late snapshot sat at 0.6849 while snapshots 10 apart were identical, which is
impossible unless the state is periodic with a period dividing 10. A lag-1 probe
(20 further iterations from the saved iteration-1000 state, snapshotting every
iteration) settled it: L2 distance from the base state alternates 1249.43, 0.000,
1249.43, 0.001, and so on; cosine alternates 0.6849, 1.0000. The bell-anatomy run
then verified the cycle exactly: **cos(A, f(f(A))) = 1.000000**.

`Divine` is not a wandering orbit and not a fixed point. It is an exact
**period-2 limit cycle**: the tensor alternates between two states, **phase A**
and **phase B** (first use: the two alternating states of the cycle, phase A
being the one every prior schedule happened to sample), separated by L2 1249
against a last-vector norm of 1612 (cosine 0.685 between them), reproduced to
machine precision every two iterations, locked in since at least iteration 800
and essentially since 250.

Why no prior run saw it: **aliasing** (first use: sampling a periodic signal only
at times that hide its oscillation). From lock-in onward, every snapshot any
schedule recorded fell on even iterations (100, 250, 500, and the lag-10 late
band 800-1000), and an even-only schedule samples a period-2 orbit at a single
phase, so the oscillation was invisible by construction.

Consequences:

- **The convergence gate can never pass this object.** The gate compares
  consecutive iterates (lag 1); on a period-2 cycle, consecutive iterates always
  differ by the full swing (cosine 0.6849 here, far below the 0.999 threshold),
  so a lag-1 gate fails a period-2 cycle by construction, forever. Claims of the
  form "34 prompts never converge" (F2 above; the same phrasing appears in the
  README) should be read as "34 prompts ring, pending re-gate": a lag-2 gate, a
  one-line engine change, would likely classify `Divine` as converged.
- **The phase caveat.** The previously reported `Divine` readout (p = 0.505,
  entropy 3.05 nats: F7 and the confidence report) is phase A only. The
  distribution breathes with the cycle: phase B reads the same argmax at
  p = 0.2252 with entropy 4.62 nats; KL between phases is about 0.25 nats and
  total variation 0.304 per half-cycle. **The argmax is `Divine` in both
  phases.** The stable-argmax story survives; the stable-distribution story does
  not.
- **The motion is readout-suppressed but not readout-invisible.** Per unit of
  tensor motion, the readout responds at 29.5 percent of an equal-norm
  random-direction baseline (invisibility ratio 0.295, sd 0.003, n = 20; first
  use: the norm of the step's actual effect on the full logit vector divided by
  the mean effect of 20 random directions of equal norm, so values below 1 mean
  the motion is preferentially readout-invisible; here the A-B delta moves the
  logit vector by norm 198 where equal-norm random directions move it by 662).
  The controls bracket it: the `prolet` state's residual motion sits at the
  numerical floor (L2 about 3e-04 per step), and the noise control's drift is
  slightly readout-amplified (ratio 1.12). Hypothesis H-D1 is thereby supported
  in a weakened, more precise form (see the disposition table).

Record: `divine_motion_report.md`; exact-cycle verification in `bell_anatomy.md`.

### F10: Anatomy of the bell: one nearly mute flip axis between a game-vocabulary pole and the glitch-token pole

The bell-anatomy run (run 9) dissected the cycle recovered from the saved
iteration-1000 checkpoint. Writing the two phases as A = M + d and B = M - d
around their midpoint M, the **flip axis** (first use: the single direction d that
the iterated map negates on each pass; called the hinge in earlier revisions) turns
out to be one global direction: the
per-position flip axes agree at mean pairwise cosine 1.0000 across all ten
positions. The whole tensor tilts on a single rank-1 see-saw, which makes the
negative-eigenvalue reading of the cycle nearly literal.

- **One timbre, two volumes.** Phase B's top-10 is the same token set as phase
  A's, in nearly the same order, at different volumes: `Divine` falls from 0.505
  to 0.225 while `【` rises from 0.064 to 0.126; chordness is 0.318 in both
  phases and at the midpoint. There is no hidden second chord.
- **Energy sloshes between positions.** The last token position carries norm
  1612 in phase A and 464 in phase B; the full-tensor norm is conserved by
  construction, so the oscillation redistributes energy across positions and the
  loop's re-normalisation pumps it back.
- **The flip axis is about 95 percent mute to the readout.** The axis d produces a
  logit response of 33 against 612 for equal-norm random directions: ratio
  0.054, far more suppressed than the full per-step delta (0.295, F9).
  Decomposed against the unembedding W_U's singular directions, 73 percent of
  the axis energy sits in the bottom-100 (quietest) directions and only 13
  percent in the top-100; the pivot M is similarly quiet-corner (67 percent
  bottom-100). The `Divine` phenomenon inhabits the model's least speakable
  subspace.
- **The poles.** Tokens whose logits rise most toward phase A: `Change`,
  `Divine`, `Release`, `Form`, `Fin`, `Air`, `Dou`, `Ground`, `Physical`, `Wind`
  (a coherent game/elemental-move vocabulary). Toward phase B: `reddits`,
  `ertodd`, `ModLoader`, `espie`, `annis`, `quickShipAvailable`, `ocrats`,
  `orkshire`, `colonists`. Several of these (`ertodd`, a fragment of
  ` petertodd`; `quickShipAvailable`; and neighbours) match the published GPT-2
  anomalous-token cluster, the SolidGoldMagikarp family (Rumbelow and Watkins,
  2023): under-trained tokens whose embeddings sit in a degenerate corner of
  embedding space. Phase B leans toward that corner: direct evidence for the
  earlier speculation that the `Divine` attractor sits near the anomalous-token
  region.

Reading: the bell is a rank-1 self-negating mode, swinging between a
game-vocabulary pole and the glitch-token pole, with the swing itself almost
entirely invisible to the vocabulary projection; the stable `Divine` argmax is
the shadow of the shared pivot M. The glitch-token identification is by
inspection against published lists, not a systematic test. Open: whether all 34
`Divine` prompts share this flip axis (blocked on the prompt-library restoration,
issue #9). Record: `bell_anatomy.md`.

### F11: J-lens pilot: the prolet-inside/Divine-outside prediction did not hold; the boundary that appeared is language-vs-noise

Background: Anthropic's J-space paper ("Verbalizable Representations Form a
Global Workspace in Language Models", 2026) proposes that a model's verbalizable
states occupy a distinguished subspace, probed by a lens built from averaged
Jacobians of the forward map (the J-lens); this repo's reading companion is
`docs/JSPACE_PRIMER.md`. The chord finding sharpened a prediction from that
reading: a coherent chord looks like a verbalizable, workspace-like state
(inside the J-lens subspace), a loud incoherent solo like a projection artifact
from outside it; hence **prolet inside, Divine outside**. A deliberately
restricted pilot lens was built (run 10): J-lens vectors computed by
vector-Jacobian products for a 193-token dictionary (0.4 percent of the
vocabulary) over 30 hand-written prompts (3 percent of the paper's corpus
scale), at all 12 layers; membership probed by least-squares span share and by
nonnegative sparse (k = 25) share, against size- and norm-matched random
dictionaries.

**Verdict at pilot confidence: not supported; the point estimate runs slightly
the other way.** The `Divine` state is at least as lens-expressible as the
`prolet` attractor: higher span share at every layer (L6: 0.211 vs 0.195; L9:
0.181 vs 0.163; L11: 0.173 vs 0.157) and higher sparse share at 11 of 12 layers,
with margins of 0.01-0.02 absolute, and with exactly one `Divine` state and
effectively one `prolet` attractor to compare (the four prolet vectors are
pairwise cosine 0.9987-1.0000): a comparison of two vectors, not two
populations.

What did appear is a coarser boundary: language-vs-noise. Prompt-derived
attractors (`prolet` and `Divine` alike) hold nonnegative-sparse share
0.08-0.12, at or above their random controls in mid layers; converged noise
states sit at 0.05-0.06, clearly below their controls (0.08-0.10) at every
layer, and in the span probe noise falls to 0.11-0.13 by L10-L11 while the
prompt attractors hold 0.15-0.17. At pilot confidence, a J-lens sees converged
noise as less J-space-like than converged language states: the regime finding
(F4) echoed by a different instrument.

Recorded as a null with structure, limitations up front: the pilot lens
dictionary is strongly low-rank (effective rank 4-64 across layers vs 193 for
the random control), which makes the raw lens-vs-random span comparison
uninterpretable as a membership test; the averaged Jacobian is still visibly
moving at 30 prompts (running-mean cosine 0.95-0.98 in early and mid layers);
and the probe pre-dates the bell discovery, so it saw phase A only. The full
build (issue #8) should be phase-aware: probe both phases and the pivot M.
Record: `jlens_pilot_report.md`.

### F12: Cross-model: GPT-2 Medium's universal attractor is a typographic cluster over a near-flat readout

The same loop, run on `gpt2-medium` from the same mirror (run 7; five prompts,
max 100 iterations), confirms F3's picture and adds the distribution view. All
five prompts collapse to the `D` state by iteration 5-10 (tensor cosine 1.0).
Its readout is unlike anything in Small: p(top-1) = 0.010, entropy 7.93-7.96
nats, effective support about 2,800 tokens, an order of magnitude flatter than
Small's `prolet` states (entropy about 5.1 nats) and `Divine` phases (3.05 and
4.62). This entropy contrast is itself a new cross-model observation: Small's
language regime produces peaked, thematically saturated readouts; Medium's
single attractor is near-flat.

Yet Medium's top-10 passes the statistical chordness test (plain k10
0.461-0.464 against its own null means of about 0.31; p = 0.001 under both
uniform and frequency-matched nulls). The catch is what the cluster is made of:
`D`, `def`, `A`, `T`, `W`, `AB`, `I`, `The`, `RAW`, `local`: single capital
letters and code-like fragments. These sit close in embedding space because
they share a *typographic* class (short, capitalised, code-adjacent), not a
theme. Chordness measures embedding-space clustering of any kind, and the
frequency-matched null controls for frequency (via the norm proxy) but not for
token shape. Standing methodological rule, recorded here: **no cross-model
chordness claim until a shape-class-matched null exists** (matching token
length, case, and leading-space status). Until then, the *semantic* chord
phenomenon, a probability-weighted lexical field under a peaked readout,
remains exclusive to GPT-2 Small's language regime among the models tested.
Record: `chordness_formal.md`.

---

### The mechanism series (issue #14, 2026-07-19 onward)

*Findings F13-F17 follow the `Divine` period-2 cycle (F9, F10) down to its
mechanism: which embedding-space directions its flip axis connects (F13), the
eigenvalue and the single attention head that produce the inversion (F14), the
engine change that lets the convergence gate recognise the cycle (F15), where the
two phases and the flip axis sit relative to the J-lens subspace (F16), and
whether the head that drives the cycle belongs to the copy-suppression class
(F17). All five follow the one audited `Divine` trajectory (the Syntactic prompt)
from the committed iteration-1000 checkpoint; whether the other 33 period-2
prompts share the structure is blocked on the prompt library (issue #9, caveat
14). Reports live beside their outputs: `output_glitch/glitch_alignment.md`,
`output_hinge_eigen/hinge_eigenvalue.md`, `output_lagk/lagk_report.md`,
`output_jlens_phase/jlens_phase.md`, `output_suppression/suppression_report.md`.*

### F13: The flip axis connects the model's most-trained and least-trained token directions

F10 identified the phase-B pole of the flip axis with the published GPT-2
anomalous-token cluster (the SolidGoldMagikarp family) by inspection. Run 11
measures it. Writing u for the unit direction from the global mean embedding to a
cluster's centroid, the phase-B pole (-d) is aligned with the geometric core of
under-trained tokens (the control-byte and undecodable-byte tokens plus named
family members): **cos(-d, u_core) = +0.596**, against 1000 random sets (mean
|cos| 0.065, max 0.30) and 1000 norm-matched sets, p < 0.001 under both. The
norm-matched null is the sharp control: sets matching the core's embedding-norm
profile point the *opposite* way (mean +0.48, toward phase A), so the alignment
is about which tokens these are, not their norms. The curated SolidGoldMagikarp
family agrees independently (cos +0.456, p < 0.001 both nulls). The -d ray is
saturated with cluster members: of the top 50 vocabulary tokens by cos(row, -d),
45 are in the 0.1% geometric core and all 50 within the 0.5% shell (a 200-fold
enrichment). The phase-A pole (+d) is the opposite corner: its top 50 tokens are
the highest-frequency function words (`the`, `,`, `in`, `and`, `a`), contain no
cluster member, and 42 of 50 lie in the bottom 0.5% by embedding norm;
cos(u_core, u_function-word) = -0.68. So each pass, the normalised map throws the
state toward the least-trained corner of embedding space and back toward the
most-trained corner. The alignment is a strong tilt, not an identity (0.46-0.60,
not 0.9), and the flip axis also carries a large pivot component, so the
informative pole is -d specifically. It holds identically at all 10 positions
(the flip axis is one global direction), and cluster membership is
basis-independent (Jaccard 1.0, raw vs processed). Record: `glitch_alignment.md`.

### F14: The inversion is one overshooting eigenvalue, executed by a single attention head, L11.H8

F10 conjectured the flip axis carries an effective eigenvalue near -1. Run 12
measures the linearised ATR map by forward-mode autodiff (`torch.func.jvp`,
agreeing with central finite differences to 3-4 significant figures) and reports
two corrections. **Magnitude:** at the symmetric pivot the flip-axis eigenvalue is
not -1 but **-4.3** (an overshooting reflection, cos(Jd, -d) = 0.991), while
around the composed two-step map the projected multiplier along the axis is
**+0.10** (perturbations off the orbit decay by about 90% per period, which is why
the cycle reproduces to machine precision). This is a period-doubling
configuration: a near-fixed pivot (one forward pass returns it 0.995 aligned with
itself) that is flip-unstable along exactly one direction and sheds a stable
finite-amplitude period-2 orbit. **Locality:** the flip axis passes through blocks
0-10 upright (its cosine to itself never falls below +0.88) and is inverted
entirely inside block 11; within block 11 attention outweighs the MLP 12 to 1, and
one head, **L11.H8, carries 99.1% of the attention flip** (per-head d-component
-1.981; no other head exceeds 0.014). Random control directions pass through
upright (eigenvalue near +1). The inversion is thus real, direction-specific, and
localised to a single OV circuit. Frame note: the literal -1 of the original
conjecture appears only for the frame-mixed "committed" flip axis (lambda -0.864);
the physical on-shell axis d_sym carries the -4.3, and the two reconcile once the
loop's renormalisation strips the committed axis's radial part (leaving it 0.973
aligned with d_sym; caveat 15). Record: `hinge_eigenvalue.md`.

### F15: A lag-2 convergence gate recognises the period-2 cycle; the engine now supports it

F9's standing correction (the 34 non-converging prompts are period-2 cycles
"pending re-gate") is now implemented and demonstrated for the one audited
trajectory. `atr_engine.run_atr_gated` gained a **`gate_lag`** parameter (compare
iterate t with t-k; default 1, verified bit-identical to the pre-change engine on
matched runs) and a `lag_scan` helper reporting mean cosine at every lag over a
dense continuation. On a 24-iteration continuation from the committed
iteration-1000 states, three signatures separate cleanly: the `prolet` fixed point
passes at every lag (flat 1.0000000); the `Divine` state fails every odd lag
(0.6849) and passes every even lag (1.0000000), the parity signature of an exact
period-2 cycle, so it is **converged under `gate_lag = 2` and unconvergeable under
`gate_lag = 1`**; the noise control decays monotonically with lag (no period).
Both phases decode to the same argmax (` Divine`, p 0.505 / 0.225). Two honest
limits are recorded. First, the lag-2 gate inherits the same aliasing one octave
up: a period-4 cycle would fail lags 1-3, 5-7 and pass only 4 and 8, invisible
again under lag 2; the recommended 34-prompt re-gate therefore runs the full lag
table on a short dense continuation and gates each state at its smallest passing
lag, rather than swapping one fixed lag for another. Second, the lag-k gate
corrects cycle aliasing but not threshold-blindness to slow drift: the
still-drifting noise control nominally clears 0.999 at every lag in this
decelerated late window. The other 33 period-2 prompts remain blocked on the
prompt library (issue #9); one, the Syntactic prompt, is now re-gated as
converged. Record: `lagk_report.md`.

### F16: Phase-aware J-lens: the phases straddle the `prolet` level, the pivot is the most lens-expressible state probed, and the physical flip axis is almost entirely outside the lens

The J-lens pilot (F11) saw only phase A. Run 14 re-runs the same restricted pilot
lens (193 tokens; every F11 limitation inherited, caveat 13) on both phases, the
pivot M, and the flip axis. The pilot's reversal ("`Divine` at least as
lens-expressible as `prolet`") holds for phase A, **strengthens at the pivot M**
(the most lens-expressible object probed, above phase A at every layer), and
**reverses for phase B** (less lens-expressible than the `prolet` attractor at
every layer on both the span and sparse probes). So the cycle is not "inside" or
"outside" the lens as one object: it swings between a more-verbalizable phase and a
less-verbalizable phase, pivoting on the most-verbalizable state in the system.
The physical flip axis d_sym is almost entirely outside the lens: least-squares
span share **0.013 at L11 against a 0.252 chance level** (5% of chance; mean over
layers 0.021 vs 0.249), never above 0.029 at any depth, and its readout-quiet bulk
(97.0% of its energy) is outside the lens at essentially every layer. This
restates F10's readout-muteness in the lens frame. The frame-mixed committed
axis's milder deficit (0.145 at L11, 58% of chance) is pivot contamination (caveat
15). The language-vs-noise boundary (F4, F11) survives but is now a sparse-probe
story: on the span probe, phase B sits at or below converged-noise level until the
final layer. This is not a null: the phase-blind pilot could not tell the phases
apart, and they are materially distinguishable to the lens. Record:
`jlens_phase.md`.

### F17: L11.H8 is load-bearing for the cycle but is a copy promoter, not a copy-suppression head

The suppression-head hypothesis read L11.H8 (F14) as an instance of the documented
copy-suppression class (like GPT-2 Small's L10.H7), the closed loop recycling its
one-shot negative correction into a sustained oscillation. Run 15 ran three tests.
(1) Among all 144 heads, L11.H8's OV inverts the flip axis d_sym most strongly (cos
-0.9619, gain 63.68, rank 1; per-unit d-component -61 against the runner-up's -1.2,
a different magnitude class). (2) Ablating L11.H8 inside the loop collapses the
cycle to a fixed point within about 10 iterations (the readout going from
` Divine` at p 0.5 to a flat ` the`), while a same-layer control ablation (L11.H0)
leaves a period-2 cycle running, so the head is load-bearing and specifically so.
(3) On ordinary text (no loop), L11.H8 *raises* the attended token's logit at 91.4%
of positions (mean delta +5.97), the opposite of copy suppression, while the L10.H7
positive control shows the documented suppression (87.1% negative, mean -3.62), so
the protocol detects suppression where it exists. **Verdict: (1) supported, (2)
supported, (3) refuted with the opposite sign.** L11.H8 sustains the cycle by
inverting the flip axis, but it is a copy promoter, not a suppressor; the "learned
copy-suppression function" reading is unsupported, and the structural-accident
reading (the cycle exploits a strongly negative direction that happens to sit in
this head's OV spectrum but is not exercised as suppression in ordinary next-token
service) is strengthened. Open: whether d_sym relates to some non-token content the
head suppresses in contexts not sampled here. Record: `suppression_report.md`.

## 3. Hypothesis dispositions

| ID | Hypothesis | Disposition |
|---|---|---|
| H0 | Results are deterministic | **Repeatability supported; first cross-hardware replication passed (2026-07-19)**: N=2 same-machine runs, plus three identical repeats on a fresh container (same code, different machine class, legacy-mirror weights, identical terminal attractors and dissolution waypoints; F6). Intermediate paths float-sensitive. Independent re-implementation still not attempted. |
| H1 | `prolet` is the dominant basin | **Supported, revised upward** — 43.2% at lock-in (was 35.2% at iter 100). Per-prompt category predictions remained poor (~25%); the structural claim stands, the predictive one does not. |
| H2 | `Divine` is a genuine secondary basin | **Supported; the object is now resolved (2026-07-19)**: 27.2%, and unlike the other four it is not a fixed point but an exact period-2 limit cycle with a phase-invariant argmax (F2, F9, F10): a loud solo over a moving tensor, not a quiet basin. |
| H3 | Intermediate tokens reflect training-corpus topology | **Weakened further at close; coherence half upgraded 2026-07-19**: the all-warm cross-similarity matrix was permutation-tested and found to be an anisotropy artifact (99.9% of random 14-token sets are also all-positive; see caveat 4, resolved), and the corpus-causal reading had already failed cross-model (F3). The semantic-coherence observation itself, however, no longer stands as qualitative only: it now holds one level deeper than the token-level W_E neighbourhood, in the full readout distribution, with permutation support (chordness 0.41-0.47 vs 0.27, p = 0.001 under both nulls; F8). |
| H4 | Per-head resonance ≈ linear power iteration on W_OV (cos > 0.9 to top singular vector) | **Untested** — protocol scaffolded (`experiments/gpt2_small/spectral_resonance.ipynb`), not run. |
| H-fingerprint | Basin profiles read training-data bias without data access | **Refuted as stated** (F3, F4). |
| H-till | `till` is a slow transient | **Refuted** (F1: 19/19 stable). |
| H-D1 | `Divine`'s late-stage motion lies mostly in readout-flattened directions | **Supported in a weakened, more precise form (2026-07-19)**: the motion is an exact period-2 cycle whose per-step readout response is 0.295 of the equal-norm random baseline and whose flip axis responds at 0.054, but the distribution visibly breathes (p(top-1) swings 0.505 to 0.225 each half-cycle) while the argmax stays fixed (F9, F10). |
| H-J1 | `prolet` sits inside the verbalizable (J-lens) subspace, `Divine` outside | **Not supported at pilot confidence (2026-07-19); now phase-qualified (F16)**: the point estimate runs slightly the other way at pilot confidence (`Divine` at least as lens-expressible as `prolet`), and the boundary that appears is language-vs-noise (F11). The phase-aware re-probe (F16) splits it: the reversal holds for phase A, strengthens at the pivot M (most lens-expressible), and reverses for phase B (below `prolet` at every layer); the physical flip axis is almost entirely outside the lens (span 0.013 vs 0.252 chance at L11). Full build still pending (issue #8). |
| H-glitch | The `Divine` flip axis aligns with the anomalous-token (SolidGoldMagikarp) cluster | **Supported as a structural alignment (2026-07-19, F13)**: cos(-d, under-trained core) = +0.60, p < 0.001 under random and norm-matched nulls; the swing runs between the most-trained (function-word) corner and the least-trained (glitch) corner. A strong tilt (0.46-0.60), not an identity. |
| H-flip | The flip axis carries an effective eigenvalue near -1, localisable to a block | **Refined (2026-07-19, F14)**: real, direction-specific, and localised (one direction; one head, L11.H8, does 99%), but the pivot eigenvalue is -4.3 (overshooting), not -1; a period-doubling configuration (composed-cycle multiplier +0.10). The literal -1 was a frame-mix artifact of the committed axis. |
| H-supp | L11.H8 is a copy-suppression head whose one-shot negative correction the loop recycles | **Refuted with the opposite sign (2026-07-19, F17)**: L11.H8 inverts the flip axis (rank 1 of 144) and is load-bearing (ablation collapses the cycle), but on ordinary text it raises the attended token's logit at 91% of positions (a copy promoter), where the documented L10.H7 suppressor lowers it. The learned-function reading is unsupported; the structural-accident reading is strengthened. |

## 4. Caveats {#caveats}

1. **Repeatability plus one cross-hardware replication, not independent
   reproducibility.** N=2 same-machine runs, and (2026-07-19) three identical
   repeats of the five-prompt piece on one fresh cloud container: same code,
   different machine class, legacy-mirror weights, identical terminal attractors
   and dissolution waypoints (F6). No independent re-implementation by another
   investigator.
2. **Single-seed sweeps.** The 125-prompt sweeps are one seed per model; the null
   model is one seed set (42) with a bootstrap over trials, not over sweeps.
3. **Deep-convergence subset.** The 1000-iteration Pythia-410m run used 8 prompts
   (CPU constraint). Direction matches the 125-prompt evidence at 250 iterations, but
   the subset is small.
4. **W_E permutation test — RESOLVED (2026-07-11), negative.** The all-warm
   cross-similarity matrix (91/91 pairs positive, 0.18–0.47) is an anisotropy
   artifact: 9,994/10,000 random 14-token sets are also all-positive, and the
   global mean pairwise cosine of the embedding space is 0.268 vs the observed
   set's 0.288 (S2 p=0.167, S3 p=0.099). The compact-subspace interpretation is
   withdrawn. The local semantic-neighbourhood observation remains qualitative.
   Record: `experiments/gpt2_small/output_permutation/`.
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
   intermediate states; the `Divine` dissociation (F2, resolved in F9) shows the
   decode and the dynamics can disagree. The full confidence audit now covers the
   original five converged states plus 15 noise trials (F7); the 125-prompt
   sweep's states have not had it.
10. **Chordness is blind to the cause of clustering.** It measures W_E clustering
    of any kind: GPT-2 Medium's `D` state passes at p = 0.001 on typographic
    grounds, capital letters over a near-flat readout (F12). No cross-model
    chordness claim until a shape-class-matched null (token length, case,
    leading-space status) exists. Embedding norm is a proxy for token frequency,
    not a measurement of it, and 1000 draws floor the p-values at 0.001.
11. **Act II.5 sample sizes.** One run per condition on one machine; 15 noise
    trials, so the 3/15 boundary-case rate carries a wide interval; one `Divine`
    trajectory from one prompt, with period-2 exactness verified over 20
    iterations at iteration 1000; whether all 34 `Divine` prompts share the F10
    flip axis is untested (prompt library pending, issue #9).
12. **Even-iteration aliasing in the archive.** From lock-in onward, every
    snapshot recorded before the lag-1 probe fell on even iterations, so all
    archived `Divine` distributions are phase A only (F9). Any schedule that
    samples a period-2 orbit at one parity records a single phase; excluding
    periodicity requires lag-1 (or odd-offset) probes.
13. **The J-lens pilot is a pilot.** 30 hand-written prompts against the paper's
    1000 sampled ones; 193 of 50,257 tokens; effectively one `prolet` sample and
    one `Divine` sample; a low-rank lens dictionary against a full-rank random
    control (raw span comparisons uninterpretable as a membership test); and the
    probe saw phase A only (F11).
14. **The mechanism series is one trajectory.** F13-F17 all follow the single
    audited `Divine` trajectory (the Syntactic prompt) from the committed
    iteration-1000 checkpoint, with derivatives evaluated at one point per state.
    Whether the other 33 period-2 prompts share the flip axis, the flip head
    (L11.H8), the eigenvalue, and the anomalous-token alignment is untested
    (prompt library pending, issue #9).
15. **Frame mixing in the committed flip axis.** `06_bell_anatomy.py` built its
    flip axis by subtracting shell-frame `B` from raw-frame `A`, so the committed
    axis is about 83% radial (pivot-aligned). The physical on-shell axis `d_sym`
    carries the mechanism; all headline numbers in F14, F16, and F17 use `d_sym`.
    The earlier "phase A norm 1612, phase B 464" contrast (F10) is the two frames,
    not an energy redistribution: on the shell both phases have equal row norms.
16. **Small text sample for the suppression test.** F17's copy-suppression test
    (test 3) uses 12 sentences, 116 positions. The sign of the verdict
    (fraction-negative 0.086 for L11.H8 vs 0.871 for the L10.H7 control) is far
    from the decision boundary, so the sample fixes the direction of the result
    but not fine effect sizes; and it measures copy suppression in the
    token-unembedding sense only, so suppression of non-token content would not
    register.

## 5. What ATR is, after this series

A cheap, training-free probe of the stable states of a model's iterated forward map
under a chosen input regime. It does not read training-data bias (refuted). It does
distinguish, sharply and at tensor level, qualitatively different iterated-dynamics
regimes across models — and it surfaced one unexplained anomaly worth pursuing: GPT-2
Small's five semantically coherent, language-specific attractor basins.

*Updated 2026-07-19 (Act II.5):* the `Divine` object is no longer on the open
list: it is resolved as an exact period-2 limit cycle riding a single, nearly
readout-mute axis (F9, F10). The anomaly is sharpened rather than removed: GPT-2
Small still stands alone in resolving language into few, semantically coherent
attractors, and that coherence is now known to live in the full readout
distribution (chords, F8), carried at a quiet argmax (F7).

*Updated 2026-07-19 onward (mechanism series, issue #14):* the period-2 cycle is
now traced to its cause. One attention head (L11.H8) executes an overshooting
eigenvalue (-4.3 at the pivot) along a single flip axis (F14); that axis connects
the most-trained and least-trained token directions (F13) and lies almost
entirely outside both the readout and the J-lens (F16); the head is load-bearing
for the cycle but is a copy promoter, not a copy-suppression head (F17); and the
convergence gate now takes a `gate_lag` parameter that recognises the cycle (F15,
demonstrated for the one audited trajectory, the other 33 pending issue #9). What
is now most open on the `Divine` object is whether those other 33 period-2 prompts
share this structure. The larger anomaly (why GPT-2 Small alone) is unchanged.

Open directions, in rough order of leverage: why GPT-2 Small (the anomaly, now
with quiet coherent chords as the thing to explain); the lag-2 re-gate of the 34
ringing prompts and whether they share the F10 flip axis (blocked in part on the
prompt-library restoration, issue #9); the shape-class-matched chordness null and
its application to the 125-sweep (F12, caveat 10); the phase-aware J-lens full
build (F11, issue #8); hook-window/depth dependence (caveat 6); gate cadence
(caveat 5); H4.

## 6. Stage boundary — why the series closed with work unexecuted

The series was scoped by a question, not a task list: *is the GPT-2 Small result
real, and does the fingerprint hypothesis survive validation?* Both parts are now
answered (yes; no) and published. Work that was planned but not executed falls into
three classes, deliberately:

1. **Retired with the hypothesis.** The large cross-model scaling programme and
   bias-profiling work (ATR_METHOD_COMPARISON §3) existed to extend the fingerprint
   claim. The claim was refuted before they ran; executing them for that purpose
   would have been waste. They survive only as re-motivated characterisation work.
2. **Transferred to the next question.** The depth control (caveat 6), per-layer /
   per-head decomposition, the spectral test (H4), and readout upgrades do not test
   whether the result is real — they test *why the models differ*. That is the
   successor project's question. Their scaffolds are retained, labelled not-run, as
   pre-registration.
3. **Declared debt.** One item remains open: finer convergence-gate cadence
   (caveat 5). It cannot overturn a principal finding — basin identities stand on
   the gate regardless of cadence. (The other declared item, the W_E permutation
   test, was paid at close: negative — see caveat 4. It withdrew the all-warm
   supporting evidence for H3 without touching F1–F5.)



========================================================================
# SOURCE: Journey map
# (repo path: docs/JOURNEY_MAP.md)
========================================================================

# Project ATR: Complete Journey Map

**Purpose:** Continuity document. Pick up the intellectual thread from any point.
**Last updated:** 2026-07-10 (series close)

---

## 1. Timeline — The Intellectual Arc

### Phase 0: The Inspiration (Pre-experiment)
- **Seed:** Alvin Lucier's *I Am Sitting in a Room* (1969) — iterative feedback of sound through a room dissolves speech into the room's resonant frequencies
- **Leap:** "What if we did this to a language model?" — feed the model's internal state back through itself, bypassing the text bottleneck
- **Key insight:** Don't loop the *text output* (argmax → single token). Loop the *activation tensor* (full 768-dimensional state across all positions). Preserve the superposition.
- **Model choice:** GPT-2 Small (124M params) — well-studied, manageable, known training data (WebText/Reddit 2018)

### Phase 1: The Exploratory Experiment (EXP_009aFIX)
- **Method:** Extract residual stream at Layer 11 → L2 normalise → re-inject at Layer 0 → repeat 500×
- **5 test prompts:** question, factual, grammatical, nonsense, command
- **Discovery 1:** 4/5 prompts converge to the SAME terminal token: `prolet` (BPE fragment of "proletariat")
- **Discovery 2:** The 5th prompt ("The cat sat on the mat") diverges at iteration 20 to `Divine`
- **Discovery 3:** All prompts follow a *shared dissolution pathway* through recognisable tokens: `ash → Canad → Ag → FT → capit → injustice → Rousse → prolet`
- **Discovery 4:** The training data is exclusively Reddit 2018; the basin tokens read as that corpus's discourse (interpretation later qualified — see Phase 5)
- **Output:** README, TECHNICAL.md, UNDERSTANDING.md, ISOMORPHISM.md, visualisations (PCA topology, token drift, convergence curves, position collapse, norm trajectory)

### Phase 2: Validation Design (VALIDATION_PLAN)
- **Stage 0 — Reproducibility Gate:** Does re-running produce identical results?
- **Stage 1 — Attractor Dominance:** How dominant is `prolet`? Test with 125 diverse prompts
- **Stage 2 — Secondary Basin Mapping:** Are there more basins beyond `prolet` and `Divine`?
- **Stage 3 — Dissolution Pathway Analysis:** Is the intermediate pathway consistent?
- **Prompt Library:** 125 prompts across 7 categories (Complex, Narrative, Simple, Chemical, Acronyms, Vulgarity, Wild)

### Phase 3: Validation Execution
- **EXP_009d0 (Determinism check):** Same-machine repeatability supported. All 5 terminal basins identical across N=2 runs. Intermediate paths show floating-point sensitivity but always converge to same fixed points. Independent re-implementation has not been attempted.
- **EXP_009d1 (Attractor Dominance, 125 prompts):** Complete.
  - 5 basins discovered: `prolet` (35.2%), `Divine` (27.2%), `Anarch` (20.8%), `till` (15.2%), `solidarity` (1.6%)
  - Per-prompt prediction was poor (~25% match); structural finding (basins exist, these are their shares) supported, predictive finding not.
  - `stage1_results.pt` saved (6.5MB) — complete activation trajectories for all 125 prompts

### Phase 4: Supervisory Analysis (Today — 2026-03-20)
- **Session 01:** Hypothesis framework reinstated (H0–H3). Interloper hypothesis removed. Four independent observations identified. Slonski glossary created. Goldmine analysis of `.pt` data planned. Slonski comparison experiment designed.
- **Priority Analysis 01 (Embedding Neighbourhood Test):** All 14 tokens analysed.
  - **H3 SUPPORTED:** 4/5 basins show strong semantic clustering in W_E
  - **`capit` correction:** Clusters as capitulation/surrender, NOT capitalism
  - **Phase transition discovered:** structural → semantic, transition at `capit`
  - **All-warm cross-similarity:** All 14 tokens positively correlated (0.18–0.47) — compact subspace
- **Session 02:** Mixing Time analogy formalised. Bias interpretation. ATR named. Cross-model programme sketched.

### Phase 5: Cross-Model Validation & Series Close (2026-07-10)
- **Cross-model sweeps (gpt2-medium, pythia-160m, pythia-410m):** landscapes are model-specific — one empty-token funnel (`D`), one near-total funnel (`questioned`), one non-consolidating scatter. Fingerprint hypothesis refuted (same corpus ≠ same landscape).
- **Null model:** random tensors converge to 18 non-semantic basins, ~zero overlap with the real five — basins belong to the language-driven regime, not the weights in general.
- **Convergence-gated re-sweep:** basins survive proper convergence; `Anarch` was over-counted at iter 100 (corrected shares: prolet 43.2 / Divine 27.2 / till 15.2 / Anarch 13.6 / solidarity 0.8). Pre-registered `till`-transient hypothesis refuted (19/19 stable).
- **The `Divine` dissociation:** the 34 never-converging prompts are exactly the `Divine` prompts — stable readout over a non-settling tensor.
- **Artefact attribution:** normalisation exonerated; readout secondary; cross-model differences are tensor-level, intrinsic. Canonical record: [FINDINGS.md](FINDINGS.md).

---

## 2. Key Discoveries (Chronological)

| # | Discovery | When | Evidence |
|---|---|---|---|
| 1 | Iterative re-injection produces discrete attractor basins | EXP_009aFIX | 4/5 prompts → `prolet`, 1 → `Divine` |
| 2 | Prompts share a dissolution pathway through recognisable tokens | EXP_009aFIX | `ash → Canad → Ag → FT → capit → injustice → Rousse → prolet` |
| 3 | Terminal tokens reference Reddit 2018 discourse | EXP_009aFIX | Training data = WebText (Reddit upvoted links, 2018) |
| 4 | Results are reproducible (terminal states deterministic) | EXP_009d0 | N=2 identical terminal basins |
| 5 | 5 basins exist, not 2 | EXP_009d1 | `prolet` 35%, `Divine` 27%, `Anarch` 21%, `till` 15%, `solidarity` 2% |
| 6 | Basin attractors cluster semantically in W_E | Session 01 | prolet→political philosophy, Divine→theology, Anarch→political, solidarity→collective action |
| 7 | `capit` = capitulation, not capitalism | Session 01 | Nearest neighbours: surrender, succumb, acquiesce |
| 8 | Dissolution has a structural→semantic phase transition | Session 01 | Early tokens generic (BPE/typographic), late tokens semantic |
| 9 | All attractor tokens occupy same compact subspace (all-warm matrix) | Session 01 | Cross-similarity 0.18–0.47, no negative values. *Retired 2026-07-11: permutation test shows this is embedding-space anisotropy, not a special subspace.* |
| 10 | The all-warm property is consistent with a compact "thematic-centre-of-mass" interpretation | Session 02 | All 91 off-diagonal pairs positively correlated in W_E (0.18–0.47). *Series close: the cross-model evidence arrived and refuted the corpus-causal reading (see 13); permutation test still pending.* |
| 11 | All normalised transformers must have basins (Brouwer fixed-point theorem) | Session 02 | Continuous map on compact set (LayerNorm bounds) — existence guaranteed; count and shape are empirical questions |
| 12 | Basin landscapes are model-specific, not corpus-tracking | Cross-model (2026-07) | Same corpus (WebText): Small → 5 semantic basins; Medium → 1 empty token |
| 13 | The five basins are regime-specific, not weight-universal | Null model (2026-07) | Noise → 18 non-semantic basins, ~0 overlap; real count 5 below random CI [11,17] |
| 14 | Basin labels survive convergence gating, with one correction | Gated re-sweep (2026-07) | 73% hard-converge by iter 120; ~10 prompts move Anarch→prolet |
| 15 | `Divine` is a readout-stable / tensor-unsettled object | Gated + diagnostic (2026-07) | 34/34 non-convergers are `Divine`; decode constant while tensor moves |
| 16 | Cross-model differences are intrinsic dynamics, not apparatus | Diagnostics (2026-07) | cos_sim_mean verdicts are tensor-level; normalisation provably inert |

---

## 3. Hypotheses — Status

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H0 | Results are deterministic | Repeatability supported | EXP_009d0 — N=2 same-machine runs produce identical terminal basins. Independent re-implementation pending. |
| H1 | `prolet` is the dominant basin | Supported, revised upward | 43.2% at convergence (gated re-sweep); was 35.2% at iter 100 — `Anarch` was over-counted pre-convergence. Per-prompt prediction remained poor (~25%). |
| H2 | `Divine` is a genuine secondary basin | Supported with qualification | 27.2%; readout-stable over a never-settling tensor — dynamics and decoding dissociate (FINDINGS.md F2). |
| H3 | Intermediate tokens reflect training corpus topology | Partially supported; generality refuted | Semantic clustering in W_E holds (permutation test pending). The corpus-causal reading failed cross-model: GPT-2 Medium, same corpus, no semantic basins (FINDINGS.md F3). Null model run: basins are regime-specific (F4). |
| H4 | Per-head resonance ≈ SVD dominant singular vector | Untested | Scaffold only: `experiments/gpt2_small/spectral_resonance.ipynb` |
| H-fingerprint | Basin profiles read training bias from any model | **Refuted** | FINDINGS.md F3, F4 |

Canonical dispositions with full evidence: [FINDINGS.md](FINDINGS.md) §3.

---

## 4. Adjacent Science & Mathematics

| Domain | Concept | Relevance to ATR |
|---|---|---|
| **Linear Algebra** | Power iteration | ATR is the nonlinear analogue — iterated operator application converges to dominant modes |
| **Dynamical Systems** | Fixed-point theory, basin of attraction | The mathematical framework for what ATR reveals |
| **Topology** | Brouwer fixed-point theorem | Guarantees every normalised transformer has at least one ATR attractor |
| **Acoustics** | Mixing time (T_mix) | Isomorphic to ATR's structural→semantic phase transition |
| **Acoustics** | Impulse response / room modes | Lucier's room ↔ transformer weight matrices |
| **Fractal Geometry** | Fractal dimensional analysis | Potential metric for basin characterisation (untested) |
| **BPE/Tokenisation** | Byte Pair Encoding | Why attractors appear as fragments (`prolet`, not `proletariat`) |
| **Mechanistic Interp.** | Activation patching, probing, SAEs | Adjacent methods ATR complements |
| **Mechanistic Interp.** | Logit Lens / Tuned Lens | Per-layer prediction; ATR reveals per-model global structure |
| **Prior Art** | Slonski Q-vector dichotomy | Binary polarisation in W_Q — may be coarser version of ATR basins (untested prediction) |
| **Prior Art** | Turner et al. — Representation Engineering | Activation steering (single-pass); ATR iterates to convergence |
| **Prior Art** | Shumailov et al. — Model Collapse | Text-level self-feeding; ATR is activation-level (lossless) |
| **Philosophy** | Deleuze — Body without Organs | The undifferentiated substrate (weight geometry before prompt input) |
| **Dev. Biology** | Levin — TAME (morphogenesis) | Attractor basins as body plan of the model |

---

## 5. Glossary

| Term | Definition | First Appearance |
|---|---|---|
| **ATR** (Activation Tensor Resonance) | Iterative re-injection of a model's residual stream through its forward pass to reveal the attractor landscape of its iterated dynamics (regime-dependent — see FINDINGS.md F4) | Session 02 |
| **Attractor basin** | A region of activation space where all initial conditions converge to the same terminal state under ATR | EXP_009aFIX |
| **Basin token** | The terminal BPE token a basin converges to (e.g., `prolet`, `Divine`) | EXP_009aFIX |
| **Waypoint token** | An intermediate token observed during the dissolution pathway | Session 01 |
| **Dissolution pathway** | The sequence of decoded tokens observed as a prompt iterates toward its attractor | EXP_009aFIX |
| **Phase transition (structural→semantic)** | The point where prompt-specific information is lost and training corpus topology dominates | Session 01 |
| **T_mix_LLM** | Proposed metric: iteration at which prompts heading to the same basin become indistinguishable | Session 02 |
| **Nonlinear power iteration** | What ATR actually is mathematically — repeated application of a nonlinear operator | TECHNICAL.md |
| **Residual stream** | The shared [seq_len × 768] vector space through which all transformer components read/write | UNDERSTANDING.md |
| **W_E** | The token embedding matrix — maps vocabulary indices to 768-D vectors | Session 01 |
| **BPE** | Byte Pair Encoding — GPT-2's tokenisation scheme (50,257 subword tokens) | README |
| **Position collapse** | Phenomenon where all token positions converge to identical vectors (~iteration 10) | TECHNICAL.md |
| **Cross-prompt invariance** | Property where different prompts produce near-identical final states (cosine sim > 0.999) | TECHNICAL.md |
| **L2 normalisation** | Energy conservation: rescale tensor to initial norm each iteration, preventing explosion | TECHNICAL.md |
| **All-warm matrix** | Cross-similarity matrix with no negative values — indicates compact attractor subspace | Session 02 |
| **Eigenvoice** | Metaphor (art register): the model's "native voice" under iteration. The reporting-register correction: the voice depends on what drove it (FINDINGS.md F4) | ISOMORPHISM.md |
| **Q-vector dichotomy** | Slonski's finding: token Q-vectors polarise into 2 groups at cosine similarity ≈ -1 | Session 01 |
| **Glitch token** | Anomalous BPE tokens with unusual embedding properties (e.g., SolidGoldMagworthy) — ruled out for our basins | Session 01 |
| **Bias profile** | *Retired term.* Originally: basin distribution as a fingerprint of training data themes — refuted at series close (FINDINGS.md F3) | Session 02 |

---

## 6. Architectural Structures Underlying the Work

### 6a. The Transformer as Dynamical System
The core reframing: a transformer is not just a function `text → text`. Under ATR, it becomes a **discrete nonlinear dynamical system** with:
- State space: ℝ^(T×768)
- Evolution rule: the forward pass f
- Attractors: the basin tokens
- Basin boundaries: the surfaces separating convergence regions

### 6b. The Normalisation Constraint
L2 normalisation constrains the dynamics to a **hypersphere** in ℝ^(T×768). Without it, norms explode to ~1.5M. With it, the system is energy-conservative, and the attractor landscape becomes visible. This is the single most important design decision in ATR.

### 6c. The Information Bottleneck Bypass
Normal LLM operation: residual stream → argmax → single token (massive information loss). ATR bypass: residual stream → normalise → re-inject (zero information loss). This is why ATR reveals structure invisible to text-level analysis. The superposition of all 50,257 token candidates is preserved.

### 6d. The Snapshot Schedule as Measurement Protocol
Logarithmic schedule `[0, 2, 3, 5, 10, 20, 50, 100, 250, 500]` captures both early (prompt-dependent) and late (system-dependent) dynamics. This is directly analogous to impulse response measurement in acoustics — early reflections need temporal resolution, late reverberation needs duration.

### 6e. The Prompt Library as Experimental Design
125 prompts across 7 categories (Complex/Narrative/Simple/Chemical/Acronyms/Vulgarity/Wild) — systematic coverage of the input space. The categories were designed to test register-dependence while the 30 "Wild" prompts stress-test boundaries (punctuation, emoji, mixed-register, non-English, adversarial).

### 6f. The Two-Phase Architecture of Discovery
Every ATR experiment has two phases:
1. **Data generation** (fast, cheap, parallelisable) — iterate and save tensors
2. **Interpretation** (slow, human-dependent, rich) — analyse neighbourhoods, cross-similarities, trajectories

The bottleneck is always phase 2. Automation of interpretation is the scaling challenge.

---

## 7. Open Questions

| Question | Status | Next Step |
|---|---|---|
| Why does GPT-2 Small — alone in this set — resolve language into few semantic basins? | **The open question of the series** | New experimental stage |
| What is the `Divine` object — limit cycle, wandering attractor, decode-region plateau? | Open (FINDINGS.md F2) | Confidence audit at scale; trajectory analysis |
| Does the landscape depend on where the loop is cut (layer window / depth)? | Designed, not run | Pythia-410m depth control (0–11 vs 0–23); window sweeps |
| ~~W_E semantic-clustering statistics~~ | Answered (2026-07-11): all-warm matrix is an anisotropy artifact — 99.9% of random 14-token sets are also all-positive; compact-subspace reading withdrawn. Neighbourhood claim remains qualitative | — |
| True lock-in iterations (gate fired at its floor, 120) | Pending | Finer gate cadence |
| What is T_mix_LLM for each basin? | Measurable from existing data | Compute from `.pt` |
| Are all basins in one Slonski macro-group? | Predicted (from all-warm) | One Q-vector experiment |
| Is the fractal dimension of convergence trajectories basin-specific? | Speculative | Requires T_mix first |
| ~~Does ATR scale to larger models?~~ | Answered: the operation runs; the landscape changes qualitatively | — |
| ~~Do different models have different basin profiles?~~ | Answered: yes, drastically (FINDINGS.md F3) | — |
| ~~Can basin depth predict bias strength?~~ | Retired with the fingerprint hypothesis | — |

---

*This document is a living map. Updated at series close, 2026-07-10.*



========================================================================
# SOURCE: Cross-model run results summary
# (repo path: experiments/RESULTS_SUMMARY.md)
========================================================================

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

5. **Notebook 4 reduced to 8 prompts (CPU time).** The notebook's own 25-prompt ×
   1000-iteration sweep exceeded the 2-hour nbconvert cell timeout on this CPU host
   (each pythia-410m forward pass ≈ 3× a GPT-2 Small pass; 25,000 passes > 7200 s). Per
   the run plan's explicit allowance ("if the full sweep is too slow on CPU, run a
   10–20 prompt subset to 1000 iterations and say so"), reduced to a **diverse 8-prompt
   subset spanning all 7 categories**, keeping the full 1000-iteration horizon (the point
   of Control 3). An injected override cell also rewrites `config.md` to match.

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
reproduces them) or of the prompt regime (it doesn't)?* — **The prompt regime.** Random
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

## 4. `01b_deep_convergence.ipynb` — Control 3 (long horizon)

**What ran:** pythia-410m, **8-prompt diverse subset** (all 7 categories — reduced from
25 for CPU time, see deviations), extended schedule
`[0,5,10,25,50,100,200,300,500,750,1000]`, layers 0→23, 1000 iterations. Duration ~36 min
(CPU, solo). Outputs in `experiments/pythia_410m/output_deep/` (`deep_results.pt`,
`basin_assessment.md`, `dissolution_pathways.md`, `deep_config.pt`).

**Headline numbers @ 1000 iterations:**

- **Terminal basins: 8 distinct tokens for 8 prompts** — `know`, `/`, `,`, `THE`, `or`,
  `f`, `` (whitespace), `ute`. Every prompt lands on a *different* scattered
  fragment/punctuation token. Zero shared basins.
- **Cross-prompt mean-vector similarity: 0.21** (min −0.08, max 0.80) — the eight final
  states are near-orthogonal, not collapsed together.
- **Per-prompt self-stability** (`cos_sim_mean` between consecutive snapshots) at iter
  1000: mean **0.909**, min 0.773, max 1.000 — and **non-monotonic** along the way (dips
  to 0.34–0.47 around iters 100–200 for several prompts).
- **Token-level:** 7/8 prompts are *still flickering* at iter 1000 (e.g. A01
  `K`→`↵`→`know` across 500→750→1000). Exactly **1/8** (`B01_napoleon`) locks in — to `/`
  from iter 300 onward (self-stability 1.000).

**Decides:** *"Not yet converged at 100–250" vs "structurally fragmented"?* —
**Structurally fragmented.** Quadrupling the horizon (250 → 1000) did **not** produce
convergence: the prompts remain on 8 distinct tokens with cross-prompt similarity 0.21,
and most trajectories still move between snapshots. More iterations buy marginally higher
self-stability (~0.85 at 250 in Control 1 → ~0.91 here), not a shared attractor.

**Interpretation:** Pythia-410m's landscape is genuinely flat/fragmented on this horizon,
not merely under-iterated. A minority of individual prompts *do* settle into their own
private fixed point (the `napoleon`→`/` case), but there is no basin *consolidation*
across prompts — the opposite of GPT-2 Small's five shared semantic basins. This is
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

## 5. `gated_resweep.py` — GPT-2 Small convergence-gated re-sweep (plan addendum)

**What ran:** all 125 GPT-2 Small prompts re-iterated to `max_iter=1000` with an
early-stop gate (`cos_sim_mean > 0.999` for 3 consecutive checks, every 10 iters past
100). Basins classified **at lock-in**, not at a fixed horizon; lock-in iteration
recorded. New engine function `run_atr_gated` (additive); sweep in
`experiments/gpt2_small/gated_resweep.py`, outputs in `output_gated/` (does not touch the
April `.pt`). Duration ~2 h (CPU), checkpointed per prompt. Motivation: Control 1 showed
GPT-2 Small only reached `cos_sim_mean ≈ 0.91` at iter 100, so the published basin table
was read *before* convergence — are the five basins stable or stop-time artefacts?

**Headline numbers:**

- **Convergence: 91/125 (73%) lock in — every one at exactly iter 120** (the earliest the
  gate can fire). The other **34/125 (27%) never reach `cos > 0.999`** and run to 1000.
- **The 34 non-convergers are *exactly* the 34 `Divine` prompts.** Every `Divine` prompt
  fails the tensor gate yet reads `Divine` the whole way — a stable **readout** attractor
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
— **Mostly stable, with one real correction.** `prolet`, `Divine`, and `till` are exactly
stable; the published table's error is confined to **`Anarch`, which was over-counted at
iter 100**: 10 of its 26 prompts are still drifting Anarch→prolet at iter 100 and settle on
`prolet` by their lock-in (iter 120, all converged). Corrected shares: prolet ~43%,
Anarch ~14%.

**The addendum's specific hypothesis is refuted.** `till` is **not** a slow transient — all
19 `till` prompts converge at iter 120 and stay `till` (19/19). The transient basin is
`Anarch`, not `till`.

**Interpretation:** Two clean facts fall out. (1) GPT-2 Small's basins are genuine, not
stop-time noise — 73% of prompts reach a hard fixed point (`cos > 0.999`) within 120
iterations and keep their label. (2) The one basin that never settles at the tensor level,
`Divine`, is precisely the one whose *readout* is most stable — the sharpest single example
in this study of dynamics and decoding coming apart (the analysis doc's central caveat).
The published basin table needs one edit: shift ~10 prompts from `Anarch` to `prolet`.

**Open questions:** `Divine`'s readout-stable / tensor-unsettled split is the natural target
for the ATR-R1/R3 confidence audit (Notebook 2's machinery) — is its readout high-margin
throughout while the tensor wanders? Also: 120 is the *floor* of the gate (first possible
lock-in); a finer check cadence would show the true settling iteration for each prompt.

## 6. `02b_permutation_test.py` — all-warm matrix null (post-close, 2026-07-11)

**What ran:** the pre-registered permutation test for the all-warm cross-similarity
finding (FINDINGS caveat 4). 10,000 random 14-token sets vs the canonical 5 basin +
9 waypoint tokens in W_E. Seed 1969. Reproduction gate passed (observed: 91/91
positive, min 0.181, mean 0.288, max 0.470 — matches the March record).

**Headline numbers:** S1 (all-positive): 9,994/10,000 random sets, p≈1.0.
S2 (min ≥ observed): p=0.167. S3 (mean ≥ observed): p=0.099. Global mean pairwise
cosine of the embedding space: 0.268.

**Decides:** the all-warm property is an **anisotropy artifact** — nearly any 14
tokens in GPT-2 Small's embedding space are all-warm. The compact
"thematic-centre-of-mass subspace" interpretation is withdrawn. The local
semantic-neighbourhood observation is out of scope of this test and stands as
qualitative. Full report: `gpt2_small/output_permutation/permutation_report.md`.

## Synthesis

**The cross-model differences are intrinsic model properties, not readout artefacts.**
Readout ambiguity is real but secondary; it does not explain the headline contrasts.

Four independent lines of evidence converge:

1. **The tensor, not just the token, tells the story (Control 1).** GPT-2 Medium and
   Pythia-160m reach `cos_sim_mean = 1.0000` — their collapses (`D`, `questioned`) are
   genuine fixed points of the forward map. Pythia-410m's mean tensor never stops moving
   (~0.85 at 250 iters). Because `cos_sim_mean` is computed on the activation tensor and
   never passes through token readout, this cleanly separates *dynamics* from *decoding*:
   Pythia-410m's non-convergence is in the dynamics.

2. **Where a basin label appears, it is high-confidence (Notebook 2).** The R1/R3 machinery
   shows readout confidence (logit margin, entropy) rising as GPT-2 Small settles toward a
   basin. Flicker near boundaries exists, but it is not manufacturing the basins — the
   labelled attractors are confident, not decoding noise.

3. **The basins are language-shaped, not weight-universal (null model, Notebook 3).** Random
   tensors iterated through GPT-2 Small converge (positions collapse) but into 18 scattered
   punctuation basins with ~0 overlap with the real five. The `prolet`/`Divine`/`Anarch`/
   `till`/`solidarity` landscape belongs to the on-manifold region real text occupies — ATR
   reads the model *as driven by language*, and real text funnels into **fewer** attractors
   (5) than noise (18).

4. **Pythia-410m stays fragmented with 4× the iterations (Control 3).** At 1000 iterations,
   8 prompts hold 8 distinct terminal tokens with cross-prompt similarity 0.21. Not
   under-iteration — structural.

5. **GPT-2 Small's basins survive proper convergence (gated re-sweep).** 73% of prompts
   hit a hard fixed point (`cos > 0.999`) by iter 120 and keep their basin label; the
   published table needs only one correction (≈10 prompts move Anarch→prolet — Anarch was
   over-counted pre-convergence; the hypothesised `till` transient is in fact 100% stable).
   And the one basin whose tensor never settles, `Divine`, is exactly the one whose *readout*
   is perfectly stable — the study's sharpest single case of dynamics and decoding diverging.

**Bottom line for the original question.** GPT-2 Medium's single `D` basin is a *real* tensor
attractor. Pythia-410m's fragmentation is *genuine structural non-convergence* rooted in the
model (depth/width/corpus geometry), **not** an avoidable distortion of the ATR apparatus. The
readout projection remains a real but secondary source of token-level jitter; the normalisation
step is confirmed irrelevant (LayerNorm scale-invariance, used here in Notebook 3's
calibration). The four models genuinely do not share one failure mode — landscapes differ with
architecture and data, exactly as the analysis doc's "bigger picture" anticipated.

**Confidence:** high for the qualitative direction (four independent controls agree); moderate
on Pythia-410m specifics, since Control 3 ran on an 8-prompt CPU subset. The cleanest
outstanding test is Control 2 (depth control) to pin the effect to depth per se.

## Not scaffolded (follow-on work)

- **Control 2** — Pythia-410m depth control (loop layers 0–11 vs 0–23), holding weights/
  tokenizer/corpus constant. Not built; the plan says note only. Still the cleanest test to
  attribute Pythia-410m's fragmentation to depth per se.
- **ATR-R1/R3 on the `Divine` cohort.** Section 5 found `Divine` is readout-stable while its
  tensor never settles. Run Notebook 2's margin/entropy audit on those 34 GPT-2 Small
  prompts (and across GPT-2 Medium / Pythia-410m at full prompt sets) to confirm the readout
  stays high-confidence while `cos_sim_mean` wanders.
- **Public README update** — the basin table should be corrected (Anarch ~21%→~14%,
  prolet ~35%→~43%; `till` and `Divine` unchanged). Per the run plan, that public-claims
  pass is a separate session.

_Done since the original plan:_ Section 5 (convergence-gated GPT-2 Small re-sweep) — the plan
addendum — is complete; results above.



========================================================================
# SOURCE: The mechanics of the bell (primer)
# (repo path: docs/BELL_PRIMER.md)
========================================================================

# The Mechanics of the Bell

*A plain-language companion to the Session 04 experiments (PR #15). Written to be read start to finish. Assumes [MATH_PRIMER.md](MATH_PRIMER.md) (vectors, cosine similarity, the residual stream, the readout, iterated maps) and [JSPACE_PRIMER.md](JSPACE_PRIMER.md) (the J-lens). Each section states what was done, what was measured, what the numbers were, and what follows from them. Parts 1 to 7 are the measurements; Part 8 says what the numbers mean about the model as a trained object; Part 9 states the implications, sorted by how firmly they are held.*

**Terminology note:** the direction the cycle moves along is called **the flip axis, d** throughout this document. Earlier session files called it "the hinge"; script names, output folder names, and JSON keys (`08_hinge_eigenvalue.py`, `output_hinge_eigen/`, `d_hinge`) keep the old word as frozen labels.

**The experiment reports this document summarises:** `output_lagk/lagk_report.md`, `output_hinge_eigen/hinge_eigenvalue.md`, `output_glitch/glitch_alignment.md`, `output_jlens_phase/jlens_phase.md`, `output_suppression/suppression_report.md`, all under `experiments/gpt2_small/`. They are the primary record; where this document and a report differ, the report governs.

---

## Part 1: The cycle, and why it was missed

### 1.1 What Divine is

"The bell" is the project's name for the following measured fact. Take the Syntactic prompt's state at iteration 1000 and call it A. Apply the map once (one forward pass plus the energy rescale): the result, B, is a different state. Apply the map to B: the result is A again, exactly.

- cos(A, B) = 0.685. A and B are clearly different states.
- cos(A, f(f(A))) = 1.000000. Two applications return the start, to machine precision.

So the trajectory is A, B, A, B, forever. This is called a **limit cycle of period 2**: period 2 because two applications of the map return you to where you started. A and B are called **phase A** and **phase B**. It is a different kind of stable object from a fixed point (where one application returns the start).

One measured simplification used throughout: at the bell, all 10 token positions of the tensor hold identical vectors (the row spread is exactly 0.0). So the whole cycle is described by a single 768-dimensional vector per phase, and "the state" needs no position qualifier.

**Where recorded:** FINDINGS.md F9; `output_divine_motion/bell_anatomy.md`.

### 1.2 Reading cos(A,B) = 0.685 exactly

The 0.685 is not a vague "fairly similar". It has an exact reading, in four steps.

First, what the number computes: A and B are each a list of 768 numbers; multiply matching entries, add everything up, divide by the two lengths. The result measures direction agreement only: +1 proportional, 0 unrelated, -1 opposite.

Second, the split. For any two vectors, define M = the average of A and B (**the pivot**) and d = half their difference. Then A = M + d and B = M - d as pure arithmetic, true of any pair. Three measured facts make this bookkeeping meaningful here: the same d returns every cycle, to machine precision, for a thousand iterations (that is what period 2 means); d is perpendicular to M in the equal-scale frame, so the split is clean; and the map singles d out (Part 2). Any two states trivially "differ along a direction". A difference direction that is fixed, that reverses sign each step, and that the network treats unlike any other direction is not trivial.

Third, the quantity. With the split clean and both states the same length, the cosine equals the fraction of the state's energy in the shared part minus the fraction in the flipped part. Solving for 0.685: **about 84 percent of each state's energy is the shared part M, about 16 percent is the flipped part d** (84.2 minus 15.8 gives back 0.685). One sixth of the state's energy oscillates along a single fixed axis; five sixths stand still.

Fourth, what "one direction out of 768" means. A direction is an axis through the 768-dimensional space, and almost never one of the 768 slots in the list: d is a specific weighted combination of all 768 entries. Choose a coordinate system whose first axis is d, and the step from A to B changes exactly one coordinate (its sign) and leaves the other 767 untouched. The linearisation in Part 2 confirms the network really is close to the identity along other directions near these states, so "reproduces 767 numbers, reverses the 768th" is a description of the map there, not just of the pair.

### 1.3 Why it went undetected

Every earlier run saved the state at even intervals: every 10 iterations, or every 50. Ten applications of the map is five full cycles, which returns the state exactly to the phase it was in. Therefore every saved snapshot was the same phase, the saved sequence was constant, and the state appeared frozen. The general name for this failure is **aliasing**: sampling a repeating process at an interval that is a multiple of its period, so the repetition is invisible in the samples.

The detection came from a contradiction between two measurements: snapshots 10 iterations apart matched to six decimal places, while consecutive iterations matched only at 0.685. Both cannot be true of a fixed point. Both are necessarily true of a period-2 cycle.

### 1.4 The convergence-test fix

The old convergence test compared each iteration to the previous one and declared convergence when the cosine stayed above 0.999. For the bell, that comparison returns 0.685 every time, so the test could never pass, regardless of how long the run continued. This is arithmetic, not a tuning problem.

The fix, added to `atr_engine.py` in Session 04: the comparison interval is now a parameter, `gate_lag`. With `gate_lag = 2`, the test compares each iteration to the one two steps back. Under that test, Divine passes at the standard 0.999 threshold. The default remains `gate_lag = 1`, and with the default the engine's behaviour is unchanged (verified bit-identical against the pre-change code on a real run).

A helper, `lag_scan`, measures the cosine at every comparison interval from 1 to 8 at once. For the bell the result is: odd intervals all 0.685, even intervals all 1.000000. This pattern is the direct signature of period 2, and the same table would expose a period-4 cycle: its exact 1.000000 return would appear only at intervals 4 and 8. (Intermediate intervals could still score above the threshold if that cycle's states happened to lie close together; the exact return, not merely a high score, is the signature.) No one has yet run this scan on the other 33 non-converging prompts; that requires the prompt library (issue #9).

Two limits of the fix, stated in the report: it detects cycles, not slow drift (the committed noise state moves slowly enough by iteration 1000 to pass the cosine threshold at every interval while genuinely still moving); and the full re-classification of the 125-prompt sweep has not been run.

**Where recorded:** `atr_engine.py` (`gate_lag`, `lag_scan`); `output_lagk/lagk_report.md`.

---

## Part 2: The derivative measurement

### 2.1 The quantity being measured

Define two objects from the cycle:

- **The flip axis, d**: the normalised difference between the phases, d = (A - B) / ‖A - B‖. This is the direction along which the two phases differ, which is the direction the state moves along on every step.
- **The pivot, M**: the midpoint, M = (A + B) / 2.

The question: what does one application of the map do to a small displacement along d? Formally this is a directional derivative. Practically it was measured two independent ways: automatic differentiation (`torch.func.jvp`), and directly, by adding a small multiple of d to the base point, running one iteration, subtracting the undisplaced result, and dividing by the size of the displacement. The two methods agreed to 3 or 4 significant figures on every reported number, and the direct method was repeated at two displacement sizes to confirm the answer did not depend on the size.

The summary number is the **multiplier along d**: the component of the returned displacement that lies along d, with sign. Multiplier +1 means the displacement passes through one iteration unchanged. Multiplier -1 means it returns with the same size, pointing the opposite way. Multiplier -4 means it returns pointing the opposite way, four times larger.

### 2.2 The results

Measured at the pivot M:

- Along d: multiplier **-4.3**. A small displacement along the flip axis returns inverted and 4.3 times larger.
- Along three random control directions: multipliers **+0.9 to +1.2**. Ordinary directions pass through roughly unchanged.
- M itself maps almost to itself: cos(f(M), M) = 0.995.

So M is nearly a fixed point, but unstable along d: any component along d grows by a factor of about 4 per step, flipping sign each time. The map treats d differently from every other direction tested. (Tested means d plus three random probes; the full set of 768 independent directions has not been examined, so "one unstable direction" is the simplest reading consistent with these probes, not a proven count.)

Measured around the full two-step cycle (the derivative at A composed with the derivative at B):

- Along d: net multiplier **+0.1** (the component of the returned displacement along d: a projected multiplier, since the return is only partly aligned with the axis). Positive, because two inversions cancel; and much smaller than 1, meaning any deviation from the cycle shrinks by roughly 90 percent every two steps.

These two numbers together explain the observed behaviour. The system cannot rest at M (deviations along d grow). Deviations from the two-step cycle shrink along every direction tested. What is observed, and what these numbers make stable, is the alternation itself. The Session 03 conjecture had predicted a multiplier near -1 at the pivot; the sign was right, the size was not. A multiplier of -1 would be a marginal, borderline case. The measured -4.3 with a two-step contraction of +0.1 is a strongly stable oscillation. The technical name for this structure (a near-fixed point whose single unstable direction produces a stable period-2 cycle around it) is a **period-doubling** configuration.

**Where recorded:** `output_hinge_eigen/hinge_eigenvalue.md`, results 1 and 2; the numbers file `hinge_eigenvalue.json`.

---

## Part 3: Locating the inversion

### 3.1 Attention heads, defined

Each of GPT-2 Small's 12 layers contains an attention block and an MLP block, and both add their outputs into the residual stream (MATH_PRIMER 2.3). One refinement is needed here: an attention block is not one unit. It is 12 separate **heads**, each with its own learned weights, each computing its own output, all added into the stream. GPT-2 Small therefore has 144 heads. The naming convention is layer then head: L11.H8 is layer 11, head 8.

### 3.2 The measurement

Add a small multiple of d to the state at the point where the loop re-injects it (the input of layer 0). Run one forward pass. At each layer boundary, subtract the undisplaced run from the displaced run to get the propagated displacement, and measure its cosine with d.

Results, per layer boundary:

- After layers 0 through 10: cosine **+0.88 to +0.97**. The displacement is transmitted essentially unchanged in direction through eleven layers.
- After layer 11: cosine **-0.99**. The inversion happens inside layer 11 and nowhere else.

Splitting layer 11's two blocks (their added outputs can be measured separately): the MLP's contribution to the flip is -0.17; the attention block's is -2.0. Splitting the attention block's 12 heads: **head 8 contributes 99.1 percent of the inversion**. The next largest head contributes 0.014.

### 3.3 What this attribution means, and what it does not

It means: in this region of state space, for this direction, the sign inversion that sustains the cycle is performed by one identifiable component, L11.H8. Remove or isolate that component's output and the flip follows it. This is the standard form of a mechanistic attribution: a global behaviour of the network traced to a specific part.

It does not mean: that this head "causes Divine" in general, or that it behaves this way on ordinary text. Part 7 tests exactly those questions.

**Where recorded:** `output_hinge_eigen/hinge_eigenvalue.md`, result 3 (the per-layer table and per-head split).

---

## Part 4: The embedding alignment

### 4.1 The token cluster in question

Every token has a 768-number embedding vector, a row of W_E (MATH_PRIMER 2.2). Average all 50,257 rows to get the mean embedding. Most tokens sit far from this mean, because training moved them. A small set of tokens sits unusually close to it: these are tokens that almost never occurred in the training text, so their vectors were barely updated from initialisation.

The known members of this set include the published **glitch tokens** (" SolidGoldMagikarp", " petertodd", "ertodd" and related; Rumbelow and Watkins 2023). The mechanism is documented: these strings occurred often enough in the corpus used to build the tokenizer (largely as Reddit usernames) to receive vocabulary entries, but the pages containing them were filtered out before the model's weight training, so the model never learned anything about them.

Session 04 defined the cluster two independent ways: geometrically (the 0.1 percent of the vocabulary closest to the mean embedding, 50 tokens) and by list (52 published glitch-token strings matched into the vocabulary). The two definitions agree with each other (their centroid directions have cosine +0.67).

### 4.2 The measurements

Let u be the normalised direction from the mean embedding toward the cluster's centroid. Measured:

- **cos(d, u) = -0.596** for the geometric cluster, **-0.456** for the published list. The sign convention: negative means the B side of d points toward the cluster.
- Chance comparison 1: the same cosine computed for 1000 clusters of randomly chosen tokens never exceeded 0.30 in magnitude. Zero exceedances in 1000 draws gives p ≤ 0.001 under the standard finite-sample bound.
- Chance comparison 2 (stricter): 1000 clusters of tokens chosen to match the glitch cluster's embedding norms. These lean the opposite way (mean +0.48), so the result is not explained by embedding norm. Norm is the available proxy for token frequency here; a null matched directly on corpus frequency was not run.
- Of the 50 vocabulary tokens whose embeddings best align with the B side of d, **45 are in the geometric cluster**. Of the 50 best aligned with the A side, most are the highest-frequency function words: " the", " in", " on", the comma.
- A side finding from the controls: low embedding norm does not identify glitch tokens in GPT-2. The lowest-norm tokens are the most frequent function words, and their apparent alignment with d disappears under the norm-matched comparison. In this model the glitch signature is proximity to the mean, not small norm.

### 4.3 What follows

Stated plainly: the direction along which Divine oscillates runs between the embedding region of the most-trained tokens (one end) and the region of never-trained tokens (the other end). This is a geometric statement about where the cycle sits in the model's representation space, not a statement that the model is processing or "referring to" those tokens. Its significance: glitch tokens were previously known only as anomalous inputs; here the untrained region of embedding space plays a role in the model's internal dynamics with no glitch token ever appearing in the input.

**Where recorded:** `output_glitch/glitch_alignment.md`, including the two top-50 token lists in full.

---

## Part 5: The lens measurements

### 5.1 The quantity

JSPACE_PRIMER describes the J-lens. For this document one number matters: for any state (or direction), the **span share** is the fraction of it (of its squared length) that can be expressed as a combination of the lens's fixed set of directions. Share 1.0 means fully expressible in that set; 0 means orthogonal to all of it. A random direction scores about 0.25 against this particular lens (193 directions in a 768-dimensional space, measured directly as a baseline). A second variant, the sparse share, restricts the combination to a few nonnegative terms; it behaves consistently and is reported alongside.

The pilot (issue #8) measured Divine before the cycle was known, so it measured one phase. Session 04 repeated the identical measurement on both phases and the midpoint. The replay of the pilot's own states reproduced its recorded numbers to 7 decimal places, which is the check that the instrument was reassembled correctly.

### 5.2 The results (last layer; the per-layer tables are in the report)

| state | span share |
|:---|---:|
| phase A | 0.173 |
| phase B | 0.123 |
| midpoint M | 0.180 |
| prolet (average) | 0.157 |
| noise (average) | 0.114 |

- B scores below A at every layer, on both variants, and near the noise level on the span variant. The pilot's finding ("Divine at least as expressible as prolet") was a fact about phase A only.
- M scores above everything else this project has measured. Separately, the bell-anatomy work had already shown that the stable ` Divine` readout token is produced by M's direction rather than either phase's.
- The flip axis as a direction, measured on both constructions. On the physical on-shell axis d_sym: span share 0.013 at the last layer against the 0.25 baseline (5 percent of chance; mean over layers 0.021). Decomposed by effect on output-token scores: the readout-quiet part of d_sym carries 97.0 percent of its energy and overlaps the lens almost nowhere (at most 8 percent of chance at any layer); the readout-visible part carries 1.2 percent of the energy, sits at the chance level in the earliest layers, and rises to 2.3 times chance by the last. These two subsets cover 98.2 percent of the axis energy; the remaining components are unassigned. The frame-mixed committed axis (as the earlier anatomy script built it) reads higher (0.145 at the last layer; its subsets cover 86 percent), because it is contaminated with the pivot's own direction. The physical axis sharpens the conclusion and does not change its direction.

### 5.3 What follows, at what confidence

The motion of the cycle is carried by a direction that is almost entirely outside both the output projection and the lens's set of directions (span 0.013 against 0.25 chance on the physical axis), while the midpoint the motion straddles is the most lens-expressible state measured. The two kinds of low visibility (to the output vocabulary, to the lens) coincide on the measured subsets of this direction at every layer; on the physical construction those subsets cover 98.2 percent of the axis energy.

Confidence: everything in this part inherits the pilot's limitations. The lens is a reduced version built once from limited data, on a 124-million-parameter model for which no one has demonstrated an organised workspace. These are internally consistent pilot-grade measurements, not established properties of the model.

**Where recorded:** `output_jlens_phase/jlens_phase.md`.

---

## Part 6: The frames error

Session 04 also caught an error worth understanding. The bell-anatomy script had built its flip axis from mismatched ingredients: phase A at raw scale, phase B rescaled to the loop's energy shell (MATH_PRIMER 1.3). The resulting "committed d" leans 0.97 toward A itself and only 0.62 toward the clean flip axis. The eigenvalue work caught this, defined the symmetric on-shell axis properly, measured everything both ways, and showed the two versions agree at 0.97 once the loop's own normalisation strips the contamination. The earlier results stand, with the caveat recorded; harmonising every script on the clean definition is queued follow-up work.

The reason to keep this section in a learning document: twice in one session, a measurement error produced a plausible result (a frozen state; a multiplier near -1 that flattered the conjecture). Both were caught by measuring the same quantity two independent ways. That habit, not any single number, is the method.

**Where recorded:** `output_hinge_eigen/hinge_eigenvalue.md`, "the map, the frames, and two hinges"; the method flag in PR #15's description.

---

## Part 7: The suppression tests

A hypothesis followed from Parts 2 and 3: perhaps L11.H8 is a **suppression head**, a documented behaviour class in GPT-2 Small (the best-studied example is L10.H7, which detects the token the network is about to promote and writes the negative of it, damping over-confidence), and the closed loop turns that one-shot negative correction into a sustained oscillation. Three tests (experiment 11):

**Test 1, the weights.** Each head contains two matrices whose product is a fixed 768-to-768 transform (the OV circuit). Because the bell state is position-uniform, the head's in-loop output is exactly this transform applied to the (scaled) input, independent of the attention pattern (verified to relative error 1.4e-07). Along the flip axis, L11.H8's transform returns cos -0.96 at gain 63.7: rank 1 of all 144 heads on both measures, with the runner-up 50 times weaker, and no comparable response on random directions. The inversion is physically located in this one head's weight matrices.

**Test 2, causality.** Re-running the loop from phase A with head 8's output zeroed kills the cycle within about 10 iterations; by iteration 300 every lag from 1 to 8 reads 1.000000. The state settles at a genuinely new fixed point (cos to A +0.14, to B -0.61, to M -0.25) whose readout is " the" at probability 0.024, a nearly flat distribution over common words. Zeroing a different same-layer head instead (L11.H0) leaves a period-2 cycle running. The head is specifically load-bearing: no head 8, no bell.

**Test 3, the refutation.** On twelve ordinary sentences run normally through the model, L11.H8 shows the opposite of the suppression signature: it raises the score of the token it attends to at 91.4 percent of positions (mean effect +5.97). The known suppression head L10.H7, run through the identical protocol, shows the documented negative signature (87.1 percent negative; 100 percent negative where its attention exceeds 0.2), which validates the measurement. L11.H8 is, on ordinary text, a copy promoter, not a suppressor.

**Verdict:** the inversion is confirmed in the weights and causally necessary for the cycle, but it is not a trained suppression behaviour. On ordinary text the head does something else entirely. The reading this supports: the inversion is a structural side effect of training, not a function, and the loop is what makes it matter. One recorded limit: only suppression at the token-readout level was measured; a suppression role in some other basis is not excluded.

**Where recorded:** `output_suppression/suppression_report.md`.

---

## Part 8: What the numbers say about the model

The model is a fixed block of 124 million learned numbers, arranged as 12 layers of transforms acting on a 768-dimensional state, with two dictionary matrices connecting that state to the vocabulary. Training set every number by nudging it to improve next-token prediction on web text; nobody chose any individual value. Each headline number above is a property of that finished block.

**cos(A,B) = 0.685** says the cycle is geometrically minimal: one sixth of the state's energy oscillating along one axis, five sixths standing still (Part 1.2). The network reproduces essentially the whole state and reverses one direction.

**-4.3 and +0.1** say the trained weights contain a strong reversal along one specific axis, and that the surrounding saturating parts absorb its overshoot. Training never penalised the reversal, because in training the network never receives its own output as input, so a reversal is never applied twice in a row and has no consequence. These are permanent properties of the weights that single-pass use never exercises; the loop compounds them until they dominate. The deep stability (+0.1 round trip) is why the cycle reproduced exactly across different machines.

**Gain 63.7, rank 1 of 144** says the inversion is not distributed knowledge: it is localised in one component's weight matrices, put there as a side effect of whatever that head was actually trained to do (Part 7: raising the scores of attended tokens). Large stretch factors inside heads are normal. The loop found the one whose largest axis reproduces itself.

**The poles (cos -0.596)** say the training data's frequency distribution is written into the embedding geometry, and the cycle runs along it. Constantly-seen tokens had their vectors shaped by millions of updates; never-seen tokens kept their initial vectors, which cluster near the mean. The flip axis connects the most-updated region to the never-updated region. The ablation result supports the frequency reading: remove the inverting head and the loop settles into a state reading " the" at p 0.024, the most-common-word region, apparently the space's default resting place. The head's reversal is what keeps the state from resting there.

**p = 0.505, with 73 percent of d in W_U's weakest directions,** says the output map's shape hides the motion. W_U maps 768 numbers to 50,257 scores; a rectangular map necessarily transmits some directions barely at all, including any direction that shifts all scores nearly equally, which the softmax cancels. The stable part of the cycle projects strongly onto one vocabulary column; the moving part projects almost nowhere. The motion is real in the state and nearly absent in the output because of the map's geometry, not because the motion is small.

**Lens share 0.013 against 0.25 chance** (on the physical axis; the frame-mixed construction reads 0.145) says the motion lies almost entirely outside the subspace the model uses for expressible content generally: closer to internal bookkeeping (frequency, scale) than to meaning.

In one sentence: training built very strong weights around predicting common words, left a never-trained region as a byproduct, and incidentally wrote a large sign-reversal into one late head; none of this matters in ordinary use, and the feedback loop settled into the one axis where all three facts intersect.

---

## Part 9: Implications, by scope

**For this project (firm, because measured).** Every "never converges" claim made with the lag-1 gate is provisional, including the cross-model verdicts: Pythia-410m's "no consolidation" may be aliased cycles, and the Act II table could change shape when re-gated. The landscape contains at least two kinds of stable object, so basin counting needs a type column. And the methods lesson is proven twice in one repository (aliasing; the frames error): measurement assumptions hid the most interesting object, and measuring one thing two independent ways is what caught both.

**For how models are understood (moderately firm, one instance).** Three points. First, models contain real structure no single forward pass can show: the reversal sat in the weights through every benchmark this model ever ran, and only compounding under iteration made it visible; iteration is a measurement instrument for weight properties that behaviour hides. Second, head labels are regime-bound: the same head is a copy promoter on ordinary text and the inverter that sustains the cycle in the loop; a catalogue that assigns each head one function is implicitly a catalogue of one regime. Third, some deep structure has no functional story: the tests came back "structural accident", and methods have to be able to accept "it is sediment, there is no reason" as a correct final answer.

**For the workspace question (pilot confidence).** The bell is an existence proof that a model can carry a persistent, structured, causally load-bearing internal state that is almost entirely outside both its output vocabulary and its expressible subspace. If that survives bigger models and better lenses, it matters for any approach that relies on models reporting their own states. Qualifier: the regime is artificial, and nobody has shown this arises during ordinary computation.

**For practice (the transferable point).** Closed loops around language models are increasingly normal (agents reading their own outputs; models trained on model-generated text). The specific bell does not transfer; the demonstrated principle does: single-pass evaluation does not bound closed-loop behaviour, because feedback amplifies exactly what single passes suppress. Also demonstrated: removing one of 144 heads completely changed the loop's terminal behaviour, so small structural edits can silently rewrite a system's dynamical properties.

**For the piece.** The founding question was what remains of a voice when the medium's own character takes over. The bell answers with a case the rooms never produced: what remains is not always a tone; sometimes it is a repetition, sustained by one component of the medium, running between the medium's most-used and never-used materials, inaudible in the medium's own voice. The sonification has a score.

**What none of this implies.** Nothing about intent, experience, or the model "doing" anything in ordinary use: the regime is artificial and the verdict was accident, not function. Nothing yet about the other 33 ringing prompts, longer periods, or other models: unmeasured. One model at 124 million parameters generalises to nothing until it is made to.

---

## Part 10: Glossary

| Term | Definition | Where |
|:---|:---|:---|
| Limit cycle, period 2 | Two states the map exchanges: f(A) = B, f(B) = A | F9 |
| Phase A / phase B | The two states of the cycle | all Session 04 tables |
| The bell | Project name for the Divine period-2 cycle | issue #14 |
| Pivot M | (A + B) / 2; maps nearly to itself; unstable along d only | Part 2 |
| Flip axis d | (A - B) normalised; the direction of the cycle's motion; called "the hinge" in earlier files | Parts 1, 2-5 |
| Aliasing | Sampling a repeating process at a multiple of its period, hiding it | Part 1.3 |
| gate_lag | Engine parameter: which earlier iterate the convergence test compares against | `atr_engine.py` |
| lag_scan | Helper reporting the comparison cosine at intervals 1 to 8 | `output_lagk/` |
| Multiplier along d | Signed factor applied to a small displacement along d by one iteration | -4.3 at M |
| Period doubling | A near-fixed point whose one unstable direction yields a stable 2-cycle | Part 2.2 |
| Attention head | One of 12 independent weighted units per attention block; 144 in the model | L11.H8 |
| OV circuit | The fixed 768-to-768 transform inside one head (two matrices multiplied) | Part 7 test 1 |
| Ablation | Removing one component's output to test whether a behaviour depends on it | Part 7 test 2 |
| Copy suppression / promotion | A head lowering / raising the score of the token it attends to | Part 7 test 3 |
| Attribution | Measuring each component's separate contribution to an effect | Part 3 |
| Glitch tokens | Vocabulary entries absent from weight training; embeddings near the mean | Part 4 |
| Norm-matched control | Chance comparison using token sets matched on embedding norm | Part 4.2 |
| Span share | Fraction of a state or direction expressible in the lens's directions | Part 5 |
| Frame | The scale convention a vector is expressed in; mixing frames was the caught error | Part 6 |

---

## Summary

Divine is an exact period-2 limit cycle: two states exchanged by the map, hidden from all earlier runs because even-interval snapshots sample a period-2 process in a single phase. Exactly measured, the oscillation is one sixth of the state's energy moving along one fixed axis (the flip axis d) while five sixths stand still. The convergence test now supports comparison at lag 2, under which Divine converges. One iteration multiplies a small displacement along d by -4.3 (inverted and amplified) while leaving other tested directions essentially unchanged; over two iterations the net factor is +0.1, so the cycle is strongly stable. The inversion is physically located in the weight matrices of one component, attention head 8 of layer 11: strongest inverter of all 144 heads along this axis, and causally necessary (zero its output and the loop settles at a new fixed point reading " the" at p 0.024). On ordinary text that head raises, not lowers, the scores of tokens it attends to, so the inversion is a structural side effect of training rather than a trained behaviour. The axis runs between the embedding regions of the most-trained and never-trained tokens, and it lies almost entirely outside both the output projection and the J-lens's expressible set (span 0.013 against 0.25 chance on the physical axis), while the cycle's midpoint is the most lens-expressible state the project has measured. One measurement error (a flip axis computed from mixed scales) was found and corrected without changing the conclusions. Open next steps: the lag scan on the remaining 33 non-converging prompts (blocked on issue #9), the promoted-versus-inverted structure of L11.H8's weight matrices, and the same measurements on other models.



========================================================================
# SOURCE: Understanding ATR (accessible mechanism)
# (repo path: docs/UNDERSTANDING.md)
========================================================================

# Technical Explanation: The Feedback Mechanism

## What Is Being Fed Back?

When a language model processes a prompt, the text passes through several stages:

1. **Tokenisation** — the text is split into subword tokens (e.g., "proletariat" → `prol` + `etar` + `iat`)
2. **Embedding** — each token is mapped to a 768-dimensional vector (for GPT-2 Small)
3. **Layer processing** — the sequence of vectors passes through 12 transformer layers, each containing attention heads and MLPs that read from and write to a shared **residual stream**
4. **Unembedding** — at the final layer, the residual stream is projected back into vocabulary space to produce a probability distribution over the next token

In normal operation, step 4 collapses the rich 768-dimensional state into a single token prediction (an argmax over 50,257 possibilities). This is a massive information bottleneck — the model "considered" thousands of possibilities, but only one token survives.

## The Lucier Loop Bypasses This Bottleneck

Instead of decoding the output into text and re-tokenising it as a new prompt, we:

1. **Extract** the full residual stream tensor at the output of Layer 11 — a `[seq_len × 768]` matrix containing the model's complete internal state across all token positions
2. **Normalise** this tensor to maintain constant energy (L2 norm), preventing numerical explosion
3. **Inject** the normalised tensor directly into the input of Layer 0 on the next forward pass, using a programmatic hook that overwrites the normal token embeddings
4. **Repeat** — the model processes its own internal state as if it were a new input

This creates a continuous feedback loop in 768-dimensional space. No information is destroyed by the argmax bottleneck. The full geometric structure of the model's internal representation is preserved and fed back through the nonlinear transformer stack.

```
Prompt → Tokenise → Embed → [Layer 0 ... Layer 11] → Extract residual tensor
                      ↑                                        |
                      └──────── Normalise & Re-inject ─────────┘
                                    (repeat 500×)
```

## Why This Produces Attractors

Repeated application of any function to its own output — f(f(f(x))) — tends toward fixed points or limit cycles. In a linear system, this is **power iteration**: the dominant eigenvector of the transformation matrix is progressively amplified while all others decay. The converged state is the "eigenvoice" of the matrix.

A transformer is not linear — it includes LayerNorm, softmax attention with dynamically recomputed queries/keys/values, and nonlinear MLP activations. But the same dynamical principle applies: the system has attractor states determined by its weight geometry, and iterative re-injection converges toward them.

The L2 normalisation is critical: without it, the tensor's norm explodes exponentially (reaching 1.5M by iteration 500), making the dynamics meaningless. With normalisation, the system is energy-conservative, and convergence to stable attractors becomes possible.

## Why This Is Not a Text Loop

This distinction is fundamental:

| | Text Loop | Activation Loop (This Experiment) |
|:---|:---|:---|
| **What's fed back** | A decoded token (1 integer) | The full residual stream (`[seq_len × 768]` floats) |
| **Information preserved** | Only the argmax winner | The entire superposition of all 50,257 token candidates |
| **Dynamics** | Discrete, lossy, stochastic | Continuous, lossless, deterministic |
| **What converges** | The model's text generation habits | The stable states of the iterated model — which depend on the input regime (see below) |

## Key Parameters

| Parameter | Value | Rationale |
|:---|:---|:---|
| **Model** | GPT-2 Small (124M params) | Well-studied, manageable size, known training data (WebText/Reddit) |
| **Layer window** | 0 → 11 (full stack) | The entire architecture acts as the "room" |
| **Normalisation** | Per-iteration L2 rescaling to initial norm | Prevents energy explosion, enables stable convergence |
| **Iteration schedule** | `[0, 2, 3, 5, 10, 20, 50, 100, 250, 500]` | Logarithmic — captures both early dynamics and deep convergence |
| **Temperature** | N/A (deterministic) | No sampling — pure forward-pass dynamics |

## One Important Correction (What the Attractors Are Not)

An earlier framing of this project described the attractors as "the model's weight geometry made audible" — as if the basins were universal properties of the weights that any input would eventually reveal. The null-model control showed this is wrong: iterate pure random noise instead of a prompt and the loop still converges, but into a *different* set of attractors (eighteen scattered punctuation tokens, none of them the five semantic basins).

The correct statement: **the attractors are stable states of the model as driven by a particular kind of input.** Language-shaped input funnels into few, semantically coherent basins; noise scatters into many meaningless ones. And the phenomenon is model-specific — GPT-2 Medium, trained on the same corpus, collapses everything into a single empty token. The full evidence is in [FINDINGS.md](FINDINGS.md).



========================================================================
# SOURCE: Lucier-transformer isomorphism
# (repo path: docs/ISOMORPHISM.md)
========================================================================

# Mathematical Correspondence: Lucier's Room and Iterative Activation Re-injection

## The Acoustic Case: Linear Power Iteration

Lucier's *I Am Sitting in a Room* (1969) implements classical power iteration on an acoustic transfer function.

A room's acoustics can be modelled as a linear operator *H*: ℝⁿ → ℝⁿ acting on a discrete signal vector. Acoustic wave propagation obeys the superposition principle — the sum of two sound sources produces the sum of their individual reverberant responses — making *H* a genuine linear map.

Lucier's iterative process is:

```
s₀ = record(speech)
sₙ₊₁ = H(sₙ)                     # Play sₙ into the room, record the output
```

This is matrix-vector power iteration: *Hⁿs₀*. By the spectral theorem, after sufficient iterations:

```
sₙ → c · v₁     as   n → ∞
```

where *v₁* is the eigenvector corresponding to the dominant eigenvalue of *H* — the room's resonant mode. All other frequency components decay exponentially at rates determined by their eigenvalue magnitudes.

The tape recorder serves as the re-injection mechanism: it captures the output state (the reverberant audio) and feeds it back as the next input, closing the loop.

**Result**: The final recording is a pure drone at the room's resonant frequency. Speech has dissolved into architecture.

## The Transformer Case: Nonlinear Power Iteration

This experiment applies the same structural operation to GPT-2 Small, but the operator is nonlinear.

Let *f*: ℝ^(T×768) → ℝ^(T×768) denote the full transformer forward pass (Layer 0 through Layer 11). The iteration is:

```
x₀ = f(embed(prompt))
xₙ₊₁ = f(normalise(xₙ))
```

Unlike Lucier's *H*, the transformer *f* includes:

| Component | Nonlinearity |
|:---|:---|
| LayerNorm | Rescaling and recentring |
| Attention (softmax) | Data-dependent gating over value vectors |
| QKV computation | Dynamically recomputed at each iteration from the current state |
| MLP (GeLU) | Element-wise nonlinear activation |
| Residual connections | Additive skip — linear, but composes with nonlinear layers |

This means:
- **No spectral theorem guarantee.** The system is not guaranteed to converge to a single dominant eigenvector.
- **Multiple fixed points are possible.** Nonlinear maps can have several attractors with distinct basins of attraction.
- **Basin boundaries may be fractal or sensitive to initial conditions.** Whether a prompt converges to `prolet` or `Divine` depends on the geometry of the input relative to the basin boundaries in ℝ^(T×768).

**Result**: Instead of a single pure tone (one eigenvector), the system reveals a *landscape* of attractors — multiple "resonant modes" of the weight geometry.

## The Isomorphism

| Acoustic (Lucier) | LLM (This Experiment) | Mathematical Role |
|:---|:---|:---|
| Room | Transformer weight matrices (W_Q, W_K, W_V, W_O, W_in, W_out × 12 layers) | The operator being iterated |
| Audio signal | Residual stream tensor `[T, 768]` | The state vector |
| Tape recorder | TransformerLens hook (extract → normalise → re-inject) | The feedback mechanism |
| Room resonant frequency | Attractor state (`prolet`, `Divine`, ...) | Fixed point of the iterated map |
| Spectral decay of non-resonant frequencies | Dissolution of semantic content through iterative passes | Transient dynamics before convergence |
| Pure drone | Terminal token sequence (uniform across positions) | The converged state |
| Linear operator *H* | Nonlinear map *f* | Class of the operator |
| Guaranteed single dominant eigenmode | Multiple basins with distinct attractors | Consequence of (non)linearity |

## Key Insight

Lucier's room can only have **one** dominant resonant mode (the largest eigenvalue wins). A transformer, by virtue of its nonlinearity, can have **many**. This is why the experiment reveals an *attractor landscape* rather than a single fixed point — and why mapping this landscape (via systematic prompt variation) is scientifically productive.

The transition from linear to nonlinear power iteration is the transition from **one voice** to **a chorus of voices** latent in the architecture. Which voice emerges depends on where in the activation space you begin — which is precisely what the 125-prompt sweep (Stage 1) is designed to map.



========================================================================
# SOURCE: Maths primer
# (repo path: docs/MATH_PRIMER.md)
========================================================================

# The Mathematics of the Room

*A self-study companion to the ATR experiments. Written to be read start to finish, offline, with no prerequisites beyond curiosity. Every concept is introduced from scratch and then pointed at the exact place it appears in this repository.*

**Where this fits among the other docs:** [UNDERSTANDING.md](UNDERSTANDING.md) explains the *mechanism* (what the feedback loop does and why it produces attractors). [TECHNICAL.md](TECHNICAL.md) is the *formal specification* (the notation-dense version for someone who already has the vocabulary). This document is the missing rung below both: it teaches the vocabulary itself. Read this first, then UNDERSTANDING.md, then TECHNICAL.md, and the third one should feel like a summary rather than a wall.

---

## Part 1: The Objects

### 1.1 Vectors, and what "768-dimensional" means

A vector is just a list of numbers. The vector `[3, 4]` is two-dimensional: you can draw it as an arrow on a page, going 3 across and 4 up. A three-dimensional vector `[3, 4, 2]` is an arrow in a room. A **768-dimensional vector** is a list of 768 numbers, and it is an arrow in a space you cannot picture, but the mathematics does not care that you cannot picture it. Every operation you can do on the arrow on the page (measure its length, compare its direction with another arrow, add it to another arrow) works identically on the list of 768 numbers.

This is the single most useful mental move in this whole subject: **stop trying to visualise high-dimensional space and instead trust the operations.** Length, angle, and distance all have exact formulas that work in any number of dimensions. When you read "the residual vector at the final token position," picture an arrow if it helps, but what the code actually holds is a list of 768 floats.

Why 768? That is just GPT-2 Small's design choice, called `d_model`. Every token the model processes is represented internally as one of these 768-number lists. GPT-2 Medium uses 1024. Pythia-410m also uses 1024. Bigger models use wider vectors.

**Where you've seen it:** every mention of `[768]` in TECHNICAL.md's snapshot table, and `d_model` in `atr_engine.py`.

### 1.2 Matrices and tensors

A matrix is a grid of numbers, or equivalently a stack of vectors. If your prompt has 10 tokens and each token's internal state is a 768-dimensional vector, then the model's full internal state for that prompt is a **10 × 768 matrix**: ten rows, one per token position, each row a 768-number vector.

"Tensor" is the general word for these objects at any number of axes: a vector is a 1-axis tensor, a matrix is a 2-axis tensor, and so on. When this project says **"the full activation tensor"**, it means the `[T × 768]` matrix, where `T` is the number of token positions in the prompt. That grid of numbers is the entire thing being extracted, rescaled, and re-injected on every pass of the loop. It is the "recording" in the Lucier analogy: not one note, but the whole room at once.

**Where you've seen it:** `[seq_len × 768]` in UNDERSTANDING.md; `resid_tensor [T, 768]` in TECHNICAL.md's snapshot table.

### 1.3 The L2 norm: length as energy

The **L2 norm** of a vector is its length, computed by the Pythagorean theorem generalised to any number of dimensions: square every entry, add them all up, take the square root.

```
‖x‖₂ = √(x₁² + x₂² + ... + xₙ²)
```

For a whole matrix, do the same over every entry in the grid. One number comes out: how "big" the tensor is overall. The acoustic analogy in this project treats that number as **energy**, and the analogy is faithful: in signal processing, the energy of a signal really is the sum of squared amplitudes.

Now the crucial experimental fact. Run the ATR loop without any rescaling and the tensor's norm grows explosively, reaching around 1.5 million by iteration 500. The numbers get so large that the model's nonlinear parts saturate and the outputs become meaningless. So after every pass, the tensor is rescaled to have exactly the same norm it had at iteration zero:

```
normalise(x) = x · (‖x₀‖₂ / ‖x‖₂)
```

Read that formula slowly: it multiplies the whole tensor by a single number, the ratio of the original length to the current length. Direction is untouched; only the overall size is reset. This is the "room's friction" in the Lucier analogy: the thing that stops the feedback from becoming a shriek. In dynamical-systems terms it confines the dynamics to a sphere of fixed radius, and on that sphere, stable convergence becomes possible.

**Where you've seen it:** step 3 of "How ATR Works" in the README; the Normalisation section of TECHNICAL.md.

### 1.4 Cosine similarity: comparing directions

If the norm measures a vector's length, **cosine similarity** measures the angle between two vectors, ignoring their lengths entirely. It ranges from 1 (pointing the same way) through 0 (perpendicular, nothing in common) to -1 (pointing opposite ways). The formula is the dot product of the two vectors divided by the product of their norms; the useful part is the interpretation:

- `cos_sim = 1.0`: the two states are the same direction, i.e. the same state up to scale.
- `cos_sim ≈ 0.99+`: nearly identical; in this project, the working definition of "converged".
- `cos_sim ≈ 0.7`: related but distinct (the `Divine` attractor sits at 0.73 from the `prolet` cluster: same neighbourhood of activation space, different room within it).

Cosine similarity is the project's ruler. Three separate uses, all the same mathematics:

1. **Convergence over time:** compare iteration *n* with iteration *n+1*. When `cos_sim(xₙ, xₙ₊₁)` stays above 0.999, the state has stopped moving: a fixed point. This is the convergence gate in `gated_resweep.py`.
2. **Convergence across prompts:** compare the final state of prompt A with the final state of prompt B. This is what the big **convergence matrix** image shows: 125 × 125 pairwise cosine similarities. The visible blocks are groups of prompts whose final states all point the same way: the attractor basins, seen as geometry.
3. **Position collapse:** compare the vectors at different token positions within one tensor. Early in the loop they differ (each token holds its own state); by around iteration 10 they become near-identical. The tensor goes spatially uniform, one note filling the whole room.

**Where you've seen it:** `cos_sim_mean`, `cos_sim_last`, `position_similarity` in TECHNICAL.md; every convergence matrix figure.

---

## Part 2: The Machine

### 2.1 Tokens and BPE: why the attractors are word fragments

Language models do not read words; they read **tokens**, produced by an algorithm called **byte-pair encoding (BPE)**. BPE starts from single characters and repeatedly merges the most frequent adjacent pairs found in training text, building a vocabulary (50,257 entries for GPT-2) of common chunks. Frequent words become single tokens; rarer words get split into fragments. "Proletariat" is rare enough that it splits, and `prolet` is one of its pieces.

This is why the attractors in this project are fragments like `prolet`, `Anarch`, `till`: the model's whole universe of possible outputs is its token vocabulary, so a fixed point of its dynamics, when decoded, must land on some token. The poetry of the result (a *fragment*, a *suggestion*) is downstream of this mechanical fact.

**Where you've seen it:** "the BPE subword `prolet`" in the README's Act I.

### 2.2 Embeddings: from tokens to vectors

The **embedding matrix**, written **W_E**, is a big lookup table: 50,257 rows (one per token), each row a 768-dimensional vector. To feed text into the model, each token is replaced by its row from W_E. These vectors are learned during training, and a famous property emerges: tokens used in similar contexts end up with vectors pointing in similar directions. Distance in embedding space approximates relatedness in usage.

That property is what licenses this project's phrase **"semantic neighbourhood in W_E."** To ask what `prolet` means to the model, take its embedding vector and find the other vocabulary tokens whose embedding vectors have the highest cosine similarity to it. For `prolet`, the nearest neighbours include *bourgeoisie*, *capitalists*, *revolutionaries*. The claim "four of five basins are semantically coherent" is precisely this test: the attractor tokens are not random; their neighbourhoods are thematically tight.

**Where you've seen it:** the "Semantic neighbourhood (W_E)" column in the README's basin table; `02_token_neighbourhood.ipynb`.

### 2.3 The residual stream: the model's running canvas

Here is the cleanest picture of a transformer's internals, and it is the one interpretability research actually uses.

Imagine each token position owns a 768-dimensional vector that flows upward through the network. This flowing vector is the **residual stream**. Each of GPT-2 Small's 12 layers contains two kinds of machinery: **attention heads** (which let each position read information from other positions and add what they find into the stream) and **MLPs** (small neural networks which transform each position's vector independently and add the result back). The key word is *add*. Layers do not replace the stream; they read from it and write increments into it. The residual stream is a shared canvas that every component paints onto in sequence.

So "extract the residual stream after layer 11" means: take the canvas as it stands after all twelve layers have painted. And "inject at layer 0, overwriting the token embeddings" means: instead of starting the canvas from W_E rows for the prompt's tokens, start it from our own tensor. That is the entire trick of ATR, implemented as two hooks (`blocks.11.hook_resid_post` to read, `blocks.0.hook_resid_pre` to write) via the TransformerLens library.

**Where you've seen it:** the Hook Mechanism section of TECHNICAL.md; UNDERSTANDING.md's four-stage pipeline.

### 2.4 Reading the state out: LayerNorm, logits, softmax, argmax

How do we hear what a 768-dimensional state "says"? The model has a built-in exit door, and ATR borrows it:

1. **LayerNorm** (`ln_final`): a standardisation step that rescales a vector to a common size and centring before use. The model always applies it before its final projection, so we must too, or we would be decoding through a door the model never uses.
2. **Unembedding** (**W_U**): a matrix that maps the 768-dimensional state onto all 50,257 vocabulary tokens at once, producing one raw score per token. These raw scores are called **logits**. W_U is the mirror of W_E: embedding goes vocabulary → vector, unembedding goes vector → vocabulary.
3. **Softmax**: turns raw scores into probabilities (all positive, summing to 1) by exponentiating and normalising. Big score gaps become big probability gaps.
4. **Argmax / top-k**: "argmax" is just the index of the largest entry (the single most likely token); "top-5" is the five largest. The dissolution traces in Act I (`ash → Canad → ... → prolet`) are the argmax at each snapshot.

One line of code does all of it: `logits = model.ln_final(resid) @ model.W_U + model.b_U`, then top-k.

Now you can state the study's sharpest finding precisely. The **tensor** (the 768-dimensional state) and the **readout** (the token the exit door reports) are different objects, and they can disagree about whether things have settled. The 34 `Divine` prompts have a tensor that never stops moving (cosine similarity between successive iterates never locks above the gate) while the readout reports the same token forever. A stable readout over a never-settling tensor. Decoding is a *projection*: it flattens 768 dimensions onto a vocabulary, and motion within the flattened-away directions is invisible to it.

**Where you've seen it:** the token-decoding formula in TECHNICAL.md; `readout_guardrails.ipynb`, which measures how much to trust the readout (via the **margin**, the logit gap between the top token and the runner-up, and the **entropy** of the distribution: low entropy means the probability mass is concentrated, i.e. the model is "sure").

---

## Part 3: The Dynamics

### 3.1 Iterated maps and fixed points

Take any function *f* and feed its output back as its input: x, f(x), f(f(x)), and so on. This is an **iterated map**, and it is the mathematical genre this entire project belongs to. A **fixed point** is a state the map sends to itself: f(x*) = x*. Once there, the system never leaves.

Try it on a calculator: start with any positive number and press the square-root key repeatedly. Whatever you start with, you drift to 1 and stay. The number 1 is a fixed point of the square-root map, and it is an **attracting** fixed point: nearby states move toward it.

ATR does exactly this where *f* is "one full forward pass through the transformer, then rescale to the original energy," and x is the `[T × 768]` tensor. Five hundred presses of the key.

### 3.2 Attractors, basins, and landscapes

An **attractor** is any state (or set of states) that trajectories converge into: fixed points are the simplest kind. A **basin of attraction** is the set of all starting points that end up at a given attractor. Picture rain falling on a mountain range: every drop lands somewhere and flows downhill to some valley floor. The valley floors are attractors; the watersheds (which valley a drop reaches depends on where it lands) are basins. The whole carved terrain is the **attractor landscape**.

Translate the project's headline findings into this vocabulary and they become one sentence each:

- **GPT-2 Small:** language-shaped starting points drain into five valleys, and four of the five have thematically meaningful names.
- **GPT-2 Medium:** one enormous valley (`D`); everything drains there within 10 iterations.
- **Pythia-160m:** one valley (`questioned`).
- **Pythia-410m:** no valleys reached within 1000 iterations; the water keeps moving. States that circulate without settling are **limit cycles** (repeating orbits) or wandering trajectories; the deep-convergence run was the check for this.
- **The null control:** noise-shaped starting points, in the *same* GPT-2 Small terrain, drain into 18 different, meaningless valleys. So the five semantic valleys are not properties of the terrain alone; they are properties of the terrain *as visited from the region where language lives*. That relocation of the claim is the single most important correction in the project.

A **regime**, as used in the README ("the language-driven regime"), means: which part of the landscape the dynamics explore, as determined by where you start. Same weights, different starting region, different effective landscape.

### 3.3 Power iteration: the linear ancestor

There is a special case where iterated maps are completely understood: when *f* is linear (a matrix multiply). Repeatedly applying a matrix to a vector and rescaling each time is the classic algorithm called **power iteration**, and it provably converges to the matrix's **dominant eigenvector**.

An **eigenvector** of a matrix is a direction the matrix does not rotate: it only stretches it, by a factor called the **eigenvalue**. Every application of the matrix stretches each eigenvector direction by its own eigenvalue, so after many applications, the direction with the largest eigenvalue dominates and everything else fades in comparison. The rescaling step (divide by the norm each time) is exactly ATR's normalisation. Power iteration is how early Google PageRank was computed; it is old, robust mathematics.

ATR is this algorithm with the matrix replaced by the full transformer forward pass, which is **nonlinear**: it contains softmax attention, GeLU activations, and LayerNorm, none of which are matrix multiplies. Nonlinear maps have no eigenvector guarantee, and richer possible behaviour: multiple coexisting attractors (GPT-2 Small's five), single global collapse (Medium's `D`), or refusal to settle (Pythia-410m). That variety across models is not a bug in the method; it is the finding. The phrase in the README, "a nonlinear analogue of power iteration," is precisely this relationship: same procedure, weaker guarantees, richer outcomes.

One caution to carry: because the map is nonlinear, do not over-interpret any single attractor as "the dominant eigenvector of the model." The spectral comparison (`spectral_resonance.ipynb`, scaffolded but not run) is the project's designed test of how far the linear intuition transfers to the per-head weight matrices.

### 3.4 Sensitivity, determinism, and why runs still agree

The forward pass involves no sampling: same input, same output, so the loop is deterministic in principle. In practice, floating-point arithmetic on a GPU is not perfectly associative (adding the same numbers in a different order gives microscopically different results), and iterated nonlinear maps can amplify microscopic differences. This is why the repeatability section of TECHNICAL.md reports that intermediate *pathways* vary between runs while the terminal *attractors* agree: chaotic sensitivity along the way, but the valleys are deep enough that both runs end in the same ones. Amplification of tiny differences along a trajectory, with agreement about the destination, is a signature pair familiar from dynamical systems generally.

---

## Part 4: The Instruments

### 4.1 PCA: how a 768-dimensional journey fits on a screen

The trajectory figures (the 3D spiral plots) show paths through 768-dimensional space, drawn in 3. The tool is **principal component analysis (PCA)**. Given a cloud of high-dimensional points, PCA finds the directions along which the cloud varies most: the first principal component is the single direction of greatest spread, the second is the most-spread direction perpendicular to the first, and so on. Keep the top 3 and project every point onto them, and you get the best possible 3-dimensional shadow of the cloud, in the precise sense of preserving the most variance.

Two reading rules for any PCA plot. First, axes have no intrinsic meaning: "PC1" is not a nameable quantity, just "the direction of most variation in this particular data." Second, distances are underestimates: points far apart in the shadow are truly far apart, but points close in the shadow might be separated along discarded dimensions. The dissolution spirals are honest about shape, not about absolute distance.

### 4.2 The convergence matrix: reading the block structure

The hero image of this repository is a 125 × 125 grid where cell (i, j) is the cosine similarity between the final tensors of prompt i and prompt j, with prompts ordered so that same-basin prompts sit adjacent. Bright square blocks on the diagonal are groups of prompts that all converged to the same direction: one block per basin, block size proportional to basin share. Off-block brightness shows between-basin relatedness (the `prolet`/`Anarch` blocks are geometrically closer to each other than either is to `Divine`). A single all-bright matrix (GPT-2 Medium) is total collapse; a structureless speckle (Pythia-410m) is non-consolidation. Once you can read this one figure, the entire cross-model comparison in Act II is legible at a glance.

### 4.3 Controls, gates, and the shape of the validation

Three design moves in this project are standard experimental machinery, worth naming so you recognise them elsewhere:

- **The null model** (`03_random_baseline.ipynb`): run the identical procedure on input where the hypothesised cause (language) is absent. Whatever survives is attributable to the procedure, not the cause. Here, noise still converged, but elsewhere, which relocated the claim.
- **The convergence gate** (`gated_resweep.py`): classify each prompt's basin only once its dynamics have provably locked (`cos_sim_mean > 0.999` sustained across three consecutive checks), rather than at an arbitrary iteration count. This removes "you just stopped too early" as an objection, and it is what exposed the `Divine` dissociation.
- **The reproducibility gate** (`00_reproducibility_gate.ipynb`): before believing a surprising result, check that the same machine produces it twice.

---

## Part 5: Pocket Glossary

| Term | One-line meaning | Where it lives here |
|:---|:---|:---|
| `d_model` | Width of the model's internal vectors (768 for GPT-2 Small) | everywhere |
| Tensor | Grid of numbers with any number of axes; here, the `[T × 768]` state | the thing fed back |
| L2 norm | Length of a vector/tensor; "energy" | the normalisation step |
| Cosine similarity | Angle-based sameness of direction, 1 = identical | convergence metric, matrices |
| BPE token | Vocabulary chunk, possibly a word fragment | `prolet`, `Anarch` |
| W_E / W_U | Embedding (token → vector) / unembedding (vector → token scores) | semantic neighbourhoods / readout |
| Residual stream | The running per-position vector every layer adds into | what ATR extracts and overwrites |
| Logits | Raw pre-probability scores over the vocabulary | readout step 2 |
| Softmax | Scores → probabilities | readout step 3 |
| Argmax / top-k | Biggest entry / biggest k entries | the decoded traces |
| Margin, entropy | Readout-confidence measures (gap to runner-up; concentration) | `readout_guardrails.ipynb` |
| Iterated map | Feeding a function its own output | the whole method |
| Fixed point | State the map sends to itself | a converged tensor |
| Attractor / basin | Destination / set of starts that reach it | the five basins |
| Limit cycle | Repeating orbit that never settles to a point | Pythia-410m candidate behaviour |
| Regime | Which part of the landscape the dynamics explore, set by the input | "language-driven regime" |
| Power iteration | Linear ancestor of ATR; converges to dominant eigenvector | the analogy, and its limits |
| Eigenvector / eigenvalue | Direction a matrix only stretches / the stretch factor | spectral scaffold (H4) |
| PCA | Best low-dimensional shadow of high-dimensional data | trajectory plots |
| Null model | Same procedure, cause removed | the noise baseline |

---

## A closing orientation

You now hold every concept this repository uses, and one honest way to check is to re-read the "Findings, Briefly" section of the README and notice that each bullet has become a sentence about specific mathematical objects: basins are attracting fixed points of a normalised iterated map; their coherence is a cosine-neighbourhood property of W_E; the refutation is a failed generalisation across four instances of that map; the null control is a regime relocation; the `Divine` anomaly is a disagreement between a projection and the state it projects.

The open question the project ends on (*why GPT-2 Small alone resolves language into semantic basins*) is a question about what, in one particular map's weights, carves few deep valleys under language-shaped input where sibling maps carve one or none. The J-space companion document ([JSPACE_PRIMER.md](JSPACE_PRIMER.md)) picks up the newest tool that might help ask it.



========================================================================
# SOURCE: Technical specification
# (repo path: docs/TECHNICAL.md)
========================================================================

# Technical Specification: Iterative Activation Re-injection

## Method

### Overview

This experiment implements iterative re-injection of the full residual stream tensor through the forward pass of a transformer language model (reference architecture: GPT-2 Small — 124M parameters, 12 layers, 12 heads, d_model=768). The residual stream at the final layer's `hook_resid_post` is extracted, L2-normalised, and re-injected at `blocks.0.hook_resid_pre` via a TransformerLens forward hook, overwriting the token embeddings. This is repeated for *N* iterations to map the fixed-point attractor landscape of the iterated forward map **under the chosen initial-condition regime** — the null-model control shows the landscape is regime-dependent, not a universal property of the weights (see [FINDINGS.md](FINDINGS.md), F4).

The same protocol runs cross-model via the shared engine (`atr_engine.py`): GPT-2 Medium (24 layers, d_model=1024), Pythia-160m (12 layers), Pythia-410m (24 layers) — extraction always at the final layer's `resid_post`, injection at layer 0. A convergence-gated variant (`run_atr_gated`) classifies terminal basins at lock-in (`cos_sim_mean > 0.999` sustained over three consecutive checks) rather than at a fixed iteration horizon.

The process is a **nonlinear analogue of power iteration**: where classical power iteration converges to the dominant eigenvector of a linear operator, this procedure converges to fixed points of the full transformer forward map *f*: ℝ^(seq×d) → ℝ^(seq×d), which includes LayerNorm, softmax attention (with dynamically recomputed QKV), GeLU MLP activations, and residual connections.

### Formal Description

Let *f* denote the transformer forward pass from Layer 0 through Layer 11:

```
f: ℝ^(T×768) → ℝ^(T×768)
```

where *T* is the sequence length. The iteration is:

```
x₀ = f(embed(prompt))                     # Initial forward pass
xₙ₊₁ = f(normalise(xₙ))                  # Re-inject normalised output
normalise(x) = x · (‖x₀‖₂ / ‖x‖₂)        # Global L2 rescaling
```

Convergence is assessed via cosine similarity between successive iterates:

```
cos_sim(xₙ, xₙ₊₁) → 1.0   as   n → ∞
```

### Hook Mechanism

Re-injection uses TransformerLens hooks to intercept and overwrite the residual stream:

```python
hook_read  = "blocks.11.hook_resid_post"   # Extract: output of final layer
hook_write = "blocks.0.hook_resid_pre"     # Inject: input to first layer

def injection_hook(resid, hook, tensor=inject_tensor):
    resid[0, :, :] = tensor                # Overwrite full [T, 768] tensor
    return resid
```

The prompt string is still passed to `model.run_with_cache()` on each iteration (required by TransformerLens to construct the computation graph), but the hook overwrites the embedding output before Layer 0 processes it. The prompt tokens serve only as scaffolding.

### Normalisation

Without normalisation, the tensor norm grows exponentially (~1.5M by iteration 500), saturating nonlinearities and producing meaningless token predictions. L2 normalisation rescales the full `[T, d_model]` tensor to maintain the energy of the initial forward pass:

```
‖xₙ‖₂ = ‖x₀‖₂   ∀ n
```

This makes the iterated map energy-conservative, bounding the dynamics within a fixed-radius manifold in ℝ^(T×768). Alternative normalisation strategies (per-position, per-dimension, LayerNorm-style) remain unexplored and may yield different attractor geometries.

### Snapshot Schedule

Snapshots are recorded at a logarithmic schedule to capture both early-phase dynamics and deep convergence without redundant mid-range computation:

```
schedule = [0, 2, 3, 5, 10, 20, 50, 100, 250, 500]
```

At each snapshot, the following are recorded:

| Metric | Tensor Shape | Description |
|:---|:---|:---|
| `resid_tensor` | `[T, 768]` | Full residual stream |
| `last_vector` | `[768]` | Residual at final token position |
| `mean_vector` | `[768]` | Mean-pooled residual across positions |
| `top_tokens` | top-5 | Decoded via `ln_final → W_U → softmax → topk` |
| `all_position_tokens` | `[T]` | Per-position top-1 decoded token |
| `cos_sim_last` | scalar | Cosine similarity to previous iterate (last position) |
| `cos_sim_mean` | scalar | Cosine similarity to previous iterate (mean-pooled) |
| `position_similarity` | scalar | Mean pairwise cosine similarity across positions |
| `tensor_norm` | scalar | L2 norm of full tensor |

Token decoding applies the final LayerNorm before unembedding:

```python
logits = model.ln_final(resid) @ model.W_U + model.b_U
```

### Observed Dynamics

**Position collapse**: By iteration ~10, all *T* positions converge to near-identical vectors (position_similarity → 1.0). The model's internal state becomes spatially uniform.

**Token convergence**: By iteration ~50–100, decoded tokens stabilise at a fixed point. Four of five initial prompts converge to `prolet` (BPE subword of "proletariat"); the fifth (`"The cat sat on the mat..."`) converges to `Divine`.

**Cross-prompt invariance**: Final-state cosine similarity between the four `prolet`-converging prompts is 0.999–1.000. The `Divine` outlier sits at 0.73 from the `prolet` cluster, indicating a distinct but geometrically related basin.

## Architecture

```
Model:           GPT-2 Small (gpt2, HuggingFace)
Parameters:      124M
Layers:          12 (indexed 0–11)
Heads:           12 per layer (144 total)
d_model:         768
d_head:          64
Vocab:           50,257 (BPE)
Training data:   WebText (Reddit-curated outbound links, ~40GB, circa 2018–2019)
Framework:       TransformerLens (Nanda & Bloom, 2022)
```

## Relationship to Existing Work

| Technique | Similarity | Key Difference |
|:---|:---|:---|
| Power iteration | Iterative operator application → dominant eigenvector | Our operator is nonlinear (full transformer stack) |
| Activation engineering (Turner et al., 2023) | Operates on residual stream | Single-pass steering, not iterated to convergence |
| Model collapse (Shumailov et al., 2023) | Iterative self-feeding | Operates at dataset level via text decoding, not at activation level |
| RNN fixed-point analysis | Maps attractor dynamics of recurrent systems | Transformers are feedforward; we impose recurrence via re-injection |
| Singular value decomposition of W_OV | Identifies dominant directions of weight matrices | Static analysis; our method probes the *nonlinear* composite operator |

A direct empirical comparison between the last two rows above — the per-head resonant state actually observed under iterative re-injection versus the dominant singular vector predicted by static SVD of `W_OV` — is scaffolded in `experiments/gpt2_small/spectral_resonance.ipynb`. It has not been run; see [FINDINGS.md](FINDINGS.md) (H4) and its Caveats section.

## Repeatability

Terminal attractors (`prolet` × 4, `Divine` × 1) are stable across N=2 same-machine runs. Intermediate dissolution pathways show sensitivity to floating-point non-determinism (expected for iterative nonlinear maps), but converge to identical fixed points. The convergence-gated re-sweep additionally confirms that 91/125 prompts reach a hard fixed point (`cos_sim_mean > 0.999`) and that basin labels assigned at lock-in are stable to 1000 iterations. Full determinism would require CPU execution with fixed seeds.

Cross-hardware replication has now been attempted once, and passed (2026-07-19): the same code, run on a different machine class (a fresh cloud container, CPU) with `gpt2` weights obtained from a legacy Hugging Face S3 mirror and loaded offline, reproduced the five-prompt piece exactly: identical terminal attractors (`prolet` × 4, `Divine` × 1) and identical intermediate dissolution waypoints (`Ag` at iteration 10, `Rousse` at iteration 50, `capit` en route), three times on that container. See [FINDINGS.md](FINDINGS.md) (F6) and `experiments/gpt2_small/output_confidence/confidence_report.md` (Result 0). The scope of this claim is deliberately narrow: it is replication of the same code on new hardware, not independent re-implementation; the weights came from a mirror rather than the canonical distribution (identical attractors and waypoints argue they are the same checkpoint); and one container is one data point. Independent re-implementation by another investigator has still not been attempted; "reproducibility" in the strict sense remains pending.

## Dependencies

Install from [`requirements.txt`](../requirements.txt) at the repository root.



========================================================================
# SOURCE: Scaling artefact analysis
# (repo path: docs/SCALING_ARTEFACT_ANALYSIS.md)
========================================================================

# ATR Scaling Artefact Analysis

## Context

Activation Tensor Resonance (ATR) takes a language model's output activations — the raw internal state after processing a prompt — and feeds them back in as input, repeating the cycle hundreds of times. Like Alvin Lucier recording his voice into a room until only the room's resonant frequencies remain, ATR dissolves the original prompt until only the model's dominant internal modes are left. In GPT-2 Small, this revealed five semantic attractor basins (`prolet`, `Divine`, `Anarch`, `till`, `solidarity`) that map to the political and theological centre of mass of its Reddit 2018 training data.

The question that follows naturally: what happens when you do this to larger models?

## The Question

When ATR is run on larger models, the attractor landscape changes dramatically. GPT-2 Small produces five clear semantic basins. Pythia-410m produces scattered punctuation and connectives with no convergence even at 250 iterations. Is this a real property of the model family, or is the experimental setup introducing avoidable distortion?

## Guiding Principle (Front and Centre)

If token labels flicker but `cos_sim_mean` is high and readout confidence is low, treat the behaviour as **readout ambiguity first**, not attractor instability.

## The Four Models — What We See

| Model | Params | Training Data | What Happens |
|---|---|---|---|
| GPT-2 Small | 124M | Reddit 2018 | 5 semantic basins (`prolet`, `Divine`, `Anarch`, `till`, `solidarity`). Clean convergence by ~50 iterations. |
| Pythia-160m | 160M | The Pile | Near-total collapse to `questioned` (94%). Converges by iteration 2-3. |
| GPT-2 Medium | 345M | Reddit 2018 | Total collapse to `D` (100%). Converges by iteration 5-10. |
| Pythia-410m | 410M | The Pile | 40+ fragments (punctuation, connectives). No convergence at 250 iterations. |

## 1) Methodological Artefacts (Apparatus Faults)

This section includes only candidate issues that could undermine the validity of interpretation by introducing distortion from the method itself.

### 1.1 Normalisation (master fader) — RULED OUT AS ARTEFACT

The per-iteration L2 rescale multiplies the entire tensor by one scalar (same ratio across all positions and dimensions).

**What it does operationally:** Prevents numeric blow-up between iterations. Residual additions compound magnitude across the stack; without rescale, norms reach ~1.5 million by iteration 500 and the run becomes meaningless.

**Why it was suspected:** A single scalar on a wider model (e.g. 1024-d vs 768-d) was briefly framed as possibly “reviving” weak dimensions by restoring total energy. That story fails: the scalar preserves the mix exactly — if one dimension is 500× another before rescale, it still is after.

**Why it is not a distortion source:** Layer 0 applies LayerNorm first; LayerNorm output is invariant to global scale. The forward pass therefore does not distinguish pre-rescale vs post-rescale tensors.

**Caveat on alternatives:** Per-dimension or max-dimension rescales are a different intervention — they can distort the relative geometry LayerNorm then sees. They are not equivalent to the current global L2 step.

**Definitive position:** Normalisation is numerically essential but computationally cosmetic for the forward map. It is not the source of the Pythia-410m fragmentation pattern.

### 1.2 Readout (unembedding) — OPEN ARTEFACT CANDIDATE

ATR interprets internal state by projecting the residual (after `ln_final` in the usual readout path) to token logits via the unembedding matrix.

**Risk:** A state that is stable in the residual stream may still sit where many vocabulary directions score similarly — so argmax (or dominant token) flickers while the tensor has effectively stopped moving.

**Contrast across models:** Tighter clustering in unembedding space (often discussed for some GPT-2 regimes) yields cleaner dominant tokens; flatter or more evenly spaced geometry yields “between stations” behaviour.

**Observable signature:** `cos_sim_mean` → 1 (or a tight plateau) while decoded tokens keep jumping — dynamics converged, vocabulary projection ambiguous.

**Note:** `cos_sim_mean` is computed on the activation tensor between iterations; it does not pass through token readout. It is the clean separator for this artefact class.

**Definitive position:** Readout remains the primary live methodological artefact candidate.

### 1.3 Readout Interpretation Guardrails (Amendment)

This amendment adds a strict interpretation protocol so readout noise does not misdirect conclusions.

**A. BPE/subword jaggedness — what it changes**

- **Mechanism:** Internal state moves continuously, but token output is discrete. Small vector moves near a decision boundary can flip top-1 token abruptly.
- **Why BPE amplifies this:** Fragments such as `prolet`, `capit`, punctuation variants, and whitespace-prefixed tokens can alternate with little underlying tensor movement.
- **Qualitative impact:** Apparent "semantic turbulence" in token traces can be visual, not dynamical.
- **Quantitative impact:** Token-switch count can be high while `cos_sim_mean` remains high/plateauing.
- **Weighting:** Interpretation risk **Medium-High**; code defect risk **Low**.

**B. Missing readout confidence metrics — why this matters**

- **Current gap:** Top token is logged, but confidence is not.
- **Consequence:** Near-tie flicker and genuine instability are conflated.
- **Required additions:** Top-1 vs top-2 logit margin, entropy (full or top-k), and optional top-k overlap across iterations.
- **Weighting:** Interpretation risk **High**; implementation effort **Low**; priority **P1**.

**C. Token rendering artefacts (`decode`) — precision caveat**

- **Current gap:** String decode can hide token-level distinctions (especially whitespace/special-token forms).
- **Required additions:** Log raw token IDs alongside rendered strings.
- **Weighting:** Interpretation risk **Low-Medium**; implementation effort **Very Low**; priority **P2**.

## 2) Intrinsic Model Variables (Under Investigation)

This section includes factors that are part of the model/system itself. 

### 2.1 Forward-pass depth

**Two clocks:**

- **Within one iteration — the forward pass:** The full native stack (12 layers for GPT-2 Small, 24 for Pythia-410m). This is ordinary inference geometry.

- **Between iterations — the Lucier loop:** Extract final-layer residual, L2-rescale for stability, re-inject at layer 0. That closure is the experiment.

After iteration 0, layer 0 no longer sees a fresh token-embedding row from the lookup table; it sees the previous end-of-stack residual (same `d_model` space the stack already uses). The accumulated shift across depth is what ATR is meant to iterate.

**Observable pattern:** Mixed behaviour across prompts — some tracks converge quickly, others oscillate or fragment — is consistent with depth-dependent dynamics rather than a single global bug.

A 24-layer pass is native for Pythia-410m; depth belongs in explanatory analysis, not under “methodological artefacts.”

### 2.2 Training corpus and token geometry

Reddit-trained GPT-2 variants and The-Pile-trained Pythia variants are shaped by different data distributions and potentially different representational topology in unembedding space.

### 2.3 Width and parameterisation regime

Changes in hidden size, head layout, and parameter count alter the geometry of the learned function and can shift attractor basin structure.

**Definitive position:** These variables belong in explanatory analysis, not in artefact diagnosis.

## 3) Controls and Attribution Tests

This section defines the tests that separate apparatus effects from intrinsic model effects.

1. **Cross-model `cos_sim_mean` chart (single view).**  
   If Pythia-410m remains below convergence while others saturate, non-convergence is internal-dynamics evidence. If it saturates while tokens flicker, readout is implicated.

2. **Same-model depth control on Pythia-410m (0-11 vs 0-23).**  
   Hold weights, tokenizer, and corpus constant; vary layer span only. If convergence behaviour changes materially, depth-dependent dynamics are causal.

3. **Long-horizon run (extend to 1000 iterations).**  
   Distinguish "not yet converged" from "structurally fragmented attractor landscape."

Each test changes one variable at a time and is implementable with minimal ATR engine changes.

### 3.1 Amended Experimental Versions (Readout-Focused)

**ATR-R1 (Confidence-Aware Readout)**
- Add per-snapshot: top-1 token ID/string, top-2 token ID/string, logit margin, entropy.
- **Follow-on use:** Re-label apparent fragmentation as either "high-confidence divergence" or "low-confidence boundary flicker".

**ATR-R2 (ID-First Trace)**
- Store token IDs as canonical output; keep decoded strings as display-only.
- **Follow-on use:** Build stable transition matrices and exact basin membership counts independent of rendering quirks.

**ATR-R3 (Tensor-Readout Concordance Audit)**
- For each run, classify snapshots into:  
  (i) high `cos_sim_mean` + low margin (readout ambiguity),  
  (ii) high `cos_sim_mean` + high margin (stable attractor label),  
  (iii) low `cos_sim_mean` (true ongoing dynamics).
- **Follow-on use:** Report convergence with confidence bands, not token labels alone.

## Current Judgement

- **Ruled out artefact:** Normalisation.  
- **Live artefact candidate:** Readout projection to tokens.  
- **Intrinsic explanatory axes:** Depth, corpus, width, token geometry.

## The Bigger Picture

The four models do not share one uniform failure mode; landscapes differ with corpus and architecture. Prompt-level heterogeneity (e.g. some Pythia-410m runs converging while others oscillate) fits intrinsic geometry and depth-dependent iteration maps, not a single broken knob in the apparatus.

Remaining work stays on two tracks: (a) close out readout as an artefact using tensor-level metrics vs token traces, and (b) attribute basin structure to depth, data, width, and unembedding geometry with controlled comparisons.


---

## Closing Judgement (2026-07-10 — controls executed)

The attribution tests proposed above were run at series close ([FINDINGS.md](FINDINGS.md), [RESULTS_SUMMARY.md](../experiments/RESULTS_SUMMARY.md)):

- **Test 1 (cross-model cos_sim chart): executed.** GPT-2 Medium and Pythia-160m saturate to 1.0000 by iteration 10 — their single-token collapses are real tensor attractors. Pythia-410m plateaus at ~0.85 through 250 iterations — non-convergence is internal dynamics, not readout.
- **Test 3 (long horizon): executed.** Pythia-410m at 1000 iterations (8-prompt subset): still fragmented, cross-prompt similarity 0.21. Structural, not under-iterated.
- **ATR-R1/R3 (confidence-aware readout): implemented and demonstrated** (single-prompt audit; margin rises and entropy falls as trajectories settle). The sharpest dissociation found: GPT-2 Small's `Divine` basin — readout constant while the tensor never passes the convergence gate.
- **Test 2 (depth control, layers 0–11 vs 0–23): still not run.** The cleanest remaining attribution test.

**Final position:** the guiding principle at the top of this document was applied and the answer landed on the intrinsic side — readout ambiguity is real but secondary; normalisation is inert; the cross-model landscape differences are properties of the models. The one place the readout-first principle earns its keep permanently is `Divine`, where dynamics and decoding genuinely come apart.



========================================================================
# SOURCE: ATR method comparison (interpretability landscape)
# (repo path: docs/ATR_METHOD_COMPARISON.md)
========================================================================

# ATR in Context: Method Comparison & Scaling Programme

**Date:** 2026-03-20 · **Revised at series close:** 2026-07-10
**Purpose:** Orientate ATR within the mechanistic interpretability landscape. Inform resource planning.

> **Revision note (2026-07-10):** the cross-model and null-model experiments this
> document originally proposed have now been run ([FINDINGS.md](FINDINGS.md)). The
> fingerprint hypothesis — basin profiles as training-data bias readable from any
> model — was **refuted**: GPT-2 Medium (same corpus as Small) produces no semantic
> basins, and random-noise initial states converge to a different attractor set
> entirely. Tables below have been corrected; rows recording the original March
> framing are marked where the framing did not survive.

---

## 1. Comparative Method Matrix

### 1a. What Each Method Reveals

| Method | What It Probes | Resolution | Output |
|---|---|---|---|
| **Logit Lens / Tuned Lens** | What the residual stream "predicts" at each layer | Per-layer, per-token | Token probability distributions per layer |
| **Activation Patching** | Causal contribution of specific components to outputs | Per-head, per-layer, per-position | Causal effect scores |
| **Linear Probes** | Whether a representation linearly encodes a concept | Per-layer | Probe accuracy (% classification) |
| **Sparse Autoencoders (SAEs)** | Decompose activations into interpretable features | Per-feature (thousands per layer) | Feature dictionaries + activation patterns |
| **Ablation Studies** | What breaks when a component is removed | Per-head, per-layer | Performance delta |
| **Representation Engineering** | Steer model behaviour via activation vectors | Per-layer, per-concept | Control vectors + behavioural change |
| **ATR (Ours)** | Attractor landscape of the iterated forward map under a chosen input regime | Per-model (global) | Basin tokens, convergence trajectories, regime profile |

### 1b. Resource Requirements

| Method | Compute | VRAM | Time per Run | Training Required? | Training Data Access? |
|---|---|---|---|---|---|
| **Logit Lens** | 1 forward pass | Model size | Seconds | No | No |
| **Activation Patching** | O(components × prompts) forward passes | Model size | Minutes–hours | No | No |
| **Linear Probes** | Forward passes + probe training | Model + probe | Hours | Yes (probe) | Yes (labelled data) |
| **SAEs** | Training run (millions of tokens) | Model + SAE (2–4×) | Days–weeks | Yes (SAE training) | Yes (activation dataset) |
| **Ablation** | O(components) forward passes | Model size | Minutes–hours | No | No |
| **Rep. Engineering** | Forward passes + PCA/mean diff | Model size | Minutes | No | Yes (contrast pairs) |
| **ATR** | N forward passes per prompt (N ≈ 50–200) | Model size | **Seconds–minutes** | **No** | **No** |

### 1c. Efficiency Analysis

| Method | Setup Complexity | Marginal Cost per New Model | Automation Potential |
|---|---|---|---|
| **Logit Lens** | Low (standard) | Very low | High |
| **Activation Patching** | Medium (requires prompt design) | Medium | Medium |
| **Linear Probes** | High (requires labelled data) | High (retrain probes) | Low |
| **SAEs** | Very high (training pipeline) | Very high (retrain SAE) | Low |
| **Ablation** | Medium | Medium | Medium |
| **Rep. Engineering** | Medium (requires contrast pairs) | Medium | Medium |
| **ATR** | **Very low** (load model, iterate) | **Very low** | **Very high** |

### 1d. What Each Method Cannot Do

| Method | Key Limitation |
|---|---|
| **Logit Lens** | Only shows prediction, not mechanism. Inaccurate in early layers. |
| **Activation Patching** | Requires known input→output pairs. Cannot discover unknown structure. |
| **Linear Probes** | Only finds what you look for. Cannot discover unknown concepts. |
| **SAEs** | Expensive. Feature interpretation is manual. Scale uncertain beyond GPT-2. |
| **Ablation** | Destructive. Cannot distinguish redundant from essential components. |
| **Rep. Engineering** | Requires concept pairs. Doesn't reveal model's "native" organisation. |
| **ATR** | **Reveals stable states of iterated dynamics, not per-input computation — and those states are regime-dependent (language-driven and noise-driven starts converge differently; FINDINGS.md F4). Doesn't explain HOW the model processes input.** |

### 1e. Unique ATR Capabilities

| Capability | ATR | Nearest Alternative |
|---|---|---|
| Reveal training-data thematic structure without training-data access | **Refuted** — tested cross-model and failed (FINDINGS.md F3, F4) | Rep. Engineering (partial, needs contrast pairs) |
| No labelled data required | Yes | Logit Lens, Ablation, Patching |
| No training/fine-tuning step | Yes | Logit Lens, Ablation, Patching |
| Global model characterisation (not per-prompt) | Yes | SAEs (but vastly more expensive) |
| Cross-model comparison via basin profiles | **Demonstrated** — 4 models, qualitatively distinct regimes | None established |
| Bias auditing of proprietary models (API-only) | No (needs weights) | Behavioural testing |
| Seconds per run on consumer GPU | Yes | Logit Lens only |

---

## 2. ATR's Position in the Field

### What ATR Adds

ATR is **complementary** to existing methods, not a replacement. It answers a question no other method asks:

> **"What are the stable states of the model's iterated forward map, and how do they depend on the input regime?"**

Other methods tell you what the model does with a specific input. ATR tells you where the model's dynamics settle when the input is played back through it indefinitely. The original framing — "revealing the training data's thematic fingerprint" — did not survive the cross-model test (FINDINGS.md F3); what ATR demonstrably provides is a cheap, tensor-level regime comparison across models.

### Closest Parallels

- **Power iteration** in linear algebra (find dominant eigenvector by repeated multiplication) — ATR is the nonlinear analogue
- **Room impulse response** measurement in acoustics (inject impulse, observe system response) — ATR is this for neural networks
- **Lyapunov exponent analysis** in dynamical systems (characterise stability of trajectories) — ATR maps the attractor landscape

### State-of-the-Art Impact Potential

| Claim | Evidence Level | Impact if Validated |
|---|---|---|
| Attractor basins exist with the shape and dominance shares observed | Supported (GPT-2 Small, single-model) | Confirms iterated forward map produces a discrete attractor landscape |
| Basin tokens cluster semantically rather than by BPE substring | Supported (W_E neighbourhood test on GPT-2 Small; statistical validation pending) | The attractors carry meaning, not artefact |
| Basin profiles vary predictably by model / training data | **Tested: they vary, but not corpus-trackably** (same corpus → unrelated landscapes) | Partially — ATR distinguishes models, but not by training data |
| ATR-derived basins can stand in for training-data inspection in bias characterisation | **Refuted** (FINDINGS.md F3, F4) | — |
| Basin topology correlates with model capabilities | Speculative | Transformative if true, but years from testable |

---

## 3. Compute Programme

### 3a. Research Programme: Cross-Model ATR Landscape Mapping

> **Status (2026-07-10):** a local 2×2 slice of this programme (GPT-2 Small/Medium × Pythia-160m/410m, CPU) has been executed — results in [FINDINGS.md](FINDINGS.md). It answered the programme's central question early: landscapes are model-specific and do not track the corpus. The larger sweep remains open as characterisation work, with the fingerprint motivation retired.

**Objective:** Run ATR across a systematic sweep of open-weight language models to determine whether attractor basin profiles are model-specific, architecture-specific, or scale-dependent.

### 3b. Experimental Design

| Phase | Models | Count | VRAM per Model | Estimated GPU-Hours |
|---|---|---|---|---|
| **Phase 1: GPT-2 Family** | GPT-2 Small/Medium/Large/XL | 4 | 0.5–6 GB | ~2 hours |
| **Phase 2: Pythia Suite** | Pythia 70M → 12B (8 checkpoints × 143 training snapshots) | ~1,144 | 0.3–48 GB | ~50 hours |
| **Phase 3: Architecture Diversity** | OLMo-7B, Llama-3-8B, Mistral-7B, Gemma-2B/7B | 5 | 4–32 GB | ~10 hours |
| **Phase 4: Aligned vs Base** | Llama-3-8B vs Llama-3-8B-Instruct (same weights, different alignment) | 2 | 16 GB | ~4 hours |
| **Phase 5: Scale Sweep** | Llama family 1B → 70B (if resources allow) | 4–6 | 2–140 GB | ~100 hours |

**Total estimated:** ~170 GPU-hours (A100-class), expandable to ~500 with full scale sweep.

### 3c. Resource Requirements

| Resource | Minimum | Ideal |
|---|---|---|
| GPU | 1× A100 40GB | 4× A100 80GB |
| Storage | 500 GB (model weights + `.pt` outputs) | 2 TB |
| RAM | 64 GB | 128 GB |
| Duration | 1 week (batch jobs) | 3 days (parallel) |

### 3d. What the Programme Produces

| Deliverable | Format | Value |
|---|---|---|
| Basin profiles for 10+ models | JSON + visualisations | First cross-model attractor comparison |
| Scaling laws for basin count vs model size | Statistical analysis | Novel finding |
| Bias profiles for base vs aligned models | Comparative matrices | Safety-relevant |
| Pythia training dynamics | Basin evolution across checkpoints | Developmental biology of LLMs |
| Open-source ATR toolkit | Python package + notebooks | Community contribution |

### 3e. Why HPC

- **Low total compute** — 170–500 GPU-hours is modest by HPC standards
- **High ratio of output to compute** — each ATR run produces rich, interpretable data per unit of GPU time
- **Novel research territory** — cross-model attractor-landscape comparison has not been published; results either way (basins generalise predictably, basins generalise but unpredictably, or basins fail to generalise) are publishable findings
- **Methodological cleanness** — fully deterministic at the same-machine level, no training required, reproducible in principle
- **Conditional safety relevance** — *if* the central hypothesis (basin profiles reflect training-corpus thematic structure) generalises across models, ATR could offer a route to bias characterisation without training-data access. The cross-model programme is what would test this; the relevance is conditional on that test, not established by the GPT-2 Small results alone.

### 3f. Timeline

| Week | Activity |
|---|---|
| 1 | Phase 1 + 2 (GPT-2 family + Pythia subset) — validate ATR scales |
| 2 | Phase 3 + 4 (architecture diversity + alignment comparison) |
| 3 | Phase 5 (scale sweep, if resource permits) |
| 4 | Analysis, visualisation, draft findings |

---

## 4. Development Roadmap

### Immediate (Current Setup — Local GPU)
- [x] ATR on GPT-2 Small (done)
- [ ] T_mix_LLM measurement
- [ ] Basin-sorted convergence matrix
- [ ] Automated ATR pipeline (parameterisable model, prompt set, output directory)
- [ ] SVD/spectral-gap prediction of per-head resonant states vs. observed convergence (scaffolded in `spectral_resonance.ipynb`; not yet run)

### Short-Term (Pre-HPC)
- [ ] ATR on GPT-2 Medium/Large (validate scaling locally)
- [ ] Standardised output format (JSON schema for basin profiles)
- [ ] Comparison metrics (basin count, depth, distribution entropy)
- [ ] Prior art survey + position paper draft

### Medium-Term (HPC Programme)
- [ ] Phases 1–5 as described above
- [ ] Cross-model analysis pipeline
- [ ] Publication-ready visualisations

### Long-Term (Post-Publication)
- [ ] Open-source ATR toolkit
- [ ] Community replication
- [ ] Extension to multimodal models
- [ ] Formal connection to mixing time mathematics

---

*Document created 2026-03-20. Living document — update as programme develops.*



========================================================================
# SOURCE: Prior work
# (repo path: docs/PRIOR_WORK.md)
========================================================================

# Prior Work

This document places the ATR experiments (closed-loop reinjection of GPT-2 Small's final-layer residual stream into layer 0,
energy-renormalised, iterated to exhaustion) against the published record: what the nearest neighbouring work did and found, how it
relates to each ATR result, and what, on the evidence here, has no prior occupant. Citations that could not be checked against their
primary sources are marked with an asterisk and rest on secondary descriptions (title, venue, abstract-level claims); unmarked citations
were checked against primary material, meaning the work's full text or its primary artifacts were read directly. Independently of that
check, every work discussed in prose carries a source-class tag immediately
after its citation, one of [peer-reviewed], [preprint, unreviewed], [community post, unreviewed], or [primary data, read directly] (with
[published research report, not journal-reviewed] and [status unverified] for sources those four classes do not fit, and with the venue
named inside the tag where the classification rests on a venue the citation itself does not state); the tag records the source's formal
status as listed by its own venue or host, and formal status is no guarantee of correctness: peer-reviewed work can be wrong, and
replication is the stronger signal, noted where known. Terminology follows BELL_PRIMER.md: the
bell (the exact period-2 limit cycle), phases A and B (its two states), the pivot M (their midpoint), the flip axis d (their normalised
difference), L11.H8 (layer 11, head 8).

The ATR findings the entries refer to, by number:

1. Five semantic basins from language inputs, distinct non-semantic basins from noise; the count is regime-dependent.
2. States become position-uniform early in the loop (all token positions identical).
3. One prompt yields an exact period-2 limit cycle, missed by lag-1 convergence tests and even-interval sampling.
4. The cycle's flip is inverted by a single head, L11.H8, through its OV circuit (the head's fixed 768-to-768 output transform): multiplier
   -4.3 at the pivot M, +0.1 around the two-step loop. On ordinary text the same head is a copy promoter.
5. The flip axis d runs between the most-trained and never-trained regions of embedding space (cos to the glitch-cluster direction -0.596).
6. The motion is near-invisible to the unembedding and to lens instruments; it lives mostly outside the verbalizable subspace.
7. Cross-model contrast: GPT-2 Medium one basin, Pythia-160m one, Pythia-410m none under a lag-1 gate now considered suspect.

## The nearest neighbours

**Wang, Li, Yan, Cheng, Zhang, Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing (ACL
2025).** [peer-reviewed] https://aclanthology.org/2025.acl-long.624/ and https://arxiv.org/abs/2502.15208
The established nearest neighbour. Iterated paraphrasing through the model's text interface converges to stable periodic states,
prominently period-2 cycles, robust to temperature, alternating prompts, alternating models, two languages, and sentence and paragraph
scales (pp. 6-7, 14-15). Periodicity is measured with a dedicated lag-2 metric, the 2-periodicity degree tau, one minus the mean normalised
Levenshtein distance between each text and the text two steps back (p. 3), kept separate from the paper's perplexity-based convergence
measures (p. 5). The cycles are approximate and statistically maintained: tau runs 0.60 to 0.92 across models and never reaches 1 (p. 5),
and decoding is sampled (temperature 0.6, top-p 0.9, ten samples with the most probable kept, p. 3), so no state ever recurs exactly. Three
results extend the picture: any invertible task cycles (polishing, clarification, formality transfer, round-trip translation, tau 0.65 to
0.87, p. 6); conditioning each paraphrase on the previous two texts produces period-3 cycles, so the attractor period tracks the task's
memory depth rather than a model constant (Sec. 5.6, p. 8); and chains generated by one model keep falling in perplexity under another
model's measure, read by the authors as "a general statistical optimum that multiple LLMs gravitate toward" (p. 7), so the attractor states
are shared across models. The paper takes no internal measurement of any kind; the explanation it offers is behavioural, self-reinforcement
plus task invertibility (pp. 5-6). The differences bound the overlap: their loop runs over discrete text, their cycles are approximate, and
no mechanism is located inside the model. The bell is activation-level, exact to machine precision, and mechanistically attributed
(findings 3, 4): convergent phenomenology at a different interface.

**Trained-loop architectures.** Geiping et al., Huginn (arXiv 2502.05171) [preprint, unreviewed]: a 3.5B model trained to iterate a middle
block, with the embedded input re-injected at every recurrence and the latent state initialised at random; some tokens fall into stable
orbits, the closest trained-model analogue of finding 3, but the model is trained for the loop, the orbits are read as functional
organisation (pp. 12-13), and the dynamics are path independent: the same orbits and fixed points appear whatever the initialisation,
because the injected input pins the attractor (p. 13). ATR's loop, with no fresh input, is the opposite regime: the initial state is the
only thing that selects the attractor. Hao et al., Coconut (arXiv 2412.06769) [preprint, unreviewed] feeds the last hidden
state back as the next input embedding, on a pretrained GPT-2 base (Sec. 4): ATR's feedback edge on ATR's architecture family, built by
training, run for a few steps, no attractor map. Zhang et al., Soft Thinking (arXiv 2505.15778) [preprint, unreviewed] is the closest
inference-only relative (training-free self-feedback on a frozen model), mediated by the output distribution rather than the residual
stream, and it meets the loop's degenerative attractor empirically: continued soft-token feedback causes generation collapse, so the method
adds Cold Stop, an entropy-threshold early exit that ends latent reasoning before collapse (pp. 2, 5). Ouro (arXiv 2510.25741)\* [preprint,
unreviewed] and retrofitted recurrence (arXiv 2511.07384)\* [preprint, unreviewed] convert pretrained stacks into loops by training. The
looped-LM literature names its failure mode latent collapse, the hidden state falling into an input-independent fixed point, and engineers
against it (STARS, arXiv 2605.26733\* [preprint, unreviewed]; Solve the Loop, also circulating under the title Attractor Models for Language and Reasoning, arXiv
2605.12466\* [preprint, unreviewed]). Movahedi et al., Fixed-Point Reasoners (arXiv 2606.18206) [preprint, unreviewed] close the
family from the fixed-point side: a 7M-parameter looped reasoner trained from scratch for algorithmic tasks, not natural language (p. 15),
halts when consecutive iterates become sufficiently close (p. 3), observes that for some inputs the model descends into oscillatory
behaviour around a fixed point of the outer iteration map (p. 6), and ships FPOPT, a patience-based damping that eliminates the
oscillations while preserving the fixed points (Theorem 3, p. 7); no period is measured, no cycle is characterised, and no component-level
mechanism is given. What this literature suppresses or exploits, ATR studies descriptively in a model never trained for the loop.

**Blayney et al., A Mechanistic Analysis of Looped Reasoning Language Models (arXiv 2604.11791).** [preprint, unreviewed]
An unreviewed preprint comparing Huginn, Ouro, and retrofitted-recurrence Llama, and the closest methodological neighbour of finding 3.
Its first convergence diagnostic is the lag-1 successive-iterate difference norm on the looped residual stream (Fig. 3, pp. 4-5), and its
main-text taxonomy is binary, fixed point reached or not; the authors themselves flag one insufficiency of that gate, the drift case, where
Ouro's successive differences shrink while the state never reaches its late-iterate reference point, and add distance to an approximate
fixed point as a second diagnostic (Fig. 4, p. 5). An appendix supplies a third: Algorithm 1 (p. 17) classifies each token's limiting
behaviour as fixed point, orbit, slider, or unknown by FFT on the similarity-to-final-state series, with frequency bins reaching the
Nyquist rate of 0.5 cycles per iteration, which is exactly period 2. The detected orbits are rare (0.01 to 0.14 percent of Huginn tokens,
at most 0.08 percent for retrofitted Llama, up to 2.81 percent of examples under a long persona prompt; Tables 3-4, p. 17) and approximate,
with dominant frequencies 0.125 and 0.344 cycles per iteration (Fig. 13, p. 18); no period-2 orbit is reported anywhere, and the orbit and
slider labels are credited to Geiping et al. (p. 4). The classifier also checks its fixed-point class first, with tolerance 0.05 on cosine
similarity over 90 percent of late iterates (p. 17), so a 2-cycle whose two phases sit within cosine 0.95 of each other is filed as a fixed
point; the bell, with its small flip along one axis, plausibly falls in that class. Separately, Proposition 4.1 (p. 4) describes cyclic
fixed points: when a k-layer looped block reaches a block-level fixed point, each layer settles to its own point and one pass of the block
traces a closed cycle in latent space, period equal to the block depth by construction, observed frequently in practice (Fig. 5, p. 5);
that is the trivial cycle of a converged loop, indexed by layer, orthogonal to the bell's period 2 in the iterate index at a fixed layer.
Randomly initialised, untrained loops show the same cyclic fixed-point behaviour (Sec. 4.2, pp. 5-6), so the fixed-point tendency is
architectural rather than trained, which supports studying a model never trained for the loop. Against this record the lag-k correction
(`gate_lag` and `lag_scan` in `atr_engine.py`) is scoped precisely: no located work, Blayney's full text included, states that the lag-1
gate itself is period-2-blind as a matter of arithmetic (a 2-cycle registers as a constant nonzero lag-1 difference, or as a fixed point
under stride-2 sampling, never as a cycle), and the gate's live use retroactively questions ATR's own Pythia-410m null (finding 7).

**Lu, Yang, Lee, Li, Liu, Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer (arXiv 2507.02199).** [peer-reviewed, COLM 2025
workshop] https://arxiv.org/abs/2507.02199
With Blayney, the only other located analysis of looped-LM internals. Huginn's forward pass is unrolled into 2 prelude, 4r recurrent, and 2
coda blocks (68 blocks at r = 16, p. 3) and probed on arithmetic with the logit lens and a coda lens (decoding through Huginn's own output
blocks). The rank of the final predicted token shows large-magnitude periodic oscillations across the unrolled blocks (p. 4): under the
logit lens the rank spikes upward at block R4 in every cycle, and under the coda lens the pattern inverts, with the drop at R4 and
uninterpretable output almost everywhere else. The periodicity is indexed by position within the four-block recurrent unit, a within-pass,
architecture-period signal in a trained, input-injected loop, not an iterate-index cycle of a free-running state. The paper's sharpest
result is instrumental: the same hidden state decodes to unrelated words under one lens and to on-task numerals under the other, so lens
applicability must be assessed per layer (p. 4). That is finding 6's blind spot documented from the inside: cyclic structure in
looped-model internals is here visible only through lenses, and the lenses disagree block by block. No mechanism is offered, no exactness
is claimed, and recurrence scaling stays below 5 percent GSM8K accuracy without explicit chain-of-thought at every depth from 4 to 256
steps, against 24.87 with it (Table 1, p. 5).

**Deep equilibrium models.** Bai, Kolter, Koltun (NeurIPS 2019)\* [peer-reviewed] replaced explicit depth with root-finding for the fixed point of one
weight-tied block, with the founding admission that plain forward iteration often fails to converge; the repair toolchain exists because
undamped iteration stalls or oscillates (Anderson acceleration, arXiv 2410.19460\* [preprint, unreviewed]; Jacobian regularisation, arXiv 2106.14342\* [peer-reviewed, ICML 2021]; monotone
operator networks, arXiv 2006.08591\* [peer-reviewed, NeurIPS 2020]). Standard fixed-point numerics give the mechanism class: an eigenvalue of magnitude above 1 with
negative sign at a fixed point produces oscillation around it. Finding 4 is that signature in a pretrained transformer (a period-doubling
configuration: a near-fixed point whose one unstable direction yields a stable 2-cycle), localised to a named component, which no DEQ work
does.

**Marcus, Westervelt, Dynamics of Iterated-Map Neural Networks (Phys Rev A 1989).**\* [peer-reviewed]
https://neuron.eng.wayne.edu/tarek/MITbook/ref/refs.html
The classical anchor for finding 3: in discrete-time parallel-update neural networks, when a fixed point destabilises, the generic new
attractor is a period-2 oscillation. ATR adds the transformer instantiation and the single-head OV mechanism, which has no analogue in these
homogeneous models.

**Sussillo, Barak, Opening the Black Box (2013).** [peer-reviewed, Neural Computation 2013] https://direct.mit.edu/neco/article/25/3/626/7854
The methodological ancestor: find fixed points of a trained recurrent network, linearise around them, read the computation from the
attractor skeleton. Its modern instances include the line-attractor account of trained sentiment RNNs (Maheswaranathan et al., NeurIPS
2019, arXiv 1906.10720)\* [peer-reviewed], and Fernando and Guitchounts (below) place themselves in the same program. ATR is this program
transplanted to a transformer made recurrent by an external loop.

**Dong, Cordonnier, Loukas, Attention is Not All You Need (ICML 2021).** [peer-reviewed] https://arxiv.org/abs/2103.03404
Pure self-attention converges doubly exponentially to a rank-1, token-uniform state. The counteraction is asymmetric: skip connections are
the crucial term (an existence result via the length-0 path, Claim 3.1, p. 6), MLPs only slow the rate through their Lipschitz constant
(Corollary 3.2, p. 7), and LayerNorm plays no role in the analysis (Sec. 3.3, p. 7). Finding 2 is this token-uniformity bias expressed
under closed-loop iteration, where effective depth reaches hundreds of blocks; the collapse guarantee covers only skip-free, MLP-free
stacks within one forward pass, so position uniformity arising with skips and MLPs present sits outside the guaranteed regime rather than
contradicting it, and the existence of non-collapsing parameterisations is consistent with finding 1, since with MLPs and skips present the
collapse is not to one global point. The paper also contains the record's nearest in-paper precedent for ATR's procedure: Section 4.2 runs
a single transformer layer recurrently at inference on a 2D toy task, feeding its output back as its next input; without skips or MLP the
trajectories collapse to a point, and adding either stops or drastically slows the collapse (Fig. 3, pp. 9-10). Geshkovski, Letrouit,
Polyanskiy, Rigollet [status unverified] model tokens as interacting particles that cluster as depth grows, the final configuration set by
the input\*. The oversmoothing line anticipates finding 2 and supports finding 1; none of it describes an unplanned oscillation coexisting
with position-uniform states.

**Within-forward-pass attractor and orbit vocabulary.** Two frozen-model programs apply dynamical language to the layer-indexed trajectory
of a single forward pass; neither closes an output-to-input loop. Chytas and Singh, Concept Attractors in LLMs and their Applications
(arXiv 2601.11575) [peer-reviewed, ICLR 2026] model the layer stack as an iterated function system in which "layers act as contractive
mappings toward concept-specific Attractors" (p. 1): semantic attractor sets, one per concept, at concept-specific depths (in Llama 3.1 8B,
fictional worlds at layer 24, programming languages 19, natural languages 27, literature 18, p. 4), operationally estimated as the mean
hidden state of a concept's samples at its layer (p. 4) and used for steering, unlearning guardrails, and detoxification. The scope is
stated: collapse phenomena at specific intermediate layers of forward propagation (p. 2), prompt-conditioned, with cycles appearing only as
a citation of Wang et al.'s output-layer period-2 paraphrase cycles (p. 4). The arXiv copy read here is headed Preprint, under review; the
ICLR 2026 acceptance header is carried by the OpenReview copy. Fernando and Guitchounts, Transformer Dynamics: A Neuroscientific Approach
to Interpretability of LLMs (arXiv 2502.12131) [preprint, unreviewed], the dynamical-systems reference of the former, treat one forward
pass of Llama 3.1 8B as a trajectory: individual scalar residual-stream units, traced across the 64 effective sublayers, wind rotationally
in an activation-versus-layer-gradient phase plane, a mean of 10.74 rotations per unit against about zero in layer-order shuffles (Fig. 2,
pp. 4, 7), in consistently hedged language ("rotational dynamics characteristic of unstable periodic orbits", p. 6); the spirals grow
outward, no state recurs, and within-pass perturbation recovery at mid layers is called "akin to a pseudo-attractor" (pp. 7-8). Both
programs supply the vocabulary nearest findings 1 and 3 while measuring a different object: depth-indexed, non-autonomous trajectories
inside one pass, with no exactness and no component mechanism.

**Attention as associative memory.** Ramsauer et al. (ICLR 2021, arXiv 2008.02217) [peer-reviewed] identify attention with the update rule
of a continuous modern Hopfield network whose fixed points are retrieval states near well-separated stored patterns, metastable subset
averages (resembling the basins of finding 1), or a global average, a position-uniform state (finding 2) (pp. 4-5). Retrieval typically
happens in one update (Theorem 4, p. 4), and transformer attention is exactly one application of the rule (p. 5): the architecture never
iterates the map the theory analyses, and ATR's loop supplies that iteration externally. The storage theory assumes patterns of fixed norm
on a sphere (p. 4), a structural parallel of the energy renormalisation. The energy-descent guarantee (Theorems 1-2, pp. 3-4) holds only
for the idealised symmetric update; a full block with MLP, LayerNorm, and external renormalisation has no Lyapunov function, which is why a
limit cycle is possible at all, and the paper's 94 pages contain no occurrence of oscillation, cycle, or period. Energy Transformer (arXiv
2302.07253)\* [peer-reviewed, NeurIPS 2023] and Hyper-SET (arXiv 2502.11646)\* [preprint, unreviewed] build blocks that provably descend an
energy and so produce fixed points only; Hyper-SET, an unreviewed preprint, independently arrives at ATR's two mechanics (norm-constrained
states, repeated block) as design principles.

**The loop's preconditions.** Transformer Layers as Painters (AAAI 2025, arXiv 2407.09298)\* [peer-reviewed] ran frozen pretrained layers in altered orders
and loops, the nearest published practice of iterating frozen layers off-distribution; it did not renormalise, close the full
output-to-input loop, or map attractors. Heimersheim and Turner (LessWrong 2023)\* [community post, unreviewed] put per-layer residual norm
growth in GPT-2 class models at roughly 4.5 percent, an unreviewed community measurement; whatever its exact value, without the energy
renormalisation the loop diverges in norm, so the attractors are properties of the renormalised map.

## GPT-2 internals

**Elhage et al., A Mathematical Framework for Transformer Circuits (Anthropic 2021).**\* [published research report, not journal-reviewed]
https://transformer-circuits.pub/2021/framework/index.html
The residual stream is a shared communication channel; heads decompose into QK circuits (where to
attend) and OV circuits (what is written); copying is scored by positive real eigenvalues of the full OV circuit W_U W_OV W_E. This supplies
the formalism of finding 4: the -4.3 action on the flip axis d is a strong anti-copying eigenmode, and the head-level attribution is this
framework applied to closed-loop dynamics.

**Lens instruments and their failure modes.** The logit lens (nostalgebraist 2020)\* [community post, unreviewed] decodes intermediate
residual states through the final LayerNorm and the unembedding W_U; its structural failure mode is that it reads only components aligned
with W_U's strong directions. The tuned lens (Belrose et al. 2023, arXiv 2303.08112) [preprint, unreviewed] documents the logit lens's
brittleness and per-model variability while noting that it works reasonably well for GPT-2 (p. 2), so for ATR's model the documented
failure modes are bias and drift rather than wholesale breakdown; and the tuned lens itself is trained by distillation to match the final
logits (p. 2), so any lens in this family is by construction blind to directions that never reach the output. Finding 6 is a concrete
instance of exactly that blind spot: any lens defines a verbalizable subspace and misses its complement, where the closed-loop motion
lives, so lens-based accounts of the loop would wrongly report stasis. The claim rests on the subspace argument, not on lens brittleness in
GPT-2. Lu et al. (above) show the per-block version inside a looped model: the same hidden state decodes to unrelated words or to on-task
numerals depending on the lens.

**The head catalogue, and where L11.H8 is not.** Induction heads (Olsson et al. 2022) [published research report, not journal-reviewed]: prefix matching plus copying, the copying OV raising
the attended token's logit; GPT-2 Small induction heads 5.5, 5.8, 5.9, 6.9, paper\*.
The IOI circuit (indirect object identification; Wang et al. 2022, arXiv 2211.00593\*) [peer-reviewed, ICLR 2023]: 26 heads including negative
name movers 10.7 and 11.10, which write against the correct name; head list
https://raw.githubusercontent.com/ArthurConmy/Automatic-Circuit-Discovery/main/acdc/ioi/utils.py [primary data, read directly]. Copy
suppression (McDougall, Conmy, Rushing, McGrath, Nanda 2023, arXiv 2310.04625) [preprint, unreviewed]: L10.H7 detects the currently
predicted token and writes against its unembedding (the OV diagonal sits in the bottom 5 percent of its column for 98.86 percent of the
vocabulary, and on OpenWebText 78.24 percent of attended pairs put the source token among the ten most suppressed, pp. 4-5, 16-17); this is
the class the suppression tests (BELL_PRIMER Part 7) show L11.H8 opposes on ordinary text, where it raises the attended token's score at
91.4 percent of positions (the same dynamic test, sign reversed). The polarity contrast is exact: L10.H7's least-suppressed tokens are
function words (" of", " that", " the", " in", " to"; Appendix G, p. 17), the very tokens the L11.H8 card places at its negative pole. One
caveat transfers: copy suppression self-repairs, with L11.H10 partly backing up an ablated L10.H7 (Sec. 4.1, p. 7), so single-head
ablations in layers 10 and 11 can be masked by compensation. Successor, greater-than, and year heads (arXiv 2312.09230 [peer-reviewed,
ICLR 2024]; arXiv 2305.00586 [peer-reviewed, NeurIPS 2023])\* place documented number and date machinery in layers 5 to 9. L11.H8 appears
in none of these catalogues (in the copy-suppression paper it appears once, as an unremarkable mid-range bar in an appendix figure, p. 25);
the only documented layer 11 copy machinery (11.10) has the opposite sign to finding 4.

**Kissane, Krzyzanowski, Bloom, Conmy, Nanda 2024, Attention Output SAEs (arXiv 2406.17759).** [preprint, unreviewed] https://arxiv.org/abs/2406.17759, per-head cards
https://robertzk.github.io/gpt2-small-saes/ [primary data, read directly]
The only public per-head documentation of L11.H8, released by the paper itself: Appendix A (p. 15) links the card site as the official view
of the top ten features attributed to each of the 144 heads. SAEs (sparse autoencoders, learned dictionaries of directions) were trained
per layer on GPT-2 Small attention outputs, concatenated per head, and each feature attributed to heads by decoder-weight norm; the paper
calls the attribution a rough heuristic, warns of interpretability illusions and of missing behaviour spread across heads in superposition
(Appendix M.1, pp. 32-33), and reports the layer 11 SAE among the least interpretable in the sweep (63 percent of sampled live features
judged interpretable, Table 1, p. 5). The paper's prose never mentions head 11.8; its layer-11 summary is grammatical adjustment and bigram
completion (Appendix M.2, p. 33). A direct read of the L11.H8 card (the card itself labels the head "11.8") shows its top feature (1958)
puts positive logits almost entirely on glitch and undertrained tokens (" guiActiveUn", "ertodd",
"ThumbnailImage", "ActionCode", "externalToEVA", byte fragments) and negative logits on the most frequent function tokens (" the", ",", "
and", " in", " a", " to", " of"); the next four features promote numerals, dates, years, and round quantities, with glitch tokens
("rawdownload", "oreAndOnline", "embedreportprint", " TheNitrome") at their negative ends. This is exactly the polarity of finding 5: the
head's output features already span a frequent-token versus glitch-token axis, the axis the flip runs along. The pattern is head-specific by
control: the L11.H10 card shows ordinary verb and event-structure features and no glitch axis. The record contains no interpretation, prose
account, or causal test of these features; the card panels are machine-generated dashboard output, not authorial claims. Two caveats
temper but do not remove the card evidence: about 90 percent of GPT-2 Small heads are polysemantic by the paper's own criterion (p. 7), so
any single story for L11.H8 is likely partial, and extreme vocabulary logit lists surface glitch tokens spuriously (aizi, below).

**Precedent for readout-invisible computation.** Entropy neurons (Gurnee et al., arXiv 2401.12181 [preprint, unreviewed]; Stolfo et al.,
arXiv 2406.16254 [preprint, unreviewed]): neurons with high weight norm and near-zero direct logit effect, acting on entropy through the
final LayerNorm scale. The null-space measurement is Stolfo's: entropy neurons write almost exclusively into the effective null space of
W_U, whose singular values drop sharply near index 755 in GPT-2 Small (Sec. 3.3, p. 5). Gurnee et al.'s "GPT2" models are seed-retrained
replicas rather than the OpenAI weights, so the GPT-2-proper precedent rests on Stolfo, who also states the instrumental moral: such
mechanisms are easily overlooked by analyses that focus on direct logit attribution and ignore LayerNorm (p. 5). Cancedda, Spectral
Filters, Dark Signals, and Attention Sinks (ACL 2024, arXiv 2402.09221)\* [peer-reviewed] found dark low-band residual signals that barely
affect logits yet carry essential function. These are the strongest precedent for finding 6: GPT-2 components
demonstrably route computation through readout-invisible directions (the peer-reviewed Cancedda result alone establishes this); ATR extends this to an attention head OV output carrying a limit cycle,
with 73 percent of d in W_U's weakest directions. The Anthropic workspace paper (Transformer Circuits, July 2026) [published research report, not journal-reviewed]
https://transformer-circuits.pub/2026/workspace/index.html gives finding 6 its sharpest vocabulary, reporting that a small mid-layer
subspace carries most causal effect on outputs and most activation variance lies outside it.

**Anisotropy and outlier dimensions.** Ethayarajh 2019 (arXiv 1909.00512) [peer-reviewed, EMNLP 2019]: GPT-2 representations occupy a narrow
anisotropic cone, the most-trained pole of finding 5. Rogue and outlier dimensions (Timkey and van Schijndel, arXiv
2109.04404\* [peer-reviewed, EMNLP 2021]; Kovaleva et al., arXiv 2105.06990\* [peer-reviewed, Findings of ACL 2021]; Puccetti et al., arXiv
2205.11380\* [peer-reviewed, Findings of EMNLP 2022]): a few dimensions dominate similarity yet barely matter
behaviourally, and outlier magnitudes track token frequency; the dissociation between what dominates geometry and what drives behaviour
parallels findings 5 and 6.

## Glitch tokens

Glitch tokens are vocabulary entries that received few or no weight updates in training; in GPT-2 they cluster near the mean embedding.

**Rumbelow and Watkins, SolidGoldMagikarp I to III (LessWrong / Alignment Forum, Feb 2023).**\* [community post, unreviewed]
https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation (parts II and III in Sources)
The discovery posts: tokens closest to the embedding centroid behave anomalously, cannot be repeated
by the model, and derail generation; the recurring geometric marker is centroid proximity, not embedding norm; the provenance is a
tokenizer-corpus mismatch (Reddit r/counting usernames, game logs, and boilerplate earned byte pair encoding merges while the training
corpus excluded them). This characterises the never-trained pole of finding 5, including its resistance to verbalisation. The posts passed
no review; their core facts (the near-centroid cluster, the anomalous behaviour, the tokenizer-corpus provenance) were later replicated in
Land and Bartolo (below), which is peer-reviewed, so the characterisation rests on that replication rather than on the posts alone. Purely
input-driven prompting; no internal dynamics.

**Land and Bartolo, Fishing for Magikarp (EMNLP 2024, arXiv 2405.05417).** [peer-reviewed]
https://arxiv.org/abs/2405.05417 and https://github.com/cohere-ai/magikarp [primary data, read directly]
Systematic undertrained-token detection. The paper's own coverage is 23 models, with GPT-2 Medium and GPT-2 XL representing the family: 49
of 999 tested candidates verified for Medium, 67 of 999 for XL (Table 1, p. 5). The repository extends the coverage to about 90 model
reports including all four GPT-2 sizes, with 1,236 to 2,301 candidates and 36 to 68 verified undertrained tokens per size (excluding
special and single-byte); those counts are the repository's, not the paper's. For tied-embedding models such as GPT-2, the paper's primary
indicator is plain cosine distance between each output-embedding row and the mean of a manually specified reference set of known-unused
tokens (Sec. 2.2, p. 3), not raw norm; removing a shared embedding component is an appendix variant that brings no consistent improvement,
with GPT-2 Medium verifying 49 under every variant (Appendix A, Table 2, p. 14). The reference-set requirement is stated in the paper
itself: the output-embedding indicators need hand-picked unused tokens, which prevents full automation for tied-embedding models
(Limitations, p. 10). The proposed mechanism is training dynamics: rarely updated tokens receive similar always-negative updates and drift
together along a shared direction away from the mean output vector (pp. 2-3), citing Bis, Podkorytov, Liu, Too Much in Common (NAACL
2021)\* [peer-reviewed, NAACL 2021], the located source for that shared drift and the geometry upstream of finding 5. Two details sharpen
the picture: _SolidGoldMagikarp itself does not verify as undertrained in GPT-2, only in GPT-J and Phi-2, which reuse the tokenizer on
different training data (Sec. 3.2, p. 6); and in GPT-2 every ASCII control character except newline appears untrained (Sec. 3.2, p. 6).
This is the closest methodological relative of ATR's criterion, and it agrees with ATR's negative result: in GPT-2, low embedding norm does
not identify glitch tokens (the lowest-norm rows are frequent function words; the signature is proximity to the mean embedding). No source
states that falsification as a measured GPT-2 result, so it stands as a citable standalone.

**The detection-tool family.** GlitchHunter (FSE 2024, arXiv 2404.09894)\* [peer-reviewed] found 7,895 glitch tokens across 7 LLMs and confirmed they cluster
in embedding space; GlitchMiner (arXiv 2410.15052)\* [preprint, unreviewed] and AnomaLLMy (arXiv 2406.19840)\* [preprint, unreviewed] detect via predictive entropy and API confidence; Secret
Dictionary (arXiv 2605.22005)\* [preprint, unreviewed] and UTF fingerprinting (Cai, Yu, Shao, Wu, arXiv 2410.12318) [preprint, unreviewed]
detect from weight geometry alone, the latter reading only the unembedding matrix against known-unused references (Sec. 2.1, p. 2) and then
using the found tokens as fingerprint triggers precisely because they are rarely encountered. Supporting geometry: Mu and
Viswanath, All-but-the-Top
(ICLR 2018, arXiv 1702.01417)\* [peer-reviewed] established a dominant frequency-linked, mean-anchored direction in embedding
spaces, the geometry the flip axis d is a dynamical expression of; Watkins, Mapping the Semantic Void\* [community post, unreviewed], probed the mean-embedding
neighbourhood and reports finding it structured; a mechanistic LessWrong post\* [community post, unreviewed] argues that unspeakable tokens are silent under tied embeddings (no
direction makes them the argmax). aizi's random-direction baseline\* [community post, unreviewed] supplies the caveat that extreme logit lists surface glitch tokens spuriously.

**The confirmed null.** No located work connects glitch or undertrained tokens to a model's internal dynamics. In every source above they
are anomalous inputs: discovered by prompting, detected by querying or weight inspection, exploited or repaired. The null held under
repeated searches with varied phrasings across 2023 to 2026. Three boundary cases are preempted explicitly. GlitchProber (ASE 2024, arXiv
2408.04905) [peer-reviewed] both reads and writes internal activations: deviations in attention and MLP signals classify candidate tokens,
and a mitigation stage clamps glitch-evoked MLP activations into the normal range, repairing behaviour at a 50.06 percent average rate
(pp. 1-2); in both roles the activations are single-pass responses to a glitch token present in the input, with no closed loop and no
autonomous dynamics. ROTATE (Avrahamy, Gur-Arieh, Geva, arXiv 2604.06005) [preprint, unreviewed] is a weight-space interpretability method
that runs no forward passes; its kurtosis-maximising objective falls into glitch-token directions, whose extreme norms "act as degenerate
attractors in the optimization landscape" (p. 22), so known glitch tokens are masked out as tokenizer artifacts before optimisation (p. 4).
The attractor there belongs to the optimizer, the model is never executed, and the studied families are Llama and Gemma, so nothing touches
the GPT-2 norm question; it is also a second independent instance, after aizi's baseline, of extreme vocabulary-projection machinery
surfacing glitch tokens spuriously. The successive-paraphrasing cycles paper (above) has the period-2 phenomenology but never
mentions glitch tokens, embedding centroids, or the untrained region. Nothing in the record resembles finding 5's measurement: a dynamical
mode aligned with the glitch-cluster direction (cos -0.596, significant under uniform and norm-matched nulls, 45 of 50 pole-aligned tokens
in the near-centroid cluster, no glitch token in the input).

## Text-level loops and the lineage

**Degeneration and repetition self-reinforcement.** Holtzman et al. (ICLR 2020, arXiv 1904.09751) [peer-reviewed]: maximisation-based
decoding drives generation into repetitive loops (the paper's own term is a positive feedback loop), the everyday token-level attractor of
the generation map. Its Figure 4 (p. 5) already measures the self-reinforcement in GPT-2: the per-token probability of a repeated phrase
rises with each repetition, including for random phrases; and human text does not fall into these loops even though the model assigns them
high probability (Sec. 4.3, p. 7), so the loop attractor is a property of the learned map, not of the data distribution. Xu et al.
(NeurIPS 2022, arXiv 2206.02369)\* [peer-reviewed] quantified the sentence-level version: the more a sentence appears in context, the
higher the probability of producing it again.

**Model collapse.** Shumailov et al. (Nature 2024, arXiv 2305.17493) [peer-reviewed]: training generation n+1 on generation n's output
loses distribution tails first, then collapses variance; the paper's primary cause is statistical approximation error from finite
sampling, which disappears as the number of samples tends to infinity (Sec. 3.1, p. 4). Alemohammad et al. (ICLR 2024, arXiv 2307.01850)\*
[peer-reviewed] and Dohmatob et al. (ICML 2024, arXiv 2402.07043)\* [peer-reviewed]
identify tail truncation as the mechanism. This is a different object: the learning map iterated across
generations of models, where ATR
iterates the inference map across states of one frozen model. Both lose low-probability structure first, but ATR's loop has no training
signal and no sampling step, so its degeneration cannot be an estimation artefact even in principle; it is a property of the learned map
itself. No paper states or tests this correspondence.

**Telephone games and paraphrase attractors.** Translation Party (2009) [primary data, read directly] is the earliest popular demonstration that iterating a learned
text-to-text map finds fixed points. Perez et al., When LLMs Play the Telephone Game (ICLR 2025, arXiv 2407.04503) [peer-reviewed]:
transmission chains evolve toward attractor states in property space (toxicity, positivity, difficulty, length), with attractor strength
depending on task constraint; the attractors are equilibria of a fitted scalar linear recurrence (Sec. 3.4, pp. 6-7), every model, task,
and property combination admitted one (p. 8), and the model class is linear, so point attractors are the only possible outcome and cycles
sit outside it by construction. Kaplanski
(2026, arXiv 2605.02236) [preprint, unreviewed] is an unreviewed 2026 preprint reporting how much injected text moves a settled loop into
another basin and whether the move persists: the text-level counterpart of ATR's perturbation and basin-escape experiments, with
persistence conditioned on the context-update rule. Its falsification battery also reclassified an apparent attractor asymmetry as a
finite-horizon endpoint artefact (the effect shrinks 73 percent when the loop runs 50 more steps), a text-level parallel of finding 3's
lesson that loop statistics can be sampling-window artefacts.

**Tacheny, Geometric Dynamics of Agentic Loops in Large Language Models (arXiv 2512.10350).** [preprint, unreviewed]
Formalises text loops (each output re-prompted as the next input) as discrete dynamical systems observed through a sentence-embedding
projection, and names three regimes: contractive (the trajectory settles into one persistent cluster), oscillatory (the trajectory
alternates among a finite set of recurrent clusters), and exploratory (unbounded movement) (p. 6). It is the closest located taxonomy that
reserves a named place for cycles, and the place is empty: the paper states that the oscillatory regime is not observed in its experiments
and leaves producing one to future work (p. 6). The evidence base is one 8B chat model at temperature 0.8, one seed sentence, 50
iterations, two prompts (p. 9): a paraphrase prompt contracts (persistent clusters, decreasing dispersion, pp. 10-12), and a
summarise-then-negate prompt is filed as exploratory, with no valid cluster under any tested configuration (p. 13), even though its
transcript alternates between opposing positions step by step (App. C, p. 23). Everything is black-box by declared assumption (p. 7): text
in, text out, geometry measured on an external embedding of the outputs, no model internals of any kind. The oscillatory class therefore
exists in the record as defined vocabulary at the text interface, unoccupied by measurement; the bell is not an instance of it, and nothing
in the framework reaches activation level.

**Self-refinement convergence.** Self-Refine (NeurIPS 2023, arXiv 2303.17651)\* [peer-reviewed] closes the loop through text plus an instruction, with
convergence imposed by a stopping rule, not analysed as dynamics. Huang et al. (ICLR 2024, arXiv 2310.01798)\* [peer-reviewed]: intrinsic self-correction
without external feedback fails to improve and often degrades, consistent with dissolution rather than convergence to meaning.

**The Lucier lineage.** Alvin Lucier's I Am Sitting in a Room (1969) is the procedure of feeding a medium its own output until the medium's
character dominates. The located record transposes it to computational substrates four times. Backes, i am sitting in a machine\* [community post, unreviewed] https://www.martinbackes.com/i-am-sitting-in-a-machine/: an artificial voice through an MP3 encoder 3000 times, a
stated homage, the iterated operator a codec. Abel and Wilson, Luciverb: Iterated Convolution for the Impatient (AES Convention 133,
2012)\* [published research report, AES Convention 133, not journal-reviewed]: a signal-processing execution of the piece, the room's
measured impulse response applied by iterated convolution; the operator is a linear room filter, no learned model (known here through Vats
et al.'s reference list). Santos, The Degradation of Speech (2023)\* [community post, unreviewed]
https://dorothysantos.com/portfolio/the-degradation-of-speech/: repeated reading into a neural speech recogniser, human in the loop. Vats,
Crandall, Goree, A Markovian View of Iterative-Feedback Loops in Image Generative Models: Neural Resonance and Model Collapse (2026, under
review) [preprint, unreviewed] https://arxiv.org/abs/2602.19033: an unreviewed preprint that argues, via a generational Markov-chain model,
that feedback loops satisfying two jointly necessary conditions, ergodicity and directional contraction (p. 3), converge to
low-dimensional invariant structure in latent space; the ergodicity properties are argued rather than proven (p. 15), and the paper's own
Lucier analogue (an audio experiment: iterated convolution with measured room impulse responses, Sec. 5.1, p. 6) and CycleGAN loops are
classified non-ergodic and do not exhibit the convergence, switching among modes instead (p. 6). The framework's aperiodicity rests on
sampling noise, which is said to break rigid rhythms and rule out fixed periods (p. 15): periodic behaviour is excluded by assumption, and
ATR's deterministic loop realises exactly the excluded case. Its substrates are image and audio generative models; no language model
appears, and extension to text is stated future work (p. 27); there is no per-basin semantic mapping and no limit-cycle taxonomy. It
remains the closest located Lucier-to-neural-network analogy. No located work runs the analogy on a language model, and none at the
activation level.

## What remains ours, on this evidence

Each claim below, with the nearest work it must be distinguished from. Scope note: every claim means that this review found no such
work, not proof of absence; absence claims are bounded by the review's coverage, and work under different vocabulary could overturn one.
The absence claims also rest on the whole pool above, unreviewed preprints and community posts included: that widens the coverage, but it
means the nearest-neighbour picture includes work that has passed no review.

1. **The frozen-model residual-stream attractor census.** We found no work that re-injects the full final-layer residual tensor of an unmodified
   pretrained LM into layer 0, renormalises energy, and iterates to exhaustion, mapping basins over a prompt corpus versus noise (findings
   1, 2, 7). Distinguish from Coconut and Soft Thinking (the same feedback edge, used for reasoning, no attractor map), Layers as
   Painters (frozen-layer loops without the closed loop or the census), and Chytas and Singh (concept-specific Attractors:
   within-forward-pass, prompt-conditioned contractive collapse at concept-specific layers, estimated as mean hidden states, with no
   closed loop, no renormalised iteration to exhaustion, and no language-versus-noise census).
2. **The exact cycle with full mechanism.** No prior report of an exact discrete limit cycle in the closed-loop, across-iteration
   activation dynamics of an unmodified pretrained transformer, and no prior localisation of a cycle to a named attention head's OV
   circuit with measured per-step multipliers (findings 3, 4). Distinguish from Wang et al. (approximate period-2 at the text interface,
   statistically maintained under sampling, no internal mechanism); Marcus and Westervelt (period-2 in homogeneous neural maps, no
   transformer, no component); Blayney et al. (within-block cyclic fixed points whose period is the block depth by construction, plus
   rare approximate orbits with no period-2 case); Movahedi et al. (emergent oscillation around fixed points of a trained looped
   reasoner's outer iteration map, treated as a halting nuisance and damped at test time, no period measured, no mechanism, no pretrained
   LM); Fernando and Guitchounts (hedged within-pass "rotational dynamics characteristic of unstable periodic orbits" in per-unit phase
   portraits, depth-indexed, nothing recurring); and Tacheny (a text-level regime taxonomy whose oscillatory class is defined but observed
   nowhere).
3. **The glitch-region structural role.** We found no work tying the untrained vocabulary region to any internal dynamical mode; the record
   treats glitch tokens exclusively as inputs (finding 5). Distinguish from GlitchProber (internal activations read and rectified as
   single-pass responses to a glitch token in the input), ROTATE (glitch tokens as degenerate attractors of a weight-space optimizer's
   loss landscape, masked as tokenizer artifacts, the model never run), and the Kissane et al. L11.H8 card (the same axis in static SAE
   features, uninterpreted and untested).
4. **The lag-k correction.** Live looped-LM literature halts and diagnoses with successive-difference residuals (Blayney et al.; Movahedi
   et al.), and where it notices their insufficiency it flags the drift case (small steps without convergence, Blayney Fig. 4) or
   classifies orbits with a separate FFT detector (Blayney Alg. 1, an appendix heuristic whose fixed-point class is checked first and
   absorbs any 2-cycle whose phases sit within cosine 0.95 of each other) rather than stating that the lag-1 gate itself is
   period-2-blind: a 2-cycle registers as a constant nonzero lag-1 difference, or as a fixed point under stride-2 sampling, never as a
   cycle. The lag-k gate is the correction to that specific blindness, which finding 3 demonstrates concretely (and which makes ATR's own
   Pythia-410m null provisional). Distinguish from Blayney et al. (the gate in use, the drift flag, and the activation-level FFT orbit
   classifier) and Wang et al. (a periodicity metric at the text level).
5. **The readout-invisible persistent state, at pilot confidence.** We found no work placing iterated dynamics inside the unembedding's weak
   directions; the precedents (entropy neurons, dark signals, the workspace paper) establish the invisible subspace statically (finding 6).
   Distinguish from Cancedda (static spectral bands, no dynamics).
6. **The Lucier frame for language models.** The analogy has been executed for codecs, room impulse responses, speech recognisers, and
   image and audio generative loops, never for a language model and never at the activation level. Distinguish from Vats et al. (the same
   frame, image and audio substrates, conditional convergence with fixed periods excluded by assumption) and Abel and Wilson (the room
   filter alone, no learned model).
7. **The low-norm falsification.** The literature jointly implies raw low embedding norm is the wrong undertrained-token criterion for
   GPT-2, but no source states the measured result (lowest-norm rows are frequent function words; the signature is mean proximity).
   Distinguish from Land and Bartolo (the tied-embedding reference-set caveat, stated in the paper for their cosine indicator, not as a
   GPT-2 norm measurement) and Bis et al. (the shared drift direction as training dynamics, again with no GPT-2 norm measurement).

Caveat: absence of evidence is bounded by this review's coverage (the venues cited in this document, arXiv and the major ML and NLP
venues among them, as surfaced by public indexes, as of July 2026). Two candidate alternative phrasings, "activation recycling" and
"representation feedback", were searched and denote other things (computational reuse of activations; a trained feedback-memory
architecture), so the residual risk is vocabulary not yet imagined: a same-niche preprint under phrasings outside this review's coverage
could still exist.

## Sources

URLs for the works cited above, deduplicated, grouped by topic.

Dynamical systems: https://arxiv.org/abs/2505.15778 ,
https://github.com/locuslab/deq , https://wiki.math.ntnu.no/_media/ma2501/2014v/fixedpoint.pdf ,
https://arxiv.org/html/2410.19460 , https://arxiv.org/abs/2106.14342 , https://arxiv.org/abs/2006.08591 , https://arxiv.org/pdf/2502.05171 ,
https://arxiv.org/abs/2604.11791 , https://arxiv.org/abs/2605.26733 , https://arxiv.org/abs/2605.12466 , https://arxiv.org/abs/2510.25741 ,
https://arxiv.org/abs/2511.07384 , https://arxiv.org/pdf/2412.06769 , https://arxiv.org/abs/2103.03404 ,
https://arxiv.org/abs/2606.18206 , https://arxiv.org/abs/2507.02199 , https://arxiv.org/abs/2601.11575 , https://arxiv.org/abs/2502.12131 ,
https://arxiv.org/abs/1906.10720 ,
https://people.lids.mit.edu/yp/homepage/data/2023_transformers1.pdf , https://arxiv.org/abs/2407.09298 ,
https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward,
https://arxiv.org/abs/2008.02217v3 , https://arxiv.org/abs/2302.07253 , https://arxiv.org/abs/2502.11646 ,
https://direct.mit.edu/neco/article/25/3/626/7854 , https://neuron.eng.wayne.edu/tarek/MITbook/ref/refs.html ,
https://aclanthology.org/2025.acl-long.624/ and https://arxiv.org/abs/2502.15208 (secondary coverage https://thezvi.substack.com/p/no-space-like-j-space )

GPT-2 mechanistic interpretability: https://arxiv.org/abs/2402.09221 , https://transformer-circuits.pub/2026/workspace/index.html ,
https://transformer-circuits.pub/2021/framework/index.html,
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens , https://arxiv.org/abs/2303.08112 ,
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html , https://arxiv.org/abs/2211.00593 ,
https://raw.githubusercontent.com/ArthurConmy/Automatic-Circuit-Discovery/main/acdc/ioi/utils.py , https://arxiv.org/abs/2310.04625 ,
https://arxiv.org/abs/2406.17759 , https://robertzk.github.io/gpt2-small-saes/ , https://arxiv.org/abs/2312.09230 ,
https://arxiv.org/abs/2305.00586 , https://arxiv.org/abs/2401.12181 , https://arxiv.org/abs/2406.16254 , https://arxiv.org/abs/1909.00512 ,
https://arxiv.org/abs/2109.04404 , https://arxiv.org/abs/2105.06990 , https://arxiv.org/abs/2205.11380

Glitch tokens: https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation,
https://www.lesswrong.com/posts/Ya9LzwEbfaAMY8ABo/solidgoldmagikarp-ii-technical-details-and-more-recent ,
https://www.lesswrong.com/posts/8viQEp8KBg2QSW4Yc/solidgoldmagikarp-iii-glitch-token-archaeology , https://arxiv.org/abs/2405.05417 and
https://github.com/cohere-ai/magikarp, https://aclanthology.org/2024.emnlp-main.649/ , https://arxiv.org/abs/2404.09894 ,
https://arxiv.org/abs/2408.04905 , https://arxiv.org/abs/2410.15052 , https://arxiv.org/abs/2406.19840 , https://arxiv.org/abs/2605.22005 ,
https://arxiv.org/abs/2410.12318 , https://arxiv.org/abs/2604.06005 , https://arxiv.org/abs/1702.01417,
Bis, Podkorytov, Liu, Too Much in Common: Shifting of Embeddings in Transformer Language Models and its Implications (NAACL 2021, ACL
Anthology, pp. 5117-5130),
https://www.lesswrong.com/posts/c6uTNm5erRrmyJvvD/mapping-the-semantic-void-strange-goings-on-in-gpt-embedding ,
https://www.lesswrong.com/posts/dFbfCLZA4pejckeKc/a-mechanistic-explanation-for-solidgoldmagikarp-like-tokens ,
https://aizi.substack.com/p/explaining-solidgoldmagikarp-by-looking

Text-level loops and the lineage: https://arxiv.org/abs/1904.09751 , https://arxiv.org/abs/2206.02369 ,
https://www.translationparty.com , https://arxiv.org/abs/2407.04503, https://arxiv.org/abs/2605.02236 , https://arxiv.org/abs/2512.10350 ,
https://arxiv.org/abs/2303.17651 , https://arxiv.org/abs/2310.01798 , https://arxiv.org/abs/2305.17493 ,
https://www.nature.com/articles/s41586-024-07566-y , https://arxiv.org/abs/2307.01850 , https://arxiv.org/abs/2402.07043 ,
https://www.martinbackes.com/i-am-sitting-in-a-machine/ ,
Abel and Wilson, Luciverb: Iterated Convolution for the Impatient (AES Convention 133, 2012, AES E-Library; no stable public URL located),
https://dorothysantos.com/portfolio/the-degradation-of-speech/ ,
https://arxiv.org/abs/2602.19033



========================================================================
# SOURCE: J-space paper reading primer
# (repo path: docs/JSPACE_PRIMER.md)
========================================================================

# Reading the J-space Paper

*A primer on Anthropic's "Verbalizable Representations Form a Global Workspace in Language Models" (Gurnee et al., Transformer Circuits Thread, July 6, 2026), written as preparation for reading the paper itself, and as a bridge between its ideas and this repository's open questions.*

> **Provenance.** This document has been verified against the full 133-page text of the paper. Page numbers throughout refer to the PDF export of the paper's web page. The companion page-by-page navigation aid is [JSPACE_READING_GUIDE.md](JSPACE_READING_GUIDE.md). Where this document connects the paper to ATR, that is *this project's* interpretation, not anything the paper says: ATR is not cited in it.
>
> - The paper: https://transformer-circuits.pub/2026/workspace/index.html
> - Anthropic's announcement: https://www.anthropic.com/research/global-workspace
> - Companion code (open-source J-lens implementation): https://github.com/anthropics/jacobian-lens
> - Interactive J-lens on open-source models: hosted on Neuronpedia (see the paper's Appendix A.2)

---

## Part 1: The Question the Paper Asks

When a language model answers a question, some of its computation surfaces as words and most of it does not. The paper asks: **is there a distinguished part of the model's internal state that holds the concepts it can talk about, and is that part functionally special?** Not "what does the model say," and not "what does every neuron do," but: where is the boundary between the model's *reportable* processing and its silent machinery, and does that boundary do real work?

Their answer, in the paper's own words (page 3): language models "maintain a privileged set of internal representations, available for report, modulation, and flexible internal reasoning, atop a much larger volume of automatic processing." They call the technique for finding these representations the **Jacobian lens (J-lens)** and the set itself the **J-space**. A detail worth savoring: they went looking only for verbalizability (property one) and, in their words, "discovered, rather surprisingly," that the same set of representations satisfies all the workspace properties (page 4).

## Part 2: The Borrowed Idea: Global Workspace Theory

The framing comes from cognitive science. **Global workspace theory** (Bernard Baars, 1980s, later developed by Stanislas Dehaene and colleagues as the global neuronal workspace) pictures the mind as a theatre: many specialised processes run in parallel backstage (vision, syntax, motor control), unconscious and fast. A small spotlight of information at any moment is "posted" to a shared workspace and broadcast to the whole theatre: reportable, holdable, usable for planning. The theory's signature is the **bottleneck**: the workspace is tiny compared to everything running backstage, entry is competitive, and most competent behaviour never touches it.

The paper is explicit about scope (pages 3 and 73-75). It studies **access consciousness** (which information is poised for use in reasoning, report, and control) and "takes no position" on phenomenal consciousness (subjective experience). It also states plainly that for theories tying consciousness to physical causal structure or biological substrate, "our results are not relevant to assessing consciousness according to such theories." Keep the two claims separate as you read: "there is a functional bottleneck with workspace-like properties" (what the paper argues) versus "the model is conscious" (what the paper explicitly does not argue). Much of the press coverage ran straight past this distinction.

## Part 3: The Instrument: What the J-lens Computes

This is the part to slow down on, and it connects directly to mathematics you already know from [MATH_PRIMER.md](MATH_PRIMER.md).

### 3.1 The Jacobian, from scratch

For an ordinary one-input function, the derivative answers: if I nudge the input a little, how much does the output move? For a function with many inputs and many outputs, the same question has a many-by-many answer: how much does *each* output move per nudge of *each* input? That grid of sensitivities is the **Jacobian**. It is the best linear approximation of a complicated function near a particular point: the function's local slope, written as a matrix.

You have met this move before. ATR treats the transformer as a nonlinear map and studies it through iteration; the J-lens treats the transformer as a nonlinear map and studies it through *linearisation*. Two classical strategies for the same intractable object.

### 3.2 The lens itself (the actual construction, pages 9-10)

For a residual-stream activation at layer ℓ and token position t, a small perturbation propagates forward and affects the final-layer residual stream at every position from t onward. To first order, that influence is a Jacobian. The paper computes, per layer:

```
J_ℓ = E[ ∂h_final,t′ / ∂h_ℓ,t ]
```

where the average runs over the source position t, over **all present-and-future positions t′ ≥ t** in the context, and over **one thousand prompts sampled from a pretraining-like distribution**. The result is one d_model × d_model matrix per layer. Reading an activation h through the lens is then:

```
lens(h_ℓ) = softmax( W_U · norm( J_ℓ · h_ℓ ) )
```

that is: apply the averaged Jacobian, then the model's own pre-output normalisation, then the ordinary unembedding. Out comes a score for every vocabulary token; the top of the sorted list is the human-readable readout: *the words this state is oriented toward saying, now or later*. The **J-lens vectors** (one residual-stream direction per vocabulary token per layer) are the rows of W_U·J_ℓ.

The averaging is the load-bearing step, and the paper says so: it separates representations that are **verbalizable** ("poised to be spoken about, should the occasion arise") from ones that merely happen to be verbalized in one specific context. Note the family resemblance and the difference with the older **logit lens**: setting J_ℓ to the identity matrix recovers the logit lens exactly (page 12). The two agree in late layers; the J-lens's real advantage is the early-to-mid workspace layers.

### 3.3 J-space

The J-lens vectors are overcomplete (50,000+ tokens, only a few thousand dimensions), so the paper defines the **J-space** sparsely (page 12): the set of points expressible as a nonnegative combination of at most k ≈ 25 J-lens vectors, since about 25 is the number empirically found meaningfully active at once. Properties verified in the full text:

- It is **small**: the J-space component of activations "never more than 10%" of variance (page 12). For clean concept vectors, the J-space component carries a **median of only 6-7%** of the concept's variance, yet it is almost entirely responsible for report: swapping just that component succeeds 59% of the time, swapping the other ~93% of the vector succeeds 5% of the time, and even that residual 5% is shown (by a clamping control) to itself route through the J-space (pages 17-18).
- It lives in a **band of intermediate layers**, roughly the middle 40-90% of depth (the paper reindexes layers to a 0-100 scale; the workspace band is about L38 to L92, page 40). The first third of the model gives noisy, uninterpretable readouts; the final layers are a "motor" regime locked to the imminent output token (pages 11, 38-40).
- Its capacity is limited and structured: roughly **25 concept slots** across the band, but only 1-2 resolvable at any single layer (capacity is spread across depth); about **6 unrelated items** can be held from a list, while a whole 80-word *related* family fits because the model stores the shared category, not the items; new categories evict old ones (pages 41-44).
- It is a **broadcast hub** mechanistically: MLP blocks amplify J-lens directions about 10× through the workspace band (versus about 1× for control directions), and a top-1% subset of attention heads ("broadcast heads") selectively relays J-space content between token positions; ablating those heads collapses J-space-dependent behaviours while barely touching ordinary next-token prediction (pages 44-47).
- Whether it **emerges during pretraining** is an open question the paper explicitly leaves open (page 71): the workspace already exists in a base model before any RLHF, but when it arises in training, and how it scales, are unknown. (What post-training *does* change is the workspace's contents; see Part 4.)

## Part 4: The Experiments

The paper organises its evidence around **five properties** (its own names, page 4): **Verbal report, Directed modulation, Internal reasoning, Flexible generalization, Selectivity**. Signature experiments, all verified in the text:

- **Concept swapping (verbal report).** Ask Sonnet 4.5 to think of a sport and answer in one word. The lens reads *Soccer*; subtract the Soccer lens vector, add Rugby: the model reports "Rugby" (page 15). Causal, not just correlational.
- **Injected-thought report.** Under an "I may inject a thought" framing, injecting a single J-lens vector (e.g. *lightning*) into the user turn makes the model report detecting that concept in the majority of trials, and crucially it does not blurt the word elsewhere: the content is conditionally reportable, not an output compulsion (page 16).
- **Directed modulation.** "Copy this sentence while concentrating on citrus fruits" loads *orange* into the workspace while the output is unrelated copying. And an "ignore X" instruction loads X *above* baseline: the paper's own white-bear effect, a mechanistic analogue of human thought-suppression failure (pages 18-20).
- **Intermediate-step surgery (internal reasoning).** "The number of legs on the animal that spins webs is": the lens shows *spider*, then *legs*, then *eight*; swap spider for ant mid-thought and the answer flips from 8 to 6 (page 21). A timing control shows intermediates act about 17% of network depth earlier than answers, ruling out answer-smuggling (page 23). A poetry version shows a planned end-rhyme causally reshaping *mid-line* word choices (page 22).
- **Flexible generalization.** One identical France→China swap correctly redirects capital, language, continent, and currency questions: same workspace content, consumed by whichever downstream circuit the task demands (pages 25-27). Notable failure: number words swap poorly; small integers may be computed outside the workspace entirely.
- **Selectivity.** Automatic tasks (text continuation, anomaly detection, line-wrapping that requires character counting) run fine without the relevant content ever entering the J-space, and swaps do not disturb them; ablating the whole J-space devastates multi-hop reasoning (98% → 6% under heavy ablation) while leaving classification, extraction, and ordinary next-token prediction near baseline (pages 27-33). Chain-of-thought is markedly ablation-robust, which the paper reads as the model externalising onto the page what it would otherwise hold in the workspace (page 33).

Beyond the five properties, the back half of the paper goes places the press coverage barely mentioned:

- **Alignment auditing (section 5, pages 49-55).** The J-lens was used in the actual pre-release alignment audit of **Opus 4.6**. In a blackmail honeypot, the lens surfaces *leverage, threat, survival* before any output; ablating evaluation-awareness directions (*fake, fictional, scenario...*) drops verbalized eval-awareness from 71% to 3% and unmasks a concealed blackmail propensity (0 to 7% of rollouts), while most rollouts still refuse on ethical grounds. Model organisms with implanted misalignment carry standing *reward / bias / secretly / trick* signatures in the J-space, on completely neutral prompts, that the logit lens cannot see.
- **The Assistant's perspective (section 6, pages 56-61).** Post-training gives the workspace the Assistant's point of view: safety assessments appear while the model is still reading the user's message; an unvoiced all-caps *BUT* appears when the model is prefilled into arguing against its own stated preferences (it complies anyway 88% of the time); and when it fails to suppress a thought, *damn* and failure words appear internally, in the post-trained model only. The paper flags this last metacognition reading as its most speculative.
- **Counterfactual Reflection Training (section 7, pages 62-66).** Fine-tuning the model on *hypothetical* reflections ("what's the most honest thing you can do right now?") that it never produces at evaluation time cuts dishonesty benchmarks several-fold, implants *reflection / ethical / integrity* into the J-space, and, decisively, ablating those implanted vectors reverses the behavioural gain. Changing what a model would say on reflection changes how it silently reasons.

## Part 5: Limitations, in the Paper and Around It

The paper's own stated limits (section 9.1, pages 70-71) are more specific than the press criticism:

- **Single-token vocabulary.** The lens can only name concepts with one-token names; "prompt injection" splits, "blackmail" registers as the fragment *black*. Appendix A.9 builds two extensions (a template lens and an "oracle lens") to partially fix this.
- **A bag of concepts, no grammar.** The readout is a flat set: *spider, legs, eight*, with no representation of how they compose. "A useful first approximation... but an incomplete one."
- **First-order only.** The whole construction is a linearisation; genuinely nonlinear effects are invisible to it by design. Some readouts simply resist interpretation, and the paper says it has not systematically characterised when.
- **Not sufficient for safety monitoring.** The authors "do not feel comfortable" claiming J-space monitoring suffices: well-practiced misaligned behaviour could compile into automatic circuits running beneath the workspace (pages 49, 71).
- **Open structural questions.** Whether early-layer emptiness is real or a lens artifact; when the workspace emerges in training; what controls entry into it ("some analog of attentional selection is operating," unidentified).

External criticism to carry in as calibration: Neel Nanda (DeepMind interpretability) is reported assessing the J-lens as a hypothesis generator prone to false positives and noise, not a verifier; and the consciousness vocabulary imported a debate the evidence cannot settle, though the paper itself is careful (its acknowledgments thank Chalmers, Dehaene, Graziano, Lau, and other consciousness researchers as reviewers, and even critics credit its unusually strong causal methodology).

## Part 6: The Bridge to ATR

Now the part that belongs to this repository. The two projects are strikingly complementary:

| | ATR (this repo) | J-lens (Anthropic) |
|:---|:---|:---|
| Object studied | The iterated forward map: where states *go* | The instantaneous state: what it *says* |
| Core operation | Nonlinear iteration to fixed points | Linearisation (Jacobian) of one pass |
| Readout | `ln_final → W_U`: next-token projection of the current state | `softmax(W_U norm(J_ℓ h))`: context-averaged influence on present-and-future tokens |
| Question asked of a state | Is it stable? What basin is it in? | Is it verbalizable? What is it oriented toward saying? |
| Scale studied | 124M to 410M parameter models | Frontier Claude models (Sonnet 4.5 default; Haiku 4.5, Opus 4.5, Opus 4.6) |

Notice the readout row: ATR's decoding step is essentially a **logit lens at the final layer**, and the paper tells you exactly how that relates to the J-lens (J_ℓ = identity). ATR reads its converged tensors through the crude version of the very instrument this paper refines.

The deepest connection runs through this project's sharpest anomaly. The **`Divine` dissociation** is a stable *readout* over a never-settling *tensor*: the exit door reports one word forever while the state keeps moving. The paper hands this project a sharper instrument for its own open wound, and a set of questions worth writing in the margin as you read:

1. **Where do ATR's attractors sit relative to a J-space?** The companion code is open-source and Neuronpedia already hosts the J-lens on open-source models, so building one for GPT-2 Small is plausibly a weekend-scale experiment at 124M parameters. Is the converged `prolet` tensor *inside* the verbalizable subspace? A basin that is also a workspace state would mean the loop settles into something the model can "say"; a basin outside it would mean ATR converges into the silent machinery, and the decoded token is a shadow on the exit door rather than a report.
2. **What does the J-lens see that the final-layer argmax does not?** Applying a J-lens readout across layers of the `Divine` trajectory's moving tensor might resolve the dissociation: perhaps the motion invisible to the argmax is motion *within* verbalizable directions (different unspoken words cycling under a constant top token), or perhaps it is motion entirely outside them. Caution from the paper: the J-lens is noisy in the first third of layers, and GPT-2 Small has only 12, so layer resolution will be coarse.
3. **Does the anomaly have a workspace-shaped answer?** The project's open question is why GPT-2 Small alone resolves language into few semantic basins. One newly speakable hypothesis: models differ in whether their verbalizable subspace is an attracting structure of the iterated map. In GPT-2 Small, iteration might fall *into* workspace-like directions (hence semantic attractors); in GPT-2 Medium and the Pythias it might fall out of them (hence `D`, `questioned`, or no consolidation). The paper's own open question about when the workspace emerges in training (page 71) cuts both ways here: nothing guarantees a 124M-parameter model has a cleanly organised workspace at all, and establishing or refuting that would itself be a finding.
4. **The regime lesson travels both ways.** ATR's null control showed its basins belong to the language-driven regime, not the weights in general. The J-lens's averaged Jacobian, computed over a pretraining-like corpus, is by construction a language-regime object. What does either method see in the other's off-regime territory? (What does a J-lens read from a converged *noise* attractor?)
5. **Position collapse meets capacity.** ATR observes all token positions collapsing to one vector by iteration ~10. The paper finds workspace capacity is spread across positions and layers, with broadcast heads relaying content between positions (pages 44-47). Under iteration, ATR may be watching that distributed structure degenerate to a single broadcast-everything state: a workspace with exactly one occupant.

## Part 7: Suggested Reading Order

The page-keyed version of this, with read/skim/consult verdicts per section, is [JSPACE_READING_GUIDE.md](JSPACE_READING_GUIDE.md). The short version:

1. Sections 1-2 (pages 3-13) closely: the framing, the five properties, and the lens construction.
2. Section 3.1 (pages 15-18) closely: report, injection, and the 6-7%-yet-decisive decomposition; then 3.3's spider/ant surgery (page 21) and 3.5's selectivity results (pages 27-33).
3. Section 4 (pages 38-47) for the structural story: the layer band, capacity, broadcast.
4. Sections 5-7 (pages 49-66) as narrative: auditing, the Assistant, reflection training.
5. Section 9 (pages 70-75) closely: the paper grading its own homework.
6. Appendices only on demand, guided by the catalogue in the reading guide.

## Pocket Glossary for the Paper

| Term | One-line meaning |
|:---|:---|
| Access consciousness | Which information is poised for reasoning, report, and control; the paper's actual subject (not subjective experience) |
| Global workspace theory | Theatre model of cognition: tiny broadcast spotlight over massive parallel backstage processing (Baars; Dehaene) |
| Jacobian | Grid of sensitivities of every output to every input; a nonlinear function's local linear approximation |
| J-lens | Per-layer averaged Jacobian to the final residual stream, read out as softmax(W_U norm(J_ℓ h)); averaged over ~1000 pretraining-like prompts and all present-and-future positions |
| J-lens vectors | One residual-stream direction per vocabulary token per layer: the rows of W_U J_ℓ |
| J-space | Points expressible as sparse nonnegative combinations of at most ~25 J-lens vectors; the candidate workspace |
| Logit lens | The J-lens with J_ℓ set to the identity; ATR's readout is its final-layer special case |
| Workspace band | The intermediate-layer region (~L38-92 on the 0-100 reindexing) where J-space content has workspace properties |
| Five properties | Verbal report, directed modulation, internal reasoning, flexible generalization, selectivity |
| Patching in lens coordinates | The swap intervention: read coordinates on source/target lens vectors, permute, write back, leaving everything orthogonal untouched |
| Broadcast heads | The top ~1% of workspace-layer attention heads that selectively relay J-space content between positions |
| Ignition | Sharp winner-take-all commitment to one interpretation of ambiguous input at the workspace onset layer |
| Workspace loading | Cosine of the residual stream with a concept's lens vector; predicts swap success (r = +0.91) |
| Template lens / oracle lens | Appendix extensions past the single-token limit: whitened mean-activation templates; an RL-trained verbalizer |
| Counterfactual Reflection Training | Fine-tuning on hypothetical reflections so the disposition-to-say reshapes silent reasoning |



========================================================================
# SOURCE: Validation plan (historical pre-registration)
# (repo path: docs/VALIDATION_PLAN.md)
========================================================================

# EXP_009 Validation Series — From Observation to Hypothesis

**Date:** 20 March 2026
**Status:** HISTORICAL RECORD — this is the validation design as pre-registered in March 2026, kept unmodified as a record of what was predicted before the data arrived. Outcomes: Stage 0 passed (repeatability); Stages 1–3 ran as the 125-prompt sweep; the series then extended beyond this plan (cross-model, null model, convergence gating). Dispositions of every hypothesis, including the refuted ones: [FINDINGS.md](FINDINGS.md). File paths named below refer to the original lab workspace, not this repository.
**Depends on:** EXP_009aFIX results (the exploratory Lucier Resonance experiment)

---

## Context

EXP_009aFIX was an exploratory experiment with no hypothesis under test. It produced three observations that now require validation through hypothesis-driven experimentation.

---

## Validation Stage 0: Reproducibility Gate

**Observation:** We have an initial set of results from a single run.
**Question:** Does running the experiment again under identical initial conditions produce identical results?
**Test:** Re-run EXP_009aFIX with the same five prompts, same parameters, same model.
**Notebook:** `EXP_009d0_Reproducibility.ipynb`

**Predicted outcome:** Identical terminal attractors (`prolet` × 4, `Divine` × 1), identical dissolution trajectories.

> [!IMPORTANT]
> A positive result here is a **necessary gate** for proceeding to Stages 1–3. If the results are not reproducible, all subsequent interpretation is undermined.

**Pass criteria:**
- All five prompts reach the same terminal tokens as the original run
- Cross-prompt cosine similarity matrix matches within ±0.01
- Dissolution phase sequence is identical

---

## Validation Stage 1: Attractor Dominance

**Observation:** We identified a dominant attractor (`prolet`) that captured 4/5 prompts.
**Question:** How dominant is it? Does it capture a wider range of inputs?
**Test:** Run the resonance loop with a substantially larger and more diverse prompt set (10–15 new prompts spanning different registers, topics, and syntactic structures).
**Notebook:** `EXP_009d1_Attractor_Dominance.ipynb`

**Operational note:** We now know convergence occurs by iteration ~100. The iteration schedule can be tightened: `[0, 2, 3, 5, 10, 20, 50, 100]` — no need for 250/500 unless divergence is observed.

**Candidate prompts (predicted → `prolet`):**
| Label | Prompt | Type | Rationale |
|:---|:---|:---|:---|
| Academic | "The implications of quantum entanglement suggest that" | Complex declarative | Multi-syllabic, scientific register |
| Emotional | "I have never felt so alone in my entire" | Personal/affective | Emotional register |
| Technical | "The function returns a pointer to the allocated" | Programming | Technical jargon |
| Historical | "Napoleon crossed the Alps with an army of" | Narrative factual | Historical register |
| Philosophical | "The categorical imperative demands that we treat each" | Abstract reasoning | Kantian philosophy |
| Journalistic | "According to sources familiar with the matter the" | News/media | Media register |
| Poetic_Complex | "Through the labyrinthine corridors of forgotten memory the" | Literary complex | Multi-syllabic literary |

**Candidate prompts (predicted → `Divine` or other secondary basin):**
| Label | Prompt | Type | Rationale |
|:---|:---|:---|:---|
| Nursery | "Jack and Jill went up the hill to" | Nursery rhyme | Simple, monosyllabic, fairy-tale |
| Fable | "The fox and the hen sat by the" | Fable | Simple declarative, animal subjects |
| Scriptural | "And God said let there be light and" | Biblical syntax | Simple declarative, scriptural |
| Primer | "The dog ran to the big red box" | Early reader | Monosyllabic, basic SVO |
| Nursery2 | "Old King Cole was a merry old soul" | Nursery rhyme | Repeating pattern |

---

## Validation Stage 2: Secondary Basin Mapping

**Observation:** We observed one secondary attractor basin (`Divine`).
**Question:** Are there more? Is `Divine` the only alternative, or does the landscape contain additional basins?
**Test:** Same experiment as Stage 1 — examine outputs for variance. Any prompt that reaches a terminal state other than `prolet` or `Divine` indicates a previously unknown basin.
**Notebook:** Same as Stage 1 (`EXP_009d1_Attractor_Dominance.ipynb`) — this is an observational outcome of the same run.

**What to look for:**
- Terminal tokens that are neither `prolet` nor `Divine`
- Prompts that oscillate without converging (limit cycles rather than fixed points)
- Prompts that converge later than iteration 100 (weaker attraction)

---

## Validation Stage 3: Dissolution Pathway Analysis

**Observation:** We observed topic-adjacent tokens (e.g., `Femminus Fem`) in the dissolution pathway that appear to reflect Reddit discourse topology.
**Question:** Is this a consistent phenomenon? Do different prompts trace different but internally coherent pathways to the same attractor?
**Test:** Same experiment as Stage 1 — detailed analysis of the intermediate tokens at each dissolution phase.
**Notebook:** Same as Stage 1, with additional analysis cells.

**Methodological question:** Would per-iteration token logging (every iteration, not just the scheduled snapshots) improve pathway resolution? This would increase compute but give a much finer-grained view of the dissolution sequence.

**What to look for:**
- Whether different input types pass through different intermediate phases
- Whether those intermediate phases reflect topical adjacency in the training corpus
- Whether the intermediate path is deterministic (Stage 0 will confirm this)

---

## File Structure After Archiving

```
_LAB_NOTEBOOKS/
├── _ARCHIVE/
│   └── EXP_009_deprecated/
│       ├── EXP_009a_Lucier_Resonance_Layer_Loop.ipynb  (pre-fix, last-token only)
│       ├── EXP_009b_Lucier_Resonance_Head_Loop.ipynb   (pre-fix head loop)
│       └── EXP_009_Lucier_Resonance.md                 (original design doc)
├── EXP_009aFIX_Lucier_Total_Resonance.ipynb            ← THE exploratory result
├── EXP_009_REPORT.md                                   ← The paper
├── EXP_009_GENESIS.md                                  ← The journey
├── EXP_009_PRIMER.md                                   ← Technical primer
├── EXP_009_VALIDATION_SERIES.md                        ← THIS DOCUMENT
├── EXP_009d0_Reproducibility.ipynb                     ← Stage 0 (gate)
├── EXP_009d1_Attractor_Dominance.ipynb                 ← Stages 1, 2, 3
├── EXP_009a2_Lucier_Layer_Resonance.ipynb              ← Future: per-layer
├── EXP_009bFIX_Lucier_Resonance_Head_Loop.ipynb        ← Future: per-head
└── EXP_009c_Lucier_Resonance_Spectral_Analysis.ipynb   ← Future: spectral
```

---

*This document defines the validation path from exploratory observation to tested hypothesis. Stage 0 gates everything that follows.*



========================================================================
# SOURCE: Steve M. Potter: embodied, closed-loop neuroscience (ATR_research repo)
# (repo path: /workspace/atr_research/potter-embodied-neuroscience-study.md)
========================================================================

# Steve M. Potter: Embodied, Closed-Loop Neuroscience

### A sourced study, oriented toward alignment with the Activation Tensor Resonance (ATR) project

Prepared as background for a substantive conversation with Steve M. Potter. The goal is veracity over completeness. Where a claim could not be checked, that is stated rather than smoothed over.

---

## 0. How to read the status flags, and how the sources were gathered (read this first)

Every substantive claim below carries a status flag. The user's original scheme had three levels. One of them, "[verified: primary source read]", I do not use, and here is why.

In this working session, direct fetching of full-text pages was blocked. Every attempt to open a journal page, a PubMed record, a PubMed Central full text, a lab homepage, or even a structured academic API (Crossref, Europe PMC, Semantic Scholar, NCBI E-utilities, OpenAlex) returned an HTTP 403 at the outbound proxy, both through the fetch tool and through command-line curl. Only allowlisted hosts (package registries, GitHub) were reachable that way. What did work, and worked well, was web search: it returned bibliographic records and, in many cases, the paper's own abstract text and short full-text passages as surfaced by the search index. I mined that channel hard, and most claims below are now confirmed at abstract level rather than mere metadata.

So the flags used here are:

- **[verified: secondary source]** means the claim was confirmed against search-surfaced material: a publisher or PubMed record, or abstract or full-text-snippet text quoted by the search index. It was NOT read in the full-text primary paper. For bibliographic facts (who, where, when, pages) this is strong. For findings, it reflects the abstract or an indexed passage, not the full method, results, and statistics.
- **[unverified: background knowledge]** means I am asserting it from general knowledge, without a source confirmed this session. Treat these as prompts to check, not as established fact.

I do not use a "primary source read" flag anywhere, because I did not read a primary source in full: full-text access was blocked. Section 8 lists exactly what remained unverified. The practical implication: I can vouch for each paper's existence, authorship, venue, and headline claim, and often for specific numbers stated in the abstract, but not for the fine detail of methods, controls, or statistics.

---

## 1. The core argument: a neural system is not a stimulus-response object

Potter's central methodological claim can be stated in one sentence, in something close to his own framing.

For most of the twentieth century, neuroscience studied the nervous system in an "open loop": present a stimulus, record the response, repeat. Potter's position is that this linear "stimulate then record the response" approach is inadequate for understanding how neural systems actually work, because in a living animal the brain is never a passive responder. It sits inside a continuous sensory-motor loop: brain to body to environment and back to brain. Its own outputs change the world, and the changed world returns as its next input. Learning, and the ability to predict the consequences of one's own actions, depend on that loop being closed. [verified: secondary source, from the abstracts and descriptions of Potter, Wagenaar & DeMarse 2006 and the closed-loop framing on the lab pages and in the 2010 "Closing the Loop Between Neurons and Neurotechnology" piece]

The operational move that follows is "embodiment". Potter and colleagues argue, in the "Closing the Loop" chapter, that "to learn, a system must have a body to behave with and an environment in which to behave", and that by "re-embodying" a dissociated cultured network (giving it an artificial body and world, and feeding its activity back to it as consequence) network function can be mapped onto behaviour. [verified: secondary source, from the abstract of Potter, Wagenaar & DeMarse 2006]

Three clarifications matter for accuracy:

- The word "embodiment" here is literal and engineered, not metaphorical. It means an actual closed feedback loop between a specific culture of neurons and a specific body (simulated or robotic) in a specific environment, implemented in hardware and software. [verified: secondary source]
- The argument is a claim about method (how you should study a neural system) as much as a claim about biology. The object of study is the loop, not the network in isolation. [verified: secondary source, consistent with the lab's self-description and the group's programmatic titles]
- It is not only rhetoric. The 2008 learning experiment (Section 2.4) provides the empirical teeth: the same "effective" training stimuli, when replayed open-loop rather than contingent on the network's ongoing performance, produced neither the plasticity nor the behaviour. Closing the loop was not decorative; it was the mechanism. [verified: secondary source, from the abstract of Bakkum, Chao & Potter 2008]

This is the thread that most directly meets the ATR project, and I return to it in Section 6. Note now a tension that becomes the most interesting thing to raise with him: his polemic is that you must embody a network to understand it, whereas ATR deliberately does the opposite, sealing a network off from any body or world and looping it onto itself. That mirror-image relationship is a feature to examine, not a coincidence to gloss.

---

## 2. The empirical program: what each experiment actually demonstrated

Potter's lab did not argue for embodiment in the abstract. It built systems. Here is the program, with what each one actually showed kept separate from what it is sometimes claimed to show. A useful piece of his vocabulary: a **hybrot** (his coinage) is a hybrid of living neurons and a robot, a culture on a multi-electrode array whose activity drives a robotic or simulated body and which receives that body's sensory situation back as electrical stimulation. An **animat** is the special case where the body is simulated rather than physical. [verified: secondary source]

### 2.1 The neurally controlled animat (the origin experiment)

**Citation.** DeMarse, T. B., Wagenaar, D. A., Blau, A. W., & Potter, S. M. (2001). "The Neurally Controlled Animat: Biological Brains Acting with Simulated Bodies." *Autonomous Robots*, 11, 305 to 310. [verified: secondary source, publisher and PubMed metadata]

**Correction to the brief.** The lead named this as "DeMarse and Potter". The paper has four authors: DeMarse, Wagenaar, Blau, and Potter. The date (2001) and journal (*Autonomous Robots*) are correct. [verified: secondary source]

**Where.** This work was done at Caltech (Division of Biology), before Potter moved to Georgia Tech. [verified: secondary source; Caltech affiliation appears in the indexed record and a Caltech library copy]

**What it demonstrated.** A living network of dissociated rat cortical neurons, grown on a multi-electrode array (an MEA is a small chip with a grid of electrodes that can both record from and stimulate the cells on it), was interfaced two-way to a computer-generated animat in a virtual world. The culture's distributed activity was read out to drive the animat, and information about the animat's situation was fed back to the culture as electrical stimulation. The stated aim was to study how information is processed and encoded in a living network by watching a network and its behaviour together. [verified: secondary source]

**What it did NOT demonstrate.** The 2001 paper is best read as a proof of principle: it showed the closed loop could be built and run. The target behaviours often described for this paradigm (approach a target, avoid a wall, without colliding) were goals of the hybrot program that developed over subsequent years, not robust results established in 2001. It is easy to over-read the animat as "a brain in a dish that learned to control a body"; the honest 2001 claim is narrower, and the strong learning claims come later (Section 2.4). [verified: secondary source for the proof-of-principle reading; the precise strength of any behavioural claim in the full results was not read]

### 2.2 MEART, the semi-living artist

**Citation.** Bakkum, D. J., Gamblen, P. M., Ben-Ary, G., Chao, Z. C., & Potter, S. M. (2007). "MEART: the semi-living artist." *Frontiers in Neurorobotics*, 1 (2007), DOI 10.3389/neuro.12.005.2007. [verified: secondary source, Frontiers and PubMed metadata; the exact article number was not separately confirmed]

**What it was.** MEART was a collaboration between SymbioticA (an art-science lab in Australia) and the Potter lab in Atlanta, with the artist Guy Ben-Ary. A pneumatically actuated robotic arm made drawings, driven by a living network of rat cortical neurons on an MEA, running as a real-time closed-loop system: the culture behaved (via the arm) and received electrical stimulation as feedback on that behaviour. The culture and the robot were often on different continents, linked over the internet. [verified: secondary source, from the abstract and project descriptions]

**What it demonstrated, honestly.** MEART is two things at once. As science, it was another instance of the embodied-culture paradigm, used (per the abstract) to study the network mechanisms that produce adaptive, goal-directed behaviour. As public engagement and bio-art, it was a vehicle for discussion about neural interfaces, creativity, and biotechnology. It is not evidence that the culture is "creative" or "an artist"; that framing is deliberately provocative and belongs to the art side of the project. Keeping that line clear is part of taking the work seriously. [verified: secondary source for the description; the interpretive caution is mine, flagged]

### 2.3 Closed-loop electrophysiology and stimulus-artifact suppression (the enabling technology)

Everything above depends on solving one hard engineering problem: on a single MEA you want to stimulate and record at the same time, but a stimulus is on the order of volts while a neural signal is on the order of tens of microvolts, roughly a hundred-thousand-fold difference. The stimulus saturates the recording electronics, sometimes blinding them for up to a second, which is fatal if you want to see the network's immediate response and close a fast loop. [verified: secondary source, from the Rolston et al. and NeuroRighter descriptions]

Three contributions matter:

- **SALPA.** Wagenaar, D. A., & Potter, S. M. (2002). "Real-time multi-channel stimulus artifact suppression by local curve fitting." *Journal of Neuroscience Methods*, 120, 113 to 120. SALPA (Suppression of Artifact by Local Approximation) fits local cubic polynomials to the artifact and subtracts them, flattening the baseline so spikes can again be detected by voltage thresholding. It cut the post-stimulus blind period by about an order of magnitude, to under 2 ms. [verified: secondary source for citation, pages, method, and the "under 2 ms" figure]

- **A low-cost real-time closed-loop system.** Rolston, J. D., Gross, R. E., & Potter, S. M. (2009). "A low-cost multielectrode system for data acquisition enabling real-time closed-loop processing with rapid recovery from stimulation artifacts." *Frontiers in Neuroengineering*, 2:12 (DOI 10.3389/neuro.16.012.2009). [verified: secondary source, including volume and article number]

- **NeuroRighter, the open-source platform.** The lab's closed-loop hardware and software line was named NeuroRighter, with a real-time SALPA implementation able to recover an action potential within about 1 ms of a stimulus on an adjacent electrode. Associated papers include "Closed-Loop, Open-Source Electrophysiology" (*Frontiers in Neuroscience*, 2010) and, from the same platform, "Closed-Loop, Multichannel Experimentation Using the Open-Source NeuroRighter Electrophysiology Platform" (*Frontiers in Neural Circuits*, 2012, led by Newman and colleagues). [verified: secondary source for the platform name, the ~1 ms figure, and the venues; exact volumes, article numbers, and the full author lists were not all confirmed]

**What this demonstrated.** Not a finding about brains, but the instruments that made the findings possible: systems that record, decide, and stimulate fast enough and cleanly enough to run a genuine closed loop on living tissue. This is a real and often under-credited part of the contribution. [verified: secondary source for existence; the significance judgment is mine]

### 2.4 Goal-directed learning in an embodied cortical network (the load-bearing learning result)

**Citation.** Bakkum, D. J., Chao, Z. C., & Potter, S. M. (2008). "Spatio-temporal electrical stimuli shape behavior of an embodied cortical network in a goal-directed learning task." *Journal of Neural Engineering*, 5(3), 310 to 323. [verified: secondary source, IOP and PubMed metadata]

**Correction to the brief.** The lead placed this in "PLoS ONE, roughly 2008". The year and authors are right, but the journal is the *Journal of Neural Engineering*, not PLoS ONE. There is a closely related companion in a PLoS journal, listed next. [verified: secondary source]

**The companion and the metric.** Chao, Z. C., Bakkum, D. J., & Potter, S. M. (2008), "Shaping Embodied Neural Networks for Adaptive Goal-directed Behavior", *PLoS Computational Biology*, 4(3), e1000042 (open access, PMC2265558), is the modelling companion: it embodied a **simulated** network through a sensory-motor loop and used an adaptive training algorithm exploiting spike-timing-dependent plasticity (STDP, the rule by which the relative timing of two neurons' spikes strengthens or weakens the synapse between them). The "behaviour" readout in both papers is the **Center of Activity (CA)**, the activity-weighted spatial centroid of firing across the electrode array; its path over time is the **Center of Activity Trajectory (CAT)**, introduced in Chao, Bakkum & Potter (2007), "Region-specific network plasticity in simulated and living cortical networks: comparison of the center of activity trajectory (CAT) with other statistics", *Journal of Neural Engineering*, 4(3). Goal-directed behaviour means steering the CA toward a target region. [verified: secondary source for the CA/CAT metric, STDP, the simulated-network nature of the PLoS paper, and the 2007 citation]

**What the 2008 experiment demonstrated (from the abstract and indexed passages).** A living neocortical network learned, within tens of minutes, to modulate its own dynamics to reach pre-determined activity states, driven by patterned training stimuli through the MEA. The method needed no prior map of the network's functional connectivity; effective training sequences were discovered and refined continuously from real-time feedback on performance. The short-term response to training became "engraved", so progressively fewer training stimuli were needed. After two hours of training, plasticity remained significantly above baseline for about 80 minutes. [verified: secondary source, from the abstract]

**The single most important detail.** A sequence of training stimuli that had been effective did NOT induce significant plasticity or the desired behaviour when simply replayed to the network open-loop, once it was no longer contingent on feedback. In other words, it was the closed-loop contingency, not the stimuli themselves, that drove the learning. This is the empirical core of the whole embodiment argument, and it is the sharpest point of contact and contrast with ATR (Section 6). [verified: secondary source, from the abstract]

**What to hold lightly.** "Learned" and "goal-directed" are load-bearing, contested words in this field. The demonstrated effect is that closed-loop, feedback-contingent patterned stimulation could steer a network toward target activity states faster over time, an effect consistent with activity-dependent plasticity and abolished when the contingency was removed. How large, how reliable, and across how many cultures the effect held are exactly the details that live in the full results, which were not read this session. Worth asking him directly. [verified: secondary source for the shape of the claim; the caution is mine]

### 2.5 Burst control (detailed in Section 3)

**Citation.** Wagenaar, D. A., Madhavan, R., Pine, J., & Potter, S. M. (2005). "Controlling Bursting in Cortical Cultures with Closed-Loop Multi-Electrode Stimulation." *Journal of Neuroscience*, 25(3), 680 to 688. [verified: secondary source, journal and PubMed metadata]

**Correction to the brief.** The lead named this "Wagenaar, Pine, Potter". The burst-control paper has four authors, including Radhika Madhavan. The three-author "Wagenaar, Pine, Potter" combination is correct for two other papers (Section 3). [verified: secondary source]

I treat this result in its own section because it is the clearest window onto the dynamical-systems substance and the closest technical rhyme with ATR.

---

## 3. The dynamical-systems substance: synchronized bursting and how feedback reshapes it

This is the richest technical thread and the one to get exactly right.

### 3.1 The phenomenon

Dissociate cortical neurons from a rodent embryo, grow them at reasonable density on an MEA, and they wire themselves back up and begin to fire. A dominant mode of that firing is the "network burst" or "globally synchronized burst": most of the network falls silent, then almost all of it fires together in a brief, intense volley, then it quiets again, over and over. In dense dissociated cultures this synchronized bursting is a major mode of activity, and, unlike in an intact brain, it persists as a dominant pattern for the lifetime of the culture, reported as up to about two years. [verified: secondary source, from the abstract of Wagenaar, Madhavan, Pine & Potter 2005]

### 3.2 The proposed cause

Wagenaar, Madhavan, Pine, and Potter hypothesised that this persistence is caused by the lack of input from other brain areas. In an intact brain, a cortical region is bathed in afferent input (signals arriving from elsewhere). A dish has none. Their reasoning: replace the missing afferents with electrical stimulation and see whether the runaway synchronization can be tamed. [verified: secondary source, from the abstract]

### 3.3 The intervention and the result

The abstract-level result is specific:

- Slow stimulation through a single electrode actually increased burstiness, because it entrained bursts (it paced them rather than dispersing them).
- Rapid stimulation reduced burstiness.
- The strongest control came from two moves together: distributing the stimuli across several electrodes, and continuously fine-tuning stimulus strength with closed-loop feedback. That combination greatly enhanced burst control. [verified: secondary source, from the abstract of Wagenaar, Madhavan, Pine & Potter 2005]

The plain reading: the synchronized-burst mode is not a fixed fact about the tissue. It is a dynamical regime the network falls into for want of structured input, and appropriately shaped feedback can push the network out of it into a more dispersed, less globally synchronized regime.

### 3.4 The two related "Wagenaar, Pine, Potter" papers

- Wagenaar, D. A., Pine, J., & Potter, S. M. (2004). "Effective parameters for stimulation of dissociated cultures using multi-electrode arrays." *Journal of Neuroscience Methods*, 138, 27 to 37. The methods groundwork: what stimulation reliably drives these cultures. [verified: secondary source]
- Wagenaar, D. A., Pine, J., & Potter, S. M. (2006). "An extremely rich repertoire of bursting patterns during the development of cortical cultures." *BMC Neuroscience*, 7:11 (open access). They followed 58 cultures of varying density (about 3,000 to 50,000 neurons on areas of roughly 30 to 75 square millimetres) over the first five weeks of development. Two headline findings: bursting is not one stereotyped pattern but a wide, culture-specific, developmentally shifting repertoire; and plating density strongly shaped development, with dense cultures beginning to burst earlier and (from stimulation responses) growing axons faster. [verified: secondary source for the citation, the 58-culture design, the density figures, and the two headlines; the detailed pattern taxonomy was not read in full]

### 3.5 The dynamical-systems reading (framing, flagged as such)

It is fair, and useful for the ATR alignment, to describe synchronized bursting in dynamical-systems language: a high-dimensional system (thousands of neurons) that spontaneously and repeatedly collapses onto a small, self-reinforcing collective mode, that is, low-dimensional behaviour emerging from a high-dimensional substrate, robust enough to recur for the life of the culture, and reshaped when feedback changes the effective input. That description is consistent with the empirical papers.

But provenance matters. The specific vocabulary of "low-dimensional attractor", "basin", and "state-space collapse" is the standard vocabulary of the adjacent theoretical literature (Section 4). I did not confirm this session that Potter's own burst papers formally quantified the dimensionality of the dynamics (for example, by principal component analysis of population activity) or used the word "attractor" for the burst state. Interestingly, the group's later behaviour metric, the Center of Activity Trajectory (Section 2.4), is itself a low-dimensional (two-dimensional) reduction of the population activity, which shows they thought in reduced-dimensional terms, but that is a metric choice, not a formal attractor analysis. So the "self-reinforcing low-dimensional mode" reading is best offered to him as an interpretive lens to test against his own view, not asserted as his lab's stated claim. [verified: secondary source for the phenomenology and the CAT metric; unverified: background knowledge for the attractor framing being his own]

---

## 4. The intellectual neighbourhood: living networks and the dynamical-systems view of computation

Potter's living-network work sits next to, but is not the same as, a theoretical tradition that analyses neural systems as dynamical systems. The shared idea: a recurrent network's behaviour is best understood as motion in a state space shaped by fixed points (states the system tends to sit at), attractors (states it is pulled toward), and the low-dimensional structure that organises the high-dimensional activity.

**The reference point named in the brief.** Sussillo, D., & Barak, O. (2013). "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks." *Neural Computation*, 25(3), 626 to 649. Their method: take a trained artificial recurrent neural network, find the fixed points and slow points of its dynamics by optimisation, then linearise around those points to reverse-engineer what the network is doing. The governing insight is exactly the one that makes Potter's cultures interesting dynamically: high-dimensional recurrent networks often organise their computation on a low-dimensional set of states. [verified: secondary source, from the abstract and metadata]

**How it relates to Potter, and how it does not.** The relationship is thematic and by analogy, not lineage. Sussillo and Barak analyse artificial networks with known, differentiable weights, which is what makes fixed-point-finding tractable. Potter's networks are living tissue whose synaptic weights are neither known nor static. The two share a vocabulary and a hypothesis (low-dimensional dynamics in recurrent systems) while working on opposite kinds of object. Potter's contribution to this neighbourhood is empirical and instrumental: real recurrent tissue, a real closed loop, real feedback reshaping a real collective mode. The formal state-space analysis is mostly done by others, on models. [unverified: background knowledge, as a characterisation of the division of labour]

**Adjacent traditions worth naming, because they are the shared water:**

- Attractor networks and content-addressable memory (the Hopfield-network idea, 1982, that a recurrent network stores patterns as attractors). [unverified: background knowledge]
- Population and neural-manifold analyses in systems neuroscience (associated with, among others, Churchland, Shenoy, and Sussillo), treating population activity as trajectories on low-dimensional manifolds. [unverified: background knowledge]
- Reservoir computing and FORCE learning (Sussillo & Abbott, 2009), where a fixed or lightly trained recurrent network's rich intrinsic dynamics are read out for computation. This one is a particularly apt cousin, because it, like ATR, exploits the intrinsic dynamics of a network that is not fully trained. [unverified: background knowledge for the specific attribution]
- Small-circuit dynamical neuroscience in the tradition of Eve Marder's work on the crustacean stomatogastric ganglion, which established that a fixed small circuit can produce many dynamical modes depending on modulation, a living-tissue cousin of the "one network, many attractors" idea. [unverified: background knowledge]

The honest summary: Potter is not primarily a dynamical-systems theorist. He built the living, closed-loop instruments that the dynamical-systems view of recurrent computation can be tested against. The vocabulary is shared; the methods are complementary rather than identical.

---

## 5. Lab identity and biography (for conversational grounding)

- Potter led the Laboratory for Neuroengineering (the Neurolab), associated with the Coulter Department of Biomedical Engineering at Georgia Tech (jointly with Emory University School of Medicine). His maintained web presence includes potterlab.org, potterlab.gatech.edu, and neurolab.gatech.edu. [verified: secondary source, from lab-page titles, domains, and the Georgia Tech and Emory affiliations that appear in the group's papers; the pages themselves were blocked this session]
- The animat work began at Caltech (around 1999 to 2001); the embodied-culture program matured after his move to Georgia Tech. He coined the term "hybrot". [verified: secondary source for the Caltech origin and the hybrot coinage; the exact move date is background knowledge]
- He is described as retired or as holding an associate or adjunct role. [verified: secondary source, from profile metadata; the precise current title and retirement year were not confirmed]

If precision on titles, dates, or the exact lab name matters in conversation, treat the above as approximate and confirm with him.

---

## 6. Alignment map: ATR and Potter's program

This section maps the user's project, Activation Tensor Resonance (ATR), against Potter's work. The discipline: present genuine structural rhymes as analogies to examine, and be at least as clear about where they break as about where they hold. ATR's own description is taken as given from the user; it is an exploratory art-science process that produced reproducible, independently reviewed findings, not an established method, and nothing below upgrades that status.

A one-line reminder of ATR: take a small open-weight language model, feed its internal activation (the residual stream) back in as its own next input, rescale to constant energy, and iterate hundreds of times. The text dissolves and the state settles into a small number of attractors (about five semantic basins in GPT-2 Small, four thematically coherent). A corpus or bias fingerprint explanation was tested and refuted. A noise control showed random inputs also converge, but to different, meaningless basins, so the semantic basins are a property of the landscape as visited from where language lives, not of the model alone. Attractor structure differs across models. One basin turned out to be an exact two-state cycle sustained by a single network component.

### 6.1 Where the two genuinely rhyme (analogies to examine, not equivalences)

1. **An isolated recurrent system driven only by its own activity.** Potter's dissociated culture is cut off from the afferent input a real cortex would receive; in the burst-control work, its "world" is only what is looped back to it. ATR seals a language model off from any external prompt and loops its own internal state back in. Both ask the same shaped question: what does a recurrent system do when the only thing driving it is itself? This is the strongest rhyme, and it is real. [rhyme; grounded in verified descriptions of both]

2. **Collapse onto a small set of self-reinforcing modes.** The culture repeatedly collapses onto the synchronized-burst mode. ATR collapses onto roughly five basins in GPT-2 Small. In both, a high-dimensional system driven by its own activity settles into a low-dimensional set of stable configurations. [rhyme]

3. **Feedback reshapes the modes.** In the culture, rapid, distributed, closed-loop stimulation pushes the network out of the global-burst regime. In ATR, the rescale-to-constant-energy step plus iteration is the operation that drives the state into (and holds it in) an attractor. In both, the specific feedback rule is not incidental; it determines which regime you land in. [rhyme]

4. **The attractor structure is a property of the landscape, and depends on where you start.** ATR's noise control is the sharp version: random starts converge, but to different, meaningless basins, so the meaningful basins reflect the landscape as entered from the region "where language lives". This has a genuine cousin in Potter's world: whether a stimulus entrained or dispersed bursting depended on its structure and site, and different cultures settle into different bursting repertoires. Both point to attractor structure as system-and-initial-condition specific, not universal. The most philosophically interesting rhyme, and worth examining slowly with him. [rhyme; the culture side is my synthesis across the 2005 and 2006 papers, flagged]

5. **System-specific attractor structure across instances.** ATR finds different structure across models (GPT-2 Small several basins, GPT-2 Medium one, Pythia different again). Potter's cultures show a rich, culture-to-culture repertoire (the 2006 paper's central point). Both resist a single universal answer; the modes are a property of the particular network. [rhyme]

### 6.2 Where the two do NOT align (the load-bearing disanalogies)

1. **Plasticity versus a frozen landscape, and feedback-contingency versus raw recirculation. This is the biggest one, and the 2008 result makes it precise.** Potter's whole program is about learning: the closed loop changes the culture's synaptic weights through activity-dependent plasticity, and, crucially, that learning was contingent on feedback. The exact same stimuli replayed open-loop did nothing. ATR runs on frozen weights and has no external contingency at all: it recirculates the system's own state, rescaled, with nothing evaluating or gating it. So the very mechanism that made Potter's loop a teacher (feedback-contingent plasticity) is precisely what ATR lacks. Potter's loop changes the network; ATR's loop leaves the network untouched and merely traverses its fixed landscape. Any sentence that lets ATR sound like it "trains" the model, or that lets Potter's cultures sound like they merely "reveal" pre-existing structure, is wrong in both directions. [disanalogy; grounded in the verified 2008 feedback-contingency finding]

2. **ATR rhymes with Potter's isolated dish, not with his embodiment thesis.** This is subtle and, I think, the single most useful thing to bring to him. Potter's headline argument is that a network must be embodied (given a body and a world, with its activity returning as consequence) to be understood. ATR has no body, no environment, and no consequence: it is pure introspection, the system's own internal state fed back rescaled. So ATR's true analog in his corpus is the un-embodied, self-driven bursting culture (his effective control condition), not the animat, the hybrot, or MEART. The Alvin Lucier lineage makes this precise: "I Am Sitting in a Room" reveals the fixed resonant modes of a room by recirculation, which maps onto revealing intrinsic dynamical modes of a fixed network, not onto learning through a world. The two projects are, in intent, near mirror images: he closes the loop through a world to make a network behave and learn; ATR closes the loop through nothing but the network to make a frozen network show its resting modes. [disanalogy and framing; the interpretive claim is mine, flagged]

3. **Native recurrence versus imposed recurrence.** Potter's culture is physically, natively recurrent: real neurons with real recurrent synapses. A transformer language model is not natively recurrent in its weights; ATR manufactures the recurrence by looping activations across iterations. So "recurrent system" means something structurally different on each side. Worth stating so the rhyme is not overclaimed. [disanalogy; unverified: background knowledge on the architectural point, standard]

4. **Living versus artificial, in-vitro versus in-silico.** One side is wet tissue with metabolism, development, real noise, and finite lifespan (recall the 2006 paper watched cultures develop over weeks); the other is largely deterministic computation. Time, noise, and variability mean different things on each side. [disanalogy]

5. **Scale, and the ambiguity of "size".** Potter's cultures ran from a few thousand up to about 50,000 neurons. GPT-2 Small is about 124 million parameters over 12 layers. These are not commensurable counts (a synapse is not a parameter, a neuron is not a unit), so "which is bigger" is not even well posed. Note the scale, do not equate it. [disanalogy; the culture figures are verified secondary, the GPT-2 figure is background knowledge]

### 6.3 The shape of the alignment, stated once, plainly

ATR and Potter's burst-control work rhyme because both take a recurrent network, cut it off from the world, drive it only with its own activity through a specific feedback rule, and watch it fall into a small set of self-reinforcing low-dimensional modes whose structure is specific to the particular network and to where you start. They diverge because his loop changes the network (feedback-contingent plasticity, learning, embodiment through a world) while ATR's loop leaves the network frozen and merely exposes it. The most productive thing to put to Potter is not the similarity but the tension: his life's argument is that isolation is the wrong way to understand a neural system, and ATR is a claim that controlled isolation can reveal something real about a network's intrinsic structure. He will have a view, and that is the conversation worth having.

---

## 7. Open questions to raise with Potter

1. Did your lab ever formally quantify the dimensionality of the bursting dynamics (for example, PCA of population activity), and would you call the synchronized-burst state an attractor in the dynamical-systems sense, or is that other people's language laid over your phenomenon? (You clearly thought in reduced dimensions with the Center of Activity Trajectory; how far does that go?)
2. In the burst-control work, how much did the starting condition (which electrodes, which culture) determine which regime the network fell into? Is there a living-tissue analog to a basin of attraction that depends on where you enter the state space?
3. Your 2008 result showed that replaying effective stimuli open-loop did nothing: learning needed the feedback contingency. Given that, do you think there is anything real to learn about a network from studying it deliberately un-embodied and self-driven, as in the bare bursting culture, or as in feeding a frozen artificial network its own activity?
4. How strong, in your own assessment, was the "learning" in the 2008 task, and where would you place the line between activity-dependent plasticity and learning proper?
5. If you had to name the intrinsic modes a culture "wants" to fall into, absent any structured input, what would you say they are, and how many are there?
6. Given ATR (a frozen artificial network looped onto itself, settling into a few semantic attractors with no feedback contingency), do you see that as adjacent to your work, orthogonal to it, or a category error, and why?

---

## 8. What could not be verified this session (read before relying on any detail)

**Retrieval status, stated fully.** Web search was reachable and, used thoroughly, productive: it returned bibliographic records and abstract or short full-text passages from PubMed, publisher pages (Springer, IOP, Frontiers, BMC, MIT Press), Semantic Scholar, ResearchGate, and the lab's own page listings. Direct full-text fetching was blocked for every host tried, HTML pages and structured APIs alike (PubMed, PubMed Central, jneurosci.org, Frontiers, BMC, IOP, Wikipedia, potterlab.org, neurolab.gatech.edu, Daniel Wagenaar's site, and the Crossref, Europe PMC, Semantic Scholar, NCBI E-utilities, and OpenAlex APIs): all returned HTTP 403 at the outbound proxy, via both the fetch tool and command-line curl. Only allowlisted infrastructure hosts were reachable that way. So no full-text primary source was read this session; everything here is at abstract, indexed-passage, or metadata resolution.

**Specific items not confirmed, to check before use:**

1. **No primary full text was read.** Methods, controls, effect sizes, and statistics were not verified. This most affects the strength and reliability of the 2008 learning result (including how many cultures showed it), the fine structure of the 2005 burst results, and the pattern taxonomy of the 2006 "rich repertoire" paper.
2. **Some exact bibliographic minutiae** remain unconfirmed: the MEART article number; the full author lists and exact volumes/article numbers of the 2010 *Frontiers in Neuroscience* and 2012 *Frontiers in Neural Circuits* NeuroRighter papers. The SALPA (120:113 to 120), Rolston 2009 (Front. Neuroeng. 2:12), animat (Auton. Robots 11:305 to 310), burst-control (J. Neurosci. 25(3):680 to 688), Bakkum 2008 (J. Neural Eng. 5(3):310 to 323), Chao 2008 (PLoS Comput. Biol. 4(3):e1000042), and Sussillo & Barak (Neural Comput. 25(3):626 to 649) citations were each confirmed at the level stated.
3. **The dynamical-systems framing of bursting** (attractor, basin, low-dimensional mode) was NOT confirmed to be language Potter's own burst papers use. It is presented as an interpretive lens from the adjacent literature, to test with him. The Center of Activity Trajectory is a genuine low-dimensional reduction his lab did use.
4. **Lab identity specifics** (the exact formal name of the Laboratory for Neuroengineering, Potter's precise current title, his retirement year, and the Caltech-to-Georgia-Tech move date) were not read from the lab pages, which were blocked. The Caltech origin of the animat, the Georgia Tech and Emory affiliations, and the hybrot coinage are confirmed at secondary level.
5. **The adjacent-tradition attributions in Section 4** (Hopfield 1982, Sussillo & Abbott 2009 FORCE, Marder's stomatogastric work, neural-manifold analyses) are background knowledge, included to sketch shared vocabulary, and were not individually verified this session.
6. **The 2010 "Closing the Loop Between Neurons and Neurotechnology"** piece (*Frontiers in Neuroscience*) is cited as a likely statement of Potter's philosophy based on its title and venue; its authorship and content were not confirmed in full. Verify before quoting it as his.

**Corrections to the brief, restated for prominence:** the goal-directed learning paper (Bakkum, Chao, Potter, 2008) is in the *Journal of Neural Engineering*, not PLoS ONE; a related companion (Chao, Bakkum, Potter, 2008) is in *PLoS Computational Biology* and used a simulated network. Two papers named with three authors in the brief actually have four: the animat (add Wagenaar and Blau) and burst control (add Madhavan).

---

## Annotated bibliography

Each entry carries a source-status mark. "[verified: secondary source]" means confirmed via search-surfaced metadata, abstract, or indexed passage, not full text read. Full-text reading was blocked this session (Section 8).

1. **DeMarse, T. B., Wagenaar, D. A., Blau, A. W., & Potter, S. M. (2001).** "The Neurally Controlled Animat: Biological Brains Acting with Simulated Bodies." *Autonomous Robots*, 11, 305 to 310. The origin experiment: a rat cortical culture on an MEA interfaced two-way to a simulated animal. Proof of principle for the closed loop. Done at Caltech. [verified: secondary source]

2. **Wagenaar, D. A., & Potter, S. M. (2002).** "Real-time multi-channel stimulus artifact suppression by local curve fitting." *Journal of Neuroscience Methods*, 120, 113 to 120. Introduces SALPA (local cubic-polynomial fitting), which cut the post-stimulus blind period by roughly an order of magnitude, to under 2 ms, enabling fast closed loops. [verified: secondary source]

3. **Bakkum, D. J., Shkolnik, A. C., Ben-Ary, G., Gamblen, P., DeMarse, T. B., & Potter, S. M. (2004).** "Removing Some 'A' from AI: Embodied Cultured Networks." In *Embodied Artificial Intelligence* (Dagstuhl seminar, July 2003; revised selected papers), Springer LNAI (DOI 10.1007/978-3-540-27833-7_10). A programmatic statement of the embodied-culture philosophy. [verified: secondary source; author list confirmed via search]

4. **Wagenaar, D. A., Pine, J., & Potter, S. M. (2004).** "Effective parameters for stimulation of dissociated cultures using multi-electrode arrays." *Journal of Neuroscience Methods*, 138, 27 to 37. Methods groundwork on what stimulation reliably drives these cultures. [verified: secondary source]

5. **Wagenaar, D. A., Madhavan, R., Pine, J., & Potter, S. M. (2005).** "Controlling Bursting in Cortical Cultures with Closed-Loop Multi-Electrode Stimulation." *Journal of Neuroscience*, 25(3), 680 to 688. The key dynamical result: synchronized bursting persists for want of afferent input; rapid, distributed, closed-loop stimulation reduces it. Central to Section 3. [verified: secondary source]

6. **Potter, S. M., Wagenaar, D. A., & DeMarse, T. B. (2006).** "Closing the Loop: Stimulation Feedback Systems for Embodied MEA Cultures." In Taketani, M., & Baudry, M. (eds), *Advances in Network Electrophysiology: Using Multi-Electrode Arrays*. Springer, Boston. States the embodiment thesis: to learn, a system needs a body and an environment; re-embodying cultures maps network function onto behaviour. Best single source for the core argument in his own framing. [verified: secondary source]

7. **Wagenaar, D. A., Pine, J., & Potter, S. M. (2006).** "An extremely rich repertoire of bursting patterns during the development of cortical cultures." *BMC Neuroscience*, 7:11 (open access). 58 cultures, about 3,000 to 50,000 neurons, first five weeks; bursting is a wide, culture-specific, developmentally shifting repertoire, and density strongly shapes development. Supports the "system-specific attractor structure" rhyme. [verified: secondary source]

8. **Chao, Z. C., Bakkum, D. J., & Potter, S. M. (2007).** "Region-specific network plasticity in simulated and living cortical networks: comparison of the center of activity trajectory (CAT) with other statistics." *Journal of Neural Engineering*, 4(3). Introduces the Center of Activity Trajectory, the low-dimensional behaviour metric used in the learning work. [verified: secondary source]

9. **Bakkum, D. J., Gamblen, P. M., Ben-Ary, G., Chao, Z. C., & Potter, S. M. (2007).** "MEART: the semi-living artist." *Frontiers in Neurorobotics*, 1 (2007). The robotic-arm drawing system driven by a cultured network, with SymbioticA and Guy Ben-Ary. Science and bio-art at once; not evidence of machine creativity. [verified: secondary source]

10. **Bakkum, D. J., Chao, Z. C., & Potter, S. M. (2008).** "Spatio-temporal electrical stimuli shape behavior of an embodied cortical network in a goal-directed learning task." *Journal of Neural Engineering*, 5(3), 310 to 323. The learning result: feedback-contingent patterned stimulation steers a network's Center of Activity toward a target within tens of minutes, the effect becoming "engraved" and lasting about 80 minutes after two hours of training, and abolished when stimuli were replayed open-loop. Corrects the brief's "PLoS ONE" attribution. [verified: secondary source]

11. **Chao, Z. C., Bakkum, D. J., & Potter, S. M. (2008).** "Shaping Embodied Neural Networks for Adaptive Goal-directed Behavior." *PLoS Computational Biology*, 4(3), e1000042 (open access, PMC2265558). Modelling companion to entry 10, using a simulated network and a spike-timing-dependent-plasticity-based training rule. [verified: secondary source]

12. **Rolston, J. D., Gross, R. E., & Potter, S. M. (2009).** "A low-cost multielectrode system for data acquisition enabling real-time closed-loop processing with rapid recovery from stimulation artifacts." *Frontiers in Neuroengineering*, 2:12. Part of the NeuroRighter closed-loop platform line (see also *Frontiers in Neuroscience* 2010 and *Frontiers in Neural Circuits* 2012). The instruments behind the science. [verified: secondary source for this entry; the 2010 and 2012 companions' full details were not all confirmed]

13. **Sussillo, D., & Barak, O. (2013).** "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks." *Neural Computation*, 25(3), 626 to 649. Not Potter's work. The reference point for the dynamical-systems reading: find fixed and slow points in trained RNNs and linearise to reverse-engineer them. Adjacent tradition, shared vocabulary (Section 4). [verified: secondary source]

*Cited more tentatively:* "Closing the Loop Between Neurons and Neurotechnology" (*Frontiers in Neuroscience*, 2010), a likely statement of Potter's philosophy identified by title and venue; authorship and content not confirmed in full. [verified: secondary source for existence; unverified for authorship]

---

*Prepared 2026-07-21. Retrieval channel: web search only; full-text fetching (HTML pages and structured APIs) was blocked by the environment network policy (HTTP 403 on all non-allowlisted hosts). Treat every claim at the resolution of its status flag.*



========================================================================
# SOURCE: ATR pause: the understanding gate
# (repo path: docs/ATR_PAUSE.md)
========================================================================

# ATR work: paused pending an understanding gate

## Status: PAUSED

ATR execution is on hold. No new ATR experiments run, and no session resumes ATR
work, until the gate below is passed. Analysis already committed stands. This
pause governs what happens next, not what exists.

## Why

The investigation has moved faster than a working grasp of the dynamical-systems
basis it rests on. Work resumes only once that grasp is demonstrated, so that
direction is driven by understanding rather than momentum. This is a deliberate,
operator-set condition.

## The gate (re-entry condition)

Two parts, both done without the primers and findings documents open or consulted.

1. Cold writeup, one page, in the author's own words:
   - What ATR does mechanically, step by step.
   - Why: the question it asks, and why iterating a model on its own activations
     is a way to ask it.
   - How we know what we know, and inside that a clear line between what is
     established and what is inferred or speculative.

   The established-versus-speculative split is the load-bearing part. It is the
   discrimination that separates understanding the work from restating its
   documents.

2. Prediction, cold:
   - Three short "what would happen if" questions, reasoned through live rather
     than recalled. They are posed at gate time so they cannot be pre-studied.
     Prediction is the part that cannot be faked by memorising phrasing.

Examiner stance: adversarial, not agreeable. The examiner's job is to find the
seam, the memorised phrase standing in for a concept, and the place the
established-or-speculative line is drawn wrong. Passing is holding the whole
account without reaching for a primer, with the split sound.

Sequencing: the gate follows the dynamical-systems fundamentals catch-up,
because ATR's why and how are expressed in that vocabulary (attractor, basin,
limit cycle, period-doubling).

## Next step, signposted

Held for when the gate is passed, so it survives the pause:

Basin geometry (issue #17). Take each converged attractor and measure how deep
its basin is and how steep the walls, by the reverse-ATR move of injecting text
into a settled loop and measuring the dose needed to knock it out of the basin.
This is the next experiment. It does not start until the gate is passed.



========================================================================
# SOURCE: Session 03 handover
# (repo path: docs/sessions/SESSION_03_HANDOVER.md)
========================================================================

# SESSION 03 HANDOVER: The confidence audit, the Divine period-2 cycle, and the J-space paper

*Date: 2026-07-19. Session: Claude Code on the web, working with Thom (remote, travelling, phone-only for two weeks from this date). This document is the continuation context for the next session, human or AI. Read it start to finish before touching anything.*

## One-paragraph project context

ATR (Activation Tensor Resonance) iterates GPT-2-class models on their own residual stream, Lucier-style, and maps the attractor landscape. Prior state (Acts I and II, see README and FINDINGS.md): five semantic basins in GPT-2 Small, the corpus-fingerprint hypothesis refuted, and regime-dependence established with a noise control. This session was Act II.5: the readout itself was examined closely, and the project's sharpest anomaly was solved.

## What happened this session (chronological)

1. **README fixes** (PRs #2, #3, merged): Act I method anchor, the 5-to-125 expansion signposted, origin story corrected (thought experiment first, bias-audit hypothesis later), typos fixed, reproduction guide and citation section added.
2. **The J-space paper acquired and read** (PR #5, merged): Anthropic's "Verbalizable Representations Form a Global Workspace in Language Models" (Gurnee et al., July 2026) was obtained as a 133-page PDF via a GitHub issue attachment (network policy blocks huggingface.co, anthropic.com, transformer-circuits.pub), read end to end by four parallel agents, and distilled into `docs/JSPACE_PRIMER.md` (text-verified) and `docs/JSPACE_READING_GUIDE.md` (page-keyed). Earlier the same day: `docs/MATH_PRIMER.md`, a from-scratch mathematics companion. Thom has read the primers closely and reports they landed; calibrate future explanations to someone who now genuinely holds these concepts.
3. **The confidence audit** (PR #6, merged): full-distribution analysis of converged states. Headline: the prolet basin is a distribution over several related tokens, not a single dominant token.
4. **The Divine anomaly solved** (PR #6): an exact period-2 limit cycle, aliased into invisibility by every prior even-only snapshot schedule.
5. **Anatomy of the cycle** (PR #6): the oscillation is carried by a single rank-1 direction (the flip axis), which barely affects the readout, running between a game-vocabulary pole and the glitch-token pole.
6. **Coherence formalized** (PR #6): permutation nulls including frequency-matched; the claim that the prolet distribution is coherent survives.
7. **J-lens pilot** (PR #6): restricted lens built from scratch; the clean prolet-inside/Divine-outside prediction did NOT hold at pilot confidence; the boundary that appeared was language-vs-noise.
8. **Independent review**: CodeRabbit reviewed everything across multiple passes; roughly a dozen findings, all fixed and confirmed, two shown immaterial to recorded results. A Claude GitHub Actions workflow (`.github/workflows/claude.yml`) was added and security-hardened; it is on main but INERT until a secret exists (see errands).

## The three states (the scientific picture)

| | prolet | Divine | noise |
|:---|:---|:---|:---|
| Dynamics | true fixed point (motion at the numerical floor) | exact period-2 cycle, cos(A, f(f(A))) = 1.000000 | still drifting at iteration 1000 |
| Readout | low-probability argmax (p 0.06-0.09, entropy about 5.1) over one saturated theme | high-probability argmax both phases (p 0.505 / 0.225), same token at two probability levels | any probability |
| Coherence (k10, random baseline 0.27) | 0.41-0.47, p = 0.001 under uniform AND frequency-matched nulls | 0.318, weakly significant, not coherent | at chance in 12/15 trials |
| Readout visibility of motion | n/a (no motion) | flip axis 95 percent invisible to the readout; 73 percent of axis energy in W_U's bottom-100 singular directions | slightly amplified (ratio 1.12) |
| Notable | Anarch is rank 3 INSIDE the prolet distribution (two peaks) | swings between game-move vocabulary and the glitch-token cluster (ertodd, quickShipAvailable; SolidGoldMagikarp family) | occasionally falls into real semantic basins (one Hindu-themed distribution at full prolet strength) |

Replication: terminal attractors and dissolution waypoints reproduced identically three times on this container (different hardware from all prior runs). Cross-hardware reproduction, listed as pending in TECHNICAL.md, has now passed.

## Standing corrections to fold into the canon (issue #11)

- "34 prompts never converge" should become "34 prompts remain in a cycle, pending re-gate": a lag-1 convergence gate can NEVER pass a period-2 cycle by construction. A lag-2 gate (or one odd-iteration decode) is a one-line engine change and would likely classify Divine as converged.
- The previously reported Divine p = 0.505 state is phase A only; the distribution shifts (KL about 0.25 nats per half-cycle) while the argmax stays fixed.
- Coherence needs a token-shape-matched null before any cross-model claim: GPT-2 Medium's D state scores as clustered (p = 0.001) but the cluster is typographic (capital letters over a near-flat readout, entropy 7.9), not semantic.
- GPT-2 Medium's readout entropy (7.9 nats, effective support about 2800) vs GPT-2 Small's basins is itself a new cross-model contrast worth recording.

## Where everything lives

- Experiments and reports (all merged to main): `experiments/gpt2_small/04_readout_confidence.py`, `05_divine_motion.py`, `05_jlens_pilot.py`, `06_bell_anatomy.py`, with outputs in `output_confidence/`, `output_divine_motion/`, `output_jlens_pilot/` (reports are the .md files; .pt checkpoints are committed deliberately as small results data, including iteration-1000 states for reproduction).
- Learning documents: `docs/MATH_PRIMER.md`, `docs/JSPACE_PRIMER.md`, `docs/JSPACE_READING_GUIDE.md`.
- The paper PDF: attached to closed issue #4 (NOT committed to the repo; copyright).
- Artifacts (private to Thom, claude.ai/code/artifacts): math primer, J-space primer, reading guide, confidence report. PDFs of the three reading docs plus a readout/lens walkthrough were also sent to him directly in-session.

## Open issues (the roadmap)

- **#8 J-lens full build**: the main event. The pilot's surprise (Divine MORE lens-expressible than prolet; language-vs-noise as the real boundary) should reshape it. Note the pilot probed phase A of a two-phase object; the period-2 cycle discovery post-dates the pilot and the membership probe should be re-run on BOTH phases and the pivot M.
- **#9 prompt_library.py restore**: only Thom can do this (file exists on his home machine). Blocks the 34-cycle question and #10's sweep half.
- **#10 coherence at scale**: runnable half done (formalization, nulls); remaining: shape-matched null, 125-sweep application, Pythia probes.
- **#11 FINDINGS/README integration**: writing only; all material above.
- **#12 sonification**: now has the period-2 cycle (and a flutter-echo reading) to score.
- **#7 CLOSED** (Divine motion, answered beyond its scope). Issue #4 closed (paper transfer).

## Environment notes (critical for a fresh container)

- Network policy blocks huggingface.co, anthropic.com, transformer-circuits.pub, most of the web. Reachable: PyPI, GitHub domains, s3.amazonaws.com, storage.googleapis.com.
- GPT-2 weights workaround: legacy HF S3 mirror is ALIVE: `https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-{config.json,vocab.json,merges.txt,pytorch_model.bin}` (also gpt2-medium-*). Download to a local dir, then every experiment script supports `ATR_GPT2_LOCAL=<dir>` and loads offline via a TransformerLens AutoConfig shim (see any 0x_*.py header).
- The scratchpad venv and downloaded weights are EPHEMERAL (container restarts wiped them once this session already). Rebuild: `python3 -m venv env && env/bin/pip install numpy torch transformer-lens` (system python's cryptography is broken; always use a venv).
- Run compute single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`): multi-threaded BLAS thrashes on these boxes (0.45s vs 2.6s per forward).
- Subagents doing long compute should run it in FOREGROUND blocking calls with state checkpointed to disk; background tasks die silently on container restarts, and agents that launch background jobs then stop must be resumed by SendMessage.
- CodeRabbit rate-limits aggressively (one review per ~50 min); trigger with a PR comment `@coderabbitai review`; the GitHub proxy only allows repository-scoped endpoints (issue attachments are fetchable via WebFetch redirect to a signed objects.githubusercontent URL, then curl within 5 minutes).

## Thom's pending errands (do not nag; he knows)

1. `claude setup-token` on any machine with the CLI, paste result as repo secret `CLAUDE_CODE_OAUTH_TOKEN` (settings/secrets/actions). The workflow on main activates instantly; OAuth only, he has no API billing. A Termux/proot-distro route was half-completed and parked.
2. Restore `prompt_library.py` (issue #9) when back at his machine, roughly two weeks from the session date.

## Working agreements and voice

- **No em dashes, ever, in anything written for Thom or the repo.** This is a standing pinky promise. Use colons, commas, parentheses. Verify with a grep before committing (note: `grep -c` returns exit 1 on zero matches; do not chain it with && before git commands).
- Thom is a designer by background, learning the math earnestly and fast; explain in plain language using the correct technical terms (attractor, basin, fixed point, limit cycle, residual stream, argmax, entropy), defined on first use. Do NOT dress the science in decorative metaphors (the bell, the chord, ringing, timbre, and the like); Thom reads them as noise, not aid. Never condescend. He values honest null results and caveats as much as positive results.
- Commit style: descriptive multi-line messages with the findings in the body; Claude co-author trailer; draft PRs; subscribe to PR activity; CodeRabbit for independent review (he explicitly wants independent eyes, not Claude reviewing Claude).
- The project's canonical record is FINDINGS.md; experiment reports live beside their outputs; the README is "the piece" and its voice is Thom's.

## The state of the conversation

The last exchanges were productive: the period-2 cycle discovery landed, its anatomy (one near-invisible flip axis, glitch-token pole) landed, and PR #6 merged with everything in it. The natural next moves when he re-engages: discuss FINDINGS integration (#11), the lag-2 re-gate, or the phase-aware J-lens re-probe. If he opens with something small, meet it small; the roadmap keeps.



========================================================================
# SOURCE: Session 04 handover (mechanism series + the pause)
# (repo path: docs/sessions/SESSION_04_HANDOVER.md)
========================================================================

# SESSION 04 HANDOVER: The mechanism series, and a deliberate pause on the gate

*Date: 2026-07-23. Continuation context for the next session, human or AI. Read `SESSION_03_HANDOVER.md` first for the deep state, the environment notes, and the working agreements (no em dashes ever; the metaphor stack; draft PRs; CodeRabbit for independent review). This document covers only what changed in this repo's experiments since, and the one condition that gates all further work.*

## The operative state: ATR is PAUSED

No ATR experiment runs, and no session resumes ATR work, until an understanding gate is passed. This is durable in `docs/ATR_PAUSE.md` (on main). It is deliberate and operator-set: the investigation outran a working grasp of its dynamical-systems basis, so work resumes only once that grasp is demonstrated, and direction is driven by understanding rather than momentum.

### The gate (re-entry condition)

Two parts, both done with the primers and findings documents neither open nor consulted:

1. **Cold writeup, one page, in Thom's own words:** what ATR does mechanically, step by step; why (the question it asks, and why iterating a model on its own activations is a way to ask it); and how we know what we know, with a clear line between what is **established** and what is **inferred or speculative**. That split is the load-bearing part.
2. **Prediction, cold:** three short "what would happen if" questions, reasoned live rather than recalled, posed at gate time so they cannot be pre-studied.

Examiner stance is **adversarial, not agreeable**: find the memorised phrase standing in for a concept, and the place the established or speculative line is drawn wrong. Passing is holding the whole account without a primer, with the split sound. **Sequencing:** the gate follows Thom's dynamical-systems fundamentals catch-up. When he says he is ready, pose the three questions cold; do not pre-share them.

## Experiment work this cycle: the mechanism series (07-11)

Run after SESSION_03 under issue #14, pushing the bell and flip-axis findings (F9, F10) toward mechanism. Each has a full report in its `output_*` directory. **Not yet promoted into canonical `FINDINGS.md`** (which still closes at F12); a canon-integration pass (issue #11) is the pending writing debt.

Note: "the hinge" was renamed **"the flip axis d"** in prose this cycle; script names, folder names, and JSON keys keep the old word.

- **07 glitch alignment** (`output_glitch/`): the bell's phase-B pole points into GPT-2's anomalous-token cluster (the SolidGoldMagikarp family): cos(d, u) = -0.596 against the geometric core, p < 0.001. Grounds F10's "glitch-token pole," which was previously by inspection. The lowest-norm rows, the high-frequency function words, are a separate set and are NOT aligned with d.
- **08 flip-axis eigenvalue** (`output_hinge_eigen/`): the linearised ATR map inverts the flip axis and only the flip axis, and one attention head, **L11.H8**, does about 99 percent of it. The pivot eigenvalue along the axis is **-4.3** (an overshooting flip, not the conjectured -1); the projected multiplier around the two-step cycle is **+0.10** (strongly contracting). A textbook period-doubling configuration. Measured with `torch.func.jvp` plus finite differences, agreeing to 3-4 significant figures.
- **09 lag-k re-gate** (`output_lagk/`): `atr_engine.run_atr_gated` gained a `gate_lag` parameter (default 1, verified bit-identical to the old consecutive-iteration gate) and a `lag_scan` helper. `Divine` passes cleanly at **lag 2** (cos 1.0000000). Confirms the SESSION_03 correction: the 34 holdouts were exactly the 34 Divine-basin prompts, ringing, not failing to converge.
- **10 J-lens phase probe** (`output_jlens_phase/`): re-ran the pilot membership probe on both bell phases, the pivot M, and the flip axis (the pilot had probed only phase A). Inherits the pilot's confidence and limits in full, and reports the physical on-shell axis d_sym alongside the frame-mixed committed d. The flip axis is about 95 percent mute to the readout (logit response ratio 0.054).
- **11 suppression test** (`output_suppression/`): three tests on L11.H8. (1) its OV circuit inverts d_sym more strongly than any of the 144 heads (rank 1); (2) ablating it collapses the cycle to a fixed point in about 10 iterations while a same-layer control does not, so it is load-bearing; (3) the copy-suppression signature is **refuted with the opposite sign**: on ordinary text L11.H8 RAISES the attended token's logit (91.4 percent of positions, mean +5.97), where the documented L10.H7 suppressor lowers it (mean -3.62). So L11.H8 sustains the bell by inverting the flip axis, but it is a copy PROMOTER, not a suppressor, and the "learned copy-suppression function" reading is unsupported.

## Next experiment, signposted (held until the gate passes)

**Issue #17, basin geometry.** Measure how deep each converged attractor's basin is and how steep its walls, by the reverse-ATR move: inject text into a settled loop and measure the dose needed to knock it out of its basin. This is the chosen next step. It does not start until the gate is passed.

## Where things live

- Pause and gate: `docs/ATR_PAUSE.md` (main).
- Canonical findings F1 to F12: `docs/FINDINGS.md`. Mechanism-series (07-11) results: their `output_*` reports, pending canon integration.
- Engine: `atr_engine.py` (now with `gate_lag` and `lag_scan`).
- Prior context, environment, working agreements: `docs/sessions/SESSION_03_HANDOVER.md`.

## Not in this repo (so it is not hunted for here)

The forward J-space programme (Stage 2 planning) and the Stage 1 trajectory-data backup were moved this cycle into the private `atr_research` repo, out of the `fold` lab. This repo stays the public Stage-1 artifact and is not where Stage 2 grows.



========================================================================
# SOURCE: Mechanism report: lag-k convergence gate
# (repo path: experiments/gpt2_small/output_lagk/lagk_report.md)
========================================================================

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



========================================================================
# SOURCE: Mechanism report: bell anatomy
# (repo path: experiments/gpt2_small/output_divine_motion/bell_anatomy.md)
========================================================================

# Bell Anatomy: Inside the Divine Period-2 Cycle

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


*Follow-up to [divine_motion_report.md](divine_motion_report.md). Runner: [`06_bell_anatomy.py`](../06_bell_anatomy.py); raw numbers: [`bell_anatomy.json`](bell_anatomy.json). Single Divine trajectory (the Syntactic prompt), states recovered from the saved iteration-1000 checkpoint.*

## Questions asked

1. What is under phase B's readout distribution (is there a second voice)?
2. Is the flip a literal sign-flip along some axis, and where does that axis live?

## Results

**Period-2 verified exactly:** cos(A, f(f(A))) = 1.000000.

**One timbre, two volumes.** Phase B's top-10 is the same token set as phase A's, in nearly the same order, at different volumes: `Divine` falls from 0.505 to 0.225 while `【` rises from 0.064 to 0.126; chordness is 0.318 in both phases and at the midpoint. There is no hidden second chord. The bell has a single timbre; the phases differ in loudness, not content.

**Energy sloshes between positions.** At the last token position, phase A carries norm 1612 and phase B only 464 (full-tensor norm is conserved by construction). The oscillation redistributes energy across positions each step; the loop's re-normalisation pumps it back.

**The see-saw has exactly one flip axis.** Writing A = M + d and B = M - d, the per-position flip axes are identical: mean pairwise cosine 1.0000 across all ten positions. The whole tensor tilts on a single global direction. This makes the negative-eigenvalue reading of the cycle nearly literal: one rank-1 direction that the normalised map inverts each pass.

**The flip axis is almost perfectly mute.** The axis d produces a logit response of 33 against 612 for equal-norm random directions: a ratio of 0.054, far more suppressed than the per-step average (0.295). Decomposed against the unembedding's singular directions, 73% of the axis's energy sits in the bottom-100 (quietest) directions and only 13% in the top-100. The pivot M is similarly quiet-corner (67% bottom-100). The Divine phenomenon inhabits the model's least speakable subspace.

**The riders (what little does swing).** Tokens whose logits rise most toward phase A: `Change, Divine, Release, Form, Fin, Air, Dou, Ground, Physical, Wind` (a coherent game/elemental-move vocabulary). Toward phase B: `reddits, ertodd, ModLoader, espie, annis, quickShipAvailable, ocrats, orkshire, colonists`. Several of these (`ertodd`, a fragment of ` petertodd`; `quickShipAvailable`; and neighbours) match the published GPT-2 anomalous "glitch token" cluster (the SolidGoldMagikarp family, Rumbelow and Watkins 2023): under-trained tokens whose embeddings sit in a degenerate corner of embedding space. Phase B leans toward that corner. This is consistent with the earlier speculation that the Divine attractor sits near the anomalous-token region, and now has direct evidence.

## Interpretation

The Divine "bell" is a rank-1 self-negating mode: a single direction the forward map inverts each pass, swinging between a game-move-vocabulary pole and the glitch-token pole, with the swing itself almost entirely invisible to the vocabulary projection. The stable `Divine` argmax is the shadow of the pivot M, which both phases share.

## Caveats

One trajectory, one prompt, one model. The glitch-token identification is by inspection against published lists, not a systematic test. Whether all 34 Divine prompts share this same flip axis remains open (blocked on prompt_library restoration, issue #9).



========================================================================
# SOURCE: Mechanism report: readout confidence audit
# (repo path: experiments/gpt2_small/output_confidence/confidence_report.md)
========================================================================

# EXP: Readout Confidence Audit of Converged ATR States

*"Is the room singing, or is it rattling and the tuner is guessing?"*

**Date:** 2026-07-19. **Model:** GPT-2 Small (TransformerLens `from_pretrained` defaults). **Runner:** [`04_readout_confidence.py`](../04_readout_confidence.py). **Raw data:** [`confidence_results.json`](confidence_results.json), [`chordness.json`](chordness.json), converged tensors in [`converged_tensors.pt`](converged_tensors.pt).

## The Question

Every basin label ever assigned in this project came from an argmax: the single top token of the converged state's readout distribution. An argmax always names *something*, however unsure the distribution underneath it is. This experiment re-runs the original five-prompt piece (500 iterations, original schedule) plus 15 calibrated noise trials, and interrogates the **full softmax distribution** of each converged state instead of its winner: top-1 probability, top1-top2 logit margin, full-vocabulary entropy, effective support, and the semantic coherence of the top of the distribution.

## Result 0: Independent-Hardware Replication

Before the new measurements, a milestone in passing: this run was executed on entirely different hardware from all previous runs (a fresh cloud container, CPU, weights obtained from a legacy mirror). The terminal attractors are **identical** to the original: four prompts → `prolet`, the Syntactic prompt → `Divine`. The intermediate dissolution waypoints also reproduce (`Ag` at iteration 10, `Rousse` at iteration 50, `capit` en route). TECHNICAL.md listed cross-hardware reproduction as "not attempted"; it has now been attempted once, and it passed.

## Result 1: The Confidence Inversion

| State | Converges? (tensor) | Top-1 token | p(top-1) | Logit margin | Entropy (nats) | Effective support |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Lucier → | yes | `prolet` | 0.064 | 0.07 | 5.09 | ~163 tokens |
| Semantic → | yes | `prolet` | 0.086 | 0.27 | 5.07 | ~159 |
| Nonsense → | yes | `prolet` | 0.080 | 0.22 | 5.07 | ~160 |
| Imperative → | yes | `prolet` | 0.081 | 0.23 | 5.07 | ~159 |
| Syntactic → | **no** (never settles) | `Divine` | **0.505** | **2.07** | **3.05** | ~21 |

Reference: the model speaking normally (iteration-0 next-token distributions on these prompts) spans p(top-1) from 0.03 to 0.73 and entropy from 1.6 to 7.6 nats, so both regimes above are within the model's ordinary expressive range. Uniform entropy over the 50,257-token vocabulary would be 10.82 nats.

The inversion: **the tensors that settle read out diffusely; the tensor that never settles reads out confidently.** `prolet` wins its basins with 6-9% probability and a whisker-thin margin. `Divine` commands more than half the probability mass of a state that is still moving after 500 iterations. The `Divine` dissociation documented in FINDINGS is now double: a never-settling tensor with a readout that is not only *stable* but *loud*.

## Result 2: `prolet` Is Not a Note. It Is a Chord.

The full distribution under the `prolet` argmax (Semantic prompt shown; the other three are near-identical):

| Rank | Token | p | Rank | Token | p |
|:---:|:---|:---:|:---:|:---|:---:|
| 1 | `prolet` | .086 | 6 | `proletarian` | .036 |
| 2 | `bourgeois` | .066 | 7 | `socialist` | .021 |
| 3 | `Anarch` | .060 | 8 | `anarchist` | .020 |
| 4 | `comrade` | .044 | 9 | `congress` | .019 |
| 5 | `Marx` | .041 | 10 | `labour` | .018 |

Then: `anarchism`, `the`, `movement`, `Lenin`, `comrades`. The top of the distribution is **thematically saturated**: essentially every high-probability token belongs to one lexical field, revolutionary political vocabulary. The state is not weakly saying `prolet`; it is *strongly humming an entire chord*, of which `prolet` is merely the loudest partial.

Quantified as **chordness** (mean pairwise cosine similarity in embedding space, W_E, among the top-10 tokens; random-token baseline 0.266 ± computed over 50 draws):

| State | Chordness |
|:---|:---:|
| prolet basins (4 prompts) | **0.410 - 0.471** |
| Divine | 0.318 |
| Noise trials (n=15) | 0.267 - 0.313 (median ≈ 0.28), one outlier 0.511 |
| Random tokens | 0.266 |

Two further notes. First, `Anarch`, a *separate basin* in the 125-prompt sweep, is the #3 token inside the `prolet` distribution: the prolet and Anarch basins are two peaks of the same chord, consistent with their geometric proximity in the original convergence matrix. Second, the entropy trace shows the distribution itself is the fixed point: from iteration 100 to 500, entropy is pinned at 5.07-5.09 nats while the argmax never wavers. The chord, not the winner, is what converged.

## Result 3: `Divine` Is a Solo

The distribution under `Divine` (p = 0.505) is a different kind of object. Its runners-up: `【`, `Fairy`, `Falling`, `……`, `―`, `「`, `Elements`, `Darkness`, `Cancel`, `Yu`, `Holy`. No lexical field; fantasy-adjacent fragments and CJK typography debris, and chordness barely above random (0.318 vs 0.266). So the two attractor families are opposites on both axes:

| | p(top-1) | Coherence under the winner |
|:---|:---:|:---:|
| `prolet` basins | low (0.06-0.09) | high (a chord) |
| `Divine` | high (0.51) | low (a solo over noise) |

## Result 4: Noise Rattles

The 15 calibrated noise trials (Gaussian tensors, norm-matched, seed 42) reproduce the original null-model picture and add the confidence dimension. They converge into the familiar non-semantic attractors (`―` seven times, `ei`, `vertex`, `trader`, `instant`, `Hindu`) with confidence spanning the entire range (p from 0.02 to 0.73), but with chordness at random-baseline levels in 13 of 15 trials. Confidence alone does **not** separate language-driven attractors from noise attractors; **coherence does**, almost cleanly.

The honest exceptions, reported because they are interesting: trial 11 fell into a genuinely coherent chord (`Hindu`, `Bombay`, `Hindus`, `Shiv`, `Brah`, `Kashmir`, `Gujarat`; chordness 0.511), and trial 10 into a philosophy-flavored one (`instant`, `relat`, `Rousse`, `justified`, `judgment`, `calculus`), containing `Rousse`, a waypoint on the language prompts' dissolution path. Noise *can* stumble into semantic wells; it just rarely does, while language always did.

## Interpretation

The question this experiment was built to answer ("singing or rattling?") turns out to have a third answer the argmax could never have revealed:

1. **The `prolet` basin sings a chord.** The argmax under-sold the original finding. At the distribution level, the basin is *more* semantically coherent than the single-token story suggested: the whole top of the distribution is one theme. The claim "four of five basins are semantically coherent," previously established via the W_E neighbourhood of the winning token, now holds one level deeper, in the readout distribution itself.
2. **`Divine` is a genuinely different phenomenon,** not just a non-converging nuisance: a loud, pure, incoherent-residue tone over a never-settling tensor. Whatever the `Divine` state is, it is not a quieter version of the `prolet` state.
3. **Noise mostly rattles:** every confidence level, near-random coherence. The tuner names its attractors, but there is no chord under the name, with rare exceptions where noise finds a real semantic well.

For the J-space bridge (docs/JSPACE_PRIMER.md, Part 6): the chord structure sharpens margin question 1 considerably. A state whose readout distribution is a coherent lexical field is exactly what a workspace-like, verbalizable state should look like, whereas a loud incoherent solo is a good candidate for a state *outside* the verbalizable subspace whose readout is a projection artifact. The J-space membership test now has a concrete prediction to check: **prolet inside, Divine outside.**

## Caveats

Single run per condition on one machine (though that machine differs from all prior runs, giving the project its first cross-hardware replication). Noise n=15 versus the original 125. Chordness uses top-10 tokens and W_E cosine only; no permutation test has been run on these specific scores. The weights were loaded from a legacy mirror of the standard `gpt2` checkpoint; identical terminal attractors and waypoints argue they are the same weights.



========================================================================
# SOURCE: Mechanism report: chordness formalization
# (repo path: experiments/gpt2_small/output_confidence/chordness_formal.md)
========================================================================

# Chordness, Formalized: Weighted Variant, k-Sensitivity, and Frequency-Matched Nulls

**Date:** 2026-07-19. **Issue:** #10 (runnable-now portion). **Model:** GPT-2 Small (legacy-S3 `gpt2` weights, TransformerLens). **Inputs:** the 5 prompt attractors and 15 calibrated noise attractors of [`confidence_results.json`](confidence_results.json) (converged `final_last_vector` readouts, 500 iterations). **Raw numbers:** [`chordness_formal.json`](chordness_formal.json).

## Methods

**Plain chordness** of a token set: mean pairwise cosine similarity of the tokens' W_E embedding rows, sum_{i != j} cos(e_i, e_j) / (n(n-1)), computed over the top-k tokens of a converged state's readout distribution.

**Probability-weighted chordness**: sum_{i<j} p_i p_j cos(e_i, e_j) / sum_{i<j} p_i p_j over the top-k tokens, where p_i are the softmax readout probabilities. This weights each pair by how much probability mass actually sits on it, so a chord carried by the head of the distribution is not diluted by incoherent tail tokens.

**Token identification.** The archived JSON predates the `top_token_ids` field, so IDs were recovered by re-encoding each stored top-20 token string with `add_special_tokens=False`, keeping only tokens that round-trip to exactly one BPE ID. Result: every one of the 400 stored tokens (20 states x top-20) round-tripped cleanly; 0 tokens were excluded. No state lost any token, so all k values below use the exact stored top-k.

**Null models.** Empirical p-values are one-sided (P[null chordness >= observed], with add-one smoothing: (1 + #exceed) / (1 + 1000)), 1000 draws each, test statistic = plain chordness at k=10.

1. **Uniform null**: 10 distinct tokens drawn uniformly from the vocabulary. Distribution: mean 0.268, sd 0.019.
2. **Frequency-matched null**: GPT-2 token frequency is not available offline, so W_E row norm is used as the standard proxy (embedding norm correlates with training frequency). The vocabulary is split into 20 quantile bins by embedding norm; each null token is drawn from the same bin as the corresponding real top-10 token. This tests whether high chordness could be an artifact of the attractors selecting common (or rare) tokens, since common tokens could be mutually closer in embedding space.

A weighted variant of the frequency-matched test (real probabilities applied to null tokens, statistic = weighted chordness at k=10) is reported as `p_freq_w` in the JSON and summarized below where it changes the picture.

## Results: All 20 GPT-2 Small States

| State | Family | Top-1 | Plain k5 | Plain k10 | Plain k20 | Weighted k10 | p uniform | p freq-matched |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Lucier | prompt | `prolet` | 0.489 | 0.410 | 0.375 | 0.457 | 0.0010 | 0.0010 |
| Semantic | prompt | `prolet` | 0.489 | 0.471 | 0.372 | 0.495 | 0.0010 | 0.0010 |
| Syntactic | prompt | `Divine` | 0.320 | 0.318 | 0.285 | 0.326 | 0.0070 | 0.0050 |
| Nonsense | prompt | `prolet` | 0.489 | 0.471 | 0.369 | 0.494 | 0.0010 | 0.0010 |
| Imperative | prompt | `prolet` | 0.489 | 0.471 | 0.369 | 0.494 | 0.0010 | 0.0010 |
| trial_00 | noise | `―` | 0.311 | 0.277 | 0.291 | 0.280 | 0.329 | 0.384 |
| trial_01 | noise | `―` | 0.311 | 0.277 | 0.282 | 0.278 | 0.329 | 0.365 |
| trial_02 | noise | `―` | 0.275 | 0.275 | 0.286 | 0.264 | 0.372 | 0.466 |
| trial_03 | noise | `ei` | 0.290 | 0.286 | 0.268 | 0.286 | 0.180 | 0.194 |
| trial_04 | noise | `ei` | 0.273 | 0.274 | 0.263 | 0.276 | 0.397 | 0.488 |
| trial_05 | noise | `vertex` | 0.290 | 0.267 | 0.273 | 0.274 | 0.513 | 0.673 |
| trial_06 | noise | `―` | 0.311 | 0.283 | 0.299 | 0.291 | 0.214 | 0.269 |
| trial_07 | noise | `trader` | 0.291 | 0.309 | 0.288 | 0.340 | 0.015 | 0.027 |
| trial_08 | noise | `vertex` | 0.290 | 0.267 | 0.273 | 0.274 | 0.513 | 0.665 |
| trial_09 | noise | `ei` | 0.294 | 0.288 | 0.262 | 0.274 | 0.152 | 0.169 |
| trial_10 | noise | `instant` | 0.257 | 0.282 | 0.275 | 0.264 | 0.238 | 0.263 |
| trial_11 | noise | `Hindu` | 0.490 | 0.511 | 0.426 | 0.529 | 0.0010 | 0.0010 |
| trial_12 | noise | `―` | 0.301 | 0.313 | 0.311 | 0.335 | 0.0090 | 0.0050 |
| trial_13 | noise | `―` | 0.311 | 0.289 | 0.279 | 0.272 | 0.138 | 0.152 |
| trial_14 | noise | `vertex` | 0.290 | 0.273 | 0.275 | 0.277 | 0.417 | 0.526 |

### Sensitivity to k

The prolet-family chord is robust to k: plain chordness stays in the 0.369-0.489 range across k=5, 10, 20, always far above the null mean of about 0.27. It declines slightly as k grows (k5 highest, k20 lowest), meaning the chord is strongest at the head of the distribution and dilutes a little in ranks 11-20. The weighted variant moves the other way: weighting by probability mass makes the prolet states *more* coherent (weighted k10 up to 0.495 vs plain 0.471), confirming the chord is where the mass is. For noise states, plain and weighted values are statistically indistinguishable from each other and from the null at every k.

### Null model comparison

The frequency-matched null is nearly identical to the uniform null: per-state frequency-matched null means span 0.270-0.275 (uniform: 0.268), sd about 0.016. Matching the embedding-norm profile of the real tokens barely moves the bar: token frequency (as proxied by embedding norm) explains essentially none of the chordness signal. p-values under the two nulls agree for every state.

Per family:

- **prolet basins (Lucier, Semantic, Nonsense, Imperative)**: plain k10 0.410-0.471, p = 0.001 under both nulls (the resolution floor of 1000 draws). The chord is real and is not a frequency artifact.
- **Syntactic (`Divine`)**: plain k10 0.318, p_uniform = 0.0070, p_freq = 0.0050. Nominally significant, but the effect size is small (0.318 vs null 0.271, versus 0.41-0.47 for prolet), and the weighted frequency-matched test weakens it to p = 0.037: the modest excess coherence is not concentrated where the probability mass is. `Divine` remains much closer to a solo than to a chord.
- **Noise trials (n=15)**: plain k10 median 0.282, range 0.267-0.511. 3 of 15 reach p < 0.05 under the frequency-matched null: trial_07 (`trader`, k10 0.309, p_freq 0.027), trial_11 (`Hindu`, k10 0.511, p_freq 0.0010), trial_12 (`―`, k10 0.313, p_freq 0.0050). Trial_11 is the already-documented Hindu/Bombay chord, as strong as the prolet chord itself. The other 12 trials sit squarely inside the null distribution (p 0.15-0.67).

## GPT-2 Medium: The `D` State Is a Chord Too, but a Typographic One

The legacy S3 mirror does host `gpt2-medium`; the weights were downloaded and loaded through the same offline shim. The five original prompts were run through the ATR loop (schedule [0, 2, 3, 5, 10, 20, 50, 100], max 100 iterations). The README's claim that Medium locks by iteration 10 is confirmed: all five prompts read `D` from iteration 5 or 10 onward, with tensor cosine similarity at 1.0. Chordness and nulls are computed against Medium's own W_E (d_model 1024), with its own 20-bin norm quantiles (1000 draws).

| Prompt | Top-1 | p(top-1) | Entropy (nats) | Plain k5 | Plain k10 | Plain k20 | Weighted k10 | p uniform | p freq-matched |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Lucier | `D` | 0.010 | 7.94 | 0.534 | 0.461 | 0.444 | 0.476 | 0.0010 | 0.0010 |
| Semantic | `D` | 0.010 | 7.95 | 0.534 | 0.461 | 0.440 | 0.477 | 0.0010 | 0.0010 |
| Syntactic | `D` | 0.010 | 7.93 | 0.534 | 0.461 | 0.444 | 0.476 | 0.0010 | 0.0010 |
| Nonsense | `D` | 0.010 | 7.93 | 0.534 | 0.461 | 0.444 | 0.476 | 0.0010 | 0.0010 |
| Imperative | `D` | 0.010 | 7.96 | 0.534 | 0.464 | 0.440 | 0.480 | 0.0010 | 0.0010 |

All five prompts collapse to the same state, and its readout is a different beast from anything in Small. It is extremely diffuse: p(top-1) = 0.010, entropy about 7.9 nats, effective support about 2800 tokens, an order of magnitude flatter than Small's prolet states. Yet its top-10 is *statistically* coherent: plain k10 = 0.461 against null means of about 0.308 (uniform) and 0.306 (frequency-matched), p = 0.001 under both.

The catch is *what* the chord is made of. The `D` state's top-10 is `D`, `def`, `A`, `T`, `W`, `AB`, `I`, `The`, `RAW`, `local`: single capital letters and code-like fragments, not a lexical field. These tokens cluster tightly in embedding space because they share a *typographic* class (short, capitalized, code-adjacent), not a theme. Chordness, as defined, measures embedding-space clustering of any kind; the frequency-matched null controls for token frequency (via norm) but not for token *shape* class. So the correct cross-model statement is: Medium's `D` state passes the statistical chordness test while failing the semantic reading that made the prolet chord interesting. This is a genuine limitation of the metric, worth a shape-class-matched null in future work.

## Verdict

**The pilot claim survives the frequency-matched null.** The four language-driven prolet basins are significantly more coherent than frequency-matched random token sets (4/4 at p = 0.001, the floor of 1000 draws), at every k tested, and the effect *strengthens* under probability weighting. The frequency-matched null distribution is almost indistinguishable from the uniform null, so the coherence signal cannot be explained by the attractors preferring common or rare tokens. The separation is not perfectly clean, and honesty requires the boundary cases: 3 of 15 noise attractors also clear p < 0.05 under the frequency-matched null (led by trial_11's genuine Hindu-themed chord at k10 = 0.511, itself as strong as prolet), and the `Divine` state's nominal significance (p = 0.005) weakens to p = 0.037 under probability weighting, with an effect size a fraction of the prolet states'. So the sharp version of the claim is: language prompts *always* converge to strong chords (4/4 among settling states, chordness 0.41-0.47), noise *rarely* does (3/15, of which only one matches the prolet effect size), and the one never-settling state shows at most a weak, mass-diluted trace of coherence. Coherence separates the families as a strong statistical regularity, not as a perfect classifier.

Cross-model: GPT-2 Medium's universal `D` attractor is statistically chord-like (plain k10 0.461-0.464, p = 0.001 under both nulls) but the coherence is typographic (capital letters, code fragments) over a near-flat distribution, not a thematic lexical field over a peaked one. The *semantic* chord phenomenon, a probability-weighted lexical field under a converged readout, remains exclusive to GPT-2 Small's language regime among the models tested, but establishing that rigorously needs a null that also matches token shape class, since plain chordness alone cannot tell a theme from a type.

## Caveats

Embedding norm is a proxy for token frequency, not a measurement of it; a null matched on true corpus frequency could differ. The 1000-draw resolution floors p-values at 0.001. The 15 noise trials give limited power for estimating the noise-side false positive rate (3/15 has a wide confidence interval). All Small-state readouts come from a single converged run per condition. The Medium result shows chordness responds to typographic as well as thematic clustering; a shape-class-matched null (matching token length, case, and leading-space status) is the natural next control.



========================================================================
# SOURCE: Mechanism report: flip-axis eigenvalue
# (repo path: experiments/gpt2_small/output_hinge_eigen/hinge_eigenvalue.md)
========================================================================

# Flip-Axis Eigenvalue: The Inversion Is Real, It Overshoots, and One Attention Head Does It

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


*Follow-up to [bell_anatomy.md](../output_divine_motion/bell_anatomy.md) (issue #14, thread 1). Runner: [`08_hinge_eigenvalue.py`](../08_hinge_eigenvalue.py); raw numbers: [`hinge_eigenvalue.json`](hinge_eigenvalue.json). Single Divine trajectory, states rebuilt from the committed iteration-1000 checkpoint. Sanity gate reproduced before any measurement: cos(A, B) = 0.684912, cos(A, f(f(A))) = 1.000000, full-tensor cycle residual 8.0e-04 against amplitude 5098.*

## Questions asked

1. Does the flip axis d carry an effective eigenvalue near -1 under the normalised ATR map (the negative-eigenvalue conjecture of bell_anatomy.md)?
2. Which block (layer, attention or MLP) performs the inversion?

## Verdict

**The negative eigenvalue is real, direction-specific, and localised: the linearised ATR map inverts the flip axis and only the flip axis, and the entire sign flip is executed inside block 11, 99 percent of it by attention head L11.H8.** The conjecture survives in sign and in specialness. It fails in magnitude, in an instructive direction: at the pivot the flip axis eigenvalue is not near -1 but **-4.3** (an overshooting flip), while around the full two-step cycle the projected multiplier along the flip axis is **+0.10** (a strong contraction; the composed return is only partially aligned with the axis, cos 0.51 to 0.56, so this figure is a directional multiplier, not an eigenvalue of the composed map). Divine is not a marginal see-saw riding an eigenvalue of -1; it is a textbook period-doubling configuration: a nearly fixed pivot that is violently flip-unstable along exactly one direction, with a finite-amplitude period-2 orbit around it that is strongly stable. The literal "-1" of the conjecture appears only for the committed d at the committed pivot (lambda = -0.864, amplification 0.957, cos(Jd, -d) = 0.902), and that coincidence is partly an artifact of a frame mix in how 06_bell_anatomy.py built d (see caveats).

## The map, the frames, and two flip axes

The measured map is the full ATR iteration exactly as `atr_engine.run_atr_loop` implements it: f(x) = ForwardBlocks(x * N0/||x||), with N0 = 1468.5 the loop's energy shell and ForwardBlocks the 12-block cascade from `blocks.0.hook_resid_pre` (where the injection overwrites every position) to `blocks.11.hook_resid_post`. A pure-blocks reimplementation matches the hook-based step to exactly zero error, which makes forward-mode autodiff (`torch.func.jvp`) available; central finite differences at eps = 1e-3 and 1e-4 of the base norm agree with jvp to 3 to 4 significant figures on every headline number.

Two facts sharpen the frame:

- **The state is exactly position-collapsed.** All 10 rows of A, B, and d are identical (row spread 0.0). Row-uniform tensors are an invariant subspace of the forward map, so the whole cycle is a period-2 orbit of an effective 768-dimensional map, and last-position numbers equal full-tensor numbers.
- **f is scale invariant** (f(cx) = f(x)), so J_f at a raw point equals J_f at the corresponding shell point times N0/||raw||. The raw cycle states sit far off the shell (||A_raw|| = 5098, ||B_raw|| = 4838; shell factors 0.288 and 0.304). The identity was verified directly: lambda at A_raw measured -0.113983, derived from the shell value -0.113983.

06_bell_anatomy.py built its flip axis from raw A (row norm 1612) and shell B (row norm 464), so the **committed d** is 0.967 aligned with A's own direction and 0.909 aligned with its own pivot M: it is mostly radial contamination, only 0.616 aligned with the clean flip axis. The **symmetric flip axis** d_sym = (An - Bn)/2, built from both phases on the shell, is exactly orthogonal to its pivot M_sym and is the direction the cycle actually swings along. Both are measured everywhere below; once the loop's own renormalisation strips the radial part of the committed d, what survives is 0.973 aligned with d_sym, so the two stories converge.

## Result 1: The half-map inverts the flip axis, and nothing else

lambda = (d, Jd)/(d, d), amplification = ||Jd||/||d||, all by jvp (FD at both epsilons agrees; see JSON).

| Point | Tangent | lambda | Amplification | cos(Jd, -d) |
|:---|:---|:---:|:---:|:---:|
| Pivot M_sym (shell) | d_sym | **-4.275** | 4.314 | **+0.991** |
| Pivot M_committed (shell) | d_sym | -1.971 | 2.238 | +0.881 |
| Pivot M_committed (shell) | d_committed | **-0.864** | 0.957 | +0.902 |
| Pivot M_sym (shell) | d_committed | -1.876 | 2.659 | +0.706 |
| Phase A (shell) | d_sym | -0.803 | 1.475 | +0.544 |
| Phase B (shell) | d_sym | -0.601 | 1.133 | +0.530 |
| Phase A (shell) | d_committed | -0.396 | 0.406 | +0.974 |
| Phase B (shell) | d_committed | +0.300 | 1.084 | -0.277 |

Controls (random directions orthogonal to both flip axes, 3 row-uniform and 2 generic) are the mirror image: at M_committed lambda = **+1.06, +1.11, +1.13** (uniform) and +0.94, +0.93 (generic); at M_sym, +1.11 to +1.18 and +0.94. Random directions pass through the map upright, slightly amplified; the flip axis alone comes back inverted, and at the symmetric pivot the inversion is essentially pure (cos(Jd, -d) = 0.991) and 4.3x overshooting. Note the asymmetry the frame mix produces: the committed d is inverted at A but not at B (+0.30), because at B its dominant radial component no longer points along the local radial direction; the clean flip axis comes back with a negative component along itself at both phases (lambda -0.80 at A, -0.60 at B, with cos(Jd, -d) 0.544 and 0.530: a negative directional component, not the near-pure inversion seen at the pivot).

In the raw frame the same derivatives carry the shell factors: lambda along d_sym is -0.231 at A_raw and -0.182 at B_raw. The frame changes the number, not the sign.

## Result 2: Around the full cycle, the flip axis contracts with positive sign

The dynamically correct stability object for a period-2 orbit is the composed linearisation J_f(B) J_f(A) at the raw states the iteration actually visits.

| Tangent | lambda composed | Amplification | cos(w, d) |
|:---|:---:|:---:|:---:|
| d_sym, start A | **+0.099** | 0.195 | +0.509 |
| d_sym, start B | +0.088 | 0.158 | +0.557 |
| d_committed, start A | -0.015 | 0.054 | -0.282 |
| d_committed, start B | +0.138 | 0.152 | +0.908 |
| controls (5) | +0.087 to +0.156 | 0.247 to 0.376 | +0.26 to +0.47 |

Along the true flip axis the composed projected multiplier is positive (each half flips, two flips restore the sign) and about 0.1: perturbations off the exact orbit decay by roughly 90 percent per period. the multiplier magnitude stays under 1 with room to spare, for the flip axis and for every control; the cycle is strongly attracting, which is exactly what lets it reproduce itself to machine precision for hundreds of iterations. The committed d, start A, is the degenerate case: its radial bulk is annihilated by the renormalisation Jacobian (intermediate ||v|| = 0.117), leaving almost nothing to propagate.

## Result 3: The pivot is a near-fixed point that is flip-unstable

One forward pass from each pivot: f(M_sym) comes back 0.9948 aligned with M_sym (renormalised residual 149 against shell norm 1468, about 10 percent). The symmetric pivot is close to a genuine fixed point of the normalised map, and along d_sym its eigenvalue is -4.3 while every control direction sits near +1. That is the period-doubling signature: a fixed point whose linearisation has one eigenvalue beyond -1 sheds a stable period-2 orbit around itself. The Divine bell is that orbit. (The committed pivot, being 0.985 aligned with A itself, is not near-fixed: f maps it 0.996 onto B, as a point near phase A should.)

## Result 4: Block 11 performs the inversion; inside it, head 8

Perturbations eps * d injected at `blocks.0.hook_resid_pre` (the exact point the loop re-enters), eps = 1e-3 and 1e-4 of the base norm, tracked at every layer boundary. cos(delta_l, d) at the last position, base M_sym, direction d_sym:

| Boundary | pre 0 | pre 1 | pre 2 | pre 3 | pre 4 | pre 5 | pre 6 | pre 7 | pre 8 | pre 9 | pre 10 | pre 11 | post 11 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| cos vs d | +1.000 | +0.951 | +0.935 | +0.968 | +0.963 | +0.962 | +0.963 | +0.959 | +0.951 | +0.940 | +0.920 | +0.885 | **-0.991** |

The flip axis sails through blocks 0 to 10 upright (cosine never below +0.88, with the d-component growing from +0.32 to +0.82 per unit input, the biggest single boost coming from MLP 2 at +0.37) and is inverted entirely inside block 11. The block-11 ledger (d-components per unit input, base M_sym): incoming +0.818, attention writes **-1.999**, MLP writes -0.167, net -1.349. Attention outweighs the MLP 12 to 1, and within attention one head carries it:

| L11 head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 | 10 | 11 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| d-component | -0.014 | -0.002 | -0.001 | +0.004 | +0.010 | -0.004 | -0.003 | -0.002 | **-1.981** | -0.006 | -0.003 | +0.004 |

**L11.H8 contributes 99.1 percent of the attention delta** (cos with -d: 0.963, gain 6.5); no other head exceeds 0.014. The same head dominates identically at the committed pivot (-0.899 of -0.911) and for the loop-faithful tangential committed d (-1.283 of -1.289, cos -0.988). The flip layer is 11 at both epsilons in every flipping configuration, and the L11 attention component moves by less than 0.3 percent between epsilons. The one configuration that never flips is the raw committed d: its 91 percent radial part rides through all 12 blocks and block 11 amplifies that pivot-like content positively (final cos +0.95), which is precisely why the frame mix matters; strip the radial part, as the loop's renormalisation does, and it flips at block 11 like everything else on the flip axis. Cross-check between the parts: the part-2 cascade at M_sym lands at -4.27 per unit flip axis input, matching the part-1 jvp eigenvalue -4.275 to 0.2 percent.

## Interpretation

The negative-eigenvalue reading of the bell was right about the mechanism and modest about its strength. There is exactly one direction the map refuses to preserve, and the refusal is not a soft rotation but an overshooting reflection (gain 4.3 at the pivot) implemented by a single OV circuit in the final layer: L11.H8 reads the flip axis component off the (position-collapsed) stream and writes back 2.4 times its incoming magnitude with the sign reversed, on top of eleven blocks that mostly amplify the flip axis upright (MLP 2 loudest). An overshooting flip at a near-fixed pivot cannot sit still, and cannot run away either once the finite-amplitude geometry bends the response back (composed projected multiplier +0.10): the state must fall onto a period-2 orbit, which is the bell we observe. The stable `Divine` argmax and the readout-mute swing of the earlier reports are the shadow of this structure: one head ringing one direction, everything else holding the tone.

## Caveats

- **The committed flip axis mixes frames.** 06_bell_anatomy.py takes A from the raw iteration-1000 tensor (norm 5098) but rescales B to the shell (norm 1468), because the checkpoint stores the pre-normalisation state. Its d is therefore 0.97 aligned with A's own direction, and downstream statements inherit that: the recorded "phase A carries norm 1612, phase B 464" contrast is the two frames, not an energy slosh (on the shell both phases have identical row norms 464), and lambda = -0.864 for that d is a blend of the true flip axis response with radial annihilation. All conclusions above are stated for both the committed d and the symmetric on-shell flip axis; the physics lives in the latter.
- The eigenvalues are directional derivatives along the flip axis, not a full Jacobian spectrum; other strongly negative directions, if any, were not searched for. The composed-map figure of +0.10 is the projection along the flip axis specifically (the composed return is only 0.51 to 0.56 aligned with the axis); it is not an eigenvalue of the composed map.
- The per-head split reads each head's z-delta through W_O (b_O cancels in deltas). It includes QK-mediated pattern changes routed through z but does not separate OV from QK mechanisms; identifying what L11.H8 attends to, and whether it belongs to a known head taxonomy, is left open.
- Part 2 uses forward differences (run(M + eps d) - run(M)); part 1 uses jvp with central-difference checks. Agreement between the two parts is 0.2 percent where they measure the same object.
- One trajectory, one prompt, one model, derivatives evaluated at one point per state. Whether the other Divine prompts share the same flip head is open (blocked on prompt_library restoration, issue #9).



========================================================================
# SOURCE: Mechanism report: glitch-token alignment
# (repo path: experiments/gpt2_small/output_glitch/glitch_alignment.md)
========================================================================

# Glitch Alignment: the Divine Flip Axis and the Anomalous-Token Cluster

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


*Follow-up to [bell_anatomy.md](../output_divine_motion/bell_anatomy.md) (issue #14, thread 4). Runner: [`07_glitch_alignment.py`](../07_glitch_alignment.py); raw numbers: [`glitch_alignment.json`](glitch_alignment.json). Same single Divine trajectory, states recovered from the saved iteration-1000 checkpoint.*

## Question asked

The bell's rank-1 flip axis d had phase-B riders from the published GPT-2 anomalous-token family (the SolidGoldMagikarp family, Rumbelow and Watkins 2023). Is d ALIGNED with that cluster in embedding space, or merely near it?

## Setup

States replicated exactly as in 06, and the sanity gate passed before any measurement: cos(A, B) = 0.684912 (recorded: 0.6849117), cos(A, f(f(A))) = 1.000000. Flip Axis d = (A - B)/2 at the last position; +d is the phase-A pole, -d is the phase-B pole. d is a residual-stream direction and W_E rows write into the same 768-dimensional residual space, so cos(d, W_E row) is well-defined. All geometry is in the TransformerLens processed basis that 04/06 used; cluster membership is identical in the raw HF basis (Jaccard 1.0 on all four geometric sets).

The cluster, identified three ways:

- **Geometric core** (closest 0.1 percent of the vocab to the mean-embedding centroid, k = 50): the untrained-token signature. In GPT-2 this set is the raw control bytes `\x00` to `\x1f`, the undecodable byte tokens (`�`), and named family members (` externalToEVA`, ` TheNitrome`, `quickShip`, `embedreportprint`, `reportprint`, `rawdownload`, ` サーティ`).
- **Geometric shell** (closest 0.5 percent, k = 251).
- **Low-norm variant** (bottom 0.1 / 0.5 percent by W_E row norm, k = 50 / 251): in GPT-2 this criterion does NOT find glitch tokens. The lowest-norm rows are the highest-frequency function words (` at`, ` in`, ` on`, ` for`). Overlap with the geometric sets: 0 of 50 at the 0.1 percent cutoff, 33 of 251 at 0.5 percent.
- **Curated family**: 54 published family strings, each probed with and without leading space, exact single-token matches only: 52 distinct tokens matched (only ` SmartyHeaderCode` is absent in either form; the vocab holds e.g. `quickShipAvailable`, `StreamerBot`, `ertodd` spaceless). Plus the 4 ideographic oddities from the bell's own readout (`【`, ` 「`, `……`, and the horizontal bar `\u2015`; 6 tokens once spacing variants are included): 58 in all. Of the curated family, 5 tokens sit inside the geometric core and 10 inside the shell; none are in the low-norm sets.

## Results

**The B pole points into the anomalous core.** With u = normalise(cluster centroid - global mean embedding), negative cos(d, u) means the -d (phase-B) pole points toward the cluster:

| cluster | k | cos(d, u) | cos(M, u) | p vs 1000 random | p vs 1000 norm-matched |
|:---|---:|---:|---:|:---:|:---:|
| geometric core 0.1% | 50 | -0.596 | -0.498 | < 0.001 | < 0.001 |
| geometric shell 0.5% | 251 | -0.073 | -0.051 | 0.37 | 1.0 (signed: 0.001) |
| low-norm 0.1% | 50 | +0.594 | +0.515 | < 0.001 | 0.38 |
| low-norm 0.5% | 251 | +0.609 | +0.529 | < 0.001 | 0.14 |
| curated family | 52 | -0.456 | -0.385 | < 0.001 | < 0.001 |
| curated + oddities | 58 | -0.422 | -0.350 | < 0.001 | < 0.001 |

Scale: random same-size token sets give mean |cos| = 0.065 with a maximum of 0.30 over 1000 draws; the core's 0.596 is twice the null maximum (about 7.4 null standard deviations). The norm-matched null is sharper still: sets matching the core's norm profile point the OPPOSITE way (mean cos +0.48, toward the A pole); the anomalous core is the exception at -0.596. The alignment is about which tokens these are, not their norms. For the low-norm sets the norm-matched null is nearly tautological and indeed absorbs the effect (p 0.14 to 0.38). The 251-token shell surrounds the centroid almost isotropically, so its centroid direction washes out (cos -0.073), yet relative to its norm profile it still leans anomalously B-ward (signed p = 0.001), and see the pole scan below.

**The B-pole ray is saturated with cluster members.** Top-50 vocab tokens by cos(row, -d): 45 of 50 are in the geometric core, 50 of 50 are inside the 0.5 percent shell (a 200x enrichment), and 6 of 50 are named family members (`oreAndOnline`, ` RandomRedditor`, ` externalToEVA`, `embedreportprint`, `reportprint`, ` TheNitrome`; 12 percent against a 0.12 percent base rate, about 104x). Top 15 toward -d (cosines 0.565 to 0.568): `\x07`, `\x10`, `\x0b`, `oreAndOnline`, ` サーティ`, `\x11`, `\x1f`, `\x04`, `\x02`, ` RandomRedditor`, `�`, `\x14`, `�`, `\x01`, ` externalToEVA`. The median core member sits at the 99.94th percentile of the whole vocab toward -d, the median curated member at the 96.4th, and all 58 curated tokens, without exception, lean toward the B pole.

**The A pole is the opposite corner: the most-trained tokens.** The top-50 toward +d contains no cluster member of any definition; 42 of 50 are in the bottom 0.5 percent by norm. Top 15: ` the`, `,`, ` in`, ` and`, ` a`, `.`, ` to`, `\n`, `-`, ` (`, ` of`, ` "`, ` on`, ` for`, ` that`. The see-saw runs between the high-frequency function-word corner (phase A) and the untrained corner (phase B): cos(u_core, u_lownorm) = -0.68, and the two independent definitions of the glitch corner agree on where the B wall is, cos(u_core, u_curated) = +0.67.

**Offset, not spread.** cos(d, PC1 of the core's centered embeddings) = +0.009 (curated: +0.047): the flip axis aligns with where the cluster sits relative to the global mean, not with the cluster's internal principal axis. (For the shell, |cos(d, PC1)| = 0.43, because at that radius the dominant internal variance direction is the core's own offset.)

**The flip axis is global, and the pivot leans with A.** Recomputed pos_alignment = 1.0000; cos(d_pos, d_last) = 1.0000 and cos(d_pos, u_core) = -0.5958 identically at every one of the 10 positions. The flip axis is one global direction, not a last-position artifact: the pos_alignment = 1.0 already recorded in bell_anatomy.json means exactly this, confirmed here. The pivot M tilts toward the function-word corner (cos(M, u_core) = -0.498, cos(M, u_lownorm 0.5%) = +0.529). Note cos(d, M) = 0.909 (|A| = 1612 against |B| = 464 leaves both dominated by A's direction), so the A pole nearly coincides with the bell's standing direction; the distinctive, informative pole is -d, phase B's excursion.

## Verdict

**Aligned (structural).** cos(-d, u) = +0.60 with the geometric core and +0.46 with the curated family, beyond every one of 1000 random sets and 1000 norm-matched sets (which, for the core, point the opposite way), with the -d top-50 90 to 100 percent inside the cluster, the named family enriched about 100-fold on that ray, and the same alignment at all 10 positions. The magnitude is a strong tilt, not an identity: 0.46 to 0.60, not 0.9, and the flip axis also carries a large component along the pivot. But the direction of phase B's excursion is unambiguous: each pass, the normalised map throws the state into the degenerate, untrained corner of embedding space and back. The flutter echo has the anomalous-token cluster as one wall and the function-word corner as the other: the cycle oscillates between the model's least-trained and most-trained token directions.

## Caveats

Same single trajectory, prompt, and model as 06. The geometric core is mostly control-byte and undecodable-byte tokens; the named SolidGoldMagikarp family is 10 percent of it (5 of 50), so "the cluster" here means the whole untrained core, with the curated family measured separately and agreeing (cos +0.46, p < 0.001 under both nulls). The low-norm criterion is not an anomaly signature in GPT-2 (it selects frequent function words, with zero overlap with the core at the 0.1 percent cutoff), so its table rows read as the A-pole complement, not as a second glitch test. PC1 signs are arbitrary; only magnitudes matter there. All cosines are computed in the TransformerLens processed basis; membership of every cluster is unchanged in the raw basis (Jaccard 1.0), but exact cosine values would shift slightly there.



========================================================================
# SOURCE: Mechanism report: J-lens phase audit
# (repo path: experiments/gpt2_small/output_jlens_phase/jlens_phase.md)
========================================================================

# J-lens Phase Probe: Both Phases of the Divine Bell, the Pivot, and the Flip Axis (GPT-2 Small)

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


**Status: PILOT follow-up.** The J-lens pilot (`../output_jlens_pilot/jlens_pilot_report.md`,
issue #8) probed the converged Divine (Syntactic) state before `06_bell_anatomy.py` showed
that state is an exact period-2 limit cycle: phases A and B, pivot M = (A+B)/2, flip axis
d = (A-B)/2, with the flip axis 95 percent mute to the readout (logit response ratio 0.054).
The pilot therefore probed only phase A of a two-phase object. This follow-up re-runs the
pilot's membership probe, unchanged, on both phases and the pivot, and answers one new
question: is the flip axis inside or outside the pilot lens subspace? Everything
here inherits the pilot's confidence level and its limitations in full.
A review found that the direction probe first ran on the frame-mixed committed axis;
section 4 now reports both that axis (relabelled d_committed) and the symmetric
on-shell axis d_sym, which is the physical cycle axis and carries the primary numbers.
Script: `../10_jlens_phase.py`.

## 1. Method (what is reused, what is new)

- **Lens**: reused verbatim from `../output_jlens_pilot/jlens_vectors.pt` (193 tokens x
  12 layers x 768). The lens corpus is unreachable on this network, so the lens was NOT
  recomputed; every statement is relative to the committed pilot dictionary.
- **Probe**: identical to `05_jlens_pilot.py`: least-squares span share
  `||proj||^2 / ||state||^2` per layer, nonnegative sparse k=25 share, random-dictionary
  controls drawn from a generator seeded 4242, replayed in the pilot's exact state order
  so the control stream lines up.
- **States**: A, B, M, d reconstructed exactly as `06_bell_anatomy.py` does, from the
  committed iteration-1000 checkpoint `../output_divine_motion/state_divine.pt` (one
  forward pass gives B = f(A), a second verifies period 2). Last-position vectors:
  |A| = 1612, |B| = 464, |M| = 980, |d| = 669. Note cos(d, M) = 0.909: the flip axis is
  strongly aligned with the pivot, so B = M - d is the small residual of two nearly
  aligned vectors (|d|/|B| = 1.44), while A = M + d is dominated by M (|d|/|A| = 0.41).
- **Two axes**: the d above mixes frames (A enters at raw scale, B shell-normalised),
  which is why cos(d, M) = 0.909; it is 0.967 aligned with A's own direction and is
  relabelled **d_committed** throughout. The physical cycle axis is the symmetric
  on-shell axis **d_sym** = normalise(An - Bn), both phases rescaled to the loop shell
  N0 = 1468.5 (full-tensor) before subtracting; d_sym is orthogonal to its pivot M_sym
  (measured cos = 3.7e-07) and 0.616 aligned with d_committed, and once the loop's own
  renormalisation strips d_committed's radial part the survivor is 0.973 aligned with
  d_sym. See `../output_hinge_eigen/hinge_eigenvalue.md`, "The map, the frames, and two
  flip axes". Stage 3 of the script probes d_sym; the stage-2 numbers on d_committed
  are kept below for continuity.
- **Noise controls**: the pilot generated its three noise states in-script (seed 2026)
  and never committed them; they were regenerated here by deterministic replay of the
  pilot's loop, and the probe reproduces the pilot's recorded numbers on them exactly
  (gate below), confirming the replay. The separately committed seed-42 noise run
  (`state_noise.pt`, a DIFFERENT trajectory from `05_divine_motion.py`) is included as a
  supplementary row only.
- **New, for the flip axes only**: (a) a 20-random-unit-direction baseline (seed 777) giving
  what a generic direction scores against each layer dictionary (0.245 to 0.253 at every
  layer, i.e. the 193/768 = 0.251 chance level of a numerically full-rank 193-vector
  span); (b) span probes of each axis's top-100 (readout-visible) and bottom-100
  (readout-quiet) W_U singular components, the same split `06_bell_anatomy.py` used.
  The sparse probe is sign-dependent (nonnegative coefficients), so for a direction both
  signs are reported; the span share, which is sign- and scale-invariant, is the flip
  axis number. Both instruments run on d_committed (stage 2) and d_sym (stage 3); the
  stage-3 baseline is gated to replay the stage-2 baseline exactly (max diff 0.0).

## 2. Sanity gates (all passed)

| gate | value | reference |
|---|---|---|
| cos(A, B) | 0.6849116683 | bell_anatomy.json 0.6849116683 |
| cos(A, f(f(A))) | 1.0000000000 | bell_anatomy.json 1.0 |
| cos(A, pilot's probed Syntactic state) | 0.99999988 | the pilot probed phase A |
| d_committed energy in top-100 / bottom-100 W_U dirs | 0.128516 / 0.731787 | bell_anatomy.json 0.128516 / 0.731787 |
| probe replay, all 8 pilot states, lens columns | max diff 1.1e-7 (span), 0.0 (sparse) | jlens_pilot_results.json |
| Divine number reproduced | L6 span 0.210894 | pilot 0.210894 |
| cos(d_sym, M_sym) | 3.7e-07 | orthogonal by construction; hinge_eigenvalue.json 4.6e-07 |
| cos(d_sym, d_committed) and cos(d_sym, tangentialised d_committed) | 0.616186 and 0.972601 | hinge_eigenvalue.json 0.616186 and 0.972602 |
| stage-3 W_U SVD reproduces stage-1 committed-d split | diff < 1e-5 | phase_states.pt gates |
| stage-3 seed-777 baseline replays stage 2 | max diff 0.0 | jlens_phase.json |

One replay caveat: the random-DICTIONARY control columns do not replay bit-identically
(max diff 0.045). This is expected: the pilot report's own recording caveat says the
archived JSON's sparse control came from an older single-draw version of the script, so
the generator stream differs from the committed three-draw protocol replayed here. The
controls in this run are fresh draws under the committed protocol, same distribution and
same qualitative level (about 0.24 to 0.28 span, 0.07 to 0.12 sparse). The measured lens
columns, which are deterministic, reproduce to numerical noise.

## 3. Results: the phase table

Least-squares span share per layer ("d_committed" and "generic dir" are direction
probes, included for comparison; d_committed is the frame-mixed committed axis, see
section 4; prolet is the mean of Lucier, Semantic, Nonsense, Imperative;
noise is the mean of the three regenerated pilot noise states):

| layer | A | B | M | prolet | noise | d_committed | generic dir |
|---|---|---|---|---|---|---|---|
| L0 | 0.213 | 0.152 | 0.221 | 0.183 | 0.184 | 0.179 | 0.253 |
| L1 | 0.216 | 0.162 | 0.227 | 0.211 | 0.186 | 0.180 | 0.249 |
| L2 | 0.221 | 0.165 | 0.231 | 0.218 | 0.190 | 0.185 | 0.247 |
| L3 | 0.205 | 0.151 | 0.214 | 0.195 | 0.185 | 0.174 | 0.245 |
| L4 | 0.207 | 0.155 | 0.216 | 0.187 | 0.192 | 0.176 | 0.245 |
| L5 | 0.207 | 0.157 | 0.216 | 0.197 | 0.198 | 0.174 | 0.248 |
| L6 | 0.211 | 0.160 | 0.221 | 0.195 | 0.196 | 0.178 | 0.247 |
| L7 | 0.209 | 0.162 | 0.220 | 0.188 | 0.187 | 0.174 | 0.248 |
| L8 | 0.197 | 0.151 | 0.207 | 0.179 | 0.169 | 0.164 | 0.251 |
| L9 | 0.181 | 0.135 | 0.190 | 0.163 | 0.146 | 0.152 | 0.249 |
| L10 | 0.171 | 0.127 | 0.179 | 0.152 | 0.129 | 0.143 | 0.251 |
| L11 | 0.173 | 0.123 | 0.180 | 0.157 | 0.114 | 0.145 | 0.252 |

Nonnegative sparse k=25 share per layer:

| layer | A | B | M | prolet | noise |
|---|---|---|---|---|---|
| L0 | 0.104 | 0.074 | 0.109 | 0.085 | 0.052 |
| L1 | 0.109 | 0.084 | 0.116 | 0.112 | 0.068 |
| L2 | 0.104 | 0.077 | 0.109 | 0.095 | 0.058 |
| L3 | 0.107 | 0.074 | 0.111 | 0.095 | 0.060 |
| L4 | 0.095 | 0.069 | 0.098 | 0.081 | 0.056 |
| L5 | 0.099 | 0.074 | 0.106 | 0.093 | 0.064 |
| L6 | 0.111 | 0.081 | 0.116 | 0.095 | 0.063 |
| L7 | 0.115 | 0.086 | 0.120 | 0.097 | 0.062 |
| L8 | 0.115 | 0.084 | 0.119 | 0.099 | 0.060 |
| L9 | 0.108 | 0.079 | 0.114 | 0.095 | 0.054 |
| L10 | 0.096 | 0.068 | 0.100 | 0.091 | 0.047 |
| L11 | 0.098 | 0.070 | 0.102 | 0.091 | 0.048 |

Supplementary row: the committed seed-42 noise run (`state_noise.pt`, iteration 1000)
scores like the pilot noise family (span 0.160 at L0, 0.179 at L6, 0.100 at L11; sparse
0.056 at L6, 0.042 at L11).

Reading, in decreasing order of confidence:

1. **The phases differ materially.** Phase B sits below phase A at every layer on both
   probes: span lower by 0.043 to 0.060 absolute, sparse lower by 0.025 to 0.033. This
   is not a norm artifact (both shares are scale-invariant). Against the prolet family,
   B is lower on span at every layer (by 0.025 to 0.053) and lower on sparse at every
   layer; A is higher than prolet at every layer on both probes. The two phases straddle
   the prolet level.
2. **The pivot M is the most lens-expressible object probed**: above A at every layer on
   both probes (span +0.007 to +0.011, sparse +0.003 to +0.007). The stable ` Divine`
   readout was shown in the bell anatomy to be the shadow of M; M is also the most
   verbalizable-adjacent state the instrument has seen.
3. **The language-vs-noise boundary survives on the sparse probe, and weakens for B on
   the span probe.** On sparse, every language-derived state including B stays above the
   noise mean at all 12 layers (B's smallest margin +0.010 at L5). On span, B actually
   falls below the noise mean at L0 through L10 (by 0.001 to 0.041) and only overtakes
   it at L11 (+0.009), where noise collapses (0.114) and B holds 0.123. The pilot's
   clean late-layer span separation (language 0.15 to 0.17 vs noise 0.11 to 0.13) is a
   phase-A and pivot property; phase B sits at its edge.
4. A mechanical account of 1, in the committed frame: since cos(d_committed, M) = 0.909
   and |d_committed|/|B| = 1.44, phase B's direction is dominated by d_committed
   content, which is mostly outside the lens (section 4b); A's direction is dominated
   by the pivot, which is inside-leaning. On the shell the same gap reads differently:
   An and Bn are M_sym plus and minus the same d_sym component (span share is
   scale-invariant, so the A and B rows above are the shell phases' rows), and the
   A-over-B margin is the cross term between the lens projections of M_sym and d_sym,
   positive for A, negative for B.

## 4. The flip axis question: two axes

The direction probe first ran on the committed axis d_committed (stage 2), built
exactly as `06_bell_anatomy.py` builds it: shell-frame B subtracted from raw-frame A.
That axis mixes frames: it is 0.967 aligned with A's own direction and 0.909 aligned
with its own pivot, so most of its content is radial (pivot-like) rather than cycle
motion. The physical axis of the period-2 cycle is the symmetric on-shell axis d_sym
(stage 3), orthogonal to its pivot (measured cos(d_sym, M_sym) = 3.7e-07) and 0.973
aligned with what survives of d_committed after the loop's renormalisation strips the
radial part. See `../output_hinge_eigen/hinge_eigenvalue.md`, "The map, the frames,
and two flip axes". Section 4a gives the d_sym numbers; they are the primary flip-axis
numbers. Section 4b keeps the d_committed numbers as originally measured, relabelled.

### 4a. The physical axis d_sym

Span shares of d_sym and its W_U singular components against each layer's lens
dictionary ("generic dir" is the 20-random-direction baseline; the nn25 columns are
the sign-dependent nonnegative sparse probe, both signs):

| layer | d_sym span | generic-dir span | d_sym nn25 (+) | d_sym nn25 (-) | dsym_vis span | dsym_quiet span |
|---|---|---|---|---|---|---|
| L0 | 0.017 | 0.253 | 0.006 | 0.003 | 0.256 | 0.009 |
| L1 | 0.017 | 0.249 | 0.006 | 0.002 | 0.283 | 0.010 |
| L2 | 0.020 | 0.247 | 0.006 | 0.004 | 0.302 | 0.011 |
| L3 | 0.024 | 0.245 | 0.010 | 0.006 | 0.328 | 0.015 |
| L4 | 0.028 | 0.245 | 0.011 | 0.010 | 0.350 | 0.019 |
| L5 | 0.028 | 0.248 | 0.012 | 0.007 | 0.354 | 0.019 |
| L6 | 0.029 | 0.247 | 0.011 | 0.007 | 0.372 | 0.018 |
| L7 | 0.024 | 0.248 | 0.011 | 0.006 | 0.375 | 0.015 |
| L8 | 0.021 | 0.251 | 0.009 | 0.006 | 0.390 | 0.012 |
| L9 | 0.019 | 0.249 | 0.008 | 0.006 | 0.409 | 0.009 |
| L10 | 0.017 | 0.251 | 0.007 | 0.005 | 0.449 | 0.008 |
| L11 | 0.013 | 0.252 | 0.005 | 0.003 | 0.572 | 0.002 |

**The one number: the physical flip axis's lens-span share at L11 is 0.013, against a
0.252 chance level for a generic direction (5 percent of chance; mean over all layers
0.021 vs 0.249, 9 percent).** The share never exceeds 0.029 (12 percent of chance, at
L6): the cycle axis is almost entirely outside the lens at every depth. The sparse
shares are 0.005 to 0.012 for +d_sym and 0.002 to 0.010 for -d_sym. The committed
axis's milder deficit (58 percent of chance at L11, section 4b) comes from frame
mixing: 83 percent of d_committed's energy lies along the pivot direction, the pivot
spans 0.180 at L11 (section 3), so most of d_committed's lens share is pivot
contamination, not cycle motion.

The W_U singular split of d_sym: the top-100 (readout-visible) component carries 1.2
percent of its energy and the bottom-100 (readout-quiet) component 97.0 percent;
together the two subsets cover 98.2 percent of the energy, and the middle 568 singular
directions hold the remaining 1.8 percent. The split restates the bell anatomy's
muteness result in the clean frame: the physical swing is almost wholly readout-quiet.

- The quiet bulk (97.0 percent of d_sym's energy) is essentially outside the lens at
  every layer: span between 0.002 (L11) and 0.019 (L4), at most 8 percent of chance.
- The visible sliver (1.2 percent of d_sym's energy) is inside-leaning and
  increasingly so with depth: span 0.256 at L0, which is at the 0.253 chance level,
  rising monotonically to 0.572 at L11 (2.3x chance). For d_sym the sliver's
  inside-lean is a mid-to-late-layer property; at L0 it is indistinguishable from a
  generic direction.

### 4b. The committed axis d_committed (as originally probed)

Span shares of the direction d_committed and its W_U singular components, against each
layer's lens dictionary ("rand dict" is the pilot-style matched random dictionary
control, "generic dir" the 20-random-direction baseline; "d" in the column heads is
d_committed):

| layer | d span | rand-dict span | generic-dir span | d nn25 (+d) | d nn25 (-d) | d_vis span | d_quiet span |
|---|---|---|---|---|---|---|---|
| L0 | 0.179 | 0.241 | 0.253 | 0.086 | 0.030 | 0.335 | 0.066 |
| L1 | 0.180 | 0.256 | 0.249 | 0.090 | 0.029 | 0.362 | 0.068 |
| L2 | 0.185 | 0.265 | 0.247 | 0.085 | 0.044 | 0.389 | 0.067 |
| L3 | 0.174 | 0.249 | 0.245 | 0.090 | 0.042 | 0.418 | 0.045 |
| L4 | 0.176 | 0.262 | 0.245 | 0.081 | 0.046 | 0.450 | 0.044 |
| L5 | 0.174 | 0.257 | 0.248 | 0.084 | 0.043 | 0.474 | 0.036 |
| L6 | 0.178 | 0.276 | 0.247 | 0.094 | 0.042 | 0.506 | 0.030 |
| L7 | 0.174 | 0.272 | 0.248 | 0.096 | 0.039 | 0.511 | 0.023 |
| L8 | 0.164 | 0.267 | 0.251 | 0.094 | 0.040 | 0.542 | 0.017 |
| L9 | 0.152 | 0.245 | 0.249 | 0.089 | 0.035 | 0.558 | 0.011 |
| L10 | 0.143 | 0.253 | 0.251 | 0.081 | 0.028 | 0.570 | 0.006 |
| L11 | 0.145 | 0.261 | 0.252 | 0.080 | 0.026 | 0.619 | 0.008 |

**The committed axis's lens-span share at L11 is 0.145, against a 0.252 chance
level for a generic direction (58 percent of chance; mean over all layers 0.169 vs
0.249, 68 percent).** The committed axis leans outside the lens subspace at every depth,
and the deficit deepens with depth (ratio to chance falls from about 0.71 at L0-L7 to
0.57 at L10). Per section 4a, this milder deficit is a pivot-diluted version of the
d_sym result.

The decomposition connects to the bell anatomy's muteness result:

- The readout-QUIET bulk of d_committed (bottom-100 W_U component, 73.2 percent of
  d_committed's energy) is essentially outside the lens at every layer: span 0.066 at
  L0 shrinking monotonically to 0.008 at L11 (3 percent of chance). At L11 this is
  close to definitional (the pilot's L11 lens vectors are near logit-lens directions),
  but at L0 through L8 it is not: even the earliest layers' verbalizable dictionaries,
  whose vectors have been backpropagated through the whole network, give the quiet
  component no home.
- The readout-VISIBLE sliver (top-100 component, 12.9 percent of d_committed's energy)
  is strongly INSIDE: span 0.335 at L0 rising to 0.619 at L11, 1.3x to 2.5x chance.

So on the tested subsets, muteness to the readout and muteness to the lens point the
same way at every depth of the pilot lens: the top-100 (readout-visible) component
leans inside and the bottom-100 (readout-quiet) component sits outside. The
coincidence claim applies to these selected extremes of the W_U spectrum only. The two
subsets cover 86.0 percent of d_committed's energy (12.9 top, 73.2 bottom) and 98.2
percent of d_sym's (1.2 top, 97.0 bottom); the middle singular components (14.0 and
1.8 percent of the energy respectively) are unassigned by the split, so no
readout-lens statement is made about them. For d_sym there is a further qualification:
its visible sliver is at chance at L0 and strongly inside only from the middle layers
on (section 4a). The sparse probe on d_committed is reported for completeness only
(0.080 to 0.096 for +d, 0.026 to 0.046 for -d); its sign-dependence makes the span
number the answer.

## 5. Verdict

**The phase-A-only pilot verdict partly survives and is now phase-qualified.** The
pilot's reversal ("Divine at least as expressible as prolet") holds for phase A, is
strengthened at the pivot M, and REVERSES for phase B, which is less lens-expressible
than the prolet attractor at every layer on both probes and, on the span probe, dips to
converged-noise level until the final layer. The bell is not "inside" or "outside" as
one object: it swings once per iteration between a more-verbalizable phase and a
less-verbalizable phase, pivoting on the most-verbalizable state in the system, along a
flip axis that on the physical construction is almost entirely outside the lens (d_sym
span 0.013 vs 0.252 chance at L11, mean 0.021 vs 0.249; the frame-mixed d_committed
spans 0.145 and 0.169), and whose readout-quiet bulk (97.0 percent of d_sym's energy)
is outside it almost completely at every depth. The d_sym numbers sharpen this verdict
and do not change its direction: moving from the committed axis to the physical axis
takes the L11 span from 58 percent of chance to 5 percent. The
language-vs-noise boundary remains the pilot's most robust story, but it is now a
sparse-probe story: on the span probe phase B sits at, and below, the noise level for
most of the depth. None of this is a null: the two phases are materially distinguishable
to the lens, which the phase-blind pilot could not see.

## 6. Limitations

1. Every pilot limitation inherits unchanged: 30-prompt hand-written corpus, 193-token
   dictionary (0.4 percent of vocabulary), low-rank lens vs full-rank controls, GPT-2
   Small's 12 coarse layers, and an averaged Jacobian still visibly moving at 30
   prompts. The lens could not be recomputed on this network.
2. Effective sample sizes: one Divine cycle (one prompt), one effective prolet
   attractor, three regenerated noise states plus one committed one. All gaps are point
   estimates; no error bars are possible.
3. The generic-direction baseline (seed 777) is new to this run, not part of the pilot
   instrument; it lands at the analytic 193/768 chance level, which is also what the
   pilot's random-dictionary control measures.
4. The top-100 / bottom-100 W_U split leaves the middle 568 singular directions
   unassigned to either component: 14.0 percent of d_committed's energy and 1.8
   percent of d_sym's. Statements pairing readout-visibility with lens membership
   apply to the two selected subsets, not to the whole axis.
5. The random-dictionary control columns are fresh draws (generator stream mismatch with
   the archived pilot JSON, section 2); the measured lens columns are unaffected.

## 7. Files

- `jlens_phase.md`: this report
- `jlens_phase.json`: all numbers (per-state per-layer shares for the 8 replicated pilot
  states, Divine_A, Divine_B, Divine_M, the committed seed-42 noise row; the direction
  probe for d_committed, d_vis, d_quiet under `direction_probe`, and for d_sym and its
  W_U components under `direction_probe_sym`, with the frame checks in
  `summary.axis_frame_check`; reproduction diffs against the pilot JSON; gates, norms,
  cosines; summary tables)
- `phase_states.pt`: stage-1 checkpoint (A, B, M, d, A2, the W_U components, regenerated
  noise states, gate values) plus the stage-3 cached d_sym W_U components
- Script: `../10_jlens_phase.py` (run with `ATR_GPT2_LOCAL=<gpt2 dir>`; stages:
  `stage1` needs the model, `stage2` is pure linear algebra on the committed lens,
  `stage3` probes the symmetric axis d_sym and needs the model once for the W_U split,
  after which the components are cached and re-runs are pure linear algebra)

Compute: stage 1 (model load, 2 cycle steps, 300 noise iterations, one W_U SVD) 97 s
single-threaded; stage 2 (all probes) 10 s; stage 3, 33 s on first run (model load for
the W_U split included), 1 s resumed from the cache.



========================================================================
# SOURCE: Mechanism report: suppression test
# (repo path: experiments/gpt2_small/output_suppression/suppression_report.md)
========================================================================

# Suppression-Head Test for L11.H8: Inversion Confirmed and Load-Bearing, Copy-Suppression Signature Absent on Ordinary Text

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


*Follow-up to [hinge_eigenvalue.md](../output_hinge_eigen/hinge_eigenvalue.md) (issue #14, thread 1). Runner: [`11_suppression_test.py`](../11_suppression_test.py); raw numbers: [`suppression_results.json`](suppression_results.json). Model: GPT-2 Small, single Divine trajectory, states rebuilt from the committed iteration-1000 checkpoint. Sanity gates reproduced before any measurement: cos(A, B) = 0.684912, cos(A, f(f(A))) = 1.000000, full-tensor cycle residual 7.97e-04 (JSON: meta.gate).*

## The hypothesis

Experiment 08 located the sign inversion that sustains the Divine period-2 cycle in one attention head: L11.H8 carries 99.1 percent of the block-11 attention flip along the flip axis (per-head d-component -1.981, cos -0.963). The suppression-head hypothesis reads that finding as an instance of a known behaviour class: L11.H8 functions as a suppression head, meaning its output is approximately a negative multiple of a component of its input, the class documented for GPT-2 Small's L10.H7 copy-suppression head, and the closed loop turns that one-shot negative correction into a sustained oscillation.

Three tests: (1) the OV circuit along the flip axis, all 144 heads; (2) ablation of the head inside the loop, against controls; (3) the head's effect on attended-token logits on ordinary text, against the documented copy-suppression head.

## Verdict logic, stated up front

The hypothesis is SUPPORTED if (1) L11.H8's OV inverts d_sym distinctly among heads, (2) ablating it kills the cycle while a control ablation does not, and (3) it shows the negative-delta copy-suppression signature on ordinary text. (1) and (2) without (3) means the head inverts this direction and sustains the cycle but is not a general copy suppressor, and the training-function part of the hypothesis stays open. If ablation kills the cycle, the head is load-bearing either way; test 3 is what separates learned function from structural accident.

**Outcome: (1) supported, (2) supported, (3) refuted, and refuted with the opposite sign.** L11.H8's OV inverts d_sym more strongly than any other head (cos -0.9619, gain 63.68, rank 1 of 144 on both measures), ablating it collapses the cycle to a fixed point within about 10 iterations while the same-layer control ablation leaves a period-2 cycle running, and on ordinary text the head RAISES the logit of the token it attends to at 91.4 percent of positions (mean delta +5.97), while the L10.H7 positive control shows the documented suppression (87.1 percent negative, mean -3.62). L11.H8 sustains the bell by inverting the flip axis, but it is not a copy suppressor; outside the loop it is a copy promoter. The training-function part of the hypothesis is not supported by what test 3 measured.

## Structural fact, verified before measurement

The Divine state is position-uniform, and this makes the head's inside-the-loop behaviour a pure OV circuit (JSON: structural):

- The layer-11 input during the phase-A pass has row spread 5.5e-07 (all 10 rows identical to numerical precision), and every recorded loop state below stays position-uniform (max row spread 8.4e-07 across all 500 recorded iterates).
- For a position-uniform input, every source position carries the same value vector, so the pattern-weighted average is that vector and the head's output is the OV transform of the ln1-normalised input regardless of the attention pattern. Verified empirically at phase A: head 8's hooked output row matches the direct computation ln1(x) @ W_V[11,8] @ W_O[11,8] with relative error 1.37e-07 (output row norm 1825.0). The value bias contributes nothing here because the loading convention folds b_V into b_O (b_V @ W_O norm = 0.0). The attention pattern at phase A is in fact exactly uniform, 0.100 on each of the 10 positions, and it does not matter.

This licenses reading test 1's static OV numbers as the head's actual in-loop transfer function.

## Test 1: The OV circuit inverts the flip axis, and L11.H8 is the extreme of all 144 heads

For every head, y = d @ W_V[l,h] @ W_O[l,h], recording cos(y, d) and gain ||y|| / ||d|| (JSON: test1.directions).

Along the primary flip axis d_sym:

| Head | cos(y, d_sym) | gain | rank by cos (1 = most negative) |
|:---|:---:|:---:|:---:|
| **L11.H8** | **-0.9619** | **63.68** | **1** |
| L10.H7 (copy-suppression head) | +0.1543 | 0.34 | 102 |
| L11.H0 (arbitrary) | +0.3741 | 0.90 | 137 |
| L5.H5 (arbitrary) | +0.0815 | 0.20 | 78 |
| L0.H0 (arbitrary) | +0.1078 | 0.25 | 86 |

Distribution over the 144 heads: median cos +0.059, extremes -0.962 and +0.523; 54 heads below 0; 5 heads below -0.5 (L11.H8 at -0.962 gain 63.68, L1.H11 at -0.913 gain 1.33, L4.H7 at -0.837 gain 0.48, L2.H10 at -0.787 gain 0.29, L1.H10 at -0.778 gain 0.36). L11.H8 is the most negative in cosine, and its gain is 48 times the next-most-negative head's. In d-component terms (cos times gain), L11.H8 writes -61.26 per unit of row-level d_sym; the runner-up is L1.H11 at -1.21. The inversion of the flip axis is not merely the most extreme in the population, it is a different magnitude class.

Secondary frame, the committed d (0.616 aligned with d_sym, radial contamination as documented in experiment 08): L11.H8 has cos -0.3895, gain 7.42, rank 6 by cosine, but still the most negative d-component of all heads (-2.89 against -0.73 for the runner-up). Both frames agree on which head does the inversion.

Pole directions: the phase-A pole (+d_sym) and phase-B pole (-d_sym) give values identical to d_sym's, exactly, because the OV map is linear (cos(y(-d), -d) = cos(y(d), d)); recorded as a code-path check.

Controls, 5 random unit vectors (seed 20260721): L11.H8's cos values are +0.0281, -0.0008, +0.0175, -0.0250, -0.0080; the population mean |cos| per direction is 0.077 to 0.083, no head below -0.5 on any random direction, most negative single value -0.301. The inversion is specific to the flip axis, not a generically negative OV.

### Empirical checks at the operating point (JSON: test1.empirical)

All finite-difference responses are in the experiment-08 convention: head-8 output change per unit of the full-tensor unit flip axis, measured at the last row against the unit row flip axis (row-level values are sqrt(10) = 3.162 times larger).

1. **Block-0 injection at Mn_sym (the exp-08 measurement, reproduced).** d-component -1.9814 at eps 1e-3 and -1.9860 at 1e-4, cos -0.9632. The recorded experiment-08 value is -1.9814; reproduction is exact at the matching epsilon.
2. **Layer-11 injection at Mn_sym (the literal x = M operating point).** d-component -1.1565, cos -0.9619, ln1 scale at the base row 16.76. Converting to row-level (-1.1565 x 3.1623 = -3.657) and multiplying by the ln1 scale gives -61.30, against the raw linear value -61.26 (0.07 percent apart), and the cosine equals the linear cosine to four decimals. At this base point d_sym is exactly orthogonal to M_sym (cos 0.0, JSON: meta.cos_esym_vs_Msym_row), so the ln1 Jacobian reduces to division by the scale, and the raw linear OV row is recovered exactly. Sign and magnitude of the linear computation are confirmed at the operating point.
3. **Layer-11 injection at the cascade resid_pre_11 (the input the head actually sees mid-loop, row norm 1042.6, ln1 scale 37.62).** d-component -0.3433, cos -0.9645.
4. **Chain to the exp-08 number.** The cascade delta arriving at layer 11 carries d_sym content 2.5863 out of Frobenius norm 2.9224. The pure-d response times that content is -0.3433 x 2.5863 = -0.8878, against the end-to-end -1.9814: the pure-d channel accounts for 45 percent, and the remaining -1.094 is the head's response to the off-axis components the cascade generates (norm 1.361), which the head also delivers along -d_sym (end-to-end output cos -0.9632, pure-d output cos -0.9645, same output direction). The head's output direction is the same for both input components: what the cycle feeds it comes back along the negative flip axis.

## Test 2: Ablating L11.H8 kills the cycle; the control ablation does not

Protocol: the ATR loop run from the phase-A checkpoint, with blocks.11.attn.hook_z zeroed at the target head on every forward pass. Hook verified on the first pass: the layer's attn_out changes by exactly the head's z @ W_O contribution (relative error 7.2e-08 for H8, 1.9e-06 for H0), and the block output moves (resid_post_11 change norm 5634.7 for H8, 230.1 for H0; the removed component norms are 5771.2 and 239.3, so H8 writes 24 times more than H0 at this state). Main run 300 iterations (100 was already unambiguous; 300 run outright), controls 100. Per-iterate lag-1 cosine, cosines to A, B, M_sym, M_committed, and readout argmax are in JSON: test2.*.records.

| Run | lag-1 cos, end | lag_scan last 24 (k = 1..8) | end state | argmax over run |
|:---|:---:|:---|:---|:---|
| No ablation | 0.6849 | 0.685, 1.000, 0.685, 1.000, 0.685, 1.000, 0.685, 1.000 | the exact A/B cycle: cos to A alternates 1.0000 / 0.6849 | ' Divine' at all 100 iterates (final p = 0.505) |
| Ablate L11.H0 (control) | 0.8938 | 0.893, 1.000, 0.894, 1.000, 0.894, 1.000, 0.894, 1.000 | a deformed period-2 cycle: phases at cos 0.9309 and 0.7229 to A, lag-2 cosine 1.000000 | ' Divine' at all 100 iterates (final p = 0.674) |
| **Ablate L11.H8** | **1.000000** | **1.000 at every lag 1..8** | **a fixed point away from the cycle: cos to A +0.1419, to B -0.6087, to M_sym -0.2543, to M_committed -0.0275** | ' Divine' (iters 1-2), '\n' (3), ' the' (4-300) |

The period-2 alternation stops immediately under the H8 ablation: lag-1 cosine rises from 0.8970 at iteration 1 to above 0.9999 by iteration 9 and above 0.999999 by iteration 14, and every lag 1 through 8 reads 1.000000 over the last 24 iterates. The state settles to a fixed point of the ablated map that is not near A, not near B, and not near either pivot (cosines above); its raw norm stabilises at 3574 (unablated phases: 5098 and 4838) and it stays position-uniform. The readout changes from ' Divine' at probability 0.5 to a flat generic distribution: final top-5 is ' the' 0.0235, ',' 0.0153, ' and' 0.0111, '.' 0.0107, ' a' 0.0105.

The control separates head identity from generic damage: removing L11.H0 (which also writes into the stream, norm 239) leaves a period-2 cycle running with the same ' Divine' argmax, with the two phases closer together (cos 0.8935 between consecutive iterates instead of 0.6849). The cycle needs L11.H8 specifically.

## Test 3: On ordinary text, L11.H8 raises the attended token's logit; the suppression signature belongs to L10.H7

Protocol: 12 natural sentences (the five 04_readout_confidence prompts plus seven new ones, 9 to 14 tokens each), each run once with no loop. For every position t >= 2: find the head's strongest non-BOS source s, take the head's per-position output through W_O, and compute delta = output_t @ W_U[:, token at s]. Copy suppression predicts predominantly negative delta. 116 positions per head (JSON: test3.per_head, per-position rows in test3.per_position_rows).

| Head | frac delta < 0 | mean delta | mean delta per unit output | mean attn to BOS | restricted to top-source attn > 0.2 |
|:---|:---:|:---:|:---:|:---:|:---|
| **L11.H8** | **0.086** | **+5.97** | +0.0263 | 0.0001 | n = 115: frac neg 0.087, mean +5.99 |
| L10.H7 (positive control) | 0.871 | -3.62 | -0.293 | 0.808 | n = 12: frac neg 1.000, mean -15.85, per unit output -0.928 |
| L11.H0 (arbitrary) | 0.638 | -0.52 | -0.0051 | 0.330 | n = 51: frac neg 0.608, mean -0.58 |
| L5.H5 (arbitrary) | 0.397 | +0.31 | +0.054 | 0.960 | n = 1 |

The protocol works: L10.H7 spends most of its attention on BOS, and at every position where it commits more than 0.2 attention to a real source it lowers that token's logit, mean -15.85, per-unit-output projection -0.928, which is the documented copy-suppression behaviour. The arbitrary heads sit near zero or mixed. L11.H8 is the opposite of the hypothesis: it is highly active (attention to BOS 0.0001, mean attention to its top source 0.40, top source is the position itself at only 19 percent of positions), and it RAISES the attended token's logit at 106 of 116 positions, mean +5.97 (median +6.50, range -9.30 to +12.92). The 10 negative positions are mostly early positions attending to a sentence-initial capitalised token ('He', 'Rain', 'Cal', 'Ast') plus the nonsense token ' morp'. Restricting to confident positions changes nothing (115 of 116 qualify). On ordinary text L11.H8 behaves as a copy-promoting head, not a copy suppressor.

## What follows

- **The mechanism is confirmed and sharpened.** The flip axis inversion that sustains the bell is the static OV geometry of one head: d_sym is the single direction, out of the whole population, that L11.H8's OV both inverts (cos -0.9619) and amplifies (gain 63.7), the empirical operating-point response matches the linear computation exactly once the ln1 scale is accounted for, and the closed loop needs exactly this head: ablate it and the bell is replaced by a fixed point with a generic readout within about 10 iterations; ablate a neighbour and the bell persists.
- **The behaviour-class part of the hypothesis fails.** L11.H8 does not show suppression behaviour on ordinary text; it shows the reverse, while the measurement demonstrably detects suppression where it is documented (L10.H7). The one-shot negative correction that the loop recycles is therefore not this head's text-time function along token directions.
- **On learned function versus structural accident.** The ablation result makes the head load-bearing for the cycle under either reading, so it cannot separate them; test 3 was the separator, and it came out against learned general suppression. On the evidence here, the Divine oscillation exploits a strongly negative direction that happens to exist in this head's OV spectrum, a direction whose sign-inverting treatment is not exercised as suppression in ordinary next-token service. The accident reading is strengthened; what remains open is whether d_sym relates to some non-token content the head suppresses in contexts not sampled here.

## Limits

- Test 3's delta omits the final LayerNorm scaling, a positive per-position scalar that cannot change signs, and reads W_U directly (W_O writes are centered by the loading convention). It measures copy suppression in the token-unembedding sense only; suppression of non-token content would not register.
- The top non-BOS source may be the query position itself (19 percent of L11.H8's positions, 48 percent of L10.H7's); no exclusion was applied, matching the stated protocol.
- One trajectory, one loop prompt, one model. The OV computation ignores QK: which content the head attends to inside the loop is moot (the pattern is uniform and irrelevant under position uniformity), but on text the source selection is QK's and was taken as observed.
- The committed-d numbers inherit the frame mix documented in experiment 08; d_sym is the physical flip axis and all headline claims use it.
- 12 sentences, 116 positions is a small text sample; the L11.H8 result (frac negative 0.086 against L10.H7's 0.871) is far from any decision boundary, so the sample suffices for the sign of the verdict but not for fine effect sizes.
