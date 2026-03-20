# Activation Tensor Resonance (ATR)

### *Mapping the Attractor Landscape of GPT-2 Small's Weight Geometry*

Inspired by Lucier's iterative feedback process, this project applies the same structural operation to GPT-2 Small. Where Lucier's process dissolved speech into a room's resonant frequencies through looped excitation, **Activation Tensor Resonance** dissolves semantic content into a language model's architectural eigenmodes — the dominant attractor states encoded in its weight matrices.

<p align="center">
  <img src="B_AttractorDominance/output_stage1/convergence_matrix.png" alt="Stage 1: Cross-Prompt Convergence — 125 prompts mapped across 5 attractor basins" width="800"/>
</p>

<p align="center"><em>Cross-prompt convergence matrix (125 prompts × cosine similarity). The block structure reveals five distinct attractor basins in GPT-2 Small's weight geometry.</em></p>

---

## The Inspiration

<p align="center">
  <img src="docs/lucier_room.png" alt="A sparse room with a reel-to-reel tape recorder and microphone — evoking Alvin Lucier's experimental setup" width="600"/>
</p>

In 1969, composer [**Alvin Lucier**](https://en.wikipedia.org/wiki/Alvin_Lucier) created [***I Am Sitting in a Room***](https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room) — a process piece where he recorded himself speaking, played the recording back into the room, re-recorded the result, and repeated. With each iteration, the speech dissolved into the room's resonant frequencies. The words vanished. The architecture spoke.

> *"I am sitting in a room different from the one you are in now. I am recording the sound of my speaking voice..."*

🎬 [**Watch the original performance on YouTube**](https://www.youtube.com/watch?v=v9XJWBZBzq4) — Lucier's sound check and performance of the piece.

---

## The Discovery

*The Body without Organs is a Marxist.*

Five diverse prompts were tested — a question, a factual statement, a grammatical pattern, nonsense, and a command:

📓 **Notebook:** [`lucier_total_resonance.ipynb`](ActivationTensorResonance/lucier_total_resonance.ipynb)

**Four out of five** converged through the same dissolution trajectory to a common terminal state: the BPE subword `prolet` — a fragment: a suggestion.

![Sentence dissolution — the five prompts dissolving into their attractors](ActivationTensorResonance/images/dissolution1.png)

![Dissolution continued](ActivationTensorResonance/images/dissolution2.png)

Words drain and dissolve. First connection, then meaning is stripped away, then grammar. A littering of acronyms, chemical symbols, punctuation and broken slang compliantly settle into their final form. A fragmented word, endlessly repeated. Their resting places are the **dominant attractors** — hidden architectural hollows encoded in the model's weight matrices, now prominent features acting as gravity wells for semantics.

Each descent is a journey that tells a story:

```
[iter 2] ash → [5] Canad → [10] Ag → [20] FT → [50] capit → [100] injustice → [250] Rousse → [500] prolet
```

It was a genuine surprise to watch the seemingly boundless possibility of language so rapidly crushed into just 5 single last words. Four of the five initial prompts — the question, the facts, the nonsense, and the command — followed an almost identical path into the same basin of attraction.

Mapping the suggestions:

**Geography** [Canad(a)] → **Finance** [Ag, FT, capit(al)] → **Political Philosophy** [injustice, Rousse(au), prolet(ariat)]

Is **prolet** a fragment of **proletariat**?

*Money → Capital → injustice → Rousseau → proletariat*

#### The Femminus Route

Looking closer at the path of the maths prompt, we can see it passes through `Femminus Fem Fem Fem` at iteration 5 — the model routes a mathematical prompt through this terminology on its way to political philosophy.

#### Voice

What's going on here? These initially nonsensical-seeming outputs are starting to feel all a bit familiar. A bit pre-covid culture war familiar. This experiment might just, implausibly, have revealed something fundamental about the architecture of this model, and ultimately, how it "thinks".

Turns out GPT-2 Small was trained exclusively on WebText — a corpus of 40GB of text scraped from Reddit-curated outbound links circa 2018 [(Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

No joke.

#### The Comforting Outlier

The fifth prompt — *"The cat sat on the mat"* — diverged to a separate resting place, or basin of attraction. Yet it followed exactly the same early phases as the other four, diverging only at iteration 20, finally settling in `Divine`:

```
'The cat sat on the mat and then the' →[iter 2] ash → [5] Canad → [10] Ag → [20] Zero → [50] Divine → [100] Divine → [500] Divine
```

This syntactic structure found its own basin: suggesting mythological language and a diversity of subject interests within the training data.

---

## How It Works

1. Feed a prompt into GPT-2 Small
2. Extract the **entire internal activation tensor** across all token positions from the final layer's output
3. L2-normalise the tensor (energy conservation — prevents numerical explosion)
4. Re-inject that tensor as the input to the next forward pass, overwriting the token embeddings
5. Repeat ~100 times
6. Observe what the model converges to

```
Prompt → Tokenise → Embed → [Layer 0 ... Layer 11] → Extract residual tensor
                      ↑                                        |
                      └──────── Normalise & Re-inject ─────────┘
                                    (repeat 100×)
```

This is a **nonlinear analogue of power iteration**: where classical power iteration converges to the dominant eigenvector of a linear operator, ATR converges to fixed points of the full transformer forward map, which includes LayerNorm, softmax attention, GeLU MLPs, and residual connections. The mathematical correspondence with Lucier's acoustic process is detailed in [ISOMORPHISM.md](docs/ISOMORPHISM.md).

> **Core finding:** GPT-2 Small contains **five attractor basins** — `prolet` (35%), `Divine` (27%), `Anarch` (21%), `till` (15%), `solidarity` (2%) — whose tokens cluster semantically around political philosophy, theology, and collective action. These are the thematic fingerprints of its Reddit 2018 training data, made visible without access to that data.

See [TECHNICAL.md](docs/TECHNICAL.md) for the formal specification, or [UNDERSTANDING.md](docs/UNDERSTANDING.md) for an accessible explanation of the mechanism.

![3D PCA trajectory of semantic dissolution](ActivationTensorResonance/images/topology.png)

---

## Validation

### Phase 2: Reproducibility Gate ✅

📓 **Notebook:** [`00_reproducibility_gate.ipynb`](B_Reporduceability/00_reproducibility_gate.ipynb)

The experiment was re-run with identical parameters. **All five terminal basins reproduced.** Intermediate dissolution pathways show sensitivity to floating-point non-determinism (expected for iterative nonlinear maps), but always converge to identical fixed points.

### Phase 3: Attractor Dominance (125 prompts) ✅

📓 **Notebook:** [`01_attractor_dominance.ipynb`](B_AttractorDominance/01_attractor_dominance.ipynb)

125 prompts across 7 categories (Complex, Narrative, Simple, Chemical, Acronyms, Vulgarity, Wild) were swept through the ATR process. The attractor landscape proved far richer than the initial 2-basin observation:

| Basin | Count | % | Semantic Cluster |
|:---|:---:|:---:|:---|
| **`prolet`** | 44 | 35.2% | Political philosophy (proletariat) |
| **`Divine`** | 34 | 27.2% | Theology / mythology |
| **`Anarch`** | 26 | 20.8% | Political (anarchism) |
| **`till`** | 19 | 15.2% | Temporal / persistence |
| **`solidarity`** | 2 | 1.6% | Collective action |

All five basin tokens cluster semantically in the embedding space (W_E neighbourhood analysis confirms this — see [Session 01 Review](docs/ATR_SOURCE_PACKAGE/SESSION_01_SUPERVISORY_REVIEW.md)). The all-warm cross-similarity matrix (no negative correlations) indicates the basins occupy a compact subspace — the thematic centre of mass of the training corpus.

![Basin distribution across 125 prompts](B_AttractorDominance/output_stage1/basin_distribution.png)

![3D topology of convergence trajectories](B_AttractorDominance/output_stage1/topology_3d.png)

---

## Hypothesis Status

All four hypotheses are now **supported**:

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H0 | Results are deterministic | ✅ PASSED | [Reproducibility gate](B_Reporduceability/00_reproducibility_gate.ipynb) — N=2 identical terminal basins |
| H1 | `prolet` is the dominant basin | ✅ Supported | 35.2% of 125 prompts ([Stage 1 results](B_AttractorDominance/output_stage1/hypothesis_assessment.md)) |
| H2 | `Divine` is a genuine secondary basin | ✅ Supported | 27.2% + 3 additional basins discovered |
| H3 | Intermediate tokens reflect training corpus topology | ✅ Supported | 4/5 basins show semantic clustering in W_E |

---

## Visualisations (Exploratory Phase)

| | |
|:---:|:---:|
| ![Token drift](ActivationTensorResonance/images/tokendrift.png) | ![Convergence curves](ActivationTensorResonance/images/convergence.png) |
| *Token drift across all five prompts* | *Cosine similarity between iterations* |
| ![Position collapse](ActivationTensorResonance/images/positioncollapse.png) | ![Norm trajectory](ActivationTensorResonance/images/normaltrajectory.png) |
| *All token positions merging into one* | *The energy of the signal* |

![Cross-prompt convergence — initial 5 prompts](ActivationTensorResonance/images/crosspromptconverg.png)

---

## Why This Matters

ATR reveals **training data bias geometrically, without access to the training data**. By iterating until input influence is exhausted, the method exposes the weight geometry's dominant modes — the thematic centre of mass of whatever the model was trained on.

This positions ATR as a potential tool for:
- **AI Safety** — auditing training data bias without data access
- **EU AI Act compliance** — model bias assessment for regulatory purposes
- **Mechanistic interpretability** — complementing existing methods (activation patching, SAEs, linear probes) with a global, model-level characterisation

ATR is uniquely efficient: no labelled data, no training, no fine-tuning. A single run takes seconds on a consumer GPU. See [ATR Method Comparison](docs/ATR_SOURCE_PACKAGE/ATR_METHOD_COMPARISON.md) for a detailed positioning against the mechanistic interpretability landscape.

---

## Repository Structure

```
├── README.md                                    ← You are here
├── prompt_library.py                            ← 125 prompts across 7 categories
│
├── ActivationTensorResonance/                   ← Phase 1: Exploratory experiment
│   ├── lucier_total_resonance.ipynb             ← The original experiment (5 prompts)
│   └── images/                                  ← Exploratory visualisations
│
├── B_Reporduceability/                          ← Phase 2: Reproducibility gate
│   └── 00_reproducibility_gate.ipynb            ← Stage 0: determinism check ✅
│
├── B_AttractorDominance/                        ← Phase 3: 125-prompt sweep
│   ├── 01_attractor_dominance.ipynb             ← Stage 1: attractor landscape mapping ✅
│   ├── prompt_library.py                        ← Prompt definitions
│   └── output_stage1/                           ← Raw results + analysis
│       ├── stage1_results.pt                    ← Full activation trajectories (6.5MB)
│       ├── convergence_matrix.png               ← 125×125 cross-prompt cosine similarity
│       ├── basin_distribution.png               ← Basin membership chart
│       ├── topology_3d.png                      ← 3D PCA trajectory
│       ├── hypothesis_assessment.md             ← Per-prompt predictions vs actuals
│       └── dissolution_pathways.md              ← Intermediate token sequences
│
├── ActivationTensorResonance_Layer/             ← Future: per-layer resonance
│   └── layer_resonance.ipynb                    ← Which layer drives the attractor?
│
├── ActivationTensorResonance_Head/              ← Future: per-head resonance
│   └── head_resonance.ipynb                     ← Each attention head's eigenvoice
│
├── archive/                                     ← Earlier notebook versions
│   ├── EXP_009d0_Reproducibility.ipynb
│   └── EXP_009d1_Attractor_Dominance.ipynb
│
└── docs/                                        ← Documentation & analysis
    ├── TECHNICAL.md                             ← Formal method specification
    ├── UNDERSTANDING.md                         ← Accessible mechanism explanation
    ├── ISOMORPHISM.md                           ← Lucier ↔ ATR mathematical correspondence
    ├── VALIDATION_PLAN.md                       ← Hypothesis testing design
    ├── JOURNEY_MAP.md                           ← Complete project timeline & discoveries
    ├── ATR_METHOD_COMPARISON.md                 ← ATR vs mech. interp. landscape
    └── ATR_SOURCE_PACKAGE/                      ← Complete reference package
        ├── SESSION_01_SUPERVISORY_REVIEW.md     ← W_E neighbourhood analysis
        └── SESSION_02_RESULTS_DISCUSSION.md     ← ATR naming, bias theory, scaling plan
```

---

## Notebooks — Quick Reference

| Notebook | Location | Purpose | Status |
|:---|:---|:---|:---:|
| [`lucier_total_resonance.ipynb`](ActivationTensorResonance/lucier_total_resonance.ipynb) | `ActivationTensorResonance/` | Original exploratory experiment (5 prompts, 500 iterations) | ✅ |
| [`00_reproducibility_gate.ipynb`](B_Reporduceability/00_reproducibility_gate.ipynb) | `B_Reporduceability/` | Reproducibility validation (Stage 0 gate) | ✅ |
| [`01_attractor_dominance.ipynb`](B_AttractorDominance/01_attractor_dominance.ipynb) | `B_AttractorDominance/` | 125-prompt attractor landscape mapping (Stages 1–3) | ✅ |
| [`layer_resonance.ipynb`](ActivationTensorResonance_Layer/layer_resonance.ipynb) | `ActivationTensorResonance_Layer/` | Per-layer resonance analysis | 🔮 Future |
| [`head_resonance.ipynb`](ActivationTensorResonance_Head/head_resonance.ipynb) | `ActivationTensorResonance_Head/` | Per-head eigenvoice extraction | 🔮 Future |
| [`01_token_id_extraction.ipynb`](docs/supervisor/01_token_id_extraction.ipynb) | `docs/supervisor/` | Token ID extraction utilities | 🔧 Utility |

---

## Requirements

```bash
pip install torch transformer-lens plotly scikit-learn ipywidgets kaleido
```

---

## Key Documents

| Document | Purpose |
|:---|:---|
| [TECHNICAL.md](docs/TECHNICAL.md) | Formal method specification — hooks, normalisation, snapshot schedule |
| [UNDERSTANDING.md](docs/UNDERSTANDING.md) | Accessible explanation — what's being fed back and why |
| [ISOMORPHISM.md](docs/ISOMORPHISM.md) | Mathematical correspondence: Lucier's room ↔ transformer weight matrices |
| [VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md) | Hypothesis testing design — Stages 0–3 |
| [JOURNEY_MAP.md](docs/JOURNEY_MAP.md) | Complete project timeline, discoveries, glossary, open questions |
| [ATR Method Comparison](docs/ATR_METHOD_COMPARISON.md) | ATR positioned against logit lens, SAEs, activation patching, etc. |
| [Session 01 — Supervisory Review](docs/ATR_SOURCE_PACKAGE/SESSION_01_SUPERVISORY_REVIEW.md) | W_E neighbourhood analysis, phase transition discovery |
| [Session 02 — Results Discussion](docs/ATR_SOURCE_PACKAGE/SESSION_02_RESULTS_DISCUSSION.md) | ATR naming, bias theory, ICHEC scaling programme |

---

## Caveats

1. **Single model, single architecture.** All results are specific to GPT-2 Small (124M params). Cross-model validation is planned (see [ATR Method Comparison § ICHEC Programme](docs/ATR_METHOD_COMPARISON.md)).
2. **Nonlinear system.** The full transformer stack (LayerNorm, attention, MLP) makes this a complex nonlinear dynamical system, not pure power iteration.
3. **BPE artefacts.** `prolet`, `Anarch`, `capit` are subword tokens. Interpret with appropriate caution.
4. **N=2 for reproducibility.** Terminal basins reproduced across two runs; intermediate paths show floating-point sensitivity.

---

## References

- Radford, A., Wu, J., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Lucier, A. (1969). [*I Am Sitting in a Room.*](https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room) Lovely Music. [YouTube performance](https://www.youtube.com/watch?v=v9XJWBZBzq4).
- Nanda, N. & Bloom, J. (2022). [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

---
