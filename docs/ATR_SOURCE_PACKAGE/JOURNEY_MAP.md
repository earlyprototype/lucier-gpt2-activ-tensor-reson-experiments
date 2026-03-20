# Project ATR: Complete Journey Map

**Purpose:** Continuity document. Pick up the intellectual thread from any point.
**Last updated:** 2026-03-20

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
- **Discovery 4:** The training data is exclusively Reddit 2018. The model's "naked voice" speaks the discourse of its training corpus
- **Output:** README, TECHNICAL.md, UNDERSTANDING.md, ISOMORPHISM.md, visualisations (PCA topology, token drift, convergence curves, position collapse, norm trajectory)

### Phase 2: Validation Design (VALIDATION_PLAN)
- **Stage 0 — Reproducibility Gate:** Does re-running produce identical results?
- **Stage 1 — Attractor Dominance:** How dominant is `prolet`? Test with 125 diverse prompts
- **Stage 2 — Secondary Basin Mapping:** Are there more basins beyond `prolet` and `Divine`?
- **Stage 3 — Dissolution Pathway Analysis:** Is the intermediate pathway consistent?
- **Prompt Library:** 125 prompts across 7 categories (Complex, Narrative, Simple, Chemical, Acronyms, Vulgarity, Wild)

### Phase 3: Validation Execution
- **EXP_009d0 (Reproducibility):** ✅ PASSED. All 5 terminal basins identical. Intermediate paths show floating-point sensitivity but always converge to same fixed points.
- **EXP_009d1 (Attractor Dominance, 125 prompts):** ✅ Complete.
  - 5 basins discovered: `prolet` (35.2%), `Divine` (27.2%), `Anarch` (20.8%), `till` (15.2%), `solidarity` (1.6%)
  - `stage1_results.pt` saved (6.5MB) — complete activation trajectories for all 125 prompts

### Phase 4: Supervisory Analysis (Today — 2026-03-20)
- **Session 01:** Hypothesis framework reinstated (H0–H3). Interloper hypothesis removed. Four independent observations identified. Slonski glossary created. Goldmine analysis of `.pt` data planned. Slonski comparison experiment designed.
- **Priority Analysis 01 (Embedding Neighbourhood Test):** All 14 tokens analysed.
  - **H3 SUPPORTED:** 4/5 basins show strong semantic clustering in W_E
  - **`capit` correction:** Clusters as capitulation/surrender, NOT capitalism
  - **Phase transition discovered:** structural → semantic, transition at `capit`
  - **All-warm cross-similarity:** All 14 tokens positively correlated (0.18–0.47) — compact subspace
- **Session 02:** Mixing Time analogy formalised. Bias interpretation. ATR named. Cross-model programme sketched. ICHEC compute access identified.

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
| 9 | All attractor tokens occupy same compact subspace (all-warm matrix) | Session 01 | Cross-similarity 0.18–0.47, no negative values |
| 10 | The all-warm property reveals inherent training corpus bias | Session 02 | Geometric manifestation: weight geometry's dominant modes = training data themes |
| 11 | All models must have basins (Brouwer fixed-point theorem) | Session 02 | Continuous map on compact set (LayerNorm bounds) |

---

## 3. Hypotheses — Status

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H0 | Results are deterministic | ✅ PASSED | EXP_009d0 |
| H1 | `prolet` is the dominant basin | ✅ Supported | 35.2% of 125 prompts |
| H2 | `Divine` is a genuine secondary basin | ✅ Supported | 27.2% + 3 more basins |
| H3 | Intermediate tokens reflect training corpus topology | ✅ Supported | 4/5 basins semantic in W_E |

---

## 4. Adjacent Science & Mathematics

| Domain | Concept | Relevance to ATR |
|---|---|---|
| **Linear Algebra** | Power iteration | ATR is the nonlinear analogue — iterated operator application converges to dominant modes |
| **Dynamical Systems** | Fixed-point theory, basin of attraction | The mathematical framework for what ATR reveals |
| **Topology** | Brouwer fixed-point theorem | Guarantees every normalised transformer has at least one ATR attractor |
| **Acoustics** | Mixing time (T_mix) | Isomorphic to ATR's structural→semantic phase transition |
| **Acoustics** | Impulse response / room modes | Lucier's room ↔ transformer weight matrices |
| **Fractal Geometry** | Fractal dimensional analysis | PI's Master's thesis — potential metric for basin characterisation |
| **BPE/Tokenisation** | Byte Pair Encoding | Why attractors appear as fragments (`prolet`, not `proletariat`) |
| **Mechanistic Interp.** | Activation patching, probing, SAEs | Adjacent methods ATR complements |
| **Mechanistic Interp.** | Logit Lens / Tuned Lens | Per-layer prediction; ATR reveals per-model global structure |
| **Prior Art** | Slonski Q-vector dichotomy | Binary polarisation in W_Q — may be coarser version of ATR basins |
| **Prior Art** | Turner et al. — Representation Engineering | Activation steering (single-pass); ATR iterates to convergence |
| **Prior Art** | Shumailov et al. — Model Collapse | Text-level self-feeding; ATR is activation-level (lossless) |
| **AI Safety** | Training data bias detection | ATR reveals training bias geometrically, without training data access |
| **EU Regulation** | AI Act — bias auditing | ATR as potential compliance tool for model bias assessment |
| **Philosophy** | Deleuze — Body without Organs | The undifferentiated substrate (weight geometry before prompt input) |
| **Dev. Biology** | Levin — TAME (morphogenesis) | Attractor basins as body plan of the model |

---

## 5. Glossary

| Term | Definition | First Appearance |
|---|---|---|
| **ATR** (Activation Tensor Resonance) | Iterative re-injection of a model's residual stream through its forward pass to reveal weight geometry attractors | Session 02 |
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
| **Eigenvoice** | Metaphor: the model's "native voice" revealed when input is exhausted — its dominant eigenmodes | ISOMORPHISM.md |
| **Q-vector dichotomy** | Slonski's finding: token Q-vectors polarise into 2 groups at cosine similarity ≈ -1 | Session 01 |
| **Glitch token** | Anomalous BPE tokens with unusual embedding properties (e.g., SolidGoldMagworthy) — ruled out for our basins | Session 01 |
| **Bias profile** | The distribution of attractor basins as a geometric fingerprint of training data themes | Session 02 |

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
| What is T_mix_LLM for each basin? | Measurable from existing data | Compute from `.pt` |
| Do basins sort cleanly in the convergence matrix? | Testable | Reorder by terminal basin |
| Are all basins in one Slonski macro-group? | Predicted (from all-warm) | One Q-vector experiment |
| Does ATR scale to larger models? | Untested | Run on GPT-2 Medium locally |
| Do different models have different basin profiles? | Untested | Cross-model ATR programme |
| Can basin depth predict bias strength? | Theoretical | Requires cross-model data |
| What statistical validation is sufficient? | Planned | Random baseline + permutation test |
| Is the fractal dimension of convergence trajectories basin-specific? | Speculative | Requires T_mix first |

---

*This document is a living map. Update after each session.*
