# Project ATR: Complete Journey Map

**Purpose:** Continuity document. Pick up the intellectual thread from any point.
**Last updated:** 2026-07-31 (alignment-review record repair; series closed 2026-07-10)

---

## 1. Timeline: The Intellectual Arc

### Phase 0: The Inspiration (Pre-experiment)
- **Seed:** Alvin Lucier's *I Am Sitting in a Room* (1969): iterative feedback of sound through a room dissolves speech into the room's resonant frequencies
- **Leap:** "What if we did this to a language model?" Feed the model's internal state back through itself, bypassing the text bottleneck
- **Key insight:** Don't loop the *text output* (argmax → single token). Loop the *activation tensor* (full 768-dimensional state across all positions). Preserve the superposition.
- **Model choice:** GPT-2 Small (124M params): well-studied, manageable, known training data (WebText/Reddit 2018)

### Phase 1: The Exploratory Experiment (EXP_009aFIX)
- **Method:** Extract residual stream at Layer 11 → L2 normalise → re-inject at Layer 0 → repeat 500×
- **5 test prompts:** question, factual, grammatical, nonsense, command
- **Discovery 1:** 4/5 prompts converge to the SAME terminal token: `prolet` (BPE fragment of "proletariat")
- **Discovery 2:** The 5th prompt ("The cat sat on the mat") diverges at iteration 20 to `Divine`
- **Discovery 3:** All prompts follow a *shared dissolution pathway* through recognisable tokens: `ash → Canad → Ag → FT → capit → injustice → Rousse → prolet`
- **Discovery 4:** The training data is exclusively Reddit 2018; the basin tokens read as that corpus's discourse (interpretation later qualified, see Phase 5)
- **Output:** README, TECHNICAL.md, UNDERSTANDING.md, ISOMORPHISM.md, visualisations (PCA topology, token drift, convergence curves, position collapse, norm trajectory)

### Phase 2: Validation Design (VALIDATION_PLAN)
- **Stage 0, Reproducibility Gate:** Does re-running produce identical results?
- **Stage 1, Attractor Dominance:** How dominant is `prolet`? Test with 125 diverse prompts
- **Stage 2, Secondary Basin Mapping:** Are there more basins beyond `prolet` and `Divine`?
- **Stage 3, Dissolution Pathway Analysis:** Is the intermediate pathway consistent?
- **Prompt Library:** 125 prompts across 7 categories (Complex, Narrative, Simple, Chemical, Acronyms, Vulgarity, Wild)

### Phase 3: Validation Execution
- **EXP_009d0 (Determinism check):** Same-machine repeatability supported. All 5 terminal basins identical across N=2 runs. Intermediate paths show floating-point sensitivity but always converge to same fixed points. Independent re-implementation has not been attempted.
- **EXP_009d1 (Attractor Dominance, 125 prompts):** Complete.
  - 5 basins discovered: `prolet` (35.2%), `Divine` (27.2%), `Anarch` (20.8%), `till` (15.2%), `solidarity` (1.6%)
  - Per-prompt prediction was poor (~25% match); structural finding (basins exist, these are their shares) supported, predictive finding not.
  - `stage1_results.pt` saved (6.5MB): complete activation trajectories for all 125 prompts

### Phase 4: Supervisory Analysis (Today, 2026-03-20)
- **Session 01:** Hypothesis framework reinstated (H0–H3). Interloper hypothesis removed. Four independent observations identified. Slonski glossary created. Goldmine analysis of `.pt` data planned. Slonski comparison experiment designed.
- **Priority Analysis 01 (Embedding Neighbourhood Test):** All 14 tokens analysed.
  - **H3 SUPPORTED:** 4/5 basins show strong semantic clustering in W_E
  - **`capit` correction:** Clusters as capitulation/surrender, NOT capitalism
  - **Phase transition discovered:** structural → semantic, transition at `capit`
  - **All-warm cross-similarity:** All 14 tokens positively correlated (0.18–0.47): compact subspace
- **Session 02:** Mixing Time analogy formalised. Bias interpretation. ATR named. Cross-model programme sketched.

### Phase 5: Cross-Model Validation & Series Close (2026-07-10)
- **Cross-model sweeps (gpt2-medium, pythia-160m, pythia-410m):** landscapes are model-specific: one empty-token funnel (`D`), one near-total funnel (`questioned`), one non-consolidating scatter. Fingerprint hypothesis refuted (same corpus ≠ same landscape).
- **Null model:** random tensors converge to 18 non-semantic basins, ~zero overlap with the real five: basins belong to the language-driven regime, not the weights in general. *Inverted 2026-07-31: this arm ran mis-calibrated and was counted before convergence (FINDINGS caveat 18); the matched-ν, gated re-run (run 17) finds noise landing in the language arm's own basins, all five reappearing at the trials' smallest passing lag, so at this injection scale the basins belong to the weights (FINDINGS F4).*
- **Convergence-gated re-sweep:** basins survive proper convergence; `Anarch` was over-counted at iter 100 (corrected shares: prolet 43.2 / Divine 27.2 / till 15.2 / Anarch 13.6 / solidarity 0.8). Pre-registered `till`-transient hypothesis refuted (19/19 stable).
- **The `Divine` dissociation:** the 34 prompts that fail the lag-1 convergence gate are exactly the `Divine` prompts: stable readout over a tensor later resolved as an exact period-2 limit cycle (FINDINGS.md F9), converged under a lag-2 gate for the audited trajectory (F15).
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
| 10 | The all-warm property is consistent with a compact "thematic-centre-of-mass" interpretation | Session 02 | All 91 off-diagonal pairs positively correlated in W_E (0.18–0.47). *Retired: the corpus-causal reading was refuted cross-model at series close (see 13), and the permutation test (2026-07-11) showed the all-warm property is an anisotropy artifact (see 9). Only the recorded cosine values stand.* |
| 11 | All normalised transformers must have basins (Brouwer fixed-point theorem) | Session 02 | Continuous map on compact set (LayerNorm bounds): existence guaranteed; count and shape are empirical questions. *Corrected 2026-07-23: the Brouwer argument is inapplicable as stated. The theorem requires a compact convex domain; the L2 shell is a sphere, not convex. Attractor existence here is an empirical observation, not a theorem.* |
| 12 | Basin landscapes are model-specific, not corpus-tracking | Cross-model (2026-07) | Same corpus (WebText): Small → 5 semantic basins; Medium → 1 empty token |
| 13 | The five basins are regime-specific, not weight-universal | Null model (2026-07) | Noise → 18 non-semantic basins, ~0 overlap; real count 5 below random CI [11,17]. *Inverted 2026-07-31: the noise arm was mis-calibrated and counted pre-convergence (caveat 18); the matched-ν, gated re-run (run 17) sends noise into the language arm's own basins (all five at the trials' smallest passing lag), `prolet` dominant among settled trials in both arms, so at this injection scale the basins are weight-native (FINDINGS F4).* |
| 14 | Basin labels survive convergence gating, with two corrections | Gated re-sweep (2026-07) | 73% pass the lag-1 gate by iter 120; ~10 prompts move Anarch→prolet, 1 moves solidarity→Anarch |
| 15 | `Divine` is a readout-stable / tensor-unsettled object | Gated + diagnostic (2026-07) | 34/34 lag-1 gate failures are `Divine`; decode constant while tensor moves. *Resolved 2026-07-19: an exact period-2 limit cycle (FINDINGS F9); fails the lag-1 gate by construction, converged at lag 2 for the audited trajectory (F15).* |
| 16 | Cross-model differences are intrinsic dynamics, not apparatus | Diagnostics (2026-07) | cos_sim_mean verdicts are tensor-level; normalisation inert up to LayerNorm's epsilon term. *Amended 2026-07-28: the inertness half is withdrawn pending re-derivation — LayerNorm ignores a global rescale but the residual path around it does not, so the block is not invariant (FINDINGS caveat 7; `experiments/preln_rescale_check.py`). The tensor-level half is unaffected, and §1.1's separate ruling that the rescale preserves the mix — and so is not the Pythia-410m fragmentation source — still stands.* |

---

## 3. Hypotheses: Status

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H0 | Results are deterministic | Repeatability supported | EXP_009d0: N=2 same-machine runs produce identical terminal basins. Independent re-implementation pending. |
| H1 | `prolet` is the dominant basin | Supported, revised upward | 43.2% at convergence (gated re-sweep); was 35.2% at iter 100. `Anarch` was over-counted pre-convergence. Per-prompt prediction remained poor (~25%). |
| H2 | `Divine` is a genuine secondary basin | Supported with qualification | 27.2%; fails the lag-1 gate by construction: an exact period-2 limit cycle, argmax stable in both phases (FINDINGS.md F9); converged under a lag-2 gate for the audited trajectory (F15). |
| H3 | Intermediate tokens reflect training corpus topology | Weakened at close; coherence half moved one level down | The permutation test was run and came back **negative** (2026-07-11): the all-warm W_E matrix is embedding-space anisotropy, not a special subspace (Discoveries 9–10 above; FINDINGS caveat 4). The corpus-causal reading failed cross-model: GPT-2 Medium, same corpus, no semantic basins (FINDINGS.md F3). Null model re-run 2026-07-31 (run 17): at matched injection scale and gated convergence, noise falls into the language arm's own basins (all five at the trials' smallest passing lag), so the regime-specific reading is inverted (FINDINGS F4 and caveat 18). The distribution-level coherence replacement (F8) is itself under challenge (FINDINGS caveat 18b). |
| H4 | Per-head resonance ≈ SVD dominant singular vector | Not supported as registered; superseded by the eigenvector rescore (2026-07-31, TC ruling in #54) | Executed 2026-07-25 (`spectral_resonance.ipynb`, issue #25 regeneration): 5/144 heads above 0.9. The registered target was the wrong object: the isolated loop is pure power iteration on W_OV transposed, whose limit is that operator's dominant eigenvector (a dominant left eigenvector of W_OV), not W_OV's top singular vector. Rescored against that eigenvector, every settling head lands where the weights predict; the five registered passes are the heads where the two directions coincide, led by the F14/F17 head L11.H8 (FINDINGS.md §3 H4; run 16, `experiments/gpt2_small/output_eigen_rescore/report.md`) |
| H-fingerprint | Basin profiles read training bias from any model | **Refuted** | FINDINGS.md F3, F4 |

Canonical dispositions with full evidence: [FINDINGS.md](FINDINGS.md) §3.

---

## 4. Adjacent Science & Mathematics

| Domain | Concept | Relevance to ATR |
|---|---|---|
| **Linear Algebra** | Power iteration | ATR is the nonlinear analogue: iterated operator application converges to dominant modes |
| **Dynamical Systems** | Fixed-point theory, basin of attraction | The mathematical framework for what ATR reveals |
| **Topology** | Brouwer fixed-point theorem | Motivating analogy only: its hypotheses do not hold on the L2 shell; attractor existence here is empirical, not guaranteed |
| **Acoustics** | Mixing time (T_mix) | Isomorphic to ATR's structural→semantic phase transition |
| **Acoustics** | Impulse response / room modes | Lucier's room ↔ transformer weight matrices |
| **Fractal Geometry** | Fractal dimensional analysis | Potential metric for basin characterisation (untested) |
| **BPE/Tokenisation** | Byte Pair Encoding | Why attractors appear as fragments (`prolet`, not `proletariat`) |
| **Mechanistic Interp.** | Activation patching, probing, SAEs | Adjacent methods ATR complements |
| **Mechanistic Interp.** | Logit Lens / Tuned Lens | Per-layer prediction; ATR reveals per-model global structure |
| **Prior Art** | Slonski Q-vector dichotomy | Binary polarisation in W_Q: may be coarser version of ATR basins (untested prediction) |
| **Prior Art** | Turner et al., Representation Engineering | Activation steering (single-pass); ATR iterates to convergence |
| **Prior Art** | Shumailov et al., Model Collapse | Text-level self-feeding; ATR is activation-level (lossless) |
| **Philosophy** | Deleuze, Body without Organs | The undifferentiated substrate (weight geometry before prompt input) |
| **Dev. Biology** | Levin, TAME (morphogenesis) | Attractor basins as body plan of the model |

---

## 5. Glossary

| Term | Definition | First Appearance |
|---|---|---|
| **ATR** (Activation Tensor Resonance) | Iterative re-injection of a model's residual stream through its forward pass to reveal the attractor landscape of its iterated dynamics (regime-dependent, see FINDINGS.md F4) | Session 02 |
| **Attractor basin** | A region of activation space where all initial conditions converge to the same terminal state under ATR | EXP_009aFIX |
| **Basin token** | The terminal BPE token a basin converges to (e.g., `prolet`, `Divine`) | EXP_009aFIX |
| **Waypoint token** | An intermediate token observed during the dissolution pathway | Session 01 |
| **Dissolution pathway** | The sequence of decoded tokens observed as a prompt iterates toward its attractor | EXP_009aFIX |
| **Phase transition (structural→semantic)** | The point where prompt-specific information is lost and training corpus topology dominates | Session 01 |
| **T_mix_LLM** | Proposed metric: iteration at which prompts heading to the same basin become indistinguishable | Session 02 |
| **Nonlinear power iteration** | What ATR actually is mathematically: repeated application of a nonlinear operator | TECHNICAL.md |
| **Residual stream** | The shared [seq_len × 768] vector space through which all transformer components read/write | UNDERSTANDING.md |
| **W_E** | The token embedding matrix: maps vocabulary indices to 768-D vectors | Session 01 |
| **BPE** | Byte Pair Encoding: GPT-2's tokenisation scheme (50,257 subword tokens) | README |
| **Position collapse** | Phenomenon where all token positions converge to identical vectors (~iteration 10) | TECHNICAL.md |
| **Cross-prompt invariance** | Property where different prompts produce near-identical final states (cosine sim > 0.999) | TECHNICAL.md |
| **L2 normalisation** | Energy conservation: rescale tensor to initial norm each iteration, preventing explosion | TECHNICAL.md |
| **All-warm matrix** | Cross-similarity matrix with no negative values: indicates compact attractor subspace | Session 02 |
| **Eigenvoice** | Metaphor (art register): the model's "native voice" under iteration. The reporting-register correction: the voice depends on what drove it (FINDINGS.md F4) | ISOMORPHISM.md |
| **Q-vector dichotomy** | Slonski's finding: token Q-vectors polarise into 2 groups at cosine similarity ≈ -1 | Session 01 |
| **Glitch token** | Anomalous BPE tokens with unusual embedding properties (the SolidGoldMagikarp family). Ruled out for the basin *identities* (Session 01), then ruled back **in** for the `Divine` object: the cycle's phase-B pole aligns with the under-trained token core at cos +0.596, p < 0.001 (FINDINGS F10, F13) | Session 01; F13 (2026-07-19) |
| **Bias profile** | *Retired term.* Originally: basin distribution as a fingerprint of training data themes, refuted at series close (FINDINGS.md F3) | Session 02 |

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
Logarithmic schedule `[0, 2, 3, 5, 10, 20, 50, 100, 250, 500]` captures both early (prompt-dependent) and late (system-dependent) dynamics. This is directly analogous to impulse response measurement in acoustics: early reflections need temporal resolution, late reverberation needs duration.

### 6e. The Prompt Library as Experimental Design
125 prompts across 7 categories (Complex/Narrative/Simple/Chemical/Acronyms/Vulgarity/Wild): systematic coverage of the input space. The categories were designed to test register-dependence while the 30 "Wild" prompts stress-test boundaries (punctuation, emoji, mixed-register, non-English, adversarial).

### 6f. The Two-Phase Architecture of Discovery
Every ATR experiment has two phases:
1. **Data generation** (fast, cheap, parallelisable): iterate and save tensors
2. **Interpretation** (slow, human-dependent, rich): analyse neighbourhoods, cross-similarities, trajectories

The bottleneck is always phase 2. Automation of interpretation is the scaling challenge.

---

## 7. Open Questions

| Question | Status | Next Step |
|---|---|---|
| Why does GPT-2 Small, alone in this set, resolve language into few semantic basins? | **The open question of the series** | New experimental stage |
| ~~What is the `Divine` object: limit cycle, wandering attractor, decode-region plateau?~~ | Answered (2026-07-19): an exact period-2 limit cycle (FINDINGS.md F9), converged under a lag-2 gate for the audited trajectory (F15) | Re-gate the other 33 prompts (unblocked: library restored, issue #24; run queued) |
| Does the landscape depend on where the loop is cut (layer window / depth)? | Designed, not run | Pythia-410m depth control (0–11 vs 0–23); window sweeps |
| ~~W_E semantic-clustering statistics~~ | Answered (2026-07-11): all-warm matrix is an anisotropy artifact: 99.9% of random 14-token sets are also all-positive; compact-subspace reading withdrawn. Neighbourhood claim remains qualitative | - |
| True lock-in iterations (gate fired at its floor, 120) | Pending | Finer gate cadence |
| What is T_mix_LLM for each basin? | Measurable from existing data | Compute from `.pt` |
| Are all basins in one Slonski macro-group? | Untested; the all-warm premise of the prediction was retired 2026-07-11 (anisotropy artifact) | One Q-vector experiment, on its own terms |
| Is the fractal dimension of convergence trajectories basin-specific? | Speculative | Requires T_mix first |
| ~~Does ATR scale to larger models?~~ | Answered: the operation runs; the landscape changes qualitatively | - |
| ~~Do different models have different basin profiles?~~ | Answered: yes, drastically (FINDINGS.md F3) | - |
| ~~Can basin depth predict bias strength?~~ | Retired with the fingerprint hypothesis | - |

---

*This document is a living map. Series closed 2026-07-10; corrections applied 2026-07-23 and 2026-07-31.*
