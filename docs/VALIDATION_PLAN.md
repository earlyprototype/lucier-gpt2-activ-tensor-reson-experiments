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
