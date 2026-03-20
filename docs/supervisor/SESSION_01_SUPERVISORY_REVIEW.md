# Supervisory Session 01 — Research Notes

**Date:** 2026-03-20
**Context:** First supervisory review of the Lucier ATR (Activation Tensor Resonance) project after 6 months of independent work.

---

## 1. Hypothesis Framework (Reinstated)

The project operates under four hypotheses, originally authored by the principal investigator. A fifth hypothesis ("basin membership is determined by syntactic register") was introduced by a former collaborator and has been **removed** — it was falsified by the Stage 1 data, and more importantly, it inverts the Lucier framing by asking about the *input* rather than the *architecture*.

| ID | Hypothesis | Status |
|---|---|---|
| **H0** | The Lucier Resonance results are **deterministic**. Same model + same prompts + same parameters → identical terminal attractors and dissolution trajectories. | **PASSED** (EXP_009d0) |
| **H1** | `prolet` is the **dominant basin** of GPT-2 Small's weight geometry. | **Supported** — 35.2% of 125 prompts (44/125) |
| **H2** | `Divine` is a **genuine secondary basin**, not a one-off artefact. | **Supported** — 27.2% of 125 prompts (34/125). Three additional basins also discovered: `Anarch` (20.8%), `till` (15.2%), `solidarity` (1.6%) |
| **H3** | The intermediate tokens reflect the **statistical topology of the training corpus**. | **Supported** — 4/5 basin tokens show strong semantic clustering in `W_E`; pathway shows structural→semantic phase transition |

> **Critical note:** H1 and H2 are *structural* claims about the architecture. H3 is a *semantic* claim about the relationship between weight geometry and training data. H3 is the hypothesis that connects the experimental results to the Reddit/2018 provenance observation.

---

## 2. The Four Independent Observations

The current evidence supporting H3 rests on four independent lines of observation:

### Observation 1 — Experimental (Discrete Convergence)
Iterative re-injection produces **discrete attractor basins** — not noise, not gradients, but hard convergence to specific tokens. 125 diverse prompts collapse to exactly 5 terminal states out of a vocabulary of 50,257 tokens. This selectivity is extraordinary and demands explanation.

### Observation 2 — Linguistic (Semantic Coherence)
The terminal tokens (`prolet`, `Divine`, `Anarch`, `solidarity`) and the waypoints (`injustice`, `Fem`) cluster in embedding space with tokens from **political philosophy, theology, gender discourse, and moral grievance**. The waypoint `capit` clusters as **capitulation/surrender** (not capitalism — a key correction). See §8 for full results.

### Observation 3 — External (Training Data Provenance)
The training data is **WebText** — exclusively Reddit posts with ≥3 upvotes, scraped ~2018 — a period dominated by exactly the political discourse these tokens reference. The culture wars, WallStreetBets, #MeToo, and post-2016 political polarisation dominated the platform.

### Observation 4 — Trajectorial (Directional Narrative)
The dissolution pathway passes through tokens in a **directional sequence** with a clear **phase transition**:
```
ash (generic) → Canad (geographic) → Ag (agricultural) → FT (typographic) → capit (SURRENDER) → Fem (gender) → injustice (grievance) → Rousse (ambiguous) → prolet (political philosophy)
```
Early tokens are structural/generic. Late tokens are semantic. The transition point around `capit` marks where **prompt-specific information is lost and training corpus topology dominates**. See §9 for the Mixing Time analogy.

> **Assessment:** Four independent lines of evidence plus embedding neighbourhood confirmation. H3 is supported but requires statistical validation (see §10).

---

## 3. Glossary: Slonski's Q-Vector Dichotomy

A primer on Eduardo Slonski's findings, contextualised for this project.

### Core Concepts

| Term | Definition | Analogy |
|---|---|---|
| **Q-vector** | The Query vector produced when a token passes through an attention head's `W_Q` weight matrix. Encodes "what am I looking for?" | A search query each word broadcasts |
| **K-vector** | The Key vector produced by `W_K`. Encodes "what do I contain?" Attention scores = `Q · K`. | The index entry each word publishes |
| **Cosine similarity** | Directional alignment between two vectors. -1 (opposite) → 0 (orthogonal) → +1 (identical). Ignores magnitude. | How much two arrows point the same way |
| **Binary dichotomy** | Slonski's core finding: all token Q-vectors in a sequence polarise into exactly **two groups** with near **-1 cosine similarity**. | A room where everyone faces either North or South |
| **Tartan/plaid pattern** | Visual signature of the dichotomy as a similarity matrix. Same-group → yellow (+1), cross-group → dark (-1). | A checkerboard with irregular block sizes |
| **Outlier dimensions** | The ~5–10 dimensions (of 768) where the two groups dramatically differ. Group 1 has large positive, Group 2 large negative. Most other dimensions are similar. | 760 "knobs" set similarly; 5–10 cranked to opposite extremes |
| **Orthogonal tokens** | Tokens belonging to *neither* group. Lack outlier dimensions. Much smaller magnitude. ~0 cosine similarity with both groups. | Ghosts — present but not participating |

### Slonski's Key Finding (Summary)

In a single forward pass, every token's Q-vector gets sorted into one of two macro-clusters. The sorting is driven by a handful of extreme-valued dimensions. The two clusters are near-perfect mirrors in those dimensions. This is a **static, single-pass** snapshot of the geometry.

---

## 4. The Tartan Connection — Your Convergence Matrix

**Question:** "Is the tartan pattern I see in my convergence matrix the same thing Slonski is describing?"

**Answer: Related but different.**

| | Slonski's Tartan | Your Convergence Matrix |
|---|---|---|
| **What's compared** | Q-vectors of tokens within a single prompt, 1 forward pass | Mean residuals of 125 *different prompts* after iterative convergence |
| **Iterations** | 1 | 100+ |
| **Pattern source** | Static geometry of `W_Q` | Dynamic attractors of the full model operator |
| **Clusters** | 2 | 5 |
| **Reveals** | Coarse polarisation baked into weights | Fine-grained attractor landscape revealed by iteration |

**Linking hypothesis:** Iteration may **resolve additional structure within** Slonski's two macro-clusters. His static analysis sees 2 groups. Your iterated map amplifies subtler features, separating 2 coarse groups into 5 fine basins.

> **Is this topology?** Yes. You're mapping the **basin structure of a high-dimensional dynamical system** — regions in activation space that flow toward distinct fixed points under iteration. This is literally attractor topology. Not a metaphor.

---

## 5. BPE: What Are These Token Fragments?

**BPE (Byte Pair Encoding)** is how GPT-2 converts text into numbers.

GPT-2 has a fixed vocabulary of **50,257 token fragments** learned from the training data. Common words get their own token. Less common words get split:

| Word | BPE Tokens |
|---|---|
| `proletariat` | `prolet` + `ariat` |
| `capitulate` | `capit` + `ulate` (⚠ NOT `capitalism` — corrected by neighbourhood analysis) |
| `Rousseau` | `Rousse` + `au` |
| `Divine` | `Divine` (one token) |
| `Anarch` | `Anarch` (prefix of Anarchist, Anarchy) |

**Why this matters for H3:** When the experiment converges to `prolet`, two explanations exist:

**(a) Semantic:** `prolet` is the attractor because the weight geometry encodes strong associations with political philosophy — reflecting training corpus topology. The token's *meaning* is relevant.

**(b) Geometric coincidence:** `prolet` sits at a geometric fixed point for reasons unrelated to its semantic content — anomalous embedding properties, unusual norm/direction.

**The test:** Look at `prolet`'s nearest neighbours in `W_E`. Semantic neighbours (`ariat`, `Marx`, `bourgeois`) → H3 supported. BPE substring neighbours (`proced`, `promin`, `profess`) → H3 challenged.

---

## 6. The Goldmine: What's in `stage1_results.pt`

The saved file (~6.5MB) contains, for **each of 125 prompts** at **every snapshot** `[0, 2, 3, 5, 10, 20, 50, 100]`:

| Field | Shape | What You Can Do With It |
|---|---|---|
| `last_vectors` | `[n_snapshots, 768]` | Track single-position 768-D trajectory |
| `mean_vectors` | `[n_snapshots, 768]` | Track whole-sequence trajectory (what convergence matrix uses) |
| `cosine_sims_*` | scalars per snapshot | Convergence speed curves |
| `position_similarity` | scalars per snapshot | Position collapse dynamics |
| `top_tokens` | nested lists | Dissolution pathway (the narrative sequence) |
| `all_position_tokens` | nested lists | Full sentence dissolution tables |
| `converged_at` | int or None | Convergence speed per prompt |

### Analyses enabled (no re-run needed):

1. **Embedding neighbourhood test** — cross-reference basin tokens with `W_E`
2. **Basin-sorted convergence matrix** — reorder by terminal basin
3. **Intermediate trajectory clustering** — PCA/UMAP of all snapshots, colour by basin
4. **Convergence speed by basin** — do `prolet` prompts converge faster?
5. **Position collapse dynamics** — when does the "sentence" fully dissolve?
6. **Slonski Q-vector comparison** — requires one new forward pass (not the `.pt` file)

---

## 7. Proposed Experiment: Slonski Q-Vector × Lucier Basin Comparison

### Rationale
Slonski: Q-vectors polarise into 2 groups via outlier dimensions (static).
Lucier ATR: Iterative re-injection reveals 5 basins (dynamic).
**Are the basins a refinement of the same structure?**

### Method
1. Load GPT-2 Small
2. For each basin token + waypoints: construct a prompt, run single forward pass, extract Q-vectors from all attention heads
3. Compute pairwise cosine similarity matrix
4. Compare against Slonski's two macro-groups

### Expected Outcomes
- **All 5 basins in one macro-group:** Fine structure *within* one half of the dichotomy
- **Basins split across both groups:** Hierarchical geometry at multiple scales
- **Basin tokens are "orthogonal":** Attractors occupy special geometric positions (potentially "glitch tokens")

---

## 8. Embedding Neighbourhood Results (Priority Analysis 01)

**Notebook:** `docs/supervisor/01_token_id_extraction.ipynb`
**Date completed:** 2026-03-20

### Basin Tokens — Neighbourhood Summary

| Token | ID | Neighbourhood Type | Key Neighbours | H3 |
|---|---|---|---|---|
| `prolet` | 22758 | Political philosophy | `bourgeoisie`, `capitalists`, `revolutionaries` | ✅ Strong |
| `Divine` | 13009 | Theology (20/20 semantic) | `Sacred`, `God`, `Holy`, `celestial`, `Arcane` | ✅ Strong |
| `Anarch` | 32229 | Political philosophy | `Marx`, `Trotsky`, `Bolshevik`, `Socialism` | ✅ Strong |
| `solidarity` | 17803 | Collective action (20/20, 0 BPE) | `sympathy`, `activism`, `protest`, `comrades` | ✅ Strong |
| `till` | 10597 | Temporal conjunctions | `until`, `whilst`, `unless`, `hitherto` | ⚠ Functional |

### Waypoint Tokens — Neighbourhood Summary

| Token | ID | Neighbourhood Type | Key Neighbours | Notes |
|---|---|---|---|---|
| `capit` | 46964 | **Capitulation** (not capitalism!) | `surrender`, `succumb`, `conced(e)` | ⚠ Corrected reading |
| `injustice` | 21942 | Moral grievance (20/20 semantic) | `oppression`, `tyranny`, `bigotry`, `atrocities` | ✅ Semantic |
| `Fem` | 31149 | Gender discourse | `Feminist`, `feminism`, `misogyn(y)`, `Gender` | ✅ Semantic |
| `Rousse` | 42849 | French/political names | `Hollande`, `Brazil` then noise | ⚠ Ambiguous |
| `Ag` | 10262 | Generic prefix / agriculture | `Agriculture`, `Farm`, `Agent` | ⚠ Suggestive |
| `FT` | 9792 | Typographic (2-letter abbrev.) | `FF`, `BT`, `PT`, `GT` | ❌ Structural |
| `ash` | 1077 | Character substring | `ASH`, `ashes`, `ashed` | ❌ Structural |
| `Canad` | 2294 | Geographic prefix | `Canada`, `Canadian` | ❌ Structural |
| `Zero` | 28667 | Glitch-adjacent | `Nitrome`, `rawdownload` in neighbours | ⚠ Suspicious |

### Embedding Norms — No Outliers

All tokens within ±1.5σ of mean norm (3.9583 ± 0.4336). None flagged as anomalous. Glitch token explanation ruled out.

### Dissolution Pathway Phase Transition

| Phase | Tokens | Character | Semantic? |
|---|---|---|---|
| **Structural** (early) | `ash`, `Canad`, `Ag`, `FT`, `Zero` | Generic, typographic, geographic | ❌ |
| **Transition** | `capit` | Capitulation / surrender | ✅ (reinterpreted) |
| **Semantic** (late) | `Fem`, `injustice`, `Rousse` | Gender, grievance, political | ✅ |
| **Terminal** | `prolet`, `Divine`, `Anarch`, `solidarity`, `till` | Basin attractors | ✅ (4/5) |

---

## 9. The Mixing Time Analogy

A structural isomorphism between the dissolution phase transition and **Mixing Time** (T_mix) in room acoustics.

### Room Acoustics
In impulse response analysis, T_mix is the moment when reverberation transitions from:
- **Early reflections** — carrying spatial information (room geometry, source position)
- **Late reverberation** — carrying only the room's modal response (eigenfrequencies, material absorption)

Before T_mix: you hear *where*. After T_mix: you hear *what the room sounds like*.

### Lucier ATR
The dissolution pathway transitions from:
- **Early iterations** — carrying prompt-specific information (input identity, positional encoding)
- **Late iterations** — carrying only the model's weight geometry (training corpus topology, attractor basins)

Before transition: the residual stream remembers *what was said*. After: it reflects only *what the model is made of*.

### Formal Definition (Proposed)
**T_mix_LLM** = the iteration number at which mean pairwise cosine similarity between prompts heading to the same basin exceeds a threshold (e.g., 0.95). This is computable from the existing `.pt` data using `mean_vectors` at each snapshot.

### Connection to PI's Prior Work
The PI's Master's thesis examined **Fractal Dimensional Analysis** as a route to better prediction of Mixing Time in hybrid artificial reverberation. The potential connection between fractal dimensional analysis of convergence trajectories and the basin structure of the attractor landscape represents a direct link between the PI's two research programmes.

> **Status:** Analogy is structurally sound and potentially mathematically isomorphic. Requires formal definition and measurement before use in publications. See §10 for validation approach.

---

## 10. Statistical Validation Plan

The neighbourhood results are currently interpreted subjectively. To convert observations into publishable evidence:

### 10a. Random Baseline Comparison
Sample 100 random tokens → compute their 20-nearest-neighbour lists → measure semantic coherence. If `solidarity`'s 20/20 semantic score never appears by chance, that's statistical significance.

### 10b. Semantic Coherence Scoring
Use external taxonomy (WordNet / sentence-BERT embeddings) to compute coherence for each neighbourhood. Quantifies "how semantic" each cluster is.

### 10c. Permutation Test
Shuffle token-to-embedding mapping → re-run attractor experiment. If same tokens emerge → purely geometric. If different → embedding content matters.

### 10d. Mixing Time Measurement
Compute T_mix_LLM from `.pt` data. Compare across basins. Check whether it depends on prompt properties.

---

*Updated priority queue: ~~Token IDs~~ ✅ → ~~Embedding neighbours~~ ✅ → Statistical validation → Sorted matrix → Q-vector comparison.*
