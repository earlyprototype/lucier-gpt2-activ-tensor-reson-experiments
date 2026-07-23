# Prior Art: Eduardo Slonski, Attention Q-Vector Binary Dichotomy

> **Source:** [X/Twitter Thread](https://x.com/EduardoSlonski/status/1745130935727894616) (27 tweets, Jan 10 2024)
> **Thread Reader:** [Unrolled](https://threadreaderapp.com/thread/1745130935727894616.html)
> **Views:** 207K+ | **Engagement:** 1K+ likes, 135 retweets
> **Date Retrieved:** 2026-03-20

---

## What Eduardo Found

Eduardo Slonski discovered a "very interesting phenomenon in the most crucial part of Transformers, the Attention Mechanism."

### The Experiment
- **Model:** Pythia 1B (EleutherAI)
- **Layer analysed:** Layer 15 (deeper layers show stronger effect)
- **Method:** Standard single forward pass: compute **cosine similarity** between all pairs of **Query (Q) vectors** across 256 tokens from a text sample
- **Visualisation tool:** Custom interactive research dashboard (unidentified, possibly internal EleutherAI tooling or custom TransformerLens frontend; see video in tweet 1)

### Key Findings

1. **Binary Dichotomy:** Tokens polarise into **two** near-opposite clusters (cosine similarity ≈ **−0.99** between group centroids)

2. **The Tartan Pattern:** The Q-vector similarity matrix produces a striking block-diagonal "tartan" or "plaid" pattern: tokens within a group are highly similar (yellow/green), tokens across groups are dissimilar (dark blue/purple)

3. **Group Identity:**
   - **Group 1 (Blue):** Standard continuation tokens
   - **Group 2 (Orange):** "Expression finishers": tokens that end multi-token expressions
   - Examples (from tweet 22): "many music genres," "cultural figures," "by his personal physician," plus words split into tokens ("sur|ges," "acqu|itted," "Jacks|ons") and multi-token phrases ("Staples Center," "Los Angeles," "Grammy Awards")

4. **Outlier Dimensions:** The dichotomy is driven by specific high-amplitude dimensions in the Q-weight matrix. Most dimensions are low-amplitude (both groups similar), but a few "outlier" dimensions have large, oppositely-signed values that separate the groups (see `weight_analysis.jpg`)

5. **Scale Invariance:** Pattern appears across all Pythia sizes (70M → 12B), becoming more pronounced in deeper layers

---

## Downloaded Assets

| File | Description |
|------|-------------|
| `heatmap_main.jpg` | The tartan/plaid cosine similarity matrix of Q-vectors (256×256 tokens) |
| `weight_analysis.jpg` | "First 50 dimensions of the two groups of weights": blue vs orange lines showing dimensional profiles |

---

## Relationship to Our Work (Lucier ATR Experiment)

### Similarities
| Feature | Eduardo (Slonski) | Us (Lucier ATR) |
|---------|-------------------|------------------|
| **Visual Pattern** | Tartan/plaid block-diagonal | Tartan/plaid block-diagonal |
| **Metric** | Cosine similarity | Cosine similarity |
| **Model Family** | Pythia (EleutherAI) | GPT-2 Small (OpenAI) |
| **Discovery** | Latent clustering/dichotomy | Attractor basin landscape |

### Critical Differences

| | Eduardo | Us |
|---|---------|-----|
| **Method** | Single forward pass | **100 iterations of re-injection** |
| **What's measured** | Q-vector similarity (static snapshot) | Terminal residual stream convergence (dynamical fixed points) |
| **# of clusters** | **2** (binary dichotomy) | **5** (prolet, Divine, Anarch, till, solidarity) |
| **Interpretation** | Where tokens *happen to sit* in Q-space | Where activation states *want to converge* under iteration |
| **Scope** | Token-level clustering within a sample | Prompt-level attractor identification across 125 prompts |

### The Key Insight

> Eduardo shows the **static landscape**: a cross-section of token geometry from a single pass.
>
> We show the **dynamical landscape**: where the system converges under repeated iteration.
>
> If a single pass shows 2 clusters, and 100 iterations reveals 5 basins, our re-injection process is **amplifying and differentiating** structure that is latent in the attention geometry but invisible to static analysis.

### Proposed Experiment (#5 in Backlog)
Compute Eduardo's Q-vector similarity matrix for our 125 prompts (single-pass, no re-injection) and compare it side-by-side with our convergence matrix. If the block structure matches → the attractors are amplified Q-geometry. If it doesn't → re-injection reveals truly novel structure.

---

## Citation

```
Slonski, Eduardo. "Attention Mechanism Binary Dichotomy in Transformers."
X/Twitter Thread, January 10, 2024.
https://x.com/EduardoSlonski/status/1745130935727894616
```
