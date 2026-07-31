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

One caution to carry: because the map is nonlinear, do not over-interpret any single attractor as "the dominant eigenvector of the model." The spectral comparison (`spectral_resonance.ipynb`, executed 2026-07-25 and rescored 2026-07-31) is the project's designed test of how far the linear intuition transfers to the per-head weight matrices, and its outcome is a working example of the eigenvector-versus-singular-vector distinction above: in the isolated per-head loop, which really is linear, every head that settled landed on its matrix's dominant eigenvector exactly (heads whose dominant eigenvalue is complex rotate instead, as the same mathematics predicts), while the registered prediction had named the top singular vector, a different direction for most heads (FINDINGS.md §3, H4; `../experiments/gpt2_small/output_eigen_rescore/report.md`).

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
| Eigenvector / eigenvalue | Direction a matrix only stretches / the stretch factor | spectral test and rescore (H4) |
| PCA | Best low-dimensional shadow of high-dimensional data | trajectory plots |
| Null model | Same procedure, cause removed | the noise baseline |

---

## A closing orientation

You now hold every concept this repository uses, and one honest way to check is to re-read the "Findings, Briefly" section of the README and notice that each bullet has become a sentence about specific mathematical objects: basins are attracting fixed points of a normalised iterated map; their coherence is a cosine-neighbourhood property of W_E; the refutation is a failed generalisation across four instances of that map; the null control is a regime relocation; the `Divine` anomaly is a disagreement between a projection and the state it projects.

The open question the project ends on (*why GPT-2 Small alone resolves language into semantic basins*) is a question about what, in one particular map's weights, carves few deep valleys under language-shaped input where sibling maps carve one or none. The J-space companion document ([JSPACE_PRIMER.md](JSPACE_PRIMER.md)) picks up the newest tool that might help ask it.
