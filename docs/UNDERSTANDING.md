# Technical Explanation: The Feedback Mechanism

## What Is Being Fed Back?

When a language model processes a prompt, the text passes through several stages:

1. **Tokenisation**: the text is split into subword tokens (e.g., "proletariat" → `prol` + `etar` + `iat`)
2. **Embedding**: each token is mapped to a 768-dimensional vector (for GPT-2 Small)
3. **Layer processing**: the sequence of vectors passes through 12 transformer layers, each containing attention heads and MLPs that read from and write to a shared **residual stream**
4. **Unembedding**: at the final layer, the residual stream is projected back into vocabulary space to produce a probability distribution over the next token

In normal operation, step 4 collapses the rich 768-dimensional state into a single token prediction (an argmax over 50,257 possibilities). This is a massive information bottleneck: the model "considered" thousands of possibilities, but only one token survives.

## The Lucier Loop Bypasses This Bottleneck

Instead of decoding the output into text and re-tokenising it as a new prompt, we:

1. **Extract** the full residual stream tensor at the output of Layer 11, a `[seq_len × 768]` matrix containing the model's complete internal state across all token positions
2. **Normalise** this tensor to maintain constant energy (L2 norm), preventing numerical explosion
3. **Inject** the normalised tensor directly into the input of Layer 0 on the next forward pass, using a programmatic hook that overwrites the normal token embeddings
4. **Repeat**: the model processes its own internal state as if it were a new input

This creates a continuous feedback loop in 768-dimensional space. No information is destroyed by the argmax bottleneck. The full geometric structure of the model's internal representation is preserved and fed back through the nonlinear transformer stack.

```
Prompt → Tokenise → Embed → [Layer 0 ... Layer 11] → Extract residual tensor
                      ↑                                        |
                      └──────── Normalise & Re-inject ─────────┘
                                    (repeat 500×)
```

## Why This Produces Attractors

Repeated application of any function to its own output, f(f(f(x))), tends toward fixed points or limit cycles. In a linear system, this is **power iteration**: the dominant eigenvector of the transformation matrix is progressively amplified while all others decay. The converged state is the "eigenvoice" of the matrix.

A transformer is not linear: it includes LayerNorm, softmax attention with dynamically recomputed queries/keys/values, and nonlinear MLP activations. But the same dynamical principle applies: the system has attractor states determined by its weight geometry, and iterative re-injection converges toward them.

The L2 normalisation is critical: without it, the tensor's norm explodes exponentially (reaching 1.5M by iteration 500), making the dynamics meaningless. With normalisation, the system is energy-conservative, and convergence to stable attractors becomes possible.

## Why This Is Not a Text Loop

This distinction is fundamental:

| | Text Loop | Activation Loop (This Experiment) |
|:---|:---|:---|
| **What's fed back** | A decoded token (1 integer) | The full residual stream (`[seq_len × 768]` floats) |
| **Information preserved** | Only the argmax winner | The entire superposition of all 50,257 token candidates |
| **Dynamics** | Discrete, lossy, stochastic | Continuous, lossless, deterministic |
| **What converges** | The model's text generation habits | The stable states of the iterated model, which depend on the input regime (see below) |

## Key Parameters

| Parameter | Value | Rationale |
|:---|:---|:---|
| **Model** | GPT-2 Small (124M params) | Well-studied, manageable size, known training data (WebText/Reddit) |
| **Layer window** | 0 → 11 (full stack) | The entire architecture acts as the "room" |
| **Normalisation** | Per-iteration L2 rescaling to initial norm | Prevents energy explosion, enables stable convergence |
| **Iteration schedule** | `[0, 2, 3, 5, 10, 20, 50, 100, 250, 500]` | Logarithmic: captures both early dynamics and deep convergence |
| **Temperature** | N/A (deterministic) | No sampling: pure forward-pass dynamics |

## One Important Correction (What the Attractors Are Not), Itself Corrected

An earlier framing of this project described the attractors as "the model's weight geometry made audible", as if the basins were universal properties of the weights that any input would eventually reveal. The first null-model control seemed to show this was wrong: pure random noise appeared to converge into a *different* set of attractors (eighteen scattered punctuation tokens, none of the five semantic basins), so the attractors were re-described as states of the model *as driven by language-shaped input*.

That control did not survive its own audit. It had run at roughly a third of the intended injection energy and was counted before it settled ([FINDINGS.md caveat 18](FINDINGS.md#caveats)), and the corrected re-run (2026-07-31, run 17: each noise tensor matched to a real prompt's exact length and injection energy, counted only at convergence) reverses the verdict: noise converges into seven basins, four of them the language arm's own five, with `prolet` dominant in both arms. So at this injection energy the original framing was closer to right than its correction: **the basins are properties of the weights, revealed by noise just as readily as by language.** Whether that holds at other injection energies is the registered follow-up (the ν-sweep). The model-specific half survives untouched: GPT-2 Medium, trained on the same corpus, still collapses everything into a single empty token. The full evidence is in [FINDINGS.md](FINDINGS.md), finding F4.
