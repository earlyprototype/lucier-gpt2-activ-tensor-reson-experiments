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
