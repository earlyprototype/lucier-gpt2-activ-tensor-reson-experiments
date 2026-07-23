# Technical Specification: Iterative Activation Re-injection

## Method

### Overview

This experiment implements iterative re-injection of the full residual stream tensor through the forward pass of a transformer language model (reference architecture: GPT-2 Small, 124M parameters, 12 layers, 12 heads, d_model=768). The residual stream at the final layer's `hook_resid_post` is extracted, L2-normalised, and re-injected at `blocks.0.hook_resid_pre` via a TransformerLens forward hook, overwriting the token embeddings. This is repeated for *N* iterations to map the fixed-point attractor landscape of the iterated forward map **under the chosen initial-condition regime**. The null-model control shows the landscape is regime-dependent, not a universal property of the weights (see [FINDINGS.md](FINDINGS.md), F4).

The same protocol runs cross-model via the shared engine (`atr_engine.py`): GPT-2 Medium (24 layers, d_model=1024), Pythia-160m (12 layers), Pythia-410m (24 layers): extraction always at the final layer's `resid_post`, injection at layer 0. A convergence-gated variant (`run_atr_gated`) classifies terminal basins at lock-in (`cos_sim_mean > 0.999` sustained over three consecutive checks) rather than at a fixed iteration horizon.

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

A direct empirical comparison between the last two rows above, the per-head resonant state actually observed under iterative re-injection versus the dominant singular vector predicted by static SVD of `W_OV`, is scaffolded in `experiments/gpt2_small/spectral_resonance.ipynb`. It has not been run; see [FINDINGS.md](FINDINGS.md) (H4) and its Caveats section.

## Repeatability

Terminal attractors (`prolet` × 4, `Divine` × 1) are stable across N=2 same-machine runs. Intermediate dissolution pathways show sensitivity to floating-point non-determinism (expected for iterative nonlinear maps), but converge to identical fixed points. The convergence-gated re-sweep additionally confirms that 91/125 prompts reach a hard fixed point (`cos_sim_mean > 0.999`) and that basin labels assigned at lock-in are stable to 1000 iterations. Full determinism would require CPU execution with fixed seeds.

Cross-hardware replication has now been attempted once, and passed (2026-07-19): the same code, run on a different machine class (a fresh cloud container, CPU) with `gpt2` weights obtained from a legacy Hugging Face S3 mirror and loaded offline, reproduced the five-prompt piece exactly: identical terminal attractors (`prolet` × 4, `Divine` × 1) and identical intermediate dissolution waypoints (`Ag` at iteration 10, `Rousse` at iteration 50, `capit` en route), three times on that container. See [FINDINGS.md](FINDINGS.md) (F6) and `experiments/gpt2_small/output_confidence/confidence_report.md` (Result 0). The scope of this claim is deliberately narrow: it is replication of the same code on new hardware, not independent re-implementation; the weights came from a mirror rather than the canonical distribution (identical attractors and waypoints argue they are the same checkpoint); and one container is one data point. Independent re-implementation by another investigator has still not been attempted; "reproducibility" in the strict sense remains pending.

## Dependencies

Install from [`requirements.txt`](../requirements.txt) at the repository root.
