# ATR Scaling Artefact Analysis

## Context

Activation Tensor Resonance (ATR) takes a language model's output activations — the raw internal state after processing a prompt — and feeds them back in as input, repeating the cycle hundreds of times. Like Alvin Lucier recording his voice into a room until only the room's resonant frequencies remain, ATR dissolves the original prompt until only the model's dominant internal modes are left. In GPT-2 Small, this revealed five semantic attractor basins (`prolet`, `Divine`, `Anarch`, `till`, `solidarity`) that map to the political and theological centre of mass of its Reddit 2018 training data.

The question that follows naturally: what happens when you do this to larger models?

## The Question

When ATR is run on larger models, the attractor landscape changes dramatically. GPT-2 Small produces five clear semantic basins. Pythia-410m produces scattered punctuation and connectives with no convergence even at 250 iterations. Is this a real property of the model family, or is the experimental setup introducing avoidable distortion?

## Guiding Principle (Front and Centre)

If token labels flicker but `cos_sim_mean` is high and readout confidence is low, treat the behaviour as **readout ambiguity first**, not attractor instability.

## The Four Models — What We See

| Model | Params | Training Data | What Happens |
|---|---|---|---|
| GPT-2 Small | 124M | Reddit 2018 | 5 semantic basins (`prolet`, `Divine`, `Anarch`, `till`, `solidarity`). Clean convergence by ~50 iterations. |
| Pythia-160m | 160M | The Pile | Near-total collapse to `questioned` (94%). Converges by iteration 2-3. |
| GPT-2 Medium | 345M | Reddit 2018 | Total collapse to `D` (100%). Converges by iteration 5-10. |
| Pythia-410m | 410M | The Pile | 40+ fragments (punctuation, connectives). No convergence at 250 iterations. |

## 1) Methodological Artefacts (Apparatus Faults)

This section includes only candidate issues that could undermine the validity of interpretation by introducing distortion from the method itself.

### 1.1 Normalisation (master fader) — RULED OUT AS ARTEFACT

The per-iteration L2 rescale multiplies the entire tensor by one scalar (same ratio across all positions and dimensions).

**What it does operationally:** Prevents numeric blow-up between iterations. Residual additions compound magnitude across the stack; without rescale, norms reach ~1.5 million by iteration 500 and the run becomes meaningless.

**Why it was suspected:** A single scalar on a wider model (e.g. 1024-d vs 768-d) was briefly framed as possibly “reviving” weak dimensions by restoring total energy. That story fails: the scalar preserves the mix exactly — if one dimension is 500× another before rescale, it still is after.

**Why it is not a distortion source:** Layer 0 applies LayerNorm first; LayerNorm output is invariant to global scale. The forward pass therefore does not distinguish pre-rescale vs post-rescale tensors.

**Caveat on alternatives:** Per-dimension or max-dimension rescales are a different intervention — they can distort the relative geometry LayerNorm then sees. They are not equivalent to the current global L2 step.

**Definitive position:** Normalisation is numerically essential but computationally cosmetic for the forward map. It is not the source of the Pythia-410m fragmentation pattern.

### 1.2 Readout (unembedding) — OPEN ARTEFACT CANDIDATE

ATR interprets internal state by projecting the residual (after `ln_final` in the usual readout path) to token logits via the unembedding matrix.

**Risk:** A state that is stable in the residual stream may still sit where many vocabulary directions score similarly — so argmax (or dominant token) flickers while the tensor has effectively stopped moving.

**Contrast across models:** Tighter clustering in unembedding space (often discussed for some GPT-2 regimes) yields cleaner dominant tokens; flatter or more evenly spaced geometry yields “between stations” behaviour.

**Observable signature:** `cos_sim_mean` → 1 (or a tight plateau) while decoded tokens keep jumping — dynamics converged, vocabulary projection ambiguous.

**Note:** `cos_sim_mean` is computed on the activation tensor between iterations; it does not pass through token readout. It is the clean separator for this artefact class.

**Definitive position:** Readout remains the primary live methodological artefact candidate.

### 1.3 Readout Interpretation Guardrails (Amendment)

This amendment adds a strict interpretation protocol so readout noise does not misdirect conclusions.

**A. BPE/subword jaggedness — what it changes**

- **Mechanism:** Internal state moves continuously, but token output is discrete. Small vector moves near a decision boundary can flip top-1 token abruptly.
- **Why BPE amplifies this:** Fragments such as `prolet`, `capit`, punctuation variants, and whitespace-prefixed tokens can alternate with little underlying tensor movement.
- **Qualitative impact:** Apparent "semantic turbulence" in token traces can be visual, not dynamical.
- **Quantitative impact:** Token-switch count can be high while `cos_sim_mean` remains high/plateauing.
- **Weighting:** Interpretation risk **Medium-High**; code defect risk **Low**.

**B. Missing readout confidence metrics — why this matters**

- **Current gap:** Top token is logged, but confidence is not.
- **Consequence:** Near-tie flicker and genuine instability are conflated.
- **Required additions:** Top-1 vs top-2 logit margin, entropy (full or top-k), and optional top-k overlap across iterations.
- **Weighting:** Interpretation risk **High**; implementation effort **Low**; priority **P1**.

**C. Token rendering artefacts (`decode`) — precision caveat**

- **Current gap:** String decode can hide token-level distinctions (especially whitespace/special-token forms).
- **Required additions:** Log raw token IDs alongside rendered strings.
- **Weighting:** Interpretation risk **Low-Medium**; implementation effort **Very Low**; priority **P2**.

## 2) Intrinsic Model Variables (Under Investigation)

This section includes factors that are part of the model/system itself. 

### 2.1 Forward-pass depth

**Two clocks:**

- **Within one iteration — the forward pass:** The full native stack (12 layers for GPT-2 Small, 24 for Pythia-410m). This is ordinary inference geometry.

- **Between iterations — the Lucier loop:** Extract final-layer residual, L2-rescale for stability, re-inject at layer 0. That closure is the experiment.

After iteration 0, layer 0 no longer sees a fresh token-embedding row from the lookup table; it sees the previous end-of-stack residual (same `d_model` space the stack already uses). The accumulated shift across depth is what ATR is meant to iterate.

**Observable pattern:** Mixed behaviour across prompts — some tracks converge quickly, others oscillate or fragment — is consistent with depth-dependent dynamics rather than a single global bug.

A 24-layer pass is native for Pythia-410m; depth belongs in explanatory analysis, not under “methodological artefacts.”

### 2.2 Training corpus and token geometry

Reddit-trained GPT-2 variants and The-Pile-trained Pythia variants are shaped by different data distributions and potentially different representational topology in unembedding space.

### 2.3 Width and parameterisation regime

Changes in hidden size, head layout, and parameter count alter the geometry of the learned function and can shift attractor basin structure.

**Definitive position:** These variables belong in explanatory analysis, not in artefact diagnosis.

## 3) Controls and Attribution Tests

This section defines the tests that separate apparatus effects from intrinsic model effects.

1. **Cross-model `cos_sim_mean` chart (single view).**  
   If Pythia-410m remains below convergence while others saturate, non-convergence is internal-dynamics evidence. If it saturates while tokens flicker, readout is implicated.

2. **Same-model depth control on Pythia-410m (0-11 vs 0-23).**  
   Hold weights, tokenizer, and corpus constant; vary layer span only. If convergence behaviour changes materially, depth-dependent dynamics are causal.

3. **Long-horizon run (extend to 1000 iterations).**  
   Distinguish "not yet converged" from "structurally fragmented attractor landscape."

Each test changes one variable at a time and is implementable with minimal ATR engine changes.

### 3.1 Amended Experimental Versions (Readout-Focused)

**ATR-R1 (Confidence-Aware Readout)**
- Add per-snapshot: top-1 token ID/string, top-2 token ID/string, logit margin, entropy.
- **Follow-on use:** Re-label apparent fragmentation as either "high-confidence divergence" or "low-confidence boundary flicker".

**ATR-R2 (ID-First Trace)**
- Store token IDs as canonical output; keep decoded strings as display-only.
- **Follow-on use:** Build stable transition matrices and exact basin membership counts independent of rendering quirks.

**ATR-R3 (Tensor-Readout Concordance Audit)**
- For each run, classify snapshots into:  
  (i) high `cos_sim_mean` + low margin (readout ambiguity),  
  (ii) high `cos_sim_mean` + high margin (stable attractor label),  
  (iii) low `cos_sim_mean` (true ongoing dynamics).
- **Follow-on use:** Report convergence with confidence bands, not token labels alone.

## Current Judgement

- **Ruled out artefact:** Normalisation.  
- **Live artefact candidate:** Readout projection to tokens.  
- **Intrinsic explanatory axes:** Depth, corpus, width, token geometry.

## The Bigger Picture

The four models do not share one uniform failure mode; landscapes differ with corpus and architecture. Prompt-level heterogeneity (e.g. some Pythia-410m runs converging while others oscillate) fits intrinsic geometry and depth-dependent iteration maps, not a single broken knob in the apparatus.

Remaining work stays on two tracks: (a) close out readout as an artefact using tensor-level metrics vs token traces, and (b) attribute basin structure to depth, data, width, and unembedding geometry with controlled comparisons.
