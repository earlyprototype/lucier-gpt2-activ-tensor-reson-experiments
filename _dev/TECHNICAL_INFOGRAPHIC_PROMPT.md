# Infographic Prompt: Technical Specification
# Audience: ML engineers, mechanistic interpretability researchers, arxiv readers
# Pair with: TECHNICAL.md

---

## Generation Prompt

Create a dense technical reference card for the Iterative Activation Re-injection method. The audience are ML engineers who know what residual streams, LayerNorm, and TransformerLens hooks are. Density is a feature — this is a spec sheet, not a tutorial.

### Content (all required, in layout order)

**Top banner:**
- Title: "Iterative Activation Re-injection"
- One-liner: "Nonlinear power iteration over the full transformer forward map"
- Architecture badge: GPT-2 Small | 124M params | 12 Layers | 768 d_model | TransformerLens

**Left column — Method:**
- Formal iteration:
  - x₀ = f(embed(prompt))
  - xₙ₊₁ = f(normalise(xₙ))
  - normalise(x) = x · (‖x₀‖₂ / ‖x‖₂)
  - Convergence: cos_sim(xₙ, xₙ₊₁) → 1.0
- Hook mechanism:
  - Read: blocks.11.hook_resid_post
  - Write: blocks.0.hook_resid_pre
  - Prompt tokens serve only as scaffolding

**Right column — Metrics:**
- Snapshot schedule: [0, 2, 3, 5, 10, 20, 50, 100, 250, 500]
- Metrics table with tensor shapes:
  - resid_tensor [T, 768]
  - last_vector [768]
  - mean_vector [768]
  - top_tokens (top-5)
  - cos_sim_last (scalar)
  - cos_sim_mean (scalar)
  - position_similarity (scalar)
  - tensor_norm (scalar)

**Bottom panel — Context:**
- Prior work comparison table:
  - Power iteration → similar (iterative operator), different (our operator is nonlinear)
  - Activation engineering → similar (operates on residual), different (single-pass, not iterated)
  - Model collapse → similar (self-feeding), different (dataset level via text, not activations)
  - RNN fixed-point analysis → similar (attractor dynamics), different (transformers are feedforward)
- Reproducibility: "prolet" × 4, "Divine" × 1 stable across N=2 runs

### Visual Style
- **Dark mode.** Charcoal/near-black background (#1a1a2e or similar), white primary text
- Electric blue (#00d4ff) for emphasis and code highlighting
- Muted amber (#ffb347) for warnings, caveats, and the reproducibility note
- Monospace font for all code and equations
- Clean sans-serif for labels and descriptions
- Grid-based layout with visible hairline dividers — engineering drawing aesthetic
- **NO decorative illustrations.** Every visual element conveys data.

### Layout
- Structured grid with clear borders between sections
- Top: full-width banner with title + architecture badge
- Middle: two-column (method left, metrics right)
- Bottom: full-width prior work comparison + reproducibility

### Text:Image Balance
- 70% text / 30% structural visual (tables, code blocks, the flow diagram)
- Every pixel earns its place by conveying information
