# Infographic Prompt: Understanding Guide
# Audience: ML-curious developers, science communicators, interdisciplinary researchers
# Pair with: UNDERSTANDING.md

---

## Generation Prompt

Create an infographic about iterative activation re-injection in transformers, aimed at people who know what neural networks are but don't work in ML daily. This is a conceptual explainer, not a technical spec.

### Content (in priority order)
1. **The Lucier analogy.** Start with this: a microphone pointed at a speaker in a room reveals the room's resonant frequencies. This experiment does the same thing with GPT-2 — feeding the model's output back into itself reveals the weight geometry's "resonant frequencies."
2. **The feedback loop diagram.** Show the full cycle: Prompt → Embed → [Layers 0–11] → Extract residual tensor → Normalise → Re-inject back to Layer 0. Repeat 500 times. Make this the hero visual — large, central, confident.
3. **Text loop vs activation loop comparison.** Two-column comparison: Text loops feed back 1 decoded token (lossy, discrete). Activation loops feed back the full 768-dimensional residual stream (lossless, continuous). The entire superposition of 50,257 candidates is preserved.
4. **The "prolet" finding.** 4 of 5 different prompts converge to the same made-up word: "prolet." The 5th converges to "Divine." Use this as a pull quote at the bottom — it's the "wow" moment.

### What NOT to include
- No equations, no tensor shapes, no code
- No architecture details (layer count, d_model, etc.)

### Visual Style
- Warm editorial science magazine aesthetic (Quanta Magazine, New Scientist)
- Muted sophisticated palette: deep teal, warm grey, cream/off-white. NOT neon.
- Clean sans-serif typography. Generous whitespace.
- Subtle grain or texture to avoid the flat, sterile AI-generated look
- Thin line-style icons — not filled, not glossy
- **Absolutely NO:** generic AI art, brain graphics, circuit-board imagery, glowing neural network visualisations, lightbulbs, gears, or puzzle pieces

### Layout
- Vertical scroll flow, top to bottom
- Hero: feedback loop diagram (large, central)
- Middle: 3 panels — "What goes in → What happens → What comes out"
- Comparison: text loop vs activation loop (side by side)
- Footer: "prolet" finding as a pull quote

### Text:Image Balance
- 40% text / 60% visual
- Max 2 sentences per text block
- No dense paragraphs — every text block earns its place
