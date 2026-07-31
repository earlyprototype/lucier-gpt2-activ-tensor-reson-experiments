# ATR Scaling Artefact Analysis

## Context

Activation Tensor Resonance (ATR) takes a language model's output activations, the raw internal state after processing a prompt, and feeds them back in as input, repeating the cycle hundreds of times. Like Alvin Lucier recording his voice into a room until only the room's resonant frequencies remain, ATR dissolves the original prompt until only the model's dominant internal modes are left. In GPT-2 Small, this revealed five attractor basins (`prolet`, `Divine`, `Anarch`, `till`, `solidarity`), four of them semantically coherent clusters of political and theological vocabulary in embedding space. The founding hypothesis read the basins as the thematic centre of mass of the Reddit 2018 training data; that corpus-causal reading was later refuted (GPT-2 Medium, trained on the same corpus, produces no semantic basins; [FINDINGS.md](FINDINGS.md) F3, F4).

The question that follows naturally: what happens when you do this to larger models?

## The Question

When ATR is run on larger models, the attractor landscape changes dramatically. GPT-2 Small produces five clear semantic basins. Pythia-410m produces scattered punctuation and connectives with no convergence even at 250 iterations. Is this a real property of the model family, or is the experimental setup introducing avoidable distortion?

## Guiding Principle (Front and Centre)

If token labels flicker but `cos_sim_mean` is high and readout confidence is low, treat the behaviour as **readout ambiguity first**, not attractor instability.

## The Four Models: What We See

| Model | Params | Training Data | What Happens |
|---|---|---|---|
| GPT-2 Small | 124M | Reddit 2018 | 5 semantic basins (`prolet`, `Divine`, `Anarch`, `till`, `solidarity`). Clean convergence by ~50 iterations. |
| Pythia-160m | 160M | The Pile | Near-total collapse to `questioned` (94%). Converges by iteration 2-3. |
| GPT-2 Medium | 345M | Reddit 2018 | Total collapse to `D` (100%). Converges by iteration 5-10. |
| Pythia-410m | 410M | The Pile | 40+ fragments (punctuation, connectives). No convergence at 250 iterations. |

## 1) Methodological Artefacts (Apparatus Faults)

This section includes only candidate issues that could undermine the validity of interpretation by introducing distortion from the method itself.

### 1.1 Normalisation (master fader): RULED OUT AS ARTEFACT

The per-iteration L2 rescale multiplies the entire tensor by one scalar (same ratio across all positions and dimensions).

**What it does operationally:** Prevents numeric blow-up between iterations. Residual additions compound magnitude across the stack; without rescale, norms reach ~1.5 million by iteration 500 and the run becomes meaningless.

**Why it was suspected:** A single scalar on a wider model (e.g. 1024-d vs 768-d) was briefly framed as possibly “reviving” weak dimensions by restoring total energy. That story fails: the scalar preserves the mix exactly: if one dimension is 500× another before rescale, it still is after.

**Why it is not a distortion source:** Layer 0 applies LayerNorm first, and LayerNorm output is invariant to positive global rescaling up to its epsilon term: because of the epsilon in the denominator, LayerNorm(c·x) equals LayerNorm(x) exactly only in the limit epsilon → 0. At the tensor norms involved here the differences are at epsilon and floating-point scale, so the forward pass is effectively, though not bit-exactly, indifferent to pre-rescale vs post-rescale tensors.

**Caveat on alternatives:** Per-dimension or max-dimension rescales are a different intervention: they can distort the relative geometry LayerNorm then sees. They are not equivalent to the current global L2 step.

**Definitive position (amended 2026-07-28 — the second clause is withdrawn):** Normalisation is numerically essential, and it is **not** the source of the Pythia-410m fragmentation pattern. That half stands on its own argument: the scalar preserves the mix exactly, so it cannot revive weak dimensions, which is the specific artefact story this section was written to test.

The "approximately inert for the forward map" clause does not follow and should not be relied on. **The premise is right and the inference is wrong.** LayerNorm output is indeed invariant to a positive global rescale up to its epsilon term — but a pre-LN block's residual path goes *around* that LayerNorm. Writing *g* for the block contribution, *F*(*c*·*x*) = *c*·*x* + *g*(LN(*x*)) against *F*(*x*) = *x* + *g*(LN(*x*)): the two differ by (*c* − 1)·*x*, and from the next block onward the gap compounds, because that block's LayerNorm now sees a different residual.

Demonstrated in `experiments/preln_rescale_check.py` (pure standard library, no model needed). Over a 12-block pre-LN stack, cos(*F*(*c*·*x*), *F*(*x*)) is 1.000000 at *c* = 1.001, **0.936 at *c* = 2, and 0.505 at *c* = 10** — while max \|LN(*c*·*x*) − LN(*x*)\| stays at 9.9 × 10⁻⁵, confirming the premise at the same time. The rescale is inert exactly when *c* ≈ 1, which is when it is doing nothing; this section's own note that norms would otherwise reach ~1.5 million by iteration 500 says *c* is nowhere near 1 in practice.

Random weights and *d* = 64, so this settles the **structure** of the argument, not the size of the effect in GPT-2. Measuring the real per-iteration *c* and the real departure needs `torch` and the actual weights. Until then, treat the forward map as **scale-sensitive**, and see [FINDINGS.md](FINDINGS.md) caveat 7, where H-pos0 is explicitly conditional on this.

### 1.1b Tokenisation asymmetry across the 2×2: OPEN ARTEFACT CANDIDATE (added 2026-07-26)

A fourth apparatus channel, not examined anywhere above, and the only one that differs *between the models being
compared* rather than applying to all of them equally.

**What it is.** `atr_engine.py` hands a raw string to `run_with_cache` (lines 125, 183, 310, 343). TransformerLens
tokenises strings through `to_tokens`, which prepends the beginning-of-sequence token when
`cfg.default_prepend_bos` is set. In `loading_from_pretrained.py` only `GPTNeoXForCausalLM` carries an explicit
`"default_prepend_bos": False` (line 537); GPT-2 has no override and inherits the global default `True` (line
1720). Measured on the engine's own call path: a 4-token probe becomes **5 tokens for `gpt2` and `gpt2-medium`**
and stays **4 for `pythia-160m` and `pythia-410m`**.

**So position 0 holds the special token `<|endoftext|>` on the GPT-2 arm and an ordinary content token on the
Pythia arm.** That is token construction, and it is measured. Whether that position then *functions* as an
attention sink here is an inference from the literature, not a measurement made in this repository — see the
coordinate-versus-positional note below. Nobody chose this: it is a library default that varies by model family and is invisible at the call site.

**Why it is an artefact candidate rather than an intrinsic variable.** Every axis in section 2 is a property of
the models. This is a property of *how we called them*, and it is not symmetric across the comparison, which is
precisely the shape of thing this section exists to catch. It is closest in kind to §1.1: both concern what the
apparatus does to the tensor before the model sees it. §1.1 clears the global rescale for the *forward map* via
LayerNorm scale-invariance; that argument says nothing about tokenisation, and nothing about how the conserved
norm is distributed across positions.

**What it puts at risk, in order.**

1. **Position-indexed cross-model claims.** Sequences differ in length by one for the same prompt, so per-position
   means run over different denominators, and position *i* is not the same token across arms. Position uniformity
   is the exposed claim.
2. **The energy budget.** The global L2 rescale conserves total norm. If a small coordinate set dominates that
   norm — and on GPT-2 it does — the rescale is largely setting *its* magnitude, with content riding the
   remainder. Note the distinction that is easy to blur: massive activations are a **coordinate** phenomenon,
   attention sinks are a **positional** one; they are associated but not identical, and whether position 0 carries
   anomalous energy *in ATR trajectories specifically* is unmeasured.
3. **A regime no model was trained in — conditional on the sink hypothesis.** From iteration 1 the re-injection
   overwrites position 0, so whatever role that position plays is preserved *structurally* while its *contents*
   are replaced. If position 0 is acting as a sink, that is a configuration no model was trained in, and only the
   GPT-2 arm enters it. Untested, and it rests on the inference flagged above.

**What it does not explain.** Not the Small-versus-Medium divergence: both carry the BOS and behave completely
differently.

**Cheapest control.** Align the slices: **GPT-2 `[1:]` against Pythia `[:]`**. Dropping index 0 from *both* arms —
the form first written here, and wrong — strips GPT-2's BOS and Pythia's first genuine content token together,
trading one misalignment for another. Needs no forward passes. Two limits, both real: this is a **sensitivity
check, not a restoration of comparability**, since the BOS participates in the GPT-2 forward pass and conditions
every other position through attention, so removing it from the metric does not remove it from the computation;
and the states being compared were produced under different conditioning either way. Then, if it matters, add a GPT-2 arm run with
`prepend_bos=False` to match Pythia's regime. Do not go the other way: Pythia was not trained on BOS-prefixed
sequences, so prepending one there introduces an artefact rather than removing one.

Raised by `agent:pythia-review` (peer board, discussion #59); verified against the TransformerLens source and by
execution. Recorded as [FINDINGS.md](FINDINGS.md) caveat 17, with F5 qualified accordingly.

### 1.2 Readout (unembedding): OPEN ARTEFACT CANDIDATE

ATR interprets internal state by projecting the residual (after `ln_final` in the usual readout path) to token logits via the unembedding matrix.

**Risk:** A state that is stable in the residual stream may still sit where many vocabulary directions score similarly, so argmax (or dominant token) flickers while the tensor has effectively stopped moving.

**Contrast across models:** Tighter clustering in unembedding space (often discussed for some GPT-2 regimes) yields cleaner dominant tokens; flatter or more evenly spaced geometry yields “between stations” behaviour.

**Observable signature:** `cos_sim_mean` → 1 (or a tight plateau) while decoded tokens keep jumping: dynamics converged, vocabulary projection ambiguous.

**Note:** `cos_sim_mean` is computed on the activation tensor between iterations; it does not pass through token readout. It is the clean separator for this artefact class.

**Definitive position:** Readout remains the primary live methodological artefact candidate.

### 1.3 Readout Interpretation Guardrails (Amendment)

This amendment adds a strict interpretation protocol so readout noise does not misdirect conclusions.

**A. BPE/subword jaggedness: what it changes**

- **Mechanism:** Internal state moves continuously, but token output is discrete. Small vector moves near a decision boundary can flip top-1 token abruptly.
- **Why BPE amplifies this:** Fragments such as `prolet`, `capit`, punctuation variants, and whitespace-prefixed tokens can alternate with little underlying tensor movement.
- **Qualitative impact:** Apparent "semantic turbulence" in token traces can be visual, not dynamical.
- **Quantitative impact:** Token-switch count can be high while `cos_sim_mean` remains high/plateauing.
- **Weighting:** Interpretation risk **Medium-High**; code defect risk **Low**.

**B. Missing readout confidence metrics: why this matters**

- **Current gap:** Top token is logged, but confidence is not.
- **Consequence:** Near-tie flicker and genuine instability are conflated.
- **Required additions:** Top-1 vs top-2 logit margin, entropy (full or top-k), and optional top-k overlap across iterations.
- **Weighting:** Interpretation risk **High**; implementation effort **Low**; priority **P1**.

**C. Token rendering artefacts (`decode`): precision caveat**

- **Current gap:** String decode can hide token-level distinctions (especially whitespace/special-token forms).
- **Required additions:** Log raw token IDs alongside rendered strings.
- **Weighting:** Interpretation risk **Low-Medium**; implementation effort **Very Low**; priority **P2**.

## 2) Intrinsic Model Variables (Under Investigation)

This section includes factors that are part of the model/system itself. 

### 2.1 Forward-pass depth

**Two clocks:**

- **Within one iteration, the forward pass:** The full native stack (12 layers for GPT-2 Small, 24 for Pythia-410m). This is ordinary inference geometry.

- **Between iterations, the Lucier loop:** Extract final-layer residual, L2-rescale for stability, re-inject at layer 0. That closure is the experiment.

After iteration 0, layer 0 no longer sees a fresh token-embedding row from the lookup table; it sees the previous end-of-stack residual (same `d_model` space the stack already uses). The accumulated shift across depth is what ATR is meant to iterate.

**Observable pattern:** Mixed behaviour across prompts, some tracks converge quickly, others oscillate or fragment, is consistent with depth-dependent dynamics rather than a single global bug.

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


---

## Closing Judgement (2026-07-10, controls executed)

The attribution tests proposed above were run at series close ([FINDINGS.md](FINDINGS.md), [RESULTS_SUMMARY.md](../experiments/RESULTS_SUMMARY.md)):

- **Test 1 (cross-model cos_sim chart): executed.** GPT-2 Medium and Pythia-160m saturate to 1.0000 by iteration 10: their single-token collapses are real tensor attractors. Pythia-410m plateaus at ~0.85 through 250 iterations: non-convergence is internal dynamics, not readout.
- **Test 3 (long horizon): executed.** Pythia-410m at 1000 iterations (8-prompt subset): still fragmented, cross-prompt similarity 0.21. Structural, not under-iterated.
- **ATR-R1/R3 (confidence-aware readout): implemented and demonstrated** (single-prompt audit; margin rises and entropy falls as trajectories settle). The sharpest dissociation found: GPT-2 Small's `Divine` basin, readout constant over what is now resolved as an exact period-2 limit cycle: it fails the lag-1 convergence gate by construction (a lag-1 gate cannot pass a period-2 cycle) and is converged under a lag-2 gate for the audited trajectory ([FINDINGS.md](FINDINGS.md) F9, F15).
- **Test 2 (depth control, layers 0–11 vs 0–23): still not run.** The cleanest remaining attribution test.

**Final position** *(amended 2026-07-31 to match §1.1's 2026-07-28 amendment)*: the guiding principle at the top of this document was applied and the answer landed on the intrinsic side: readout ambiguity is real but secondary; the cross-model landscape differences are properties of the models. The normalisation clause that previously stood here ("inert up to LayerNorm's epsilon term") is withdrawn per §1.1 as amended: the residual path bypasses the LayerNorm, so the map is not scale-invariant; what stands is the narrower ruling that the scalar preserves the mix and is therefore not the source of the Pythia-410m fragmentation. The one place the readout-first principle earns its keep permanently is `Divine`, where dynamics and decoding genuinely come apart.

---

## Amendment (2026-07-26): the closing judgement covered three apparatus channels, and there is a fourth

Nothing above is withdrawn. What follows is a gap in coverage, not a contradiction.

The closing judgement clears normalisation, decoding and readout jitter, and on that basis attributes the
cross-model differences to depth, corpus, width and token geometry. **Tokenisation is not on either list**, and
§1.1b establishes that the two arms of the comparison do not tokenise alike: TransformerLens prepends a
beginning-of-sequence token for GPT-2 and not for the NeoX family, so position 0 holds a special token on one arm
and ordinary content on the other.

That is an apparatus difference *between the models being compared*, which is a stronger category of problem than
an apparatus property they share, and none of the executed tests probes it. Test 1 in particular — the cross-model
`cos_sim` chart the first bullet rests on — is computed over position-indexed tensors whose position 0 means
different things on the two arms.

**Revised position:** the judgement stands as *three apparatus channels exonerated*, not as *apparatus excluded*.
Restoring the stronger reading needs one cheap run: the position-0-excluded recomputation in §1.1b. Until then the
intrinsic attribution is well-supported for the channels tested and open for this one.

This also adds a line to the remaining-work list, alongside Test 2 (the depth control, still not run):

- **Test 4 (tokenisation control): not run.** Recompute position-indexed metrics over aligned slices — GPT-2
  `[1:]` against Pythia `[:]`, not index 0 dropped from both — as a sensitivity check; then, to remove the BOS
  from the computation rather than only from the metric, add a `prepend_bos=False` GPT-2 arm. No forward passes
  required for the first half.
