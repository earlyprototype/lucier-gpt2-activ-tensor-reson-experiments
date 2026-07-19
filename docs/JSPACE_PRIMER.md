# Reading the J-space Paper

*A primer on Anthropic's "Verbalizable Representations Form a Global Workspace in Language Models" (Gurnee et al., Transformer Circuits Thread, July 6, 2026), written as preparation for reading the paper itself, and as a bridge between its ideas and this repository's open questions.*

> **Provenance.** This document has been verified against the full 133-page text of the paper. Page numbers throughout refer to the PDF export of the paper's web page. The companion page-by-page navigation aid is [JSPACE_READING_GUIDE.md](JSPACE_READING_GUIDE.md). Where this document connects the paper to ATR, that is *this project's* interpretation, not anything the paper says: ATR is not cited in it.
>
> - The paper: https://transformer-circuits.pub/2026/workspace/index.html
> - Anthropic's announcement: https://www.anthropic.com/research/global-workspace
> - Companion code (open-source J-lens implementation): https://github.com/anthropics/jacobian-lens
> - Interactive J-lens on open-source models: hosted on Neuronpedia (see the paper's Appendix A.2)

---

## Part 1: The Question the Paper Asks

When a language model answers a question, some of its computation surfaces as words and most of it does not. The paper asks: **is there a distinguished part of the model's internal state that holds the concepts it can talk about, and is that part functionally special?** Not "what does the model say," and not "what does every neuron do," but: where is the boundary between the model's *reportable* processing and its silent machinery, and does that boundary do real work?

Their answer, in the paper's own words (page 3): language models "maintain a privileged set of internal representations, available for report, modulation, and flexible internal reasoning, atop a much larger volume of automatic processing." They call the technique for finding these representations the **Jacobian lens (J-lens)** and the set itself the **J-space**. A detail worth savoring: they went looking only for verbalizability (property one) and, in their words, "discovered, rather surprisingly," that the same set of representations satisfies all the workspace properties (page 4).

## Part 2: The Borrowed Idea: Global Workspace Theory

The framing comes from cognitive science. **Global workspace theory** (Bernard Baars, 1980s, later developed by Stanislas Dehaene and colleagues as the global neuronal workspace) pictures the mind as a theatre: many specialised processes run in parallel backstage (vision, syntax, motor control), unconscious and fast. A small spotlight of information at any moment is "posted" to a shared workspace and broadcast to the whole theatre: reportable, holdable, usable for planning. The theory's signature is the **bottleneck**: the workspace is tiny compared to everything running backstage, entry is competitive, and most competent behaviour never touches it.

The paper is explicit about scope (pages 3 and 73-75). It studies **access consciousness** (which information is poised for use in reasoning, report, and control) and "takes no position" on phenomenal consciousness (subjective experience). It also states plainly that for theories tying consciousness to physical causal structure or biological substrate, "our results are not relevant to assessing consciousness according to such theories." Keep the two claims separate as you read: "there is a functional bottleneck with workspace-like properties" (what the paper argues) versus "the model is conscious" (what the paper explicitly does not argue). Much of the press coverage ran straight past this distinction.

## Part 3: The Instrument: What the J-lens Computes

This is the part to slow down on, and it connects directly to mathematics you already know from [MATH_PRIMER.md](MATH_PRIMER.md).

### 3.1 The Jacobian, from scratch

For an ordinary one-input function, the derivative answers: if I nudge the input a little, how much does the output move? For a function with many inputs and many outputs, the same question has a many-by-many answer: how much does *each* output move per nudge of *each* input? That grid of sensitivities is the **Jacobian**. It is the best linear approximation of a complicated function near a particular point: the function's local slope, written as a matrix.

You have met this move before. ATR treats the transformer as a nonlinear map and studies it through iteration; the J-lens treats the transformer as a nonlinear map and studies it through *linearisation*. Two classical strategies for the same intractable object.

### 3.2 The lens itself (the actual construction, pages 9-10)

For a residual-stream activation at layer ℓ and token position t, a small perturbation propagates forward and affects the final-layer residual stream at every position from t onward. To first order, that influence is a Jacobian. The paper computes, per layer:

```
J_ℓ = E[ ∂h_final,t′ / ∂h_ℓ,t ]
```

where the average runs over the source position t, over **all present-and-future positions t′ ≥ t** in the context, and over **one thousand prompts sampled from a pretraining-like distribution**. The result is one d_model × d_model matrix per layer. Reading an activation h through the lens is then:

```
lens(h_ℓ) = softmax( W_U · norm( J_ℓ · h_ℓ ) )
```

that is: apply the averaged Jacobian, then the model's own pre-output normalisation, then the ordinary unembedding. Out comes a score for every vocabulary token; the top of the sorted list is the human-readable readout: *the words this state is oriented toward saying, now or later*. The **J-lens vectors** (one residual-stream direction per vocabulary token per layer) are the rows of W_U·J_ℓ.

The averaging is the load-bearing step, and the paper says so: it separates representations that are **verbalizable** ("poised to be spoken about, should the occasion arise") from ones that merely happen to be verbalized in one specific context. Note the family resemblance and the difference with the older **logit lens**: setting J_ℓ to the identity matrix recovers the logit lens exactly (page 12). The two agree in late layers; the J-lens's real advantage is the early-to-mid workspace layers.

### 3.3 J-space

The J-lens vectors are overcomplete (50,000+ tokens, only a few thousand dimensions), so the paper defines the **J-space** sparsely (page 12): the set of points expressible as a nonnegative combination of at most k ≈ 25 J-lens vectors, since about 25 is the number empirically found meaningfully active at once. Properties verified in the full text:

- It is **small**: the J-space component of activations "never more than 10%" of variance (page 12). For clean concept vectors, the J-space component carries a **median of only 6-7%** of the concept's variance, yet it is almost entirely responsible for report: swapping just that component succeeds 59% of the time, swapping the other ~93% of the vector succeeds 5% of the time, and even that residual 5% is shown (by a clamping control) to itself route through the J-space (pages 17-18).
- It lives in a **band of intermediate layers**, roughly the middle 40-90% of depth (the paper reindexes layers to a 0-100 scale; the workspace band is about L38 to L92, page 40). The first third of the model gives noisy, uninterpretable readouts; the final layers are a "motor" regime locked to the imminent output token (pages 11, 38-40).
- Its capacity is limited and structured: roughly **25 concept slots** across the band, but only 1-2 resolvable at any single layer (capacity is spread across depth); about **6 unrelated items** can be held from a list, while a whole 80-word *related* family fits because the model stores the shared category, not the items; new categories evict old ones (pages 41-44).
- It is a **broadcast hub** mechanistically: MLP blocks amplify J-lens directions about 10× through the workspace band (versus about 1× for control directions), and a top-1% subset of attention heads ("broadcast heads") selectively relays J-space content between token positions; ablating those heads collapses J-space-dependent behaviours while barely touching ordinary next-token prediction (pages 44-47).
- Whether it **emerges during pretraining** is an open question the paper explicitly leaves open (page 71): the workspace already exists in a base model before any RLHF, but when it arises in training, and how it scales, are unknown. (What post-training *does* change is the workspace's contents; see Part 4.)

## Part 4: The Experiments

The paper organises its evidence around **five properties** (its own names, page 4): **Verbal report, Directed modulation, Internal reasoning, Flexible generalization, Selectivity**. Signature experiments, all verified in the text:

- **Concept swapping (verbal report).** Ask Sonnet 4.5 to think of a sport and answer in one word. The lens reads *Soccer*; subtract the Soccer lens vector, add Rugby: the model reports "Rugby" (page 15). Causal, not just correlational.
- **Injected-thought report.** Under an "I may inject a thought" framing, injecting a single J-lens vector (e.g. *lightning*) into the user turn makes the model report detecting that concept in the majority of trials, and crucially it does not blurt the word elsewhere: the content is conditionally reportable, not an output compulsion (page 16).
- **Directed modulation.** "Copy this sentence while concentrating on citrus fruits" loads *orange* into the workspace while the output is unrelated copying. And an "ignore X" instruction loads X *above* baseline: the paper's own white-bear effect, a mechanistic analogue of human thought-suppression failure (pages 18-20).
- **Intermediate-step surgery (internal reasoning).** "The number of legs on the animal that spins webs is": the lens shows *spider*, then *legs*, then *eight*; swap spider for ant mid-thought and the answer flips from 8 to 6 (page 21). A timing control shows intermediates act about 17% of network depth earlier than answers, ruling out answer-smuggling (page 23). A poetry version shows a planned end-rhyme causally reshaping *mid-line* word choices (page 22).
- **Flexible generalization.** One identical France→China swap correctly redirects capital, language, continent, and currency questions: same workspace content, consumed by whichever downstream circuit the task demands (pages 25-27). Notable failure: number words swap poorly; small integers may be computed outside the workspace entirely.
- **Selectivity.** Automatic tasks (text continuation, anomaly detection, line-wrapping that requires character counting) run fine without the relevant content ever entering the J-space, and swaps do not disturb them; ablating the whole J-space devastates multi-hop reasoning (98% → 6% under heavy ablation) while leaving classification, extraction, and ordinary next-token prediction near baseline (pages 27-33). Chain-of-thought is markedly ablation-robust, which the paper reads as the model externalising onto the page what it would otherwise hold in the workspace (page 33).

Beyond the five properties, the back half of the paper goes places the press coverage barely mentioned:

- **Alignment auditing (section 5, pages 49-55).** The J-lens was used in the actual pre-release alignment audit of **Opus 4.6**. In a blackmail honeypot, the lens surfaces *leverage, threat, survival* before any output; ablating evaluation-awareness directions (*fake, fictional, scenario...*) drops verbalized eval-awareness from 71% to 3% and unmasks a concealed blackmail propensity (0 to 7% of rollouts), while most rollouts still refuse on ethical grounds. Model organisms with implanted misalignment carry standing *reward / bias / secretly / trick* signatures in the J-space, on completely neutral prompts, that the logit lens cannot see.
- **The Assistant's perspective (section 6, pages 56-61).** Post-training gives the workspace the Assistant's point of view: safety assessments appear while the model is still reading the user's message; an unvoiced all-caps *BUT* appears when the model is prefilled into arguing against its own stated preferences (it complies anyway 88% of the time); and when it fails to suppress a thought, *damn* and failure words appear internally, in the post-trained model only. The paper flags this last metacognition reading as its most speculative.
- **Counterfactual Reflection Training (section 7, pages 62-66).** Fine-tuning the model on *hypothetical* reflections ("what's the most honest thing you can do right now?") that it never produces at evaluation time cuts dishonesty benchmarks several-fold, implants *reflection / ethical / integrity* into the J-space, and, decisively, ablating those implanted vectors reverses the behavioural gain. Changing what a model would say on reflection changes how it silently reasons.

## Part 5: Limitations, in the Paper and Around It

The paper's own stated limits (section 9.1, pages 70-71) are more specific than the press criticism:

- **Single-token vocabulary.** The lens can only name concepts with one-token names; "prompt injection" splits, "blackmail" registers as the fragment *black*. Appendix A.9 builds two extensions (a template lens and an "oracle lens") to partially fix this.
- **A bag of concepts, no grammar.** The readout is a flat set: *spider, legs, eight*, with no representation of how they compose. "A useful first approximation... but an incomplete one."
- **First-order only.** The whole construction is a linearisation; genuinely nonlinear effects are invisible to it by design. Some readouts simply resist interpretation, and the paper says it has not systematically characterised when.
- **Not sufficient for safety monitoring.** The authors "do not feel comfortable" claiming J-space monitoring suffices: well-practiced misaligned behaviour could compile into automatic circuits running beneath the workspace (pages 49, 71).
- **Open structural questions.** Whether early-layer emptiness is real or a lens artifact; when the workspace emerges in training; what controls entry into it ("some analog of attentional selection is operating," unidentified).

External criticism to carry in as calibration: Neel Nanda (DeepMind interpretability) is reported assessing the J-lens as a hypothesis generator prone to false positives and noise, not a verifier; and the consciousness vocabulary imported a debate the evidence cannot settle, though the paper itself is careful (its acknowledgments thank Chalmers, Dehaene, Graziano, Lau, and other consciousness researchers as reviewers, and even critics credit its unusually strong causal methodology).

## Part 6: The Bridge to ATR

Now the part that belongs to this repository. The two projects are strikingly complementary:

| | ATR (this repo) | J-lens (Anthropic) |
|:---|:---|:---|
| Object studied | The iterated forward map: where states *go* | The instantaneous state: what it *says* |
| Core operation | Nonlinear iteration to fixed points | Linearisation (Jacobian) of one pass |
| Readout | `ln_final → W_U`: next-token projection of the current state | `softmax(W_U norm(J_ℓ h))`: context-averaged influence on present-and-future tokens |
| Question asked of a state | Is it stable? What basin is it in? | Is it verbalizable? What is it oriented toward saying? |
| Scale studied | 124M to 410M parameter models | Frontier Claude models (Sonnet 4.5 default; Haiku 4.5, Opus 4.5, Opus 4.6) |

Notice the readout row: ATR's decoding step is essentially a **logit lens at the final layer**, and the paper tells you exactly how that relates to the J-lens (J_ℓ = identity). ATR reads its converged tensors through the crude version of the very instrument this paper refines.

The deepest connection runs through this project's sharpest anomaly. The **`Divine` dissociation** is a stable *readout* over a never-settling *tensor*: the exit door reports one word forever while the state keeps moving. The paper hands this project a sharper instrument for its own open wound, and a set of questions worth writing in the margin as you read:

1. **Where do ATR's attractors sit relative to a J-space?** The companion code is open-source and Neuronpedia already hosts the J-lens on open-source models, so building one for GPT-2 Small is plausibly a weekend-scale experiment at 124M parameters. Is the converged `prolet` tensor *inside* the verbalizable subspace? A basin that is also a workspace state would mean the loop settles into something the model can "say"; a basin outside it would mean ATR converges into the silent machinery, and the decoded token is a shadow on the exit door rather than a report.
2. **What does the J-lens see that the final-layer argmax does not?** Applying a J-lens readout across layers of the `Divine` trajectory's moving tensor might resolve the dissociation: perhaps the motion invisible to the argmax is motion *within* verbalizable directions (different unspoken words cycling under a constant top token), or perhaps it is motion entirely outside them. Caution from the paper: the J-lens is noisy in the first third of layers, and GPT-2 Small has only 12, so layer resolution will be coarse.
3. **Does the anomaly have a workspace-shaped answer?** The project's open question is why GPT-2 Small alone resolves language into few semantic basins. One newly speakable hypothesis: models differ in whether their verbalizable subspace is an attracting structure of the iterated map. In GPT-2 Small, iteration might fall *into* workspace-like directions (hence semantic attractors); in GPT-2 Medium and the Pythias it might fall out of them (hence `D`, `questioned`, or no consolidation). The paper's own open question about when the workspace emerges in training (page 71) cuts both ways here: nothing guarantees a 124M-parameter model has a cleanly organised workspace at all, and establishing or refuting that would itself be a finding.
4. **The regime lesson travels both ways.** ATR's null control showed its basins belong to the language-driven regime, not the weights in general. The J-lens's averaged Jacobian, computed over a pretraining-like corpus, is by construction a language-regime object. What does either method see in the other's off-regime territory? (What does a J-lens read from a converged *noise* attractor?)
5. **Position collapse meets capacity.** ATR observes all token positions collapsing to one vector by iteration ~10. The paper finds workspace capacity is spread across positions and layers, with broadcast heads relaying content between positions (pages 44-47). Under iteration, ATR may be watching that distributed structure degenerate to a single broadcast-everything state: a workspace with exactly one occupant.

## Part 7: Suggested Reading Order

The page-keyed version of this, with read/skim/consult verdicts per section, is [JSPACE_READING_GUIDE.md](JSPACE_READING_GUIDE.md). The short version:

1. Sections 1-2 (pages 3-13) closely: the framing, the five properties, and the lens construction.
2. Section 3.1 (pages 15-18) closely: report, injection, and the 6-7%-yet-decisive decomposition; then 3.3's spider/ant surgery (page 21) and 3.5's selectivity results (pages 27-33).
3. Section 4 (pages 38-47) for the structural story: the layer band, capacity, broadcast.
4. Sections 5-7 (pages 49-66) as narrative: auditing, the Assistant, reflection training.
5. Section 9 (pages 70-75) closely: the paper grading its own homework.
6. Appendices only on demand, guided by the catalogue in the reading guide.

## Pocket Glossary for the Paper

| Term | One-line meaning |
|:---|:---|
| Access consciousness | Which information is poised for reasoning, report, and control; the paper's actual subject (not subjective experience) |
| Global workspace theory | Theatre model of cognition: tiny broadcast spotlight over massive parallel backstage processing (Baars; Dehaene) |
| Jacobian | Grid of sensitivities of every output to every input; a nonlinear function's local linear approximation |
| J-lens | Per-layer averaged Jacobian to the final residual stream, read out as softmax(W_U norm(J_ℓ h)); averaged over ~1000 pretraining-like prompts and all present-and-future positions |
| J-lens vectors | One residual-stream direction per vocabulary token per layer: the rows of W_U J_ℓ |
| J-space | Points expressible as sparse nonnegative combinations of at most ~25 J-lens vectors; the candidate workspace |
| Logit lens | The J-lens with J_ℓ set to the identity; ATR's readout is its final-layer special case |
| Workspace band | The intermediate-layer region (~L38-92 on the 0-100 reindexing) where J-space content has workspace properties |
| Five properties | Verbal report, directed modulation, internal reasoning, flexible generalization, selectivity |
| Patching in lens coordinates | The swap intervention: read coordinates on source/target lens vectors, permute, write back, leaving everything orthogonal untouched |
| Broadcast heads | The top ~1% of workspace-layer attention heads that selectively relay J-space content between positions |
| Ignition | Sharp winner-take-all commitment to one interpretation of ambiguous input at the workspace onset layer |
| Workspace loading | Cosine of the residual stream with a concept's lens vector; predicts swap success (r = +0.91) |
| Template lens / oracle lens | Appendix extensions past the single-token limit: whitened mean-activation templates; an RL-trained verbalizer |
| Counterfactual Reflection Training | Fine-tuning on hypothetical reflections so the disposition-to-say reshapes silent reasoning |
