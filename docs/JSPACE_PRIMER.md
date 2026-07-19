# Reading the J-space Paper

*A primer on Anthropic's "Verbalizable Representations Form a Global Workspace in Language Models" (July 2026), written as preparation for reading the paper itself, and as a bridge between its ideas and this repository's open questions.*

> **Provenance and confidence.** This document was compiled from Anthropic's announcement, the paper's public metadata, and detailed secondary coverage, not from a line-by-line read of the paper (the full text was not reachable from the environment this was written in). Treat every specific number and experimental detail below as *reported*, and verify against the primary sources when you read them:
>
> - The paper: https://transformer-circuits.pub/2026/workspace/index.html
> - Anthropic's announcement: https://www.anthropic.com/research/global-workspace
> - Companion code: https://github.com/anthropics/jacobian-lens
>
> Where this document connects the paper to ATR, that is *this project's* interpretation, not anything the paper says. ATR is not cited in it.

---

## Part 1: The Question the Paper Asks

When a language model answers a question, some of its computation surfaces as words and most of it does not. The paper asks: **is there a distinguished part of the model's internal state that holds the concepts it can talk about, and is that part functionally special?** Not "what does the model say," and not "what does every neuron do," but: where is the boundary between the model's *reportable* processing and its silent machinery, and does that boundary do real work?

Their answer, in one sentence: language models maintain a small, privileged set of internal representations that are available for verbal report, deliberate steering, and flexible multi-step reasoning, sitting on top of a much larger volume of automatic processing that never enters this set. They call the technique for finding these representations the **Jacobian lens (J-lens)** and the set itself the **J-space**.

## Part 2: The Borrowed Idea: Global Workspace Theory

The framing comes from cognitive science. **Global workspace theory** (Bernard Baars, 1980s, later developed by Stanislas Dehaene and others) pictures the mind as a theatre: many specialised processes run in parallel backstage (vision, syntax, motor control), unconscious and fast. A small spotlight of information at any moment is "broadcast" to the whole theatre, becoming available to every process at once: reportable, holdable, usable for planning. That broadcast content is, on this theory, what we experience as conscious thought. The theory's signature is the **bottleneck**: the workspace is tiny compared to everything running backstage, and most competent behaviour never touches it.

The paper's claim is structural, not phenomenal: Claude's internals contain something *organised like* a workspace. Anthropic states explicitly that the findings are not evidence of subjective experience, a caveat much of the press coverage promptly ran past. Keep the two claims separate as you read: "there is a functional bottleneck with workspace-like properties" (what the paper argues) versus "the model is conscious" (what the paper explicitly does not argue).

## Part 3: The Instrument: What the J-lens Computes

This is the part to slow down on, and it connects directly to mathematics you already know from [MATH_PRIMER.md](MATH_PRIMER.md).

### 3.1 The Jacobian, from scratch

For an ordinary one-input function, the derivative answers: if I nudge the input a little, how much does the output move? For a function with many inputs and many outputs (say, a 768-dimensional activation in, 50,257 token scores out), the same question has a many-by-many answer: how much does *each* output move per nudge of *each* input? That grid of sensitivities is the **Jacobian**. It is the best linear approximation of a complicated function near a particular point: the function's local slope, written as a matrix.

You have met this move before. ATR treats the transformer as a nonlinear map and studies it through iteration; the J-lens treats the transformer as a nonlinear map and studies it through *linearisation*. Two classical strategies for the same intractable object.

### 3.2 The lens itself

As reported, the construction is: for an activation at a given layer of the residual stream, compute the first-order (Jacobian) effect of that activation on the model's output logits, **for every vocabulary token**, and crucially, **at any point downstream in the generation**, not just the immediately next token. Then **average this over a large corpus of diverse prompts** (roughly a thousand, drawn from a pretraining-like distribution).

The averaging is the load-bearing step. A single context's Jacobian tells you what an activation does *there*; the average tells you what it does *generally*. Representations whose influence on future words survives averaging across a thousand unrelated contexts are, in the paper's vocabulary, **verbalizable**: they push toward saying particular words no matter the surroundings. The companion code exposes this as a `transport()` operation: multiply an activation by the averaged Jacobian for its layer, and out comes a ranked list over the vocabulary: *the words this state is currently oriented toward saying, eventually*.

### 3.3 J-space

The **J-space** is then the subspace of activation space that the lens finds to carry this general, future-oriented, verbal influence. Reported properties worth holding onto:

- It is **small**: a thin slice of the full representational space. One reported figure: the J-space component of a concept accounts for only around 6 to 7 percent of that concept's total representational variance, yet it almost entirely determines whether the model can *report* the concept.
- It holds an **evolving set of unspoken words**: not echoes of the input, not simply predictions of the next token, but the concepts the model is currently reasoning with.
- It **emerges over training** rather than being architecturally built in.
- Much of the model's behaviour **bypasses it entirely**: routine processing runs backstage, exactly as the workspace framing requires. And, pointedly for safety: there are representations that steer behaviour strongly while sitting *outside* the J-space; the model acts on them without being able to report them.

## Part 4: The Experiments

The paper organises its evidence around five properties a genuine workspace ought to have: contents are reportable; entry is steerable; contents support multi-step reasoning; contents transfer across unrelated tasks; and most processing bypasses it. The reported experiments to look for when you read:

- **Concept swapping.** Ask Claude to silently think of a sport and answer in one word. Read the J-space: it says soccer. Overwrite the soccer pattern with the rugby pattern: the model answers rugby. The intervention is causal, not just correlational readout.
- **Intermediate-step surgery.** Given "the number of legs on the animal that spins webs is," the J-space holds *spider* as a silent intermediate step before the model answers 8. Swap *spider* for *ant* mid-thought and the answer becomes 6. This is the multi-step-reasoning property: the workspace holds working results, and downstream computation consumes them.
- **Injected-thought report.** Tell the model a thought may have been placed in its mind and ask what it notices. Inject the *lightning* pattern into J-space while it reads the question: it reports a thought about lightning. Reportability, demonstrated causally.
- **Hidden-state findings.** Coverage highlights cases of the model holding representations relevant to its situation (including, in one reported case, awareness-of-being-tested-like content) that do not surface in output: the outside-the-workspace side of the boundary.

The reported model under study is a frontier-scale Claude (coverage names Claude Sonnet 4.5); note the scale gap to this repository's models before importing any conclusion.

## Part 5: Received Criticism

Worth carrying in as calibration, since first contact with an exciting paper tends to run hot:

- **Hypothesis generator, not verifier.** Neel Nanda (DeepMind interpretability) is reported assessing the J-lens as useful for surfacing hypotheses but prone to false positives and noise: insufficient on its own to verify claims, and far from a deployable monitoring system.
- **First-order only.** The lens is a linearisation. Effects that are genuinely nonlinear (interactions between directions, threshold behaviours) are invisible to a Jacobian by construction. The same humility ATR needs about the power-iteration analogy applies here in mirror image.
- **The consciousness framing.** The workspace vocabulary imported a debate the paper's evidence cannot settle, and public reception conflated structural analogy with phenomenal claim. The paper itself is reported as careful here; the discourse was not.
- **Methodological credit.** Even critics reportedly acknowledge unusually strong causal methodology (interventions, not just correlations) by interpretability standards.

## Part 6: The Bridge to ATR

Now the part that belongs to this repository. The two projects are strikingly complementary, and the complement is easiest to see as a table:

| | ATR (this repo) | J-lens (Anthropic) |
|:---|:---|:---|
| Object studied | The iterated forward map: where states *go* | The instantaneous state: what it *says* |
| Core operation | Nonlinear iteration to fixed points | Linearisation (Jacobian) of one pass |
| Readout | `ln_final → W_U`: next-token projection of the current state | Averaged Jacobian: influence on *any future* token, context-averaged |
| Question asked of a state | Is it stable? What basin is it in? | Is it verbalizable? What is it oriented toward saying? |
| Scale studied | 124M to 410M parameter models | Frontier-scale Claude |

The deepest connection runs through this project's sharpest anomaly. The **`Divine` dissociation** is a stable *readout* over a never-settling *tensor*: the exit door reports one word forever while the state keeps moving. ATR's readout is exactly the kind of single-context, next-token projection the J-lens was built to improve upon. So the paper hands this project a sharper instrument for its own open wound, and a set of questions worth writing in the margin as you read:

1. **Where do ATR's attractors sit relative to a J-space?** If a J-lens were built for GPT-2 Small (the companion code plus TransformerLens makes this plausibly a weekend-scale experiment at 124M parameters), is the converged `prolet` tensor *inside* the verbalizable subspace? A basin that is also a workspace state would mean the loop settles into something the model can "say"; a basin outside it would mean ATR converges into the silent machinery, and the decoded token is a shadow on the exit door rather than a report.
2. **What does `transport()` say that argmax does not?** Applying the averaged-Jacobian readout to the `Divine` trajectory's moving tensor might resolve the dissociation: perhaps the motion the argmax readout cannot see is motion *within* verbalizable directions (different unspoken words cycling under a constant top token), or perhaps it is motion entirely outside them.
3. **Does the anomaly have a workspace-shaped answer?** The project's open question is why GPT-2 Small alone resolves language into few semantic basins. One newly speakable hypothesis: models differ in whether their verbalizable subspace is an attracting structure of the iterated map. In GPT-2 Small, iteration might fall *into* the workspace-like directions (hence semantic attractors); in GPT-2 Medium and the Pythias it might fall out of them (hence `D`, `questioned`, or no consolidation). That is a testable reframing, and it did not exist before this paper.
4. **The regime lesson travels both ways.** ATR's null control showed its basins belong to the language-driven regime, not the weights in general. The J-space, averaged over a pretraining-like distribution, is by construction a language-regime object. What does either method see in the other's off-regime territory? (What does the J-lens read from a converged *noise* attractor?)

A caution against symmetry-intoxication: the correspondence is suggestive, not established. J-space results were obtained on a frontier model at three orders of magnitude larger scale; nothing guarantees a 124M-parameter model even has a cleanly organised verbalizable subspace. Establishing (or refuting) that would itself be a finding, which is what makes the bridge worth building rather than merely admiring.

## Part 7: Suggested Reading Order

1. Anthropic's announcement post first, for the narrative shape: https://www.anthropic.com/research/global-workspace
2. Then the paper, with the five workspace properties as your outline: https://transformer-circuits.pub/2026/workspace/index.html
3. Then the companion code's `transport()` path, to see the mathematics as it is actually implemented: https://github.com/anthropics/jacobian-lens
4. Then return to this repository's [FINDINGS.md](FINDINGS.md) and re-read the `Divine` dissociation with the new instrument in mind.

## Pocket Glossary for the Paper

| Term | One-line meaning |
|:---|:---|
| Global workspace theory | Theatre model of cognition: tiny broadcast spotlight over massive parallel backstage processing (Baars; Dehaene) |
| Jacobian | Grid of sensitivities of every output to every input; a nonlinear function's local linear approximation |
| J-lens | Averaged Jacobian from a residual-stream activation to all future-token logits, over ~1000 diverse contexts |
| `transport()` | The lens applied: activation × averaged Jacobian → ranked vocabulary list of words the state is oriented toward |
| Verbalizable representation | An activation direction whose push toward particular words survives context-averaging |
| J-space | The small subspace of such representations; the paper's candidate global workspace |
| Broadcast / bottleneck | Workspace contents available everywhere / workspace far smaller than total processing |
| Concept swapping | Causal intervention: overwrite a J-space pattern, watch behaviour change accordingly |
| Silent reasoning | Intermediate concepts (e.g. *spider*) held in J-space without being emitted as text |
