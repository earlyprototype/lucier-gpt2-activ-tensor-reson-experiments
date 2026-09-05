# SESSION 06 READING NOTE: the latent-reasoning taxonomy, and where this loop sits in it

*Date: 2026-09-03. This is a reading note, not a results record. It parses a conversation between TC and a Claude Code session into something that can be read later, in order, in one sitting. Nothing here changes `FINDINGS.md`. It accompanies the pull request that adds the taxonomy entry to `docs/PRIOR_WORK.md` and one paragraph to the README. Working agreements from `SESSION_03_HANDOVER.md` apply: no em dashes, plain language, technical terms defined where they are used. Every claim below is marked as established (from the committed record or from a source read directly), inferred (a conclusion drawn from established facts), or speculation.*

## Start here: five sentences

The looping operation this project runs is, in the vocabulary of a July 2025 survey of "latent reasoning", vertical recurrence with every control removed: no fresh input on any pass, no carried state, no training, and a stop rule that waits for a fixed point. The survey names that regime once, as "self-iteration in the absence of input tokens", and lists no evidence for it. The survey contains no dynamical-systems vocabulary at all: the words collapse, attractor, equilibrium and oscillation do not occur in its 38 pages. The failure mode that later looped-model papers call latent collapse, a hidden state falling into an input-independent fixed point, is this project's headline measurement. None of the works the survey maps closes the output-to-input loop on a frozen, unmodified pretrained model, and none maps basins, so the record's scoped novelty claims one and two stand after the check.

## 1. What prompted this

TC asked a general Claude chat how chain of thought works. The answer said that chain of thought happens through ordinary autoregressive generation: the same weights are run again on a sequence that is one token longer each time, with no loop between layers inside a single forward pass. It added that research exists on looped or recurrent-depth transformers, and on latent reasoning that feeds hidden states back in instead of decoding them to tokens. A follow-up question produced a literature map with GPT-2 and Pythia touchpoints, and TC then pointed at the survey the map ended with: Zhu et al., A Survey on Latent Reasoning, arXiv 2507.06203.

The pause for thought was that the map looks like this project, on this project's models. This note records what was found when the map and the survey were checked against the record.

## 2. Three structural differences between chain of thought and this loop

Established, from the engine (`atr_engine.py`, the `run_atr_loop` and gated-run functions) and from `docs/TECHNICAL.md`: each pass takes the residual stream at the output of the last layer, for every token position at once, rescales it to a fixed energy, and overwrites the input to layer 0 with it. The prompt tokens are scaffolding only after the first pass. The sequence length never changes.

Chain of thought differs in three ways.

1. **Chain of thought passes through a token at every step.** Each step decodes to one discrete symbol out of 50,257 and re-embeds it. This loop never decodes inside the loop.
2. **Chain of thought appends.** Every past token stays in the context, so the state grows. This loop replaces the whole state each pass, so the only memory is the state itself.
3. **Chain of thought runs at natural scale.** This loop pins the energy of the fed-back state at 67 to 72 times what layer 0 naturally receives on GPT-2 Small (`FINDINGS.md`, abstract), and about 220 times on GPT-2 Medium (the sibling repository's operator report of 2026-07-31).

Inferred: together these are why one process accumulates information and the other destroys it. The record's clearest number is that 54 of 125 prompts on GPT-2 Small end at one vector, identical to seven significant figures across unrelated prompts (`ALIGNMENT_REVIEW.md` section 1, from issue #98 and finding F11), so the settled state carries no information about the input. The looped-model literature has a name for that outcome, latent collapse, and engineers against it; `docs/PRIOR_WORK.md` records the engineering (Soft Thinking's entropy-threshold early exit, and the 2026 STARS and Attractor Models preprints). The record's own sentence is the positioning: what that literature suppresses or exploits, this project studies descriptively in a model never trained for the loop.

## 3. Where the loop sits in the survey's taxonomy

Established, from the survey read in full.

The survey divides latent reasoning into two families. **Vertical recurrence**, which it also calls activation-based, applies the same layers again to add depth at one time step. **Horizontal recurrence**, which it also calls hidden-state-based, evolves a compressed hidden state across a growing sequence, as linear-attention and test-time-training models do. This loop is vertical recurrence in its pure form: the whole stack is the looped block, every position is overwritten, and the sequence never grows.

Within vertical recurrence the survey separates two branches. **Architectural looping** covers Universal Transformer, CoTFormer, Recursive Transformer, AlgoFormer and Recurrent-Depth (Huginn). **Explicit hidden-state feedback** covers Coconut, where the last hidden state is inserted as a new sequence element. The record files this project beside Coconut because the feedback edge, last hidden state into the input embedding, is the same. That is right about the edge. The topology, fixed length with replacement and the whole block looped, belongs to the architectural branch. Both statements are now in the prior-work entry.

The survey's design table for the architectural branch (its Table 1) has five axes. This loop sits at one corner of all five.

| Axis in the survey's table | What the surveyed architectures do | What this loop does |
|:--|:--|:--|
| Per-iteration input | Huginn re-injects the embedded input every pass; the record notes this is why Huginn's attractors do not depend on the starting state | None. The input is consumed once, on the first pass |
| Carried hidden state | Most keep a key-value cache across iterations | None. The whole sequence is recomputed each pass |
| Dynamic stop | Recursive Transformer stops when the largest per-token change falls below a threshold; Huginn explores fixed-point criteria | A convergence gate on the cosine between successive iterates, with the lag-k correction that recognises a two-step cycle |
| Depth embedding | Tried by early designs, dropped by later ones | None |
| Training for the loop | All of them | None |

One sentence in the survey names this regime. Discussing Soft Reasoning on its page 17, it says such methods "suggest significant potential, particularly for enabling self-iteration in the absence of input tokens", and that no evidence for it yet exists. Self-iteration in the absence of input tokens is this apparatus. Whether the survey's authors would count a frozen model as a method is a fair question; the phrase is still the nearest name the field has given the object.

## 4. What the survey does not contain

Established, by counting over the extracted full text.

- The words collapse, attractor, equilibrium and oscillation occur zero times.
- "Fixed point" occurs nine times, every time as a design choice: CODI "effectively learns a fixed-point iteration in activation space", Huginn's stop criterion, implicit fixed-point recurrent networks, Infini-attention's associative memory. Never as an object that was measured.
- "Converge" occurs eight times, as a stop criterion or as a figure of speech about the field.
- The survey is dated July 2025. The papers that name latent collapse and engineer against it are dated 2026 and are already in `docs/PRIOR_WORK.md`.

Inferred: the gap this project occupies is visible in the survey by its absence. The survey describes what the loop is built to do. Nobody in it asks what the loop does when nothing is training it.

## 5. The bandwidth premise, read against finding 1

Established: the survey's first figure claims a bandwidth advantage for latent reasoning, about 15 bits per token against about 40,960 bits per hidden state, a ratio of roughly 2,700 to 1.

Inferred: finding 1's shared attractor is what that channel carries under free iteration of a model never trained for the loop, at this apparatus's injection scale: nothing about the input. Bandwidth available is not bandwidth retained. Training is what makes the wide channel carry anything; the untrained map, at this scale, drives it to a point or a two-step cycle. The standing caveat travels with this: the record establishes it at 67 to 72 times natural scale, the ν-sweep (`SESSION_05_HANDOVER.md`) shows the five basins exist only in a band from about 50 to about 300 times natural, and below that band the loop steps through other words or oscillates rather than settling.

## 6. Stage 2 in the survey's vocabulary

The sibling repository (`ATR_research`, Stage 2) borrowed a three-band theory of layers from the Anthropic workspace paper: an input-parsing band, a workspace band, a motor band. The survey's interpretability section rests on the same three-band picture under other names: shallow layers parse, intermediate layers reason, deep layers decide. The survey's Prelude, Loop, Coda structure, which it says trained looped architectures have converged on, is a window loop that injects at one layer and reads at a deeper one.

Inferred: Stage 2's census of every such cut on GPT-2 Medium is, in the survey's terms, a census of every Prelude, Loop, Coda partition of a frozen model. The sibling repository's operator report records what that census found: the collapse to the single token "D" lives in one cut of 300 and needs about 220 times natural injection strength; the word-producing cuts are isolated cells, not a band; and the lens instrument finds no coherent band. In the survey's words, trained loops converge on a Prelude, Loop, Coda structure; Stage 2 says the untrained model does not come with one. That sentence belongs in the sibling repository's own record and is offered here only as a way of stating the result to a reader who knows this literature.

## 7. The token-level loop: what everyone already does, and the nuance

TC asked whether feeding tokens back into a model is not simply what everyone does when they first get access to one. Yes, and the record already covers that version. Everyone's version appends: keep generating, or paste the output back in as the next prompt, or run the telephone game, or train the next model on the last one's output. Its attractors are the repetition loops Holtzman et al. measured in GPT-2 and the paraphrase cycles Wang et al. found, both recorded in `docs/PRIOR_WORK.md` under text-level loops, and model collapse across training generations, recorded there too.

The version raised in this conversation is different in one mechanical respect and is not, as a phenomenon, new. It holds the sequence length fixed and replaces tokens rather than appending them, which turns the loop into a map on a finite set, so it must end in a cycle and its cycles can be counted. Two variants, both inferred from the causal mask, which lets each position see only itself and the positions before it:

- **Shifted.** Each position receives the token predicted by the position before it. This is Jacobi decoding: its unique fixed point is greedy generation from the first token, reached within one pass per position. Known, and in use as a speed technique (PCCOT, in the reading list). Not exploratory.
- **Unshifted.** Each position is replaced by its own predicted next token. Position 0 then runs greedy generation with a one-token context window, a map on the 50,257-token vocabulary whose cycles can be enumerated exactly with 50,257 single-token forward passes and no convergence gate. Whether anyone has published that census is not known to this session; it is the kind of thing done informally and rarely written up.

The nuance is therefore modest and specific. As a standalone phenomenon the token loop is not interesting. As the discretised twin of this loop on the same weights, it is the cheapest test of the inference in section 2, that the token bottleneck is what prevents collapse. It also runs at natural scale by construction, which removes the loudness question and, for the same reason, makes it a different map rather than a control. It was registered on TC's ruling of 2026-09-03 as issue #141, and does not start until a run number is claimed on the Identifier registry.

## 8. Reading list

Status tags follow `docs/PRIOR_WORK.md`. "Read level" says what this session actually read: full text, abstract only, or the survey's description of it. Works already in the prior-work record before this session are marked.

| Work | Identifier | Status | Read level | Why it matters here |
|:--|:--|:--|:--|:--|
| Zhu et al., A Survey on Latent Reasoning | arXiv 2507.06203 | preprint | full text | The field's map. Sections 2.1, 3.1, Table 1 and page 17 are the parts that place this loop |
| Geiping et al., Huginn (Recurrent-Depth) | arXiv 2502.05171 | preprint | in record | The trained loop nearest to this one; re-injects the input every pass, which pins its attractors |
| Hao et al., Coconut | arXiv 2412.06769 | preprint | in record | The same feedback edge, on GPT-2, built by training, a few steps, no attractor map |
| Zhang et al., Soft Thinking | arXiv 2505.15778 | preprint | in record | Training-free feedback through the output distribution; meets collapse and adds an early exit |
| Blayney et al., Mechanistic Analysis of Looped Reasoning LMs | arXiv 2604.11791 | preprint | in record | The other located analysis of looped-model internals; the lag-1 gate in use |
| Lu et al., Latent Chain-of-Thought? | arXiv 2507.02199 | peer-reviewed | in record | Lens readouts on Huginn's recurrent block disagree block by block |
| Shen et al., CODI | arXiv 2502.21074 | preprint | abstract | Coconut's class, with self-distillation; the first latent method to match explicit chain of thought on GSM8K at GPT-2 scale |
| Wu, Teng, Tu, PCCOT | arXiv 2506.18582 | preprint | survey's description | Jacobi iteration over continuous thoughts; the fixed-point iteration used for speed |
| Zeng et al., Pondering LM | arXiv 2505.20674 | preprint | survey's description | Softmax-weighted vocabulary embedding fed back, built in at pretraining |
| Zhu et al., Soft Reasoning | arXiv 2505.24688 | preprint | survey's description | Source of the "self-iteration in the absence of input tokens" sentence |
| Mohtashami et al., CoTFormer | arXiv 2310.10845 | preprint | survey's description | Design-table entry; interleaves activations back into the sequence |
| Bae et al., Relaxed Recursive Transformers | arXiv 2410.20672 | preprint | survey's description | Design-table entry; early exit on maximum change |
| Gao et al., AlgoFormer | arXiv 2402.13572 | preprint | survey's description | Design-table entry; fixed iteration count |
| Schöne et al., Implicit language models are RNNs | ICML 2025 | peer-reviewed | survey's description | A state block iterated to convergence at every token; horizontal, not vertical |
| Saunshi et al., Reasoning with latent thoughts | arXiv 2502.17416 | preprint | not read | The survey's theoretical anchor: a looped model can simulate chain-of-thought steps |
| Deng et al., Implicit CoT via Knowledge Distillation | arXiv 2311.01460 | preprint | abstract | Boundary of the term: reasoning trained into a single pass, no loop |
| Deng, Choi, Shieber, From Explicit CoT to Implicit CoT | arXiv 2405.14838 | preprint | abstract | Boundary of the term; GPT-2 Small at 9-by-9 multiplication. Its GSM8K figure is for Mistral 7B, not GPT-2 |
| Yu et al., SpiralFormer | arXiv 2602.11698 | preprint | abstract | Looped transformers trained from scratch, 160M to 1.4B; the Pythia benchmarking claim is not in the abstract |

Suggested order for a first reading: the survey's sections 2.1 and 3.1 with Table 1, then the Huginn paper's pages 12 and 13 on orbits and path independence, then Soft Thinking's pages 2 and 5 on collapse and the early exit, then the rest as needed.

## 9. Candidate routes, unregistered

Everything in this section is speculation. No hypothesis number or experiment identifier is allocated here, and none of it starts without a ruling and a registered specification.

For this repository and the Stage 2 repository:

1. **A driven-loop arm.** Add the one control the surveyed architectures all keep and this loop drops: re-inject the prompt's own embedding on every pass, alongside the fed-back state, as Huginn does. Huginn's path independence predicts that a driven frozen loop has prompt-pinned attractors. If it does, "the basins belong to the weights" is a statement about the autonomous loop only, and the driven loop is the object the field actually studies. Cheap: one engine option and the existing prompt library. Registered on TC's ruling of 2026-09-03 as issue #140.
2. **Natural scale as the primary regime.** Every trained loop in the survey runs at the scale its block naturally receives. The ν-sweep shows the five basins exist only between about 50 and 300 times that scale, and that below the band the loop steps through other words or oscillates. A driven loop at natural scale, route 1, is the closest this apparatus can come to the field's object without training.
3. **The token-level twin.** Section 7. Value is relative to this loop, not absolute. Registered as issue #141.
4. **Stage 2's census stated in the survey's terms.** One sentence in the Stage 2 outline: the window census is a census of Prelude, Loop, Coda partitions of a frozen model, and it found no partition that behaves as a reasoning block. That belongs in the sibling repository's own record.

For the plasticity repository:

5. **The gradient-state family is the missing prior art.** The survey's horizontal branch includes test-time training, Titans and Atlas, where the hidden state is a set of fast weights updated by gradient steps at inference on a reconstruction objective. The plasticity repository's claim register (its C-42 row) already lists fast weights among the searches never run. These are those searches by name. Oja's rule is itself a gradient-like update, ascent on the variance of the activations passing through the site, so the plasticity loop is a gradient-state recurrence whose objective is the model's own activity rather than a supplied target. A symmetric positioning follows and is offered as interpretation only: this project is vertical recurrence with the input removed; the plasticity project is gradient-state recurrence with the target removed.
6. **Plasticity in a driven loop.** If route 1 produces prompt-pinned attractors, the plasticity question becomes whether an unconstrained local rule deepens or moves an input-pinned attractor, which is closer to what the trained-loop literature would call learning to reason about that input. Speculation stacked on speculation; recorded so it is not lost.

## 10. Decisions taken and open

Taken in this session, under TC's rulings of 2026-09-03:

- The taxonomy entry and the eleven added works are in `docs/PRIOR_WORK.md`, with the coverage caveat extended and the source URLs added.
- One positioning paragraph is in the README's "What This Is Now" section. This repository was chosen over the sibling repositories because it is the public piece, it holds the prior-work record, and it is where a reader arriving from the latent-reasoning literature lands. The Stage 2 outline and the plasticity README can each take one sentence later, through their own records.

Taken later the same day, on TC's ruling: both experiments are registered in this repository, the driven-loop arm as issue #140 and the token-level twin as issue #141. Execution waits on a run-number claim on the Identifier registry and, for #140, on the engine option landing in its own pull request first.

Open, for TC:

- Whether the plasticity repository's prior-art file takes the gradient-state family (route 5).

## 11. What this note does not do

It reports no new measurement. It allocates no identifier. It does not alter any finding, caveat or claim in `FINDINGS.md`. The statements about the sibling repositories are readings of their committed records and are suggestions for those records, not changes to them.
