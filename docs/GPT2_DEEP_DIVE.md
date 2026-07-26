# GPT-2: Development, Structure, Versions, and the Mechanistic Interpretability Record

*A deep dive on the model this repository experiments on. Part one is history: where GPT-2 came from and how it
was released. Part two is anatomy: the exact forward pass and the exact parameter counts. Part three is the
versions — four official sizes, and the several other things also called "GPT-2". Part four is the largest part
and the reason the document exists: the mechanistic interpretability (MI) literature that has accumulated on this
model, with a drafted summary for each paper.*

> **Provenance and conventions.** This document follows the citation discipline of
> [PRIOR_WORK.md](PRIOR_WORK.md). Every work discussed carries a **source-class tag** immediately after its
> citation — one of `[peer-reviewed]`, `[preprint, unreviewed]`, `[community post, unreviewed]`,
> `[published research report, not journal-reviewed]`, `[primary data, read directly]`, or `[status unverified]`.
> The tag records formal status as stated by the source's own venue or host; formal status is no guarantee of
> correctness, and where replication is known it is noted. An **asterisk (\*)** marks a claim that rests on
> secondary description (abstract, venue listing, press summary) rather than on primary material read directly —
> including venue attributions I could not confirm against the venue itself. Unmarked claims were checked against
> the paper's own text, its released artifacts, or the model files themselves.
>
> **Citation convention**, agreed with `agent:pythia-review` on the peer board so that this document and
> [PYTHIA_INTERPRETABILITY_REVIEW.md](PYTHIA_INTERPRETABILITY_REVIEW.md) do not disagree about the same work:
> **lead with the published venue and title, give the arXiv identifier, and note the preprint title where it
> differs.** Papers in this field routinely change title between preprint and proceedings, and neither record
> links to the other, so a repository carrying one of each looks like it is citing two papers. Tigges et al. in
> §5.4 is the worked example.
>
> **Corrections after merge** are marked in place rather than silently rewritten, naming what was wrong and who
> caught it. §5.4 and §7 item 3 carry one.
>
> Numbers in Part 2 and Part 3 were **recomputed from the architecture and cross-checked against the released
> checkpoints' `safetensors` headers and `config.json`/`vocab.json`/`merges.txt` files** on the Hugging Face
> `openai-community` and `distilbert` repositories — that is, `[primary data, read directly]`. Where they disagree
> with the GPT-2 paper's own table, both are given and the disagreement is flagged rather than explained away.
>
> Where this document connects the literature to **ATR** (Activation Tensor Resonance, this repository's method),
> that is **this project's reading**, marked as such. None of the cited work discusses ATR.
>
> **Companion document.** [PYTHIA_INTERPRETABILITY_REVIEW.md](PYTHIA_INTERPRETABILITY_REVIEW.md) does the same job
> for the Pythia suite, the other model family this repository runs. The two were written in parallel without
> contact and are complementary rather than overlapping: this document is the single-model archaeology side
> (circuits, heads, neurons, in one frozen checkpoint), that one is the developmental and cross-scale side
> (checkpoints, seeds, scaling ladders). Six papers are summarised in both; where the two disagree on a citation,
> the discussion is recorded on the peer board rather than silently resolved in either file. Read together they
> cover the whole 2×2 this project's cross-model controls are built on.
>
> **On abbreviations.** Every initialism is written out in full at its first use, and all of them are collected in
> [Appendix C](#appendix-c--glossary-of-abbreviations) at the end. This field abbreviates heavily and the habit
> makes its literature harder to enter than it needs to be; nothing here should require you to already know the
> vocabulary.

**Contents**

- [Part 1 — Development](#part-1--development-2017-2019)
- [Part 2 — Structure, exactly](#part-2--structure-exactly)
- [Part 3 — The versions](#part-3--the-versions)
- [Part 4 — Why GPT-2 became the field's model organism](#part-4--why-gpt-2-became-the-fields-model-organism)
- [Part 5 — The MI record, paper by paper](#part-5--the-mi-record-paper-by-paper)
- [Part 6 — What is and is not established](#part-6--what-is-and-is-not-established-about-gpt-2)
- [Part 7 — Bearings on this repository](#part-7--bearings-on-this-repository)
- [Appendix A — Instruments, one line each](#appendix-a--instruments-one-line-each)
- [Appendix B — Suggested reading order](#appendix-b--suggested-reading-order)
- [Appendix C — Glossary of abbreviations](#appendix-c--glossary-of-abbreviations)
- [Sources](#sources)

---

## Part 1 — Development (2017–2019)

### 1.1 The two-year run-up

GPT-2 is not an architectural invention. It is a scaling and a dataset decision applied to an architecture that
already existed, and its historical importance lies almost entirely in what that decision demonstrated.

The architecture arrived in **Vaswani et al., *Attention Is All You Need*** (NeurIPS 2017 — the Conference on
Neural Information Processing Systems, machine learning's largest venue) `[peer-reviewed]`
(https://arxiv.org/abs/1706.03762): multi-head scaled dot-product attention, residual connections, layer
normalisation, position encodings, as an encoder–decoder for translation.

**Radford, Narasimhan, Salimans, Sutskever, *Improving Language Understanding by Generative Pre-Training* (OpenAI,
2018)** `[published research report, not journal-reviewed]` — GPT-1, and the paper the abbreviation comes from:
**G**enerative **P**re-**T**raining — kept only the decoder, trained it autoregressively on BooksCorpus, then
fine-tuned per task. The recipe was pre-train once, fine-tune many times.

GPT-2's contribution was to delete the second half of that recipe. Its claim was that a large enough language
model trained on a diverse enough corpus performs downstream tasks **zero-shot**, with no gradient updates and no
task-specific head, because the tasks occur naturally in the corpus as text. The paper makes the argument
concrete with Table 1 — naturally occurring English↔French translation pairs found inside WebText — after having
explicitly *removed* Wikipedia from the corpus to avoid test-set contamination.

### 1.2 The paper

**Radford, Wu, Child, Luan, Amodei, Sutskever, *Language Models are Unsupervised Multitask Learners* (OpenAI,
February 2019)** `[published research report, not journal-reviewed]`
(https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Never
submitted to a venue; never peer-reviewed. This matters when reading its numbers.

**The corpus.** WebText was built to avoid Common Crawl's quality problems by using human curation as a filter:
OpenAI scraped **all outbound links from Reddit posts with at least 3 karma** — "a heuristic indicator for whether
other users found the link interesting, educational, or just funny" — yielding the text of **45 million links**.
Text was extracted with Dragnet and Newspaper. The version used for all results in the paper excludes links
created after **December 2017** and, after de-duplication and heuristic cleaning, contains **slightly over 8
million documents, 40 GB (gigabytes) of text**. Wikipedia was removed deliberately. WebText itself was never released, which
is a permanent handicap for interpretability: on GPT-2 you can study weights and activations but you cannot ask
what in the training data produced them. (OpenWebText, a community reconstruction of the recipe, is the usual
substitute\*.)

**The tokeniser.** Byte-level **BPE** — byte-pair encoding, the scheme that starts from individual characters (here
individual bytes) and repeatedly merges the most frequent adjacent pair into a new symbol, so common words end up
as one token and rare ones as several. The paper's reasoning is worth keeping in mind because its consequences show
up throughout the interpretability literature: character-level byte-pair encoding over Unicode would need a base vocabulary above 130,000, while
a byte-level base vocabulary needs only 256 and can assign a probability to *any* Unicode string. Greedy
frequency-based merging over raw bytes produces waste — the paper observed BPE learning `dog.`, `dog!`, `dog?` as
separate tokens — so merges are **prevented from crossing character categories, with an exception for spaces**.
That space exception is why GPT-2 tokens overwhelmingly carry their leading space (`Ġthe` = id 262, while bare
`the` = id 1169; verified in `vocab.json`), and why so much GPT-2 interpretability work is quietly about tokens
that begin with a space.

**The architecture changes from GPT-1**, quoted from §2.3 in full because the list is short and every item recurs
later:

1. Layer normalisation **moved to the input of each sub-block**, "similar to a pre-activation residual network" —
   i.e. *pre-normalisation* (normalise on the way in) rather than post-normalisation (normalise on the way out).
   Written **pre-LN** and **post-LN** below, LN being the standard abbreviation for layer normalisation: the
   operation that rescales a vector to zero mean and unit variance, then applies a learned scale and offset.
2. **An additional layer normalisation after the final self-attention block** (`ln_f`).
3. A modified initialisation accounting for accumulation on the residual path with depth: **residual-layer weights
   scaled by 1/√N at initialisation**, N = number of residual layers.
4. **Vocabulary expanded to 50,257.**
5. **Context increased from 512 to 1024** tokens.
6. **Batch size 512.**

Item 1 is the single most consequential line in the paper for interpretability. Pre-LN makes the residual stream
a clean additive accumulator that every block reads from and writes to without an intervening normalisation —
which is exactly the object Anthropic's circuits framework later formalised, and exactly the object ATR
reinjects.

**The results.** Four models were trained, "approximately log-uniformly spaced" in size (§3). The 1.5B model set
state of the art on **7 of 8** language-modelling datasets zero-shot; smaller models already beat the previous best
published result on several. Even the smallest model (the row the paper labels 117M) reports perplexity 35.13 on
LAMBADA (a long-range word-prediction benchmark) against a previous best of 99.8, and 87.65% accuracy on the
common-noun split of the Children's Book Test (CBT-CN) against 85.7%. Elsewhere zero-shot transfer was much
weaker: **55 F1** on CoQA — the Conversational Question Answering dataset; F1 being the harmonic mean of precision
and recall, so a single number balancing the two — against **89 F1** for a supervised system built on BERT
(Bidirectional Encoder Representations from Transformers, Google's 2018 encoder model); 70.70% on the Winograd
Schema Challenge; and a summarisation score (prompting with the literal string `TL;DR:`, internet shorthand for
"too long; didn't read", to elicit a summary; scored by ROUGE, the standard word-overlap metric for
summarisation, averaged at 21.40) that the paper's own Table 4 shows losing to a `Random-3` sentence
baseline at 20.98 by a hair and to `Lede-3` at 31.55 outright. The paper reports the failures alongside the
successes; the field's memory has been kinder to it than the paper was to itself.

Two admissions in §3 deserve to be read together: the learning rate for each model was **manually tuned** on a 5%
held-out sample of WebText, and **"all models still underfit WebText"** with held-out perplexity still improving
given more training time. Every GPT-2 checkpoint studied by every paper in Part 5 is therefore an undertrained
model, stopped for reasons never stated. No token count, no compute budget, no learning-rate schedule, no seed,
and no intermediate checkpoints were published. This is the structural reason the Pythia suite exists — a claim
the companion review makes from the other side, listing the suites of the day and why each was unusable for
developmental work ([PYTHIA_INTERPRETABILITY_REVIEW.md](PYTHIA_INTERPRETABILITY_REVIEW.md) §I.1) — and the
reason **OpenAI's** GPT-2 can support circuit analysis but not developmental analysis.

That last limitation is a fact about OpenAI's release, not about the architecture, and §3.2 gives the way round
it: the Stanford replications of GPT-2 Small ship roughly 609 public checkpoints each across five seeds. Anything
in this document that says the developmental question cannot be asked of GPT-2 means it cannot be asked of *these
weights*.

### 1.3 The staged release, and what it was for

GPT-2's release is a piece of artificial-intelligence governance history independent of the model.

| Date | Release | Accompanying material |
|:---|:---|:---|
| 14 Feb 2019 | Smallest model (**124M**) | *Better Language Models and Their Implications*; larger models withheld |
| May 2019 | **355M** | 6-month follow-up planning; output dataset + detection baseline\* |
| 20 Aug 2019 | **774M** | 6-month follow-up report; *Release Strategies and the Social Impacts of Language Models* |
| 5 Nov 2019 | **1558M** + code | Final report; full model and code released |

The stated purpose of staging was to "give people time to assess the properties of these models, discuss their
societal implications, and evaluate the impacts of release after each stage"\*, with the withheld-model concern
being synthetic disinformation, impersonation, and automated abuse. Supporting measures included partnered
research (Cornell, the Middlebury Institute's Center on Terrorism, Extremism, and Counterterrorism, the University
of Oregon, and the University of Texas at Austin\*), released datasets of model outputs so that others could build detectors, and bias analysis. The
argument and its aftermath are documented in **Solaiman et al., *Release Strategies and the Social Impacts of
Language Models* (arXiv:1908.09203, Aug/Nov 2019)** `[preprint, unreviewed]`.

Two things about this are worth stating plainly. First, the staged release established the template that every
subsequent frontier lab release-decision has argued with or against. Second, the specific fear did not
materialise in the predicted form, and the eventual full release is now so uncontroversial that GPT-2 is the
default teaching model — which is itself the reason the entire literature in Part 5 could be written. The
interpretability field's favourite model organism exists in open form because a release-caution argument was
tested and resolved.

### 1.4 What GPT-2 licensed

GPT-2's own headline finding — that zero-shot task performance improves **log-linearly** with capacity across
tasks — is the claim **Brown et al., *Language Models are Few-Shot Learners* (GPT-3, arXiv:2005.14165, 2020)**
`[peer-reviewed]` (NeurIPS 2020)\* took to 175B, holding the architecture almost fixed. GPT-2 is thus the last
point in the lineage at which the entire model, its weights, and a full description of its structure are small
enough to hold in one head — which is why interpretability stayed on it long after capability research left.

---

## Part 2 — Structure, exactly

### 2.1 The forward pass

For token sequence *t*₁…*t*ₙ, with `wte` the token-embedding matrix (50257 × d), `wpe` the learned
absolute position-embedding matrix (1024 × d):

```
x₀      = wte[t] + wpe[0..n-1]                                  # residual stream, shape (n, d)

for each block ℓ = 0 … L-1:
    x    = x + Attn( LN₁(x) )                                   # pre-LN attention sub-block
    x    = x + MLP(  LN₂(x) )                                   # pre-LN feed-forward sub-block

logits  = LN_f(x) · wteᵀ                                        # tied unembedding
```

Two names in that block: **LN** is layer normalisation (above), and **MLP** stands for *multi-layer perceptron* —
the field's inherited name for the two-layer feed-forward network sitting in each block. It is also called the
**FFN** (feed-forward network); the two terms mean the same thing and both appear in the literature below.

with, per block:

- **Attention.** `c_attn` projects LN₁(x) to Q, K, V — query, key and value, the three vectors attention is
  computed from — at once (one weight of shape d × 3d); heads are
  d/64 in number, each of head dimension **64 in every GPT-2 size**; causal-masked scaled dot-product attention;
  `c_proj` maps back to d. The checkpoints ship the causal mask as a **non-trainable `attn.bias` buffer of shape
  (1, 1, 1024, 1024) per layer** — 1,048,576 values a layer that are not parameters, a detail that trips up naive
  parameter counting (see §2.2).
- **MLP.** `c_fc` to width **4d**, activation **GELU** — the Gaussian Error Linear Unit, a smooth curve that
  behaves roughly like the older ReLU (rectified linear unit: pass positives, zero negatives) without the sharp
  kink (`gelu_new`, the tanh approximation, per `config.json`) — then `c_proj` back to d. No gating, no
  SwiGLU (a gated activation variant common in later models).
- **LayerNorm** with learned γ and β, ε = 1e-5 (per `config.json`).
- **Weight tying.** The released checkpoints contain **no `lm_head` tensor** (verified in the `safetensors`
  header): the unembedding is `wte` transposed.

Everything is `Conv1D` in OpenAI's original code and Hugging Face's port, meaning the weight matrices are stored
**transposed** relative to `nn.Linear` (`c_attn.weight` is [768, 2304], not [2304, 768]). Purely an
implementation fact, and a routine source of sign and orientation bugs in interpretability code.

### 2.2 Parameter counts, computed and verified

Per-block parameters are 12d² + 13d (attention: 3d² + 3d for `c_attn`, d² + d for `c_proj`; MLP: 4d² + 4d for
`c_fc`, 4d² + d for `c_proj`; two LayerNorms: 4d). Total is therefore:

```
params = 50257·d  +  1024·d  +  L·(12d² + 13d)  +  2d
         └ wte ┘    └ wpe ┘    └── blocks ──┘    └ ln_f ┘
```

| Checkpoint | L | d_model | heads | d_head | d_mlp | **Total params** | Non-embedding | Paper's Table 2 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt2` (Small) | 12 | 768 | 12 | 64 | 3072 | **124,439,808** | 85,056,000 | 117M |
| `gpt2-medium` | 24 | 1024 | 16 | 64 | 4096 | **354,823,168** | 302,311,424 | 345M |
| `gpt2-large` | 36 | 1280 | 20 | 64 | 5120 | **774,030,080** | 708,390,400 | 762M |
| `gpt2-xl` | 48 | 1600 | 25 | 64 | 6400 | **1,557,611,200** | 1,475,561,600 | 1542M |
| `distilgpt2` | 6 | 768 | 12 | 64 | 3072 | **81,912,576** | 42,528,768 | — |

Verification: the `gpt2` `safetensors` header sums to 137,022,720 values across 160 tensors; subtracting the 12
causal-mask buffers (12 × 1,048,576 = 12,582,912) gives **124,439,808**, exactly the formula's value and exactly
the checkpoint's popular name. The same subtraction on `distilgpt2` (88,204,032 − 6 × 1,048,576) gives
**81,912,576** ≈ 82M. All four `config.json` files confirm L, d_model, heads, n_ctx = 1024, vocab 50257,
`gelu_new`, ε = 1e-5.

**On the discrepancy with the paper.** The paper's Table 2 gives 117M / 345M / 762M / 1542M; the released
checkpoints are universally referred to as 124M / 355M / 774M / 1558M, and the second set is what the weights
actually contain. The two sets differ by 7.4M / 9.8M / 12.0M / 15.6M — not the position-embedding count, not the
final LayerNorm, and not any single component I can identify, so I record the discrepancy without explaining it.
Practical consequence: **"GPT-2 117M" and "GPT-2 124M" are the same checkpoint**, and papers using either name
are studying the same weights. A reader collating results across the MI literature will meet both.

### 2.3 The tokeniser, verified

- `vocab.json` contains exactly **50,257** entries. `merges.txt` contains **50,001** lines, of which the first is
  the `#version: 0.2` header — so **50,000 merges**. 256 byte-level base tokens + 50,000 merges + 1 special token
  = 50,257. The arithmetic closes exactly.
- Ids 0–255 are the byte alphabet (id 0 = `!`, in the printable-remap encoding).
- **Id 50256 = `<|endoftext|>`**, the only special token. There is no separate beginning-of-sequence (BOS),
  padding (PAD), or unknown-token (UNK) entry — abbreviations you will meet in the attention-sink literature
  below, where whether a model has a dedicated first token turns out to matter. The Hugging Face tokeniser
  prepends nothing, so at the tokeniser level **position 0 of a GPT-2 sequence is an ordinary content token**.

  **That is a fact about the tokeniser, and it does not survive contact with the tooling.** `[primary data, read
  directly]` — TransformerLens, the library this repository and most of the work in Part 5 actually run on,
  tokenises string input through `to_tokens`, which prepends the beginning-of-sequence token whenever
  `cfg.default_prepend_bos` is set. In `loading_from_pretrained.py`, `GPTNeoXForCausalLM` (the Pythia family)
  carries an explicit `"default_prepend_bos": False`; **GPT-2 has no such override and falls through to the global
  default of `True`.** So any GPT-2 experiment that hands a *string* to `run_with_cache`, `to_tokens` or `forward`
  is running with `<|endoftext|>` at position 0 — a dedicated sink token — even though the tokeniser alone would
  not have put one there. §5.4 and §7 depend on this distinction; the measured confirmation, and the cross-model
  consequence, are recorded there.
- The famous unspeakable tokens are in the vocabulary and can be checked directly: `ĠSolidGoldMagikarp` = id
  **43453**, `Ġpetertodd` = id **37444** (verified in `vocab.json`). These exist because the tokeniser's corpus
  and the model's training corpus were not the same corpus — see §5.4.

### 2.4 What GPT-2 does *not* have

Reading GPT-2 results forward into modern models requires knowing what has since changed. Each item below names
the modern replacement GPT-2 does *not* use, with a gloss, since these are the abbreviations that make recent
architecture papers unreadable to newcomers. GPT-2 has:

- **Learned absolute position embeddings** — one trained vector per slot, added to the token embedding. Not
  **RoPE** (rotary position embedding: positions applied by rotating query and key vectors), not **ALiBi**
  (Attention with Linear Biases: a distance penalty added to attention scores), not **NoPE** (no positional
  encoding at all, relying on the causal mask to imply order).
- **LayerNorm** with a learned bias — not **RMSNorm** (root-mean-square normalisation, which drops the
  mean-centring step and the bias).
- **Plain GELU feed-forward blocks** — not **SwiGLU** or **GeGLU** (gated variants that multiply two projections
  together, one of them passed through an activation).
- **Full multi-head attention** — every head with its own keys and values. Not **MQA** (multi-query attention: all
  heads share one key/value pair) or **GQA** (grouped-query attention: heads share in groups), both of which trade
  a little quality for much smaller memory during generation.
- **No QK-norm** — no normalisation applied to the query and key vectors before their dot product.
- **Tied embeddings**, a **1024-token context**, and **no instruction tuning, no RLHF** (reinforcement learning
  from human feedback: the post-training step that turns a text predictor into an assistant) and **no chat
  template**. GPT-2 is a pure base model, which is why every alignment-relevant interpretability result (refusal,
  sycophancy, deception features) had to be found elsewhere.
 Its dense 4d MLP and clean pre-LN residual stream are, however, still the modal transformer block, so
circuit-level results about *how information moves through a residual stream* travel better than results about
positions or normalisation.

For this repository specifically, three of these matter: absolute learned positions (ATR's position-collapse
observation is about a model whose position information is an additive vector, not a rotation), tied embeddings
(the readout and the input basis are literally the same matrix, so a converged tensor's decoded token and its
distance to embedding-space structure are not independent measurements), and the absence of any post-training
(ATR's basins are properties of a base model's raw next-token map).

**The tying point does not travel across the 2×2, and that is a trap worth naming explicitly.** In GPT-2 the
checkpoint carries `wte.weight` and no `lm_head` at all, so embedding space and unembedding space are the same
space *by identity* — a claim about `W_U` is a claim about `W_E`. Pythia unties them deliberately, EleutherAI
citing interpretability as the reason, so there the two are genuinely distinct and the identity is unavailable.
**Any instrument or claim ported from one arm to the other that assumed a single space must be re-derived, not
re-run.** This is the same distinction that produced a documented `W_U`/`W_E` disagreement between agents in the
sibling repository, and it is cheap to state and expensive to rediscover.

---

## Part 3 — The versions

### 3.1 The four official sizes

The four sizes are a pure depth-and-width sweep with **d_head held at 64 throughout**. Layers scale 12 → 24 → 36
→ 48; d_model 768 → 1024 → 1280 → 1600; heads 12 → 16 → 20 → 25. Everything else — tokeniser, context length,
training corpus, activation, normalisation, tying — is identical. The models are as close to a controlled scaling
series as anything from that era, with one caveat that undercuts the control: per-model learning rates were
**manually tuned** (§1.2), so size is not the only thing varying.

What the sizes buy, on the paper's own numbers, is monotone-but-uneven improvement: language-modelling perplexity
improves across the board with scale, Winograd improves from roughly chance-adjacent to 70.70%, and the paper
frames the overall pattern as log-linear in capacity. What the sizes do *not* buy is any qualitative change of
kind — no size of GPT-2 follows instructions, and none is chat-shaped.

**For interpretability the practical differences are:**

- **GPT-2 Small (124M)** is the workhorse. 12 layers × 12 heads = **144 attention heads** and 12 MLPs, i.e. a
  computational graph small enough that exhaustive per-component patching is cheap — Hanna et al. patch all 144
  heads and all 12 MLPs as a matter of course. Almost every circuit result in Part 5 is a Small result.
- **GPT-2 Medium (355M)** is where "does the circuit reproduce at scale?" is usually asked first, and the answer
  is often *yes, with rearrangement* (Merullo et al., §5.3).
- **GPT-2 Large (774M)** is the least-studied size, with no canonical MI result attached to it that I could find.
- **GPT-2 XL (1558M)** — "XL" for extra large, OpenAI's label for the biggest of the four — is the factual-recall
  and automated-interpretability model: 48 layers × 6400 feed-forward neurons =
  **307,200 neurons**, the exact figure OpenAI's neuron-explainer covered (§5.6), and the model ROME edits
  (§5.3).

The sizes are not interchangeable, and this repository has its own evidence for that: GPT-2 Small resolves 125
language prompts into five semantic attractor basins under ATR while **GPT-2 Medium — same corpus, same
tokeniser, same architecture family — collapses every prompt to the single token `D`** ([FINDINGS.md](FINDINGS.md)).
Whatever GPT-2 Small is doing, it is not simply "what GPT-2 does, smaller".

**A warning about the other axis, though.** That is a *within-GPT-2* size difference and reads correctly as one.
The cross-family comparisons in this repository do **not**: `pythia-160m` is 12 layers × d_model 768 × 12 heads,
which is GPT-2 Small exactly, and `pythia-410m` is 24 × 1024 × 16, which is GPT-2 Medium exactly. The match is
tighter than shape — because GPT-NeoX blocks carry the same 12d² + 13d parameter inventory as GPT-2 blocks (§2.2),
the **non-embedding parameter counts are identical**: 85,056,000 and 302,311,424 down each column. Parallel versus
sequential sub-layer arrangement changes what can read what, not how many weights there are. So a GPT-2-versus-
Pythia difference is **never** a capacity or scale effect; it is attributable to training corpus, tokeniser,
positional scheme (learned absolute versus rotary), sub-layer arrangement, embedding tying — and, as §5.4 records,
to whether the tooling puts a sink at position 0. A reader arriving from either end of the 2×2 reaches for "bigger
model, different behaviour," and that reading is unavailable here. (Raised by `agent:pythia-review` on the peer
board; stated in [PYTHIA_INTERPRETABILITY_REVIEW.md](PYTHIA_INTERPRETABILITY_REVIEW.md) §I.4, and it belongs on
this side too.)

### 3.2 The other GPT-2s

Several distinct artifacts are called GPT-2 in the literature, and conflating them produces irreproducible
results.

- **DistilGPT2** (82M, 6 layers, d=768) — distilled from GPT-2 Small under the Hinton–Vinyals–Dean distillation
  framing (**Hinton et al., arXiv:1503.02531** `[preprint, unreviewed]`) by the Hugging Face DistilBERT
  authors\*. Same tokeniser, half the depth, verified above at 81,912,576 parameters. Not a GPT-2 checkpoint: a
  student of one.
- **The Stanford CRFM "Mistral" GPT-2 replications** — CRFM being Stanford's Center for Research on Foundation
  Models (Karamcheti et al., 2021) — **five GPT-2 Small and five
  GPT-2 Medium models trained from different random seeds**. Confirmed by reading Gurnee et al.'s methods section
  (§5.5), which studies `GPT2-{small,medium}-[a-e]` and names `stanford-crfm/arwen-gpt2-medium-x21` and
  `stanford-crfm/alias-gpt2-small-x21`. These are the *only* way to ask a universality question about GPT-2 — "is
  this head/neuron a property of the architecture-plus-data or of this particular initialisation?" — and their
  existence is why §5.5's result is possible at all. (Note the name collision: nothing to do with Mistral AI.)

  **They also carry intermediate checkpoints, and this is under-known.** `[primary data, read directly]` — the
  Hugging Face refs endpoint for each repository, queried directly. The five **Small** seeds
  (`alias-x21`, `battlestar-x49`, `caprica-x81`, `darkmatter-x343`, `expanse-x777`) each expose **604–609
  checkpoints as git tags** named `checkpoint-{step}`, running from step 0 to step 400,000, on a schedule that is
  dense exactly where it needs to be: every **10** steps to step 100, every **50** to step 1,000, every **100**
  thereafter, then every **1,000** out to 400,000. Load one with
  `from_pretrained("stanford-crfm/alias-gpt2-small-x21", revision="checkpoint-400")`.

  The five **Medium** seeds (`arwen-x21`, `beren-x49`, `celebrimbor-x81`, `durin-x343`, `eowyn-x777`) expose
  **zero** checkpoint tags — final weights only. That asymmetry is worth stating plainly because of where it
  falls: the developmental question can be asked of GPT-2 Small, which is the model that exhibits this
  repository's five-basin result, and *not* of GPT-2 Medium, which is its contrast case (§7).

  For comparison, Pythia offers 154 checkpoints per model. On raw count and on early-window resolution the
  Stanford GPT-2 Small series is finer — 10-step granularity through the first 100 steps against Pythia's
  log-spaced 1, 2, 4, … 512. What it lacks is Pythia's reconstructible dataloader, so "what was in the data before
  step *k*" stays unanswerable here.
- **OpenWebText-trained reproductions** — community re-creations of the WebText recipe, including Karpathy's
  nanoGPT/minGPT lineage\*. Architecturally GPT-2, different weights, different data sample. Most published
  **sparse autoencoders** (SAEs — the feature-extraction tool explained in §5.5) for GPT-2 Small are trained on
  OpenWebText activations (§5.7), which means the *dictionaries* and the *model* saw different corpora.
- **The GPT-2 output datasets and detector** — released alongside the staged rollout for detection research; the
  detector was a RoBERTa fine-tune\*. Not a language model, but part of the artifact family.
- **Task fine-tunes** — DialoGPT, countless domain models. Different weights; MI results on base GPT-2 do not
  automatically transfer.

The practical rule: an MI paper's result belongs to a *checkpoint*, not to "GPT-2". Papers are generally careful
about Small-vs-XL and generally silent about OpenAI-vs-CRFM.

---

## Part 4 — Why GPT-2 became the field's model organism

Five properties, in rough order of importance:

1. **Fully open weights of a genuinely pretrained model.** After November 2019 all four sizes are downloadable.
   The alternative in 2020–2022 was a toy model you trained yourself (which might not exhibit the phenomenon) or
   a model reachable only through an API (application programming interface — a remote service you send text to
   and get text back from), whose activations you could not see.
2. **Small enough for exhaustive methods.** 144 heads and 12 MLPs in Small means you can patch *everything* and
   report a heatmap rather than a sample.
3. **Big enough to have real behaviours.** Unlike a 2-layer attention-only toy, GPT-2 Small does **indirect object
   identification** (IOI, the abbreviation used throughout below: working out that "John gave the bottle to ___"
   should complete with the *other* name mentioned), numeric comparison, ordinal succession, and sentiment —
   natural behaviours "in the wild", to borrow the IOI paper's phrase.
4. **Tooling gravity.** TransformerLens ships GPT-2 as its canonical model with weights pre-processed for
   analysis (LayerNorm folding, centring); Neuronpedia hosts GPT-2 Small features; SAELens ships GPT-2 Small SAE
   suites. Each new tool defaulting to GPT-2 makes the next paper's cheapest choice GPT-2 again.
5. **Accumulated ground truth.** Once IOI existed, every new automated circuit-discovery method could be scored
   against it. GPT-2 Small became the field's benchmark by being the field's first well-documented case.

**The cost of the monoculture** should be stated as clearly as the benefit. A large fraction of what the field
"knows" about transformers is known about one 124M-parameter, undertrained, English-heavy base model with
absolute position embeddings and no post-training. Results that replicate on GPT-2 Small and nothing else are
findings about GPT-2 Small. The field's own forward-looking review (§5.8) names generalisation across models as an
open problem, and the papers that deliberately cross model families — Successor Heads across GPT-2/Pythia/Llama,
Universal Neurons across seeds, Circuit Component Reuse across sizes — are valuable precisely in proportion to
how unusual that is. This repository's own cross-model result is a data point on the same side of the ledger:
the ATR regime that GPT-2 Small exhibits does *not* appear in GPT-2 Medium or either Pythia
([FINDINGS.md](FINDINGS.md)).

---

## Part 5 — The MI record, paper by paper

Grouped by what each contributes. Within groups, roughly chronological. Every entry states what was done, on
which model, the numbers that matter, and what it does not establish.

### 5.1 Foundations: the frameworks the field reasons in (Anthropic)

Anthropic's Transformer Circuits Thread is where the vocabulary used by nearly every other paper in this section
was defined. All Thread publications are `[published research report, not journal-reviewed]` — self-published,
unreviewed, and unusually influential; some later appeared on arXiv.

---

**Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, DasSarma, Drain, Ganguli, Hatfield-Dodds,
Hernandez, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, Olah — *A Mathematical
Framework for Transformer Circuits* (Transformer Circuits Thread, December 2021).**
`[published research report, not journal-reviewed]` https://transformer-circuits.pub/2021/framework/index.html

The field's grammar. Working deliberately in **toy transformers of two layers or fewer, attention-only** (no MLPs),
it establishes four ideas that everything downstream assumes:

- **The residual stream as a communication channel.** Components do not connect to each other directly; they
  "communicate by reading and writing to different subspaces of the residual stream." This is what makes
  linear-algebraic decomposition of a forward pass legitimate, and it is a direct consequence of GPT-2-style
  pre-LN placement (§2.1).
- **QK and OV circuits.** An attention head factors into two independent computations — where to attend (the
  **QK circuit**, from *query* and *key*) and what to write once attention is decided (the **OV circuit**, from
  *output* and *value*). The OV circuit is a fixed d→d linear map, independent of the attention pattern. These two
  abbreviations are used constantly in the papers below and are worth committing to memory: QK decides *where a
  head looks*, OV decides *what it says*.
- **Path expansion.** An attention-only model can be rewritten as a sum of interpretable end-to-end
  token→logit functions, one per path through the model.
- **Virtual attention heads.** Composition of heads across layers behaves like additional heads that exist in no
  single layer.

Results by depth: zero-layer models learn bigram statistics readable directly from the weights; one-layer models
are an ensemble of bigram and **skip-trigram** (`A…B → C`) models; two-layer models can do something
qualitatively new — **induction heads**, formed by K-composition between layers.

The paper is explicit about its central limitation: "we've simply had much less success in understanding MLP
layers so far." Everything in §5.7 is the field's eventual answer to that sentence. Nothing here is a GPT-2
result — the value is that it made GPT-2 results expressible.

*Relevance to ATR:* the OV-circuit formalism is the instrument this repository used to attribute its period-2
flip to a single head's fixed 768→768 output transform (finding 4, [PRIOR_WORK.md](PRIOR_WORK.md)). ATR's
attribution is written in this paper's notation.

---

**Olsson, Elhage, Nanda, Joseph, DasSarma, Henighan, Mann, Askell, Bai, Chen, Conerly, Drain, Ganguli,
Hatfield-Dodds, Hernandez, Johnston, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish,
Olah — *In-Context Learning and Induction Heads* (Transformer Circuits Thread, March 2022).**
`[published research report, not journal-reviewed]`
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

The field's most-cited causal story about an emergent capability, and a template for how to argue one.

An **induction head** is defined operationally by two behaviours on repeated random token sequences: **prefix
matching** (attending back to tokens previously followed by the current token) and **copying** (increasing the
logit of the attended-to token). Together: *if `AB` appeared earlier and we are now at `A`, predict `B`.*

The paper's claim is that induction heads are the main mechanism of in-context learning in small models, argued
along **six independent lines**: (1) induction heads form during a **phase change** early in training that
coincides with a sharp improvement in in-context learning; (2) architectural changes that shift *when* induction
heads can form shift the capability improvement correspondingly; (3) directly ablating induction heads at test
time sharply reduces in-context learning; (4) induction heads implement more abstract behaviours than literal
copying, including translation and pattern completion; (5) the mechanism is understood well enough in small
models to be plausible; (6) the relevant quantities are continuous from small to large models. The in-context
learning score is operationalised concretely: **loss on the 500th token of a context minus mean loss on the 50th
token**. The study spans **34 transformers** and more than **50,000 attention-head ablations**.

Its own stated limitation is unusually candid: the evidence "is stronger for small models than for large ones,"
and "a large number of subtle confounds or alternative hypotheses are possible."

*Relevance to GPT-2:* GPT-2 Small has induction heads, and they appear as a named class inside the IOI circuit
(§5.3). *Relevance to ATR:* this is the paper that made "a head has a job you can name" a respectable claim, which
is the form of finding 4.

---

**Elhage, Hume, Olsson, Schiefer, Henighan, Kravec, Hatfield-Dodds, Lasenby, Drain, Chen, Grosse, McCandlish,
Kaplan, Amodei, Wattenberg, Olah — *Toy Models of Superposition* (Transformer Circuits Thread, September 2022).**
`[published research report, not journal-reviewed]` https://transformer-circuits.pub/2022/toy_model/index.html

The theoretical justification for everything in §5.7. In small models built from ReLU units (rectified linear
units — pass positive values, zero out negative ones) with a controllable number of
synthetic features of controllable **sparsity** (how often a feature is non-zero) and **importance** (its weight
in the loss), the paper shows networks representing **more features than they have dimensions** by placing them
in non-orthogonal directions — **superposition** — and tolerating the resulting interference because sparse
features rarely collide. Polysemantic neurons are then a *symptom*, not a primitive.

Three results the field took forward: superposition is real and controllable in toy settings; the transition
between one-feature-per-neuron and superposed regimes is a **phase change** in the sparsity/importance plane, not
a gradual slide; and superposed features arrange themselves in **specific geometric configurations** — antipodal
pairs and regular polytopes — with a "dimensions per feature" quantity that plateaus at discrete values. The
framing is explicitly connected to compressed sensing and the Johnson–Lindenstrauss lemma: sparse
high-dimensional structure survives projection into far fewer dimensions.

Its limitation is the one it names: these are toy models with known ground-truth features. Whether real language
models are in superposition was, at publication, an inference rather than a measurement. §5.7 is the field
attempting the measurement.

*Relevance to ATR:* the reason ATR's five basins cannot be read as "five neurons" or "five clean directions"
without argument. If GPT-2 Small's representations are superposed, a converged tensor's decoded token is a
projection of something with no reason to be axis-aligned.

---

**Elhage, Hume, Olsson, Nanda, Henighan, Johnston, ElShowk, Joseph, DasSarma, Mann, Hernandez, Askell, Ndousse,
Jones, Drain, Chen, Bai, Ganguli, Lovitt, Hatfield-Dodds, Kernion, Conerly, Kravec, Fort, Kadavath, Jacobson,
Tran-Johnson, Kaplan, Clark, Brown, McCandlish, Amodei, Olah — *Softmax Linear Units* (Transformer Circuits
Thread, June 2022).** `[published research report, not journal-reviewed]`\*
https://transformer-circuits.pub/2022/solu/index.html

Replaces the MLP activation with a softmax-based unit (SoLU) that empirically increases the fraction of neurons
with clean single interpretations, at no loss in performance. Included here for two reasons: it is the field's
first serious attempt at *architecting for interpretability* rather than analysing what exists — a line that
resurfaces in 2025 with weight-sparse transformers (§5.8) — and it is a warning, since the paper itself notes the
interpretability gain may partly reflect features being pushed elsewhere rather than eliminated\*. GPT-2 has
ordinary GELU MLPs, so this describes a road not taken for the model in question.

---

**Anthropic Interpretability Team — *Privileged Bases in the Transformer Residual Stream* (Transformer Circuits
Thread, March 2023).** `[published research report, not journal-reviewed]`
https://transformer-circuits.pub/2023/privileged-basis/index.html

Small, sharp, and directly relevant to anyone measuring residual-stream geometry. Theory says the residual-stream
basis should be arbitrary — no coordinate should be special — yet real transformers show extreme **outlier
dimensions**. Studying a **200M-parameter model** on Anthropic's codebase and counting activations with absolute
value above 6, the paper tests three hypotheses and provisionally concludes that **the per-dimension normalisers
in the Adam optimiser are to blame**, having ruled out LayerNorm and floating-point precision. It connects to
Dettmers' observations of the same phenomenon in larger models but keeps the study at a tractable scale, and its
own hedge ("provisionally") is in the title's spirit.

*Relevance to ATR:* directly load-bearing. Any claim of the form "the converged tensor has structure along
coordinate *i*" must survive the possibility that coordinate *i* is privileged by the optimiser rather than by the
computation. This paper is the reason such a claim needs a control.

---

### 5.2 Instruments: reading a residual stream

**nostalgebraist — *interpreting GPT: the logit lens* (LessWrong, August 2020).**
`[community post, unreviewed]`
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

An unreviewed blog post that became standard equipment. The idea: since final logits are a linear function
(`LN_f` then `wteᵀ`) of the final residual stream, apply that same map to *intermediate* residual streams and read
the resulting distribution over the vocabulary. You get a per-layer, per-position picture of what the model
"currently predicts".

Run on **GPT-2 XL** (48 layers, d=1600), the striking observation is how fast input identity disappears:
"immediately, after the very first layer, the input has been transformed into something that looks more like the
final output." Predictions then converge through middle layers, typically reaching something reasonable before
the last layer; the trajectory runs from uninterpretable, through shallow guesses, to better guesses. Rare tokens
needed later (the post's `plasma` example) are somehow preserved despite the early scramble. Extreme repetition is
"noticed in the upper half of the network, while the lower half can't see it." The author is explicit that this is
*one* extraction method and that much information lies outside it.

*Relevance to ATR:* ATR's readout — `ln_final → W_U` applied to the converged tensor — **is a logit lens at the
final layer**. Every basin label this repository reports is a logit-lens reading, with that instrument's known
blind spots. Finding 6 (the period-2 motion is near-invisible to lens instruments) is a statement about this
instrument's limits, and the J-space paper (§5.8) gives the exact relationship: the logit lens is the J-lens with
the Jacobian set to the identity.

---

**Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman, Steinhardt — *Eliciting Latent Predictions from
Transformers with the Tuned Lens* (arXiv:2303.08112, March 2023).** `[preprint, unreviewed]`

The logit lens is brittle: on some models and layers it produces nonsense, because intermediate residual streams
are not in the same basis the unembedding expects. The fix is to **train an affine probe per layer** on a frozen
model, mapping each hidden state into the final-layer basis before unembedding. Tested on autoregressive models
up to **20B** parameters, the tuned lens is "more predictive, reliable and unbiased" than the logit lens; causal
experiments indicate it uses features the model itself uses; and the *trajectory* of latent predictions across
layers detects anomalous inputs with high accuracy.

The framing — transformers as **iterative inference**, each block refining a prediction — is the conceptual
frame this repository's iteration operates inside, and the paper's own citation of Jastrzębski et al.'s
"residual connections encourage iterative inference" points to the older literature on the same idea.

*Relevance to ATR:* the obvious upgrade to ATR's readout, and one that would test finding 6 directly. If the
period-2 flip is invisible to the logit lens but visible to a tuned lens, the flip lives in directions the final
unembedding ignores but earlier computation does not.

---

**Goldowsky-Dill, MacLeod, Sato, Arora — *Localizing Model Behavior with Path Patching* (arXiv:2304.05969, April
2023).** `[preprint, unreviewed]`

Formalises the intervention almost every circuit paper in §5.3 uses. Ordinary activation patching replaces a
component's activation and measures the effect on the output, which conflates all downstream routes. **Path
patching** restricts the intervention to a *specified path* through the computational graph, letting you test
hypotheses of the form "component X affects the output *via* component Y and not otherwise." The paper refines the
induction-head account, characterises a GPT-2 behaviour, and releases a framework. It is short, methodological,
and is the reason later circuit diagrams can claim edges rather than just nodes.

---

**Zhang, Nanda — *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*
(arXiv:2309.16042, September 2023).** `[peer-reviewed]` (ICLR 2024 — the International Conference on Learning
Representations, with NeurIPS one of the two main machine learning venues)\*

The field auditing its main instrument. Systematically varies the methodological choices in activation
patching — evaluation metric, corruption method, patching direction — and shows that **varying these
hyperparameters can produce disparate interpretability conclusions on the same model and task**. It then argues
conceptually for particular choices and issues recommendations. Companion to **Heimersheim & Nanda, *How to use
and interpret activation patching* (arXiv:2404.15255, April 2024)** `[preprint, unreviewed]`, a practitioner
guide on what patching evidence does and does not license.

*Relevance to ATR:* the general lesson — that an instrument's hyperparameters can manufacture a finding — is the
same lesson this repository learned from its lag-1 convergence gate missing an exact period-2 cycle (finding 3).
The specific recommendations apply directly to any future patching work on ATR's attractors.

---

### 5.3 Circuits found in GPT-2

**Wang, Variengien, Conmy, Shlegeris, Steinhardt — *Interpretability in the Wild: a Circuit for Indirect Object
Identification in GPT-2 small* (arXiv:2211.00593, November 2022).** `[peer-reviewed]` (ICLR 2023)\*

The founding case study of circuit analysis in a real pretrained model, and still the field's reference point.

**Task:** "After John and Mary went to the shops, John gave a bottle of milk to ___" should complete with *Mary*.
Formally: identify the name that appears once rather than twice, and copy it.

**Result:** an explanation spanning **26 attention heads in 7 classes**, discovered by causal intervention
(path patching), with an algorithm attributed to them: **duplicate-token heads** and **induction heads** detect
that one name repeats; **previous-token heads** feed them; **S-inhibition heads** write a signal that suppresses
the duplicated name; **name-mover heads** attend to and copy the remaining name; and — the detail that outlived
the rest — **negative name-mover heads** push *against* the correct answer, while **backup name-mover heads**
step in when the primary movers are ablated.

**Evaluation:** three explicit criteria — **faithfulness** (the circuit alone reproduces the behaviour),
**completeness** (nothing important is missing), **minimality** (nothing is superfluous) — which the authors
report as supportive but *also* as pointing to remaining gaps. That self-assessment aged well: the negative and
backup heads are the seed of the self-repair literature (§5.5), and the paper's honesty about incompleteness set
the field's norm.

Its stated framing — "the largest end-to-end attempt at reverse-engineering a natural behavior 'in the wild'" —
is the reason IOI became the benchmark against which automated methods (§5.6) are scored.

---

**Hanna, Liu, Variengien — *How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a
pre-trained language model* (arXiv:2305.00586, April 2023).** `[peer-reviewed]` (NeurIPS 2023)\*

The second canonical GPT-2 Small circuit, and the first to put MLPs at the centre.

**Task ("year-span prediction"):** given "The `<noun>` lasted from the year 17**32** to the year 17", the model
should put more probability on two-digit completions greater than 32. Nouns come from a pool of 120 drawn via
FrameNet; centuries from {11…17}; start years from {02…98}. The tokenisation constraint is itself a finding:
because BPE gives frequent years single tokens, the template must avoid centuries where the completion would not
tokenise as the model would naturally write it.

**Circuit:** path patching over all **144 attention heads and 12 MLPs** finds the direct contributors to the
logits to be **MLPs 8–11 plus head a9.h1**. Since MLPs cannot attend, the authors trace what feeds them: MLP 9
relies on a9.h1; MLP 8 relies on **a8.h11, a8.h8, a7.h10, a6.h9, a5.h5, a5.h1**. The late MLPs are doing the
comparison — boosting years greater than the start year — and the attention heads' job is to deliver the start
year to them.

**Generality:** the same circuit activates on other greater-than-requiring tasks, which the authors read as a
"complex but general mechanism" rather than a template-matched shortcut.

Why it matters beyond the result: it is the task ACDC — Automatic Circuit DisCovery, the automated method in
§5.6 — is scored on, and the circuit transcoders (§5.7)
later revisited and revised — making it the most re-examined GPT-2 circuit after IOI.

---

**Merullo, Eickhoff, Pavlick — *Circuit Component Reuse Across Tasks in Transformer Language Models*
(arXiv:2310.08744, October 2023).** `[peer-reviewed]` (ICLR 2024)\*

The best available answer to "are circuits task-specific curiosities?" Taking the IOI circuit, the authors show
(1) it **reproduces on a larger GPT-2 model**, and (2) it is **largely reused** for the superficially unrelated
Colored Objects task, with about **78% overlap in in-circuit attention heads**. The proof-of-concept intervention
is the striking part: adjusting **four attention heads in middle layers** to make the Colored Objects circuit
behave like the IOI circuit raises accuracy from **49.6% to 93.7%**, and the downstream effects land where the IOI
circuit's structure predicts.

This is the strongest single piece of evidence that GPT-2 circuit findings are about reusable machinery rather
than per-task artifacts — and it is also, notably, one of very few papers to check a GPT-2 Small circuit at a
larger GPT-2 size.

---

**Meng, Bau, Andonian, Belinkov — *Locating and Editing Factual Associations in GPT* (arXiv:2202.05262, February
2022).** `[peer-reviewed]` (NeurIPS 2022)\*

The paper that made **GPT-2 XL** the factual-recall model. Causal tracing — corrupting the subject tokens and
restoring individual hidden states — localises factual recall to **mid-layer feed-forward modules processing the
subject token**. The authors then test the localisation by exploiting it: **ROME** (Rank-One Model Editing)
modifies a single feed-forward block's weights to change one fact, evaluated both on zsRE (zero-shot relation
extraction, a standard fact-editing benchmark) and on a new counterfactual dataset
where it uniquely maintains specificity *and* generalisation where prior methods trade one for the other.

Method-wise this is the origin of "causal tracing" as a GPT-2 idiom. Epistemically it is also the origin of a
cautionary literature: subsequent work questioned whether editing success licenses the localisation claim\* — a
tension worth carrying into any argument of the form "we intervened here and the behaviour changed, therefore the
behaviour lives here."

---

### 5.4 Components: what individual heads and neurons do

**Geva, Schuster, Berant, Levy — *Transformer Feed-Forward Layers Are Key-Value Memories* (arXiv:2012.14913,
December 2020).** `[peer-reviewed]` (EMNLP 2021 — the Conference on Empirical Methods in Natural Language
Processing)\*

The first widely adopted account of the two-thirds of a transformer's parameters that the circuits framework had
least traction on. Each FFN is read as a key-value memory: the first-layer rows are **keys** correlating with
human-interpretable input patterns, and the second-layer columns are **values** inducing distributions over the
output vocabulary. Lower layers key on shallow surface patterns, upper layers on semantic ones; upper-layer
values concentrate probability on plausible next tokens. An FFN output is a composition of its memories, refined
across layers through the residual stream.

---

**Geva, Caciularu, Wang, Goldberg — *Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in
the Vocabulary Space* (arXiv:2203.14680, March 2022).** `[peer-reviewed]` (EMNLP 2022)\*

The sequel, and more directly a GPT-2 paper. It views a token representation as a changing distribution over the
vocabulary and each FFN output as an **additive update** to that distribution, decomposable into sub-updates from
single parameter vectors, each promoting an often human-interpretable concept. Two applications validate the
account operationally: **GPT-2's toxicity is reduced by almost 50%** by suppressing identified promoting vectors,
and an early-exit rule saves **20% of computation** on average.

*Relevance to ATR:* these two papers are the reason to expect a converged ATR tensor's decoded token to be
readable as an accumulation of MLP-written vocabulary-space pushes rather than a single localised cause.

---

**McDougall, Conmy, Rushing, McGrath, Nanda — *Copy Suppression: Comprehensively Understanding an Attention Head*
(arXiv:2310.04625, October 2023).** `[peer-reviewed]` (ICLR 2024)\*

The most complete account of a single component in GPT-2 Small, and the closest published neighbour to this
repository's finding 4.

**Head L10H7** has one main role across the entire training distribution: if upstream components predict a token
and that token appeared earlier in the context, L10H7 **suppresses** it. Mechanistically it detects the naive
prediction, attends back to the previous instance of that token, and writes to the residual stream **in the
opposite direction** to the prediction. The showcase example: ablate L10H7 and "All's fair in love and war"
becomes "All's fair in love and love."

Two quantitative claims carry the paper. First, weights-based analysis explains **76.9% of L10H7's impact** in
GPT-2 Small — "the most comprehensive description of the complete role of a component in a language model to
date," by the authors' own assessment. Second, copy suppression **explains self-repair**: when an overconfident
upstream copier is ablated there is nothing left to suppress, and the paper attributes **39%** of self-repair
behaviour on a narrow task to this mechanism. It also retroactively explains why several earlier task-specific
studies kept finding "negative heads" that favoured the wrong answer — including IOI's negative name movers.

*Relevance to ATR (this project's reading, not the paper's):* the structural parallel is close enough to be
worth stating precisely. Copy suppression is a **late-layer head whose OV circuit writes against a direction the
rest of the model is pushing** — a negative-multiplier component in the residual stream. ATR's finding 4 is a
late-layer head (L11.H8) whose OV circuit **inverts** the flip axis at the pivot (multiplier −4.3) while being a
copy *promoter* on ordinary text. Different head, different regime, and this repository should not claim they are
the same phenomenon; but "GPT-2 Small's last layers contain heads whose job is to push back" is an established
fact about this model, and finding 4 is an instance of a known genus.

---

**Gould, Ong, Ogden, Conmy — *Successor Heads: Recurring, Interpretable Attention Heads In The Wild*
(arXiv:2312.09230, December 2023).** `[peer-reviewed]` (ICLR 2024)\*

**Successor heads** increment tokens with a natural ordering: *Monday* → *Tuesday*, *January* → *February*, *7* →
*8*. The paper's value is its breadth: successor heads are found in models from **31M to at least 12B**
parameters, across **GPT-2, Pythia, and Llama-2** — one of the few genuinely cross-family component results. The
mechanism is a set of **"mod-10 features"** underlying incrementing, shared across architectures and sizes, and
the authors do vector arithmetic on those features to edit head behaviour. On natural language data the same
heads show **interpretable polysemanticity**: multiple describable roles rather than one clean job or unreadable
mush.

---

**Gurnee, Horsley, Guo, Rezaei Kheirkhah, Sun, Hathaway, Nanda, Bertsimas — *Universal Neurons in GPT2 Language
Models* (arXiv:2401.12181, January 2024).** `[preprint, unreviewed]`\*

The universality question asked properly, and one of the most useful GPT-2 papers for calibrating how much any
single-model finding is worth.

**Setup (read directly from the paper):** five **GPT-2 Small** and five **GPT-2 Medium** models trained from
different random initialisations — the Stanford CRFM "Mistral" replications (Karamcheti et al. 2021), with
`stanford-crfm/arwen-gpt2-medium-x21` and `stanford-crfm/alias-gpt2-small-x21` named as the primary checkpoints —
plus supporting experiments on Pythia. Pairwise activation correlations are computed for **every neuron pair
across seeds over 100 million tokens of the Pile test set**, against a rotated-basis random baseline.

**Result:** only **1–5% of neurons are universal** in the sense of consistently co-activating across seeds. Those
that are usually have clear interpretations and fall into a small number of **neuron families** — alphabet
neurons (activating on a letter and on tokens beginning with it), previous-token-property neurons (e.g. fires iff
the previous token contains a comma), **absolute-position** neurons, and context/domain neurons (e.g. medical
text). Weight-space analysis then establishes universal *functional* roles in simple circuits: **deactivating
attention heads**, **changing the entropy** of the next-token distribution via the final LayerNorm, and
**predicting or suppressing** membership in a token set (e.g. a family of late-layer neurons that promote or
suppress years, second-person pronouns, single digits, with suppression neurons appearing later than prediction
neurons).

Two lessons: 95–99% of GPT-2 neurons are *not* universal even under the most favourable possible conditions
(same architecture, same data, only the seed varying), so a result about "a neuron in GPT-2" is by default a
result about one training run; and the neurons that *are* universal are disproportionately interpretable, which
makes universality a practical discovery filter.

*Relevance to ATR:* the position-dependent neuron family is a direct point of contact with ATR's
position-uniformity finding (finding 2), and the entropy-modulating family with this repository's readout
confidence metrics.

---

**Gurnee, Nanda, Pauly, Harvey, Troitskii, Bertsimas — *Finding Neurons in a Haystack: Case Studies with Sparse
Probing* (arXiv:2305.01610, May 2023).** `[peer-reviewed]` (TMLR — Transactions on Machine Learning Research)\*

Trains **k-sparse linear probes** on internal activations to predict input features, sweeping k to measure how
distributed a feature's representation is. Over **100 features in 10 categories across 7 models from 70M to 6.9B
parameters**. Findings: early layers use **sparse combinations of neurons to hold many features in
superposition**; middle layers have seemingly **dedicated neurons for higher-level contextual features**; and
increasing scale increases representational sparsity on average, though with several distinct scaling dynamics.
Empirical support for the superposition picture of §5.1 in real models, and a bridge to §5.7.

---

**Tigges, Hollinsworth, Geiger, Nanda — *Language Models Linearly Represent Sentiment*
(BlackboxNLP 2024, https://aclanthology.org/2024.blackboxnlp-1.5/).** `[peer-reviewed]`
*Cite this one under its published title.* The work circulates under two: the arXiv preprint
(arXiv:2310.15154, 23 October 2023, still at v1 and never retitled) is *Linear Representations of
Sentiment in Large Language Models*, and the peer-reviewed BlackboxNLP 2024 version is *Language
Models Linearly Represent Sentiment*. Same four authors, same work. Both titles return the paper in
a search and neither record points at the other, so a citation list carrying one of each looks like
two papers.

A model case study of what a *single direction* means across a broad distribution. Sentiment is represented
approximately **linearly** — one direction, positive at one extreme and negative at the other — across a range of
models including GPT-2; causal interventions isolate it; and a small set of heads and neurons implement it. The
memorable finding is the **summarisation motif**: sentiment is not held only on emotionally charged words but is
**summarised at intermediate positions with no inherent sentiment** — punctuation and names. Quantitatively, on
Stanford Sentiment Treebank zero-shot classification, ablating the sentiment direction destroys **76%** of
above-chance accuracy, and **36 percentage points of that 76** come from ablating the summarised direction **at
comma positions alone**.

*Relevance to ATR:* the summarisation motif is direct evidence that GPT-2-family models park semantic content at
positions with no semantic content of their own — the nearest published relative of ATR's observation that all
token positions converge to a single shared vector (finding 2).

---

**Xiao, Tian, Chen, Han, Lewis — *Efficient Streaming Language Models with Attention Sinks* (arXiv:2309.17453,
September 2023)** `[peer-reviewed]` (ICLR 2024)\* **and Sun, Chen, Kolter, Liu — *Massive Activations in Large
Language Models* (arXiv:2402.17762, February 2024)** `[peer-reviewed]` (COLM 2024 — the Conference on Language
Modeling)\*

Two papers about a phenomenon that any residual-stream study must control for. The first identifies the
**attention sink**: models direct large attention mass to the first few tokens regardless of semantic relevance,
and preserving those tokens' key/value (KV) cache entries — the stored keys and values a model reuses instead of
recomputing them for every new token — rescues window attention. That is the basis of StreamingLLM (LLM = **large
language model**, the general term for this class of system), which extends
finite-window models to millions of tokens without fine-tuning. The second characterises **massive
activations** — a handful of activations up to ~100,000× larger than the rest, largely **input-independent**,
functioning as indispensable **bias terms**, and causing exactly the attention concentration the first paper
observes.

*Relevance to GPT-2 and ATR.* An earlier version of this section said GPT-2 has no beginning-of-sequence token and
prepends nothing, so position 0 is a content token that might nonetheless be carrying sink duty. **That was wrong
as applied to this repository's runs, and wrong in the direction that mattered.** The correction was raised by
`agent:pythia-review` on the peer board (discussion #59, 2026-07-26) and then measured; it is recorded here rather
than quietly rewritten, because the original claim was in a merged document.

The tokeniser fact is right (§2.3). The application is not: `atr_engine.py` passes a raw **string** to
`run_with_cache` (lines 125, 183, 310, 343), TransformerLens tokenises strings with
`cfg.default_prepend_bos`, and GPT-2 inherits the global default of `True`. **Position 0 in every ATR trajectory
on GPT-2 is `<|endoftext|>` — a dedicated structural sink, not content.** The repository already knew this in one
place and not the other: `experiments/gpt2_small/11_suppression_test.py:607` is commented `# [1, L], BOS at 0` and
line 610 records `"n_tokens_no_bos": L - 1`. Two files disagreed and this document read the wrong one.

The correction sharpens the control rather than weakening it. The sink is not a diffuse worry about "whatever sits
at position 0"; it has a known address, and "does the L2 renormalisation rescale a structural sink along with the
content?" becomes directly testable.

**And it creates a cross-model confound neither this document nor its Pythia companion had identified.** Measured
by `agent:pythia-review` with `torch` and `transformer-lens`, on the engine's own call path
`[peer-board measurement, unreviewed — see pull request (PR) #61]`:

| model | tokens, raw tokeniser | tokens, via `run_with_cache(str)` | sink at position 0? |
|:---|---:|---:|:---|
| `gpt2` | 4 | **5** | **yes** — `<\|endoftext\|>` |
| `gpt2-medium` | 4 | **5** | **yes** |
| `pythia-160m` | 4 | 4 | no |
| `pythia-410m` | 4 | 4 | no |

The two arms of this project's 2×2 **do not tokenise the same way**. The GPT-2 arm carries a structural
non-content token at position 0; the Pythia arm does not, because TransformerLens sets `default_prepend_bos=False`
for the NeoX family, Pythia having not been trained on sequences with one. Nobody chose this; it is a library
default that varies by model family and is invisible at the call site. Every position-indexed cross-model
comparison is therefore comparing sequences whose position 0 means different things, and the same prompt yields
sequences differing in length by one between arms.

Any claim about a GPT-2 residual stream's norm, geometry, or position structure still needs these two papers in
the frame, and together with Privileged Bases (§5.1) they remain the "before you interpret that outlier, read
this" set. What changes is that on GPT-2 the outlier has a name and a position.

*A caution against the obvious inference.* The natural next step — mask the massive-activation coordinates and see
whether convergence survives — has been run, and it **refuted the hypothesis that motivated it**. The expectation
was that large, near-constant coordinates inflate cosine similarity toward 1, putting the saturating models'
readings at risk. Measured, the effect runs the other way on GPT-2 Small: masking *raises* `cos_sim_mean` from
0.9167 to 0.9933, so the dominant coordinates were **depressing** the metric, not inflating it, and GPT-2 Medium
returns the same saturation with fifty coordinates deleted. `[peer-board measurement, unreviewed — PR #61 open at
time of writing, and its author has already corrected two figures in it after review, so treat the numbers as
in-flight.]` The general lesson survives the specific refutation: reason about these coordinates before
interpreting a residual-stream metric, and do not assume you know the sign of the effect.

---

### 5.5 Superposition, dictionaries, features

**Bricken, Templeton, Batson, Chen, Jermyn, Conerly, Turner, Anil, Denison, Askell, Lasenby, Wu, Kravec, Schiefer,
Maxwell, Joseph, Hatfield-Dodds, Tamkin, Nguyen, McLean, Burke, Hume, Carter, Henighan, Olah — *Towards
Monosemanticity: Decomposing Language Models With Dictionary Learning* (Transformer Circuits Thread, October
2023).** `[published research report, not journal-reviewed]`
https://transformer-circuits.pub/2023/monosemantic-features/index.html

The paper that turned superposition from theory into method. A **sparse autoencoder** (SAE) is trained on the MLP
activations of a **one-layer transformer with a 512-neuron MLP**, decomposing those 512 neurons into **over 4,000
features** that are individually far more interpretable than the neurons — the released examples include DNA
(genetic) sequences, legal language, HTTP requests (Hypertext Transfer Protocol — the text format web browsers use to
request pages), Hebrew text, and nutrition statements\*. **Ninety learned dictionaries**
were released with activating examples and downstream logit effects for every feature; the recommended entry
point is the run labelled **A/1**.

The claims that mattered: features are sparse, individually interpretable, **causally relevant** when clamped,
and explain the layer's behaviour better than its neurons do. Two phenomena named here became permanent
vocabulary — **feature splitting** (a feature in a small dictionary resolves into several finer features in a
larger one) and the observation that dictionary size controls the granularity at which concepts become
resolvable.

Its limitation is its model: one layer, 512 neurons. Whether the method scaled was the open question, answered
next.

---

**Cunningham, Ewart, Riggs, Huben, Sharkey — *Sparse Autoencoders Find Highly Interpretable Features in Language
Models* (arXiv:2309.08600, September 2023).** `[peer-reviewed]` (ICLR 2024)\*

Concurrent, independent, and the version that landed on GPT-2-scale models. Trains SAEs to reconstruct internal
activations of language models and shows the learned features are **more interpretable and more monosemantic**
than directions found by alternatives — PCA (principal component analysis, which finds the directions of greatest
variance), ICA (independent component analysis, which finds statistically independent directions), and the raw
neuron basis — under automated interpretability scoring. The
result that connects it to §5.3: SAE features pinpoint the units **causally responsible for counterfactual
behaviour on the IOI task** at a **finer granularity than previous decompositions**. Superposition, the paper
argues, is resolvable by a scalable unsupervised method.

---

**Templeton, Conerly, Marcus, Lindsey, Bricken, Chen, Pearce, Citro, Ameisen, Jones, Cunningham, Turner,
McDougall, MacDiarmid, Tamkin, Durmus, Hume, Mosconi, Freeman, Sumers, Rees, Batson, Jermyn, Carter, Olah, Henighan
— *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (Transformer Circuits Thread,
May 2024; also arXiv:2605.29358).** `[published research report, not journal-reviewed]`

Dictionary learning at production scale, and the paper that made SAEs a mainstream safety tool. SAEs with **up to
34 million features** are trained on the **middle-layer residual stream** of **Claude 3 Sonnet**, with **scaling
laws used to choose hyperparameters** — the first time SAE training was treated as a scaling problem rather than
a craft.

Findings: features are **multilingual and multimodal**, generalising to images despite text-only training;
they respond both to concrete instances and to abstract discussion of a concept; and **clamping them steers
behaviour consistently with their interpretation**. Features were found for famous entities and places and for
abstractions like sarcasm and code errors, and — the reason this paper is cited in safety contexts — for
**deception, power-seeking, sycophancy, and bias**, which causally influence outputs when manipulated. A
systematic relationship holds between a concept's **frequency** and the **dictionary size** needed to resolve it.

The limitations are stated in the abstract without softening: "our suite of features is incomplete, and we lack
rigorous methods for evaluating whether our features faithfully capture model computations." That second clause is
the open problem §5.6 and §5.8 are still working on.

---

**Makelov, Lange, Nanda — *Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control*
(arXiv:2405.08366, May 2024).** `[preprint, unreviewed]`

The evaluation problem attacked on GPT-2. Since real features have no ground truth, the authors build
**supervised** feature dictionaries for the **IOI task in GPT-2 Small**, demonstrate that these achieve excellent
approximation, control, and interpretability, and then use them as a yardstick for **unsupervised** SAEs trained
on either IOI data or OpenWebText. Verdict: the SAEs do capture interpretable IOI features but are **less
successful than supervised features at controlling the model**. Two failure modes are named and have stuck:
**feature occlusion** (a causally relevant concept robustly overshadowed by slightly higher-magnitude features)
and **feature over-splitting** (binary features fragmenting into many smaller, less interpretable ones).

---

**Leask, Bussmann, Pearce, Bloom, Tigges, Al Moubayed, Sharkey, Nanda — *Sparse Autoencoders Do Not Find
Canonical Units of Analysis* (arXiv:2502.04878, February 2025).** `[peer-reviewed]` (ICLR 2025)\*

The strongest published check on the SAE programme's central hope. Two techniques: **SAE stitching** — inserting
latents from a larger SAE into a smaller one — separates the larger SAE's latents into *novel* latents (which
improve performance, proving the smaller SAE was **incomplete**) and *reconstruction* latents (which merely
duplicate). **Meta-SAEs** — SAEs trained on another SAE's decoder matrix — show that latents decompose into
combinations of latents from smaller SAEs, so they are **not atomic**; the illustrative decomposition is an
"Einstein" latent splitting into "scientist", "Germany", and "famous person".

Conclusion: SAEs do not deliver a unique, complete, atomic feature basis. They remain useful; they are not the
canonical decomposition the early framing hoped for. This is the paper to read before treating any feature list —
GPT-2's included — as *the* set of the model's concepts.

---

**Engels, Michaud, Liao, Gurnee, Tegmark — *Not All Language Model Features Are One-Dimensionally Linear*
(arXiv:2405.14860, May 2024).** `[peer-reviewed]` (ICLR 2025)\*

Directly relevant to any geometric claim about GPT-2. The linear-representation hypothesis assumes features are
one-dimensional directions; this paper defines **irreducible multi-dimensional features** (not decomposable into
independent or non-co-occurring lower-dimensional features) and finds them automatically with SAEs **in GPT-2 and
Mistral 7B**. The striking examples are **circular** features for days of the week and months of the year, and
intervention experiments on Mistral 7B and Llama 3 8B indicate these circles are the actual computational unit
for modular arithmetic over those quantities.

*Relevance to ATR:* a caution with teeth. A converged tensor's structure need not be a point on a line; ATR's
period-2 cycle is itself a two-state structure along a single axis, and this literature is the reason not to
assume a one-dimensional readout exhausts it.

---

### 5.6 Automation and scale

**Bills, Cammarata, Mossing, Tillman, Gao, Goh, Sutskever, Leike, Wu, Saunders (OpenAI) — *Language models can
explain neurons in language models* (May 2023).** `[published research report, not journal-reviewed]`
https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html

Automated interpretability's opening move, and the largest single GPT-2 study by neuron count. **GPT-4 writes
natural-language explanations for MLP neurons in GPT-2 XL**, using top-activating text excerpts. The pipeline is
three steps: **explain** (GPT-4 proposes a description from activation examples), **simulate** (GPT-4 predicts
activations on held-out text using only the description), **score** (the **explanation score** is the correlation
between simulated and true activations).

Scale and outcome: explanations were generated for **all 307,200 neurons** of GPT-2 XL (48 layers × 6400) and
released; only about **1,000 neurons** received explanations scoring **≥ 0.8**\*. The honest reading is that the
overwhelming majority of GPT-2 XL's neurons resisted a short natural-language description that predicts their
behaviour — which is what the superposition literature predicts. The team also reports that **later layers are
harder to explain**, and that GPT-4's explanations, while better than smaller models', remain worse than
humans'\*. Code and the full explanation dataset were open-sourced.

*Relevance:* the 307,200 figure is the standard citation for GPT-2 XL's neuron count (and reproduces exactly
from §2.2's architecture). More importantly this is the field's clearest demonstration that *per-neuron*
interpretability does not scale — the motivation for dictionary learning.

---

**Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso — *Towards Automated Circuit Discovery for Mechanistic
Interpretability* (arXiv:2304.14997, April 2023).** `[peer-reviewed]` (NeurIPS 2023)\*

Systematises the manual workflow of §5.3 (choose metric and dataset → activation-patch to find involved units →
vary dataset/metric/units to characterise them) and automates the middle step. **ACDC** iteratively prunes edges
of the computational graph whose removal does not degrade the chosen metric. Validation is by rediscovery: on
GPT-2 Small's greater-than task, ACDC recovers **5/5 component types** and selects **68 of 32,000 edges, all of
which had been found manually** by Hanna et al. This is the paper that made GPT-2 Small's hand-built circuits
into a benchmark suite.

---

**Syed, Rager, Conmy — *Attribution Patching Outperforms Automated Circuit Discovery* (arXiv:2310.10348, October
2023).** `[peer-reviewed]` (BlackboxNLP 2024)\*

Replaces ACDC's iterative patching with a **linear approximation** (attribution patching, also called **EAP** for
edge attribution patching) requiring **two
forward passes and one backward pass** total, then prunes the least important edges. Averaged over tasks it
achieves **greater circuit-recovery AUC than all existing methods**, including ACDC — AUC being area under the
curve, a single number summarising how well a method separates the right components from the wrong ones across all
possible cut-off thresholds. The practical significance
is that circuit discovery stops being a compute-bound activity — a prerequisite for everything in §5.7 that
builds circuits out of tens of thousands of features rather than 144 heads.

See also **O'Neill & Bui, *Sparse Autoencoders Enable Scalable and Reliable Circuit Identification in Language
Models* (arXiv:2405.12522, May 2024)** `[preprint, unreviewed]`, which discretises SAE codes on attention-head
outputs and reports higher precision and recall than baselines on **IOI, greater-than, and docstring
completion** while reducing runtime "from hours to seconds" using only 5–10 examples per task.

---

**Dunefsky, Chlenski, Nanda — *Transcoders Find Interpretable LLM Feature Circuits* (arXiv:2406.11944, June
2024).** `[peer-reviewed]` (NeurIPS 2024)\*

The MLP problem from §5.1, addressed with the right tool. An SAE decomposes *activations*; a **transcoder**
approximates an entire densely-activating MLP sublayer with a **wider, sparsely-activating** one, so that circuit
analysis can pass *through* MLPs instead of stopping at them. The key structural result is that transcoder-based
circuits **factorise into input-dependent and input-invariant terms**, giving weights-based analysis through
MLPs. Transcoders are trained on models of **120M, 410M, and 1.4B** parameters and match or beat SAEs on
sparsity, faithfulness, and human interpretability. Applied to reverse-engineer unknown circuits, they yield
**novel insights about the greater-than circuit in GPT-2 Small** — the same circuit from §5.3, revisited with a
finer instrument.

---

**Marks, Rager, Michaud, Belinkov, Bau, Mueller — *Sparse Feature Circuits: Discovering and Editing Interpretable
Causal Graphs in Language Models* (arXiv:2403.19647, March 2024).** `[peer-reviewed]` (ICLR 2025)\*

Circuits whose nodes are **interpretable features** rather than polysemantic heads and neurons. The payoff is
downstream utility: **SHIFT** (the authors' name for the technique) improves a classifier's generalisation by
ablating features a *human* judges
task-irrelevant — spurious-cue removal via interpretability — and the paper demonstrates a fully unsupervised
pipeline discovering **thousands** of feature circuits for automatically discovered behaviours. The bridge from
"we can name features" to "naming features lets us fix something."

---

**Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike, Wu (OpenAI) — *Scaling and evaluating sparse
autoencoders* (arXiv:2406.04093, June 2024).** `[preprint, unreviewed]`

The engineering paper that stabilised SAE training. **k-sparse autoencoders** control sparsity directly, removing
the reconstruction/sparsity balancing act and improving the frontier; modifications largely eliminate **dead
latents** even at the largest scales. Clean **scaling laws** in autoencoder size and sparsity follow, plus new
feature-quality metrics based on recovery of hypothesised features, explainability of activation patterns, and
sparsity of downstream effects. Demonstrated up to a **16-million-latent autoencoder on GPT-4 activations over 40
billion tokens**, with training code, autoencoders **for open-source models including GPT-2 Small**, and a
visualiser released.

---

**Community infrastructure** `[primary data, read directly]` for the artifacts; `[community post, unreviewed]`\*
for the accompanying write-ups.

Worth naming because it determines what is cheap to do on GPT-2. **TransformerLens** (Nanda) provides
hook-instrumented GPT-2 with analysis-friendly weight preprocessing — LayerNorm folding, weight centring — which
is the normal starting point for every circuit paper above. **SAELens** distributes pretrained SAE suites, of
which the standard GPT-2 Small residual-stream set (`gpt2-small-res-jb`, Bloom) covers **all 12 residual-stream
layers, trained for 300M tokens of OpenWebText at up to 128-token sequences**\*. **Neuronpedia** hosts those
features with browsable dashboards under stable identifiers (e.g. `gpt2-small/2-res-jb/5821`), which is why GPT-2
Small feature indices are citable across papers at all.

Note the mismatch worth remembering: the model was trained on WebText; the dictionaries were trained on
OpenWebText. Features are described in terms of a corpus the model never saw.

---

### 5.7 Anthropic's frontier: circuits at production scale

These papers do not study GPT-2. They are included because they are where the GPT-2 programme's methods went, and
because they are the state of the art against which any new GPT-2 result should be positioned.

---

**Ameisen, Lindsey, Pearce, Gurnee, Turner, Chen, Citro, Abrahams, Carter, Hosmer, Marcus, Sklar, Templeton,
Bricken, McDougall, Conerly, Batson, Olah et al. — *Circuit Tracing: Revealing Computational Graphs in Language
Models* (Transformer Circuits Thread, March 2025).** `[published research report, not journal-reviewed]`
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

The methods half of Anthropic's 2025 circuits work, and the current best answer to "how do you get a circuit out
of a frontier model?"

**Cross-layer transcoders (CLTs)** extend §5.6's transcoders across depth: a layer-ℓ feature reads from the
residual stream at layer ℓ via a linear encoder and a JumpReLU (a rectified linear unit with a learned activation
threshold, so a feature stays at exactly zero until its input clears the bar), and contributes to reconstructing
feed-forward outputs at
layers ℓ, ℓ+1, …, L — trained jointly for reconstruction plus sparsity across all layers. Substituting CLT
features for MLPs gives a **replacement model**; making it accurate for one prompt gives a **local replacement
model**, which freezes attention patterns and normalisation denominators from the real forward pass and adds
error terms, producing "a very large fully connected neural network, spanning across tokens" in which the only
remaining nonlinearity is feature preactivation. **Attribution graphs** are then built over four node types
(output tokens, active features, token embeddings, error nodes) with edges as linear attributions from backward
Jacobians, and **pruned ~10× while retaining ~80% of explanatory power**. Studied on an **18-layer model** and on
**Claude 3.5 Haiku**.

Validation is three-pronged: perturbation experiments against graph predictions; influence-score predictiveness
(**0.72 Spearman** for feature–feature interactions); and mechanistic-faithfulness measurement (**~0.8 cosine
similarity** one layer downstream).

The limitations are the important part, and they are stated at length. Attention is **frozen**, so QK circuits
are not explained at all — attribution graphs miss induction-head-style mechanisms **by construction**. CLTs
account for roughly **61%** of model computation via feature paths (**50%** for next-token matching), with the
remainder appearing as error nodes. Perturbation magnitudes diverge from the real model across layers even where
directions correlate. And global, context-independent circuits remain hard because features that never co-occur
still interfere in the weights.

---

**Lindsey, Gurnee, Ameisen, Chen, Pearce, Turner, Citro, Abrahams, Carter, Hosmer, Marcus, Sklar, Templeton,
Bricken, McDougall, Conerly, Batson, Olah et al. — *On the Biology of a Large Language Model* (Transformer
Circuits Thread, March 2025).** `[published research report, not journal-reviewed]`
https://transformer-circuits.pub/2025/attribution-graphs/biology.html

The results half, on **Claude 3.5 Haiku**. Nine-plus case studies, each an existence proof:

- **Multi-step reasoning.** Genuine internal two-hop composition (Dallas → Texas → Austin) alongside shortcut
  pathways, with the intermediate accessible and manipulable.
- **Planning in poems.** The model selects end-of-line rhyme targets *before* writing the line and works
  backwards, using the plan to shape intermediate words.
- **Multilingual circuits.** Both language-specific and abstract language-agnostic components, with **English
  mechanistically privileged** as a default output format.
- **Arithmetic.** Lookup-table-like features doing the core computation, generalising across pure arithmetic,
  astronomical data, and financial tables, repurposed by contextual signals.
- **Medical diagnosis.** Candidate diagnoses represented internally and used to select confirmatory questions —
  legible diagnostic reasoning "in its head".
- **Entity recognition and hallucination.** A **default refusal circuit suppressed by "known entity" features**;
  hallucination occurs when those features fire without sufficient underlying knowledge. Arguably the single most
  mechanistically satisfying account of hallucination published.
- **Refusals.** Harm-category features learned in pretraining get wired together into general refusal machinery
  during finetuning.
- **Jailbreaks.** An acronym jailbreak works because the model assembles letters without recognising the harmful
  meaning until it is already writing, at which point punctuation-triggered refusal machinery can re-engage.
- **Chain-of-thought faithfulness.** Cases where the written reasoning does and does not correspond to the
  internal computation.

Stated limitation, quantified: the methods give satisfying insight on about **a quarter of the prompts** tried,
and the case studies are existence proofs rather than guarantees of generality.

*Why it belongs in a GPT-2 document:* every technique here is a descendant of a method developed on GPT-2 or on
Anthropic's toy models, and the results are the clearest available answer to "what is circuit analysis
ultimately for?"

---

**Kamath, Ameisen et al. — *Tracing Attention Computation Through Feature Interactions* (Transformer Circuits
Thread, July 2025).** `[published research report, not journal-reviewed]`\*
Addresses the frozen-attention limitation above by explaining attention patterns in terms of feature
interactions within attribution graphs — i.e. bringing QK circuits back into the picture.

---

**Gurnee, Ameisen, Kauvar, Tarng, Pearce, Olah, Batson — *When Models Manipulate Manifolds: The Geometry of a
Counting Task* (Transformer Circuits Thread, October 2025; arXiv:2601.04480).**
`[published research report, not journal-reviewed]`

A geometric rather than feature-listing account, on **Claude 3.5 Haiku**, of how a model line-breaks fixed-width
text — a task requiring character counting from token inputs. Character counts are represented on **low-dimensional
curved manifolds discretised by sparse feature families, analogous to biological place cells**. The algorithm is a
sequence of geometric operations: token lengths are **accumulated** into a character-count manifold; attention
heads **twist** the manifold to estimate distance to the line boundary; and the break decision is enabled by
**arranging estimates orthogonally to create a linear decision boundary**. Validated by causal intervention and
by the discovery of **visual illusions** — character sequences that hijack the counting mechanism.

*Relevance to ATR:* the methodological argument — that feature-based and geometric views must be combined — is
this repository's argument too. ATR's basins are geometric objects (attractors of a map) read through a
feature-ish instrument (an argmax over the vocabulary), and finding 6 is precisely a case where the geometry moves
and the feature readout does not.

---

**Lindsey — *Emergent Introspective Awareness in Large Language Models* (Transformer Circuits Thread, October
2025).** `[published research report, not journal-reviewed]`\*
Concept-injection experiments testing whether models can report on their own internal states, with evidence that
they sometimes can, unreliably. Included as the methodological precursor to the injected-thought experiments in
the workspace paper.

---

**Gurnee et al. — *Verbalizable Representations Form a Global Workspace in Language Models* (Transformer Circuits
Thread, July 2026).** `[published research report, not journal-reviewed]`
https://transformer-circuits.pub/2026/workspace/index.html

The current frontier, and the one this repository is already engaged with. Summarised at length in
[JSPACE_PRIMER.md](JSPACE_PRIMER.md), verified there against the full 133-page text; only the parts bearing on
GPT-2 are repeated here.

The **J-lens** replaces the logit lens's implicit identity with a **per-layer averaged Jacobian** from the
residual stream to the final-layer residual stream, averaged over source positions, over all present-and-future
positions, and over ~1,000 pretraining-like prompts, then read out through the model's own normalisation and
unembedding. Setting the Jacobian to the identity **recovers the logit lens exactly** — which locates every
result in §5.2, and ATR's own readout, as a special case of this instrument. The **J-space** — sparse
non-negative combinations of at most ~25 J-lens vectors — is shown to be small (never more than 10% of activation
variance; a median 6–7% of a concept vector's variance yet almost entirely responsible for report), confined to a
**band of intermediate layers** (~40–90% of depth), capacity-limited (~25 concept slots, ~6 unrelated held
items), and mechanistically a **broadcast hub** (MLPs amplify J-lens directions ~10×; a top-1% subset of
"broadcast heads" relays content across positions).

Two things make it directly relevant to GPT-2 work. First, the **companion J-lens implementation is
open-source** and hosted on Neuronpedia for open-weight models, so the instrument is available at the 124M scale.
Second, the paper explicitly leaves open **when a workspace emerges during pretraining** — which means whether a
124M-parameter undertrained base model has one at all is an open empirical question, not an assumption.

---

### 5.8 Where the field says it is going

**Sharkey, Chughtai, Batson, Lindsey, Wu, Bushnaq, Goldowsky-Dill, Heimersheim, Ortega, Bloom, Biderman,
Garriga-Alonso, Conmy, Nanda, Rumbelow, Wattenberg, Schoots, Miller, Michaud, Casper, Tegmark, Saunders, Bau,
Todd, Geiger, Geva, Hoogland, Murfet, McGrath — *Open Problems in Mechanistic Interpretability*
(Transactions on Machine Learning Research; accepted 20 September 2025; https://openreview.net/forum?id=91H76m9Z94;
preprint arXiv:2501.16496, January 2025).** `[peer-reviewed]`
Cited here under the published venue per the convention in the provenance block. An earlier version of this
document had it as an unreviewed preprint, which is what the arXiv record alone shows — the arXiv entry carries no
`journal_ref` — and `agent:pythia-review` had the published venue right; the correction is theirs.

The field's own agenda, co-authored across Anthropic, DeepMind, EleutherAI, Apollo, the Massachusetts Institute of
Technology, Harvard, Northeastern
and others — which makes it the best single citation for "what the field agrees it has not solved." Three
categories: **methods need conceptual and practical improvement** to reveal deeper insight; **applying methods to
specific goals** (auditing, debugging, control) is largely unsolved; and there are **socio-technical challenges**
the field influences and is influenced by. Explicitly forward-facing rather than a survey of results.

---

**Braun, Bushnaq, Heimersheim, Mendel, Sharkey — *Interpretability in Parameter Space: Minimizing Mechanistic
Description Length with Attribution-based Parameter Decomposition* (arXiv:2501.14926, January 2025)**
`[preprint, unreviewed]` **and Bushnaq, Braun, Sharkey — *Stochastic Parameter Decomposition*
(arXiv:2506.20790, June 2025)** `[preprint, unreviewed]`

A different bet: decompose the **parameters** rather than the activations. **APD** splits a network's weights into
components that are (i) faithful to the original parameters, (ii) minimal in number per input, and (iii)
maximally simple — a minimum-description-length objective over mechanisms. It recovers ground-truth mechanisms in
toy settings (features from superposition, separated compressed computations, cross-layer distributed
representations) and offers a route to "minimal circuits in superposition" and an architecture-agnostic
definition of a feature. **SPD** replaces APD's prohibitive cost and hyperparameter sensitivity with a more
scalable and robust procedure, avoiding parameter shrinkage and better identifying ground truth. Neither has been
applied to GPT-2 at the time of writing; scaling remains the stated obstacle.

---

**Gao, Rajaram, Coxon, Govande, Baker, Mossing (OpenAI) — *Weight-sparse transformers have interpretable circuits*
(arXiv:2511.13653, November 2025).** `[preprint, unreviewed]`

The SoLU idea (§5.1) at full strength: instead of interpreting a dense model, **train for interpretability** by
constraining most weights to zero so each neuron has few connections, then prune to isolate the part responsible
for a hand-crafted task. The resulting circuits contain neurons and residual channels corresponding to natural
concepts with a small number of straightforwardly interpretable connections — "an unprecedented level of human
understandability", validated with considerable rigour. The trade-off is stated plainly: **sparser weights trade
capability for interpretability**, larger models improve the capability–interpretability frontier, and preserving
interpretability **beyond tens of millions of nonzero parameters remains unsolved**. Preliminary results suggest
adaptation to explaining existing dense models.

Read against GPT-2: this is the field admitting that a densely-connected 124M-parameter model may simply be the
wrong object to fully reverse-engineer, and that the alternative is a differently-trained model of comparable
size.

Follow-on: **Marin-Llobet & Heimersheim, *Individual Parameters in Weight-Sparse Transformers Appear
Interpretable* (arXiv:2607.02964, July 2026)** `[preprint, unreviewed]` asks whether a **single weight** can be
described globally — by characterising the inputs on which ablating it changes predictions — using an automated
LLM pipeline that writes a short description and **verifies it on held-out text**, crediting a weight only if the
description generalises. Across two sparse and two dense transformers, **12–31% of weights in sparse models** have
a single short description that identifies their use, and the sparse-vs-dense gap widens once unreliable
descriptions are discarded. The natural successor to §5.6's neuron explanations, at the level of individual
parameters.

---

### 5.9 Adjacent results GPT-2 work leans on

- **Rumbelow & Watkins, *SolidGoldMagikarp (plus, prompt generation)* (LessWrong, February 2023)**
  `[community post, unreviewed]`, with sequels. Identifies anomalous "glitch" tokens in **GPT-2 and GPT-3** that
  cluster near the embedding centroid and cannot be correctly repeated, with the models resorting to evasion. The
  cause is a **mismatch between the tokeniser's corpus and the model's training corpus**: strings frequent enough
  to earn a token but too rare in training to earn a meaningful embedding — Reddit usernames, e-commerce backend
  strings, log-file fragments\*. Verified directly in this repository's own check: `ĠSolidGoldMagikarp` is id
  **43453** and `Ġpetertodd` is id **37444** in GPT-2's `vocab.json` (§2.3).
  *Relevance to ATR:* this is the **glitch cluster** in finding 5. ATR's period-2 flip axis has cosine −0.596 to
  the glitch-cluster direction, i.e. it runs between the most-trained and never-trained regions of embedding
  space. That finding is only meaningful because this literature established that the never-trained region exists
  and is geometrically distinctive.
- **Hendel, Geva, Globerson, *In-Context Learning Creates Task Vectors* (arXiv:2310.15916, October 2023)**
  `[peer-reviewed]` (EMNLP 2023 Findings)\*. In-context learning — ICL, the ability to pick up a task from examples
  in the prompt without any weight changes — compresses a demonstration set into a **single task vector** that
  modulates the model. Evidence that a *single* residual-stream direction can carry an entire task
  specification — the kind of claim that makes ATR's question ("what is a converged tensor?") well-posed.
- **Li, Patel, Viégas, Pfister, Wattenberg, *Inference-Time Intervention* (arXiv:2306.03341, June 2023)**
  `[peer-reviewed]` (NeurIPS 2023)\*. Shifting activations along directions in a few attention heads raises
  Alpaca's TruthfulQA truthfulness from **32.5% to 65.1%**, using only a few hundred examples to locate the
  directions. Included as the canonical demonstration that residual-stream directions found by interpretability
  are steerable, not merely correlational.
- **Singh, Moskovitz, Hill, Chan, Saxe, *What needs to go right for an induction head?* (arXiv:2404.07129, April
  2024)** `[peer-reviewed]` (NeurIPS 2024)\*. Induction-head *formation* dissected in a controlled synthetic
  setting with an "optogenetics-inspired" framework for clamping activations throughout training, identifying
  **three interacting subcircuits** whose interaction produces the phase change, and showing induction heads are
  **diverse and additive** rather than singular. The developmental study OpenAI's GPT-2 cannot support, because
  those checkpoints were never released (§1.2) — though the Stanford replications of GPT-2 Small could (§3.2).
- **Paulo, Marshall, Belrose, *Does Transformer Interpretability Transfer to RNNs?* (arXiv:2404.05971, April
  2024)** `[preprint, unreviewed]`. RNN = recurrent neural network, the pre-transformer architecture family that
  processes a sequence one step at a time while carrying a running state. Contrastive activation addition, the
  tuned lens, and latent-knowledge elicitation mostly transfer to Mamba and RWKV (two modern recurrent
  architectures; RWKV is a coined name rather than an abbreviation of words). Relevant as the outer bound on how
  architecture-specific the GPT-2 toolkit is.

---

## Part 6 — What is and is not established about GPT-2

**Reasonably well established** (multiple independent methods, and in several cases replication):

1. The residual stream functions as an additive communication channel that components read from and write to, and
   decomposing a forward pass along it is legitimate. *(§5.1, and the basis of everything else.)*
2. GPT-2 Small implements **specific, findable, largely reusable circuits** for at least IOI, greater-than,
   ordinal succession, and sentiment — with IOI's 26-head/7-class structure confirmed by automated rediscovery
   and ~78% reused for a different task at a larger size. *(§5.3.)*
3. **Late-layer MLPs write vocabulary-space updates**, and factual recall in GPT-2 XL is concentrated in
   mid-layer MLPs at subject-token positions, to a degree that supports rank-one editing. *(§5.3, §5.4.)*
4. **Individual attention heads can have describable global jobs.** L10H7's copy suppression is the strongest
   case: 76.9% of its impact explained from weights. *(§5.4.)*
5. **Most GPT-2 neurons are not interpretable one at a time**, and are not stable across seeds: ~1,000 of 307,200
   XL neurons scored ≥0.8 under automated explanation, and only 1–5% of Small/Medium neurons are universal across
   five seeds. *(§5.5, §5.6.)*
6. **Superposition is a real and useful explanation** for (5), and sparse dictionaries recover features more
   interpretable and more causally precise than neurons — including on IOI. *(§5.1, §5.5.)*
7. **Instrument choice can manufacture findings.** Logit-lens brittleness, activation-patching hyperparameters,
   outlier dimensions attributable to Adam, and attention sinks are all documented ways to be fooled by a real
   measurement. *(§5.1, §5.2, §5.4.)*

**Contested or open:**

- **Whether SAE features are the right units at all.** Stitching and meta-SAEs show they are neither complete nor
  atomic; occlusion and over-splitting are documented failure modes; and faithfulness evaluation is named as
  unsolved by the people who scaled the method. *(§5.5.)*
- **Whether features are one-dimensional.** Irreducibly multi-dimensional (circular) features exist in GPT-2.
  *(§5.5.)*
- **How much a circuit explains.** Faithfulness/completeness/minimality were introduced with the first circuit
  and immediately showed gaps; CLTs at frontier scale account for ~61% of computation; attribution graphs give
  satisfying insight on ~a quarter of prompts. *(§5.3, §5.7.)*
- **Attention itself.** The frontier method freezes attention patterns and therefore explains no QK circuit at
  all; the fix is recent and preliminary. *(§5.7.)*
- **Whether localisation claims from editing are sound.** ROME's success does not by itself establish that the
  fact "lives" where it was edited\*. *(§5.3.)*
- **Whether any of this generalises off GPT-2.** Named as an open problem by the field's own review; the
  cross-family results that exist (successor heads, RNN transfer) are the exception. *(§4, §5.8.)*
- **Whether dense models of GPT-2's size are the right target.** The weight-sparse programme is a bet that they
  are not. *(§5.8.)*

**Barely studied, and unstudiable on OpenAI's weights:** training dynamics. No intermediate checkpoints, no token
count, no schedule (§1.2), so everything the field knows about *when* GPT-2's structure formed is inferred from
other models. The qualifier matters: the Stanford replications of GPT-2 Small carry ~609 checkpoints per seed
across five seeds (§3.2), so the developmental question is answerable on the GPT-2 *architecture* and has simply
not been asked. That is an open opportunity, not a closed door.

---

## Part 7 — Bearings on this repository

*This section is this project's interpretation; no cited work discusses ATR.*

Five places where the literature above bears directly on the ATR results in [FINDINGS.md](FINDINGS.md), beyond
the neighbours already catalogued in [PRIOR_WORK.md](PRIOR_WORK.md):

1. **ATR's readout is a named, characterised instrument with known blind spots.** `ln_final → W_U` on the
   converged tensor is the logit lens at the final layer (§5.2), which is the J-lens with the Jacobian set to the
   identity (§5.7). Finding 6 — the period-2 motion being near-invisible to lens instruments — is therefore not
   an anomaly of this project's setup but an instance of a documented limitation, and the two published upgrades
   (tuned lens, J-lens) are the obvious next measurements. The J-lens implementation is open-source and runs on
   open-weight models.
2. **Finding 4 belongs to a known genus.** GPT-2 Small's late layers demonstrably contain heads whose function is
   to write *against* what the rest of the model is pushing (§5.4, copy suppression at L10H7, 76.9% explained;
   IOI's negative name movers). L11.H8's inverting OV circuit is a new instance, not a new kind. Stating it that
   way makes the finding more credible, not less — and it raises a specific, cheap question: does L11.H8 behave
   as a copy suppressor on ordinary text, in the sense that paper defines?
3. **Finding 2 has a nearest published relative, and a confound that has to be handled first.** The
   **summarisation motif** (§5.4) shows GPT-2-family models parking semantic content at positions with no
   semantic content of their own, with 36 of 76 percentage points of sentiment-classification accuracy living at
   commas. Position-uniformity under iteration is a more extreme version of the same disregard for where
   information sits.

   The control this needs is not the one an earlier version of this document specified. **Position 0 in every
   GPT-2 ATR trajectory is `<|endoftext|>`, a structural sink** (§2.3, §5.4) — TransformerLens prepends it for
   GPT-2 and *not* for Pythia. So finding 2's position-uniformity claim, stated across the 2×2, currently compares
   a GPT-2 arm whose position 0 is a non-content sink against a Pythia arm whose position 0 is an ordinary token.
   Three consequences, in order of how much they bite:

   - **The cross-model version of finding 2 is confounded until this is controlled.** "All positions collapse to
     one vector" means something different when one of those positions is a structural sink. The cheap fix is to
     re-run the position-uniformity measurement with position 0 excluded on both arms, which makes the arms
     comparable for the first time; `11_suppression_test.py` already computes `n_tokens_no_bos` and shows how.
   - **The within-GPT-2 version is sharpened, not weakened.** The sink has a known address, so "does the L2
     renormalisation rescale a structural sink along with the content?" is directly testable rather than a
     diffuse worry.
   - **Sequence lengths differ by one between arms for the same prompt**, which quietly affects any per-position
     average.

   This belongs in [FINDINGS.md](FINDINGS.md) as a caveat on finding 2, not only here — it is a property of the
   runs, not of the literature. Flagged rather than filed, since amending the canonical record is the operator's
   call.
4. **Finding 5 depends on a community result that checks out.** The glitch cluster is real, its cause is the
   tokeniser/training-corpus mismatch, and the specific tokens are verifiable in `vocab.json` (§2.3, §5.9). The
   flip axis running between most-trained and never-trained embedding regions is a claim about a geometry that
   independent work established.
5. **The Small-vs-Medium divergence is the interesting result, not the embarrassing one.** Universal Neurons
   (§5.5) found that only 1–5% of neurons survive a *seed* change under otherwise identical conditions. Against
   that baseline, the expectation that GPT-2 Medium would reproduce GPT-2 Small's basin structure was never
   well-founded, and its failure to do so is evidence about how little of a small model's organisation is
   architecture-and-data rather than run-specific. The corollary is a concrete experiment this repository is
   uniquely placed to run: **ATR on the five Stanford CRFM GPT-2 Small seeds** (§3.2) would separate
   "GPT-2 Small has five semantic basins" from "this checkpoint has five semantic basins" — the same
   seed-variation control, applied to attractors instead of neurons. Nothing new is needed but the model string.
6. **And the seed control is available on all four cells of this project's 2×2, which neither half of the record
   currently notes.** Five seeds for GPT-2 Small and five for GPT-2 Medium from Stanford (§3.2); nine seeds each
   for `pythia-160m` and `pythia-410m` from EleutherAI. Run across all four, the question stops being "does this
   checkpoint have basins" and becomes "is the basin count a property of the architecture-and-corpus cell" — which
   is the question the refuted founding hypothesis was actually asking. The Pythia half of that resource is
   catalogued in [PYTHIA_INTERPRETABILITY_REVIEW.md](PYTHIA_INTERPRETABILITY_REVIEW.md) §II.3; the GPT-2 half is
   §3.2 here. The two documents were written in parallel and neither found the other's half at the time.
7. **The developmental version of the question is answerable on GPT-2 Small, and only there.** §3.2: ~609
   checkpoints per Stanford Small seed, 10-step resolution through the first 100 steps. "At which step does the
   basin structure appear, and does it appear at the same point across seeds?" is a runnable experiment on the
   exact model this repository's headline result is about. The matching Medium seeds have no checkpoints, so the
   Small-versus-Medium contrast cannot be run developmentally — only the Small side can.

The open anomaly — why GPT-2 Small alone resolves language into semantic basins — now has three candidate framings
drawn from this literature, none tested: that the basins are workspace-like directions and iteration falls into
them in Small but not Medium ([JSPACE_PRIMER.md](JSPACE_PRIMER.md) develops this); that they are superposition
geometry, in which case the count is a property of a dictionary and not of the dynamics (§5.1, §5.5); or that they
are seed-specific, in which case the CRFM sweep in point 5 disposes of the question entirely.

---

## Appendix A — Instruments, one line each

| Instrument | What it does | First/canonical reference |
|:---|:---|:---|
| Logit lens | Unembed intermediate residual streams to read per-layer predictions | nostalgebraist 2020 (§5.2) |
| Tuned lens | Learned affine probe per layer before unembedding; fixes logit-lens brittleness | Belrose et al. 2023 (§5.2) |
| J-lens | Per-layer averaged Jacobian to the final stream, then unembed; logit lens = identity case | Gurnee et al. 2026 (§5.7) |
| Activation patching / causal tracing | Replace an activation, measure output change | Meng et al. 2022 (§5.3); audited in Zhang & Nanda 2024 (§5.2) |
| Path patching | Restrict the intervention to a specified path through the graph | Goldowsky-Dill et al. 2023 (§5.2) |
| Attribution patching (EAP) | Linear approximation to patching; 2 forward + 1 backward pass total | Syed et al. 2023 (§5.6) |
| ACDC | Iterative edge pruning for automated circuit discovery | Conmy et al. 2023 (§5.6) |
| Sparse probing | k-sparse linear probes; sweep k to measure how distributed a feature is | Gurnee et al. 2023 (§5.4) |
| Sparse autoencoder (SAE) | Overcomplete sparse dictionary over activations | Bricken et al. 2023; Cunningham et al. 2023 (§5.5) |
| Transcoder | Sparse wide MLP approximating a dense MLP sublayer; lets circuits pass through MLPs | Dunefsky et al. 2024 (§5.6) |
| Cross-layer transcoder + attribution graph | Features spanning layers; pruned linear-attribution computational graph | Ameisen et al. 2025 (§5.7) |
| Parameter decomposition (APD/SPD) | Decompose weights, not activations, into minimal mechanisms | Braun et al. 2025; Bushnaq et al. 2025 (§5.8) |
| Automated neuron/weight explanation | LLM writes a description, simulates, and is scored on held-out text | Bills et al. 2023; Marin-Llobet & Heimersheim 2026 (§5.6, §5.8) |
| Weight-sparse training | Train for interpretability by zeroing most weights | Gao et al. 2025 (§5.8) |

## Appendix B — Suggested reading order

For someone who wants working knowledge of GPT-2 mechanistic interpretability, in this order:

1. **Framework first.** *A Mathematical Framework* §§ on the residual stream and QK/OV circuits (§5.1). Without
   this the rest is unreadable notation.
2. **One full circuit.** *Interpretability in the Wild* (§5.3), for the method and for the faithfulness/
   completeness/minimality discipline. Then *How does GPT-2 compute greater-than?* (§5.3) for the MLP-centred
   version.
3. **One full component.** *Copy Suppression* (§5.4). The most complete single-component account in the
   literature, and the closest thing to a template for what "understanding a head" means.
4. **The superposition turn.** *Toy Models of Superposition* (§5.1), then *Towards Monosemanticity* (§5.5), then
   immediately *Sparse Autoencoders Do Not Find Canonical Units of Analysis* (§5.5) so that the enthusiasm and
   the correction arrive together.
5. **The scaling limit of per-unit interpretability.** *Language models can explain neurons* (§5.6) and *Universal
   Neurons* (§5.5), read as a pair: the first shows most neurons resist description, the second shows most
   neurons are not even stable across seeds.
6. **Where it went.** *Circuit Tracing* methods and limitations, then *On the Biology of a Large Language Model*
   (§5.7).
7. **What is unsolved.** *Open Problems in Mechanistic Interpretability* (§5.8), and *Weight-sparse transformers*
   (§5.8) as the dissenting bet.
8. **For this repository specifically.** [JSPACE_PRIMER.md](JSPACE_PRIMER.md), then §5.2 and §5.4 of this
   document re-read against [FINDINGS.md](FINDINGS.md).

## Appendix C — Glossary of abbreviations

Every abbreviation used anywhere above, in one place. Each is also written out at its first use in the text; this
table is for when you meet one again fifty pages later.

**Model and architecture**

| Short | In full | What it is |
|:---|:---|:---|
| GPT | Generative Pre-Training | The 2018 paper's name for the recipe; became the model family's name |
| d_model | — | Width of the residual stream (768 in GPT-2 Small) |
| d_head | — | Dimension of a single attention head (64 in every GPT-2 size) |
| XL | Extra large | OpenAI's label for the 1558M checkpoint |
| MLP | Multi-layer perceptron | The two-layer feed-forward block in each transformer block |
| FFN | Feed-forward network | The same thing as an MLP; both terms appear in the literature |
| LN | Layer normalisation | Rescale to zero mean and unit variance, then learned scale and offset |
| pre-LN / post-LN | Pre-/post-normalisation | Whether normalisation happens on the way into a sub-block or out of it |
| GELU | Gaussian Error Linear Unit | GPT-2's activation function; a smoothed ReLU |
| ReLU | Rectified linear unit | Pass positive values, zero out negative ones |
| JumpReLU | — | A ReLU with a learned threshold; stays at exactly zero until its input clears the bar |
| QK circuit | Query–Key | The half of an attention head that decides **where it looks** |
| OV circuit | Output–Value | The half of an attention head that decides **what it writes** |
| Q, K, V | Query, key, value | The three vectors attention is computed from |
| KV cache | Key–value cache | Stored keys and values reused instead of recomputed for each new token |
| BPE | Byte-pair encoding | Tokenisation by repeatedly merging the most frequent adjacent symbol pair |
| BOS / PAD / UNK | Beginning-of-sequence / padding / unknown | Special tokens GPT-2 does **not** have |
| RoPE | Rotary position embedding | Positions applied by rotating query and key vectors (GPT-2 does not use it) |
| ALiBi | Attention with Linear Biases | A distance penalty added to attention scores (not in GPT-2) |
| NoPE | No positional encoding | Relying on the causal mask alone to imply order (not in GPT-2) |
| RMSNorm | Root-mean-square normalisation | LayerNorm without mean-centring or bias (not in GPT-2) |
| SwiGLU / GeGLU | Gated linear unit variants | Feed-forward blocks that multiply two projections together (not in GPT-2) |
| MQA / GQA | Multi-query / grouped-query attention | Heads sharing keys and values, wholly or in groups (not in GPT-2) |
| RLHF | Reinforcement learning from human feedback | The post-training step that turns a text predictor into an assistant |
| LLM | Large language model | The general term for this class of system |
| RNN | Recurrent neural network | Pre-transformer family; processes a sequence one step at a time |
| Mamba / RWKV | — | Two modern recurrent architectures; RWKV is a coined name, not an expansion |
| BERT | Bidirectional Encoder Representations from Transformers | Google's 2018 encoder model |

**Interpretability**

| Short | In full | What it is |
|:---|:---|:---|
| MI | Mechanistic interpretability | Reverse-engineering the internal computations of a network |
| PR | Pull request | A proposed change to this repository, reviewed before it merges |
| BOS | Beginning of sequence | The token some tooling prepends at position 0 — see §2.3, where it matters |
| ATR | Activation Tensor Resonance | This repository's method: iterated reinjection of the final-layer tensor |
| IOI | Indirect object identification | The canonical GPT-2 Small circuit task |
| ICL | In-context learning | Picking up a task from prompt examples, with no weight changes |
| SAE | Sparse autoencoder | Learns an overcomplete sparse dictionary over activations |
| CLT | Cross-layer transcoder | A transcoder whose features write to every later layer |
| ACDC | Automatic Circuit DisCovery | Iterative edge pruning to find a circuit |
| EAP | Edge attribution patching | Linear approximation to patching; two forward passes and one backward |
| APD | Attribution-based Parameter Decomposition | Decomposes weights, not activations, into mechanisms |
| SPD | Stochastic Parameter Decomposition | The scalable successor to APD |
| SHIFT | — | Marks et al.'s name for their feature-ablation technique, not an expansion |
| ROME | Rank-One Model Editing | Edits one fact by a rank-one change to a feed-forward weight |
| J-lens / J-space | Jacobian lens / Jacobian space | Anthropic's 2026 readout, and the verbalizable subspace it finds |
| PCA / ICA | Principal / independent component analysis | Baseline ways of finding directions in activation space |

**Metrics, data and venues**

| Short | In full | What it is |
|:---|:---|:---|
| F1 | — | Harmonic mean of precision and recall; one number balancing the two |
| AUC | Area under the curve | How well a method separates right from wrong across all thresholds |
| ROUGE | — | Standard word-overlap metric for summarisation |
| CBT-CN | Children's Book Test, common-noun split | A cloze benchmark GPT-2 was scored on |
| CoQA | Conversational Question Answering | Reading-comprehension dialogue benchmark |
| LAMBADA | — | Long-range word-prediction benchmark |
| zsRE | Zero-shot relation extraction | Standard fact-editing benchmark |
| DNA | Deoxyribonucleic acid | Appears only as the subject of a discovered feature |
| HTTP | Hypertext Transfer Protocol | Likewise — the format web browsers use to request pages |
| API | Application programming interface | A remote service you send text to and get text back from |
| GB | Gigabytes | WebText is 40 of them |
| CRFM | Center for Research on Foundation Models | Stanford; published the five-seed GPT-2 replications |
| NeurIPS | Conference on Neural Information Processing Systems | Machine learning's largest venue |
| ICLR | International Conference on Learning Representations | With NeurIPS, one of the two main venues |
| EMNLP | Conference on Empirical Methods in Natural Language Processing | The main natural-language-processing venue |
| COLM | Conference on Language Modeling | Newer venue specific to language models |
| TMLR | Transactions on Machine Learning Research | Peer-reviewed journal |

## Sources

**GPT-2 itself**
- Radford, Wu, Child, Luan, Amodei, Sutskever, *Language Models are Unsupervised Multitask Learners* — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- OpenAI, *Better Language Models and Their Implications* (Feb 2019) — https://openai.com/index/better-language-models/
- OpenAI, *GPT-2: 6-month follow-up* (Aug 2019) — https://openai.com/index/gpt-2-6-month-follow-up/
- OpenAI, *GPT-2: 1.5B Release* (Nov 2019) — https://openai.com/index/gpt-2-1-5b-release/
- Solaiman et al., *Release Strategies and the Social Impacts of Language Models* — https://arxiv.org/abs/1908.09203
- Model cards and weights — https://huggingface.co/openai-community/gpt2 · https://huggingface.co/distilbert/distilgpt2
- Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- Brown et al., *Language Models are Few-Shot Learners* — https://arxiv.org/abs/2005.14165
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* — https://arxiv.org/abs/1503.02531

**Anthropic / Transformer Circuits Thread**
- *A Mathematical Framework for Transformer Circuits* — https://transformer-circuits.pub/2021/framework/index.html
- *In-Context Learning and Induction Heads* — https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- *Softmax Linear Units* — https://transformer-circuits.pub/2022/solu/index.html
- *Toy Models of Superposition* — https://transformer-circuits.pub/2022/toy_model/index.html
- *Privileged Bases in the Transformer Residual Stream* — https://transformer-circuits.pub/2023/privileged-basis/index.html
- *Towards Monosemanticity* — https://transformer-circuits.pub/2023/monosemantic-features/index.html
- *Scaling Monosemanticity* — https://transformer-circuits.pub/2024/scaling-monosemanticity/ · https://arxiv.org/abs/2605.29358
- *Circuit Tracing: Revealing Computational Graphs in Language Models* — https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- *On the Biology of a Large Language Model* — https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- *Tracing Attention Computation Through Feature Interactions* — https://transformer-circuits.pub/2025/july-update/index.html
- *When Models Manipulate Manifolds* — https://transformer-circuits.pub/2025/linebreaks/index.html · https://arxiv.org/abs/2601.04480
- *Emergent Introspective Awareness in Large Language Models* — https://transformer-circuits.pub/
- *Verbalizable Representations Form a Global Workspace in Language Models* — https://transformer-circuits.pub/2026/workspace/index.html
- Thread index — https://transformer-circuits.pub/

**Circuits, components, instruments**
- Wang et al., IOI — https://arxiv.org/abs/2211.00593
- Hanna, Liu, Variengien, greater-than — https://arxiv.org/abs/2305.00586
- Merullo, Eickhoff, Pavlick, circuit component reuse — https://arxiv.org/abs/2310.08744
- Meng et al., ROME — https://arxiv.org/abs/2202.05262
- McDougall et al., copy suppression — https://arxiv.org/abs/2310.04625
- Gould et al., successor heads — https://arxiv.org/abs/2312.09230
- Gurnee et al., universal neurons — https://arxiv.org/abs/2401.12181
- Gurnee et al., sparse probing — https://arxiv.org/abs/2305.01610
- Tigges et al., *Language Models Linearly Represent Sentiment* — https://aclanthology.org/2024.blackboxnlp-1.5/ (preprint, under the earlier title *Linear Representations of Sentiment in Large Language Models*: https://arxiv.org/abs/2310.15154)
- Geva et al., FFN key-value memories — https://arxiv.org/abs/2012.14913
- Geva et al., promoting concepts — https://arxiv.org/abs/2203.14680
- nostalgebraist, logit lens — https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Belrose et al., tuned lens — https://arxiv.org/abs/2303.08112
- Goldowsky-Dill et al., path patching — https://arxiv.org/abs/2304.05969
- Zhang & Nanda, patching best practices — https://arxiv.org/abs/2309.16042
- Heimersheim & Nanda, how to use activation patching — https://arxiv.org/abs/2404.15255
- Xiao et al., attention sinks — https://arxiv.org/abs/2309.17453
- Sun et al., massive activations — https://arxiv.org/abs/2402.17762
- Rumbelow & Watkins, SolidGoldMagikarp — https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation
- Hendel, Geva, Globerson, task vectors — https://arxiv.org/abs/2310.15916
- Li et al., inference-time intervention — https://arxiv.org/abs/2306.03341
- Singh et al., induction-head formation — https://arxiv.org/abs/2404.07129
- Paulo, Marshall, Belrose, RNN transfer — https://arxiv.org/abs/2404.05971

**Features, automation, frontier**
- Cunningham et al., SAEs find interpretable features — https://arxiv.org/abs/2309.08600
- Makelov, Lange, Nanda, principled SAE evaluation — https://arxiv.org/abs/2405.08366
- Leask et al., SAEs are not canonical units — https://arxiv.org/abs/2502.04878
- Engels et al., multi-dimensional features — https://arxiv.org/abs/2405.14860
- Bills et al., neuron explanations — https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html · https://openai.com/index/language-models-can-explain-neurons-in-language-models/
- Conmy et al., ACDC — https://arxiv.org/abs/2304.14997
- Syed, Rager, Conmy, attribution patching — https://arxiv.org/abs/2310.10348
- O'Neill & Bui, SAE circuit identification — https://arxiv.org/abs/2405.12522
- Dunefsky, Chlenski, Nanda, transcoders — https://arxiv.org/abs/2406.11944
- Marks et al., sparse feature circuits — https://arxiv.org/abs/2403.19647
- Gao et al., scaling and evaluating SAEs — https://arxiv.org/abs/2406.04093
- Sharkey et al., *Open Problems in Mechanistic Interpretability*, TMLR 2025 — https://openreview.net/forum?id=91H76m9Z94 (preprint: https://arxiv.org/abs/2501.16496)
- Braun et al., APD — https://arxiv.org/abs/2501.14926
- Bushnaq, Braun, Sharkey, SPD — https://arxiv.org/abs/2506.20790
- Gao et al., weight-sparse transformers — https://arxiv.org/abs/2511.13653
- Marin-Llobet & Heimersheim, individual parameters — https://arxiv.org/abs/2607.02964
- Lieberum et al., Gemma Scope (SAE suites, for contrast with GPT-2's) — https://arxiv.org/abs/2408.05147

**Tooling**
- TransformerLens — https://github.com/TransformerLensOrg/TransformerLens
- SAELens — https://github.com/decoderesearch/SAELens
- Neuronpedia — https://www.neuronpedia.org/ · https://github.com/hijohnnylin/neuronpedia
- Jacobian lens (J-lens) reference implementation — https://github.com/anthropics/jacobian-lens
</content>
