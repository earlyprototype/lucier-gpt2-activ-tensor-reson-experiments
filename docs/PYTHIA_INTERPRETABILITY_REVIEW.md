# Pythia: development, structure, versions, and the mechanistic interpretability record

A review of EleutherAI's Pythia suite — why it was built, how it is put together, how its
versions differ, and what mechanistic interpretability (MI) work has been done on it — written
for operators of this repository, which runs its cross-model controls on `pythia-160m` and
`pythia-410m`.

**On the name.** The request named "Pithia". No model suite by that spelling exists. The subject
is **Pythia**, EleutherAI's scaling suite (arXiv 2304.01373), named for the oracle at Delphi. The
spelling is worth pinning because EleutherAI's automated-interpretability library is called
**Delphi** and its SAE library **Sparsify** — the Delphic naming runs through the whole EleutherAI
interpretability stack, and searches on the misspelling return nothing.

---

## How to read the citations

Following `PRIOR_WORK.md`. Every work carries a **source-class tag** recording its formal status
as listed by its own venue or host — formal status is no guarantee of correctness, and replication
is the stronger signal, noted where known. Citations marked with an **asterisk (\*)** were *not*
checked against primary sources and rest on secondary descriptions (title, venue, abstract-level
claims, search-index summaries); unmarked citations were checked against primary material — the
paper's full text, its HTML rendering, or its primary artifacts read directly.

Primary material read directly for this review: the Pythia paper (ar5iv HTML), the Pythia
repository README, the Anthropic Circuits Updates — September 2024 page, the Transformer Circuits
Thread publication index, and the Cunningham et al. abstract page. Everything else is
secondary-sourced and asterisked. **This matters for the numbers below**: hyperparameter tables and
version details are verified; the per-paper result figures in Part III are, with the exceptions
noted, taken from abstracts and summaries and should be checked against the papers before being
quoted in a write-up.

---

# Part I — What Pythia is, and the problem it was built to solve

## I.1 The motivating gap

By late 2022 the most consequential result in mechanistic interpretability was developmental:
Olsson et al., *In-context Learning and Induction Heads* (Anthropic, Transformer Circuits Thread,
March 2022)\* [published research report, not journal-reviewed]
(https://arxiv.org/abs/2209.11895) showed that in-context learning ability appears in a **phase
change** early in training, visible as a bump in the loss curve, and that induction heads —
attention heads implementing `[A][B] … [A] → [B]` — form at the same moment. The causal argument
rested on perturbing the architecture so the bump moved, and observing that induction-head
formation moved with it.

That argument requires **dense checkpoints across training at multiple scales**. Anthropic had
them; nobody outside did. The public suites of the day were unusable for the purpose:

- **GPT-3** — no weights.
- **OPT** — checkpoints exist, but data ordering was not held constant across sizes.
- **BLOOM** — multilingual, which confounds term-frequency and memorization analyses.
- **GPT-Neo / GPT-J / GPT-NeoX-20B** — final weights only, and not a controlled scaling ladder.

Pythia was built to close exactly this gap. Its design is best read as a list of *constraints
accepted at a cost to benchmark performance* in order to make developmental and cross-scale
analysis valid.

## I.2 The controlled-suite design

Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*
(ICML 2023) [peer-reviewed] — https://arxiv.org/abs/2304.01373, read via
https://ar5iv.labs.arxiv.org/html/2304.01373.

The suite is **16 models**: 8 sizes × {Pile, deduplicated Pile}. Four properties do the work:

1. **Identical data, identical order, across every size.** Not merely the same corpus — the same
   sequence of batches. A 70M and a 12B model see token *n* at step *n* together. This is what
   makes "the 2.8B learns this at step 65,000 and the 410M never does" a statement about scale
   rather than about data luck.
2. **154 checkpoints per model.** Prior public suites offered ≤30. Schedule: `step0`
   (initialisation), then log-spaced `step{1,2,4,8,…,512}` (10 checkpoints), then every 1,000
   steps from `step1000` to `step143000` (143 checkpoints). 1 + 10 + 143 = 154. The dense early
   log-spacing is deliberate: induction-head formation and other early phase changes happen inside
   the first few hundred steps and are invisible on a linear grid.
3. **A reconstructible dataloader.** `utils/batch_viewer.py` in the Pythia repository lets a
   researcher recover *exactly which sequences were in batch k*. This converts correlational claims
   ("models are better on frequent terms") into checkable ones ("this term appeared 412 times
   before step 65,000").
4. **Uniform architecture and batch size across scale.** All models use batch size 1024 sequences
   × 2048 tokens = **2,097,152 tokens per step**, including the 70M. This runs counter to the
   received wisdom that small models need small batches; the paper reports a 4–8× wall-clock
   speedup for the small models with no convergence problem, and — more importantly for MI — it
   removes batch size as a confound when comparing across the ladder.

## I.3 Architecture

Pythia is a GPT-NeoX-family, fully dense, decoder-only transformer. The choices that matter for
interpretability work:

| Choice | Detail | Why it matters for MI |
|---|---|---|
| **Parallel attention + MLP** | Attention and feed-forward sublayers computed in parallel from the same layernormed residual (as GPT-J/PaLM), not sequentially | Attention and MLP contributions to a layer's residual write are *independent*, which simplifies path decomposition — but it also means an MLP cannot read that layer's attention output, changing which circuits are even expressible. Conventional wisdom said this regresses quality below ~6B; Pythia's results contradict that. |
| **Untied input/output embeddings** | `W_E` and `W_U` are separate matrices | **Directly motivated by interpretability.** In GPT-2 the tied weights make embedding space and unembedding space the same space by identity — a fact that has caused real confusion in this repo's sibling work (see the `W_U`/`W_E` collision recorded on the peer board). In Pythia they are genuinely distinct, so "this direction is in unembedding space" is a substantive claim. |
| **Rotary positional embeddings** | Su et al. (2021), applied to the first 25% of head dimensions | Position is applied inside the QK circuit rather than added to the residual stream, so there is no additive positional component in the residual to decompose away. Relevant to this repo's finding 2 (position-uniformity of ATR states). |
| **Fully dense attention** | No alternating sparse/dense layers as in GPT-3 | Every layer has the same structure; no layer-index confound. |
| **Flash Attention** | Used in the current (v1) suite | Numerically not bit-identical to a naive attention implementation. Worth knowing when chasing small reproducibility discrepancies. |
| **Tokenizer** | GPT-NeoX BPE trained on the Pile, vocab 50,432 (padded), sequence length 2048 | Different from GPT-2's tokenizer. Cross-model token-level comparisons with GPT-2 are not one-to-one. |

## I.4 Hyperparameters

From the repository README (read directly), corroborated by the paper's Table 1:

| Size | Non-emb. params | Layers | d_model | Heads | d_head | Batch (tokens) | LR |
|---|---|---|---|---|---|---|---|
| 14M | — | 6 | 128 | 4 | 32 | 2M | 1.0e-3 |
| 31M | — | 6 | 256 | 8 | 32 | 2M | 1.0e-3 |
| 70M | 18.9M | 6 | 512 | 8 | 64 | 2M | 1.0e-3 |
| 160M | 85.1M | 12 | 768 | 12 | 64 | 2M | 6.0e-4 |
| **410M** | 302.3M | **24** | **1024** | **16** | 64 | 2M | 3.0e-4 |
| 1B | 805.7M | 16 | 2048 | 8 | **256** | 2M | 3.0e-4 |
| 1.4B | 1.2B | 24 | 2048 | 16 | 128 | 2M | 2.0e-4 |
| 2.8B | 2.5B | 32 | 2560 | 32 | 80 | 2M | 1.6e-4 |
| 6.9B | 6.4B | 32 | 4096 | 32 | 128 | 2M | 1.2e-4 |
| 12B | 11.3B | 36 | 5120 | 40 | 128 | 2M | 1.2e-4 |

Two anomalies in that table are load-bearing and easy to miss:

- **`pythia-1b` is not on the ladder.** It has 16 layers where 410M has 24, and 8 heads of
  dimension **256** where every other model uses 64–128. It is shaped unlike its neighbours.
  Any "trend across scale" plot that includes 1B is measuring two things at once. Several
  published Pythia scaling figures quietly drop it; if yours doesn't, say so.
- **The two models this repository runs are shape-matched to their GPT-2 counterparts.**
  `pythia-160m` is 12 layers × d_model 768 × 12 heads — GPT-2 Small exactly. `pythia-410m` is
  24 layers × d_model 1024 × 16 heads — GPT-2 Medium exactly. The 2×2 design in
  `CROSS_MODEL_RUN_PLAN.md` is therefore a genuine factorial and a better-controlled one than it
  may have been designed to be: layer count, width and head count are held *identical* down each
  column, and what varies is training corpus (WebText vs Pile), tokenizer, positional scheme
  (learned absolute vs rotary), sublayer arrangement (sequential vs parallel), and embedding tying
  (tied vs untied). Any cross-model difference is attributable to that list and not to capacity.

Training: Adam (β₁=0.9, β₂=0.95, ε=1e-8, weight decay 0.01), ~300B tokens (143,000 × 2,097,152 ≈
299.9B), matching GPT-3 and OPT's token budget. Deduplicated Pile is ~207B tokens, so the deduped
models see roughly **1.5 epochs**. Hardware: A100-40GB, 32–256 GPUs per model, ~136,070 GPU-hours
for one complete suite.

---

# Part II — Versions: what differs, and which one you are actually running

This is the part most likely to silently corrupt a result, because the Hugging Face IDs are
similar and the defaults are not obvious.

## II.1 The v0 → v1 retrain

The originally released suite had **inconsistencies across runs** that were fixed by retraining
the whole thing. The originals survive as `EleutherAI/pythia-{160m,410m,1.4b}-v0`. What changed:

| | v0 (original) | v1 (current, unsuffixed) |
|---|---|---|
| Batch size | 4M tokens for 160M/410M/1.4B | **2M tokens uniformly, all sizes** |
| Steps | 71,500 | **143,000** |
| Checkpoint interval | every 500 steps | every 1,000 steps (plus log-spaced early) |
| LR schedule | differed by model size | **uniform: decay to 0.1× max LR for every size** |
| Flash Attention | no | yes |
| Step labels on HF | renamed for consistency with the 2M-batch numbering | native |

The LR-schedule fix is the one that matters most for scaling claims: in v0, a difference between
the 410M and the 2.8B could be a difference in learning-rate schedule rather than in scale. The
v0 models are still useful — **as an ablation**, since they are the same architecture and data at
a different batch size and schedule.

**Practical consequence:** a "Pythia" result published in the first half of 2023 may be a v0
result. If a paper reports 71,500 steps or 4M-token batches, it is v0.

## II.2 The January 2023 renaming

On **20 January 2023** the suite was renamed so that parameter counts **include embedding and
unembedding parameters**, aligning with other suites and better reflecting on-device memory. One
model was additionally corrected because a training typo had made it smaller than intended.

Before the rename, models carried GPT-3/OPT-equivalent names keyed to non-embedding counts. Older
literature therefore refers to Pythia models by names that no longer exist. The clearest surviving
example: Michaud et al. (2023) describe evaluating Pythia models "ranging from 19m to 6.4b
non-embedding parameters"\* — those are today's **70M** and **6.9B**.

The exact old→new mapping table was not recoverable verbatim from the sources read here, and is
**not reproduced** rather than guessed. Use the non-embedding column in §I.4 as the bridge: a
pre-2023 Pythia name is almost always the non-embedding count or its GPT-3-equivalent round number.

## II.3 The four axes of variation

Beyond v0/v1, four independent axes exist. All are on Hugging Face under `EleutherAI/`.

**1. Deduplication.** `pythia-{size}` (Pile, ~300B tokens, ~1 epoch) vs `pythia-{size}-deduped`
(deduplicated Pile, 207B tokens, ~1.5 epochs). Available for 70M and up. The paper's own finding
is that **deduplication shows no clear benefit** to downstream performance — which contradicted
some prior work and aligned with the GPT-NeoX-20B result. For MI purposes the deduped models are
the correct choice when memorization is the object of study, and an arbitrary choice otherwise;
what you must not do is mix them within one comparison.

**2. Random seeds.** Seeds 1–9 exist for **14M, 31M, 70M, 160M, and 410M** (standard, non-deduped
only), named `pythia-{size}-seed{n}`. This is the suite's **universality instrument**: it answers
"does this circuit appear in *a* model, or in *this* model?" — the single most common unanswered
objection to a circuit-level finding. It is also, for this repository, the most obviously
underused resource: the question of whether the five ATR basins are a property of the weight
geometry or of one particular training run is directly testable on nine 410M seeds without
retraining anything.

**3. Extra small sizes.** **14M and 31M** were added after the paper, explicitly "at the request of
alignment researchers interested in scaling sparse autoencoders." They are not in the paper's
tables. They are the cheapest place to run a full SAE pipeline end to end.

**4. The intervention models.** The gender-bias case study retrained 70M–6.9B from a mid-training
checkpoint with masculine pronouns swapped for feminine in the final 7% and 21% of training. These
are counterfactual-data models — a genuinely rare artifact class, since almost nobody can afford to
train the counterfactual.

## II.4 Known defects, stated plainly

Two are recorded in the repository README and should be carried into any cross-scale claim:

- **6.9B and 12B accidentally used a different initialisation**, because the init value was not
  specified in their config files. The two largest models are therefore not initialisation-matched
  to the rest of the ladder.
- **One model was smaller than intended** due to a training typo, corrected at the January 2023
  renaming.

Neither invalidates the suite. Both mean that a clean monotone trend across all eight sizes should
be treated with suspicion at the top end, and that 1B (§I.4) should be treated with suspicion in
the middle.

## II.5 What this repository is running

The notebooks call `HookedTransformer.from_pretrained("pythia-410m")` and `"pythia-160m"`. That
resolves through TransformerLens to the **v1, non-deduped, final-checkpoint (`step143000`)**
models. Three consequences worth recording in the results summaries:

1. **No version is pinned in the code.** The alias is stable in practice, but the run provenance
   currently rests on TransformerLens's mapping rather than on an explicit revision. Pinning
   `revision="step143000"` costs nothing and makes the record self-describing.
2. **TransformerLens preprocesses weights by default** — LayerNorm folding, centering of writing
   weights, and (unless disabled) centering of the unembedding. Since the readout-confidence metrics
   in `readout_guardrails.ipynb` are logit-derived, it is worth recording which preprocessing was
   active. On the arithmetic: centering the unembedding shifts all logits by a per-position
   constant, which leaves the top-1 token, the top-1-vs-top-2 **margin**, and the softmax entropy
   unchanged, while changing raw logit values. So the guardrail metrics should be invariant to it —
   but that is a statement worth *verifying rather than assuming*, and it is cheap to verify.
3. **Pythia's untied embeddings mean the `W_E`/`W_U` identity that holds in GPT-2 does not hold
   here.** Any diagnostic ported from the GPT-2 side of the 2×2 that assumed the two spaces are the
   same must be re-derived, not re-run.

---

# Part III — The mechanistic interpretability record

## III.0 Why Pythia became the default MI substrate

Three properties, in descending order of importance to the field:

- **Checkpoints** make *developmental* claims testable — when does a circuit form, and does it
  form in stages? No other open suite of this size offered 154.
- **The dataloader index** makes *data-attribution* claims testable — this behaviour appeared at
  step *k*; what was in the data before step *k*?
- **The scale ladder with everything else held fixed** makes *scaling* claims testable — the same
  probe, the same corpus, the same order, eight sizes.

GPT-2 remains the substrate for single-model circuit archaeology (IOI, greater-than, copy
suppression) because it is small, ubiquitous, and heavily tooled. Pythia is the substrate for
everything with a *training-time* or *cross-scale* axis. That division of labour is visible in the
summaries below, and it is exactly the division this repository straddles.

---

## III.1 Drafted summaries

Each entry: citation, source class, what was done, on which models, the result, and what it is
worth. **The result figures are abstract-level unless the entry says otherwise.**

---

### 1. Biderman et al. (2023) — *Pythia: A Suite for Analyzing LLMs Across Training and Scaling*

**ICML 2023** [peer-reviewed] · https://arxiv.org/abs/2304.01373 · *read directly (ar5iv HTML)*

The suite paper, and three case studies chosen to demonstrate what the suite makes possible.

**Case study 1 — memorization is a Poisson process.** Using the (32,32)-memorization criterion,
the authors ask whether *where* a sequence sits in training order affects whether it is memorized.
Q-Q plots show memorized sequences distributed as a Poisson point process across training steps —
i.e. **training order has essentially no effect**. The practical corollary is negative and useful:
you cannot protect sensitive data by scheduling it early or late. Detection via partial checkpoints
remains the only lever.

**Case study 2 — term frequency effects emerge at a specific point.** Reproducing Razeghi et al.
and Kandpal et al., the authors find few-shot accuracy correlates with pretraining term frequency —
and then locate the emergence: a **phase transition around step 65,000 (≈45% of training)**, in
models **≥2.8B only**. Smaller models and earlier checkpoints show no such correlation. For
arithmetic, the gap between top-decile and bottom-decile operand frequency widens over training,
most in the largest models.

**Case study 3 — counterfactual data intervention on gender bias.** Retraining 70M–6.9B from a
checkpoint with pronouns swapped in the last 7% / 21% of training reduces stereotypical bias on
WinoBias and CrowS-Pairs at every scale, with **larger models showing stronger intervention
effects**, while LAMBADA perplexity stays roughly flat. Bias mitigation without catastrophic
forgetting, demonstrated causally rather than correlationally.

**What it is worth.** The case studies are behavioural, not mechanistic — and the paper contains
**no sustained induction-head or circuit analysis**, despite the untied-embedding choice being
justified by interpretability. That gap is the honest characterisation: Pythia is *infrastructure
for* MI rather than MI itself, and the mechanistic payoff arrives in the papers below.

**Relevance here.** The 65,000-step phase transition is a template for the question this repo
should be asking of ATR: *if* attractor structure is a real property, at which checkpoint does it
appear, and does it appear at the same relative point across sizes?

---

### 2. Gurnee, Nanda, Pauly, Harvey, Troitskii, Bertsimas (2023) — *Finding Neurons in a Haystack: Case Studies with Sparse Probing*

**TMLR 2023** [peer-reviewed]\* · https://arxiv.org/abs/2305.01610

Trains **k-sparse linear probes** on internal activations to predict whether a feature is present
in the input, sweeping *k* to measure how many neurons a feature is spread across. Over **100
features in 10 categories**, across **7 models spanning two orders of magnitude** (up to 6.9B) —
the Pythia suite.

**Findings.** Three layer-dependent regimes: **early layers** represent many features as sparse
combinations of polysemantic neurons, each firing for large collections of unrelated n-grams and
local patterns; **middle layers** contain apparently dedicated neurons for higher-level contextual
features; **increasing scale raises representational sparsity on average**, but with several
distinct scaling dynamics rather than one trend.

**What it is worth.** This is generally credited as the **first demonstration of neuron
superposition "in the wild"** in real LLMs — moving Anthropic's *Toy Models of Superposition*
hypothesis from constructed toy settings to trained language models. It is the empirical bridge
between the superposition theory and everything that followed in dictionary learning.

**Limitations to carry.** A sparse probe finding a feature is decodable does not establish the
model *uses* it. The paper is a representational result, not a causal one.

---

### 3. Cunningham, Ewart, Riggs, Huben, Sharkey (2023) — *Sparse Autoencoders Find Highly Interpretable Features in Language Models*

**ICLR 2024** [peer-reviewed] · https://arxiv.org/abs/2309.08600 · *abstract read directly; result figures secondary\**

Trains **sparse autoencoders** — overcomplete dictionaries with an L1 sparsity penalty — on the
internal activations of **Pythia-70M and Pythia-410M** (residual streams of width 512 and 1024
respectively), and on MLP sites. The hypothesis under test is Anthropic's: polysemanticity is
caused by superposition, so an overcomplete sparse basis should recover monosemantic directions
that the neuron basis cannot.

**Findings.** SAE features score higher on **automated interpretability** than PCA, ICA, or the
neuron basis. In a causal test on the **indirect object identification** task, the SAE
decomposition localises the features responsible for counterfactual behaviour **more finely than
previous decompositions** — the causal claim, not just a correlational one. The paper also reports
the reconstruction cost (replacing layer-2 Pythia-70M residual activations with SAE
reconstructions raises perplexity\*).

**What it is worth.** Released within weeks of Anthropic's *Towards Monosemanticity*, and the
open-model counterpart to it: Anthropic demonstrated dictionary learning on a proprietary one-layer
transformer; Cunningham et al. demonstrated it on a real, public, multi-layer model anyone could
download. The two together are why SAEs became the field's dominant method for two years.

**Limitations to carry.** Automated interpretability scores are a weak metric — later work found
they fail to distinguish trained from randomly-initialised transformers in some settings\*. The
reconstruction-perplexity gap is the standing objection to all SAE work: the dictionary is not the
model.

---

### 4. Gould, Ong, Ogden, Conmy (2024) — *Successor Heads: Recurring, Interpretable Attention Heads In The Wild*

**ICLR 2024** [peer-reviewed]\* · https://arxiv.org/abs/2312.09230

Identifies **successor heads**: attention heads whose OV circuit performs *incrementation* over
ordinal sequences. Input the representation of `Monday`, `first`, `January`, or `one`; the head's
output raises the likelihood of `Tuesday`, `second`, `February`, `two`.

**Findings.** Successor heads are found across **GPT-2, Pythia (including 410M), and Llama-2**,
from models as small as **31M parameters up to 12B** — the same functional head class recurring
across architectures, tokenizers, and three orders of magnitude of scale. The mechanism rests on a
set of **"mod-10 features"** in the ordinal representations that the OV circuit acts on.

**What it is worth.** One of the strongest **universality** results in MI: not "a circuit exists in
this model" but "this circuit class recurs wherever we look." That is the claim MI needs in order
to be a science rather than a collection of case studies.

**The Anthropic connection — and it is a real one.** Anthropic's **Circuits Updates — September
2024** (https://transformer-circuits.pub/2024/september-update/index.html, *read directly*)
[published research report, not journal-reviewed] contains a section that **explicitly replicates
Gould et al. (2023)** on an internal 18-layer Anthropic model, using four complementary analyses
including a novel ICA-based method on the OV circuit. Their top-scoring successor head maps
**about 80% of ordinal tokens** to their successors correctly and shows a strong super-diagonal
component. This is the clearest documented instance of a Pythia-established mechanistic result
being carried *into* Anthropic and confirmed on a proprietary model — universality tested across
the open/closed boundary.

---

### 5. Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, Nanda (2024) — *Confidence Regulation Neurons in Language Models*

**NeurIPS 2024** [peer-reviewed]\* · https://arxiv.org/abs/2406.16254

Asks how models represent and regulate **uncertainty** in the next-token distribution, via two
neuron classes.

**Entropy neurons.** Characterised by unusually high weight norm, they write into the
**unembedding null space** — affecting the residual stream *norm*, and hence the final LayerNorm
scale, and hence the logit scale — with minimal direct effect on the logits themselves. Net effect:
they scale logits down, raising entropy, without changing what the model believes is most likely.
Observed across models up to **7B**, including Pythia.

**Token frequency neurons.** Boost or suppress each token's logit in proportion to its **log
frequency**, shifting the output distribution toward or away from the unigram prior.

**The unifying observation.** Output entropy is negatively correlated with KL divergence from the
empirical token-frequency distribution: as a model becomes less confident, its output distribution
falls back toward the unigram distribution.

**What it is worth, and why it is the most operationally relevant entry here.** This repository
asked whether Pythia-410m's apparent attractor *fragmentation* was genuine structure or **readout
ambiguity** — token flicker over a converged tensor — and closed the question on the intrinsic side
(`SCALING_ARTEFACT_ANALYSIS.md`, closing judgement 2026-07-10). Stolfo et al. supply the mechanism
that made the guiding principle worth adopting in the first place: a dedicated set of neurons
exists whose whole function is to rescale logits *without changing the argmax*, modulating readout
confidence independently of represented content. That is mechanistic vindication of the
`readout_guardrails.ipynb` design — margin and entropy must be logged alongside token identity
because the two are decoupled **by construction in the model**, not merely as a measurement
nuisance.

The forward-looking item: entropy neurons act **through the residual-stream norm**, via the final
LayerNorm scale. ATR's per-iteration L2 rescale sets that norm by fiat every iteration.
`SCALING_ARTEFACT_ANALYSIS.md` §1.1 rules the rescale inert for the *forward map* — correctly, since
layer 0 applies LayerNorm — but the entropy-neuron channel operates at the *other* end of the
stack, after `ln_final`, where no such invariance argument applies. Whether the rescale interacts
with confidence readout at the terminal LayerNorm is a distinct question from the one §1.1
answers, and it is cheap to check: log pre-rescale residual norm per iteration alongside margin and
entropy.

---

### 6. Feng & Steinhardt (2024) — *How do Language Models Bind Entities in Context?*

**ICLR 2024** [peer-reviewed]\* · https://arxiv.org/abs/2310.17191

Attacks the **binding problem**: given "a green square and a blue circle," how does a model keep
green attached to square?

**Findings.** Via causal interventions, the authors identify a **binding ID mechanism**: activations
carry **binding ID vectors** attached to entities and to attributes, and the model resolves binding
by matching them. Binding IDs occupy a **continuous subspace** in which distance between two IDs
tracks how discriminable they are. The mechanism appears **in every sufficiently large model of
both the Pythia and LLaMA families** — the cross-family generality is the paper's strongest claim.

**What it is worth.** One of the few results showing a genuinely **symbolic-looking** structure —
a variable-binding scheme — implemented in a linear-algebraic substrate, and one of the cleanest
uses of Pythia's scale ladder to establish an *emergence threshold* ("sufficiently large").

---

### 7. Tigges, Hollinsworth, Geiger, Nanda (2024) — *Language Models Linearly Represent Sentiment*

**BlackboxNLP 2024** [peer-reviewed]\* · https://arxiv.org/abs/2310.15154

Finds that **sentiment is represented linearly** — a single direction in activation space, positive
at one extreme and negative at the other, capturing the feature across a range of tasks — in GPT-2
and Pythia models. The direction is isolated by causal intervention and shown causal on both toy
tasks and Stanford Sentiment Treebank.

**The summarization motif.** The paper's most interesting finding is positional: sentiment is *not*
carried only on emotionally charged words. It is additionally **summarized at intermediate
positions with no inherent sentiment** — punctuation, names, separators. Information is aggregated
at syntactic waypoints and read from there.

**Relevance here.** The summarization motif is a *positional* claim about where information lives
in the residual stream, and it is the natural published comparison for this repo's **finding 2**
(ATR states become position-uniform early in the loop). Summarization says sentiment concentrates
at particular positions; ATR position-uniformity says iterated re-injection destroys positional
differentiation entirely. Whether the latter is the former's endpoint under iteration is an open,
testable question.

---

### 8. Nanda, Rajamanoharan, Kramár, Shah (2023) — *Fact Finding* (four-post sequence)

**AI Alignment Forum** [community post, unreviewed]\* ·
https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall

An attempt to reverse-engineer **factual recall** in **Pythia-2.8B**, using the athlete→sport task
(which of three sports does this athlete play).

**Findings.** Early MLPs implement something like a **lookup table**, but the authors' headline
conclusion is a reframing: early MLPs act as a **"multi-token embedding"** — their role is to select
the right *unit of analysis* from the last few tokens (assembling a multi-token entity name into a
single usable representation) rather than to store facts in a clean key-value form. The sequence is
notable for reporting a **partial negative result**: the clean lookup-table picture inherited from
the ROME/knowledge-editing literature did not survive contact with the model.

**What it is worth.** A rare, well-documented account of a circuit-analysis project that did *not*
resolve cleanly, published as such. Unreviewed, and the authors frame their conclusion as a best
guess — treat the "multi-token embedding" framing as a hypothesis with supporting evidence, not a
settled mechanism.

---

### 9. Michaud, Liu, Girit, Tegmark (2023) — *The Quantization Model of Neural Scaling*

**NeurIPS 2023** [peer-reviewed]\* · https://arxiv.org/abs/2303.13506

Proposes that network knowledge and skills are **quantized into discrete chunks ("quanta")**, learned
in order of decreasing use frequency. If quanta frequencies follow a power law, both the smooth
power-law loss curve *and* the sudden emergence of individual capabilities follow — emergence and
smooth scaling as two views of the same process.

**Empirically**, on the Pythia suite (models described in the pre-2023 naming as **19M to 6.4B
non-embedding parameters**, i.e. today's 70M–6.9B), the authors **auto-discover diverse model
capabilities** by clustering on the internal structure of the models, and find tentative evidence
that their use frequencies follow the predicted power law.

**What it is worth.** The most serious attempt to connect *mechanistic* decomposition to *scaling
laws* — to explain why the loss curve has the shape it does in terms of what the model contains. The
empirical support is explicitly tentative, and the quanta-discovery procedure is the weakest link.

---

### 10. Biderman, Prashanth, Sutawika, Schoelkopf, Anthony, Purohit, Raff (2023) — *Emergent and Predictable Memorization in Large Language Models*

**NeurIPS 2023** [peer-reviewed]\* · https://arxiv.org/abs/2304.11158

Asks a practical question: can you predict **which sequences a large model will memorize** before
paying to train it, by extrapolating from cheaper runs? Measures memorization across the Pythia
suite and fits scaling laws for forecasting it.

**Findings.** **Intermediate checkpoints of the target model are better predictors than smaller
fully-trained models.** The compute-optimal predictor changes with budget: in low-compute regimes
larger models predict better; from ~1% of the full budget, equicompute models perform the same
regardless of parameter count; from ~10%, the *smallest* model trained for that budget is the best
predictor. The authors also report evidence of emergent or semi-emergent memorization behaviour
with scale.

**What it is worth.** A direct use of the checkpoint infrastructure that no other public suite
supports, and a result with an operational consequence: run a small trial, forecast, decide. Its
negative implication pairs with the Pythia paper's Poisson finding — you can *predict* memorization
but not *prevent* it by reordering.

---

### 11. Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman, Steinhardt (2023) — *Eliciting Latent Predictions from Transformers with the Tuned Lens*

**preprint / widely used** [preprint, unreviewed]\* · https://arxiv.org/abs/2303.08112

Treats the transformer as **iterative inference** and trains an **affine probe per block** on a
frozen model, decoding every hidden state into a vocabulary distribution. This is a repair of the
**logit lens**, which applies the unembedding directly to intermediate residuals and is known to be
brittle — failing entirely on some models.

**Findings.** Tested on autoregressive models up to **20B** (the Pythia suite and GPT-NeoX-20B), the
tuned lens is **more predictive, more reliable, and less biased** than the logit lens. Causal
experiments indicate the lens uses **similar features to the model itself** rather than inventing
its own. The trajectory of latent predictions across layers can detect adversarial/malicious inputs
with high accuracy.

**Relevance here, and it is direct.** This repository's finding 6 is that ATR motion is *near-
invisible to the unembedding and to lens instruments* — it lives outside the verbalizable subspace.
That is a claim about a class of instruments, and its strength depends on which lens was used. The
tuned lens exists precisely because the logit lens is brittle in a model-specific way, and it is
**trained on Pythia**, with public weights. A finding-6 replication under the tuned lens is a cheap,
high-value control: if the motion is invisible to a *trained* decoder as well as an untrained one,
the "outside the verbalizable subspace" claim is much stronger. If it is not, the claim was about
the logit lens.

---

### 12. Sharkey, Chughtai, Batson, Lindsey, Wu, Bushnaq, Goldowsky-Dill, Heimersheim, Ortega, Bloom, **Biderman**, Garriga-Alonso, Conmy, Nanda, Rumbelow, Wattenberg, Schoots, Miller, Michaud, Casper, Tegmark, Saunders, Bau, Todd, Geiger, Geva, Hoogland, Murfet, McGrath (2025) — *Open Problems in Mechanistic Interpretability*

**TMLR, September 2025** [peer-reviewed]\* · https://arxiv.org/abs/2501.16496

A 29-author, 89-page field survey of what MI has not solved. Affiliations span **Anthropic**
(Joshua Batson, Jack Lindsey, Jeff Wu), **EleutherAI** (Stella Biderman), Google DeepMind, Apollo
Research, MIT, Northeastern, Tel Aviv, Timaeus, FAR AI, Goodfire and others.

**What it is worth here.** It is the one document in this review **co-authored by Anthropic
interpretability staff and Pythia's lead author**, and it is the correct citation for the claim
that the Pythia-based open-model tradition and the Anthropic closed-model tradition are one field
rather than two. Read it for the standing problem list — the limits of SAEs, the absence of
validated evaluation for interpretability claims, and the gap between feature-level description and
mechanistic explanation — all of which apply to entries 2, 3 and 5 above.

---

### 13. *Hidden Dynamics of Massive Activations in Transformer Training* (2025)

**preprint** [preprint, unreviewed]\* · https://arxiv.org/abs/2508.03616

Studies how **massive activations** — the extreme-magnitude residual-stream coordinates associated
with **attention sinks** — develop *over training*, using the Pythia suite (described as 9
decoder-only models, 14M–12B, 150+ checkpoints each). Proposes a mathematical framework for their
emergence, and characterises how the trajectories depend on scale.

**What it is worth.** Attention sinks and massive activations are the best-known example of a
phenomenon that is *not* a feature in the interpretability sense but dominates the geometry of the
residual stream — and they were originally noticed as "outlier dimensions" in BERT-scale models,
with emergence linked to token frequency. This is the developmental treatment, and it is only
possible because of Pythia's checkpoints.

**Relevance here, and it cuts in a specific direction.** A small number of coordinates with
enormous magnitude dominate any **cosine similarity computed on the raw residual stream** — this
repo's primary tensor-level convergence metric (`cos_sim_mean` in `cos_sim_diagnostic.ipynb`), and
the instrument on which the closing judgement in `SCALING_ARTEFACT_ANALYSIS.md` rests. Because sink
coordinates are large and roughly state-independent, they **inflate** cosine similarity toward 1.
That asymmetry matters for how the existing result should be read:

- **Pythia-410m plateaus at ~0.85.** Masking the top-*k* magnitude coordinates can only push this
  *lower*. The fragmentation conclusion is therefore robust to the confound — masking would sharpen
  it, not overturn it.
- **GPT-2 Medium and Pythia-160m saturate to 1.0000 by iteration 10.** This is the reading at risk.
  A saturating cosine is exactly what sink dominance produces, so "their single-token collapses are
  real tensor attractors" is the claim that a masked re-computation would test. If the saturation
  survives masking, the finding is stronger than currently stated; if it does not, the two
  single-funnel models are a different phenomenon from what the record says.

This is a cheap re-run over existing `stage1_results.pt` files — no model forward passes — and it
tests the stronger of the two closing claims rather than the weaker one.

---

# Part IV — Anthropic and Pythia: what the record actually shows

The request asked for Anthropic to be among the sources. Here is the honest shape of that
relationship, stated precisely, because it is more interesting than a simple citation.

## IV.1 The negative finding, stated first

I surveyed the **complete publication index of the Transformer Circuits Thread**
(https://transformer-circuits.pub/, *read directly*), Anthropic's interpretability venue, from
*A Mathematical Framework for Transformer Circuits* (December 2021) through the most recent
Circuits Updates. **No Anthropic interpretability publication takes Pythia as its object of study.**
Anthropic's public MI work is conducted on (a) purpose-built toy models and (b) Claude models —
Claude 3 Sonnet, Claude 3.5 Haiku, Claude Sonnet 4.5.

This is not an oversight and should not be papered over. Anthropic has its own checkpoints, its own
frontier models, and no need for a public scaling ladder. Their constraint is the opposite of the
academic one: they can study models nobody else can access, and cannot publish weights.

## IV.2 The four real connections

**1. Pythia exists because of an Anthropic result.** The developmental methodology Pythia
operationalises is Anthropic's. Olsson et al.'s induction-head phase change (2022) is the
canonical demonstration that *when* a capability forms is a mechanistic question — and it was
unreplicable outside Anthropic for want of checkpoints. Pythia's 154-checkpoint schedule, with its
dense early log-spacing precisely where the induction bump occurs, is that constraint made
concrete. The suite is, in a real sense, infrastructure built so that Anthropic's developmental
claims could be checked by other people.

**2. Anthropic replicated a Pythia-established circuit on its own models.** Circuits Updates —
September 2024 (*read directly*) replicates **Gould et al.'s successor heads** — established on
GPT-2, Pythia and Llama — on an internal 18-layer Anthropic model, reaching ~80% correct ordinal
succession on the top head via OV-circuit analysis. This is the traffic running open→closed:
a result found on public models, confirmed on a private one. It is the strongest available evidence
that circuit-level findings on Pythia generalise to frontier-scale proprietary systems.

**3. Superposition: hypothesis at Anthropic, confirmation on Pythia, method back at Anthropic.**
The sequence is clean and worth laying out:

| Step | Work | Models |
|---|---|---|
| Hypothesis | Elhage et al., *Toy Models of Superposition* (Anthropic, Sept 2022)\* [published research report] | constructed toy models |
| Confirmation in the wild | Gurnee et al., *Finding Neurons in a Haystack* (2023)\* | **Pythia** |
| Method, closed | Bricken et al., *Towards Monosemanticity* (Anthropic, Oct 2023)\* [published research report] | Anthropic 1-layer transformer |
| Method, open | Cunningham et al. (2023) | **Pythia-70M, Pythia-410M** |
| Method, at scale | Templeton et al., *Scaling Monosemanticity* (Anthropic, May 2024)\* [published research report] | Claude 3 Sonnet |

Note the personnel edge: **Hoagy Cunningham**, first author of the Pythia SAE paper, is a
co-author of Anthropic's *Scaling Monosemanticity*\*. The open-model work was a route into the
closed-model programme, not a parallel track.

**4. Co-authorship.** *Open Problems in Mechanistic Interpretability* (TMLR 2025, entry 12) carries
three Anthropic authors and Pythia's lead author on the same paper. Whatever the institutional
separation of substrates, the field is one field.

## IV.2a The current edge: J-space, and what it means for Pythia here

Anthropic's *Verbalizable Representations Form a Global Workspace in Language Models* (Gurnee et al.,
Transformer Circuits Thread, 6 July 2026) — this repository's active follow-on focus, with a
verified primer at `docs/JSPACE_PRIMER.md` — is studied on Claude, in keeping with §IV.1. But the
companion **J-lens implementation is open-sourced and model-agnostic**
(https://github.com/anthropics/jacobian-lens): it fits on any open-weights HuggingFace decoder
transformer via `jlens.fit(model, prompts=…)`, computing `J_ℓ = E[∂h_final/∂h_ℓ]` from a corpus of
prompts and reading activations as `unembed(J_ℓ h)`. The shipped examples use **Qwen**; the library
ships **no precomputed lens for Pythia** and is marked not maintained\*.

The consequence for this repository is concrete. **Finding 6** — that ATR's motion is near-invisible
to the unembedding and to lens instruments, living mostly outside the verbalizable subspace — is
stated on the GPT-2 side. J-space is the instrument that makes "verbalizable subspace" a measured
quantity rather than a metaphor, and fitting it on `pythia-410m` is mechanically supported today.
That would put finding 6 on both arms of the 2×2 using Anthropic's own instrument, on a model with
untied embeddings where `W_E` and `W_U` are genuinely distinct spaces — which is exactly the setting
in which "outside the verbalizable subspace" is a non-degenerate claim. Cost: a Jacobian average
over ~1k prompts per layer, 24 layers at d_model 1024. This is the single highest-value item in
Part VI and is listed there as item 2.

## IV.3 One asymmetry worth naming

Anthropic's open-sourced circuit-tracing tools (2025) target open-weight models — **Gemma-2-2B and
Llama-3.2-1B**, not Pythia\*. The reason is generational: Pythia (2023, ~300B tokens, Pile-only) is
now well behind current small open models on capability, and attribution-graph work wants models
that can actually do multi-step reasoning. **Pythia's comparative advantage has narrowed to exactly
the thing nothing has replaced: dense checkpoints, known data order, a reconstructible dataloader,
and nine seeds per size at the small end.** For developmental and universality questions it remains
without a real competitor. For "what can a good small model do and how," it has been superseded.

---

# Part V — What the Pythia MI record has not settled

Recorded so that a future session does not mistake volume for closure:

1. **No end-to-end circuit for a non-trivial behaviour has been established on a large Pythia model
   and independently replicated.** The successes are components (successor heads, entropy neurons),
   representations (binding IDs, sentiment directions), or partial (Fact Finding).
2. **The seed axis is barely used.** Nine seeds each at 14M–410M make universality directly testable,
   and almost nobody has done it. This is the cheapest available high-value experiment in the whole
   suite.
3. **SAE features remain unvalidated as *the model's* units.** The reconstruction-perplexity gap
   never closed, and automated-interpretability scores are a contested metric (entry 3, entry 12).
4. **Developmental MI mostly stopped at induction heads.** The checkpoints support asking "when does
   circuit X form" for any X; the literature has asked it for very few.
5. **The 1B shape anomaly and the 6.9B/12B initialisation defect are under-reported** in papers that
   plot trends across the full ladder.

---

# Part VI — What this bears on directly in this repository

The experimental series is closed (Stages 0–5), and the artefact-vs-intrinsic question landed on
the intrinsic side at the 2026-07-10 closing judgement. Nothing below reopens that. These are items
the Pythia literature makes available *cheaply* against caveats and open threads the record already
names — the standing single-seed caveat, the un-run depth control, and the J-space follow-on.
Ordered by decisiveness per unit cost. None requires retraining.

1. **Use the seed models to retire the single-seed caveat** (§II.3). `README.md` lists
   "single-seed sweeps" as the first headline caveat, and `SCALING_ARTEFACT_ANALYSIS.md` closes with
   basin structure attributed to *depth, corpus, width, token geometry* — an attribution that a
   single training run cannot separate from run-specific weight geometry. **Nine independent seeds
   exist for `pythia-410m` and `pythia-160m`**, same data, same order, same hyperparameters, public,
   free. Running the existing 125-prompt sweep across `pythia-410m-seed{1..9}` answers directly
   whether fragmentation is a property of Pythia-410m-the-architecture or of one set of weights. No
   new method, no new code beyond the model string. **This is the strongest control the suite offers
   this project and it is currently unused.**

2. **Fit a J-lens on `pythia-410m` and put finding 6 on both arms of the 2×2** (§IV.2a, entry 11).
   Finding 6 — motion invisible to the unembedding and to lens instruments — is an *instrument*
   claim, currently resting on the GPT-2 side and on untrained lenses. Two public instruments bear
   on it, and they are complementary: the **tuned lens** (Belrose et al., trained affine probes,
   fitted on the Pythia suite, weights public) tests whether a *trained* decoder also fails to see
   the motion; the **J-lens** (Anthropic, open-source, fits on any HF decoder) measures the
   verbalizable subspace directly, which is the concept finding 6 actually invokes. Pythia's untied
   embeddings make this a sharper test than GPT-2 permits, since there `W_E` and `W_U` are the same
   matrix.

3. **Re-compute `cos_sim_mean` with top-*k* magnitude coordinates masked** (entry 13). Cheap — reads
   existing `stage1_results.pt`, no forward passes. Tests the *stronger* of the two closing claims:
   GPT-2 Medium and Pythia-160m saturating to 1.0000 is also what sink dominance would produce. The
   Pythia-410m plateau at 0.85 can only fall under masking, so that conclusion is safe either way.

4. **Log residual-stream norm alongside margin and entropy** (entry 5). Entropy neurons regulate
   confidence through the norm at the *terminal* LayerNorm — downstream of the point where §1.1's
   invariance argument applies. One extra scalar per iteration in the guardrail traces.

5. **Pin the model revision and record the version explicitly** (§II.5). `revision="step143000"`,
   plus a line in each results summary stating v1 / non-deduped / final checkpoint. Free, and it
   makes the record self-describing when someone reproduces this in two years against a Hugging Face
   repo that has moved. Related: the un-run **Test 2 depth control** (loop layers 0–11 vs 0–23,
   flagged as "the cleanest remaining attribution test") is a Pythia-410m experiment — 24 layers is
   what makes the half-depth comparison available at all.

A sixth item, larger, noted without recommendation: **Pythia's checkpoints make ATR a developmental
question.** "At which checkpoint does attractor structure appear, and at the same relative point
across sizes?" is the Pythia-shaped version of this project's central question, and it cannot be
asked of GPT-2 at all — no public checkpoint series exists. It is also expensive: a sweep per
checkpoint, on CPU. If it is ever run, the Pythia paper's own template applies — the interesting
answer is not *whether* but *at which step*, and the log-spaced early checkpoints are where phase
changes have historically been found.

---

## Sources

**Primary suite and infrastructure**
- Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, ICML 2023 — https://arxiv.org/abs/2304.01373 · HTML: https://ar5iv.labs.arxiv.org/html/2304.01373 · PMLR: https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf
- EleutherAI, Pythia repository (versions, variants, defects) — https://github.com/EleutherAI/pythia
- Model cards, e.g. https://huggingface.co/EleutherAI/pythia-410m · https://huggingface.co/EleutherAI/pythia-70m-deduped
- EleutherAI Sparsify (k-sparse SAEs/transcoders) — https://github.com/EleutherAI/sparsify
- EleutherAI Delphi (automated interpretability) — https://github.com/EleutherAI/delphi

**Anthropic**
- Transformer Circuits Thread (publication index) — https://transformer-circuits.pub/
- Circuits Updates — September 2024 (successor-head replication) — https://transformer-circuits.pub/2024/september-update/index.html
- Olsson et al., *In-context Learning and Induction Heads*, 2022 — https://arxiv.org/abs/2209.11895
- Elhage et al., *Toy Models of Superposition*, 2022 — https://transformer-circuits.pub/2022/toy_model/index.html
- Bricken et al., *Towards Monosemanticity*, 2023 — https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Templeton et al., *Scaling Monosemanticity*, 2024 — https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Anthropic, open-sourcing circuit-tracing tools, 2025 — https://www.anthropic.com/research/open-source-circuit-tracing
- Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*, 2026 — https://transformer-circuits.pub/2026/workspace/index.html · announcement: https://www.anthropic.com/research/global-workspace · code: https://github.com/anthropics/jacobian-lens · primer in this repo: [JSPACE_PRIMER.md](JSPACE_PRIMER.md)

**Mechanistic interpretability on Pythia**
- Gurnee et al., *Finding Neurons in a Haystack*, TMLR 2023 — https://arxiv.org/abs/2305.01610
- Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features*, ICLR 2024 — https://arxiv.org/abs/2309.08600
- Gould et al., *Successor Heads*, ICLR 2024 — https://arxiv.org/abs/2312.09230
- Stolfo et al., *Confidence Regulation Neurons in Language Models*, NeurIPS 2024 — https://arxiv.org/abs/2406.16254
- Feng & Steinhardt, *How do Language Models Bind Entities in Context?*, ICLR 2024 — https://arxiv.org/abs/2310.17191
- Tigges et al., *Language Models Linearly Represent Sentiment*, BlackboxNLP 2024 — https://arxiv.org/abs/2310.15154 · https://aclanthology.org/2024.blackboxnlp-1.5/
- Nanda et al., *Fact Finding* (AF sequence), 2023 — https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall
- Michaud et al., *The Quantization Model of Neural Scaling*, NeurIPS 2023 — https://arxiv.org/abs/2303.13506
- Biderman et al., *Emergent and Predictable Memorization in LLMs*, NeurIPS 2023 — https://arxiv.org/abs/2304.11158
- Belrose et al., *Eliciting Latent Predictions with the Tuned Lens*, 2023 — https://arxiv.org/abs/2303.08112
- Sharkey et al., *Open Problems in Mechanistic Interpretability*, TMLR 2025 — https://arxiv.org/abs/2501.16496
- *Hidden Dynamics of Massive Activations in Transformer Training*, 2025 — https://arxiv.org/abs/2508.03616

**Related (context)**
- Sussillo & Barak (2013), fixed-point analysis of RNNs — see `docs/PRIOR_WORK.md`
- This repository: `docs/SCALING_ARTEFACT_ANALYSIS.md`, `CROSS_MODEL_RUN_PLAN.md`, `docs/PRIOR_WORK.md`
