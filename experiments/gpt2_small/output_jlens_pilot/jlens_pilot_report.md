# J-lens Pilot: Restricted Jacobian Lens and J-space Membership Probe (GPT-2 Small)

**Status: PILOT.** This is a deliberately restricted, pilot-scale version of the J-lens
construction described in docs/JSPACE_PRIMER.md Part 3, built to get a first answer to
question 1 of the primer's Part 6: where do ATR's converged attractors sit relative to a
verbalizable (J-lens) subspace? Every number below carries pilot confidence only. The
limitations section is not boilerplate: the corpus is 3% of the paper's, the token set is
0.4% of the vocabulary, and the "prolet family" turns out to be effectively a single
attractor. Script: `05_jlens_pilot.py`. Issue: #8.

## 1. Methods

### 1.1 Pilot J-lens construction

The paper's lens is `J_l = E[dh_final,t' / dh_l,t]`, averaged over source positions t,
present-and-future positions t' >= t, and 1000 pretraining-sampled prompts, giving one
d_model x d_model matrix per layer, with J-lens vectors as the rows of `W_U J_l`. The
pilot computes the J-lens *vectors* directly, for a restricted token set, via
vector-Jacobian products:

- For each corpus prompt, one forward pass with gradients enabled (no `torch.no_grad`),
  caching the residual stream at `blocks.l.hook_resid_post` for all 12 layers.
- For each token t in the restricted set, the scalar `s_t = sum over positions t' of
  logit_t(t')` where `logit = ln_final(resid_final) @ W_U[:, t]` (the constant `b_U`
  drops out of the gradient). One backward pass of `s_t` yields `ds_t/dh_l,pos` for every
  layer and position at once. Causal masking automatically restricts each position's
  gradient to contributions from t' >= pos.
- Gradients are averaged over positions within a prompt, then over the 30 prompts:
  `v_t,l = E_prompts[ E_pos[ ds_t/dh_l,pos ] ]`, one vector per (token, layer),
  shape [193 tokens, 12 layers, 768].

Deviations from the paper's construction, beyond scale:

- The scalar sums logits over downstream positions rather than averaging over t' >= pos,
  so positions with more downstream context get proportionally more weight inside a
  prompt. Accepted as a pilot approximation.
- Gradients are taken at `hook_resid_post` of each block. At layer 11 post the gradient
  path is only ln_final and the unembedding, so v_t,11 is essentially a context-modulated
  logit-lens direction; this serves as a built-in sanity anchor (its cross-prompt
  stability is 1.0000, see 3.1).
- The lens is kept as raw gradient vectors; the paper's readout normalisation
  (`softmax(W_U norm(J_l h))`) is not applied because the membership probe works in
  residual space, not readout space.

Efficiency notes (this is what made CPU feasible): all model parameters were frozen so
backward computes activation gradients only; the graph is rooted at a detached
`blocks.0.hook_resid_pre` leaf; tokens were batched 25 at a time by replicating the
prompt across the batch dimension, one backward per batch; and the full [batch, seq,
50257] unembed matmul was skipped (`return_type=None`), with `s_t` computed from
`ln_final.hook_normalized` dotted with single W_U columns. A naive per-token backward
with live parameter gradients measured 10.6 s per backward (projected ~8 hours); the
final configuration ran the whole 193 tokens x 30 prompts gradient loop in 7.1 minutes
single-threaded (this container's 4 visible CPUs are throttled; 1 thread measured ~5x
faster than 4).

### 1.2 Corpus (the pilot's main limitation)

30 hand-written prompts, 8 to 20 tokens each, across eight registers (factual,
narrative, question, instruction, casual chat, code-like, list-like, nonsense), standing
in for the paper's 1000 pretraining-sampled prompts. The full list is in
`jlens_pilot_results.json` under `corpus`. A hand-written corpus of this size cannot
claim to estimate the pretraining-distribution expectation; the averaged Jacobian here is
"averaged over 30 sentences I wrote," nothing stronger.

### 1.3 Restricted token set

Union of three groups, 193 tokens total after deduplication (one overlap):

| group | size | source |
|---|---|---|
| readout | 42 | top-20 readout tokens of each of the 5 converged states, recomputed from `output_confidence/converged_tensors.pt` (the archived JSON predates the id-first format); heavy overlap across states collapses 100 slots to 42 unique ids |
| common | 102 | common English word tokens (leading-space single-token encodings) |
| random_control | 50 | uniform random vocabulary ids, seed 1234 |

### 1.4 States probed and probe definition

Eight states, last-position vector of each converged tensor:

- **prolet**: Lucier, Semantic, Nonsense, Imperative (converged prompt states)
- **divine**: Syntactic (the Divine state: stable readout over a never-settling tensor)
- **noise**: 3 fresh Gaussian states (norm 397, seq 10), run through 100 ATR iterations
  with the same injection loop as `04_readout_confidence.py`, seed 2026

Caveat discovered during analysis: the four prolet vectors are essentially one attractor
(pairwise cosine 0.9987 to 1.0000), while Syntactic sits apart (cosine ~0.73 to each).
The prolet "family" is therefore effectively a single sample, and its four rows in the
tables below are near-duplicates, not replicates.

Per layer l, with dictionary D_l = the 193 pilot J-lens vectors:

- **Span probe (least squares)**: project the state onto span(D_l); record
  `||proj||^2 / ||state||^2` (variance share).
- **Sparse probe (nonnegative, k=25)**: minimise `||D_l^T c - h||^2` subject to c >= 0
  and at most 25 nonzero entries, via projected gradient (300 steps, hard top-k
  threshold); record the approximation's variance share. This mirrors the paper's sparse
  J-space definition (nonnegative combinations of at most ~25 J-lens vectors).
- **Control**: equal-sized random dictionary with matched per-vector norms (3 seeds
  averaged for the span probe, 1 seed for the sparse probe).

## 2. Results

### 2.1 Averaged-Jacobian stability (running mean at 15 vs 30 prompts)

Cosine between the running-mean J-lens vector after 15 prompts and after all 30, per
token, summarised per layer:

| layer | mean cos | median | min |
|---|---|---|---|
| L0 | 0.963 | 0.965 | 0.916 |
| L1 | 0.951 | 0.952 | 0.923 |
| L2 | 0.957 | 0.957 | 0.939 |
| L3 | 0.961 | 0.961 | 0.938 |
| L4 | 0.965 | 0.965 | 0.945 |
| L5 | 0.971 | 0.971 | 0.950 |
| L6 | 0.978 | 0.978 | 0.966 |
| L7 | 0.982 | 0.983 | 0.974 |
| L8 | 0.988 | 0.988 | 0.978 |
| L9 | 0.992 | 0.992 | 0.983 |
| L10 | 0.995 | 0.996 | 0.989 |
| L11 | 1.0000 | 1.0000 | 1.0000 |

The average is reasonably but not fully converged at 30 prompts: mean cosines of 0.95 to
0.98 in early and mid layers mean the direction is still moving as prompts accumulate,
exactly where the workspace band would live. L11's perfect stability confirms the sanity
anchor (that gradient barely depends on context). Per-prompt raw gradient norms are
logged in the JSON (`raw_grad_norms_per_prompt_layer_mean`) and vary smoothly by a factor
of ~2 across prompts with no outlier prompt dominating the average.

### 2.2 Membership probe: per-layer variance shares by family

Family means. Columns: least-squares span share | random-dictionary span share (3 seeds)
| nonnegative sparse k=25 share | random-dictionary sparse share.

**prolet (Lucier, Semantic, Nonsense, Imperative; effectively one attractor)**

| layer | lens span | rand span | lens nn-25 | rand nn-25 |
|---|---|---|---|---|
| L0 | 0.183 | 0.248 | 0.085 | 0.084 |
| L1 | 0.211 | 0.247 | 0.112 | 0.089 |
| L2 | 0.218 | 0.252 | 0.095 | 0.089 |
| L3 | 0.195 | 0.258 | 0.095 | 0.095 |
| L4 | 0.187 | 0.252 | 0.081 | 0.095 |
| L5 | 0.197 | 0.259 | 0.093 | 0.078 |
| L6 | 0.195 | 0.265 | 0.095 | 0.079 |
| L7 | 0.188 | 0.264 | 0.097 | 0.075 |
| L8 | 0.179 | 0.263 | 0.099 | 0.091 |
| L9 | 0.163 | 0.249 | 0.095 | 0.077 |
| L10 | 0.152 | 0.257 | 0.091 | 0.090 |
| L11 | 0.157 | 0.244 | 0.091 | 0.079 |

**divine (Syntactic)**

| layer | lens span | rand span | lens nn-25 | rand nn-25 |
|---|---|---|---|---|
| L0 | 0.213 | 0.248 | 0.104 | 0.086 |
| L1 | 0.216 | 0.250 | 0.109 | 0.117 |
| L2 | 0.221 | 0.240 | 0.104 | 0.078 |
| L3 | 0.205 | 0.285 | 0.107 | 0.071 |
| L4 | 0.207 | 0.274 | 0.095 | 0.074 |
| L5 | 0.207 | 0.261 | 0.099 | 0.067 |
| L6 | 0.211 | 0.248 | 0.111 | 0.093 |
| L7 | 0.209 | 0.274 | 0.115 | 0.089 |
| L8 | 0.197 | 0.251 | 0.115 | 0.090 |
| L9 | 0.181 | 0.250 | 0.108 | 0.083 |
| L10 | 0.171 | 0.247 | 0.096 | 0.077 |
| L11 | 0.173 | 0.269 | 0.098 | 0.092 |

**noise (3 converged Gaussian states)**

| layer | lens span | rand span | lens nn-25 | rand nn-25 |
|---|---|---|---|---|
| L0 | 0.184 | 0.253 | 0.052 | 0.085 |
| L1 | 0.186 | 0.254 | 0.068 | 0.085 |
| L2 | 0.190 | 0.245 | 0.058 | 0.087 |
| L3 | 0.185 | 0.246 | 0.060 | 0.087 |
| L4 | 0.192 | 0.257 | 0.056 | 0.096 |
| L5 | 0.198 | 0.259 | 0.064 | 0.090 |
| L6 | 0.196 | 0.253 | 0.063 | 0.089 |
| L7 | 0.187 | 0.250 | 0.062 | 0.084 |
| L8 | 0.169 | 0.253 | 0.060 | 0.078 |
| L9 | 0.146 | 0.266 | 0.054 | 0.085 |
| L10 | 0.129 | 0.243 | 0.047 | 0.101 |
| L11 | 0.114 | 0.255 | 0.048 | 0.080 |

### 2.3 Reading the tables

Three observations, in decreasing order of confidence:

1. **The lens span captures LESS variance than the size-matched random dictionary, for
   every state at every layer** (roughly 0.11 to 0.22 vs 0.24 to 0.27). This is not a
   probe failure; it is because the pilot J-lens dictionary is strongly low-rank. All 193
   vectors are images of unembedding directions under (roughly) the same averaged
   Jacobian transpose, so they crowd into a shared subspace: participation-ratio
   effective rank runs from 4 (L0) through 25-42 (mid layers) to 64 (L11), against 193
   for the random dictionary. A random 193-dim subspace of a 768-dim space captures
   ~25% of a generic vector, which is exactly what the control shows. The honest
   comparison is therefore *within* the lens columns, across families and layers, not
   lens vs random span. The random control did its job by exposing this.

2. **The clearest family separation is language-regime vs noise, and it is in the sparse
   probe.** Prompt-derived attractors (prolet and divine) sit at 0.08 to 0.12 sparse
   share, at or above their random-dictionary sparse controls in mid layers; the
   converged noise states sit at 0.05 to 0.06, clearly BELOW their controls (0.08 to
   0.10) at every layer. In the span probe the same split appears in late layers: noise
   falls to 0.11 to 0.13 by L10-L11 while prompt attractors hold 0.15 to 0.17. At pilot
   confidence, ATR's language-driven attractors have some nonnegative-sparse J-lens
   structure that converged noise lacks. This echoes the repo's earlier null-control
   finding (basins belong to the language regime) from a completely different instrument,
   and answers the primer's question 4 in miniature: a pilot J-lens sees converged noise
   as *less* J-space-like than converged language states.

3. **Divine vs prolet: the prediction goes the wrong way, weakly.** The Divine
   (Syntactic) state has a slightly HIGHER lens span share than the prolet attractor at
   every single layer (e.g. L6: 0.211 vs 0.195; L9: 0.181 vs 0.163; L11: 0.173 vs
   0.157) and a slightly higher sparse share at most layers (L7: 0.115 vs 0.097). The
   margins are small (0.01 to 0.02 absolute) and there is exactly one Divine state and
   effectively one prolet attractor, so this is a comparison of two vectors, not two
   populations.

## 3. Verdict on "prolet inside, Divine outside" (pilot confidence only)

**Not supported by this pilot; the point estimate runs slightly in the opposite
direction.** The hypothesis from JSPACE_PRIMER.md Part 6 was that the prolet attractor
might live inside the verbalizable subspace (the loop settling into something the model
can "say") while the Divine state's never-settling tensor might live outside it (its
constant readout a shadow on the exit door). In this restricted probe, the Divine state
is at least as expressible in pilot J-lens coordinates as the prolet attractor at every
layer, by both probes, by a small margin. What the pilot does weakly support is a
coarser version of the inside/outside story drawn at the regime boundary: prompt-derived
attractors (prolet AND Divine alike) carry more sparse J-lens structure than converged
noise, which sits below even a random-dictionary control. With one Divine state, one
effective prolet attractor, a 30-prompt hand-written corpus, a 193-token lens, and an
averaged Jacobian that is still visibly moving at 30 prompts (mean cosine 0.95 to 0.98
in the very layers that matter), none of this rises above pilot confidence. A null or
reversed result here is information, not failure: it says the interesting boundary may
be language-vs-noise, not prolet-vs-Divine.

## 4. Limitations (prominent, as promised)

1. **Corpus**: 30 hand-written prompts vs the paper's 1000 pretraining-sampled ones. The
   stability check shows the average has not fully converged (mean cos 0.95 to 0.98 at
   L1 to L5 between 15 and 30 prompts). Everything downstream inherits this.
2. **Token set**: 193 of 50257 tokens (0.4%). The span and sparse probes can only see
   membership in the restricted dictionary; a state could be perfectly expressible in
   J-lens vectors of tokens we did not include.
3. **Effective sample sizes**: 1 Divine state, 1 effective prolet attractor (the four
   prolet vectors are pairwise cosine > 0.998), 3 noise states. No error bars are
   possible on the family comparisons; treat every gap as a point estimate.
4. **Low-rank dictionary vs full-rank control**: the random control matches size and
   norms but not spectrum, which makes the raw lens-vs-random span comparison
   uninterpretable as a membership test (section 2.3, point 1). A spectrum-matched
   control would be the first upgrade for a full run.
5. **Construction deviations**: sum rather than average over downstream positions;
   gradients at resid_post; no readout normalisation; single-token lens only (the
   paper's own stated limitation applies with force at GPT-2 scale).
6. **GPT-2 Small may not have a cleanly organised workspace at all** (primer Part 6,
   question 3). The paper's workspace claims are made for frontier models; at 124M
   parameters the pilot may be probing structure that simply is not there, and 12 layers
   give very coarse depth resolution.
7. **Sparse probe quality**: the nonnegative k=25 approximation uses a simple projected
   gradient with hard thresholding, which is not guaranteed to find the optimal support.

## 5. Files

- `jlens_pilot_report.md`: this report
- `jlens_pilot_results.json`: full per-state per-layer shares, corpus, token-set sizes,
  stability numbers, per-prompt raw gradient norms, compute time
- `jlens_vectors.pt`: token ids and strings, token groups, layers, hook names, the
  [193, 12, 768] J-lens vector tensor, prompt count, corpus
- Script: `../05_jlens_pilot.py` (run with `ATR_GPT2_LOCAL=<gpt2 dir>`; `--probe` for
  the timing probe)

Compute: 7.1 minutes for the 193 x 30 gradient loop (single CPU thread), ~2 minutes for
noise convergence and probes. Total wall clock under 10 minutes after optimisation; the
naive construction measured 500x slower.
