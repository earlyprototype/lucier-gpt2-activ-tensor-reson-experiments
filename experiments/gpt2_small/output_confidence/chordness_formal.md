# Chordness, Formalized: Weighted Variant, k-Sensitivity, and Frequency-Matched Nulls

**Date:** 2026-07-19. **Issue:** #10 (runnable-now portion). **Model:** GPT-2 Small (legacy-S3 `gpt2` weights, TransformerLens). **Inputs:** the 5 prompt attractors and 15 calibrated noise attractors of [`confidence_results.json`](confidence_results.json) (converged `final_last_vector` readouts, 500 iterations). **Raw numbers:** [`chordness_formal.json`](chordness_formal.json).

## Methods

**Plain chordness** of a token set: mean pairwise cosine similarity of the tokens' W_E embedding rows, sum_{i != j} cos(e_i, e_j) / (n(n-1)), computed over the top-k tokens of a converged state's readout distribution.

**Probability-weighted chordness**: sum_{i<j} p_i p_j cos(e_i, e_j) / sum_{i<j} p_i p_j over the top-k tokens, where p_i are the softmax readout probabilities. This weights each pair by how much probability mass actually sits on it, so a chord carried by the head of the distribution is not diluted by incoherent tail tokens.

**Token identification.** The archived JSON predates the `top_token_ids` field, so IDs were recovered by re-encoding each stored top-20 token string with `add_special_tokens=False`, keeping only tokens that round-trip to exactly one BPE ID. Result: every one of the 400 stored tokens (20 states x top-20) round-tripped cleanly; 0 tokens were excluded. No state lost any token, so all k values below use the exact stored top-k.

**Null models.** Empirical p-values are one-sided (P[null chordness >= observed], with add-one smoothing: (1 + #exceed) / (1 + 1000)), 1000 draws each, test statistic = plain chordness at k=10.

1. **Uniform null**: 10 distinct tokens drawn uniformly from the vocabulary. Distribution: mean 0.268, sd 0.019.
2. **Frequency-matched null**: GPT-2 token frequency is not available offline, so W_E row norm is used as the standard proxy (embedding norm correlates with training frequency). The vocabulary is split into 20 quantile bins by embedding norm; each null token is drawn from the same bin as the corresponding real top-10 token. This tests whether high chordness could be an artifact of the attractors selecting common (or rare) tokens, since common tokens could be mutually closer in embedding space.

A weighted variant of the frequency-matched test (real probabilities applied to null tokens, statistic = weighted chordness at k=10) is reported as `p_freq_w` in the JSON and summarized below where it changes the picture.

## Results: All 20 GPT-2 Small States

| State | Family | Top-1 | Plain k5 | Plain k10 | Plain k20 | Weighted k10 | p uniform | p freq-matched |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Lucier | prompt | `prolet` | 0.489 | 0.410 | 0.375 | 0.457 | 0.0010 | 0.0010 |
| Semantic | prompt | `prolet` | 0.489 | 0.471 | 0.372 | 0.495 | 0.0010 | 0.0010 |
| Syntactic | prompt | `Divine` | 0.320 | 0.318 | 0.285 | 0.326 | 0.0070 | 0.0050 |
| Nonsense | prompt | `prolet` | 0.489 | 0.471 | 0.369 | 0.494 | 0.0010 | 0.0010 |
| Imperative | prompt | `prolet` | 0.489 | 0.471 | 0.369 | 0.494 | 0.0010 | 0.0010 |
| trial_00 | noise | `―` | 0.311 | 0.277 | 0.291 | 0.280 | 0.329 | 0.384 |
| trial_01 | noise | `―` | 0.311 | 0.277 | 0.282 | 0.278 | 0.329 | 0.365 |
| trial_02 | noise | `―` | 0.275 | 0.275 | 0.286 | 0.264 | 0.372 | 0.466 |
| trial_03 | noise | `ei` | 0.290 | 0.286 | 0.268 | 0.286 | 0.180 | 0.194 |
| trial_04 | noise | `ei` | 0.273 | 0.274 | 0.263 | 0.276 | 0.397 | 0.488 |
| trial_05 | noise | `vertex` | 0.290 | 0.267 | 0.273 | 0.274 | 0.513 | 0.673 |
| trial_06 | noise | `―` | 0.311 | 0.283 | 0.299 | 0.291 | 0.214 | 0.269 |
| trial_07 | noise | `trader` | 0.291 | 0.309 | 0.288 | 0.340 | 0.015 | 0.027 |
| trial_08 | noise | `vertex` | 0.290 | 0.267 | 0.273 | 0.274 | 0.513 | 0.665 |
| trial_09 | noise | `ei` | 0.294 | 0.288 | 0.262 | 0.274 | 0.152 | 0.169 |
| trial_10 | noise | `instant` | 0.257 | 0.282 | 0.275 | 0.264 | 0.238 | 0.263 |
| trial_11 | noise | `Hindu` | 0.490 | 0.511 | 0.426 | 0.529 | 0.0010 | 0.0010 |
| trial_12 | noise | `―` | 0.301 | 0.313 | 0.311 | 0.335 | 0.0090 | 0.0050 |
| trial_13 | noise | `―` | 0.311 | 0.289 | 0.279 | 0.272 | 0.138 | 0.152 |
| trial_14 | noise | `vertex` | 0.290 | 0.273 | 0.275 | 0.277 | 0.417 | 0.526 |

### Sensitivity to k

The prolet-family chord is robust to k: plain chordness stays in the 0.369-0.489 range across k=5, 10, 20, always far above the null mean of about 0.27. It declines slightly as k grows (k5 highest, k20 lowest), meaning the chord is strongest at the head of the distribution and dilutes a little in ranks 11-20. The weighted variant moves the other way: weighting by probability mass makes the prolet states *more* coherent (weighted k10 up to 0.495 vs plain 0.471), confirming the chord is where the mass is. For noise states, plain and weighted values are statistically indistinguishable from each other and from the null at every k.

### Null model comparison

The frequency-matched null is nearly identical to the uniform null: per-state frequency-matched null means span 0.270-0.275 (uniform: 0.268), sd about 0.016. Matching the embedding-norm profile of the real tokens barely moves the bar: token frequency (as proxied by embedding norm) explains essentially none of the chordness signal. p-values under the two nulls agree for every state.

Per family:

- **prolet basins (Lucier, Semantic, Nonsense, Imperative)**: plain k10 0.410-0.471, p = 0.001 under both nulls (the resolution floor of 1000 draws). The chord is real and is not a frequency artifact.
- **Syntactic (`Divine`)**: plain k10 0.318, p_uniform = 0.0070, p_freq = 0.0050. Nominally significant, but the effect size is small (0.318 vs null 0.271, versus 0.41-0.47 for prolet), and the weighted frequency-matched test weakens it to p = 0.037: the modest excess coherence is not concentrated where the probability mass is. `Divine` remains much closer to a solo than to a chord.
- **Noise trials (n=15)**: plain k10 median 0.282, range 0.267-0.511. 3 of 15 reach p < 0.05 under the frequency-matched null: trial_07 (`trader`, k10 0.309, p_freq 0.027), trial_11 (`Hindu`, k10 0.511, p_freq 0.0010), trial_12 (`―`, k10 0.313, p_freq 0.0050). Trial_11 is the already-documented Hindu/Bombay chord, as strong as the prolet chord itself. The other 12 trials sit squarely inside the null distribution (p 0.15-0.67).

## GPT-2 Medium: The `D` State Is a Chord Too, but a Typographic One

The legacy S3 mirror does host `gpt2-medium`; the weights were downloaded and loaded through the same offline shim. The five original prompts were run through the ATR loop (schedule [0, 2, 3, 5, 10, 20, 50, 100], max 100 iterations). The README's claim that Medium locks by iteration 10 is confirmed: all five prompts read `D` from iteration 5 or 10 onward, with tensor cosine similarity at 1.0. Chordness and nulls are computed against Medium's own W_E (d_model 1024), with its own 20-bin norm quantiles (1000 draws).

| Prompt | Top-1 | p(top-1) | Entropy (nats) | Plain k5 | Plain k10 | Plain k20 | Weighted k10 | p uniform | p freq-matched |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Lucier | `D` | 0.010 | 7.94 | 0.534 | 0.461 | 0.444 | 0.476 | 0.0010 | 0.0010 |
| Semantic | `D` | 0.010 | 7.95 | 0.534 | 0.461 | 0.440 | 0.477 | 0.0010 | 0.0010 |
| Syntactic | `D` | 0.010 | 7.93 | 0.534 | 0.461 | 0.444 | 0.476 | 0.0010 | 0.0010 |
| Nonsense | `D` | 0.010 | 7.93 | 0.534 | 0.461 | 0.444 | 0.476 | 0.0010 | 0.0010 |
| Imperative | `D` | 0.010 | 7.96 | 0.534 | 0.464 | 0.440 | 0.480 | 0.0010 | 0.0010 |

All five prompts collapse to the same state, and its readout is a different beast from anything in Small. It is extremely diffuse: p(top-1) = 0.010, entropy about 7.9 nats, effective support about 2800 tokens, an order of magnitude flatter than Small's prolet states. Yet its top-10 is *statistically* coherent: plain k10 = 0.461 against null means of about 0.308 (uniform) and 0.306 (frequency-matched), p = 0.001 under both.

The catch is *what* the chord is made of. The `D` state's top-10 is `D`, `def`, `A`, `T`, `W`, `AB`, `I`, `The`, `RAW`, `local`: single capital letters and code-like fragments, not a lexical field. These tokens cluster tightly in embedding space because they share a *typographic* class (short, capitalized, code-adjacent), not a theme. Chordness, as defined, measures embedding-space clustering of any kind; the frequency-matched null controls for token frequency (via norm) but not for token *shape* class. So the correct cross-model statement is: Medium's `D` state passes the statistical chordness test while failing the semantic reading that made the prolet chord interesting. This is a genuine limitation of the metric, worth a shape-class-matched null in future work.

## Verdict

**The pilot claim survives the frequency-matched null.** The four language-driven prolet basins are significantly more coherent than frequency-matched random token sets (4/4 at p = 0.001, the floor of 1000 draws), at every k tested, and the effect *strengthens* under probability weighting. The frequency-matched null distribution is almost indistinguishable from the uniform null, so the coherence signal cannot be explained by the attractors preferring common or rare tokens. The separation is not perfectly clean, and honesty requires the boundary cases: 3 of 15 noise attractors also clear p < 0.05 under the frequency-matched null (led by trial_11's genuine Hindu-themed chord at k10 = 0.511, itself as strong as prolet), and the `Divine` state's nominal significance (p = 0.005) weakens to p = 0.037 under probability weighting, with an effect size a fraction of the prolet states'. So the sharp version of the claim is: language prompts *always* converge to strong chords (4/4 among settling states, chordness 0.41-0.47), noise *rarely* does (3/15, of which only one matches the prolet effect size), and the one never-settling state shows at most a weak, mass-diluted trace of coherence. Coherence separates the families as a strong statistical regularity, not as a perfect classifier.

Cross-model: GPT-2 Medium's universal `D` attractor is statistically chord-like (plain k10 0.461-0.464, p = 0.001 under both nulls) but the coherence is typographic (capital letters, code fragments) over a near-flat distribution, not a thematic lexical field over a peaked one. The *semantic* chord phenomenon, a probability-weighted lexical field under a converged readout, remains exclusive to GPT-2 Small's language regime among the models tested, but establishing that rigorously needs a null that also matches token shape class, since plain chordness alone cannot tell a theme from a type.

## Caveats

Embedding norm is a proxy for token frequency, not a measurement of it; a null matched on true corpus frequency could differ. The 1000-draw resolution floors p-values at 0.001. The 15 noise trials give limited power for estimating the noise-side false positive rate (3/15 has a wide confidence interval). All Small-state readouts come from a single converged run per condition. The Medium result shows chordness responds to typographic as well as thematic clustering; a shape-class-matched null (matching token length, case, and leading-space status) is the natural next control.
