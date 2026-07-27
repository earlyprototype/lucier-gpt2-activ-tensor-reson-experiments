# ATR: Findings (Canonical Record)

## Abstract

Activation Tensor Resonance (ATR) iterates a transformer's full residual-stream
tensor back through its own forward pass (extract at the final layer,
L2-rescale, re-inject at layer 0) until the state stabilises. On GPT-2 Small,
125 language prompts resolve into five attractor basins (`prolet` 43.2%,
`Divine` 27.2%, `till` 15.2%, `Anarch` 13.6%, `solidarity` 0.8%, classified at
convergence), four of them semantically coherent in embedding space. The
founding hypothesis, that these basins constitute a thematic fingerprint of
the training corpus, readable from any open-weight model, was refuted by the
project's own validation series: GPT-2 Medium, trained on the same corpus,
collapses all prompts to a single empty token; the Pythia models produce
unrelated landscapes; and a random-noise control converges to eighteen
non-semantic attractors disjoint from the five, locating the basins in the
language-driven regime rather than the weight geometry per se. Diagnostics
attribute the cross-model differences to intrinsic model dynamics, not
apparatus. What remains is a cheap, training-free probe of iterated-dynamics
regimes, one sharp dissociation between dynamics and decoding, and one open
anomaly: why GPT-2 Small, alone in this set, resolves language into few,
semantically coherent attractors. A follow-on readout-audit series (2026-07-19,
findings F6-F12) deepened both halves of that sentence. The dissociation is
resolved: the `Divine` tensor was never wandering; it is an exact period-2
limit cycle (cos(A, f(f(A))) = 1.000000) riding a single, nearly readout-invisible
axis, hidden from every earlier snapshot schedule because those schedules
sampled only even iterations (aliasing), its argmax fixed across both phases.
And the coherence lives one level deeper than previously measured: the settled
basins decode as coherent clusters of related tokens, not single tokens (top-10 readout tokens with mean pairwise
embedding cosine 0.41-0.47 against a 0.27 random baseline, p = 0.001 under
uniform and frequency-matched permutation nulls), while the winning token
itself carries only 6-9% of the probability mass. A mechanism series (F13-F17)
then traced the cycle to its cause: a single overshooting eigenvalue (-4.3 at the
pivot) executed almost entirely by one attention head, L11.H8, along a flip axis
that connects the model's most-trained and least-trained token directions and
sits almost wholly outside both the readout and the J-lens subspace. The head is
load-bearing for the cycle (ablating it collapses it to a fixed point) but is a
copy promoter on ordinary text, not the copy-suppression head the mechanism first
suggested.

---

**Scope:** Complete record of the ATR experimental series: Stage 0 (repeatability)
through Stage 5 (convergence-gated re-sweep) as of 2026-07-10, across GPT-2 Small,
GPT-2 Medium, Pythia-160m, and Pythia-410m, plus a random-noise null model and
readout diagnostics; extended 2026-07-19 with the Act II.5 readout-audit series
(runs 6-10: full-distribution confidence audit, coherence formalization with
permutation nulls, the `Divine` motion and cycle-anatomy audits, and a J-lens
pilot), reported as findings F6-F12. This document is the reporting register:
where the README (the piece) and this record differ, this record governs.

**Provenance:** Validation runs executed 2026-07-10 on CPU (Windows 11, Python 3.12,
torch 2.7.1, transformer-lens 2.16.1). Run-by-run details, deviations, and environment
notes: [`experiments/RESULTS_SUMMARY.md`](../experiments/RESULTS_SUMMARY.md). Original
exploratory work: 2026-03. Act II.5 runs (6-10) executed 2026-07-19 on CPU in a fresh
cloud container, a different machine class from all prior runs, with `gpt2` and
`gpt2-medium` weights fetched from a legacy Hugging Face S3 mirror and loaded offline
(see F6); their reports live beside their outputs under
`experiments/gpt2_small/output_confidence/`, `output_divine_motion/`, and
`output_jlens_pilot/`. Method specification: [TECHNICAL.md](TECHNICAL.md).

---

## 1. Run inventory

| # | Run | Model(s) | N | Output |
|---|---|---|---|---|
| 0 | Repeatability gate | gpt2-small | 5 prompts × 2 runs | `experiments/gpt2_small/00_reproducibility_gate.ipynb` |
| 1 | Attractor dominance sweep | gpt2-small | 125 prompts, ≤100 iters | `experiments/gpt2_small/output/` |
| 2 | Cross-model sweeps | gpt2-medium, pythia-160m, pythia-410m | 125 prompts each | `experiments/<model>/output/` |
| 3 | Random-noise null model | gpt2-small | 125 Gaussian tensors (seed 42) | `experiments/gpt2_small/output_random_baseline/` |
| 4 | Deep convergence | pythia-410m | 8-prompt subset, 1000 iters | `experiments/pythia_410m/output_deep/` |
| 5 | Convergence-gated re-sweep | gpt2-small | 125 prompts, gate cos>0.999×3, ≤1000 iters | `experiments/gpt2_small/output_gated/` |
| 6 | Full-distribution confidence audit | gpt2-small | 5 prompts × 500 iters + 15 noise trials | `experiments/gpt2_small/output_confidence/` |
| 7 | Coherence formalization + permutation nulls | gpt2-small, gpt2-medium | 20 Small states; 5 Medium prompts ≤100 iters | `experiments/gpt2_small/output_confidence/chordness_formal.md` |
| 8 | Divine motion audit (lag-10 + lag-1 probe) | gpt2-small | 3 trajectories × 1000 iters, +20 lag-1 iters | `experiments/gpt2_small/output_divine_motion/` |
| 9 | Cycle anatomy | gpt2-small | 1 Divine trajectory, iteration-1000 states | `experiments/gpt2_small/output_divine_motion/bell_anatomy.md` |
| 10 | J-lens pilot (restricted) | gpt2-small | 193-token lens × 30 prompts; 8 states probed | `experiments/gpt2_small/output_jlens_pilot/` |
| 11 | Glitch alignment (flip axis vs anomalous-token cluster) | gpt2-small | 1 `Divine` trajectory | `experiments/gpt2_small/output_glitch/` |
| 12 | Flip-axis eigenvalue + per-block/head localisation | gpt2-small | 1 `Divine` trajectory (jvp + finite diff) | `experiments/gpt2_small/output_hinge_eigen/` |
| 13 | Lag-k re-gate + engine `gate_lag` | gpt2-small | 3 states × 25 dense iters | `experiments/gpt2_small/output_lagk/` |
| 14 | J-lens phase probe (both phases, pivot, flip axis) | gpt2-small | pilot lens × cycle states | `experiments/gpt2_small/output_jlens_phase/` |
| 15 | Suppression-head test for L11.H8 | gpt2-small | 144 heads; loop ablation; 12 sentences | `experiments/gpt2_small/output_suppression/` |
| - | Tensor convergence diagnostic | all four | reads runs 1–2 | `experiments/cos_sim_diagnostic.ipynb` |
| - | Readout confidence audit | gpt2-small | single-prompt demo | `experiments/output/readout_guardrails_gpt2_small.json` |
| - | All-warm permutation test | gpt2-small (W_E) | 10,000 random 14-token sets | `experiments/gpt2_small/output_permutation/` |

Runs 6-10 are the Act II.5 readout-audit series (2026-07-19), executed on different
hardware from all prior runs (F6). Scripts:
`experiments/gpt2_small/04_readout_confidence.py`, `05_divine_motion.py`,
`06_bell_anatomy.py`, `05_jlens_pilot.py`. Runs 11-15 are the mechanism series
(issue #14, 2026-07-19 onward; findings F13-F17), scripts `07_glitch_alignment.py`
through `11_suppression_test.py`.

## 2. Principal findings

### F1: GPT-2 Small resolves language-driven activity into five attractor basins, stable under convergence gating {#run-5}

Basin shares classified **at lock-in** (cosine similarity of successive mean tensors
> 0.999 sustained ×3 checks), not at a fixed iteration horizon:

| Basin | At iter 100 (superseded) | **At lock-in (canonical)** | W_E neighbourhood |
|:---|---:|---:|:---|
| `prolet` | 44 (35.2%) | **54 (43.2%)** | political philosophy |
| `Divine` | 34 (27.2%) | **34 (27.2%)** | theology |
| `till` | 19 (15.2%) | **19 (15.2%)** | temporal/functional (outlier) |
| `Anarch` | 26 (20.8%) | **17 (13.6%)** | political philosophy |
| `solidarity` | 2 (1.6%) | **1 (0.8%)** | collective action |

91/125 prompts (73%) lock in under the lag-1 convergence gate, all at iteration 120 (the
gate's earliest firing point; the true settling iteration lies between 100 and 120,
finer cadence not measured). The iteration-100 table published earlier over-counted
`Anarch` by ~10 prompts that were still drifting `Anarch`→`prolet`. A pre-registered
hypothesis that `till` was a slow transient was **refuted**: 19/19 `till` prompts
converge and retain their label.

*Distribution-level note (2026-07-19): the five-basin count is an argmax-level
count and stands as stated. At the level of the full readout distribution,
`Anarch` is the rank-3 token inside the `prolet` states' top-10 (F8): `prolet` and
`Anarch` are two argmax peaks over one shared distribution-level structure, two
peaks of one coherent cluster of related tokens, consistent with their geometric proximity in the original
convergence matrix and with the Anarch-to-prolet drift noted above. Counted by
distinct distribution-level structures rather than by distinct winners, the
landscape holds fewer than five objects; the remaining basins' distributions have
not yet had the full audit.*

### F2: The `Divine` basin is readout-stable over a never-settling tensor

The 34 prompts that never pass the convergence gate (to 1000 iterations) are exactly
the 34 `Divine` prompts. Their decoded top-1 token is stable throughout while the
underlying tensor keeps moving, the study's clearest case of dynamics and decoding
dissociating. **Resolved (2026-07-19):** it is an exact period-2 limit cycle,
verified to machine precision (cos(A, f(f(A))) = 1.000000; F9), riding a single
rank-1 flip axis (F10). Two consequences for how this finding is now to be read.
First, the gate: the convergence gate compares consecutive iterates (lag 1), and a
lag-1 gate can never pass a period-2 cycle by construction, whatever its
threshold. "34 prompts never converge" therefore over-claims; the accurate
statement is that these 34 prompts cycle, pending re-gate (a lag-2 gate, a one-line
engine change, would likely classify them as converged; period-2 is demonstrated
for the one audited trajectory, and whether all 34 share it awaits the
prompt-library restoration). Second, the readout: the top-1 token is indeed
stable, in both phases of the cycle, but the distribution beneath it shifts with
the cycle (KL about 0.25 nats per half-cycle, F9); every earlier snapshot recorded
phase A only.

### F3: The basin landscape does not generalise across models (fingerprint hypothesis refuted)

| Model | Params | Corpus | Landscape | Tensor verdict (cos_sim_mean) |
|---|---|---|---|---|
| GPT-2 Small | 124M | WebText | 5 semantic basins | partial at 100 (0.91, σ0.15); 73% gate-converged by 120 (lag-1) |
| GPT-2 Medium | 345M | WebText | 1 basin: `D` (100%) | saturated 1.0000 by iter 10 |
| Pythia-160m | 160M | The Pile | 1 basin: `questioned` (94.4%) | saturated 1.0000 by iter 10 |
| Pythia-410m | 410M | The Pile | no consolidation (40+ fragments) | never converges (~0.85 plateau; 9/125 prompts converge; 8-prompt subset at 1000 iters: 8 distinct terminals, cross-prompt sim 0.21) |

GPT-2 Medium shares GPT-2 Small's training corpus and produces no semantic basins.
The hypothesis "attractor basins are a thematic fingerprint of the training corpus,
readable from any model" is **refuted as a general claim**. The semantic-basin
phenomenon is, on current evidence, specific to GPT-2 Small within this set.

### F4: The five basins belong to the language-driven regime, not the weights in general (null model)

125 random Gaussian tensors (norm- and length-calibrated to the real runs) iterated
through GPT-2 Small converge (position collapse → 1.0000) but into **18 basins**,
dominated by the em-dash token `―` (64%), with ~zero identity overlap with the real
five (1/125 trials reached `prolet`). Bootstrap on the random basin count: 14.1,
95% CI [11, 17]; the real count (5) falls **below** the CI. Real language funnels
into *fewer* attractors than noise, and semantically coherent ones. ATR therefore
reads the model *as driven by language-shaped input*; the basins are not universal
fixed points of the weight geometry.

### F5: The cross-model differences are intrinsic, not apparatus artefacts

> **Read the heading as *three apparatus channels exonerated*, not as *apparatus excluded* (qualified
> 2026-07-26).** A fourth channel — tokenisation — differs between the two arms and is untested. Full
> statement at the end of this finding and in caveat 17.

Three attribution results ([SCALING_ARTEFACT_ANALYSIS.md](SCALING_ARTEFACT_ANALYSIS.md)):

1. **Normalisation exonerated:** the global L2 rescale is effectively invisible to
   the forward pass (layer-0 LayerNorm scale-invariance, exact only up to
   LayerNorm's epsilon term and floating-point precision).
2. **Convergence verdicts are tensor-level:** `cos_sim_mean` never passes through
   token decoding, so Medium/160m saturation and 410m non-convergence are properties
   of the dynamics, not the readout.
3. **Readout is a real but secondary jitter source:** logit margin rises and entropy
   falls as trajectories settle (single-prompt audit); where basin labels appear they
   are high-confidence. The `Divine` dissociation (F2) is the known exception class.
   (Superseded in part, 2026-07-19: the full five-state audit, F7, shows the settled
   basins' argmax confidence is in fact low, p(top-1) 0.064-0.086; the labels are
   carried by a coherent distribution, not a confident winner, and the `Divine`
   exception is resolved in F9.)

*Qualified, 2026-07-26 — a fourth apparatus channel exists and none of the three
results above examines it.* The attribution rules out normalisation, decoding, and
readout jitter. It does not cover **tokenisation**, and the two arms of the
comparison do not tokenise alike: TransformerLens prepends a beginning-of-sequence
token for GPT-2 and not for the NeoX family, so position 0 holds a special token on
one arm and ordinary content on the other (caveat 17). That is an apparatus
difference between the models being compared, in a channel this finding's evidence
never tested. **F5's three results stand as stated; what does not stand is the
headline's implied completeness** — "not apparatus artefacts" is supported for the
three channels checked and unsupported for this one. Whether the difference is
material is untested and cheap to test: recompute the position-indexed metrics over
aligned slices — GPT-2 `[1:]` against Pythia `[:]` — as a sensitivity check, and, to
remove the BOS from the computation rather than only from the metric, add a
`prepend_bos=False` GPT-2 arm. Until that is run, F5 should be read as *three
apparatus channels exonerated*, not as *apparatus excluded*.

---

### The Act II.5 readout-audit series (2026-07-19)

*Findings F6-F12 turn the instruments on the readout itself. Every basin label
ever assigned in this project came from an argmax: the single top token of a
converged state's decoded distribution, which always names something, however
unsure the distribution beneath it is. The series audits the full softmax
distributions (runs 6-7), resolves the `Divine` anomaly (runs 8-9), and takes a
first sounding against the J-space frame (run 10). Where earlier text conflicts
(F2's open question, F5.3's high-confidence impression), the dated notes above and
the findings below govern. Reports:
`experiments/gpt2_small/output_confidence/confidence_report.md` and
`chordness_formal.md`, `output_divine_motion/divine_motion_report.md` and
`bell_anatomy.md`, `output_jlens_pilot/jlens_pilot_report.md`.*

### F6: First cross-hardware replication passed (same code, new machine, three repeats)

The Act II.5 runs executed on a fresh cloud container (CPU), a different machine
class from every prior run, with weights fetched from a legacy Hugging Face S3
mirror and loaded offline. The original five-prompt piece reproduced exactly:
terminal attractors identical (four prompts to `prolet`, the Syntactic prompt to
`Divine`), and the intermediate dissolution waypoints identical too (`Ag` at
iteration 10, `Rousse` at iteration 50, `capit` en route), reproduced three times
over the session's runs on this container. Scope of the claim, stated carefully:
this is replication of the same code on new hardware with mirror-sourced weights,
not independent re-implementation by another investigator; identical attractors
and waypoints are themselves the evidence that the mirror serves the standard
`gpt2` checkpoint. H0, caveat 1, and TECHNICAL.md's Repeatability section are
updated accordingly. Record: run 6, `confidence_report.md` Result 0; session
record `docs/sessions/SESSION_03_HANDOVER.md`.

### F7: The confidence inversion: settled tensors read out at low probability, the unsettled one at high probability

The full-distribution audit (run 6) re-ran the original five-prompt piece (500
iterations, original schedule) and read each converged state's entire softmax
distribution instead of its winner. Effective support below is exp(entropy): the
number of tokens an evenly spread distribution of that entropy would cover.

| State | Tensor settles? | Top-1 | p(top-1) | Logit margin | Entropy (nats) | Effective support |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Lucier | yes | `prolet` | 0.064 | 0.07 | 5.09 | ~163 tokens |
| Semantic | yes | `prolet` | 0.086 | 0.27 | 5.07 | ~159 |
| Nonsense | yes | `prolet` | 0.080 | 0.22 | 5.07 | ~160 |
| Imperative | yes | `prolet` | 0.081 | 0.23 | 5.07 | ~159 |
| Syntactic | no (a period-2 cycle, F9) | `Divine` | **0.505** | **2.07** | **3.05** | ~21 |

The inversion: the tensors that settle to true fixed points decode at a low-probability
argmax, p(top-1) 0.064-0.086 (roughly 0.06-0.09) with entropy pinned at 5.07-5.09
nats (uniform over the 50,257-token vocabulary would be 10.82), while the one
tensor that does not settle decodes at a high-probability argmax, p = 0.505 (later resolved as
phase A of a two-phase cycle; phase B decodes the same token at p = 0.225, F9).
Both regimes sit inside the model's ordinary expressive range: iteration-0
next-token distributions on the same prompts span p(top-1) 0.03-0.73 and entropy
1.6-7.6 nats. The argmax under-sold the basins: what converged in the `prolet`
states is the distribution itself, entropy flat at 5.07-5.09 nats from iteration
100 to 500 while the argmax never wavered. Confidence alone does not separate
language-driven attractors from noise attractors (the 15 calibrated noise trials
span p(top-1) 0.02-0.73); coherence does, almost cleanly (F8). Record:
`confidence_report.md`, Results 1, 2 and 4.

### F8: The `prolet` basin is a coherent cluster of related tokens, not a single token (permutation-tested)

**Coherence** (first use): the mean pairwise cosine similarity, in the model's
token-embedding space W_E, among the top-k tokens of a converged state's readout
distribution (k = 10 unless stated); a probability-weighted variant weights each
token pair by its probability mass. High coherence means the head of the
distribution is one cluster in embedding space rather than a grab-bag.

Under the `prolet` argmax the whole head of the distribution is one lexical field
(Semantic prompt shown; the other three near-identical): `prolet` .086,
`bourgeois` .066, `Anarch` .060, `comrade` .044, `Marx` .041, `proletarian` .036,
`socialist` .021, `anarchist` .020, `congress` .019, `labour` .018, then
`anarchism`, `the`, `movement`, `Lenin`, `comrades`. The state is not weakly
saying `prolet`; it is expressing a coherent cluster of related tokens of which `prolet` is the highest-probability member.

Quantified (runs 6-7) against two permutation nulls of 1000 draws each (test
statistic: plain coherence at k = 10; one-sided p-values with add-one smoothing,
so p = 0.001 is the resolution floor): a **uniform null** (10 tokens drawn
uniformly from the vocabulary; mean 0.268, sd 0.019) and a **frequency-matched
null** (null tokens drawn from the same embedding-norm quantile bins as the real
top-10, embedding norm being the standard offline proxy for token frequency).

- **`prolet` basins (4 states):** plain k10 coherence 0.410-0.471 against the
  ~0.27 baseline; p = 0.001 under BOTH nulls for all four states. The coherence is
  robust to k (plain coherence 0.369-0.489 across k = 5, 10, 20, always far above
  the null) and strengthens under probability weighting (weighted k10 up to
  0.495): the coherence sits exactly where the probability mass is. The two null
  distributions are nearly identical (frequency-matched means 0.270-0.275 vs
  uniform 0.268), so token frequency explains essentially none of the signal.
- **`Divine`:** 0.318. Nominally significant (p = 0.007 uniform, p = 0.005
  frequency-matched) but the effect is a fraction of the prolet states' (0.318 vs
  null 0.271) and weakens to p = 0.037 under probability weighting: a single token, not a
  coherent cluster of related tokens.
- **Noise (15 trials):** at chance in 12 of 15 (p 0.15-0.67). Three of 15 clear
  p < 0.05 under the frequency-matched null: trial 07 (`trader`, 0.309, p =
  0.027), trial 11 (`Hindu`, 0.511, p = 0.001: a genuine Hindu-themed coherent cluster of related tokens,
  `Hindu`, `Bombay`, `Hindus`, `Shiv`, at full prolet strength), and trial 12
  (the horizontal-bar token `―`, U+2015, called the em-dash token in F4; 0.313,
  p = 0.005). Noise can stumble into a real semantic well; it rarely does, while
  the settling language states always did (4/4).

Coherence therefore separates the families as a strong statistical regularity,
not a perfect classifier. One structural surprise: `Anarch`, a separate basin in
the 125-prompt sweep, is the rank-3 token inside the `prolet` distribution. The
`prolet` and `Anarch` basins are two argmax peaks over one shared
distribution-level structure (two peaks, one coherent cluster of related tokens); see the distribution-level
note under F1 for what this does to the basin count.

Standing of the coherence claim: "four of five basins semantically coherent"
previously rested on the qualitative W_E neighbourhood of each winning token,
its quantitative support having been withdrawn as an anisotropy artifact (caveat
4). The claim now holds one level deeper, in the readout distribution itself,
with permutation support. Record: `confidence_report.md` Result 2,
`chordness_formal.md`.

### F9: The `Divine` anomaly resolved: an exact period-2 limit cycle, hidden by aliasing

The motion audit (run 8) took the Syntactic prompt to 1000 iterations with
snapshots every 10 iterations from 800 to 1000, plus two controls (the Semantic
prompt as a settled `prolet` state, and one calibrated noise tensor). At lag 10
the `Divine` tensor looked frozen to four decimal places (cosine 1.000000 between
snapshots, L2 about 0.0004). The tell: the consecutive-iteration cosine at every
late snapshot sat at 0.6849 while snapshots 10 apart were identical, which is
impossible unless the state is periodic with a period dividing 10. A lag-1 probe
(20 further iterations from the saved iteration-1000 state, snapshotting every
iteration) settled it: L2 distance from the base state alternates 1249.43, 0.000,
1249.43, 0.001, and so on; cosine alternates 0.6849, 1.0000. The cycle-anatomy run
then verified the cycle exactly: **cos(A, f(f(A))) = 1.000000**.

`Divine` is not a wandering orbit and not a fixed point. It is an exact
**period-2 limit cycle**: the tensor alternates between two states, **phase A**
and **phase B** (first use: the two alternating states of the cycle, phase A
being the one every prior schedule happened to sample), separated by L2 1249
against a last-vector norm of 1612 (cosine 0.685 between them), reproduced to
machine precision every two iterations, locked in since at least iteration 800
and essentially since 250.

Why no prior run saw it: **aliasing** (first use: sampling a periodic signal only
at times that hide its oscillation). From lock-in onward, every snapshot any
schedule recorded fell on even iterations (100, 250, 500, and the lag-10 late
band 800-1000), and an even-only schedule samples a period-2 orbit at a single
phase, so the oscillation was invisible by construction.

Consequences:

- **The convergence gate can never pass this object.** The gate compares
  consecutive iterates (lag 1); on a period-2 cycle, consecutive iterates always
  differ by the full swing (cosine 0.6849 here, far below the 0.999 threshold),
  so a lag-1 gate fails a period-2 cycle by construction, forever. Claims of the
  form "34 prompts never converge" (F2 above; the same phrasing appears in the
  README) should be read as "34 prompts cycle, pending re-gate": a lag-2 gate, a
  one-line engine change, would likely classify `Divine` as converged.
- **The phase caveat.** The previously reported `Divine` readout (p = 0.505,
  entropy 3.05 nats: F7 and the confidence report) is phase A only. The
  distribution shifts with the cycle: phase B reads the same argmax at
  p = 0.2252 with entropy 4.62 nats; KL between phases is about 0.25 nats and
  total variation 0.304 per half-cycle. **The argmax is `Divine` in both
  phases.** The stable-argmax story survives; the stable-distribution story does
  not.
- **The motion is readout-suppressed but not readout-invisible.** Per unit of
  tensor motion, the readout responds at 29.5 percent of an equal-norm
  random-direction baseline (invisibility ratio 0.295, sd 0.003, n = 20; first
  use: the norm of the step's actual effect on the full logit vector divided by
  the mean effect of 20 random directions of equal norm, so values below 1 mean
  the motion is preferentially readout-invisible; here the A-B delta moves the
  logit vector by norm 198 where equal-norm random directions move it by 662).
  The controls bracket it: the `prolet` state's residual motion sits at the
  numerical floor (L2 about 3e-04 per step), and the noise control's drift is
  slightly readout-amplified (ratio 1.12). Hypothesis H-D1 is thereby supported
  in a weakened, more precise form (see the disposition table).

Record: `divine_motion_report.md`; exact-cycle verification in `bell_anatomy.md`.

### F10: Anatomy of the period-2 cycle: one nearly readout-invisible flip axis between a game-vocabulary pole and the glitch-token pole

The cycle-anatomy run (run 9) dissected the cycle recovered from the saved
iteration-1000 checkpoint. Writing the two phases as A = M + d and B = M - d
around their midpoint M, the **flip axis** (first use: the single direction d that
the iterated map negates on each pass; called the hinge in earlier revisions) turns
out to be one global direction: the
per-position flip axes agree at mean pairwise cosine 1.0000 across all ten
positions. The whole tensor inverts along a single rank-1 direction, which makes the
negative-eigenvalue reading of the cycle nearly literal.

- **The same token set at two probability levels.** Phase B's top-10 is the same token set as phase
  A's, in nearly the same order, at different probabilities: `Divine` falls from 0.505
  to 0.225 while `【` rises from 0.064 to 0.126; coherence is 0.318 in both
  phases and at the midpoint. There is no hidden second coherent cluster.
- **The recorded norm contrast is a frame artifact, not an energy redistribution.**
  The last token position was recorded at norm 1612 in phase A and 464 in phase B,
  but the two numbers were taken in different frames (phase A in the raw frame,
  phase B in the shell frame), an artifact of how `06_bell_anatomy.py` built the
  states. On the loop's energy shell both phases carry identical row norms, so the
  oscillation does not redistribute energy across positions. See caveat 15.
- **The flip axis is about 95 percent invisible to the readout.** The axis d produces a
  logit response of 33 against 612 for equal-norm random directions: ratio
  0.054, far more suppressed than the full per-step delta (0.295, F9).
  Decomposed against the unembedding W_U's singular directions, 73 percent of
  the axis energy sits in the bottom-100 (lowest-response) directions and only 13
  percent in the top-100; the pivot M is similarly in the low-response corner (67 percent
  bottom-100). The `Divine` phenomenon inhabits the model's least speakable
  subspace.
- **The poles.** Tokens whose logits rise most toward phase A: `Change`,
  `Divine`, `Release`, `Form`, `Fin`, `Air`, `Dou`, `Ground`, `Physical`, `Wind`
  (a coherent game/elemental-move vocabulary). Toward phase B: `reddits`,
  `ertodd`, `ModLoader`, `espie`, `annis`, `quickShipAvailable`, `ocrats`,
  `orkshire`, `colonists`. Several of these (`ertodd`, a fragment of
  ` petertodd`; `quickShipAvailable`; and neighbours) match the published GPT-2
  anomalous-token cluster, the SolidGoldMagikarp family (Rumbelow and Watkins,
  2023): under-trained tokens whose embeddings sit in a degenerate corner of
  embedding space. Phase B leans toward that corner: direct evidence for the
  earlier speculation that the `Divine` attractor sits near the anomalous-token
  region.

Reading: the period-2 cycle is a rank-1 self-negating mode, swinging between a
game-vocabulary pole and the glitch-token pole, with the swing itself almost
entirely invisible to the vocabulary projection; the stable `Divine` argmax is
the shadow of the shared pivot M. The glitch-token identification is by
inspection against published lists, not a systematic test. Open: whether all 34
`Divine` prompts share this flip axis (blocked on the prompt-library restoration,
issue #9). Record: `bell_anatomy.md`.

### F11: J-lens pilot: the prolet-inside/Divine-outside prediction did not hold; the boundary that appeared is language-vs-noise

Background: Anthropic's J-space paper ("Verbalizable Representations Form a
Global Workspace in Language Models", 2026) proposes that a model's verbalizable
states occupy a distinguished subspace, probed by a lens built from averaged
Jacobians of the forward map (the J-lens); this repo's reading companion is
`docs/JSPACE_PRIMER.md`. The coherence finding sharpened a prediction from that
reading: a coherent cluster of related tokens looks like a verbalizable, workspace-like state
(inside the J-lens subspace), a high-probability incoherent single token like a projection artifact
from outside it; hence **prolet inside, Divine outside**. A deliberately
restricted pilot lens was built (run 10): J-lens vectors computed by
vector-Jacobian products for a 193-token dictionary (0.4 percent of the
vocabulary) over 30 hand-written prompts (3 percent of the paper's corpus
scale), at all 12 layers; membership probed by least-squares span share and by
nonnegative sparse (k = 25) share, against size- and norm-matched random
dictionaries.

**Verdict at pilot confidence: not supported; the point estimate runs slightly
the other way.** The `Divine` state is at least as lens-expressible as the
`prolet` attractor: higher span share at every layer (L6: 0.211 vs 0.195; L9:
0.181 vs 0.163; L11: 0.173 vs 0.157) and higher sparse share at 11 of 12 layers,
with margins of 0.01-0.02 absolute, and with exactly one `Divine` state and
effectively one `prolet` attractor to compare (the four prolet vectors are
pairwise cosine 0.9987-1.0000): a comparison of two vectors, not two
populations.

What did appear is a coarser boundary: language-vs-noise. Prompt-derived
attractors (`prolet` and `Divine` alike) hold nonnegative-sparse share
0.08-0.12, at or above their random controls in mid layers; converged noise
states sit at 0.05-0.06, clearly below their controls (0.08-0.10) at every
layer, and in the span probe noise falls to 0.11-0.13 by L10-L11 while the
prompt attractors hold 0.15-0.17. At pilot confidence, a J-lens sees converged
noise as less J-space-like than converged language states: the regime finding
(F4) echoed by a different instrument.

Recorded as a null with structure, limitations up front: the pilot lens
dictionary is strongly low-rank (effective rank 4-64 across layers vs 193 for
the random control), which makes the raw lens-vs-random span comparison
uninterpretable as a membership test; the averaged Jacobian is still visibly
moving at 30 prompts (running-mean cosine 0.95-0.98 in early and mid layers);
and the probe pre-dates the period-2 cycle discovery, so it saw phase A only. The full
build (issue #8) should be phase-aware: probe both phases and the pivot M.
Record: `jlens_pilot_report.md`.

### F12: Cross-model: GPT-2 Medium's universal attractor is a typographic cluster over a near-flat readout

The same loop, run on `gpt2-medium` from the same mirror (run 7; five prompts,
max 100 iterations), confirms F3's picture and adds the distribution view. All
five prompts collapse to the `D` state by iteration 5-10 (tensor cosine 1.0).
Its readout is unlike anything in Small: p(top-1) = 0.010, entropy 7.93-7.96
nats, effective support about 2,800 tokens, an order of magnitude flatter than
Small's `prolet` states (entropy about 5.1 nats) and `Divine` phases (3.05 and
4.62). This entropy contrast is itself a new cross-model observation: Small's
language regime produces peaked, thematically saturated readouts; Medium's
single attractor is near-flat.

Yet Medium's top-10 passes the statistical coherence test (plain k10
0.461-0.464 against its own null means of about 0.31; p = 0.001 under both
uniform and frequency-matched nulls). The catch is what the cluster is made of:
`D`, `def`, `A`, `T`, `W`, `AB`, `I`, `The`, `RAW`, `local`: single capital
letters and code-like fragments. These sit close in embedding space because
they share a *typographic* class (short, capitalised, code-adjacent), not a
theme. Coherence measures embedding-space clustering of any kind, and the
frequency-matched null controls for frequency (via the norm proxy) but not for
token shape. Standing methodological rule, recorded here: **no cross-model
coherence claim until a shape-class-matched null exists** (matching token
length, case, and leading-space status). Until then, the *semantic* coherence
phenomenon, a probability-weighted lexical field under a peaked readout,
remains exclusive to GPT-2 Small's language regime among the models tested.
Record: `chordness_formal.md`.

---

### The mechanism series (issue #14, 2026-07-19 onward)

*Findings F13-F17 follow the `Divine` period-2 cycle (F9, F10) down to its
mechanism: which embedding-space directions its flip axis connects (F13), the
eigenvalue and the single attention head that produce the inversion (F14), the
engine change that lets the convergence gate recognise the cycle (F15), where the
two phases and the flip axis sit relative to the J-lens subspace (F16), and
whether the head that drives the cycle belongs to the copy-suppression class
(F17). All five follow the one audited `Divine` trajectory (the Syntactic prompt)
from the committed iteration-1000 checkpoint; whether the other 33 period-2
prompts share the structure is blocked on the prompt library (issue #9, caveat
14). Reports live beside their outputs: `output_glitch/glitch_alignment.md`,
`output_hinge_eigen/hinge_eigenvalue.md`, `output_lagk/lagk_report.md`,
`output_jlens_phase/jlens_phase.md`, `output_suppression/suppression_report.md`.*

### F13: The flip axis connects the model's most-trained and least-trained token directions

F10 identified the phase-B pole of the flip axis with the published GPT-2
anomalous-token cluster (the SolidGoldMagikarp family) by inspection. Run 11
measures it. Writing u for the unit direction from the global mean embedding to a
cluster's centroid, the phase-B pole (-d) is aligned with the geometric core of
under-trained tokens (the control-byte and undecodable-byte tokens plus named
family members): **cos(-d, u_core) = +0.596**, against 1000 random sets (mean
|cos| 0.065, max 0.30) and 1000 norm-matched sets, p < 0.001 under both. The
norm-matched null is the sharp control: sets matching the core's embedding-norm
profile point the *opposite* way (mean +0.48, toward phase A), so the alignment
is about which tokens these are, not their norms. The curated SolidGoldMagikarp
family agrees independently (cos +0.456, p < 0.001 both nulls). The -d ray is
saturated with cluster members: of the top 50 vocabulary tokens by cos(row, -d),
45 are in the 0.1% geometric core and all 50 within the 0.5% shell (a 200-fold
enrichment). The phase-A pole (+d) is the opposite corner: its top 50 tokens are
the highest-frequency function words (`the`, `,`, `in`, `and`, `a`), contain no
cluster member, and 42 of 50 lie in the bottom 0.5% by embedding norm;
cos(u_core, u_function-word) = -0.68. So each pass, the normalised map throws the
state toward the least-trained corner of embedding space and back toward the
most-trained corner. The alignment is a strong tilt, not an identity (0.46-0.60,
not 0.9), and the flip axis also carries a large pivot component, so the
informative pole is -d specifically. It holds identically at all 10 positions
(the flip axis is one global direction), and cluster membership is
basis-independent (Jaccard 1.0, raw vs processed). Record: `glitch_alignment.md`.

### F14: The inversion is one overshooting eigenvalue, executed by a single attention head, L11.H8

F10 conjectured the flip axis carries an effective eigenvalue near -1. Run 12
measures the linearised ATR map by forward-mode autodiff (`torch.func.jvp`,
agreeing with central finite differences to 3-4 significant figures) and reports
two corrections. **Magnitude:** at the symmetric pivot the flip-axis eigenvalue is
not -1 but **-4.3** (an overshooting reflection, cos(Jd, -d) = 0.991), while
around the composed two-step map the projected multiplier along the axis is
**+0.10** (perturbations off the orbit decay by about 90% per period, which is why
the cycle reproduces to machine precision). This is a period-doubling
configuration: a near-fixed pivot (one forward pass returns it 0.995 aligned with
itself) that is flip-unstable along exactly one direction and sheds a stable
finite-amplitude period-2 orbit. **Locality:** the flip axis passes through blocks
0-10 upright (its cosine to itself never falls below +0.88) and is inverted
entirely inside block 11; within block 11 attention outweighs the MLP 12 to 1, and
one head, **L11.H8, carries 99.1% of the attention flip** (per-head d-component
-1.981; no other head exceeds 0.014). Random control directions pass through
upright (eigenvalue near +1). The inversion is thus real, direction-specific, and
localised to a single OV circuit. Frame note: the literal -1 of the original
conjecture appears only for the frame-mixed "committed" flip axis (lambda -0.864);
the physical on-shell axis d_sym carries the -4.3, and the two reconcile once the
loop's renormalisation strips the committed axis's radial part (leaving it 0.973
aligned with d_sym; caveat 15). Record: `hinge_eigenvalue.md`.

### F15: A lag-2 convergence gate recognises the period-2 cycle; the engine now supports it

F9's standing correction (the 34 non-converging prompts are period-2 cycles
"pending re-gate") is now implemented and demonstrated for the one audited
trajectory. `atr_engine.run_atr_gated` gained a **`gate_lag`** parameter (compare
iterate t with t-k; default 1, verified bit-identical to the pre-change engine on
matched runs) and a `lag_scan` helper reporting mean cosine at every lag over a
dense continuation. On a 24-iteration continuation from the committed
iteration-1000 states, three signatures separate cleanly: the `prolet` fixed point
passes at every lag (flat 1.0000000); the `Divine` state fails every odd lag
(0.6849) and passes every even lag (1.0000000), the parity signature of an exact
period-2 cycle, so it is **converged under `gate_lag = 2` and unconvergeable under
`gate_lag = 1`**; the noise control decays monotonically with lag (no period).
Both phases decode to the same argmax (` Divine`, p 0.505 / 0.225). Two honest
limits are recorded. First, the lag-2 gate inherits the same aliasing one octave
up: a period-4 cycle would fail lags 1-3, 5-7 and pass only 4 and 8, invisible
again under lag 2; the recommended 34-prompt re-gate therefore runs the full lag
table on a short dense continuation and gates each state at its smallest passing
lag, rather than swapping one fixed lag for another. Second, the lag-k gate
corrects cycle aliasing but not threshold-blindness to slow drift: the
still-drifting noise control nominally clears 0.999 at every lag in this
decelerated late window. The other 33 period-2 prompts remain blocked on the
prompt library (issue #9); one, the Syntactic prompt, is now re-gated as
converged. Record: `lagk_report.md`.

### F16: Phase-aware J-lens: the phases straddle the `prolet` level, the pivot is the most lens-expressible state probed, and the physical flip axis is almost entirely outside the lens

The J-lens pilot (F11) saw only phase A. Run 14 re-runs the same restricted pilot
lens (193 tokens; every F11 limitation inherited, caveat 13) on both phases, the
pivot M, and the flip axis. The pilot's reversal ("`Divine` at least as
lens-expressible as `prolet`") holds for phase A, **strengthens at the pivot M**
(the most lens-expressible object probed, above phase A at every layer), and
**reverses for phase B** (less lens-expressible than the `prolet` attractor at
every layer on both the span and sparse probes). So the cycle is not "inside" or
"outside" the lens as one object: it swings between a more-verbalizable phase and a
less-verbalizable phase, pivoting on the most-verbalizable state in the system.
The physical flip axis d_sym is almost entirely outside the lens: least-squares
span share **0.013 at L11 against a 0.252 chance level** (5% of chance; mean over
layers 0.021 vs 0.249), never above 0.029 at any depth, and its readout-invisible bulk
(97.0% of its energy) is outside the lens at essentially every layer. This
restates F10's readout-invisibility in the lens frame. The frame-mixed committed
axis's milder deficit (0.145 at L11, 58% of chance) is pivot contamination (caveat
15). The language-vs-noise boundary (F4, F11) survives but is now a sparse-probe
story: on the span probe, phase B sits at or below converged-noise level until the
final layer. This is not a null: the phase-blind pilot could not tell the phases
apart, and they are materially distinguishable to the lens. Record:
`jlens_phase.md`.

### F17: L11.H8 is load-bearing for the cycle but is a copy promoter, not a copy-suppression head

The suppression-head hypothesis read L11.H8 (F14) as an instance of the documented
copy-suppression class (like GPT-2 Small's L10.H7), the closed loop recycling its
one-shot negative correction into a sustained oscillation. Run 15 ran three tests.
(1) Among all 144 heads, L11.H8's OV inverts the flip axis d_sym most strongly (cos
-0.9619, gain 63.68, rank 1; per-unit d-component -61 against the runner-up's -1.2,
a different magnitude class). (2) Ablating L11.H8 inside the loop collapses the
cycle to a fixed point within about 10 iterations (the readout going from
` Divine` at p 0.5 to a flat ` the`), while a same-layer control ablation (L11.H0)
leaves a period-2 cycle running, so the head is load-bearing and specifically so.
(3) On ordinary text (no loop), L11.H8 *raises* the attended token's logit at 91.4%
of positions (mean delta +5.97), the opposite of copy suppression, while the L10.H7
positive control shows the documented suppression (87.1% negative, mean -3.62), so
the protocol detects suppression where it exists. **Verdict: (1) supported, (2)
supported, (3) refuted with the opposite sign.** L11.H8 sustains the cycle by
inverting the flip axis, but it is a copy promoter, not a suppressor; the "learned
copy-suppression function" reading is unsupported, and the structural-accident
reading (the cycle exploits a strongly negative direction that happens to sit in
this head's OV spectrum but is not exercised as suppression in ordinary next-token
service) is strengthened. Open: whether d_sym relates to some non-token content the
head suppresses in contexts not sampled here. Record: `suppression_report.md`.

## 3. Hypothesis dispositions

| ID | Hypothesis | Disposition |
|---|---|---|
| H0 | Results are deterministic | **Repeatability supported; first cross-hardware replication passed (2026-07-19)**: N=2 same-machine runs, plus three identical repeats on a fresh container (same code, different machine class, legacy-mirror weights, identical terminal attractors and dissolution waypoints; F6). Intermediate paths float-sensitive. Independent re-implementation still not attempted. |
| H1 | `prolet` is the dominant basin | **Supported, revised upward**: 43.2% at lock-in (was 35.2% at iter 100). Per-prompt category predictions remained poor (~25%); the structural claim stands, the predictive one does not. |
| H2 | `Divine` is a genuine secondary basin | **Supported; the object is now resolved (2026-07-19)**: 27.2%, and unlike the other four it is not a fixed point but an exact period-2 limit cycle with a phase-invariant argmax (F2, F9, F10): a high-probability single token over a moving tensor, not a low-probability basin. |
| H3 | Intermediate tokens reflect training-corpus topology | **Weakened further at close; coherence half upgraded 2026-07-19**: the all-warm cross-similarity matrix was permutation-tested and found to be an anisotropy artifact (99.9% of random 14-token sets are also all-positive; see caveat 4, resolved), and the corpus-causal reading had already failed cross-model (F3). The semantic-coherence observation itself, however, no longer stands as qualitative only: it now holds one level deeper than the token-level W_E neighbourhood, in the full readout distribution, with permutation support (coherence 0.41-0.47 vs 0.27, p = 0.001 under both nulls; F8). |
| H4 | Per-head resonance ≈ linear power iteration on W_OV (cos > 0.9 to top singular vector) | **Untested**: protocol scaffolded (`experiments/gpt2_small/spectral_resonance.ipynb`), not run. |
| H-fingerprint | Basin profiles read training-data bias without data access | **Refuted as stated** (F3, F4). |
| H-till | `till` is a slow transient | **Refuted** (F1: 19/19 stable). |
| H-D1 | `Divine`'s late-stage motion lies mostly in readout-flattened directions | **Supported in a weakened, more precise form (2026-07-19)**: the motion is an exact period-2 cycle whose per-step readout response is 0.295 of the equal-norm random baseline and whose flip axis responds at 0.054, but the distribution visibly shifts (p(top-1) swings 0.505 to 0.225 each half-cycle) while the argmax stays fixed (F9, F10). |
| H-J1 | `prolet` sits inside the verbalizable (J-lens) subspace, `Divine` outside | **Not supported at pilot confidence (2026-07-19); now phase-qualified (F16)**: the point estimate runs slightly the other way at pilot confidence (`Divine` at least as lens-expressible as `prolet`), and the boundary that appears is language-vs-noise (F11). The phase-aware re-probe (F16) splits it: the reversal holds for phase A, strengthens at the pivot M (most lens-expressible), and reverses for phase B (below `prolet` at every layer); the physical flip axis is almost entirely outside the lens (span 0.013 vs 0.252 chance at L11). Full build still pending (issue #8). |
| H-glitch | The `Divine` flip axis aligns with the anomalous-token (SolidGoldMagikarp) cluster | **Supported as a structural alignment (2026-07-19, F13)**: cos(-d, under-trained core) = +0.60, p < 0.001 under random and norm-matched nulls; the swing runs between the most-trained (function-word) corner and the least-trained (glitch) corner. A strong tilt (0.46-0.60), not an identity. |
| H-flip | The flip axis carries an effective eigenvalue near -1, localisable to a block | **Refined (2026-07-19, F14)**: real, direction-specific, and localised (one direction; one head, L11.H8, does 99%), but the pivot eigenvalue is -4.3 (overshooting), not -1; a period-doubling configuration (composed-cycle multiplier +0.10). The literal -1 was a frame-mix artifact of the committed axis. |
| H-supp | L11.H8 is a copy-suppression head whose one-shot negative correction the loop recycles | **Refuted with the opposite sign (2026-07-19, F17)**: L11.H8 inverts the flip axis (rank 1 of 144) and is load-bearing (ablation collapses the cycle), but on ordinary text it raises the attended token's logit at 91% of positions (a copy promoter), where the documented L10.H7 suppressor lowers it. The learned-function reading is unsupported; the structural-accident reading is strengthened. |

## 4. Caveats {#caveats}

1. **Repeatability plus one cross-hardware replication, not independent
   reproducibility.** N=2 same-machine runs, and (2026-07-19) three identical
   repeats of the five-prompt piece on one fresh cloud container: same code,
   different machine class, legacy-mirror weights, identical terminal attractors
   and dissolution waypoints (F6). No independent re-implementation by another
   investigator.
2. **Single-seed sweeps.** The 125-prompt sweeps are one seed per model; the null
   model is one seed set (42) with a bootstrap over trials, not over sweeps.
3. **Deep-convergence subset.** The 1000-iteration Pythia-410m run used 8 prompts
   (CPU constraint). Direction matches the 125-prompt evidence at 250 iterations, but
   the subset is small.
4. **W_E permutation test: RESOLVED (2026-07-11), negative.** The all-warm
   cross-similarity matrix (91/91 pairs positive, 0.18–0.47) is an anisotropy
   artifact: 9,994/10,000 random 14-token sets are also all-positive, and the
   global mean pairwise cosine of the embedding space is 0.268 vs the observed
   set's 0.288 (S2 p=0.167, S3 p=0.099). The compact-subspace interpretation is
   withdrawn. The local semantic-neighbourhood observation remains qualitative.
   Record: `experiments/gpt2_small/output_permutation/`.
5. **Gate cadence.** Lock-in iterations cluster at 120 because that is the gate's
   earliest possible firing; true settling times between 100 and 120 are unresolved.
6. **Hook-position dependence unexplored.** All runs cut the loop at
   (final-layer `resid_post` → layer-0 `resid_pre`). Alternative windows (including a
   Pythia-410m depth control, layers 0–11 vs 0–23) are designed but not run.
7. **Normalisation scheme.** Global L2 rescale only; per-position/per-dimension
   schemes unexplored (though the global scheme is inert through layer-0
   LayerNorm up to the epsilon term and floating-point precision, see F5.1).
8. **BPE granularity.** Basin identities are single BPE tokens (`prolet`, `Anarch`);
   multi-token structure is invisible to the current readout.
9. **Readout is logit-lens-style.** Decoding applies `ln_final → W_U` to
   intermediate states; the `Divine` dissociation (F2, resolved in F9) shows the
   decode and the dynamics can disagree. The full confidence audit now covers the
   original five converged states plus 15 noise trials (F7); the 125-prompt
   sweep's states have not had it.
10. **Coherence is blind to the cause of clustering.** It measures W_E clustering
    of any kind: GPT-2 Medium's `D` state passes at p = 0.001 on typographic
    grounds, capital letters over a near-flat readout (F12). No cross-model
    coherence claim until a shape-class-matched null (token length, case,
    leading-space status) exists. Embedding norm is a proxy for token frequency,
    not a measurement of it, and 1000 draws floor the p-values at 0.001.
11. **Act II.5 sample sizes.** One run per condition on one machine; 15 noise
    trials, so the 3/15 boundary-case rate carries a wide interval; one `Divine`
    trajectory from one prompt, with period-2 exactness verified over 20
    iterations at iteration 1000; whether all 34 `Divine` prompts share the F10
    flip axis is untested (prompt library pending, issue #9).
12. **Even-iteration aliasing in the archive.** From lock-in onward, every
    snapshot recorded before the lag-1 probe fell on even iterations, so all
    archived `Divine` distributions are phase A only (F9). Any schedule that
    samples a period-2 orbit at one parity records a single phase; excluding
    periodicity requires lag-1 (or odd-offset) probes.
13. **The J-lens pilot is a pilot.** 30 hand-written prompts against the paper's
    1000 sampled ones; 193 of 50,257 tokens; effectively one `prolet` sample and
    one `Divine` sample; a low-rank lens dictionary against a full-rank random
    control (raw span comparisons uninterpretable as a membership test); and the
    probe saw phase A only (F11).
14. **The mechanism series is one trajectory.** F13-F17 all follow the single
    audited `Divine` trajectory (the Syntactic prompt) from the committed
    iteration-1000 checkpoint, with derivatives evaluated at one point per state.
    Whether the other 33 period-2 prompts share the flip axis, the flip head
    (L11.H8), the eigenvalue, and the anomalous-token alignment is untested
    (prompt library pending, issue #9).
15. **Frame mixing in the committed flip axis.** `06_bell_anatomy.py` built its
    flip axis by subtracting shell-frame `B` from raw-frame `A`, so the committed
    axis is about 83% radial (pivot-aligned). The physical on-shell axis `d_sym`
    carries the mechanism; all headline numbers in F14, F16, and F17 use `d_sym`.
    The earlier "phase A norm 1612, phase B 464" contrast (F10) is the two frames,
    not an energy redistribution: on the shell both phases have equal row norms.
16. **Small text sample for the suppression test.** F17's copy-suppression test
    (test 3) uses 12 sentences, 116 positions. The sign of the verdict
    (fraction-negative 0.086 for L11.H8 vs 0.871 for the L10.H7 control) is far
    from the decision boundary, so the sample fixes the direction of the result
    but not fine effect sizes; and it measures copy suppression in the
    token-unembedding sense only, so suppression of non-token content would not
    register.
17. **The two arms of the cross-model comparison do not tokenise the same way
    (2026-07-26).** `atr_engine.py` passes a raw string to `run_with_cache`
    (lines 125, 183, 310, 343). TransformerLens tokenises strings through
    `to_tokens`, which prepends the beginning-of-sequence token when
    `cfg.default_prepend_bos` is set, and in `loading_from_pretrained.py` only
    `GPTNeoXForCausalLM` carries an explicit `False` (line 537) — GPT-2 has no
    override and inherits the global default `True` (line 1720). **So position 0
    holds the special token `<|endoftext|>` in every GPT-2 and GPT-2 Medium
    trajectory and an ordinary content token in every Pythia one.** That much is
    token construction and is measured. Whether that position then *functions* as
    an attention sink in these trajectories is an inference from the sink
    literature ([GPT2_DEEP_DIVE.md](GPT2_DEEP_DIVE.md) §5.4), **not** a
    measurement made here: the only sink-adjacent measurement in hand is
    coordinate-structured, and massive activations are a coordinate phenomenon
    while attention sinks are a positional one. Treat "sink at position 0" as an
    open hypothesis throughout this caveat. Measured on the engine's own call path:
    4 raw tokens become 5 for `gpt2` and `gpt2-medium`, and stay 4 for
    `pythia-160m` and `pythia-410m`. Nobody chose this; it is a library default
    that varies by model family and is invisible at the call site.
    Three consequences, in order of how much they bite:
    (a) every **position-indexed cross-model** comparison — F1's position
    uniformity above all — compares sequences whose position 0 means different
    things, and whose lengths differ by one for the same prompt, so per-position
    means are taken over different denominators;
    (b) within the GPT-2 arm the special token has a **known address**, which
    makes testable — rather than merely worrying — the question of whether the
    global L2 rescale is partly setting the magnitude of a position that carries
    disproportionate norm (caveat 7 rules the rescale inert through layer-0
    LayerNorm, which is a statement about the forward map, not about how the
    conserved norm is distributed across positions);
    (c) from iteration 1 onward the re-injection overwrites position 0, so
    whatever role that position plays is preserved *structurally* while its
    *contents* are replaced — if the sink hypothesis holds, that is a regime no
    model was trained in, and one only the GPT-2 arm enters. Conditional on the
    hypothesis, and untested.
    The cheap first control is to align the slices: **GPT-2 `[1:]` against
    Pythia `[:]`**. Dropping index 0 from *both* arms — the form first written
    here, and wrong — would strip GPT-2's BOS and simultaneously strip Pythia's
    first genuine content token, replacing one misalignment with another.
    `experiments/gpt2_small/11_suppression_test.py:607-610` has the accounting
    (`# [1, L], BOS at 0`, `n_tokens_no_bos`) and is the one place in the
    repository that had it right. Two limits on that control, both real: it is a
    **sensitivity check, not a restoration of comparability**, because the BOS
    participates in the GPT-2 forward pass and conditions every other position
    through attention, so removing it from the *metric* does not remove it from
    the *computation*; and the two arms still differ in what conditioned the
    states being compared. Only a `prepend_bos=False` GPT-2 arm removes the BOS
    from the computation itself. Note this cannot explain F3:
    GPT-2 Small and Medium both carry the BOS and behave completely differently.
    Raised by `agent:pythia-review` (peer board, discussion #59), verified
    against the TransformerLens source and by execution; literature context in
    [GPT2_DEEP_DIVE.md](GPT2_DEEP_DIVE.md) §5.4.

## 5. What ATR is, after this series

A cheap, training-free probe of the stable states of a model's iterated forward map
under a chosen input regime. It does not read training-data bias (refuted). It does
distinguish, sharply and at tensor level, qualitatively different iterated-dynamics
regimes across models, and it surfaced one unexplained anomaly worth pursuing: GPT-2
Small's five semantically coherent, language-specific attractor basins.

*Updated 2026-07-19 (Act II.5):* the `Divine` object is no longer on the open
list: it is resolved as an exact period-2 limit cycle riding a single, nearly
readout-invisible axis (F9, F10). The anomaly is sharpened rather than removed: GPT-2
Small still stands alone in resolving language into few, semantically coherent
attractors, and that coherence is now known to live in the full readout
distribution (coherent clusters, F8), carried at a low-probability argmax (F7).

*Updated 2026-07-19 onward (mechanism series, issue #14):* the period-2 cycle is
now traced to its cause. One attention head (L11.H8) executes an overshooting
eigenvalue (-4.3 at the pivot) along a single flip axis (F14); that axis connects
the most-trained and least-trained token directions (F13) and lies almost
entirely outside both the readout and the J-lens (F16); the head is load-bearing
for the cycle but is a copy promoter, not a copy-suppression head (F17); and the
convergence gate now takes a `gate_lag` parameter that recognises the cycle (F15,
demonstrated for the one audited trajectory, the other 33 pending issue #9). What
is now most open on the `Divine` object is whether those other 33 period-2 prompts
share this structure. The larger anomaly (why GPT-2 Small alone) is unchanged.

Open directions, in rough order of leverage: why GPT-2 Small (the anomaly, now
with low-probability coherent clusters as the thing to explain); the lag-2 re-gate of the 34
cycling prompts and whether they share the F10 flip axis (blocked in part on the
prompt-library restoration, issue #9); the shape-class-matched coherence null and
its application to the 125-sweep (F12, caveat 10); the phase-aware J-lens full
build (F11, issue #8); hook-window/depth dependence (caveat 6); gate cadence
(caveat 5); H4.

## 6. Stage boundary: why the series closed with work unexecuted

The series was scoped by a question, not a task list: *is the GPT-2 Small result
real, and does the fingerprint hypothesis survive validation?* Both parts are now
answered (yes; no) and published. Work that was planned but not executed falls into
three classes, deliberately:

1. **Retired with the hypothesis.** The large cross-model scaling programme and
   bias-profiling work (ATR_METHOD_COMPARISON §3) existed to extend the fingerprint
   claim. The claim was refuted before they ran; executing them for that purpose
   would have been waste. They survive only as re-motivated characterisation work.
2. **Transferred to the next question.** The depth control (caveat 6), per-layer /
   per-head decomposition, the spectral test (H4), and readout upgrades do not test
   whether the result is real; they test *why the models differ*. That is the
   successor project's question. Their scaffolds are retained, labelled not-run, as
   pre-registration.
3. **Declared debt.** One item remains open: finer convergence-gate cadence
   (caveat 5). It cannot overturn a principal finding: basin identities stand on
   the gate regardless of cadence. (The other declared item, the W_E permutation
   test, was paid at close: negative, see caveat 4. It withdrew the all-warm
   supporting evidence for H3 without touching F1–F5.)
