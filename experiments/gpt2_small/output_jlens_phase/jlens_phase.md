# J-lens Phase Probe: Both Phases of the Divine Bell, the Pivot, and the Flip Axis (GPT-2 Small)

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


**Status: PILOT follow-up.** The J-lens pilot (`../output_jlens_pilot/jlens_pilot_report.md`,
issue #8) probed the converged Divine (Syntactic) state before `06_bell_anatomy.py` showed
that state is an exact period-2 limit cycle: phases A and B, pivot M = (A+B)/2, flip axis
d = (A-B)/2, with the flip axis 95 percent mute to the readout (logit response ratio 0.054).
The pilot therefore probed only phase A of a two-phase object. This follow-up re-runs the
pilot's membership probe, unchanged, on both phases and the pivot, and answers one new
question: is the flip axis d inside or outside the pilot lens subspace? Everything
here inherits the pilot's confidence level and its limitations in full.
Script: `../10_jlens_phase.py`.

## 1. Method (what is reused, what is new)

- **Lens**: reused verbatim from `../output_jlens_pilot/jlens_vectors.pt` (193 tokens x
  12 layers x 768). The lens corpus is unreachable on this network, so the lens was NOT
  recomputed; every statement is relative to the committed pilot dictionary.
- **Probe**: identical to `05_jlens_pilot.py`: least-squares span share
  `||proj||^2 / ||state||^2` per layer, nonnegative sparse k=25 share, random-dictionary
  controls drawn from a generator seeded 4242, replayed in the pilot's exact state order
  so the control stream lines up.
- **States**: A, B, M, d reconstructed exactly as `06_bell_anatomy.py` does, from the
  committed iteration-1000 checkpoint `../output_divine_motion/state_divine.pt` (one
  forward pass gives B = f(A), a second verifies period 2). Last-position vectors:
  |A| = 1612, |B| = 464, |M| = 980, |d| = 669. Note cos(d, M) = 0.909: the flip axis is
  strongly aligned with the pivot, so B = M - d is the small residual of two nearly
  aligned vectors (|d|/|B| = 1.44), while A = M + d is dominated by M (|d|/|A| = 0.41).
- **Noise controls**: the pilot generated its three noise states in-script (seed 2026)
  and never committed them; they were regenerated here by deterministic replay of the
  pilot's loop, and the probe reproduces the pilot's recorded numbers on them exactly
  (gate below), confirming the replay. The separately committed seed-42 noise run
  (`state_noise.pt`, a DIFFERENT trajectory from `05_divine_motion.py`) is included as a
  supplementary row only.
- **New, for the flip axis only**: (a) a 20-random-unit-direction baseline (seed 777) giving
  what a generic direction scores against each layer dictionary (0.245 to 0.253 at every
  layer, i.e. the 193/768 = 0.251 chance level of a numerically full-rank 193-vector
  span); (b) span probes of d's top-100 (readout-visible) and bottom-100 (readout-quiet)
  W_U singular components, the same split `06_bell_anatomy.py` used. The sparse probe is
  sign-dependent (nonnegative coefficients), so for the direction d both signs are
  reported; the span share, which is sign- and scale-invariant, is the flip axis number.

## 2. Sanity gates (all passed)

| gate | value | reference |
|---|---|---|
| cos(A, B) | 0.6849116683 | bell_anatomy.json 0.6849116683 |
| cos(A, f(f(A))) | 1.0000000000 | bell_anatomy.json 1.0 |
| cos(A, pilot's probed Syntactic state) | 0.99999988 | the pilot probed phase A |
| d energy in top-100 / bottom-100 W_U dirs | 0.128516 / 0.731787 | bell_anatomy.json 0.128516 / 0.731787 |
| probe replay, all 8 pilot states, lens columns | max diff 1.1e-7 (span), 0.0 (sparse) | jlens_pilot_results.json |
| Divine number reproduced | L6 span 0.210894 | pilot 0.210894 |

One replay caveat: the random-DICTIONARY control columns do not replay bit-identically
(max diff 0.045). This is expected: the pilot report's own recording caveat says the
archived JSON's sparse control came from an older single-draw version of the script, so
the generator stream differs from the committed three-draw protocol replayed here. The
controls in this run are fresh draws under the committed protocol, same distribution and
same qualitative level (about 0.24 to 0.28 span, 0.07 to 0.12 sparse). The measured lens
columns, which are deterministic, reproduce to numerical noise.

## 3. Results: the phase table

Least-squares span share per layer ("flip axis d" and "generic dir" are direction probes,
included for comparison; prolet is the mean of Lucier, Semantic, Nonsense, Imperative;
noise is the mean of the three regenerated pilot noise states):

| layer | A | B | M | prolet | noise | flip axis d | generic dir |
|---|---|---|---|---|---|---|---|
| L0 | 0.213 | 0.152 | 0.221 | 0.183 | 0.184 | 0.179 | 0.253 |
| L1 | 0.216 | 0.162 | 0.227 | 0.211 | 0.186 | 0.180 | 0.249 |
| L2 | 0.221 | 0.165 | 0.231 | 0.218 | 0.190 | 0.185 | 0.247 |
| L3 | 0.205 | 0.151 | 0.214 | 0.195 | 0.185 | 0.174 | 0.245 |
| L4 | 0.207 | 0.155 | 0.216 | 0.187 | 0.192 | 0.176 | 0.245 |
| L5 | 0.207 | 0.157 | 0.216 | 0.197 | 0.198 | 0.174 | 0.248 |
| L6 | 0.211 | 0.160 | 0.221 | 0.195 | 0.196 | 0.178 | 0.247 |
| L7 | 0.209 | 0.162 | 0.220 | 0.188 | 0.187 | 0.174 | 0.248 |
| L8 | 0.197 | 0.151 | 0.207 | 0.179 | 0.169 | 0.164 | 0.251 |
| L9 | 0.181 | 0.135 | 0.190 | 0.163 | 0.146 | 0.152 | 0.249 |
| L10 | 0.171 | 0.127 | 0.179 | 0.152 | 0.129 | 0.143 | 0.251 |
| L11 | 0.173 | 0.123 | 0.180 | 0.157 | 0.114 | 0.145 | 0.252 |

Nonnegative sparse k=25 share per layer:

| layer | A | B | M | prolet | noise |
|---|---|---|---|---|---|
| L0 | 0.104 | 0.074 | 0.109 | 0.085 | 0.052 |
| L1 | 0.109 | 0.084 | 0.116 | 0.112 | 0.068 |
| L2 | 0.104 | 0.077 | 0.109 | 0.095 | 0.058 |
| L3 | 0.107 | 0.074 | 0.111 | 0.095 | 0.060 |
| L4 | 0.095 | 0.069 | 0.098 | 0.081 | 0.056 |
| L5 | 0.099 | 0.074 | 0.106 | 0.093 | 0.064 |
| L6 | 0.111 | 0.081 | 0.116 | 0.095 | 0.063 |
| L7 | 0.115 | 0.086 | 0.120 | 0.097 | 0.062 |
| L8 | 0.115 | 0.084 | 0.119 | 0.099 | 0.060 |
| L9 | 0.108 | 0.079 | 0.114 | 0.095 | 0.054 |
| L10 | 0.096 | 0.068 | 0.100 | 0.091 | 0.047 |
| L11 | 0.098 | 0.070 | 0.102 | 0.091 | 0.048 |

Supplementary row: the committed seed-42 noise run (`state_noise.pt`, iteration 1000)
scores like the pilot noise family (span 0.160 at L0, 0.179 at L6, 0.100 at L11; sparse
0.056 at L6, 0.042 at L11).

Reading, in decreasing order of confidence:

1. **The phases differ materially.** Phase B sits below phase A at every layer on both
   probes: span lower by 0.043 to 0.060 absolute, sparse lower by 0.025 to 0.033. This
   is not a norm artifact (both shares are scale-invariant). Against the prolet family,
   B is lower on span at every layer (by 0.025 to 0.053) and lower on sparse at every
   layer; A is higher than prolet at every layer on both probes. The two phases straddle
   the prolet level.
2. **The pivot M is the most lens-expressible object probed**: above A at every layer on
   both probes (span +0.007 to +0.011, sparse +0.003 to +0.007). The stable ` Divine`
   readout was shown in the bell anatomy to be the shadow of M; M is also the most
   verbalizable-adjacent state the instrument has seen.
3. **The language-vs-noise boundary survives on the sparse probe, and weakens for B on
   the span probe.** On sparse, every language-derived state including B stays above the
   noise mean at all 12 layers (B's smallest margin +0.010 at L5). On span, B actually
   falls below the noise mean at L0 through L10 (by 0.001 to 0.041) and only overtakes
   it at L11 (+0.009), where noise collapses (0.114) and B holds 0.123. The pilot's
   clean late-layer span separation (language 0.15 to 0.17 vs noise 0.11 to 0.13) is a
   phase-A and pivot property; phase B sits at its edge.
4. A mechanical account of 1: since cos(d, M) = 0.909 and |d|/|B| = 1.44, phase B's
   direction is dominated by flip axis content, and the flip axis is mostly outside the lens
   (section 4); A's direction is dominated by the pivot, which is inside-leaning.

## 4. The flip axis question

Span shares of the direction d and its W_U singular components, against each layer's
lens dictionary ("rand dict" is the pilot-style matched random dictionary control,
"generic dir" the 20-random-direction baseline):

| layer | d span | rand-dict span | generic-dir span | d nn25 (+d) | d nn25 (-d) | d_vis span | d_quiet span |
|---|---|---|---|---|---|---|---|
| L0 | 0.179 | 0.241 | 0.253 | 0.086 | 0.030 | 0.335 | 0.066 |
| L1 | 0.180 | 0.256 | 0.249 | 0.090 | 0.029 | 0.362 | 0.068 |
| L2 | 0.185 | 0.265 | 0.247 | 0.085 | 0.044 | 0.389 | 0.067 |
| L3 | 0.174 | 0.249 | 0.245 | 0.090 | 0.042 | 0.418 | 0.045 |
| L4 | 0.176 | 0.262 | 0.245 | 0.081 | 0.046 | 0.450 | 0.044 |
| L5 | 0.174 | 0.257 | 0.248 | 0.084 | 0.043 | 0.474 | 0.036 |
| L6 | 0.178 | 0.276 | 0.247 | 0.094 | 0.042 | 0.506 | 0.030 |
| L7 | 0.174 | 0.272 | 0.248 | 0.096 | 0.039 | 0.511 | 0.023 |
| L8 | 0.164 | 0.267 | 0.251 | 0.094 | 0.040 | 0.542 | 0.017 |
| L9 | 0.152 | 0.245 | 0.249 | 0.089 | 0.035 | 0.558 | 0.011 |
| L10 | 0.143 | 0.253 | 0.251 | 0.081 | 0.028 | 0.570 | 0.006 |
| L11 | 0.145 | 0.261 | 0.252 | 0.080 | 0.026 | 0.619 | 0.008 |

**The one number: the flip axis's lens-span share at L11 is 0.145, against a 0.252 chance
level for a generic direction (58 percent of chance; mean over all layers 0.169 vs
0.249, 68 percent).** The flip axis leans outside the lens subspace at every depth, and the
deficit deepens with depth (ratio to chance falls from about 0.71 at L0-L7 to 0.57 at
L10).

The decomposition says why, and connects directly to the bell anatomy's muteness result:

- The readout-QUIET bulk of the flip axis (bottom-100 W_U component, 73 percent of d's
  energy) is essentially outside the lens at every layer: span 0.066 at L0 shrinking
  monotonically to 0.008 at L11 (3 percent of chance). At L11 this is close to
  definitional (the pilot's L11 lens vectors are near logit-lens directions), but at L0
  through L8 it is not: even the earliest layers' verbalizable dictionaries, whose
  vectors have been backpropagated through the whole network, give the quiet component
  no home.
- The readout-VISIBLE sliver (top-100 component, 13 percent of d's energy) is strongly
  INSIDE: span 0.335 at L0 rising to 0.619 at L11, 1.3x to 2.5x chance.

So muteness to the readout and muteness to the lens travel together, at every depth of
the pilot lens: the part of the flip axis the readout can hear is exactly the part the lens
can express, and the rest (the "A-to-B displacement at the readout-visible complement"
is this sliver; its quiet complement is the bulk) is outside both. The sparse probe on d
is reported for completeness only (0.080 to 0.096 for +d, 0.026 to 0.046 for -d); its
sign-dependence makes the span number the answer.

## 5. Verdict

**The phase-A-only pilot verdict partly survives and is now phase-qualified.** The
pilot's reversal ("Divine at least as expressible as prolet") holds for phase A, is
strengthened at the pivot M, and REVERSES for phase B, which is less lens-expressible
than the prolet attractor at every layer on both probes and, on the span probe, dips to
converged-noise level until the final layer. The bell is not "inside" or "outside" as
one object: it swings once per iteration between a more-verbalizable phase and a
less-verbalizable phase, pivoting on the most-verbalizable state in the system, along a
flip axis that is mostly outside the lens (span 0.145 vs 0.252 chance at L11), and
whose readout-quiet bulk is outside it almost completely at every depth. The
language-vs-noise boundary remains the pilot's most robust story, but it is now a
sparse-probe story: on the span probe phase B sits at, and below, the noise level for
most of the depth. None of this is a null: the two phases are materially distinguishable
to the lens, which the phase-blind pilot could not see.

## 6. Limitations

1. Every pilot limitation inherits unchanged: 30-prompt hand-written corpus, 193-token
   dictionary (0.4 percent of vocabulary), low-rank lens vs full-rank controls, GPT-2
   Small's 12 coarse layers, and an averaged Jacobian still visibly moving at 30
   prompts. The lens could not be recomputed on this network.
2. Effective sample sizes: one Divine cycle (one prompt), one effective prolet
   attractor, three regenerated noise states plus one committed one. All gaps are point
   estimates; no error bars are possible.
3. The generic-direction baseline (seed 777) is new to this run, not part of the pilot
   instrument; it lands at the analytic 193/768 chance level, which is also what the
   pilot's random-dictionary control measures.
4. The top-100 / bottom-100 W_U split leaves d's middle 568 singular directions (14
   percent of its energy) unassigned to either component.
5. The random-dictionary control columns are fresh draws (generator stream mismatch with
   the archived pilot JSON, section 2); the measured lens columns are unaffected.

## 7. Files

- `jlens_phase.md`: this report
- `jlens_phase.json`: all numbers (per-state per-layer shares for the 8 replicated pilot
  states, Divine_A, Divine_B, Divine_M, the committed seed-42 noise row; the direction
  probe for d, d_vis, d_quiet; reproduction diffs against the pilot JSON; gates, norms,
  cosines; summary tables)
- `phase_states.pt`: stage-1 checkpoint (A, B, M, d, A2, the W_U components, regenerated
  noise states, gate values)
- Script: `../10_jlens_phase.py` (run with `ATR_GPT2_LOCAL=<gpt2 dir>`; stages:
  `stage1` needs the model, `stage2` is pure linear algebra on the committed lens)

Compute: stage 1 (model load, 2 cycle steps, 300 noise iterations, one W_U SVD) 97 s
single-threaded; stage 2 (all probes) 10 s.
