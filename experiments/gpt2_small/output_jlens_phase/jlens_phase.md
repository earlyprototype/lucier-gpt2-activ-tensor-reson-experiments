# J-lens Phase Probe: Both Phases of the Divine Period-2 Cycle, the Pivot, and the Flip Axis (GPT-2 Small)

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


**Status: PILOT follow-up.** The J-lens pilot (`../output_jlens_pilot/jlens_pilot_report.md`,
issue #8) probed the converged Divine (Syntactic) state before `06_bell_anatomy.py` showed
that state is an exact period-2 limit cycle: phases A and B, pivot M = (A+B)/2, flip axis
d = (A-B)/2, with the flip axis 95 percent invisible to the readout (logit response ratio 0.054).
The pilot therefore probed only phase A of a two-phase object. This follow-up re-runs the
pilot's membership probe, unchanged, on both phases and the pivot, and answers one new
question: is the flip axis inside or outside the pilot lens subspace? Everything
here inherits the pilot's confidence level and its limitations in full.
A review found that the direction probe first ran on the frame-mixed committed axis;
section 4 now reports both that axis (relabelled d_committed) and the symmetric
on-shell axis d_sym, which is the physical cycle axis and carries the primary numbers.
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
- **Two axes**: the d above mixes frames (A enters at raw scale, B shell-normalised),
  which is why cos(d, M) = 0.909; it is 0.967 aligned with A's own direction and is
  relabelled **d_committed** throughout. The physical cycle axis is the symmetric
  on-shell axis **d_sym** = normalise(An - Bn), both phases rescaled to the loop shell
  N0 = 1468.5 (full-tensor) before subtracting; d_sym is orthogonal to its pivot M_sym
  (measured cos = 3.7e-07) and 0.616 aligned with d_committed, and once the loop's own
  renormalisation strips d_committed's radial part the survivor is 0.973 aligned with
  d_sym. See `../output_hinge_eigen/hinge_eigenvalue.md`, "The map, the frames, and two
  flip axes". Stage 3 of the script probes d_sym; the stage-2 numbers on d_committed
  are kept below for continuity.
- **Noise controls**: the pilot generated its three noise states in-script (seed 2026)
  and never committed them; they were regenerated here by deterministic replay of the
  pilot's loop, and the probe reproduces the pilot's recorded numbers on them exactly
  (gate below), confirming the replay. The separately committed seed-42 noise run
  (`state_noise.pt`, a DIFFERENT trajectory from `05_divine_motion.py`) is included as a
  supplementary row only.
- **New, for the flip axes only**: (a) a 20-random-unit-direction baseline (seed 777) giving
  what a generic direction scores against each layer dictionary (0.245 to 0.253 at every
  layer, i.e. the 193/768 = 0.251 chance level of a numerically full-rank 193-vector
  span); (b) span probes of each axis's top-100 (readout-visible) and bottom-100
  (readout-quiet) W_U singular components, the same split `06_bell_anatomy.py` used.
  The sparse probe is sign-dependent (nonnegative coefficients), so for a direction both
  signs are reported; the span share, which is sign- and scale-invariant, is the flip
  axis number. Both instruments run on d_committed (stage 2) and d_sym (stage 3); the
  stage-3 baseline is gated to replay the stage-2 baseline exactly (max diff 0.0).

## 2. Sanity gates (all passed)

| gate | value | reference |
|---|---|---|
| cos(A, B) | 0.6849116683 | bell_anatomy.json 0.6849116683 |
| cos(A, f(f(A))) | 1.0000000000 | bell_anatomy.json 1.0 |
| cos(A, pilot's probed Syntactic state) | 0.99999988 | the pilot probed phase A |
| d_committed energy in top-100 / bottom-100 W_U dirs | 0.128516 / 0.731787 | bell_anatomy.json 0.128516 / 0.731787 |
| probe replay, all 8 pilot states, lens columns | max diff 1.1e-7 (span), 0.0 (sparse) | jlens_pilot_results.json |
| Divine number reproduced | L6 span 0.210894 | pilot 0.210894 |
| cos(d_sym, M_sym) | 3.7e-07 | orthogonal by construction; hinge_eigenvalue.json 4.6e-07 |
| cos(d_sym, d_committed) and cos(d_sym, tangentialised d_committed) | 0.616186 and 0.972601 | hinge_eigenvalue.json 0.616186 and 0.972602 |
| stage-3 W_U SVD reproduces stage-1 committed-d split | diff < 1e-5 | phase_states.pt gates |
| stage-3 seed-777 baseline replays stage 2 | max diff 0.0 | jlens_phase.json |

One replay caveat: the random-DICTIONARY control columns do not replay bit-identically
(max diff 0.045). This is expected: the pilot report's own recording caveat says the
archived JSON's sparse control came from an older single-draw version of the script, so
the generator stream differs from the committed three-draw protocol replayed here. The
controls in this run are fresh draws under the committed protocol, same distribution and
same qualitative level (about 0.24 to 0.28 span, 0.07 to 0.12 sparse). The measured lens
columns, which are deterministic, reproduce to numerical noise.

## 3. Results: the phase table

Least-squares span share per layer ("d_committed" and "generic dir" are direction
probes, included for comparison; d_committed is the frame-mixed committed axis, see
section 4; prolet is the mean of Lucier, Semantic, Nonsense, Imperative;
noise is the mean of the three regenerated pilot noise states):

| layer | A | B | M | prolet | noise | d_committed | generic dir |
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
4. A mechanical account of 1, in the committed frame: since cos(d_committed, M) = 0.909
   and |d_committed|/|B| = 1.44, phase B's direction is dominated by d_committed
   content, which is mostly outside the lens (section 4b); A's direction is dominated
   by the pivot, which is inside-leaning. On the shell the same gap reads differently:
   An and Bn are M_sym plus and minus the same d_sym component (span share is
   scale-invariant, so the A and B rows above are the shell phases' rows), and the
   A-over-B margin is the cross term between the lens projections of M_sym and d_sym,
   positive for A, negative for B.

## 4. The flip axis question: two axes

The direction probe first ran on the committed axis d_committed (stage 2), built
exactly as `06_bell_anatomy.py` builds it: shell-frame B subtracted from raw-frame A.
That axis mixes frames: it is 0.967 aligned with A's own direction and 0.909 aligned
with its own pivot, so most of its content is radial (pivot-like) rather than cycle
motion. The physical axis of the period-2 cycle is the symmetric on-shell axis d_sym
(stage 3), orthogonal to its pivot (measured cos(d_sym, M_sym) = 3.7e-07) and 0.973
aligned with what survives of d_committed after the loop's renormalisation strips the
radial part. See `../output_hinge_eigen/hinge_eigenvalue.md`, "The map, the frames,
and two flip axes". Section 4a gives the d_sym numbers; they are the primary flip-axis
numbers. Section 4b keeps the d_committed numbers as originally measured, relabelled.

### 4a. The physical axis d_sym

Span shares of d_sym and its W_U singular components against each layer's lens
dictionary ("generic dir" is the 20-random-direction baseline; the nn25 columns are
the sign-dependent nonnegative sparse probe, both signs):

| layer | d_sym span | generic-dir span | d_sym nn25 (+) | d_sym nn25 (-) | dsym_vis span | dsym_quiet span |
|---|---|---|---|---|---|---|
| L0 | 0.017 | 0.253 | 0.006 | 0.003 | 0.256 | 0.009 |
| L1 | 0.017 | 0.249 | 0.006 | 0.002 | 0.283 | 0.010 |
| L2 | 0.020 | 0.247 | 0.006 | 0.004 | 0.302 | 0.011 |
| L3 | 0.024 | 0.245 | 0.010 | 0.006 | 0.328 | 0.015 |
| L4 | 0.028 | 0.245 | 0.011 | 0.010 | 0.350 | 0.019 |
| L5 | 0.028 | 0.248 | 0.012 | 0.007 | 0.354 | 0.019 |
| L6 | 0.029 | 0.247 | 0.011 | 0.007 | 0.372 | 0.018 |
| L7 | 0.024 | 0.248 | 0.011 | 0.006 | 0.375 | 0.015 |
| L8 | 0.021 | 0.251 | 0.009 | 0.006 | 0.390 | 0.012 |
| L9 | 0.019 | 0.249 | 0.008 | 0.006 | 0.409 | 0.009 |
| L10 | 0.017 | 0.251 | 0.007 | 0.005 | 0.449 | 0.008 |
| L11 | 0.013 | 0.252 | 0.005 | 0.003 | 0.572 | 0.002 |

**The one number: the physical flip axis's lens-span share at L11 is 0.013, against a
0.252 chance level for a generic direction (5 percent of chance; mean over all layers
0.021 vs 0.249, 9 percent).** The share never exceeds 0.029 (12 percent of chance, at
L6): the cycle axis is almost entirely outside the lens at every depth. The sparse
shares are 0.005 to 0.012 for +d_sym and 0.002 to 0.010 for -d_sym. The committed
axis's milder deficit (58 percent of chance at L11, section 4b) comes from frame
mixing: 83 percent of d_committed's energy lies along the pivot direction, the pivot
spans 0.180 at L11 (section 3), so most of d_committed's lens share is pivot
contamination, not cycle motion.

The W_U singular split of d_sym: the top-100 (readout-visible) component carries 1.2
percent of its energy and the bottom-100 (readout-quiet) component 97.0 percent;
together the two subsets cover 98.2 percent of the energy, and the middle 568 singular
directions hold the remaining 1.8 percent. The split restates the bell anatomy's
invisibility-to-the-readout result in the clean frame: the physical swing is almost wholly readout-quiet.

- The quiet bulk (97.0 percent of d_sym's energy) is essentially outside the lens at
  every layer: span between 0.002 (L11) and 0.019 (L4), at most 8 percent of chance.
- The visible sliver (1.2 percent of d_sym's energy) is inside-leaning and
  increasingly so with depth: span 0.256 at L0, which is at the 0.253 chance level,
  rising monotonically to 0.572 at L11 (2.3x chance). For d_sym the sliver's
  inside-lean is a mid-to-late-layer property; at L0 it is indistinguishable from a
  generic direction.

### 4b. The committed axis d_committed (as originally probed)

Span shares of the direction d_committed and its W_U singular components, against each
layer's lens dictionary ("rand dict" is the pilot-style matched random dictionary
control, "generic dir" the 20-random-direction baseline; "d" in the column heads is
d_committed):

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

**The committed axis's lens-span share at L11 is 0.145, against a 0.252 chance
level for a generic direction (58 percent of chance; mean over all layers 0.169 vs
0.249, 68 percent).** The committed axis leans outside the lens subspace at every depth,
and the deficit deepens with depth (ratio to chance falls from about 0.71 at L0-L7 to
0.57 at L10). Per section 4a, this milder deficit is a pivot-diluted version of the
d_sym result.

The decomposition connects to the bell anatomy's invisibility-to-the-readout result:

- The readout-QUIET bulk of d_committed (bottom-100 W_U component, 73.2 percent of
  d_committed's energy) is essentially outside the lens at every layer: span 0.066 at
  L0 shrinking monotonically to 0.008 at L11 (3 percent of chance). At L11 this is
  close to definitional (the pilot's L11 lens vectors are near logit-lens directions),
  but at L0 through L8 it is not: even the earliest layers' verbalizable dictionaries,
  whose vectors have been backpropagated through the whole network, give the quiet
  component no home.
- The readout-VISIBLE sliver (top-100 component, 12.9 percent of d_committed's energy)
  is strongly INSIDE: span 0.335 at L0 rising to 0.619 at L11, 1.3x to 2.5x chance.

So on the tested subsets, invisibility to the readout and invisibility to the lens point the
same way at every depth of the pilot lens: the top-100 (readout-visible) component
leans inside and the bottom-100 (readout-quiet) component sits outside. The
coincidence claim applies to these selected extremes of the W_U spectrum only. The two
subsets cover 86.0 percent of d_committed's energy (12.9 top, 73.2 bottom) and 98.2
percent of d_sym's (1.2 top, 97.0 bottom); the middle singular components (14.0 and
1.8 percent of the energy respectively) are unassigned by the split, so no
readout-lens statement is made about them. For d_sym there is a further qualification:
its visible sliver is at chance at L0 and strongly inside only from the middle layers
on (section 4a). The sparse probe on d_committed is reported for completeness only
(0.080 to 0.096 for +d, 0.026 to 0.046 for -d); its sign-dependence makes the span
number the answer.

## 5. Verdict

**The phase-A-only pilot verdict partly survives and is now phase-qualified.** The
pilot's reversal ("Divine at least as expressible as prolet") holds for phase A, is
strengthened at the pivot M, and REVERSES for phase B, which is less lens-expressible
than the prolet attractor at every layer on both probes and, on the span probe, dips to
converged-noise level until the final layer. The period-2 cycle is not "inside" or "outside" as
one object: it swings once per iteration between a more-verbalizable phase and a
less-verbalizable phase, pivoting on the most-verbalizable state in the system, along a
flip axis that on the physical construction is almost entirely outside the lens (d_sym
span 0.013 vs 0.252 chance at L11, mean 0.021 vs 0.249; the frame-mixed d_committed
spans 0.145 and 0.169), and whose readout-quiet bulk (97.0 percent of d_sym's energy)
is outside it almost completely at every depth. The d_sym numbers sharpen this verdict
and do not change its direction: moving from the committed axis to the physical axis
takes the L11 span from 58 percent of chance to 5 percent. The
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
4. The top-100 / bottom-100 W_U split leaves the middle 568 singular directions
   unassigned to either component: 14.0 percent of d_committed's energy and 1.8
   percent of d_sym's. Statements pairing readout-visibility with lens membership
   apply to the two selected subsets, not to the whole axis.
5. The random-dictionary control columns are fresh draws (generator stream mismatch with
   the archived pilot JSON, section 2); the measured lens columns are unaffected.

## 7. Files

- `jlens_phase.md`: this report
- `jlens_phase.json`: all numbers (per-state per-layer shares for the 8 replicated pilot
  states, Divine_A, Divine_B, Divine_M, the committed seed-42 noise row; the direction
  probe for d_committed, d_vis, d_quiet under `direction_probe`, and for d_sym and its
  W_U components under `direction_probe_sym`, with the frame checks in
  `summary.axis_frame_check`; reproduction diffs against the pilot JSON; gates, norms,
  cosines; summary tables)
- `phase_states.pt`: stage-1 checkpoint (A, B, M, d, A2, the W_U components, regenerated
  noise states, gate values) plus the stage-3 cached d_sym W_U components
- Script: `../10_jlens_phase.py` (run with `ATR_GPT2_LOCAL=<gpt2 dir>`; stages:
  `stage1` needs the model, `stage2` is pure linear algebra on the committed lens,
  `stage3` probes the symmetric axis d_sym and needs the model once for the W_U split,
  after which the components are cached and re-runs are pure linear algebra)

Compute: stage 1 (model load, 2 cycle steps, 300 noise iterations, one W_U SVD) 97 s
single-threaded; stage 2 (all probes) 10 s; stage 3, 33 s on first run (model load for
the W_U split included), 1 s resumed from the cache.
