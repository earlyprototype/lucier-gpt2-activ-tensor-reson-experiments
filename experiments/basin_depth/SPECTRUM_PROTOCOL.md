# The local Jacobian spectrum of the ATR loop map at settled attractors

Registration document. Written before execution. Nothing in this document is a
result. The one number quoted as measured is the smoke test in section 12,
which reproduces prior art and establishes only that the machinery runs.

## 0. Limits, up front

Read these before the motivation, because several of them bound what the
experiment can possibly show.

1. This measures the leading 24 of roughly 7679 eigenvalues. Nothing is
   learned about the bulk. No trace, no determinant, no dimension count, no
   claim about the spectrum as a whole.
2. A Jacobian is a local linear object. Basin boundaries are global and
   nonlinear. A spectrum cannot locate a boundary. The prediction registered in
   section 9 is precisely the claim that the local object nevertheless orders
   the global escape thresholds, and that claim may simply be false.
3. One model, GPT-2 Small, CPU, float32 throughout. The matvec carries a
   relative error near 1e-6, so eigenvalues below about 1e-3 in modulus are
   reported as "below the numerical floor" and not as numbers.
4. The states are the escape ladder's states: two prompts per label per level,
   selected from the nu sweep's checkpoints. That is a probe, not a sample.
   FINDINGS caveat 19 applies to every cross-state comparison.
5. Convergence is only ever to the engine gate's tolerance. This protocol
   refines each base point and reports its fixed-point residual, but a state
   that is really a very slow drift will present as a fixed point carrying an
   eigenvalue at 1, and the spectrum cannot tell those apart.
6. Period detection is by lag scan to lag 8. A period above 8 aliases, exactly
   as F15 records for the lag-2 gate one octave down.
7. Every `Divine` number inherits caveat 14: one audited trajectory, not 34.
8. The cross-experiment analysis in section 9 is partly blocked. The escape
   ladder does not archive its random direction tensors and cannot regenerate
   them (section 9.2). This was discovered while designing the protocol and is
   not a defect of the ladder's own result.

## 1. What this measures and why

The loop rescales every iterate to a fixed Frobenius norm, so the state never
leaves a sphere of dimension (positions times 768) minus 1, about 7679 for a
ten-position prompt. On that sphere a settled attractor is a fixed point or a
short cycle of the induced map. Near it the map linearises to a matrix, and the
eigenvalues of that matrix say what the dynamics do in each direction:
modulus below 1 contracting, above 1 expanding, negative real flipping sides
each pass (the configuration that produces the period-2 `Divine` cycle),
complex spiralling.

The record already contains exactly one such number, measured in one direction:
F14's flip-axis eigenvalue. Everything else about local stability in this
project is inferred from behaviour, not measured. This experiment measures the
leading part of the spectrum properly, at several attractors, and uses it to
make one prediction that a separate running experiment can refute.

## 2. Prior art

`experiments/gpt2_small/08_hinge_eigenvalue.py` and its output
`output_hinge_eigen/hinge_eigenvalue.json`. It computes directional derivatives
of the loop map by `torch.func.jvp` on

```
f(x) = G( nu * x / ||x||_F )
```

where `G` is the block cascade 0 to 11 applied to a [T, 768] tensor injected at
`blocks.0.hook_resid_pre`, and `nu` is the run's Frobenius pin. The injection
overwrites every position, so the cascade is the whole forward map from that
point and is prompt-independent given the sequence length. The script verifies
this against the hook path to zero error, and this protocol repeats that check.

What it reports per direction `t` (unit) at a base point `x`:
`lambda = <t, J_f(x) t>`, the amplification `||J_f(x) t||`, and
`cos(J_f(x) t, -t)`. Central finite differences at `eps_rel` 1e-3 and 1e-4
serve as robustness checks. Its headline numbers:

| object | quantity | value |
|:--|:--|--:|
| `d_committed` at `Mn_committed` | `<t, J_f t>` | -0.863580 |
| `d_sym` at `Mn_sym` | `<t, J_f t>` | -4.275078 |
| `d_sym`, composed A to B to A | `<t, J_f(B) J_f(A) t>` | +0.099339 |

This protocol reuses that JVP machinery unchanged and adds three things it did
not have: the correct on-shell linearisation (section 4), the composed map as
the object for cycles (section 5), and a matrix-free eigensolver over the whole
tangent space instead of a handful of named directions (section 8).

## 3. The map, exactly

Let `nu` be the pin the state was settled at (the escape ladder's `target`;
`renorm="seed_j"` at the historical level, a numeric pin at the multiplier
levels). The loop's state at injection time is always on the shell
`||y||_F = nu`, so the dynamics the loop actually runs is the **on-shell map**

```
S(y) = nu * G(y) / ||G(y)||_F        for  ||y||_F = nu
```

and it relates to the prior art's `f` by `S(y) = nu * f(y) / ||f(y)||`, since
`f(y) = G(y)` when `y` is already on the shell.

Base points are constructed to be the same objects the escape ladder perturbs:
`y = nu * x_hat`, where `x` is the raw converged tensor returned by
`converged_state` and `nu` is that call's `target`. The ladder rotates that
same on-shell point, so the two experiments share a base point exactly and no
frame conversion is needed between them.

## 4. Design question 1: the on-shell linearisation and the radial projection

There are three distinct corrections, and each has its own reason.

**(a) Input projection.** Differentiating `f`:

```
J_f(x) = J_G(x_n) * (nu / ||x||) * P_x ,    P_x = I - x_hat x_hat^T
```

so `J_f(x) x = 0` exactly. A purely radial perturbation is undone by the
rescale before the network sees anything, and the derivative has that null
direction built in. This is automatic if you differentiate `f` rather than `G`,
which is why the map under test must include the renormalisation step and not
be the raw forward pass. It is nonetheless enforced explicitly on every input
vector, because an eigensolver let loose on the full ambient space will find the
radial null vector, return a spurious eigenvalue 0, and spend Krylov dimension
on it. **Gate G2** measures the null directly:
`||J_f(y) y_hat|| / ||J_f(y) t||` must be below 1e-4.

**(b) Output projection.** The response `J_f(y) v` is not tangent to the shell
at the image point. The loop's next rescale removes its radial part. So

```
DS(y) v = ( nu / ||f(y)|| ) * Q_{f(y)} * J_f(y) v ,    Q_z = I - z_hat z_hat^T
```

with `v` tangent at `y`. At a fixed point `f(y*)` is parallel to `y*`, so
`Q_{f(y*)} = P_{y*}` and `DS(y*)` maps the tangent space at `y*` into itself.
Only then is it an operator with eigenvalues. Without the output projection the
operator does not map the tangent space to itself and its "eigenvalues" are not
multipliers of the sphere dynamics at all.

**(c) Output scale.** The factor `nu / ||f(y)||` is the derivative of the
magnitude-restoring rescale and is not optional; it is typically far from 1
(measured 0.294 at the `Divine` committed pivot), so omitting it rescales the
whole spectrum.

The prior art omitted (b) and (c): its numbers are Rayleigh quotients of `J_f`,
not multipliers of `S`. That is a defensible convention for a single named
direction and an indefensible one for a spectrum. **Both conventions are
reported for every eigenvector**, labelled `lambda_onshell` and
`lambda_rayleigh_Jf`, so the new numbers are comparable with the committed
record in either direction.

One consequence worth stating plainly. `Q` and `P` coincide only at a fixed
point. At a point the map does not fix, `DS` is not an endomorphism and has no
eigenvalues. The `Divine` pivot is such a point (this protocol's smoke test
measures `cos(f(M), M) = 0.851` for the committed pivot). F14's -4.3 is
therefore a Rayleigh quotient of a half map at a point the loop does not fix.
It is a real and informative number about the local flow, and it is not an
eigenvalue of anything in this protocol. This protocol does not reproduce it and
does not need to.

## 5. Design question 2: one pass or the composed two-pass map

Determine the period first, then choose the object. Never the other way round.

**Period detection.** Continue the settled state for 9 further passes, compute
`atr_engine.lag_scan` on the mean vectors, and take `p` = the smallest lag whose
mean cosine exceeds the engine threshold 0.999. This is F15's rule, and using
anything else reintroduces the aliasing F9 and F15 document.

**`p = 1` (fixed point).** Linearise the one-pass map: the operator is `DS(y*)`
on the tangent space at `y*`.

**`p = 2` (cycle, the `Divine` case).** The one-pass map is not a fixed point at
all. It maps phase A to phase B, so `DS(A)` maps the tangent space at A to the
tangent space at B and has no eigenvalues. The right object is the composed map
`S2 = S o S`, whose derivative at A is

```
DS2(A) = DS(B) DS(A) ,   B = S(A)
```

an operator on the tangent space at A. Its eigenvalues are the Floquet
multipliers of the cycle. Since `spectrum(DS(B) DS(A)) = spectrum(DS(A) DS(B))`,
the choice of phase does not change the spectrum, and the protocol computes both
phases at one cycle state and checks that they agree, as a free consistency
test.

**`p > 2`.** The same construction with the `p`-fold composition, but this is
out of scope for the first run: flagged and skipped, with the state named in the
report.

**Comparability across periods.** A composed multiplier is a per-period rate.
Everything that compares fixed points with cycles uses the per-pass rate
`rho_k = |lambda_k| ** (1/p)`. Both are reported.

**Naming a sign.** For a one-pass map a negative real eigenvalue is a flip mode,
and a modulus above 1 there is the period-doubling configuration F14 found. For
a composed two-pass map a negative real eigenvalue is a period-4 flip mode, and
that is what the report will call it.

## 6. Design question 3: real against complex eigenvalues

The operator is real and nonsymmetric, so eigenvalues come as real values and
conjugate pairs. Both solver paths return a complex array: strictly real
eigenvalues come back with zero imaginary part, and a pair comes back with
eigenvectors packed as `vr + i vi` and `vr - i vi`. This is true of
`numpy.linalg.eig` on the real Hessenberg matrix of the built-in Arnoldi and of
`scipy.sparse.linalg.eigs` on ARPACK's real nonsymmetric driver alike, so the
reporting rules below do not depend on which path ran.

Reporting rules, fixed in advance:

- Sort by modulus, descending. Ties broken by argument.
- Each conjugate pair is reported **once**, as one row, marked as a pair, with
  modulus, argument in degrees, and passes per revolution
  `p * 360 / |arg_deg|`. Reporting both members would double-count the mode and
  inflate any subsequent count of "how many marginal directions there are".
- For a real eigenvalue the eigenvector is taken real (imaginary part must be
  below 1e-6 relative, and this is asserted).
- For a pair, the object that matters geometrically is the real invariant
  2-plane `span(Re v, Im v)`. It is Gram-Schmidt orthonormalised to `q1, q2` and
  stored. Any cosine of a probe direction `u` against that mode is the subspace
  cosine `sqrt(<u,q1>^2 + <u,q2>^2)`, not a cosine against either member.
- Every eigenpair carries its own residual `||A v - lambda v|| / (|lambda|
  ||v||)`, computed with fresh matvecs after the solve. Anything above 1e-3 is
  reported as unconverged and excluded from the prediction test.

## 7. Design question 5: finite-difference step and the sensitivity check

The JVP is the primary instrument. `torch.func.jvp` gives the exact directional
derivative to float32 roundoff and costs 2.85 warm forward passes on this box
(measured, section 12). Central finite differences cost 2 forward passes and
carry both truncation error and cancellation error, so they are used only as an
independent-implementation check, never inside the Arnoldi loop.

The prior art's own two-epsilon data settles the step size. For `d_committed` at
the committed pivot: JVP -0.863580, FD at `eps_rel` 1e-3 gives -0.863613 (five
significant figures), FD at 1e-4 gives -0.860108 (0.40 percent off). The same
pattern holds for `d_sym` at the symmetric pivot: -4.275078, -4.274697,
-4.262007. So on this box in float32, `eps_rel` 1e-3 is the well-conditioned
choice and 1e-4 is already cancellation-limited. No smaller step is used.

**Gate G4, the sensitivity check.** After the solve, for the top 5 eigenvectors,
recompute the Rayleigh quotient by central FD at `eps_rel` 1e-3 and 1e-4.

- JVP against FD at 1e-3 must agree to within 1 percent relative, same sign.
  Failure halts the run for that state.
- FD at 1e-3 against FD at 1e-4 is reported, not gated. A disagreement above 2
  percent means the eigenvalue's last reported digit is not resolved, and the
  report says so for that eigenvalue.

## 8. The eigensolver

Matrix-free. The operator is a function of a flat vector of length
`N = T * 768` that casts to float32, projects onto the tangent space at the base
point, applies one or two JVP-based `DS` steps, projects again, and casts back
to float64.

**Solver choice, and a dependency note.** `scipy` is not installed in this
repo's `env/`, verified while writing this protocol. Rather than add a
dependency to a repo mid-series, the script ships its own explicitly restarted
Arnoldi (modified Gram-Schmidt with one reorthogonalisation pass, `numpy.linalg.eig`
on the small Hessenberg matrix, Ritz residuals from `|h_{m+1,m} e_m^T y|` for
free). If `scipy.sparse.linalg` is importable the script uses
`eigs` on a `LinearOperator` instead and records which path it took;
the two are cross-checked whenever both are available. Installing scipy is a
legitimate alternative and would give ARPACK's implicit restarting, which is
better conditioned; the built-in path is chosen so the experiment can run today
without touching the environment.

- `K = 24` eigenvalues, largest modulus. These are the least contracting and any
  expanding directions, which is exactly what the prediction is about.
- Arnoldi basis size `m = 96` (4K), with 4 explicit restarts. Total cost is then
  exactly `(1 + 4) * 96 = 480` matvecs, known in advance rather than estimated.
- Requested accuracy: a Ritz pair counts as converged at residual below 1e-6
  relative. Not tighter, because the matvec's own relative error is near 1e-6.
  Unconverged pairs are reported with their residuals and excluded from the
  prediction test.
- `v0` seeded deterministically, and **gate G5** reruns the whole solve at a
  second `v0` seed and requires the top-10 moduli to agree to 1e-3 relative.
  Krylov convergence is start-vector dependent and this is the cheapest honest
  check that the returned set is the leading set and not a Krylov accident.
- A matvec counter is stored with the result, so the report quotes the real cost
  rather than an estimate.
- **Gate G0**, plumbing: the identical solver path is run on a small dense random
  real nonsymmetric matrix and compared against `numpy.linalg.eig`. This catches
  convention, packing and conjugate-pair bugs in seconds and costs no forward
  passes.

Two structural notes that the report must carry.

**Row-uniform splitting.** The settled states are position-collapsed (the
`Divine` state to `row_spread` 4e-7). The row-uniform subspace, 768 dimensional,
is then invariant under the map, so the operator block-diagonalises into
row-uniform modes and row-mean-free modes. Each eigenvector's `row_spread` is
reported. If eigenvectors come back mixed, the collapse is not exact enough to
split the operator, and that is itself a finding to record rather than a bug to
hide.

**Base-point refinement.** The engine's gate passes at mean-vector cosine 0.999,
which is loose for a linearisation. Before any solve, the base point is refined
by iterating `S` (or `S o S` for a cycle) up to 20 further passes, and the
residual `||S^p(y) - y||_F / nu` is recorded. **Gate G6**: the residual must
fall below 1e-4, else the state is flagged and its spectrum is reported as
provisional.

## 9. The falsifiable prediction

### 9.1 Statement

The escape ladder (`experiments/basin_depth/01_escape_ladder.py`, running now)
measures, for each settled state and each of several unit tangent directions,
the smallest rotation in degrees after which the loop does not return to the
same attractor.

**Prediction P1.** Within a single attractor, the directions whose multipliers
are closest to 1 in modulus, that is the least contracting and most marginal
ones, are the directions that escape at the smallest angles. Operationally,
define a direction's marginality as the overlap-weighted per-pass rate

```
rho(u) = sum_k w_k rho_k / sum_k w_k ,   w_k = |<u, e_k>|^2 ,  rho_k = |lambda_k|^(1/p)
```

over the leading K modes, using the invariant-plane subspace cosine for
conjugate pairs. The prediction is that the escape threshold `theta(u)` and
`rho(u)` are negatively rank-correlated within each state.

**Confirms P1**: Spearman correlation at or below -0.5, one-sided p below 0.05,
pooled over at least three states after within-state ranking, AND the
eigen-direction with the largest `rho` escapes at a strictly smaller rung than
the median random-direction threshold in the same state, in a majority of
states.

**Refutes P1**: Spearman above -0.2 or positive in sign, OR the most marginal
eigen-direction escapes at or above the random median in a majority of states.

Anything between those two is ambiguous and is reported as ambiguous, with the
numbers, and not spun in either direction.

**Prediction P2** (cheap, needs no new ladder). At the `Divine` cycle the flip
axis should be an eigenvector of the composed two-pass on-shell operator: it
should have the largest overlap with some single leading eigenvector or
invariant plane. If it is not close to any single leading eigenvector, then
F14's rank-1 reading of the cycle is incomplete, and this protocol will say so.

One frame detail that P2 must get right, measured during this design. `d_sym`
is not tangent to the shell at phase A: `cos(d_sym, A_hat) = 0.3969`, so only
0.9179 of it is tangential. The tangent-space object, and the only thing the
composed on-shell operator can act on, is the tangentially projected axis, which
is 0.9179 aligned with `d_sym` itself. The escape ladder orthogonalises its
directions against the state in exactly the same way, so the ladder's flip-axis
direction and this protocol's flip-axis direction are the same vector, and P2
and P1 are talking about the same object. What P2 does **not** do is try to
recover the prior art's composed +0.0993 from the tangent operator: that number
is a Rayleigh quotient of an unprojected `J_f` chain at the raw states, along the
full `d_sym` including its radial 40 percent. It is reproduced in gate G3 in its
own frame, and it is not the same quantity as any tangent-space multiplier.

**Prediction P3** (structural, free). At a position-collapsed attractor every
leading eigenvector should be either row-uniform or row-mean-free, never mixed.

### 9.2 The blocker, and why the test has to be run along the eigenvectors

Two problems with correlating against the escape ladder's existing directions.

**Irrecoverable directions.** The ladder does not archive its direction tensors.
Its per-state generator is seeded `SEED + hash(key) % 10000`, and `hash` on a
`str` is Python's salted hash. Verified during this design: the same key gives
9535 in one process and 3854 in the next, with `PYTHONHASHSEED` unset. The four
random directions per state therefore cannot be reconstructed from the committed
checkpoints by anyone, including a rerun of the same script. The two named
directions, the flip axis and the glitch direction, are stored in
`output/directions.pt` and are exactly reproducible.

A separate one-line change should archive the direction tensors in future
ladders. That is a recommendation, not part of this protocol, and it must not
disturb the run in flight.

**Vanishing overlap.** Even with the tensors, a random direction carries almost
no eigen-information. In the flattened space, `N = 7680` for a ten-position
state; a random unit direction's expected squared overlap with a fixed
K-dimensional subspace is `K / N = 24 / 7680 = 0.0031`, a root-mean-square
cosine of 0.056 to the whole leading subspace and 0.0114 to any single
eigenvector. Four such directions per state cannot resolve a correlation against
`rho`. This is not a fixable sampling problem; it is the geometry.

**Consequence for the design.** The prediction is tested in three tiers.

- **Tier A, free.** The two named directions per state have stored tensors.
  Report their overlap with every leading mode, their `rho(u)`, and their
  committed escape threshold. Two points per state, up to 32 points overall.
  Underpowered on its own, reported as such, but it costs nothing.
- **Tier B, free.** The ladder's random-direction thresholds form the **null
  band**. Their distribution does not depend on the lost seed, so they remain
  usable in aggregate. Predicted, and checkable: because random directions have
  near-zero overlap with the leading modes, their thresholds should be
  relatively homogeneous within a state, and eigen-direction thresholds should
  be more spread out than they are.
- **Tier C, Stage 3, the real test.** Run the identical ladder, same rungs, same
  gate, same escape criterion, along the eigenvectors themselves: the 4 modes
  with the largest `rho` and the 4 with the smallest, per state. This is the
  version of the test that has power, and it is the version the confirm and
  refute criteria above are written for. Tier A and Tier B are supporting
  evidence.

## 10. Pre-stated interpretations

Written before any spectrum is computed, so that no shape of result can be
narrated after the fact.

| observed leading spectrum | reading fixed in advance |
|:--|:--|
| all moduli comfortably below 1, say max below 0.9 | a strongly contracting attractor; escape must be a nonlinear, not a marginal-direction, phenomenon, and P1 should fail at that state. Predicts large escape thresholds. |
| leading modulus in 0.95 to 1.00 | a slow manifold. The direction to watch. Predicts a small escape threshold along that mode and a late lock-in for that state. |
| leading modulus 1 to numerical precision | a neutral direction, candidate for a symmetry of the construction rather than of the model. Must be checked against the eigenvector's shape (row-uniform, position-permutation-like) before the word symmetry is used at all. |
| any modulus above 1 at a state the gate called converged | either the gate is wrong about that state or the state is a saddle the loop has not yet left. Reported loudly, cross-checked against a 200-pass continuation, and not quietly dropped. |
| complex pair with modulus near 1 | a slow spiral. Free cross-check: the state's lag scan should then show a long quasi-period rather than a flat 1.0 at every lag. |
| large negative real eigenvalue at a `p = 1` fixed point | the state sits near a period doubling, the same configuration F14 found at the `Divine` pivot. Predicts that a small change of pin should tip that state into a cycle, which the nu sweep can check. |
| `Divine` composed spectrum with one mode at about +0.10 aligned with `d_sym` | F14 confirmed and generalised: the cycle's stability really is rank-1 dominated. |
| `Divine` composed spectrum whose leading modes are all unrelated to `d_sym` | F14's rank-1 reading is incomplete. Reported as a correction to the record, not buried. |

## 11. States and modes

States: exactly the escape ladder's states, taken from
`01_escape_ladder.sweep_states()`, restricted to those whose zero-perturbation
control passed. As of writing, 16 of 17 controls have passed, the exception
being `m056_solidarity_G08_period`, which the ladder itself already excludes.
Sharing the state list exactly is what makes the cross-experiment comparison
legitimate.

Script modes, following the conventions of `01_escape_ladder.py` and
`nu_sweep/01_stage_a.py`:

- `--verify` runs gates G0, G1, G2, G3, G4 on the committed `Divine` state and
  writes `verify.json`. Workers refuse to run until it has passed.
- `--worker i --num-workers n` runs a slice of the states, one checkpoint file
  per state, resumable, temp-file-then-rename.
- `--eigen-ladder --worker i` runs Stage 3 (Tier C) for states that already have
  a spectrum checkpoint.
- `--report-only` regenerates `spectrum.md` from the checkpoints alone. No
  number in the report is computed anywhere but from stored data.

## 12. Verification gate, and the smoke test already run

No new number is trusted until the machinery reproduces the committed one.

**Gate G1, the prior-art gate.** On the committed `Divine` iteration-1000 state
(`output_divine_motion/state_divine.pt`), rebuild the committed frame exactly as
`08_hinge_eigenvalue.py` does (raw A, shell Bn, `d_committed = (A - Bn)/2`,
`M_committed = (A + Bn)/2`, then normalised to the shell), and require

```
<d_hat, J_f(Mn_committed) d_hat> = -0.863580  to within 1e-3 absolute
||J_f(Mn_committed) d_hat||      =  0.957522  to within 1e-3
cos(J_f d_hat, -d_hat)           =  0.901890  to within 1e-3
```

**This gate has already been run as the design smoke test** (one JVP, two
forward passes) and it passes to six decimals:

```
cos(A, f(f(A))) = 1.0000005            period-2 cycle intact
lambda          = -0.863580            expected -0.863580
||J t||         =  0.957522            expected  0.957522
cos(Jt, -t)     = +0.901890            expected  0.901890
radial null     =  8.3e-07             gate G2, threshold 1e-4
cos(f(M), M)    =  0.851010            the committed pivot is not a fixed point
on-shell scale  =  0.294174            nu / ||f(M)||
```

**Gate G3, the composed gate.** In the prior art's convention, the composed
two-pass Rayleigh quotient along `d_sym` starting at raw A must reproduce
+0.099339 to within 1e-3. This gates the composition order and the chaining of
JVPs, which is the part of the machinery most likely to be silently wrong.

**Gates G0, G2, G4, G5, G6** are as specified in sections 8, 4, 7, 8, 8. All of
them must pass, per state where they are per-state, before that state's spectrum
enters the report. A state failing G6 is reported as provisional and excluded
from the prediction test.

**The operator path was also exercised end to end during design**, on the same
committed `Divine` state, at a cost of about twenty forward passes. It behaves:

```
period detected            2, lag table 1:0.684912 2:1.000000 3:0.684912 4:1.000000 ...
                           the exact parity signature F9 and F15 record
base residual after refine 2.20e-07  (gate G6 threshold 1e-4)
tangent dimension N        7680 ambient, one radial direction projected out
matvec output tangency     1.19e-07 relative
one composed matvec        finite, well scaled
```

Gate G0 also passes standalone, with maximum absolute error 8.3e-15 against
`numpy.linalg.eig` on a dense 60 by 60 nonsymmetric matrix, and the conjugate
pairs are correctly deduplicated to one mode each.

## 13. Compute cost

Measured on this box, single-threaded, ten-position state, warm:

| unit | cost |
|:--|--:|
| one forward pass through the block cascade | 0.177 s |
| one `torch.func.jvp` | 0.504 s, that is 2.85 forward passes |
| one full matvec of the `p = 1` operator | 0.556 s, that is 3.1 forward passes |
| one full matvec of the composed `p = 2` operator | 1.040 s, that is 5.9 forward passes |

The matvec figures are measured on the real operator at the committed `Divine`
state, warm, single-threaded, and they include the tangent projections and the
on-shell rescale, not just the JVP.

JVPs per eigenvalue: with the built-in explicitly restarted Arnoldi the count is
fixed in advance, not estimated. `m = 96` basis vectors and 4 restarts is
`(1 + 4) * 96 = 480` matvecs for `K = 24` eigenvalues, that is **20 matvecs per
eigenvalue**, and one matvec is 1 JVP for a fixed point or 2 JVPs for a cycle.
So 480 JVPs (fixed point) or 960 JVPs (cycle) per solve, which is 1370 or 2740
forward-pass equivalents. The counter in the checkpoint records the actual
figure, and if a solve is stopped early on full convergence the report quotes
the lower number.

Per state:

| item | matvecs | `p = 1` | `p = 2` |
|:--|--:|--:|--:|
| main solve | 480 | 4.4 min | 8.3 min |
| gate G5 second solve | 480 | 4.4 min | 8.3 min |
| eigenpair residuals, 2 per complex pair | about 40 | 0.4 min | 0.7 min |
| gate G4 finite differences, 20 forward passes | - | 0.1 min | 0.1 min |
| base refinement and period scan, up to 50 passes | - | 0.2 min | 0.2 min |
| **per state** | about 1000 | **about 9.5 min** | **about 17.6 min** |

The state list is 16 states (section 11), of which the two `Divine` ones are
expected to be `p = 2`. Total: about **2.8 core-hours**. Across four workers that
is about 45 minutes of wall time, but this protocol is not to be started while
the escape ladder holds the cores.

Stage 3, the eigen-ladder, costed separately because it is much the more
expensive half: 8 rungs times 8 eigen-directions times up to 200 gated
iterations at about 0.25 s per iteration is up to 53 min per state. Scoped to
three states, that is up to 2.7 core-hours. Stage 3 runs only after Stages 1 and
2 are in and only if the spectrum shows any spread in `rho` worth testing; if
every leading mode has essentially the same modulus, P1 is untestable at that
state and the ladder is not run there. That decision rule is registered here so
it cannot be made after seeing the thresholds.

Expressed in forward passes, the whole of Stages 1 and 2 is about 56,000
forward-pass equivalents; Stage 3 adds up to 38,000.

## 14. Outputs

In `experiments/basin_depth/output_spectrum/`:

```
verify.json                  the gate results, written by --verify
checkpoints/<key>.pt         one spectrum per state (resume unit)
eigen_ladder/<key>.pt        Stage 3 ladders, one per state
spectrum.pt                  combined archive, assembled by --report-only
spectrum.md                  every headline number, regenerated from data only
```

The report states, per state: the period `p` and its lag table, the base-point
residual, the gate results, the leading spectrum with both conventions, the
per-eigenvector row spread and residual, the overlaps of the named directions,
and the prediction test with its pre-stated verdict applied mechanically.
