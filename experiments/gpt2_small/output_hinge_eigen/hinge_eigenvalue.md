# Flip-Axis Eigenvalue: The Inversion Is Real, It Overshoots, and One Attention Head Does It

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


*Follow-up to [bell_anatomy.md](../output_divine_motion/bell_anatomy.md) (issue #14, thread 1). Runner: [`08_hinge_eigenvalue.py`](../08_hinge_eigenvalue.py); raw numbers: [`hinge_eigenvalue.json`](hinge_eigenvalue.json). Single Divine trajectory, states rebuilt from the committed iteration-1000 checkpoint. Sanity gate reproduced before any measurement: cos(A, B) = 0.684912, cos(A, f(f(A))) = 1.000000, full-tensor cycle residual 8.0e-04 against amplitude 5098.*

## Questions asked

1. Does the flip axis d carry an effective eigenvalue near -1 under the normalised ATR map (the negative-eigenvalue conjecture of bell_anatomy.md)?
2. Which block (layer, attention or MLP) performs the inversion?

## Verdict

**The negative eigenvalue is real, direction-specific, and localised: the linearised ATR map inverts the flip axis and only the flip axis, and the entire sign flip is executed inside block 11, 99 percent of it by attention head L11.H8.** The conjecture survives in sign and in specialness. It fails in magnitude, in an instructive direction: at the pivot the flip axis eigenvalue is not near -1 but **-4.3** (an overshooting flip), while around the full two-step cycle the projected multiplier along the flip axis is **+0.10** (a strong contraction; the composed return is only partially aligned with the axis, cos 0.51 to 0.56, so this figure is a directional multiplier, not an eigenvalue of the composed map). Divine is not a marginal see-saw riding an eigenvalue of -1; it is a textbook period-doubling configuration: a nearly fixed pivot that is violently flip-unstable along exactly one direction, with a finite-amplitude period-2 orbit around it that is strongly stable. The literal "-1" of the conjecture appears only for the committed d at the committed pivot (lambda = -0.864, amplification 0.957, cos(Jd, -d) = 0.902), and that coincidence is partly an artifact of a frame mix in how 06_bell_anatomy.py built d (see caveats).

## The map, the frames, and two flip axes

The measured map is the full ATR iteration exactly as `atr_engine.run_atr_loop` implements it: f(x) = ForwardBlocks(x * N0/||x||), with N0 = 1468.5 the loop's energy shell and ForwardBlocks the 12-block cascade from `blocks.0.hook_resid_pre` (where the injection overwrites every position) to `blocks.11.hook_resid_post`. A pure-blocks reimplementation matches the hook-based step to exactly zero error, which makes forward-mode autodiff (`torch.func.jvp`) available; central finite differences at eps = 1e-3 and 1e-4 of the base norm agree with jvp to 3 to 4 significant figures on every headline number.

Two facts sharpen the frame:

- **The state is exactly position-collapsed.** All 10 rows of A, B, and d are identical (row spread 0.0). Row-uniform tensors are an invariant subspace of the forward map, so the whole cycle is a period-2 orbit of an effective 768-dimensional map, and last-position numbers equal full-tensor numbers.
- **f is scale invariant** (f(cx) = f(x)), so J_f at a raw point equals J_f at the corresponding shell point times N0/||raw||. The raw cycle states sit far off the shell (||A_raw|| = 5098, ||B_raw|| = 4838; shell factors 0.288 and 0.304). The identity was verified directly: lambda at A_raw measured -0.113983, derived from the shell value -0.113983.

06_bell_anatomy.py built its flip axis from raw A (row norm 1612) and shell B (row norm 464), so the **committed d** is 0.967 aligned with A's own direction and 0.909 aligned with its own pivot M: it is mostly radial contamination, only 0.616 aligned with the clean flip axis. The **symmetric flip axis** d_sym = (An - Bn)/2, built from both phases on the shell, is exactly orthogonal to its pivot M_sym and is the direction the cycle actually swings along. Both are measured everywhere below; once the loop's own renormalisation strips the radial part of the committed d, what survives is 0.973 aligned with d_sym, so the two stories converge.

## Result 1: The half-map inverts the flip axis, and nothing else

lambda = (d, Jd)/(d, d), amplification = ||Jd||/||d||, all by jvp (FD at both epsilons agrees; see JSON).

| Point | Tangent | lambda | Amplification | cos(Jd, -d) |
|:---|:---|:---:|:---:|:---:|
| Pivot M_sym (shell) | d_sym | **-4.275** | 4.314 | **+0.991** |
| Pivot M_committed (shell) | d_sym | -1.971 | 2.238 | +0.881 |
| Pivot M_committed (shell) | d_committed | **-0.864** | 0.957 | +0.902 |
| Pivot M_sym (shell) | d_committed | -1.876 | 2.659 | +0.706 |
| Phase A (shell) | d_sym | -0.803 | 1.475 | +0.544 |
| Phase B (shell) | d_sym | -0.601 | 1.133 | +0.530 |
| Phase A (shell) | d_committed | -0.396 | 0.406 | +0.974 |
| Phase B (shell) | d_committed | +0.300 | 1.084 | -0.277 |

Controls (random directions orthogonal to both flip axes, 3 row-uniform and 2 generic) are the mirror image: at M_committed lambda = **+1.06, +1.11, +1.13** (uniform) and +0.94, +0.93 (generic); at M_sym, +1.11 to +1.18 and +0.94. Random directions pass through the map upright, slightly amplified; the flip axis alone comes back inverted, and at the symmetric pivot the inversion is essentially pure (cos(Jd, -d) = 0.991) and 4.3x overshooting. Note the asymmetry the frame mix produces: the committed d is inverted at A but not at B (+0.30), because at B its dominant radial component no longer points along the local radial direction; the clean flip axis comes back with a negative component along itself at both phases (lambda -0.80 at A, -0.60 at B, with cos(Jd, -d) 0.544 and 0.530: a negative directional component, not the near-pure inversion seen at the pivot).

In the raw frame the same derivatives carry the shell factors: lambda along d_sym is -0.231 at A_raw and -0.182 at B_raw. The frame changes the number, not the sign.

## Result 2: Around the full cycle, the flip axis contracts with positive sign

The dynamically correct stability object for a period-2 orbit is the composed linearisation J_f(B) J_f(A) at the raw states the iteration actually visits.

| Tangent | lambda composed | Amplification | cos(w, d) |
|:---|:---:|:---:|:---:|
| d_sym, start A | **+0.099** | 0.195 | +0.509 |
| d_sym, start B | +0.088 | 0.158 | +0.557 |
| d_committed, start A | -0.015 | 0.054 | -0.282 |
| d_committed, start B | +0.138 | 0.152 | +0.908 |
| controls (5) | +0.087 to +0.156 | 0.247 to 0.376 | +0.26 to +0.47 |

Along the true flip axis the composed projected multiplier is positive (each half flips, two flips restore the sign) and about 0.1: perturbations off the exact orbit decay by roughly 90 percent per period. the multiplier magnitude stays under 1 with room to spare, for the flip axis and for every control; the cycle is strongly attracting, which is exactly what lets it reproduce itself to machine precision for hundreds of iterations. The committed d, start A, is the degenerate case: its radial bulk is annihilated by the renormalisation Jacobian (intermediate ||v|| = 0.117), leaving almost nothing to propagate.

## Result 3: The pivot is a near-fixed point that is flip-unstable

One forward pass from each pivot: f(M_sym) comes back 0.9948 aligned with M_sym (renormalised residual 149 against shell norm 1468, about 10 percent). The symmetric pivot is close to a genuine fixed point of the normalised map, and along d_sym its eigenvalue is -4.3 while every control direction sits near +1. That is the period-doubling signature: a fixed point whose linearisation has one eigenvalue beyond -1 sheds a stable period-2 orbit around itself. The Divine bell is that orbit. (The committed pivot, being 0.985 aligned with A itself, is not near-fixed: f maps it 0.996 onto B, as a point near phase A should.)

## Result 4: Block 11 performs the inversion; inside it, head 8

Perturbations eps * d injected at `blocks.0.hook_resid_pre` (the exact point the loop re-enters), eps = 1e-3 and 1e-4 of the base norm, tracked at every layer boundary. cos(delta_l, d) at the last position, base M_sym, direction d_sym:

| Boundary | pre 0 | pre 1 | pre 2 | pre 3 | pre 4 | pre 5 | pre 6 | pre 7 | pre 8 | pre 9 | pre 10 | pre 11 | post 11 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| cos vs d | +1.000 | +0.951 | +0.935 | +0.968 | +0.963 | +0.962 | +0.963 | +0.959 | +0.951 | +0.940 | +0.920 | +0.885 | **-0.991** |

The flip axis sails through blocks 0 to 10 upright (cosine never below +0.88, with the d-component growing from +0.32 to +0.82 per unit input, the biggest single boost coming from MLP 2 at +0.37) and is inverted entirely inside block 11. The block-11 ledger (d-components per unit input, base M_sym): incoming +0.818, attention writes **-1.999**, MLP writes -0.167, net -1.349. Attention outweighs the MLP 12 to 1, and within attention one head carries it:

| L11 head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 | 10 | 11 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| d-component | -0.014 | -0.002 | -0.001 | +0.004 | +0.010 | -0.004 | -0.003 | -0.002 | **-1.981** | -0.006 | -0.003 | +0.004 |

**L11.H8 contributes 99.1 percent of the attention delta** (cos with -d: 0.963, gain 6.5); no other head exceeds 0.014. The same head dominates identically at the committed pivot (-0.899 of -0.911) and for the loop-faithful tangential committed d (-1.283 of -1.289, cos -0.988). The flip layer is 11 at both epsilons in every flipping configuration, and the L11 attention component moves by less than 0.3 percent between epsilons. The one configuration that never flips is the raw committed d: its 91 percent radial part rides through all 12 blocks and block 11 amplifies that pivot-like content positively (final cos +0.95), which is precisely why the frame mix matters; strip the radial part, as the loop's renormalisation does, and it flips at block 11 like everything else on the flip axis. Cross-check between the parts: the part-2 cascade at M_sym lands at -4.27 per unit flip axis input, matching the part-1 jvp eigenvalue -4.275 to 0.2 percent.

## Interpretation

The negative-eigenvalue reading of the bell was right about the mechanism and modest about its strength. There is exactly one direction the map refuses to preserve, and the refusal is not a soft rotation but an overshooting reflection (gain 4.3 at the pivot) implemented by a single OV circuit in the final layer: L11.H8 reads the flip axis component off the (position-collapsed) stream and writes back 2.4 times its incoming magnitude with the sign reversed, on top of eleven blocks that mostly amplify the flip axis upright (MLP 2 loudest). An overshooting flip at a near-fixed pivot cannot sit still, and cannot run away either once the finite-amplitude geometry bends the response back (composed projected multiplier +0.10): the state must fall onto a period-2 orbit, which is the bell we observe. The stable `Divine` argmax and the readout-mute swing of the earlier reports are the shadow of this structure: one head ringing one direction, everything else holding the tone.

## Caveats

- **The committed flip axis mixes frames.** 06_bell_anatomy.py takes A from the raw iteration-1000 tensor (norm 5098) but rescales B to the shell (norm 1468), because the checkpoint stores the pre-normalisation state. Its d is therefore 0.97 aligned with A's own direction, and downstream statements inherit that: the recorded "phase A carries norm 1612, phase B 464" contrast is the two frames, not an energy slosh (on the shell both phases have identical row norms 464), and lambda = -0.864 for that d is a blend of the true flip axis response with radial annihilation. All conclusions above are stated for both the committed d and the symmetric on-shell flip axis; the physics lives in the latter.
- The eigenvalues are directional derivatives along the flip axis, not a full Jacobian spectrum; other strongly negative directions, if any, were not searched for. The composed-map figure of +0.10 is the projection along the flip axis specifically (the composed return is only 0.51 to 0.56 aligned with the axis); it is not an eigenvalue of the composed map.
- The per-head split reads each head's z-delta through W_O (b_O cancels in deltas). It includes QK-mediated pattern changes routed through z but does not separate OV from QK mechanisms; identifying what L11.H8 attends to, and whether it belongs to a known head taxonomy, is left open.
- Part 2 uses forward differences (run(M + eps d) - run(M)); part 1 uses jvp with central-difference checks. Agreement between the two parts is 0.2 percent where they measure the same object.
- One trajectory, one prompt, one model, derivatives evaluated at one point per state. Whether the other Divine prompts share the same flip head is open (blocked on prompt_library restoration, issue #9).
