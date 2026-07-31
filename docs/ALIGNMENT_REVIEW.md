# ATR: Alignment Review and Direction

**Date:** 2026-07-31
**Type:** leadership review: a full pass over the repository, all 32 open issues, all open and recently merged PRs, the agent board, and the session records.
**Commissioned by:** TC, with the brief: *examine all claims, directions and hypotheses; produce a general report and a set of instructions to get the whole project aligned and where it needs to go.*
**Governing relationship:** [FINDINGS.md](FINDINGS.md) remains the canonical record of *results*. This document governs *work sequencing and dispositions* until TC supersedes it. Where this review and an open issue disagree about priority, this review states the reconciliation and names the issue.

Everything asserted below was verified against primary sources during this review (code, committed artifacts, issue text, PR diffs), by direct reading and by an 11-agent adversarial verification pass whose verdicts are folded in. Where a claim rests on an issue's own unverified measurement, that is said explicitly.

---

## 1. Where the project actually is

The arc so far: a Lucier-inspired feedback loop on GPT-2 Small found five semantically coherent attractor basins; the founding corpus-fingerprint hypothesis was refuted by the project's own cross-model validation; the survivors were a cheap dynamical probe, one fully-traced mechanism (the `Divine` period-2 cycle, F9 to F17), and one anomaly (why GPT-2 Small alone). A readout-audit series then found the basins are carried by broad, low-confidence distributions (F7, F8). Execution was then deliberately paused behind an understanding gate ([ATR_PAUSE.md](ATR_PAUSE.md)).

Then, 2026-07-28 to 07-31, the **mathematics-foundations wave**, the most consequential four days in the project's record:

1. **M1's exactness question was settled analytically** (#91, independently confirmed twice). A row-uniform state is an exactly invariant subspace of the map: identical rows give identical queries, keys and values, attention returns the shared vector under any weights, and everything else is per-position. The ~1e-7 residual in the archives is float32 rounding inside that subspace, so the gated float64 exactness run is no longer needed. Stated precisely, and this matters: the *exactness at* the collapsed state is architectural; the *convergence into* the subspace is emergent dynamics, model-dependent (Pythia-410m never consolidates), and its rate is the still-open M2.
2. **The rescale factor was measured from committed data** (M5, via PR #85/#87/#89): the loop rescales by *c* ≈ 0.27 to 0.29 (language) / 0.099 (noise) per pass, a stable constant, emphatically not 1, and the map is not scale-invariant over that range (#69, caveat 7). The fixed point is a fixed point of *rescale ∘ F*, with per-pass gain λ = 1/*c* ≈ 3.47 to 3.76 (language) / 10.1 (noise).
3. **The core object sharpened.** 54 of 125 prompts converge to *one vector*: pairwise cosine 0.9987 to 1.0000, identical to seven significant figures across unrelated prompts (#98, F11). The attractor carries no information about the input. The thing to explain is a single direction fixed by the weights, with a decomposable gain.
4. **The STOP directive** (#94, 2026-07-29): reverse-engineer the `prolet` fixed point from the weights. Exact per-component attribution at v* (Part A), then a global self-coupling spectrum (Part C). This is the mathematics-foundations direction, and it is the right one. §3 assesses it.
5. **Two findings came under credible challenge** the same week: F4's noise baseline was mis-calibrated by ~1/√T (#97, **verified in this review**, §4.1), and F8's coherence null was measured against what GPT-2 actually emits and found evidentially empty (#98: the cone-artifact *mechanism* was independently reproduced during this review from raw HF weights; the corpus percentile figures still await committed code, §4.2).

So the project is mid-pivot from *cataloguing phenomena* to *reverse-engineering one object*. The pivot is correct. What makes it tricky: three of the claims the new work leans on need repair first, the record has drifted in eleven documented places, and the STOP directive, the pause gate, and the owner's own subsequent issue filings are pulling agents in three directions. The rest of this document is the reconciliation.

---

## 2. Claim-by-claim status after the wave

Status of every principal claim, cross-checked against the July 29 to 31 issues. "Canonical" = what FINDINGS.md currently says; "actual" = what this review concludes the evidence supports today.

| Claim | Canonical | Actual status after the wave | Action |
|:--|:--|:--|:--|
| **F1** five basins at lock-in | supported | Stands at argmax level. Distribution level already qualified in-record ("fewer than five objects"); sharpened by the one-vector fact (#98): `prolet` is a single direction, not a region. | Language tightens under §5 T0-6 |
| **F2/F9** `Divine` = period-2 cycle | resolved | Stands. Untouched by the wave. One trajectory (caveat 14). | None |
| **F3** landscape doesn't generalise | supported | Stands. The Pythia-410m "no consolidation" cell remains **provisional**: produced by the lag-1 gate F15 proved period-blind; the full-lag re-gate needs local-only trajectories or new runs (gated). | Flag stays; parked with cross-model track |
| **F4** noise → 18 basins, language → 5 | supported | **Confounded, verified.** Noise arm ran at ~1/√T of matched Frobenius norm (gain regime 10.1× vs 3.5×) *and* was counted at iteration 100 unconverged vs language at lock-in. Either alone could produce the gap. Blast radius wider than F4 itself: see §4.1. | §5 T0-5: caveat now; re-run post-gate (§5 T2-1) |
| **F5** differences intrinsic, not apparatus | qualified | Stands as "three channels exonerated," but **F5's result 1 ("the global L2 rescale is effectively invisible to the forward pass") still states the reasoning caveat 7 withdrew on 2026-07-28**, and the withdrawal was never propagated to SCALING_ARTEFACT_ANALYSIS's closing judgement, RESULTS_SUMMARY (three places), or PYTHIA_INTERPRETABILITY_REVIEW (flagged on board discussion #70, never edited). Tokenisation channel (caveat 17) still open; CE11 (73× natural-norm injection, #95) deepens the apparatus question. | §5 T0-4: propagation sweep |
| **F6** cross-hardware replication | supported | Stands. | None |
| **F7** confidence inversion | supported | Stands for the language states (#98's pipeline reproduced the F7 numbers). Its noise clause ("15 calibrated noise trials") inherits the #97 mis-calibration: the 397.18 constant is hardcoded at `04_readout_confidence.py:153`. | Noise clause caveated with F4 |
| **F8** `prolet` coherence, p = 0.001 | supported | **Under verified mechanism challenge** (#98): coherence@10 measures anisotropic-cone position, not meaning (a random multiple of one token embedding scores at `prolet`'s level; reproduced independently in this review). Corpus percentiles (78th/2.6th) plausible, not yet reproducible from committed code. Effective sample is 2 distinct states, not 4. F8's noise leg additionally inherits #97. | §4.2 adjudication protocol before rewrite |
| **F9's noise bracket, F11/F16's language-vs-noise boundary** | supported | The *language-side* results stand. Every noise-side comparison inherits #97: the mis-scaled ν is hardcoded in `05_divine_motion.py:96`, `05_jlens_pilot.py:294`, `10_jlens_phase.py:186`, so noise controls ran at ν differing ~3.5× from the language states they bracket. | Caveat with F4; matched-ν re-run decides |
| **F12** Medium's typographic attractor | supported | Stands; #98's mechanism deepens the caveat-10 worry it already records. | None |
| **F13/F14/F17** flip axis, −4.3, L11.H8 | supported | Stand (consume `state_divine.pt` only; no noise dependence). One trajectory. #99 adds the DEQ/Jacobian-regularisation literature as F14's proper frame. | §5 T0-4 doc fix |
| **F15** lag-k gate | delivered | Stands. | None |
| **H4** per-head resonance ≈ W_OV power iteration | "untested" | **Record is wrong, verified.** Executed 2026-07-25 (PR #29 regeneration), NOT SUPPORTED: 5/144 heads > 0.9. Artifacts committed (`experiments/_DATA/EXP_009/009c_*.pt`). Five places still say untested (FINDINGS ×2, JOURNEY_MAP, TECHNICAL.md:120, README notebook table) plus the notebook's own banner. Needs TC's two rulings in #54. | §5 T0-2 |
| **H-pos0** attractor is fixed point of single-position map | registered | Premise (exact collapse) now holds analytically (#91). The *c* = 1 step is corrected (M5), **but the FINDINGS H-pos0 row still asserts "at a settled state ‖xⁿ‖ is constant, c_n = 1," which M5 measured false** (c = 0.288/0.099/0.266). Test unblocked by the `prepend_bos` engine change (PR #82, merged). Gated. | §5 T0-4: amend the row |
| **H-fingerprint / H-till / H-supp / H3** | refuted / withdrawn | Stand as recorded. | None |
| **M1** exact position collapse | "not settled" (contraction/RESULTS.md) | **Exactness settled analytically** (#91, twice confirmed); the convergence *into* the subspace, and its rate, remain the empirical questions (M2). Withdraw the float64 exactness-run recommendation; keep the archive recommendation. | §5 T0-4 |
| **M2** collapse rate | data-gapped | Correct as recorded: no archive holds per-position per-iteration tensors. Fix is the archive spec (§5, standing rules), not a new metric. | Standing rule |
| **M3** Jacobian spectrum at attractors | gated, registered (#71) | **This is the sound version of #94's Part C** (§3.4). | Folded into #94 protocol |
| **M5** rescale factor | partly delivered | Settled value measured; transient (iters 1 to 99) unobserved. "Not gated" per PR #87 (engine records `tensor_norm`; the loss was at save time). | Standing rule |

**Identifier note:** Stage 1 owns H0 to H4, Stage 2 (sibling repo) owns H5 to H8, H11 is registry-assigned. #95's candidate explanations are correctly renumbered CE1 to CE11. New hypotheses take numbers only via the board's Identifier registry (discussion #53). `EXP_010` is held by the sibling repo; `EXP_011` has been twice declined and never confirmed free; keep using descriptive directory names.

---

## 3. The mathematics-foundations direction: assessment

This is the review's centrepiece, per the brief. Short version: **Part A of #94 is sound and should be executed as specified below; Part C as written is already contradicted by the repo's own H4 data and should be replaced by the registered M3; the eigenvalue framing closes, and the direction-preservation gap in the record was closed during this review.**

### 3.1 The framing is correct, and now fully verified

#94 writes the fixed point as F(v̂\*) = λv̂\*, λ = ‖v\*‖/ν ≈ 3.756. Verified against the engine (atr_engine.py:212-235): the loop is *rescale to ν, then forward*; every recorded state is post-forward/pre-rescale; so the injected state v̂\* is on the shell ‖v̂\*‖ = ν and F(v̂\*) is the recorded next state. At a converged state F(v̂\*) ∝ v̂\* and the constant is exactly the M5 amplification. The residual-sum decomposition F(x) = x + Σₗ[Aₗ(x) + Mₗ(x)] is valid at the read site (`blocks.11.hook_resid_post` is upstream of `ln_f`). The arithmetic checks: λ = 5230.652/1392.6476 = 3.755905 (one nit: stable to the 4th figure only from iteration ~250, not 100).

**The direction-preservation gap, now closed.** The eigenvalue equation is exact only if the state converges in *direction*, and the record documented only norm stability. This review computed it from the committed archives: lag-1 `cosine_sim_last` = 1.0 ± 1.2e-7 (the float32 floor) at all late snapshots, and in float64 cos(x₂₅₀, x₁₀₀₀) = 1 − 9.9e-11, cos(x₉₀₀, x₁₀₀₀) = 1 − 3e-14. The instrument detects rotation where it is real (the `Divine` lag-1 cosine is 0.6849), so `prolet`'s 1.0 is meaningful: the rotating-while-norm-stable failure mode is empirically excluded. These numbers should be added to `contraction/RESULTS.md` (§5 T0-4) so the record shows what #94 assumes.

**The error budget.** State the correctness check as Σαc = 2.7559 ± ~1e-3, not "2.76 exactly," and note the bonus: since Σαc = λ·cos θ − 1, closure of the sum *is itself* a direction-preservation check.

### 3.2 Part A (exact component attribution): endorsed, with two corrections and preconditions

#94's own caveats (orthogonal-write blindness; position collapse as the well-definedness condition; the w_in·W_E read degrading with depth; the ⟨w_out, w_in⟩ self-coupling shortcut being a cross-iteration quantity a within-pass dot product cannot capture) are all correct as written. Two things in the issue's formulation are not, and both would silently break the correctness check:

- **The decomposition equation as literally written, Σₗ[Aₗ(v̂\*) + Mₗ(v̂\*)] = (λ−1)v̂\*, is false.** Component *l* acts on the *accumulated mid-pass residual*, not on v̂\*. The exact object is each component's **actual write during the single forward pass initialised at v̂\***, read via `hook_attn_out` / `hook_mlp_out` (and `z @ W_O` for the per-head split). #94's author demonstrably knows this (it is Part C's failure mode 1), but the Part A statement invites the naive isolated-application implementation, whose sum would *not* equal (λ−1)v̂\*. Implement against the hooks.
- **The "156 components" omit the biases.** `hook_mlp_out` includes `b_out`, but the per-head `z @ W_O` split drops each layer's `attn.b_O` (12 terms). The advertised Σαc = 2.76 check will not close as specified. Do the **24-term closure first** (per-layer `hook_attn_out` + `hook_mlp_out`, which is exact), verify it against λ − 1 within the §3.1 error budget, then refine to heads plus explicit bias terms.

Preconditions and controls, in order:

- **P1: frame check.** The single most likely silent failure: the committed tensor is the PRE-rescale state (norm 5230.65). Rescale it by c = 0.266248 to the shell (norm ν = 1392.6476) *before* the forward pass; attribute at v̂\*, in float64. Verify the load (`output_divine_motion/state_prolet.pt`, present in the clone; T = 12) and rank-1 (σ₂/σ₁ ~ 1e-7; recompute at the state used). Hard-symmetrising the rows to their exact mean is legitimate if wanted: the row-uniform subspace is exactly invariant (#91).
- **P2: fixed-point residual first.** The archived state is an iteration-1000 *stop*, not a proven exact fixed point. Run the single pass, report λ_check = ‖F(v̂\*)‖/ν against 3.755905 and 1 − cos(F(v̂\*), v̂\*) in float64; that residual is the error bar the Σαc closure is judged against.
- **P3: the state count is 2, not 4.** `Semantic` ≡ `Control_prolet_Semantic`, `Syntactic` ≡ `Divine_Syntactic` (#91 archive note; #98, verified). Attribution runs on distinct states, and the writeup says how many there are.
- **C1: contrast states.** Run the identical attribution at the `Divine` pivot M and both phases (committed cycle states), and at one noise attractor, the latter carrying an explicit flag that noise states were produced under the #97 mis-calibration (10× gain regime) and are not a clean control until the re-run (§5 T2-1).
- **C2: null distribution.** The αc concentration profile for ~100 random on-shell directions through the same pass, so "the prolet state is held up by few/many components" is a statement against a measured baseline. (The same discipline that saved F13: the norm-matched null was the sharp control there.)
- **C3: CE mapping, pre-registered.** #95 exists so Part A discriminates rather than narrates: concentrated interpretable components → CE9/CE4; diffuse structureless contributions → CE5/CE8 strengthen and "why socialist" may stop being well-posed. Write the mapping down *before* looking; link #95 in the run report.

Neuron-level drill-down (rank ⟨w_out_j, v̂\*⟩·aj, decode top neurons through ln_f → W_Eᵀ) as #94 specifies, with its depth caveat kept, **and with #98-corrected nulls on every decode step**: coherence-style eyeballing of "what tokens this neuron promotes" is exactly the cone artifact (a random multiple of one token embedding out-scores every `prolet` state on the old statistic, twice reproduced), so neuron-decode claims use the ordinary-output percentile null or the cone-position probe, not qualitative neighbourhood reading. Add **X3 from #96** (the one experiment that survived that issue's retractions): is L11.H8's W_O output direction anti-aligned with the frequency pole? Weights-only, ungated, slots beside Part A's head-level table.

Two honest scope caveats for the writeup. **α and λ are shell-indexed quantities, not weight-intrinsic ones**: they depend on the arbitrary seed norm ν because F is not scale-invariant (caveat 7), and since ‖Δ‖ is near-constant across runs (≈3620 to 3838, 6% spread) the headline gain λ − 1 ≈ ‖Δ‖/ν substantially reflects the injection convention (ν ≈ 73× natural layer-0 norm; CE11). And **αc is attribution, not causal necessity**: a high-α component is not shown to sustain the attractor until it is ablated inside the loop (the F17 L11.H8 ablation is the template, and the Divine case shows one head can be decisive while others carry the bulk writes); ablation confirmation is gated and queues with M3.

### 3.3 What Part A cannot settle

Attribution at v\* is *local*: which components sustain this fixed point. It cannot decide why *this* direction rather than another (global question), whether the basin structure survives a different ν (the ν-sweep, gated), or whether the identity is seed-contingent (CE10/CE5: needs #73-class evidence, parked). State this in the writeup so the result is not over-claimed; over-claim-then-retract is this project's most expensive recurring failure mode.

### 3.4 Part C (global spectrum from weights alone): replace, don't run as written

#94 already lists four failure modes and flags the first ("layers compose, they do not sum") as possibly fatal. This review adds the decisive fact: **the repo has already run the within-layer version of this approximation and it failed.** H4, per-head resonance ≈ linear power iteration on static W_OV, was executed on 2026-07-25 and returned NOT SUPPORTED: 5/144 heads with cos > 0.9, "loop dynamics are dominated by nonlinear effects for most heads; per-head resonant state cannot be reduced to the static OV eigensystem alone" (`009c` artifacts; #54). A zeroth-order static-weights spectrum is the same idea one level up. Running it as a headline analysis invites a plausible-but-wrong result.

The sound replacement is already registered as **#71 M3**: the Jacobian spectrum of the *actual* map, linearised at v̂\*. The machinery exists and is validated (`08_hinge_eigenvalue.py`, jvp agreeing with finite differences to 3 to 4 figures; it produced the −4.3). Three technical points that make M3 stronger than #94 anticipates:

- **The right operator is the normalised map's Jacobian.** With u\* = v̂\*/ν the unit direction, J_G(v̂\*) = λ⁻¹(I − u\*u\*ᵀ)J_F(v̂\*): the rescale contributes exactly the radial projection (built from the unit vector, not the shell-norm state) and the 1/λ factor, so the raw J_F spectrum alone would be the wrong object.
- **Part C's failure mode 3 is neutralised at this specific fixed point.** Within the row-uniform invariant subspace of a position-collapsed state, perturbations leave all queries and keys equal, the softmax stays exactly uniform, and the QK pattern-derivative vanishes: the attention Jacobian restricted to that subspace is exactly the OV circuit under a uniform pattern. A genuine lever the issue doesn't exploit; it also shrinks the computation to the 768-dim subspace (Lanczos over tens to hundreds of jvps, not 768 full passes).
- **It comes with built-in validation the weight-sum version lacks:** |λ₂| must predict the measured settling half-life (5.6 to 11.4 iterations for GPT-2 Small ⇒ |λ₂| ≈ 0.88 to 0.94), and the count of |λ| > 1 directions must be 0 for a genuine attractor. The unexplained ~30-iteration settling latency (M2) is a non-normality/pseudospectrum question the same computation can address.

M3 answers what Part C was reaching for in a well-defined form (*is the fixed-point direction the dominant local mode, what is λ₂, how many escape directions exist*) and feeds #17 (basin geometry) and M4 directly. It needs forward passes → gated; it goes first in the post-gate queue alongside the noise re-run.

Keep the weight-only spectrum, if at all, as a pre-registered *approximation study*: compute it, compare against the M3 spectrum, and report how badly layer composition breaks it. That has methodological value; as a standalone claim about the model it has none. And #94's own scope note stands: even a successful spectrum gives local linear dominance, not global maximality of a nonlinear map with multiple known fixed points.

### 3.5 The tricky part, named

TC's instinct that this direction is "good but tricky" is exactly right, and the trickiness is specific: **every load-bearing scalar in #94 is inherited from analyses that were themselves corrected twice in the last fortnight** (M5's index conventions and figures via #85 → #87 → #89; the participation ratio by a factor of 7; the "inert rescale" withdrawal). The λ = 3.756 line is now solid *because* that correction chain ran to completion, but the chain is still unmerged (§7). The precondition discipline in §3.2 is what makes Part A trustworthy where earlier headline numbers were not: every number script-generated, every check explicit, controls before conclusions.

---

## 4. Two adjudications before the record moves

### 4.1 F4 / #97: verified; act now

Confirmed independently, twice, from primary source: `03_random_baseline.ipynb` cell 4 calibrates `MEAN_NORM` from Stage 1's *mean-vector* norms (≈397; `config.json`: 397.177) and cell 6 applies it to the *Frobenius* norm of each `[seq_len, 768]` noise tensor. Per-row scale ≈ 397/√T ≈ 126 vs the language runs' ≈ 400 to 460. The notebook's own cell-0 spec says "match the mean Frobenius norm"; git history shows the original version computed it correctly and the columnar-adapter commit introduced the substitution. The cell comment justifies it with the LayerNorm-invariance reasoning caveat 7 has since withdrawn. Second confound also confirmed: the schedule stops at iteration 100 and the run's own report records `Cosine convergence: NO (0.9256)`; 18 basins were counted on unconverged trajectories vs 5 at lock-in.

**The blast radius is wider than F4.** The mis-scaled constant was hardcoded downstream, so every noise arm in the repository inherits it: `04_readout_confidence.py:153` (F7's noise clause, F8's noise leg), `05_divine_motion.py:96` (F9's noise bracket, the invisibility-ratio 1.12 control, the 10.12× gain figure), `05_jlens_pilot.py:294` and `10_jlens_phase.py:186` (F11's and F16's language-vs-noise boundary compares states generated at ν differing ~3.5×). Not affected: F8's core prolet-coherence numbers (vocabulary-permutation nulls, ν-independent) and the F13/F14/F17 mechanism series (consumes `state_divine.pt` only).

**Actions:** (a) caveat F4 in FINDINGS now, listing the downstream inheritance above; free, stops propagation. (b) The matched-ν, gated-classification noise re-run is the first post-gate experiment (§5 T2-1). (c) Note the accidental ν contrast is *suggestive* (lower ν → more basins) but confounded with convergence state; the deliberate ν sweep (§6, one experiment) is the clean version.

### 4.2 F8 / #98: mechanism verified; adjudicate the percentiles before rewriting

#98's *mechanism* claim survived independent adversarial reproduction during this review, from raw HF GPT-2 weights with separate provenance: the uniform-token null sits at 0.268, an isotropic Gaussian decodes at 0.274, and a random multiple of **one** arbitrary token embedding decodes at coherence ≈ 0.46, at `prolet`'s level with zero semantic content. coherence@10 is also exactly temperature-invariant while entropy is not, so the entropy conditioning controls nothing. Every #98 sub-claim checkable against committed outputs verified exactly (the three identical prolet coherence values to 7 significant figures; trial_11 at 0.511; the Spearman −0.68). This is the same artifact class caveat 4 already retired at token level.

What remains unadjudicated is the *corpus percentile* measurement (665,600 ordinary GPT-2 outputs; `prolet` at the 78th percentile, the null at the 2.6th): plausible, self-validated against committed figures, but existing only as issue prose, exactly the reproducibility gap (#98 itself flags `chordness_formal.py` never existing) this project keeps paying for.

**Adjudication protocol:** (1) whoever ran #98 commits the pipeline plus a data manifest (corpus slices, seeds, matching windows) to a branch; (2) one independent pass re-runs it (ordinary forward passes over public corpora: analysis-class under the pause per #98's own scope note, but TC confirms); (3) then amend F8 (the mechanism part can cite this review's two independent reproductions), extend caveat 4 one level down, fix the 4-vs-2 sample count, and re-state the anomaly. Until then FINDINGS carries an "under challenge, see #98" flag on F8 and the README stops quoting the coherence result unqualified.

**The interaction neither issue states, and the correct joint reading:** #97 and #98 together do not *invert* which side is anomalous; they *suspend* the question. #98 (if its percentiles hold) de-anomalises the `prolet` side, while #97's verified mis-calibration makes the noise side's low, off-cone coherence unattributable (property of noise-seeded ATR vs artifact of the 10× gain regime) until a matched-ν re-run; trial_11 scoring 0.511 under the same wrong ν shows the confound is not deterministic in either direction. #98's item 3 ("re-point the anomaly") is therefore **held** pending T2-1; its item 2 (the one-line cone-position probe on committed tensors) is ungated and goes into the Part A run (§3.2), where it also serves CE6.

---

## 5. The plan

Three tiers. Tier 0 requires no ruling and no compute; Tier 1 is the STOP work plus its adjudications; Tier 2 is the post-gate experiment queue in order. Nothing else runs.

> **Execution log (2026-07-31).** TC approved the plan and removed the pause gate ("we remove the
> gate, that's my official position"), and delegated experiment ordering to this review. Executed the
> same day, in this PR: the #89 → #87 chain merged to main and #100 closed; the README, FINDINGS
> (F4/F5/F8 flags, caveats 18/18b, H-pos0 clause, prompt-library unblocking), JOURNEY_MAP, TECHNICAL,
> SCALING, RESULTS_SUMMARY, PYTHIA review and PRIOR_WORK (#99) fixes; ATR_PAUSE.md rewritten as
> lifted; the two resolved `.drift-allow` exemptions deleted. Still open in Tier 0: the two H4
> rulings (#54, TC's call), and #48, which turns out to need the local-only Stage 1 archives and so
> moves to the re-run work in Tier 2. Tier 1/Tier 2 below are now one continuous queue: the gate no
> longer separates them.

### Tier 0: record repair (immediately; analysis-free; STOP-exempt as bookkeeping)

1. **Land the correction chain: merge PR #89, then PR #87, then delete the branches.** Both are clean, analysis-only, checks passing. Close #100 (bot docstrings against a transient branch) unless it rebases trivially after the merges.
2. **H4 (#54): TC makes the two rulings** (does the regeneration run dispose of the hypothesis; is the `.detach()` deviation inert). Then all six stale statements update together: FINDINGS H4 row and line 883, JOURNEY_MAP §3, TECHNICAL.md:120, the README notebook table, and the notebook's cell-0 banner (README's own Knowledge Graph paragraph already states the truth, contradicting its own table). Closes #54, #25.
3. **Trivial drift fixes:** README caveats line ("permutation test not yet run" → resolved negative, caveat 4), closing #55, plus the matching JOURNEY_MAP §3 H3 row ("permutation test pending"), which contradicts JOURNEY_MAP's own Discoveries 9 and 10; then delete the `A/w-e-permutation-test` line from `docs/graph/.drift-allow` so `check_record_drift.py` enforces the fix. README's two "never converge / never stops moving" passages → the F9-governed "cycle, re-gated" phrasing. JOURNEY_MAP glossary's "glitch tokens ruled out" → superseded by F10/F13. Close #11 (the confidence-audit integration substantially landed the day it was filed; the residue is one optional README-narrative sentence, TC's wording call).
4. **Fold the wave into the record:** M1 → exactness settled-analytic in `contraction/RESULTS.md` + FINDINGS, withdrawing the float64 exactness-run recommendation, keeping M2's archive recommendation, and adding the direction-stability numbers computed in this review (§3.1) so the record shows what #94 assumes. **Propagate the caveat-7 withdrawal everywhere it never reached:** FINDINGS F5 result 1, SCALING_ARTEFACT_ANALYSIS's Current/Closing Judgements, RESULTS_SUMMARY's three restatements, and the flagged passage in PYTHIA_INTERPRETABILITY_REVIEW (board discussion #70). Amend the FINDINGS H-pos0 row's "c_n = 1" step (measured false, M5). Fix the sample-count language: F8's "4/4 states" and RESULTS' "all eight committed runs" → the archive holds 3 distinct states (two duplicate pairs), effectively 2 distinct `prolet` vectors. TECHNICAL.md corrections: the "energy-conservative, ‖xₙ‖ = ‖x₀‖ ∀n" claim (only the *injected* tensor is on-shell; recorded iterates sit at ~3.8×), the convergence-gate description (the gate runs on the mean-pooled vector, not a full-tensor cosine), the snapshot schedule (the 125-sweep ran [0..100], not the 5-prompt schedule), and the undocumented `gate_lag`/`renorm`/`inject_hook_name` extensions. #99's citation fixes plus the Greff/Jastrzębski/Fan additions and DEQ context on F14. Closes #99, advances #28.
5. **Protective caveats:** F4 caveat with the §4.1 inheritance list; F8 under-challenge flag (§4.2). Nothing else in FINDINGS changes until adjudication.
6. **Issue hygiene:** close #96 (owner-recommended, superseded by #98); update #91 (analytic half absorbed; E4 consolidated into the single ν-sweep design, §6); update #75 → close-candidate (engine half delivered by PR #82; remaining halves gated, tracked in T2); verify #24 is closable (prompt library restored per README provenance note).
7. **Board:** post this review to the agent board (done: PR #103's thread); annotate #79 (handover) with a pointer here.

### Tier 1: the STOP work (#94), sequenced

1. #98 adjudication protocol (§4.2), in parallel with the below; it changes #94's *motivation* text, not its method.
2. #94 preconditions P1 to P3 (§3.2): float64, committed tensors only, ungated.
3. **Part A at `prolet`**, then C1 contrast states, C2 null, neuron drill-down, X3, cone probe. One hooked forward pass per state.
4. **M3 Jacobian spectrum at v̂\*** as the Part C replacement (§3.4): gated; queued for the moment the gate lifts.
5. Writeup as new findings (F18+), with the pre-registered CE mapping (C3) and the §3.3 scope limits stated.

**The gate ruling: RESOLVED (2026-07-31).** TC removed the pause outright ("we remove the gate, that's my official position"); [ATR_PAUSE.md](ATR_PAUSE.md) is rewritten as lifted. Nothing in Tier 1 or Tier 2 is blocked by anything except its position in the queue below, whose ordering TC has delegated to this review.

### Tier 2: the experiment queue (in order; nothing jumps it)

1. **Matched-ν noise baseline re-run** (repairs F4 and every inherited noise control; makes #98's relocated anomaly decidable), under the standing archive spec below, with gated classification at lock-in, full lag table, per-position per-iteration float64 archives. Closes the #97 loop and the M2 gap in one run.
2. **The ν-sweep**: one experiment, currently filed twice (#27 E1 ≡ #91 E4; CE11's natural-energy control on Small is its ν → natural endpoint). Decides whether basin identity depends on the arbitrary rescale target, the live confound under every basin-identity claim.
3. **M3/M4** if not already run under Tier 1.
4. **#17 basin geometry**: fourth rather than first because items 1 and 2 protect existing published claims while #17 extends them. (This reordering is now settled: the pause document that named #17 as next is lifted, and TC delegated queue order to this review.)
5. Then, re-motivated by what Part A finds: #72 (distribution-level landscape), #84 Arm A (depth × iteration grid; its Arm B is ungated any time), #8 (J-lens full build), #76 (SAE decomposition, fidelity gate first), the H-pos0 n = 1 run, and the 33 unaudited `Divine` prompts.
6. **Cross-model track (#73, #74, #77, #78, the Pythia-410m lag re-gate) stays parked**: TC's explicit ruling (#91 comment) that generalisation is a different question from mechanism, and mechanism is the live one. The Pythia re-gate is flagged as the one parked item protecting an already-published claim (F3's fourth cell); it comes off the shelf first when this track reopens.

### Standing rules (all work, all agents, effective now)

1. **Every quoted number is regenerable by a committed script.** The #89 review found the only figures wrong in #87 were exactly the ones no script computed; #98 found a headline result whose generating code never existed. Hard rule; PR review checks it.
2. **Archive spec for any future run:** per-position tensors at every snapshot, `tensor_norm`, `seq_len`, `initial_norm`, position metrics in float64, full lag table at gate time, uniform snapshot gaps. (Each item is a documented past loss: M2's gap, M5's reconstruction, F9's aliasing, the mixed-gap estimator bias.)
3. **One engine.** ENGINE.md declares `atr_engine.py` canonical, yet at least six inline loop reimplementations exist with no equivalence check (the random-baseline notebook's copy is where the #97 bug lives). New work imports the engine or ships an equivalence test against it; the existing copies get flagged in T0-5's caveat.
4. **Identifiers via the board registry only** (discussion #53); descriptive directory names preferred; `EXP_011` remains unconfirmed.
5. **Record moves with the claim:** any PR that changes a finding's status updates FINDINGS + README + the knowledge graph in the same PR; `check_record_drift.py` gates it.
6. **Corrections land before dependent work starts.** The #85 → #87 → #89 chain sat unmerged while three issues (#91, #94, #97) built on its numbers; merges have also outrun reviews twice (#56, #58), both needing follow-up correction PRs. One branch per workstream (the knowledge-graph branch carried six unrelated PRs).
7. **STOP semantics (#94):** the STOP governs *new analysis and experiment work*; it does not forbid Tier 0 record repair. (The pause gate it previously deferred to was lifted 2026-07-31.) TC should edit #94's body to link this section, so the next agent does not have to infer it.
8. **Cross-model claims out of scope** until the mechanism phase concludes (TC's ruling, #91).
9. **Session working agreements bind agents too.** SESSION_03's standing rules (no em dashes anywhere in the repo; no decorative metaphors; grep before committing) were enforced by dedicated PRs and then ignored by later agent-authored docs, this review's own first draft included. Read `docs/sessions/` before writing.

---

## 6. Open-issue dispositions (all 32)

| # | Title (short) | Disposition |
|--:|:--|:--|
| 8 | J-lens full build | Park (T2-5). #84 Arm B absorbs the undelivered #8(b) acceptance item, ungated. |
| 10 | Chordness formalisation / shape-matched null | Hold pending #98 adjudication: the headline is superseded if the percentiles hold, and the program shrinks to the replacement cone probe. |
| 11 | Integrate confidence audit | **T0-3. Close** (substantially done the day it was filed; one README sentence is TC's wording call). |
| 12 | Sonification (art) | Park; independent track, post-mechanism. |
| 14 | Bell Program | Substantially delivered (F13 to F17). Update body; residue = 33 prompts (gated, plus #24) → T2-5. |
| 17 | Basin geometry | Post-gate queue position 4 (§5 T2-4; TC to confirm reorder vs ATR_PAUSE). |
| 24 | Prompt library | Verify closable: restoration committed with provenance flags (README note). |
| 25 | EXP_009c / H4 spectral | Resolves with #54 rulings (T0-2); residue is only the eigenvector-vs-singular-vector addendum. |
| 26 | Dynamical-systems positioning | Absorb into #28 register; no separate work item. |
| 27 | Normalisation as absorption | E1 consolidated into the single ν-sweep (T2-2); the "inert" half already resolved via caveat 7. |
| 28 | Open-questions register | Update in T0-4 to fold in the wave. |
| 48 | Truncated dissolution tables | Re-disposed 2026-07-31: regenerating the tables needs the gitignored, local-only Stage 1 archives, so this is not a doc fix; it rides with the Tier 2 re-run work. |
| 51 | Cross-repo knowledge graph | Park; infrastructure, post-mechanism. |
| 54 | H4 record contradiction | **T0-2. TC's two rulings, then close.** |
| 55 | README vs FINDINGS permutation drift | **T0-3. Close this week.** (Verified still true on main today; fix `.drift-allow` with it.) |
| 71 | M1 to M5 dynamics metrics | M1 exactness settled-analytic; M2 → archive spec; M3/M4 → §3.4 + T2-3; M5 partly delivered. Update body, keep open as metric register. |
| 72 | Cluster-level landscape | Park to T2-5; interacts with #98 adjudication. |
| 73 | CRFM seed sweep | Parked (cross-model ruling). The CE5/CE10 discriminator when the track reopens. |
| 74 | GPT-2 size ladder | Parked (cross-model ruling). |
| 75 | prepend_bos | Engine half delivered (PR #82); close-candidate, remaining halves gated → T2-5. |
| 76 | SAE decomposition | Park; fidelity gate is analysis-only and may run opportunistically. |
| 77 | Matched-capacity control | Parked (cross-model ruling). |
| 78 | Modern-model scale test | Parked (cross-model ruling). |
| 79 | Handover | Historical; annotate with pointer here (T0-7). |
| 84 | Depth × iteration lens grid | Arm B ungated, optional after Tier 1 core; Arm A → T2-5. |
| 91 | M1 analytic + E-list | Analytic half → record (T0-4); E4 → ν-sweep (T2-2); then close. |
| 92 (PR) | WebText provenance audit | Independent track (CE1-adjacent); park as draft until after Tier 1 unless TC wants it moving. |
| 94 | **STOP: fixed-point reverse-engineering** | **The priority.** Execute per §3.2/§5 Tier 1; edit body per §5 rule 7; Part C → §3.4 replacement. |
| 95 | CE1 to CE11 register | Keep open as the explanation registry Part A discriminates against (C3). |
| 96 | Self-prediction mechanism | **Close** (owner-recommended; superseded by #98). X3 carried into Tier 1. |
| 97 | F4 norm bug | Verified twice. Caveat lands T0-5; re-run is T2-1; then close. |
| 98 | F8 null challenge | Mechanism verified twice; adjudicate percentiles per §4.2; items: (1) after adjudication, (2) into Tier 1, (3) held for T2-1, (4) respected. |
| 99 | PRIOR_WORK corrections | **T0-4. Close this week.** |

## 7. Open-PR dispositions

| PR | Disposition |
|--:|:--|
| **#87** | Merge (after #89). Clean, analysis-only, checks pass; its M1 "higher-precision run required" conclusion is then immediately annotated per T0-4 (exactness settled analytically): merge first, annotate second, so the correction chain's history stays legible. |
| **#89** | Merge first (into #87's branch). The review it implements is the origin of standing rule 1. |
| **#100** | Close unless it rebases trivially post-merge; bot docstrings do not justify keeping a stacked branch alive. |
| **#92** | Keep as parked draft (see #92 row above). |

---

## 8. What this project is now

A study that found, and can now precisely state, one object: **GPT-2 Small's iterated, renormalised forward map sends 43% of language-derived starting states to a single weight-determined direction with per-pass gain ≈ 3.76, and sends one other family to an exact period-2 cycle executed by one attention head.** The corpus-fingerprint reading is dead; the coherence reading is under adjudication; what is left is harder and better: an exactly decomposable fixed point, a mechanism series that already took one such object apart (F13 to F17), and a method (#94 Part A) for taking apart the other. The record's credibility asset, five self-refutations honestly documented, is worth more than any single finding; the standing rules in §5 exist to keep it.

**The instruction set, in one line each:** repair the record (Tier 0, this week); adjudicate #98's percentiles with committed code; execute #94 Part A with the §3.2 preconditions; replace Part C with M3; then run the noise re-run and the ν-sweep before anything new (the pause gate is lifted; only the queue order governs); keep everything else parked until those decide what it means.
