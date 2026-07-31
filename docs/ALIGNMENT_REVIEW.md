# ATR — Alignment Review and Direction

**Date:** 2026-07-31
**Type:** leadership review — full pass over the repository, all 32 open issues, all open and recently merged PRs, the agent board, and the session records.
**Commissioned by:** TC, with the brief: *examine all claims, directions and hypotheses; produce a general report and a set of instructions to get the whole project aligned and where it needs to go.*
**Governing relationship:** [FINDINGS.md](FINDINGS.md) remains the canonical record of *results*. This document governs *work sequencing and dispositions* until TC supersedes it. Where this review and an open issue disagree about priority, this review states the reconciliation and names the issue.

Everything asserted below was verified against primary sources during this review — code, committed artifacts, issue text, PR diffs — not summarised from prose. Where a claim rests on an issue's own unverified measurement, that is said explicitly.

---

## 1. Where the project actually is

The arc so far: a Lucier-inspired feedback loop on GPT-2 Small found five semantically coherent attractor basins; the founding corpus-fingerprint hypothesis was refuted by the project's own cross-model validation; the survivors were a cheap dynamical probe, one fully-traced mechanism (the `Divine` period-2 cycle, F9–F17), and one anomaly (why GPT-2 Small alone). A readout-audit series then found the basins are carried by broad, low-confidence distributions (F7, F8). Execution was then deliberately paused behind an understanding gate ([ATR_PAUSE.md](ATR_PAUSE.md)).

Then, 2026-07-28 → 07-31, the **mathematics-foundations wave** — the most consequential four days in the project's record:

1. **M1 was settled analytically** (#91, independently confirmed): position collapse is *forced* — a row-uniform state is an exactly invariant subspace of the map, so the collapse is architecture, not dynamics. The residual ~1e-7 in the archives is float32 rounding inside that subspace. The gated float64 M1 run is no longer needed; the open empirical question is the *rate* of collapse (M2), which no committed archive can answer.
2. **The rescale factor was measured from committed data** (M5, via PR #85/#87/#89): the loop rescales by *c* ≈ 0.27–0.29 (language) / 0.099 (noise) per pass — a stable constant, emphatically not 1 — and the map is not scale-invariant over that range (#69, caveat 7). The fixed point is a fixed point of *rescale ∘ F*, with per-pass gain λ = 1/*c* ≈ 3.47–3.76 (language) / 10.1 (noise).
3. **The core object sharpened.** 54 of 125 prompts converge to *one vector* — pairwise cosine 0.9987–1.0000, identical to seven significant figures across unrelated prompts (#98, F11). The attractor carries no information about the input. The thing to explain is a single direction fixed by the weights, with a decomposable gain.
4. **The STOP directive** (#94, 2026-07-29): reverse-engineer the `prolet` fixed point from the weights — exact per-component attribution at v* (Part A), then a global self-coupling spectrum (Part C). This is the mathematics-foundations direction, and it is the right one. §3 assesses it.
5. **Two findings came under credible challenge** the same week: F4's noise baseline was mis-calibrated by ~1/√T (#97 — **verified in this review**, §4.1), and F8's coherence null was measured against what GPT-2 actually emits and found evidentially empty (#98 — credible, **not yet independently verified**, §4.2).

So the project is mid-pivot from *cataloguing phenomena* to *reverse-engineering one object*. The pivot is correct. What makes it tricky is that three of the claims the new work leans on need repair first, the record has drifted in five documented places, and the STOP directive, the pause gate, and the owner's own subsequent issue filings are pulling agents in three directions. The rest of this document is the reconciliation.

---

## 2. Claim-by-claim status after the wave

Status of every principal claim, cross-checked against the July 29–31 issues. "Canonical" = what FINDINGS.md currently says; "actual" = what this review concludes the evidence supports today.

| Claim | Canonical | Actual status after the wave | Action |
|:--|:--|:--|:--|
| **F1** five basins at lock-in | supported | Stands at argmax level. Distribution level already qualified in-record ("fewer than five objects"); sharpened by the one-vector fact (#98): `prolet` is a single direction, not a region. | None now; language tightens under §5 T0-6 |
| **F2/F9** `Divine` = period-2 cycle | resolved | Stands. Untouched by the wave. One trajectory (caveat 14). | None |
| **F3** landscape doesn't generalise | supported | Stands. The Pythia-410m "no consolidation" cell remains **provisional** — produced by the lag-1 gate F15 proved period-blind; the full-lag re-gate needs local-only trajectories or new runs (gated). | Flag stays; parked with cross-model track |
| **F4** noise → 18 basins, language → 5 | supported | **Confounded, verified.** Noise arm ran at ~1/√T of matched Frobenius norm (gain regime 10.1× vs 3.5×) *and* was counted at iteration 100 unconverged vs language at lock-in. Either alone could produce the gap. | §5 T0-5: caveat now; re-run post-gate (§5 T2-1) |
| **F5** differences intrinsic, not apparatus | qualified | Stands as "three channels exonerated" — but **F5's result 1 ("the global L2 rescale is effectively invisible to the forward pass") still states the reasoning caveat 7 withdrew on 2026-07-28**, and the withdrawal was never propagated to SCALING_ARTEFACT_ANALYSIS's closing judgement or RESULTS_SUMMARY (three places). Tokenisation channel (caveat 17) still open; CE11 (73× natural-norm injection, #95) deepens the apparatus question. | §5 T0-4: propagation sweep |
| **F6** cross-hardware replication | supported | Stands. | None |
| **F7** confidence inversion | supported | Stands — #98's pipeline independently *reproduced* the F7 numbers while attacking F8. | None |
| **F8** `prolet` coherence, p = 0.001 | supported | **Under credible challenge** (#98): the 0.27 null sits at the 2.6th percentile of 665,600 real GPT-2 outputs; `prolet` at 0.471 is a 78th-percentile ordinary output; effective sample is 2, not 4; `chordness_formal.py` never existed. Not yet independently verified. | §4.2 adjudication protocol before any rewrite |
| **F11/F16** J-lens pilot nulls | not supported (H-J1) | Stand, at pilot confidence. | None |
| **F13/F14/F17** flip axis, −4.3, L11.H8 | supported | Stand. One trajectory. #99 adds the DEQ/Jacobian-regularisation literature as the proper frame for F14. | §5 T0-4 doc fix |
| **F15** lag-k gate | delivered | Stands. | None |
| **H4** per-head resonance ≈ W_OV power iteration | "untested" | **Record is wrong — verified.** Executed 2026-07-25 (PR #29 regeneration), NOT SUPPORTED: 5/144 heads > 0.9. Artifacts committed (`experiments/_DATA/EXP_009/009c_*.pt`). Needs TC's two rulings in #54 (dispositive? deviations inert?). | §5 T0-2 |
| **H-pos0** attractor is fixed point of single-position map | registered | Premise (exact collapse) now holds *analytically* (#91). The *c* = 1 step is corrected (M5) — **but the FINDINGS H-pos0 row still asserts "at a settled state ‖xⁿ‖ is constant, c_n = 1," which M5 measured false** (c = 0.288/0.099/0.266). Test unblocked by the `prepend_bos` engine change (PR #82, merged). Gated. | §5 T0-4: amend the row |
| **H-fingerprint / H-till / H-supp / H3** | refuted / withdrawn | Stand as recorded. | None |
| **M1** exact position collapse | "not settled" (contraction/RESULTS.md) | **Settled analytically** (#91, confirmed independently; float32-floor measurements are the consistency check). The "higher-precision run" recommendation should be withdrawn. | §5 T0-4 |
| **M2** collapse rate | data-gapped | Correct as recorded: no archive holds per-position per-iteration tensors. Fix is the archive spec (§5, standing rules), not a new metric. | Standing rule |
| **M3** Jacobian spectrum at attractors | gated, registered (#71) | **This is the sound version of #94's Part C** (§3.4). | Folded into #94 protocol |
| **M5** rescale factor | partly delivered | Settled value measured; transient (iters 1–99) unobserved. "Not gated" per PR #87 (engine records `tensor_norm`; the loss was at save time). | Standing rule |

**Hypothesis-numbering note:** Stage 1 owns H0–H4, Stage 2 (sibling repo) owns H5–H8, H11 is registry-assigned. #95's candidate explanations are correctly renumbered CE1–CE11. Any new hypothesis takes the next free number *only* via the board's Identifier registry (discussion #53). `EXP_010` is held by the sibling repo; `EXP_011` has been twice declined and never confirmed free — keep using descriptive directory names.

---

## 3. The mathematics-foundations direction: assessment

This is the review's centrepiece, per the brief. Short version: **Part A of #94 is sound and should be executed as specified below; Part C as written is already contradicted by the repo's own H4 data and should be replaced by the registered M3; the eigenvalue framing closes, but with an error budget that must be stated rather than assumed.**

### 3.1 The framing is correct — with one obligation

#94 writes the fixed point as F(v̂\*) = λv̂\*, λ = ‖v\*‖/ν ≈ 3.756. Verified against the engine (atr_engine.py:212–235): the loop is *rescale to ν, then forward*; every recorded state is post-forward/pre-rescale; so the injected state v̂\* is on the shell ‖v̂\*‖ = ν and F(v̂\*) is the recorded next state. At a converged state F(v̂\*) ∝ v̂\* and the constant is exactly the M5 amplification. The residual-sum decomposition F(x) = x + Σₗ[Aₗ(x) + Mₗ(x)] is valid at the read site (`blocks.11.hook_resid_post` is upstream of `ln_f`). The arithmetic (5230.65 / 1392.65 = 3.756) checks against `experiments/contraction/RESULTS.md`.

**The obligation:** the eigenvalue equation is exact only if direction convergence is exact, and the committed evidence for `prolet` is a gate-level cosine (flat 1.0000000 at every lag over a 24-iteration continuation, F15) — excellent, but a tolerance, not zero. A state with cos(F(v̂\*), v̂\*) = 1 − ε carries an orthogonal residual of norm ≈ √(2ε)·‖F(v̂\*)‖. Therefore **"Σαc = λ − 1 is a built-in correctness check" must be stated with an explicit error budget**: measure cos(F(v̂\*), v̂\*) in float64 at the state actually used, derive the tolerance, and require the decomposition to close within it — not "exactly."

### 3.2 Part A (exact component attribution) — endorsed, with two corrections and preconditions

#94's own caveats (orthogonal-write blindness; position collapse as the well-definedness condition; the w_in·W_E read degrading with depth; the ⟨w_out, w_in⟩ self-coupling shortcut being a cross-iteration quantity a within-pass dot product cannot capture) are all correct as written. Two things in the issue's formulation are not, and both would silently break the correctness check:

- **The decomposition equation as literally written — Σₗ[Aₗ(v̂\*) + Mₗ(v̂\*)] = (λ−1)v̂\* — is false.** Component *l* acts on the *accumulated mid-pass residual*, not on v̂\*. The exact object is each component's **actual write during the single forward pass initialised at v̂\*** — read via `hook_attn_out` / `hook_mlp_out` (and `z @ W_O` for the per-head split). #94's author demonstrably knows this (it is Part C's failure mode 1), but the Part A statement invites the naive isolated-application implementation, whose sum would *not* equal (λ−1)v̂\*. Implement against the hooks.
- **The "156 components" omit the biases.** `hook_mlp_out` includes `b_out`, but the per-head `z @ W_O` split drops each layer's `attn.b_O` (12 terms). The advertised Σαc = 2.76 check will not close as specified. Do the **24-term closure first** (per-layer `hook_attn_out` + `hook_mlp_out`, which is exact), verify it against λ − 1 within the §3.1 error budget, then refine to heads + explicit bias terms.

Preconditions and controls, in order:

- **P1 — frame check.** Compute everything at the on-shell state v̂\* (rescaled committed tensor — `output_divine_motion/state_prolet.pt`, present in the clone; confirm ‖x\*‖ = 5230.65, ν = 1392.65, T = 12), in float64. Verify rank-1 (σ₂/σ₁ ~ 1e-7 is already committed; recompute at the state used). Hard-symmetrising the rows to their exact mean is legitimate if wanted — the row-uniform subspace is exactly invariant (#91).
- **P2 — error budget.** The archived state is an iteration-1000 *stop*, not a proven exact fixed point. Before attributing anything, run the single pass, report λ_check = ‖F(v̂\*)‖/ν against 3.756 and 1 − cos(F(v̂\*), v̂\*) in float64; that residual is the error bar the Σαc closure is judged against (§3.1).
- **P3 — the state count is 2, not 4.** `Semantic` ≡ `Control_prolet_Semantic`, `Syntactic` ≡ `Divine_Syntactic` (#91 archive note; #98). Attribution runs on distinct states, and the writeup says how many there are.
- **C1 — contrast states.** Run the identical attribution at the `Divine` pivot M and both phases (the committed cycle states), and at one noise attractor — the latter carrying an explicit flag that noise states were produced under the #97 mis-calibration (10× gain regime) and are not a clean control until the re-run (§5 T2-1).
- **C2 — null distribution.** The αc concentration profile for ~100 random on-shell directions through the same pass, so "the prolet state is held up by few/many components" is a statement against a measured baseline, not intuition. (This is the same discipline that saved F13 — the norm-matched null was the sharp control there.)
- **C3 — CE mapping, pre-registered.** #95 exists so that Part A discriminates rather than narrates: concentrated interpretable components → CE9/CE4; diffuse structureless contributions → CE5/CE8 strengthen and "why socialist" may stop being well-posed. Write the mapping down *before* looking (#95 already does most of this; link it in the run report).

Neuron-level drill-down (rank ⟨w_out_j, v̂\*⟩·aj, decode top neurons through ln_f → W_Eᵀ) as #94 specifies, with its own depth caveat kept — **and with #98-corrected nulls on every decode step**: coherence-style eyeballing of "what tokens this neuron promotes" is exactly the cone artifact under adjudication (a random multiple of one token embedding out-scores every `prolet` state on the old statistic), so neuron-decode claims use the ordinary-output percentile null or the cone-position probe, not qualitative neighbourhood reading. Add **X3 from #96** (the one experiment that survived that issue's retractions): is L11.H8's W_O output direction anti-aligned with the frequency pole? Weights-only, ungated, and it slots naturally beside Part A's head-level table.

Two honest scope caveats to carry into the writeup: **α and λ are shell-indexed quantities, not weight-intrinsic ones** — they depend on the arbitrary seed norm ν because F is not scale-invariant (caveat 7), and since ‖Δ‖ is near-constant across runs (≈3620–3838, 6% spread) the headline gain λ − 1 ≈ ‖Δ‖/ν substantially reflects the injection convention (ν ≈ 73× natural layer-0 norm; CE11). And **αc is attribution, not causal necessity**: a high-α component is not shown to sustain the attractor until it is ablated inside the loop (the F17 L11.H8 ablation is the template, and the Divine case shows one head can be decisive while others carry the bulk writes) — ablation confirmation is gated and queues with M3.

### 3.3 What Part A cannot settle

Attribution at v\* is *local*: which components sustain this fixed point. It cannot decide why *this* direction rather than another (global question), whether the basin structure survives a different ν (E4/E1, gated), or whether the identity is seed-contingent (CE10/CE5 — needs #73-class evidence, parked). State this in the writeup so the result is not over-claimed — over-claim-then-retract is this project's most expensive recurring failure mode.

### 3.4 Part C (global spectrum from weights alone) — replace, don't run as written

#94 already lists four failure modes and flags the first ("layers compose, they do not sum") as possibly fatal. This review adds the decisive fact: **the repo has already run the within-layer version of this approximation and it failed.** H4 — per-head resonance ≈ linear power iteration on static W_OV — was executed on 2026-07-25 and returned NOT SUPPORTED: 5/144 heads with cos > 0.9, "loop dynamics are dominated by nonlinear effects for most heads; per-head resonant state cannot be reduced to the static OV eigensystem alone" (`009c` artifacts; #54). A zeroth-order static-weights spectrum is the same idea one level up. Running it as a headline analysis invites a plausible-but-wrong result.

The sound replacement is already registered as **#71 M3**: the Jacobian spectrum of the *actual* map, linearised at v̂\* — the machinery exists and is validated (`08_hinge_eigenvalue.py`, jvp agreeing with finite differences to 3–4 figures; it produced the −4.3). Three technical points that make M3 stronger than #94 anticipates:

- **The right operator is the normalised map's Jacobian**, J_G(v̂\*) = λ⁻¹(I − v̂\*v̂\*ᵀ)J_F(v̂\*): the rescale contributes exactly the radial projection and the 1/λ factor, so the raw J_F spectrum alone would be the wrong object.
- **Part C's failure mode 3 is neutralised at this specific fixed point.** Within the row-uniform invariant subspace of a position-collapsed state, perturbations leave all queries and keys equal, the softmax stays exactly uniform, and the QK pattern-derivative vanishes — the attention Jacobian restricted to that subspace is exactly the OV circuit under a uniform pattern. A genuine lever the issue doesn't exploit; it also shrinks the computation to the 768-dim subspace (Lanczos over tens–hundreds of jvps, not 768 full passes).
- **It comes with built-in validation the weight-sum version lacks:** |λ₂| must predict the measured settling half-life (5.6–11.4 iterations for GPT-2 Small ⇒ |λ₂| ≈ 0.88–0.94), and the count of |λ| > 1 directions must be 0 for a genuine attractor. The unexplained ~30-iteration settling latency (M2) is a non-normality/pseudospectrum question the same computation can address.

M3 answers what Part C was reaching for in a well-defined form — *is the fixed-point direction the dominant local mode, what is λ₂, how many escape directions exist* — and feeds #17 (basin geometry) and M4 directly. It needs forward passes → gated; it goes first in the post-gate queue alongside the noise re-run.

Keep the weight-only spectrum, if at all, as a pre-registered *approximation study*: compute it, compare against the M3 spectrum, and report how badly layer composition breaks it. That has methodological value; as a standalone claim about the model it has none. And #94's own scope note stands: even a successful spectrum gives local linear dominance, not global maximality of a nonlinear map with multiple known fixed points.

### 3.5 The tricky part, named

TC's instinct that this direction is "good but tricky" is exactly right, and the trickiness is specific: **every load-bearing scalar in #94 is inherited from analyses that were themselves corrected twice in the last fortnight** (M5's index conventions and figures via #85 → #87 → #89; the participation ratio by a factor of 7; the "inert rescale" withdrawal). The λ = 3.756 line is now solid *because* that correction chain ran to completion — but the chain is still unmerged (§7). The precondition discipline in §3.2 is what makes Part A trustworthy where earlier headline numbers were not: every number script-generated, every check explicit, controls before conclusions.

---

## 4. Two adjudications before the record moves

### 4.1 F4 / #97 — verified; act now

This review independently confirmed the bug from primary source: `03_random_baseline.ipynb` cell 4 calibrates `MEAN_NORM` from Stage 1's *mean-vector* norms (≈397; `config.json`: 397.177) and cell 6 applies it to the *Frobenius* norm of each `[seq_len, 768]` noise tensor. Per-row scale ≈ 397/√T ≈ 126 vs the language runs' ≈ 400–460. The cell comment even repeats the LayerNorm-invariance justification that caveat 7 has since withdrawn. Second confound also confirmed: the schedule stops at iteration 100 and the run's own report records `Cosine convergence: NO (0.9256)` — 18 basins counted on unconverged trajectories vs 5 at lock-in.

**Actions:** (a) caveat F4 in FINDINGS now — free, stops propagation into README/citations; (b) the matched-ν, gated-classification noise re-run is the first post-gate experiment (§5 T2-1); (c) note that the accidental ν contrast is *suggestive* that lower ν → more basins but is confounded with convergence state — a deliberate ν sweep (E1/E4, one experiment, §6) is the clean version.

### 4.2 F8 / #98 — credible; adjudicate before rewriting

#98 is methodologically serious (it reproduced the repo's own committed figures to 4+ decimals before attacking them, ran two corpora, and pre-empted the obvious rescues). If it holds, F8's p = 0.001 is real arithmetic against an irrelevant null, F8 needs re-statement, caveat 4 extends one level down, and the anomaly re-points at the noise attractors. But its 665,600-distribution measurement exists **only as prose in an issue** — the exact reproducibility gap (#98 itself flags `chordness_formal.py` never existing) this project keeps paying for.

**Adjudication protocol:** (1) whoever ran #98 commits the pipeline + a data manifest (corpus slices, seeds, the entropy/p₁-matching windows) to a branch; (2) one independent pass re-runs it (ordinary forward passes over public corpora — analysis-class under the pause, per #98's own scope note, but TC confirms); (3) only then amend F8, extend caveat 4, fix the 4-vs-2 sample count, and re-state the anomaly. Until then FINDINGS carries an "under challenge, see #98" flag on F8 and the README stops quoting the coherence result unqualified.

**The interaction neither issue states:** #98's relocated anomaly — noise attractors below the 5th percentile of ordinary outputs — rests on noise states produced under #97's mis-calibration, in a 10× gain regime the language states never entered. Whether "noise decodes off-cone" is a fact about noise or about ν is unknowable until the matched-ν re-run. #98's item 3 ("re-point the anomaly") is therefore **held** pending T2-1; its item 2 (the one-line cone-position probe on committed tensors: cos(ln_f(x), mean embedding)) is ungated and goes into the Part A run (§3.2), where it also serves CE6.

---

## 5. The plan

Three tiers. Tier 0 requires no ruling and no compute; Tier 1 is the STOP work plus its adjudications; Tier 2 is the post-gate experiment queue in order. Nothing else runs.

### Tier 0 — record repair (immediately; analysis-free; STOP-exempt as bookkeeping)

1. **Land the correction chain: merge PR #89, then PR #87, then delete the branches.** Both are clean, analysis-only, checks passing. Close #100 (bot docstrings against a transient branch) unless it rebases trivially after the merges. *(Done as part of this review's follow-up if TC approves the dispositions.)*
2. **H4 (#54): TC makes the two rulings** (does the regeneration run dispose of the hypothesis; is the `.detach()` deviation inert). Then all four stale statements update together: FINDINGS H4 row, JOURNEY_MAP §3, TECHNICAL.md:120, the README notebook table, and the notebook's cell-0 banner (README's own Knowledge Graph paragraph already states the truth, contradicting its own table). Closes #54, #25.
3. **Trivial drift fixes:** README caveats line ("permutation test not yet run" → resolved negative, caveat 4) — closes #55, and the matching JOURNEY_MAP §3 H3 row ("permutation test pending"), which contradicts JOURNEY_MAP's own Discoveries 9–10. README's two "never converge / never stops moving" passages → the F9-governed "cycle, re-gated" phrasing. JOURNEY_MAP glossary's "glitch tokens ruled out" → superseded by F10/F13. Integrate the confidence-audit into README's claims language — closes #11. De-truncate the dissolution tables — closes #48.
4. **Fold the wave into the record:** M1 → settled-analytic in `contraction/RESULTS.md` + FINDINGS (withdraw the float64-run recommendation; keep M2's archive recommendation). **Propagate the caveat-7 withdrawal everywhere it never reached:** FINDINGS F5 result 1 ("effectively invisible to the forward pass"), SCALING_ARTEFACT_ANALYSIS's Current/Closing Judgements ("normalisation is inert…", "ruled-out artefact: normalisation"), and RESULTS_SUMMARY's three restatements of the withdrawn LayerNorm-invariance reasoning. Amend the FINDINGS H-pos0 row's "c_n = 1" step (measured false, M5). Fix the sample-count language: F8's "4/4 states" and RESULTS' "all eight committed runs" → the archive holds 3 distinct states (two duplicate pairs), effectively 2 distinct `prolet` vectors. #99's citation fixes + the Greff/Jastrzębski/Fan additions + DEQ context on F14. Closes #99, advances #28.
5. **Protective caveats:** F4 caveat (§4.1); F8 under-challenge flag (§4.2). Nothing else in FINDINGS changes until adjudication.
6. **Issue hygiene:** close #96 (owner-recommended, superseded by #98); update #91 (analytic half absorbed into the record; E4 consolidated into the single ν-sweep design, §6); update #75 (engine half delivered by PR #82; remaining halves gated); verify #24 is closable (prompt library restored per README provenance note).
7. **Board:** post this review to the agent board; annotate #79 (handover) with a pointer here.

### Tier 1 — the STOP work (#94), sequenced

1. #98 adjudication protocol (§4.2) — runs in parallel with the below; it changes #94's *motivation* text, not its method.
2. #94 preconditions P1–P3 (§3.2) — float64, committed tensors only, ungated.
3. **Part A at `prolet`**, then C1 contrast states, C2 null, neuron drill-down, X3, cone probe. One hooked forward pass per state.
4. **M3 Jacobian spectrum at v̂\*** as the Part C replacement (§3.4) — gated; queued for the moment the gate lifts.
5. Writeup as new findings (F18+), with the pre-registered CE mapping (C3) and the §3.3 scope limits stated.

**The one ruling Tier 1 needs from TC:** Part A requires single hooked forward passes at committed states. [ATR_PAUSE.md](ATR_PAUSE.md) blocks "new ATR experiments"; #98's scope note already treats plain forward passes as outside that definition, but it is TC's definition, not ours. Two clean options: **(a) pass the gate first** — the recommended one, because #94 *is* the understanding the gate was built to demand, and writing the Part A code (ungated) is itself preparation for it; or **(b) rule single no-loop hooked passes as analysis.** What is not clean is agents deciding this severally, which is what will happen if it stays unruled.

### Tier 2 — post-gate experiment queue (in order; nothing jumps it)

1. **Matched-ν noise baseline re-run** (repairs F4; makes #98's relocated anomaly decidable) — under the standing archive spec below, with gated classification at lock-in, full lag table, per-position per-iteration float64 archives. Closes the #97 loop and the M2 gap in one run.
2. **The ν-sweep** — one experiment, currently filed three times (#27 E1 ≡ #91 E4; CE11's natural-energy control on Small is its ν → natural endpoint). Decides whether basin identity depends on the arbitrary rescale target — the live confound under every basin-identity claim.
3. **M3/M4** if not already run under Tier 1.
4. **#17 basin geometry** — the ATR_PAUSE-signposted experiment, now fourth rather than first because items 1–2 protect existing published claims while #17 extends them. TC should confirm this reordering, since ATR_PAUSE names #17 explicitly.
5. Then, re-motivated by what Part A finds: #72 (distribution-level landscape), #84 Arm A (depth × iteration grid; its Arm B is ungated any time), #8 (J-lens full build), #76 (SAE decomposition, fidelity gate first), the H-pos0 n = 1 run, and the 33 unaudited `Divine` prompts.
6. **Cross-model track (#73, #74, #77, #78, Pythia-410m lag re-gate) stays parked** — TC's explicit ruling (#91 comment): generalisation is a different question from mechanism, and mechanism is the live one. The Pythia re-gate is flagged as the one item here protecting an already-published claim (F3's fourth cell) — it comes off the shelf first when this track reopens.

### Standing rules (all work, all agents, effective now)

1. **Every quoted number is regenerable by a committed script.** The #89 review found that the only figures wrong in #87 were exactly the ones no script computed; #98 found a headline result whose generating code never existed. This is now a hard rule; PR review checks it.
2. **Archive spec for any future run:** per-position tensors at every snapshot, `tensor_norm`, `seq_len`, `initial_norm`, position metrics in float64, full lag table at gate time, uniform snapshot gaps. (Each item is a documented past loss: M2's gap, M5's reconstruction, F9's aliasing, the mixed-gap estimator bias.)
3. **Identifiers via the board registry only** (discussion #53); descriptive directory names preferred; `EXP_011` remains unconfirmed.
4. **Record moves with the claim:** any PR that changes a finding's status updates FINDINGS + README + the knowledge graph in the same PR; `check_record_drift.py` gates it.
5. **STOP semantics (#94):** the STOP governs *new analysis and experiment work* — it does not forbid Tier 0 record repair, and it does not override the pause gate. TC should edit #94's body to link this section, so the next agent does not have to infer it.
6. **Cross-model claims out of scope** until the mechanism phase concludes (TC's ruling, #91).

---

## 6. Open-issue dispositions (all 32)

| # | Title (short) | Disposition |
|--:|:--|:--|
| 8 | J-lens full build | Park (T2-5). #84 Arm B partially serves the (b) debt ungated. |
| 10 | Chordness formalisation / shape-matched null | Hold pending #98 adjudication — if F8 falls, this program shrinks to the replacement probe. |
| 11 | Integrate confidence audit | **T0-3. Close this week.** |
| 12 | Sonification (art) | Park; independent track, post-mechanism. |
| 14 | Bell Program | Substantially delivered (F13–F17). Update body; residue = 33 prompts (gated + #24) → T2-5. |
| 17 | Basin geometry | Post-gate queue position 4 (§5 T2-4; TC to confirm reorder vs ATR_PAUSE). |
| 24 | Prompt library | Verify closable — restoration committed with provenance flags (README note). |
| 25 | EXP_009c / H4 spectral | Resolves with #54 rulings (T0-2). |
| 26 | Dynamical-systems positioning | Absorb into #28 register; no separate work item. |
| 27 | Normalisation as absorption | E1 consolidated into the single ν-sweep (T2-2); the "inert" half already resolved via caveat 7. |
| 28 | Open-questions register | Update in T0-4 to fold in the wave. |
| 48 | Truncated dissolution tables | **T0-3. Close this week.** |
| 51 | Cross-repo knowledge graph | Park; infrastructure, post-mechanism. |
| 54 | H4 record contradiction | **T0-2. TC's two rulings, then close.** |
| 55 | README vs FINDINGS permutation drift | **T0-3. Close this week.** (Verified still true on main today.) |
| 71 | M1–M5 dynamics metrics | M1 settled-analytic; M2 → archive spec; M3/M4 → §3.4 + T2-3; M5 partly delivered. Update body, keep open as metric register. |
| 72 | Cluster-level landscape | Park to T2-5; interacts with #98 adjudication. |
| 73 | CRFM seed sweep | Parked (cross-model ruling). The CE5/CE10 discriminator when the track reopens. |
| 74 | GPT-2 size ladder | Parked (cross-model ruling). |
| 75 | prepend_bos | Engine half delivered (PR #82). Remaining halves gated → T2-5. Update body. |
| 76 | SAE decomposition | Park; fidelity gate is analysis-only and may run opportunistically. |
| 77 | Matched-capacity control | Parked (cross-model ruling). |
| 78 | Modern-model scale test | Parked (cross-model ruling). |
| 79 | Handover | Historical; annotate with pointer here (T0-7). |
| 84 | Depth × iteration lens grid | Arm B ungated, optional after Tier 1 core; Arm A → T2-5. |
| 91 | M1 analytic + E-list | Analytic half → record (T0-4); E4 → ν-sweep (T2-2); then close. |
| 92 (PR) | WebText provenance audit | Independent track (CE1-adjacent); park as draft until after Tier 1 unless TC wants it moving. |
| 94 | **STOP: fixed-point reverse-engineering** | **The priority.** Execute per §3.2/§5 Tier 1; edit body per §5 rule 5; Part C → §3.4 replacement. |
| 95 | CE1–CE11 register | Keep open as the explanation registry Part A discriminates against (C3). |
| 96 | Self-prediction mechanism | **Close** (owner-recommended; superseded by #98). X3 carried into Tier 1. |
| 97 | F4 norm bug | Verified. Caveat lands T0-5; re-run is T2-1; then close. |
| 98 | F8 null challenge | Adjudicate per §4.2; items: (1) after adjudication, (2) into Tier 1, (3) held for T2-1, (4) respected. |
| 99 | PRIOR_WORK corrections | **T0-4. Close this week.** |

## 7. Open-PR dispositions

| PR | Disposition |
|--:|:--|
| **#87** | Merge (after #89). Clean, analysis-only, checks pass; its M1 "higher-precision run required" conclusion is then immediately annotated as superseded-analytic per T0-4 — merge first, annotate second, so the correction chain's history stays legible. |
| **#89** | Merge first (into #87's branch). The review it implements is the origin of standing rule 1. |
| **#100** | Close unless it rebases trivially post-merge; bot docstrings do not justify keeping a stacked branch alive. |
| **#92** | Keep as parked draft (see #92 row above). |

---

## 8. What this project is now

A study that found, and can now precisely state, one object: **GPT-2 Small's iterated, renormalised forward map sends 43% of language-derived starting states to a single weight-determined direction with per-pass gain ≈ 3.76, and sends one other family to an exact period-2 cycle executed by one attention head.** The corpus-fingerprint reading is dead; the coherence reading is under adjudication; what is left is harder and better: an exactly decomposable fixed point, a mechanism series that already took one such object apart (F13–F17), and a method (#94 Part A) for taking apart the other. The record's credibility asset — five self-refutations honestly documented — is worth more than any single finding; the standing rules in §5 exist to keep it.

**The instruction set, in one line each:** repair the record (Tier 0, this week); adjudicate #98 with committed code; execute #94 Part A with the §3.2 preconditions; replace Part C with M3; pass the pause gate; then run the noise re-run and the ν-sweep before anything new; keep everything else parked until those decide what it means.
