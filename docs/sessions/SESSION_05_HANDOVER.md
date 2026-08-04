# SESSION 05 HANDOVER: run 18, the ν-sweep, and what the pin turned out to be doing

*Date: 2026-08-02. Continuation context for the next session, human or AI. Read `SESSION_03_HANDOVER.md` first for the deep state, the environment notes and the working agreements (no em dashes anywhere; plain language with the technical terms defined on first use and no decorative metaphors; CodeRabbit for independent review). `SESSION_04_HANDOVER.md` covers the mechanism series and the pause, which was lifted on 2026-07-31. This document covers what changed since PR #103.*

## What landed

Three PRs merged this session, in order.

**PR #107** delivered two workstreams held over from the previous session: the H4 record replacement (run 16) and the matched-ν noise baseline (run 17). Run 17 is the important one: at matched injection scale, pure noise lands in the language arm's own basins, all five reappearing and 97 of 125 trials landing in them. This inverted F4 and killed the "noise relocates the basins" reading for good. It also opened the question that consumed the rest of the session.

**PR #110** was a one-file docstring restoration. No behaviour change.

**PR #114** is run 18, the ν-sweep, registered in issues #113 and #116 before execution. 1,425 trials across twenty-one injection scales. It is the substance of this session and is described below.

Draft PR #92 (the WebText provenance audit) was closed unmerged by operator decision; issue #102 carries that work, since its forensic design has to be rebuilt.

## Run 18: what the sweep found

The engine pins every iterate's Frobenius norm to a fixed target, ν. That pin was never a considered choice: the historical protocol used whatever the first forward pass happened to produce, which is about 70 times the size layer 0 naturally receives. The sweep asked what the pin was doing.

**The five-basin landscape lives in a band**, roughly 50x to 300x the natural entry scale. Both edges are bracketed at all 125 prompts: the lower inside (48x, 56x), where the share landing in the five goes 39/125 to 125/125; the upper inside (256x, 384x), 84/125 to 47/125. The historical pin sits comfortably inside at 71x, so every published result was taken in a valid regime, but a regime, not the only one.

**Below the band the landscape is stratified, not empty.** `arbit` at 2-4x, the horizontal bar at 6-8x and again at 16-24x, `vertex` at 12x (invisible at both its neighbours), fragmentation into 14 labels at 32x, and at 40x a periodic shelf where only 33/125 lock and 92 pass solely at lag 2. The lowest strata point measurably into the anomalous-token cluster on F13's instruments with matched nulls (at 4x: top-50 alignment fraction 0.90 against the geometric cluster where chance is 0.005, and 0.18 against the curated SolidGoldMagikarp family where chance is 0.001). The band itself anti-aligns with that cluster and sits in the low-norm function-word region instead. Note also that the old mis-calibrated noise arm's norm, 397, sits inside the horizontal-bar stratum, and its dominant basin was that bar: consistent with it having been a real basin of its stratum rather than pure artefact.

**The two edges are governed by different quantities.** This came from a TC objection in session: the single-pass gain is a ratio of magnitudes and says nothing about direction, while everything the project reports at the end is a direction. The engine had been archiving the directional measure all along, as each trial's per-iteration `cos_sim_mean_lag1`. Converted to degrees turned per pass and re-analysed archive-only (`06_angular_profile.py`, no re-runs), the per-prompt transition scatter inverts by edge: the lower edge clusters in degrees (CV 0.082 against 0.123 in magnitude), the upper in magnitude (0.091 against 0.510 in degrees). In magnitude coordinates the sweep is one smooth decline from 54x to 1.01x and both edges look like the same event. In degrees it is a hill: 45 degrees a pass at 8x, 66 at 24x, 73 at 48x, peaking at 77 inside the band, then collapsing to 9 at 256x. **Below the band the loop is not failing to move; it is moving as much as it ever does.** The mechanism first proposed for the lower edge (insufficient change per pass) is refuted by its own data.

**The split inside the band is content-borne.** The shared-pin control pinned all 125 prompts to one identical number (1393.70, the median library exit norm). The multi-basin split did not collapse; it widened to six labels with 99 percent in the five. So what you feed the loop still selects which basin it finds, which is the first solid evidence of that since the scale question opened.

FINDINGS caveat 19 carries all of this. Records and regenerating scripts: `experiments/nu_sweep/`. An interactive walk through the profile was built for TC and is not part of the repo record.

## Corrections made in flight, worth carrying forward

- **The Q2 scatter test as registered was malformed.** It required exactly one in-five crossing per prompt; the band has two edges, so most prompts cross twice, and the first run silently discarded 23 of 25 prompts and reported a number computed over the surviving two. Corrected to measure each edge separately, with the grid bias on the multiplier coordinate stated in the report.
- **The knowledge graph drifted inside prose.** The run-18 node was written before Stage C finished and kept the superseded edges and trial count. `check_record_drift.py` could not catch it because it compares structured fields, not prose inside graph descriptions. Worth knowing that this class of drift is unguarded.
- **Quarter-width sampling near a boundary is not load-bearing.** Of three soft boundaries tested at both 25 and 125 prompts, two moved materially (256x from 44 to 67 percent; 48x from 56 to 31 percent), once in each direction. Recorded in caveat 19. Do not accept a boundary claim from 25 prompts.

## Open, in the order I would take them

1. **Find out which parts of the model do the turning.** Every pass, the state's direction swings by some angle, and that angle is what governs the band's lower edge. But the model is not one thing doing the swinging: it is 144 attention heads and 12 feed-forward blocks, each writing its own vector into the running state every pass, and the measured angle is the sum of all of them. Our measure collapses that into a single arrow, so it cannot distinguish two heads doing nearly all the work from 156 components each nudging slightly. Splitting the arrow back into per-component shares would say which. It is worth doing because the project has precedent for a concentrated answer: experiment 11 found one head, layer 11 head 8, doing about 99 percent of the work that sustains the `Divine` cycle. If the turn is similarly concentrated, that is the mechanism behind the lower edge. This is the unanswered half of TC's objection; the machinery exists from experiments 08 and 11. Not yet registered.
2. **#17, basin geometry (escape thresholds).** Annotated this session with corrected context: its original motivation (comparing language basin depth against noise basin depth) is refuted by run 17, since there are no separate noise basins. Replacement motivation supplied, plus one new mandatory design parameter, namely that every threshold must state the injection scale it was measured at. Ready to run as annotated.
3. **#116 holds the parked cross-model prediction** for Pythia 160M, recorded before any such run: the architectural features should transfer, the glitch strata should not. Executes only on TC reopening the cross-model track (standing rule 8).
4. The rest of the Tier 2 queue in `ALIGNMENT_REVIEW.md` section 5, with #94 Part A next.

## Environment notes for a fresh container

Everything in SESSION_03's environment section still holds. Two additions from this session:

- **Launch long compute with the harness's background-task mechanism, not a detached shell command.** A batch of workers started with `nohup ... &` inside a foreground call died when their shell exited, losing about half an hour. Nothing was lost from the record because every trial checkpoints individually, which is the design that saved it; keep that pattern for anything long.
- **CodeRabbit rate-limits hard under a busy branch** and auto-pauses reviews when commits land faster than it can review. Trigger a single final review with a `@coderabbitai review` comment once the branch is actually finished, which is also when the review is most useful.

The venv and downloaded weights remain ephemeral. Rebuild per SESSION_03; the legacy S3 mirror for GPT-2 weights was alive this session.

## The state of the conversation

TC engaged closely and technically throughout, and two of his interventions changed results rather than presentation: the magnitude-versus-direction objection that produced the two-edge finding, and the observation that the strata sitting at powers of two was an artefact of the sampling grid rather than a fact about the model, which was correct. He asked for the work to be explained in plain language more than once; the explanations that worked were the ones that rebuilt the analogy (a room, a volume knob) and then said explicitly where it stopped holding. He also asked, directly, whether any of this is useful information. That question deserves a real answer from whoever picks this up, and the honest one given was: the methodological warning transfers, the weight-geometry results are a small real contribution to a small real niche, and the record-keeping apparatus is the most transferable thing here.
