# Glitch Alignment: the Divine Hinge and the Anomalous-Token Cluster

*Follow-up to [bell_anatomy.md](../output_divine_motion/bell_anatomy.md) (issue #14, thread 4). Runner: [`07_glitch_alignment.py`](../07_glitch_alignment.py); raw numbers: [`glitch_alignment.json`](glitch_alignment.json). Same single Divine trajectory, states recovered from the saved iteration-1000 checkpoint.*

## Question asked

The bell's rank-1 hinge d had phase-B riders from the published GPT-2 anomalous-token family (the SolidGoldMagikarp family, Rumbelow and Watkins 2023). Is d ALIGNED with that cluster in embedding space, or merely near it?

## Setup

States replicated exactly as in 06, and the sanity gate passed before any measurement: cos(A, B) = 0.684912 (recorded: 0.6849117), cos(A, f(f(A))) = 1.000000. Hinge d = (A - B)/2 at the last position; +d is the phase-A pole, -d is the phase-B pole. d is a residual-stream direction and W_E rows write into the same 768-dimensional residual space, so cos(d, W_E row) is well-defined. All geometry is in the TransformerLens processed basis that 04/06 used; cluster membership is identical in the raw HF basis (Jaccard 1.0 on all four geometric sets).

The cluster, identified three ways:

- **Geometric core** (closest 0.1 percent of the vocab to the mean-embedding centroid, k = 50): the untrained-token signature. In GPT-2 this set is the raw control bytes `\x00` to `\x1f`, the undecodable byte tokens (`�`), and named family members (` externalToEVA`, ` TheNitrome`, `quickShip`, `embedreportprint`, `reportprint`, `rawdownload`, ` サーティ`).
- **Geometric shell** (closest 0.5 percent, k = 251).
- **Low-norm variant** (bottom 0.1 / 0.5 percent by W_E row norm, k = 50 / 251): in GPT-2 this criterion does NOT find glitch tokens. The lowest-norm rows are the highest-frequency function words (` at`, ` in`, ` on`, ` for`). Overlap with the geometric sets: 0 of 50 at the 0.1 percent cutoff, 33 of 251 at 0.5 percent.
- **Curated family**: 54 published family strings, each probed with and without leading space, exact single-token matches only: 52 distinct tokens matched (only ` SmartyHeaderCode` is absent in either form; the vocab holds e.g. `quickShipAvailable`, `StreamerBot`, `ertodd` spaceless). Plus the 4 ideographic oddities from the bell's own readout (`【`, ` 「`, `……`, and the horizontal bar `\u2015`; 6 tokens once spacing variants are included): 58 in all. Of the curated family, 5 tokens sit inside the geometric core and 10 inside the shell; none are in the low-norm sets.

## Results

**The B pole points into the anomalous core.** With u = normalise(cluster centroid - global mean embedding), negative cos(d, u) means the -d (phase-B) pole points toward the cluster:

| cluster | k | cos(d, u) | cos(M, u) | p vs 1000 random | p vs 1000 norm-matched |
|:---|---:|---:|---:|:---:|:---:|
| geometric core 0.1% | 50 | -0.596 | -0.498 | < 0.001 | < 0.001 |
| geometric shell 0.5% | 251 | -0.073 | -0.051 | 0.37 | 1.0 (signed: 0.001) |
| low-norm 0.1% | 50 | +0.594 | +0.515 | < 0.001 | 0.38 |
| low-norm 0.5% | 251 | +0.609 | +0.529 | < 0.001 | 0.14 |
| curated family | 52 | -0.456 | -0.385 | < 0.001 | < 0.001 |
| curated + oddities | 58 | -0.422 | -0.350 | < 0.001 | < 0.001 |

Scale: random same-size token sets give mean |cos| = 0.065 with a maximum of 0.30 over 1000 draws; the core's 0.596 is twice the null maximum (about 7.4 null standard deviations). The norm-matched null is sharper still: sets matching the core's norm profile point the OPPOSITE way (mean cos +0.48, toward the A pole); the anomalous core is the exception at -0.596. The alignment is about which tokens these are, not their norms. For the low-norm sets the norm-matched null is nearly tautological and indeed absorbs the effect (p 0.14 to 0.38). The 251-token shell surrounds the centroid almost isotropically, so its centroid direction washes out (cos -0.073), yet relative to its norm profile it still leans anomalously B-ward (signed p = 0.001), and see the pole scan below.

**The B-pole ray is saturated with cluster members.** Top-50 vocab tokens by cos(row, -d): 45 of 50 are in the geometric core, 50 of 50 are inside the 0.5 percent shell (a 200x enrichment), and 6 of 50 are named family members (`oreAndOnline`, ` RandomRedditor`, ` externalToEVA`, `embedreportprint`, `reportprint`, ` TheNitrome`; 12 percent against a 0.12 percent base rate, about 104x). Top 15 toward -d (cosines 0.565 to 0.568): `\x07`, `\x10`, `\x0b`, `oreAndOnline`, ` サーティ`, `\x11`, `\x1f`, `\x04`, `\x02`, ` RandomRedditor`, `�`, `\x14`, `�`, `\x01`, ` externalToEVA`. The median core member sits at the 99.94th percentile of the whole vocab toward -d, the median curated member at the 96.4th, and all 58 curated tokens, without exception, lean toward the B pole.

**The A pole is the opposite corner: the most-trained tokens.** The top-50 toward +d contains no cluster member of any definition; 42 of 50 are in the bottom 0.5 percent by norm. Top 15: ` the`, `,`, ` in`, ` and`, ` a`, `.`, ` to`, `\n`, `-`, ` (`, ` of`, ` "`, ` on`, ` for`, ` that`. The see-saw runs between the high-frequency function-word corner (phase A) and the untrained corner (phase B): cos(u_core, u_lownorm) = -0.68, and the two independent definitions of the glitch corner agree on where the B wall is, cos(u_core, u_curated) = +0.67.

**Offset, not spread.** cos(d, PC1 of the core's centered embeddings) = +0.009 (curated: +0.047): the hinge aligns with where the cluster sits relative to the global mean, not with the cluster's internal principal axis. (For the shell, |cos(d, PC1)| = 0.43, because at that radius the dominant internal variance direction is the core's own offset.)

**The hinge is global, and the pivot leans with A.** Recomputed pos_alignment = 1.0000; cos(d_pos, d_last) = 1.0000 and cos(d_pos, u_core) = -0.5958 identically at every one of the 10 positions. The hinge is one global direction, not a last-position artifact: the pos_alignment = 1.0 already recorded in bell_anatomy.json means exactly this, confirmed here. The pivot M tilts toward the function-word corner (cos(M, u_core) = -0.498, cos(M, u_lownorm 0.5%) = +0.529). Note cos(d, M) = 0.909 (|A| = 1612 against |B| = 464 leaves both dominated by A's direction), so the A pole nearly coincides with the bell's standing direction; the distinctive, informative pole is -d, phase B's excursion.

## Verdict

**Aligned (structural).** cos(-d, u) = +0.60 with the geometric core and +0.46 with the curated family, beyond every one of 1000 random sets and 1000 norm-matched sets (which, for the core, point the opposite way), with the -d top-50 90 to 100 percent inside the cluster, the named family enriched about 100-fold on that ray, and the same alignment at all 10 positions. The magnitude is a strong tilt, not an identity: 0.46 to 0.60, not 0.9, and the hinge also carries a large component along the pivot. But the direction of phase B's excursion is unambiguous: each pass, the normalised map throws the state into the degenerate, untrained corner of embedding space and back. The flutter echo has the anomalous-token cluster as one wall and the function-word corner as the other: the cycle oscillates between the model's least-trained and most-trained token directions.

## Caveats

Same single trajectory, prompt, and model as 06. The geometric core is mostly control-byte and undecodable-byte tokens; the named SolidGoldMagikarp family is 10 percent of it (5 of 50), so "the cluster" here means the whole untrained core, with the curated family measured separately and agreeing (cos +0.46, p < 0.001 under both nulls). The low-norm criterion is not an anomaly signature in GPT-2 (it selects frequent function words, with zero overlap with the core at the 0.1 percent cutoff), so its table rows read as the A-pole complement, not as a second glitch test. PC1 signs are arbitrary; only magnitudes matter there. All cosines are computed in the TransformerLens processed basis; membership of every cluster is unchanged in the raw basis (Jaccard 1.0), but exact cosine values would shift slightly there.
