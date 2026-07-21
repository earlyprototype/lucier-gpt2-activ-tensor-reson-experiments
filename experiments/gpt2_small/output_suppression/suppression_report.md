# Suppression-Head Test for L11.H8: Inversion Confirmed and Load-Bearing, Copy-Suppression Signature Absent on Ordinary Text

*Follow-up to [hinge_eigenvalue.md](../output_hinge_eigen/hinge_eigenvalue.md) (issue #14, thread 1). Runner: [`11_suppression_test.py`](../11_suppression_test.py); raw numbers: [`suppression_results.json`](suppression_results.json). Model: GPT-2 Small, single Divine trajectory, states rebuilt from the committed iteration-1000 checkpoint. Sanity gates reproduced before any measurement: cos(A, B) = 0.684912, cos(A, f(f(A))) = 1.000000, full-tensor cycle residual 7.97e-04 (JSON: meta.gate).*

## The hypothesis

Experiment 08 located the sign inversion that sustains the Divine period-2 cycle in one attention head: L11.H8 carries 99.1 percent of the block-11 attention flip along the hinge (per-head d-component -1.981, cos -0.963). The suppression-head hypothesis reads that finding as an instance of a known behaviour class: L11.H8 functions as a suppression head, meaning its output is approximately a negative multiple of a component of its input, the class documented for GPT-2 Small's L10.H7 copy-suppression head, and the closed loop turns that one-shot negative correction into a sustained oscillation.

Three tests: (1) the OV circuit along the hinge, all 144 heads; (2) ablation of the head inside the loop, against controls; (3) the head's effect on attended-token logits on ordinary text, against the documented copy-suppression head.

## Verdict logic, stated up front

The hypothesis is SUPPORTED if (1) L11.H8's OV inverts d_sym distinctly among heads, (2) ablating it kills the cycle while a control ablation does not, and (3) it shows the negative-delta copy-suppression signature on ordinary text. (1) and (2) without (3) means the head inverts this direction and sustains the cycle but is not a general copy suppressor, and the training-function part of the hypothesis stays open. If ablation kills the cycle, the head is load-bearing either way; test 3 is what separates learned function from structural accident.

**Outcome: (1) supported, (2) supported, (3) refuted, and refuted with the opposite sign.** L11.H8's OV inverts d_sym more strongly than any other head (cos -0.9619, gain 63.68, rank 1 of 144 on both measures), ablating it collapses the cycle to a fixed point within about 10 iterations while the same-layer control ablation leaves a period-2 cycle running, and on ordinary text the head RAISES the logit of the token it attends to at 91.4 percent of positions (mean delta +5.97), while the L10.H7 positive control shows the documented suppression (87.1 percent negative, mean -3.62). L11.H8 sustains the bell by inverting the hinge, but it is not a copy suppressor; outside the loop it is a copy promoter. The training-function part of the hypothesis is not supported by what test 3 measured.

## Structural fact, verified before measurement

The Divine state is position-uniform, and this makes the head's inside-the-loop behaviour a pure OV circuit (JSON: structural):

- The layer-11 input during the phase-A pass has row spread 5.5e-07 (all 10 rows identical to numerical precision), and every recorded loop state below stays position-uniform (max row spread 8.4e-07 across all 500 recorded iterates).
- For a position-uniform input, every source position carries the same value vector, so the pattern-weighted average is that vector and the head's output is the OV transform of the ln1-normalised input regardless of the attention pattern. Verified empirically at phase A: head 8's hooked output row matches the direct computation ln1(x) @ W_V[11,8] @ W_O[11,8] with relative error 1.37e-07 (output row norm 1825.0). The value bias contributes nothing here because the loading convention folds b_V into b_O (b_V @ W_O norm = 0.0). The attention pattern at phase A is in fact exactly uniform, 0.100 on each of the 10 positions, and it does not matter.

This licenses reading test 1's static OV numbers as the head's actual in-loop transfer function.

## Test 1: The OV circuit inverts the hinge, and L11.H8 is the extreme of all 144 heads

For every head, y = d @ W_V[l,h] @ W_O[l,h], recording cos(y, d) and gain ||y|| / ||d|| (JSON: test1.directions).

Along the primary hinge d_sym:

| Head | cos(y, d_sym) | gain | rank by cos (1 = most negative) |
|:---|:---:|:---:|:---:|
| **L11.H8** | **-0.9619** | **63.68** | **1** |
| L10.H7 (copy-suppression head) | +0.1543 | 0.34 | 102 |
| L11.H0 (arbitrary) | +0.3741 | 0.90 | 137 |
| L5.H5 (arbitrary) | +0.0815 | 0.20 | 78 |
| L0.H0 (arbitrary) | +0.1078 | 0.25 | 86 |

Distribution over the 144 heads: median cos +0.059, extremes -0.962 and +0.523; 54 heads below 0; 5 heads below -0.5 (L11.H8 at -0.962 gain 63.68, L1.H11 at -0.913 gain 1.33, L4.H7 at -0.837 gain 0.48, L2.H10 at -0.787 gain 0.29, L1.H10 at -0.778 gain 0.36). L11.H8 is the most negative in cosine, and its gain is 48 times the next-most-negative head's. In d-component terms (cos times gain), L11.H8 writes -61.26 per unit of row-level d_sym; the runner-up is L1.H11 at -1.21. The inversion of the hinge is not merely the most extreme in the population, it is a different magnitude class.

Secondary frame, the committed d (0.616 aligned with d_sym, radial contamination as documented in experiment 08): L11.H8 has cos -0.3895, gain 7.42, rank 6 by cosine, but still the most negative d-component of all heads (-2.89 against -0.73 for the runner-up). Both frames agree on which head does the inversion.

Pole directions: the phase-A pole (+d_sym) and phase-B pole (-d_sym) give values identical to d_sym's, exactly, because the OV map is linear (cos(y(-d), -d) = cos(y(d), d)); recorded as a code-path check.

Controls, 5 random unit vectors (seed 20260721): L11.H8's cos values are +0.0281, -0.0008, +0.0175, -0.0250, -0.0080; the population mean |cos| per direction is 0.077 to 0.083, no head below -0.5 on any random direction, most negative single value -0.301. The inversion is specific to the hinge direction, not a generically negative OV.

### Empirical checks at the operating point (JSON: test1.empirical)

All finite-difference responses are in the experiment-08 convention: head-8 output change per unit of the full-tensor unit hinge, measured at the last row against the unit row hinge (row-level values are sqrt(10) = 3.162 times larger).

1. **Block-0 injection at Mn_sym (the exp-08 measurement, reproduced).** d-component -1.9814 at eps 1e-3 and -1.9860 at 1e-4, cos -0.9632. The recorded experiment-08 value is -1.9814; reproduction is exact at the matching epsilon.
2. **Layer-11 injection at Mn_sym (the literal x = M operating point).** d-component -1.1565, cos -0.9619, ln1 scale at the base row 16.76. Converting to row-level (-1.1565 x 3.1623 = -3.657) and multiplying by the ln1 scale gives -61.30, against the raw linear value -61.26 (0.07 percent apart), and the cosine equals the linear cosine to four decimals. At this base point d_sym is exactly orthogonal to M_sym (cos 0.0, JSON: meta.cos_esym_vs_Msym_row), so the ln1 Jacobian reduces to division by the scale, and the raw linear OV row is recovered exactly. Sign and magnitude of the linear computation are confirmed at the operating point.
3. **Layer-11 injection at the cascade resid_pre_11 (the input the head actually sees mid-loop, row norm 1042.6, ln1 scale 37.62).** d-component -0.3433, cos -0.9645.
4. **Chain to the exp-08 number.** The cascade delta arriving at layer 11 carries d_sym content 2.5863 out of Frobenius norm 2.9224. The pure-d response times that content is -0.3433 x 2.5863 = -0.8878, against the end-to-end -1.9814: the pure-d channel accounts for 45 percent, and the remaining -1.094 is the head's response to the off-hinge components the cascade generates (norm 1.361), which the head also delivers along -d_sym (end-to-end output cos -0.9632, pure-d output cos -0.9645, same output direction). The head's output direction is the same for both input components: what the cycle feeds it comes back along the negative hinge.

## Test 2: Ablating L11.H8 kills the cycle; the control ablation does not

Protocol: the ATR loop run from the phase-A checkpoint, with blocks.11.attn.hook_z zeroed at the target head on every forward pass. Hook verified on the first pass: the layer's attn_out changes by exactly the head's z @ W_O contribution (relative error 7.2e-08 for H8, 1.9e-06 for H0), and the block output moves (resid_post_11 change norm 5634.7 for H8, 230.1 for H0; the removed component norms are 5771.2 and 239.3, so H8 writes 24 times more than H0 at this state). Main run 300 iterations (100 was already unambiguous; 300 run outright), controls 100. Per-iterate lag-1 cosine, cosines to A, B, M_sym, M_committed, and readout argmax are in JSON: test2.*.records.

| Run | lag-1 cos, end | lag_scan last 24 (k = 1..8) | end state | argmax over run |
|:---|:---:|:---|:---|:---|
| No ablation | 0.6849 | 0.685, 1.000, 0.685, 1.000, 0.685, 1.000, 0.685, 1.000 | the exact A/B cycle: cos to A alternates 1.0000 / 0.6849 | ' Divine' at all 100 iterates (final p = 0.505) |
| Ablate L11.H0 (control) | 0.8938 | 0.893, 1.000, 0.894, 1.000, 0.894, 1.000, 0.894, 1.000 | a deformed period-2 cycle: phases at cos 0.9309 and 0.7229 to A, lag-2 cosine 1.000000 | ' Divine' at all 100 iterates (final p = 0.674) |
| **Ablate L11.H8** | **1.000000** | **1.000 at every lag 1..8** | **a fixed point away from the cycle: cos to A +0.1419, to B -0.6087, to M_sym -0.2543, to M_committed -0.0275** | ' Divine' (iters 1-2), '\n' (3), ' the' (4-300) |

The period-2 alternation stops immediately under the H8 ablation: lag-1 cosine rises from 0.8970 at iteration 1 to above 0.9999 by iteration 9 and above 0.999999 by iteration 14, and every lag 1 through 8 reads 1.000000 over the last 24 iterates. The state settles to a fixed point of the ablated map that is not near A, not near B, and not near either pivot (cosines above); its raw norm stabilises at 3574 (unablated phases: 5098 and 4838) and it stays position-uniform. The readout changes from ' Divine' at probability 0.5 to a flat generic distribution: final top-5 is ' the' 0.0235, ',' 0.0153, ' and' 0.0111, '.' 0.0107, ' a' 0.0105.

The control separates head identity from generic damage: removing L11.H0 (which also writes into the stream, norm 239) leaves a period-2 cycle running with the same ' Divine' argmax, with the two phases closer together (cos 0.8935 between consecutive iterates instead of 0.6849). The cycle needs L11.H8 specifically.

## Test 3: On ordinary text, L11.H8 raises the attended token's logit; the suppression signature belongs to L10.H7

Protocol: 12 natural sentences (the five 04_readout_confidence prompts plus seven new ones, 9 to 14 tokens each), each run once with no loop. For every position t >= 2: find the head's strongest non-BOS source s, take the head's per-position output through W_O, and compute delta = output_t @ W_U[:, token at s]. Copy suppression predicts predominantly negative delta. 116 positions per head (JSON: test3.per_head, per-position rows in test3.per_position_rows).

| Head | frac delta < 0 | mean delta | mean delta per unit output | mean attn to BOS | restricted to top-source attn > 0.2 |
|:---|:---:|:---:|:---:|:---:|:---|
| **L11.H8** | **0.086** | **+5.97** | +0.0263 | 0.0001 | n = 115: frac neg 0.087, mean +5.99 |
| L10.H7 (positive control) | 0.871 | -3.62 | -0.293 | 0.808 | n = 12: frac neg 1.000, mean -15.85, per unit output -0.928 |
| L11.H0 (arbitrary) | 0.638 | -0.52 | -0.0051 | 0.330 | n = 51: frac neg 0.608, mean -0.58 |
| L5.H5 (arbitrary) | 0.397 | +0.31 | +0.054 | 0.960 | n = 1 |

The protocol works: L10.H7 spends most of its attention on BOS, and at every position where it commits more than 0.2 attention to a real source it lowers that token's logit, mean -15.85, per-unit-output projection -0.928, which is the documented copy-suppression behaviour. The arbitrary heads sit near zero or mixed. L11.H8 is the opposite of the hypothesis: it is highly active (attention to BOS 0.0001, mean attention to its top source 0.40, top source is the position itself at only 19 percent of positions), and it RAISES the attended token's logit at 106 of 116 positions, mean +5.97 (median +6.50, range -9.30 to +12.92). The 10 negative positions are mostly early positions attending to a sentence-initial capitalised token ('He', 'Rain', 'Cal', 'Ast') plus the nonsense token ' morp'. Restricting to confident positions changes nothing (115 of 116 qualify). On ordinary text L11.H8 behaves as a copy-promoting head, not a copy suppressor.

## What follows

- **The mechanism is confirmed and sharpened.** The hinge inversion that sustains the bell is the static OV geometry of one head: d_sym is the single direction, out of the whole population, that L11.H8's OV both inverts (cos -0.9619) and amplifies (gain 63.7), the empirical operating-point response matches the linear computation exactly once the ln1 scale is accounted for, and the closed loop needs exactly this head: ablate it and the bell is replaced by a fixed point with a generic readout within about 10 iterations; ablate a neighbour and the bell persists.
- **The behaviour-class part of the hypothesis fails.** L11.H8 does not show suppression behaviour on ordinary text; it shows the reverse, while the measurement demonstrably detects suppression where it is documented (L10.H7). The one-shot negative correction that the loop recycles is therefore not this head's text-time function along token directions.
- **On learned function versus structural accident.** The ablation result makes the head load-bearing for the cycle under either reading, so it cannot separate them; test 3 was the separator, and it came out against learned general suppression. On the evidence here, the Divine oscillation exploits a strongly negative direction that happens to exist in this head's OV spectrum, a direction whose sign-inverting treatment is not exercised as suppression in ordinary next-token service. The accident reading is strengthened; what remains open is whether d_sym relates to some non-token content the head suppresses in contexts not sampled here.

## Limits

- Test 3's delta omits the final LayerNorm scaling, a positive per-position scalar that cannot change signs, and reads W_U directly (W_O writes are centered by the loading convention). It measures copy suppression in the token-unembedding sense only; suppression of non-token content would not register.
- The top non-BOS source may be the query position itself (19 percent of L11.H8's positions, 48 percent of L10.H7's); no exclusion was applied, matching the stated protocol.
- One trajectory, one loop prompt, one model. The OV computation ignores QK: which content the head attends to inside the loop is moot (the pattern is uniform and irrelevant under position uniformity), but on text the source selection is QK's and was taken as observed.
- The committed-d numbers inherit the frame mix documented in experiment 08; d_sym is the physical hinge and all headline claims use it.
- 12 sentences, 116 positions is a small text sample; the L11.H8 result (frac negative 0.086 against L10.H7's 0.871) is far from any decision boundary, so the sample suffices for the sign of the verdict but not for fine effect sizes.
