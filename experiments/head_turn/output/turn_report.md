# Turn attribution: results

Registered before execution in issue #119. Raw data:
`turn_results.pt` (per-pass per-component signed shares, float64);
fresh gated runs for the stage-C levels in `plain_gated_results.pt`.
A component's share fraction is its signed share divided by that
pass's total turn, so fractions sum to 1 per pass; the closure
assert held on every instrumented pass.

How to read the two windows. In passes 1-20 the state is moving and
the net turn is of the same order as the individual writes, so a
share fraction reads as ordinary dominance. In the 20 passes around
lock-in the net turn is a small residual of large writes that nearly
cancel, so fractions there are amplified by the shrinking
denominator: the cancellation ratio line (the sum of every
component's absolute share, over the net turn) states that
amplification directly, and a component ranked last by signed share
is the largest CANCELLING contributor, not the least active one.

## m008, passes 1-20

- Passes used 200, skipped for near-zero turn 0.
- Mean cancellation ratio: 5.2 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 171.4%, top 5 246.6%, top 20 278.5%.
- Largest components by mean absolute share: `L11.H8` 180.9%, `L2.MLP` 104.3%, `attn-bias` 36.0%.
- L11.H8's rank by mean SIGNED share fraction: 1 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L11.H8` | 100.3% |
| 2 | `L0.MLP` | 17.8% |
| 3 | `L11.MLP` | 14.0% |
| 4 | `attn-bias` | 4.0% |
| 5 | `L11.H0` | 3.4% |
| 6 | `L10.MLP` | 2.9% |
| 7 | `L11.H3` | 2.7% |
| 8 | `L11.H11` | 2.4% |
| 9 | `L11.H2` | 1.6% |
| 10 | `L11.H9` | 1.4% |

## m008, the 20 passes around lock-in

- Passes used 153, skipped for near-zero turn 47.
- Mean cancellation ratio: 2528.7 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 33572.2%, top 5 64778.6%, top 20 90559.5%.
- Largest components by mean absolute share: `L11.H8` 88348.4%, `L2.MLP` 33440.9%, `L11.MLP` 17613.7%.
- L11.H8's rank by mean SIGNED share fraction: 157 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L2.MLP` | 33440.9% |
| 2 | `L11.MLP` | 17613.7% |
| 3 | `L0.MLP` | 4188.5% |
| 4 | `L11.H2` | 2986.3% |
| 5 | `L7.H6` | 2016.7% |
| 6 | `L11.H9` | 1922.0% |
| 7 | `L10.H10` | 1792.4% |
| 8 | `L11.H0` | 1743.5% |
| 9 | `L4.MLP` | 1731.0% |
| 10 | `L8.H2` | 1666.6% |

## m040, passes 1-20

- Passes used 200, skipped for near-zero turn 0.
- Mean cancellation ratio: 2.2 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 91.6%, top 5 124.4%, top 20 140.0%.
- Largest components by mean absolute share: `L11.H8` 76.0%, `L2.MLP` 27.5%, `L11.MLP` 21.3%.
- L11.H8's rank by mean SIGNED share fraction: 1 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L11.H8` | 75.1% |
| 2 | `L11.MLP` | 14.4% |
| 3 | `L0.MLP` | 7.7% |
| 4 | `L11.H0` | 2.8% |
| 5 | `L10.MLP` | 1.6% |
| 6 | `L11.H9` | 1.5% |
| 7 | `L11.H2` | 1.1% |
| 8 | `L11.H11` | 1.1% |
| 9 | `L11.H1` | 0.9% |
| 10 | `L11.H5` | 0.9% |

## m040, the 20 passes around lock-in

- Passes used 40, skipped for near-zero turn 0.
- Mean cancellation ratio: 271.7 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 7896.5%, top 5 11945.4%, top 20 13212.8%.
- Largest components by mean absolute share: `L11.H8` 7896.5%, `L11.MLP` 4070.7%, `attn-bias` 3404.3%.
- L11.H8's rank by mean SIGNED share fraction: 1 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L11.H8` | 7896.5% |
| 2 | `attn-bias` | 3404.3% |
| 3 | `L11.H3` | 291.9% |
| 4 | `L4.H7` | 187.3% |
| 5 | `L11.H4` | 165.3% |
| 6 | `L6.H6` | 125.0% |
| 7 | `L6.MLP` | 116.1% |
| 8 | `L10.H10` | 114.9% |
| 9 | `L1.H8` | 111.9% |
| 10 | `L9.MLP` | 111.2% |

## m056, passes 1-20

- Passes used 200, skipped for near-zero turn 0.
- Mean cancellation ratio: 1.9 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 79.2%, top 5 114.1%, top 20 129.8%.
- Largest components by mean absolute share: `L11.H8` 71.9%, `attn-bias` 19.4%, `L2.MLP` 19.0%.
- L11.H8's rank by mean SIGNED share fraction: 1 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L11.H8` | 71.9% |
| 2 | `L11.MLP` | 12.1% |
| 3 | `L0.MLP` | 7.9% |
| 4 | `L11.H0` | 2.7% |
| 5 | `L10.MLP` | 2.2% |
| 6 | `L11.H9` | 1.3% |
| 7 | `L11.H11` | 1.1% |
| 8 | `L11.H2` | 1.0% |
| 9 | `L11.H1` | 1.0% |
| 10 | `L11.H5` | 0.9% |

## m056, the 20 passes around lock-in

- Passes used 176, skipped for near-zero turn 24.
- Mean cancellation ratio: 1210.9 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 26915.8%, top 5 43769.6%, top 20 54656.0%.
- Largest components by mean absolute share: `L11.MLP` 26915.8%, `L11.H8` 21062.9%, `attn-bias` 10923.4%.
- L11.H8's rank by mean SIGNED share fraction: 157 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L11.MLP` | 26915.8% |
| 2 | `L2.MLP` | 7925.9% |
| 3 | `L11.H0` | 4868.3% |
| 4 | `L8.H2` | 2002.5% |
| 5 | `L1.MLP` | 1622.4% |
| 6 | `L7.H6` | 1267.8% |
| 7 | `L11.H5` | 1235.1% |
| 8 | `L3.MLP` | 950.1% |
| 9 | `L6.H2` | 844.6% |
| 10 | `L9.H7` | 817.1% |

## historical, passes 1-20

- Passes used 200, skipped for near-zero turn 0.
- Mean cancellation ratio: 2.1 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 77.0%, top 5 117.6%, top 20 137.2%.
- Largest components by mean absolute share: `L11.H8` 69.0%, `attn-bias` 26.9%, `L11.MLP` 20.8%.
- L11.H8's rank by mean SIGNED share fraction: 1 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L11.H8` | 68.2% |
| 2 | `L11.MLP` | 12.0% |
| 3 | `L0.MLP` | 7.3% |
| 4 | `L11.H0` | 3.0% |
| 5 | `L10.MLP` | 2.5% |
| 6 | `L11.H11` | 1.3% |
| 7 | `L11.H9` | 1.1% |
| 8 | `L11.H1` | 1.1% |
| 9 | `L11.H2` | 0.9% |
| 10 | `L11.H5` | 0.9% |

## historical, the 20 passes around lock-in

- Passes used 176, skipped for near-zero turn 4.
- Mean cancellation ratio: 1060.9 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 18943.4%, top 5 35201.8%, top 20 46392.1%.
- Largest components by mean absolute share: `L11.H8` 20648.7%, `L11.MLP` 18943.4%, `attn-bias` 12458.7%.
- L11.H8's rank by mean SIGNED share fraction: 157 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `L11.MLP` | 18943.4% |
| 2 | `L2.MLP` | 7604.8% |
| 3 | `L11.H0` | 4813.2% |
| 4 | `L8.H2` | 1720.1% |
| 5 | `L11.H5` | 1355.8% |
| 6 | `L0.MLP` | 1353.0% |
| 7 | `L7.H6` | 1328.4% |
| 8 | `L6.H2` | 1065.2% |
| 9 | `L1.H11` | 1023.8% |
| 10 | `L10.H3` | 857.6% |

## m384, passes 1-20

- Passes used 200, skipped for near-zero turn 0.
- Mean cancellation ratio: 6.4 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 109.7%, top 5 217.0%, top 20 299.5%.
- Largest components by mean absolute share: `L2.MLP` 138.3%, `attn-bias` 80.7%, `L11.H8` 56.5%.
- L11.H8's rank by mean SIGNED share fraction: 2 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `attn-bias` | 54.9% |
| 2 | `L11.H8` | 51.9% |
| 3 | `L0.MLP` | 15.7% |
| 4 | `L11.H3` | 11.7% |
| 5 | `L11.H9` | 10.3% |
| 6 | `L11.H2` | 9.5% |
| 7 | `L10.H10` | 7.4% |
| 8 | `L10.MLP` | 6.8% |
| 9 | `L10.H0` | 5.1% |
| 10 | `L9.H8` | 3.0% |

## m384, the 20 passes around lock-in

- Passes used 185, skipped for near-zero turn 15.
- Mean cancellation ratio: 84.6 (1.0 would mean no cancellation at all).
- Mean concentration of the total turn: top 1 component 1069.0%, top 5 2551.1%, top 20 3619.2%.
- Largest components by mean absolute share: `L2.MLP` 1680.6%, `attn-bias` 1057.6%, `L11.H8` 847.7%.
- L11.H8's rank by mean SIGNED share fraction: 2 of 157 (last place means largest cancelling contributor).

| rank | component | mean share of the turn |
|--:|:--|--:|
| 1 | `attn-bias` | 1057.6% |
| 2 | `L11.H8` | 847.5% |
| 3 | `L11.H3` | 244.7% |
| 4 | `L0.MLP` | 214.2% |
| 5 | `L11.H2` | 177.2% |
| 6 | `L10.H10` | 153.0% |
| 7 | `L9.H2` | 104.3% |
| 8 | `L11.H6` | 93.5% |
| 9 | `L10.H0` | 90.8% |
| 10 | `L8.H11` | 70.3% |

## Registered expectation checks

1. Concentration (top 5 above 50% inside the band, passes 1-20): m056 114.1% (holds); historical 117.6% (holds).
2. Ranking identity across the lower edge (top-10 overlap, m040 vs m056, passes 1-20): 10/10 shared.
3. No prediction was offered about L11.H8; its rank is reported per level above.

## Scope

Direct contributions only; no ablation, so no causal claim about
any component. The turn is measured on the mean vector across
positions, matching run 18's instrument. Five levels and ten
prompts is a probe, not a sweep. Interpretation lands in issue
#119 and the findings record, not here.
