# Which parts of the model do the turning?

Registered before execution in issue #119. Raw data:
`turn_attribution.pt` (per-trial checkpoints in `checkpoints/`).
The residual stream is additive, so each pass's motion splits
exactly into the writes of 144 attention heads, 12 feed-forward
blocks and 12 attention output biases.

**Both coordinates of the motion are reported throughout.** A pass
moves the state in two ways: it turns the direction and it changes
the size. These are two coordinates of one motion, not competing
explanations. Run 18 was read as showing the five-basin band's two
edges dominated by one coordinate each, the lower by the turn and
the upper by the size, and that reading motivated this experiment;
it was withdrawn on 2026-08-04 as methodologically unsound
(FINDINGS caveat 19), and which quantity governs either edge is
now open. Both shares are reported here regardless, and neither
column below bears on that open question. Every component gets
both shares: the share of the direction change (its write
perpendicular to the state,
projected onto the realised turn) and the share of the size change
(its write parallel to the state). Both are signed, and both sum
to 1 by construction, which the script asserts every pass.

Which coordinate the apparatus controls matters for reading the
size column. The loop rescales every iterate to the pin, so within
a run the size is held by hand and only the direction is free; the
size coordinate re-enters as the pin itself, which is the sweep's
axis. The size shares below therefore describe what the model
would do to the size if the rescale were not undoing it each pass.

**Direct contributions only.** A component's write also changes
what later components read, so a small direct share does not mean
a component is unimportant. No ablations were run, so nothing
here is a causal claim.

## Concentration in both coordinates

| level | phase | passes | motion | size per pass | top 1 | top 5 | top 20 | cancellation | top component |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| m008 | early | 200 | direction | 30.8&deg; | 100.3% | 139.2% | 159.4% | 5.35 | `L11H8` |
| m008 | early | 200 | size | x18.23 | 29.7% | 80.6% | 98.8% | 4.66 | `L11H8` |
| m008 | late | 190 | direction | 0.0&deg; | undefined | undefined | undefined | undefined | net turn below 1 deg |
| m008 | late | 190 | size | x24.39 | 128.7% | 156.5% | 166.1% | 2.45 | `L11H8` |
| m040 | early | 200 | direction | 59.1&deg; | 75.1% | 101.7% | 112.4% | 2.22 | `L11H8` |
| m040 | early | 200 | size | x5.34 | 145.9% | 440.4% | 571.1% | 17.28 | `L11MLP` |
| m040 | late | 160 | direction | 42.2&deg; | 72.1% | 98.5% | 111.0% | 1.94 | `L11H8` (38 passes dropped) |
| m040 | late | 198 | size | x5.78 | 43.4% | 77.5% | 109.2% | 2.12 | `L11MLP` |
| m056 | early | 200 | direction | 60.6&deg; | 71.9% | 96.7% | 107.1% | 2.01 | `L11H8` |
| m056 | early | 200 | size | x4.06 | 32.1% | 86.0% | 112.6% | 4.42 | `L11H8` |
| m056 | late | 190 | direction | 0.0&deg; | undefined | undefined | undefined | undefined | net turn below 1 deg |
| m056 | late | 190 | size | x4.29 | 39.6% | 69.9% | 92.4% | 1.27 | `L11MLP` |
| historical | early | 200 | direction | 52.8&deg; | 68.2% | 93.0% | 103.4% | 2.22 | `L11H8` |
| historical | early | 200 | size | x3.41 | 197.3% | 582.1% | 844.1% | 21.93 | `L11attn_bias` |
| historical | late | 20 | direction | 4.5&deg; | 87.8% | 114.3% | 124.1% | 2.48 | `L11H8` (171 passes dropped) |
| historical | late | 191 | size | x3.60 | 42.8% | 68.4% | 91.7% | 1.35 | `L11MLP` |
| m384 | early | 155 | direction | 3.7&deg; | 48.9% | 118.4% | 158.5% | 5.26 | `L11H8` (45 passes dropped) |
| m384 | early | 200 | size | x1.32 | 46.7% | 93.3% | 133.1% | 2.59 | `L2MLP` |
| m384 | late | 190 | direction | 0.1&deg; | undefined | undefined | undefined | undefined | net turn below 1 deg |
| m384 | late | 190 | size | x1.35 | 51.8% | 93.4% | 127.2% | 2.34 | `L2MLP` |

The cancellation column is the mean of the absolute shares summed
over all components. It is 1.0 when every component pushes the
same way and rises as components fight each other, so a large
value means the net turn is a small residue of much larger
opposing contributions.

## Ranked components per level (early passes), both coordinates

- **m008**, direction: `L11H8` +100.3%, `L0MLP` +17.8%, `L11MLP` +14.0%, `L11attn_bias` +3.6%, `L11H0` +3.4%, `L10MLP` +2.9%
- **m008**, size: `L11H8` +29.7%, `L11MLP` +28.0%, `L0MLP` +8.1%, `L11H0` +7.9%, `L2MLP` +7.0%, `L11attn_bias` +3.5%
- **m040**, direction: `L11H8` +75.1%, `L11MLP` +14.4%, `L0MLP` +7.7%, `L11H0` +2.8%, `L10MLP` +1.6%, `L11H9` +1.5%
- **m040**, size: `L11MLP` +145.9%, `L2MLP` +142.9%, `L0MLP` +93.7%, `L11H0` +37.1%, `L11H9` +20.8%, `L11H1` +13.6%
- **m056**, direction: `L11H8` +71.9%, `L11MLP` +12.1%, `L0MLP` +7.9%, `L11H0` +2.7%, `L10MLP` +2.2%, `L11H9` +1.3%
- **m056**, size: `L11H8` +32.1%, `L11MLP` +20.4%, `L11attn_bias` +14.5%, `L0MLP` +11.6%, `L11H0` +7.3%, `L11H9` +3.8%
- **historical**, direction: `L11H8` +68.2%, `L11MLP` +12.0%, `L0MLP` +7.3%, `L11H0` +3.0%, `L10MLP` +2.5%, `L11H11` +1.3%
- **historical**, size: `L11attn_bias` +197.3%, `L11MLP` +162.9%, `L0MLP` +132.2%, `L11H9` +45.8%, `L11H2` +43.8%, `L11H0` +39.8%
- **m384**, direction: `L11H8` +48.9%, `L11attn_bias` +41.5%, `L0MLP` +11.0%, `L10H10` +8.7%, `L11H3` +8.2%, `L11H2` +6.6%
- **m384**, size: `L2MLP` +46.7%, `L11H0` +14.3%, `L11MLP` +12.0%, `L11H8` +11.4%, `L0MLP` +9.0%, `L11H4` +6.2%

Whether the same components lead in both coordinates is itself informative: a component that grows the state without turning it, or turns it without growing it, is doing a different job from one that does both.

## The pre-stated expectations (issue #119)

1. **Concentration**: the top 5 components account for 96.7% at m056, 93.0% at historical on early passes. The pre-stated threshold was more than 50 percent inside the band: MET.
2. **The ranking across the lower edge**: 5 of the top 5 components are shared between 40x (outside the band) and 56x (inside). Outside: `L11H8`, `L11MLP`, `L0MLP`, `L11H0`, `L10MLP`. Inside: `L11H8`, `L11MLP`, `L0MLP`, `L11H0`, `L10MLP`. The registration expected the composition to differ across this edge; a fully shared ranking would mean the turn's composition is not what changes there.
3. **Layer 11 head 8**: no prediction was registered for it. Its rank is reported like any other component's, in the tables above.

## Reading

Every number above is regenerated by re-running this script;
nothing is hand-computed. This loop is verified against the
canonical engine by the committed contract check before any trial
runs. The turn is measured on the mean vector across positions,
matching run 18's instrument, so this says nothing about
per-position structure. Five levels and ten prompts is a probe:
run 18 found that quarter-width sampling near a boundary moved
materially in two of three cases (FINDINGS caveat 19), so no
boundary claim may rest on these counts without widening.
