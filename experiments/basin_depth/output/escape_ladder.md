# How far must a settled state be rotated before it does not come home?

Registered before execution in issue #17 (2026-08-04 registration
comment). Raw data: `escape_ladder.pt`. The loop rescales every
iterate, so a state lives on a sphere and a perturbation is a
rotation; thresholds are therefore in DEGREES, which is
dimensionless and comparable with run 18's finding that inside the
band the loop itself rotates the state about 77 degrees on an
ordinary pass. A threshold is the smallest ladder rung whose run
settles on a different token from the unperturbed attractor.

## The zero-perturbation control

This control was NOT in the registered protocol and was added after the first ladders ran. A rung counts as an escape when the re-entered run settles on a different token from the attractor, which is only meaningful if re-entering with NO perturbation returns that attractor. Without the control, a state whose re-entry fails to reproduce its own settled point is indistinguishable from a basin so shallow that one degree dislodges it.

Of 10 states with both a ladder and a control, 10 passed and 0 failed. Failed states are excluded from every number below and named here:

- none

### States the apparatus cannot resolve

The control also measures HOW EXACTLY the loop returns an unperturbed state, and that floor is not the same everywhere. Inside the band a state comes back to five or six decimal places. Near the upper edge it does not. Where the floor is worse than 0.999, a basin crossing cannot be told apart from the loop's own drift, so those states are excluded and named here rather than given a number that would only reflect the apparatus:

- `m056_solidarity_G08_period (returns only to 0.9017)`
- `m256_solidarity_A02_medical (returns only to 0.9923)`
- `m256_solidarity_A04_climate (returns only to 0.9898)`
- `m256_the_A03_neuro (returns only to 0.7619)`
- `m256_the_A05_evolution (returns only to 0.9734)`
- `m256_._A01_physics (returns only to 0.9890)`
- `m256_._A06_epistemology (returns only to 0.9937)`

**This is a result, not only a nuisance.** The convergence gate asks only that consecutive passes be similar. Approaching the upper edge the loop tends toward the identity map, so consecutive passes are similar whether or not anything has settled, and the gate stops being informative exactly there. A state can pass it while still drifting. Any claim that trials lock in at high injection should be read with that in mind.

Escape elsewhere is judged against each state's own floor: a rung escapes when it lands more than 10 times further from the attractor than the unperturbed control does. A single fixed cutoff cannot serve both groups, since 0.999 is a hundred times looser than the floor for one and tighter than the floor for the other.

## How much the cutoff matters

Escape is judged by whether the settled state returns to the
attractor, measured as a cosine. Choosing where to put that cutoff
is a judgement, and the returned cosines do NOT fall into two clean
clusters with a gap to cut in: about half land within a hair of
exactly 1 and the rest trail away continuously. So rather than
defend one number, here is the whole answer at four cutoffs. The
cosines are stored raw in the archive, so any other cutoff can be
applied without recomputing anything.

| cutoff | as an angle | median threshold | probes that never escaped |
|--:|--:|--:|--:|
| 0.9999 | 0.8 deg | 64 deg | 17 of 60 |
| 0.999 | 2.6 deg | 90 deg | 21 of 60 |
| 0.99 | 8.1 deg | 90 deg | 21 of 60 |
| 0.9 | 25.8 deg | 90 deg | 25 of 60 |

The tables below use 0.999. **If the median moves a
lot across those rows, no single depth figure from this experiment
should be quoted without its cutoff.**

## Escape thresholds per state, in both coordinates

Angles are the primary measure, because the pin removes the size
degree of freedom inside a run and only the direction is free.
But the same push has an exact size in the state's own units: a
rotation by theta moves the state a distance 2 sin(theta/2) times
its norm, so the displacement column below is that number, and the
two columns describe one push rather than two rival measures.
For reference 1 degree is 0.017 of the norm, 8 degrees 0.140,
64 degrees 1.060, and 90 degrees 1.414.

| level | basin | prompt | random directions | flip axis | glitch direction | median displacement |
|:--|:--|:--|:--|--:|--:|--:|
| m056 | `prolet` | A01_physics | 90 to 90 deg, median 90 | >90 | >90 | 1.414 |
| m056 | `prolet` | A02_medical | 64 to 90 deg, median 77 | >90 | >90 | 1.245 |
| m056 | `solidarity` | G09_space | 1 to 16 deg, median 5 | 2 | 16 | 0.087 |
| m056 | `Anarch` | G07_the | 32 to 90 deg, median 64 | >90 | >90 | 1.060 |
| historical | `prolet` | A01_physics | 90 to 90 deg, median 90 | >90 | >90 | 1.414 |
| historical | `prolet` | A02_medical | 64 to 90 deg, median 90 | >90 | >90 | 1.414 |
| historical | `Divine` | A08_linguistics | 32 to 90 deg, median 64 | 90 | 16 | 1.060 |
| historical | `Divine` | A14_kant | 4 to 8 deg, median 6 | 8 | 16 | 0.140 |
| historical | `Anarch` | A03_neuro | 90 to 90 deg, median 90 | >90 | >90 | 1.414 |
| historical | `Anarch` | A05_evolution | 90 to 90 deg, median 90 | >90 | >90 | 1.414 |

## The pre-stated expectations (issue #17)

1. **Spread across directions within a state**: random-direction thresholds run 1 to 90 degrees overall, median 90. 6 of 40 random probes did not escape at any rung up to 90 degrees.
2. **Named against random directions**: median threshold 16 degrees along the flip axis and the anomalous-cluster direction, against 90 along random ones. A lower named median would mean escape is easiest along structure the record already identified (F13, F14); parity means that connection is absent and is reported as absent.
4. **The token proxy against the state test**: of 480 rungs, 65 (14%) are classified differently by the printed token than by the settled state. The first version of this experiment used the token, which sees one position of about ten and reports only which logit is largest; this figure is how much that proxy cost.
3. **Across the band**: median threshold by level, m056 64 deg, historical 90 deg. The two outer levels sit near the band's edges and the historical pin sits mid-band, so smaller thresholds at the outer levels would mean the edges are basins growing shallow, and flat thresholds would mean the basins do not thin out but simply stop.

## Reading

Every number above is regenerated by re-running this script;
nothing is hand-computed. Converged states are regenerated by
iteration and each is checked against the terminal token the sweep
committed for the same level and prompt; a state that does not
match is skipped and named rather than used. The flip axis is
recomputed and gated against experiment 07's committed scalars.
Escape is judged by the terminal token, carrying the standing
single-phase caveat for periodic trials. The ladder locates a
threshold only to within a factor of about two, and two prompts per
basin is a probe: run 18 found small samples near a boundary can
mislead (FINDINGS caveat 19), so no edge claim rests on these
counts without widening.
