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

Of 17 states with both a ladder and a control, 16 passed and 1 failed. Failed states are excluded from every number below and named here:

- `m056_solidarity_G08_period (settles 'solidarity', re-entry gives 'till')`

A failing control is itself informative: the state sits close enough to a boundary that an identical tensor, re-entered through a separate code path, lands elsewhere. That is a real property of the state, not only a defect of the method, but it cannot be read as an escape threshold.

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
| 0.9999 | 0.8 deg | 1 deg | 17 of 96 |
| 0.999 | 2.6 deg | 1 deg | 21 of 96 |
| 0.99 | 8.1 deg | 24 deg | 22 of 96 |
| 0.9 | 25.8 deg | 90 deg | 33 of 96 |

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
| m056 | `solidarity` | G09_space | 1 to 8 deg, median 2 | 1 | 4 | 0.044 |
| m056 | `Anarch` | G07_the | 64 to 90 deg, median 64 | >90 | >90 | 1.060 |
| historical | `prolet` | A01_physics | 90 to 90 deg, median 90 | >90 | >90 | 1.414 |
| historical | `prolet` | A02_medical | 64 to 90 deg, median 90 | >90 | >90 | 1.414 |
| historical | `Divine` | A08_linguistics | 64 to 90 deg, median 77 | 90 | 16 | 1.245 |
| historical | `Divine` | A14_kant | 16 to 64 deg, median 48 | 16 | 16 | 0.416 |
| historical | `Anarch` | A03_neuro | 90 to 90 deg, median 90 | >90 | >90 | 1.414 |
| historical | `Anarch` | A05_evolution | 90 to 90 deg, median 90 | >90 | >90 | 1.414 |
| m256 | `solidarity` | A02_medical | 1 to 1 deg, median 1 | 1 | 1 | 0.017 |
| m256 | `solidarity` | A04_climate | 1 to 1 deg, median 1 | 1 | 1 | 0.017 |
| m256 | `the` | A03_neuro | 1 to 1 deg, median 1 | 1 | 1 | 0.017 |
| m256 | `the` | A05_evolution | 1 to 1 deg, median 1 | 1 | 1 | 0.017 |
| m256 | `.` | A01_physics | 1 to 1 deg, median 1 | 1 | 1 | 0.017 |
| m256 | `.` | A06_epistemology | 1 to 1 deg, median 1 | 1 | 1 | 0.017 |

## The pre-stated expectations (issue #17)

1. **Spread across directions within a state**: random-direction thresholds run 1 to 90 degrees overall, median 16. 7 of 64 random probes did not escape at any rung up to 90 degrees.
2. **Named against random directions**: median threshold 1 degrees along the flip axis and the anomalous-cluster direction, against 16 along random ones. A lower named median would mean escape is easiest along structure the record already identified (F13, F14); parity means that connection is absent and is reported as absent.
4. **The token proxy against the state test**: of 768 rungs, 284 (37%) are classified differently by the printed token than by the settled state. The first version of this experiment used the token, which sees one position of about ten and reports only which logit is largest; this figure is how much that proxy cost.
3. **Across the band**: median threshold by level, m056 64 deg, historical 90 deg, m256 1 deg. The two outer levels sit near the band's edges and the historical pin sits mid-band, so smaller thresholds at the outer levels would mean the edges are basins growing shallow, and flat thresholds would mean the basins do not thin out but simply stop.

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
