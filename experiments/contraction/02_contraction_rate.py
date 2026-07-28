"""M2 (#71): contraction rate, from committed data only. No forward passes.

M2 as worded asks for log(1 - position_similarity) against iteration. That plot
cannot be drawn from anything in this repository. `position_similarity` needs the
full [positions, d_model] tensor at each iteration, and no archive keeps it:

  * output_divine_motion/snapshots_*.pt  keeps last_vector and mean_vector, both 1-D
  * output_divine_motion/state_*.pt      keeps the full tensor, final iteration only
  * output_confidence/converged_tensors.pt   full tensor, converged only
  * sink_geometry/output/trajectories.pt keeps the per-position mean, per iteration

So the per-iteration record is of the state's *mean over positions*, not of the
spread between positions. The contraction rate of that trajectory is measurable,
and it is the same kind of number M2 wants -- fit log(1 - cos) against iteration
and read the slope -- but it is a different quantity, and this script does not
pretend otherwise.

Two estimators are reported, because the obvious one is biased:

  vs last iterate   1 - cos(v_t, v_T).  This is what M2 literally describes, but
                    it is anchored on the last recorded state rather than a true
                    fixed point, so it is driven to zero at t = T by construction
                    and steepens near the end regardless of the dynamics. The
                    final 20% of points are dropped to blunt that; it is still
                    the weaker of the two.
  step-to-step      1 - cos(v_t, v_t+1).  References no fixed point at all.
                    This is the primary number.

Where the two agree, the rate is real. R^2 is reported throughout: a geometric
approach is a straight line on these axes, so a low R^2 means the process is not
a single rate and the slope should not be quoted as if it were.

Run:  python experiments/contraction/02_contraction_rate.py
"""

import glob
import math
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAJECTORIES = os.path.join(REPO, "experiments", "sink_geometry", "output", "trajectories.pt")
SNAPSHOTS = os.path.join(REPO, "experiments", "gpt2_small", "output_divine_motion", "snapshots_*.pt")

# Below this, 1 - cos is float64 noise rather than signal.
FLOOR = 1e-10


def least_squares(points):
    """points: [(x, y)]. Returns (slope, r_squared, n) or None."""
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return slope, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), n


def unit(trajectory):
    trajectory = trajectory.to(torch.float64)
    return trajectory / trajectory.norm(dim=1, keepdim=True).clamp(min=1e-30)


def step_points(trajectory, iterations, mask_top=0):
    """log(1 - cos(v_t, v_t+1)) against the midpoint of each step."""
    trajectory = trajectory.to(torch.float64).clone()
    if mask_top:
        biggest = trajectory.abs().mean(0).argsort(descending=True)[:mask_top]
        trajectory[:, biggest] = 0.0
    v = unit(trajectory)
    points = []
    for i in range(len(v) - 1):
        deviation = 1.0 - (v[i] @ v[i + 1]).clamp(max=1.0).item()
        if deviation > FLOOR:
            points.append(((iterations[i] + iterations[i + 1]) / 2, math.log(deviation)))
    return points


def anchored_points(trajectory, iterations):
    """log(1 - cos(v_t, v_T)), final 20% dropped."""
    v = unit(trajectory)
    deviations = (1.0 - (v[:-1] @ v[-1]).clamp(max=1.0)).tolist()
    keep = max(1, int(len(deviations) * 0.8))
    return [
        (iterations[i], math.log(d))
        for i, d in enumerate(deviations[:keep])
        if d > FLOOR
    ]


def fmt(fit):
    if fit is None:
        return "       no fit"
    slope, r2, _ = fit
    return f"{slope:+8.4f} R2={r2:.3f}"


def half_life(fit):
    if fit is None or fit[0] >= 0:
        return "     n/a"
    return f"{math.log(0.5) / fit[0]:8.2f}"


def report_trajectories():
    bundle = torch.load(TRAJECTORIES, map_location="cpu", weights_only=False)
    manifest = bundle["manifest"]

    print("Contraction rate of the state trajectory, per model, per prompt")
    print(f"  source: {os.path.relpath(TRAJECTORIES, REPO)}")
    print(f"  manifest: {dict(manifest)}")
    print()

    header = (
        f"{'model':<14}{'prompt':>7}{'step-to-step':>20}{'vs last iterate':>20}"
        f"{'half-life':>11}{'first half':>20}{'second half':>20}"
    )
    print(header)
    print("-" * len(header))

    summary = {}
    for model in manifest["models"]:
        fits = []
        for index, means in enumerate(bundle["traj"][model]["means"]):
            iterations = list(range(means.shape[0]))
            steps = step_points(means, iterations)
            step_fit = least_squares(steps)
            anchor_fit = least_squares(anchored_points(means, iterations))
            midpoint = len(steps) // 2
            first = least_squares(steps[:midpoint])
            second = least_squares(steps[midpoint:])
            print(
                f"{model:<14}{index:>7}{fmt(step_fit):>20}{fmt(anchor_fit):>20}"
                f"{half_life(step_fit):>11}{fmt(first):>20}{fmt(second):>20}"
            )
            if step_fit:
                fits.append(step_fit)
        summary[model] = fits
        print()
    return summary


def report_masking():
    """Does the massive-activation finding explain GPT-2 Small's poor fit?

    experiments/sink_geometry/RESULTS.md found that a handful of oversized
    dimensions dominate GPT-2 Small's cosines, and there *hide* convergence
    rather than inflate it. If the same dimensions are why GPT-2 Small will not
    fit a single rate here, masking them should straighten it -- and should leave
    the other three models, which that run found insensitive to masking,
    unchanged. This is a prediction that can fail.
    """
    bundle = torch.load(TRAJECTORIES, map_location="cpu", weights_only=False)
    print("Does masking the oversized dimensions straighten the fit?")
    print()
    header = f"{'model':<14}{'prompt':>7}" + "".join(
        f"{'mask ' + str(k):>20}" for k in (0, 10, 50)
    )
    print(header)
    print("-" * len(header))
    for model in bundle["manifest"]["models"]:
        for index, means in enumerate(bundle["traj"][model]["means"]):
            iterations = list(range(means.shape[0]))
            cells = [
                fmt(least_squares(step_points(means, iterations, mask_top=k)))
                for k in (0, 10, 50)
            ]
            print(f"{model:<14}{index:>7}" + "".join(f"{c:>20}" for c in cells))
        print()


def report_long_runs():
    """Cross-check on the 1000-iteration GPT-2 Small runs.

    These are recorded on a deliberately non-uniform schedule (0, 100, 250, 500,
    then every 10 from 800). The x axis must be the recorded iteration, not the
    snapshot index -- treating the index as the iteration compresses 800 steps
    into 4 and inflates the slope by two orders of magnitude.
    """
    print("Cross-check: the 1000-iteration runs (non-uniform schedule)")
    print(f"  source: {os.path.relpath(SNAPSHOTS, REPO)}")
    print()
    header = f"{'run':<26}{'snapshots':>11}{'step-to-step':>20}{'vs last iterate':>20}"
    print(header)
    print("-" * len(header))
    for path in sorted(glob.glob(SNAPSHOTS)):
        run = torch.load(path, map_location="cpu", weights_only=False)
        snapshots = run["snapshots"]
        iterations = [s["iteration"] for s in snapshots]
        means = torch.stack([s["mean_vector"] for s in snapshots])
        print(
            f"{run['label']:<26}{len(iterations):>11}"
            f"{fmt(least_squares(step_points(means, iterations))):>20}"
            f"{fmt(least_squares(anchored_points(means, iterations))):>20}"
        )
    print()


def main():
    print("M2 -- contraction rate (committed data, no forward passes)")
    print()
    print("NOTE: log(1 - position_similarity) vs iteration, which M2 asks for, is")
    print("      not drawable from committed data -- no archive keeps the full")
    print("      per-position tensor per iteration. What follows is the contraction")
    print("      of the per-position mean, which is a different quantity.")
    print()
    report_trajectories()
    report_masking()
    report_long_runs()


if __name__ == "__main__":
    main()
