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
  step-to-step      1 - cos(v_t, v_t+D).  References no fixed point at all.
                    This is the primary number, but it is only comparable across
                    steps of equal width D: the deviation carries a factor
                    (1 - exp(-lambda*D))^2, so gaps of 10 and 100 iterations are
                    not the same quantity. step_points() enforces one width and
                    reports what that discards.

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
    """log(1 - cos(v_t, v_t+D)) against the midpoint of each step.

    Only steps of a single, constant width D are returned. This matters and is
    easy to miss: for a geometric approach to a fixed point the deviation
    carries a factor (1 - exp(-lambda*D))^2, so a 100-iteration gap and a
    10-iteration gap are not on the same scale, and fitting them together biases
    the slope no matter how the x axis is labelled. Putting the recorded
    iteration on the x axis fixes the *spacing* of the points but not the
    *quantity* being plotted.

    The uniformly-spaced trajectories (D = 1 throughout) are unaffected. The
    divine_motion long runs are recorded on gaps of 10, 100, 150, 250 and 300
    and are not, which is what this guard exists for. Where several widths are
    available the one yielding the most usable points is kept, and the caller is
    told which via the returned width.

    Returns (points, width, dropped) -- dropped counts usable points discarded
    for having the wrong width, so a caller can report the truncation instead of
    silently presenting a partial fit as a whole one.
    """
    trajectory = trajectory.to(torch.float64).clone()
    if mask_top:
        biggest = trajectory.abs().mean(0).argsort(descending=True)[:mask_top]
        trajectory[:, biggest] = 0.0
    v = unit(trajectory)

    by_width = {}
    for i in range(len(v) - 1):
        deviation = 1.0 - (v[i] @ v[i + 1]).clamp(max=1.0).item()
        if deviation <= FLOOR:
            continue
        width = iterations[i + 1] - iterations[i]
        midpoint = (iterations[i] + iterations[i + 1]) / 2
        by_width.setdefault(width, []).append((midpoint, math.log(deviation)))

    if not by_width:
        return [], None, 0
    width = max(by_width, key=lambda w: len(by_width[w]))
    total = sum(len(p) for p in by_width.values())
    return by_width[width], width, total - len(by_width[width])


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
    bundle = torch.load(TRAJECTORIES, map_location="cpu", weights_only=True)
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
            steps, _width, _dropped = step_points(means, iterations)
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
    bundle = torch.load(TRAJECTORIES, map_location="cpu", weights_only=True)
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
                fmt(least_squares(step_points(means, iterations, mask_top=k)[0]))
                for k in (0, 10, 50)
            ]
            print(f"{model:<14}{index:>7}" + "".join(f"{c:>20}" for c in cells))
        print()


def report_long_runs():
    """Cross-check on the 1000-iteration GPT-2 Small runs.

    These are recorded on a deliberately non-uniform schedule (0, 100, 250, 500,
    then every 10 from 800), which costs the step estimator twice over.

    First, the x axis must be the recorded iteration, not the snapshot index --
    treating the index as the iteration compresses 800 steps into 4 and inflates
    the slope by two orders of magnitude.

    Second, and less obvious: the *quantity* also depends on the gap width, so
    even with a correct x axis the widths cannot be mixed. step_points() keeps
    only one width; this reports which, and how much was dropped, so a fit over
    a fifth of the run is not mistaken for a fit over all of it.

    The anchored estimator is unaffected -- it measures every point against one
    fixed reference, so gap width does not enter.
    """
    print("Cross-check: the 1000-iteration runs (non-uniform schedule)")
    print(f"  source: {os.path.relpath(SNAPSHOTS, REPO)}")
    print()
    header = (
        f"{'run':<26}{'snaps':>7}{'gap used':>10}{'dropped':>9}"
        f"{'step-to-step':>20}{'vs last iterate':>20}"
    )
    print(header)
    print("-" * len(header))
    for path in sorted(glob.glob(SNAPSHOTS)):
        run = torch.load(path, map_location="cpu", weights_only=True)
        snapshots = run["snapshots"]
        iterations = [s["iteration"] for s in snapshots]
        means = torch.stack([s["mean_vector"] for s in snapshots])
        steps, width, dropped = step_points(means, iterations)
        print(
            f"{run['label']:<26}{len(iterations):>7}"
            f"{('-' if width is None else str(width)):>10}{dropped:>9}"
            f"{fmt(least_squares(steps)):>20}"
            f"{fmt(least_squares(anchored_points(means, iterations))):>20}"
        )
    print()
    print("  'dropped' counts points above the floor discarded for having a")
    print("  different gap width. Where it is large, the step fit covers only")
    print("  part of the run and the anchored column is the better guide.")
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
