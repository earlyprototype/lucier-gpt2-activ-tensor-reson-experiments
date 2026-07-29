"""M1 (#71): position_similarity at full precision, from committed data only.

No forward passes. Reads tensors already in the repository and recomputes the
metric that `atr_engine.py` reports, so the numbers here are directly comparable
to what a live run prints.

The question M1 exists to settle is whether the reported position collapse is
exact or merely close: whether `position_similarity` is 1.000000... or 0.9999.
The published figures are rounded to 4 dp, at which everything reads 1.0000.

Two things are computed for each committed converged tensor:

  * the engine's own metric, verbatim, in float32 (its native path);
  * the same metric in float64, so float32 accumulation order cannot be what
    produces or hides the answer.

and then the follow-on question H-pos0 actually needs. H-pos0 says position 0's
forward map is autonomous up to one scalar c_n per position. If every position
holds the same *direction*, what separates them is exactly that scalar -- so the
spread of c_n at convergence is the quantity to read, not just the cosine.

Run:  python experiments/contraction/01_position_collapse_precision.py
"""

import glob
import math
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERGED = os.path.join(
    REPO, "experiments", "gpt2_small", "output_confidence", "converged_tensors.pt"
)
STATES = os.path.join(REPO, "experiments", "gpt2_small", "output_divine_motion", "state_*.pt")

FLOAT32_EPS = 2.0 ** -23  # 1.19e-07

# ULP multiples used by the noise-floor comparison below.
ULP_LEVELS = (1, 4, 16, 64)


def engine_position_similarity(tensor):
    """Verbatim from atr_engine.py, 'Position collapse metric'."""
    seq_len = tensor.shape[0]
    pos_norms = tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized_positions = tensor / pos_norms
    pos_sim_matrix = normalized_positions @ normalized_positions.T
    mask = ~torch.eye(seq_len, dtype=torch.bool, device=pos_sim_matrix.device)
    return pos_sim_matrix[mask].mean().item(), pos_sim_matrix, mask


def measure(name, tensor, iteration):
    tensor = tensor.to(torch.float32)
    f32, _, _ = engine_position_similarity(tensor)
    f64, matrix, mask = engine_position_similarity(tensor.to(torch.float64))

    deviation = 1.0 - f64
    worst_pair = matrix[mask].min().item()
    # 1 - cos ~ theta^2 / 2 for small theta
    theta = math.sqrt(max(0.0, 2.0 * deviation))

    # The per-position scalar c_n, taken relative to position 0.
    norms = tensor.to(torch.float64).norm(dim=1)
    c_n = norms / norms[0]

    # A tensor whose positions are all one direction is rank 1.
    singular = torch.linalg.svdvals(tensor.to(torch.float64))

    return {
        "name": name,
        "positions": tensor.shape[0],
        "iteration": iteration,
        "f32": f32,
        "f64": f64,
        "deviation": deviation,
        "worst_pair": worst_pair,
        "theta": theta,
        "c_n_spread": (c_n.max() - c_n.min()).item(),
        "sigma_ratio": (singular[1] / singular[0]).item(),
    }


def rms_residual_scale(deviation):
    """RMS angular-residual scale: d = sqrt(1 - position_similarity).

    An order-of-magnitude summary of how far apart the positions are, and
    nothing more. The geometry behind it is exact -- 1 - cos = theta^2 / 2, so
    d is the angle up to a factor of sqrt(2) -- but reading d as a PER-COMPONENT
    relative error is not: that step needs the disagreement to be spread evenly
    and independently across coordinates, and for these tensors it is not (one
    coordinate carries 4-55% of it). So d is reported as a scale, in units of
    float32 epsilon for legibility, and the per-coordinate question is answered
    by `per_coordinate_disagreement` instead.
    """
    return math.sqrt(max(deviation, 0.0))


def per_coordinate_disagreement(tensor):
    """Coordinate-by-coordinate relative disagreement, assuming nothing.

    `rms_residual_scale` above summarises the angle into one number, but
    turning that into a per-component statement assumes the disagreement is
    spread evenly over coordinates, which it is not. This measures the
    disagreement per coordinate instead.

    Under relative rounding every coordinate carries |du_k| / |u_k| ~ eps
    regardless of its size, so the disagreement being CONCENTRATED proves
    nothing on its own: relative rounding of a state whose energy is 91% in ten
    coordinates is concentrated too. What separates rounding from structure is
    whether the relative disagreement is flat at a few eps across coordinates.

    Scale is divided out first -- H-pos0 allows each position its own scalar
    (see M5), so what is at issue is the direction, not the length.

    Returns quantiles of |du_k| / |u_k| in units of eps, plus how far the
    worst-relative coordinates sit below typical magnitude and what share of the
    angle they carry. Those last two matter because a coordinate near zero has a
    huge relative error for a negligible absolute one.
    """
    t = tensor.to(torch.float64)
    unit = t / t.norm(dim=1, keepdim=True)
    seq_len = unit.shape[0]

    relative, magnitude, contribution = [], [], []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            scale = torch.maximum(unit[i].abs(), unit[j].abs())
            keep = scale > 0
            gap = (unit[i] - unit[j]).abs()
            relative.append(gap[keep] / scale[keep] / FLOAT32_EPS)
            magnitude.append(scale[keep])
            contribution.append(gap[keep] ** 2)

    rel = torch.cat(relative)
    mag = torch.cat(magnitude)
    con = torch.cat(contribution)
    tail = rel > 100
    return {
        "median": rel.quantile(0.5).item(),
        "p90": rel.quantile(0.9).item(),
        "p99": rel.quantile(0.99).item(),
        "tail_fraction": tail.float().mean().item(),
        "tail_magnitude_ratio": (
            mag[tail].median().item() / mag.median().item() if tail.any() else float("nan")
        ),
        "tail_angle_share": (
            con[tail].sum().item() / con.sum().item() if tail.any() else 0.0
        ),
    }


def noise_floor(tensor, levels=ULP_LEVELS, seed=0):
    """Secondary cross-check: a synthetic sensitivity sweep, NOT a ULP baseline.

    Read `per_coordinate_disagreement` above first -- that is the primary
    evidence, and it needs none of this.

    This perturbs an exactly collapsed tensor by relative Gaussian noise of size
    k * FLOAT32_EPS and reports the deviation that produces. Being relative, it
    does scale with each component's magnitude, so it is not tied to numbers
    near 1. But Gaussian noise of standard deviation eps is NOT float32
    round-to-nearest, whose relative error is bounded by eps/2 and roughly
    uniform (standard deviation about eps / (2*sqrt(3)), some 3.5x smaller). Nor
    is a single rounding the right comparison: the archived state is the output
    of a twelve-layer forward pass, so its accumulated error is worth several
    roundings, not one.

    So the k column that matches an observation should NOT be read as "the
    collapse is exact to k ULPs". What the sweep is good for is showing how
    steeply the deviation moves with the assumed error scale -- a factor of 4 in
    k moves it by ~15x -- which is why the conclusion survives the constant
    being uncertain by a factor of a few.
    """
    generator = torch.Generator().manual_seed(seed)
    t64 = tensor.to(torch.float64)
    direction = t64.mean(0)
    direction = direction / direction.norm()
    exact = t64.norm(dim=1, keepdim=True) * direction.unsqueeze(0)

    out = []
    for k in levels:
        jitter = torch.randn(
            exact.shape, dtype=torch.float64, generator=generator
        )
        perturbed = exact * (1.0 + k * FLOAT32_EPS * jitter)
        out.append(1.0 - engine_position_similarity(perturbed)[0])
    return out


def load_runs():
    runs = []
    for label, tensor in torch.load(
        CONVERGED, map_location="cpu", weights_only=True
    ).items():
        runs.append((label, tensor, "converged"))
    for path in sorted(glob.glob(STATES)):
        state = torch.load(path, map_location="cpu", weights_only=True)
        runs.append((state["label"], state["current_tensor"], str(state["iteration"])))
    return runs


def main():
    rows = [measure(*run) for run in load_runs()]

    print("M1 -- position_similarity at full precision (committed data, no forward passes)")
    print(f"  {os.path.relpath(CONVERGED, REPO)}")
    print(f"  {os.path.relpath(STATES, REPO)}")
    print()

    header = (
        f"{'run':<26}{'pos':>4}{'iter':>10}"
        f"{'engine (f32)':>16}{'float64':>18}{'1 - sim':>12}{'worst pair':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<26}{r['positions']:>4}{r['iteration']:>10}"
            f"{r['f32']:>16.10f}{r['f64']:>18.12f}"
            f"{r['deviation']:>12.2e}{r['worst_pair']:>16.12f}"
        )

    print()
    header2 = f"{'run':<26}{'angle (rad)':>14}{'sigma2/sigma1':>16}{'c_n spread':>14}"
    print(header2)
    print("-" * len(header2))
    for r in rows:
        print(
            f"{r['name']:<26}{r['theta']:>14.2e}"
            f"{r['sigma_ratio']:>16.2e}{r['c_n_spread']:>14.2e}"
        )

    exact = [r["name"] for r in rows if r["f64"] == 1.0]
    at_4dp = sum(1 for r in rows if round(r["f64"], 4) == 1.0)
    worst = max(rows, key=lambda r: r["deviation"])
    largest_angle = max(r["theta"] for r in rows)

    print()
    print(f"runs measured                    : {len(rows)}")
    print(f"reading 1.0000 at 4 dp           : {at_4dp}/{len(rows)}")
    print(f"exactly 1.0 in float64           : {exact or 'none'}")
    print(f"largest 1 - position_similarity  : {worst['deviation']:.2e}  ({worst['name']})")
    print(f"largest angle between positions  : {largest_angle:.2e} rad")
    print(f"float32 epsilon                  : {FLOAT32_EPS:.2e}")
    print()
    if largest_angle < 10 * FLOAT32_EPS:
        print(
            "The angle between positions is within an order of magnitude of the\n"
            "precision the tensors were archived at, so the raw figures above cannot\n"
            "by themselves separate 'exactly parallel' from 'parallel to about one\n"
            "part in 1e7'. The comparison below settles how much of that is real."
        )
    else:
        print(
            "The angle between positions is well above float32 epsilon, so it is a\n"
            "measurement rather than a storage artefact."
        )

    # First cut: the scale of the residual. An RMS summary, not per-coordinate.
    print()
    print("Scale of the residual: d = sqrt(1 - position_similarity), an RMS")
    print("angular summary. NOT a per-coordinate figure -- see the next table.")
    print()
    header3 = f"{'run':<26}{'1 - sim':>12}{'d':>12}{'d / eps32':>12}"
    print(header3)
    print("-" * len(header3))
    within = 0
    for r in rows:
        d = rms_residual_scale(r["deviation"])
        if d <= 2 * FLOAT32_EPS:
            within += 1
        print(f"{r['name']:<26}{r['deviation']:>12.2e}{d:>12.2e}{d / FLOAT32_EPS:>12.2f}")

    print()
    print(f"float32 epsilon                          : {FLOAT32_EPS:.3e}")
    print(f"runs with d within 2 eps                 : {within}/{len(rows)}")
    print(
        "\nd is an RMS summary: turning it into a per-coordinate statement assumes\n"
        "the disagreement is spread evenly, and it is not. The next table drops\n"
        "that assumption."
    )

    # Per coordinate, assuming nothing about how the disagreement is spread.
    print()
    print("Per coordinate: relative disagreement |du_k| / |u_k| in units of eps,")
    print("scale divided out (each position may keep its own scalar -- see M5).")
    print()
    header4 = (
        f"{'run':<26}{'median':>9}{'p90':>9}{'p99':>9}"
        f"{'>100 eps':>10}{'their |u|':>11}{'their angle':>13}"
    )
    print(header4)
    print("-" * len(header4))
    for name, tensor, _ in load_runs():
        m = per_coordinate_disagreement(tensor)
        print(
            f"{name:<26}{m['median']:>9.2f}{m['p90']:>9.2f}{m['p99']:>9.2f}"
            f"{m['tail_fraction'] * 100:>9.2f}%{m['tail_magnitude_ratio']:>11.4f}"
            f"{m['tail_angle_share'] * 100:>12.2f}%"
        )

    print(
        "\nThe typical coordinate agrees to about 2 eps. The heavy p99 is confined to\n"
        "coordinates far below typical magnitude ('their |u|', as a fraction of the\n"
        "median) which together carry ~1% of the angle: a small-denominator artefact,\n"
        "not a finding. Where the angle actually lives, agreement is at the few-eps\n"
        "level -- the scale float32 arithmetic operates at.\n"
        "\nThis bounds the residual; it does not prove the absence of structure below\n"
        "that scale, and nothing here should be read as proving it."
    )

    # SECONDARY: sensitivity sweep. See noise_floor's docstring for what this is
    # and, more importantly, what it is not.
    print()
    print("Secondary, synthetic: deviation of an exactly collapsed tensor carrying")
    print("relative Gaussian noise of k x eps32. NOT a ULP baseline -- Gaussian(eps)")
    print("is not round-to-nearest, and the state is a 12-layer pass, not one")
    print("rounding. Shown for the slope: k x4 moves the deviation ~15x, so the")
    print("reading above survives the error scale being off by a factor of a few.")
    print()
    header4 = f"{'run':<26}{'observed':>13}" + "".join(
        f"{'k=' + str(k):>13}" for k in ULP_LEVELS
    )
    print(header4)
    print("-" * len(header4))
    for name, tensor, _ in load_runs():
        observed = 1.0 - engine_position_similarity(
            tensor.to(torch.float32).to(torch.float64)
        )[0]
        floors = noise_floor(tensor.to(torch.float32))
        print(f"{name:<26}{observed:>13.2e}" + "".join(f"{f:>13.2e}" for f in floors))


if __name__ == "__main__":
    main()
