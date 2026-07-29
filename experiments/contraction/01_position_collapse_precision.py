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


def noise_floor(tensor, levels=ULP_LEVELS, seed=0):
    """What deviation would an EXACTLY collapsed tensor show anyway?

    Saying "the residual sits at float32 epsilon, so we cannot tell exact from
    1-part-in-1e7" is true but dodges the question that matters: is any of the
    measured deviation real, or is all of it the price of doing the arithmetic
    in float32 at all?

    The wrong null is to build a rank-1 tensor and round it to float32. Because
    the per-position norms here agree to ~1e-7, every row rounds to nearly the
    same float32 vector -- some come out bit-identical, deviation exactly 0.
    That measures storage, and storage is not where the deviation comes from.

    The deviation comes from arithmetic. Each iteration is a forward pass whose
    matmuls accumulate over 768+ dimensions in float32; even if the true map
    sends every position to one vector, each position's arithmetic takes a
    different path and lands a few ULPs away. So the null is an exactly
    collapsed tensor perturbed per-component at float32 ULP scale, which is what
    a float32 forward pass leaves behind whatever the underlying dynamics do.
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

    # How much of the measured deviation is unavoidable float32 arithmetic?
    print()
    print("Against the float32 noise floor: deviation of an EXACTLY collapsed tensor")
    print("carrying k ULPs of per-component arithmetic noise.")
    print()
    header3 = f"{'run':<26}{'observed':>13}" + "".join(
        f"{'k=' + str(k):>13}" for k in ULP_LEVELS
    )
    print(header3)
    print("-" * len(header3))
    at_one_ulp = 0
    for name, tensor, _ in load_runs():
        observed = 1.0 - engine_position_similarity(
            tensor.to(torch.float32).to(torch.float64)
        )[0]
        floors = noise_floor(tensor.to(torch.float32))
        if observed <= 4 * floors[0]:
            at_one_ulp += 1
        print(
            f"{name:<26}{observed:>13.2e}"
            + "".join(f"{f:>13.2e}" for f in floors)
        )

    print()
    print(f"runs within 4x of the one-ULP floor: {at_one_ulp}/{len(rows)}")
    print(
        "\nWhere observed sits on the k=1 column, one unit in the last place of\n"
        "float32 per component -- the smallest non-zero perturbation the format\n"
        "can hold -- reproduces the measurement on its own. There is then nothing\n"
        "left to attribute to a genuine residual, and the collapse is exact as far\n"
        "as the question can be posed in float32."
    )


if __name__ == "__main__":
    main()
