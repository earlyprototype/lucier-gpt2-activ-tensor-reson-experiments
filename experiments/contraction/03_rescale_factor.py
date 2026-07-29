"""M5 (#71): the per-iteration rescale factor c_n -- from committed data.

M5 is filed `GATED` `EXPERIMENT` on the grounds that the ratio is "currently
never recorded". That is not true of the engine: `atr_engine.py` stores
`tensor_norm` in every snapshot including iteration 0, so

    c_{n+1} = snapshots[0]["tensor_norm"] / snapshots[n]["tensor_norm"]

is available directly from any run whose snapshots were saved intact. No engine
change and no forward pass is needed.

What is missing is that two experiment scripts discarded those fields at save
time -- `05_divine_motion.py:118` reimplements the snapshot and keeps 7 of the
engine's 20 fields; `sink_geometry/02_masking_control.py:86-88` keeps only the
per-position mean. This script therefore reconstructs c from what those slimmed
archives did retain, which works because of the order of operations in the loop
(atr_engine.py:211-216):

    for i in 1..max_iter:
        current_norm = ||current_tensor||          # PRE-rescale
        current_tensor *= initial_norm / current_norm
        ... forward pass ...
        current_tensor = new state                 # NOT rescaled
        ... snapshot recorded here ...

Every recorded state is post-forward and pre-rescale. So the norm of a recorded
snapshot is exactly the denominator of c for the following iteration:

    c_{n+1} = initial_norm / ||x_n||

The archived snapshots dropped `tensor_norm`, but kept `last_norm`, and once
positions have collapsed ||x|| = sqrt(seq_len) * ||any position||. Step 1 below
checks that identity against the state files, which record both, before anything
depends on it. `initial_norm` is recorded directly in the state files.

The result bears on H-pos0 as stated in #75, which asserts that at a settled
state c_n = 1. Measured, it is a stable constant that is not 1. See RESULTS.md.

Run:  python experiments/contraction/03_rescale_factor.py
"""

import glob
import math
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(REPO, "experiments", "gpt2_small", "output_divine_motion")
STATES = os.path.join(DIR, "state_*.pt")

# The collapse identity is exact only to float32; anything under this is noise.
IDENTITY_TOL = 1e-5


def load_pairs():
    """Each run's (label, state, snapshots), where both files exist."""
    pairs = []
    for path in sorted(glob.glob(STATES)):
        state = torch.load(path, map_location="cpu", weights_only=True)
        snap_path = path.replace("state_", "snapshots_")
        if not os.path.exists(snap_path):
            continue
        snaps = torch.load(snap_path, map_location="cpu", weights_only=True)
        pairs.append((state["label"], state, snaps["snapshots"]))
    return pairs


def check_identity(pairs):
    """||tensor|| == sqrt(seq_len) * last_norm, once positions have collapsed.

    Everything downstream reconstructs the tensor norm from `last_norm` this
    way. If it does not hold, none of the c_n values below mean anything, so it
    is checked rather than assumed.
    """
    print("Step 1 -- verify ||tensor|| == sqrt(seq_len) * last_norm after collapse")
    print()
    all_ok = True
    for label, state, _ in pairs:
        tensor = state["current_tensor"]
        seq_len = tensor.shape[0]
        direct = tensor.norm().item()
        reconstructed = math.sqrt(seq_len) * tensor[-1].norm().item()
        rel_err = abs(direct - reconstructed) / direct
        ok = rel_err < IDENTITY_TOL
        all_ok &= ok
        print(
            f"  {label:<26} ||T||={direct:>9.2f}  "
            f"sqrt({seq_len})*last={reconstructed:>9.2f}   "
            f"rel err {rel_err:.1e}   {'OK' if ok else 'MISMATCH'}"
        )
    print()
    if not all_ok:
        raise SystemExit(
            "  identity FAILED -- the reconstruction below would be invalid. Stopping."
        )
    print("  identity holds; the reconstruction below is sound.")
    print()
    return all_ok


def settled_values(pairs):
    print("Step 2 -- c at the settled state")
    print()
    header = (
        f"{'run':<26}{'seq':>4}{'initial_norm':>14}{'settled ||x||':>15}"
        f"{'c':>10}{'amplification':>15}"
    )
    print(header)
    print("-" * len(header))
    for label, state, _ in pairs:
        tensor = state["current_tensor"]
        initial = state["initial_norm"]
        settled = tensor.norm().item()
        print(
            f"{label:<26}{tensor.shape[0]:>4}{initial:>14.2f}{settled:>15.2f}"
            f"{initial / settled:>10.4f}{settled / initial:>15.4f}"
        )
    print()


def trajectory(pairs):
    """c_n across the run.

    Only the post-collapse values are exact: before collapse the positions are
    not parallel, so reconstructing ||tensor|| from `last_norm` overestimates or
    underestimates it. Iteration 0 is shown for context and marked, not used.
    """
    print("Step 3 -- does c_n vary, or settle to a constant?")
    print("  (iteration 0 is pre-collapse, so its value is an artefact of the")
    print("   reconstruction and is marked * rather than trusted)")
    print()
    for label, state, snaps in pairs:
        seq_len = state["current_tensor"].shape[0]
        initial = state["initial_norm"]
        print(f"  {label}  (seq_len={seq_len}, initial_norm={initial:.2f})")
        values = [
            (s["iteration"], initial / (math.sqrt(seq_len) * s["last_norm"]))
            for s in snaps
        ]
        shown = values[:4] + [None] + values[-3:]
        cells = []
        for v in shown:
            if v is None:
                cells.append("...")
            else:
                cells.append(f"n={v[0]}:{v[1]:.4f}" + ("*" if v[0] == 0 else ""))
        print("    " + "  ".join(cells))
        settled = [v for n, v in values if n >= 100]
        if settled:
            print(
                f"    from n=100 on: min {min(settled):.4f}  max {max(settled):.4f}  "
                f"spread {max(settled) - min(settled):.1e}"
            )
    print()


def main():
    pairs = load_pairs()
    if not pairs:
        raise SystemExit("no state_*.pt / snapshots_*.pt pairs found")
    print("M5 -- per-iteration rescale factor c_n (committed data, no forward passes)")
    print(f"  source: {os.path.relpath(DIR, REPO)}")
    print()
    check_identity(pairs)
    settled_values(pairs)
    trajectory(pairs)
    print("What this does NOT cover -- all limits of these archives, not of the")
    print("engine, which recorded every one of these and had it thrown away:")
    print("  * iterations 1-99, which 05_divine_motion.py's schedule does not sample,")
    print("    so the approach of c_n to its settled value is unobserved")
    print("  * any run whose seq_len or initial_norm was not archived")
    print("  * the pre-collapse regime, where the reconstruction is not exact")


if __name__ == "__main__":
    main()
