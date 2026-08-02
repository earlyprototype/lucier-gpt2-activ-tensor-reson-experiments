"""Nu-sweep Stage B: full-width brackets at the band edges (issue #113).

Stage A (25 prompts x 10 levels) found the five-basin band statistic at 0
percent through m032, 100 percent at m064, the historical pin and m128, and
44 percent at m256. The pre-stated rule brackets each 50 percent crossing at
full sweep width: this stage runs ALL 125 library prompts at the four levels
in the two crossing pairs (m032, m064) and (m128, m256).

Mechanically this is Stage A widened: same engine call, same gate, same
checkpoint directory and naming, so the 25-prompt trials Stage A already ran
at these levels are reused verbatim and only the remaining prompts compute.
The registered per-level statistic and the F15 classification rule are
unchanged (see 01_stage_a.py, whose machinery this file imports).

Run from the repo root:

    python3 experiments/nu_sweep/02_stage_b.py --worker 0 --num-workers 4
    ...
    python3 experiments/nu_sweep/02_stage_b.py --report-only

Outputs (in experiments/nu_sweep/output/):
    checkpoints/<level>_<pid>.pt   shared with Stage A (resume unit)
    stage_b_results.pt             combined archive for the four levels
    stage_b_report.md              every headline number, from the data
"""

import argparse
import importlib.util
import itertools
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "nu_sweep_stage_a", HERE / "01_stage_a.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

# Stage B widens the four crossing levels to the full library. Everything
# else (gate, engine call, checkpoint naming, F15 rule, band threshold) is
# Stage A's, imported above.
sa.N_PROMPTS = 125
sa.LEVELS = ["m032", "m064", "m128", "m256"]
sa.MULTIPLIERS = [32, 64, 128, 256]

OUT_DIR = sa.OUT_DIR
STAGE_A_PAIRS = {("m032", "m064"), ("m128", "m256")}


def write_report():
    """Assemble stage_b_results.pt and regenerate stage_b_report.md; every
    published number originates here."""
    import prompt_library
    results, missing = sa.collect()
    if missing:
        print(f"[report] WARNING: {len(missing)} trials missing; the report "
              f"marks itself partial. First few: {missing[:6]}")
    config = {
        "experiment": "nu_sweep_stage_b", "registration": "issue #113",
        "prompts": list(prompt_library.PROMPT_LIBRARY)[:sa.N_PROMPTS],
        "layers": f"{sa.LAYER_START}->{sa.LAYER_END}",
        "levels": sa.LEVELS, "multipliers": sa.MULTIPLIERS,
        "gate": {"max_iter": sa.MAX_ITER, "threshold": sa.THRESHOLD,
                 "patience": sa.PATIENCE, "check_every": sa.CHECK_EVERY,
                 "check_start": sa.CHECK_START, "gate_lag": 1},
        "band_threshold": sa.BAND_THRESHOLD,
        "torch": str(torch.__version__),
    }
    torch.save({"config": config, "results": results},
               OUT_DIR / "stage_b_results.pt")

    stats = {}
    for level in sa.LEVELS:
        rows = results[level]
        if not rows:
            continue
        labels = {}
        in_five = 0
        locked = []
        for r in rows.values():
            lag = sa.smallest_passing_lag(r)
            tok = r["terminal_token"].strip()
            labels[tok] = labels.get(tok, 0) + 1
            if lag is not None and tok in sa.REAL_FIVE:
                in_five += 1
            if r["converged"]:
                locked.append(r["lock_in_iter"])
        stats[level] = {
            "n": len(rows),
            "mean_pin": statistics.mean(
                r["target_norm"] for r in rows.values()),
            "share_in_five": in_five / len(rows),
            "n_in_five": in_five,
            "locked": len(locked),
            "median_lock_in": statistics.median(locked) if locked else None,
            "labels": labels,
        }
    ordered = [lv for lv in sa.LEVELS if lv in stats]
    crossings = [
        (a, b) for a, b in itertools.pairwise(ordered)
        if (a, b) in STAGE_A_PAIRS
        and (stats[a]["share_in_five"] > sa.BAND_THRESHOLD)
        != (stats[b]["share_in_five"] > sa.BAND_THRESHOLD)]

    lines = [
        "# Nu-sweep Stage B: full-width band-edge brackets",
        "",
        "Registered before execution in issue #113. Raw data:",
        "`stage_b_results.pt` (per-trial checkpoints in `checkpoints/`,",
        "shared with Stage A; the 25-prompt Stage A trials at these levels",
        "are reused verbatim). All 125 library prompts at the four levels",
        "of Stage A's two 50 percent crossings; registered gate settings;",
        "F15 classification at each trial's smallest passing lag.",
        "",
    ]
    if missing:
        lines += [f"**PARTIAL: {len(missing)} of "
                  f"{len(sa.LEVELS) * sa.N_PROMPTS} trials missing; numbers "
                  "below cover completed trials only.**", ""]
    lines += [
        "## Per-level summary at full width",
        "",
        "| level | mean pin | share in real five | locked (lag 1) "
        "| median lock-in | distinct labels |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for level in ordered:
        s = stats[level]
        med = (f"{s['median_lock_in']:.0f}"
               if s["median_lock_in"] is not None else "n/a")
        lines.append(
            f"| {level} | {s['mean_pin']:.0f} | "
            f"{s['share_in_five']:.0%} ({s['n_in_five']}/{s['n']}) | "
            f"{s['locked']}/{s['n']} | {med} | {len(s['labels'])} |")
    lines += ["", "## Basin table per level", ""]
    for level in ordered:
        s = stats[level]
        top = sorted(s["labels"].items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(f"- **{level}** (mean pin {s['mean_pin']:.0f}): "
                     + ", ".join(f"`{t}` {c}" for t, c in top))
    lines += [
        "",
        "## Edge determination (pre-stated rule)",
        "",
        f"- Band statistic: share of trials whose terminal label at their "
        f"smallest passing lag is one of {sorted(sa.REAL_FIVE)}.",
        f"- Stage A crossing pairs, re-tested at full width: "
        f"{sorted(STAGE_A_PAIRS)}.",
        f"- Crossings confirmed at full width: "
        f"{[f'{a} to {b}' for a, b in crossings] if crossings else 'none'}.",
        "",
        "## Reading",
        "",
        "Every number above is regenerated by re-running this script;",
        "nothing is hand-computed. Interpretation lands in issue #113 and",
        "the findings record, not here. Periodic-trial labels carry the",
        "same single-phase readout provenance as run 17's; no per-trial",
        "phase audit exists for them.",
    ]
    with open(OUT_DIR / "stage_b_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'stage_b_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None,
                    help="worker index into the level x prompt grid")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--report-only", action="store_true",
                    help="assemble results and regenerate the report")
    args = ap.parse_args()
    if args.report_only:
        write_report()
    elif args.worker is not None:
        sa.run_worker(args.worker, args.num_workers)
    else:
        ap.print_help()
