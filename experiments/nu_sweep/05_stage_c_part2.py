"""Nu-sweep Stage C Part 2: full-width brackets at the new crossings (#116).

Stage C's fine scan moved both 50 percent crossings: the lower edge to
(m040, m048) and the upper edge to (m256, m384). The registered Part 2 rule
(issue #116, inherited from Stage B) runs BOTH endpoints of every crossing
pair at all 125 library prompts, unless an endpoint is already at full
width. m256 already is (Stage B).

The first pass widened m040, m048 and m384, and its own numbers moved the
lower crossing again: m048 read 56 percent in the five on 25 prompts and 31
percent on 125, so the crossing is now (m048, m056) and the same rule pulls
m056 in. This file therefore widens m040, m048, m056 and m384. Re-running it
is safe and cheap: completed trials are skipped by their checkpoints.

Mechanically identical to Stage B: Stage A's engine call, gate, F15 rule,
shared checkpoint directory and naming, and level_stats. The 25-prompt Stage
C trials at these levels are reused verbatim.

Run from the repo root:

    python3 experiments/nu_sweep/05_stage_c_part2.py --worker 0 --num-workers 4
    ...
    python3 experiments/nu_sweep/05_stage_c_part2.py --report-only

Outputs (in experiments/nu_sweep/output/):
    checkpoints/<level>_<pid>.pt   shared with Stages A, B and C
    stage_c_part2_results.pt       archive for the three widened levels
    stage_c_part2_report.md        every headline number, from the data
"""

import argparse
import importlib.util
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

# The endpoints of Stage C's two crossings that are not already full width.
sa.N_PROMPTS = 125
sa.LEVELS = ["m040", "m048", "m056", "m384"]
sa.MULTIPLIERS = [40, 48, 56, 384]

OUT_DIR = sa.OUT_DIR
# The crossing pairs this stage tests, both endpoints included; m256 is
# carried from Stage B's full-width run.
CROSSING_PAIRS = [("m040", "m048"), ("m048", "m056"), ("m256", "m384")]


def write_report():
    """Assemble the archive and regenerate the report over the three widened
    levels plus m256 (already full width, needed to state both crossings);
    every published number originates here."""
    import prompt_library
    results, missing = sa.collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if missing:
        print(f"[report] WARNING: {len(missing)} trials missing; the report "
              f"marks itself partial. First few: {missing[:6]}")
    config = {
        "experiment": "nu_sweep_stage_c_part2", "registration": "issue #116",
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
               OUT_DIR / "stage_c_part2_results.pt")

    # m256's full-width trials live in the shared checkpoint directory from
    # Stage B; load them so both crossings can be stated at full width.
    all_levels = {}
    for ckpt in sorted(sa.CKPT_DIR.glob("*.pt")):
        r = torch.load(ckpt, map_location="cpu", weights_only=True)
        all_levels.setdefault(r["level"], {})[r["pid"]] = r
    shown = ["m040", "m048", "m056", "m256", "m384"]
    stats = {lv: sa.level_stats(all_levels[lv])
             for lv in shown if all_levels.get(lv)}

    lines = [
        "# Nu-sweep Stage C Part 2: full-width brackets at the new crossings",
        "",
        "Registered before execution in issue #116 (Part 2 rule, inherited",
        "from Stage B). Raw data: `stage_c_part2_results.pt`; checkpoints",
        "shared with Stages A, B and C, so the 25-prompt Stage C trials at",
        "these levels are reused verbatim. Stage C's fine scan moved both",
        "crossings: the lower edge to (m040, m048), the upper to (m256,",
        "m384). m256 was already at full width from Stage B.",
        "",
        "## Per-level summary at full width",
        "",
        "| level | n | share in real five | locked (lag-1 gate) "
        "| median lock-in | distinct labels | no-lock smallest lags |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for lv in shown:
        if lv not in stats:
            continue
        s = stats[lv]
        med = (f"{s['median_lock_in']:.0f}"
               if s["median_lock_in"] is not None else "n/a")
        nolock = (", ".join(
            f"lag {k}: {v}" for k, v in sorted(s["nolock_lags"].items()))
            if s["nolock_lags"] else "all locked")
        lines.append(
            f"| {lv} | {s['n']} | "
            f"{s['share_in_five']:.0%} ({s['n_in_five']}/{s['n']}) | "
            f"{s['locked']}/{s['n']} | {med} | {len(s['labels'])} | "
            f"{nolock} |")
    lines += ["", "## Basin table per level", ""]
    for lv in shown:
        if lv not in stats:
            continue
        top = sorted(stats[lv]["labels"].items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(f"- **{lv}**: "
                     + ", ".join(f"`{t}` {c}" for t, c in top[:10])
                     + (f", plus {len(top) - 10} more labels"
                        if len(top) > 10 else ""))
    lines += ["", "## Crossing confirmation at full width", ""]
    for a, b in CROSSING_PAIRS:
        if a not in stats or b not in stats:
            lines.append(f"- {a} to {b}: incomplete, not stated.")
            continue
        sa_, sb = stats[a], stats[b]
        crossed = ((sa_["share_in_five"] > sa.BAND_THRESHOLD)
                   != (sb["share_in_five"] > sa.BAND_THRESHOLD))
        lines.append(
            f"- {a} to {b}: {sa_['share_in_five']:.0%} to "
            f"{sb['share_in_five']:.0%} at full width, "
            f"{'crossing CONFIRMED' if crossed else 'crossing NOT confirmed'} "
            f"against the {sa.BAND_THRESHOLD:.0%} rule.")
    lines += [
        "",
        "## Reading",
        "",
        "Every number above is regenerated by re-running this script;",
        "nothing is hand-computed. Interpretation lands in issue #116 and",
        "the findings record, not here. Periodic-trial labels carry run",
        "17's single-phase readout provenance; no per-trial phase audit",
        "exists for them.",
    ]
    with open(OUT_DIR / "stage_c_part2_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'stage_c_part2_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        write_report()
    elif args.worker is not None:
        sa.run_worker(args.worker, args.num_workers)
    else:
        ap.print_help()
