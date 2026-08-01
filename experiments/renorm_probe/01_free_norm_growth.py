"""Free-norm growth diagnostic (issue #112, probe 1).

What does the loop do with the tensor's size when nothing pins it? Every
registered run rescales each iterate back to a fixed Frobenius norm (the
first pass's exit norm, nu). The record infers that this pin is necessary
from the measured single-pass gain (about 3.5x at the language scale, about
10x at the old mis-calibrated scale); this probe measures the free-running
behaviour directly instead of inferring it.

Pre-stated expectation (registered in #112 before execution, falsifiable):
the two gain numbers suggest growth closer to additive than multiplicative.
Each block reads a LayerNorm-normalised copy of its input (scale-blind) but
writes its output onto the running residual stream unnormalised, so each
pass should add a roughly constant-sized increment. If that holds, the
free norm grows roughly linearly and the per-pass gain falls toward 1 as
the tensor grows. If growth is instead multiplicative, the norm explodes
within about seventy passes.

Protocol: first 5 prompts of the prompt library in library order (no
selection), 200 iterations each, renorm="none" (engine mode added for this
probe under the one-engine standing rule), convergence gate disabled (its
first check is placed beyond the iteration ceiling), per-iteration metrics
recorded (tensor norm, float64 position similarity, lag-1 cosine of the
mean vector), terminal capture with the full lag table.

Run from the repo root:

    python3 experiments/renorm_probe/01_free_norm_growth.py
    python3 experiments/renorm_probe/01_free_norm_growth.py --report-only

Outputs (in experiments/renorm_probe/output/):
    free_norm_results.pt   per-prompt engine results with metrics
    free_norm_report.md    every headline number, regenerated from the data
"""

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from atr_engine import run_atr_gated  # noqa: E402
import prompt_library  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"

LAYER_START, LAYER_END = 0, 11
MAX_ITER = 200
N_PROMPTS = 5
GATE_OFF = 10 ** 9  # first gate check placed beyond the ceiling: never fires
THRESHOLD = 0.999


def smallest_passing_lag(lag_table, threshold=THRESHOLD):
    """Return the smallest lag whose terminal cosine clears the gate threshold, or None."""
    if not lag_table:
        return None
    for lag in sorted(lag_table):
        if lag_table[lag] > threshold:
            return lag
    return None


def run():
    """Run the five free-norm trials and write the report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2")
    pids = list(prompt_library.PROMPT_LIBRARY)[:N_PROMPTS]
    config = {
        "experiment": "free_norm_growth",
        "registration": "issue #112, probe 1",
        "prompts": pids, "layers": f"{LAYER_START}->{LAYER_END}",
        "max_iter": MAX_ITER, "renorm": "none", "gate": "disabled",
        "torch": str(torch.__version__),
    }
    results = {}
    t0 = time.time()
    for idx, pid in enumerate(pids):
        r = run_atr_gated(
            model, prompt_library.PROMPT_LIBRARY[pid], LAYER_START, LAYER_END,
            max_iter=MAX_ITER, check_start=GATE_OFF, renorm="none",
            capture_terminal=True, record_metrics=True)
        results[pid] = r
        print(f"[{idx + 1}/{len(pids)}] {pid}: norm {r['seed_norm_at_j']:.0f} -> "
              f"{r['metrics'][-1]['tensor_norm']:.0f} over {MAX_ITER} iters, "
              f"terminal {r['terminal_token']!r} ({time.time() - t0:.0f}s)")
        torch.save({"config": config, "results": results},
                   OUT_DIR / "free_norm_results.pt")
    write_report(config, results)


def write_report(config, results):
    """Regenerate free_norm_report.md from the per-trial results; every published number originates here."""
    lines = [
        "# Free-norm growth diagnostic: results",
        "",
        "Registered before execution in issue #112 (probe 1). Raw data:",
        "`free_norm_results.pt`. The loop ran with no rescale at all",
        '(`renorm="none"`), gate disabled, 200 iterations per prompt.',
        "",
        "## Norm trajectory per prompt (Frobenius norm of the full tensor)",
        "",
        "| prompt | iter 0 | iter 1 | iter 10 | iter 50 | iter 100 | iter 200 |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    summary = {}
    for pid, r in results.items():
        norms = [m["tensor_norm"] for m in r["metrics"]]
        nu0 = r["seed_norm_at_j"]
        full = [nu0] + norms  # index = iteration
        def at(i):
            return full[i] if i < len(full) else full[-1]
        lines.append(
            f"| {pid} | {at(0):.0f} | {at(1):.0f} | {at(10):.0f} "
            f"| {at(50):.0f} | {at(100):.0f} | {at(200):.0f} |")
        diffs = [full[i] - full[i - 1] for i in range(1, len(full))]
        ratios = [full[i] / full[i - 1] for i in range(1, len(full)) if full[i - 1] > 0]
        lag = smallest_passing_lag(r.get("lag_scan") or {})
        summary[pid] = {
            "early_inc": sum(diffs[:10]) / 10, "late_inc": sum(diffs[-10:]) / 10,
            "early_ratio": sum(ratios[:10]) / 10, "late_ratio": sum(ratios[-10:]) / 10,
            "final_pos_sim": r["metrics"][-1]["position_similarity_f64"],
            "final_lag1": r["metrics"][-1]["cos_sim_mean_lag1"],
            "smallest_lag": lag, "terminal": r["terminal_token"].strip(),
        }
    lines += [
        "",
        "## Growth character per prompt",
        "",
        "Additive growth means a roughly constant increment per pass and a",
        "per-pass ratio falling toward 1; multiplicative growth means a",
        "roughly constant ratio above 1 and explosion within about seventy",
        "passes.",
        "",
        "| prompt | mean increment, passes 1-10 | mean increment, last 10 "
        "| mean ratio, passes 1-10 | mean ratio, last 10 |",
        "|:--|--:|--:|--:|--:|",
    ]
    for pid, s in summary.items():
        lines.append(
            f"| {pid} | {s['early_inc']:+.0f} | {s['late_inc']:+.0f} "
            f"| {s['early_ratio']:.3f} | {s['late_ratio']:.3f} |")
    lines += [
        "",
        "## Does the direction still settle without the pin?",
        "",
        "| prompt | final position similarity (float64) | final lag-1 cosine "
        "| smallest passing lag at end | terminal token |",
        "|:--|--:|--:|--:|:--|",
    ]
    for pid, s in summary.items():
        lag = s["smallest_lag"] if s["smallest_lag"] is not None else "none"
        lines.append(
            f"| {pid} | {s['final_pos_sim']:.6f} | {s['final_lag1']:.6f} "
            f"| {lag} | `{s['terminal']}` |")
    late_ratios = [s["late_ratio"] for s in summary.values()]
    lines += [
        "",
        "## Reading",
        "",
        "Every number above is regenerated by re-running this script; nothing",
        "is hand-computed. The registered expectation was additive growth",
        f"(issue #112). Observed: the per-pass ratio over the last ten passes",
        f"is {min(late_ratios):.3f} to {max(late_ratios):.3f} across the five",
        "prompts, against 3.5 measured for a single pass at the registered",
        "injection scale. Interpretation is deferred to the issue and the",
        "findings record; this file states the measurements.",
    ]
    with open(OUT_DIR / "free_norm_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'free_norm_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the report from saved results, no model run")
    args = ap.parse_args()
    if args.report_only:
        saved = torch.load(OUT_DIR / "free_norm_results.pt",
                           map_location="cpu", weights_only=True)
        write_report(saved["config"], saved["results"])
    else:
        run()
