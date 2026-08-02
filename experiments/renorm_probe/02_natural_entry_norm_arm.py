"""Natural entry-norm arm (issue #112, probe 2).

The historical loop pins every iterate to the first pass's EXIT norm, which
is far larger than what layer 0 naturally receives, so from iteration 1 the
model runs outside the input range it was trained on. The engine has carried
the alternative since the layer-window controls: renorm="natural_i" pins to
the natural size of layer 0's own input for that prompt. This probe runs the
registered gate protocol under that pin for the first 10 prompts of the
library, in library order, and compares terminal outcomes against the
committed record.

What would count as what (registered in #112 before execution): same basins
under the natural pin means basin identity is robust across at least two pin
sizes; a different landscape means the published five-basin structure is a
property of the out-of-range regime specifically, and basin-identity claims
inherit a scale caveat pending the nu-sweep.

Run from the repo root:

    python3 experiments/renorm_probe/02_natural_entry_norm_arm.py
    python3 experiments/renorm_probe/02_natural_entry_norm_arm.py --report-only

Outputs (in experiments/renorm_probe/output/):
    natural_arm_results.pt   per-prompt engine results with metrics
    natural_arm_report.md    every headline number, regenerated from the data
"""

import argparse
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from atr_engine import run_atr_gated  # noqa: E402
import prompt_library  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
PATHWAYS = (REPO_ROOT / "experiments" / "gpt2_small" / "output" /
            "dissolution_pathways.md")

LAYER_START, LAYER_END = 0, 11
MAX_ITER, THRESHOLD, PATIENCE = 1000, 0.999, 3
CHECK_EVERY, CHECK_START = 10, 100
N_PROMPTS = 10
REAL_FIVE = {"prolet", "Divine", "till", "Anarch", "solidarity"}


def smallest_passing_lag(lag_table, threshold=THRESHOLD):
    """Return the smallest lag whose terminal cosine clears the gate threshold, or None."""
    if not lag_table:
        return None
    for lag in sorted(lag_table):
        if lag_table[lag] > threshold:
            return lag
    return None


def historical_at_100(pids):
    """Terminal tokens at iteration 100 for the given prompts, parsed from the
    committed dissolution pathways table (the historical exit-norm record).
    The per-prompt lock-in map was never committed, so iteration 100 is the
    only per-prompt historical readout available for comparison."""
    text = PATHWAYS.read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith("|")]
    header = None
    last_data = {}
    for ln in rows:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and cells[0] == "Iter":
            header = cells
            continue
        if header and cells and re.fullmatch(r"\*\*\d+\*\*", cells[0]):
            it = int(cells[0].strip("*"))
            for name, val in zip(header[1:], cells[1:]):
                if name in pids:
                    last_data[name] = (it, val.strip("`"))
    return {pid: tok for pid, (it, tok) in last_data.items() if it == 100}


def run():
    """Run the ten natural entry-norm trials and write the report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2")
    pids = list(prompt_library.PROMPT_LIBRARY)[:N_PROMPTS]
    config = {
        "experiment": "natural_entry_norm_arm",
        "registration": "issue #112, probe 2",
        "prompts": pids, "layers": f"{LAYER_START}->{LAYER_END}",
        "renorm": "natural_i",
        "gate": {"max_iter": MAX_ITER, "threshold": THRESHOLD,
                 "patience": PATIENCE, "check_every": CHECK_EVERY,
                 "check_start": CHECK_START, "gate_lag": 1},
        "torch": str(torch.__version__),
    }
    results = {}
    t0 = time.time()
    for idx, pid in enumerate(pids):
        r = run_atr_gated(
            model, prompt_library.PROMPT_LIBRARY[pid], LAYER_START, LAYER_END,
            max_iter=MAX_ITER, threshold=THRESHOLD, patience=PATIENCE,
            check_every=CHECK_EVERY, check_start=CHECK_START,
            renorm="natural_i", capture_terminal=True, record_metrics=True)
        results[pid] = r
        gate = f"locked@{r['lock_in_iter']}" if r["converged"] else "no-lock"
        print(f"[{idx + 1}/{len(pids)}] {pid}: target {r['target_norm']:.0f} "
              f"(exit seed {r['seed_norm_at_j']:.0f}), {r['terminal_token']!r} "
              f"{gate} ({time.time() - t0:.0f}s)")
        torch.save({"config": config, "results": results},
                   OUT_DIR / "natural_arm_results.pt")
    write_report(config, results)


def write_report(config, results):
    """Regenerate natural_arm_report.md from the per-trial results; every published number originates here."""
    hist = historical_at_100(list(results))
    lines = [
        "# Natural entry-norm arm: results",
        "",
        "Registered before execution in issue #112 (probe 2). Raw data:",
        "`natural_arm_results.pt`. The loop ran the registered gate protocol",
        'with `renorm="natural_i"`: each iterate is pinned to the natural',
        "size of layer 0's own input for that prompt, instead of the",
        "historical pin at the first pass's exit size.",
        "",
        "## Per-prompt outcomes",
        "",
        "The ratio column states how far above the model's natural entry",
        "scale the historical runs injected: exit-norm pin divided by",
        "natural entry pin. The final column is the same prompt's decoded",
        "token at iteration 100 in the committed historical record (the",
        "per-prompt lock-in map was never committed).",
        "",
        "| prompt | natural pin | historical exit pin | ratio | gate "
        "| smallest passing lag | terminal token | historical at 100 |",
        "|:--|--:|--:|--:|:--|--:|:--|:--|",
    ]
    tokens = {}
    for pid, r in results.items():
        lag = smallest_passing_lag(r.get("lag_scan") or {})
        gate = (f"locked at {r['lock_in_iter']}" if r["converged"]
                else f"no lock by {r['n_iters']}")
        tok = r["terminal_token"].strip()
        tokens[tok] = tokens.get(tok, 0) + 1
        ratio = (r["seed_norm_at_j"] / r["target_norm"]
                 if r["target_norm"] else float("nan"))
        lines.append(
            f"| {pid} | {r['target_norm']:.0f} | {r['seed_norm_at_j']:.0f} "
            f"| {ratio:.1f}x | {gate} | {lag if lag is not None else 'none'} "
            f"| `{tok}` | `{hist.get(pid, '?')}` |")
    overlap = sorted(set(tokens) & REAL_FIVE)
    n_locked = sum(1 for r in results.values() if r["converged"])
    lines += [
        "",
        "## Summary",
        "",
        f"- Locked in under the lag-1 gate: {n_locked}/{len(results)}.",
        f"- Distinct terminal tokens: {len(tokens)}: "
        + ", ".join(f"`{t}` ({c})" for t, c in
                    sorted(tokens.items(), key=lambda kv: -kv[1])) + ".",
        f"- Overlap with the five language-arm basins "
        f"{sorted(REAL_FIVE)}: {overlap if overlap else 'none'}.",
        "",
        "## Reading",
        "",
        "Every number above is regenerated by re-running this script;",
        "nothing is hand-computed. The registered disposition rule (issue",
        "#112): same basins under this pin means basin identity is robust",
        "across two pin sizes; a different landscape means the published",
        "five-basin structure belongs to the out-of-range injection regime",
        "specifically. Interpretation lands in the issue and the findings",
        "record, not here.",
    ]
    with open(OUT_DIR / "natural_arm_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'natural_arm_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the report from saved results, no model run")
    args = ap.parse_args()
    if args.report_only:
        saved = torch.load(OUT_DIR / "natural_arm_results.pt",
                           map_location="cpu", weights_only=True)
        write_report(saved["config"], saved["results"])
    else:
        run()
