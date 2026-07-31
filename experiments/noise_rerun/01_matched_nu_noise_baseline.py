"""Matched-nu, convergence-gated random-noise baseline.

The corrected form of the Stage 1 null model (EXP_009d3), repairing the two
confounds recorded in FINDINGS.md caveat 18 / issue #97:

1. **Calibration.** The original run calibrated the noise tensors' FROBENIUS
   norm to the language runs' MEAN-VECTOR norm (a per-position-scale statistic,
   ~397), so the noise arm ran at ~1/sqrt(T) of the language arms' injection
   scale, in a ~10x gain regime against their ~3.5x. Here each trial is
   **pair-matched**: trial k takes prompt k's exact sequence length and exact
   iteration-0 Frobenius norm, measured fresh in Stage A. No proxy, no
   distribution fitting.

2. **Convergence state.** The original 18-basin count was read at iteration
   100 with the run's own report recording cosine convergence NO (0.9256),
   while the 5-basin language count is at gated lock-in. Here every trial runs
   under the engine's convergence gate (lag-1, threshold 0.999, patience 3,
   ceiling 1000 iterations), terminal states carry the full lag table
   (F15's protocol: classify each state at its smallest passing lag, so
   period-2 attractors are not misread as non-convergence), and basins are
   counted **at lock-in only**.

Archive spec (ALIGNMENT_REVIEW.md section 5, standing rule 2): per-iteration
position_similarity in float64, tensor_norm and lag-1 cosine per iteration,
seq_len and seed norm per trial, the full terminal tensor per trial, uniform
per-iteration cadence (no snapshot-gap mixing). Engine path only (standing
rule 3): the loop is `atr_engine.run_atr_gated(seed_tensor=...)`, verified
bit-identical to the prompt path when seeded with a prompt's own iteration-0
tensor; there is no inline loop copy in this file.

Run from the repo root:

    python3 experiments/noise_rerun/01_matched_nu_noise_baseline.py            # full 125
    python3 experiments/noise_rerun/01_matched_nu_noise_baseline.py --smoke    # 2 trials, 150-iter ceiling

Outputs (in experiments/noise_rerun/output/):
    calibration.json   per-prompt seq_len + Frobenius norm, and the run config
    results.pt         per-trial engine results, metrics, terminal tensors
    report.md          every headline number, regenerated from the data
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from atr_engine import run_atr_gated  # noqa: E402
import prompt_library  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 42
LAYER_START, LAYER_END = 0, 11
MAX_ITER, THRESHOLD, PATIENCE = 1000, 0.999, 3
CHECK_EVERY, CHECK_START = 10, 100
REAL_FIVE = {"prolet", "Divine", "till", "Anarch", "solidarity"}
# The original, mis-calibrated run's headline numbers, for the comparison table.
OLD_RUN = {"basins_at_iter100": 18, "dominant": "―", "dominant_share": 0.64}


def stage_a_calibration(model):
    """One un-hooked forward pass per prompt: seq_len + Frobenius norm at the read site."""
    read_site = f"blocks.{LAYER_END}.hook_resid_post"
    rows = []
    for pid, prompt in prompt_library.PROMPT_LIBRARY.items():
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, names_filter=lambda n: n == read_site)
        t = cache[read_site][0]
        rows.append({"prompt_id": pid, "seq_len": int(t.shape[0]),
                     "frobenius_norm": float(t.norm().item())})
    return rows


def stage_b_noise(calibration, n_trials):
    """Pair-matched Gaussian tensors: trial k <- prompt k's seq_len and norm."""
    gen = torch.Generator().manual_seed(SEED)
    trials = []
    for k, row in enumerate(calibration[:n_trials]):
        t = torch.randn(row["seq_len"], 768, generator=gen)
        t = t * (row["frobenius_norm"] / t.norm().item())
        trials.append({"trial_id": f"R{k + 1:03d}", "matched_to": row["prompt_id"],
                       "seq_len": row["seq_len"],
                       "target_frobenius": row["frobenius_norm"], "tensor": t})
    return trials


def smallest_passing_lag(lag_table, threshold=THRESHOLD):
    """Return the smallest lag whose terminal cosine clears the gate threshold, or None."""
    if not lag_table:
        return None
    for lag in sorted(lag_table):
        if lag_table[lag] > threshold:
            return lag
    return None


def run(n_trials, max_iter):
    """Calibrate against the prompt library, run every noise trial, write the report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2")

    print(f"[stage A] calibrating from {len(prompt_library.PROMPT_LIBRARY)} prompts")
    calibration = stage_a_calibration(model)
    norms = [r["frobenius_norm"] for r in calibration]
    config = {
        "experiment": "matched_nu_noise_baseline",
        "repairs": "FINDINGS caveat 18 / issue #97",
        "seed": SEED, "n_trials": n_trials, "layers": f"{LAYER_START}->{LAYER_END}",
        "gate": {"max_iter": max_iter, "threshold": THRESHOLD, "patience": PATIENCE,
                 "check_every": CHECK_EVERY, "check_start": CHECK_START, "gate_lag": 1},
        "calibration_mode": "pair-matched (trial k <- prompt k's seq_len and Frobenius norm)",
        "frobenius_mean": sum(norms) / len(norms),
        "frobenius_min": min(norms), "frobenius_max": max(norms),
        "torch": torch.__version__,
    }
    with open(OUT_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "per_prompt": calibration}, f, indent=2)
    print(f"[stage A] Frobenius mean {config['frobenius_mean']:.1f} "
          f"(range {config['frobenius_min']:.1f}-{config['frobenius_max']:.1f}); "
          f"the old run injected at 397.18")

    trials = stage_b_noise(calibration, n_trials)
    results = {}
    t0 = time.time()
    for idx, tr in enumerate(trials):
        r = run_atr_gated(
            model, None, LAYER_START, LAYER_END, max_iter=max_iter,
            threshold=THRESHOLD, patience=PATIENCE, check_every=CHECK_EVERY,
            check_start=min(CHECK_START, max_iter), capture_terminal=True,
            seed_tensor=tr["tensor"], record_metrics=True)
        results[tr["trial_id"]] = {
            "matched_to": tr["matched_to"], "seq_len": tr["seq_len"],
            "target_frobenius": tr["target_frobenius"], "result": r,
        }
        gate = "locked@%s" % r["lock_in_iter"] if r["converged"] else "no-lock"
        print(f"[{idx + 1}/{len(trials)}] {tr['trial_id']} <- {tr['matched_to']}: "
              f"{r['terminal_token']!r} {gate} ({time.time() - t0:.0f}s)")
        if (idx + 1) % 10 == 0 or idx + 1 == len(trials):
            torch.save({"config": config, "results": results},
                       OUT_DIR / "results.pt")
    write_report(config, results)


def write_report(config, results):
    """Regenerate report.md from the per-trial results; every published number originates here."""
    conv = {k: v for k, v in results.items() if v["result"]["converged"]}
    unconv = {k: v for k, v in results.items() if not v["result"]["converged"]}
    periodic = {}
    for k, v in unconv.items():
        lag = smallest_passing_lag(v["result"].get("lag_scan") or {})
        if lag and lag > 1:
            periodic[k] = lag
    basin_counts = {}
    for v in conv.values():
        tok = v["result"]["terminal_token"].strip()
        basin_counts[tok] = basin_counts.get(tok, 0) + 1
    basins = sorted(basin_counts.items(), key=lambda kv: -kv[1])
    overlap = sorted({b for b, _ in basins} & REAL_FIVE)
    n = len(results)
    dom = basins[0] if basins else ("-", 0)
    lock_iters = sorted(v["result"]["lock_in_iter"] for v in conv.values())
    med_lock = lock_iters[len(lock_iters) // 2] if lock_iters else None

    # F15's classification rule is "smallest passing lag", so the periodic
    # trials are basins too, labeled by the readout of the captured terminal
    # iterate. The language arm labels its own period-2 basin (Divine) the
    # same single-phase way.
    periodic_counts = {}
    for k in periodic:
        tok = unconv[k]["result"]["terminal_token"].strip()
        periodic_counts[tok] = periodic_counts.get(tok, 0) + 1
    periodic_basins = sorted(periodic_counts.items(), key=lambda kv: -kv[1])
    combined_counts = dict(basin_counts)
    for tok, c in periodic_counts.items():
        combined_counts[tok] = combined_counts.get(tok, 0) + c
    combined = sorted(combined_counts.items(), key=lambda kv: -kv[1])
    n_passing = len(conv) + len(periodic)
    combined_overlap = sorted(set(combined_counts) & REAL_FIVE)
    in_five = sum(c for tok, c in combined if tok in REAL_FIVE)
    unclassified = len(unconv) - len(periodic)

    lines = [
        "# Matched-nu noise baseline: results",
        "",
        f"Corrected re-run of the Stage 1 null model (repairs FINDINGS caveat 18 / issue #97).",
        f"Config and per-prompt calibration: `calibration.json`. Raw data: `results.pt`.",
        "",
        "## Headline",
        "",
        f"- Trials: {n}, pair-matched to the 125 real prompts' seq_len and Frobenius norm",
        f"  (injection at Frobenius mean {config['frobenius_mean']:.1f}; the old run used 397.18).",
        f"- **Converged at lock-in: {len(conv)}/{n}**"
        + (f" (median lock-in iteration {med_lock})" if med_lock else "") + ".",
        f"- **Distinct terminal basins at lock-in: {len(basin_counts)}** "
        f"(the old run reported 18, counted unconverged at iteration 100).",
        f"- Dominant basin: `{dom[0]}` at {dom[1] / max(len(conv), 1):.0%} of converged trials "
        f"(old run: `{OLD_RUN['dominant']}` at {OLD_RUN['dominant_share']:.0%}).",
        f"- Overlap with the real five {sorted(REAL_FIVE)}: "
        f"{overlap if overlap else 'none'}.",
        f"- Not locked by iteration {config['gate']['max_iter']}: {len(unconv)}"
        + (f", of which {len(periodic)} pass at a higher lag (periodic attractors): "
           f"{ {k: v for k, v in sorted(periodic.items())} }" if periodic else "") + ".",
        "",
        "## Basin table (converged trials only)",
        "",
        "| terminal token | trials | share of converged |",
        "|:--|--:|--:|",
    ]
    for tok, c in basins:
        lines.append(f"| `{tok}` | {c} | {c / max(len(conv), 1):.1%} |")
    if periodic_basins:
        lines += [
            "",
            "## Periodic trials (pass at lag 2), labeled by terminal readout",
            "",
            "| terminal token | trials | share of periodic |",
            "|:--|--:|--:|",
        ]
        for tok, c in periodic_basins:
            lines.append(f"| `{tok}` | {c} | {c / len(periodic):.1%} |")
        lines += [
            "",
            "Label provenance: each periodic trial's label is the argmax readout of",
            "the captured terminal iterate, which is one phase of the cycle. The",
            "language arm's `Divine` label has the same single-phase provenance,",
            "with run 8 (F2) additionally auditing that readout as stable across",
            "the cycle; no per-trial phase audit exists yet for these noise trials.",
        ]
    lines += [
        "",
        "## All trials at their smallest passing lag (the F15 classification rule)",
        "",
        f"Basis: {n_passing}/{n} trials pass at lag 1 or lag 2"
        + (f"; {unclassified} pass at no scanned lag and are excluded" if unclassified else "") + ".",
        "",
        "| terminal token | trials | share of passing |",
        "|:--|--:|--:|",
    ]
    for tok, c in combined:
        lines.append(f"| `{tok}` | {c} | {c / max(n_passing, 1):.1%} |")
    lines += [
        "",
        f"- Distinct labels over all passing trials: **{len(combined_counts)}**.",
        f"- Overlap with the real five {sorted(REAL_FIVE)}: "
        f"{combined_overlap if combined_overlap else 'none'}.",
        f"- Passing trials landing in the real five: **{in_five}/{n_passing}"
        f" ({in_five / max(n_passing, 1):.1%})**.",
        "",
        "## Reading",
        "",
        "Every number above is regenerated by re-running this script; nothing is",
        "hand-computed. Comparisons against the language arm's five basins are now",
        "at matched injection scale and matched convergence state, which the",
        "original run's comparison was not (either confound alone could have",
        "produced its 5-vs-18 gap). The per-iteration float64 position-similarity",
        "curves for every trial are in `results.pt` under `metrics`, closing the",
        "M2 archive gap for this arm.",
    ]
    with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 trials, 150-iteration ceiling")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate report.md from the saved results.pt, no model run")
    args = ap.parse_args()
    if args.report_only:
        saved = torch.load(OUT_DIR / "results.pt", map_location="cpu",
                           weights_only=False)
        write_report(saved["config"], saved["results"])
    else:
        n = 2 if args.smoke else (args.trials or len(prompt_library.PROMPT_LIBRARY))
        run(n, 150 if args.smoke else MAX_ITER)
