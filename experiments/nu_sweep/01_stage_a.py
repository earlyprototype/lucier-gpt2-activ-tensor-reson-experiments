"""Nu-sweep Stage A: coarse map of the injection-scale band (issue #113).

The engine pins every iterate's Frobenius norm (nu in the record) to a fixed
target. The groundwork (issue #112) fixed three anchor points: the published
five-basin landscape is present at the historical pin (each prompt's first
pass exit norm, about 70x the model's natural entry scale), absent at the
natural entry scale, and absent with no pin at all. This stage runs the first
25 prompts of the library, in library order, at ten pin levels: multipliers
1, 2, 4, 8, 16, 32, 64, 128, 256 of each prompt's own natural entry norm
(layer 0's resid_pre size for that prompt), plus the exact historical pin
(renorm="seed_j") as the continuity anchor with the committed record.

Pre-stated per-level statistic (issue #113): the share of trials whose
terminal label at their smallest passing lag (the F15 rule) is one of the
five language basins. The band is the contiguous set of levels where that
share exceeds 50 percent; Stage B brackets each crossing at full sweep width.

Contract check (runs before any sweep trial, per the registration): a numeric
pin equal to a run's own seed norm must reproduce renorm="seed_j"
bit-identically: same terminal token, same lock-in, and every per-iteration
metric float equal. The sweep refuses to start if this fails.

Run from the repo root (workers split the level x prompt grid; each is
single-threaded, so run up to one per core):

    python3 experiments/nu_sweep/01_stage_a.py --contract
    python3 experiments/nu_sweep/01_stage_a.py --worker 0 --num-workers 4
    ...
    python3 experiments/nu_sweep/01_stage_a.py --report-only

Outputs (in experiments/nu_sweep/output/):
    contract_check.json      the pre-run bit-identity contract result
    checkpoints/<level>_<pid>.pt   one file per completed trial (resume unit)
    stage_a_results.pt       combined archive (assembled by --report-only)
    stage_a_report.md        every headline number, regenerated from the data

If huggingface.co is unreachable, set ATR_GPT2_LOCAL to a directory
containing the standard gpt2 files (config.json, pytorch_model.bin,
vocab.json, merges.txt) and the script loads offline.
"""

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Single-thread BLAS: multi-threaded thrashing costs 5x per forward here,
# and the worker model parallelises across processes instead.
torch.set_num_threads(1)

import prompt_library  # noqa: E402
from atr_engine import run_atr_gated  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
CKPT_DIR = OUT_DIR / "checkpoints"

LAYER_START, LAYER_END = 0, 11
MAX_ITER, THRESHOLD, PATIENCE = 1000, 0.999, 3
CHECK_EVERY, CHECK_START = 10, 100
N_PROMPTS = 25
MULTIPLIERS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LEVELS = [f"m{m:03d}" for m in MULTIPLIERS] + ["historical"]
REAL_FIVE = {"prolet", "Divine", "till", "Anarch", "solidarity"}
BAND_THRESHOLD = 0.50


def load_model():
    """Load GPT-2 Small on CPU, offline via ATR_GPT2_LOCAL when set."""
    local = os.environ.get("ATR_GPT2_LOCAL")
    if local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
        hf_model = GPT2LMHeadModel.from_pretrained(local)
        tokenizer = GPT2TokenizerFast.from_pretrained(local)
        import transformer_lens.loading_from_pretrained as lfp
        _cfg = GPT2Config.from_pretrained(local)

        class _Shim:
            @staticmethod
            def from_pretrained(name, *a, **k):
                return _cfg

        lfp.AutoConfig = _Shim
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(
            "gpt2", hf_model=hf_model, tokenizer=tokenizer, device="cpu")
    else:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()
    return model


def natural_norm(model, prompt):
    """The natural size of layer 0's own input for this prompt: the Frobenius
    norm of blocks.0.hook_resid_pre on an un-hooked forward pass, which is the
    same quantity renorm="natural_i" pins to inside the engine."""
    name = f"blocks.{LAYER_START}.hook_resid_pre"
    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt, names_filter=lambda n: n == name)
    return cache[name][0].norm().item()


def gate_kwargs():
    """The registered gate settings (issue #113), one source of truth."""
    return dict(max_iter=MAX_ITER, threshold=THRESHOLD, patience=PATIENCE,
                check_every=CHECK_EVERY, check_start=CHECK_START,
                capture_terminal=True, record_metrics=True)


def run_contract(model):
    """Numeric-pin bit-identity contract (issue #113): renorm=<seed norm as a
    number> must reproduce renorm="seed_j" exactly. Writes contract_check.json
    and returns True on pass."""
    pid = next(iter(prompt_library.PROMPT_LIBRARY))
    prompt = prompt_library.PROMPT_LIBRARY[pid]
    kw = gate_kwargs()
    kw["max_iter"] = 150  # enough to include gate checks; both runs identical
    a = run_atr_gated(model, prompt, LAYER_START, LAYER_END,
                      renorm="seed_j", **kw)
    b = run_atr_gated(model, prompt, LAYER_START, LAYER_END,
                      renorm=a["target_norm"], **kw)
    checks = {
        "terminal_token": a["terminal_token"] == b["terminal_token"],
        "terminal_token_id": a["terminal_token_id"] == b["terminal_token_id"],
        "lock_in_iter": a["lock_in_iter"] == b["lock_in_iter"],
        "n_iters": a["n_iters"] == b["n_iters"],
        "target_norm": a["target_norm"] == b["target_norm"],
        "metrics_length": len(a["metrics"]) == len(b["metrics"]),
        "metrics_bit_identical": all(
            ma["position_similarity_f64"] == mb["position_similarity_f64"]
            and ma["tensor_norm"] == mb["tensor_norm"]
            and ma["cos_sim_mean_lag1"] == mb["cos_sim_mean_lag1"]
            # strict=False by intent: a length mismatch must record a False
            # check above, not raise here before the result file is written.
            for ma, mb in zip(a["metrics"], b["metrics"], strict=False)),
    }
    result = {
        "prompt": pid,
        "iterations_compared": len(a["metrics"]),
        "checks": checks,
        "passed": all(checks.values()),
        "torch": str(torch.__version__),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "contract_check.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[contract] passed={result['passed']} "
          f"({result['iterations_compared']} iterations compared)")
    return result["passed"]


def contract_passed():
    """True if a committed-or-fresh contract_check.json records a pass."""
    p = OUT_DIR / "contract_check.json"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("passed") is True


def grid():
    """The full (level, pid) grid in deterministic order."""
    pids = list(prompt_library.PROMPT_LIBRARY)[:N_PROMPTS]
    return [(level, pid) for level in LEVELS for pid in pids]


def run_worker(worker, num_workers):
    """Run this worker's slice of the grid, one checkpoint file per trial."""
    if not contract_passed():
        sys.exit("[worker] contract_check.json missing or failed; run "
                 "--contract first (issue #113 requires it before any "
                 "sweep trial)")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    naturals = {}
    todo = [(lv, pid) for i, (lv, pid) in enumerate(grid())
            if i % num_workers == worker]
    t0 = time.time()
    for n, (level, pid) in enumerate(todo):
        ckpt = CKPT_DIR / f"{level}_{pid}.pt"
        if ckpt.exists():
            continue
        prompt = prompt_library.PROMPT_LIBRARY[pid]
        if pid not in naturals:
            naturals[pid] = natural_norm(model, prompt)
        if level == "historical":
            renorm = "seed_j"
            multiplier = None
        else:
            multiplier = int(level[1:])
            renorm = multiplier * naturals[pid]
        r = run_atr_gated(model, prompt, LAYER_START, LAYER_END,
                          renorm=renorm, **gate_kwargs())
        r.update({"pid": pid, "level": level, "multiplier": multiplier,
                  "natural_norm": naturals[pid]})
        tmp = ckpt.with_suffix(".tmp")
        torch.save(r, tmp)
        tmp.rename(ckpt)
        gate = (f"locked@{r['lock_in_iter']}" if r["converged"]
                else f"no-lock/{r['n_iters']}")
        print(f"[w{worker}] {n + 1}/{len(todo)} {level} {pid}: "
              f"pin {r['target_norm']:.0f} {r['terminal_token']!r} {gate} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[w{worker}] slice complete ({time.time() - t0:.0f}s)")


def smallest_passing_lag(r):
    """The F15 rule: lag 1 if the gate locked, else the smallest lag in the
    terminal lag table clearing the gate threshold, else None."""
    if r["converged"]:
        return 1
    table = r.get("lag_scan") or {}
    for lag in sorted(table):
        if table[lag] > THRESHOLD:
            return lag
    return None


def collect():
    """Load every checkpoint into {level: {pid: result}}, reporting gaps."""
    results = {level: {} for level in LEVELS}
    missing = []
    for level, pid in grid():
        ckpt = CKPT_DIR / f"{level}_{pid}.pt"
        if ckpt.exists():
            results[level][pid] = torch.load(
                ckpt, map_location="cpu", weights_only=True)
        else:
            missing.append(f"{level}_{pid}")
    return results, missing


def write_report():
    """Assemble stage_a_results.pt and regenerate stage_a_report.md; every
    published number originates here."""
    results, missing = collect()
    if missing:
        print(f"[report] WARNING: {len(missing)} trials missing; the report "
              f"marks itself partial. First few: {missing[:6]}")
    config = {
        "experiment": "nu_sweep_stage_a", "registration": "issue #113",
        "prompts": list(prompt_library.PROMPT_LIBRARY)[:N_PROMPTS],
        "layers": f"{LAYER_START}->{LAYER_END}",
        "levels": LEVELS, "multipliers": MULTIPLIERS,
        "gate": {"max_iter": MAX_ITER, "threshold": THRESHOLD,
                 "patience": PATIENCE, "check_every": CHECK_EVERY,
                 "check_start": CHECK_START, "gate_lag": 1},
        "band_threshold": BAND_THRESHOLD,
        "torch": str(torch.__version__),
    }
    torch.save({"config": config, "results": results},
               OUT_DIR / "stage_a_results.pt")

    def level_sort_key(level):
        rows = results[level]
        pins = [r["target_norm"] for r in rows.values()]
        return statistics.mean(pins) if pins else float("inf")

    ordered = sorted((lv for lv in LEVELS if results[lv]), key=level_sort_key)
    stats = {}
    for level in ordered:
        rows = results[level]
        labels = {}
        in_five = 0
        locked = []
        gains = []
        for r in rows.values():
            lag = smallest_passing_lag(r)
            tok = r["terminal_token"].strip()
            labels[tok] = labels.get(tok, 0) + 1
            if lag is not None and tok in REAL_FIVE:
                in_five += 1
            if r["converged"]:
                locked.append(r["lock_in_iter"])
            if r["metrics"] and r["target_norm"]:
                gains.append(r["metrics"][0]["tensor_norm"] / r["target_norm"])
        stats[level] = {
            "n": len(rows),
            "mean_pin": statistics.mean(
                r["target_norm"] for r in rows.values()),
            "share_in_five": in_five / len(rows),
            "n_in_five": in_five,
            "locked": len(locked),
            "median_lock_in": statistics.median(locked) if locked else None,
            "labels": labels,
            "mean_gain": statistics.mean(gains) if gains else None,
        }
    band = [lv for lv in ordered
            if stats[lv]["share_in_five"] > BAND_THRESHOLD]
    crossings = [
        (a, b) for a, b in itertools.pairwise(ordered)
        if (stats[a]["share_in_five"] > BAND_THRESHOLD)
        != (stats[b]["share_in_five"] > BAND_THRESHOLD)]

    lines = [
        "# Nu-sweep Stage A: results",
        "",
        "Registered before execution in issue #113. Raw data:",
        "`stage_a_results.pt` (per-trial checkpoints in `checkpoints/`).",
        "Ten pin levels: multipliers of each prompt's natural entry norm,",
        "plus the exact historical pin (`renorm=\"seed_j\"`) as the",
        "continuity anchor. First 25 prompts of the library in library",
        "order; registered gate settings; F15 classification at each",
        "trial's smallest passing lag.",
        "",
    ]
    if missing:
        lines += [f"**PARTIAL: {len(missing)} of {len(grid())} trials "
                  "missing; numbers below cover completed trials only.**", ""]
    lines += [
        "## Per-level summary (levels ordered by mean pin size)",
        "",
        "| level | mean pin | share in real five | locked (lag 1) "
        "| median lock-in | distinct labels | mean single-pass gain |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for level in ordered:
        s = stats[level]
        med = (f"{s['median_lock_in']:.0f}"
               if s["median_lock_in"] is not None else "n/a")
        gain = f"{s['mean_gain']:.2f}" if s["mean_gain"] else "n/a"
        lines.append(
            f"| {level} | {s['mean_pin']:.0f} | "
            f"{s['share_in_five']:.0%} ({s['n_in_five']}"
            f"/{s['n']}) | {s['locked']}/{s['n']} | {med} | "
            f"{len(s['labels'])} | {gain} |")
    lines += ["", "## Basin table per level", ""]
    for level in ordered:
        s = stats[level]
        top = sorted(s["labels"].items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(f"- **{level}** (mean pin {s['mean_pin']:.0f}): "
                     + ", ".join(f"`{t}` {c}" for t, c in top))
    lines += [
        "",
        "## Band determination (pre-stated rule)",
        "",
        f"- Band statistic: share of trials whose terminal label at their "
        f"smallest passing lag is one of {sorted(REAL_FIVE)}.",
        f"- Levels above the {BAND_THRESHOLD:.0%} threshold, in pin order: "
        f"{band if band else 'none'}.",
        f"- Threshold crossings between adjacent levels (Stage B brackets "
        f"each at full sweep width): "
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
    with open(OUT_DIR / "stage_a_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'stage_a_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", action="store_true",
                    help="run the pre-sweep bit-identity contract check")
    ap.add_argument("--worker", type=int, default=None,
                    help="worker index into the level x prompt grid")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--report-only", action="store_true",
                    help="assemble results and regenerate the report")
    args = ap.parse_args()
    if args.contract:
        ok = run_contract(load_model())
        sys.exit(0 if ok else 1)
    elif args.report_only:
        write_report()
    elif args.worker is not None:
        run_worker(args.worker, args.num_workers)
    else:
        ap.print_help()
