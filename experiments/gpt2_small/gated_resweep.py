"""Section 5 — GPT-2 Small convergence-gated re-sweep.

Control 1 showed gpt2-small only reached cos_sim_mean ~0.91 at iter 100, so the
published 5-basin table was read *before* convergence. This re-runs all 125
prompts with an early-stop gate (cos_sim_mean > 0.999 for 3 consecutive checks,
every 10 iters past 100), classifies each terminal basin *at lock-in*, and tests
whether the basin shares (especially `till`, 15.2%) are stable or stop-time
artefacts.

Does NOT touch the April stage1_results.pt — writes to output_gated/.
Checkpoints after every prompt (resume-safe): re-running skips completed prompts.

Env: GATED_LIMIT=N processes only the first N prompts (pilot). Default: all 125.
Run from repo root or this dir; paths resolve to repo root via atr_engine.py.
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Resolve repo root (dir containing atr_engine.py) for imports + output paths.
_here = Path(__file__).resolve().parent
for _cand in (_here, *_here.parents):
    if (_cand / "atr_engine.py").exists():
        ROOT = _cand
        break
sys.path.insert(0, str(ROOT))

import torch
from transformer_lens import HookedTransformer

from atr_engine import run_atr_gated
import prompt_library as pl

OUT = ROOT / "experiments" / "gpt2_small" / "output_gated"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "gated_results.pt"

MAX_ITER = 1000
THRESHOLD = 0.999
PATIENCE = 3
CHECK_EVERY = 10
CHECK_START = 100
LIMIT = int(os.environ.get("GATED_LIMIT", "0")) or None

ALL_CATS = {**pl.COMPLEX, **pl.NARRATIVE, **pl.SIMPLE, **pl.CHEMICAL,
            **pl.ACRONYMS, **pl.VULGARITY, **pl.WILD}
prompt_ids = list(ALL_CATS.keys())
if LIMIT:
    prompt_ids = prompt_ids[:LIMIT]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | prompts: {len(prompt_ids)} | max_iter={MAX_ITER} "
      f"| gate: cos>{THRESHOLD} x{PATIENCE} every {CHECK_EVERY} past {CHECK_START}",
      flush=True)

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.eval()
LAYER_START, LAYER_END = 0, model.cfg.n_layers - 1

results = {}
if CKPT.exists():
    results = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Resuming: {len(results)} prompts already done", flush=True)

t0 = time.time()
for idx, pid in enumerate(prompt_ids):
    if pid in results:
        continue
    prompt = ALL_CATS[pid]
    ts = time.time()
    with torch.no_grad():
        r = run_atr_gated(model, prompt, LAYER_START, LAYER_END,
                          max_iter=MAX_ITER, threshold=THRESHOLD,
                          patience=PATIENCE, check_every=CHECK_EVERY,
                          check_start=CHECK_START)
    results[pid] = r
    torch.save(results, CKPT)  # checkpoint after every prompt
    dt = time.time() - ts
    lock = r["lock_in_iter"] if r["converged"] else f"NO({r['n_iters']})"
    print(f"[{idx+1}/{len(prompt_ids)}] {pid:<16} basin={r['terminal_token'].strip()!r:<14} "
          f"lock_in={lock} cos={r['final_cos_sim_mean']:.4f} ({dt:.1f}s)", flush=True)

print(f"\nSweep done: {len(results)} prompts in {(time.time()-t0)/60:.1f} min", flush=True)

# ---- Analysis: iter-100 (published) vs lock-in basins ----
stage1 = ROOT / "experiments" / "gpt2_small" / "output" / "stage1_results.pt"
at100 = {}
if stage1.exists():
    s1 = torch.load(stage1, map_location="cpu", weights_only=False)
    for pid, col in s1.items():
        # columnar: top_tokens is a list per scheduled iter; last = iter 100
        at100[pid] = col["top_tokens"][-1][0][0].strip()

done = [p for p in prompt_ids if p in results]
gated_basin = {p: results[p]["terminal_token"].strip() for p in done}
n = len(done)

c100 = Counter(at100[p] for p in done if p in at100)
cgat = Counter(gated_basin[p] for p in done)
n_conv = sum(1 for p in done if results[p]["converged"])
lock_iters = [results[p]["lock_in_iter"] for p in done if results[p]["converged"]]

lines = ["# Section 5 — GPT-2 Small Convergence-Gated Re-sweep\n"]
lines.append(f"- Prompts: {n}  |  gate: cos_sim_mean > {THRESHOLD} x{PATIENCE} "
             f"(every {CHECK_EVERY} iters past {CHECK_START}), max_iter={MAX_ITER}")
lines.append(f"- Locked in (converged): {n_conv}/{n}  |  ran to {MAX_ITER}: {n - n_conv}/{n}")
if lock_iters:
    import statistics
    lines.append(f"- Lock-in iteration: min={min(lock_iters)}, median="
                 f"{int(statistics.median(lock_iters))}, max={max(lock_iters)}")
lines.append("\n## Basin shares: iter 100 (published) vs at lock-in\n")
lines.append("| Basin | @100 | @lock-in | delta |")
lines.append("|:---|---:|---:|---:|")
basins = sorted(set(c100) | set(cgat), key=lambda b: -(c100.get(b, 0) + cgat.get(b, 0)))
for b in basins:
    a, g = c100.get(b, 0), cgat.get(b, 0)
    if a == 0 and g == 0:
        continue
    lines.append(f"| `{b}` | {a} ({100*a/n:.1f}%) | {g} ({100*g/n:.1f}%) | {g-a:+d} |")

# till hypothesis
till_100 = [p for p in done if at100.get(p) == "till"]
if till_100:
    moved = Counter(gated_basin[p] for p in till_100)
    lines.append(f"\n## `till` hypothesis (was {len(till_100)} prompts @100)\n")
    lines.append("Where the iter-100 `till` prompts end at lock-in:")
    for tok, c in moved.most_common():
        lines.append(f"- `{tok}`: {c}")
    stayed = moved.get("till", 0)
    lines.append(f"\n**`till` retention: {stayed}/{len(till_100)}** "
                 f"({'collapses — slow transient confirmed' if stayed < len(till_100)/2 else 'largely stable'}).")

(OUT / "gated_report.md").write_text("\n".join(lines), encoding="utf-8")
print(f"[SAVED] {OUT / 'gated_report.md'}", flush=True)
print("\n".join(lines))
