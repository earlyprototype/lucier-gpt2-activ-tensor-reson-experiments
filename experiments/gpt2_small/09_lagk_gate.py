"""
EXP: Lag-k re-gate census of the committed iteration-1000 states (issue #14)

The convergence gate used everywhere so far (atr_engine.run_atr_gated,
gated_resweep.py) compares consecutive iterates: cos_sim_mean between t and
t-1 must hold above threshold. A period-2 limit cycle can NEVER pass that
comparison, however locked the cycle is: consecutive iterates of the Divine
bell sit at cosine 0.6849 forever. Session 03's standing correction
(docs/sessions/SESSION_03_HANDOVER.md): "34 prompts never converge" should
read "34 prompts ring, pending re-gate", and the fix is a one-line engine
change. That change now exists: run_atr_gated takes gate_lag (default 1,
exact historical behaviour), comparing iterate t with iterate t - gate_lag;
and atr_engine.lag_scan measures mean cosine at every lag 1..max_lag over a
dense run of iterates (pure tensor arithmetic, no model).

This script is the census demonstration on the three committed states:

  1. Load the iteration-1000 loop states saved by 05_divine_motion.py
     (output_divine_motion/state_{divine,prolet,noise}.pt) and continue each
     for 24 further iterations with the exact ATR map, recording EVERY
     iterate (dense: no schedule, no aliasing).
  2. Sanity gate on the Divine continuation before anything is measured:
     cos(A, f(A)) must reproduce 0.6849 and cos(A, f(f(A))) must reproduce
     1.0000 (the bell_anatomy.json values); abort if replication drifts.
  3. lag_scan (k = 1..8) on each state's iterates, for both the mean vector
     (the gate's metric) and the last vector; pass/fail at the engine's own
     default threshold; lag-1 detail for the final pairs; phase A and phase B
     readouts for Divine at two consecutive iterates.

Expected shape: prolet passes at k = 1 (true fixed point), Divine fails
k = 1 and passes exactly at the even lags (period 2), noise decays
monotonically with lag (drift, no period). A hypothetical period-4 ringer
would pass only at k = 4 and 8: the pattern across lags, not any single
lag, is the census instrument.

Outputs: output_lagk/lagk_results.json (lag tables, gate verdicts, sanity
numbers; the report lagk_report.md is written from these).

Run:  python 09_lagk_gate.py      (from experiments/gpt2_small/)

If huggingface.co is unreachable, set ATR_GPT2_LOCAL to a directory
containing the standard gpt2 files (config.json, pytorch_model.bin,
vocab.json, merges.txt) and the script will load offline.
"""
import os, sys, json, inspect

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "output_lagk")
SRC = os.path.join(HERE, "output_divine_motion")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

# Single-thread BLAS: multi-threaded thrashing costs 5x per forward here.
torch.set_num_threads(1)

LOCAL = os.environ.get("ATR_GPT2_LOCAL")
if LOCAL:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
    hf_model = GPT2LMHeadModel.from_pretrained(LOCAL)
    tokenizer = GPT2TokenizerFast.from_pretrained(LOCAL)
    import transformer_lens.loading_from_pretrained as lfp
    _cfg = GPT2Config.from_pretrained(LOCAL)
    class _Shim:
        @staticmethod
        def from_pretrained(name, *a, **k):
            return _cfg
    lfp.AutoConfig = _Shim
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf_model,
                                              tokenizer=tokenizer, device="cpu")
else:
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()

from atr_engine import lag_scan, run_atr_gated, get_readout_detail

N_CONT = 24          # continuation iterations per state, every iterate kept
MAX_LAG = 8
BASE_ITER = 1000     # the committed states are the loop at iteration 1000
L0, L1 = 0, model.cfg.n_layers - 1

# The standard threshold, read from the engine itself (same value the
# 125-prompt gated re-sweep ran at).
_sig = inspect.signature(run_atr_gated).parameters
THRESHOLD = _sig["threshold"].default
GATE_LAG_DEFAULT = _sig["gate_lag"].default

# Sanity-gate reference numbers from the committed bell anatomy.
with open(os.path.join(SRC, "bell_anatomy.json")) as fh:
    BELL = json.load(fh)
SANITY_LAG1_TOL = 0.005   # |cos(A, f(A)) - 0.684912| must be inside this
SANITY_LAG2_MIN = 0.9999  # cos(A, f(f(A))) must exceed this

KEYS = ["divine", "prolet", "noise"]


def continue_state(key):
    """Continue one committed state N_CONT iterations, keeping every iterate.

    Loop mechanics are identical to atr_engine.run_atr_gated and to
    05_divine_motion.stage_chunk: L2-normalise the tensor to its initial
    norm, inject at blocks.0.hook_resid_pre (overwriting the embeddings
    entirely), read blocks.11.hook_resid_post. The noise run only needs a
    token scaffold of the right length (the hook overwrites it).
    """
    state = torch.load(os.path.join(SRC, f"state_{key}.pt"), weights_only=True)
    assert state["iteration"] == BASE_ITER, (key, state["iteration"])
    label, prompt = state["label"], state["prompt"]
    current = state["current_tensor"]
    initial_norm = state["initial_norm"]
    hook_read = f"blocks.{L1}.hook_resid_post"
    hook_write = f"blocks.{L0}.hook_resid_pre"
    run_input = (prompt if key != "noise"
                 else torch.full((1, current.shape[0]), 262))

    lasts = [current[-1, :].clone()]
    means = [current.mean(dim=0).clone()]
    for i in range(BASE_ITER + 1, BASE_ITER + N_CONT + 1):
        cn = current.norm().item()
        if cn > 0:
            current = current * (initial_norm / cn)
        inject = current.clone()
        def hookfn(resid, hook, tensor=inject):
            resid[0, :, :] = tensor
            return resid
        model.add_hook(hook_write, hookfn)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    run_input, names_filter=lambda n: n == hook_read)
        finally:
            model.reset_hooks()
        current = cache[hook_read][0].clone()
        lasts.append(current[-1, :].clone())
        means.append(current.mean(dim=0).clone())
    print(f"[{label}] continued {BASE_ITER} -> {BASE_ITER + N_CONT}, "
          f"{len(lasts)} dense iterates", flush=True)
    return {"label": label, "prompt": prompt,
            "last": torch.stack(lasts), "mean": torch.stack(means)}


def cos_pairs(stack, lag):
    """Per-pair cosine at one lag over a [n, d] stack of iterates."""
    return F.cosine_similarity(stack[lag:], stack[:-lag], dim=-1)


def slim_readout(vec):
    d = get_readout_detail(model, vec)
    return {"top1": d["top_token_strings"][0], "top1_id": d["top_token_ids"][0],
            "top1_prob": d["top_token_probs"][0], "entropy_nats": d["entropy"]}


# ---- 1. Continuations (Divine first: its sanity gate guards the rest) ----
runs = {}
for key in KEYS:
    runs[key] = continue_state(key)
    if key == "divine":
        lv = runs[key]["last"]
        cos1 = float(F.cosine_similarity(lv[0], lv[1], dim=0))
        cos2 = float(F.cosine_similarity(lv[0], lv[2], dim=0))
        ok = (abs(cos1 - BELL["cosAB"]) <= SANITY_LAG1_TOL
              and cos2 >= SANITY_LAG2_MIN)
        print(f"  SANITY cos(A, f(A)) = {cos1:.6f} (bell_anatomy {BELL['cosAB']:.6f}), "
              f"cos(A, f(f(A))) = {cos2:.6f} (bell_anatomy {BELL['cosAA2']:.6f}) "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        sanity = {"cos_A_fA": cos1, "cos_A_ffA": cos2,
                  "expected_cosAB": BELL["cosAB"],
                  "expected_cosAA2": BELL["cosAA2"],
                  "tol_lag1": SANITY_LAG1_TOL, "min_lag2": SANITY_LAG2_MIN,
                  "passed": ok}
        if not ok:
            print("SANITY GATE FAILED: the ATR map replication does not "
                  "reproduce the committed period-2 numbers. Fix the map "
                  "before trusting any census below.", flush=True)
            sys.exit(1)

# ---- 2. Lag census ----
results = {"meta": {
    "issue": 14,
    "date": "2026-07-19",
    "base_iteration": BASE_ITER,
    "n_continuation_iters": N_CONT,
    "n_dense_iterates": N_CONT + 1,
    "max_lag": MAX_LAG,
    "threshold": THRESHOLD,
    "threshold_source": "atr_engine.run_atr_gated default (gated_resweep.py "
                        "ran the 125-prompt sweep at the same value)",
    "gate_lag_default": GATE_LAG_DEFAULT,
    "gate_metric": "cos_sim_mean: cosine between mean vectors of iterates "
                   "gate_lag apart (lag tables for the last vector are "
                   "recorded alongside)",
    "source_states": [f"output_divine_motion/state_{k}.pt" for k in KEYS],
    "sanity_gate": sanity,
}, "runs": {}}

for key in KEYS:
    run = runs[key]
    label = run["label"]
    scan_mean = lag_scan(run["mean"], MAX_LAG)
    scan_last = lag_scan(run["last"], MAX_LAG)
    min_mean = {k: float(cos_pairs(run["mean"], k).min()) for k in scan_mean}
    passes = {k: bool(scan_mean[k] > THRESHOLD) for k in scan_mean}
    passing = [k for k in sorted(passes) if passes[k]]
    smallest = passing[0] if passing else None

    n = run["last"].shape[0]
    last_pairs = []
    for j in range(n - 5, n - 1):
        a, b = run["last"][j], run["last"][j + 1]
        last_pairs.append({
            "iter_from": BASE_ITER + j, "iter_to": BASE_ITER + j + 1,
            "cos_last": float(F.cosine_similarity(a, b, dim=0)),
            "l2_last": float((b - a).norm()),
        })

    entry = {
        "prompt": run["prompt"],
        "lag_mean_cos_meanvec": {str(k): v for k, v in scan_mean.items()},
        "lag_mean_cos_lastvec": {str(k): v for k, v in scan_last.items()},
        "lag_min_cos_meanvec": {str(k): v for k, v in min_mean.items()},
        "passes_at_lag": {str(k): v for k, v in passes.items()},
        "smallest_passing_lag": smallest,
        "final_lag1_pairs": last_pairs,
        "final_readout": slim_readout(run["last"][-1]),
    }
    if key == "divine":
        entry["phase_readouts"] = {
            f"iter_{BASE_ITER + N_CONT - 1}_phase_B": slim_readout(run["last"][-2]),
            f"iter_{BASE_ITER + N_CONT}_phase_A": slim_readout(run["last"][-1]),
        }
        entry["regate_verdict"] = {
            "gate_lag": 2,
            "threshold": THRESHOLD,
            "mean_cos_at_lag_2": scan_mean[2],
            "min_cos_at_lag_2": min_mean[2],
            "mean_cos_at_lag_1": scan_mean[1],
            "converged_at_gate_lag_2": bool(min_mean[2] > THRESHOLD),
            "converged_at_gate_lag_1": bool(min_mean[1] > THRESHOLD),
            "note": "every lag-2 pair in the window clears the threshold, so "
                    "any patience/check_every schedule locks in; at lag 1 the "
                    "cosine is pinned at cos(A, B) and can never pass",
        }
    results["runs"][label] = entry

    print(f"\n[{label}] smallest passing lag at threshold {THRESHOLD}: {smallest}")
    print("  lag : mean cos (mean vec) | min cos | pass")
    for k in sorted(scan_mean):
        print(f"  {k:>3} : {scan_mean[k]:.7f} | {min_mean[k]:.7f} | "
              f"{'yes' if passes[k] else 'no'}", flush=True)

with open(os.path.join(OUT, "lagk_results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print(f"\nDONE. Results in {OUT}", flush=True)
