"""Board #60's control, corrected and cached.

v1 bug: averaged 11 transitions over a divisor of 10, inflating every figure by
1.1x and producing "cosines" above 1. Fixed by counting the terms.

Generation and analysis are now separate: trajectories are cached to disk so the
masking analysis can be re-run without another sweep.
"""
import hashlib
import json
import os
import sys

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

sys.path.insert(0, REPO)
from atr_engine import run_atr_loop          # noqa: E402
from prompt_library import PROMPT_LIBRARY    # noqa: E402

MODELS = {"gpt2": "gpt2", "gpt2-medium": "gpt2-medium",
          "pythia-160m": "EleutherAI/pythia-160m",
          "pythia-410m": "EleutherAI/pythia-410m"}
N_PROMPTS, MAX_ITER, TAIL = 5, 60, 10
KS = [1, 2, 5, 10, 20, 50]

prompts = [v for v in PROMPT_LIBRARY.values() if isinstance(v, str)][:N_PROMPTS]
cache_path = os.path.join(OUT, "trajectories.pt")


def load(tl_name, hf_name):
    if "pythia" in tl_name:
        hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.float32)
        if not hasattr(hf, "embed_out") and hasattr(hf, "lm_head"):
            hf.embed_out = hf.lm_head
        return HookedTransformer.from_pretrained(tl_name, hf_model=hf, device="cpu")
    return HookedTransformer.from_pretrained(tl_name, device="cpu")


# ---- generate (or reuse) ------------------------------------------------
# The cache is only valid for the inputs that produced it. Reusing it across a
# change to MAX_ITER, the prompt set, the model list, or the engine would
# silently publish stale measurements under new-looking parameters, so the
# manifest is compared and a mismatch forces regeneration.
def manifest():
    return {"models": sorted(MODELS), "max_iter": MAX_ITER,
            "n_prompts": len(prompts),
            "prompts_sha": hashlib.sha256(
                "␟".join(prompts).encode()).hexdigest()[:16],
            "engine_sha": hashlib.sha256(
                open(os.path.join(REPO, "atr_engine.py"), "rb").read()
            ).hexdigest()[:16]}


want = manifest()
traj = None
if os.path.exists(cache_path):
    # weights_only: the cache is committed to the repo, so it is a file a third
    # party can modify. Unrestricted torch.load unpickles, i.e. executes.
    blob = torch.load(cache_path, map_location="cpu", weights_only=True)
    if blob.get("manifest") == want:
        traj = blob["traj"]
        print(f"reusing cached trajectories: {cache_path}")
    else:
        print("cache manifest mismatch — regenerating.")
        for k in set(want) | set(blob.get("manifest", {})):
            if blob.get("manifest", {}).get(k) != want.get(k):
                print(f"    {k}: cached={blob.get('manifest', {}).get(k)!r} "
                      f"now={want.get(k)!r}")

if traj is None:
    traj = {}
    for name, hf_name in MODELS.items():
        print(f"sweeping {name} ...", flush=True)
        model = load(name, hf_name)
        last = model.cfg.n_layers - 1
        traj[name] = {"d_model": model.cfg.d_model, "n_layers": model.cfg.n_layers,
                      "means": []}
        for p in prompts:
            snaps = run_atr_loop(model, p, 0, last, MAX_ITER,
                                 list(range(MAX_ITER + 1)), verbose=False)
            traj[name]["means"].append(
                torch.stack([s["mean_vector"].float().detach() for s in snaps]))
        del model
    torch.save({"manifest": want, "traj": traj}, cache_path)
    print(f"cached -> {cache_path}")


# ---- analyse ------------------------------------------------------------
def tail_cos(M, tail=TAIL):
    """Mean cosine between consecutive iterates over the last `tail` transitions."""
    vals = [torch.nn.functional.cosine_similarity(M[t], M[t - 1], dim=0).item()
            for t in range(len(M) - tail, len(M))]
    assert len(vals) == tail, f"{len(vals)} != {tail}"
    return sum(vals) / len(vals)


report = {}
for name in MODELS:
    d = traj[name]
    rows, agg = [], {}
    for pi, M in enumerate(d["means"]):
        prof = M.abs().mean(0)
        order = prof.argsort(descending=True)
        e = prof ** 2
        # 12 significant figures, not 5: a saturating model reported as 1.00000
        # is otherwise indistinguishable from one at 0.999996, and the whole
        # question here is whether the alignment is exact or merely close.
        r = {"prompt": pi,
             "unmasked": round(tail_cos(M), 12),
             "energy_top10": round(e[order[:10]].sum().item() / e.sum().item(), 4)}
        for k in KS:
            r[f"top{k}"] = round(tail_cos(M[:, order[k:]]), 12)
        rows.append(r)
    for key in ["unmasked"] + [f"top{k}" for k in KS] + ["energy_top10"]:
        agg[key] = round(sum(r[key] for r in rows) / len(rows), 12)
    # Distance from 1 for the saturating cases, where the leading digits hide it.
    agg["unmasked_1_minus_cos"] = round(1.0 - agg["unmasked"], 12)
    agg["masked_top10_1_minus_cos"] = round(1.0 - agg["top10"], 12)
    report[name] = {"per_prompt": rows, "mean": agg,
                    "d_model": d["d_model"], "n_layers": d["n_layers"]}

    print(f"\n{name}  (d_model={d['d_model']}, {d['n_layers']} layers)")
    print(f"  energy in top-10 coords: {agg['energy_top10']:.1%}")
    print(f"  1-cos: unmasked {agg['unmasked_1_minus_cos']:.3e}   "
          f"mask-top10 {agg['masked_top10_1_minus_cos']:.3e}")
    print(f"  {'unmasked':>10}: {agg['unmasked']:.9f}")
    for k in KS:
        delta = agg[f"top{k}"] - agg["unmasked"]
        print(f"  {'mask top'+str(k):>10}: {agg[f'top{k}']:.5f}   ({delta:+.5f})")

json.dump(report, open(os.path.join(OUT, "masking_control.json"), "w"), indent=2)
print(f"\nwrote {OUT}/masking_control.json")
