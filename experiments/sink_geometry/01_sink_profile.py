"""Sink probe, corrected.

Fix 1: the BOS control now compares against a *raw* HuggingFace tokenizer.
       TransformerLens wraps the tokenizer and makes it prepend BOS too, so
       comparing TL-tokenizer against TL-run_with_cache compares a thing to
       itself and always reports "no BOS added". That was a bug in v1.

Fix 2: transformers 5.x renamed GPTNeoX's `embed_out` to `lm_head`; this TL
       release still reaches for `embed_out`. Alias it before conversion.
"""
import json
import sys

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

sys.path.insert(0, REPO)
from prompt_library import PROMPT_LIBRARY  # noqa: E402

MODELS = {
    "gpt2":         "gpt2",
    "gpt2-medium":  "gpt2-medium",
    "pythia-160m":  "EleutherAI/pythia-160m",
    "pythia-410m":  "EleutherAI/pythia-410m",
}
N_PROMPTS = 12
PROBE = "The quick brown fox"

prompts = [v for v in list(PROMPT_LIBRARY.values()) if isinstance(v, str)][:N_PROMPTS]
report = {}


def load(tl_name, hf_name):
    if "pythia" in tl_name:
        hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.float32)
        if not hasattr(hf, "embed_out") and hasattr(hf, "lm_head"):
            hf.embed_out = hf.lm_head          # transformers 5.x rename
        return HookedTransformer.from_pretrained(tl_name, hf_model=hf, device="cpu")
    return HookedTransformer.from_pretrained(tl_name, device="cpu")


for name, hf_name in MODELS.items():
    print(f"\n{'='*72}\n{name}\n{'='*72}", flush=True)
    model = load(name, hf_name)
    last = model.cfg.n_layers - 1
    site = f"blocks.{last}.hook_resid_post"
    e = {"n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model,
         "read_site": site}

    # ---- Q1: BOS, against a RAW tokenizer -------------------------------
    raw_tok = AutoTokenizer.from_pretrained(hf_name)
    n_raw = len(raw_tok.encode(PROBE))
    with torch.no_grad():
        _, c = model.run_with_cache(PROBE, names_filter=lambda n: n == site)
    n_engine = c[site].shape[1]                 # exactly atr_engine.py's call
    strs = model.to_str_tokens(PROBE)
    e["bos"] = {
        "raw_hf_tokens": n_raw,
        "atr_engine_path_tokens": n_engine,
        "delta": n_engine - n_raw,
        "prepends_bos": n_engine == n_raw + 1,
        "first_3_str_tokens": strs[:3],
        "cfg.default_prepend_bos": getattr(model.cfg, "default_prepend_bos", None),
    }
    print(f"  BOS: raw={n_raw} engine={n_engine} -> prepends_bos="
          f"{e['bos']['prepends_bos']}  first3={strs[:3]}", flush=True)

    # ---- Q2: magnitude structure at the read site ------------------------
    states = []
    for p in prompts:
        with torch.no_grad():
            _, c = model.run_with_cache(p, names_filter=lambda n: n == site)
        states.append(c[site][0].float())

    # Skip position 0 only where it is actually BOS. Q1 above establishes that
    # TransformerLens prepends for GPT-2 and not for the NeoX family, so an
    # unconditional s[1:] would silently drop a real content token from every
    # Pythia profile and bias the cross-family comparison this section rests on.
    has_bos = e["bos"]["prepends_bos"]
    content = torch.stack([(s[1:] if has_bos else s).abs().mean(0) for s in states])
    pos0 = torch.stack([s[0].abs() for s in states])
    prof = content.mean(0)
    med = prof.median().item()
    tv, ti = prof.topk(10)

    per_top = [set(content[i].topk(10).indices.tolist()) for i in range(len(states))]
    shared = set.intersection(*per_top)

    energy = prof ** 2
    tot = energy.sum().item()
    e["magnitude"] = {
        "median_abs": round(med, 3),
        "max_abs": round(tv[0].item(), 1),
        "ratio_max_to_median": round(tv[0].item() / med, 1),
        "top10_coord_idx": ti.tolist(),
        "top10_shared_by_all_prompts": len(shared),
        "energy_frac_top1": round(energy.topk(1).values.sum().item() / tot, 4),
        "energy_frac_top5": round(energy.topk(5).values.sum().item() / tot, 4),
        "energy_frac_top10": round(energy.topk(10).values.sum().item() / tot, 4),
        "energy_frac_top20": round(energy.topk(20).values.sum().item() / tot, 4),
    }
    e["position_0"] = {
        "is_bos": has_bos,          # position 0 is BOS for GPT-2, content for Pythia
        "pos0_norm": round(pos0.norm(dim=1).mean().item(), 1),
        "rest_norm": round(torch.stack(
            [(s[1:] if has_bos else s).norm(dim=1).mean()
             for s in states]).mean().item(), 1),
    }
    e["position_0"]["ratio"] = round(
        e["position_0"]["pos0_norm"] / e["position_0"]["rest_norm"], 3)

    m = e["magnitude"]
    print(f"  site {site}  d_model={model.cfg.d_model}", flush=True)
    print(f"  max/median = {m['ratio_max_to_median']}x   "
          f"top-10 coords shared by all {len(states)} prompts: "
          f"{m['top10_shared_by_all_prompts']}/10", flush=True)
    print(f"  energy fraction  top1={m['energy_frac_top1']}  "
          f"top5={m['energy_frac_top5']}  top10={m['energy_frac_top10']}  "
          f"top20={m['energy_frac_top20']}", flush=True)
    tag = "BOS" if has_bos else "content"
    print(f"  position-0 ({tag}) norm {e['position_0']['pos0_norm']} vs rest "
          f"{e['position_0']['rest_norm']} (ratio {e['position_0']['ratio']})",
          flush=True)

    report[name] = e
    del model, states

out = os.path.join(OUT, "sink_profile.json")
json.dump(report, open(out, "w"), indent=2)
print(f"\nwrote {out}")
