"""
EXP: Readout confidence audit of converged ATR states ("singing or rattling?")

Re-runs the original five-prompt piece (lucier_total_resonance) to 500
iterations plus calibrated noise trials (03_random_baseline config), then
interrogates each converged state's FULL readout distribution instead of
its argmax:

  - top-20 tokens with probabilities
  - top-1 probability and top1-top2 logit margin
  - full-vocabulary entropy (nats) and effective support exp(H)
  - the same metrics tracked across the iteration schedule
  - "chordness": mean pairwise W_E cosine among the top-10 tokens,
    against a random-token baseline
  - reference: the model's genuine next-token distribution at iteration 0

Results and interpretation: output_confidence/confidence_report.md

Run:  python 04_readout_confidence.py
      (from experiments/gpt2_small/; expects atr_engine.py at repo root)

If huggingface.co is unreachable, set ATR_GPT2_LOCAL to a directory
containing the standard gpt2 files (config.json, pytorch_model.bin,
vocab.json, merges.txt) and the script will load offline.
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "output_confidence")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

import torch

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

from atr_engine import run_atr_loop, get_readout_detail

SCHEDULE = [0, 2, 3, 5, 10, 20, 50, 100, 250, 500]
MAX_ITER = 500
L0, L1 = 0, model.cfg.n_layers - 1
V = model.cfg.d_vocab
UNIFORM_H = math.log(V)

PROMPTS = {
    "Lucier":     "Am I sitting in a room different from the one you are in now",
    "Semantic":   "The Eiffel Tower is located in the city of",
    "Syntactic":  "The cat sat on the mat and then the",
    "Nonsense":   "Flurb glex morp wintly skade",
    "Imperative": "Calculate the sum of all prime numbers below",
}


def full_readout(vec, k=20):
    """Confidence metrics + top-k from the full distribution for one [768] vector."""
    with torch.no_grad():
        normalized = model.ln_final(vec)
        logits = normalized @ model.W_U + model.b_U
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, k)
        top_logits = logits[top_i]
        H = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
        return {
            "top_tokens": [model.tokenizer.decode([int(i)]) for i in top_i],
            "top_probs": [float(p) for p in top_p],
            "top1_prob": float(top_p[0]),
            "top10_mass": float(top_p[:10].sum()),
            "logit_margin": float(top_logits[0] - top_logits[1]),
            "entropy_nats": H,
            "effective_support": math.exp(H),
            "uniform_entropy": UNIFORM_H,
        }


def chordness(token_strings):
    """Mean pairwise W_E cosine among single-token strings (semantic coherence)."""
    ids = []
    for t in token_strings:
        enc = model.tokenizer.encode(t, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    if len(ids) < 3:
        return None
    with torch.no_grad():
        E = model.W_E[ids]
        E = E / E.norm(dim=-1, keepdim=True)
        sim = E @ E.T
        n = len(ids)
        return float((sim.sum() - n) / (n * (n - 1)))


results = {"prompts": {}, "noise": {}, "meta": {
    "schedule": SCHEDULE, "max_iter": MAX_ITER, "vocab": V,
    "uniform_entropy_nats": UNIFORM_H,
}}
tensors_to_save = {}

# ---- The five original prompts ----
for label, prompt in PROMPTS.items():
    print(f"=== PROMPT: {label} ===", flush=True)
    snaps = run_atr_loop(model, prompt, L0, L1, MAX_ITER, SCHEDULE, verbose=False)
    trace = [{
        "iteration": s["iteration"],
        "top1": s["top_tokens"][0][0],
        "top1_prob": s["top_tokens"][0][1],
        "logit_margin": s.get("top_logit_margin_last"),
        "entropy_nats": s.get("entropy_last"),
        "cos_sim_last": s["cosine_sim_last"],
        "cos_sim_mean": s["cosine_sim_mean"],
        "position_similarity": s["position_similarity"],
    } for s in snaps]
    final = snaps[-1]
    fr = full_readout(final["last_vector"])
    results["prompts"][label] = {
        "prompt": prompt,
        "trace": trace,
        "final_last_vector": fr,
        "final_mean_vector": full_readout(final["mean_vector"]),
        "baseline_iter0": full_readout(snaps[0]["last_vector"]),
        "final_all_position_tokens": final["all_position_tokens"],
        "chordness_top10": chordness(fr["top_tokens"][:10]),
    }
    tensors_to_save[label] = final["tensor"]
    print(f"  top1={fr['top_tokens'][0]!r} p={fr['top1_prob']:.3f} "
          f"margin={fr['logit_margin']:.2f} H={fr['entropy_nats']:.2f} "
          f"chord={results['prompts'][label]['chordness_top10']:.3f}", flush=True)

# ---- Calibrated noise trials (matches output_random_baseline/config.json) ----
N_NOISE = 15
NOISE_NORM_MEAN, NOISE_NORM_STD, NOISE_SEQ = 397.17687817382813, 43.860541764212016, 10
torch.manual_seed(42)
hook_read = f"blocks.{L1}.hook_resid_post"
hook_write = f"blocks.{L0}.hook_resid_pre"
scaffold_tokens = torch.full((1, NOISE_SEQ), 262)  # hook overwrites embeddings

for trial in range(N_NOISE):
    target_norm = float(torch.normal(torch.tensor(NOISE_NORM_MEAN),
                                     torch.tensor(NOISE_NORM_STD)))
    x = torch.randn(NOISE_SEQ, model.cfg.d_model)
    x = x * (target_norm / x.norm())
    initial_norm = x.norm().item()
    current = x.clone()
    trace = []
    for i in range(1, MAX_ITER + 1):
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
                    scaffold_tokens, names_filter=lambda n: n == hook_read)
        finally:
            model.reset_hooks()
        prev = current
        current = cache[hook_read][0].clone()
        if i in SCHEDULE:
            d = get_readout_detail(model, current[-1, :])
            cos = torch.nn.functional.cosine_similarity(
                current[-1, :].unsqueeze(0), prev[-1, :].unsqueeze(0)).item()
            trace.append({"iteration": i, "top1": d["top_token_strings"][0],
                          "top1_prob": d["top_token_probs"][0],
                          "logit_margin": d["top_logit_margin"],
                          "entropy_nats": d["entropy"], "cos_sim_last": cos})
    fr = full_readout(current[-1, :])
    results["noise"][f"trial_{trial:02d}"] = {
        "trace": trace,
        "final_last_vector": fr,
        "final_mean_vector": full_readout(current.mean(dim=0)),
        "chordness_top10": chordness(fr["top_tokens"][:10]),
    }
    print(f"noise {trial:02d}: top1={fr['top_tokens'][0]!r} p={fr['top1_prob']:.3f} "
          f"H={fr['entropy_nats']:.2f} "
          f"chord={results['noise'][f'trial_{trial:02d}']['chordness_top10']}", flush=True)

# ---- Random-token chordness baseline ----
import random
random.seed(0)
vals = []
for _ in range(50):
    ids = random.sample(range(model.W_E.shape[0]), 10)
    with torch.no_grad():
        E = model.W_E[ids]
        E = E / E.norm(dim=-1, keepdim=True)
        sim = E @ E.T
        vals.append(float((sim.sum() - 10) / 90))
results["meta"]["chordness_random_baseline"] = sum(vals) / len(vals)

with open(os.path.join(OUT, "confidence_results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
torch.save(tensors_to_save, os.path.join(OUT, "converged_tensors.pt"))
print("DONE. Results in", OUT, flush=True)
