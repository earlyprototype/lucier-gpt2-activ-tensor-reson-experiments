"""
EXP: Divine anomaly, readout-invisible motion test (issue #7, H-D1)

The confidence audit (output_confidence/confidence_report.md) found that the
Syntactic prompt's "Divine" state never passes the convergence gate yet reads
out one token at p = 0.505. Hypothesis H-D1: the late-stage motion of that
tensor lies (mostly) in directions the readout map (ln_final -> W_U) flattens
away. The tensor dances; the shadow stands still.

Method:
  1. Run the Syntactic prompt to 1000 iterations with a dense late schedule:
     [0, 100, 250, 500] plus every 10 iterations from 800 to 1000.
  2. Controls: one prolet-basin prompt (Semantic) and one calibrated noise
     tensor (Gaussian, seq_len 10, norm 397.18, torch.manual_seed(42)),
     same schedule (noise loop copies the 03/04 noise-trial pattern).
  3. For each successive late snapshot pair (t, t+10), at the last token
     position:
       a. Tensor motion: cosine similarity and L2 between last_vectors
          (mean_vector motion also recorded in the JSON).
       b. Readout motion: KL(p_new || p_old) and total-variation distance
          between the FULL softmax readouts, plus p(top1) and entropy drift.
       c. Invisibility ratio: the delta's actual effect on logits,
          ||logits(v + delta) - logits(v)||, divided by the mean effect of
          20 random directions of equal norm. Well below 1 means the motion
          is preferentially readout-invisible.

Staged, resumable execution (each chunk saves loop state to disk so runs
survive interruption; processes are single-threaded so the three runs can
execute in parallel on separate cores):

  python 05_divine_motion.py chunk divine 300   # advance Syntactic loop
  python 05_divine_motion.py chunk prolet 300   # advance Semantic control
  python 05_divine_motion.py chunk noise 300    # advance noise control
  (repeat until each prints RUN_COMPLETE; state_<key>.pt holds progress)
  python 05_divine_motion.py analyse            # measurements + tables

Outputs: output_divine_motion/snapshots_<label>.pt (checkpoints),
         output_divine_motion/divine_motion_results.json,
         output_divine_motion/divine_motion_tables.md (raw tables; the
         report divine_motion_report.md is written from these).

If huggingface.co is unreachable, set ATR_GPT2_LOCAL to a directory
containing the standard gpt2 files (config.json, pytorch_model.bin,
vocab.json, merges.txt) and the script will load offline.
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "output_divine_motion")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

# On this hardware 4 BLAS threads thrash (2.6 s/forward vs 0.45 s with 1):
# single-thread each process and parallelise across processes instead.
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

MAX_ITER = 1000
SCHEDULE = [0, 100, 250, 500] + list(range(800, MAX_ITER + 1, 10))
LATE_START = 800
L0, L1 = 0, model.cfg.n_layers - 1
N_RANDOM_DIRS = 20
RANDOM_DIR_SEED = 1234

RUNS = {
    "divine": ("Divine_Syntactic", "The cat sat on the mat and then the"),
    "prolet": ("Control_prolet_Semantic",
               "The Eiffel Tower is located in the city of"),
    "noise": ("Control_noise", None),
}
NOISE_NORM, NOISE_SEQ, NOISE_SEED = 397.17687817382813, 10, 42


def full_logits_probs(vec):
    """Full-vocab logits and softmax for one [d_model] vector."""
    with torch.no_grad():
        logits = model.ln_final(vec) @ model.W_U + model.b_U
        return logits, torch.softmax(logits, dim=-1)


def readout_summary(vec):
    logits, probs = full_logits_probs(vec)
    top_p, top_i = torch.topk(probs, 5)
    H = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    return {
        "top1_token": model.tokenizer.decode([int(top_i[0])]),
        "top1_id": int(top_i[0]),
        "top1_prob": float(top_p[0]),
        "entropy_nats": H,
    }


def make_snapshot(i, tensor, cos):
    """Slim snapshot with just the fields the analysis needs."""
    lv = tensor[-1, :].clone()
    rs = readout_summary(lv)
    return {
        "iteration": i,
        "last_vector": lv.cpu(),
        "mean_vector": tensor.mean(dim=0).clone().cpu(),
        "last_norm": float(lv.norm()),
        "top_tokens": [(rs["top1_token"], rs["top1_prob"])],
        "entropy_last": rs["entropy_nats"],
        "cosine_sim_last": cos,
    }


def init_state(key):
    """Iteration-0 state for a run. Prompt runs mirror atr_engine.run_atr_loop
    (initial un-hooked forward pass); the noise run mirrors the 03/04
    noise-trial pattern (Gaussian tensor, fixed norm, torch.manual_seed(42))."""
    label, prompt = RUNS[key]
    hook_read = f"blocks.{L1}.hook_resid_post"
    if key == "noise":
        torch.manual_seed(NOISE_SEED)
        x = torch.randn(NOISE_SEQ, model.cfg.d_model)
        current = x * (NOISE_NORM / x.norm())
    else:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, names_filter=lambda n: n == hook_read)
        current = cache[hook_read][0].clone()
    snaps = [make_snapshot(0, current, 1.0)] if 0 in SCHEDULE else []
    return {
        "label": label, "prompt": prompt, "iteration": 0,
        "current_tensor": current, "initial_norm": current.norm().item(),
        "prev_last": current[-1, :].clone(),
        "snapshots": snaps,
    }


def stage_chunk(key, n_iters):
    """Advance one run by up to n_iters iterations, checkpointing state.

    Loop mechanics are identical to atr_engine.run_atr_loop: L2-normalise the
    tensor to its initial norm, inject at blocks.L0.hook_resid_pre (which
    overwrites the embeddings entirely), read blocks.L1.hook_resid_post.
    """
    state_path = os.path.join(OUT, f"state_{key}.pt")
    done_path = os.path.join(OUT, f"snapshots_{key}.pt")
    if os.path.exists(done_path):
        print(f"[{key}] RUN_COMPLETE (already finished)", flush=True)
        return
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=False)
    else:
        state = init_state(key)
    label, prompt = state["label"], state["prompt"]
    hook_read = f"blocks.{L1}.hook_resid_post"
    hook_write = f"blocks.{L0}.hook_resid_pre"
    # Hook overwrites the whole residual tensor, so the noise run only needs
    # a token scaffold of the right length (matches 03/04).
    run_input = prompt if key != "noise" else torch.full((1, NOISE_SEQ), 262)
    current = state["current_tensor"]
    initial_norm = state["initial_norm"]
    prev_last = state["prev_last"]
    start = state["iteration"] + 1
    end = min(state["iteration"] + n_iters, MAX_ITER)

    for i in range(start, end + 1):
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
        if i in SCHEDULE:
            cos = float(F.cosine_similarity(
                current[-1, :].unsqueeze(0), prev_last.unsqueeze(0)))
            state["snapshots"].append(make_snapshot(i, current, cos))
        prev_last = current[-1, :].clone()

    state.update(iteration=end, current_tensor=current, prev_last=prev_last)
    torch.save(state, state_path)
    print(f"[{label}] advanced to iteration {end} "
          f"({len(state['snapshots'])} snapshots)", flush=True)
    if end >= MAX_ITER:
        torch.save({"label": label, "prompt": prompt,
                    "snapshots": state["snapshots"]}, done_path)
        print(f"[{key}] RUN_COMPLETE", flush=True)


def pair_measurements(v0, v1, m0, m1, gen):
    """All issue-#7 measurements for one successive snapshot pair."""
    delta = v1 - v0
    delta_norm = float(delta.norm())
    cos_last = float(F.cosine_similarity(v1.unsqueeze(0), v0.unsqueeze(0)))
    cos_mean = float(F.cosine_similarity(m1.unsqueeze(0), m0.unsqueeze(0)))
    l2_mean = float((m1 - m0).norm())

    logits0, p0 = full_logits_probs(v0)
    logits1, p1 = full_logits_probs(v1)
    kl = float((p1 * (torch.log(p1.clamp_min(1e-12))
                      - torch.log(p0.clamp_min(1e-12)))).sum())
    tv = float(0.5 * (p1 - p0).abs().sum())
    r0, r1 = readout_summary(v0), readout_summary(v1)

    # Invisibility ratio: actual logit motion vs equal-norm random directions.
    # logits(v0 + delta) is exactly logits(v1), so the actual effect is direct.
    actual_effect = float((logits1 - logits0).norm())
    rand_effects = []
    if delta_norm > 0:
        for _ in range(N_RANDOM_DIRS):
            r = torch.randn(v0.shape[0], generator=gen)
            r = r * (delta_norm / r.norm())
            rl, _ = full_logits_probs(v0 + r)
            rand_effects.append(float((rl - logits0).norm()))
    rand_mean = sum(rand_effects) / len(rand_effects) if rand_effects else float("nan")
    rand_std = (sum((x - rand_mean) ** 2 for x in rand_effects)
                / len(rand_effects)) ** 0.5 if rand_effects else float("nan")
    ratio = actual_effect / rand_mean if rand_effects and rand_mean > 0 else float("nan")

    return {
        "tensor_cos_last": cos_last,
        "tensor_l2_last": delta_norm,
        "tensor_cos_mean": cos_mean,
        "tensor_l2_mean": l2_mean,
        "readout_kl_nats": kl,
        "readout_tv": tv,
        "p_top1_from": r0["top1_prob"],
        "p_top1_to": r1["top1_prob"],
        "p_top1_drift": r1["top1_prob"] - r0["top1_prob"],
        "entropy_from": r0["entropy_nats"],
        "entropy_to": r1["entropy_nats"],
        "entropy_drift": r1["entropy_nats"] - r0["entropy_nats"],
        "top1_from": r0["top1_token"],
        "top1_to": r1["top1_token"],
        "logit_effect_actual": actual_effect,
        "logit_effect_random_mean": rand_mean,
        "logit_effect_random_std": rand_std,
        "invisibility_ratio": ratio,
    }


def analyse_snapshots(snaps, label):
    """Late-band pairwise measurements plus trajectory-shape summary."""
    late = [s for s in snaps if s["iteration"] >= LATE_START]
    late.sort(key=lambda s: s["iteration"])
    gen = torch.Generator().manual_seed(RANDOM_DIR_SEED)
    pairs = []
    for a, b in zip(late[:-1], late[1:]):
        m = pair_measurements(a["last_vector"], b["last_vector"],
                              a["mean_vector"], b["mean_vector"], gen)
        m["iter_from"], m["iter_to"] = a["iteration"], b["iteration"]
        pairs.append(m)
        print(f"  [{label}] {a['iteration']}->{b['iteration']}: "
              f"cos={m['tensor_cos_last']:.6f} L2={m['tensor_l2_last']:.3f} "
              f"KL={m['readout_kl_nats']:.2e} TV={m['readout_tv']:.2e} "
              f"ratio={m['invisibility_ratio']:.3f}", flush=True)

    # Trajectory shape over the late band: path length vs net displacement.
    path_len = sum(p["tensor_l2_last"] for p in pairs)
    net_disp = float((late[-1]["last_vector"] - late[0]["last_vector"]).norm())
    ratios = [p["invisibility_ratio"] for p in pairs
              if not math.isnan(p["invisibility_ratio"])]
    return {
        "trace": [{
            "iteration": s["iteration"],
            "top1": s["top_tokens"][0][0],
            "top1_prob": s["top_tokens"][0][1],
            "entropy_nats": s.get("entropy_last"),
            "cos_sim_last": s["cosine_sim_last"],
            "last_norm": s["last_norm"],
        } for s in snaps],
        "late_pairs": pairs,
        "late_band_summary": {
            "path_length_last": path_len,
            "net_displacement_last": net_disp,
            "wander_ratio": path_len / net_disp if net_disp > 0 else float("inf"),
            "mean_tensor_l2": path_len / len(pairs) if pairs else float("nan"),
            "mean_invisibility_ratio": sum(ratios) / len(ratios) if ratios else float("nan"),
            "min_invisibility_ratio": min(ratios) if ratios else float("nan"),
            "max_invisibility_ratio": max(ratios) if ratios else float("nan"),
            "total_kl": sum(p["readout_kl_nats"] for p in pairs),
            "total_tv": sum(p["readout_tv"] for p in pairs),
            "p_top1_at_800": pairs[0]["p_top1_from"] if pairs else None,
            "p_top1_at_1000": pairs[-1]["p_top1_to"] if pairs else None,
            "entropy_at_800": pairs[0]["entropy_from"] if pairs else None,
            "entropy_at_1000": pairs[-1]["entropy_to"] if pairs else None,
        },
    }


def stage_probe(key, n_iters=20):
    """Lag-1 probe: continue a finished run from its saved state (iteration
    MAX_ITER), snapshotting EVERY iteration, and measure per-iteration motion,
    readout motion, invisibility ratio, and the orbit period.

    Motivation: the lag-10 schedule turned out to sample the Divine orbit
    phase-locked (snapshots 10 apart are identical while consecutive-iteration
    cosine sits at 0.685). Only lag-1 sampling can see the true motion.
    """
    state = torch.load(os.path.join(OUT, f"state_{key}.pt"), weights_only=False)
    label, prompt = state["label"], state["prompt"]
    hook_read = f"blocks.{L1}.hook_resid_post"
    hook_write = f"blocks.{L0}.hook_resid_pre"
    run_input = prompt if key != "noise" else torch.full((1, NOISE_SEQ), 262)
    current = state["current_tensor"]
    initial_norm = state["initial_norm"]
    base_iter = state["iteration"]

    vecs = [(base_iter, current[-1, :].clone(), current.mean(dim=0).clone())]
    for i in range(base_iter + 1, base_iter + n_iters + 1):
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
        vecs.append((i, current[-1, :].clone(), current.mean(dim=0).clone()))

    gen = torch.Generator().manual_seed(RANDOM_DIR_SEED)
    pairs = []
    for (i0, v0, m0), (i1, v1, m1) in zip(vecs[:-1], vecs[1:]):
        m = pair_measurements(v0, v1, m0, m1, gen)
        m["iter_from"], m["iter_to"] = i0, i1
        pairs.append(m)
        print(f"  [{label} lag-1] {i0}->{i1}: cos={m['tensor_cos_last']:.4f} "
              f"L2={m['tensor_l2_last']:.2f} KL={m['readout_kl_nats']:.3e} "
              f"TV={m['readout_tv']:.3e} top1={m['top1_to']!r} "
              f"p={m['p_top1_to']:.4f} ratio={m['invisibility_ratio']:.3f}",
              flush=True)

    # Orbit period: L2 distance from the base vector at every lag.
    base_v = vecs[0][1]
    lags = [{"lag": i1 - vecs[0][0], "l2_from_base": float((v1 - base_v).norm()),
             "cos_with_base": float(F.cosine_similarity(
                 v1.unsqueeze(0), base_v.unsqueeze(0)))}
            for (i1, v1, _) in vecs[1:]]
    for d in lags:
        print(f"  [{label}] lag {d['lag']:>2}: L2 from base = "
              f"{d['l2_from_base']:.3f}, cos = {d['cos_with_base']:.6f}",
              flush=True)

    phases = [dict(readout_summary(v1), iteration=i1) for (i1, v1, _) in vecs]

    probe_path = os.path.join(OUT, "probe_lag1_results.json")
    probe = {}
    if os.path.exists(probe_path):
        with open(probe_path) as fh:
            probe = json.load(fh)
    probe[label] = {"base_iteration": base_iter, "pairs": pairs,
                    "lags_from_base": lags, "phase_readouts": phases}
    with open(probe_path, "w") as fh:
        json.dump(probe, fh, indent=1)
    print(f"probe saved for {label}", flush=True)


def stage_analyse():
    results = {"meta": {
        "issue": 7,
        "schedule": SCHEDULE,
        "max_iter": MAX_ITER,
        "late_band": [LATE_START, MAX_ITER],
        "n_random_dirs": N_RANDOM_DIRS,
        "random_dir_seed": RANDOM_DIR_SEED,
        "kl_direction": "KL(p_new || p_old), nats",
        "noise_config": {"seed": NOISE_SEED, "seq_len": NOISE_SEQ,
                         "norm": NOISE_NORM},
    }, "runs": {}}
    for key in ["divine", "prolet", "noise"]:
        ck = torch.load(os.path.join(OUT, f"snapshots_{key}.pt"),
                        weights_only=False)
        label = ck["label"]
        results["runs"][label] = analyse_snapshots(ck["snapshots"], label)
        results["runs"][label]["prompt"] = ck["prompt"]

    with open(os.path.join(OUT, "divine_motion_results.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    # ---- Markdown tables (raw, consumed by divine_motion_report.md) ----
    lines = []
    for label in ["Divine_Syntactic", "Control_prolet_Semantic", "Control_noise"]:
        run = results["runs"][label]
        lines.append(f"### {label}\n")
        lines.append("| Pair | Tensor cos | Tensor L2 | Readout KL (nats) | "
                     "Readout TV | p(top1) | Entropy (nats) | Invisibility ratio |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for p in run["late_pairs"]:
            lines.append(
                f"| {p['iter_from']}-{p['iter_to']} "
                f"| {p['tensor_cos_last']:.6f} | {p['tensor_l2_last']:.2f} "
                f"| {p['readout_kl_nats']:.2e} | {p['readout_tv']:.2e} "
                f"| {p['p_top1_to']:.4f} | {p['entropy_to']:.4f} "
                f"| {p['invisibility_ratio']:.3f} |")
        s = run["late_band_summary"]
        lines.append("")
        lines.append(f"Late-band summary: path length {s['path_length_last']:.2f}, "
                     f"net displacement {s['net_displacement_last']:.2f}, "
                     f"wander ratio {s['wander_ratio']:.2f}, "
                     f"mean invisibility ratio {s['mean_invisibility_ratio']:.3f} "
                     f"(range {s['min_invisibility_ratio']:.3f} to "
                     f"{s['max_invisibility_ratio']:.3f}), "
                     f"total KL {s['total_kl']:.2e}, total TV {s['total_tv']:.2e}.\n")
    with open(os.path.join(OUT, "divine_motion_tables.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("DONE. Results in", OUT, flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "chunk":
        n = int(sys.argv[3]) if len(sys.argv) >= 4 else 250
        stage_chunk(sys.argv[2], n)
    elif len(sys.argv) >= 3 and sys.argv[1] == "probe":
        n = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
        stage_probe(sys.argv[2], n)
    elif len(sys.argv) >= 2 and sys.argv[1] == "analyse":
        stage_analyse()
    else:
        print("usage: 05_divine_motion.py chunk {divine|prolet|noise} [n_iters]"
              " | probe {divine|prolet|noise} [n_iters] | analyse")
        sys.exit(1)
