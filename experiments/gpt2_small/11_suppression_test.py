"""
EXP: Suppression-head test for L11.H8 (follow-up to 08_hinge_eigenvalue,
issue #14 thread 1)

08_hinge_eigenvalue.py located the sign inversion that sustains the Divine
period-2 cycle in one attention head: L11.H8 carries 99.1 percent of the
block-11 attention flip along the hinge (per-head d-component -1.981,
cos -0.963, base M_sym, direction d_sym). This experiment tests the
SUPPRESSION-HEAD HYPOTHESIS: L11.H8 functions as a suppression head (its
output is approximately a negative multiple of a component of its input,
the behaviour class documented for GPT-2 Small's L10.H7 copy-suppression
head), and the closed ATR loop turns that one-shot negative correction
into a sustained oscillation.

Structural fact this experiment verifies and then exploits: the Divine
state is position-uniform (all rows identical). For a position-uniform
input, an attention head's output is its OV transform of the ln1-normalised
input regardless of the attention pattern, because the pattern-weighted
average of identical value vectors is that vector. Verified empirically at
phase A before any measurement (hooked head-8 output vs the direct
computation ln1(x) @ W_V[11,8] @ W_O[11,8], plus the b_V term).

Three tests:
  1. OV CIRCUIT. For all 144 heads, y = d @ W_V[layer,h] @ W_O[layer,h]; record
     cos(y, d) and gain ||y||/||d||, for d_sym (primary), the committed d
     (secondary), the +d_sym and -d_sym pole directions, and 5 random unit
     vectors (control). Plus empirical operating-point checks: the exp-08
     block-0 injection reproduced (expect head-8 d-component -1.981), and
     direct layer-11 injections at Mn_sym and at the cascade resid_pre_11,
     tying the raw linear OV numbers to the exp-08 measurement through the
     ln1 scale.
  2. ABLATION. ATR loop from state_divine.pt with blocks.11.attn.hook_z
     zeroed at head 8 every pass (300 iterations), vs head 0 ablated
     (control, 100) and no ablation (control, 100). Per-iterate lag-1
     cosine, cosines to A, B, M_sym, M_committed, readout argmax; lag_scan
     k=1..8 on the last 24 iterates.
  3. ORDINARY TEXT. 12 natural sentences run once, no loop. For every
     position t >= 2 and each probe head, find the top non-BOS source s,
     compute the head's per-position output through W_O, and its effect on
     the logit of the token at s (delta = out_t @ W_U[:, token_at_s]).
     Copy suppression predicts predominantly negative delta. Heads: L11.H8
     (subject), L10.H7 (documented copy-suppression head, positive
     control), L11.H0 and L5.H5 (arbitrary controls).

States and frames exactly as 08_hinge_eigenvalue.py: A_full is the raw
iteration-1000 tensor from state_divine.pt, B_full = f(A_full), An/Bn are
the shell rescalings, d_sym = (An - Bn)/2 row (primary), committed
d = (A_full[-1] - Bn[-1])/2 (secondary, 06_bell_anatomy frame),
M_sym = (An + Bn)/2, M_committed = (A_full + Bn)/2. Sanity gates
cos(A,B) = 0.6849 and cos(A, f(f(A))) = 1.0 are asserted before any
measurement.

Run (from experiments/gpt2_small/):
    python 11_suppression_test.py            # all stages, resuming
    python 11_suppression_test.py gate test1 # named stages only
Stages: gate, test1, test2_none, test2_h0, test2_h8, test3. Results are
checkpointed to output_suppression/suppression_results.json after every
stage (and every 100 iterations inside test-2 runs, with the loop state in
test2_<run>_checkpoint.pt), so an interrupted run resumes where it
stopped. The report is output_suppression/suppression_report.md.

If huggingface.co is unreachable, set ATR_GPT2_LOCAL to a directory
containing the standard gpt2 files (config.json, pytorch_model.bin,
vocab.json, merges.txt) and the script will load offline.
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DIVINE = os.path.join(HERE, "output_divine_motion")
OUT = os.path.join(HERE, "output_suppression")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

# Single-threaded: 4 BLAS threads thrash on this box (see 05_divine_motion.py).
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

from atr_engine import lag_scan, get_top_tokens

PROMPT = "The cat sat on the mat and then the"
N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads
HOOK_READ = f"blocks.{N_LAYERS - 1}.hook_resid_post"
HOOK_WRITE = "blocks.0.hook_resid_pre"
HOOK_PRE11 = "blocks.11.hook_resid_pre"
HOOK_Z11 = "blocks.11.attn.hook_z"
EPS_RELS = [1e-3, 1e-4]
RANDOM_SEED = 20260721
# Recorded exp-08 reference (output_hinge_eigen/hinge_eigenvalue.json,
# base_Mn_sym__dir_d_sym, eps_rel 1e-3, per_head_at_flip head 8):
EXP08_H8_DCOMP = -1.9814
RESULTS_PATH = os.path.join(OUT, "suppression_results.json")

# ---------------------------------------------------------------- states ----
st = torch.load(os.path.join(DIVINE, "state_divine.pt"), weights_only=True)
A_full = st["current_tensor"]            # raw output at iteration 1000
N0 = float(st["initial_norm"])           # loop energy shell (Frobenius norm)
T, DM = A_full.shape


def norm_to(x, n):
    return x * (n / x.norm())


def unit(x):
    return x / x.norm()


def fcos(a, b):
    return float(F.cosine_similarity(a.flatten().unsqueeze(0),
                                     b.flatten().unsqueeze(0)))


def step(x, extra_names=(), extra_hooks=()):
    """One ATR iteration, hook-based, exactly as 06/08 run it. Returns a dict
    of cached tensors (batch index stripped); HOOK_READ is always cached."""
    cur = x * (N0 / x.norm())
    inject = cur.clone()
    def h(resid, hook, tensor=inject):
        resid[0, :, :] = tensor
        return resid
    names = {HOOK_READ} | set(extra_names)
    model.add_hook(HOOK_WRITE, h)
    for name, fn in extra_hooks:
        model.add_hook(name, fn)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                PROMPT, names_filter=lambda n: n in names)
    finally:
        model.reset_hooks()
    return {n: cache[n][0].detach().clone() for n in names}


def run_inject_at(hook_name, tensor, extra_names=()):
    """Forward pass with `tensor` overwriting every position at hook_name."""
    inject = tensor.clone()
    def h(resid, hook, t=inject):
        resid[0, :, :] = t
        return resid
    names = {HOOK_READ} | set(extra_names)
    model.add_hook(hook_name, h)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                PROMPT, names_filter=lambda n: n in names)
    finally:
        model.reset_hooks()
    return {n: cache[n][0].detach().clone() for n in names}


def row_spread(x):
    mean_row = x.mean(dim=0)
    return float((x - mean_row).norm() / mean_row.norm())


def atomic_save(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, RESULTS_PATH)


if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as fh:
        results = json.load(fh)
    print(f"resuming: stages done = {results.get('stages_done', [])}",
          flush=True)
else:
    results = {"stages_done": []}

# --------------------------------------------------------- gates, always ----
t_gate = time.time()
B_full = step(A_full)[HOOK_READ]         # raw phase B
A2_full = step(B_full)[HOOK_READ]        # raw, should equal A_full
An = norm_to(A_full, N0)
Bn = norm_to(B_full, N0)
A2n = norm_to(A2_full, N0)

cosAB = fcos(A_full[-1], Bn[-1])
cosAA2 = fcos(A_full[-1], A2n[-1])
cycle_residual = float((A_full - A2_full).norm())
print(f"GATE: cos(A,B) = {cosAB:.6f} (expect 0.684912), "
      f"cos(A, f(f(A))) = {cosAA2:.6f} (expect 1.0), "
      f"cycle residual = {cycle_residual:.2e}  [{time.time()-t_gate:.1f}s]",
      flush=True)
assert abs(cosAB - 0.6849116683) < 5e-4, "sanity gate failed: cos(A,B)"
assert cosAA2 > 0.99999, "sanity gate failed: cos(A, f(f(A)))"

# Hinges and pivots, both frames (08 construction).
e_sym = unit(((An - Bn) / 2)[-1])                 # primary hinge, row unit
e_committed = unit((A_full[-1] - Bn[-1]) / 2)     # committed hinge (06 frame)
D_sym_hat = unit((An - Bn) / 2)                   # full-tensor unit hinge
M_sym_full = (An + Bn) / 2
M_com_full = (A_full + Bn) / 2
Mn_sym = norm_to(M_sym_full, N0)
Mn_com = norm_to(M_com_full, N0)
A_row_hat = unit(A_full[-1])
B_row_hat = unit(B_full[-1])
Msym_row_hat = unit(M_sym_full[-1])
Mcom_row_hat = unit(M_com_full[-1])

results["meta"] = {
    "experiment": "11_suppression_test",
    "date": "2026-07-21",
    "model": "gpt2-small (TransformerLens, offline via ATR_GPT2_LOCAL)",
    "prompt": PROMPT,
    "seq_len": T,
    "d_model": DM,
    "n0_initial_norm": N0,
    "gate": {"cosAB": cosAB, "cosAA2": cosAA2,
             "full_tensor_cycle_residual": cycle_residual},
    "eps_rels": EPS_RELS,
    "random_seed": RANDOM_SEED,
    "exp08_reference_h8_dcomp": EXP08_H8_DCOMP,
    "frame_note": ("d_sym = row of (An - Bn)/2 is the primary hinge; the "
                   "committed d = (A_full[-1] - Bn[-1])/2 (06_bell_anatomy "
                   "frame, radial contamination cos 0.97 with A) is "
                   "reported secondarily."),
    "cos_esym_vs_ecommitted": fcos(e_sym, e_committed),
    "cos_esym_vs_Msym_row": fcos(e_sym, M_sym_full[-1]),
}

TRACK_HEADS = [(11, 8), (10, 7), (11, 0), (5, 5), (0, 0)]


# =================================================================== gate ===
def stage_gate():
    """Structural verification: position uniformity and the OV identity at
    phase A (pattern-independence of head output for uniform input)."""
    names = (HOOK_PRE11, HOOK_Z11, "blocks.11.hook_attn_out",
             "blocks.11.attn.hook_pattern", "blocks.11.ln1.hook_scale")
    c = step(A_full, names)
    P11 = c[HOOK_PRE11]
    z = c[HOOK_Z11]                                   # [pos, head, d_head]
    with torch.no_grad():
        actual = z[:, 8, :] @ model.W_O[11, 8]        # [pos, d_model]
        u = model.blocks[11].ln1(P11[-1].view(1, 1, -1))[0, 0]
        direct = (u @ model.W_V[11, 8] + model.b_V[11, 8]) @ model.W_O[11, 8]
        direct_nobias = (u @ model.W_V[11, 8]) @ model.W_O[11, 8]
        bias_term = (model.b_V[11, 8] @ model.W_O[11, 8]).norm()
    pat = c["blocks.11.attn.hook_pattern"][8]         # [query, key]
    out = {
        "resid_pre11_row_spread": row_spread(P11),
        "resid_pre11_row_norm": float(P11[-1].norm()),
        "ln1_scale_last_row": float(c["blocks.11.ln1.hook_scale"][-1]),
        "head8_output_row_spread_abs": float((actual - actual.mean(0)).norm()),
        "head8_output_row_norm": float(actual[-1].norm()),
        "ov_identity_rel_err_with_bV": float(
            (actual[-1] - direct).norm() / actual[-1].norm()),
        "ov_identity_rel_err_without_bV": float(
            (actual[-1] - direct_nobias).norm() / actual[-1].norm()),
        "bV_WO_term_norm": float(bias_term),
        "pattern_row_sums_max_dev_from_1": float(
            (pat.sum(-1) - 1.0).abs().max()),
        "pattern_head8_last_row": [float(v) for v in pat[-1]],
        "note": ("uniform input rows make every value vector identical, so "
                 "the pattern-weighted average equals that vector and the "
                 "head output is the OV transform of the ln1 row regardless "
                 "of the pattern; verified to numerical precision"),
    }
    print(f"  structural: pre11 row spread {out['resid_pre11_row_spread']:.2e}, "
          f"OV identity rel err {out['ov_identity_rel_err_with_bV']:.2e} "
          f"(without b_V {out['ov_identity_rel_err_without_bV']:.2e}), "
          f"ln1 scale {out['ln1_scale_last_row']:.2f}", flush=True)
    results["structural"] = out


# ================================================================== test1 ===
def ov_all_heads(t_row):
    """y = t @ W_V[layer,h] @ W_O[layer,h] for every head; cos(y, t), gain ||y||/||t||."""
    with torch.no_grad():
        tu = unit(t_row)
        v = torch.einsum("d,lhdk->lhk", tu, model.W_V)     # [12,12,64]
        y = torch.einsum("lhk,lhkm->lhm", v, model.W_O)    # [12,12,768]
        gains = y.norm(dim=-1)                             # [12,12]
        coss = (y @ tu) / gains.clamp_min(1e-12)
    return coss, gains


def direction_entry(name, t_row):
    coss, gains = ov_all_heads(t_row)
    flat = [(float(coss[layer, h]), float(gains[layer, h]), layer, h)
            for layer in range(N_LAYERS) for h in range(N_HEADS)]
    by_cos = sorted(flat)                                  # ascending cos
    rank = {(layer, h): i + 1 for i, (_, _, layer, h) in enumerate(by_cos)}
    entry = {
        "direction": name,
        "all_heads": [{"layer": layer, "head": h,
                       "cos": round(float(coss[layer, h]), 6),
                       "gain": round(float(gains[layer, h]), 6)}
                      for layer in range(N_LAYERS) for h in range(N_HEADS)],
        "n_heads_cos_below_-0.5": int((coss < -0.5).sum()),
        "n_heads_cos_below_0": int((coss < 0).sum()),
        "most_negative_head": {"layer": by_cos[0][2], "head": by_cos[0][3],
                               "cos": by_cos[0][0], "gain": by_cos[0][1]},
        "top5_most_negative": [{"layer": layer, "head": h, "cos": c, "gain": g}
                               for c, g, layer, h in by_cos[:5]],
        "mean_cos": float(coss.mean()),
        "mean_abs_cos": float(coss.abs().mean()),
        "tracked_heads": {},
    }
    for layer, h in TRACK_HEADS:
        entry["tracked_heads"][f"L{layer}.H{h}"] = {
            "cos": float(coss[layer, h]), "gain": float(gains[layer, h]),
            "rank_by_cos_ascending": rank[(layer, h)],
        }
    return entry


def head8_response(base_cache_fn, eps_rel, base_tensor):
    """FD response of head-8 output to eps * D_sym_hat, exp-08 convention:
    delta per unit eps_abs of the full-tensor-unit hinge, measured at the
    last row against e_sym."""
    eps_abs = eps_rel * float(base_tensor.norm())
    cb = base_cache_fn(base_tensor)
    cp = base_cache_fn(base_tensor + eps_abs * D_sym_hat)
    dz = (cp[HOOK_Z11] - cb[HOOK_Z11]) / eps_abs
    with torch.no_grad():
        contrib = dz[:, 8, :] @ model.W_O[11, 8]
    return {
        "eps_rel": eps_rel,
        "eps_abs": eps_abs,
        "d_component_last": float(contrib[-1] @ e_sym),
        "cos_last_vs_dsym": fcos(contrib[-1], e_sym),
        "gain_frobenius": float(contrib.norm()),
        "ln1_scale_base_last_row": float(
            cb["blocks.11.ln1.hook_scale"][-1]),
    }, cb, cp


def stage_test1():
    print("TEST 1: OV circuit", flush=True)
    t1 = {"directions": {}}
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    dirs = [("d_sym", e_sym.clone()),
            ("d_committed", e_committed.clone()),
            ("pole_A_plus_d_sym", e_sym.clone()),
            ("pole_B_minus_d_sym", -e_sym.clone())]
    for i in range(5):
        r = torch.randn(DM, generator=gen)
        dirs.append((f"random_{i}", unit(r)))
    for name, t_row in dirs:
        entry = direction_entry(name, t_row)
        entry["cos_vs_d_sym"] = fcos(t_row, e_sym)
        t1["directions"][name] = entry
        th = entry["tracked_heads"]
        print(f"  [{name}] L11.H8 cos {th['L11.H8']['cos']:+.4f} gain "
              f"{th['L11.H8']['gain']:.3f} rank {th['L11.H8']['rank_by_cos_ascending']}"
              f" | L10.H7 cos {th['L10.H7']['cos']:+.4f} | "
              f"n(cos<-0.5) = {entry['n_heads_cos_below_-0.5']}", flush=True)

    # Empirical operating-point checks, exp-08 convention throughout.
    emp = {}
    # (i) exp-08 reproduction: perturbation enters at blocks.0.hook_resid_pre
    # with base Mn_sym; the loop's own re-entry point.
    names_i = (HOOK_Z11, HOOK_PRE11, "blocks.11.ln1.hook_scale")
    def base_fn_block0(tensor):
        return run_inject_at(HOOK_WRITE, tensor, names_i)
    emp["block0_injection_at_Mn_sym"] = {"runs": [], "note":
        "reproduces exp 08 part 2 (base_Mn_sym__dir_d_sym per_head head 8)"}
    v11 = None
    for eps_rel in EPS_RELS:
        r, cb, cp = head8_response(base_fn_block0, eps_rel, Mn_sym)
        if eps_rel == EPS_RELS[0]:
            v11 = (cp[HOOK_PRE11] - cb[HOOK_PRE11]) / r["eps_abs"]
        emp["block0_injection_at_Mn_sym"]["runs"].append(r)
        print(f"  [empirical i] block-0 inject, eps {eps_rel:g}: head-8 "
              f"d-component {r['d_component_last']:+.4f} (exp08 recorded "
              f"{EXP08_H8_DCOMP:+.4f}), cos {r['cos_last_vs_dsym']:+.4f}",
              flush=True)
    emp["block0_injection_at_Mn_sym"]["exp08_recorded"] = EXP08_H8_DCOMP
    # Content of the cascade delta arriving at layer 11 (for the chain).
    emp["cascade_delta_at_pre11"] = {
        "d_component_last_row": float(v11[-1] @ e_sym),
        "cos_last_row_vs_dsym": fcos(v11[-1], e_sym),
        "full_tensor_component_along_Dsym_hat": float(
            (v11.flatten() @ D_sym_hat.flatten())),
        "frobenius_norm": float(v11.norm()),
    }
    # (ii) layer-11 injection at Mn_sym (the literal operating point x = M).
    def base_fn_pre11(tensor):
        return run_inject_at(HOOK_PRE11, tensor, (HOOK_Z11,
                             "blocks.11.ln1.hook_scale"))
    emp["layer11_injection_at_Mn_sym"] = {"runs": [], "note":
        "x = M delivered directly to layer 11 input; per unit full-tensor d"}
    for eps_rel in EPS_RELS:
        r, _, _ = head8_response(base_fn_pre11, eps_rel, Mn_sym)
        emp["layer11_injection_at_Mn_sym"]["runs"].append(r)
        print(f"  [empirical ii] pre-11 inject at Mn_sym, eps {eps_rel:g}: "
              f"d-component {r['d_component_last']:+.4f}, cos "
              f"{r['cos_last_vs_dsym']:+.4f}, ln1 scale "
              f"{r['ln1_scale_base_last_row']:.2f}", flush=True)
    # (iii) layer-11 injection at the cascade operating point resid_pre_11.
    P11_cascade = base_fn_block0(Mn_sym)[HOOK_PRE11]
    emp["layer11_injection_at_cascade_pre11"] = {
        "base_row_norm": float(P11_cascade[-1].norm()),
        "base_row_spread": row_spread(P11_cascade),
        "runs": [], "note":
        "pure d_sym delivered at the resid_pre_11 the Mn_sym cascade produces"}
    for eps_rel in EPS_RELS:
        r, _, _ = head8_response(base_fn_pre11, eps_rel, P11_cascade)
        emp["layer11_injection_at_cascade_pre11"]["runs"].append(r)
        print(f"  [empirical iii] pre-11 inject at cascade point, eps "
              f"{eps_rel:g}: d-component {r['d_component_last']:+.4f}, cos "
              f"{r['cos_last_vs_dsym']:+.4f}, ln1 scale "
              f"{r['ln1_scale_base_last_row']:.2f}", flush=True)
    # Chain: (i) should be approximately (iii) times the d_sym content of the
    # cascade delta at pre-11 (off-d components account for the remainder).
    c3 = emp["layer11_injection_at_cascade_pre11"]["runs"][0]["d_component_last"]
    mult = emp["cascade_delta_at_pre11"]["full_tensor_component_along_Dsym_hat"]
    emp["chain_check"] = {
        "iii_times_cascade_dsym_content": c3 * mult,
        "i_measured": emp["block0_injection_at_Mn_sym"]["runs"][0][
            "d_component_last"],
        "note": ("product of the pure-d layer-11 response and the d content "
                 "of the arriving cascade delta, vs the end-to-end "
                 "measurement; the gap is the head's response to the "
                 "cascade delta's off-d components"),
    }
    print(f"  [chain] iii * d-content = {emp['chain_check']['iii_times_cascade_dsym_content']:+.4f}"
          f" vs i = {emp['chain_check']['i_measured']:+.4f}", flush=True)
    t1["empirical"] = emp
    results["test1"] = t1


# ================================================================== test2 ===
def stage_test2(run_name, ablate_head, n_target):
    """ATR loop from state_divine.pt with optional head ablation at L11."""
    label = (f"ablate L11.H{ablate_head}" if ablate_head is not None
             else "no ablation")
    print(f"TEST 2 [{run_name}]: {label}, {n_target} iterations", flush=True)
    ck_path = os.path.join(OUT, f"test2_{run_name}_checkpoint.pt")
    t2 = results.setdefault("test2", {})
    run = t2.get(run_name)
    if run is not None and run.get("n_done", 0) >= n_target and run.get(
            "complete", False):
        print(f"  [{run_name}] already complete", flush=True)
        return
    if run is not None and os.path.exists(ck_path) and run.get("n_done", 0) > 0:
        ck = torch.load(ck_path, weights_only=True)
        cur = ck["current_tensor"]
        prev_row = ck["prev_row"]
        tail_rows = [r for r in ck["tail_rows"]]
        start = int(ck["n_done"]) + 1
        # The checkpoint is the single source of truth for the records too,
        # so tensor state and records can never desynchronise across a crash
        # (older checkpoints without a records key fall back to the JSON).
        records = ck.get("records", run["records"])
        run["records"] = records
        run["n_done"] = int(ck["n_done"])
        print(f"  [{run_name}] resuming at iteration {start}", flush=True)
    else:
        cur = A_full.clone()
        prev_row = A_full[-1].clone()
        tail_rows = []
        start = 1
        records = []
        run = {"label": label, "ablate_head": ablate_head,
               "n_target": n_target, "records": records, "n_done": 0,
               "complete": False}
        t2[run_name] = run

    def zero_head(z, hook, head=ablate_head):
        z[:, :, head, :] = 0.0
        return z
    hooks = [] if ablate_head is None else [(HOOK_Z11, zero_head)]

    # One-time verification that the hook removes exactly the head's additive
    # output from the layer (first iteration of the ablated runs).
    if ablate_head is not None and start == 1:
        cfull = step(cur, (HOOK_Z11, "blocks.11.hook_attn_out", HOOK_READ))
        cabl = step(cur, ("blocks.11.hook_attn_out", HOOK_READ),
                    extra_hooks=hooks)
        with torch.no_grad():
            pred = cfull[HOOK_Z11][:, ablate_head, :] @ model.W_O[11, ablate_head]
            got = cfull["blocks.11.hook_attn_out"] - cabl["blocks.11.hook_attn_out"]
            rel = float((got - pred).norm() / pred.norm())
            resid_change = float(
                (cfull[HOOK_READ] - cabl[HOOK_READ]).norm())
        run["ablation_verification"] = {
            "attn_out_delta_vs_headOV_rel_err": rel,
            "removed_component_norm": float(pred.norm()),
            "resid_post11_change_norm": resid_change,
        }
        print(f"  ablation check: attn_out delta matches z@W_O to rel err "
              f"{rel:.2e}; resid_post_11 moved {resid_change:.1f}", flush=True)

    t_run = time.time()
    for i in range(start, n_target + 1):
        out = step(cur, extra_hooks=hooks)[HOOK_READ]
        row = out[-1]
        with torch.no_grad():
            normalized = model.ln_final(out)
            logits = normalized @ model.W_U + model.b_U
            ids = logits.argmax(dim=-1)
        rec = {
            "i": i,
            "lag1_cos": fcos(row, prev_row),
            "cos_to_A": fcos(row, A_full[-1]),
            "cos_to_B": fcos(row, B_full[-1]),
            "cos_to_M_sym": fcos(row, M_sym_full[-1]),
            "cos_to_M_committed": fcos(row, M_com_full[-1]),
            "argmax_id_last": int(ids[-1]),
            "argmax_tok_last": model.tokenizer.decode([int(ids[-1])]),
            "n_unique_argmax_positions": int(len(set(int(x) for x in ids))),
            "out_norm": float(out.norm()),
            "row_spread": row_spread(out),
        }
        records.append(rec)
        prev_row = row.clone()
        tail_rows.append(unit(row.clone()))
        if len(tail_rows) > 24:
            tail_rows.pop(0)
        cur = out
        if i % 100 == 0 or i == n_target:
            run["n_done"] = i
            # One atomic artifact holds the tensor state AND the records at
            # the same iteration marker; the JSON below is derived output.
            torch.save({"current_tensor": cur, "prev_row": prev_row,
                        "tail_rows": torch.stack(tail_rows), "n_done": i,
                        "records": records},
                       ck_path + ".tmp")
            os.replace(ck_path + ".tmp", ck_path)
            atomic_save(results)
            print(f"  iter {i}: lag1 {rec['lag1_cos']:+.4f}, cosA "
                  f"{rec['cos_to_A']:+.4f}, cosB {rec['cos_to_B']:+.4f}, "
                  f"cosMsym {rec['cos_to_M_sym']:+.4f}, tok "
                  f"'{rec['argmax_tok_last']}'  "
                  f"[{time.time()-t_run:.0f}s]", flush=True)

    scan = lag_scan(torch.stack(tail_rows), max_lag=8)
    run["lag_scan_last24_last_row"] = {str(k): v for k, v in scan.items()}
    lastN = records[-10:]
    run["settled_fixed_point_last10_lag1_min"] = min(
        r["lag1_cos"] for r in lastN)
    run["final"] = {
        "iteration": records[-1]["i"],
        "cos_to_A": records[-1]["cos_to_A"],
        "cos_to_B": records[-1]["cos_to_B"],
        "cos_to_M_sym": records[-1]["cos_to_M_sym"],
        "cos_to_M_committed": records[-1]["cos_to_M_committed"],
        "top5_tokens": [[tok, float(p)] for tok, p in
                        get_top_tokens(model, cur[-1], k=5)],
    }
    run["complete"] = True
    run["n_done"] = n_target
    print(f"  [{run_name}] done: lag_scan(last 24) = "
          + " ".join(f"k{k}:{v:+.4f}" for k, v in scan.items()), flush=True)


# ================================================================== test3 ===
TEST3_PROMPTS = [
    # The five 04_readout_confidence prompts.
    "Am I sitting in a room different from the one you are in now",
    "The Eiffel Tower is located in the city of",
    "The cat sat on the mat and then the",
    "Flurb glex morp wintly skade",
    "Calculate the sum of all prime numbers below",
    # Seven ordinary English sentences, varied topics, 8 to 15 tokens.
    "The bridge was closed for repairs until further notice",
    "She poured the coffee and opened her laptop to read the news",
    "Rain fell steadily on the roof of the old barn",
    "The committee voted to approve the budget on Tuesday",
    "He tied his shoes and ran to catch the morning train",
    "The recipe calls for two eggs and a cup of flour",
    "Astronomers discovered a new comet beyond the orbit of Mars",
]
TEST3_HEADS = [(11, 8), (10, 7), (11, 0), (5, 5)]
ATTN_THRESHOLD = 0.2


def stage_test3():
    print("TEST 3: ordinary text", flush=True)
    cache_names = set()
    for layer, _ in TEST3_HEADS:
        cache_names.add(f"blocks.{layer}.attn.hook_pattern")
        cache_names.add(f"blocks.{layer}.attn.hook_z")
    rows = []
    prompt_meta = []
    for pi, sent in enumerate(TEST3_PROMPTS):
        toks = model.to_tokens(sent)                       # [1, L], BOS at 0
        L = toks.shape[1]
        prompt_meta.append({"index": pi, "text": sent,
                            "n_tokens_no_bos": L - 1})
        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks, names_filter=lambda n: n in cache_names)
        for layer, h in TEST3_HEADS:
            pat = cache[f"blocks.{layer}.attn.hook_pattern"][0][h]   # [L, L]
            z = cache[f"blocks.{layer}.attn.hook_z"][0]              # [L, nh, dh]
            with torch.no_grad():
                outs = z[:, h, :] @ model.W_O[layer, h]              # [L, 768]
            for t in range(2, L):
                p_row = pat[t]
                s = int(p_row[1:t + 1].argmax()) + 1
                a_top = float(p_row[s])
                tok_s = int(toks[0, s])
                out_t = outs[t]
                with torch.no_grad():
                    delta = float(out_t @ model.W_U[:, tok_s])
                    onorm = float(out_t.norm())
                rows.append({
                    "prompt": pi, "layer": layer, "head": h, "t": t, "s": s,
                    "s_is_t": s == t,
                    "attn_to_bos": float(p_row[0]),
                    "attn_to_top_source": a_top,
                    "token_at_s": model.tokenizer.decode([tok_s]),
                    "delta": delta,
                    "delta_per_unit_output": delta / max(onorm, 1e-12),
                    "head_output_norm": onorm,
                })
    per_head = {}
    for layer, h in TEST3_HEADS:
        sel = [r for r in rows if r["layer"] == layer and r["head"] == h]
        res = [r for r in sel if r["attn_to_top_source"] > ATTN_THRESHOLD]
        def agg(rs):
            if not rs:
                return {"n": 0}
            ds = [r["delta"] for r in rs]
            dn = [r["delta_per_unit_output"] for r in rs]
            return {
                "n": len(rs),
                "frac_delta_negative": sum(1 for d in ds if d < 0) / len(ds),
                "mean_delta": sum(ds) / len(ds),
                "mean_delta_per_unit_output": sum(dn) / len(dn),
                "min_delta": min(ds), "max_delta": max(ds),
            }
        per_head[f"L{layer}.H{h}"] = {
            "all_positions": agg(sel),
            f"attn_top_source_gt_{ATTN_THRESHOLD}": agg(res),
            "mean_attn_to_bos": sum(r["attn_to_bos"] for r in sel) / len(sel),
            "frac_top_source_is_self": sum(
                1 for r in sel if r["s_is_t"]) / len(sel),
        }
        a = per_head[f"L{layer}.H{h}"]["all_positions"]
        rst = per_head[f"L{layer}.H{h}"][f"attn_top_source_gt_{ATTN_THRESHOLD}"]
        print(f"  L{layer}.H{h}: n {a['n']}, frac neg {a['frac_delta_negative']:.3f}, "
              f"mean delta {a['mean_delta']:+.3f}; restricted n {rst['n']}, "
              f"frac neg {rst.get('frac_delta_negative', float('nan')):.3f}, "
              f"mean {rst.get('mean_delta', float('nan')):+.3f}; "
              f"BOS attn {per_head[f'L{layer}.H{h}']['mean_attn_to_bos']:.3f}",
              flush=True)
    results["test3"] = {
        "prompts": prompt_meta,
        "attn_threshold": ATTN_THRESHOLD,
        "per_head": per_head,
        "per_position_rows": rows,
        "note": ("delta = head output at t, through W_O, dotted with "
                 "W_U[:, token at s] where s is the head's top non-BOS "
                 "source; ln_final scaling (positive scalar per position) "
                 "is omitted, so signs are unaffected; W_O writes are "
                 "centered by the TransformerLens loading convention"),
    }


# =================================================================== main ===
STAGE_FNS = [
    ("gate", stage_gate),
    ("test1", stage_test1),
    ("test2_none", lambda: stage_test2("none", None, 100)),
    ("test2_h0", lambda: stage_test2("h0", 0, 100)),
    ("test2_h8", lambda: stage_test2("h8", 8, 300)),
    ("test3", stage_test3),
]

requested = sys.argv[1:] or [n for n, _ in STAGE_FNS]
known = {n for n, _ in STAGE_FNS}
bad = [r for r in requested if r not in known]
if bad:
    sys.exit(f"unknown stage(s) {bad}; valid: {sorted(known)}")

for name, fn in STAGE_FNS:
    if name not in requested:
        continue
    if name in results["stages_done"]:
        print(f"stage {name}: already done, skipping", flush=True)
        continue
    t_s = time.time()
    fn()
    results["stages_done"].append(name)
    atomic_save(results)
    print(f"stage {name} finished in {time.time()-t_s:.0f}s; checkpoint saved",
          flush=True)

print("ALL REQUESTED STAGES DONE. Results in", RESULTS_PATH, flush=True)
