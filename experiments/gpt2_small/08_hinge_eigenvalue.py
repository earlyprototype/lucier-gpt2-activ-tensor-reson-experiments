"""
EXP: Hinge eigenvalue of the Divine period-2 cycle, and which block inverts it
(issue #14, thread 1)

The bell anatomy (output_divine_motion/bell_anatomy.md) reads the Divine cycle
as a rank-1 self-negating mode: writing A = M + d and B = M - d, one hinge
direction d that the normalised ATR map inverts each pass, an effective
eigenvalue near -1. This experiment tests that conjecture directly and asks
which block (layer, attention or MLP) performs the inversion.

The map under test is the full ATR iteration exactly as atr_engine.run_atr_loop
implements it:

    f(x) = ForwardBlocks( x * (initial_norm / ||x||_F) )

where ForwardBlocks is blocks 0..11 of GPT-2 Small applied to a tensor injected
at blocks.0.hook_resid_pre (the injection overwrites every position, so the
forward from that point is exactly the block cascade; verified in-script
against the hook-based step to zero error).

States and directions:
  A_full = state_divine.pt current_tensor (raw output at iteration 1000)
  B_full = f(A_full) (raw), A2_full = f(B_full) (sanity gate: period 2)
  Committed hinge (exactly as 06_bell_anatomy.py builds it): Bn = B_full
  rescaled to initial_norm, d = (A_full[-1] - Bn[-1]) / 2, with full-tensor
  version D = (A_full - Bn) / 2 and pivot M_full = (A_full + Bn) / 2.
  NOTE: this mixes frames. A_full is raw (norm 5098) while Bn is on the shell
  (norm 1468.5), so the committed d is dominated by A's own direction
  (cos(D, A_full) = 0.97). The symmetric on-shell hinge is also measured:
  An = A_full rescaled to initial_norm, D_sym = (An - Bn) / 2,
  M_sym = (An + Bn) / 2. Both are reported everywhere.
  The state is position-collapsed (all 10 rows identical), so both hinges are
  row-uniform and the row-uniform subspace is invariant under f; last-position
  numbers and full-tensor numbers coincide.

Part 1, eigenvalue along the hinge. Directional derivative J v of f at the
pivot and at the cycle states, by forward-mode autodiff (torch.func.jvp) with
central finite differences at two epsilons (1e-3 and 1e-4 of the base point
norm) as robustness checks. Reported per point and tangent: lambda = <t, J t>
(t unit), amplification ||J t||, and cos(J t, -t). Because f is scale
invariant (f(cx) = f(x)), J_f(cx) = J_f(x) / c: evaluations at shell points
(norm = initial_norm, the tensor the loop actually injects) and at raw points
differ by the exact scalar initial_norm / ||raw||, which is measured and
verified. The dynamically correct object for cycle stability is the composed
linearisation J_f(B_raw) J_f(A_raw) at the raw states the iteration actually
visits; it is measured directly (jvp chain and FD chain), both cycle phases,
plus random orthogonal control directions (3 row-uniform, 2 generic).

Part 2, layer attribution. With the model at the normalised pivot Mn (and at
the symmetric pivot as a robustness base), inject Mn + eps * t at
blocks.0.hook_resid_pre (the exact point the loop re-enters; the loop's own
renormalisation sits immediately upstream of this point and is part 1's
business), run once with caches, and track delta_l = run(Mn + eps t)_l -
run(Mn)_l at every layer boundary (resid_pre 0..11 and final resid_post),
plus per-layer hook_attn_out and hook_mlp_out deltas and a per-head split
(z @ W_O) at the flip layer. Two epsilons (1e-3 and 1e-4 of the base norm).

Run (from experiments/gpt2_small/):
    python 08_hinge_eigenvalue.py
Writes output_hinge_eigen/hinge_eigenvalue.json (partial checkpoint after
part 1, final after part 2). The report is output_hinge_eigen/
hinge_eigenvalue.md.

If huggingface.co is unreachable, set ATR_GPT2_LOCAL to a directory
containing the standard gpt2 files (config.json, pytorch_model.bin,
vocab.json, merges.txt) and the script will load offline.
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DIVINE = os.path.join(HERE, "output_divine_motion")
OUT = os.path.join(HERE, "output_hinge_eigen")
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

PROMPT = "The cat sat on the mat and then the"
N_LAYERS = model.cfg.n_layers
HOOK_READ = f"blocks.{N_LAYERS - 1}.hook_resid_post"
HOOK_WRITE = "blocks.0.hook_resid_pre"
EPS_RELS = [1e-3, 1e-4]
CONTROL_SEED = 20260719

# ---------------------------------------------------------------- states ----
st = torch.load(os.path.join(DIVINE, "state_divine.pt"), weights_only=True)
A_full = st["current_tensor"]            # raw output at iteration 1000
N0 = float(st["initial_norm"])           # loop energy shell (Frobenius norm)
T, DM = A_full.shape


def step(x):
    """One ATR iteration, hook-based, exactly as 06_bell_anatomy.py runs it."""
    cur = x * (N0 / x.norm())
    inject = cur.clone()
    def h(resid, hook, tensor=inject):
        resid[0, :, :] = tensor
        return resid
    model.add_hook(HOOK_WRITE, h)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                PROMPT, names_filter=lambda n: n == HOOK_READ)
    finally:
        model.reset_hooks()
    return cache[HOOK_READ][0].clone()


def norm_to(x, n):
    return x * (n / x.norm())


def unit(x):
    return x / x.norm()


def fcos(a, b):
    return float(F.cosine_similarity(a.flatten().unsqueeze(0),
                                     b.flatten().unsqueeze(0)))


B_full = step(A_full)                    # raw phase B
A2_full = step(B_full)                   # raw, should equal A_full

Bn = norm_to(B_full, N0)
An = norm_to(A_full, N0)
A2n = norm_to(A2_full, N0)
A_last = A_full[-1]                      # 06's A (raw frame)
B_last = Bn[-1]                          # 06's B (shell frame)
A2_last = A2n[-1]

cosAB = fcos(A_last, B_last)
cosAA2 = fcos(A_last, A2_last)
cycle_residual = float((A_full - A2_full).norm())
print(f"GATE: cos(A,B) = {cosAB:.6f} (expect 0.684912), "
      f"cos(A, f(f(A))) = {cosAA2:.6f} (expect 1.0), "
      f"full-tensor cycle residual = {cycle_residual:.2e}", flush=True)
assert abs(cosAB - 0.6849116683) < 5e-4, "sanity gate failed: cos(A,B)"
assert cosAA2 > 0.99999, "sanity gate failed: cos(A, f(f(A)))"

# Committed hinge and pivot, exactly as 06_bell_anatomy.py builds them.
d_committed = (A_last - B_last) / 2
M_committed_last = (A_last + B_last) / 2
D_committed = (A_full - Bn) / 2          # full-tensor version (06 line 99)
M_committed_full = (A_full + Bn) / 2
# Symmetric on-shell hinge and pivot.
D_sym = (An - Bn) / 2
M_sym_full = (An + Bn) / 2

Mn_committed = norm_to(M_committed_full, N0)
Mn_sym = norm_to(M_sym_full, N0)

Dc_hat = unit(D_committed)
Ds_hat = unit(D_sym)

# Position collapse: how uniform are the rows?
def row_spread(x):
    mean_row = x.mean(dim=0)
    return float((x - mean_row).norm() / mean_row.norm())

geometry = {
    "N0_initial_norm": N0,
    "norm_A_raw": float(A_full.norm()),
    "norm_B_raw": float(B_full.norm()),
    "norm_M_committed_full": float(M_committed_full.norm()),
    "norm_M_sym_full": float(M_sym_full.norm()),
    "norm_d_committed_last": float(d_committed.norm()),
    "norm_D_committed_full": float(D_committed.norm()),
    "norm_D_sym_full": float(D_sym.norm()),
    "per_position_norm_A_raw": float(A_full[-1].norm()),
    "per_position_norm_Bn": float(Bn[-1].norm()),
    "row_spread_A_raw": row_spread(A_full),
    "row_spread_Bn": row_spread(Bn),
    "row_spread_D_committed": row_spread(D_committed),
    "cos_Dcommitted_vs_Dsym": fcos(D_committed, D_sym),
    "cos_Dcommitted_vs_Araw": fcos(D_committed, A_full),
    "cos_Dcommitted_vs_Mcommitted": fcos(D_committed, M_committed_full),
    "cos_Dsym_vs_An": fcos(D_sym, An),
    "cos_Dsym_vs_Bn": fcos(D_sym, Bn),
    "cos_Dsym_vs_Msym": fcos(D_sym, M_sym_full),
    "cos_Dsym_vs_Mcommitted": fcos(D_sym, M_committed_full),
    "cos_Mcommitted_vs_Araw": fcos(M_committed_full, A_full),
    "cos_Msym_vs_An": fcos(M_sym_full, An),
    "shell_factor_at_A_raw": N0 / float(A_full.norm()),
    "shell_factor_at_B_raw": N0 / float(B_full.norm()),
    "shell_factor_at_M_committed": N0 / float(M_committed_full.norm()),
}
print("geometry:", json.dumps(
    {k: round(v, 6) for k, v in geometry.items()}, indent=1), flush=True)

# ------------------------------------------------- differentiable ATR map ----
def f_map(x2d):
    """Full ATR iteration as a pure function of the [T, d_model] state:
    renormalise to the shell, then the block cascade (equals the hook-based
    step because the injection overwrites every position of resid_pre 0)."""
    xn = x2d * (N0 / x2d.norm())
    resid = xn.unsqueeze(0)
    for blk in model.blocks:
        resid = blk(resid)
    return resid[0]


with torch.no_grad():
    equiv = float((f_map(A_full) - B_full).norm())
print(f"pure-forward equivalence: ||f_map(A) - step(A)|| = {equiv:.3e}",
      flush=True)
assert equiv < 1e-3 * float(B_full.norm()), "f_map does not match hook step"


def jvp_dir(x, t):
    """J_f(x) t by forward-mode autodiff. Exact directional derivative."""
    _, jv = torch.func.jvp(f_map, (x,), (t,))
    return jv.detach()


def fd_dir(x, t_unit, h):
    """Central finite difference (f(x + h t) - f(x - h t)) / (2 h)."""
    with torch.no_grad():
        return (f_map(x + h * t_unit) - f_map(x - h * t_unit)) / (2.0 * h)


def stats(t_unit, Jt):
    return {
        "lambda": float(t_unit.flatten() @ Jt.flatten()),
        "amplification": float(Jt.norm()),
        "cos_Jt_vs_minus_t": fcos(Jt, -t_unit),
    }


def measure_halfmap(x, t_unit, label):
    base_norm = float(x.norm())
    out = {"base_norm": base_norm}
    out["jvp"] = stats(t_unit, jvp_dir(x, t_unit))
    for rel in EPS_RELS:
        h = rel * base_norm
        out[f"fd_eps_{rel:g}"] = dict(stats(t_unit, fd_dir(x, t_unit, h)),
                                      eps_abs=h)
    print(f"  [halfmap] {label}: lambda_jvp = {out['jvp']['lambda']:+.4f}, "
          f"amp = {out['jvp']['amplification']:.4f}, "
          f"cos(Jt, -t) = {out['jvp']['cos_Jt_vs_minus_t']:+.4f}", flush=True)
    return out


def measure_composed(xA, xB, t_unit, label, with_fd=True):
    """w = J_f(xB) J_f(xA) t, jvp chain plus optional FD chain."""
    out = {}
    v = jvp_dir(xA, t_unit)
    w = jvp_dir(xB, v)
    out["jvp"] = {
        "lambda_composed": float(t_unit.flatten() @ w.flatten()),
        "amplification_composed": float(w.norm()),
        "cos_w_vs_t": fcos(w, t_unit),
        "intermediate_v_norm": float(v.norm()),
        "cos_v_vs_minus_t": fcos(v, -t_unit),
    }
    if with_fd:
        for rel in EPS_RELS:
            v1 = fd_dir(xA, t_unit, rel * float(xA.norm()))
            vn = float(v1.norm())
            w1 = vn * fd_dir(xB, unit(v1), rel * float(xB.norm()))
            out[f"fd_eps_{rel:g}"] = {
                "lambda_composed": float(t_unit.flatten() @ w1.flatten()),
                "amplification_composed": float(w1.norm()),
                "cos_w_vs_t": fcos(w1, t_unit),
                "intermediate_v_norm": vn,
                "cos_v_vs_minus_t": fcos(v1, -t_unit),
            }
    j = out["jvp"]
    print(f"  [composed] {label}: lambda = {j['lambda_composed']:+.4f}, "
          f"amp = {j['amplification_composed']:.4f}, "
          f"cos(w, t) = {j['cos_w_vs_t']:+.4f}, "
          f"|v| = {j['intermediate_v_norm']:.4f}", flush=True)
    return out


# ------------------------------------------------------- control tangents ----
gen = torch.Generator().manual_seed(CONTROL_SEED)
controls = []
basis = [Dc_hat.flatten(), Ds_hat.flatten()]
for i in range(5):
    row_uniform = i < 3
    if row_uniform:
        r = torch.randn(DM, generator=gen)
        r = r.unsqueeze(0).expand(T, DM).clone()
    else:
        r = torch.randn(T, DM, generator=gen)
    rf = r.flatten()
    for b in basis:
        rf = rf - (rf @ b) * b
    rf = rf / rf.norm()
    basis.append(rf)
    controls.append({
        "name": f"control_{i}_{'uniform' if row_uniform else 'generic'}",
        "row_uniform": row_uniform,
        "tangent": rf.view(T, DM),
        "cos_vs_Mn_committed": fcos(rf.view(T, DM), Mn_committed),
    })

# ------------------------------------------------------------------ part 1 ----
print("PART 1: eigenvalue along the hinge", flush=True)
part1 = {"half_map": {}, "composed_cycle": {}, "controls": []}

for dname, that in [("d_committed", Dc_hat), ("d_sym", Ds_hat)]:
    blockres = {}
    blockres["at_M_committed_shell"] = measure_halfmap(
        Mn_committed, that, f"{dname} at Mn_committed")
    blockres["at_M_sym_shell"] = measure_halfmap(
        Mn_sym, that, f"{dname} at Mn_sym")
    blockres["at_A_shell"] = measure_halfmap(An, that, f"{dname} at An")
    blockres["at_B_shell"] = measure_halfmap(Bn, that, f"{dname} at Bn")
    blockres["raw_frame_derived"] = {
        "note": ("f is scale invariant, so J_f at a raw point equals J_f at "
                 "the shell point times initial_norm/||raw||; multiply shell "
                 "lambda and amplification by the factors below"),
        "factor_at_A_raw": geometry["shell_factor_at_A_raw"],
        "factor_at_B_raw": geometry["shell_factor_at_B_raw"],
        "lambda_at_A_raw": (blockres["at_A_shell"]["jvp"]["lambda"]
                            * geometry["shell_factor_at_A_raw"]),
        "lambda_at_B_raw": (blockres["at_B_shell"]["jvp"]["lambda"]
                            * geometry["shell_factor_at_B_raw"]),
    }
    part1["half_map"][dname] = blockres

# Direct raw-point evaluation to verify the scale-invariance identity.
raw_jv = stats(Dc_hat, jvp_dir(A_full, Dc_hat))
part1["scale_invariance_check"] = {
    "lambda_at_A_raw_direct_jvp": raw_jv["lambda"],
    "lambda_at_A_raw_derived_from_shell":
        part1["half_map"]["d_committed"]["raw_frame_derived"]["lambda_at_A_raw"],
    "amplification_at_A_raw_direct": raw_jv["amplification"],
}
print(f"  [scale check] direct raw lambda = {raw_jv['lambda']:+.5f} vs derived "
      f"{part1['scale_invariance_check']['lambda_at_A_raw_derived_from_shell']:+.5f}",
      flush=True)

# Composed cycle map at the raw states the iteration actually visits.
for dname, that in [("d_committed", Dc_hat), ("d_sym", Ds_hat)]:
    part1["composed_cycle"][dname] = {
        "start_A": measure_composed(A_full, B_full, that,
                                    f"{dname} A->B->A"),
        "start_B": measure_composed(B_full, A_full, that,
                                    f"{dname} B->A->B", with_fd=False),
    }

for c in controls:
    entry = {
        "name": c["name"],
        "row_uniform": c["row_uniform"],
        "cos_vs_Mn_committed": c["cos_vs_Mn_committed"],
        "at_M_committed_shell": measure_halfmap(
            Mn_committed, c["tangent"], c["name"] + " at Mn_committed"),
        "at_M_sym_shell": {"jvp": stats(c["tangent"],
                                        jvp_dir(Mn_sym, c["tangent"]))},
        "composed_start_A": measure_composed(
            A_full, B_full, c["tangent"], c["name"] + " A->B->A",
            with_fd=False),
    }
    print(f"  [halfmap] {c['name']} at Mn_sym: lambda_jvp = "
          f"{entry['at_M_sym_shell']['jvp']['lambda']:+.4f}, amp = "
          f"{entry['at_M_sym_shell']['jvp']['amplification']:.4f}", flush=True)
    part1["controls"].append(entry)

# Supplementary: how close is each pivot to a fixed point of f? A flip-unstable
# near-fixed pivot plus a contracting composed map is the period-doubling
# picture; these residuals locate the pivot relative to that reading.
with torch.no_grad():
    fMc = f_map(Mn_committed)
    fMs = f_map(Mn_sym)
part1["pivot_fixed_point_residuals"] = {
    "Mn_committed": {
        "cos_fM_vs_M": fcos(fMc, Mn_committed),
        "cos_fM_vs_An": fcos(fMc, An),
        "cos_fM_vs_Bn": fcos(fMc, Bn),
        "norm_fM": float(fMc.norm()),
        "residual_norm_after_renorm": float(
            (norm_to(fMc, N0) - Mn_committed).norm()),
    },
    "Mn_sym": {
        "cos_fM_vs_M": fcos(fMs, Mn_sym),
        "cos_fM_vs_An": fcos(fMs, An),
        "cos_fM_vs_Bn": fcos(fMs, Bn),
        "norm_fM": float(fMs.norm()),
        "residual_norm_after_renorm": float(
            (norm_to(fMs, N0) - Mn_sym).norm()),
    },
}
print("  [pivot residuals]", json.dumps(
    {k: {kk: round(vv, 4) for kk, vv in v.items()}
     for k, v in part1["pivot_fixed_point_residuals"].items()}), flush=True)

results = {
    "meta": {
        "issue": 14,
        "thread": 1,
        "date": "2026-07-19",
        "model": "gpt2-small (TransformerLens, offline via ATR_GPT2_LOCAL)",
        "prompt": PROMPT,
        "seq_len": T,
        "d_model": DM,
        "gate": {"cosAB": cosAB, "cosAA2": cosAA2,
                 "full_tensor_cycle_residual": cycle_residual},
        "geometry": geometry,
        "eps_rels": EPS_RELS,
        "control_seed": CONTROL_SEED,
        "pure_forward_equivalence_l2": equiv,
        "frame_note": ("06_bell_anatomy.py builds d from raw A (norm "
                       f"{geometry['norm_A_raw']:.0f}) and shell B (norm "
                       f"{N0:.0f}); the committed d is therefore "
                       f"{geometry['cos_Dcommitted_vs_Araw']:.3f} aligned "
                       "with A's own direction and only "
                       f"{geometry['cos_Dcommitted_vs_Dsym']:.3f} aligned "
                       "with the symmetric on-shell hinge (An - Bn)/2. "
                       "Both are measured throughout."),
    },
    "part1": part1,
    "part2": None,
}
with open(os.path.join(OUT, "hinge_eigenvalue.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print("part 1 checkpoint saved", flush=True)

# ------------------------------------------------------------------ part 2 ----
print("PART 2: layer attribution", flush=True)

CACHE_NAMES = set()
for layer in range(N_LAYERS):
    CACHE_NAMES.add(f"blocks.{layer}.hook_resid_pre")
    CACHE_NAMES.add(f"blocks.{layer}.hook_attn_out")
    CACHE_NAMES.add(f"blocks.{layer}.hook_mlp_out")
    CACHE_NAMES.add(f"blocks.{layer}.attn.hook_z")
CACHE_NAMES.add(HOOK_READ)


def run_cached(inject):
    """Forward with the tensor injected at blocks.0.hook_resid_pre (the exact
    point run_atr_loop re-enters), caching every layer boundary."""
    inject = inject.clone()
    def h(resid, hook, tensor=inject):
        resid[0, :, :] = tensor
        return resid
    model.add_hook(HOOK_WRITE, h)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                PROMPT, names_filter=lambda n: n in CACHE_NAMES)
    finally:
        model.reset_hooks()
    return {n: cache[n][0].detach().clone() for n in CACHE_NAMES}


def layer_tables(base_cache, pert_cache, eps_abs, e_row, D_hat):
    """Per-boundary and per-block deltas, d components measured at the last
    position against the unit row direction e_row, full-tensor against D_hat."""
    boundaries = []
    for layer in range(N_LAYERS + 1):
        name = (f"blocks.{layer}.hook_resid_pre" if layer < N_LAYERS else HOOK_READ)
        delta = (pert_cache[name] - base_cache[name]) / eps_abs
        boundaries.append({
            "boundary": (f"resid_pre_{layer}" if layer < N_LAYERS
                         else f"resid_post_{N_LAYERS - 1}"),
            "cos_last_vs_d": fcos(delta[-1], e_row),
            "cos_full_vs_D": fcos(delta, D_hat),
            "gain_frobenius_per_unit_input": float(delta.norm()),
            "d_component_last": float(delta[-1] @ e_row),
        })
    blocks = []
    for layer in range(N_LAYERS):
        da = (pert_cache[f"blocks.{layer}.hook_attn_out"]
              - base_cache[f"blocks.{layer}.hook_attn_out"]) / eps_abs
        dm = (pert_cache[f"blocks.{layer}.hook_mlp_out"]
              - base_cache[f"blocks.{layer}.hook_mlp_out"]) / eps_abs
        blocks.append({
            "layer": layer,
            "attn_d_component_last": float(da[-1] @ e_row),
            "attn_cos_last_vs_d": fcos(da[-1], e_row),
            "attn_gain": float(da.norm()),
            "mlp_d_component_last": float(dm[-1] @ e_row),
            "mlp_cos_last_vs_d": fcos(dm[-1], e_row),
            "mlp_gain": float(dm.norm()),
        })
    # First boundary where the cumulative d component goes negative.
    flip = None
    for prev, cur in zip(boundaries[:-1], boundaries[1:]):
        if prev["cos_last_vs_d"] > 0 and cur["cos_last_vs_d"] < 0:
            flip = int(prev["boundary"].split("_")[-1])
            break
    return boundaries, blocks, flip


def per_head_split(base_cache, pert_cache, layer, eps_abs, e_row):
    zb = base_cache[f"blocks.{layer}.attn.hook_z"]
    zp = pert_cache[f"blocks.{layer}.attn.hook_z"]
    dz = (zp - zb) / eps_abs                      # [pos, head, d_head]
    heads = []
    for h_i in range(model.cfg.n_heads):
        contrib = dz[:, h_i, :] @ model.W_O[layer, h_i]   # [pos, d_model]
        heads.append({
            "head": h_i,
            "d_component_last": float(contrib[-1] @ e_row),
            "cos_last_vs_d": fcos(contrib[-1], e_row),
            "gain": float(contrib.norm()),
        })
    return heads


part2 = {"configs": []}
e_committed = unit(D_committed[-1])
e_sym = unit(D_sym[-1])
# Loop-faithful committed-d tangent: perturbing the pre-normalisation state by
# d makes the loop inject the tangentially projected d (the renormalisation
# strips the radial part before the network sees it).
Mc_hat = unit(Mn_committed)
Dc_tang = Dc_hat - float(Dc_hat.flatten() @ Mc_hat.flatten()) * Mc_hat
Dc_tang = unit(Dc_tang)
part2["d_committed_tangential_note"] = {
    "definition": ("unit component of d_committed orthogonal to the committed "
                   "pivot: what the loop renormalisation passes through when "
                   "the pre-normalisation state is perturbed by d_committed"),
    "cos_vs_d_sym": fcos(Dc_tang, Ds_hat),
    "tangential_fraction_of_d_committed": float(
        (Dc_hat - float(Dc_hat.flatten() @ Mc_hat.flatten()) * Mc_hat).norm()),
}
CONFIGS = [
    ("base_Mn_committed__dir_d_committed", Mn_committed, Dc_hat, e_committed),
    ("base_Mn_committed__dir_d_committed_tangential", Mn_committed, Dc_tang,
     unit(Dc_tang[-1])),
    ("base_Mn_committed__dir_d_sym", Mn_committed, Ds_hat, e_sym),
    ("base_Mn_sym__dir_d_sym", Mn_sym, Ds_hat, e_sym),
]

base_caches = {}
for cfg_name, base, D_hat, e_row in CONFIGS:
    base_key = id(base)
    if base_key not in base_caches:
        base_caches[base_key] = run_cached(base)
        got = base_caches[base_key]["blocks.0.hook_resid_pre"]
        inj_err = float((got - base).norm())
        assert inj_err < 1e-4, f"injection not reflected in cache: {inj_err}"
    base_cache = base_caches[base_key]
    cfg_out = {"name": cfg_name, "base_norm": float(base.norm()),
               "eps_runs": []}
    for rel in EPS_RELS:
        eps_abs = rel * float(base.norm())
        pert_cache = run_cached(base + eps_abs * D_hat)
        boundaries, blocks, flip = layer_tables(
            base_cache, pert_cache, eps_abs, e_row, D_hat)
        run_out = {"eps_rel": rel, "eps_abs": eps_abs,
                   "boundaries": boundaries, "blocks": blocks,
                   "flip_layer": flip}
        if flip is not None:
            run_out["per_head_at_flip"] = per_head_split(
                base_cache, pert_cache, flip, eps_abs, e_row)
        cfg_out["eps_runs"].append(run_out)
        print(f"  [{cfg_name}] eps_rel {rel:g}: flip layer = {flip}",
              flush=True)
        print("    boundary cos:", " ".join(
            f"{b['cos_last_vs_d']:+.3f}" for b in boundaries), flush=True)
    part2["configs"].append(cfg_out)

results["part2"] = part2
with open(os.path.join(OUT, "hinge_eigenvalue.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print("DONE. Results in", OUT, flush=True)

# ------------------------------------------------------------- summary ----
hm = part1["half_map"]
cc = part1["composed_cycle"]
print("SUMMARY")
for dname in ["d_committed", "d_sym"]:
    m = hm[dname]["at_M_committed_shell"]["jvp"]
    a = hm[dname]["at_A_shell"]["jvp"]
    b = hm[dname]["at_B_shell"]["jvp"]
    comp = cc[dname]["start_A"]["jvp"]
    print(f"  {dname}: lambda at Mn = {m['lambda']:+.4f} "
          f"(cos flip {m['cos_Jt_vs_minus_t']:+.3f}), at An = {a['lambda']:+.4f}, "
          f"at Bn = {b['lambda']:+.4f}; composed = "
          f"{comp['lambda_composed']:+.4f} (cos {comp['cos_w_vs_t']:+.3f})",
          flush=True)
for cfg in part2["configs"]:
    flips = [r["flip_layer"] for r in cfg["eps_runs"]]
    print(f"  {cfg['name']}: flip layer(s) = {flips}", flush=True)
