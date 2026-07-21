"""
EXP: Phase-aware J-lens membership probe of the Divine bell (issue #8 follow-up)

The J-lens pilot (05_jlens_pilot.py, output_jlens_pilot/) probed the converged
Syntactic ("Divine") state against a pilot J-lens dictionary and found no
prolet-inside / Divine-outside split, only a weak language-vs-noise boundary.
After that pilot, 06_bell_anatomy.py showed the Divine state is an exact
period-2 limit cycle: phases A and B, pivot M = (A+B)/2, hinge d = (A-B)/2,
with the hinge 95 percent mute to the readout (logit response ratio 0.054).
The pilot therefore probed only phase A of a two-phase object.

This script re-runs the pilot's membership probe, UNCHANGED, on both phases
and the pivot, and asks one new question: is the hinge direction d inside or
outside the pilot lens subspace?

Stage 1 (needs the model; a few forward passes only):
  - Reconstruct A, B, M, d exactly as 06_bell_anatomy.py does: A is the last
    position of the saved iteration-1000 tensor (state_divine.pt), B is the
    last position of f(A) renormalised to the loop's initial norm, A2 = f(f(A))
    verifies period 2.
  - Sanity gates (hard exit on failure): cos(A, B) and cos(A, f(f(A))) must
    match bell_anatomy.json; A must match the state the pilot probed
    (converged_tensors.pt "Syntactic", last position).
  - Regenerate the pilot's three noise states bit-identically (the pilot
    generated them in-script, seed 2026; they were never committed).
  - Decompose d against the unembedding's singular directions (top-100 and
    bottom-100 splits, mirroring 06_bell_anatomy.py) and gate against the
    recorded d_top100 / d_bot100 energies.
  - Checkpoint everything to output_jlens_phase/phase_states.pt so stage 2
    never needs the model.

Stage 2 (no model; pure linear algebra on the committed lens):
  - Replay the pilot probe verbatim (least-squares span share, nonnegative
    sparse k=25 share, random-dictionary controls from a generator seeded
    4242) on the pilot's eight states IN THE PILOT'S ORDER, so the random
    control stream aligns; gate the lens columns against
    jlens_pilot_results.json (reproduction of the pilot's Divine number).
  - Continue the same probe onto phase B, pivot M, and (supplementary) the
    committed 05_divine_motion noise state (seed 42, iteration 1000).
  - Probe the direction d with the span probe (sign-invariant); the sparse
    probe is reported for both +d and -d because nonnegativity is
    sign-dependent. A 20-random-direction baseline (seed 777) calibrates
    what a generic unit direction scores against each layer dictionary.
    The top-100 (readout-visible) and bottom-100 (readout-quiet) W_U
    components of d are span-probed separately.

Stage 3 (review fix; pure linear algebra unless the W_U cache is missing):
  - The stage-2 direction probe ran on the COMMITTED flip axis d, which
    06_bell_anatomy.py builds by mixing frames (raw-scale A with
    shell-normalised B), so that d is 0.909 aligned with its own pivot M:
    mostly radial contamination. The physical cycle axis is the SYMMETRIC
    on-shell axis d_sym = normalise(An - Bn), both phases rescaled to the
    loop shell N0, orthogonal to its pivot M_sym (see
    output_hinge_eigen/hinge_eigenvalue.md, "The map, the frames, and two
    flip axes"). Stage 3 reconstructs d_sym from the stage-1 checkpoint,
    gates it against hinge_eigenvalue.json (orthogonality to M_sym;
    cos(d_sym, d_committed) = 0.616; cos with the tangentialised committed
    d = 0.973), and runs the same probes on it: per-layer span share,
    sparse share for both signs, the seed-777 generic-direction baseline
    (gated to replay stage 2 exactly), and span probes of d_sym's top-100
    and bottom-100 W_U singular components (the components need the model
    once; they are cached back into phase_states.pt, so re-runs are pure
    linear algebra). Results are appended to jlens_phase.json under
    direction_probe_sym; the stage-2 numbers are kept and relabelled
    d_committed in the report.

The lens itself is REUSED from output_jlens_pilot/jlens_vectors.pt; its
corpus is unreachable on this network, so no lens recomputation is attempted.

Outputs: output_jlens_phase/{phase_states.pt, jlens_phase.json} plus
markdown-ready tables on stdout (report: output_jlens_phase/jlens_phase.md).

Run:  ATR_GPT2_LOCAL=<gpt2 dir> python 10_jlens_phase.py [stage1|stage2|stage3|all]
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "output_jlens_phase")
DM = os.path.join(HERE, "output_divine_motion")
JPILOT = os.path.join(HERE, "output_jlens_pilot")
CONF = os.path.join(HERE, "output_confidence")
HEIG = os.path.join(HERE, "output_hinge_eigen")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F
# Same hardware note as 05/06: multi-threaded BLAS is slower here. One thread.
torch.set_num_threads(1)

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
STATES_PT = os.path.join(OUT, "phase_states.pt")
PROMPT = "The cat sat on the mat and then the"


def load_model():
    """Offline-loading shim copied from 04_readout_confidence.py."""
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
    return model


def stage1():
    t0 = time.time()
    model = load_model()
    print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(DM, "bell_anatomy.json")) as fh:
        bell = json.load(fh)

    st = torch.load(os.path.join(DM, "state_divine.pt"), weights_only=True)
    A_full = st["current_tensor"]
    initial_norm = st["initial_norm"]
    L1 = model.cfg.n_layers - 1
    hook_read = f"blocks.{L1}.hook_resid_post"
    hook_write = "blocks.0.hook_resid_pre"

    # step() and norm_to() copied from 06_bell_anatomy.py
    def step(x):
        cur = x * (initial_norm / x.norm())
        inject = cur.clone()
        def h(resid, hook, tensor=inject):
            resid[0, :, :] = tensor
            return resid
        model.add_hook(hook_write, h)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    PROMPT, names_filter=lambda n: n == hook_read)
        finally:
            model.reset_hooks()
        return cache[hook_read][0].clone()

    def norm_to(x, n):
        return x * (n / x.norm())

    B_full = step(A_full)
    A2_full = step(B_full)
    Bn = norm_to(B_full, initial_norm)
    A2n = norm_to(A2_full, initial_norm)
    A = A_full[-1].clone()
    B = Bn[-1].clone()
    A2 = A2n[-1].clone()
    M = (A + B) / 2
    d = (A - B) / 2

    cosAB = float(F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0)))
    cosAA2 = float(F.cosine_similarity(A.unsqueeze(0), A2.unsqueeze(0)))
    conv = torch.load(os.path.join(CONF, "converged_tensors.pt"), weights_only=True)
    syn_pilot = conv["Syntactic"][-1]
    cosA_pilot = float(F.cosine_similarity(A.unsqueeze(0), syn_pilot.unsqueeze(0)))

    print(f"gate cos(A,B)        = {cosAB:.10f}  (bell_anatomy: {bell['cosAB']:.10f})")
    print(f"gate cos(A,f(f(A)))  = {cosAA2:.10f}  (bell_anatomy: {bell['cosAA2']:.10f})")
    print(f"gate cos(A, pilot Syntactic last) = {cosA_pilot:.10f}")
    if not (abs(cosAB - bell["cosAB"]) < 5e-4 and cosAA2 > 0.999999
            and cosA_pilot > 0.999999):
        raise SystemExit("SANITY GATE FAILED (stage 1): A/B reconstruction does "
                         "not match bell_anatomy.json or the pilot state. Stopping.")
    print("stage-1 reconstruction gates PASSED", flush=True)

    # ---- Pilot noise states, regenerated exactly (05_jlens_pilot.py) ----
    # The pilot created these in-script and never saved them; the generation
    # is deterministic (global seed 2026, eval model, no RNG use in forwards),
    # so an exact replay reproduces them bit-for-bit.
    N_LAYERS, D_MODEL = model.cfg.n_layers, model.cfg.d_model
    NOISE_NORM, NOISE_SEQ, NOISE_ITERS = 397.0, 10, 100
    scaffold_tokens = torch.full((1, NOISE_SEQ), 262)
    noise_states = {}
    torch.manual_seed(2026)
    for trial in range(3):
        x = torch.randn(NOISE_SEQ, D_MODEL)
        x = x * (NOISE_NORM / x.norm())
        init_n = x.norm().item()
        current = x.clone()
        for _ in range(NOISE_ITERS):
            cn = current.norm().item()
            if cn > 0:
                current = current * (init_n / cn)
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
            current = cache[hook_read][0].clone()
        noise_states[f"Noise_{trial}"] = current[-1, :].clone()
        print(f"noise trial {trial} regenerated "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- W_U singular split of the hinge (mirrors 06_bell_anatomy.py) ----
    with torch.no_grad():
        U, _S, _Vh = torch.linalg.svd(model.W_U, full_matrices=False)
        unit = U.T @ (d / d.norm())
        d_top100_energy = float((unit[:100] ** 2).sum())
        d_bot100_energy = float((unit[-100:] ** 2).sum())
        cvec = U.T @ d
        d_vis = (U[:, :100] @ cvec[:100]).clone()     # readout-visible part
        d_quiet = (U[:, -100:] @ cvec[-100:]).clone()  # readout-quiet part
    print(f"gate d_top100 energy = {d_top100_energy:.6f} (bell_anatomy: {bell['d_top100']:.6f})")
    print(f"gate d_bot100 energy = {d_bot100_energy:.6f} (bell_anatomy: {bell['d_bot100']:.6f})")
    if (abs(d_top100_energy - bell["d_top100"]) > 1e-3
            or abs(d_bot100_energy - bell["d_bot100"]) > 1e-3):
        raise SystemExit("SANITY GATE FAILED (stage 1): W_U singular split of d "
                         "does not reproduce bell_anatomy.json. Stopping.")
    print("stage-1 W_U split gate PASSED", flush=True)

    # committed seed-42 noise run from 05_divine_motion.py (supplementary row)
    stn = torch.load(os.path.join(DM, "state_noise.pt"), weights_only=True)
    noise_committed = stn["current_tensor"][-1, :].clone()

    torch.save({
        "A": A, "B": B, "M": M, "d": d, "A2": A2,
        "d_vis_top100": d_vis, "d_quiet_bot100": d_quiet,
        "noise_states": noise_states,
        "noise_committed_seed42": noise_committed,
        "gates": {
            "cosAB": cosAB, "cosAA2": cosAA2, "cosA_vs_pilot_syntactic": cosA_pilot,
            "bell_cosAB": bell["cosAB"], "bell_cosAA2": bell["cosAA2"],
            "d_top100_energy": d_top100_energy, "d_bot100_energy": d_bot100_energy,
            "bell_d_top100": bell["d_top100"], "bell_d_bot100": bell["d_bot100"],
        },
        "norms": {"A": float(A.norm()), "B": float(B.norm()),
                  "M": float(M.norm()), "d": float(d.norm()),
                  "A_full": float(A_full.norm()), "initial_norm": initial_norm,
                  "d_vis": float(d_vis.norm()), "d_quiet": float(d_quiet.norm())},
        "cosines": {"cos_AB": cosAB, "cos_AM": float(F.cosine_similarity(
                        A.unsqueeze(0), M.unsqueeze(0))),
                    "cos_BM": float(F.cosine_similarity(
                        B.unsqueeze(0), M.unsqueeze(0))),
                    "cos_dM": float(F.cosine_similarity(
                        d.unsqueeze(0), M.unsqueeze(0)))},
    }, STATES_PT)
    print(f"stage 1 complete, states checkpointed ({time.time()-t0:.0f}s)", flush=True)


# ---- Probe machinery copied verbatim from 05_jlens_pilot.py ----
def lstsq_share(D, h):
    """Variance share of h captured by span of dictionary rows D [n, d]."""
    sol = torch.linalg.lstsq(D.T, h.unsqueeze(1))
    proj = D.T @ sol.solution
    return float(proj.squeeze(1).norm() ** 2 / h.norm() ** 2)


def nn_sparse_share(D, h, k=25, steps=300):
    """Nonnegative sparse approximation, at most k atoms, projected gradient.

    min_c ||D^T c - h||^2 s.t. c >= 0, ||c||_0 <= k. Simple hard-threshold
    projected gradient; pilot quality only.
    """
    Dt = D.T  # [d, n]
    G = Dt.T @ Dt
    lip = float(torch.linalg.eigvalsh(G).max())
    step = 1.0 / max(lip, 1e-8)
    c = torch.zeros(D.shape[0])
    b = Dt.T @ h
    for _ in range(steps):
        grad = G @ c - b
        c = c - step * grad
        c = c.clamp_min(0.0)
        if int((c > 0).sum()) > k:
            thresh = torch.topk(c, k).values.min()
            c[c < thresh] = 0.0
    approx = Dt @ c
    return float(approx.norm() ** 2 / h.norm() ** 2)


def random_dict_like(D, gen):
    R = torch.randn(D.shape, generator=gen)
    R = R / R.norm(dim=-1, keepdim=True) * D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return R


def probe_state(jlens, h, n_layers, gen):
    """One state through the pilot probe: identical logic and generator
    consumption order to the 05_jlens_pilot.py per-state loop."""
    per_layer = []
    for layer_idx in range(n_layers):
        D = jlens[:, layer_idx, :]
        entry = {
            "layer": layer_idx,
            "lstsq_share": lstsq_share(D, h),
            "nn_sparse_k25_share": nn_sparse_share(D, h),
        }
        rand_shares, rand_nn = [], []
        for _ in range(3):
            R = random_dict_like(D, gen)
            rand_shares.append(lstsq_share(R, h))
            rand_nn.append(nn_sparse_share(R, h))
        entry["random_lstsq_share_mean"] = sum(rand_shares) / len(rand_shares)
        entry["random_nn_sparse_k25_share"] = sum(rand_nn) / len(rand_nn)
        per_layer.append(entry)
    return per_layer


def stage2():
    t0 = time.time()
    jl = torch.load(os.path.join(JPILOT, "jlens_vectors.pt"), weights_only=True)
    jlens = jl["vectors"]                      # [193, 12, 768]
    n_layers = len(jl["layers"])
    ph = torch.load(STATES_PT, weights_only=True)
    conv = torch.load(os.path.join(CONF, "converged_tensors.pt"), weights_only=True)
    with open(os.path.join(JPILOT, "jlens_pilot_results.json")) as fh:
        pilot = json.load(fh)

    # ---- Replicated prefix: pilot's eight states, pilot order, gen 4242 ----
    # Order matters: the pilot drew its random dictionaries from one generator
    # while looping states in this order, so replaying in the same order makes
    # the random span controls reproduce too (the random SPARSE control in the
    # recorded JSON came from an older single-draw version of the pilot script
    # and is expected to differ; see the pilot report's recording caveat).
    gen = torch.Generator().manual_seed(4242)
    pilot_states = {label: conv[label][-1, :].clone() for label in
                    ["Lucier", "Semantic", "Syntactic", "Nonsense", "Imperative"]}
    for k in ["Noise_0", "Noise_1", "Noise_2"]:
        pilot_states[k] = ph["noise_states"][k]

    results = {"per_state": {}, "reproduction_check": {}}
    LENS_TOL = 1e-6
    for label, h in pilot_states.items():
        per_layer = probe_state(jlens, h, n_layers, gen)
        results["per_state"][label] = {"per_layer": per_layer,
                                       "state_norm": float(h.norm())}
        ref = pilot["per_state"][label]["per_layer"]
        diffs = {}
        for col in ["lstsq_share", "nn_sparse_k25_share",
                    "random_lstsq_share_mean", "random_nn_sparse_k25_share"]:
            diffs[col] = max(abs(per_layer[layer_idx][col] - ref[layer_idx][col])
                             for layer_idx in range(n_layers))
        results["reproduction_check"][label] = diffs
        print(f"replicated {label}: L6 lstsq={per_layer[6]['lstsq_share']:.6f} "
              f"(pilot {ref[6]['lstsq_share']:.6f}), max lens col diff "
              f"{max(diffs['lstsq_share'], diffs['nn_sparse_k25_share']):.2e}",
              flush=True)
        if max(diffs["lstsq_share"], diffs["nn_sparse_k25_share"]) > LENS_TOL:
            raise SystemExit(f"SANITY GATE FAILED (stage 2): lens columns for "
                             f"{label} do not reproduce jlens_pilot_results.json. "
                             f"Stopping before any new numbers.")
    print("stage-2 reproduction gate PASSED (all eight pilot states)", flush=True)

    # ---- New states: phases, pivot, supplementary committed noise ----
    for label, h in [("Divine_A", ph["A"]), ("Divine_B", ph["B"]),
                     ("Divine_M", ph["M"]),
                     ("Noise_committed_seed42", ph["noise_committed_seed42"])]:
        per_layer = probe_state(jlens, h, n_layers, gen)
        results["per_state"][label] = {"per_layer": per_layer,
                                       "state_norm": float(h.norm())}
        print(f"probed {label}: L6 lstsq={per_layer[6]['lstsq_share']:.4f} "
              f"L11 lstsq={per_layer[11]['lstsq_share']:.4f}", flush=True)

    # ---- The hinge question: direction d against the lens ----
    # Span share is sign- and scale-invariant, so it is THE hinge number.
    # The nonnegative sparse probe depends on sign (c >= 0), so both signs
    # are recorded. Baseline: 20 random unit directions (seed 777), mean span
    # share per layer, i.e. what a generic direction scores against D_l.
    gen777 = torch.Generator().manual_seed(777)
    base_dirs = torch.randn(20, jlens.shape[2], generator=gen777)
    base_dirs = base_dirs / base_dirs.norm(dim=-1, keepdim=True)

    direction_probe = {}
    for label, v in [("d_hinge", ph["d"]),
                     ("d_vis_top100", ph["d_vis_top100"]),
                     ("d_quiet_bot100", ph["d_quiet_bot100"])]:
        per_layer = []
        for layer_idx in range(n_layers):
            D = jlens[:, layer_idx, :]
            entry = {"layer": layer_idx, "lstsq_share": lstsq_share(D, v)}
            if label == "d_hinge":
                entry["nn_sparse_k25_share_plus"] = nn_sparse_share(D, v)
                entry["nn_sparse_k25_share_minus"] = nn_sparse_share(D, -v)
                rand_shares = []
                for _ in range(3):
                    R = random_dict_like(D, gen)
                    rand_shares.append(lstsq_share(R, v))
                entry["random_lstsq_share_mean"] = sum(rand_shares) / 3
            entry["random_direction_lstsq_mean"] = float(
                sum(lstsq_share(D, base_dirs[i]) for i in range(20)) / 20)
            per_layer.append(entry)
        direction_probe[label] = {"per_layer": per_layer, "norm": float(v.norm())}
        print(f"direction {label}: L6 span={per_layer[6]['lstsq_share']:.4f} "
              f"L11 span={per_layer[11]['lstsq_share']:.4f} "
              f"(generic dir L11 {per_layer[11]['random_direction_lstsq_mean']:.4f})",
              flush=True)
    results["direction_probe"] = direction_probe

    # ---- Summary block ----
    def col(label, key="lstsq_share"):
        return [results["per_state"][label]["per_layer"][layer_idx][key]
                for layer_idx in range(n_layers)]
    prolet_labels = ["Lucier", "Semantic", "Nonsense", "Imperative"]
    noise_labels = ["Noise_0", "Noise_1", "Noise_2"]
    def mean_cols(labels, key):
        cols = [col(lb, key) for lb in labels]
        return [sum(c[layer_idx] for c in cols) / len(cols) for layer_idx in range(n_layers)]
    d_span = [direction_probe["d_hinge"]["per_layer"][layer_idx]["lstsq_share"]
              for layer_idx in range(n_layers)]
    generic = [direction_probe["d_hinge"]["per_layer"][layer_idx]["random_direction_lstsq_mean"]
               for layer_idx in range(n_layers)]
    summary = {
        "span_by_layer": {
            "Divine_A": col("Divine_A"), "Divine_B": col("Divine_B"),
            "Divine_M": col("Divine_M"),
            "prolet_mean": mean_cols(prolet_labels, "lstsq_share"),
            "noise_mean": mean_cols(noise_labels, "lstsq_share"),
            "d_hinge": d_span, "generic_direction": generic,
        },
        "sparse_by_layer": {
            "Divine_A": col("Divine_A", "nn_sparse_k25_share"),
            "Divine_B": col("Divine_B", "nn_sparse_k25_share"),
            "Divine_M": col("Divine_M", "nn_sparse_k25_share"),
            "prolet_mean": mean_cols(prolet_labels, "nn_sparse_k25_share"),
            "noise_mean": mean_cols(noise_labels, "nn_sparse_k25_share"),
        },
        "hinge_headline": {
            "d_span_L11": d_span[11], "d_span_L6": d_span[6],
            "d_span_mean_all_layers": sum(d_span) / len(d_span),
            "generic_direction_span_L11": generic[11],
            "generic_direction_span_mean": sum(generic) / len(generic),
        },
    }
    results["summary"] = summary

    results["meta"] = {
        "issue": 8,
        "script": "10_jlens_phase.py",
        "lens_file": "output_jlens_pilot/jlens_vectors.pt (reused, not recomputed)",
        "probe": "identical to 05_jlens_pilot.py (lstsq span, nonnegative sparse "
                 "k=25, random-dictionary controls, generator seed 4242, pilot "
                 "state order for the replicated prefix)",
        "state_construction": "A, B, M, d exactly as 06_bell_anatomy.py "
                              "(state_divine.pt, iteration 1000)",
        "noise_states": "pilot's seed-2026 states regenerated in-script "
                        "(deterministic replay); the pilot never committed them. "
                        "Noise_committed_seed42 is the separate committed "
                        "05_divine_motion run (state_noise.pt, iteration 1000), "
                        "a different noise trajectory, included as a "
                        "supplementary row only.",
        "gates": ph["gates"],
        "norms": ph["norms"],
        "cosines": ph["cosines"],
        "notes": [
            "Divine_A equals the pilot's probed Syntactic state up to "
            "cos > 0.9999998 (L2 diff 0.025 at norm 1612), so its rows "
            "duplicate the Syntactic rows within numerical noise.",
            "The random sparse control differs from the recorded pilot JSON "
            "by design: the pilot report documents that its recorded value "
            "came from a single draw while the committed script (and this "
            "replay) averages three.",
            "Span shares are scale- and sign-invariant; the sparse probe is "
            "sign-dependent, hence both signs reported for d.",
        ],
    }

    with open(os.path.join(OUT, "jlens_phase.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"stage 2 complete ({time.time()-t0:.0f}s). JSON written.", flush=True)

    # ---- Markdown-ready tables ----
    def fmt(x):
        return f"{x:.3f}"
    print("\n#### span table")
    print("| layer | A | B | M | prolet | noise | hinge d | generic dir |")
    print("|---|---|---|---|---|---|---|---|")
    s = summary["span_by_layer"]
    for layer_idx in range(n_layers):
        print(f"| L{layer_idx} | {fmt(s['Divine_A'][layer_idx])} | {fmt(s['Divine_B'][layer_idx])} | "
              f"{fmt(s['Divine_M'][layer_idx])} | {fmt(s['prolet_mean'][layer_idx])} | "
              f"{fmt(s['noise_mean'][layer_idx])} | {fmt(s['d_hinge'][layer_idx])} | "
              f"{fmt(s['generic_direction'][layer_idx])} |")
    print("\n#### sparse table")
    print("| layer | A | B | M | prolet | noise |")
    print("|---|---|---|---|---|---|")
    sp = summary["sparse_by_layer"]
    for layer_idx in range(n_layers):
        print(f"| L{layer_idx} | {fmt(sp['Divine_A'][layer_idx])} | {fmt(sp['Divine_B'][layer_idx])} | "
              f"{fmt(sp['Divine_M'][layer_idx])} | {fmt(sp['prolet_mean'][layer_idx])} | "
              f"{fmt(sp['noise_mean'][layer_idx])} |")
    print("\n#### hinge detail")
    print("| layer | d span | rand-dict span | generic-dir span | d nn25 (+d) | "
          "d nn25 (-d) | d_vis span | d_quiet span |")
    print("|---|---|---|---|---|---|---|---|")
    dp = direction_probe
    for layer_idx in range(n_layers):
        e = dp["d_hinge"]["per_layer"][layer_idx]
        print(f"| L{layer_idx} | {fmt(e['lstsq_share'])} | "
              f"{fmt(e['random_lstsq_share_mean'])} | "
              f"{fmt(e['random_direction_lstsq_mean'])} | "
              f"{fmt(e['nn_sparse_k25_share_plus'])} | "
              f"{fmt(e['nn_sparse_k25_share_minus'])} | "
              f"{fmt(dp['d_vis_top100']['per_layer'][layer_idx]['lstsq_share'])} | "
              f"{fmt(dp['d_quiet_bot100']['per_layer'][layer_idx]['lstsq_share'])} |")
    print("\nhinge headline:", json.dumps(summary["hinge_headline"], indent=1))


def stage3():
    """Probe the symmetric on-shell flip axis d_sym (review fix).

    The committed axis probed in stage 2 mixes frames (raw A with shell B)
    and is 0.909 aligned with its own pivot; d_sym is the frame-clean cycle
    axis, orthogonal to M_sym. Appends direction_probe_sym to
    jlens_phase.json; stage-2 results are untouched.
    """
    t0 = time.time()
    json_path = os.path.join(OUT, "jlens_phase.json")
    if not os.path.exists(json_path):
        raise SystemExit("stage 3 needs jlens_phase.json; run stage2 first.")
    ph = torch.load(STATES_PT, weights_only=True)

    def cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    # ---- Construct d_sym from the checkpointed states ----
    # Stage 1 stored A in the raw frame (last row of the iteration-1000
    # tensor, full norm 5098) and B already on the shell (full norm N0).
    # The state is exactly position-collapsed, so rescaling the stored row
    # by N0 / ||A_full|| equals taking the last row of norm_to(A_full, N0),
    # which is how 08_hinge_eigenvalue.py builds An.
    N0 = ph["norms"]["initial_norm"]
    An = ph["A"] * (N0 / ph["norms"]["A_full"])
    Bn = ph["B"]
    D_sym = (An - Bn) / 2
    M_sym = (An + Bn) / 2
    d_sym = D_sym / D_sym.norm()

    with open(os.path.join(HEIG, "hinge_eigenvalue.json")) as fh:
        heig = json.load(fh)
    geo = heig["meta"]["geometry"]
    tang_ref = heig["part2"]["d_committed_tangential_note"]["cos_vs_d_sym"]

    dc_hat = ph["d"] / ph["d"].norm()
    mc_hat = ph["M"] / ph["M"].norm()
    dc_tang = dc_hat - float(dc_hat @ mc_hat) * mc_hat
    dc_tang = dc_tang / dc_tang.norm()

    cos_dsym_msym = cos(d_sym, M_sym)
    cos_dsym_dcom = cos(d_sym, ph["d"])
    cos_dsym_tang = cos(d_sym, dc_tang)
    print(f"gate cos(d_sym, M_sym)             = {cos_dsym_msym:+.2e}  "
          f"(hinge_eigenvalue: {geo['cos_Dsym_vs_Msym']:.2e})")
    print(f"gate cos(d_sym, d_committed)       = {cos_dsym_dcom:.6f}  "
          f"(hinge_eigenvalue: {geo['cos_Dcommitted_vs_Dsym']:.6f})")
    print(f"gate cos(d_sym, tangential d_comm) = {cos_dsym_tang:.6f}  "
          f"(hinge_eigenvalue: {tang_ref:.6f})")
    if not (abs(cos_dsym_msym) < 1e-4
            and abs(cos_dsym_dcom - geo["cos_Dcommitted_vs_Dsym"]) < 1e-3
            and abs(cos_dsym_tang - tang_ref) < 1e-3):
        raise SystemExit("SANITY GATE FAILED (stage 3): d_sym reconstruction "
                         "does not match hinge_eigenvalue.json. Stopping.")
    print("stage-3 d_sym construction gates PASSED", flush=True)

    # ---- W_U singular split of d_sym (model needed once; then cached) ----
    if "dsym_vis_top100" not in ph:
        model = load_model()
        print(f"model loaded for W_U split ({time.time()-t0:.0f}s)", flush=True)
        with torch.no_grad():
            U, _S, _Vh = torch.linalg.svd(model.W_U, full_matrices=False)
            unit_c = U.T @ dc_hat
            top_c = float((unit_c[:100] ** 2).sum())
            bot_c = float((unit_c[-100:] ** 2).sum())
            if (abs(top_c - ph["gates"]["d_top100_energy"]) > 1e-5
                    or abs(bot_c - ph["gates"]["d_bot100_energy"]) > 1e-5):
                raise SystemExit("SANITY GATE FAILED (stage 3): fresh W_U SVD "
                                 "does not reproduce the stage-1 committed-d "
                                 "split. Stopping.")
            coef = U.T @ d_sym
            ph["dsym_vis_top100"] = (U[:, :100] @ coef[:100]).clone()
            ph["dsym_quiet_bot100"] = (U[:, -100:] @ coef[-100:]).clone()
            ph["dsym_wu_energies"] = {
                "top100": float((coef[:100] ** 2).sum()),
                "bot100": float((coef[-100:] ** 2).sum()),
            }
        torch.save(ph, STATES_PT)
        print("stage-3 W_U gate PASSED; d_sym components cached into "
              "phase_states.pt", flush=True)
    top_e = ph["dsym_wu_energies"]["top100"]
    bot_e = ph["dsym_wu_energies"]["bot100"]
    print(f"d_sym energy in top-100 / bottom-100 W_U dirs = "
          f"{top_e:.6f} / {bot_e:.6f} (covered {top_e + bot_e:.6f})", flush=True)

    # ---- Probes: identical machinery, same seed-777 baseline as stage 2 ----
    jl = torch.load(os.path.join(JPILOT, "jlens_vectors.pt"), weights_only=True)
    jlens = jl["vectors"]
    n_layers = len(jl["layers"])
    with open(json_path) as fh:
        results = json.load(fh)

    gen777 = torch.Generator().manual_seed(777)
    base_dirs = torch.randn(20, jlens.shape[2], generator=gen777)
    base_dirs = base_dirs / base_dirs.norm(dim=-1, keepdim=True)

    random_means = [float(sum(lstsq_share(jlens[:, layer_idx, :], base_dirs[i])
                              for i in range(20)) / 20)
                    for layer_idx in range(n_layers)]
    direction_probe_sym = {}
    for label, v in [("d_sym", d_sym),
                     ("dsym_vis_top100", ph["dsym_vis_top100"]),
                     ("dsym_quiet_bot100", ph["dsym_quiet_bot100"])]:
        per_layer = []
        for layer_idx in range(n_layers):
            D = jlens[:, layer_idx, :]
            entry = {"layer": layer_idx, "lstsq_share": lstsq_share(D, v)}
            if label == "d_sym":
                entry["nn_sparse_k25_share_plus"] = nn_sparse_share(D, v)
                entry["nn_sparse_k25_share_minus"] = nn_sparse_share(D, -v)
            entry["random_direction_lstsq_mean"] = random_means[layer_idx]
            per_layer.append(entry)
        direction_probe_sym[label] = {"per_layer": per_layer,
                                      "norm": float(v.norm())}
        print(f"direction {label}: L6 span={per_layer[6]['lstsq_share']:.4f} "
              f"L11 span={per_layer[11]['lstsq_share']:.4f} "
              f"(generic dir L11 {per_layer[11]['random_direction_lstsq_mean']:.4f})",
              flush=True)

    # Baseline replay gate: the seed-777 generic-direction column must equal
    # the one stage 2 recorded (it is a per-layer property, not a per-
    # direction one), proving the two stages are on the same footing.
    max_dg = max(
        abs(direction_probe_sym["d_sym"]["per_layer"][layer_idx]
            ["random_direction_lstsq_mean"]
            - results["direction_probe"]["d_hinge"]["per_layer"][layer_idx]
            ["random_direction_lstsq_mean"])
        for layer_idx in range(n_layers))
    print(f"gate seed-777 baseline vs stage 2: max diff {max_dg:.2e}")
    if max_dg > 1e-6:
        raise SystemExit("SANITY GATE FAILED (stage 3): the generic-direction "
                         "baseline does not replay stage 2. Stopping.")
    print("stage-3 baseline replay gate PASSED", flush=True)

    # ---- Summary block and JSON update (idempotent keys, no appends) ----
    ds = [direction_probe_sym["d_sym"]["per_layer"][layer_idx]["lstsq_share"]
          for layer_idx in range(n_layers)]
    gn = [direction_probe_sym["d_sym"]["per_layer"][layer_idx]
          ["random_direction_lstsq_mean"] for layer_idx in range(n_layers)]
    vis = [direction_probe_sym["dsym_vis_top100"]["per_layer"][layer_idx]
           ["lstsq_share"] for layer_idx in range(n_layers)]
    qt = [direction_probe_sym["dsym_quiet_bot100"]["per_layer"][layer_idx]
          ["lstsq_share"] for layer_idx in range(n_layers)]
    results["direction_probe_sym"] = direction_probe_sym
    results["summary"]["sym_axis_headline"] = {
        "d_sym_span_L11": ds[11], "d_sym_span_L6": ds[6],
        "d_sym_span_mean_all_layers": sum(ds) / len(ds),
        "generic_direction_span_L11": gn[11],
        "generic_direction_span_mean": sum(gn) / len(gn),
        "dsym_vis_span_L0": vis[0], "dsym_vis_span_L11": vis[11],
        "dsym_quiet_span_L0": qt[0], "dsym_quiet_span_L11": qt[11],
        "wu_energy_top100": top_e, "wu_energy_bot100": bot_e,
        "wu_energy_covered": top_e + bot_e,
    }
    results["summary"]["axis_frame_check"] = {
        "cos_dsym_Msym": cos_dsym_msym,
        "cos_dsym_dcommitted": cos_dsym_dcom,
        "cos_dsym_dcommitted_tangential": cos_dsym_tang,
        "norm_Dsym_last": float(D_sym.norm()),
        "norm_Msym_last": float(M_sym.norm()),
        "committed_wu_energy_top100": ph["gates"]["d_top100_energy"],
        "committed_wu_energy_bot100": ph["gates"]["d_bot100_energy"],
        "committed_wu_energy_covered": (ph["gates"]["d_top100_energy"]
                                        + ph["gates"]["d_bot100_energy"]),
    }
    results["meta"]["frame_note"] = (
        "direction_probe keys d_hinge, d_vis_top100, d_quiet_bot100 are the "
        "COMMITTED flip axis d_committed, built exactly as 06_bell_anatomy.py "
        "builds it (raw-frame A mixed with shell-frame B; cos(d, M) = 0.909, "
        "mostly radial contamination). direction_probe_sym is the symmetric "
        "on-shell axis d_sym = normalise(An - Bn), both phases on the loop "
        "shell N0, orthogonal to its pivot M_sym; it is the physical cycle "
        "axis. See output_hinge_eigen/hinge_eigenvalue.md, 'The map, the "
        "frames, and two flip axes'.")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"stage 3 complete ({time.time()-t0:.0f}s). JSON updated.", flush=True)

    # ---- Markdown-ready table ----
    def fmt(x):
        return f"{x:.3f}"
    print("\n#### symmetric axis detail")
    print("| layer | d_sym span | generic-dir span | d_sym nn25 (+) | "
          "d_sym nn25 (-) | dsym_vis span | dsym_quiet span |")
    print("|---|---|---|---|---|---|---|")
    for layer_idx in range(n_layers):
        e = direction_probe_sym["d_sym"]["per_layer"][layer_idx]
        print(f"| L{layer_idx} | {fmt(e['lstsq_share'])} | "
              f"{fmt(e['random_direction_lstsq_mean'])} | "
              f"{fmt(e['nn_sparse_k25_share_plus'])} | "
              f"{fmt(e['nn_sparse_k25_share_minus'])} | "
              f"{fmt(vis[layer_idx])} | {fmt(qt[layer_idx])} |")
    print("\nsym axis headline:",
          json.dumps(results["summary"]["sym_axis_headline"], indent=1))


if __name__ == "__main__":
    if STAGE in ("stage1", "all") and not os.path.exists(STATES_PT):
        stage1()
    elif STAGE in ("stage1", "all"):
        print("phase_states.pt exists; skipping stage 1", flush=True)
    if STAGE in ("stage2", "all"):
        stage2()
    if STAGE in ("stage3", "all"):
        stage3()
