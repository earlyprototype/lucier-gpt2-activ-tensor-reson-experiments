"""What does the loop map look like, to first order, at a settled attractor?

Belongs at `experiments/basin_depth/03_spectrum.py`. Registered before
execution in `spectrum_protocol.md`; read that first, because several of this
script's choices are only defensible with its reasoning attached.

THE OBJECT. The loop rescales every iterate to a fixed Frobenius norm nu, so
the state lives on a sphere of dimension (positions x 768) minus 1, about 7679
here. The dynamics on that sphere is

    S(y) = nu * G(y) / ||G(y)||_F        for  ||y||_F = nu

where G is the block cascade 0..11 applied to a tensor injected at
blocks.0.hook_resid_pre. The injection overwrites every position, so G is the
whole forward map from that point and does not depend on the prompt beyond its
length; the script asserts this against the hook path, as 08_hinge_eigenvalue.py
did. At a settled attractor S linearises, and the eigenvalues of that
linearisation say what the loop does locally: modulus below 1 contracting,
above 1 expanding, negative real flipping sides each pass, complex spiralling.

THE MAP MUST INCLUDE THE RENORMALISATION, and there are three corrections, not
one. Writing f(x) = G(nu x / ||x||) for the prior art's map:

  (a) J_f(x) x = 0 exactly. A radial perturbation is undone by the rescale
      before the network sees it. Automatic once the rescale is inside the
      differentiated function, enforced anyway so the eigensolver does not
      waste Krylov dimension on a spurious zero.
  (b) The response is not tangent to the shell at the image point, and the
      loop's next rescale strips its radial part:
          DS(y) v = (nu / ||f(y)||) * Q_{f(y)} * J_f(y) v ,   Q_z = I - z^ z^^T
      Only at a fixed point does Q coincide with the projection at y, and only
      then is DS an operator on one space with eigenvalues at all.
  (c) The factor nu / ||f(y)|| is not optional. It is 0.294 at the Divine
      committed pivot, so dropping it rescales the whole spectrum.

08_hinge_eigenvalue.py reported Rayleigh quotients of J_f, that is (a) only.
Both conventions are reported here, lambda_onshell and lambda_rayleigh_Jf, so
the numbers stay comparable with the committed record in either direction.

ONE PASS OR TWO. Determined by measurement, never assumed. The period p is the
smallest lag whose mean-vector cosine passes the engine threshold on a dense
continuation (the F15 rule). For p = 1 the object is DS at the fixed point. For
p = 2 the one-pass map is not a fixed point at all, it swaps the phases, so
DS(A) maps one tangent space to another and has no eigenvalues; the object is
DS2(A) = DS(B) DS(A), whose eigenvalues are the cycle's Floquet multipliers.
Fixed points and cycles are compared on the per-pass rate |lambda|^(1/p).

LIMITS, UP FRONT. This measures the leading 24 of about 7679 eigenvalues and
says nothing about the bulk. A Jacobian is local; basin boundaries are global
and nonlinear, and a spectrum cannot locate one. Everything is float32, so the
matvec carries a relative error near 1e-6 and eigenvalues below about 1e-3 in
modulus are reported as below the numerical floor rather than as numbers. The
states are two prompts per label per level, a probe and not a sample (FINDINGS
caveat 19). Divine numbers inherit caveat 14, one audited trajectory of 34.

VERIFICATION BEFORE ANY NEW NUMBER. --verify reproduces 08_hinge_eigenvalue's
committed flip-axis eigenvalue -0.863580 and its composed +0.099339 from this
script's own JVP path, plus a plumbing check of the eigensolver against a dense
matrix and a finite-difference sensitivity check. Workers refuse to run until
verify.json records a pass.

Run from the repo root:

    ATR_GPT2_LOCAL=... python3 experiments/basin_depth/03_spectrum.py --verify
    ATR_GPT2_LOCAL=... python3 experiments/basin_depth/03_spectrum.py --worker 0 --num-workers 4
    ATR_GPT2_LOCAL=... python3 experiments/basin_depth/03_spectrum.py --eigen-ladder --worker 0 --num-workers 4
    python3 experiments/basin_depth/03_spectrum.py --report-only

Outputs (in experiments/basin_depth/output_spectrum/):
    verify.json                 gate results; workers refuse without a pass
    checkpoints/<key>.pt        one spectrum per state (the resume unit)
    eigen_ladder/<key>.pt       stage 3 ladders along eigenvectors
    spectrum.pt                 combined archive
    spectrum.md                 every headline number, from the data only
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

# The script is written to sit at experiments/basin_depth/. ATR_REPO lets it be
# exercised from elsewhere (a scratchpad copy) without editing paths.
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ATR_REPO", HERE.parents[1]))
sys.path.insert(0, str(REPO_ROOT))
torch.set_num_threads(1)  # 4 BLAS threads thrash on this box; workers parallelise


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sa = _load("nu_sweep_stage_a", "experiments/nu_sweep/01_stage_a.py")
el = _load("escape_ladder", "experiments/basin_depth/01_escape_ladder.py")

from atr_engine import lag_scan  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "basin_depth" / "output_spectrum"
CKPT_DIR = OUT_DIR / "checkpoints"
LADDER_DIR = OUT_DIR / "eigen_ladder"
VERIFY = OUT_DIR / "verify.json"
DIVINE_STATE = (REPO_ROOT / "experiments" / "gpt2_small" / "output_divine_motion"
                / "state_divine.pt")

K_EIGS = 24              # leading eigenvalues per state
ARNOLDI_M = 96           # Krylov basis size, 4 * K_EIGS
ARNOLDI_RESTARTS = 4     # total matvecs = (1 + restarts) * m, fixed in advance
RITZ_TOL = 1e-6          # a Ritz pair below this residual counts as converged
V0_SEEDS = (20260805, 91125)   # second seed is gate G5
EPS_RELS = (1e-3, 1e-4)  # 1e-3 is the well-conditioned step here; see protocol
MAX_LAG = 8
REFINE_PASSES = 20
REFINE_TOL = 1e-4        # gate G6, ||S^p(y) - y|| / nu
FD_GATE_REL = 0.01       # gate G4, jvp against FD at eps_rel 1e-3
N_LADDER_EIGS = 4        # stage 3: this many from each end of the leading set

# Committed numbers from experiments/gpt2_small/output_hinge_eigen/
# hinge_eigenvalue.json. Nothing new is trusted until these come back.
PRIOR_ART = {
    "lambda_d_committed_at_Mn_committed": -0.8635796904563904,
    "amplification_d_committed_at_Mn_committed": 0.9575217366218567,
    "cos_Jt_minus_t_d_committed_at_Mn_committed": 0.9018904566764832,
    "lambda_d_sym_at_Mn_sym": -4.275077819824219,
    "lambda_composed_d_sym_start_A": 0.09933854639530182,
    "cos_AB": 0.6849116683,
}
GATE_ABS = 1e-3


# ------------------------------------------------------------------ helpers --
def unit(x):
    return x / x.norm()


def fcos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)))


def row_spread(x):
    """0 when every position holds the same vector; the collapse diagnostic."""
    mean_row = x.mean(dim=0)
    denom = float(mean_row.norm())
    return float((x - mean_row).norm() / denom) if denom > 0 else float("nan")


def make_f_map(model, nu):
    """The ATR iteration as a pure differentiable function of [T, d_model].

    Renormalise to the shell, then the block cascade. Equal to the hook-based
    step because the injection overwrites every position of resid_pre 0, which
    the caller is expected to assert once per state."""
    def f_map(x2d):
        xn = x2d * (nu / x2d.norm())
        resid = xn.unsqueeze(0)
        for blk in model.blocks:
            resid = blk(resid)
        return resid[0]
    return f_map


def hook_step(model, prompt, nu, x):
    """One iteration through the engine's own hook path, for the equality check."""
    cur = x * (nu / x.norm())
    read = f"blocks.{sa.LAYER_END}.hook_resid_post"
    write = f"blocks.{sa.LAYER_START}.hook_resid_pre"

    def h(resid, hook, tensor=cur.clone()):
        resid[0, :, :] = tensor
        return resid

    model.add_hook(write, h)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, names_filter=lambda n: n == read)
    finally:
        model.reset_hooks()
    return cache[read][0].clone()


# --------------------------------------------------------- the on-shell map --
class OnShellMap:
    """S and its derivative at one or more base points on the shell.

    f(base) is cached per base point, so a matvec costs exactly one jvp per
    composition step and no extra forward pass."""

    def __init__(self, f_map, nu):
        self.f_map = f_map
        self.nu = nu
        self._cache = {}
        self.n_jvp = 0

    def image(self, base):
        key = id(base)
        if key not in self._cache:
            with torch.no_grad():
                fb = self.f_map(base)
            self._cache[key] = (fb, unit(fb).flatten(), self.nu / float(fb.norm()))
        return self._cache[key]

    def step(self, base):
        """S(base): the next on-shell iterate."""
        fb, _, scale = self.image(base)
        return fb * scale

    def ds(self, base, v):
        """DS(base) v for v tangent at base. Output projection (b) and scale (c)."""
        _, u_hat, scale = self.image(base)
        _, jv = torch.func.jvp(self.f_map, (base,), (v,))
        self.n_jvp += 1
        jv = jv.detach().flatten()
        jv = jv - (jv @ u_hat) * u_hat
        return (scale * jv).view(base.shape)

    def jf(self, base, v):
        """Raw J_f(base) v, the prior art's convention: (a) only."""
        _, jv = torch.func.jvp(self.f_map, (base,), (v,))
        self.n_jvp += 1
        return jv.detach()

    def fd(self, base, v_unit, eps_rel):
        """Central finite difference of f along a unit direction."""
        h = eps_rel * float(base.norm())
        with torch.no_grad():
            return (self.f_map(base + h * v_unit)
                    - self.f_map(base - h * v_unit)) / (2.0 * h)


class TangentOperator:
    """The linearised loop map restricted to the tangent space at bases[0].

    p = 1: DS(y). p = 2: DS(B) DS(A) with bases = [A, B]. Input and output are
    both projected onto the tangent space at bases[0]; the intermediate
    projection is already done inside OnShellMap.ds."""

    def __init__(self, osm, bases):
        self.osm = osm
        self.bases = bases
        self.shape = bases[0].shape
        self.n = int(np.prod(self.shape))
        self.y_hat = unit(bases[0]).flatten()
        self.n_matvec = 0

    def project(self, v_flat):
        return v_flat - (v_flat @ self.y_hat) * self.y_hat

    def apply(self, v_flat_t):
        v = self.project(v_flat_t).view(self.shape)
        for base in self.bases:
            v = self.osm.ds(base, v)
        self.n_matvec += 1
        return self.project(v.flatten())

    def matvec_np(self, v_np):
        v = torch.from_numpy(np.ascontiguousarray(v_np)).float()
        return self.apply(v).double().numpy()

    def rayleigh_jf(self, v_flat_t):
        """The raw J_f Rayleigh quotient along a tangent direction, correction
        (a) only, for continuity with the committed record's convention.

        NOT the same number as 08_hinge_eigenvalue's composed +0.099339. That
        one is an unprojected chain at the RAW cycle states, along the full
        d_sym including its radial part (cos(d_sym, A_hat) = 0.397, so 40
        percent of it is radial). Gate G3 reproduces it in its own frame. Here
        the input is projected onto the tangent space first, because that is
        the only space the on-shell operator acts on, and the two quantities
        are therefore different objects and must not be compared directly."""
        v = self.project(v_flat_t)
        v = v / v.norm()
        w = v.view(self.shape)
        for base in self.bases:
            w = self.osm.jf(base, w)
        return float(v @ w.flatten())


# ---------------------------------------------------------------- eigensolver --
def arnoldi_eigs(matvec, n, k, m, restarts, seed):
    """Explicitly restarted Arnoldi for the k largest-modulus eigenvalues.

    Modified Gram-Schmidt with one reorthogonalisation pass (the matvec is
    float32-noisy, so a single MGS pass loses orthogonality fast). Ritz
    residuals come free from |h_{m+1,m}| |e_m^T y|. Cost is exactly
    (1 + restarts) * m matvecs, which is why the protocol can quote a fixed
    number of JVPs per eigenvalue rather than an estimate.

    Returns (values, vectors, residuals, n_matvec); vectors are columns of an
    (n, k) complex array."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    n_mv = 0
    vals = vecs = resid = None
    for sweep in range(restarts + 1):
        V = np.zeros((m + 1, n))
        H = np.zeros((m + 1, m))
        V[0] = v
        m_eff = m
        for j in range(m):
            w = matvec(V[j])
            n_mv += 1
            for _pass in range(2):
                proj = V[:j + 1] @ w
                H[:j + 1, j] += proj
                w -= proj @ V[:j + 1]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-12:
                m_eff = j + 1
                break
            V[j + 1] = w / H[j + 1, j]
        Hm = H[:m_eff, :m_eff]
        ev, evec = np.linalg.eig(Hm)
        order = np.argsort(-np.abs(ev))[:k]
        vals = ev[order]
        Y = evec[:, order]
        vecs = V[:m_eff].T.astype(np.complex128) @ Y
        # ||A x - theta x|| for a Ritz pair, in the Arnoldi identity.
        resid = np.abs(H[m_eff, m_eff - 1] * Y[-1, :]) / np.maximum(
            np.linalg.norm(vecs, axis=0), 1e-30)
        if sweep < restarts:
            # Restart from the leading Ritz block. Real part of the sum keeps
            # the start vector real, which keeps the Krylov space real.
            v = np.real(vecs).sum(axis=1)
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                break
            v /= nv
    return vals, vecs, resid, n_mv


def dedupe_pairs(vals, vecs, resid):
    """One row per mode: a conjugate pair is reported once, with its plane.

    Reporting both members of a pair would double-count the mode and inflate
    any count of how many marginal directions a state has."""
    modes, used = [], set()
    order = sorted(range(len(vals)), key=lambda i: (-abs(vals[i]), np.angle(vals[i])))
    for i in order:
        if i in used:
            continue
        lam = vals[i]
        partner = None
        if abs(lam.imag) > 1e-9 * max(abs(lam), 1e-30):
            for j in order:
                if j != i and j not in used and abs(vals[j] - np.conj(lam)) < \
                        1e-6 * max(abs(lam), 1e-30):
                    partner = j
                    break
        used.add(i)
        if partner is not None:
            used.add(partner)
        v = vecs[:, i]
        if partner is None:
            imag_frac = float(np.linalg.norm(v.imag) / max(np.linalg.norm(v), 1e-30))
            basis = [np.real(v) / max(np.linalg.norm(np.real(v)), 1e-30)]
        else:
            imag_frac = None
            q1 = np.real(v)
            q1 = q1 / max(np.linalg.norm(q1), 1e-30)
            q2 = np.imag(v)
            q2 = q2 - (q2 @ q1) * q1
            q2 = q2 / max(np.linalg.norm(q2), 1e-30)
            basis = [q1, q2]
        modes.append({
            "re": float(lam.real), "im": float(lam.imag),
            "modulus": float(abs(lam)),
            "arg_deg": float(math.degrees(np.angle(lam))),
            "is_pair": partner is not None,
            "ritz_residual": float(resid[i]),
            "real_eigvec_imag_fraction": imag_frac,
            "basis": [b.astype(np.float32) for b in basis],
        })
    return modes


def classify(mode, p):
    """The word the report uses for this mode. Fixed in advance."""
    mod, re = mode["modulus"], mode["re"]
    if mode["is_pair"]:
        turns = p * 360.0 / max(abs(mode["arg_deg"]), 1e-9)
        kind = "expanding spiral" if mod > 1 else "spiral"
        return f"{kind}, {turns:.1f} passes per revolution"
    if re < 0:
        return ("flip (period doubling)" if p == 1 else "period-4 flip mode") + (
            ", expanding" if mod > 1 else "")
    return "expanding" if mod > 1 else "contracting"


# ------------------------------------------------------------ base points ---
def detect_period(osm, y, max_lag=MAX_LAG):
    """F15's rule: the smallest lag whose mean-vector cosine passes the gate.

    A single fixed lag aliases every period that does not divide it, which is
    the mistake F9 and F15 document one octave down, so the whole table is
    scanned and the smallest passing lag is taken."""
    iterates, cur = [], y
    for _ in range(max_lag + 2):
        cur = osm.step(cur)
        iterates.append(cur.mean(dim=0).clone())
    table = lag_scan([y.mean(dim=0)] + iterates, max_lag=max_lag)
    p = next((lag for lag in sorted(table) if table[lag] > sa.THRESHOLD), None)
    return p, {int(k): float(v) for k, v in table.items()}


def refine(osm, y, p, passes=REFINE_PASSES, tol=REFINE_TOL):
    """Tighten the base point, because the engine gate passes at cosine 0.999
    and that is loose for a linearisation. Gate G6 on the returned residual."""
    nu = osm.nu
    best = y
    resid = float("inf")
    for _ in range(passes):
        cur = best
        for _ in range(p):
            cur = osm.step(cur)
        new_resid = float((cur - best).norm()) / nu
        if new_resid >= resid:
            break
        best, resid = cur, new_resid
        if resid < tol:
            break
    return best, resid


# ----------------------------------------------------------------- gates ----
def gate_g0():
    """Plumbing: the same solver on a dense matrix against numpy.linalg.eig.

    Costs no forward passes and catches conjugate-pair packing and ordering
    bugs, which are the likeliest silent errors in the whole script."""
    rng = np.random.default_rng(7)
    n, k = 60, 8
    A = rng.standard_normal((n, n)) / math.sqrt(n)
    ref = np.sort_complex(np.linalg.eig(A)[0])
    ref = ref[np.argsort(-np.abs(ref))][:k]
    vals, _, _, _ = arnoldi_eigs(lambda v: A @ v, n, k, m=48, restarts=3, seed=1)
    vals = vals[np.argsort(-np.abs(vals))]
    err = float(np.max(np.abs(np.sort_complex(vals) - np.sort_complex(ref))))
    return {"max_abs_error": err, "passed": bool(err < 1e-8)}


def gate_prior_art(model):
    """G1, G2, G3, G4 on the committed Divine state, in the committed frame.

    Rebuilds the frame exactly as 08_hinge_eigenvalue.py does: raw A, shell Bn,
    d_committed = (A - Bn)/2, and the symmetric on-shell pair for the composed
    check. Everything here is a number the record already contains."""
    st = torch.load(DIVINE_STATE, weights_only=True)
    A_full = st["current_tensor"]
    nu = float(st["initial_norm"])
    f_map = make_f_map(model, nu)
    osm = OnShellMap(f_map, nu)

    with torch.no_grad():
        B_full = f_map(A_full)
        A2_full = f_map(B_full)
    Bn = B_full * (nu / B_full.norm())
    An = A_full * (nu / A_full.norm())
    D_committed = (A_full - Bn) / 2
    M_committed = (A_full + Bn) / 2
    Mn_committed = M_committed * (nu / M_committed.norm())
    D_sym = (An - Bn) / 2
    Mn_sym = (An + Bn) / 2
    Mn_sym = Mn_sym * (nu / Mn_sym.norm())
    dc, ds = unit(D_committed), unit(D_sym)

    out = {"cos_A_B": fcos(A_full[-1], Bn[-1]),
           "cos_A_ffA": fcos(A_full, A2_full)}

    # G1: the prior art's flip-axis eigenvalue, in the prior art's convention.
    jt = osm.jf(Mn_committed, dc)
    g1 = {"lambda": float(dc.flatten() @ jt.flatten()),
          "amplification": float(jt.norm()),
          "cos_Jt_minus_t": fcos(jt, -dc)}
    g1["passed"] = bool(
        abs(g1["lambda"] - PRIOR_ART["lambda_d_committed_at_Mn_committed"]) < GATE_ABS
        and abs(g1["amplification"]
                - PRIOR_ART["amplification_d_committed_at_Mn_committed"]) < GATE_ABS
        and abs(g1["cos_Jt_minus_t"]
                - PRIOR_ART["cos_Jt_minus_t_d_committed_at_Mn_committed"]) < GATE_ABS)
    out["G1_prior_art_flip_axis"] = g1

    # G1b: the symmetric pivot's -4.275, the other committed half-map number.
    jts = osm.jf(Mn_sym, ds)
    g1b = {"lambda": float(ds.flatten() @ jts.flatten())}
    g1b["passed"] = bool(abs(g1b["lambda"]
                             - PRIOR_ART["lambda_d_sym_at_Mn_sym"]) < GATE_ABS)
    out["G1b_prior_art_sym_pivot"] = g1b

    # G2: the radial null direction. J_f x = 0 exactly, to float32.
    jr = osm.jf(Mn_committed, unit(Mn_committed))
    g2 = {"ratio": float(jr.norm()) / float(jt.norm())}
    g2["passed"] = bool(g2["ratio"] < 1e-4)
    out["G2_radial_null"] = g2

    # G3: the composed two-pass chain, at the raw states the loop visits.
    v = osm.jf(A_full, ds)
    w = osm.jf(B_full, v)
    g3 = {"lambda_composed": float(ds.flatten() @ w.flatten())}
    g3["passed"] = bool(abs(g3["lambda_composed"]
                            - PRIOR_ART["lambda_composed_d_sym_start_A"]) < GATE_ABS)
    out["G3_composed_chain"] = g3

    # G4: finite-difference sensitivity. eps_rel 1e-3 is the gate; 1e-4 is
    # reported because it is already cancellation-limited in float32.
    g4 = {"jvp": g1["lambda"]}
    for rel in EPS_RELS:
        fd = osm.fd(Mn_committed, dc, rel)
        g4[f"fd_eps_{rel:g}"] = float(dc.flatten() @ fd.flatten())
    g4["rel_diff_1e-3"] = abs(g4["fd_eps_0.001"] - g4["jvp"]) / abs(g4["jvp"])
    g4["rel_diff_1e-4"] = abs(g4["fd_eps_0.0001"] - g4["jvp"]) / abs(g4["jvp"])
    g4["passed"] = bool(g4["rel_diff_1e-3"] < FD_GATE_REL)
    out["G4_fd_sensitivity"] = g4

    # Context, not gated: the committed pivot is not a fixed point, so its
    # half-map numbers are Rayleigh quotients and not multipliers of anything.
    with torch.no_grad():
        fM = f_map(Mn_committed)
    out["pivot_context"] = {
        "cos_fM_M": fcos(fM, Mn_committed),
        "onshell_scale_nu_over_norm_fM": nu / float(fM.norm()),
        "note": ("the committed pivot is only near-fixed, so the -0.864 and "
                 "-4.275 above are half-map Rayleigh quotients; this protocol "
                 "does not treat them as eigenvalues"),
    }
    out["n_jvp"] = osm.n_jvp
    return out


def run_verify():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "G0_solver_plumbing": gate_g0()}
    print(f"[verify] G0 solver plumbing: "
          f"{'PASS' if res['G0_solver_plumbing']['passed'] else 'FAIL'} "
          f"(max error {res['G0_solver_plumbing']['max_abs_error']:.2e})")
    model = sa.load_model()
    res.update(gate_prior_art(model))
    keys = [k for k in res if k.startswith("G")]
    res["passed"] = all(res[k]["passed"] for k in keys)
    for k in keys:
        print(f"[verify] {k}: {'PASS' if res[k]['passed'] else 'FAIL'} "
              + json.dumps({a: (round(b, 6) if isinstance(b, float) else b)
                            for a, b in res[k].items() if a != "passed"}))
    with open(VERIFY, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[verify] overall {'PASS' if res['passed'] else 'FAIL'} -> {VERIFY}")
    return res["passed"]


def verify_passed():
    if not VERIFY.exists():
        return False
    with open(VERIFY, encoding="utf-8") as f:
        return bool(json.load(f).get("passed"))


# ------------------------------------------------------------- one state ----
def states_to_probe():
    """The escape ladder's states, restricted to those whose zero-perturbation
    control passed. Sharing the list exactly is what makes the cross-experiment
    comparison in the report legitimate."""
    out = []
    for level, mult, label, pid in el.sweep_states():
        key = f"{level}_{label}_{pid}"
        ctrl = el.CONTROL_DIR / f"{key}.pt"
        if not ctrl.exists():
            continue
        if not torch.load(ctrl, map_location="cpu", weights_only=True)["passed"]:
            continue
        out.append((key, level, mult, label, pid))
    return out


def spectrum_for_state(model, prompt, state, target, key, named):
    """Everything this protocol measures at one attractor."""
    nu = float(target)
    y = state * (nu / state.norm())
    f_map = make_f_map(model, nu)
    osm = OnShellMap(f_map, nu)

    # The cascade must equal the engine's hook path, or none of this is the
    # loop's own map. 08_hinge_eigenvalue.py made the same check.
    with torch.no_grad():
        equiv = float((f_map(y) - hook_step(model, prompt, nu, y)).norm())
    assert equiv < 1e-3 * nu, f"f_map does not match the hook step: {equiv}"

    p, lag_table = detect_period(osm, y)
    rec = {"key": key, "nu": nu, "period": p, "lag_table": lag_table,
           "forward_equivalence_l2": equiv,
           "row_spread_state": row_spread(y)}
    if p is None or p > 2:
        rec["skipped"] = (f"period {p} is out of scope for this run; the "
                          "p-fold composition is defined but not run")
        return rec

    y, resid = refine(osm, y, p)
    rec["base_residual"] = resid
    rec["G6_base_refined"] = bool(resid < REFINE_TOL)
    bases = [y] if p == 1 else [y, osm.step(y)]
    op = TangentOperator(osm, bases)

    solves = []
    for seed in V0_SEEDS:
        t0 = time.time()
        vals, vecs, ritz, n_mv = arnoldi_eigs(
            op.matvec_np, op.n, K_EIGS, ARNOLDI_M, ARNOLDI_RESTARTS, seed)
        solves.append({"seed": seed, "vals": vals, "vecs": vecs, "ritz": ritz,
                       "n_matvec": n_mv, "seconds": time.time() - t0})
        print(f"    solve seed {seed}: {n_mv} matvecs, "
              f"{solves[-1]['seconds']:.0f}s, "
              f"|lambda| max {abs(vals).max():.4f}", flush=True)

    # G5: the two start vectors must agree on the leading set.
    m0 = np.sort(np.abs(solves[0]["vals"]))[::-1][:10]
    m1 = np.sort(np.abs(solves[1]["vals"]))[::-1][:10]
    rec["G5_seed_agreement"] = {
        "max_rel_diff_top10": float(np.max(np.abs(m0 - m1) / np.maximum(m0, 1e-12))),
    }
    rec["G5_seed_agreement"]["passed"] = bool(
        rec["G5_seed_agreement"]["max_rel_diff_top10"] < 1e-3)

    s = solves[0]
    modes = dedupe_pairs(s["vals"], s["vecs"], s["ritz"])
    for mode in modes:
        b = torch.from_numpy(np.stack(mode["basis"])).float()
        # A fresh residual with real matvecs, independent of the Arnoldi
        # identity, which can flatter a pair that has not settled.
        if mode["is_pair"]:
            a1 = op.apply(b[0]).numpy()
            a2 = op.apply(b[1]).numpy()
            lam = complex(mode["re"], mode["im"])
            z = (b[0].numpy() + 1j * b[1].numpy())
            az = a1 + 1j * a2
            mode["residual_direct"] = float(
                np.linalg.norm(az - lam * z) / max(abs(lam) * np.linalg.norm(z), 1e-30))
        else:
            a1 = op.apply(b[0]).numpy()
            mode["residual_direct"] = float(
                np.linalg.norm(a1 - mode["re"] * b[0].numpy())
                / max(abs(mode["re"]), 1e-30))
        mode["converged"] = bool(mode["residual_direct"] < 1e-3)
        mode["rho_per_pass"] = mode["modulus"] ** (1.0 / p)
        mode["kind"] = classify(mode, p)
        mode["row_spread"] = row_spread(b[0].view(y.shape))
        # The prior art's convention along the same direction, for continuity.
        mode["lambda_rayleigh_Jf"] = op.rayleigh_jf(b[0])
        mode["basis"] = b  # stored as a torch tensor in the checkpoint

    # G4 per state, on the leading five real basis vectors.
    fd_checks = []
    for mode in modes[:5]:
        t = unit(op.project(mode["basis"][0]).view(y.shape))
        entry = {"modulus": mode["modulus"],
                 "jvp": op.rayleigh_jf(t.flatten())}
        for rel in EPS_RELS:
            w = t
            for base in bases:
                w = osm.fd(base, unit(w), rel) * float(w.norm())
            entry[f"fd_eps_{rel:g}"] = float(t.flatten() @ w.flatten())
        denom = max(abs(entry["jvp"]), 1e-9)
        entry["rel_diff_1e-3"] = abs(entry["fd_eps_0.001"] - entry["jvp"]) / denom
        entry["rel_diff_1e-4"] = abs(entry["fd_eps_0.0001"] - entry["jvp"]) / denom
        fd_checks.append(entry)
    rec["G4_fd_sensitivity"] = fd_checks
    rec["G4_passed"] = bool(all(e["rel_diff_1e-3"] < FD_GATE_REL
                                for e in fd_checks if abs(e["jvp"]) > 1e-3))

    # Tier A: the two named escape-ladder directions, which are the only ones
    # with recoverable tensors. See the report for why the random ones are not.
    overlaps = {}
    for name in ("flip_axis", "glitch"):
        if name not in named:
            continue
        u = named[name].float().unsqueeze(0).expand(y.shape[0], -1).reshape(-1)
        u = op.project(u)
        if float(u.norm()) < 1e-9:
            continue
        u = u / u.norm()
        per_mode = []
        for i, mode in enumerate(modes):
            b = mode["basis"]
            c = math.sqrt(sum(float(u @ b[j]) ** 2 for j in range(b.shape[0])))
            per_mode.append(c)
        w = np.array(per_mode) ** 2
        rhos = np.array([m["rho_per_pass"] for m in modes])
        overlaps[name] = {
            "cos_per_mode": per_mode,
            "leading_subspace_cos": float(math.sqrt(float(w.sum()))),
            "rho_weighted": float((w * rhos).sum() / max(w.sum(), 1e-30)),
        }
    rec["named_direction_overlaps"] = overlaps
    rec["random_direction_expected_cos"] = {
        "per_eigenvector": 1.0 / math.sqrt(op.n),
        "leading_subspace": math.sqrt(len(modes) / op.n),
        "note": ("a random ladder direction carries essentially no "
                 "eigen-information at this dimension; this is why tier C "
                 "runs the ladder along the eigenvectors themselves"),
    }
    rec["modes"] = modes
    rec["n_matvec_total"] = sum(x["n_matvec"] for x in solves) + len(modes) * 2
    rec["n_jvp_total"] = osm.n_jvp
    rec["solve_seconds"] = [x["seconds"] for x in solves]
    return rec


def run_worker(worker, num_workers):
    if not verify_passed():
        sys.exit("[worker] verify.json missing or failed; run --verify first. "
                 "No new number is trusted until the machinery reproduces "
                 "08_hinge_eigenvalue's committed -0.863580.")
    import prompt_library
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    named = torch.load(el.DIRECTIONS, map_location="cpu", weights_only=False)
    model = sa.load_model()
    naturals = {}
    todo = [s for i, s in enumerate(states_to_probe()) if i % num_workers == worker]
    t0 = time.time()
    for n, (key, level, mult, label, pid) in enumerate(todo):
        ckpt = CKPT_DIR / f"{key}.pt"
        if ckpt.exists():
            continue
        prompt = prompt_library.PROMPT_LIBRARY[pid]
        if pid not in naturals:
            naturals[pid] = sa.natural_norm(model, prompt)
        pin = None if mult is None else mult * naturals[pid]
        state, target, lock_in, token, ok = el.converged_state(
            model, prompt, pin, expect_token=label)
        if not ok:
            print(f"[w{worker}] SKIP {key}: regenerated state reads {token!r}, "
                  f"the sweep recorded {label!r}", flush=True)
            continue
        print(f"[w{worker}] {n + 1}/{len(todo)} {key}", flush=True)
        rec = spectrum_for_state(model, prompt, state, target, key, named)
        rec.update({"level": level, "multiplier": mult, "label": label,
                    "pid": pid, "lock_in": lock_in, "home_token": token.strip()})
        tmp = ckpt.with_suffix(".tmp")
        torch.save(rec, tmp)
        tmp.rename(ckpt)
        if "modes" in rec:
            top = rec["modes"][0]
            print(f"[w{worker}] {key}: p={rec['period']}, leading |lambda| "
                  f"{top['modulus']:.4f} ({top['kind']}), "
                  f"rho/pass {top['rho_per_pass']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[w{worker}] slice complete ({time.time() - t0:.0f}s)")


# ------------------------------------------------- stage 3, the eigen-ladder --
def run_eigen_ladder(worker, num_workers):
    """Tier C: the identical escape ladder, run along the eigenvectors.

    Same rungs, same gate, same escape criterion as 01_escape_ladder.py, so the
    thresholds are directly comparable with the random-direction null band that
    experiment already measured. Registered decision rule: a state is skipped
    when its leading rho values have no spread worth testing, because P1 is
    untestable there; the rule is applied mechanically below and not after
    seeing any threshold."""
    import prompt_library
    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    model = sa.load_model()
    naturals = {}
    keys = sorted(p.stem for p in CKPT_DIR.glob("*.pt"))
    todo = [k for i, k in enumerate(keys) if i % num_workers == worker]
    for key in todo:
        out = LADDER_DIR / f"{key}.pt"
        if out.exists():
            continue
        rec = torch.load(CKPT_DIR / f"{key}.pt", map_location="cpu",
                         weights_only=False)
        modes = [m for m in rec.get("modes", []) if m["converged"]]
        if len(modes) < 2 * N_LADDER_EIGS:
            print(f"[e{worker}] SKIP {key}: only {len(modes)} converged modes")
            continue
        rhos = [m["rho_per_pass"] for m in modes]
        if max(rhos) - min(rhos) < 0.05:
            print(f"[e{worker}] SKIP {key}: leading rho spread "
                  f"{max(rhos) - min(rhos):.3f} is below the registered 0.05; "
                  "P1 is untestable at this state")
            continue
        pid, label = rec["pid"], rec["label"]
        prompt = prompt_library.PROMPT_LIBRARY[pid]
        if pid not in naturals:
            naturals[pid] = sa.natural_norm(model, prompt)
        pin = None if rec["multiplier"] is None else rec["multiplier"] * naturals[pid]
        state, target, _, token, ok = el.converged_state(
            model, prompt, pin, expect_token=label)
        if not ok:
            print(f"[e{worker}] SKIP {key}: state no longer reproduces")
            continue
        home_mean = state.mean(dim=0).clone()
        home_last = state[-1, :].clone()
        chosen = list(range(N_LADDER_EIGS)) + list(
            range(len(modes) - N_LADDER_EIGS, len(modes)))
        ladders = {}
        for i in sorted(set(chosen)):
            mode = modes[i]
            v = mode["basis"][0].reshape(-1).float()
            flat = state.reshape(-1)
            x_hat = flat / flat.norm()
            v = v - torch.dot(v, x_hat) * x_hat
            v = v / v.norm()
            rungs = el.run_ladder(model, prompt, state, target, v,
                                  token.strip(), home_mean, home_last)
            ladders[f"mode{i:02d}"] = {
                "rungs": rungs, "modulus": mode["modulus"],
                "rho_per_pass": mode["rho_per_pass"], "kind": mode["kind"],
                "threshold": el.threshold_of(rungs)}
            print(f"[e{worker}] {key} mode{i:02d} rho {mode['rho_per_pass']:.4f}: "
                  f"threshold {ladders[f'mode{i:02d}']['threshold']} deg", flush=True)
        tmp = out.with_suffix(".tmp")
        torch.save({"key": key, "ladders": ladders}, tmp)
        tmp.rename(out)


# ---------------------------------------------------------------- report ----
def spearman(xs, ys):
    """Rank correlation, written out because scipy is not in this env."""
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def write_report():
    recs = [torch.load(p, map_location="cpu", weights_only=False)
            for p in sorted(CKPT_DIR.glob("*.pt"))]
    ladders = {p.stem: torch.load(p, map_location="cpu", weights_only=False)
               for p in sorted(LADDER_DIR.glob("*.pt"))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"k_eigs": K_EIGS, "arnoldi_m": ARNOLDI_M,
                           "restarts": ARNOLDI_RESTARTS, "v0_seeds": V0_SEEDS,
                           "eps_rels": EPS_RELS},
                "spectra": recs, "eigen_ladders": ladders},
               OUT_DIR / "spectrum.pt")
    verify = json.load(open(VERIFY, encoding="utf-8")) if VERIFY.exists() else None

    L = ["# The local Jacobian spectrum of the loop map at settled attractors",
         "",
         "Registered before execution in `spectrum_protocol.md`. Raw data:",
         "`spectrum.pt`. Every number below is regenerated from the",
         "checkpoints; nothing is hand-computed.", ""]

    L += ["## Verification gate", ""]
    if verify is None:
        L += ["**No verify.json. Nothing below should be read.**", ""]
    else:
        L += [f"Overall: **{'PASS' if verify['passed'] else 'FAIL'}**.", "",
              "| gate | what it checks | result |", "|:--|:--|:--|"]
        names = {
            "G0_solver_plumbing": "the eigensolver against a dense matrix",
            "G1_prior_art_flip_axis":
                "reproduces the committed flip-axis eigenvalue -0.863580",
            "G1b_prior_art_sym_pivot": "reproduces the committed -4.275078",
            "G2_radial_null": "the rescale's null direction, J_f y = 0",
            "G3_composed_chain": "reproduces the committed composed +0.099339",
            "G4_fd_sensitivity": "jvp against central differences at 1e-3",
        }
        for k, what in names.items():
            if k in verify:
                L.append(f"| `{k}` | {what} | "
                         f"{'PASS' if verify[k]['passed'] else 'FAIL'} |")
        L += ["", "No new number in this report is to be read if the gate row "
              "for `G1_prior_art_flip_axis` says FAIL.", ""]

    L += ["## Leading spectrum per state", "",
          "`p` is the period at the smallest passing lag (the F15 rule).",
          "`lambda` is the on-shell multiplier, which includes the",
          "renormalisation's output projection and scale; `rho` is the",
          "per-pass rate `|lambda|^(1/p)`, which is what makes cycles and",
          "fixed points comparable. `lambda(Jf)` is the raw Rayleigh quotient",
          "of `J_f` along the same tangent direction, the committed record's",
          "convention, given for continuity; it is not the same object as the",
          "committed composed +0.099339, which is an unprojected chain at the",
          "raw states and is reproduced separately in gate G3.", "",
          "| state | p | base residual | leading |lambda| | rho/pass | kind |"
          " row spread | matvecs |",
          "|:--|--:|--:|--:|--:|:--|--:|--:|"]
    for r in recs:
        if "modes" not in r:
            L.append(f"| `{r['key']}` | {r.get('period')} | - | - | - | "
                     f"{r.get('skipped', 'no spectrum')} | - | - |")
            continue
        m = r["modes"][0]
        L.append(f"| `{r['key']}` | {r['period']} | {r['base_residual']:.2e} | "
                 f"{m['modulus']:.4f} | {m['rho_per_pass']:.4f} | {m['kind']} | "
                 f"{m['row_spread']:.2e} | {r['n_matvec_total']} |")

    for r in recs:
        if "modes" not in r:
            continue
        L += ["", f"### `{r['key']}`", "",
              f"Period {r['period']}, lag table "
              + ", ".join(f"{k}:{v:.5f}" for k, v in sorted(r['lag_table'].items()))
              + ".",
              f"Base-point residual {r['base_residual']:.2e} against the 1e-4 "
              f"gate: {'pass' if r['G6_base_refined'] else 'FAIL, provisional'}.",
              f"Seed agreement (G5) max relative difference over the top ten "
              f"moduli: {r['G5_seed_agreement']['max_rel_diff_top10']:.2e}, "
              f"{'pass' if r['G5_seed_agreement']['passed'] else 'FAIL'}.",
              f"Finite-difference sensitivity (G4): "
              f"{'pass' if r['G4_passed'] else 'FAIL'}; per-mode relative "
              "differences "
              + ", ".join(f"{e['rel_diff_1e-3']:.1e}/{e['rel_diff_1e-4']:.1e}"
                          for e in r["G4_fd_sensitivity"])
              + " at eps_rel 1e-3 and 1e-4.",
              "",
              "| # | Re | Im | \\|lambda\\| | rho/pass | kind | pair | residual"
              " | row spread | lambda(Jf) |",
              "|--:|--:|--:|--:|--:|:--|:--|--:|--:|--:|"]
        for i, m in enumerate(r["modes"]):
            floor = "below floor" if m["modulus"] < 1e-3 else f"{m['modulus']:.5f}"
            L.append(
                f"| {i} | {m['re']:+.5f} | {m['im']:+.5f} | {floor} | "
                f"{m['rho_per_pass']:.5f} | {m['kind']} | "
                f"{'yes' if m['is_pair'] else 'no'} | "
                f"{m['residual_direct']:.1e}{'' if m['converged'] else ' UNCONV'} | "
                f"{m['row_spread']:.1e} | {m['lambda_rayleigh_Jf']:+.5f} |")
        ov = r.get("named_direction_overlaps", {})
        if ov:
            L += ["", "Named escape-ladder directions against this eigenbasis:", ""]
            for name, o in ov.items():
                L.append(f"- `{name}`: cosine with the leading "
                         f"{len(r['modes'])}-mode subspace "
                         f"{o['leading_subspace_cos']:.4f}, largest single-mode "
                         f"cosine {max(o['cos_per_mode']):.4f} at mode "
                         f"{o['cos_per_mode'].index(max(o['cos_per_mode']))}, "
                         f"overlap-weighted rho {o['rho_weighted']:.4f}.")
            exp = r["random_direction_expected_cos"]
            L.append(f"- For scale, a random direction's expected cosine is "
                     f"{exp['per_eigenvector']:.4f} with any single "
                     f"eigenvector and {exp['leading_subspace']:.4f} with the "
                     f"whole leading subspace.")

    L += ["", "## The registered prediction", "",
          "P1: within an attractor, the directions whose multipliers are",
          "closest to 1 in modulus escape at the smallest angles. Confirmed at",
          "Spearman at or below -0.5 with the most marginal eigen-direction",
          "escaping below the random median in a majority of states; refuted",
          "above -0.2 or at positive sign. Anything between is ambiguous and",
          "is reported as ambiguous.", ""]
    if not ladders:
        L += ["**No eigen-ladders yet, so P1 is untested.** Stage 3 supplies",
              "the only version of this test with power: a random ladder",
              "direction's expected cosine with a single eigenvector is",
              "1/sqrt(N), about 0.011 here, so the ladder's own random",
              "directions cannot resolve the correlation. They are also",
              "irrecoverable: the ladder does not archive its direction",
              "tensors and seeds its generator with Python's salted string",
              "hash, so the same key gives a different seed in every process.",
              ""]
    else:
        rows = []
        for key, lad in ladders.items():
            for name, d in lad["ladders"].items():
                if d["threshold"] is not None:
                    rows.append((key, d["rho_per_pass"], d["threshold"]))
        L += ["| state | rho/pass | escape threshold (deg) |", "|:--|--:|--:|"]
        for key, rho, th in rows:
            L.append(f"| `{key}` | {rho:.4f} | {th} |")
        rs = spearman([r[1] for r in rows], [r[2] for r in rows])
        if rs is not None:
            verdict = ("CONFIRMED" if rs <= -0.5 else
                       "REFUTED" if rs > -0.2 else "AMBIGUOUS")
            L += ["", f"Pooled Spearman correlation between rho and escape "
                  f"threshold: **{rs:+.3f}** over {len(rows)} eigen-directions. "
                  f"Applying the pre-stated cut-offs mechanically: **{verdict}**. "
                  "Note that this is the pooled figure; the registered "
                  "criterion also requires the per-state comparison against "
                  "the random median, which is tabulated above."]
    L += ["", "## Reading", "",
          "The spectrum is the leading 24 of about 7679 eigenvalues and says",
          "nothing about the bulk. A Jacobian is local and cannot locate a",
          "basin boundary; the registered prediction is exactly the claim that",
          "the local object nevertheless orders the global thresholds, and it",
          "may be false. Everything is float32, so moduli below 1e-3 are",
          "reported as below the numerical floor. States are two prompts per",
          "label per level, a probe and not a sample (FINDINGS caveat 19), and",
          "any `Divine` state inherits caveat 14."]
    with open(OUT_DIR / "spectrum.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] {OUT_DIR / 'spectrum.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="run the gates; workers refuse until this passes")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--eigen-ladder", action="store_true",
                    help="stage 3: run the escape ladder along the eigenvectors")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if run_verify() else 1)
    elif args.report_only:
        write_report()
    elif args.eigen_ladder and args.worker is not None:
        run_eigen_ladder(args.worker, args.num_workers)
    elif args.worker is not None:
        run_worker(args.worker, args.num_workers)
    else:
        ap.print_help()
