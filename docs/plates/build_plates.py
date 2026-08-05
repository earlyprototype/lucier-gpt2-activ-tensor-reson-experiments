"""Rebuild the five plates from committed run archives (no model needed).

The plates are visual work drawn from this project's data. Nothing here runs
the model or produces new measurements: every number comes from archives
already committed, so the plates can be regenerated on any machine with the
repo and torch, in seconds.

    python3 docs/plates/build_plates.py

Each plate is a template in this directory with a single `__DATA__` token,
plus a JSON payload under data/. The build inlines the payload and writes the
self-contained page. Templates and payloads are both committed, so a plate can
be edited without recomputing its data and its data can be recomputed without
touching its design.

WHAT EACH PLATE DRAWS, and where its honesty limits are stated:

  i-iii_room.html      Plates I to III. Every settled state as a point cloud
                       (1,425 runs, 21 injection levels); the settling itself
                       as traces across four architectures; the sweep as a
                       curve in loudness, growth and turn. Projections by
                       principal components, variance retained printed on the
                       page.

  iv_bodies.html       Plate IV. Five prompts rendered as deforming bodies
                       over sixty passes. The surface is a fixed linear
                       reading of the state, so two bodies look alike exactly
                       when the states do. IMPORTANT: the shape coefficients
                       are whitened for legibility, which changes apparent
                       distances, so the separation figures reported on that
                       page are computed from the RAW state vectors instead.
                       An earlier version measured separation in the display's
                       own coordinates and reported the opposite conclusion.

  v_cyanotypes.html    Plate V. All 144 attention heads as specimen outlines
                       built from their own singular spectra, with the
                       stability class and leading eigenvalue from the earlier
                       per-head census. Each head was looped in isolation,
                       which is not what a head does inside the working model.

The 1024-dimensional models cannot share a projection basis with the
768-dimensional ones, so each model is projected separately in every plate
that spans architectures.
"""

import json
import math
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
sys.path.insert(0, str(ROOT))

NU_CKPT = ROOT / "experiments" / "nu_sweep" / "output" / "checkpoints"
TRAJ = ROOT / "experiments" / "sink_geometry" / "output" / "trajectories.pt"
ANGULAR = ROOT / "experiments" / "nu_sweep" / "output" / "angular_profile.json"
SPECTRAL = ROOT / "experiments" / "_DATA" / "EXP_009" / "009c_spectral_data.pt"
CENSUS = (ROOT / "experiments" / "gpt2_small" / "output_eigen_rescore"
          / "results.json")

K_SHAPE = 48        # shape components per body, plate IV
H_MODES = 20        # harmonics per specimen, plate V


def pca3(X, q=3):
    """Centre and project onto leading components; return coords and the
    fraction of variance they carry."""
    c = X - X.mean(0)
    _, S, V = torch.pca_lowrank(c, q=max(q + 3, 6))
    Y = c @ V[:, :q]
    var = (S ** 2 / (S ** 2).sum())[:q]
    return Y, [round(float(v), 4) for v in var], V


def unit_box(Y):
    """Scale coordinates into [-1, 1] on every axis."""
    lo, hi = Y.min(0).values, Y.max(0).values
    return (Y - lo) / (hi - lo) * 2 - 1


def spread(X):
    """Mean angular separation (1 minus cosine) over all distinct pairs."""
    Xn = torch.nn.functional.normalize(X, dim=1)
    G = Xn @ Xn.T
    n = G.shape[0]
    return float((1 - G[~torch.eye(n, dtype=bool)]).mean())


def build_cloud():
    """Plate I: every settled state, projected to three dimensions."""
    rows = []
    for p in sorted(NU_CKPT.glob("*.pt")):
        r = torch.load(p, map_location="cpu", weights_only=True)
        mult = r.get("multiplier")
        if mult is None and r.get("natural_norm"):
            mult = float(r["target_norm"]) / float(r["natural_norm"])
        rows.append((r["terminal_mean_vec"].float(), r["terminal_token"].strip(),
                     mult, r["level"], bool(r["converged"])))
    Y, var, _ = pca3(torch.stack([r[0] for r in rows]))
    Yn = unit_box(Y)
    return {"n": len(rows), "var": var,
            "pts": [[round(float(c), 4) for c in Yn[i]] for i in range(len(rows))],
            "tok": [r[1] for r in rows],
            "mult": [round(r[2], 3) if r[2] is not None else None for r in rows],
            "level": [r[3] for r in rows], "conv": [r[4] for r in rows]}


def build_traj():
    """Plate II: settling traces, projected per model."""
    d = torch.load(TRAJ, map_location="cpu", weights_only=False)
    out = {}
    for model, info in d["traj"].items():
        M = torch.stack([t.float() for t in info["means"]])
        Y, var, _ = pca3(M.reshape(-1, M.shape[-1]))
        Yn = unit_box(Y).reshape(M.shape[0], M.shape[1], 3)
        out[model] = {"var": round(sum(var), 4), "d_model": info["d_model"],
                      "n_layers": info["n_layers"],
                      "traj": [[[round(float(v), 4) for v in pt] for pt in run]
                               for run in Yn]}
    return out


def build_sweep():
    """Plate III: the sweep curve, straight from the angular profile."""
    with open(ANGULAR, encoding="utf-8") as f:
        prof = json.load(f)["profile"]
    keep = ("level", "multiplier", "gain", "turn_mean_10", "share_in_five",
            "dominant")
    return [{k: r[k] for k in keep} for r in prof]


def build_morph():
    """Plate IV: shape coefficients per pass, plus separation from RAW states.

    The coefficients are whitened so all 48 components contribute visibly to
    the geometry rather than the loudest two dominating. That is a display
    choice and it changes apparent distances, so the reported separation is
    measured on the raw state vectors, never on these coefficients."""
    d = torch.load(TRAJ, map_location="cpu", weights_only=False)
    out = {}
    for model, info in d["traj"].items():
        M = torch.stack([t.float() for t in info["means"]])
        flat = M.reshape(-1, M.shape[-1])
        c = flat - flat.mean(0)
        _, S, V = torch.pca_lowrank(c, q=K_SHAPE + 4)
        C = (c @ V[:, :K_SHAPE]).reshape(M.shape[0], M.shape[1], K_SHAPE)
        C = C / C.reshape(-1, K_SHAPE).std(0, keepdim=True).clamp_min(1e-6)
        dm = M.shape[-1]
        out[model] = {
            "K": K_SHAPE, "d_model": info["d_model"],
            "n_layers": info["n_layers"],
            "var": round(float((S ** 2 / (S ** 2).sum())[:K_SHAPE].sum()), 4),
            "spread_early": round(spread(M[:, :3].reshape(-1, dm)), 4),
            "spread_late": round(spread(M[:, -3:].reshape(-1, dm)), 4),
            "curve": [round(spread(M[:, i]), 4) for i in range(M.shape[1])],
            "amp": round(0.42 / C.pow(2).sum(-1).div(2).sqrt().mean().item(), 5),
            "coef": [[[round(float(v), 3) for v in st] for st in run]
                     for run in C]}
    return out


def build_heads():
    """Plate V: one specimen outline per attention head.

    Amplitudes are the singular values relative to the leading one, so the
    outline shows how quickly that head's spectrum falls away. Phases are a
    fixed deterministic read of its dominant direction. Class and leading
    eigenvalue are reproduced from the earlier census, not recomputed."""
    import numpy as np
    spec = torch.load(SPECTRAL, map_location="cpu", weights_only=False)
    with open(CENSUS, encoding="utf-8") as f:
        census = {h["head"]: h for h in json.load(f)["per_head"]}
    heads = []
    for layer in range(12):
        for head in range(12):
            key = f"L{layer}_H{head}"
            s, e = spec[key], census[key]
            sv = np.asarray(s["singular_values"], dtype=float)
            dv = np.asarray(s["dominant_vector"], dtype=float)
            heads.append({
                "k": key, "L": layer, "H": head,
                "a": [round(float(x), 4)
                      for x in sv[1:H_MODES + 1] / max(sv[0], 1e-9)],
                "p": [round(float(math.atan2(dv[i * 7 + 3], dv[i * 7 + 11])), 3)
                      for i in range(H_MODES)],
                "gap": round(float(s["spectral_gap"]), 3),
                "lam": round(float(e["lam1_modulus"]), 4),
                "cls": e["class"], "sign": e["lam1_sign"],
                "rot": (round(e["rotation_plane_fraction"], 4)
                        if e["rotation_plane_fraction"] is not None else None),
                "tok": s["top_tokens"][0][0]})
    return {"H": H_MODES, "heads": heads}


PLATES = [
    ("_tpl_room.html", "i-iii_room.html",
     lambda: {"cloud": build_cloud(), "traj": build_traj(),
              "sweep": build_sweep()}),
    ("_tpl_bodies.html", "iv_bodies.html",
     lambda: {"K": K_SHAPE, "m": build_morph()}),
    ("_tpl_cyanotypes.html", "v_cyanotypes.html", build_heads),
]


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    for tpl, out, fn in PLATES:
        payload = fn()
        blob = json.dumps(payload, separators=(",", ":"))
        html = (HERE / tpl).read_text(encoding="utf-8")
        if "__DATA__" not in html:
            sys.exit(f"[plates] {tpl} has no __DATA__ token")
        (HERE / out).write_text(html.replace("__DATA__", blob),
                                encoding="utf-8")
        (DATA / (out.rsplit(".", 1)[0] + ".json")).write_text(blob,
                                                              encoding="utf-8")
        print(f"[plates] {out}  ({len(blob) // 1024} KB of data)")
    print("[plates] done; open any of the html files directly in a browser")
