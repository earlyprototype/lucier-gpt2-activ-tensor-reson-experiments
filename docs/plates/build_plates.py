"""Rebuild the six plates from committed run archives (no model needed).

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

See README.md in this directory for the full log: what each plate is, where
each is published, and what each got wrong before it got it right.

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

  vi_river.html        Plate VI. The token flow across a thousand passes as a
                       braided cable, six models including a noise control.
                       The only plate with no projection at all: nodes carry
                       their own pass number, so nothing is distorted. The
                       arrangement within each column is a drawing choice and
                       carries no measurement.

The 1024-dimensional models cannot share a projection basis with the
768-dimensional ones, so each model is projected separately in every plate
that spans architectures.
"""

import json
import math
import pathlib
import sys

# The full rebuild needs torch (archives + the seeded projection below);
# --inline-only re-inlines committed payloads and must work without it.
try:
    import torch
except ModuleNotFoundError:
    torch = None

# torch.pca_lowrank uses randomised SVD, so without this the same archive
# produces a different projection on every build: axes flip sign, points move,
# and the committed payloads churn. A plate must be reproducible from its
# inputs or it is not evidence of anything.
if torch is not None:
    torch.manual_seed(20260805)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
sys.path.insert(0, str(ROOT))

NU_CKPT = ROOT / "experiments" / "nu_sweep" / "output" / "checkpoints"
TRAJ = ROOT / "experiments" / "sink_geometry" / "output" / "trajectories.pt"
ANGULAR = ROOT / "experiments" / "nu_sweep" / "output" / "angular_profile.json"
SPECTRAL = ROOT / "experiments" / "_DATA" / "EXP_009" / "009c_spectral_data.pt"
DISSOLUTION = ROOT / "docs" / "graph" / "_data" / "dissolution.json"
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


def build_river():
    """Plate VI: the token flow as an alluvial diagram.

    First attempt arranged each pass on a ring in three dimensions. It was
    unreadable: on a ring, strands wrap behind the cable and cross each other
    regardless of how the nodes are ordered, so the picture said "tangle" when
    the data says "narrowing". Barycentre ordering reduced the crossings and
    did not fix the form. The form was the problem, so this is flat.

    Horizontal position is the pass number and is read straight from the
    archive. Vertical position is a drawing choice: nodes are stacked in each
    column with height proportional to how many prompts hold that word, sorted
    by basin so a strand keeps its neighbours, then relaxed by barycentre
    ordering to reduce crossings. The stack height therefore shows the
    surviving population directly, which is the narrowing the piece is about.
    """
    basins = ["prolet", "Divine", "Anarch", "till", "solidarity"]
    with open(DISSOLUTION, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for name, m in d["models"].items():
        nodes, edges = m["nodes"], m["edges"]
        iters = sorted({n["iter"] for n in nodes})
        idx = {it: i for i, it in enumerate(iters)}
        cols = {}
        for n in nodes:
            cols.setdefault(n["iter"], []).append(n)
        for it in cols:
            cols[it].sort(key=lambda n: (basins.index(n["basin"])
                                         if n["basin"] in basins else 9,
                                         -n.get("count", 1), n["token"]))
        nbr = {}
        for e in edges:
            nbr.setdefault(e["from"], []).append(e["to"])
            nbr.setdefault(e["to"], []).append(e["from"])
        rank = {n["id"]: j for it in cols for j, n in enumerate(cols[it])}
        for sweep in range(30):
            for it in (iters if sweep % 2 == 0 else list(reversed(iters))):
                ns = cols[it]
                keyed = []
                for j, n in enumerate(ns):
                    xs = [rank[o] / max(len(cols[int(o.split("|")[1])]) - 1, 1)
                          for o in nbr.get(n["id"], []) if o in rank]
                    keyed.append((sum(xs) / len(xs) if xs
                                  else rank[n["id"]] / max(len(ns) - 1, 1), n))
                keyed.sort(key=lambda t: t[0])
                cols[it] = [n for _, n in keyed]
                for j, n in enumerate(cols[it]):
                    rank[n["id"]] = j

        # widest column sets the scale, so every column is comparable
        weight = {it: sum(n.get("count", 1) for n in ns) for it, ns in cols.items()}
        gap = 0.012
        widest = max(weight[it] + gap * max(len(cols[it]) - 1, 0) * 40
                     for it in cols)
        band = {}
        for it, ns in cols.items():
            total = weight[it] / widest + gap * max(len(ns) - 1, 0)
            y = -total / 2
            for n in ns:
                h = (n.get("count", 1) / widest)
                band[n["id"]] = [round(y, 5), round(y + h, 5)]
                y += h + gap
        pos = {nid: round(idx[int(nid.split("|")[1])]
                          / max(len(iters) - 1, 1), 5) for nid in band}
        # each node's outgoing flows partition its band, so ribbons meet cleanly
        offs_out, offs_in = {}, {}
        drawn = []
        for e in sorted(edges, key=lambda e: (e["from"], e["to"])):
            if e["from"] not in band or e["to"] not in band:
                continue
            c = e.get("count", 1) / widest
            a0 = offs_out.get(e["from"], band[e["from"]][0])
            b0 = offs_in.get(e["to"], band[e["to"]][0])
            drawn.append({"x0": pos[e["from"]], "x1": pos[e["to"]],
                          "y0": round(a0, 5), "y1": round(a0 + c, 5),
                          "z0": round(b0, 5), "z1": round(b0 + c, 5),
                          "c": e.get("count", 1), "bs": e.get("basin", "")})
            offs_out[e["from"]] = a0 + c
            offs_in[e["to"]] = b0 + c
        # colour by basin RANK within this model, since the models do not
        # share a basin vocabulary: GPT-2 small ends in five words, GPT-2
        # medium in one letter, Pythia 410M in nothing at all.
        shares = m.get("basin_shares") or {}
        ranked = sorted(shares, key=lambda k: -shares[k])[:5]
        rank_of = {b: i for i, b in enumerate(ranked)}
        for f in drawn:
            f["r"] = rank_of.get(f["bs"], -1)
        out[name] = {
            "label": m.get("label", name), "kind": m.get("kind", ""),
            "note": m.get("note", ""),
            "iters": iters, "basins": m.get("basins", {}),
            "shares": {k: round(v, 1) for k, v in shares.items()},
            "ranked": ranked, "n_basins": len(m.get("basins") or {}),
            "n_nodes": len(nodes),
            "cols": [{"it": it, "x": idx[it] / max(len(iters) - 1, 1),
                      "n": len(cols[it])} for it in iters],
            "nodes": [{"t": n["token"], "x": pos[n["id"]],
                       "y": band[n["id"]], "c": n.get("count", 1),
                       "b": n["basin"], "r": rank_of.get(n["basin"], -1),
                       "term": bool(n.get("terminal"))}
                      for it in iters for n in cols[it]],
            "flows": drawn}
    return out


PLATES = [
    ("_tpl_room.html", "i-iii_room.html",
     lambda: {"cloud": build_cloud(), "traj": build_traj(),
              "sweep": build_sweep()}),
    ("_tpl_bodies.html", "iv_bodies.html",
     lambda: {"K": K_SHAPE, "m": build_morph()}),
    ("_tpl_cyanotypes.html", "v_cyanotypes.html", build_heads),
    ("_tpl_river.html", "vi_river.html", build_river),
]


if __name__ == "__main__":
    # --inline-only: re-inline the COMMITTED payloads (data/*.json) into the
    # templates without recomputing anything. This is the path for editing a
    # plate's design: no torch, no archives, byte-identical data. The full
    # build below remains the path when the numbers themselves change.
    if "--inline-only" in sys.argv:
        for tpl, out, _fn in PLATES:
            blob = (DATA / (out.rsplit(".", 1)[0] + ".json")).read_text(
                encoding="utf-8")
            html = (HERE / tpl).read_text(encoding="utf-8")
            if "__DATA__" not in html:
                sys.exit(f"[plates] {tpl} has no __DATA__ token")
            (HERE / out).write_text(html.replace("__DATA__", blob),
                                    encoding="utf-8")
            print(f"[plates] {out}  (re-inlined committed payload)")
        sys.exit(0)
    if torch is None:
        sys.exit("[plates] the full rebuild needs torch; "
                 "use --inline-only to re-inline the committed payloads")
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
