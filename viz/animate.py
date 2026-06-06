"""
MP4 animations of the attractor-landscape prototypes.

  anim1  Basin-population wells -- camera reveal: edge-on thin line rising to a
         three-quarter view while rotating, so the 5 wells deepen into view.
  anim3  Dynamical velocity field -- ~70 prompt particles released onto the
         surface and flowing downhill (gradient descent) until they pool into
         the basins, while the camera slowly orbits. Illustrative, not faithful.

Outputs MP4s into viz/out/.
"""
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from landscapes import (
    parse_terminal_basins, parse_pathways,
    build_population_surface, build_velocity_surface,
    make_wells, WELL_POS, style_3d, BASIN_COLOR, BASINS, ITERS, OUT,
)

FPS = 30


def _frame(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def ease(t):  # smoothstep 0..1
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# anim1 : population wells -- camera reveal
# --------------------------------------------------------------------------
def anim1_reveal(terminals, n=210):
    X, Y, Z, facecolors, pos, counts = build_population_surface(terminals, g=240)
    total = sum(counts.values())

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=False)
    style_3d(ax)
    ax.set_zlim(-0.42, 0.05)
    labels = []
    for b in BASINS:
        cx, cy = pos[b]; d = counts[b] / total
        labels.append((ax.text(cx, cy, -d - 0.03, f"{b}\n{counts[b]}",
                       color=BASIN_COLOR[b], ha="center", va="top",
                       fontsize=9, fontweight="bold"), -d - 0.03))
    fig.suptitle("Basin-population wells  ·  depth = share of 125 prompts captured",
                 color="#e8eaf0", fontsize=13, y=0.9)

    path = OUT / "anim1_population_wells.mp4"
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=16) as w:
        for i in range(n):
            t = i / (n - 1)
            # phase 1 (0-0.45): rise from edge-on (elev 2) to 40 while easing
            # phase 2 (0.45-1): hold elev, keep orbiting
            if t < 0.45:
                elev = 2 + ease(t / 0.45) * 38
            else:
                elev = 40
            azim = -90 + 300 * t                       # continuous orbit
            ax.view_init(elev=elev, azim=azim)
            # labels fade in during the reveal, hide while near edge-on
            vis = ease(min(1.0, max(0.0, (t - 0.30) / 0.25)))
            for txt, _ in labels:
                txt.set_alpha(vis)
            w.append_data(_frame(fig))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# anim3 : velocity field -- particles flowing downhill into the basins
# --------------------------------------------------------------------------
def anim3_flow(paths, terminals, n=240):
    X, Y, Z, xs, ys, facecolors, xy, basins, vel = build_velocity_surface(
        paths, terminals, g=200)

    # gradient of the surface for downhill flow
    dZy, dZx = np.gradient(Z, ys, xs)

    def grad_at(px, py):
        ix = np.clip(np.searchsorted(xs, px), 0, len(xs) - 1)
        iy = np.clip(np.searchsorted(ys, py), 0, len(ys) - 1)
        return dZx[iy, ix], dZy[iy, ix]

    def z_at(px, py):
        ix = np.clip(np.searchsorted(xs, px), 0, len(xs) - 1)
        iy = np.clip(np.searchsorted(ys, py), 0, len(ys) - 1)
        return Z[iy, ix]

    # release particles from a scattered cloud above the terrain
    rng = np.random.default_rng(3)
    m = len(xy)
    P = xy + rng.normal(0, 0.9, xy.shape)             # start near but jittered
    cols = [BASIN_COLOR[b] for b in basins]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    path = OUT / "anim3_velocity_flow.mp4"
    lr, damp = 0.06, 0.78
    V = np.zeros_like(P)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=16) as w:
        for i in range(n):
            ax.clear()
            ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=2, cstride=2,
                            linewidth=0, antialiased=True, shade=False, alpha=0.95)
            # integrate downhill motion (momentum + damping)
            for k in range(m):
                gx, gy = grad_at(P[k, 0], P[k, 1])
                V[k] = damp * V[k] - lr * np.array([gx, gy])
                P[k] += V[k]
            P[:, 0] = np.clip(P[:, 0], xs[0], xs[-1])
            P[:, 1] = np.clip(P[:, 1], ys[0], ys[-1])
            zs = np.array([z_at(px, py) for px, py in P]) + 0.015
            ax.scatter(P[:, 0], P[:, 1], zs, c=cols, s=22,
                       edgecolors="white", linewidths=0.3, depthshade=False)
            style_3d(ax)
            ax.view_init(elev=38, azim=-58 + 60 * (i / (n - 1)))
            fig.suptitle("Dynamical velocity field  ·  prompts flowing downhill into the basins",
                         color="#e8eaf0", fontsize=13, y=0.9)
            w.append_data(_frame(fig))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# anim2 : the wells carving out in real time as the iterations run
# --------------------------------------------------------------------------
def _pretty(tok):
    return {"↵": "newline", "―": "—", "": "·"}.get(tok, tok)


def per_iter_stats(paths):
    """For each scheduled iteration return: basin commit counts, the modal
    top-token (what the 'room' is muttering) and its unison fraction."""
    from collections import Counter
    n = len(ITERS); total = len(paths)
    counts = [{b: 0 for b in BASINS} for _ in range(n)]
    modal, unison = [], []
    for k in range(n):
        cnt = Counter()
        for toks in paths.values():
            t = toks[k]
            if t:
                cnt[t] += 1
            if t in BASINS:
                counts[k][t] += 1
        if cnt:
            tok, c = cnt.most_common(1)[0]
            modal.append(tok); unison.append(c / total)
        else:
            modal.append(""); unison.append(0.0)
    return counts, total, modal, unison


def anim2_evolve(paths):
    """Grow the five wells from a flat plain through the real iteration schedule.

    We DWELL on the transit phase (iters 0..20) because that is where the
    tension lives: the prompts lose their words, fall silent, then chant a
    meaningless fragment in unison. That unison drives a standing-wave ripple
    on the surface (resonance building) -- which only collapses into the five
    wells once the basins commit at iters 50..100."""
    counts, total, modal, unison = per_iter_stats(paths)
    depth_keys = [{b: counts[k][b] / total for b in BASINS} for k in range(len(ITERS))]
    committed = [sum(counts[k].values()) / total for k in range(len(ITERS))]
    n_seg = len(ITERS) - 1                          # 7 segments, s in [0..7]

    CYAN = np.array([0.35, 0.65, 1.0])              # one palette for the whole piece
    INK = "#cfd4e2"                                 # uniform label/caption colour

    # Build the per-frame s-timeline: long dwells early, plus a held beat at
    # iter 20 (the calm before the crystallisation).
    s_tl = []
    def hold(sv, n): s_tl.extend([sv] * n)
    def ramp(a, b, n): s_tl.extend(a + (i / n) * (b - a) for i in range(n))
    hold(0.0, 30)                                   # sit in the churn at iter 0
    # dwell early; let the late carving (segments 5,6) breathe at the same rate
    seg_frames = [30, 24, 28, 32, 36, 64, 72]
    for k in range(n_seg):
        ramp(k, k + 1, seg_frames[k])
        if k == 4:                                  # arrived at iter 20
            hold(5.0, 40)                           # the held breath before the snap
    hold(float(n_seg), 50)                          # rest on the final wells
    frames = len(s_tl)

    def lerp_node(arr, s):
        lo = int(np.floor(s)); hi = min(lo + 1, n_seg); f = s - lo
        return arr[lo] * (1 - f) + arr[hi] * f

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    title = fig.text(0.5, 0.93, "", color="#e8eaf0", fontsize=14, ha="center")
    say = fig.text(0.5, 0.86, "", color=INK, fontsize=15, ha="center",
                   fontstyle="italic")
    sub = fig.text(0.5, 0.075,
                   "depth = prompts committed to a basin  ·  ripple = unison of the "
                   "transit chant  ·  70-prompt subset",
                   color="#7d8295", fontsize=9, ha="center")

    path = OUT / "anim2_wells_evolve.mp4"
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=16) as w:
        for i, s in enumerate(s_tl):
            lo = int(np.floor(s)); hi = min(lo + 1, n_seg); f = ease(s - lo)
            depths = {b: depth_keys[lo][b] * (1 - f) + depth_keys[hi][b] * f
                      for b in BASINS}
            X, Y, Z, fc = make_wells(depths, g=220, mono=CYAN)

            # standing-wave churn: amplitude = unison * (1 - committed)
            u = lerp_node(unison, s); cm = lerp_node(committed, s)
            amp = 0.085 * u * (1 - cm)
            ph = i * 0.30
            churn = (np.sin(1.7 * X + ph) * np.sin(1.3 * Y - 0.7 * ph)
                     + 0.6 * np.sin(2.6 * X - 1.1 * ph) * np.sin(2.1 * Y + ph))
            Z = Z + amp * churn
            # make the crests glow in the SAME cyan so the resonance reads on the plain
            strength = amp / 0.085
            cr = np.clip(churn, 0, None)
            cr = (cr / (cr.max() + 1e-9))[..., None]
            fc = fc.copy()
            fc[..., :3] = np.clip(fc[..., :3] + 0.60 * strength * cr * CYAN, 0, 1)

            ax.clear()
            ax.plot_surface(X, Y, Z, facecolors=fc, rstride=2, cstride=2,
                            linewidth=0, antialiased=True, shade=False)
            style_3d(ax)
            ax.set_zlim(-0.42, 0.16)
            ax.view_init(elev=40, azim=-55 + 18 * (i / (frames - 1)))

            # smoothly interpolate the iteration readout between sparse readings
            it_val = round(ITERS[lo] * (1 - f) + ITERS[hi] * f)
            # use the same lo/hi/f logic for modal token to stay consistent
            tok = _pretty(modal[lo] if f < 0.5 else modal[hi])
            title.set_text(f"The wells forming  ·  iteration {it_val:>3} / 100")
            say.set_text(f'the room says:  “{tok}”')
            w.append_data(_frame(fig))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# gallery : textless, 1/4 speed, side-on 360 turn, palindrome loop
# --------------------------------------------------------------------------
def anim2_gallery(paths, speed=4):
    """The installation cut. No text. Everything at 1/4 speed (4x frames so the
    motion stays smooth). The wells form, the camera drops to a direct side-on
    profile and turns a slow 360, then the WHOLE piece plays in reverse -- a
    palindrome that loops seamlessly for ever.

    Rendered as an atomic, resumable PNG sequence so an interrupted run (these
    containers can pause) just continues where it left off."""
    counts, total, modal, unison = per_iter_stats(paths)
    depth_keys = [{b: counts[k][b] / total for b in BASINS} for k in range(len(ITERS))]
    committed = [sum(counts[k].values()) / total for k in range(len(ITERS))]
    n_seg = len(ITERS) - 1
    CYAN = np.array([0.35, 0.65, 1.0])

    # forward timeline of the well-forming (no final hold; the turn follows on)
    s_tl = []
    def hold(sv, n): s_tl.extend([sv] * n)
    def ramp(a, b, n): s_tl.extend(a + (i / n) * (b - a) for i in range(n))
    hold(0.0, 30 * speed)
    seg_frames = [f * speed for f in [30, 24, 28, 32, 36, 64, 72]]
    for k in range(n_seg):
        ramp(k, k + 1, seg_frames[k])
        if k == 4:
            hold(5.0, 40 * speed)
    ev = len(s_tl)

    def lerp_node(arr, s):
        lo = int(np.floor(s)); hi = min(lo + 1, n_seg); ff = s - lo
        return arr[lo] * (1 - ff) + arr[hi] * ff

    Xf, Yf, Zf, fcf = make_wells(depth_keys[-1], g=220, mono=CYAN)
    az_end = -55 + 18
    transition = 8 * speed
    spin = 100 * speed
    side_elev = 3.0

    # one ordered list of forward frames: ("form", i) | ("trans", j) | ("spin", j)
    spec = ([("form", i) for i in range(ev)]
            + [("trans", j) for j in range(transition)]
            + [("spin", j) for j in range(spin)])
    Nf = len(spec)

    fdir = OUT / "_frames"
    fdir.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw(X, Y, Z, fc, elev, azim):
        ax.clear()
        ax.plot_surface(X, Y, Z, facecolors=fc, rstride=2, cstride=2,
                        linewidth=0, antialiased=True, shade=False)
        ax.set_axis_off()
        ax.set_zlim(-0.42, 0.16)
        ax.view_init(elev=elev, azim=azim)

    for idx, (kind, a) in enumerate(spec):
        fp = fdir / f"{idx:05d}.png"
        if fp.exists():
            continue
        if kind == "form":
            i = a; s = s_tl[i]
            lo = int(np.floor(s)); hi = min(lo + 1, n_seg); f = ease(s - lo)
            depths = {b: depth_keys[lo][b] * (1 - f) + depth_keys[hi][b] * f
                      for b in BASINS}
            X, Y, Z, fc = make_wells(depths, g=220, mono=CYAN)
            u = lerp_node(unison, s); cm = lerp_node(committed, s)
            amp = 0.085 * u * (1 - cm)
            ph = i * 0.30 / speed
            churn = (np.sin(1.7 * X + ph) * np.sin(1.3 * Y - 0.7 * ph)
                     + 0.6 * np.sin(2.6 * X - 1.1 * ph) * np.sin(2.1 * Y + ph))
            Z = Z + amp * churn
            strength = amp / 0.085
            cr = np.clip(churn, 0, None); cr = (cr / (cr.max() + 1e-9))[..., None]
            fc = fc.copy()
            fc[..., :3] = np.clip(fc[..., :3] + 0.60 * strength * cr * CYAN, 0, 1)
            draw(X, Y, Z, fc, 40, -55 + 18 * (i / (ev - 1)))
        elif kind == "trans":
            t = ease((a + 1) / transition)
            draw(Xf, Yf, Zf, fcf, 40 + (side_elev - 40) * t, az_end)
        else:  # spin
            draw(Xf, Yf, Zf, fcf, side_elev, az_end + 360 * ((a + 1) / spin))
        tmp = fdir / f"{idx:05d}.tmp.png"
        imageio.imwrite(tmp, _frame(fig))
        tmp.replace(fp)                                 # atomic
        if idx % 100 == 0:
            print(f"frame {idx}/{Nf}", flush=True)
    plt.close(fig)
    print(f"forward frames ready: {Nf}", flush=True)

    # palindrome: forward, then forward reversed (skip duplicate end/start)
    out = OUT / "anim2_wells_loop.mp4"
    order = list(range(Nf)) + list(range(Nf - 2, 0, -1))
    with imageio.get_writer(out, fps=30, codec="libx264", quality=9,
                            macro_block_size=16) as w:
        for idx in order:
            w.append_data(imageio.imread(fdir / f"{idx:05d}.png"))
    print(f"LOOP done: {out} ({len(order)} frames)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    terminals = parse_terminal_basins()
    paths = parse_pathways()
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        print("->", anim1_reveal(terminals))
    if "2" in which:
        print("->", anim2_evolve(paths))
    if "3" in which:
        print("->", anim3_flow(paths, terminals))
    if "g" in which:
        print("->", anim2_gallery(paths))
