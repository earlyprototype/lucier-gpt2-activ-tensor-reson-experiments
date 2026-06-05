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
    style_3d, BASIN_COLOR, BASINS, OUT,
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


if __name__ == "__main__":
    terminals = parse_terminal_basins()
    paths = parse_pathways()
    p1 = anim1_reveal(terminals)
    print("->", p1)
    p3 = anim3_flow(paths, terminals)
    print("->", p3)
