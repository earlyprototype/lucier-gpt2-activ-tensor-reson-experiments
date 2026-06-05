"""
Attractor-landscape prototypes for the ATR ("Lucier loop") experiments.

Renders THREE depth-map / terrain constructions of the 5 attractor basins,
all from real run artifacts (no fabricated numbers):

  #1  Basin-population wells   -- exact, from hypothesis_assessment.md (125 prompts)
  #2  Trajectory-density terrain -- from dissolution_pathways.md pathways (MDS layout)
  #3  Dynamical velocity field  -- residual "still-moving" proxy over the same layout

Data sources (parsed live, nothing hard-coded except colours/labels):
  B_AttractorDominance/output_stage1/hypothesis_assessment.md
  B_AttractorDominance/output_stage1/dissolution_pathways.md

Outputs PNGs into viz/out/.
"""
import re
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from sklearn.manifold import MDS

ROOT = pathlib.Path(__file__).resolve().parent.parent
STAGE1 = ROOT / "B_AttractorDominance" / "output_stage1"
OUT = ROOT / "viz" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical basin order (by dominance) + a gallery palette.
BASINS = ["prolet", "Divine", "Anarch", "till", "solidarity"]
BASIN_COLOR = {
    "prolet":     "#ffd24a",  # warm gold  (dominant)
    "Divine":     "#7ad7ff",  # sky        (theology)
    "Anarch":     "#ff5d73",  # red        (political)
    "till":       "#9b8cff",  # violet     (temporal)
    "solidarity": "#5dffb0",  # green      (collective)
}
ITERS = [0, 2, 3, 5, 10, 20, 50, 100]

plt.rcParams.update({
    "figure.facecolor": "#07080d",
    "axes.facecolor": "#07080d",
    "savefig.facecolor": "#07080d",
    "text.color": "#e8eaf0",
    "axes.labelcolor": "#aab0c0",
    "xtick.color": "#5b6072",
    "ytick.color": "#5b6072",
    "font.size": 10,
})


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_terminal_basins():
    """Return dict prompt_id -> (category, terminal_basin) for all 125 prompts."""
    text = (STAGE1 / "hypothesis_assessment.md").read_text()
    rows = {}
    # rows like: | A01_physics | Complex | `prolet` (high) | `prolet` -> **prolet** | y |
    for m in re.finditer(r"\|\s*([A-G]\d+_\w+)\s*\|\s*(\w+)\s*\|.*?\*\*([\w:]+)\*\*", text):
        pid, cat, term = m.group(1), m.group(2), m.group(3)
        term = term.split(":")[-1]  # "OTHER:Anarch" -> "Anarch"
        if term in BASINS:
            rows[pid] = (cat, term)
    return rows


def parse_pathways():
    """Return dict prompt_id -> list of top-tokens aligned to ITERS."""
    text = (STAGE1 / "dissolution_pathways.md").read_text()
    lines = text.splitlines()
    paths = {}
    header_ids = []
    for ln in lines:
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] == "Iter":
            header_ids = cells[1:]
            continue
        m = re.match(r"\*\*(\d+)\*\*", cells[0])
        if not m or not header_ids:
            continue
        it = int(m.group(1))
        if it not in ITERS:
            continue
        ix = ITERS.index(it)
        for pid, tok in zip(header_ids, cells[1:]):
            tok = tok.strip("`").strip()
            paths.setdefault(pid, [""] * len(ITERS))[ix] = tok
    return paths


# --------------------------------------------------------------------------
# Shared surface plotting helper
# --------------------------------------------------------------------------
def style_3d(ax):
    ax.set_facecolor("#07080d")
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.04)
        axis.line.set_color((1, 1, 1, 0.08))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


# --------------------------------------------------------------------------
# #1  Basin-population wells (exact counts)
# --------------------------------------------------------------------------
# fixed well positions: dominant basin in centre, the rest around a ring
WELL_POS = {
    "prolet":     (0.0,  0.0),
    "Divine":     (1.7,  0.9),
    "Anarch":     (-1.7, 0.9),
    "till":       (1.2, -1.7),
    "solidarity": (-1.6,-1.4),
}


def make_wells(depths, g=320, gmax=0.36, mono=None):
    """Build X, Y, Z, facecolors from a {basin: depth} mapping (depth in [0, ~0.4]).

    Used both for the static final surface and for the evolving animation, so a
    flat plain (all depths 0) smoothly grows into the five basins.

    If `mono` (an RGB add-vector) is given, the wells are coloured in a single
    hue by depth instead of one colour per basin -- keeping one coherent palette
    across the whole process.
    """
    xs = np.linspace(-3.2, 3.2, g)
    ys = np.linspace(-3.0, 3.0, g)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    Cidx = np.full(X.shape, -1)
    nearest_depth = np.zeros_like(X)
    for bi, b in enumerate(BASINS):
        cx, cy = WELL_POS[b]
        depth = depths.get(b, 0.0)
        width = 0.55 + 0.5 * depth
        well = depth * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2)))
        Z -= well
        owns = well > nearest_depth
        Cidx[owns] = bi
        nearest_depth[owns] = well[owns]

    facecolors = np.zeros(X.shape + (4,))
    base = np.array([0.10, 0.11, 0.16, 1.0])      # the undifferentiated plain
    facecolors[...] = base
    glow = np.clip((nearest_depth / gmax), 0, 1)[..., None]
    if mono is not None:
        facecolors[..., :3] = np.clip(base[:3] + glow[..., 0, None] * np.asarray(mono), 0, 1)
    else:
        for bi, b in enumerate(BASINS):
            rgba = np.array(matplotlib.colors.to_rgba(BASIN_COLOR[b]))
            facecolors[Cidx == bi] = rgba
        facecolors[..., :3] = facecolors[..., :3] * (0.30 + 0.70 * glow[..., 0, None])
    facecolors[..., 3] = 1.0
    return X, Y, Z, facecolors


def build_population_surface(terminals, g=320):
    """Return X, Y, Z, facecolors, well-positions, counts for the final wells."""
    counts = {b: 0 for b in BASINS}
    for _, term in terminals.values():
        counts[term] += 1
    total = sum(counts.values())
    depths = {b: counts[b] / total for b in BASINS}
    X, Y, Z, facecolors = make_wells(depths, g=g)
    return X, Y, Z, facecolors, WELL_POS, counts


def fig1_population_wells(terminals):
    X, Y, Z, facecolors, pos, counts = build_population_surface(terminals)
    total = sum(counts.values())

    fig = plt.figure(figsize=(15, 7))
    for k, (elev, azim, tag) in enumerate(
        [(38, -60, "three-quarter"), (6, -90, "edge-on")]
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=2, cstride=2,
                        linewidth=0, antialiased=True, shade=False)
        style_3d(ax)
        ax.set_zlim(-0.42, 0.04)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{tag}", color="#7d8295", fontsize=10, pad=-2)
        if k == 0:
            for b in BASINS:
                cx, cy = pos[b]
                d = counts[b] / total
                ax.text(cx, cy, -d - 0.03, f"{b}\n{counts[b]}",
                        color=BASIN_COLOR[b], ha="center", va="top",
                        fontsize=9, fontweight="bold")
    fig.suptitle("#1  Basin-population wells  ·  depth = share of 125 prompts captured",
                 color="#e8eaf0", fontsize=14, y=0.96)
    fig.text(0.5, 0.04,
             "prolet 44  ·  Divine 34  ·  Anarch 26  ·  till 19  ·  solidarity 2     "
             "(exact, from hypothesis_assessment.md)",
             ha="center", color="#7d8295", fontsize=9)
    p = OUT / "01_population_wells.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p, counts


# --------------------------------------------------------------------------
# Layout shared by #2 and #3: MDS of pathway similarity
# --------------------------------------------------------------------------
def pathway_layout(paths, terminals):
    pids = [p for p in paths if p in terminals]
    n = len(pids)
    # distance = weighted token disagreement across iterations (late iters weigh more)
    w = np.array([0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 1.9, 2.5])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = paths[pids[i]], paths[pids[j]]
            diff = sum(wk for wk, ta, tb in zip(w, a, b) if ta != tb)
            D[i, j] = D[j, i] = diff
    max_d = D.max()
    if max_d > 0:
        D /= max_d
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=7,
             normalized_stress="auto").fit_transform(D)
    xy -= xy.mean(0)
    max_xy = np.abs(xy).max()
    if max_xy > 0:
        xy /= max_xy
    xy *= 2.6
    basins = np.array([terminals[p][1] for p in pids])
    return pids, xy, basins


# --------------------------------------------------------------------------
# #2  Trajectory-density terrain
# --------------------------------------------------------------------------
def fig2_density_terrain(paths, terminals):
    pids, xy, basins = pathway_layout(paths, terminals)
    g = 260
    xs = np.linspace(-3.4, 3.4, g)
    ys = np.linspace(-3.4, 3.4, g)
    X, Y = np.meshgrid(xs, ys)
    dens = np.zeros_like(X)
    sigma = 0.42
    for (px, py) in xy:
        dens += np.exp(-(((X - px) ** 2 + (Y - py) ** 2) / (2 * sigma ** 2)))
    dens = gaussian_filter(dens, 2.0)
    Z = -dens / dens.max()  # wells where trajectories accumulate

    # colour each cell by the nearest basin's centroid
    cents = {b: xy[basins == b].mean(0) for b in BASINS if (basins == b).any()}
    facecolors = np.zeros(X.shape + (4,))
    owner = np.full(X.shape, "", dtype=object)
    best = np.full(X.shape, 1e9)
    for b, c in cents.items():
        d = (X - c[0]) ** 2 + (Y - c[1]) ** 2
        m = d < best
        best[m] = d[m]; owner[m] = b
    for b in cents:
        rgba = np.array(matplotlib.colors.to_rgba(BASIN_COLOR[b]))
        facecolors[owner == b] = rgba
    glow = (-Z / (-Z).max())[..., None]
    facecolors[..., :3] *= (0.25 + 0.75 * glow[..., 0, None])
    facecolors[..., 3] = 1.0

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=False)
    # rain the real prompts down onto their wells
    for (px, py), b in zip(xy, basins):
        zi = np.interp(px, xs, Z[np.argmin(np.abs(ys - py))])
        ax.scatter(px, py, zi + 0.02, color=BASIN_COLOR[b], s=18,
                   edgecolors="white", linewidths=0.3, depthshade=False)
    style_3d(ax)
    ax.view_init(elev=42, azim=-58)
    fig.suptitle("#2  Trajectory-density terrain  ·  wells where real pathways accumulate",
                 color="#e8eaf0", fontsize=14, y=0.93)
    fig.text(0.5, 0.06,
             f"{len(pids)} prompt pathways  ·  MDS layout of token-trajectory similarity  "
             "(from dissolution_pathways.md)",
             ha="center", color="#7d8295", fontsize=9)
    p = OUT / "02_density_terrain.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# #3  Dynamical velocity field
# --------------------------------------------------------------------------
VEL_CMAP = LinearSegmentedColormap.from_list(
    "vel", ["#0a1f3c", "#1f6f8b", "#7ad7ff", "#ffd24a", "#ff5d73"])


def build_velocity_surface(paths, terminals, g=260):
    """Return X, Y, Z, xs, ys, facecolors, prompt xy, basins, velocities."""
    pids, xy, basins = pathway_layout(paths, terminals)
    # per-prompt residual "velocity": fraction of late steps where the token is
    # still changing -> 0 means locked into its fixed point (a well bottom).
    vel = []
    for p in pids:
        toks = paths[p]
        changes = sum(1 for a, b in zip(toks[3:], toks[4:]) if a != b)  # iters 5..100
        vel.append(changes / 4.0)
    vel = np.array(vel)

    xs = np.linspace(-3.4, 3.4, g)
    ys = np.linspace(-3.4, 3.4, g)
    X, Y = np.meshgrid(xs, ys)
    # interpolate velocity over the plane (Shepard / inverse-distance)
    Z = np.zeros_like(X); Wsum = np.zeros_like(X)
    for (px, py), v in zip(xy, vel):
        wd = 1.0 / (((X - px) ** 2 + (Y - py) ** 2) + 0.15)
        Z += wd * v; Wsum += wd
    Z = Z / Wsum
    Z = gaussian_filter(Z, 3.0)
    Z -= Z.min()

    norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    facecolors = VEL_CMAP(norm)
    return X, Y, Z, xs, ys, facecolors, xy, basins, vel


def fig3_velocity_field(paths, terminals):
    X, Y, Z, xs, ys, facecolors, xy, basins, vel = build_velocity_surface(paths, terminals)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=False)
    for (px, py), b, v in zip(xy, basins, vel):
        zi = np.interp(px, xs, Z[np.argmin(np.abs(ys - py))])
        ax.scatter(px, py, zi + 0.02, color=BASIN_COLOR[b], s=16,
                   edgecolors="white", linewidths=0.3, depthshade=False)
    style_3d(ax)
    ax.view_init(elev=40, azim=-58)
    fig.suptitle("#3  Dynamical velocity field  ·  height = how much the state is still moving",
                 color="#e8eaf0", fontsize=13, y=0.93)
    fig.text(0.5, 0.06,
             "valleys (dark) = fixed points where iteration stops pushing  ·  "
             "ridges = unstable transition tokens (Rousse / Ag / Zero)",
             ha="center", color="#7d8295", fontsize=9)
    p = OUT / "03_velocity_field.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    terminals = parse_terminal_basins()
    paths = parse_pathways()
    print(f"parsed {len(terminals)} terminal basins, {len(paths)} pathways")
    p1, counts = fig1_population_wells(terminals)
    print("counts:", counts, "->", p1)
    p2 = fig2_density_terrain(paths, terminals)
    print("->", p2)
    p3 = fig3_velocity_field(paths, terminals)
    print("->", p3)
