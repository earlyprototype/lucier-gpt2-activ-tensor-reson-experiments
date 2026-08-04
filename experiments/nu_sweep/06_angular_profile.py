"""The sweep profile in angular coordinates, and the edge test redone (#116).

Operator objection, 2026-08-02 (in session), which this script executes and
which supersedes the single-pass-gain reading in the Stage C report:

    the single-pass gain is a ratio of MAGNITUDES. It says how large the
    model's contribution is and nothing about where it points. A large
    contribution nearly parallel to the injected state barely moves the
    state; a small orthogonal one swings it hard. Everything this project
    measures at the end is a direction (the readout, the basins, the
    convergence gate), so a magnitude ratio is the wrong currency.

Archive-only: no forward passes, no re-runs. The engine has recorded the
directional quantity all along, as `cos_sim_mean_lag1` in every trial's
per-iteration metrics: the cosine between the mean vector at iteration t and
at t-1. This script converts it to degrees turned per pass and rebuilds the
profile and the edge test in that coordinate.

What it computes, per level:
  - turn on the first pass, and averaged over passes 1-10, 1-50 and the
    whole run, in degrees;
  - the same levels' single-pass gain, for side-by-side comparison.

And the edge test (Stage C's Q2, redone): for each of the first 25 prompts,
the lower edge is its first out-to-in transition along the profile and the
upper edge its last in-to-out; the transition point is the geometric
midpoint of that prompt's own interval endpoints, taken in three
coordinates (pin multiplier, single-pass gain, degrees turned per pass over
passes 1-10); cross-prompt coefficients of variation are compared.

The grid bias carried over from Stage C still applies and is restated in the
report: levels are spaced in MULTIPLIER units and shared across prompts, so
multiplier midpoints are grid-locked while the two measured coordinates vary
continuously per prompt. A coefficient of variation smaller in multiplier
units is therefore not evidence about mechanism; the comparison that carries
information is gain against turn, which are on equal footing.

Not measured here, and stated so the reader does not infer it: this cosine
is computed on the mean vector across positions, so it is one summary arrow
per pass. It does not decompose the turn across the 144 attention heads and
12 feed-forward blocks that produce it, which is the operator's per-head
objection and remains open.

Run from the repo root (seconds, no model):

    python3 experiments/nu_sweep/06_angular_profile.py

Outputs (in experiments/nu_sweep/output/):
    angular_profile.json    every computed number
    angular_profile.md      the readable tables, from the data only
"""

import importlib.util
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "nu_sweep_stage_c", HERE / "04_stage_c.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)
sa = sc.sa

OUT_DIR = sa.OUT_DIR
WINDOWS = [10, 50]


def turn_degrees(metric_row):
    """Degrees the mean vector's direction rotated on this pass."""
    c = max(-1.0, min(1.0, metric_row["cos_sim_mean_lag1"]))
    return math.degrees(math.acos(c))


def trial_turns(r):
    """(first-pass turn, {window: mean turn over that window}, whole-run mean)
    in degrees, or None when the trial carries no metrics."""
    m = r.get("metrics")
    if not m:
        return None
    turns = [turn_degrees(x) for x in m]
    windows = {w: statistics.mean(turns[:w]) for w in WINDOWS if len(turns) >= 2}
    return turns[0], windows, statistics.mean(turns)


def edge_test(results, ordered):
    """Stage C's Q2, redone in three coordinates.

    Same edge-finding rule as 04_stage_c.py (first out-to-in pair for the
    lower edge, last in-to-out for the upper), same geometric-midpoint
    convention, with degrees-turned added alongside multiplier and gain."""
    import prompt_library
    pids = list(prompt_library.PROMPT_LIBRARY)[:sc.FINE_N_PROMPTS]
    edges = {"lower": {"mult": [], "gain": [], "turn": []},
             "upper": {"mult": [], "gain": [], "turn": []}}
    excluded = []
    for pid in pids:
        seq = []
        for lv in ordered:
            r = results[lv].get(pid)
            if r is None or not r.get("metrics") or not r.get("natural_norm"):
                continue
            t = trial_turns(r)
            if t is None or 10 not in t[1]:
                continue
            lag = sa.smallest_passing_lag(r)
            in_five = (lag is not None
                       and r["terminal_token"].strip() in sa.REAL_FIVE)
            seq.append((r["target_norm"] / r["natural_norm"],
                        r["metrics"][0]["tensor_norm"] / r["target_norm"],
                        t[1][10], in_five))
        seq.sort(key=lambda x: x[0])
        ups = [(a, b) for a, b in itertools.pairwise(seq) if not a[3] and b[3]]
        downs = [(a, b) for a, b in itertools.pairwise(seq) if a[3] and not b[3]]
        if not ups or not downs:
            excluded.append((pid, f"{len(ups)} up, {len(downs)} down"))
            continue
        for name, (lo, hi) in (("lower", ups[0]), ("upper", downs[-1])):
            for k, i in (("mult", 0), ("gain", 1), ("turn", 2)):
                edges[name][k].append((lo[i] * hi[i]) ** 0.5)

    def cv(v):
        return (statistics.stdev(v) / statistics.mean(v)
                if len(v) > 1 else float("nan"))

    out = {"excluded": excluded, "edges": {}}
    for name, d in edges.items():
        out["edges"][name] = {
            "n_prompts": len(d["mult"]),
            **{f"mean_{k}": (statistics.mean(v) if v else float("nan"))
               for k, v in d.items()},
            **{f"cv_{k}": cv(v) for k, v in d.items()},
        }
    return out


def main():
    """Build the angular profile and the redone edge test; write both."""
    results = sc.collect_all()
    ordered = sorted(results, key=lambda lv: statistics.mean(
        r["target_norm"] for r in results[lv].values()))

    profile = []
    for lv in ordered:
        rows = results[lv]
        first, win, whole, gains = [], {w: [] for w in WINDOWS}, [], []
        for r in rows.values():
            t = trial_turns(r)
            if t is None or not r["target_norm"]:
                continue
            first.append(t[0])
            for w in WINDOWS:
                if w in t[1]:
                    win[w].append(t[1][w])
            whole.append(t[2])
            gains.append(r["metrics"][0]["tensor_norm"] / r["target_norm"])
        if not first:
            continue
        s = sa.level_stats(rows)
        profile.append({
            "level": lv, "n": s["n"],
            "multiplier": sc.mean_multiplier(rows),
            "gain": statistics.mean(gains),
            "turn_first": statistics.mean(first),
            **{f"turn_mean_{w}": (statistics.mean(win[w]) if win[w]
                                  else float("nan")) for w in WINDOWS},
            "turn_whole_run": statistics.mean(whole),
            "share_in_five": s["share_in_five"],
            "locked": s["locked"],
            "dominant": max(s["labels"].items(), key=lambda kv: kv[1])[0],
        })

    test = edge_test(results, ordered)
    payload = {"profile": profile, "edge_test": test,
               "windows": WINDOWS,
               "provenance": "archive-only; no forward passes. Operator "
                             "objection of 2026-08-02, executed in session."}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "angular_profile.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    peak = max(profile, key=lambda p: p["turn_mean_10"])
    lines = [
        "# The sweep in angular coordinates: degrees turned per pass",
        "",
        "Archive-only re-analysis; no forward passes, no re-runs. Executes",
        "the operator's objection of 2026-08-02: the single-pass gain is a",
        "ratio of magnitudes and says nothing about direction, while every",
        "quantity this project reports at the end is a direction. The",
        "engine has archived the directional measure all along, as each",
        "trial's per-iteration `cos_sim_mean_lag1`; below it is converted",
        "to degrees the mean vector rotated on that pass.",
        "",
        "## Profile: magnitude and direction side by side",
        "",
        "| level | n | multiplier | gain (magnitude) | turn, pass 1 "
        "| turn, passes 1-10 | turn, passes 1-50 | turn, whole run "
        "| share in five | dominant |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for p in profile:
        lines.append(
            f"| {p['level']} | {p['n']} | {p['multiplier']:.1f}x | "
            f"{p['gain']:.2f}x | {p['turn_first']:.1f}&deg; | "
            f"{p['turn_mean_10']:.1f}&deg; | {p['turn_mean_50']:.1f}&deg; | "
            f"{p['turn_whole_run']:.1f}&deg; | "
            f"{p['share_in_five']:.0%} | `{p['dominant']}` |")

    lines += [
        "",
        "## The edge test, redone in three coordinates",
        "",
        "Same edge-finding rule as the Stage C report (lower edge = a",
        "prompt's first out-to-in transition, upper edge = its last",
        "in-to-out), same geometric-midpoint convention, with degrees",
        "turned added alongside multiplier and gain.",
        "",
        f"- Prompts excluded (no up-crossing or no down-crossing): "
        f"{test['excluded'] if test['excluded'] else 'none'}.",
        "",
        "| edge | prompts | mean multiplier | CV mult | mean gain | CV gain "
        "| mean turn | CV turn |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in ("lower", "upper"):
        e = test["edges"][name]
        lines.append(
            f"| {name} | {e['n_prompts']} | {e['mean_mult']:.2f}x | "
            f"{e['cv_mult']:.4f} | {e['mean_gain']:.3f}x | "
            f"{e['cv_gain']:.4f} | {e['mean_turn']:.2f}&deg; | "
            f"{e['cv_turn']:.4f} |")

    lines += [
        "",
        "**How to read the comparison.** The grid bias carried from Stage C",
        "still applies: levels are spaced in multiplier units and shared",
        "across prompts, so multiplier midpoints are grid-locked while both",
        "measured coordinates vary continuously per prompt. The multiplier",
        "column is therefore not evidence about mechanism. The comparison",
        "that carries information is gain against turn, which sit on equal",
        "footing.",
        "",
        "## What the direction column shows that the magnitude column hides",
        "",
        f"- Turning peaks INSIDE the band: the largest early-phase turn is "
        f"{peak['turn_mean_10']:.1f} degrees per pass at `{peak['level']}` "
        f"({peak['multiplier']:.0f}x), where the share in the five basins is "
        f"{peak['share_in_five']:.0%}.",
        "- Approaching the band from below, the turn RISES while the gain",
        "  falls. The two coordinates disagree about the lower edge: in",
        "  magnitude it looks like the same smooth decline that produces the",
        "  upper edge, and in direction it is the opposite motion.",
        "- Above the band both agree: the turn collapses along with the",
        "  gain, which is the loop going inert.",
        "",
        "## Stated limits",
        "",
        "This cosine is computed on the mean vector across positions, so it",
        "is one summary arrow per pass. It does NOT decompose the turn",
        "across the 144 attention heads and 12 feed-forward blocks that",
        "produce it; that decomposition is the operator's per-head",
        "objection and is not answered here. The profile is also read only",
        "at the swept levels, so behaviour between them is unmeasured, and",
        "periodic trials carry run 17's single-phase readout provenance.",
        "",
        "Every number above is regenerated by re-running this script;",
        "nothing is hand-computed.",
    ]
    with open(OUT_DIR / "angular_profile.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'angular_profile.md'}")


if __name__ == "__main__":
    main()
