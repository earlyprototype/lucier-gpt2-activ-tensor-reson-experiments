"""Nu-sweep Stage C: the fine scan, in dimensionless coordinates (issue #116).

Registered before execution in issue #116; operator go of 2026-08-02 (in
session). Reuses Stage A's engine call, gate, F15 rule, checkpoint directory
and naming, and level_stats (one definition of the band statistic).

Parts executed by this script:

  Part 1  Fine levels, first 25 prompts in library order: m006, m012, m024
          (strata boundaries; the Q5 interval), m040, m048, m056 (lower band
          edge interior), m384, m512, m768, m1024 (upper edge hunt).
  Part 3  Shared-pin control: ALL 125 prompts at one shared numeric pin, the
          median first-pass exit norm over the full library, computed by
          --calibrate and recorded in shared_pin.json before any shared
          trial runs.
  Part 4  --report-only: per-level table over every level on disk (Stages A,
          B and C combined), each boundary stated in BOTH dimensionless
          coordinates (pin multiplier over natural entry norm; mean
          single-pass gain), and the Q2 scatter comparison: per-prompt
          in-five transition midpoints expressed in each coordinate, with
          the cross-prompt coefficient of variation of both. Pre-stated
          prediction (issue #116): scatter is smaller in the gain
          coordinate; equal or larger refutes the replacement-ratio
          hypothesis.

Part 2 (full-width brackets of any new 50 percent crossing) is a separate
run after Part 1's numbers exist, per the registration.

Run from the repo root:

    python3 experiments/nu_sweep/04_stage_c.py --calibrate
    python3 experiments/nu_sweep/04_stage_c.py --worker 0 --num-workers 4
    ...
    python3 experiments/nu_sweep/04_stage_c.py --report-only

Outputs (in experiments/nu_sweep/output/):
    shared_pin.json          the shared-pin calibration (per-prompt + median)
    checkpoints/<level>_<pid>.pt   shared with Stages A and B (resume unit)
    stage_c_results.pt       combined archive of the Stage C levels
    stage_c_report.md        every headline number, from the data only
"""

import argparse
import importlib.util
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "nu_sweep_stage_a", HERE / "01_stage_a.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

OUT_DIR = sa.OUT_DIR
CKPT_DIR = sa.CKPT_DIR
SHARED_PIN = OUT_DIR / "shared_pin.json"

FINE_LEVELS = ["m006", "m012", "m024", "m040", "m048", "m056",
               "m384", "m512", "m768", "m1024"]
FINE_N_PROMPTS = 25
SHARED_LEVEL = "shared"
SHARED_N_PROMPTS = 125


def exit_norm(model, prompt):
    """A prompt's first-pass exit norm: the Frobenius norm of
    blocks.11.hook_resid_post on an un-hooked forward pass, the quantity the
    historical renorm="seed_j" pins to."""
    name = f"blocks.{sa.LAYER_END}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt, names_filter=lambda n: n == name)
    return cache[name][0].norm().item()


def calibrate():
    """Measure every library prompt's first-pass exit norm and record the
    median as the shared pin (issue #116 Part 3). Deterministic; atomic
    write so concurrent callers converge on identical content."""
    import prompt_library
    model = sa.load_model()
    per_prompt = {}
    for pid in prompt_library.PROMPT_LIBRARY:
        per_prompt[pid] = exit_norm(model, prompt_library.PROMPT_LIBRARY[pid])
    med = statistics.median(per_prompt.values())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SHARED_PIN.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"median": med, "n_prompts": len(per_prompt),
                   "per_prompt": per_prompt}, f, indent=2)
    tmp.rename(SHARED_PIN)
    print(f"[calibrate] shared pin = median exit norm = {med:.2f} "
          f"over {len(per_prompt)} prompts -> {SHARED_PIN}")
    return med


def shared_pin_value():
    """The recorded shared pin; refuses to guess if calibration is absent."""
    if not SHARED_PIN.exists():
        sys.exit("[stage-c] shared_pin.json missing; run --calibrate first "
                 "(issue #116 Part 3 requires the pin recorded before any "
                 "shared trial)")
    with open(SHARED_PIN, encoding="utf-8") as f:
        return json.load(f)["median"]


def grid():
    """Stage C's (level, pid) work list: fine levels on the first 25
    prompts, the shared level on all 125."""
    import prompt_library
    pids = list(prompt_library.PROMPT_LIBRARY)
    work = [(lv, pid) for lv in FINE_LEVELS for pid in pids[:FINE_N_PROMPTS]]
    work += [(SHARED_LEVEL, pid) for pid in pids[:SHARED_N_PROMPTS]]
    return work


def run_worker(worker, num_workers):
    """Run this worker's slice, one checkpoint per trial. Mirrors Stage A's
    loop with one extra pin branch for the shared level; the statistic and
    everything downstream stay Stage A's."""
    import prompt_library
    if not sa.contract_passed():
        sys.exit("[worker] contract_check.json missing or failed")
    shared = shared_pin_value()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = sa.load_model()
    naturals = {}
    todo = [(lv, pid) for i, (lv, pid) in enumerate(grid())
            if i % num_workers == worker]
    t0 = time.time()
    for n, (level, pid) in enumerate(todo):
        ckpt = CKPT_DIR / f"{level}_{pid}.pt"
        if ckpt.exists():
            continue
        prompt = prompt_library.PROMPT_LIBRARY[pid]
        if pid not in naturals:
            naturals[pid] = sa.natural_norm(model, prompt)
        if level == SHARED_LEVEL:
            multiplier = None
            renorm = shared
        else:
            multiplier = int(level[1:])
            renorm = multiplier * naturals[pid]
        r = sa.run_atr_gated(model, prompt, sa.LAYER_START, sa.LAYER_END,
                             renorm=renorm, **sa.gate_kwargs())
        r.update({"pid": pid, "level": level, "multiplier": multiplier,
                  "natural_norm": naturals[pid]})
        tmp = ckpt.with_suffix(".tmp")
        torch.save(r, tmp)
        tmp.rename(ckpt)
        gate = (f"locked@{r['lock_in_iter']}" if r["converged"]
                else f"no-lock/{r['n_iters']}")
        print(f"[w{worker}] {n + 1}/{len(todo)} {level} {pid}: "
              f"pin {r['target_norm']:.0f} {r['terminal_token']!r} {gate} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[w{worker}] slice complete ({time.time() - t0:.0f}s)")


def collect_all():
    """Every checkpoint on disk (Stages A, B, C), grouped by level."""
    results = {}
    for ckpt in sorted(CKPT_DIR.glob("*.pt")):
        r = torch.load(ckpt, map_location="cpu", weights_only=True)
        results.setdefault(r["level"], {})[r["pid"]] = r
    return results


def mean_gain(rows):
    """Mean single-pass gain at injection over a level's trials."""
    gains = [r["metrics"][0]["tensor_norm"] / r["target_norm"]
             for r in rows.values() if r.get("metrics") and r["target_norm"]]
    return statistics.mean(gains) if gains else None


def mean_multiplier(rows):
    """Mean pin-over-natural multiplier over a level's trials (measured,
    not nominal, so the historical and shared levels get real values)."""
    vals = [r["target_norm"] / r["natural_norm"] for r in rows.values()
            if r.get("natural_norm")]
    return statistics.mean(vals) if vals else None


def q2_scatter(results, ordered):
    """The Q2 test (issue #116), corrected in execution.

    As registered this scan looked for a prompt with exactly ONE in-five
    flip along the profile. That was wrong: the band has two edges, so every
    prompt flips twice (out to in at the lower edge, in to out at the upper),
    and the first run of this function discarded 23 of 25 prompts. The
    corrected scan locates each edge separately: the LOWER crossing is the
    first out-to-in adjacent pair, the UPPER crossing the last in-to-out
    pair. For each edge and prompt, the transition point is the geometric
    midpoint of that prompt's own interval endpoints, taken in each
    coordinate (pin multiplier; the prompt's own measured single-pass gain),
    and the cross-prompt coefficient of variation is compared.

    Stated limitation, load-bearing for the reading: the level grid is
    defined in MULTIPLIER units and shared by every prompt, so a prompt's
    multiplier midpoint can only take one of the few discrete grid values,
    while its gain midpoint varies continuously with the prompt. The
    comparison is therefore biased toward the multiplier coordinate looking
    tighter, and a coefficient of variation smaller in multiplier units is
    NOT evidence against the replacement-ratio hypothesis. Only the reverse
    finding (gain tighter despite the bias) would be evidence for it. A fair
    test needs per-prompt bisection in gain units."""
    import prompt_library
    pids = list(prompt_library.PROMPT_LIBRARY)[:FINE_N_PROMPTS]
    edges = {"lower": {"mult": [], "gain": []},
             "upper": {"mult": [], "gain": []}}
    excluded = []
    for pid in pids:
        seq = []
        for lv in ordered:
            r = results[lv].get(pid)
            if r is None or not r.get("metrics"):
                continue
            lag = sa.smallest_passing_lag(r)
            in_five = (lag is not None
                       and r["terminal_token"].strip() in sa.REAL_FIVE)
            mult = (r["target_norm"] / r["natural_norm"]
                    if r.get("natural_norm") else None)
            gain = r["metrics"][0]["tensor_norm"] / r["target_norm"]
            if mult:
                seq.append((mult, gain, in_five))
        seq.sort(key=lambda t: t[0])
        ups = [(a, b) for a, b in itertools.pairwise(seq)
               if not a[2] and b[2]]
        downs = [(a, b) for a, b in itertools.pairwise(seq)
                 if a[2] and not b[2]]
        if not ups or not downs:
            excluded.append((pid, f"{len(ups)} up, {len(downs)} down"))
            continue
        for name, pair in (("lower", ups[0]), ("upper", downs[-1])):
            (m0, g0, _), (m1, g1, _) = pair
            edges[name]["mult"].append((m0 * m1) ** 0.5)
            edges[name]["gain"].append((g0 * g1) ** 0.5)

    def cv(vals):
        return (statistics.stdev(vals) / statistics.mean(vals)
                if len(vals) > 1 else float("nan"))

    out = {"excluded": excluded, "edges": {}}
    for name, d in edges.items():
        out["edges"][name] = {
            "n_prompts": len(d["mult"]),
            "cv_multiplier": cv(d["mult"]), "cv_gain": cv(d["gain"]),
            "mean_multiplier": (statistics.mean(d["mult"])
                                if d["mult"] else float("nan")),
            "mean_gain": (statistics.mean(d["gain"])
                          if d["gain"] else float("nan")),
            "distinct_multiplier_midpoints": len(set(
                round(v, 6) for v in d["mult"])),
            "midpoints_multiplier": d["mult"], "midpoints_gain": d["gain"],
        }
    return out


def write_report():
    """Assemble stage_c_results.pt and regenerate stage_c_report.md over the
    combined level set; every published number originates here."""
    results = collect_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_c_levels = set(FINE_LEVELS) | {SHARED_LEVEL}
    torch.save({"config": {"experiment": "nu_sweep_stage_c",
                           "registration": "issue #116",
                           "fine_levels": FINE_LEVELS,
                           "shared_level": SHARED_LEVEL,
                           "torch": str(torch.__version__)},
                "results": {lv: results[lv] for lv in results
                            if lv in stage_c_levels}},
               OUT_DIR / "stage_c_results.pt")

    ordered = sorted(
        results,
        key=lambda lv: statistics.mean(
            r["target_norm"] for r in results[lv].values()))
    stats = {lv: sa.level_stats(results[lv]) for lv in ordered}
    crossings = [
        (a, b) for a, b in itertools.pairwise(ordered)
        if (stats[a]["share_in_five"] > sa.BAND_THRESHOLD)
        != (stats[b]["share_in_five"] > sa.BAND_THRESHOLD)]
    scatter = q2_scatter(results, ordered)

    lines = [
        "# Nu-sweep Stage C: the fine scan, combined-profile report",
        "",
        "Registered before execution in issue #116. Raw data:",
        "`stage_c_results.pt` (checkpoints shared with Stages A and B; this",
        "table spans every level on disk from all three stages). The",
        "`shared` level pins all 125 prompts to ONE number, the median",
        "first-pass exit norm over the library (`shared_pin.json`).",
        "Coordinates are dimensionless: the measured pin-over-natural",
        "multiplier and the measured mean single-pass gain.",
        "",
        "## Per-level summary, all stages, ordered by mean pin",
        "",
        "| level | n | mean multiplier | mean gain | share in real five "
        "| locked (lag-1 gate) | distinct labels | dominant label |",
        "|:--|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for lv in ordered:
        s = stats[lv]
        rows = results[lv]
        top = max(s["labels"].items(), key=lambda kv: kv[1])
        mm = mean_multiplier(rows)
        mg = mean_gain(rows)
        lines.append(
            f"| {lv} | {s['n']} | "
            f"{mm:.1f} | {mg:.2f} | "
            f"{s['share_in_five']:.0%} ({s['n_in_five']}/{s['n']}) | "
            f"{s['locked']}/{s['n']} | {len(s['labels'])} | "
            f"`{top[0]}` {top[1]} |")
    lines += ["", "## Basin table, Stage C levels only", ""]
    for lv in ordered:
        if lv not in stage_c_levels:
            continue
        s = stats[lv]
        top = sorted(s["labels"].items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(f"- **{lv}**: "
                     + ", ".join(f"`{t}` {c}" for t, c in top[:10])
                     + (f", plus {len(top) - 10} more labels"
                        if len(top) > 10 else ""))
    lines += [
        "",
        "## Band-statistic crossings over the combined profile",
        "",
        f"- Crossing intervals (50 percent rule, pre-stated): "
        f"{[f'{a} to {b}' for a, b in crossings] if crossings else 'none'}.",
        "",
        "## Q2: transition scatter in the two coordinates (pre-stated test,",
        "corrected in execution)",
        "",
        "The registered scan looked for prompts with exactly ONE in-five",
        "flip. That was wrong: the band has two edges, so every prompt",
        "flips twice, and the first run discarded 23 of 25 prompts. The",
        "corrected scan measures each edge separately.",
        "",
        f"- Prompts excluded (no up-crossing or no down-crossing): "
        f"{scatter['excluded'] if scatter['excluded'] else 'none'}.",
        "",
        "| edge | prompts | mean transition (multiplier) | CV multiplier "
        "| mean transition (gain) | CV gain | distinct multiplier values |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in ("lower", "upper"):
        e = scatter["edges"][name]
        lines.append(
            f"| {name} | {e['n_prompts']} | {e['mean_multiplier']:.2f} | "
            f"{e['cv_multiplier']:.4f} | {e['mean_gain']:.3f} | "
            f"{e['cv_gain']:.4f} | {e['distinct_multiplier_midpoints']} |")
    lines += [
        "",
        "**Reading the comparison, with its limitation stated first.** The",
        "level grid is defined in MULTIPLIER units and shared by every",
        "prompt, so a prompt's multiplier midpoint can take only one of the",
        "few discrete grid values (the last column counts them), while its",
        "gain midpoint varies continuously with the prompt. The comparison",
        "is therefore biased toward multiplier looking tighter. A CV smaller",
        "in multiplier units is NOT evidence against the replacement-ratio",
        "hypothesis; only the reverse (gain tighter despite the bias) would",
        "be evidence for it. A fair test needs per-prompt bisection in gain",
        "units, which this grid cannot deliver.",
        "",
        "## Reading",
        "",
        "Every number above is regenerated by re-running this script;",
        "nothing is hand-computed. Interpretation lands in issue #116 and",
        "the findings record, not here. Periodic-trial labels carry run",
        "17's single-phase readout provenance; no per-trial phase audit",
        "exists for them.",
    ]
    with open(OUT_DIR / "stage_c_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'stage_c_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="record the shared pin (median library exit norm)")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    elif args.report_only:
        write_report()
    elif args.worker is not None:
        run_worker(args.worker, args.num_workers)
    else:
        ap.print_help()
