"""Turn attribution: which components do the turning? (issue #119)

Run 18 established that the per-pass change in the state's DIRECTION, not its
size, governs the lower edge of the five-basin band. This experiment splits
that per-pass turn into its per-component shares: the model is 144 attention
heads and 12 feed-forward (MLP) blocks, each of which writes its own vector
onto the running residual stream every pass, so the total change decomposes
exactly into per-component terms plus the attention output biases.

The instrument matches run 18's: the mean vector across positions. Per
scheduled pass, with m_in the injected tensor's mean vector and m_out the
read-back tensor's mean vector, the realised turn direction t_hat is the unit
part of m_out perpendicular to m_in, and the total turn T is that part's
length. A component's signed share is its mean-vector write projected onto
t_hat; because the residual stream is additive, the shares (heads + MLPs +
the attention-bias bucket) sum to T exactly, which the probe asserts in
float64 every scheduled pass (closure residual recorded).

Direct contributions only: a component's write also changes what later
components read, and this decomposition does not see that. No ablations here
(registered scope limit).

Protocol (registered in #119 before execution):
  - Levels: m008, m040, m056, historical (about 71x natural), m384;
    multipliers are of each prompt's own natural entry norm, run 18's rule.
  - Prompts: first 10 of the library in library order.
  - Passes: 1..20, plus [lock-10, lock+10) where the trial locks.
  - Lock iterations come from the committed run 18 archive where it holds the
    trial (stage A: m008, historical); the stage C levels' per-trial archive
    was never committed, so m040/m056/m384 get fresh plain gated runs here,
    saved to plain_gated_results.pt (which incidentally repairs part of that
    archive gap for these ten prompts).

Run from the repo root:

    python3 experiments/head_turn/01_turn_attribution.py
    python3 experiments/head_turn/01_turn_attribution.py --report-only

Outputs (in experiments/head_turn/output/):
    plain_gated_results.pt  fresh gated runs for the three stage-C levels
    turn_results.pt         per-pass per-component shares, all levels
    turn_report.md          every headline number, regenerated from the data
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atr_engine import run_atr_gated  # noqa: E402
import prompt_library  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "nu_sweep_stage_a", REPO_ROOT / "experiments" / "nu_sweep" / "01_stage_a.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

OUT_DIR = HERE / "output"
SWEEP_OUT = REPO_ROOT / "experiments" / "nu_sweep" / "output"

LAYER_START, LAYER_END = 0, 11
MAX_ITER, THRESHOLD, PATIENCE = 1000, 0.999, 3
CHECK_EVERY, CHECK_START = 10, 100
GATE_OFF = 10 ** 9  # instrumented replay: the gate never fires

# (level, multiplier of the natural entry norm; None = the historical
# seed_j pin at the first pass's exit norm)
LEVELS = [("m008", 8), ("m040", 40), ("m056", 56),
          ("historical", None), ("m384", 384)]
N_PROMPTS = 10
EARLY_PASSES = list(range(1, 21))
LOCK_HALF_WIDTH = 10
# Concentration statistics skip passes whose total turn is below this floor,
# RELATIVE to the injected mean vector's length (a settled fixed point turns
# by an amount near the float32 rounding floor, and the top-k fraction of
# rounding noise is itself noise); skipped counts are reported.
TURN_FLOOR_REL = 1e-3


def component_labels(model):
    """156 component labels + the attention-bias bucket, in share order."""
    labels = [f"L{l}.H{h}" for l in range(LAYER_START, LAYER_END + 1)
              for h in range(model.cfg.n_heads)]
    labels += [f"L{l}.MLP" for l in range(LAYER_START, LAYER_END + 1)]
    labels.append("attn-bias")
    return labels


def probe_hook_names(model):
    """Per-head result and per-block MLP write hooks for the window."""
    names = []
    for l in range(LAYER_START, LAYER_END + 1):
        names.append(f"blocks.{l}.attn.hook_result")
        names.append(f"blocks.{l}.hook_mlp_out")
    return names


class TurnProbe:
    """Accumulates per-pass signed component shares of the realised turn."""

    def __init__(self, model, wanted_passes):
        self.model = model
        self.wanted = set(wanted_passes)
        self.rows = []
        self._bias = model.b_O[LAYER_START:LAYER_END + 1].to(torch.float64).sum(0)

    def __call__(self, i, inject_tensor, current_tensor, cache):
        if i not in self.wanted:
            return
        m_in = inject_tensor.to(torch.float64).mean(0)
        m_out = current_tensor.to(torch.float64).mean(0)
        u = m_in / m_in.norm().clamp(min=1e-30)
        delta = m_out - m_in
        d_perp = delta - (delta @ u) * u
        turn = d_perp.norm().item()
        t_hat = d_perp / max(turn, 1e-30)
        shares = torch.empty(
            (LAYER_END - LAYER_START + 1) * (self.model.cfg.n_heads + 1) + 1,
            dtype=torch.float64)
        k = 0
        for l in range(LAYER_START, LAYER_END + 1):
            res = cache[f"blocks.{l}.attn.hook_result"][0].to(torch.float64).mean(0)
            shares[k:k + self.model.cfg.n_heads] = res @ t_hat
            k += self.model.cfg.n_heads
        for l in range(LAYER_START, LAYER_END + 1):
            mlp = cache[f"blocks.{l}.hook_mlp_out"][0].to(torch.float64).mean(0)
            shares[k] = mlp @ t_hat
            k += 1
        shares[k] = self._bias @ t_hat
        closure_residual = shares.sum().item() - turn
        # Additivity self-check (registered): shares must sum to the turn.
        # Tolerance scales with the state's size, because the identity holds
        # only up to float32 rounding of 24 successive residual additions.
        m_in_norm = m_in.norm().item()
        assert abs(closure_residual) <= max(1e-3 * turn, 1e-4 * m_in_norm), (
            f"closure failed at pass {i}: sum(shares)-T = {closure_residual}")
        cos = (m_out @ u).item() / max(m_out.norm().item(), 1e-30)
        self.rows.append({
            "iteration": i,
            "turn": turn,
            "turn_rel": turn / max(m_in_norm, 1e-30),
            "angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
            "closure_residual": closure_residual,
            "shares": shares.clone(),
        })


def archive_lock(level, pid):
    """(lock_in_iter, converged, natural_norm, terminal_token) from the
    committed stage A archive, or None if the archive lacks the trial."""
    path = SWEEP_OUT / "stage_a_results.pt"
    if not path.exists():
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    r = d["results"].get(level, {}).get(pid)
    if r is None:
        return None
    return {"lock_in_iter": r["lock_in_iter"], "converged": r["converged"],
            "natural_norm": r.get("natural_norm"),
            "terminal_token": r["terminal_token"], "source": "stage_a_archive"}


def run():
    """Resolve lock windows, run the instrumented passes, write the report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = sa.load_model()
    model.set_use_attn_result(True)
    pids = list(prompt_library.PROMPT_LIBRARY)[:N_PROMPTS]
    labels = component_labels(model)
    hooks = probe_hook_names(model)
    gate_kw = dict(max_iter=MAX_ITER, threshold=THRESHOLD, patience=PATIENCE,
                   check_every=CHECK_EVERY, check_start=CHECK_START,
                   capture_terminal=True, record_metrics=True)

    plain_path = OUT_DIR / "plain_gated_results.pt"
    plain = (torch.load(plain_path, map_location="cpu", weights_only=False)
             if plain_path.exists() else {"results": {}})

    config = {
        "experiment": "turn_attribution", "registration": "issue #119",
        "levels": [lv for lv, _ in LEVELS], "prompts": pids,
        "layers": f"{LAYER_START}->{LAYER_END}",
        "early_passes": [EARLY_PASSES[0], EARLY_PASSES[-1]],
        "lock_half_width": LOCK_HALF_WIDTH, "turn_floor_rel": TURN_FLOOR_REL,
        "component_labels": labels, "torch": str(torch.__version__),
    }
    results = {}
    naturals = {}
    t0 = time.time()
    for level, mult in LEVELS:
        results[level] = {}
        for pid in pids:
            prompt = prompt_library.PROMPT_LIBRARY[pid]
            if pid not in naturals:
                naturals[pid] = sa.natural_norm(model, prompt)

            lock = archive_lock(level, pid)
            if lock is None:
                key = f"{level}_{pid}"
                if key not in plain["results"]:
                    renorm = "seed_j" if mult is None else mult * naturals[pid]
                    r = run_atr_gated(model, prompt, LAYER_START, LAYER_END,
                                      renorm=renorm, **gate_kw)
                    r.update({"pid": pid, "level": level,
                              "natural_norm": naturals[pid]})
                    plain["results"][key] = r
                    torch.save(plain, plain_path)
                    print(f"[plain] {level} {pid}: "
                          f"{'locked@%s' % r['lock_in_iter'] if r['converged'] else 'no-lock'} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                r = plain["results"][key]
                lock = {"lock_in_iter": r["lock_in_iter"],
                        "converged": r["converged"],
                        "natural_norm": r["natural_norm"],
                        "terminal_token": r["terminal_token"],
                        "source": "fresh_gated_run"}
            if lock["natural_norm"] is not None:
                # Archive-vs-recomputed consistency: same un-hooked pass.
                assert abs(lock["natural_norm"] - naturals[pid]) < 0.5, (
                    f"natural norm drift for {pid}: archive "
                    f"{lock['natural_norm']}, recomputed {naturals[pid]}")

            schedule = list(EARLY_PASSES)
            if lock["converged"]:
                li = lock["lock_in_iter"]
                schedule += [j for j in range(li - LOCK_HALF_WIDTH,
                                              li + LOCK_HALF_WIDTH)
                             if j > 0]
            schedule = sorted(set(schedule))
            probe = TurnProbe(model, schedule)
            renorm = "seed_j" if mult is None else mult * naturals[pid]
            r = run_atr_gated(model, prompt, LAYER_START, LAYER_END,
                              max_iter=max(schedule), check_start=GATE_OFF,
                              renorm=renorm, pass_probe=probe,
                              probe_names=hooks)
            results[level][pid] = {
                "lock": lock, "schedule": schedule,
                "natural_norm": naturals[pid],
                "target_norm": r["target_norm"],
                "passes": probe.rows,
            }
            torch.save({"config": config, "results": results},
                       OUT_DIR / "turn_results.pt")
            print(f"[probe] {level} {pid}: {len(probe.rows)} passes "
                  f"instrumented, lock source {lock['source']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    write_report(config, results)


def _window_rows(trial):
    """Split a trial's probe rows into (early, lock) windows."""
    early = [row for row in trial["passes"] if row["iteration"] <= 20]
    lock = [row for row in trial["passes"] if row["iteration"] > 20]
    return early, lock


def _aggregate(rows_by_trial, n_components):
    """Mean share fraction per component, mean top-k concentrations, and the
    number of passes skipped for a near-zero turn."""
    frac_sum = torch.zeros(n_components, dtype=torch.float64)
    top_counts = {1: [], 5: [], 20: []}
    used = skipped = 0
    for rows in rows_by_trial:
        for row in rows:
            if row["turn_rel"] <= TURN_FLOOR_REL:
                skipped += 1
                continue
            frac = row["shares"] / row["turn"]
            frac_sum += frac
            used += 1
            ranked = torch.sort(frac, descending=True).values
            for k in top_counts:
                top_counts[k].append(ranked[:k].sum().item())
    if used == 0:
        return None
    return {
        "mean_frac": frac_sum / used,
        "top": {k: sum(v) / len(v) for k, v in top_counts.items()},
        "used": used, "skipped": skipped,
    }


def write_report(config, results):
    """Regenerate turn_report.md; every published number originates here."""
    labels = config["component_labels"]
    lines = [
        "# Turn attribution: results",
        "",
        "Registered before execution in issue #119. Raw data:",
        "`turn_results.pt` (per-pass per-component signed shares, float64);",
        "fresh gated runs for the stage-C levels in `plain_gated_results.pt`.",
        "A component's share fraction is its signed share divided by that",
        "pass's total turn, so fractions sum to 1 per pass; the closure",
        "assert held on every instrumented pass.",
        "",
    ]
    top10_by_key = {}
    for level in config["levels"]:
        trials = results[level].values()
        for window, name in ((0, "passes 1-20"), (1, "the 20 passes around lock-in")):
            rows_by_trial = [_window_rows(t)[window] for t in trials]
            agg = _aggregate(rows_by_trial, len(labels))
            if agg is None:
                lines += [f"## {level}, {name}", "",
                          "No instrumented passes in this window (no trial "
                          "locks at this level).", ""]
                continue
            order = torch.argsort(agg["mean_frac"], descending=True)
            top10 = [(labels[i], agg["mean_frac"][i].item())
                     for i in order[:10]]
            top10_by_key[(level, window)] = [t for t, _ in top10]
            l11h8_rank = next(i for i, idx in enumerate(order.tolist())
                              if labels[idx] == "L11.H8") + 1
            lines += [
                f"## {level}, {name}",
                "",
                f"- Passes used {agg['used']}, skipped for near-zero turn "
                f"{agg['skipped']}.",
                f"- Mean concentration of the total turn: top 1 component "
                f"{agg['top'][1]:.1%}, top 5 {agg['top'][5]:.1%}, "
                f"top 20 {agg['top'][20]:.1%}.",
                f"- L11.H8's rank by mean share fraction: {l11h8_rank} "
                f"of {len(labels)}.",
                "",
                "| rank | component | mean share of the turn |",
                "|--:|:--|--:|",
            ]
            for n, (lab, frac) in enumerate(top10, 1):
                lines.append(f"| {n} | `{lab}` | {frac:.1%} |")
            lines.append("")

    lines += ["## Registered expectation checks", ""]
    e1 = []
    for level in ("m056", "historical"):
        trials = results[level].values()
        agg = _aggregate([_window_rows(t)[0] for t in trials], len(labels))
        if agg:
            e1.append((level, agg["top"][5]))
    lines.append(
        "1. Concentration (top 5 above 50% inside the band, passes 1-20): "
        + "; ".join(f"{lv} {v:.1%} ({'holds' if v > 0.5 else 'fails'})"
                    for lv, v in e1) + ".")
    a = top10_by_key.get(("m040", 0))
    b = top10_by_key.get(("m056", 0))
    if a and b:
        overlap = len(set(a) & set(b))
        lines.append(
            f"2. Ranking identity across the lower edge (top-10 overlap, "
            f"m040 vs m056, passes 1-20): {overlap}/10 shared.")
    lines += [
        "3. No prediction was offered about L11.H8; its rank is reported "
        "per level above.",
        "",
        "## Scope",
        "",
        "Direct contributions only; no ablation, so no causal claim about",
        "any component. The turn is measured on the mean vector across",
        "positions, matching run 18's instrument. Five levels and ten",
        "prompts is a probe, not a sweep. Interpretation lands in issue",
        "#119 and the findings record, not here.",
    ]
    with open(OUT_DIR / "turn_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'turn_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the report from saved results, no model run")
    args = ap.parse_args()
    if args.report_only:
        saved = torch.load(OUT_DIR / "turn_results.pt", map_location="cpu",
                           weights_only=False)
        write_report(saved["config"], saved["results"])
    else:
        run()
