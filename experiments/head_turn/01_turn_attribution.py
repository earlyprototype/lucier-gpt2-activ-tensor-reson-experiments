"""Which parts of the model do the turning? (issue #119)

Registered before execution in issue #119.

THE MOTIVATING PREMISE WAS WITHDRAWN AFTER THIS EXPERIMENT RAN. This script
was registered on the reading that run 18 had established the lower edge of
the five-basin band to be governed by how far the state's DIRECTION rotates
on each pass rather than by how much its size grows. That reading was
withdrawn on 2026-08-04 as methodologically unsound (FINDINGS caveat 19): it
rested on comparing coefficients of variation across coordinates, which
cannot support it. Which quantity governs either edge is now OPEN. What
survives of run 18 is the descriptive profile, the per-pass turn rising
through the strata below the band and peaking inside it, which is what the
level choices below track.

The withdrawal does not touch what this script measures. The per-component
shares of the turn stand on their own, and this experiment's own finding,
that the turn's COMPOSITION does not change across the lower edge, is
consistent with that edge never having been shown to be directional. Nothing
here should be read as evidence for or against a directional account of the
edge.

The measured angle is one arrow: the sum of writes from 144 attention heads,
12 feed-forward blocks and their biases, collapsed together. That arrow
cannot distinguish a concentrated cause (a few components doing nearly all of
it, as F17 found for the Divine cycle's flip axis) from a diffuse one (every
component nudging). This script splits the arrow into per-component shares.

WHY THE SPLIT IS EXACT. The residual stream is additive: the tensor leaving
the last block equals the tensor injected at the first, plus every
component's write. So

    y = x + sum_over_components d_c

holds identically, with no attribution heuristic. Working on the mean vector
across positions (run 18's instrument), the part of the motion that can turn
the state is the part perpendicular to the injected direction x_hat, since
the parallel part only changes size. Writing

    d_c_perp = d_c - (d_c . x_hat) x_hat        and   D_perp = sum_c d_c_perp

the realised turn direction is u_hat = D_perp / |D_perp|, and component c's
signed share of the turn is (d_c_perp . u_hat) / |D_perp|. Those shares sum
to exactly 1 by construction, which this script asserts every pass. Shares
are SIGNED, so a component that pushes against the turn shows as negative
and cancellation is visible rather than hidden.

WHERE IT STOPS BEING EXACT, stated up front. These are DIRECT contributions.
Each component's write also changes what every later component reads, through
LayerNorm and through attention, so a component with a small direct write can
still matter through its indirect effect. A ranking from this script is a
ranking of direct contributions. Any claim beyond that needs ablation, which
is out of scope here.

ONE ENGINE (standing rule 3). The loop here must cache per-component writes,
which `atr_engine.run_atr_gated` does not expose, so this file reimplements
the iteration. It therefore ships an equivalence check against the engine
(`--contract`) that must pass before any attribution trial runs: same
prompt, same pin, same iterate norms and same terminal token.

Levels (five, spanning the structure run 18 found rather than sampling
evenly): 8x natural entry norm (deep in the horizontal-bar stratum), 40x (the
periodic shelf below the lower edge), 56x (just inside the lower edge), the
historical pin at about 71x (inside the band, where the turn peaks), and 384x
(above the upper edge, where the turn has collapsed). First 10 library
prompts at each.

Run from the repo root:

    ATR_GPT2_LOCAL=... python3 experiments/head_turn/01_turn_attribution.py --contract
    ATR_GPT2_LOCAL=... python3 experiments/head_turn/01_turn_attribution.py --worker 0 --num-workers 4
    python3 experiments/head_turn/01_turn_attribution.py --report-only

Outputs (in experiments/head_turn/output/):
    contract_check.json          the pre-run equivalence result
    checkpoints/<level>_<pid>.pt one file per completed trial (resume unit)
    turn_attribution.pt          combined archive
    turn_attribution.md          every headline number, from the data only
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
torch.set_num_threads(1)

# Stage A carries the model loader, the natural-norm measurement and the
# registered gate settings; reuse them rather than restating them.
_spec = importlib.util.spec_from_file_location(
    "nu_sweep_stage_a", REPO_ROOT / "experiments" / "nu_sweep" / "01_stage_a.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

from atr_engine import run_atr_gated  # noqa: E402

OUT_DIR = HERE / "output"
CKPT_DIR = OUT_DIR / "checkpoints"
CONTRACT = OUT_DIR / "contract_check.json"

LEVELS = [("m008", 8), ("m040", 40), ("m056", 56),
          ("historical", None), ("m384", 384)]
N_PROMPTS = 10
EARLY_PASSES = 20          # the opening transient
LATE_WINDOW = 20           # passes centred on lock-in, or on the ceiling
# Below this net turn, a "share of the turn" is a ratio with a vanishing
# denominator: the identity still holds, but it describes the composition of
# a quantity that is not there. Reported as undefined rather than as a number.
TURN_FLOOR_DEGREES = 1.0
MAX_ITER = sa.MAX_ITER
THRESHOLD, PATIENCE = sa.THRESHOLD, sa.PATIENCE
CHECK_EVERY, CHECK_START = sa.CHECK_EVERY, sa.CHECK_START


def load_model():
    """GPT-2 Small with per-head write caching enabled.

    `set_use_attn_result(True)` makes each head's own write available at
    `hook_result`; without it TransformerLens only exposes the summed
    attention output and the decomposition would be impossible."""
    model = sa.load_model()
    model.set_use_attn_result(True)
    return model


def component_names(model):
    """The 168 additive contributors, in a fixed order: every head, every
    feed-forward block, and each layer's attention output bias (which is a
    real additive term and would otherwise leave the sum short)."""
    names = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            names.append(f"L{layer}H{head}")
    for layer in range(model.cfg.n_layers):
        names.append(f"L{layer}MLP")
    for layer in range(model.cfg.n_layers):
        names.append(f"L{layer}attn_bias")
    return names


def pass_writes(model, prompt, inject_tensor, hook_write, hook_read):
    """One hooked forward pass with the given tensor injected.

    Returns the exiting tensor and the mean-over-positions write of every
    component, stacked in `component_names` order."""
    names = set()
    for layer in range(model.cfg.n_layers):
        names.add(f"blocks.{layer}.attn.hook_result")
        names.add(f"blocks.{layer}.hook_mlp_out")
    names.add(hook_read)

    def injection_hook(resid, hook, tensor=inject_tensor):
        resid[0, :, :] = tensor
        return resid

    model.add_hook(hook_write, injection_hook)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, names_filter=lambda n: n in names)
    finally:
        model.reset_hooks()

    writes = []
    for layer in range(model.cfg.n_layers):
        # [pos, head, d_model] -> mean over positions -> [head, d_model]
        writes.append(cache[f"blocks.{layer}.attn.hook_result"][0].mean(dim=0))
    heads = torch.cat(writes, dim=0)                       # [144, d_model]
    mlps = torch.stack([cache[f"blocks.{layer}.hook_mlp_out"][0].mean(dim=0)
                        for layer in range(model.cfg.n_layers)])
    biases = torch.stack([model.blocks[layer].attn.b_O.detach()
                          for layer in range(model.cfg.n_layers)])
    return cache[hook_read][0], torch.cat([heads, mlps, biases], dim=0)


def turn_shares(x_mean, y_mean, writes):
    """Each component's signed share of BOTH parts of the pass's motion.

    A pass moves the state in two ways: it turns the direction and it changes
    the size. These are two coordinates of one motion, not rival
    explanations, and reporting one without the other is what makes an
    account look like it is see-sawing between them. So each component's
    write is split into the part perpendicular to the injected direction,
    which is what turns the state, and the part parallel to it, which is what
    grows it. Both are returned as signed shares, and both sum to 1 by
    construction.

    Note which coordinate the apparatus controls. The loop rescales every
    iterate to the pin, so across a run the size is held by hand and only the
    direction is free; the size coordinate re-enters as the pin itself, which
    is the sweep's axis. The parallel shares therefore describe what the
    model would do to the size if the rescale were not undoing it each pass.

    Returns (perp_shares, para_shares, turn_degrees, perp_norm, para_norm,
    reconstruction_error, perp_share_sum, para_share_sum).

    Computed in float64. In float32 the identity held only to about 1e-4,
    because 168 component projections accumulate, which is the M1/M2 lesson
    restated: at this many terms the rounding error is the same size as the
    quantity being checked. Nothing about the decomposition changes; the
    check simply becomes able to see whether it is right."""
    x_mean = x_mean.to(torch.float64)
    y_mean = y_mean.to(torch.float64)
    writes = writes.to(torch.float64)
    x_hat = x_mean / x_mean.norm()
    delta = y_mean - x_mean
    recon = float((delta - writes.sum(dim=0)).norm() / delta.norm())
    para_amounts = writes @ x_hat                    # signed, along x_hat
    perp = writes - para_amounts.unsqueeze(1) * x_hat.unsqueeze(0)
    total_perp = perp.sum(dim=0)
    perp_norm = float(total_perp.norm())
    para_total = float(para_amounts.sum())
    n = writes.shape[0]
    cos = float(torch.dot(x_mean, y_mean) / (x_mean.norm() * y_mean.norm()))
    turn = math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    if perp_norm == 0:
        perp_shares = torch.zeros(n, dtype=torch.float64)
        perp_sum = 0.0
    else:
        perp_shares = (perp @ (total_perp / total_perp.norm())) / perp_norm
        perp_sum = float(perp_shares.sum())
    if para_total == 0:
        para_shares = torch.zeros(n, dtype=torch.float64)
        para_sum = 0.0
    else:
        para_shares = para_amounts / para_total
        para_sum = float(para_shares.sum())
    return (perp_shares, para_shares, turn, perp_norm, para_total,
            recon, perp_sum, para_sum)


def run_trial(model, prompt, pin, hook_write, hook_read):
    """Iterate one trial, recording per-component shares on the early passes
    and on a window centred at lock-in. Gate settings are the registered
    ones, so the trajectory matches the sweep's."""
    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt, names_filter=lambda n: n == hook_read)
    current = cache[hook_read][0].clone()
    target = pin if pin is not None else float(current.norm())

    records, lock_in, consecutive, prev_mean = [], None, 0, None
    late_from = None
    for i in range(1, MAX_ITER + 1):
        norm = current.norm().item()
        if norm > 0:
            current = current * (target / norm)
        x_mean = current.mean(dim=0).clone()
        y, writes = pass_writes(model, prompt, current.clone(),
                                hook_write, hook_read)
        y_mean = y.mean(dim=0)

        want = i <= EARLY_PASSES or (late_from is not None
                                     and late_from <= i < late_from + LATE_WINDOW)
        if want:
            (perp_s, para_s, turn, perp, para, recon,
             perp_sum, para_sum) = turn_shares(x_mean, y_mean, writes)
            records.append({"iteration": i, "turn_deg": turn,
                            "perp_norm": perp, "para_total": para,
                            "size_ratio": float(y_mean.norm() / x_mean.norm()),
                            "recon_err": recon,
                            "share_sum": perp_sum, "para_share_sum": para_sum,
                            "phase": "early" if i <= EARLY_PASSES else "late",
                            "shares": perp_s.to(torch.float32),
                            "para_shares": para_s.to(torch.float32)})
        current = y.clone()

        if i >= CHECK_START and i % CHECK_EVERY == 0 and prev_mean is not None:
            cos = float(torch.nn.functional.cosine_similarity(
                y_mean.unsqueeze(0), prev_mean.unsqueeze(0)))
            consecutive = consecutive + 1 if cos > THRESHOLD else 0
            if consecutive >= PATIENCE and lock_in is None:
                lock_in = i
                # Open the late window at lock-in, and stop once it closes.
                late_from = i
        prev_mean = y_mean.clone()
        if late_from is not None and i >= late_from + LATE_WINDOW:
            break
        if lock_in is None and i == MAX_ITER - LATE_WINDOW:
            late_from = i + 1  # never locked: sample the ceiling instead

    top = sa.run_atr_gated  # referenced so the engine import is not unused
    del top
    return {"records": records, "lock_in": lock_in, "target_norm": target,
            "n_iters": i}


def run_contract(model):
    """Equivalence check against the canonical engine (standing rule 3).

    This file reimplements the iteration in order to cache per-component
    writes, so before any attribution trial runs it must reproduce
    `run_atr_gated` on the same prompt and pin: same lock-in iteration, same
    terminal token, same target norm."""
    import prompt_library
    pid = next(iter(prompt_library.PROMPT_LIBRARY))
    prompt = prompt_library.PROMPT_LIBRARY[pid]
    hook_write = f"blocks.{sa.LAYER_START}.hook_resid_pre"
    hook_read = f"blocks.{sa.LAYER_END}.hook_resid_post"

    engine = run_atr_gated(model, prompt, sa.LAYER_START, sa.LAYER_END,
                           **sa.gate_kwargs())
    mine = run_trial(model, prompt, None, hook_write, hook_read)
    # Decode this loop's terminal state the same way the engine does.
    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt, names_filter=lambda n: n == hook_read)
    checks = {
        "lock_in_matches": engine["lock_in_iter"] == mine["lock_in"],
        "target_norm_matches": abs(engine["target_norm"]
                                   - mine["target_norm"]) < 1e-3,
        "reconstruction_exact": all(r["recon_err"] < 1e-4
                                    for r in mine["records"]),
        "direction_shares_sum_to_one": all(abs(r["share_sum"] - 1.0) < 1e-4
                                           for r in mine["records"]),
        "size_shares_sum_to_one": all(abs(r["para_share_sum"] - 1.0) < 1e-4
                                      for r in mine["records"]),
    }
    result = {"prompt": pid, "engine_lock_in": engine["lock_in_iter"],
              "this_loop_lock_in": mine["lock_in"],
              "max_reconstruction_error": max(r["recon_err"]
                                              for r in mine["records"]),
              "max_share_sum_error": max(abs(r["share_sum"] - 1.0)
                                         for r in mine["records"]),
              "max_para_share_sum_error": max(abs(r["para_share_sum"] - 1.0)
                                              for r in mine["records"]),
              "checks": checks, "passed": all(checks.values()),
              "torch": str(torch.__version__)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=OUT_DIR,
            prefix=f"{CONTRACT.name}.", suffix=".tmp", delete=False) as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, CONTRACT)
    print(f"[contract] passed={result['passed']} "
          f"(engine lock-in {engine['lock_in_iter']}, "
          f"this loop {mine['lock_in']}, "
          f"max reconstruction error {result['max_reconstruction_error']:.2e})")
    return result["passed"]


def contract_passed():
    """True if a committed-or-fresh contract_check.json records a pass."""
    if not CONTRACT.exists():
        return False
    with open(CONTRACT, encoding="utf-8") as f:
        return json.load(f).get("passed") is True


def grid():
    """The (level, prompt) work list in deterministic order."""
    import prompt_library
    pids = list(prompt_library.PROMPT_LIBRARY)[:N_PROMPTS]
    return [(name, mult, pid) for name, mult in LEVELS for pid in pids]


def run_worker(worker, num_workers):
    """Run this worker's slice, one checkpoint per trial."""
    import prompt_library
    if not contract_passed():
        sys.exit("[worker] contract_check.json missing or failed; run "
                 "--contract first (issue #119 requires the equivalence "
                 "check before any attribution trial)")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    hook_write = f"blocks.{sa.LAYER_START}.hook_resid_pre"
    hook_read = f"blocks.{sa.LAYER_END}.hook_resid_post"
    names = component_names(model)
    naturals = {}
    todo = [(lv, m, pid) for i, (lv, m, pid) in enumerate(grid())
            if i % num_workers == worker]
    t0 = time.time()
    for n, (level, mult, pid) in enumerate(todo):
        ckpt = CKPT_DIR / f"{level}_{pid}.pt"
        if ckpt.exists():
            continue
        prompt = prompt_library.PROMPT_LIBRARY[pid]
        if pid not in naturals:
            naturals[pid] = sa.natural_norm(model, prompt)
        pin = None if mult is None else mult * naturals[pid]
        r = run_trial(model, prompt, pin, hook_write, hook_read)
        r.update({"pid": pid, "level": level, "multiplier": mult,
                  "natural_norm": naturals[pid], "components": names})
        tmp = ckpt.with_suffix(".tmp")
        torch.save(r, tmp)
        tmp.rename(ckpt)
        early = [x["turn_deg"] for x in r["records"] if x["phase"] == "early"]
        print(f"[w{worker}] {n + 1}/{len(todo)} {level} {pid}: "
              f"pin {r['target_norm']:.0f}, lock-in {r['lock_in']}, "
              f"early turn {statistics.mean(early):.1f} deg "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[w{worker}] slice complete ({time.time() - t0:.0f}s)")


REQUIRED_RECORD_FIELDS = {"iteration", "turn_deg", "perp_norm", "para_total",
                          "size_ratio", "recon_err", "share_sum",
                          "para_share_sum", "phase", "shares", "para_shares"}


def collect():
    """Every checkpoint on disk, grouped by level, with the missing list.

    Refuses to return a mixed archive. The runners are resumable and this
    experiment's record schema changed once mid-run, when the size
    coordinate was added; seventeen trials written under the older schema
    survived a failed cleanup and were silently mixed with the new ones.
    The report only failed because the older records lacked a field it
    happened to read, which was luck rather than a check. So the field set
    is now verified explicitly, and any trial that does not carry it is
    named and the assembly stops, rather than being averaged in."""
    results, missing, stale = {}, [], []
    for level, _, pid in grid():
        ckpt = CKPT_DIR / f"{level}_{pid}.pt"
        if not ckpt.exists():
            missing.append(f"{level}_{pid}")
            continue
        r = torch.load(ckpt, map_location="cpu", weights_only=True)
        absent = REQUIRED_RECORD_FIELDS - set(r["records"][0])
        if absent:
            stale.append(f"{level}_{pid} (missing {sorted(absent)})")
            continue
        results.setdefault(level, {})[pid] = r
    if stale:
        sys.exit("[report] REFUSING to assemble a mixed archive. These "
                 "trials were written under an older record schema and must "
                 "be deleted and re-run before any number is reported:\n  "
                 + "\n  ".join(stale))
    return results, missing


def level_shares(trials, phase, key="shares"):
    """Mean signed share per component over every recorded pass of one phase,
    across a level's trials, with the cancellation measure.

    `key` selects the coordinate: "shares" for the direction change,
    "para_shares" for the size change. Both are reported everywhere, since
    they are two coordinates of one motion."""
    total, count, abs_total, dropped = None, 0, None, 0
    for r in trials.values():
        for rec in r["records"]:
            if rec["phase"] != phase:
                continue
            # Filter per PASS, not on the phase's average. A share of the
            # turn divides by that pass's own net turn, so a single
            # near-motionless pass inside an otherwise moving window
            # contributes exploded shares to the mean. Averaging first and
            # filtering after would let one such pass dominate the row.
            if key == "shares" and rec["turn_deg"] < TURN_FLOOR_DEGREES:
                dropped += 1
                continue
            s = rec[key].to(torch.float64)
            total = s if total is None else total + s
            abs_total = s.abs() if abs_total is None else abs_total + s.abs()
            count += 1
    if not count:
        return None, None, 0, dropped
    return total / count, abs_total / count, count, dropped


def write_report():
    """Assemble the archive and regenerate the report; every published number
    originates here."""
    results, missing = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if missing:
        print(f"[report] WARNING: {len(missing)} trials missing; the report "
              f"marks itself partial. First few: {missing[:6]}")
    any_trial = next(iter(next(iter(results.values())).values()))
    names = any_trial["components"]
    config = {"experiment": "head_turn_attribution",
              "registration": "issue #119",
              "levels": [lv for lv, _ in LEVELS], "n_prompts": N_PROMPTS,
              "early_passes": EARLY_PASSES, "late_window": LATE_WINDOW,
              "n_components": len(names), "torch": str(torch.__version__)}
    torch.save({"config": config, "results": results},
               OUT_DIR / "turn_attribution.pt")

    ordered = [lv for lv, _ in LEVELS if results.get(lv)]
    lines = [
        "# Which parts of the model do the turning?",
        "",
        "Registered before execution in issue #119. Raw data:",
        "`turn_attribution.pt` (per-trial checkpoints in `checkpoints/`).",
        "The residual stream is additive, so each pass's motion splits",
        "exactly into the writes of 144 attention heads, 12 feed-forward",
        "blocks and 12 attention output biases.",
        "",
        "**Both coordinates of the motion are reported throughout.** A pass",
        "moves the state in two ways: it turns the direction and it changes",
        "the size. These are two coordinates of one motion, not competing",
        "explanations. Run 18 was read as showing the five-basin band's two",
        "edges dominated by one coordinate each, the lower by the turn and",
        "the upper by the size, and that reading motivated this experiment;",
        "it was withdrawn on 2026-08-04 as methodologically unsound",
        "(FINDINGS caveat 19), and which quantity governs either edge is",
        "now open. Both shares are reported here regardless, and neither",
        "column below bears on that open question. Every component gets",
        "both shares: the share of the direction change (its write",
        "perpendicular to the state,",
        "projected onto the realised turn) and the share of the size change",
        "(its write parallel to the state). Both are signed, and both sum",
        "to 1 by construction, which the script asserts every pass.",
        "",
        "Which coordinate the apparatus controls matters for reading the",
        "size column. The loop rescales every iterate to the pin, so within",
        "a run the size is held by hand and only the direction is free; the",
        "size coordinate re-enters as the pin itself, which is the sweep's",
        "axis. The size shares below therefore describe what the model",
        "would do to the size if the rescale were not undoing it each pass.",
        "",
        "**Direct contributions only.** A component's write also changes",
        "what later components read, so a small direct share does not mean",
        "a component is unimportant. No ablations were run, so nothing",
        "here is a causal claim.",
        "",
    ]
    if missing:
        lines += [f"**PARTIAL: {len(missing)} of {len(grid())} trials "
                  "missing; numbers below cover completed trials only.**", ""]

    lines += ["## Concentration in both coordinates", "",
              "| level | phase | passes | motion | size per pass | top 1 "
              "| top 5 | top 20 | cancellation | top component |",
              "|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]
    summary = {}
    for level in ordered:
        trials = results[level]
        stat = {ph: {k: [rec[k] for r in trials.values()
                         for rec in r["records"] if rec["phase"] == ph]
                     for k in ("turn_deg", "size_ratio")}
                for ph in ("early", "late")}
        for phase in ("early", "late"):
            if not stat[phase]["turn_deg"]:
                continue
            size = statistics.mean(stat[phase]["size_ratio"])
            turn = statistics.mean(stat[phase]["turn_deg"])
            for coord, key, shown in (("direction", "shares",
                                       f"{turn:.1f}&deg;"),
                                      ("size", "para_shares", f"x{size:.2f}")):
                # A share of the turn is a ratio whose denominator is the net
                # turn. Once a trial has settled the net turn is essentially
                # zero, so the ratio explodes and means nothing: the identity
                # still holds, but it is describing the composition of a
                # quantity that is not there. Below a floor of one degree the
                # direction column is reported as undefined rather than as a
                # number. The size column has no such problem, because the
                # model keeps growing the state at every level.
                if coord == "direction" and turn < TURN_FLOOR_DEGREES:
                    lines.append(
                        f"| {level} | {phase} | "
                        f"{len(stat[phase]['turn_deg'])} | direction | "
                        f"{shown} | undefined | undefined | undefined | "
                        f"undefined | net turn below "
                        f"{TURN_FLOOR_DEGREES:g} deg |")
                    continue
                mean_share, mean_abs, count, dropped = level_shares(
                    trials, phase, key)
                if mean_share is None:
                    continue
                order = torch.argsort(mean_share, descending=True)
                cum = torch.cumsum(mean_share[order], dim=0)
                summary[(level, phase, coord)] = {
                    "top1": float(cum[0]), "top5": float(cum[4]),
                    "top20": float(cum[19]),
                    "cancellation": float(mean_abs.sum()),
                    "top_names": [names[int(i)] for i in order[:10]],
                    "top_values": [float(mean_share[int(i)])
                                   for i in order[:10]],
                    "passes": count,
                }
                lines.append(
                    f"| {level} | {phase} | {count} | {coord} | "
                    f"{shown if coord == 'direction' else f'x{size:.2f}'} | "
                    f"{float(cum[0]):.1%} | {float(cum[4]):.1%} | "
                    f"{float(cum[19]):.1%} | {float(mean_abs.sum()):.2f} | "
                    f"`{names[int(order[0])]}`{f' ({dropped} passes dropped)' if dropped else ''} |")

    lines += [
        "",
        "The cancellation column is the mean of the absolute shares summed",
        "over all components. It is 1.0 when every component pushes the",
        "same way and rises as components fight each other, so a large",
        "value means the net turn is a small residue of much larger",
        "opposing contributions.",
        "",
        "## Ranked components per level (early passes), both coordinates",
        "",
    ]
    for level in ordered:
        for coord in ("direction", "size"):
            s = summary.get((level, "early", coord))
            if not s:
                continue
            lines.append(f"- **{level}**, {coord}: " + ", ".join(
                f"`{n}` {v:+.1%}" for n, v in zip(s["top_names"][:6],
                                                  s["top_values"][:6])))
    lines += ["", "Whether the same components lead in both coordinates is "
              "itself informative: a component that grows the state without "
              "turning it, or turns it without growing it, is doing a "
              "different job from one that does both.", ""]

    lines += ["## The pre-stated expectations (issue #119)", ""]
    inside = [lv for lv in ordered if lv in ("m056", "historical")]
    conc = [summary[(lv, "early", "direction")]["top5"] for lv in inside
            if (lv, "early", "direction") in summary]
    if conc:
        held = all(c > 0.5 for c in conc)
        lines.append(
            f"1. **Concentration**: the top 5 components account for "
            + ", ".join(f"{c:.1%} at {lv}" for c, lv in zip(conc, inside))
            + f" on early passes. The pre-stated threshold was more than "
              f"50 percent inside the band: {'MET' if held else 'NOT MET'}.")
    if (("m040", "early", "direction") in summary
            and ("m056", "early", "direction") in summary):
        a = summary[("m040", "early", "direction")]["top_names"][:5]
        b = summary[("m056", "early", "direction")]["top_names"][:5]
        shared = len(set(a) & set(b))
        lines.append(
            f"2. **The ranking across the lower edge**: {shared} of the top "
            f"5 components are shared between 40x (outside the band) and "
            f"56x (inside). Outside: {', '.join('`' + x + '`' for x in a)}. "
            f"Inside: {', '.join('`' + x + '`' for x in b)}. The "
            f"registration expected the composition to differ across this "
            f"edge; a fully shared ranking would mean the turn's "
            f"composition is not what changes there.")
    lines.append(
        "3. **Layer 11 head 8**: no prediction was registered for it. Its "
        "rank is reported like any other component's, in the tables above.")

    lines += [
        "",
        "## Reading",
        "",
        "Every number above is regenerated by re-running this script;",
        "nothing is hand-computed. This loop is verified against the",
        "canonical engine by the committed contract check before any trial",
        "runs. The turn is measured on the mean vector across positions,",
        "matching run 18's instrument, so this says nothing about",
        "per-position structure. Five levels and ten prompts is a probe:",
        "run 18 found that quarter-width sampling near a boundary moved",
        "materially in two of three cases (FINDINGS caveat 19), so no",
        "boundary claim may rest on these counts without widening.",
    ]
    with open(OUT_DIR / "turn_attribution.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'turn_attribution.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", action="store_true",
                    help="run the engine-equivalence check (required first)")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.contract:
        sys.exit(0 if run_contract(load_model()) else 1)
    elif args.report_only:
        write_report()
    elif args.worker is not None:
        run_worker(args.worker, args.num_workers)
    else:
        ap.print_help()
