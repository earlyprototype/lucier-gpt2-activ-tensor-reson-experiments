#!/usr/bin/env python3
"""
build_dissolution_graph.py — DISSOLUTION GRAPH data generator.

Parses the `dissolution_pathways.md` tables emitted by the ATR (Lucier) attractor
sweeps and turns them into a layered transition graph, one per model:

    prompts -> intermediate tokens (per iteration band) -> terminal attractor basins

Reads (whatever exists on disk; missing models are skipped with a warning):

    experiments/gpt2_small/output/dissolution_pathways.md
    experiments/gpt2_medium/output/dissolution_pathways.md
    experiments/pythia_160m/output/dissolution_pathways.md
    experiments/pythia_410m/output/dissolution_pathways.md
    experiments/pythia_410m/output_deep/dissolution_pathways.md
    experiments/gpt2_small/output_random_baseline/dissolution_pathways_random.md

Writes:

    docs/graph/_data/dissolution.json

Idempotent, takes no arguments, never crashes on an absent model.

IMPORTANT PROVENANCE NOTE (do not "fix" this by scaling):
the published markdown tables show only the FIRST ~10 prompts of each register
(the notebook truncated the printed columns), so the graph is built from the
70-prompt visible subset of the 125-prompt sweep. Basin shares computed here are
therefore shares OF THE VISIBLE SUBSET at the iteration-100 snapshot, which is a
different quantity from both the README's convergence-GATED shares (prolet 43.2%)
and FINDINGS' full-125 iteration-100 snapshot (prolet 35.2%). The script prints
all three side by side so the discrepancy is visible rather than massaged.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent           # docs/graph
REPO = HERE.parent.parent                        # repo root
OUT_DIR = HERE / "_data"
OUT_PATH = OUT_DIR / "dissolution.json"

# Token used to display a decoded token that is the empty string (the tables
# contain literal empty backtick cells for tokens that decode to "").
EMPTY_TOKEN = "∅"  # ∅

# --------------------------------------------------------------------------
# Model registry.  `prefix` is the short node-id namespace, e.g. "s|10|capit".
# --------------------------------------------------------------------------

MODEL_SPECS = [
    {
        "key": "gpt2-small",
        "prefix": "s",
        "label": "GPT-2 Small (124M)",
        "params": "124M",
        "corpus": "WebText",
        "path": "experiments/gpt2_small/output/dissolution_pathways.md",
        "kind": "model",
        "note": "Five semantic attractor basins (F1).",
    },
    {
        "key": "gpt2-medium",
        "prefix": "m",
        "label": "GPT-2 Medium (345M)",
        "params": "345M",
        "corpus": "WebText",
        "path": "experiments/gpt2_medium/output/dissolution_pathways.md",
        "kind": "model",
        "note": "Single funnel: every prompt collapses to `D`, locked by iter 10 (F3).",
    },
    {
        "key": "pythia-160m",
        "prefix": "p160",
        "label": "Pythia-160m",
        "params": "160M",
        "corpus": "The Pile",
        "path": "experiments/pythia_160m/output/dissolution_pathways.md",
        "kind": "model",
        "note": "Single funnel: `questioned`, saturated by iter 10 (F3).",
    },
    {
        "key": "pythia-410m",
        "prefix": "p410",
        "label": "Pythia-410m",
        "params": "410M",
        "corpus": "The Pile",
        "path": "experiments/pythia_410m/output/dissolution_pathways.md",
        "kind": "model",
        "note": "No consolidation; fragments never settle (F3).",
    },
    {
        "key": "pythia-410m-deep",
        "prefix": "p410d",
        "label": "Pythia-410m (deep, 1000 iters)",
        "params": "410M",
        "corpus": "The Pile",
        "path": "experiments/pythia_410m/output_deep/dissolution_pathways.md",
        "kind": "model",
        "note": "8-prompt deep-convergence subset run out to 1000 iterations.",
    },
    {
        "key": "noise-null",
        "prefix": "n",
        "label": "Null model: random Gaussian tensors",
        "params": "n/a (GPT-2 Small weights)",
        "corpus": "none — calibrated Gaussian noise (seed=42)",
        "path": "experiments/gpt2_small/output_random_baseline/dissolution_pathways_random.md",
        "kind": "null-model",
        "register_override": "Noise",
        "note": "Null control: noise through GPT-2 Small converges into non-semantic basins (F4).",
    },
]

# Documented reference values, for the sanity-check print only. Sourced from
# README.md and docs/FINDINGS.md#f1 / #f3 / #f4. Never used to alter parsed data.
DOCUMENTED = {
    "gpt2-small": {
        "source": "FINDINGS.md F1 / README.md",
        "gated_125": {"prolet": 43.2, "Divine": 27.2, "till": 15.2,
                      "Anarch": 13.6, "solidarity": 0.8},
        "iter100_125": {"prolet": 35.2, "Divine": 27.2, "Anarch": 20.8,
                        "till": 15.2, "solidarity": 1.6},
    },
    "gpt2-medium": {
        "source": "FINDINGS.md F3",
        "iter100_125": {"D": 100.0},
    },
    "pythia-160m": {
        "source": "FINDINGS.md F3",
        "iter100_125": {"questioned": 94.4},
    },
    "pythia-410m": {
        "source": "FINDINGS.md F3",
        "iter100_125": None,  # "no consolidation (40+ fragments)"
        "expectation": "no consolidation — 40+ distinct fragments, no dominant basin",
    },
    "noise-null": {
        "source": "FINDINGS.md F4",
        "iter100_125": {"―": 64.0},
        "expectation": "18 non-semantic basins, dominated by em-dash ― (64%)",
    },
}

# --------------------------------------------------------------------------
# Markdown parsing
# --------------------------------------------------------------------------

SECTION_RE = re.compile(r"^##\s+(.*?)\s*$")
SECTION_N_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<n>\d+)\s+prompts?\)\s*$")
# Fallback for the null-baseline heading, "## 125 Random Gaussian Tensors (seed=42)".
SECTION_LEADING_N_RE = re.compile(r"^(?P<n>\d+)\s+(?P<name>\S.*)$")
ITER_CELL_RE = re.compile(r"^\*{0,2}(\d+)\*{0,2}$")
BACKTICK_RE = re.compile(r"`([^`]*)`")


def split_row(line: str) -> list[str]:
    """Split a markdown table row into its cells (naive pipe split)."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_separator(line: str) -> bool:
    body = line.strip().strip("|")
    return bool(body) and all(set(c.strip()) <= set(":- ") for c in body.split("|"))


def parse_data_row(line: str, n_cols: int) -> tuple[int, list[str]] | None:
    """
    Parse `| **10** | `capit` | `Ag` | ... |` into (10, ["capit", "Ag", ...]).

    Cells may themselves contain a pipe character (e.g. the token `|`), so the
    token cells are recovered with a backtick regex rather than a pipe split,
    which is robust to that case. Falls back to a pipe split if the backtick
    extraction does not yield the expected column count.
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    body = stripped[1:]
    first, sep, rest = body.partition("|")
    if not sep:
        return None
    m = ITER_CELL_RE.match(first.strip())
    if not m:
        return None
    it = int(m.group(1))

    if rest.rstrip().endswith("|"):
        rest = rest.rstrip()[:-1]

    tokens = BACKTICK_RE.findall(rest)
    if len(tokens) != n_cols:
        # Fallback: pipe split, then strip backticks.
        alt = [c.strip().strip("`") for c in rest.split("|")]
        alt = [c for c in alt]
        if len(alt) == n_cols:
            tokens = alt
        else:
            return (it, tokens)  # caller warns on the mismatch
    return (it, tokens)


def parse_pathways(path: Path, register_override: str | None = None):
    """
    Parse a dissolution_pathways.md file.

    Returns (trajectories, registers, declared_counts, warnings) where
      trajectories: {prompt_id: {iter: token}}
      registers:    {prompt_id: register_name}   (insertion-ordered register list built by caller)
      declared_counts: {register_name: declared N prompts}
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    trajectories: dict[str, dict[int, str]] = {}
    prompt_register: dict[str, str] = {}
    register_order: list[str] = []
    declared: dict[str, int] = {}
    warnings: list[str] = []

    register = None
    header: list[str] | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        sm = SECTION_RE.match(line)
        if sm:
            raw = sm.group(1).strip()
            nm = SECTION_N_RE.match(raw)
            lm = SECTION_LEADING_N_RE.match(raw)
            if nm:
                name = nm.group("name").strip()
                n = int(nm.group("n"))
            elif lm:
                name = lm.group("name").strip()
                n = int(lm.group("n"))
            else:
                name = raw
                n = None
            if register_override:
                name = register_override
            register = name
            if register not in register_order:
                register_order.append(register)
            if n is not None:
                # A register may be spread over several sections; keep the max
                # declared count (they repeat the register total each time).
                declared[register] = max(declared.get(register, 0), n)
            header = None
            i += 1
            continue

        stripped = line.strip()
        if stripped.startswith("|") and not is_separator(stripped):
            cells = split_row(stripped)
            if cells and cells[0].strip().lower() == "iter":
                header = [c.strip() for c in cells[1:]]
                if register is None:
                    register = register_override or "Unlabelled"
                    if register not in register_order:
                        register_order.append(register)
                for p in header:
                    trajectories.setdefault(p, {})
                    prompt_register.setdefault(p, register)
                i += 1
                continue

            if header is not None:
                parsed = parse_data_row(stripped, len(header))
                if parsed is not None:
                    it, tokens = parsed
                    if len(tokens) != len(header):
                        warnings.append(
                            f"{path}: line {i+1}: expected {len(header)} cells, "
                            f"got {len(tokens)} — row skipped"
                        )
                    else:
                        for p, tok in zip(header, tokens):
                            trajectories[p][it] = tok if tok != "" else EMPTY_TOKEN
                    i += 1
                    continue
        i += 1

    return trajectories, prompt_register, register_order, declared, warnings


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def build_model_graph(spec, trajectories, prompt_register, register_order, declared):
    prefix = spec["prefix"]

    prompts = [p for p in trajectories if trajectories[p]]
    # Stable, human-meaningful ordering (A01_physics, A02_medical, ...).
    prompts.sort()

    iters = sorted({it for p in prompts for it in trajectories[p]})
    if not iters:
        return None
    terminal_iter = iters[-1]

    # Terminal basin per prompt (its token at the last iteration it has).
    terminal_of: dict[str, str] = {}
    for p in prompts:
        traj = trajectories[p]
        last = max(traj)
        terminal_of[p] = traj[last]

    def nid(it: int, token: str) -> str:
        return f"{prefix}|{it}|{token}"

    # ---- nodes -----------------------------------------------------------
    node_prompts: dict[str, list[str]] = defaultdict(list)
    node_meta: dict[str, tuple[int, str]] = {}
    for p in prompts:
        for it, tok in trajectories[p].items():
            key = nid(it, tok)
            node_prompts[key].append(p)
            node_meta[key] = (it, tok)

    nodes = []
    for key in sorted(node_prompts, key=lambda k: (node_meta[k][0], -len(node_prompts[k]), node_meta[k][1])):
        it, tok = node_meta[key]
        members = sorted(node_prompts[key])
        regs = sorted({prompt_register.get(p, "?") for p in members})
        basin_counts = Counter(terminal_of[p] for p in members)
        basin, basin_n = basin_counts.most_common(1)[0]
        nodes.append({
            "id": key,
            "token": tok,
            "iter": it,
            "count": len(members),
            "registers": regs,
            "terminal": it == terminal_iter,
            "basin": basin,
            "basin_purity": round(basin_n / len(members), 4),
            "prompts": members,
        })
    node_ids = {n["id"] for n in nodes}

    # ---- edges -----------------------------------------------------------
    edge_prompts: dict[tuple[str, str], list[str]] = defaultdict(list)
    edge_meta: dict[tuple[str, str], tuple[int, int]] = {}
    for p in prompts:
        traj = trajectories[p]
        seq = sorted(traj)
        for a, b in zip(seq, seq[1:]):
            src = nid(a, traj[a])
            dst = nid(b, traj[b])
            edge_prompts[(src, dst)].append(p)
            edge_meta[(src, dst)] = (a, b)

    edges = []
    for (src, dst) in sorted(edge_prompts, key=lambda e: (edge_meta[e][0], -len(edge_prompts[e]), e)):
        members = sorted(edge_prompts[(src, dst)])
        basin_counts = Counter(terminal_of[p] for p in members)
        basin, _ = basin_counts.most_common(1)[0]
        a, b = edge_meta[(src, dst)]
        edges.append({
            "from": src,
            "to": dst,
            "from_iter": a,
            "to_iter": b,
            "count": len(members),
            "prompts": members,
            "basin": basin,
            "registers": sorted({prompt_register.get(p, "?") for p in members}),
        })

    # ---- basins ----------------------------------------------------------
    basin_counter = Counter(terminal_of[p] for p in prompts)
    total = len(prompts)
    basins = {tok: n for tok, n in basin_counter.most_common()}
    basin_shares = {tok: round(100.0 * n / total, 1) for tok, n in basin_counter.most_common()}

    registers = [r for r in register_order]
    register_counts = Counter(prompt_register.get(p, "?") for p in prompts)

    model = {
        "label": spec["label"],
        "params": spec["params"],
        "corpus": spec["corpus"],
        "kind": spec["kind"],
        "note": spec.get("note", ""),
        "node_prefix": prefix,
        "source": spec["path"],
        "prompts": total,
        "prompts_documented": sum(declared.values()) if declared else None,
        "iterations": iters,
        "terminal_iter": terminal_iter,
        "basins": basins,
        "basin_shares": basin_shares,
        "basin_count": len(basins),
        "registers": registers,
        "register_counts": {r: register_counts.get(r, 0) for r in registers},
        "prompt_terminals": {p: terminal_of[p] for p in prompts},
        "nodes": nodes,
        "edges": edges,
    }
    return model, node_ids


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("DISSOLUTION GRAPH builder")
    print("=" * 72)

    models: dict[str, dict] = {}
    generated_from: list[str] = []
    all_warnings: list[str] = []

    for spec in MODEL_SPECS:
        path = REPO / spec["path"]
        if not path.exists():
            print(f"  WARNING: missing source, skipping model '{spec['key']}': {spec['path']}")
            continue

        trajectories, prompt_register, register_order, declared, warnings = parse_pathways(
            path, spec.get("register_override")
        )
        all_warnings.extend(warnings)
        for w in warnings:
            print(f"  WARNING: {w}")

        if not trajectories:
            print(f"  WARNING: no tables parsed, skipping model '{spec['key']}': {spec['path']}")
            continue

        built = build_model_graph(spec, trajectories, prompt_register, register_order, declared)
        if built is None:
            print(f"  WARNING: no iterations parsed, skipping model '{spec['key']}'")
            continue

        model, _ = built
        models[spec["key"]] = model
        generated_from.append(spec["path"])

    if not models:
        print("ERROR: no source files found at all — nothing to emit.")
        return 1

    # Global iteration union, plus the canonical snapshot schedule.
    canonical = [0, 2, 3, 5, 10, 20, 50, 100]
    all_iters = sorted({it for m in models.values() for it in m["iterations"]})

    payload = {
        "metadata": {
            "domain": "dissolution",
            "version": "1.0",
            "title": "Dissolution pathways: prompts funnelling into terminal attractor basins",
            "generated": date.today().isoformat(),
            "generated_from": generated_from,
            "iterations": canonical,
            "iterations_observed": all_iters,
            "empty_token_symbol": EMPTY_TOKEN,
            "provenance_note": (
                "The published dissolution_pathways.md tables print only the first ~10 "
                "prompts of each register, so this graph is built from the visible subset "
                "of the full 125-prompt sweep (see per-model 'prompts' vs "
                "'prompts_documented'). Basin shares here are therefore shares of the "
                "visible subset at the final snapshot iteration, not the convergence-gated "
                "125-prompt shares quoted in README.md (prolet 43.2%) nor the full-125 "
                "iteration-100 snapshot in FINDINGS.md F1 (prolet 35.2%)."
            ),
            "documented_reference": DOCUMENTED,
        },
        "models": models,
    }

    # ---- validation ------------------------------------------------------
    print()
    print("Validating edge endpoints...")
    errors = 0
    for key, m in models.items():
        ids = {n["id"] for n in m["nodes"]}
        dangling = [e for e in m["edges"] if e["from"] not in ids or e["to"] not in ids]
        if dangling:
            errors += len(dangling)
            print(f"  ERROR [{key}]: {len(dangling)} edge endpoint(s) do not resolve to an emitted node")
            for e in dangling[:5]:
                print(f"    {e['from']} -> {e['to']}")
        # per-node count must equal the number of prompts listed
        for n in m["nodes"]:
            if n["count"] != len(n["prompts"]):
                errors += 1
                print(f"  ERROR [{key}]: node {n['id']} count/prompts mismatch")
        # basin totals must equal prompt count
        if sum(m["basins"].values()) != m["prompts"]:
            errors += 1
            print(f"  ERROR [{key}]: basin totals {sum(m['basins'].values())} != prompts {m['prompts']}")
    if errors:
        print(f"  VALIDATION FAILED with {errors} error(s).")
        return 2
    print("  OK — every edge endpoint resolves to an emitted node.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- reporting -------------------------------------------------------
    print()
    print("-" * 72)
    print("PER-MODEL COUNTS")
    print("-" * 72)
    print(f"{'model':<20} {'prompts':>8} {'declared':>9} {'nodes':>7} {'edges':>7} {'basins':>7}  iterations")
    for key, m in models.items():
        decl = m["prompts_documented"] if m["prompts_documented"] is not None else "-"
        print(f"{key:<20} {m['prompts']:>8} {str(decl):>9} {len(m['nodes']):>7} "
              f"{len(m['edges']):>7} {m['basin_count']:>7}  {m['iterations']}")

    print()
    print("-" * 72)
    print("TERMINAL BASIN HISTOGRAMS (at each model's final snapshot iteration)")
    print("-" * 72)
    for key, m in models.items():
        print()
        print(f"### {key} — {m['label']}  "
              f"[{m['prompts']} prompts parsed of {m['prompts_documented']} in the sweep, "
              f"iter {m['terminal_iter']}]")
        for tok, n in m["basins"].items():
            share = m["basin_shares"][tok]
            bar = "#" * max(1, int(round(share / 2)))
            print(f"    {tok!r:<20} {n:>4}  {share:>5.1f}%  {bar}")

    print()
    print("-" * 72)
    print("SANITY CHECK vs README.md / docs/FINDINGS.md")
    print("-" * 72)
    for key, m in models.items():
        ref = DOCUMENTED.get(key)
        if not ref:
            print(f"\n{key}: no documented reference on file — parsed values reported as-is.")
            continue
        print(f"\n{key}  (reference: {ref['source']})")
        if ref.get("expectation"):
            print(f"    documented expectation : {ref['expectation']}")
        if ref.get("gated_125"):
            print(f"    documented @ lock-in   : "
                  + ", ".join(f"{k} {v}%" for k, v in ref["gated_125"].items()))
        if ref.get("iter100_125"):
            print(f"    documented @ iter100   : "
                  + ", ".join(f"{k} {v}%" for k, v in ref["iter100_125"].items()))
        print(f"    parsed here (n={m['prompts']:>3})  : "
              + ", ".join(f"{k} {v}%" for k, v in m["basin_shares"].items()))
        # Directional note on the known, expected discrepancy.
        if ref.get("gated_125") and ref.get("iter100_125"):
            for tok in ref["gated_125"]:
                if tok in m["basin_shares"]:
                    here = m["basin_shares"][tok]
                    gated = ref["gated_125"][tok]
                    snap = ref["iter100_125"].get(tok)
                    if snap is not None and abs(here - gated) > 5:
                        print(f"      note: {tok} here {here}% vs gated {gated}% vs "
                              f"documented iter-100 {snap}% — snapshot-vs-gated difference "
                              f"is expected (README quotes convergence-gated shares).")

    print()
    print(f"Wrote {OUT_PATH}")
    if all_warnings:
        print(f"({len(all_warnings)} parse warning(s) above.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
