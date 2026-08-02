#!/usr/bin/env python3
"""
build_isomorphism_graph.py — ISOMORPHISM GRAPH data generator.

Turns docs/ISOMORPHISM.md into a paired graph: Lucier's acoustic apparatus on one
side, the transformer's on the other, joined by `analogous-to` edges that state the
shared mathematical role — and, more usefully, `breaks-down-at` edges recording the
places the correspondence stops holding.

This graph is hand-authored rather than parsed. Everything in it is a transcription
of a sentence that already exists in the repository; the SOURCES table below records
which sentence, and every node carries a `doc_ref` so a reader can check it. Nothing
here is derived from data files, so the script takes no arguments, reads nothing, and
produces the same output every time it is run.

Reads (as prose, by the author, not at runtime):

    docs/ISOMORPHISM.md      the isomorphism table and the nonlinearity table
    docs/TECHNICAL.md        hook mechanism, normalisation, the formal iteration
    docs/MATH_PRIMER.md      power iteration and its limits
    docs/UNDERSTANDING.md    the bottleneck bypass, the regime correction
    docs/FINDINGS.md         F1, F4, F9, F15
    docs/JOURNEY_MAP.md      Key Discovery 11 and its 2026-07-23 correction
    README.md                the piece

Writes:

    docs/graph/_data/isomorphism.json

Schema is the shared "evidence graph" contract (metadata / claims / runs / sources /
relationships), so the same viewer renders this graph and entities.json. Two extra
fields ride along on the claims: `side` ("acoustic" | "transformer" | "shared"), for
two-column layout, and `role` ("pair" | "nonlinearity" | "breakdown"), for styling
the places the analogy fails.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent            # docs/graph
REPO = HERE.parent.parent                         # repo root
OUT_DIR = HERE / "_data"
OUT_PATH = OUT_DIR / "isomorphism.json"

GENERATED = "2026-07-25"


def load_sibling(name: str):
    """Import a module sitting next to this one, by path (not via sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "docs_graph_" + name, str(HERE / (name + ".py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Shared with check_record_drift.py and build_evidence_graph.py: one anchor
# rule for the whole of docs/graph/, so a document renamed under one generator
# cannot stay green under the other.
heading_anchors = load_sibling("check_record_drift").heading_anchors

# --------------------------------------------------------------------------
# Phases.  Ids, labels and starts are kept identical to entities.json so the
# viewer's timeline scrubber behaves the same on both graphs.  JOURNEY_MAP.md
# dates only Phase 4 and Phase 5; the earlier starts are placeholders.
# --------------------------------------------------------------------------

JM_TIMELINE = "docs/JOURNEY_MAP.md#1-timeline-the-intellectual-arc"

PHASES = [
    {"id": "phase-0", "label": "Phase 0: The Inspiration (Pre-experiment)",
     "start": "2026-03-01", "doc_ref": JM_TIMELINE},
    {"id": "phase-1", "label": "Phase 1: The Exploratory Experiment (EXP_009aFIX)",
     "start": "2026-03-01", "doc_ref": JM_TIMELINE},
    {"id": "phase-2", "label": "Phase 2: Validation Design (VALIDATION_PLAN)",
     "start": "2026-03-01", "doc_ref": JM_TIMELINE},
    {"id": "phase-3", "label": "Phase 3: Validation Execution",
     "start": "2026-03-01", "doc_ref": JM_TIMELINE},
    {"id": "phase-4", "label": "Phase 4: Supervisory Analysis (Today, 2026-03-20)",
     "start": "2026-03-20", "doc_ref": JM_TIMELINE},
    {"id": "phase-5", "label": "Phase 5: Cross-Model Validation & Series Close (2026-07-10)",
     "start": "2026-07-10", "doc_ref": JM_TIMELINE},
]

PHASE_NOTE = (
    "JOURNEY_MAP.md dates Phase 4 (2026-03-20) and Phase 5 (2026-07-10) only; "
    "phases 0-3 share a placeholder start of 2026-03-01, matching entities.json."
)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

SOURCES = [
    {
        "id": "doc-isomorphism",
        "title": "ISOMORPHISM.md — Lucier's room and iterative activation re-injection",
        "type": "doc",
        "path": "docs/ISOMORPHISM.md",
        "description": "The correspondence itself: the linear acoustic case, the nonlinear "
                       "transformer case, the nonlinearity table and the isomorphism table "
                       "this graph is a transcription of.",
    },
    {
        "id": "doc-technical",
        "title": "TECHNICAL.md — method specification",
        "type": "doc",
        "path": "docs/TECHNICAL.md",
        "description": "Hook read/write points, the L2 rescaling rule, the formal iteration "
                       "x_{n+1} = f(normalise(x_n)), and the gate_lag parameter.",
    },
    {
        "id": "doc-understanding",
        "title": "UNDERSTANDING.md — the feedback mechanism",
        "type": "doc",
        "path": "docs/UNDERSTANDING.md",
        "description": "Why the loop is closed before the unembedding rather than after it, "
                       "and the correction about what the attractors are not.",
    },
    {
        "id": "doc-math-primer",
        "title": "MATH_PRIMER.md — the maths from scratch",
        "type": "doc",
        "path": "docs/MATH_PRIMER.md",
        "description": "Iterated maps, fixed points, basins, and power iteration as the "
                       "linear ancestor of ATR, with its guarantees spelled out.",
    },
    {
        "id": "doc-findings",
        "title": "FINDINGS.md — the canonical record",
        "type": "doc",
        "path": "docs/FINDINGS.md",
        "description": "Run inventory, principal findings F1-F17, hypothesis dispositions "
                       "and caveats. F4, F9 and F15 anchor the breakdown points in this graph.",
    },
    {
        "id": "doc-journey-map",
        "title": "JOURNEY_MAP.md — timeline, discoveries, glossary",
        "type": "doc",
        "path": "docs/JOURNEY_MAP.md",
        "description": "Key Discoveries table (including number 11, the Brouwer claim and its "
                       "2026-07-23 correction) and the adjacent-mathematics table.",
    },
    {
        "id": "doc-readme",
        "title": "README.md — the piece",
        "type": "doc",
        "path": "README.md",
        "description": "The narrative account of the operation, including the description of "
                       "the L2 rescale as 'the room's friction'.",
    },
    {
        "id": "prior-lucier-1969",
        "title": "Lucier, A. (1969). I Am Sitting in a Room",
        "type": "prior-work",
        "path": "https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room",
        "description": "The source piece: speech recorded, replayed into a room, re-recorded, "
                       "repeated, until only the room's resonances remain.",
    },
    {
        "id": "prior-transformerlens",
        "title": "Nanda, N. & Bloom, J. (2022). TransformerLens",
        "type": "prior-work",
        "path": "https://github.com/TransformerLensOrg/TransformerLens",
        "description": "The hook library that supplies the re-injection mechanism: the "
                       "apparatus standing in for Lucier's tape recorder.",
    },
]

# --------------------------------------------------------------------------
# Runs and models
# --------------------------------------------------------------------------

RUNS = [
    {
        "id": "model-gpt2-small",
        "label": "GPT-2 Small (124M)",
        "type": "model",
        "description": "12 layers, 12 heads, d_model 768, 50,257 BPE tokens, trained on "
                       "WebText. The room whose acoustics this graph maps.",
    },
    {
        "id": "run-03-random-baseline",
        "label": "Run 3: random-noise null model",
        "type": "null-model",
        "description": "125 Gaussian tensors (seed 42) iterated through GPT-2 Small in place "
                       "of prompts; first reported as 18 non-semantic attractors with no "
                       "overlap with the five. Superseded 2026-07-31: this arm ran "
                       "mis-calibrated and was counted before convergence (FINDINGS caveat "
                       "18); the matched-nu re-run (run 17) finds noise landing in the "
                       "language arm's own basins.",
        "script": "experiments/gpt2_small/03_random_baseline.ipynb",
        "output_dir": "experiments/gpt2_small/output_random_baseline/",
        "n": "125 Gaussian tensors, seed 42",
    },
    {
        "id": "run-17-matched-nu-noise",
        "label": "Run 17: matched-nu noise re-run (2026-07-31)",
        "type": "null-model",
        "description": "The corrected null control: 125 Gaussian tensors, each pair-matched "
                       "to one real prompt's exact sequence length and iteration-0 Frobenius "
                       "norm, run under the engine's convergence gate. 90/125 lock in to 7 "
                       "basins, four of them the language arm's own; the 35 period-2 trials "
                       "decode to till, i, player and Divine itself, so at each trial's "
                       "smallest passing lag all five language basins reappear and 97/125 "
                       "trials land in them (FINDINGS F4, corrected).",
        "script": "experiments/noise_rerun/01_matched_nu_noise_baseline.py",
        "output_dir": "experiments/noise_rerun/output/",
        "n": "125 Gaussian tensors, pair-matched, seed 42",
    },
    {
        "id": "run-18-nu-sweep",
        "label": "Run 18: the nu-sweep (2026-08-02)",
        "type": "run",
        "description": "The injection-scale scan (registered issues #113/#116): the pin nu "
                       "swept over ten multipliers of each prompt's natural entry norm plus "
                       "the exact historical pin (Stage A, 250 trials), then the crossing "
                       "levels at all 125 prompts (Stage B, 650 trials total). The five-basin "
                       "landscape is a band: 0/125 in the five at 32x, 125/125 at 64x, "
                       "123/125 at 128x, 84/125 at 256x, so the lower edge is sharp inside "
                       "(32x, 64x) and the upper edge lies beyond the swept range. Below the "
                       "band: strata (arbit at 2-4x, the horizontal bar at 8-16x, "
                       "fragmentation at 32x) whose lowest members point into the "
                       "anomalous-token cluster (FINDINGS caveat 19, resolved wording).",
        "script": "experiments/nu_sweep/01_stage_a.py",
        "output_dir": "experiments/nu_sweep/output/",
        "n": "1,025 trials over Stages A and B, gated, F15-classified",
    },
    {
        "id": "run-08-divine-motion",
        "label": "Run 8: Divine motion audit (lag-10 + lag-1 probe)",
        "type": "run",
        "description": "Three trajectories to 1000 iterations with a dense lag-1 probe from the "
                       "saved iteration-1000 state. Found the alternation the log-spaced "
                       "schedules had aliased away, and verified the cycle exactly: "
                       "cos(A, f(f(A))) = 1.000000.",
        "script": "experiments/gpt2_small/05_divine_motion.py",
        "output_dir": "experiments/gpt2_small/output_divine_motion/",
        "n": "3 trajectories x 1000 iters, + 20 lag-1 iters",
        "date": "2026-07-19",
    },
]

# --------------------------------------------------------------------------
# Claims: the paired concepts, the nonlinearities, and the breakdown points.
#
# Tuple order:
#   (id, label, type, status, side, role, phase, doc_ref, asserted, retired,
#    description)
# --------------------------------------------------------------------------

ISO = "docs/ISOMORPHISM.md"
FIND = "docs/FINDINGS.md"

CLAIMS_RAW = [
    # ---------------------------------------------------------------- acoustic
    (
        "ac-room", "Room", "concept", "supported", "acoustic", "pair", "phase-0",
        f"{ISO}#the-acoustic-case-linear-power-iteration", None, None,
        "The physical space in Lucier's 1969 piece, modelled as a linear operator H acting "
        "on a discrete signal vector. Acoustic propagation obeys superposition — two sources "
        "sum to the sum of their reverberant responses — which is what makes H a genuine "
        "linear map.",
    ),
    (
        "ac-audio-signal", "Audio signal", "concept", "supported", "acoustic", "pair", "phase-0",
        f"{ISO}#the-isomorphism", None, None,
        "The recorded speech, s_n: the thing that is played, captured and played again. It is "
        "the state the iteration carries forward, and the only thing that changes from pass "
        "to pass.",
    ),
    (
        "ac-tape-recorder", "Tape recorder", "concept", "supported", "acoustic", "pair", "phase-0",
        f"{ISO}#the-acoustic-case-linear-power-iteration", None, None,
        "The re-injection mechanism: it captures the reverberant output and feeds it back as "
        "the next input, closing the loop. Without it there is no iteration, only a room.",
    ),
    (
        "ac-room-friction", "Room friction (energy loss per pass)", "concept", "qualified",
        "acoustic", "pair", "phase-0",
        "README.md#how-atr-works", None, None,
        "The README's name for what the L2 rescale stands in for: the loss that keeps a real "
        "room's replayed signal from growing without bound. Held loosely — real damping is "
        "frequency-dependent, whereas the ATR rescale is a single global factor applied to "
        "the whole tensor.",
    ),
    (
        "ac-resonant-frequency", "Room resonant frequency", "concept", "qualified",
        "acoustic", "pair", "phase-0",
        f"{ISO}#the-acoustic-case-linear-power-iteration", None, None,
        "The dominant eigenmode v1 of H, the direction the normalised iterates approach. "
        "ISOMORPHISM.md is explicit that this convergence is conditional, not automatic: for "
        "a real room the single-dominant-mode account is an analogy and an empirical pattern, "
        "not a theorem.",
    ),
    (
        "ac-spectral-decay", "Spectral decay of non-resonant frequencies", "concept",
        "supported", "acoustic", "pair", "phase-0",
        f"{ISO}#the-isomorphism", None, None,
        "The transient phase: every component other than the dominant one shrinks relative to "
        "it, at rates set by the eigenvalue ratios. This is what a listener hears as the words "
        "draining out of the recording.",
    ),
    (
        "ac-pure-drone", "Pure drone", "concept", "supported", "acoustic", "pair", "phase-0",
        f"{ISO}#the-acoustic-case-linear-power-iteration", None, None,
        "The end state of the piece: a steady tone at the room's resonant frequency. Speech "
        "has dissolved into architecture, and further passes change nothing audible.",
    ),
    (
        "ac-linear-operator-h", "Linear operator H", "concept", "supported",
        "acoustic", "pair", "phase-0",
        f"{ISO}#the-acoustic-case-linear-power-iteration", None, None,
        "H: R^n -> R^n, the room's transfer function. Fixed once the room is built, identical "
        "on every pass, and additive: H(a + b) = H(a) + H(b). Every guarantee on the acoustic "
        "side descends from those three properties.",
    ),
    (
        "ac-dominant-eigenmode", "Single dominant eigenmode", "concept", "qualified",
        "acoustic", "pair", "phase-0",
        f"{ISO}#key-insight", None, None,
        "The consequence of linearity: for a diagonalisable H with a unique "
        "largest-magnitude eigenvalue, and a start with a nonzero component along it, the "
        "normalised iterates approach one direction. One room, one answer — under those "
        "conditions, and only those.",
    ),

    # ------------------------------------------------------------- transformer
    (
        "tr-weights", "Transformer weight matrices", "concept", "supported",
        "transformer", "pair", "phase-1",
        f"{ISO}#the-isomorphism", None, None,
        "W_Q, W_K, W_V, W_O, W_in, W_out across all 12 layers of GPT-2 Small: the operator "
        "being iterated. The whole stack, layer 0 through layer 11, acts as the room.",
    ),
    (
        "tr-residual-stream", "Residual stream tensor [T, 768]", "concept", "supported",
        "transformer", "pair", "phase-1",
        "docs/TECHNICAL.md#formal-description", None, None,
        "The state vector: the full [T, 768] residual stream across all token positions, read "
        "at blocks.11.hook_resid_post. Looping this rather than the decoded token is the "
        "design decision the whole project rests on.",
    ),
    (
        "tr-hook", "TransformerLens hook (extract, normalise, re-inject)", "concept",
        "supported", "transformer", "pair", "phase-1",
        "docs/TECHNICAL.md#hook-mechanism", None, None,
        "The feedback mechanism: a forward hook at blocks.0.hook_resid_pre overwrites the "
        "token embeddings with the normalised tensor extracted from the final layer. The "
        "prompt string is still passed on every iteration, but only as scaffolding for the "
        "computation graph.",
    ),
    (
        "tr-l2-normalisation", "L2 rescaling to initial energy", "concept", "supported",
        "transformer", "pair", "phase-1",
        "docs/TECHNICAL.md#normalisation", None, None,
        "normalise(x) = x * (||x0|| / ||x||), applied to the whole tensor every pass. Without "
        "it the norm reaches about 1.5M by iteration 500 and saturates the nonlinearities; "
        "with it the iteration is energy-conservative and confined to a fixed-radius shell.",
    ),
    (
        "tr-attractor-state", "Attractor state (prolet, Divine, ...)", "concept", "qualified",
        "transformer", "pair", "phase-1",
        f"{ISO}#the-isomorphism", None, None,
        "What the iteration settles into: a fixed point for the prolet-class prompts, and an "
        "exact period-2 limit cycle for Divine. 'Attractor' here has to be read in the "
        "dynamical-systems sense, not as a synonym for fixed point.",
    ),
    (
        "tr-dissolution", "Iterative dissolution of semantic content", "concept", "supported",
        "transformer", "pair", "phase-1",
        f"{ISO}#the-isomorphism", None, None,
        "The transient: connection goes, then meaning, then grammar, along a shared pathway of "
        "decoded waypoints (ash -> Canad -> Ag -> FT -> capit -> injustice -> Rousse -> "
        "prolet) before the state reaches a fixed point or cycle.",
    ),
    (
        "tr-terminal-token", "Terminal token sequence (uniform across positions)", "concept",
        "supported", "transformer", "pair", "phase-1",
        "docs/TECHNICAL.md#observed-dynamics", None, None,
        "The readout at the attractor: by iteration ~10 all T positions have collapsed to "
        "near-identical vectors, so the decode is one token repeated across the sequence. The "
        "drone, in text.",
    ),
    (
        "tr-nonlinear-map-f", "Nonlinear map f", "concept", "supported",
        "transformer", "pair", "phase-1",
        f"{ISO}#the-transformer-case-nonlinear-power-iteration", None, None,
        "f: R^(T x 768) -> R^(T x 768), the full forward pass from layer 0 to layer 11. Same "
        "structural position as H, but it contains LayerNorm, softmax attention, GeLU and "
        "dynamically recomputed QKV, none of which are matrix multiplies.",
    ),
    (
        "tr-multiple-basins", "Multiple basins with distinct attractors", "concept",
        "supported", "transformer", "pair", "phase-5",
        f"{FIND}#run-5", "2026-07-10", None,
        "The consequence of nonlinearity: not one dominant mode but a landscape. GPT-2 Small "
        "resolves 125 language prompts into five basins at lock-in — prolet 43.2%, Divine "
        "27.2%, till 15.2%, Anarch 13.6%, solidarity 0.8%.",
    ),

    # ------------------------------------------- the nonlinearities themselves
    (
        "nl-layernorm", "LayerNorm", "concept", "supported",
        "transformer", "nonlinearity", "phase-1",
        f"{ISO}#the-transformer-case-nonlinear-power-iteration", None, None,
        "Rescaling and recentring, applied before every sublayer. The first place in the "
        "stack where the additivity that makes the acoustic account work is lost.",
    ),
    (
        "nl-softmax-attention", "Attention (softmax)", "concept", "supported",
        "transformer", "nonlinearity", "phase-1",
        f"{ISO}#the-transformer-case-nonlinear-power-iteration", None, None,
        "Data-dependent gating over value vectors: the softmax weights are a function of the "
        "state being processed, so the mixing pattern is different on every pass.",
    ),
    (
        "nl-qkv-recompute", "QKV computation (recomputed each iteration)", "concept",
        "supported", "transformer", "nonlinearity", "phase-1",
        f"{ISO}#the-transformer-case-nonlinear-power-iteration", None, None,
        "Queries, keys and values are recomputed at each iteration from the current state. "
        "The operator applied at step n+1 is not the operator applied at step n.",
    ),
    (
        "nl-gelu-mlp", "MLP (GeLU)", "concept", "supported",
        "transformer", "nonlinearity", "phase-1",
        f"{ISO}#the-transformer-case-nonlinear-power-iteration", None, None,
        "Element-wise nonlinear activation inside every block's feedforward layer: a "
        "pointwise bend applied 12 times per pass.",
    ),
    (
        "nl-residual-connections", "Residual connections", "concept", "supported",
        "transformer", "nonlinearity", "phase-1",
        f"{ISO}#the-transformer-case-nonlinear-power-iteration", None, None,
        "The additive skip path. Linear in itself — the one component that behaves like air — "
        "but it composes with the nonlinear sublayers, so it carries no guarantees of its own.",
    ),

    # -------------------------------------------------------- breakdown points
    (
        "concept-no-spectral-guarantee", "No spectral theorem guarantee", "concept",
        "supported", "shared", "breakdown", "phase-1",
        f"{ISO}#the-transformer-case-nonlinear-power-iteration", None, None,
        "The load-bearing break. Power iteration's convergence proof is a theorem about "
        "matrices; f is not a matrix, so the system is not guaranteed to converge to a single "
        "dominant eigenvector, or to converge at all. What the acoustic side gets by theorem, "
        "the transformer side has to earn by measurement.",
    ),
    (
        "f9-divine-period-2", "F9: Divine is an exact period-2 limit cycle", "finding",
        "supported", "transformer", "breakdown", "phase-5",
        f"{FIND}#f9-the-divine-anomaly-resolved-an-exact-period-2-limit-cycle-hidden-by-aliasing",
        "2026-07-19", None,
        "The Divine state alternates between two phases separated by L2 1249 (cosine 0.685), "
        "returning to itself exactly every second pass: cos(A, f(f(A))) = 1.000000. Every "
        "earlier snapshot schedule sampled only even iterations and so recorded one phase, "
        "which is why it looked frozen for four months.",
    ),
    (
        "concept-regime-dependence", "The attractors belong to a regime, not to the weights",
        "finding", "corrected", "transformer", "breakdown", "phase-5",
        f"{FIND}#f4-the-five-basins-belong-to-the-language-driven-regime-not-the-weights-in-general-null-model",
        "2026-07-10", None,
        "As first recorded: pure noise seemed to converge into eighteen non-semantic "
        "attractors with no overlap with the five, so the basins looked like properties of "
        "the model as driven by language. Inverted 2026-07-31 (run 17): that noise arm was "
        "mis-calibrated and counted before convergence; at matched injection scale and "
        "gated convergence, noise falls into the language arm's own basins, all five "
        "reappearing at the trials' smallest passing lag. At the "
        "tested scale the basins belong to the weights; other scales await the nu-sweep.",
    ),
    (
        "concept-readout-bypass", "The loop is closed before the readout, not after it",
        "concept", "supported", "shared", "breakdown", "phase-0",
        "docs/UNDERSTANDING.md#the-lucier-loop-bypasses-this-bottleneck", None, None,
        "Lucier's loop passes through the air, carrying everything the room emitted. ATR's "
        "loop is cut upstream of the unembedding, deliberately bypassing the argmax that "
        "would collapse 50,257 candidates to one token. The two apparatuses re-inject "
        "different kinds of object.",
    ),
    (
        "claim-brouwer-basins", "Every normalised transformer must have basins (Brouwer)",
        "hypothesis", "corrected", "shared", "breakdown", "phase-4",
        "docs/JOURNEY_MAP.md#2-key-discoveries-chronological", "2026-03-20", "2026-07-23",
        "Session 02's attempt to promote the room analogy into a theorem: a continuous map on "
        "a compact set has a fixed point, LayerNorm bounds the state, therefore attractors "
        "are guaranteed and only their number and depth are empirical. Corrected 2026-07-23; "
        "kept in the graph because the correction is the interesting part.",
    ),
    (
        "concept-l2-shell-not-convex", "The L2 shell is a sphere, and a sphere is not convex",
        "concept", "supported", "transformer", "breakdown", "phase-5",
        "docs/JOURNEY_MAP.md#2-key-discoveries-chronological", "2026-07-23", None,
        "Brouwer's theorem requires a compact convex domain. L2 rescaling confines the state "
        "to a fixed-radius sphere in R^(T x 768), which is not convex, and the normalised map "
        "is undefined at zero. Attractor existence in these models is an observation, not a "
        "guarantee.",
    ),
]

CLAIM_FIELDS = (
    "id", "label", "type", "status", "side", "role", "phase", "doc_ref",
    "asserted", "retired", "description",
)

# doc_ref path prefix -> source id, for the generated documented-in edges.
DOC_SOURCE_MAP = {
    "docs/ISOMORPHISM.md": "doc-isomorphism",
    "docs/TECHNICAL.md": "doc-technical",
    "docs/UNDERSTANDING.md": "doc-understanding",
    "docs/MATH_PRIMER.md": "doc-math-primer",
    "docs/FINDINGS.md": "doc-findings",
    "docs/JOURNEY_MAP.md": "doc-journey-map",
    "README.md": "doc-readme",
}

# --------------------------------------------------------------------------
# Relationships.  Tuple order: (from, to, type, description, weight)
# --------------------------------------------------------------------------

RELS_RAW = [
    # ---- the isomorphism table, one edge per row.  The description is the
    # ---- third column: the shared mathematical role.
    ("ac-room", "tr-weights", "analogous-to",
     "Shared role: the operator being iterated. A room is fixed once built and the weights "
     "are fixed once trained; in both cases the iteration applies the same object over and "
     "over to whatever is handed to it.", 8),
    ("ac-audio-signal", "tr-residual-stream", "analogous-to",
     "Shared role: the state vector. Both are the quantity that changes from pass to pass "
     "and gets carried forward — a signal in R^n there, a [T, 768] tensor here.", 8),
    ("ac-tape-recorder", "tr-hook", "analogous-to",
     "Shared role: the feedback mechanism. Both capture the output state and present it as "
     "the next input; without either device the system is not iterated at all.", 8),
    ("ac-room-friction", "tr-l2-normalisation", "analogous-to",
     "Shared role: the energy budget that keeps the iteration bounded. The README names the "
     "L2 rescale 'the room's friction' — in both systems the state would otherwise grow "
     "without limit under repeated re-injection.", 5),
    ("ac-resonant-frequency", "tr-attractor-state", "analogous-to",
     "Shared role: attractor of the iterated map. The room mode is the direction the acoustic "
     "iterates approach; prolet, Divine and the rest are what the transformer iterates settle "
     "into (a fixed point, or a period-2 cycle for Divine).", 8),
    ("ac-spectral-decay", "tr-dissolution", "analogous-to",
     "Shared role: transient dynamics before the attractor. Non-dominant acoustic components "
     "shrink at rates set by eigenvalue ratios; semantic content drains through the "
     "dissolution pathway. Both are the part of the process you can hear happening.", 8),
    ("ac-pure-drone", "tr-terminal-token", "analogous-to",
     "Shared role: the stable attractor readout. A steady tone at the room's resonance; a "
     "single token repeated across every position once the state stops moving.", 8),
    ("ac-linear-operator-h", "tr-nonlinear-map-f", "analogous-to",
     "Shared role: the map being iterated — and the row where the correspondence changes "
     "character. Same structural position, different class of operator: H is linear, f is "
     "not, and every break recorded in this graph descends from that difference.", 9),
    ("ac-dominant-eigenmode", "tr-multiple-basins", "analogous-to",
     "Shared role: the consequence of (non)linearity. Linearity buys one dominant mode under "
     "stated conditions; nonlinearity permits several coexisting attractors with distinct "
     "basins, which is the landscape the 125-prompt sweep maps.", 9),

    # ---- where the analogy breaks: the nonlinearity table
    ("ac-linear-operator-h", "nl-layernorm", "breaks-down-at",
     "H is additive by the acoustic superposition principle. LayerNorm recentres and rescales "
     "its input, so f(a + b) is not f(a) + f(b), and the transfer-function reading of the "
     "stack fails before the first attention head runs.", 7),
    ("ac-room", "nl-softmax-attention", "breaks-down-at",
     "A room does not listen. Softmax attention gates the value vectors on the data itself, "
     "so the mixing pattern the 'room' applies is a function of what is being played into "
     "it — the operator is state-dependent in a way no acoustic space is.", 7),
    ("ac-room", "nl-qkv-recompute", "breaks-down-at",
     "The room's geometry is fixed for the duration of the piece. Q, K and V are recomputed "
     "at every iteration from the current state, so the transformer is not one room played "
     "many times but a different room on each pass.", 7),
    ("ac-linear-operator-h", "nl-gelu-mlp", "breaks-down-at",
     "GeLU applies a pointwise bend to every coordinate, twelve times per pass. Air is linear "
     "to a very good approximation at these amplitudes, so this component has no acoustic "
     "counterpart at all.", 7),
    ("ac-linear-operator-h", "nl-residual-connections", "breaks-down-at",
     "The one near-miss: the skip path is additive, so it behaves like superposition. But it "
     "composes with the nonlinear sublayers rather than replacing them, so the linear-looking "
     "part of the architecture does not restore the linear model.", 5),

    # ---- why the nonlinearities cost the guarantee
    ("nl-layernorm", "concept-no-spectral-guarantee", "supports",
     "Rescaling and recentring break additivity, and the spectral theorem is a statement "
     "about additive maps.", 5),
    ("nl-softmax-attention", "concept-no-spectral-guarantee", "supports",
     "Data-dependent gating means there is no single matrix whose eigenvectors could be the "
     "thing the iteration converges to.", 5),
    ("nl-qkv-recompute", "concept-no-spectral-guarantee", "supports",
     "Recomputing QKV each pass makes the effective operator state-dependent, so 'the "
     "dominant eigenvalue of H' has no fixed referent.", 5),
    ("nl-gelu-mlp", "concept-no-spectral-guarantee", "supports",
     "An element-wise nonlinear activation puts the composite map outside the class the "
     "power-iteration proof covers.", 5),
    ("nl-residual-connections", "concept-no-spectral-guarantee", "supports",
     "Additive in isolation, but composed with nonlinear sublayers, so it cannot be used to "
     "recover a linear reading of the stack.", 3),

    # ---- the consequences of losing the guarantee
    ("ac-dominant-eigenmode", "concept-no-spectral-guarantee", "breaks-down-at",
     "Power iteration singles out one direction only for a diagonalisable operator with a "
     "unique largest-magnitude eigenvalue and a start that overlaps it. None of those "
     "hypotheses can be checked for f, so the single-mode expectation does not transfer.", 8),
    ("concept-no-spectral-guarantee", "tr-multiple-basins", "supports",
     "With no theorem forcing one dominant direction, several attractors with distinct basins "
     "become possible — and five of them are what GPT-2 Small actually shows.", 6),
    ("tr-l2-normalisation", "tr-multiple-basins", "supports",
     "The rescaling is what makes the landscape visible at all: unnormalised, the norm reaches "
     "~1.5M and saturates the nonlinearities, and no basin structure can be read off the "
     "result.", 5),

    # ---- the period-2 cycle: a room mode is a fixed point, this is not
    ("ac-resonant-frequency", "f9-divine-period-2", "breaks-down-at",
     "A room mode is a fixed point: play the drone back into the room and the drone comes out "
     "again. The Divine state returns to itself only every second pass, alternating between "
     "two phases 1249 apart in L2. Nothing in the resonance picture predicts an object that "
     "needs two passes to come home.", 9),
    ("f9-divine-period-2", "tr-attractor-state", "qualifies",
     "The attractor set is not only fixed points. Divine is a genuine attractor of f that no "
     "lag-1 convergence gate can ever pass, which is why 34 prompts were reported for months "
     "as never converging.", 7),
    ("f9-divine-period-2", "ac-pure-drone", "qualifies",
     "The drone survives at the readout — the argmax is Divine in both phases — but the "
     "distribution beneath it shifts by about 0.25 nats KL per half-cycle. A steady readout "
     "no longer licenses the inference that the state is steady.", 6),
    ("f9-divine-period-2", "run-08-divine-motion", "evidenced-by",
     "The lag-1 probe from the saved iteration-1000 state: L2 distance from the base state "
     "alternates 1249.43, 0.000, 1249.43, 0.001, and the cycle-anatomy run then verified "
     "cos(A, f(f(A))) = 1.000000.", 6),

    # ---- the regime correction, itself corrected: at matched injection
    # scale the room analogy PASSES this test (run 17, 2026-07-31).
    ("ac-resonant-frequency", "concept-regime-dependence", "breaks-down-at",
     "Excite a room with anything broadband and the same modes rise: they are a property of "
     "the space. The five basins first seemed to fail that test (noise appeared to raise "
     "eighteen different, non-semantic attractors), but the corrected matched-nu control "
     "(run 17, 2026-07-31) reverses it: noise raises the language arm's own five basins, "
     "so at the tested injection scale the analogy passes and the basins do belong to the "
     "room. The edge is kept as the record of the correction; the nu-sweep (run 18, "
     "2026-08-02) then answered the scale question: the analogy passes only inside a "
     "measured band of injection scales, 64x to at least 256x the natural entry scale, "
     "and below the band the room resolves different words entirely (FINDINGS caveat "
     "19).", 9),
    ("concept-regime-dependence", "run-03-random-baseline", "evidenced-by",
     "The original evidence, now superseded and kept as the historical record: 125 "
     "mis-calibrated Gaussian tensors, counted "
     "pre-convergence, appeared to give 18 non-semantic basins. Run 17's matched-nu, gated "
     "re-run inverts the reading (FINDINGS F4).", 6),
    ("concept-regime-dependence", "run-17-matched-nu-noise", "evidenced-by",
     "The current evidence for the corrected reading: pair-matched, convergence-gated noise "
     "lands in the language arm's own basins (all five at each trial's smallest passing "
     "lag, 97/125 trials), so at the tested injection scale the basins belong to the "
     "weights (FINDINGS F4).", 9),
    ("concept-regime-dependence", "run-18-nu-sweep", "evidenced-by",
     "The scale dimension of regime dependence, measured: the five-basin landscape holds "
     "from 64x to at least 256x the natural entry scale (lower edge sharp at full sweep "
     "width, 0/125 to 125/125 in one doubling), and below the band the iterated map "
     "resolves stratified non-basin attractors, the lowest pointing into the "
     "anomalous-token cluster (FINDINGS caveat 19, resolved).", 9),

    # ---- the apparatus difference
    ("ac-tape-recorder", "concept-readout-bypass", "breaks-down-at",
     "The tape recorder re-injects the fully rendered signal, everything the room emitted "
     "through the air. The hook re-injects the residual stream before the unembedding, "
     "deliberately skipping the argmax bottleneck — so the ATR loop has no acoustic "
     "counterpart at the point where it is actually closed.", 7),

    # ---- the retired theorem
    ("claim-brouwer-basins", "ac-resonant-frequency", "builds-on",
     "The Session 02 argument tried to turn the room analogy into a theorem: if a room always "
     "has modes, a normalised transformer must always have attractors.", 4),
    ("concept-l2-shell-not-convex", "claim-brouwer-basins", "corrects",
     "Brouwer requires a compact convex domain. L2 rescaling puts the state on a sphere, "
     "which is not convex, and the normalised map is undefined at zero — so the fixed-point "
     "theorem cannot be invoked here. Corrected 2026-07-23.", 9),
    ("concept-l2-shell-not-convex", "tr-l2-normalisation", "relates-to",
     "The surface the argument founders on is exactly the constraint the L2 rescaling "
     "imposes: the design decision that makes the dynamics legible is the same one that "
     "denies them a fixed-point guarantee.", 5),

    # ---- provenance
    ("ac-tape-recorder", "prior-lucier-1969", "cites",
     "The re-injection mechanism is Lucier's: record, play back into the room, re-record, "
     "repeat.", 2),
    ("tr-hook", "prior-transformerlens", "cites",
     "The read and write points (blocks.11.hook_resid_post, blocks.0.hook_resid_pre) are "
     "TransformerLens hooks; the library is what makes the loop a few lines of code.", 2),
    ("run-08-divine-motion", "model-gpt2-small", "run-on",
     "The motion audit and its lag-1 probe were run on GPT-2 Small, the model the whole "
     "isomorphism is stated for.", 2),
    ("run-03-random-baseline", "model-gpt2-small", "run-on",
     "The null model uses the same weights as the language sweep, changing only what is fed "
     "into them.", 2),
]


def build_documented_in_edges(claims):
    """One documented-in edge per claim, pointing at the doc its doc_ref names."""
    edges = []
    for c in claims:
        ref = c.get("doc_ref")
        if not ref:
            continue
        path = ref.split("#", 1)[0]
        src = DOC_SOURCE_MAP.get(path)
        if src is None:
            raise SystemExit(f"ERROR: no source registered for doc_ref path {path!r}")
        edges.append({
            "from": c["id"],
            "to": src,
            "type": "documented-in",
            "description": f"Stated in {ref} — the sentence this node is a transcription of.",
            "weight": 2,
        })
    return edges


def build():
    claims = [dict(zip(CLAIM_FIELDS, row)) for row in CLAIMS_RAW]

    relationships = [
        {"from": f, "to": t, "type": ty, "description": d, "weight": w}
        for (f, t, ty, d, w) in RELS_RAW
    ]
    relationships += build_documented_in_edges(claims)

    # Carry the run ids named by evidenced-by edges onto the claims themselves.
    for c in claims:
        ev = [r["to"] for r in relationships
              if r["from"] == c["id"] and r["type"] == "evidenced-by"]
        if ev:
            c["evidence"] = ev

    graph = {
        "metadata": {
            "domain": "isomorphism",
            "version": "1.0",
            "created": GENERATED,
            "last_updated": GENERATED,
            "generated": GENERATED,
            "title": "Lucier's room and the transformer: the correspondence, and where it breaks",
            "subtitle": "Paired acoustic and transformer concepts joined by shared mathematical "
                        "role, with the nonlinearities and findings that stop the analogy being "
                        "an identity.",
            "phases": PHASES,
            "phase_note": PHASE_NOTE,
            "layout_hint": {
                "columns": ["acoustic", "shared", "transformer"],
                "field": "side",
                "note": "Nodes carry side (acoustic | transformer | shared) and role "
                        "(pair | nonlinearity | breakdown). A two-column layout with the "
                        "shared column between reads left-to-right as ISOMORPHISM.md does.",
            },
            "generated_from": [
                "docs/ISOMORPHISM.md",
                "docs/TECHNICAL.md",
                "docs/UNDERSTANDING.md",
                "docs/MATH_PRIMER.md",
                "docs/FINDINGS.md",
                "docs/JOURNEY_MAP.md",
                "README.md",
            ],
            "provenance_note": "Hand-authored from the documents listed above, not parsed from "
                               "data. Every node's doc_ref names the passage it transcribes; "
                               "no number or claim in this file originates here.",
        },
        "claims": claims,
        "runs": RUNS,
        "sources": SOURCES,
        "relationships": relationships,
    }
    return graph


def validate(graph):
    """Every edge endpoint must resolve; ids must be unique; vocabularies must hold."""
    ids = {}
    for bucket in ("claims", "runs", "sources"):
        for node in graph[bucket]:
            nid = node["id"]
            if nid in ids:
                raise SystemExit(f"ERROR: duplicate node id {nid!r}")
            ids[nid] = bucket

    node_types = {
        "claims": {"hypothesis", "finding", "concept"},
        "runs": {"run", "model", "null-model"},
        "sources": {"doc", "artefact", "prior-work"},
    }
    # Kept identical to build_evidence_graph.ALLOWED_STATUS. The two graphs are
    # rendered by the same viewer.html, coloured from the same
    # visual_config.json and documented by the same README table, so a value
    # legal in one and illegal in the other is a bug waiting to happen -- which
    # is what "not-supported" was, having been added on the evidence side only.
    # "not-supported" = the evidence failed to back the claim without
    # contradicting it, which is a different verdict from "refuted".
    statuses = {"supported", "refuted", "not-supported", "qualified", "retired",
                "corrected", "open", "untested"}
    edge_types = {
        "supports", "refutes", "qualifies", "corrects", "retires", "supersedes", "tests",
        "produced-by", "run-on", "evidenced-by", "documented-in",
        "analogous-to", "breaks-down-at", "builds-on", "cites", "relates-to",
    }
    phase_ids = {p["id"] for p in graph["metadata"]["phases"]}

    problems = []
    for bucket, allowed in node_types.items():
        for node in graph[bucket]:
            if node["type"] not in allowed:
                problems.append(f"{node['id']}: bad type {node['type']!r}")
    for c in graph["claims"]:
        if c["status"] not in statuses:
            problems.append(f"{c['id']}: bad status {c['status']!r}")
        if c["phase"] not in phase_ids:
            problems.append(f"{c['id']}: unknown phase {c['phase']!r}")
        if c["side"] not in {"acoustic", "transformer", "shared"}:
            problems.append(f"{c['id']}: bad side {c['side']!r}")
        if not c["description"].strip():
            problems.append(f"{c['id']}: empty description")

    unresolved = 0
    for r in graph["relationships"]:
        for end in ("from", "to"):
            if r[end] not in ids:
                problems.append(f"edge {r['from']} -> {r['to']}: unresolved {end} {r[end]!r}")
                unresolved += 1
        if r["type"] not in edge_types:
            problems.append(f"edge {r['from']} -> {r['to']}: bad type {r['type']!r}")
        if not r.get("description", "").strip():
            problems.append(f"edge {r['from']} -> {r['to']}: empty description")

    # Every path this graph cites must exist, and every '#anchor' must name a
    # heading that is still there. Both graphs feed the same viewer; a renamed
    # document leaving a dead link behind a green build is the failure mode.
    anchor_cache: dict[str, set] = {}

    def anchors_of(rel_path: str) -> set:
        if rel_path not in anchor_cache:
            try:
                anchor_cache[rel_path] = heading_anchors(
                    (REPO / rel_path).read_text(encoding="utf-8"))
            except OSError:
                anchor_cache[rel_path] = set()
        return anchor_cache[rel_path]

    def check_path(owner: str, field: str, value):
        if not value or value.startswith(("http://", "https://")):
            return
        rel_path, _, fragment = value.partition("#")
        rel_path = rel_path.rstrip("/")
        if not rel_path:
            return
        if not (REPO / rel_path).exists():
            problems.append(
                f"{owner}.{field} points at {rel_path!r} which does not exist on disk")
            return
        if fragment and rel_path.endswith(".md") and fragment not in anchors_of(rel_path):
            problems.append(
                f"{owner}.{field} points at {value!r}: {rel_path} exists, but no "
                f"heading in it anchors as #{fragment}")

    for c in graph["claims"]:
        check_path(c["id"], "doc_ref", c.get("doc_ref"))
    for r in graph["runs"]:
        for field in ("script", "output_dir", "output_path", "doc_ref"):
            check_path(r["id"], field, r.get(field))
    for s in graph["sources"]:
        check_path(s["id"], "path", s.get("path"))
        check_path(s["id"], "doc_ref", s.get("doc_ref"))
    for p in graph["metadata"]["phases"]:
        check_path(p["id"], "doc_ref", p.get("doc_ref"))

    return ids, problems, unresolved


def main():
    graph = build()
    ids, problems, unresolved = validate(graph)

    if problems:
        print("VALIDATION FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    claims = graph["claims"]
    rels = graph["relationships"]
    n_nodes = len(claims) + len(graph["runs"]) + len(graph["sources"])

    print("=" * 72)
    print("ISOMORPHISM GRAPH")
    print("=" * 72)
    print(f"nodes  : {n_nodes}   "
          f"(claims {len(claims)}, runs/models {len(graph['runs'])}, "
          f"sources {len(graph['sources'])})")
    print(f"edges  : {len(rels)}")
    print()

    print("claims by side  : " + ", ".join(
        f"{k} {v}" for k, v in sorted(Counter(c["side"] for c in claims).items())))
    print("claims by role  : " + ", ".join(
        f"{k} {v}" for k, v in sorted(Counter(c["role"] for c in claims).items())))
    print("claims by status: " + ", ".join(
        f"{k} {v}" for k, v in sorted(Counter(c["status"] for c in claims).items())))
    print("edges by type   : " + ", ".join(
        f"{k} {v}" for k, v in sorted(Counter(r["type"] for r in rels).items())))
    print()

    print("-" * 72)
    print("PAIRS (analogous-to)")
    print("-" * 72)
    label = {c["id"]: c["label"] for c in claims}
    for r in rels:
        if r["type"] == "analogous-to":
            print(f"  {label[r['from']]:<45} <-> {label[r['to']]}")
    print()

    print("-" * 72)
    print("BREAKDOWN POINTS (breaks-down-at)")
    print("-" * 72)
    for r in rels:
        if r["type"] == "breaks-down-at":
            print(f"  {label[r['from']]:<45} -/- {label[r['to']]}")
    print()

    print("-" * 72)
    print("CORRECTED / RETIRED")
    print("-" * 72)
    for c in claims:
        if c["status"] in {"corrected", "retired", "refuted"}:
            print(f"  [{c['status']}] {c['label']}  (stopped standing {c['retired']})")
    print()

    print(f"endpoint check : all {len(rels)} edges resolve "
          f"({unresolved} unresolved) against {len(ids)} node ids")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
