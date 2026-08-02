#!/usr/bin/env python3
"""
build_evidence_graph.py
=======================

Builds the ATR project's EVIDENCE GRAPH: hypotheses, findings, discoveries,
concepts, runs, models, docs, artefacts and prior work, wired together with
signed epistemic edges (supports / refutes / qualifies / corrects / retires /
supersedes / tests) plus structural and associative plumbing.

Emits, next to this file:
    _data/entities.json
    _data/visual_config.json

Design rule: PARSE the markdown where the markdown is regular
(the run inventory table, the hypothesis disposition table, the F-heading
list, the Key Discoveries table, the Adjacent Science table, the glossary,
the phase headings).  Where the source is irregular, a curated Python literal
is used, and every curated fact carries a doc_ref pointing at the passage it
came from.

Identity (node ids) is curated and keyed on the parsed row keys, so that if a
source table changes shape the build fails loudly instead of silently
renaming nodes and dangling every edge that referenced them.

No arguments.  Idempotent: byte-identical output for identical inputs.
Validates itself before writing:
  * every relationship endpoint resolves to a real node id
  * every claim.status is in the allowed vocabulary
  * every relationship.type is in the allowed vocabulary
  * every doc_ref / path / script / output_dir resolves on disk, *including*
    its '#anchor': a doc_ref that names a heading the document no longer has is
    a dead link in the viewer and on the published site, and used to pass
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections import Counter, OrderedDict

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(HERE, "_data")



# A markdown table cell escapes a literal pipe as `\|`. Both patterns are
# compiled once here rather than inline, so the split and the unescape can never
# drift apart into disagreeing about what an escape looks like.
_TABLE_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_TABLE_CELL_ESCAPE = re.compile(r"\\\|")


def load_sibling(name: str):
    """Import a module sitting next to this one, by path.

    By path rather than by name so it works however the builder is invoked --
    `python3 docs/graph/build_evidence_graph.py`, from another directory, or
    imported by a test -- without depending on sys.path.
    """
    spec = importlib.util.spec_from_file_location(
        "docs_graph_" + name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The one definition of "which anchors does this document offer", borrowed from
# the drift checker rather than reimplemented here. Two anchor resolvers that
# disagree is precisely the class of drift docs/graph/ exists to catch, and a
# private second copy in the generator would be the one place the checker could
# not see it. See check_record_drift.heading_anchors for the rule itself (both
# GitHub's literal slug and its space-collapsed form are accepted, so
# "Science & Mathematics" validates as `science--mathematics`).
heading_anchors = load_sibling("check_record_drift").heading_anchors

FINDINGS_MD = os.path.join(REPO, "docs", "FINDINGS.md")
JOURNEY_MD = os.path.join(REPO, "docs", "JOURNEY_MAP.md")
README_MD = os.path.join(REPO, "README.md")

# Build stamp is a constant, not "today", so re-running is byte-identical.
BUILD_DATE = "2026-07-25"
SOURCE_LAST_UPDATED = "2026-07-31"  # FINDINGS.md H4 row: the #54 rescore ruling landed

# Dates that appear throughout the record.
D_SERIES_CLOSE = "2026-07-10"  # FINDINGS.md provenance; JOURNEY_MAP Phase 5
D_PERMUTATION = "2026-07-11"   # RESULTS_SUMMARY.md section 6; permutation_report.md
D_ACT_II_5 = "2026-07-19"      # FINDINGS.md scope note; confidence_report.md
D_POST_CLOSE = "2026-07-23"    # JOURNEY_MAP.md header; bell_anatomy.md correction
D_POS0 = "2026-07-28"          # FINDINGS.md H-pos0 row; GPT2_DEEP_DIVE.md section 2.5
D_REGEN = "2026-07-25"         # REGENERATION_REPORT.md; spectral_resonance.ipynb execution
D_RESCORE = "2026-07-31"       # FINDINGS.md H4 row; output_eigen_rescore/report.md; issue #54 ruling
D_SUPERVISORY = "2026-03-20"   # JOURNEY_MAP Phase 4 heading
D_EXPLORATORY = "2026-03-01"   # month anchor: FINDINGS "Original exploratory work: 2026-03"

ALLOWED_STATUS = {
    "supported", "refuted", "not-supported", "qualified", "retired",
    "corrected", "open", "untested",
}
ALLOWED_REL = {
    # epistemic (signed, carry force)
    "supports", "refutes", "qualifies", "corrects", "retires", "supersedes", "tests",
    # structural (neutral plumbing)
    "produced-by", "run-on", "evidenced-by", "documented-in",
    # associative
    "analogous-to", "breaks-down-at", "builds-on", "cites", "relates-to",
    # associative: dependency between open work and what it gates.  `blocks`
    # runs blocker -> gated thing, `blocked-by` is the same relation written
    # from the gated end; only one of the two is drawn per pair, in whichever
    # direction the record states it.
    "blocks", "blocked-by",
}
ALLOWED_CLAIM_TYPE = {"hypothesis", "finding", "concept", "question"}
ALLOWED_RUN_TYPE = {"run", "model", "null-model"}
ALLOWED_SOURCE_TYPE = {"doc", "artefact", "prior-work"}


# --------------------------------------------------------------------------
# Generic markdown helpers
# --------------------------------------------------------------------------

def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def section(text: str, heading_re: str, level: int) -> str:
    """Return the block from the line matching heading_re up to the next
    heading of the same or a shallower level."""
    lines = text.splitlines()
    start = None
    pat = re.compile(heading_re)
    for i, line in enumerate(lines):
        if pat.match(line):
            start = i
            break
    if start is None:
        raise SystemExit(f"FATAL: heading not found: {heading_re}")
    out = [lines[start]]
    stop = re.compile(r"^#{1,%d} " % level)
    for line in lines[start + 1:]:
        if stop.match(line):
            break
        out.append(line)
    return "\n".join(out)


def parse_table(block: str):
    """Parse the first pipe table in `block` into a list of OrderedDicts."""
    header = None
    rows = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            if rows:
                break          # table finished
            continue
        # Split on unescaped pipes only, then unescape. A markdown table cell
        # writes a literal pipe as `\|`, and splitting on every `|` cuts the row
        # in two at that point: the rest of the cell becomes extra columns, and
        # everything past the header's width is discarded by the truncation
        # below. Silently.
        #
        # H-pos0 is the first claim in the record to need one -- its test is
        # "seeded at `<\|endoftext\|>`" -- and it lost its Test instruction, its
        # Falsifiers and its Rationale from the graph while reading correctly in
        # FINDINGS.md. That is precisely the record/graph divergence the drift
        # check exists to catch, arriving through the parser rather than through
        # the record, where nothing was watching for it.
        cells = [_TABLE_CELL_ESCAPE.sub("|", c.strip())
                 for c in _TABLE_CELL_SPLIT.split(line.strip("|"))]
        if header is None:
            header = cells
            continue
        if all(c and set(c) <= set("-: ") for c in cells):
            continue           # the ---|--- separator row
        while len(cells) < len(header):
            cells.append("")
        # strict=True is safe -- and worth having -- because the two sequences
        # are made the same length on the two lines above: short rows are padded
        # with empty cells, long rows are truncated to the header. If that ever
        # stops being true, a silently dropped column becomes a loud error.
        rows.append(OrderedDict(zip(header, cells[:len(header)], strict=True)))
    if not rows:
        raise SystemExit("FATAL: no table rows parsed from block")
    return rows


def demark(text: str) -> str:
    """Strip markdown emphasis / links / code fences from a table cell."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links -> label
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"(?<!\w)\*(?!\s)([^*]+?)\*(?!\w)", r"\1", text)  # *emph*
    text = text.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def gh_anchor(heading: str) -> str:
    """GitHub-flavoured heading anchor."""
    h = heading.strip()
    h = re.sub(r"\{#[^}]*\}", "", h)
    h = h.replace("`", "").replace("*", "").lower()
    h = re.sub(r"[^\w\s-]", "", h)
    h = re.sub(r"\s+", "-", h.strip())
    return h


def sentence(text: str) -> str:
    """Ensure a cell reads as a sentence when concatenated with another."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def kebab(text: str, words: int = 6) -> str:
    t = demark(text).lower()
    t = re.sub(r"[^\w\s-]", " ", t)
    parts = [p for p in re.split(r"[\s_-]+", t) if p]
    return "-".join(parts[:words])


# --------------------------------------------------------------------------
# 1. Phases  (parsed from JOURNEY_MAP.md "### Phase N: ...")
# --------------------------------------------------------------------------

# Phases 0-3 carry no day-level date in the record; FINDINGS.md states
# "Original exploratory work: 2026-03", so the month start is used as the
# anchor and metadata.date_precision records that.
PHASE_START = {
    "phase-0": D_EXPLORATORY,
    "phase-1": D_EXPLORATORY,
    "phase-2": D_EXPLORATORY,
    "phase-3": D_EXPLORATORY,
    "phase-4": D_SUPERVISORY,
    "phase-5": D_SERIES_CLOSE,
}


def parse_phases(journey: str):
    phases = []
    for m in re.finditer(r"^### Phase (\d+): (.+)$", journey, re.M):
        num, rest = m.group(1), m.group(2).strip()
        pid = "phase-%s" % num
        date_m = re.search(r"\d{4}-\d{2}-\d{2}", rest)
        phases.append(OrderedDict([
            ("id", pid),
            ("label", "Phase %s: %s" % (num, rest)),
            ("start", date_m.group(0) if date_m else PHASE_START[pid]),
            ("doc_ref", "docs/JOURNEY_MAP.md#1-timeline-the-intellectual-arc"),
        ]))
    if len(phases) != 6:
        raise SystemExit("FATAL: expected Phase 0-5, parsed %d" % len(phases))
    return phases


# --------------------------------------------------------------------------
# 2. Findings F1..F17  (headings parsed; status/date/summary curated)
# --------------------------------------------------------------------------

FINDING_ID = {
    "F1": "f1-five-attractor-basins",
    "F2": "f2-divine-readout-stable",
    "F3": "f3-fingerprint-refuted",
    "F4": "f4-null-model-regime",
    "F5": "f5-intrinsic-not-apparatus",
    "F6": "f6-cross-hardware-replication",
    "F7": "f7-confidence-inversion",
    "F8": "f8-prolet-coherent-cluster",
    "F9": "f9-divine-period-2",
    "F10": "f10-cycle-anatomy-flip-axis",
    "F11": "f11-jlens-pilot-null",
    "F12": "f12-medium-typographic-cluster",
    "F13": "f13-flip-axis-glitch-alignment",
    "F14": "f14-flip-eigenvalue-l11h8",
    "F15": "f15-lag2-gate",
    "F16": "f16-jlens-phase-probe",
    "F17": "f17-l11h8-copy-promoter",
}

# status, asserted date, evidence run ids, and a 1-3 sentence prose summary.
# Every summary is a compression of the F-section named in doc_ref, read in full.
FINDING_META = {
    "F1": dict(
        status="qualified", asserted=D_SERIES_CLOSE,
        evidence=["run-5-gated-resweep", "run-1-attractor-dominance"],
        description=(
            "GPT-2 Small's 125-prompt sweep resolves into five attractor basins classified "
            "at lock-in (prolet 43.2%, Divine 27.2%, till 15.2%, Anarch 13.6%, solidarity 0.8%), "
            "with 91/125 prompts (73%) passing the lag-1 gate, all at iteration 120, which is the "
            "gate's earliest possible firing point rather than a measured settling time: the true "
            "iteration lies somewhere between 100 and 120 and finer cadence was never run (caveat "
            "5). The earlier iteration-100 table over-counted Anarch by ~10 prompts still drifting "
            "to prolet, and a 2026-07-19 distribution-level note adds that prolet and Anarch are two "
            "argmax peaks over one shared structure. One seed (caveat 2)."),
    ),
    "F2": dict(
        status="corrected", asserted=D_SERIES_CLOSE,
        evidence=["run-5-gated-resweep", "run-1-attractor-dominance"],
        description=(
            "The 34 prompts that never pass the lag-1 convergence gate are exactly the 34 Divine "
            "prompts: a stable top-1 readout over a tensor that keeps moving, the series' clearest "
            "dissociation of dynamics from decoding. Resolved 2026-07-19 as an exact period-2 limit "
            "cycle, so \"never converge\" over-claims and should read \"cycle, pending re-gate\"."),
    ),
    "F3": dict(
        status="supported", asserted=D_SERIES_CLOSE,
        evidence=["run-2-cross-model-sweeps", "run-1-attractor-dominance",
                  "run-4-deep-convergence"],
        description=(
            "GPT-2 Medium, trained on the same WebText corpus, collapses every prompt to the single "
            "token D; Pythia-160m funnels into questioned and Pythia-410m never consolidates. The "
            "basin landscape does not generalise across models, refuting the fingerprint hypothesis "
            "as a general claim. The sweeps are one seed per model (caveat 2), and the 410m deep "
            "run is an 8-prompt subset taken under a CPU constraint (caveat 3)."),
    ),
    "F4": dict(
        # "corrected" on the F2/H-flip precedent: the registered reading was
        # overturned by the finding's own repair. The original arm's numbers
        # (18 basins, em-dash 64%) were manufactured by a mis-calibrated
        # injection norm and a pre-convergence count (caveat 18); run 17
        # repeats the control pair-matched and gated, and inverts it.
        status="corrected", asserted=D_SERIES_CLOSE,
        evidence=["run-3-random-noise-null", "run-17-matched-nu-noise"],
        description=(
            "As registered: 125 calibrated Gaussian tensors iterated through GPT-2 Small converge "
            "into 18 non-semantic basins dominated by the horizontal-bar token (64%), locating the "
            "five basins in the language-driven regime. Inverted 2026-07-31 by the matched-nu, "
            "gated re-run (run 17): at pair-matched injection norms 90/125 noise trials lock in to "
            "7 basins, four of them the language arm's own (prolet, solidarity, Anarch, till; 85.6% "
            "of converged trials), the em-dash basin never appears, and all 35 unconverged trials "
            "pass at lag 2, the Divine cycle's period-2 signature. Labeled at their smallest "
            "passing lag (F15's rule) those periodic trials decode to till, i, player and Divine "
            "itself, so all five language basins reappear and 97/125 trials (77.6%) land in them. "
            "At this injection scale the basins belong "
            "to the weights, not the input regime; the nu-sweep is the registered follow-up."),
    ),
    "F5": dict(
        status="corrected", asserted=D_SERIES_CLOSE,
        evidence=["run-cos-sim-diagnostic", "run-readout-guardrails"],
        description=(
            "Three attribution results place the cross-model differences in the dynamics rather than "
            "the apparatus: the global L2 rescale is inert through layer-0 LayerNorm up to epsilon, "
            "cos_sim_mean verdicts never pass through token decoding, and readout jitter is real but "
            "secondary. The third clause was superseded in part on 2026-07-19 by the full "
            "five-state confidence audit."),
    ),
    "F6": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-6-full-distribution-confidence"],
        description=(
            "The Act II.5 runs reproduced the original five-prompt piece exactly on a fresh cloud "
            "container of a different machine class, with weights from a legacy Hugging Face S3 "
            "mirror: identical terminal attractors and identical dissolution waypoints (Ag at 10, "
            "Rousse at 50), three times over. This is same-code replication on new hardware, not "
            "independent re-implementation."),
    ),
    "F7": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-6-full-distribution-confidence"],
        description=(
            "Reading whole softmax distributions instead of winners inverts the expected picture: the "
            "four settled prolet states decode at p(top-1) 0.064-0.086 with entropy pinned at "
            "5.07-5.09 nats, while the one unsettled state decodes at p = 0.505. Confidence alone "
            "does not separate language attractors from noise attractors; coherence does."),
    ),
    "F8": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-6-full-distribution-confidence", "run-7-coherence-formalization"],
        description=(
            "Under the prolet argmax the whole head of the distribution is one lexical field "
            "(bourgeois, Anarch, comrade, Marx, proletarian...), with plain k=10 coherence 0.410-0.471 "
            "against a ~0.27 baseline and p = 0.001 under both uniform and frequency-matched "
            "permutation nulls. Divine by contrast is a single token (0.318, weakening to p = 0.037 "
            "under probability weighting), and noise is at chance in 12 of 15 trials."),
    ),
    "F9": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-8-divine-motion-audit", "run-9-cycle-anatomy"],
        description=(
            "The Divine tensor is an exact period-2 limit cycle, verified at cos(A, f(f(A))) = "
            "1.000000, alternating between two phases separated by L2 1249 against a norm of 1612 "
            "(cosine 0.685). Every prior schedule sampled only even iterations, so the oscillation "
            "was invisible by construction, and a lag-1 gate can never pass such an object."),
    ),
    "F10": dict(
        status="corrected", asserted=D_ACT_II_5,
        evidence=["run-9-cycle-anatomy"],
        description=(
            "Writing the phases as A = M + d and B = M - d, the flip axis d is a single global rank-1 "
            "direction (per-position axes agree at mean pairwise cosine 1.0000) that is about 95% "
            "invisible to the readout (logit response ratio 0.054, 73% of its energy in W_U's "
            "bottom-100 directions). Its poles run between a game/elemental vocabulary and the "
            "published GPT-2 anomalous-token cluster, identified by inspection against published "
            "lists, not by a systematic test. One trajectory: whether all 34 Divine prompts share "
            "this flip axis is open, blocked on the prompt library (issue #9, caveat 11)."),
    ),
    "F11": dict(
        status="qualified", asserted=D_ACT_II_5,
        evidence=["run-10-jlens-pilot"],
        description=(
            "The restricted J-lens pilot did not support the prolet-inside/Divine-outside prediction; "
            "the point estimate runs slightly the other way (Divine higher on span share at every "
            "layer, higher on sparse share at 11 of 12, margins 0.01-0.02), on a comparison of two "
            "vectors rather than two populations. What appeared instead is a coarser language-vs-noise "
            "boundary, with converged noise states clearly below their random controls. Recorded as a "
            "null with structure: the pilot lens dictionary is strongly low-rank, which makes the raw "
            "lens-vs-random span comparison uninterpretable as a membership test, and the probe saw "
            "phase A only (caveat 13)."),
    ),
    "F12": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-7-coherence-formalization"],
        description=(
            "GPT-2 Medium's universal D attractor reads out near-flat (p(top-1) = 0.010, entropy "
            "7.93-7.96 nats, effective support ~2,800 tokens) yet still passes the coherence test at "
            "p = 0.001, on a cluster of single capitals and code fragments. Hence the standing rule: "
            "no cross-model coherence claim until a shape-class-matched null exists."),
    ),
    "F13": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-11-glitch-alignment"],
        description=(
            "The phase-B pole of the flip axis aligns with the geometric core of under-trained tokens "
            "at cos(-d, u_core) = +0.596, p < 0.001 under both random and norm-matched nulls, and 45 "
            "of the top 50 tokens along -d are in the 0.1% geometric core. The phase-A pole is the "
            "opposite corner: the highest-frequency function words. A strong tilt (0.46-0.60), not "
            "an identity, and measured on the single audited Divine trajectory (caveat 14)."),
    ),
    "F14": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-12-flip-axis-eigenvalue"],
        description=(
            "Forward-mode autodiff puts the flip-axis eigenvalue at -4.3 at the symmetric pivot (an "
            "overshooting reflection, not the conjectured -1), with the composed two-step multiplier "
            "at +0.10: a period-doubling configuration. The inversion happens entirely inside block "
            "11, where one head, L11.H8, carries 99.1% of the attention flip. All headline numbers "
            "are for the physical on-shell axis d_sym, not the frame-mixed committed axis (caveat "
            "15), and are derivatives at one point on the single audited trajectory (caveat 14)."),
    ),
    "F15": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-13-lagk-regate"],
        description=(
            "atr_engine.run_atr_gated gained a gate_lag parameter and a lag_scan helper, and a dense "
            "24-iteration continuation separates three signatures cleanly: the prolet fixed point "
            "passes at every lag, the Divine state fails every odd lag (0.6849) and passes every even "
            "one, and the noise control decays monotonically. The audited Divine trajectory is "
            "converged under gate_lag = 2 and unconvergeable under gate_lag = 1. Two honest limits: "
            "a fixed lag-2 gate inherits the same aliasing one octave up (a period-4 cycle would be "
            "invisible again), so the recommended re-gate runs the full lag table and gates at the "
            "smallest passing lag; and the lag-k gate does not fix threshold-blindness to slow "
            "drift. The other 33 period-2 prompts remain blocked on the prompt library (issue #9)."),
    ),
    "F16": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-14-jlens-phase-probe"],
        description=(
            "Re-probing both phases, the pivot and the flip axis: the pilot's reversal holds for phase "
            "A, strengthens at the pivot M (the most lens-expressible object probed), and reverses for "
            "phase B, which sits below the prolet attractor at every layer. The physical flip axis "
            "d_sym is almost entirely outside the lens, span share 0.013 at L11 against a 0.252 chance "
            "level. This is the same restricted 193-token pilot lens, so every F11 limitation is "
            "inherited (caveat 13), on the single audited trajectory (caveat 14)."),
    ),
    "F17": dict(
        status="supported", asserted=D_ACT_II_5,
        evidence=["run-15-suppression-test"],
        description=(
            "L11.H8's OV inverts the flip axis most strongly of all 144 heads (cos -0.9619, rank 1) "
            "and ablating it inside the loop collapses the cycle to a fixed point within ~10 "
            "iterations, while a same-layer control ablation leaves the cycle running. But on ordinary "
            "text it raises the attended token's logit at 91.4% of positions: a copy promoter, not the "
            "copy-suppression head the mechanism first suggested. The copy test is 12 sentences and "
            "116 positions, far from the decision boundary but small (caveat 16), and it measures "
            "suppression in the token-unembedding sense only; the cycle tests follow the single "
            "audited trajectory (caveat 14)."),
    ),
}


def parse_findings(findings_md: str):
    claims = []
    seen = []
    for m in re.finditer(r"^### (F(\d+)):\s*(.+)$", findings_md, re.M):
        key = m.group(1)
        heading_body = m.group(3).strip()
        anchor_m = re.search(r"\{#([^}]+)\}", heading_body)
        anchor = anchor_m.group(1) if anchor_m else gh_anchor(m.group(0)[4:])
        title = demark(re.sub(r"\{#[^}]*\}", "", heading_body)).strip()
        if key not in FINDING_ID:
            raise SystemExit("FATAL: unmapped finding heading %r" % key)
        meta = FINDING_META[key]
        seen.append(key)
        claims.append(OrderedDict([
            ("id", FINDING_ID[key]),
            ("label", "%s: %s" % (key, title)),
            ("type", "finding"),
            ("status", meta["status"]),
            ("description", meta["description"]),
            ("phase", "phase-5"),
            ("asserted", meta["asserted"]),
            ("retired", None),
            ("doc_ref", "docs/FINDINGS.md#%s" % anchor),
            ("evidence", list(meta["evidence"])),
        ]))
    missing = set(FINDING_ID) - set(seen)
    if missing:
        raise SystemExit("FATAL: findings not found in FINDINGS.md: %s" % sorted(missing))
    return claims


# --------------------------------------------------------------------------
# 3. Hypotheses  (table parsed; status curated from the disposition prose)
# --------------------------------------------------------------------------

HYP_ID = {
    "H0": "h0-determinism",
    "H1": "h1-prolet-dominant",
    "H2": "h2-divine-secondary-basin",
    "H3": "h3-corpus-topology",
    "H4": "h4-head-power-iteration",
    "H-fingerprint": "h-fingerprint",
    "H-till": "h-till",
    "H-D1": "h-d1",
    "H-J1": "h-j1",
    "H-glitch": "h-glitch",
    "H-flip": "h-flip",
    "H-supp": "h-supp",
    "H-pos0": "h-pos0",
}

# status is read off the bolded lead of each disposition cell in
# docs/FINDINGS.md#3-hypothesis-dispositions; phase is where the hypothesis
# was first put on the record (JOURNEY_MAP section 3 / the finding that raised it).
#
# The status vocabulary is eight words, and the three negative ones are distinct:
#   refuted        the evidence contradicts the claim
#   not-supported  the evidence failed to back the claim, without contradicting it
#                  (a null result)
#   qualified      the claim survives, in a narrowed or mixed form
# "not-supported" was added because H-J1 previously had to be filed as "qualified",
# which put a pilot-confidence null in the same bucket as genuine partial support.
# Cases where the one-word status still compresses a longer disposition are recorded
# here so the compression is auditable rather than silent:
#   H-D1  "Supported in a weakened, more precise form" -> supported, with the
#         weakening carried by the F10 -> H-D1 `qualifies` edge.
#   H-J1  "Not supported at pilot confidence (2026-07-19); now phase-qualified (F16)"
#         -> not-supported. The hypothesis as stated (a static prolet-inside /
#         Divine-outside split) failed: "the point estimate runs slightly the other
#         way". "refuted" would overstate a pilot-confidence null. The "phase-qualified"
#         clause is not partial survival of the stated claim; it is a different,
#         phase-indexed claim raised by the later F16 re-probe, with the full build
#         still pending (issue #8). The F16 -> H-J1 `qualifies` edge carries that half.
#   H3    "Weakened further at close; coherence half upgraded 2026-07-19" -> stays
#         qualified, deliberately. This one is NOT a null and must not be re-filed as
#         not-supported: the corpus-causal half failed cross-model (F3) and the
#         all-warm matrix was shown to be an anisotropy artifact, but the
#         semantic-coherence half was UPGRADED with permutation support ("coherence
#         0.41-0.47 vs 0.27, p = 0.001 under both nulls; F8"). A claim that lost one
#         half and strengthened the other is the textbook "qualified", not a null.
#   H4    "Not supported as registered; superseded by the corrected-target rescore"
#         -> not-supported, on the H-J1 precedent: the registered claim (settled
#         state ~ TOP SINGULAR VECTOR, cos > 0.9, most heads) failed at 5/144.
#         The rescore's finding (settled state = DOMINANT EIGENVECTOR for every
#         settling head) is a different, corrected claim raised by run 16 under
#         TC's #54 ruling, carried in the disposition prose and the run-16
#         evidence edge, not by the one-word status. "refuted" would be wrong:
#         the corrected claim is the registered idea aimed at the right object,
#         and it largely succeeds.
# The `asserted` dates below are NOT in the record: the documents date dispositions,
# never the moment a hypothesis was raised. They are placements, not measurements;
# see metadata.date_precision.
HYP_META = {
    "H0":            dict(status="supported", phase="phase-2", asserted=D_EXPLORATORY, retired=None),
    "H1":            dict(status="supported", phase="phase-2", asserted=D_EXPLORATORY, retired=None),
    "H2":            dict(status="supported", phase="phase-2", asserted=D_EXPLORATORY, retired=None),
    "H3":            dict(status="qualified", phase="phase-2", asserted=D_EXPLORATORY, retired=None),
    "H4":            dict(status="not-supported", phase="phase-4", asserted=D_SUPERVISORY, retired=None),
    "H-fingerprint": dict(status="refuted",   phase="phase-4", asserted=D_SUPERVISORY, retired=D_SERIES_CLOSE),
    "H-till":        dict(status="refuted",   phase="phase-5", asserted=D_SERIES_CLOSE, retired=D_SERIES_CLOSE),
    "H-D1":          dict(status="supported", phase="phase-5", asserted=D_SERIES_CLOSE, retired=None),
    "H-J1":          dict(status="not-supported", phase="phase-5", asserted=D_ACT_II_5, retired=None),
    "H-glitch":      dict(status="supported", phase="phase-5", asserted=D_ACT_II_5, retired=None),
    "H-flip":        dict(status="corrected", phase="phase-5", asserted=D_ACT_II_5, retired=None),
    "H-supp":        dict(status="refuted",   phase="phase-5", asserted=D_ACT_II_5, retired=D_ACT_II_5),
    # H-pos0 is the first hypothesis on the record derived from the architecture
    # rather than from a run, so it carries no evidence edges: the causal-mask
    # argument is deductive, and the sequence-length-1 test that would make it
    # empirical has not been run (and needs an engine change first). H4 carried
    # "untested" for the same reason until run 16 gave it evidence and the #54
    # ruling moved it to "not-supported"; H-pos0 has no such run yet.
    "H-pos0":        dict(status="untested",  phase="phase-5", asserted=D_POS0, retired=None),
}

HYP_EVIDENCE = {
    "H0": ["run-0-repeatability-gate", "run-6-full-distribution-confidence"],
    "H1": ["run-1-attractor-dominance", "run-5-gated-resweep"],
    "H2": ["run-5-gated-resweep", "run-8-divine-motion-audit", "run-9-cycle-anatomy"],
    "H3": ["run-token-neighbourhood", "run-all-warm-permutation",
           "run-7-coherence-formalization"],
    "H4": ["run-16-eigen-rescore"],
    "H-fingerprint": ["run-2-cross-model-sweeps", "run-3-random-noise-null"],
    "H-till": ["run-5-gated-resweep"],
    "H-D1": ["run-8-divine-motion-audit", "run-9-cycle-anatomy"],
    "H-J1": ["run-10-jlens-pilot", "run-14-jlens-phase-probe"],
    "H-glitch": ["run-11-glitch-alignment"],
    "H-flip": ["run-12-flip-axis-eigenvalue"],
    "H-supp": ["run-15-suppression-test"],
    "H-pos0": [],
}


def parse_hypotheses(findings_md: str):
    block = section(findings_md, r"^## 3\. Hypothesis dispositions", 2)
    rows = parse_table(block)
    claims = []
    for row in rows:
        key = row["ID"].strip()
        if key not in HYP_ID:
            raise SystemExit("FATAL: unmapped hypothesis id %r" % key)
        meta = HYP_META[key]
        statement = demark(row["Hypothesis"])
        disposition = demark(row["Disposition"])
        claims.append(OrderedDict([
            ("id", HYP_ID[key]),
            ("label", "%s: %s" % (key, statement)),
            ("type", "hypothesis"),
            ("status", meta["status"]),
            ("description", "%s Disposition: %s" % (statement + ".", disposition)),
            ("phase", meta["phase"]),
            ("asserted", meta["asserted"]),
            ("retired", meta["retired"]),
            ("doc_ref", "docs/FINDINGS.md#3-hypothesis-dispositions"),
            ("evidence", list(HYP_EVIDENCE[key])),
        ]))
    if len(claims) != len(HYP_ID):
        raise SystemExit("FATAL: hypothesis table has %d rows, expected %d"
                         % (len(claims), len(HYP_ID)))
    return claims


# --------------------------------------------------------------------------
# 4. Key Discoveries  (table parsed, including the *Retired/Corrected* notes)
# --------------------------------------------------------------------------

DISC_ID = {
    "1": "disc-1-discrete-attractor-basins",
    "2": "disc-2-shared-dissolution-pathway",
    "3": "disc-3-reddit-2018-discourse",
    "4": "disc-4-reproducible-terminals",
    "5": "disc-5-five-basins-not-two",
    "6": "disc-6-semantic-clustering-we",
    "7": "disc-7-capit-is-capitulation",
    "8": "disc-8-structural-semantic-transition",
    "9": "disc-9-all-warm-compact-subspace",
    "10": "disc-10-thematic-centre-of-mass",
    "11": "disc-11-brouwer-guarantees-basins",
    "12": "disc-12-landscapes-model-specific",
    "13": "disc-13-basins-regime-specific",
    "14": "disc-14-labels-survive-gating",
    "15": "disc-15-divine-readout-stable",
    "16": "disc-16-differences-intrinsic",
}

DISC_PHASE = {
    "EXP_009aFIX": "phase-1",
    "EXP_009d0": "phase-3",
    "EXP_009d1": "phase-3",
    "Session 01": "phase-4",
    "Session 02": "phase-4",
}

DISC_EVIDENCE = {
    "1": ["run-original-piece"], "2": ["run-original-piece"], "3": ["run-original-piece"],
    "4": ["run-0-repeatability-gate"], "5": ["run-1-attractor-dominance"],
    "6": ["run-token-neighbourhood"], "7": ["run-token-neighbourhood"],
    "8": ["run-token-neighbourhood"], "9": ["run-token-neighbourhood"],
    "10": ["run-token-neighbourhood"], "11": [],
    "12": ["run-2-cross-model-sweeps"], "13": ["run-3-random-noise-null"],
    "14": ["run-5-gated-resweep"], "15": ["run-5-gated-resweep"],
    "16": ["run-cos-sim-diagnostic"],
}

# Overrides for discoveries whose disposition is recorded elsewhere than in an
# inline *annotation* on the Key Discoveries row.
DISC_OVERRIDE = {
    # JOURNEY_MAP Phase 1: "(interpretation later qualified, see Phase 5)"
    "3": dict(status="qualified", note=(
        "Qualified at series close: JOURNEY_MAP Phase 1 marks the interpretation "
        "\"later qualified, see Phase 5\", and F3 shows GPT-2 Medium reading the same "
        "corpus into one empty token.")),
    # FINDINGS caveat 4: quantitative support withdrawn, neighbourhood claim
    # "remains qualitative"; FINDINGS F8 relocates it to the readout distribution.
    "6": dict(status="qualified", note=(
        "Qualified 2026-07-11: the permutation test withdrew the compact-subspace "
        "support and FINDINGS caveat 4 records the local neighbourhood observation as "
        "qualitative only; F8 re-establishes clustering one level deeper, in the readout "
        "distribution.")),
    # FINDINGS F1 / RESULTS_SUMMARY section 5: the iteration-100 shares this row
    # records were corrected at lock-in by the convergence-gated re-sweep.
    "5": dict(status="corrected", note=(
        "Corrected 2026-07-10: the convergence-gated re-sweep re-classified the shares at "
        "lock-in (prolet 43.2%, Divine 27.2%, till 15.2%, Anarch 13.6%, solidarity 0.8%); "
        "the iteration-100 numbers recorded here over-counted Anarch by ~10 prompts.")),
    # The row's annotation gives no date of its own; the permutation test that
    # settled the second leg ran 2026-07-11 (RESULTS_SUMMARY section 6).
    "10": dict(retired=D_PERMUTATION),
}

# "Inverted" joined the vocabulary 2026-07-31, when run 17 reversed Key
# Discovery 13's reading; it maps to "corrected" like "Resolved" does.
ANNOT_RE = re.compile(
    r"\*(Retired|Corrected|Resolved|Inverted)(?:\s+(\d{4}-\d{2}-\d{2}))?\s*:", re.I)
ANNOT_STATUS = {"retired": "retired", "corrected": "corrected",
                "resolved": "corrected", "inverted": "corrected"}


def parse_discoveries(journey: str):
    block = section(journey, r"^## 2\. Key Discoveries", 2)
    rows = parse_table(block)
    claims = []
    for row in rows:
        num = row["#"].strip()
        if num not in DISC_ID:
            raise SystemExit("FATAL: unmapped discovery number %r" % num)
        raw = row["Discovery"].strip()
        when = demark(row["When"])
        evidence_text = demark(row["Evidence"])

        status, retired = "supported", None
        note = ""
        for idx, cell in enumerate((raw, row["Evidence"])):
            m = ANNOT_RE.search(cell)
            if m:
                status = ANNOT_STATUS[m.group(1).lower()]
                if m.group(2):
                    retired = m.group(2) if status == "retired" else None
                note = demark(cell[m.start():]).strip()
                if idx == 1:
                    # The annotation lives in the Evidence cell, so it is already
                    # inside evidence_text. Cut it out, or it is emitted twice:
                    # once inline in the evidence prose and again as the note
                    # appended below. (The Discovery cell gets the same treatment
                    # via the ANNOT_RE.split on `headline`.)
                    evidence_text = demark(cell[:m.start()]).strip()
                break
        ov = DISC_OVERRIDE.get(num, {})
        status = ov.get("status", status)
        retired = ov.get("retired", retired)
        if status not in ("retired", "refuted"):
            retired = ov.get("retired", None)

        headline = demark(ANNOT_RE.split(raw)[0]) if ANNOT_RE.search(raw) else demark(raw)
        headline = headline.rstrip(" .*")
        description = "%s Recorded from %s; evidence: %s." % (
            headline + ".", when, evidence_text.rstrip("."))
        if note:
            description += " " + note
        if ov.get("note"):
            description += " " + ov["note"]

        phase = DISC_PHASE.get(when.split(" (")[0].strip(), "phase-5")
        asserted = PHASE_START[phase]
        claims.append(OrderedDict([
            ("id", DISC_ID[num]),
            ("label", "Discovery %s: %s" % (num, headline)),
            ("type", "finding"),
            ("status", status),
            ("description", description),
            ("phase", phase),
            ("asserted", asserted),
            ("retired", retired),
            ("doc_ref", "docs/JOURNEY_MAP.md#2-key-discoveries-chronological"),
            ("evidence", list(DISC_EVIDENCE[num])),
        ]))
    if len(claims) != len(DISC_ID):
        raise SystemExit("FATAL: Key Discoveries table has %d rows, expected %d"
                         % (len(claims), len(DISC_ID)))
    return claims


# --------------------------------------------------------------------------
# 5. Concepts  (Adjacent Science table + glossary, both parsed)
# --------------------------------------------------------------------------

# Adjacent Science rows.  Rows whose Domain is "Prior Art" are routed to
# sources[] as prior-work instead of to claims[].
ADJACENT_ID = {
    "Power iteration": "concept-power-iteration",
    "Fixed-point theory, basin of attraction": "concept-fixed-point-theory",
    "Brouwer fixed-point theorem": "concept-brouwer-fixed-point",
    "Mixing time (T_mix)": "concept-mixing-time",
    "Impulse response / room modes": "concept-impulse-response",
    "Fractal dimensional analysis": "concept-fractal-dimension",
    "Byte Pair Encoding": "concept-byte-pair-encoding",
    "Activation patching, probing, SAEs": "concept-activation-patching",
    "Logit Lens / Tuned Lens": "concept-logit-lens",
    "Deleuze, Body without Organs": "concept-body-without-organs",
    "Levin, TAME (morphogenesis)": "concept-tame-morphogenesis",
}
ADJACENT_PRIOR = {
    "Slonski Q-vector dichotomy": "prior-slonski-q-vector",
    "Turner et al., Representation Engineering": "prior-turner-repeng",
    "Shumailov et al., Model Collapse": "prior-shumailov-model-collapse",
}
ADJACENT_STATUS = {
    # JOURNEY_MAP's own Relevance cell: "its hypotheses do not hold on the L2
    # shell", corrected 2026-07-23 (Key Discovery 11 annotation).
    "concept-brouwer-fixed-point": ("corrected", D_POST_CLOSE),
}
PRIOR_URL = {
    "prior-shumailov-model-collapse": "https://arxiv.org/abs/2305.17493",
}


def parse_adjacent(journey: str):
    block = section(journey, r"^## 4\. Adjacent Science & Mathematics", 2)
    rows = parse_table(block)
    concepts, priors = [], []
    for row in rows:
        name = demark(row["Concept"])
        domain = demark(row["Domain"])
        relevance = demark(row["Relevance to ATR"])
        if name in ADJACENT_PRIOR:
            pid = ADJACENT_PRIOR[name]
            priors.append(OrderedDict([
                ("id", pid),
                ("title", name),
                ("type", "prior-work"),
                ("path", PRIOR_URL.get(
                    pid, "docs/JOURNEY_MAP.md#4-adjacent-science--mathematics")),
                ("description", "%s (%s). Relevance to ATR: %s"
                 % (name, domain, sentence(relevance))),
                ("doc_ref", "docs/JOURNEY_MAP.md#4-adjacent-science--mathematics"),
                ("cited_by", "doc-journey-map"),
            ]))
            continue
        if name not in ADJACENT_ID:
            raise SystemExit("FATAL: unmapped adjacent-science concept %r" % name)
        cid = ADJACENT_ID[name]
        status, retired = ADJACENT_STATUS.get(cid, (None, None))
        if status is None:
            status = "untested" if "untested" in relevance.lower() else "open"
        if status not in ("retired", "refuted"):
            retired = None      # a corrected claim still stands, in amended form
        concepts.append(OrderedDict([
            ("id", cid),
            ("label", name),
            ("type", "concept"),
            ("status", status),
            ("description", "%s (%s). Relevance to ATR: %s"
             % (name, domain, sentence(relevance))),
            ("phase", "phase-4"),
            ("asserted", D_SUPERVISORY),
            ("retired", retired),
            ("doc_ref", "docs/JOURNEY_MAP.md#4-adjacent-science--mathematics"),
        ]))
    return concepts, priors


# Glossary terms.  Three terms duplicate Adjacent-Science concepts and are
# skipped here (BPE, Nonlinear power iteration, Q-vector dichotomy).
GLOSSARY_ID = {
    "ATR (Activation Tensor Resonance)": "concept-atr",
    "Attractor basin": "concept-attractor-basin",
    "Basin token": "concept-basin-token",
    "Waypoint token": "concept-waypoint-token",
    "Dissolution pathway": "concept-dissolution-pathway",
    "Phase transition (structural→semantic)": "concept-structural-semantic-transition",
    "T_mix_LLM": "concept-t-mix-llm",
    "Residual stream": "concept-residual-stream",
    "W_E": "concept-w-e",
    "Position collapse": "concept-position-collapse",
    "Cross-prompt invariance": "concept-cross-prompt-invariance",
    "L2 normalisation": "concept-l2-normalisation",
    "All-warm matrix": "concept-all-warm-matrix",
    "Eigenvoice": "concept-eigenvoice",
    "Glitch token": "concept-glitch-token",
    "Bias profile": "concept-bias-profile",
}
GLOSSARY_SKIP = {"BPE", "Nonlinear power iteration", "Q-vector dichotomy"}
GLOSSARY_STATUS = {
    # Retired 2026-07-11 by the W_E permutation test (FINDINGS caveat 4).
    "concept-all-warm-matrix": ("retired", D_PERMUTATION),
    # The glossary marks this a "*Retired term.*", refuted at series close (F3).
    "concept-bias-profile": ("retired", D_SERIES_CLOSE),
    # The glossary carries its own "reporting-register correction" to F4.
    "concept-eigenvoice": ("corrected", D_SERIES_CLOSE),
    # Session 01 "ruled out for our basins"; F10/F13 put the Divine cycle's
    # phase-B pole in exactly that region.
    "concept-glitch-token": ("corrected", D_ACT_II_5),
    # "Proposed metric", never computed (JOURNEY_MAP section 7).
    "concept-t-mix-llm": ("untested", None),
}


def parse_glossary(journey: str):
    block = section(journey, r"^## 5\. Glossary", 2)
    rows = parse_table(block)
    concepts = []
    for row in rows:
        term = demark(row["Term"])
        if term in GLOSSARY_SKIP:
            continue
        if term not in GLOSSARY_ID:
            raise SystemExit("FATAL: unmapped glossary term %r" % term)
        cid = GLOSSARY_ID[term]
        status, retired = GLOSSARY_STATUS.get(cid, ("open", None))
        if status not in ("retired", "refuted"):
            retired = None      # a corrected claim still stands, in amended form
        first = demark(row["First Appearance"])
        concepts.append(OrderedDict([
            ("id", cid),
            ("label", term),
            ("type", "concept"),
            ("status", status),
            ("description", "%s First appearance: %s."
             % (sentence(demark(row["Definition"])), first)),
            ("phase", DISC_PHASE.get(first, "phase-4")),
            ("asserted", PHASE_START[DISC_PHASE.get(first, "phase-4")]),
            ("retired", retired),
            ("doc_ref", "docs/JOURNEY_MAP.md#5-glossary"),
        ]))
    return concepts


# Concepts introduced by the Act II.5 / mechanism series ("first use:" in
# FINDINGS.md), plus the two external frames the series argues against.
SERIES_CONCEPTS = [
    dict(id="concept-period-2-limit-cycle", label="Period-2 limit cycle",
         status="supported", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f9-the-divine-anomaly-resolved-an-exact-period-2-limit-cycle-hidden-by-aliasing",
         description=("The tensor alternates between two states, phase A and phase B, reproduced "
                      "to machine precision every two iterations: cos(A, f(f(A))) = 1.000000. Not a "
                      "wandering orbit and not a fixed point.")),
    dict(id="concept-phase-a-b", label="Phase A / phase B",
         status="supported", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f9-the-divine-anomaly-resolved-an-exact-period-2-limit-cycle-hidden-by-aliasing",
         description=("First use in F9: the two alternating states of the cycle, phase A being the "
                      "one every prior snapshot schedule happened to sample. Separated by L2 1249, "
                      "cosine 0.685, KL about 0.25 nats per half-cycle.")),
    dict(id="concept-flip-axis", label="Flip axis (d)",
         status="supported", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f10-anatomy-of-the-period-2-cycle-one-nearly-readout-invisible-flip-axis-between-a-game-vocabulary-pole-and-the-glitch-token-pole",
         description=("First use in F10: the single direction the iterated map negates on each pass, "
                      "called the hinge in earlier revisions. One global rank-1 direction; the "
                      "physical on-shell version is d_sym.")),
    dict(id="concept-aliasing", label="Aliasing (even-iteration sampling)",
         status="supported", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f9-the-divine-anomaly-resolved-an-exact-period-2-limit-cycle-hidden-by-aliasing",
         description=("First use in F9: sampling a periodic signal only at times that hide its "
                      "oscillation. Every archived snapshot from lock-in onward fell on even "
                      "iterations, so a period-2 orbit was recorded at a single phase.")),
    dict(id="concept-coherence", label="Coherence (top-k embedding clustering)",
         status="supported", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f8-the-prolet-basin-is-a-coherent-cluster-of-related-tokens-not-a-single-token-permutation-tested",
         description=("First use in F8: the mean pairwise cosine similarity in W_E among the top-k "
                      "tokens of a converged state's readout distribution, with a probability-weighted "
                      "variant. High coherence means the head of the distribution is one cluster.")),
    dict(id="concept-invisibility-ratio", label="Invisibility ratio",
         status="supported", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f9-the-divine-anomaly-resolved-an-exact-period-2-limit-cycle-hidden-by-aliasing",
         description=("First use in F9: the norm of a step's actual effect on the full logit vector "
                      "divided by the mean effect of 20 random directions of equal norm. Values below "
                      "1 mean the motion is preferentially readout-invisible; the cycle step sits at "
                      "0.295 and the flip axis at 0.054.")),
    dict(id="concept-lag1-convergence-gate", label="Lag-1 convergence gate",
         status="corrected", phase="phase-5", asserted=D_SERIES_CLOSE,
         doc_ref="docs/FINDINGS.md#f15-a-lag-2-convergence-gate-recognises-the-period-2-cycle-the-engine-now-supports-it",
         description=("The original convergence test: cosine similarity of successive mean tensors "
                      "above 0.999 for three consecutive checks. It compares consecutive iterates, so "
                      "it fails a period-2 cycle by construction whatever its threshold.")),
    dict(id="concept-jspace-workspace", label="J-space / the Jacobian lens",
         status="open", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f11-j-lens-pilot-the-prolet-insidedivine-outside-prediction-did-not-hold-the-boundary-that-appeared-is-language-vs-noise",
         description=("Anthropic's J-space proposal: a model's verbalizable states occupy a "
                      "distinguished subspace, probed by a lens built from averaged Jacobians of the "
                      "forward map. This repo's reading companion is docs/JSPACE_PRIMER.md.")),
    dict(id="concept-copy-suppression", label="Copy suppression (head class)",
         status="open", phase="phase-5", asserted=D_ACT_II_5,
         doc_ref="docs/FINDINGS.md#f17-l11h8-is-load-bearing-for-the-cycle-but-is-a-copy-promoter-not-a-copy-suppression-head",
         description=("The documented head class (GPT-2 Small's L10.H7) that detects the currently "
                      "predicted token and writes against its unembedding. The hypothesis that "
                      "L11.H8 belongs to it was tested and refuted with the opposite sign.")),
]


def series_concepts():
    out = []
    for c in SERIES_CONCEPTS:
        out.append(OrderedDict([
            ("id", c["id"]), ("label", c["label"]), ("type", "concept"),
            ("status", c["status"]), ("description", c["description"]),
            ("phase", c["phase"]), ("asserted", c["asserted"]), ("retired", None),
            ("doc_ref", c["doc_ref"]),
        ]))
    return out


# --------------------------------------------------------------------------
# 5b. Questions  (the things the record explicitly leaves open)
# --------------------------------------------------------------------------
#
# A `question` is a claim type in its own right: not a hypothesis (which asserts
# something testable) and not a concept (which is vocabulary).  It exists because
# `open` was useless as a work signal - every `open` claim in this graph was a
# concept, `open` being what a concept falls into when there is no epistemic
# verdict to record.  `type == "question" and status == "open"` is the query that
# returns work.
#
# HARVESTING RULE, and it is strict: a question node must correspond to something
# the record actually leaves open, and the sentence that leaves it open is quoted
# verbatim in the description.  Nothing here is inferred, extrapolated, or added
# because it "would obviously be next".  Sources used, all read in full:
#   * docs/FINDINGS.md section 6 (stage boundary: work closed unexecuted)
#   * docs/FINDINGS.md section 5 (open directions, in rough order of leverage)
#   * docs/FINDINGS.md section 4 (caveats 1, 5, 6, 10, 11, 14)
#   * docs/FINDINGS.md section 3 (hypothesis dispositions: pending-work clauses)
#   * docs/FINDINGS.md F10 / F11 / F12 / F15 bodies
#   * docs/JOURNEY_MAP.md section 7 (Open Questions), non-struck rows only
#
# Every question here is `open`.  JOURNEY_MAP section 7 marks one row "Untested";
# that word is quoted in the description rather than used as the status, because
# `untested` in this graph means "asserted, never put to a run", and a question
# asserts nothing.  Keeping one status across the set also keeps the work query a
# single predicate.
#
# Struck-through rows of JOURNEY_MAP section 7 are deliberately NOT emitted: they
# are answered, and their answers are already in the graph as findings with the
# edges that settled them (F9 for the `Divine` object, F3 for basin profiles, the
# permutation test for the W_E statistics).

A_CAVEATS = "docs/FINDINGS.md#caveats"
A_OPEN_QUESTIONS = "docs/JOURNEY_MAP.md#7-open-questions"
A_F10 = ("docs/FINDINGS.md#f10-anatomy-of-the-period-2-cycle-one-nearly-readout-"
         "invisible-flip-axis-between-a-game-vocabulary-pole-and-the-glitch-token-pole")
A_F11 = ("docs/FINDINGS.md#f11-j-lens-pilot-the-prolet-insidedivine-outside-prediction-"
         "did-not-hold-the-boundary-that-appeared-is-language-vs-noise")
A_F12 = ("docs/FINDINGS.md#f12-cross-model-gpt-2-mediums-universal-attractor-is-a-"
         "typographic-cluster-over-a-near-flat-readout")
A_F15 = ("docs/FINDINGS.md#f15-a-lag-2-convergence-gate-recognises-the-period-2-cycle-"
         "the-engine-now-supports-it")

QUESTIONS = [
    dict(
        id="q-why-gpt2-small",
        label="Q: Why does GPT-2 Small alone resolve language into few semantic basins?",
        asserted=D_SERIES_CLOSE, doc_ref=A_OPEN_QUESTIONS,
        description=(
            "JOURNEY_MAP section 7 files this one at a status of its own: \"The open "
            "question of the series\", next step \"New experimental stage\". FINDINGS "
            "section 5 puts it first in the open directions, ordered by leverage: \"why "
            "GPT-2 Small (the anomaly, now with low-probability coherent clusters as the "
            "thing to explain)\", and adds that the anomaly was sharpened rather than "
            "removed by Act II.5.")),
    dict(
        id="q-flip-axis-generality",
        label="Q: Do all 34 Divine prompts share the F10 flip axis, head and eigenvalue?",
        asserted=D_ACT_II_5, doc_ref=A_F10,
        description=(
            "F10 states it in as many words: \"Open: whether all 34 Divine prompts share "
            "this flip axis (blocked on the prompt-library restoration, issue #9).\" "
            "Caveat 14 widens it to the whole mechanism series: \"Whether the other 33 "
            "period-2 prompts share the flip axis, the flip head (L11.H8), the "
            "eigenvalue, and the anomalous-token alignment is untested (prompt library "
            "pending, issue #9).\" FINDINGS section 5 calls this \"what is now most open "
            "on the Divine object\".")),
    dict(
        id="q-lag2-regate-33",
        label="Q: Do the other 33 period-2 prompts re-gate as converged?",
        asserted=D_ACT_II_5, doc_ref=A_F15,
        description=(
            "F15 demonstrates the lag-k gate for one trajectory and records the rest as "
            "outstanding: \"The other 33 period-2 prompts remain blocked on the prompt "
            "library (issue #9); one, the Syntactic prompt, is now re-gated as "
            "converged.\" The procedure is on record too - the re-gate \"runs the full "
            "lag table on a short dense continuation and gates each state at its smallest "
            "passing lag\" - so this is blocked on an artefact, not on a method. "
            "JOURNEY_MAP section 7 gives the same next step: \"Re-gate the other 33 "
            "prompts (blocked on issue #9)\".")),
    dict(
        id="q-prompt-library",
        label="Q: Is the 125-prompt library restored (issue #9)?",
        asserted=D_ACT_II_5, doc_ref=A_CAVEATS,
        status="retired", retired="2026-07-31",
        description=(
            "Caveats 11 and 14 both filed the same gate in the same three words, \"prompt "
            "library pending, issue #9\", and FINDINGS section 5 named it as the blocker "
            "on two separate directions at once: the re-gate of the 34 cycling prompts "
            "and the flip-axis generality question. This node was a restoration task "
            "rather than a question about the model; it is in the graph so that the two "
            "threads it gated point at one blocker instead of repeating one sentence in "
            "two descriptions, where nothing can pair them. Answered 2026-07-31: yes. "
            "prompt_library.py is restored on main (issue #24, provenance-flagged full "
            "restoration), the record's blocked-on language is lifted (PR #103), and the "
            "two formerly gated threads are queued in ALIGNMENT_REVIEW.md section 5.")),
    dict(
        id="q-jlens-full-build",
        label="Q: What does the phase-aware full J-lens build show (issue #8)?",
        asserted=D_ACT_II_5, doc_ref=A_F11,
        description=(
            "F11 specifies the outstanding work: \"The full build (issue #8) should be "
            "phase-aware: probe both phases and the pivot M.\" H-J1's disposition closes "
            "on the same debt - \"Full build still pending (issue #8)\" - and FINDINGS "
            "section 5 lists \"the phase-aware J-lens full build (F11, issue #8)\" among "
            "the open directions. The pilot's own limits (caveat 13) are what the full "
            "build exists to lift.")),
    dict(
        id="q-independent-reimplementation",
        label="Q: Does an independent re-implementation reproduce the result?",
        asserted=D_ACT_II_5, doc_ref=A_CAVEATS,
        description=(
            "H0's disposition ends on the gap: \"Independent re-implementation still not "
            "attempted.\" Caveat 1 makes the same point its heading - \"Repeatability "
            "plus one cross-hardware replication, not independent reproducibility\" - and "
            "closes \"No independent re-implementation by another investigator.\" What "
            "exists is same-code replication on new hardware (F6), which is a different "
            "claim.")),
    dict(
        id="q-gate-cadence",
        label="Q: What are the true settling iterations between 100 and 120?",
        asserted=D_SERIES_CLOSE, doc_ref=A_CAVEATS,
        description=(
            "Caveat 5: \"Lock-in iterations cluster at 120 because that is the gate's "
            "earliest possible firing; true settling times between 100 and 120 are "
            "unresolved.\" FINDINGS section 6 files it as the series' one remaining "
            "declared debt - \"One item remains open: finer convergence-gate cadence "
            "(caveat 5)\" - and bounds it in the same breath: \"It cannot overturn a "
            "principal finding: basin identities stand on the gate regardless of "
            "cadence.\" So it carries no blocks edge: open work that gates nothing.")),
    dict(
        id="q-hook-window-depth",
        label="Q: Does the landscape depend on where the loop is cut (window / depth)?",
        asserted=D_SERIES_CLOSE, doc_ref=A_CAVEATS,
        description=(
            "Caveat 6: \"Hook-position dependence unexplored. All runs cut the loop at "
            "(final-layer resid_post to layer-0 resid_pre). Alternative windows "
            "(including a Pythia-410m depth control, layers 0-11 vs 0-23) are designed "
            "but not run.\" JOURNEY_MAP section 7 carries the row at status \"Designed, "
            "not run\", and FINDINGS section 5 lists \"hook-window/depth dependence "
            "(caveat 6)\" among the open directions.")),
    dict(
        id="q-shape-class-null",
        label="Q: What does a shape-class-matched coherence null show?",
        asserted=D_ACT_II_5, doc_ref=A_F12,
        description=(
            "F12 records the block as a standing methodological rule: \"no cross-model "
            "coherence claim until a shape-class-matched null exists (matching token "
            "length, case, and leading-space status)\". Caveat 10 repeats it, and "
            "FINDINGS section 5 lists \"the shape-class-matched coherence null and its "
            "application to the 125-sweep (F12, caveat 10)\" among the open directions. "
            "Until it exists, the semantic-coherence phenomenon \"remains exclusive to "
            "GPT-2 Small's language regime among the models tested\".")),
    dict(
        id="q-tmix-llm",
        label="Q: What is T_mix_LLM for each basin?",
        asserted=D_SERIES_CLOSE, doc_ref=A_OPEN_QUESTIONS,
        description=(
            "JOURNEY_MAP section 7: \"What is T_mix_LLM for each basin? | Measurable from "
            "existing data | Compute from .pt\". The glossary files the metric itself as "
            "a \"proposed metric\" that was never computed, so the question is open with "
            "its data already on disk - the cheapest item on the open list.")),
    dict(
        id="q-slonski-macro-group",
        label="Q: Are all basins in one Slonski macro-group?",
        asserted=D_SERIES_CLOSE, doc_ref=A_OPEN_QUESTIONS,
        description=(
            "JOURNEY_MAP section 7: \"Are all basins in one Slonski macro-group? | "
            "Untested; the all-warm premise of the prediction was retired 2026-07-11 "
            "(anisotropy artifact) | One Q-vector experiment, on its own terms\". Kept at "
            "exactly the strength the record gives it: the question outlived its own "
            "premise and stands only as a question, with a named one-experiment next "
            "step.")),
    dict(
        id="q-fractal-dimension",
        label="Q: Is the fractal dimension of convergence trajectories basin-specific?",
        asserted=D_SERIES_CLOSE, doc_ref=A_OPEN_QUESTIONS,
        description=(
            "JOURNEY_MAP section 7: \"Is the fractal dimension of convergence "
            "trajectories basin-specific? | Speculative | Requires T_mix first\". The "
            "weakest item on the open list, recorded at that strength: speculative, and "
            "gated on a metric nobody has computed yet.")),
]


def questions():
    out = []
    for q in QUESTIONS:
        out.append(OrderedDict([
            ("id", q["id"]),
            ("label", q["label"]),
            ("type", "question"),
            # A question is `open` while it stands: the status finally means work
            # rather than "a concept with no epistemic verdict".  An answered
            # question takes `retired` plus a `retires` edge from whatever
            # answered it, so the answer stays a followable node rather than a
            # status word.  q-prompt-library is the first to take that path
            # (answered 2026-07-31: the library is restored, issue #24).
            ("status", q.get("status", "open")),
            ("description", q["description"]),
            ("phase", "phase-5"),
            ("asserted", q["asserted"]),
            ("retired", q.get("retired")),
            ("doc_ref", q["doc_ref"]),
        ]))
    ids = [q["id"] for q in out]
    if len(ids) != len(set(ids)):
        raise SystemExit("FATAL: duplicate question id")
    return out


# --------------------------------------------------------------------------
# 6. Runs  (run inventory table parsed; ids/scripts/dates curated)
# --------------------------------------------------------------------------

RUN_ID = {
    "0": "run-0-repeatability-gate",
    "1": "run-1-attractor-dominance",
    "2": "run-2-cross-model-sweeps",
    "3": "run-3-random-noise-null",
    "4": "run-4-deep-convergence",
    "5": "run-5-gated-resweep",
    "6": "run-6-full-distribution-confidence",
    "7": "run-7-coherence-formalization",
    "8": "run-8-divine-motion-audit",
    "9": "run-9-cycle-anatomy",
    "10": "run-10-jlens-pilot",
    "11": "run-11-glitch-alignment",
    "12": "run-12-flip-axis-eigenvalue",
    "13": "run-13-lagk-regate",
    "14": "run-14-jlens-phase-probe",
    "15": "run-15-suppression-test",
    "16": "run-16-eigen-rescore",
    "17": "run-17-matched-nu-noise",
    "Tensor convergence diagnostic": "run-cos-sim-diagnostic",
    "Readout confidence audit": "run-readout-guardrails",
    "All-warm permutation test": "run-all-warm-permutation",
}

# Runs that are analysis over committed artifacts and raw weights, with no
# ATR loop executed. The generic run templates must not call these ATR runs.
ANALYSIS_ONLY_RUNS = {"run-16-eigen-rescore"}

RUN_SCRIPT = {
    "run-0-repeatability-gate": "experiments/gpt2_small/00_reproducibility_gate.ipynb",
    "run-1-attractor-dominance": "experiments/gpt2_small/01_attractor_dominance.ipynb",
    "run-2-cross-model-sweeps": "experiments/gpt2_medium/01_attractor_dominance.ipynb",
    "run-3-random-noise-null": "experiments/gpt2_small/03_random_baseline.ipynb",
    "run-4-deep-convergence": "experiments/pythia_410m/01b_deep_convergence.ipynb",
    "run-5-gated-resweep": "experiments/gpt2_small/gated_resweep.py",
    "run-6-full-distribution-confidence": "experiments/gpt2_small/04_readout_confidence.py",
    "run-7-coherence-formalization": None,   # no separate script named in the record
    "run-8-divine-motion-audit": "experiments/gpt2_small/05_divine_motion.py",
    "run-9-cycle-anatomy": "experiments/gpt2_small/06_bell_anatomy.py",
    "run-10-jlens-pilot": "experiments/gpt2_small/05_jlens_pilot.py",
    "run-11-glitch-alignment": "experiments/gpt2_small/07_glitch_alignment.py",
    "run-12-flip-axis-eigenvalue": "experiments/gpt2_small/08_hinge_eigenvalue.py",
    "run-13-lagk-regate": "experiments/gpt2_small/09_lagk_gate.py",
    "run-14-jlens-phase-probe": "experiments/gpt2_small/10_jlens_phase.py",
    "run-15-suppression-test": "experiments/gpt2_small/11_suppression_test.py",
    "run-16-eigen-rescore": "experiments/gpt2_small/12_eigen_rescore.py",
    "run-17-matched-nu-noise": "experiments/noise_rerun/01_matched_nu_noise_baseline.py",
    "run-cos-sim-diagnostic": "experiments/cos_sim_diagnostic.ipynb",
    "run-readout-guardrails": "experiments/readout_guardrails.ipynb",
    "run-all-warm-permutation": "experiments/gpt2_small/02b_permutation_test.py",
}

RUN_OUTPUT_DIR = {
    "run-1-attractor-dominance": "experiments/gpt2_small/output/",
    "run-3-random-noise-null": "experiments/gpt2_small/output_random_baseline/",
    "run-4-deep-convergence": "experiments/pythia_410m/output_deep/",
    "run-5-gated-resweep": "experiments/gpt2_small/output_gated/",
    "run-6-full-distribution-confidence": "experiments/gpt2_small/output_confidence/",
    "run-7-coherence-formalization": "experiments/gpt2_small/output_confidence/",
    "run-8-divine-motion-audit": "experiments/gpt2_small/output_divine_motion/",
    "run-9-cycle-anatomy": "experiments/gpt2_small/output_divine_motion/",
    "run-10-jlens-pilot": "experiments/gpt2_small/output_jlens_pilot/",
    "run-11-glitch-alignment": "experiments/gpt2_small/output_glitch/",
    "run-12-flip-axis-eigenvalue": "experiments/gpt2_small/output_hinge_eigen/",
    "run-13-lagk-regate": "experiments/gpt2_small/output_lagk/",
    "run-14-jlens-phase-probe": "experiments/gpt2_small/output_jlens_phase/",
    "run-15-suppression-test": "experiments/gpt2_small/output_suppression/",
    "run-16-eigen-rescore": "experiments/gpt2_small/output_eigen_rescore/",
    "run-17-matched-nu-noise": "experiments/noise_rerun/output/",
    "run-readout-guardrails": "experiments/output/",
    "run-all-warm-permutation": "experiments/gpt2_small/output_permutation/",
}

# Execution dates, from FINDINGS "Provenance", RESULTS_SUMMARY, and the
# individual reports' own Date lines.  Runs 0-1 predate the validation series
# (FINDINGS: "Original exploratory work: 2026-03") and carry the month anchor.
RUN_DATE = {
    "run-0-repeatability-gate": D_EXPLORATORY,
    "run-1-attractor-dominance": D_EXPLORATORY,
    "run-2-cross-model-sweeps": D_SERIES_CLOSE,
    "run-3-random-noise-null": D_SERIES_CLOSE,
    "run-4-deep-convergence": D_SERIES_CLOSE,
    "run-5-gated-resweep": D_SERIES_CLOSE,
    "run-6-full-distribution-confidence": D_ACT_II_5,
    "run-7-coherence-formalization": D_ACT_II_5,
    "run-8-divine-motion-audit": D_ACT_II_5,
    "run-9-cycle-anatomy": D_ACT_II_5,
    "run-10-jlens-pilot": D_ACT_II_5,
    "run-11-glitch-alignment": D_ACT_II_5,
    "run-12-flip-axis-eigenvalue": D_ACT_II_5,
    "run-13-lagk-regate": D_ACT_II_5,
    "run-14-jlens-phase-probe": D_ACT_II_5,
    "run-15-suppression-test": D_ACT_II_5,
    "run-16-eigen-rescore": D_RESCORE,
    "run-17-matched-nu-noise": D_RESCORE,
    "run-cos-sim-diagnostic": D_SERIES_CLOSE,
    "run-readout-guardrails": D_SERIES_CLOSE,
    "run-all-warm-permutation": D_PERMUTATION,
}

RUN_PHASE = {
    "run-0-repeatability-gate": "phase-3",
    "run-1-attractor-dominance": "phase-3",
}

MODEL_KEYS = [
    ("gpt2-small", "model-gpt2-small"),
    ("gpt2-medium", "model-gpt2-medium"),
    ("pythia-160m", "model-pythia-160m"),
    ("pythia-410m", "model-pythia-410m"),
]


def parse_runs(findings_md: str):
    block = section(findings_md, r"^## 1\. Run inventory", 2)
    rows = parse_table(block)
    runs = []
    run_models = {}
    for row in rows:
        num = row["#"].strip()
        name = demark(row["Run"])
        key = num if num != "-" else name
        if key not in RUN_ID:
            raise SystemExit("FATAL: unmapped run inventory row %r / %r" % (num, name))
        rid = RUN_ID[key]

        models_cell = demark(row["Model(s)"]).lower()
        if "all four" in models_cell:
            mids = [m for _, m in MODEL_KEYS]
        else:
            mids = [m for k, m in MODEL_KEYS if k in models_cell]
        if not mids:
            raise SystemExit("FATAL: no model resolved for run %s (%r)" % (rid, models_cell))
        run_models[rid] = mids

        label = ("Run %s: %s" % (num, name)) if num != "-" else name
        out_cell = row["Output"]
        out_path = None
        m = re.search(r"`([^`]+)`", out_cell)
        if m and "<" not in m.group(1):
            out_path = m.group(1)

        entry = OrderedDict([
            ("id", rid),
            ("label", label),
            ("type", "run"),
            ("description", (
                "%s: analysis over %s, on %s; no forward passes, no ATR loop."
                if rid in ANALYSIS_ONLY_RUNS else
                "%s ATR run over %s, on %s.") % (
                name, demark(row["N"]), demark(row["Model(s)"]))),
        ])
        if RUN_SCRIPT.get(rid):
            entry["script"] = RUN_SCRIPT[rid]
        if RUN_OUTPUT_DIR.get(rid):
            entry["output_dir"] = RUN_OUTPUT_DIR[rid]
        elif out_path and out_path.endswith("/"):
            entry["output_dir"] = out_path
        if out_path and not out_path.endswith("/") and out_path != RUN_SCRIPT.get(rid):
            entry["output_path"] = out_path
        entry["n"] = demark(row["N"])
        entry["date"] = RUN_DATE[rid]
        entry["phase"] = RUN_PHASE.get(rid, "phase-5")
        entry["doc_ref"] = "docs/FINDINGS.md#1-run-inventory"
        runs.append(entry)
    if len(runs) != len(RUN_ID):
        raise SystemExit("FATAL: run inventory has %d rows, expected %d"
                         % (len(runs), len(RUN_ID)))
    return runs, run_models


# Runs named outside the FINDINGS inventory table (README notebook table /
# JOURNEY_MAP phases).  Curated, each with a doc_ref.
EXTRA_RUNS = [
    OrderedDict([
        ("id", "run-original-piece"),
        ("label", "EXP_009aFIX: the original five-prompt piece"),
        ("type", "run"),
        ("description", ("The exploratory experiment: five prompts (question, fact, "
                         "nursery grammar, nonsense, command) iterated 500 times through GPT-2 "
                         "Small, decoding the nearest vocabulary token at every step.")),
        ("script", "experiments/gpt2_small/lucier_total_resonance.ipynb"),
        ("n", "5 prompts x 500 iterations"),
        ("date", D_EXPLORATORY),
        ("phase", "phase-1"),
        ("doc_ref", "docs/JOURNEY_MAP.md#1-timeline-the-intellectual-arc"),
    ]),
    OrderedDict([
        ("id", "run-token-neighbourhood"),
        ("label", "Priority Analysis 01: embedding neighbourhood test"),
        ("type", "run"),
        ("description", ("Session 01's W_E analysis of all 14 canonical tokens (5 basin "
                         "tokens plus 9 waypoints): nearest neighbours, the capit correction, the "
                         "structural-to-semantic transition, and the all-warm cross-similarity "
                         "matrix.")),
        ("script", "experiments/gpt2_small/02_token_neighbourhood.ipynb"),
        ("n", "14 tokens (5 basins + 9 waypoints) in W_E"),
        ("date", D_SUPERVISORY),
        ("phase", "phase-4"),
        ("doc_ref", "docs/JOURNEY_MAP.md#1-timeline-the-intellectual-arc"),
    ]),
    OrderedDict([
        ("id", "run-spectral-scaffold"),
        ("label", "Spectral resonance protocol (H4, executed 2026-07-25)"),
        ("type", "run"),
        ("description", ("Pre-registered protocol for H4: per-head resonance against the top "
                         "singular vector of W_OV. Executed end to end 2026-07-25 in the issue "
                         "#25 artifact regeneration (5/144 heads above 0.9, NOT SUPPORTED as "
                         "registered); superseded 2026-07-31 by the corrected-target rescore, "
                         "run 16, per the operator ruling in issue #54.")),
        ("script", "experiments/gpt2_small/spectral_resonance.ipynb"),
        ("n", "144 heads, executed 2026-07-25"),
        ("date", D_REGEN),
        ("phase", "phase-4"),
        ("doc_ref", "docs/FINDINGS.md#3-hypothesis-dispositions"),
    ]),
]

MODELS = [
    OrderedDict([
        ("id", "model-gpt2-small"), ("label", "GPT-2 Small (124M)"), ("type", "model"),
        ("description", ("124M parameters, 12 layers, d_model 768, trained on WebText "
                         "(Reddit-curated outbound links, 2018). The only model in the set that "
                         "resolves language into five semantically coherent basins; partial "
                         "convergence at iteration 100 (cos_sim_mean 0.91), 73% gate-converged by "
                         "iteration 120.")),
        ("n", "125 prompts"), ("date", None), ("phase", "phase-1"),
        ("doc_ref", "docs/FINDINGS.md#f3-the-basin-landscape-does-not-generalise-across-models-fingerprint-hypothesis-refuted"),
    ]),
    OrderedDict([
        ("id", "model-gpt2-medium"), ("label", "GPT-2 Medium (345M)"), ("type", "model"),
        ("description", ("345M parameters, same WebText corpus as GPT-2 Small. Collapses all "
                         "125 prompts to a single basin, the token D, saturating at cos_sim_mean "
                         "1.0000 by iteration 10; its readout is near-flat (entropy 7.93-7.96 "
                         "nats).")),
        ("n", "125 prompts"), ("date", None), ("phase", "phase-5"),
        ("doc_ref", "docs/FINDINGS.md#f3-the-basin-landscape-does-not-generalise-across-models-fingerprint-hypothesis-refuted"),
    ]),
    OrderedDict([
        ("id", "model-pythia-160m"), ("label", "Pythia-160m"), ("type", "model"),
        ("description", ("160M parameters, trained on The Pile. One basin, questioned (94.4%), "
                         "saturating at cos_sim_mean 1.0000 by iteration 10.")),
        ("n", "125 prompts"), ("date", None), ("phase", "phase-5"),
        ("doc_ref", "docs/FINDINGS.md#f3-the-basin-landscape-does-not-generalise-across-models-fingerprint-hypothesis-refuted"),
    ]),
    OrderedDict([
        ("id", "model-pythia-410m"), ("label", "Pythia-410m"), ("type", "model"),
        ("description", ("410M parameters, trained on The Pile. No consolidation: 40+ fragments, "
                         "never converging (~0.85 plateau, 9/125 prompts converge); the 8-prompt "
                         "1000-iteration subset ends at 8 distinct terminals with cross-prompt "
                         "similarity 0.21.")),
        ("n", "125 prompts; 8-prompt deep subset"), ("date", None), ("phase", "phase-5"),
        ("doc_ref", "docs/FINDINGS.md#f3-the-basin-landscape-does-not-generalise-across-models-fingerprint-hypothesis-refuted"),
    ]),
    OrderedDict([
        ("id", "null-model-gaussian-noise"),
        ("label", "Random-noise null model (Gaussian, seed 42)"),
        ("type", "null-model"),
        ("description", ("125 random Gaussian tensors iterated through GPT-2 Small, intended as "
                         "norm- and length-calibrated to the real runs; caveat 18 later found the "
                         "calibration wrong (a per-position statistic applied as the Frobenius "
                         "target) and the 18-basin count read before convergence. Superseded "
                         "2026-07-31 by run 17, the pair-matched, gated re-run, which finds noise "
                         "landing in the language arm's own basins, all five at smallest "
                         "passing lag (F4).")),
        ("n", "125 Gaussian tensors (seed 42); 15 calibrated trials in the confidence audit"),
        ("date", D_SERIES_CLOSE), ("phase", "phase-5"),
        ("doc_ref", "docs/FINDINGS.md#f4-the-five-basins-belong-to-the-language-driven-regime-not-the-weights-in-general-null-model"),
    ]),
]


# --------------------------------------------------------------------------
# 7. Sources: docs, artefacts, prior work
# --------------------------------------------------------------------------

DOCS = [
    ("doc-readme", "README.md: the piece", "README.md",
     "The narrative account: headline results, the basin share table, the four-model "
     "comparison, the notebook quick-reference and the reference list. Where the README "
     "and FINDINGS.md differ, FINDINGS.md governs."),
    ("doc-findings", "FINDINGS.md: canonical record", "docs/FINDINGS.md",
     "The reporting register: run inventory, principal findings F1-F17, hypothesis "
     "dispositions, 16 caveats, what ATR is after the series, and why the series closed "
     "with work unexecuted."),
    ("doc-journey-map", "JOURNEY_MAP.md: the timeline", "docs/JOURNEY_MAP.md",
     "Continuity document: Phases 0-5, the 16 Key Discoveries with their retirements and "
     "corrections, hypothesis status, adjacent science, glossary and open questions."),
    ("doc-technical", "TECHNICAL.md: method specification", "docs/TECHNICAL.md",
     "The formal specification of the loop: extract at the final layer's hook_resid_post, "
     "L2-normalise, re-inject at blocks.0.hook_resid_pre, repeat."),
    ("doc-understanding", "UNDERSTANDING.md: mechanism explained", "docs/UNDERSTANDING.md",
     "The accessible account of what is actually fed back and why the loop has attractors."),
    ("doc-math-primer", "MATH_PRIMER.md", "docs/MATH_PRIMER.md",
     "The mathematics from scratch, tied to the exact places each concept appears in this "
     "repository."),
    ("doc-isomorphism", "ISOMORPHISM.md: Lucier correspondence", "docs/ISOMORPHISM.md",
     "The formal correspondence between Lucier's room (linear power iteration on an acoustic "
     "transfer function) and the transformer loop, and exactly where the analogy breaks."),
    ("doc-scaling-artefact", "SCALING_ARTEFACT_ANALYSIS.md", "docs/SCALING_ARTEFACT_ANALYSIS.md",
     "The artefact-versus-intrinsic attribution: normalisation exonerated, convergence "
     "verdicts tensor-level, readout a secondary jitter source."),
    ("doc-validation-plan", "VALIDATION_PLAN.md (historical)", "docs/VALIDATION_PLAN.md",
     "The pre-registered validation design of March 2026, kept unmodified as a record of what "
     "was predicted before the data arrived."),
    ("doc-method-comparison", "ATR_METHOD_COMPARISON.md", "docs/ATR_METHOD_COMPARISON.md",
     "ATR placed in the mechanistic-interpretability landscape, plus the cross-model scaling "
     "programme it originally proposed, revised at series close."),
    ("doc-jspace-primer", "JSPACE_PRIMER.md", "docs/JSPACE_PRIMER.md",
     "Reading companion for Anthropic's J-space paper and the bridge from it to this "
     "project's open questions."),
    ("doc-jspace-reading-guide", "JSPACE_READING_GUIDE.md", "docs/JSPACE_READING_GUIDE.md",
     "Page-keyed navigation aid for the 133-page J-space PDF."),
    ("doc-prior-work", "PRIOR_WORK.md", "docs/PRIOR_WORK.md",
     "The ATR results placed against the published record, with source-class tags and an "
     "explicit statement of what has no prior occupant."),
    ("doc-bell-primer", "BELL_PRIMER.md", "docs/BELL_PRIMER.md",
     "Plain-language companion to the mechanism series: what was done, what was measured, "
     "what the numbers were, and how firmly each implication is held."),
    ("doc-atr-pause", "ATR_PAUSE.md", "docs/ATR_PAUSE.md",
     "The standing pause: no new ATR experiments run until the understanding gate is passed."),
    ("doc-results-summary", "experiments/RESULTS_SUMMARY.md", "experiments/RESULTS_SUMMARY.md",
     "Run-by-run record of the validation series: environment, deviations from the run plan, "
     "and per-notebook headline numbers."),
    ("doc-cross-model-run-plan", "CROSS_MODEL_RUN_PLAN.md", "CROSS_MODEL_RUN_PLAN.md",
     "The execution plan for the cross-model validation branch."),
]

ARTEFACTS = [
    ("art-convergence-matrix-small", "GPT-2 Small convergence matrix",
     "experiments/gpt2_small/output/convergence_matrix.png",
     "125 prompts x cosine similarity after iteration: five blocks, the five basins.",
     "run-1-attractor-dominance"),
    ("art-convergence-matrix-medium", "GPT-2 Medium convergence matrix",
     "experiments/gpt2_medium/output/convergence_matrix.png",
     "One block: every prompt ends at the token D.", "run-2-cross-model-sweeps"),
    ("art-convergence-matrix-160m", "Pythia-160m convergence matrix",
     "experiments/pythia_160m/output/convergence_matrix.png",
     "One block: 94.4% of prompts end at questioned.", "run-2-cross-model-sweeps"),
    ("art-convergence-matrix-410m", "Pythia-410m convergence matrix",
     "experiments/pythia_410m/output/convergence_matrix.png",
     "No blocks: the room never settles, 40+ terminal fragments.", "run-2-cross-model-sweeps"),
    ("art-basin-distribution-small", "GPT-2 Small basin distribution",
     "experiments/gpt2_small/output/basin_distribution.png",
     "The 125-prompt basin shares as first published, at iteration 100.",
     "run-1-attractor-dominance"),
    ("art-topology-3d-small", "GPT-2 Small 3D PCA topology",
     "experiments/gpt2_small/output/topology_3d.png",
     "PCA trajectory of the 125 prompts dissolving toward their attractors.",
     "run-1-attractor-dominance"),
    ("art-dissolution-pathways-small", "GPT-2 Small dissolution pathways",
     "experiments/gpt2_small/output/dissolution_pathways.md",
     "The decoded waypoint sequences for the 125-prompt sweep.",
     "run-1-attractor-dominance"),
    ("art-stage1-hypothesis-assessment", "Stage 1 hypothesis assessment",
     "experiments/gpt2_small/output/hypothesis_assessment.md",
     "The 125-prompt sweep's own assessment of the pre-registered hypotheses.",
     "run-1-attractor-dominance"),
    ("art-gated-report", "Convergence-gated re-sweep report",
     "experiments/gpt2_small/output_gated/gated_report.md",
     "91/125 lock in, all at iteration 120; the 34 non-convergers are exactly the Divine "
     "prompts; basin shares at lock-in.", "run-5-gated-resweep"),
    ("art-prompt-library", "The restored 125-prompt library",
     "prompt_library.py",
     "Provenance-flagged full restoration of the original 125 prompts (issue #24): "
     "every entry recovered verbatim from committed records, zero re-authored "
     "prompts, per-prompt provenance flags in the module's PROVENANCE dict.", None),
    ("art-random-baseline-report", "Random-baseline (null model) report",
     "experiments/gpt2_small/output_random_baseline/random_baseline_report.md",
     "125 Gaussian trials, seed 42: 18 basins, bootstrap count 14.1 with 95% CI [11, 17].",
     "run-3-random-noise-null"),
    ("art-deep-basin-assessment", "Pythia-410m deep-run basin assessment",
     "experiments/pythia_410m/output_deep/basin_assessment.md",
     "The 8-prompt, 1000-iteration deep run: 8 distinct terminals, cross-prompt similarity 0.21.",
     "run-4-deep-convergence"),
    ("art-readout-guardrails-json", "Readout guardrails (single-prompt demo)",
     "experiments/output/readout_guardrails_gpt2_small.json",
     "Logit margin and entropy as a trajectory settles, on one prompt: the original readout "
     "confidence check.", "run-readout-guardrails"),
    ("art-confidence-report", "Readout confidence audit report",
     "experiments/gpt2_small/output_confidence/confidence_report.md",
     "The full-distribution audit of the five converged states plus 15 noise trials: the "
     "confidence inversion, and Result 0, the cross-hardware replication.",
     "run-6-full-distribution-confidence"),
    ("art-chordness-formal", "Coherence formalized (permutation nulls)",
     "experiments/gpt2_small/output_confidence/chordness_formal.md",
     "Weighted coherence, k-sensitivity, and the uniform and frequency-matched permutation "
     "nulls, for GPT-2 Small and GPT-2 Medium.", "run-7-coherence-formalization"),
    ("art-divine-motion-report", "Divine motion audit report",
     "experiments/gpt2_small/output_divine_motion/divine_motion_report.md",
     "The lag-10 schedule that showed a frozen tensor, and the lag-1 probe that caught the "
     "alternation.", "run-8-divine-motion-audit"),
    ("art-bell-anatomy", "Cycle anatomy report",
     "experiments/gpt2_small/output_divine_motion/bell_anatomy.md",
     "The exact-cycle verification, the flip axis, the two poles, and the 2026-07-23 correction "
     "of the phase-norm contrast as a frame artifact.", "run-9-cycle-anatomy"),
    ("art-jlens-pilot-report", "J-lens pilot report",
     "experiments/gpt2_small/output_jlens_pilot/jlens_pilot_report.md",
     "The restricted 193-token pilot lens over 30 prompts, its span and sparse membership "
     "probes, and its stated limitations.", "run-10-jlens-pilot"),
    ("art-glitch-alignment", "Glitch alignment report",
     "experiments/gpt2_small/output_glitch/glitch_alignment.md",
     "cos(-d, under-trained core) = +0.596 against random and norm-matched nulls, and the "
     "function-word identity of the opposite pole.", "run-11-glitch-alignment"),
    ("art-hinge-eigenvalue", "Flip-axis eigenvalue report",
     "experiments/gpt2_small/output_hinge_eigen/hinge_eigenvalue.md",
     "The jvp measurement of the -4.3 pivot eigenvalue, the +0.10 composed multiplier, the "
     "block-11 localisation and the L11.H8 attribution.", "run-12-flip-axis-eigenvalue"),
    ("art-lagk-report", "Lag-k re-gate report",
     "experiments/gpt2_small/output_lagk/lagk_report.md",
     "The full lag table for the fixed point, the cycle and the noise control, and the "
     "gate_lag engine change.", "run-13-lagk-regate"),
    ("art-jlens-phase", "J-lens phase probe report",
     "experiments/gpt2_small/output_jlens_phase/jlens_phase.md",
     "Both phases, the pivot M and the flip axis against the pilot lens at all 12 layers.",
     "run-14-jlens-phase-probe"),
    ("art-suppression-report", "Suppression-head test report",
     "experiments/gpt2_small/output_suppression/suppression_report.md",
     "The 144-head OV ranking, the in-loop ablation, and the ordinary-text copy test against "
     "the L10.H7 positive control.", "run-15-suppression-test"),
    ("art-permutation-report", "All-warm permutation test report",
     "experiments/gpt2_small/output_permutation/permutation_report.md",
     "10,000 random 14-token sets, seed 1969: 9,994 are also all-positive, so the all-warm "
     "matrix is an anisotropy artifact.", "run-all-warm-permutation"),
    ("art-atr-engine", "atr_engine.py (the loop engine)", "atr_engine.py",
     "The hooks, metrics and gated-run machinery, including run_atr_gated's gate_lag parameter "
     "and the lag_scan helper added by the lag-k re-gate.", "run-13-lagk-regate"),
]

PRIOR_WORK = [
    ("prior-radford-2019", "Radford, Wu et al. (2019), Language Models are Unsupervised "
     "Multitask Learners",
     "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
     "The GPT-2 paper: the source for WebText, 40GB of text scraped from Reddit-curated "
     "outbound links circa 2018, which both GPT-2 Small and GPT-2 Medium were trained on.",
     "doc-readme"),
    ("prior-biderman-2023", "Biderman et al. (2023), Pythia: A Suite for Analyzing Large "
     "Language Models Across Training and Scaling", "https://arxiv.org/abs/2304.01373",
     "The Pythia suite: the source for the two comparison models trained on The Pile.",
     "doc-readme"),
    ("prior-lucier-1969", "Lucier, A. (1969), I Am Sitting in a Room",
     "https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room",
     "The seed: iterative feedback of recorded speech through a room until only the room's "
     "resonant frequencies remain.", "doc-readme"),
    ("prior-nanda-bloom-transformerlens", "Nanda, N. & Bloom, J. (2022), TransformerLens",
     "https://github.com/TransformerLensOrg/TransformerLens",
     "The hook library every ATR run is built on: forward hooks at hook_resid_post and "
     "blocks.0.hook_resid_pre.", "doc-readme"),
    ("prior-anthropic-jspace-2026", "Anthropic (2026), Verbalizable Representations Form a "
     "Global Workspace in Language Models",
     "https://transformer-circuits.pub/2026/workspace/index.html",
     "The J-space paper: verbalizable states occupy a distinguished subspace, probed by a lens "
     "built from averaged Jacobians. The source of hypothesis H-J1.", "doc-readme"),
    ("prior-rumbelow-watkins-2023", "Rumbelow, J. & Watkins, M. (2023), SolidGoldMagikarp I-III",
     "https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation",
     "The anomalous-token discovery posts: tokens closest to the embedding centroid behave "
     "anomalously; the provenance is a tokenizer-corpus mismatch.", "doc-prior-work"),
    ("prior-land-bartolo-2024", "Land, S. & Bartolo, M. (2024), Fishing for Magikarp (EMNLP)",
     "https://arxiv.org/abs/2405.05417",
     "Systematic undertrained-token detection across 23 models; the peer-reviewed replication "
     "the anomalous-token characterisation rests on.", "doc-prior-work"),
    ("prior-mcdougall-2023", "McDougall, Conmy, Rushing, McGrath, Nanda (2023), Copy Suppression",
     "https://arxiv.org/abs/2310.04625",
     "The copy-suppression head class: GPT-2 Small's L10.H7 detects the currently predicted "
     "token and writes against its unembedding. The class H-supp proposed L11.H8 belonged to.",
     "doc-prior-work"),
    ("prior-elhage-2021", "Elhage et al. (2021), A Mathematical Framework for Transformer Circuits",
     "https://transformer-circuits.pub/2021/framework/index.html",
     "QK and OV circuit decomposition, and copying scored by positive real eigenvalues of the "
     "full OV circuit: the formalism the -4.3 flip-axis measurement is stated in.",
     "doc-prior-work"),
    ("prior-nostalgebraist-2020", "nostalgebraist (2020), the logit lens",
     "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens",
     "Decoding intermediate residual states through the final LayerNorm and W_U; its structural "
     "failure mode is that it reads only components aligned with W_U's strong directions.",
     "doc-prior-work"),
    ("prior-wang-2025", "Wang, Li, Yan, Cheng, Zhang (2025), Unveiling Attractor Cycles in Large "
     "Language Models (ACL)", "https://arxiv.org/abs/2502.15208",
     "The established nearest neighbour: iterated paraphrasing through the text interface "
     "converges to stable period-2 cycles, measured with a lag-2 metric. Approximate, "
     "text-level, and with no internal measurement.", "doc-prior-work"),
]


def build_sources(adjacent_priors):
    sources = []
    for sid, title, path, desc in DOCS:
        sources.append(OrderedDict([
            ("id", sid), ("title", title), ("type", "doc"),
            ("path", path), ("description", desc),
        ]))
    for aid, title, path, desc, _run in ARTEFACTS:
        sources.append(OrderedDict([
            ("id", aid), ("title", title), ("type", "artefact"),
            ("path", path), ("description", desc),
        ]))
    for pid, title, url, desc, cited_by in PRIOR_WORK:
        sources.append(OrderedDict([
            ("id", pid), ("title", title), ("type", "prior-work"),
            ("path", url), ("description", desc), ("cited_by", cited_by),
        ]))
    for entry in adjacent_priors:
        e = OrderedDict(entry)
        e.pop("doc_ref", None)
        sources.append(e)
    return sources


# --------------------------------------------------------------------------
# 8. Relationships
# --------------------------------------------------------------------------

def rel(frm, to, typ, description, weight=3, asserted=None):
    r = OrderedDict([("from", frm), ("to", to), ("type", typ),
                     ("description", description), ("weight", weight)])
    if asserted:
        r["asserted"] = asserted
    return r


F1, F2, F3, F4, F5 = (FINDING_ID["F%d" % i] for i in range(1, 6))
F6, F7, F8, F9, F10 = (FINDING_ID["F%d" % i] for i in range(6, 11))
F11, F12, F13, F14, F15 = (FINDING_ID["F%d" % i] for i in range(11, 16))
F16, F17 = FINDING_ID["F16"], FINDING_ID["F17"]

H0, H1, H2, H3, H4 = (HYP_ID["H%d" % i] for i in range(5))
HFP = HYP_ID["H-fingerprint"]
HTILL = HYP_ID["H-till"]
HD1 = HYP_ID["H-D1"]
HJ1 = HYP_ID["H-J1"]
HGL = HYP_ID["H-glitch"]
HFLIP = HYP_ID["H-flip"]
HSUPP = HYP_ID["H-supp"]

D = DISC_ID


def curated_relationships():
    R = []

    # ---- findings -> hypotheses ------------------------------------------
    R += [
        rel(F1, H1, "supports",
            "The gated re-sweep puts prolet at 43.2% of 125 prompts at lock-in, up from the "
            "35.2% recorded at iteration 100, so the dominance claim survives proper "
            "convergence and strengthens.", 8, D_SERIES_CLOSE),
        rel(F1, HTILL, "refutes",
            "The pre-registered slow-transient reading of till fails: 19 of 19 till prompts "
            "converge under the gate and retain their label. What was still drifting at "
            "iteration 100 was Anarch, about 10 of whose prompts move on to prolet; Anarch "
            "itself survives the gate as a basin at 13.6%.",
            9, D_SERIES_CLOSE),
        rel(F1, H2, "supports",
            "Divine holds 27.2% of prompts at lock-in, unchanged from the iteration-100 count, "
            "so it is a genuine second basin and not a stop-time artefact.", 6, D_SERIES_CLOSE),
        rel(F2, H2, "supports",
            "All 34 Divine prompts hold the same decoded top-1 token throughout, which is what "
            "a genuine secondary basin looks like at the readout.", 6, D_SERIES_CLOSE),
        rel(F3, HFP, "refutes",
            "GPT-2 Medium shares GPT-2 Small's training corpus and produces no semantic basins "
            "at all, so basin profiles cannot be read as a thematic fingerprint of the corpus "
            "from any model.", 10, D_SERIES_CLOSE),
        rel(F4, HFP, "refutes",
            "The refutation survives F4's inversion with its sign flipped. Originally: noise "
            "found 18 disjoint basins, so the five seemed input-specific. Run 17's matched-nu "
            "control: noise finds all five (97/125 trials at smallest passing lag), so at this "
            "injection scale the basins do not require language-shaped input; the noise still "
            "passes through the trained weights, so what refutes the corpus reading is F3's "
            "cross-model table (same corpus, no shared basins), which run 17 leaves untouched. "
            "Either way, not a fingerprint.", 9, D_RESCORE),
        rel(F3, H3, "qualifies",
            "The corpus-causal half of H3 fails cross-model: same corpus, different landscape. "
            "Only the embedding-space clustering observation survives.", 8, D_SERIES_CLOSE),
        rel(F4, H3, "qualifies",
            "Inverted 2026-07-31 (run 17): the matched-nu control finds all five basins "
            "under noise (97/125 trials at smallest passing lag), so any corpus topology H3 "
            "reads is a property of the weights "
            "at this injection scale, not of language-driven input.", 6, D_RESCORE),
        rel(F8, H3, "supports",
            "The semantic-coherence half of H3 is upgraded: it now holds in the full readout "
            "distribution with permutation support (0.41-0.47 against 0.27, p = 0.001 under "
            "both nulls), not just as a qualitative W_E neighbourhood.", 7, D_ACT_II_5),
        rel(F6, H0, "supports",
            "Three identical repeats on a fresh cloud container of a different machine class "
            "reproduce the terminal attractors and the intermediate waypoints exactly, "
            "extending determinism beyond the original same-machine N=2.", 8, D_ACT_II_5),
        rel(F9, H2, "supports",
            "The second basin is resolved as an object: an exact period-2 limit cycle with a "
            "phase-invariant argmax, so H2's basin is real but is a cycle rather than a fixed "
            "point.", 8, D_ACT_II_5),
        rel(F9, HD1, "supports",
            "The late-stage motion does sit mostly in readout-flattened directions: the "
            "per-step readout response is 0.295 of an equal-norm random baseline.",
            7, D_ACT_II_5),
        rel(F10, HD1, "qualifies",
            "The flip axis itself responds at 0.054, more suppressed still, but the "
            "distribution beneath the argmax visibly shifts (p(top-1) 0.505 to 0.225), so "
            "'readout-flattened' holds in a weakened, more precise form.", 7, D_ACT_II_5),
        rel(F11, HJ1, "refutes",
            "The pilot's point estimate runs the other way: the Divine state is at least as "
            "lens-expressible as the prolet attractor, on higher span share at every layer and "
            "higher sparse share at 11 of 12, margins 0.01-0.02 absolute. Not supported, at "
            "pilot confidence only.", 4, D_ACT_II_5),
        rel(F16, HJ1, "qualifies",
            "The phase-aware re-probe splits the prediction rather than settling it: the "
            "reversal holds for phase A, strengthens at the pivot M, and reverses for phase B.",
            7, D_ACT_II_5),
        rel(F13, HGL, "supports",
            "The alignment is measured, not inspected: cos(-d, under-trained core) = +0.596, "
            "p < 0.001 under both random and norm-matched nulls, with 45 of the top 50 tokens "
            "along -d inside the 0.1% geometric core.", 9, D_ACT_II_5),
        rel(F14, HFLIP, "supports",
            "The inversion is real, direction-specific and localisable exactly as H-flip "
            "predicted: one direction, one block (11), one head (L11.H8, 99.1% of the "
            "attention flip).", 8, D_ACT_II_5),
        rel(F14, HFLIP, "corrects",
            "The magnitude in H-flip is wrong: the pivot eigenvalue is -4.3, an overshooting "
            "reflection, not the conjectured -1. The literal -1 appears only for the "
            "frame-mixed committed axis (-0.864).", 9, D_ACT_II_5),
        rel(F17, HSUPP, "refutes",
            "L11.H8 raises the attended token's logit at 91.4% of positions on ordinary text, "
            "the opposite sign to copy suppression, while the L10.H7 positive control shows "
            "the documented behaviour. The learned-function reading is unsupported.",
            10, D_ACT_II_5),
        rel(F14, HSUPP, "supports",
            "The first two clauses of H-supp survive: L11.H8 does invert the flip axis and is "
            "the head that executes the cycle's inversion.", 5, D_ACT_II_5),
        rel(F2, HD1, "supports",
            "The original observation behind H-D1: the tensor moves while the decoded top-1 "
            "token does not, so the motion must lie in directions the readout flattens.",
            5, D_SERIES_CLOSE),
        rel(F5, H0, "supports",
            "The convergence verdicts are tensor-level and never pass through token decoding, "
            "so the reported terminal states are not decoding coincidences.", 4, D_SERIES_CLOSE),
    ]

    # ---- findings -> findings --------------------------------------------
    R += [
        rel(F9, F2, "corrects",
            "The anomaly is resolved: the Divine tensor is an exact period-2 limit cycle, so "
            "'34 prompts never converge' over-claims and must be read as '34 prompts cycle, "
            "pending re-gate'. The dissociation survives, but as aliasing plus periodicity, "
            "not as endless wandering.", 10, D_ACT_II_5),
        rel(F9, F7, "corrects",
            "F7's headline Divine readout (p = 0.505, entropy 3.05 nats) is phase A only; "
            "phase B decodes the same argmax at p = 0.2252 with entropy 4.62. The "
            "stable-argmax story survives, the stable-distribution story does not.",
            8, D_ACT_II_5),
        rel(F7, F5, "corrects",
            "F5's third clause read the settled basins as high-confidence from a single-prompt "
            "audit; the full five-state audit shows p(top-1) 0.064-0.086. Superseded in part.",
            8, D_ACT_II_5),
        rel(F14, F10, "corrects",
            "Two corrections to the cycle anatomy: the flip-axis eigenvalue is -4.3, not the "
            "conjectured -1, and the phase A / phase B norm contrast (1612 vs 464) is a "
            "frame-mixing artifact of how 06_bell_anatomy.py built the states, not an energy "
            "redistribution across positions.", 9, D_POST_CLOSE),
        rel(F16, F11, "corrects",
            "The pilot verdict was phase-blind: it read phase A and reported one number for "
            "the cycle. The phases are materially distinguishable to the lens and straddle the "
            "prolet level, so the single comparison conflated two different objects.",
            7, D_ACT_II_5),
        rel(F8, F1, "qualifies",
            "Anarch is the rank-3 token inside the prolet states' top-10, so prolet and Anarch "
            "are two argmax peaks over one shared distribution-level structure: counted by "
            "distinct structures rather than distinct winners, the landscape holds fewer than "
            "five objects.", 8, D_ACT_II_5),
        rel(F7, F1, "qualifies",
            "The settled basins' argmax confidence is low (p(top-1) 0.064-0.086), so the basin "
            "labels are carried by a coherent distribution rather than a confident winner.",
            5, D_ACT_II_5),
        rel(F4, F1, "qualifies",
            "Inverted 2026-07-31 (run 17): F1's basins are not regime-specific after all; noise "
            "at matched injection scale finds all five, so the landscape belongs to the "
            "weights at this scale, with scale-dependence the registered open question "
            "(nu-sweep).", 6, D_RESCORE),
        rel(F12, F8, "qualifies",
            "Coherence measures embedding-space clustering of any kind: GPT-2 Medium's D state "
            "passes at p = 0.001 on purely typographic grounds, so no cross-model coherence "
            "claim stands until a shape-class-matched null exists.", 7, D_ACT_II_5),
        rel(F5, F3, "supports",
            "The cross-model landscape differences are tensor-level and survive the "
            "normalisation and readout controls, so F3's table is about the models, not the "
            "apparatus.", 8, D_SERIES_CLOSE),
        rel(F12, F3, "supports",
            "The Medium re-run on new hardware confirms F3's single-basin picture and adds the "
            "distribution view: a near-flat readout, effective support ~2,800 tokens.",
            6, D_ACT_II_5),
        rel(F11, F4, "supports",
            "A second instrument saw the same regime boundary: a J-lens read converged noise "
            "as less J-space-like than converged language states. Its support attached to the "
            "ORIGINAL reading, and its noise leg inherited the caveat-18 mis-calibration, so "
            "it awaits re-derivation against the run-17 states.", 6, D_ACT_II_5),
        rel(F16, F10, "supports",
            "F10's readout-invisibility restated in the lens frame: the physical flip axis "
            "holds span share 0.013 at L11 against a 0.252 chance level, 97.0% of its energy "
            "outside the lens.", 6, D_ACT_II_5),
        rel(F15, F9, "supports",
            "The parity signature confirms the cycle independently: the Divine state fails "
            "every odd lag at 0.6849 and passes every even lag at 1.0000000.", 8, D_ACT_II_5),
        rel(F13, F10, "supports",
            "F10 identified the phase-B pole with the anomalous-token cluster by inspection "
            "against published lists; F13 measures it against random and norm-matched nulls "
            "and it holds.", 8, D_ACT_II_5),
        rel(F17, F14, "supports",
            "The head attribution is independently confirmed: among all 144 heads L11.H8's OV "
            "inverts d_sym most strongly (rank 1, per-unit d-component -61 against the "
            "runner-up's -1.2), and ablating it collapses the cycle.", 8, D_ACT_II_5),
        rel(F9, F1, "qualifies",
            "The 34 prompts F1 counts as unconverged are cycling, not drifting, so the gated "
            "sweep's convergence rate is a property of the lag-1 gate as much as of the "
            "prompts.", 6, D_ACT_II_5),
        # builds-on chain (associative)
        rel(F9, F2, "builds-on",
            "The motion audit was commissioned specifically to answer F2's open question about "
            "where the never-settling tensor goes.", 5, D_ACT_II_5),
        rel(F10, F9, "builds-on",
            "Having established that the cycle exists, the anatomy run dissects it into pivot "
            "and flip axis from the saved iteration-1000 checkpoint.", 5, D_ACT_II_5),
        rel(F13, F10, "builds-on",
            "F13 takes F10's by-inspection glitch identification and turns it into a measured "
            "alignment with nulls.", 4, D_ACT_II_5),
        rel(F14, F10, "builds-on",
            "F14 takes F10's conjecture that the flip axis carries an eigenvalue near -1 and "
            "measures the linearised map.", 4, D_ACT_II_5),
        rel(F15, F9, "builds-on",
            "F15 implements the re-gate that F9 named as the standing correction to the "
            "convergence claim.", 4, D_ACT_II_5),
        rel(F16, F11, "builds-on",
            "F16 re-runs the same restricted pilot lens, inheriting every one of its "
            "limitations, on both phases plus the pivot and the flip axis.", 4, D_ACT_II_5),
        rel(F17, F14, "builds-on",
            "F17 tests the suppression reading that F14's head attribution suggested.",
            4, D_ACT_II_5),
        rel(F8, F7, "builds-on",
            "Having found that confidence does not separate the families, the coherence "
            "measure was formalised on the same distributions to find something that does.",
            4, D_ACT_II_5),
        rel(F7, F5, "builds-on",
            "F7 is the full five-state version of the single-prompt readout audit F5's third "
            "clause rested on.", 4, D_ACT_II_5),
    ]

    # ---- findings / runs -> discoveries -----------------------------------
    R += [
        rel(F1, D["5"], "corrects",
            "The convergence-gated re-sweep corrects this row's iteration-100 shares: Anarch "
            "was over-counted at 20.8% and falls to 13.6% (~10 prompts still drifting to "
            "prolet), prolet rises 35.2% to 43.2%, and solidarity falls 1.6% to 0.8%.",
            9, D_SERIES_CLOSE),
        rel("run-5-gated-resweep", "run-1-attractor-dominance", "corrects",
            "The re-sweep re-classified run 1's basin table at lock-in instead of at iteration "
            "100, moving about 10 prompts from Anarch to prolet and 1 from solidarity to "
            "Anarch.", 8, D_SERIES_CLOSE),
        rel("run-all-warm-permutation", D["9"], "retires",
            "10,000 random 14-token sets, seed 1969: 9,994 are also all-positive and the "
            "global mean pairwise cosine is 0.268 against the observed set's 0.288. The "
            "all-warm matrix is embedding-space anisotropy, not a special compact subspace.",
            10, D_PERMUTATION),
        rel("run-all-warm-permutation", D["10"], "retires",
            "The thematic-centre-of-mass reading rested on the all-warm property, which the "
            "permutation test shows is an artifact (S2 p = 0.167, S3 p = 0.099). Only the "
            "recorded cosine values stand.", 10, D_PERMUTATION),
        rel(F3, D["10"], "retires",
            "The other leg of the same interpretation, the corpus-causal reading, was refuted "
            "cross-model at series close: same corpus, different landscape.", 8, D_SERIES_CLOSE),
        rel("concept-brouwer-fixed-point", D["11"], "corrects",
            "Post-close correction of 2026-07-23: Brouwer's theorem requires a compact convex "
            "domain and the L2 shell is a sphere, not convex, so the theorem is inapplicable "
            "as stated. Attractor existence here is an empirical observation, not a guarantee.",
            9, D_POST_CLOSE),
        rel(F9, D["15"], "corrects",
            "The readout-stable / tensor-unsettled object is identified: an exact period-2 "
            "limit cycle that fails the lag-1 gate by construction, converged at lag 2 for the "
            "audited trajectory.", 9, D_ACT_II_5),
        rel(D["12"], D["3"], "qualifies",
            "The Reddit-2018 reading of the terminal tokens does not survive the cross-model "
            "table: GPT-2 Medium heard the same corpus and produces one empty token.",
            7, D_SERIES_CLOSE),
        rel("run-all-warm-permutation", D["6"], "qualifies",
            "The test's scope is the all-warm cross-similarity matrix, not the local "
            "neighbourhoods; the semantic-clustering observation survives it but stands as "
            "qualitative only.", 6, D_PERMUTATION),
        rel(F8, D["6"], "supports",
            "The clustering claim is re-established one level deeper, in the readout "
            "distribution itself, with permutation support under two nulls.", 7, D_ACT_II_5),
        rel(F6, D["4"], "supports",
            "Reproducibility extends from N=2 same-machine runs to three repeats on a "
            "different machine class with mirror-sourced weights, terminal attractors and "
            "waypoints identical.", 7, D_ACT_II_5),
        rel(F1, D["14"], "supports",
            "The gated re-sweep is the evidence for this row: 73% of prompts pass the lag-1 "
            "gate by iteration 120 and keep their label.", 7, D_SERIES_CLOSE),
        rel(F3, D["12"], "supports",
            "F3's four-model table is the finding this discovery records.", 7, D_SERIES_CLOSE),
        rel(F4, D["13"], "corrects",
            "F4's null model was the evidence this discovery recorded (18 non-semantic "
            "basins, ~zero overlap), and F4's own repair overturned it: run 17's matched-nu, "
            "gated re-run sends noise into the language arm's own five basins, so the "
            "regime-specific reading this discovery states is inverted at the tested scale.",
            7, D_RESCORE),
        rel(F5, D["16"], "supports",
            "F5's three attribution results are what this discovery records.", 7, D_SERIES_CLOSE),
        rel(F2, D["15"], "supports",
            "F2 is the finding this discovery records.", 6, D_SERIES_CLOSE),
        rel(F1, D["1"], "supports",
            "The 125-prompt sweep confirms at scale what five hand-picked prompts suggested: "
            "iterated re-injection produces discrete attractor basins.", 6, D_SERIES_CLOSE),
        rel(D["5"], D["1"], "builds-on",
            "Scaling from five prompts to 125 turned the two observed basins into five.",
            5, D_EXPLORATORY),
        rel(D["9"], D["6"], "builds-on",
            "The all-warm matrix was read as the global counterpart of the per-token "
            "neighbourhood clustering.", 4, D_SUPERVISORY),
        rel(D["10"], D["9"], "builds-on",
            "The thematic-centre-of-mass interpretation was the reading placed on the all-warm "
            "property.", 5, D_SUPERVISORY),
        rel(D["8"], D["2"], "builds-on",
            "The structural-to-semantic transition is a reading of the shared dissolution "
            "pathway's token sequence.", 4, D_SUPERVISORY),
        rel(D["7"], D["2"], "qualifies",
            "The waypoint capit clusters as capitulation, not capitalism, which changes what "
            "the pathway's late-stage tokens are taken to mean.", 5, D_SUPERVISORY),
    ]

    # ---- findings / runs -> concepts --------------------------------------
    R += [
        rel(F15, "concept-lag1-convergence-gate", "supersedes",
            "The engine now takes a gate_lag parameter: gating at the state's own period "
            "classifies the cycle as converged, where the lag-1 gate could never pass it. The "
            "recommended re-gate runs the full lag table and gates at the smallest passing "
            "lag.", 9, D_ACT_II_5),
        rel("concept-lag1-convergence-gate", "concept-period-2-limit-cycle", "breaks-down-at",
            "On a period-2 cycle consecutive iterates always differ by the full swing (cosine "
            "0.6849 here), so a lag-1 gate fails the object by construction, whatever its "
            "threshold.", 8, D_ACT_II_5),
        rel("concept-logit-lens", "concept-flip-axis", "breaks-down-at",
            "The flip axis produces a logit response of 33 against 612 for equal-norm random "
            "directions, so a logit-lens readout registers almost none of the cycle's motion.",
            7, D_ACT_II_5),
        rel(F9, "concept-period-2-limit-cycle", "supports",
            "cos(A, f(f(A))) = 1.000000, verified over 20 iterations from the committed "
            "iteration-1000 state: the object is exactly periodic, not approximately so.",
            9, D_ACT_II_5),
        rel(F9, "concept-aliasing", "supports",
            "Every schedule from lock-in onward sampled even iterations only, which records a "
            "period-2 orbit at a single phase; the oscillation was invisible by construction.",
            8, D_ACT_II_5),
        rel(F10, "concept-flip-axis", "supports",
            "The per-position flip axes agree at mean pairwise cosine 1.0000, so the whole "
            "tensor inverts along a single global rank-1 direction.", 8, D_ACT_II_5),
        rel(F9, "concept-phase-a-b", "supports",
            "The lag-1 probe alternates L2 distance 1249.43 / 0.000 and cosine 0.6849 / 1.0000, "
            "which is what two alternating states look like.", 7, D_ACT_II_5),
        rel(F8, "concept-coherence", "supports",
            "Coherence is defined and then tested against two permutation nulls of 1000 draws, "
            "separating the language attractors from noise as a strong statistical "
            "regularity.", 8, D_ACT_II_5),
        rel(F12, "concept-coherence", "qualifies",
            "Coherence is blind to the cause of clustering: GPT-2 Medium's typographic cluster "
            "passes at p = 0.001, so the measure needs a shape-class-matched null before it "
            "travels across models.", 7, D_ACT_II_5),
        rel(F9, "concept-invisibility-ratio", "supports",
            "The measure is defined and calibrated here: 0.295 for the cycle step, against a "
            "prolet control at the numerical floor and a noise control slightly amplified at "
            "1.12.", 6, D_ACT_II_5),
        rel(F3, "concept-bias-profile", "retires",
            "The term is retired with the hypothesis it named: basin distribution as a "
            "fingerprint of training-data themes does not survive the cross-model table.",
            9, D_SERIES_CLOSE),
        rel(F4, "concept-eigenvoice", "corrects",
            "The correction has itself been corrected. First: there is no single native voice, "
            "and what the loop settles into depends on what drove it. Run 17 inverted the "
            "second half: converged noise lands in four of the language arm's five basins "
            "(and the period-2 trials decode to the fifth, Divine, itself), so at this "
            "injection scale the weights do carry native voices, several rather than the "
            "metaphor's one.", 7, D_RESCORE),
        rel(F13, "concept-glitch-token", "corrects",
            "The Session 01 glossary note ruled glitch tokens out for the basins; the Divine "
            "cycle's phase-B pole is measured at cos +0.596 to the under-trained core, so the "
            "anomalous-token region is implicated after all, in the dynamics rather than the "
            "basin identities.", 8, D_ACT_II_5),
        rel("run-all-warm-permutation", "concept-all-warm-matrix", "retires",
            "The property the term names is generic to the embedding space: 9,994 of 10,000 "
            "random 14-token sets are also all-warm, so it indicates nothing about the "
            "attractors.", 9, D_PERMUTATION),
        rel(F1, "concept-attractor-basin", "supports",
            "Five basins, each classified at lock-in and each retaining its members under the "
            "convergence gate.", 6, D_SERIES_CLOSE),
        rel(F4, "concept-attractor-basin", "qualifies",
            "Basins exist for noise inputs too, and at matched injection scale they are "
            "largely the same ones (run 17): the notion is weight-native at this scale, with "
            "scale-dependence the open question (nu-sweep).", 6, D_RESCORE),
        rel(F1, "concept-byte-pair-encoding", "relates-to",
            "Basin identities are single BPE tokens (prolet, Anarch), so any multi-token "
            "structure behind them is invisible to the current readout (caveat 8).",
            4, D_SERIES_CLOSE),
        rel(F2, "concept-logit-lens", "relates-to",
            "The readout is logit-lens-style, applying ln_final and W_U to intermediate states, "
            "and F2 is the case where the decode and the dynamics disagree (caveat 9).",
            5, D_SERIES_CLOSE),
        rel(F11, "concept-jspace-workspace", "tests",
            "The pilot builds a restricted J-lens (193 tokens, 30 prompts, all 12 layers) and "
            "probes membership by least-squares span share and nonnegative sparse share.",
            6, D_ACT_II_5),
        rel(F16, "concept-jspace-workspace", "tests",
            "The phase-aware re-probe puts both phases, the pivot and the flip axis against the "
            "same lens, and finds the pivot the most lens-expressible object in the system.",
            6, D_ACT_II_5),
        rel(F17, "concept-copy-suppression", "relates-to",
            "The suppression protocol detects the documented class where it exists (L10.H7: "
            "87.1% negative, mean -3.62) and finds the opposite sign for L11.H8.",
            6, D_ACT_II_5),
        rel(HSUPP, "concept-copy-suppression", "builds-on",
            "The hypothesis read L11.H8 as an instance of the documented copy-suppression "
            "class, with the closed loop recycling its one-shot negative correction.",
            6, D_ACT_II_5),
        rel(HJ1, "concept-jspace-workspace", "builds-on",
            "H-J1 is the prediction the J-space reading generated: a coherent cluster should "
            "look verbalizable and a peaked single token should not.", 6, D_ACT_II_5),
        rel(HGL, "concept-glitch-token", "builds-on",
            "H-glitch asks whether the flip axis points at the anomalous-token cluster.",
            5, D_ACT_II_5),
        rel(F14, "concept-flip-axis", "qualifies",
            "There are two axes, not one: the frame-mixed committed axis (lambda -0.864) and "
            "the physical on-shell d_sym (-4.3). All headline mechanism numbers use d_sym.",
            7, D_POST_CLOSE),
        rel("concept-atr", "concept-power-iteration", "analogous-to",
            "ATR is the nonlinear analogue: where power iteration converges to the dominant "
            "eigenvector of a linear operator, ATR converges to stable states of the full "
            "forward map.", 6),
        rel("concept-atr", "concept-impulse-response", "analogous-to",
            "Lucier's room is to its impulse response what the transformer's weights are to "
            "its attractor landscape; the snapshot schedule is the acoustic measurement "
            "protocol transplanted.", 6),
        rel("concept-structural-semantic-transition", "concept-mixing-time", "analogous-to",
            "The point where prompt-specific information is lost and the landscape takes over "
            "is the loop's mixing time, and T_mix_LLM is the proposed measurement of it.", 5),
        rel("concept-atr", "concept-fixed-point-theory", "relates-to",
            "Fixed-point theory and basins of attraction are the framework the loop's results "
            "are stated in: state space, evolution rule, attractors, basin boundaries.", 6),
        rel("concept-l2-normalisation", "concept-atr", "relates-to",
            "The energy renormalisation is the single most important design decision: without "
            "it norms explode to ~1.5M and no landscape is visible.", 7),
        rel("concept-position-collapse", "concept-cross-prompt-invariance", "relates-to",
            "Positions collapse to identical vectors around iteration 10, which is the "
            "precondition for different prompts ending at near-identical final states.", 5),
        rel("concept-t-mix-llm", "concept-structural-semantic-transition", "relates-to",
            "T_mix_LLM is the proposed metric for exactly this transition: the iteration at "
            "which prompts heading to the same basin become indistinguishable.", 4),
        rel("concept-fractal-dimension", "concept-attractor-basin", "relates-to",
            "Fractal dimensional analysis is listed as a potential, untested metric for "
            "characterising basin geometry.", 3),
        rel("concept-activation-patching", "concept-atr", "relates-to",
            "Patching, probing and SAEs are the adjacent single-pass methods ATR complements "
            "with an iterated-dynamics view.", 4),
        rel("concept-body-without-organs", "concept-atr", "relates-to",
            "The undifferentiated substrate the loop is imagined to expose: the weight geometry "
            "before prompt input.", 3),
        rel("concept-tame-morphogenesis", "concept-attractor-basin", "relates-to",
            "Levin's morphogenetic framing reads attractor basins as something like a body plan "
            "for the model.", 3),
        rel("concept-w-e", "concept-coherence", "relates-to",
            "Coherence is measured entirely in W_E: the mean pairwise cosine among the top-k "
            "tokens' embedding rows.", 5),
        rel("concept-basin-token", "concept-attractor-basin", "relates-to",
            "The basin token is the terminal BPE token that names a basin, which is what every "
            "basin label in this project is.", 4),
        rel("concept-waypoint-token", "concept-dissolution-pathway", "relates-to",
            "Waypoints are the intermediate tokens the dissolution pathway passes through: ash, "
            "Canad, Ag, FT, capit, injustice, Rousse.", 4),
        rel("concept-residual-stream", "concept-atr", "relates-to",
            "The residual stream is the object the loop moves: extracted at the final layer, "
            "rescaled, and written back over the token embeddings at layer 0.", 6),
        rel(F6, "concept-dissolution-pathway", "supports",
            "The replication reproduced not just the terminal attractors but the intermediate "
            "waypoints, Ag at iteration 10 and Rousse at 50, three times over.", 6, D_ACT_II_5),
    ]

    # ---- runs -> hypotheses (tests) ---------------------------------------
    R += [
        rel("run-0-repeatability-gate", H0, "tests",
            "Two same-machine runs over five prompts, checking whether terminal basins are "
            "identical.", 6),
        rel("run-1-attractor-dominance", H1, "tests",
            "125 prompts across seven registers, to see how dominant prolet is once the choice "
            "of starting point is taken out of any one person's hands.", 6),
        rel("run-1-attractor-dominance", H2, "tests",
            "The same sweep asks whether Divine is a genuine second basin or an artefact of one "
            "hand-picked prompt.", 5),
        rel("run-2-cross-model-sweeps", HFP, "tests",
            "The decisive test of the fingerprint claim: the same 125 prompts through a "
            "same-corpus model and two Pile-trained models.", 9),
        rel("run-3-random-noise-null", HFP, "tests",
            "The null-model control asks whether the basins are in the weights at all, by "
            "removing language from the input.", 8),
        rel("run-5-gated-resweep", HTILL, "tests",
            "The re-sweep's pre-registered addendum: is till a slow transient that would "
            "disappear under proper convergence?", 7),
        rel("run-8-divine-motion-audit", HD1, "tests",
            "Three trajectories to 1000 iterations with a lag-1 probe, measuring how much of "
            "the motion the readout can see.", 7),
        rel("run-10-jlens-pilot", HJ1, "tests",
            "A deliberately restricted pilot lens over 193 tokens and 30 prompts, probing "
            "membership for the prolet and Divine attractors.", 6),
        rel("run-14-jlens-phase-probe", HJ1, "tests",
            "The same lens applied phase-aware, to both phases, the pivot and the flip axis.",
            6),
        rel("run-11-glitch-alignment", HGL, "tests",
            "The flip axis measured against the under-trained core and the curated "
            "SolidGoldMagikarp family, with random and norm-matched nulls.", 8),
        rel("run-12-flip-axis-eigenvalue", HFLIP, "tests",
            "The linearised ATR map measured by forward-mode autodiff, cross-checked against "
            "central finite differences, then decomposed by block and head.", 8),
        rel("run-15-suppression-test", HSUPP, "tests",
            "Three tests: the 144-head OV ranking, in-loop ablation with a same-layer control, "
            "and the ordinary-text copy test against L10.H7.", 9),
        rel("run-all-warm-permutation", H3, "tests",
            "The pre-registered null for the all-warm matrix: 10,000 random 14-token sets "
            "against the canonical 14.", 7),
        rel("run-spectral-scaffold", H4, "tests",
            "The pre-registered protocol for H4: per-head resonance against the top singular "
            "vector of W_OV. Executed 2026-07-25 (issue #25 regeneration); superseded by the "
            "run-16 eigenvector rescore per the #54 ruling.", 4),
        rel("run-6-full-distribution-confidence", H0, "tests",
            "Result 0 of the confidence audit is a replication check: same code, new machine "
            "class, mirror-sourced weights, three repeats.", 6),
    ]

    # ---- docs / prior work ------------------------------------------------
    R += [
        rel(F5, "doc-scaling-artefact", "documented-in",
            "The three attribution results (normalisation, tensor-level verdicts, readout "
            "jitter) are worked through in full in this document.", 4),
        rel(F11, "doc-jspace-primer", "documented-in",
            "The J-space reading that generated the prolet-inside / Divine-outside prediction "
            "is set out here.", 4),
        rel(F16, "doc-jspace-primer", "documented-in",
            "The lens construction the phase probe re-uses is described in Part 3 of the "
            "primer.", 4),
        rel("doc-jspace-reading-guide", "doc-jspace-primer", "builds-on",
            "The reading guide is the page-keyed map for the paper the primer explains.", 3),
        rel("doc-bell-primer", "doc-math-primer", "builds-on",
            "The mechanism-series companion assumes the vocabulary the maths primer teaches: "
            "vectors, cosine similarity, the residual stream, iterated maps.", 3),
        rel("doc-bell-primer", "doc-jspace-primer", "builds-on",
            "It also assumes the J-lens as the primer sets it out.", 3),
        rel("doc-readme", "doc-findings", "cites",
            "The README is the piece; where it and the canonical record differ, the record "
            "governs.", 5),
        rel("doc-journey-map", "doc-findings", "cites",
            "The journey map defers to FINDINGS section 3 for canonical hypothesis "
            "dispositions with full evidence.", 5),
        rel("doc-scaling-artefact", "doc-findings", "cites",
            "The artefact analysis records that the corpus-causal reading it opens with was "
            "later refuted in F3 and F4.", 4),
        rel("doc-technical", "prior-nanda-bloom-transformerlens", "cites",
            "The method specification is written against TransformerLens hook names: "
            "hook_resid_post and blocks.0.hook_resid_pre.", 4),
        rel("doc-isomorphism", "prior-lucier-1969", "cites",
            "The isomorphism document formalises Lucier's acoustic process as linear power "
            "iteration and then marks where the transformer case departs from it.", 5),
        rel("run-0-repeatability-gate", "doc-validation-plan", "builds-on",
            "Stage 0 of the pre-registered validation design: does re-running produce identical "
            "results?", 5),
        rel("run-1-attractor-dominance", "doc-validation-plan", "builds-on",
            "Stages 1-3 of the pre-registered design, run as a single 125-prompt sweep.", 5),
        rel(F3, "doc-method-comparison", "supersedes",
            "The refutation retired the large cross-model scaling and bias-profiling programme "
            "this document proposed: it existed to extend the fingerprint claim, which was "
            "refuted before it ran. The document carries a 2026-07-10 revision note saying so.",
            7, D_SERIES_CLOSE),
        rel(F13, "prior-rumbelow-watkins-2023", "cites",
            "The phase-B pole's named members (ertodd, quickShipAvailable and neighbours) are "
            "the published anomalous-token family from these posts.", 7),
        rel(F13, "prior-land-bartolo-2024", "cites",
            "The peer-reviewed replication that the near-centroid cluster, not embedding norm, "
            "is the marker of under-trained tokens in GPT-2.", 6),
        rel(F17, "prior-mcdougall-2023", "cites",
            "L10.H7, the documented copy-suppression head, is the positive control the "
            "protocol is validated against.", 7),
        rel(F14, "prior-elhage-2021", "cites",
            "The OV-circuit formalism the -4.3 flip-axis action is stated in; copying is "
            "scored by positive real eigenvalues of the full OV circuit.", 6),
        rel(F9, "prior-wang-2025", "analogous-to",
            "The nearest published neighbour: period-2 attractor cycles found by iterating a "
            "model through its text interface. Theirs are approximate, statistical, and "
            "text-level; this one is exact to machine precision and mechanistically located.",
            7),
        rel(F11, "prior-anthropic-jspace-2026", "cites",
            "The J-space paper is the source of the verbalizable-subspace claim the pilot lens "
            "was built to probe.", 6),
        rel(F3, "prior-radford-2019", "cites",
            "The GPT-2 paper establishes that Small and Medium share WebText, which is what "
            "makes the same-corpus / different-landscape contrast decisive.", 6),
        rel(F3, "prior-biderman-2023", "cites",
            "The Pythia suite supplies the two Pile-trained comparison models.", 4),
        rel("concept-logit-lens", "prior-nostalgebraist-2020", "cites",
            "The readout used throughout is logit-lens-style, with the documented failure mode "
            "that it reads only components aligned with W_U's strong directions.", 5),
        rel("concept-glitch-token", "prior-rumbelow-watkins-2023", "cites",
            "The discovery posts that named the anomalous-token family and located it near the "
            "embedding centroid.", 5),
        rel("concept-copy-suppression", "prior-mcdougall-2023", "cites",
            "The paper that documents the class and its GPT-2 Small exemplar L10.H7.", 5),
        rel("concept-jspace-workspace", "prior-anthropic-jspace-2026", "cites",
            "The paper the concept is taken from, with the Jacobian-lens construction it "
            "proposes.", 5),
        rel("concept-power-iteration", "prior-lucier-1969", "analogous-to",
            "Lucier's room implements classical power iteration on an acoustic transfer "
            "function; the tape loop is the iteration and the room modes are the dominant "
            "eigenvectors.", 5),
    ]

    return R


def question_relationships():
    """Wire every question to what raised it and to what it gates.

    Direction rule: one edge per dependency, drawn the way the record phrases it.
    Where the source says "X is blocked on Y" the edge is X -blocked-by-> Y;
    where the source leads with the blocker ("no claim until Y exists") it is
    Y -blocks-> X.  Drawing both would double the dependency count and make one
    stated blocker look like two.
    """
    R = []

    # ---- the shared blocker: issue #9, the prompt library -----------------
    # This is the pairing that prose cannot show.  F10 and F15 sit in different
    # sections and never mention each other, yet they are one artefact away from
    # both moving.
    # Resolved 2026-07-31: the library is restored (issue #24), so the two
    # blocked-by edges become relates-to edges that keep the pairing and its
    # history visible, and the artefact that answered the question carries a
    # retires edge into it.
    R += [
        rel("q-flip-axis-generality", "q-prompt-library", "relates-to",
            "Formerly blocked-by: F10's own words were that the flip-axis generality "
            "question is \"blocked on the prompt-library restoration, issue #9\", "
            "repeated in caveats 11 and 14. Blocker cleared 2026-07-31: the library is "
            "restored (issue #24) and the 34-prompt run is queued in ALIGNMENT_REVIEW.md "
            "section 5.", 8, "2026-07-31"),
        rel("q-lag2-regate-33", "q-prompt-library", "relates-to",
            "Formerly blocked-by: F15's own words were \"The other 33 period-2 prompts "
            "remain blocked on the prompt library (issue #9)\". Blocker cleared "
            "2026-07-31: the library is restored (issue #24) and the re-gate is queued "
            "in ALIGNMENT_REVIEW.md section 5.", 8, "2026-07-31"),
        rel("art-prompt-library", "q-prompt-library", "retires",
            "The restored library answers the question: all 125 prompts recovered "
            "verbatim from committed records, every entry flagged original (issue #24). "
            "The two threads the question gated are queued in ALIGNMENT_REVIEW.md "
            "section 5.", 6, "2026-07-31"),
    ]

    # ---- the other stated dependencies ------------------------------------
    R += [
        rel("q-fractal-dimension", "q-tmix-llm", "blocked-by",
            "JOURNEY_MAP section 7 states the order in two words: the fractal-dimension "
            "question \"Requires T_mix first\".", 4, D_SERIES_CLOSE),
        rel("q-jlens-full-build", "h-j1", "blocks",
            "H-J1 cannot move past a pilot-confidence null while the instrument that "
            "would settle it is unbuilt: its disposition ends \"Full build still pending "
            "(issue #8)\", and F11 specifies that the build \"should be phase-aware: "
            "probe both phases and the pivot M\".", 8, D_ACT_II_5),
        rel("q-shape-class-null", "concept-coherence", "blocks",
            "F12 states the block as a standing rule rather than a wish: \"no cross-model "
            "coherence claim until a shape-class-matched null exists\". The measure "
            "cannot travel outside GPT-2 Small until that null is built.", 7, D_ACT_II_5),
        rel("q-independent-reimplementation", "h0-determinism", "blocks",
            "H0 currently stands on repeatability plus one cross-hardware replication. "
            "Caveat 1's heading is the block: \"Repeatability plus one cross-hardware "
            "replication, not independent reproducibility\", and H0's disposition ends "
            "\"Independent re-implementation still not attempted\".", 6, D_ACT_II_5),
        rel("q-hook-window-depth", "q-why-gpt2-small", "blocks",
            "FINDINGS section 6 assigns the depth control to the successor question "
            "explicitly: the depth control, the per-layer / per-head decomposition, the "
            "spectral test and readout upgrades \"do not test whether the result is real; "
            "they test why the models differ. That is the successor project's "
            "question.\"", 5, D_SERIES_CLOSE),
    ]

    # ---- questions -> the claims that raised them -------------------------
    R += [
        rel("q-why-gpt2-small", F3, "relates-to",
            "F3 is what makes the question a question: same corpus, different landscape, "
            "and GPT-2 Small alone resolves language into few semantic basins.",
            7, D_SERIES_CLOSE),
        rel("q-why-gpt2-small", F8, "relates-to",
            "F8 sharpened what has to be explained: not a confident winner but a "
            "low-probability argmax over a coherent lexical field.", 6, D_ACT_II_5),
        rel("q-flip-axis-generality", F10, "relates-to",
            "F10 measured the flip axis on one trajectory and left its generality open in "
            "the same paragraph.", 8, D_ACT_II_5),
        rel("q-flip-axis-generality", F13, "relates-to",
            "Caveat 14 names the anomalous-token alignment as one of the four properties "
            "whose generality across the other 33 prompts is untested.", 5, D_ACT_II_5),
        rel("q-flip-axis-generality", F14, "relates-to",
            "Caveat 14 names the eigenvalue and the flip head L11.H8 as two more of those "
            "four properties.", 5, D_ACT_II_5),
        rel("q-flip-axis-generality", F17, "relates-to",
            "F17's ablation and copy test follow the same single audited trajectory, so "
            "the head's role in the other 33 prompts is untested too (caveat 14).",
            5, D_ACT_II_5),
        rel("q-lag2-regate-33", F15, "relates-to",
            "F15 implemented the lag-k gate and demonstrated it for one prompt; this is "
            "the remainder of that work.", 8, D_ACT_II_5),
        rel("q-lag2-regate-33", F9, "relates-to",
            "F9 is the finding that made the re-gate necessary: the 34 non-convergers "
            "cycle rather than drift, so they were failed by construction.", 6, D_ACT_II_5),
        rel("q-jlens-full-build", F11, "relates-to",
            "The pilot is what the full build replaces; every F11 limitation (caveat 13) "
            "is a specification for it.", 7, D_ACT_II_5),
        rel("q-jlens-full-build", F16, "relates-to",
            "F16 re-probed with the same restricted pilot lens, so it inherits the "
            "pilot's limits and does not discharge the full build.", 6, D_ACT_II_5),
        rel("q-independent-reimplementation", F6, "relates-to",
            "F6 is the strongest replication on record and is explicitly not this: "
            "\"same-code replication on new hardware, not independent "
            "re-implementation\".", 6, D_ACT_II_5),
        rel("q-gate-cadence", F1, "relates-to",
            "F1's lock-in iterations are all 120 because that is the gate's floor, which "
            "is what leaves the true settling times unresolved (caveat 5).",
            6, D_SERIES_CLOSE),
        rel("q-hook-window-depth", F3, "relates-to",
            "The cross-model differences are the thing a window or depth change would "
            "have to be ruled out of; all four sweeps cut the loop at one place.",
            5, D_SERIES_CLOSE),
        rel("q-shape-class-null", F12, "relates-to",
            "F12 is where the rule is stated: GPT-2 Medium's typographic cluster passes "
            "the coherence test at p = 0.001, so shape has to be controlled for.",
            7, D_ACT_II_5),
        rel("q-shape-class-null", F8, "relates-to",
            "F8's permutation nulls control for frequency via the norm proxy but not for "
            "token shape (caveat 10), which is the gap this question names.",
            5, D_ACT_II_5),
        rel("q-tmix-llm", "concept-t-mix-llm", "relates-to",
            "The question is the metric's own definition, still uncomputed: JOURNEY_MAP "
            "section 7 marks it \"Measurable from existing data\".", 5, D_SERIES_CLOSE),
        rel("q-slonski-macro-group", "prior-slonski-q-vector", "relates-to",
            "The Q-vector dichotomy is the framework the question is posed in, and the "
            "next step JOURNEY_MAP names is \"One Q-vector experiment, on its own "
            "terms\".", 4, D_SERIES_CLOSE),
        rel("q-slonski-macro-group", D["9"], "relates-to",
            "The prediction's premise was the all-warm compact subspace, retired "
            "2026-07-11 by the permutation test; the question survives it only as a "
            "question.", 4, D_PERMUTATION),
        rel("q-fractal-dimension", "concept-fractal-dimension", "relates-to",
            "The concept is listed in the Adjacent Science table as a potential, untested "
            "metric for basin geometry; this is the question that would use it.",
            3, D_SERIES_CLOSE),
    ]

    # ---- documented-in ----------------------------------------------------
    for q in QUESTIONS:
        doc = ("doc-findings" if q["doc_ref"].startswith("docs/FINDINGS.md")
               else "doc-journey-map")
        where = ("the canonical record" if doc == "doc-findings"
                 else "the journey map's Open Questions table")
        R.append(rel(q["id"], doc, "documented-in",
                     "The passage this question is quoted from is in %s." % where, 2))

    return R


def structural_relationships(claims, runs, run_models, sources):
    """produced-by, run-on, evidenced-by, documented-in - generated, but each
    with a description naming both endpoints and what the link is."""
    R = []
    by_id = {c["id"]: c for c in claims}
    run_by_id = {r["id"]: r for r in runs}

    # finding/hypothesis evidence -> produced-by
    for c in claims:
        for run_id in c.get("evidence", []) or []:
            r = run_by_id[run_id]
            R.append(rel(
                c["id"], run_id, "produced-by",
                "%s rests on %s (%s)." % (c["label"].split(":")[0], r["label"], r.get("n", "")),
                4, r.get("date")))

    # run -> model
    for r in runs:
        for mid in run_models.get(r["id"], []):
            R.append(rel(
                r["id"], mid, "run-on",
                ("%s analysed %s from its weights and committed artifacts (%s)."
                 if r["id"] in ANALYSIS_ONLY_RUNS else
                 "%s iterated the ATR loop on %s (%s).") % (
                    r["label"], next(m["label"] for m in runs if m["id"] == mid),
                    r.get("n", "")),
                2, r.get("date")))

    # noise null model
    R += [
        rel("run-3-random-noise-null", "null-model-gaussian-noise", "run-on",
            "Run 3 is the null-model control: 125 calibrated Gaussian tensors driven through "
            "GPT-2 Small's loop in place of prompts.", 5, D_SERIES_CLOSE),
        rel("run-6-full-distribution-confidence", "null-model-gaussian-noise", "run-on",
            "The confidence audit carried 15 calibrated noise trials alongside the five prompt "
            "states, as the comparison family for coherence.", 4, D_ACT_II_5),
        rel("run-8-divine-motion-audit", "null-model-gaussian-noise", "run-on",
            "One calibrated noise tensor served as the drifting control against which the "
            "cycle's invisibility ratio was bracketed.", 3, D_ACT_II_5),
        rel("run-13-lagk-regate", "null-model-gaussian-noise", "run-on",
            "The noise control supplies the third lag signature: monotonic decay with lag, no "
            "period.", 3, D_ACT_II_5),
        rel(FINDING_ID["F4"], "null-model-gaussian-noise", "evidenced-by",
            "The 18 non-semantic basins and the bootstrap CI [11, 17] were properties of this "
            "null model's ORIGINAL, mis-calibrated arm; run 17's matched-nu re-run superseded "
            "them and inverted F4's reading.", 7, D_SERIES_CLOSE),
    ]

    # artefacts -> runs, findings -> artefacts
    art_run = {a[0]: a[4] for a in ARTEFACTS}
    art_title = {a[0]: a[1] for a in ARTEFACTS}
    art_desc_override = {
        # atr_engine.py predates every run; run 13 is the run that changed it.
        "art-atr-engine": ("The lag-k re-gate added run_atr_gated's gate_lag parameter and the "
                           "lag_scan helper to the engine, verified bit-identical to the "
                           "pre-change engine at gate_lag = 1."),
    }
    for aid, run_id in sorted(art_run.items()):
        if run_id is None:
            # art-prompt-library: a restored source file, not a run product.
            continue
        R.append(rel(aid, run_id, "produced-by",
                     art_desc_override.get(
                         aid, "%s is the output record of %s."
                              % (art_title[aid], run_by_id[run_id]["label"])),
                     2, run_by_id[run_id].get("date")))

    finding_artefacts = {
        "F1": ["art-gated-report", "art-convergence-matrix-small",
               "art-basin-distribution-small", "art-topology-3d-small"],
        "F2": ["art-gated-report"],
        "F3": ["art-convergence-matrix-medium", "art-convergence-matrix-160m",
               "art-convergence-matrix-410m", "art-deep-basin-assessment"],
        "F4": ["art-random-baseline-report"],
        "F5": ["art-readout-guardrails-json"],
        "F6": ["art-confidence-report"],
        "F7": ["art-confidence-report"],
        "F8": ["art-confidence-report", "art-chordness-formal"],
        "F9": ["art-divine-motion-report", "art-bell-anatomy"],
        "F10": ["art-bell-anatomy"],
        "F11": ["art-jlens-pilot-report"],
        "F12": ["art-chordness-formal"],
        "F13": ["art-glitch-alignment"],
        "F14": ["art-hinge-eigenvalue"],
        "F15": ["art-lagk-report", "art-atr-engine"],
        "F16": ["art-jlens-phase"],
        "F17": ["art-suppression-report"],
    }
    for key, arts in finding_artefacts.items():
        fid = FINDING_ID[key]
        for aid in arts:
            R.append(rel(fid, aid, "evidenced-by",
                         "%s is reported in full in %s." % (key, art_title[aid]),
                         3, by_id[fid]["asserted"]))

    # documented-in hubs
    for key, fid in sorted(FINDING_ID.items(), key=lambda kv: int(kv[0][1:])):
        R.append(rel(fid, "doc-findings", "documented-in",
                     "%s is a principal finding of the canonical record." % key, 2))
    for key, hid in HYP_ID.items():
        R.append(rel(hid, "doc-findings", "documented-in",
                     "%s's disposition is fixed in the canonical record's hypothesis table."
                     % key, 2))
    for num, did in sorted(DISC_ID.items(), key=lambda kv: int(kv[0])):
        R.append(rel(did, "doc-journey-map", "documented-in",
                     "Discovery %s is row %s of the Key Discoveries table." % (num, num), 2))
    for c in claims:
        if c["type"] == "concept" and c["doc_ref"].startswith("docs/JOURNEY_MAP.md"):
            R.append(rel(c["id"], "doc-journey-map", "documented-in",
                         "%s is defined in the journey map." % c["label"], 1))
    for key in ("F13", "F14", "F15", "F16", "F17"):
        R.append(rel(FINDING_ID[key], "doc-bell-primer", "documented-in",
                     "%s is one of the five mechanism-series measurements the Bell primer "
                     "walks through in plain language." % key, 3))

    # prior work cited by its listing document
    for pid, _t, _u, _d, cited_by in PRIOR_WORK:
        R.append(rel(cited_by, pid, "cites",
                     "Listed and characterised in this document's reference section.", 1))
    for s in sources:
        if s["type"] == "prior-work" and "cited_by" in s and s["id"] not in {
                p[0] for p in PRIOR_WORK}:
            R.append(rel(s["cited_by"], s["id"], "cites",
                         "Listed in the Adjacent Science & Mathematics table as prior art "
                         "bearing on ATR.", 1))

    # runs recorded in the run-by-run summary
    for rid in ("run-3-random-noise-null", "run-4-deep-convergence", "run-5-gated-resweep",
                "run-cos-sim-diagnostic", "run-readout-guardrails",
                "run-all-warm-permutation"):
        R.append(rel(rid, "doc-results-summary", "documented-in",
                     "Environment, deviations and headline numbers for this run are recorded "
                     "in the validation series' run-by-run summary.", 2))
    R.append(rel("run-2-cross-model-sweeps", "doc-cross-model-run-plan", "documented-in",
                 "The cross-model sweep was executed against this run plan.", 2))

    return R


# --------------------------------------------------------------------------
# 9. Visual config
# --------------------------------------------------------------------------

def visual_config():
    return OrderedDict([
        ("version", "1.0"),
        ("domain", "evidence"),
        ("colors", OrderedDict([
            ("hypothesis", "#6B4C8A"),
            ("finding", "#2E7D5B"),
            ("concept", "#B9812F"),
            ("question", "#A8477A"),
            ("run", "#5B7DB1"),
            ("model", "#B3423F"),
            ("null-model", "#8A8F94"),
            ("doc", "#7f8c8d"),
            ("artefact", "#C7CDD1"),
            ("prior-work", "#B0B7BC"),
            ("default", "#7f8c8d"),
        ])),
        ("status_colors", OrderedDict([
            ("supported", "#2E7D5B"),
            ("refuted", "#B3423F"),
            # Muted, desaturated brick: negative in hue like "refuted" but
            # visibly weaker. Nearest neighbours are refuted (dE76 28.8) and
            # retired (30.4), both comfortably above the ~20 confusion floor.
            ("not-supported", "#8F5A57"),
            ("qualified", "#B9812F"),
            ("retired", "#8A8F94"),
            ("corrected", "#5B7DB1"),
            ("open", "#6B4C8A"),
            ("untested", "#9AA3A8"),
            ("default", "#7f8c8d"),
        ])),
        ("shapes", OrderedDict([
            ("hypothesis", "diamond"),
            ("finding", "dot"),
            ("concept", "hexagon"),
            # `circle` draws the label inside, so an open question cannot be
            # mistaken for another finding's `dot` at a glance.
            ("question", "circle"),
            ("run", "square"),
            ("model", "triangle"),
            ("null-model", "triangleDown"),
            ("doc", "box"),
            ("artefact", "ellipse"),
            ("prior-work", "star"),
            ("default", "dot"),
        ])),
        ("edge_styles", OrderedDict([
            ("supports", OrderedDict([("style", "solid"), ("dashes", False),
                                      ("color", "#2E7D5B"), ("arrow", True), ("width", 2)])),
            ("refutes", OrderedDict([("style", "dashed"), ("dashes", [8, 4]),
                                     ("color", "#B3423F"), ("arrow", True), ("width", 2)])),
            ("qualifies", OrderedDict([("style", "dashed"), ("dashes", [8, 4]),
                                       ("color", "#B9812F"), ("arrow", True), ("width", 2)])),
            ("corrects", OrderedDict([("style", "dotted"), ("dashes", [2, 4]),
                                      ("color", "#5B7DB1"), ("arrow", True), ("width", 2)])),
            ("retires", OrderedDict([("style", "dotted"), ("dashes", [2, 4]),
                                     ("color", "#8A8F94"), ("arrow", True), ("width", 2)])),
            ("supersedes", OrderedDict([("style", "dotted"), ("dashes", [2, 4]),
                                        ("color", "#5B7DB1"), ("arrow", True), ("width", 2)])),
            ("tests", OrderedDict([("style", "solid"), ("dashes", False),
                                   ("color", "#6B4C8A"), ("arrow", True), ("width", 2)])),
            ("produced-by", OrderedDict([("style", "solid"), ("dashes", False),
                                         ("color", "#C7CDD1"), ("arrow", True), ("width", 1)])),
            ("run-on", OrderedDict([("style", "solid"), ("dashes", False),
                                    ("color", "#C7CDD1"), ("arrow", True), ("width", 1)])),
            ("evidenced-by", OrderedDict([("style", "solid"), ("dashes", False),
                                          ("color", "#C7CDD1"), ("arrow", True), ("width", 1)])),
            ("documented-in", OrderedDict([("style", "solid"), ("dashes", False),
                                           ("color", "#C7CDD1"), ("arrow", True), ("width", 1)])),
            ("analogous-to", OrderedDict([("style", "dashed"), ("dashes", [8, 4]),
                                          ("color", "#5B7DB1"), ("arrow", False), ("width", 1)])),
            ("breaks-down-at", OrderedDict([("style", "dashed"), ("dashes", [8, 4]),
                                            ("color", "#5B7DB1"), ("arrow", False), ("width", 1)])),
            ("builds-on", OrderedDict([("style", "solid"), ("dashes", False),
                                       ("color", "#B0B7BC"), ("arrow", True), ("width", 1)])),
            ("cites", OrderedDict([("style", "solid"), ("dashes", False),
                                   ("color", "#B0B7BC"), ("arrow", True), ("width", 1)])),
            ("relates-to", OrderedDict([("style", "solid"), ("dashes", False),
                                        ("color", "#B0B7BC"), ("arrow", True), ("width", 1)])),
            # Dependency edges share the question colour so open work reads as
            # one family, and are dashed with an arrow because which end is
            # blocked is the entire content of the edge.
            ("blocks", OrderedDict([("style", "dashed"), ("dashes", [6, 3]),
                                    ("color", "#A8477A"), ("arrow", True), ("width", 2)])),
            ("blocked-by", OrderedDict([("style", "dashed"), ("dashes", [6, 3]),
                                        ("color", "#A8477A"), ("arrow", True), ("width", 2)])),
        ])),
        ("node_sizes", OrderedDict([
            ("hypothesis", 26),
            ("finding", 22),
            ("concept", 16),
            ("question", 20),
            ("run", 18),
            ("model", 24),
            ("null-model", 18),
            ("doc", 14),
            ("artefact", 11),
            ("prior-work", 12),
            ("default", 14),
        ])),
        ("physics", OrderedDict([
            ("enabled", True),
            ("solver", "forceAtlas2Based"),
            ("forceAtlas2Based", OrderedDict([
                ("gravitationalConstant", -72),
                ("centralGravity", 0.008),
                ("springLength", 165),
                ("springConstant", 0.06),
                ("damping", 0.5),
                ("avoidOverlap", 0.35),
            ])),
            ("stabilization", OrderedDict([
                ("enabled", True),
                ("iterations", 420),
                ("updateInterval", 25),
                ("fit", True),
            ])),
            ("minVelocity", 0.75),
            ("maxVelocity", 40),
            ("timestep", 0.4),
        ])),
        ("edge_weight_range", OrderedDict([("min", 1), ("max", 10),
                                           ("min_width", 0.5), ("max_width", 5)])),
    ])


# --------------------------------------------------------------------------
# 10. Validation
# --------------------------------------------------------------------------

def validate(graph):
    errors = []
    ids = {}
    for bucket in ("claims", "runs", "sources"):
        for node in graph[bucket]:
            if node["id"] in ids:
                errors.append("duplicate node id %r (in %s and %s)"
                              % (node["id"], ids[node["id"]], bucket))
            ids[node["id"]] = bucket

    for c in graph["claims"]:
        if c["status"] not in ALLOWED_STATUS:
            errors.append("claim %s has status %r, not in the allowed vocabulary"
                          % (c["id"], c["status"]))
        if c["type"] not in ALLOWED_CLAIM_TYPE:
            errors.append("claim %s has type %r" % (c["id"], c["type"]))
        if not c.get("description"):
            errors.append("claim %s has an empty description" % c["id"])
        for run_id in c.get("evidence", []) or []:
            if run_id not in ids:
                errors.append("claim %s: evidence run %r does not resolve" % (c["id"], run_id))
    for r in graph["runs"]:
        if r["type"] not in ALLOWED_RUN_TYPE:
            errors.append("run %s has type %r" % (r["id"], r["type"]))
    for s in graph["sources"]:
        if s["type"] not in ALLOWED_SOURCE_TYPE:
            errors.append("source %s has type %r" % (s["id"], s["type"]))

    dangling = []
    for i, e in enumerate(graph["relationships"]):
        if e["type"] not in ALLOWED_REL:
            errors.append("relationship %d (%s -> %s) has type %r, not in the allowed "
                          "vocabulary" % (i, e["from"], e["to"], e["type"]))
        if not e.get("description", "").strip():
            errors.append("relationship %d (%s -> %s) has an empty description"
                          % (i, e["from"], e["to"]))
        for end in ("from", "to"):
            if e[end] not in ids:
                dangling.append("%s (%s of edge %d: %s -%s-> %s)"
                                % (e[end], end, i, e["from"], e["type"], e["to"]))
    if dangling:
        errors.append("DANGLING RELATIONSHIP ENDPOINTS (%d):\n    %s"
                      % (len(dangling), "\n    ".join(dangling)))

    # every path on disk -- and every anchor inside the document it names
    anchor_cache = {}

    def anchors_of(rel_path):
        if rel_path not in anchor_cache:
            try:
                with open(os.path.join(REPO, rel_path), encoding="utf-8") as fh:
                    anchor_cache[rel_path] = heading_anchors(fh.read())
            except OSError:
                anchor_cache[rel_path] = set()
        return anchor_cache[rel_path]

    def check_path(owner, field, value):
        if not value:
            return
        if value.startswith("http://") or value.startswith("https://"):
            return
        rel_path, _, fragment = value.partition("#")
        rel_path = rel_path.rstrip("/")
        if not rel_path:
            return
        if not os.path.exists(os.path.join(REPO, rel_path)):
            errors.append("%s.%s points at %r which does not exist on disk"
                          % (owner, field, rel_path))
            return
        # A file that exists but whose named heading does not is a dead link
        # everywhere the graph is rendered, and silently so.
        if fragment and rel_path.endswith(".md") and fragment not in anchors_of(rel_path):
            errors.append(
                "%s.%s points at %r: %s exists, but no heading in it anchors as "
                "#%s (the section was renamed or removed -- re-point the reference "
                "in the source document, do not hand-edit the graph)"
                % (owner, field, value, rel_path, fragment))

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

    if errors:
        sys.stderr.write("\nVALIDATION FAILED (%d problem(s)):\n" % len(errors))
        for err in errors:
            sys.stderr.write("  - %s\n" % err)
        sys.stderr.write("\n")
        raise SystemExit(1)


# --------------------------------------------------------------------------
# 11. Main
# --------------------------------------------------------------------------

def main():
    findings_md = read(FINDINGS_MD)
    journey_md = read(JOURNEY_MD)
    read(README_MD)  # presence check: the reference list and notebook table are cited

    phases = parse_phases(journey_md)

    claims = []
    claims += parse_findings(findings_md)
    claims += parse_hypotheses(findings_md)
    claims += parse_discoveries(journey_md)
    adj_concepts, adj_priors = parse_adjacent(journey_md)
    claims += adj_concepts
    claims += parse_glossary(journey_md)
    claims += series_concepts()
    claims += questions()

    runs, run_models = parse_runs(findings_md)
    runs += EXTRA_RUNS
    runs += MODELS
    # The two executed EXTRA_RUNS are GPT-2 Small work too (JOURNEY_MAP Phase 0:
    # "Model choice: GPT-2 Small"; Phase 4's neighbourhood test reads GPT-2 Small's
    # W_E).  Without this they were the only executed runs with no run-on edge.
    run_models["run-original-piece"] = ["model-gpt2-small"]
    run_models["run-token-neighbourhood"] = ["model-gpt2-small"]

    sources = build_sources(adj_priors)

    relationships = curated_relationships()
    relationships += question_relationships()
    relationships += structural_relationships(claims, runs, run_models, sources)

    # stable ordering: idempotent output
    type_order = {"finding": 0, "hypothesis": 1, "concept": 2, "question": 3}
    claims.sort(key=lambda c: (type_order[c["type"]], c["id"]))
    runs.sort(key=lambda r: ({"run": 0, "model": 1, "null-model": 2}[r["type"]], r["id"]))
    sources.sort(key=lambda s: ({"doc": 0, "artefact": 1, "prior-work": 2}[s["type"]], s["id"]))
    relationships.sort(key=lambda e: (e["type"], e["from"], e["to"]))

    graph = OrderedDict([
        ("metadata", OrderedDict([
            ("domain", "evidence"),
            ("version", "1.0"),
            ("created", BUILD_DATE),
            ("last_updated", BUILD_DATE),
            ("source_last_updated", SOURCE_LAST_UPDATED),
            ("title", "ATR Evidence Graph: hypotheses, findings, runs and self-corrections"),
            ("description",
             "The Activation Tensor Resonance series as an evidence graph: 17 principal "
             "findings, 16 key discoveries, 12 hypotheses, the concepts they run on and the "
             "questions the record leaves open, wired to the runs, models and artefacts that "
             "produced them, with every recorded correction, retirement and supersession "
             "carried as a signed edge and every stated blocker as a blocks / blocked-by "
             "edge."),
            ("generator", "docs/graph/build_evidence_graph.py"),
            ("sources_parsed", ["docs/FINDINGS.md", "docs/JOURNEY_MAP.md"]),
            ("sources_checked_not_parsed", ["README.md"]),
            ("date_precision",
             "Phases 0-3 and the runs inside them carry the month anchor 2026-03-01; "
             "FINDINGS.md records only 'Original exploratory work: 2026-03'. Runs 11-15 are "
             "stamped 2026-07-19 because FINDINGS.md dates the mechanism series '2026-07-19 "
             "onward' and gives no per-run dates; the stamp is the series start, not each run's "
             "execution day. Hypothesis `asserted` dates are placements, not record: the sources "
             "date dispositions, never the moment a hypothesis was raised. Question "
             "`asserted` dates are placements on the same footing: the caveats and "
             "JOURNEY_MAP's Open Questions table carry no dates of their own, so each "
             "question is stamped with the document revision that states it (2026-07-10 at "
             "series close, 2026-07-19 for the Act II.5 caveats and F-sections). Every other "
             "date, "
             "including every date on a corrects / retires / supersedes edge, is day-precise as "
             "stated in the record."),
            ("phases", phases),
        ])),
        ("claims", claims),
        ("runs", runs),
        ("sources", sources),
        ("relationships", relationships),
    ])

    validate(graph)

    os.makedirs(DATA_DIR, exist_ok=True)
    ent_path = os.path.join(DATA_DIR, "entities.json")
    vis_path = os.path.join(DATA_DIR, "visual_config.json")
    with open(ent_path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(vis_path, "w", encoding="utf-8") as fh:
        json.dump(visual_config(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # ---- summary ----------------------------------------------------------
    node_types = Counter([c["type"] for c in graph["claims"]]
                         + [r["type"] for r in graph["runs"]]
                         + [s["type"] for s in graph["sources"]])
    statuses = Counter(c["status"] for c in graph["claims"])
    edge_types = Counter(e["type"] for e in graph["relationships"])
    total_nodes = len(graph["claims"]) + len(graph["runs"]) + len(graph["sources"])

    print("ATR evidence graph built")
    print("  %s" % ent_path)
    print("  %s" % vis_path)
    print("")
    print("nodes: %d   edges: %d" % (total_nodes, len(graph["relationships"])))
    print("")
    print("by node type:")
    for t, n in sorted(node_types.items(), key=lambda kv: (-kv[1], kv[0])):
        print("  %-12s %3d" % (t, n))
    print("")
    print("claim status histogram:")
    for s, n in sorted(statuses.items(), key=lambda kv: (-kv[1], kv[0])):
        print("  %-12s %3d" % (s, n))
    print("")
    print("by edge type:")
    epistemic = {"supports", "refutes", "qualifies", "corrects",
                 "retires", "supersedes", "tests"}
    for t, n in sorted(edge_types.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "*" if t in epistemic else " "
        print("  %s %-15s %3d" % (mark, t, n))
    print("")
    print("self-correction edges (corrects / retires / supersedes):")
    for e in graph["relationships"]:
        if e["type"] in ("corrects", "retires", "supersedes"):
            print("  %-12s %s -> %s  [%s]" % (
                e["type"], e["from"], e["to"], e.get("asserted", "undated")))
    print("")
    print("open questions (type=question, status=open) and their dependencies:")
    q_ids = [c["id"] for c in graph["claims"] if c["type"] == "question"]
    for qid in q_ids:
        claim = next(c for c in graph["claims"] if c["id"] == qid)
        print("  %s" % claim["label"])
        print("      %-14s %s" % ("id", qid))
        print("      %-14s %s" % ("doc_ref", claim["doc_ref"]))
        gates, gated_by = [], []
        for e in graph["relationships"]:
            if e["type"] == "blocks" and e["from"] == qid:
                gates.append(e["to"])
            elif e["type"] == "blocked-by" and e["to"] == qid:
                gates.append(e["from"])
            elif e["type"] == "blocks" and e["to"] == qid:
                gated_by.append(e["from"])
            elif e["type"] == "blocked-by" and e["from"] == qid:
                gated_by.append(e["to"])
        print("      %-14s %s" % ("blocks", ", ".join(sorted(gates)) or "-"))
        print("      %-14s %s" % ("blocked by", ", ".join(sorted(gated_by)) or "-"))
    print("")
    print("dependency edges (blocks / blocked-by):")
    for e in graph["relationships"]:
        if e["type"] in ("blocks", "blocked-by"):
            print("  %-11s %s -> %s  [%s]" % (
                e["type"], e["from"], e["to"], e.get("asserted", "undated")))
    print("")
    print("phases: %s" % ", ".join("%s (%s)" % (p["id"], p["start"])
                                   for p in graph["metadata"]["phases"]))


if __name__ == "__main__":
    main()
