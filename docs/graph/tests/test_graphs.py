"""Structural and content tests for the three generated graph files.

    docs/graph/_data/entities.json      the evidence graph
    docs/graph/_data/isomorphism.json   the acoustic/transformer correspondence
    docs/graph/_data/dissolution.json   six token-flow graphs, one per model

The `_data/*.json` files are build products (see docs/graph/build_*.py). If they
have not been generated, every test here skips rather than fails - a fresh
checkout with no build should be green.

Run from the repo root:

    python3 -m pytest docs/graph/tests -q
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

# docs/graph/tests/test_graphs.py -> docs/graph/tests -> docs/graph -> docs -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "graph" / "_data"

EVIDENCE_FILES = ["entities.json", "isomorphism.json"]

# --- Vocabularies, transcribed from docs/graph/README.md "The vocabularies" --

# Four types. `question` is what the record explicitly leaves open - distinct
# from a hypothesis, which asserts something testable, and from a concept, which
# is only vocabulary. It exists because `open` was useless as a work signal:
# every `open` claim in this graph used to be a concept, `open` being what a
# concept falls into when there is no epistemic verdict to record.
CLAIM_TYPES = {"hypothesis", "finding", "concept", "question"}
RUN_TYPES = {"run", "model", "null-model"}
SOURCE_TYPES = {"doc", "artefact", "prior-work"}

# Eight values, ordered as a severity gradient. The three negative ones are
# distinct: "refuted" = the evidence contradicts the claim; "not-supported" =
# the evidence failed to back it, without contradicting it (a null result);
# "qualified" = it survives in a narrowed or mixed form.
CLAIM_STATUSES = {
    "supported",
    "refuted",
    "not-supported",
    "qualified",
    "retired",
    "corrected",
    "open",
    "untested",
}

# Canonical order, shared by viewer.html's STATUS_ORDER, visual_config.json's
# status_colors and the README table. "not-supported" sits immediately after
# "refuted" so the list reads as a severity gradient.
STATUS_ORDER = [
    "supported",
    "refuted",
    "not-supported",
    "qualified",
    "retired",
    "corrected",
    "open",
    "untested",
]

STATUS_COLOURS = {
    "supported": "#2E7D5B",
    "refuted": "#B3423F",
    "not-supported": "#8F5A57",
    "qualified": "#B9812F",
    "retired": "#8A8F94",
    "corrected": "#5B7DB1",
    "open": "#6B4C8A",
    "untested": "#9AA3A8",
    "default": "#7f8c8d",
}

EPISTEMIC_EDGES = {
    "supports",
    "refutes",
    "qualifies",
    "corrects",
    "retires",
    "supersedes",
    "tests",
}
STRUCTURAL_EDGES = {"produced-by", "run-on", "evidenced-by", "documented-in"}
# `blocks` / `blocked-by` are the same relation read from opposite ends: a
# dependency between open work and what it gates. Associative, not epistemic - a
# blocker changes whether a claim can be settled yet, never whether it is true.
DEPENDENCY_EDGES = {"blocks", "blocked-by"}
ASSOCIATIVE_EDGES = {
    "analogous-to",
    "breaks-down-at",
    "builds-on",
    "cites",
    "relates-to",
} | DEPENDENCY_EDGES
EDGE_TYPES = EPISTEMIC_EDGES | STRUCTURAL_EDGES | ASSOCIATIVE_EDGES

# The statuses a `question` may carry. There is deliberately no "answered"
# value: an answered question takes `retired` and the answer is the claim on the
# other end of its incoming retires/supersedes edge.
QUESTION_STATUSES = {"open", "untested", "retired"}

# The questions extracted from the record, pinned. Every one of these
# corresponds to a sentence in docs/FINDINGS.md or docs/JOURNEY_MAP.md that
# leaves something open, quoted in the node's own description. Adding another is
# a judgement call about the record and belongs in this pin, so that a builder
# cannot quietly manufacture open work.
EXPECTED_QUESTIONS = [
    "q-flip-axis-generality",
    "q-fractal-dimension",
    "q-gate-cadence",
    "q-hook-window-depth",
    "q-independent-reimplementation",
    "q-jlens-full-build",
    "q-lag2-regate-33",
    "q-prompt-library",
    "q-shape-class-null",
    "q-slonski-macro-group",
    "q-tmix-llm",
    "q-why-gpt2-small",
]

EVIDENCE_TOP_LEVEL_KEYS = {"metadata", "claims", "runs", "sources", "relationships"}
DISSOLUTION_TOP_LEVEL_KEYS = {"metadata", "models"}

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Fields whose value, when present, names something on disk.
PATH_FIELDS = ("doc_ref", "path", "script", "output_dir")


# --- Loading ---------------------------------------------------------------


def _load(name):
    """Load a generated graph file, or skip the test if it has not been built."""
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO_ROOT)} not generated - run docs/graph/build_*.py")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{name} is not valid JSON: {exc}")


@pytest.fixture(scope="module")
def entities():
    return _load("entities.json")


@pytest.fixture(scope="module")
def isomorphism():
    return _load("isomorphism.json")


@pytest.fixture(scope="module")
def dissolution():
    return _load("dissolution.json")


@pytest.fixture(scope="module")
def evidence_graphs():
    return {name: _load(name) for name in EVIDENCE_FILES}


def _graph(evidence_graphs, name):
    return evidence_graphs[name]


def _node_ids(graph):
    ids = []
    for collection in ("claims", "runs", "sources"):
        ids.extend(entity["id"] for entity in graph[collection])
    return ids


def _looks_like_repo_path(value):
    """True for values that name something inside this repository."""
    if not isinstance(value, str) or not value.strip():
        return False
    if value.startswith(("http://", "https://", "#", "mailto:")):
        return False
    base = value.split("#", 1)[0].strip()
    if not base:
        return False
    if "/" in base:
        return True
    # bare filenames at the repo root, e.g. README.md, atr_engine.py
    return bool(re.search(r"\.[A-Za-z0-9]{1,6}$", base))


# ===========================================================================
# Evidence-schema graphs: entities.json and isomorphism.json
# ===========================================================================


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_evidence_file_is_valid_json_object(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    assert isinstance(graph, dict), f"{name}: top level must be an object"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_evidence_file_has_required_top_level_keys(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    missing = EVIDENCE_TOP_LEVEL_KEYS - set(graph)
    assert not missing, f"{name}: missing top-level keys {sorted(missing)}"
    for collection in ("claims", "runs", "sources", "relationships"):
        assert isinstance(graph[collection], list), f"{name}: {collection} must be a list"
        assert graph[collection], f"{name}: {collection} is empty"
    assert isinstance(graph["metadata"], dict)


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_metadata_has_domain_and_version(evidence_graphs, name):
    metadata = _graph(evidence_graphs, name)["metadata"]
    for key in ("domain", "version", "title", "phases"):
        assert key in metadata, f"{name}: metadata missing {key!r}"
    assert isinstance(metadata["phases"], list) and metadata["phases"]


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_no_duplicate_node_ids(evidence_graphs, name):
    ids = _node_ids(_graph(evidence_graphs, name))
    duplicates = sorted(i for i, n in Counter(ids).items() if n > 1)
    assert not duplicates, f"{name}: duplicate node ids {duplicates}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_every_node_has_id_and_label(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    for collection, label_key in (("claims", "label"), ("runs", "label"), ("sources", "title")):
        for entity in graph[collection]:
            assert entity.get("id"), f"{name}/{collection}: node without an id"
            assert entity.get(label_key), (
                f"{name}/{collection}/{entity.get('id')}: missing {label_key!r}"
            )


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_every_relationship_endpoint_resolves(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    ids = set(_node_ids(graph))
    dangling = [
        (rel["from"], rel["type"], rel["to"])
        for rel in graph["relationships"]
        if rel["from"] not in ids or rel["to"] not in ids
    ]
    assert not dangling, f"{name}: relationships with unresolved endpoints: {dangling[:10]}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_every_claim_status_is_in_the_vocabulary(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    bad = [
        (claim["id"], claim.get("status"))
        for claim in graph["claims"]
        if claim.get("status") not in CLAIM_STATUSES
    ]
    assert not bad, f"{name}: claims with a status outside the vocabulary: {bad}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_every_relationship_type_is_in_the_vocabulary(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    bad = sorted(
        {rel["type"] for rel in graph["relationships"] if rel["type"] not in EDGE_TYPES}
    )
    assert not bad, f"{name}: relationship types outside the vocabulary: {bad}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_every_node_type_is_in_the_vocabulary(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    for collection, allowed in (
        ("claims", CLAIM_TYPES),
        ("runs", RUN_TYPES),
        ("sources", SOURCE_TYPES),
    ):
        bad = [
            (entity["id"], entity.get("type"))
            for entity in graph[collection]
            if entity.get("type") not in allowed
        ]
        assert not bad, f"{name}/{collection}: types outside the vocabulary: {bad}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_every_relationship_has_a_description(evidence_graphs, name):
    """The generators' rule: an edge with no specific sentence should not be drawn."""
    graph = _graph(evidence_graphs, name)
    bare = [
        (rel["from"], rel["type"], rel["to"])
        for rel in graph["relationships"]
        if not str(rel.get("description", "")).strip()
    ]
    assert not bare, f"{name}: edges with no description: {bare[:10]}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_no_duplicate_or_self_referential_relationships(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    keys = [(rel["from"], rel["to"], rel["type"]) for rel in graph["relationships"]]
    duplicates = sorted(k for k, n in Counter(keys).items() if n > 1)
    assert not duplicates, f"{name}: duplicate edges {duplicates}"
    loops = [k for k in keys if k[0] == k[1]]
    assert not loops, f"{name}: self-referential edges {loops}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_claim_phases_resolve_to_declared_phases(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    phase_ids = {phase["id"] for phase in graph["metadata"]["phases"]}
    bad = [
        (claim["id"], claim["phase"])
        for claim in graph["claims"]
        if claim.get("phase") and claim["phase"] not in phase_ids
    ]
    assert not bad, f"{name}: claims referencing undeclared phases: {bad}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_dates_are_iso_formatted(evidence_graphs, name):
    graph = _graph(evidence_graphs, name)
    bad = []
    for collection in ("claims", "runs"):
        for entity in graph[collection]:
            for key in ("asserted", "retired", "date"):
                value = entity.get(key)
                if value is None:
                    continue
                if not ISO_DATE_RE.match(str(value)):
                    bad.append((entity["id"], key, value))
    assert not bad, f"{name}: non-ISO dates: {bad}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_repo_paths_exist_on_disk(evidence_graphs, name):
    """Every doc_ref / path / script / output_dir that looks like a repo path
    must actually resolve, so the graph cannot cite a file that is not there."""
    graph = _graph(evidence_graphs, name)
    missing = []
    for collection in ("claims", "runs", "sources"):
        for entity in graph[collection]:
            for key in PATH_FIELDS:
                value = entity.get(key)
                if not _looks_like_repo_path(value):
                    continue
                base = value.split("#", 1)[0].strip()
                if not (REPO_ROOT / base).exists():
                    missing.append((collection, entity["id"], key, value))
    assert not missing, f"{name}: referenced paths not on disk: {missing}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_every_claim_is_connected(evidence_graphs, name):
    """A claim nobody points at and which points at nobody is a graph leak."""
    graph = _graph(evidence_graphs, name)
    touched = set()
    for rel in graph["relationships"]:
        touched.add(rel["from"])
        touched.add(rel["to"])
    orphans = [claim["id"] for claim in graph["claims"] if claim["id"] not in touched]
    assert not orphans, f"{name}: claims with no edges at all: {orphans}"


def test_isomorphism_pairs_have_a_side(isomorphism):
    """The isomorphism graph is two columns; every paired claim declares which."""
    sides = {claim.get("side") for claim in isomorphism["claims"] if claim.get("role") == "pair"}
    assert sides <= {"acoustic", "transformer"}, f"unexpected sides: {sides}"
    assert sides == {"acoustic", "transformer"}, "both columns must be populated"


def test_isomorphism_analogies_cross_the_two_sides(isomorphism):
    """An `analogous-to` edge that stays on one side is not an isomorphism claim."""
    side = {claim["id"]: claim.get("side") for claim in isomorphism["claims"]}
    same_side = [
        (rel["from"], rel["to"])
        for rel in isomorphism["relationships"]
        if rel["type"] == "analogous-to"
        and side.get(rel["from"]) is not None
        and side.get(rel["from"]) == side.get(rel["to"])
    ]
    assert not same_side, f"analogous-to edges that do not cross sides: {same_side}"


# ===========================================================================
# dissolution.json
# ===========================================================================


def test_dissolution_is_valid_json_object(dissolution):
    assert isinstance(dissolution, dict)


def test_dissolution_has_required_top_level_keys(dissolution):
    missing = DISSOLUTION_TOP_LEVEL_KEYS - set(dissolution)
    assert not missing, f"dissolution.json: missing top-level keys {sorted(missing)}"
    assert isinstance(dissolution["models"], dict) and dissolution["models"]
    metadata = dissolution["metadata"]
    for key in ("domain", "version", "title", "iterations", "iterations_observed"):
        assert key in metadata, f"dissolution.json: metadata missing {key!r}"
    assert dissolution["metadata"]["domain"] == "dissolution"


def test_dissolution_does_not_use_the_evidence_schema(dissolution):
    """README is explicit that dissolution.json is not a claim graph."""
    for key in ("claims", "runs", "sources", "relationships"):
        assert key not in dissolution, f"dissolution.json unexpectedly has {key!r}"


def _models(dissolution):
    return sorted(dissolution["models"])


def test_dissolution_covers_the_expected_models(dissolution):
    assert _models(dissolution) == [
        "gpt2-medium",
        "gpt2-small",
        "noise-null",
        "pythia-160m",
        "pythia-410m",
        "pythia-410m-deep",
    ]


@pytest.fixture(params=[
    "gpt2-small",
    "gpt2-medium",
    "pythia-160m",
    "pythia-410m",
    "pythia-410m-deep",
    "noise-null",
])
def model(request, dissolution):
    models = dissolution["models"]
    if request.param not in models:
        pytest.skip(f"dissolution.json has no model {request.param!r}")
    return request.param, models[request.param]


def test_model_has_required_keys(model):
    name, data = model
    for key in (
        "label",
        "kind",
        "node_prefix",
        "source",
        "iterations",
        "terminal_iter",
        "basins",
        "basin_shares",
        "basin_count",
        "nodes",
        "edges",
    ):
        assert key in data, f"{name}: missing {key!r}"
    assert data["nodes"], f"{name}: no nodes"
    assert data["edges"], f"{name}: no edges"
    assert data["kind"] in {"model", "null-model"}


def test_model_has_no_duplicate_node_ids(model):
    name, data = model
    ids = [node["id"] for node in data["nodes"]]
    duplicates = sorted(i for i, n in Counter(ids).items() if n > 1)
    assert not duplicates, f"{name}: duplicate node ids {duplicates}"


def test_model_node_ids_encode_prefix_and_iteration(model):
    name, data = model
    prefix = data["node_prefix"]
    for node in data["nodes"]:
        parts = node["id"].split("|")
        assert len(parts) >= 3, f"{name}: malformed node id {node['id']!r}"
        assert parts[0] == prefix, f"{name}: node {node['id']!r} has the wrong prefix"
        assert int(parts[1]) == node["iter"], (
            f"{name}: node {node['id']!r} disagrees with its iter field {node['iter']}"
        )


def test_model_edges_reference_emitted_nodes(model):
    """Every edge endpoint must be a node that was actually emitted for this model."""
    name, data = model
    ids = {node["id"] for node in data["nodes"]}
    dangling = [
        (edge["from"], edge["to"])
        for edge in data["edges"]
        if edge["from"] not in ids or edge["to"] not in ids
    ]
    assert not dangling, f"{name}: edges pointing at nodes that were not emitted: {dangling[:10]}"


def test_model_edge_iteration_bands_are_monotonic(model):
    """Flow runs forward in iteration: to_iter must be strictly later than from_iter."""
    name, data = model
    bad = [
        (edge["from"], edge["to"], edge["from_iter"], edge["to_iter"])
        for edge in data["edges"]
        if not edge["to_iter"] > edge["from_iter"]
    ]
    assert not bad, f"{name}: edges that do not advance in iteration: {bad[:10]}"


def test_model_edge_bands_agree_with_their_endpoints(model):
    name, data = model
    nodes = {node["id"]: node for node in data["nodes"]}
    # Endpoints that do not resolve are test_model_edges_reference_emitted_nodes'
    # business; skip them here so a broken generator fails with that test's
    # message rather than a KeyError out of this one. pytest gives no ordering
    # guarantee, so this test cannot assume that one has already run.
    bad = []
    for edge in data["edges"]:
        source = nodes.get(edge["from"])
        target = nodes.get(edge["to"])
        if source is None or target is None:
            continue
        if source["iter"] != edge["from_iter"] or target["iter"] != edge["to_iter"]:
            bad.append(edge["from"] + " -> " + edge["to"])
    assert not bad, f"{name}: edge bands disagree with node iters: {bad[:10]}"


def test_model_iterations_are_observed_and_sorted(model, dissolution):
    name, data = model
    observed = set(dissolution["metadata"]["iterations_observed"])
    assert set(data["iterations"]) <= observed, (
        f"{name}: iterations not present in metadata.iterations_observed"
    )
    assert data["iterations"] == sorted(data["iterations"]), f"{name}: iterations unsorted"
    assert data["terminal_iter"] == max(data["iterations"]), (
        f"{name}: terminal_iter is not the last iteration"
    )


def test_model_node_iterations_are_declared(model):
    name, data = model
    declared = set(data["iterations"])
    bad = sorted({node["iter"] for node in data["nodes"] if node["iter"] not in declared})
    assert not bad, f"{name}: nodes at undeclared iterations {bad}"


def test_model_terminal_nodes_sit_at_the_terminal_iteration(model):
    name, data = model
    terminals = [node for node in data["nodes"] if node.get("terminal")]
    assert terminals, f"{name}: no terminal nodes"
    assert all(node["iter"] == data["terminal_iter"] for node in terminals), (
        f"{name}: a terminal node is not at terminal_iter"
    )


def test_model_terminal_nodes_are_sinks(model):
    name, data = model
    nodes = {node["id"]: node for node in data["nodes"]}
    # Same as above: an edge from an id that was never emitted is the dangling-
    # endpoint test's finding, not this one's, so skip it instead of raising.
    leaking = [
        edge["from"]
        for edge in data["edges"]
        if nodes.get(edge["from"], {}).get("terminal")
    ]
    assert not leaking, f"{name}: edges leaving a terminal node: {leaking[:5]}"


def test_model_basins_match_the_terminal_histogram(model):
    name, data = model
    histogram = {
        node["token"]: node["count"] for node in data["nodes"] if node.get("terminal")
    }
    assert histogram == data["basins"], (
        f"{name}: declared basins disagree with the terminal histogram"
    )
    assert data["basin_count"] == len(data["basins"]), f"{name}: basin_count is wrong"


def test_model_basin_shares_are_percentages_that_sum_to_100(model):
    name, data = model
    shares = data["basin_shares"]
    assert set(shares) == set(data["basins"]), f"{name}: basin_shares keys differ from basins"
    assert all(0 < value <= 100 for value in shares.values()), f"{name}: share out of range"
    total = sum(shares.values())
    assert abs(total - 100.0) <= 1.0, f"{name}: basin shares sum to {total}, not ~100"


def test_model_source_file_exists_on_disk(model):
    name, data = model
    assert (REPO_ROOT / data["source"]).exists(), (
        f"{name}: source table {data['source']} is not on disk"
    )


def test_dissolution_generated_from_paths_exist(dissolution):
    missing = [
        path
        for path in dissolution["metadata"]["generated_from"]
        if not (REPO_ROOT / path).exists()
    ]
    assert not missing, f"dissolution.json: generated_from paths not on disk: {missing}"


# ===========================================================================
# CONTENT: ground truth pinned from docs/FINDINGS.md
# ===========================================================================


def _claim(graph, claim_id):
    for claim in graph["claims"]:
        if claim["id"] == claim_id:
            return claim
    raise AssertionError(f"no claim with id {claim_id!r}")


def _edges(graph, edge_type=None, source=None, target=None):
    out = []
    for rel in graph["relationships"]:
        if edge_type is not None and rel["type"] != edge_type:
            continue
        if source is not None and rel["from"] != source:
            continue
        if target is not None and rel["to"] != target:
            continue
        out.append(rel)
    return out


def test_content_h_fingerprint_is_refuted_by_f3_and_f4(entities):
    """FINDINGS.md section 3: 'H-fingerprint | ... | Refuted as stated (F3, F4).'"""
    hypothesis = _claim(entities, "h-fingerprint")
    assert hypothesis["type"] == "hypothesis"
    assert hypothesis["status"] == "refuted"

    refuters = {rel["from"] for rel in _edges(entities, "refutes", target="h-fingerprint")}
    assert refuters == {"f3-fingerprint-refuted", "f4-null-model-regime"}, (
        f"H-fingerprint should be refuted by exactly F3 and F4, got {sorted(refuters)}"
    )
    for finding in refuters:
        assert _claim(entities, finding)["status"] == "supported"


def test_content_h_fingerprint_was_tested_before_it_was_refuted(entities):
    """The refutation has to be attached to the runs that produced it."""
    testers = {rel["from"] for rel in _edges(entities, "tests", target="h-fingerprint")}
    assert "run-2-cross-model-sweeps" in testers
    assert "run-3-random-noise-null" in testers


def test_content_f9_corrects_and_resolves_f2(entities):
    """F9 resolved the Divine anomaly and corrected F2's reading of it."""
    f9 = _claim(entities, "f9-divine-period-2")
    f2 = _claim(entities, "f2-divine-readout-stable")

    assert f9["status"] == "supported"
    assert f2["status"] == "corrected", "F2 must be marked corrected once F9 landed"

    corrections = {rel["to"] for rel in _edges(entities, "corrects", source="f9-divine-period-2")}
    assert "f2-divine-readout-stable" in corrections, "F9 must carry a corrects edge to F2"

    # and the resolution is recorded as period-2 machinery, not a fixed point
    supported = {rel["to"] for rel in _edges(entities, "supports", source="f9-divine-period-2")}
    assert "concept-period-2-limit-cycle" in supported


def test_content_every_corrected_claim_records_its_correction(entities):
    """`corrected` must be earned by a corrects/supersedes edge on the claim -
    incoming for a claim that was corrected by a later result, outgoing for a
    claim that carries the correction of its own earlier reading (the only case
    is `concept-brouwer-fixed-point`, whose JOURNEY_MAP relevance cell was
    corrected and which itself corrects Discovery 11)."""
    corrected = {c["id"] for c in entities["claims"] if c["status"] == "corrected"}
    recorded = set()
    for edge_type in ("corrects", "supersedes"):
        for rel in _edges(entities, edge_type):
            recorded.add(rel["to"])
            recorded.add(rel["from"])
    orphaned = sorted(corrected - recorded)
    assert not orphaned, f"claims marked corrected with no correction recorded: {orphaned}"


@pytest.mark.parametrize("name", EVIDENCE_FILES)
def test_no_description_repeats_a_sentence_verbatim(evidence_graphs, name):
    """Regression guard for the discovery-annotation duplication bug: when the
    *Corrected .../Retired ...* annotation sat in the Evidence cell rather than
    the Discovery cell, build_evidence_graph.parse_discoveries emitted it twice -
    once inline in the evidence prose and once again as the appended note."""
    graph = _graph(evidence_graphs, name)
    offenders = []
    for collection in ("claims", "runs", "sources"):
        for entity in graph[collection]:
            text = str(entity.get("description", ""))
            sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", text) if len(s.strip()) > 25]
            seen = set()
            for sentence in sentences:
                if sentence in seen:
                    offenders.append((entity["id"], sentence[:80]))
                    break
                seen.add(sentence)
    assert not offenders, f"{name}: descriptions repeating a sentence verbatim: {offenders}"


def test_content_every_refuted_claim_has_something_refuting_it(entities):
    refuted = {c["id"] for c in entities["claims"] if c["status"] == "refuted"}
    targets = {rel["to"] for rel in _edges(entities, "refutes")}
    orphaned = sorted(refuted - targets)
    assert not orphaned, f"claims marked refuted with nothing refuting them: {orphaned}"


def test_content_other_refuted_hypotheses_from_findings_section_3(entities):
    """FINDINGS.md section 3 also records H-till and H-supp as refuted."""
    assert _claim(entities, "h-till")["status"] == "refuted"
    assert _claim(entities, "h-supp")["status"] == "refuted"


def test_content_h_j1_is_not_supported_not_refuted_and_not_qualified(entities):
    """FINDINGS.md section 3: H-J1's disposition opens with the literal words
    "**Not supported at pilot confidence (2026-07-19); now phase-qualified (F16)**".

    This is a null result, not a contradiction: "the point estimate runs slightly
    the other way at pilot confidence", so "refuted" would overstate it. Nor is it
    partial survival of the stated claim - the "phase-qualified" clause names a
    different, phase-indexed claim raised by the later F16 re-probe, with the full
    build still pending (issue #8). It was filed as "qualified" only because the
    vocabulary had no null value; it now has one.
    """
    h_j1 = _claim(entities, "h-j1")
    assert h_j1["status"] == "not-supported"
    assert h_j1["status"] != "qualified", "regression: the pre-vocabulary compression"


def test_content_h3_stays_qualified_because_it_is_mixed_not_null(entities):
    """DELIBERATE: H3 is *not* "not-supported". Do not "fix" this.

    FINDINGS.md section 3 reads "**Weakened further at close; coherence half
    upgraded 2026-07-19**". Half of it failed - the all-warm matrix "was
    permutation-tested and found to be an anisotropy artifact" and "the
    corpus-causal reading had already failed cross-model (F3)". But the other
    half was strengthened, not merely retained: the semantic-coherence
    observation "no longer stands as qualitative only", holding "in the full
    readout distribution, with permutation support (coherence 0.41-0.47 vs 0.27,
    p = 0.001 under both nulls; F8)".

    A claim that lost one half and had the other upgraded is the textbook case
    for "qualified". "not-supported" would assert the evidence failed to back it,
    which the F8 permutation result directly contradicts.
    """
    assert _claim(entities, "h3-corpus-topology")["status"] == "qualified"


def test_content_not_supported_claims_are_only_the_ones_audited(entities):
    """The 2026-07 status-vocabulary sweep read all 12 dispositions in
    FINDINGS.md section 3 and all 33 findings, and moved exactly one claim.
    If a builder adds another, that is a judgement call and belongs in this pin.
    """
    nulls = sorted(c["id"] for c in entities["claims"]
                   if c["status"] == "not-supported")
    assert nulls == ["h-j1"], f"unaudited not-supported claims: {nulls}"


def test_content_f11_is_qualified_because_a_null_finding_still_stands(entities):
    """F11 records a null, but the *finding* is not "not-supported".

    The distinction the vocabulary draws is about how a claim stands, and F11 -
    "the restricted J-lens pilot did not support the prolet-inside/Divine-outside
    prediction" - is a real measured result that stands, narrowed by its own
    limits ("the probe saw phase A only"). The null it reports is carried by
    H-J1's status, not by F11's; marking both would double-count it.
    """
    assert _claim(entities, "f11-jlens-pilot-null")["status"] == "qualified"


def test_content_gpt2_medium_dissolution_is_a_single_token(dissolution):
    """FINDINGS.md F3: GPT-2 Medium -> '1 basin: `D` (100%)'."""
    medium = dissolution["models"]["gpt2-medium"]
    terminals = [node for node in medium["nodes"] if node.get("terminal")]

    assert len(terminals) == 1, (
        f"GPT-2 Medium should funnel to one terminal token, got "
        f"{[n['token'] for n in terminals]}"
    )
    assert terminals[0]["token"] == "D"
    assert medium["basin_count"] == 1
    assert medium["basin_shares"] == {"D": 100.0}


def test_content_gpt2_small_has_the_five_semantic_basins(dissolution):
    """FINDINGS.md F1/F3: GPT-2 Small -> five semantic basins, prolet dominant."""
    small = dissolution["models"]["gpt2-small"]
    assert set(small["basins"]) == {"prolet", "Divine", "Anarch", "till", "solidarity"}
    assert small["basin_count"] == 5
    dominant = max(small["basin_shares"], key=small["basin_shares"].get)
    assert dominant == "prolet"


def test_content_pythia_160m_collapses_to_questioned(dissolution):
    """FINDINGS.md F3: Pythia-160m -> '1 basin: `questioned` (94.4%)'."""
    model = dissolution["models"]["pythia-160m"]
    dominant = max(model["basin_shares"], key=model["basin_shares"].get)
    assert dominant == "questioned"
    assert model["basin_shares"][dominant] > 80


def test_content_pythia_410m_does_not_consolidate(dissolution):
    """FINDINGS.md F3: Pythia-410m -> 'no consolidation (40+ fragments)'."""
    model = dissolution["models"]["pythia-410m"]
    assert model["basin_count"] >= 20, "Pythia-410m should show a fragmented landscape"
    assert max(model["basin_shares"].values()) < 25, "no basin should dominate"


def test_content_noise_null_is_a_null_model_dominated_by_the_em_dash(dissolution):
    """FINDINGS.md F4: the random-tensor null is dominated by the em-dash token."""
    null = dissolution["models"]["noise-null"]
    assert null["kind"] == "null-model"
    dominant = max(null["basin_shares"], key=null["basin_shares"].get)
    assert dominant == "―", f"expected the horizontal-bar token, got {dominant!r}"
    assert "prolet" not in null["basins"], "the noise null must not reach the real basins"


def test_content_the_two_webtext_models_share_no_basin(dissolution):
    """F3's core point: same corpus, disjoint landscapes."""
    small = set(dissolution["models"]["gpt2-small"]["basins"])
    medium = set(dissolution["models"]["gpt2-medium"]["basins"])
    assert not (small & medium), (
        f"GPT-2 Small and Medium should share no terminal basin, shared: {small & medium}"
    )


def test_content_isomorphism_records_where_the_analogy_breaks(isomorphism):
    """The isomorphism graph is only honest if it carries its own failures."""
    breaks = _edges(isomorphism, "breaks-down-at")
    assert breaks, "isomorphism.json must record at least one breaks-down-at edge"
    for rel in breaks:
        assert str(rel.get("description", "")).strip(), (
            "a breaks-down-at edge must state the reason the analogy fails"
        )


# ===========================================================================
# Questions: the open work the record names, and what gates it
# ===========================================================================


def _questions(graph):
    return [claim for claim in graph["claims"] if claim["type"] == "question"]


def _dependency_edges(graph):
    return [rel for rel in graph["relationships"] if rel["type"] in DEPENDENCY_EDGES]


def test_questions_are_exactly_the_pinned_set(entities):
    """A question node must be traceable to a sentence in the record that leaves
    something open. If the builder invents one, this fails."""
    found = sorted(claim["id"] for claim in _questions(entities))
    assert found == EXPECTED_QUESTIONS, (
        f"question set drifted from the audited list; added: "
        f"{sorted(set(found) - set(EXPECTED_QUESTIONS))}, "
        f"removed: {sorted(set(EXPECTED_QUESTIONS) - set(found))}"
    )


def test_every_question_id_is_prefixed(entities):
    for claim in _questions(entities):
        assert claim["id"].startswith("q-"), f"{claim['id']} is not a q- id"


def test_every_question_uses_the_question_status_slice(entities):
    bad = [
        (claim["id"], claim["status"])
        for claim in _questions(entities)
        if claim["status"] not in QUESTION_STATUSES
    ]
    assert not bad, f"questions with a status outside {sorted(QUESTION_STATUSES)}: {bad}"


def test_every_open_question_is_actually_open(entities):
    """All 12 stand open at the close; none has been answered since."""
    assert all(claim["status"] == "open" for claim in _questions(entities))
    assert all(claim.get("retired") is None for claim in _questions(entities))


def test_open_is_no_longer_a_dead_work_signal(entities):
    """THE POINT OF THE `question` TYPE.

    Before it existed, every claim with status `open` was a concept - `open` is
    what a concept falls into when nobody has an epistemic verdict to record - so
    filtering on `open` returned vocabulary and no work. Now `type == "question"
    and status == "open"` is a real work queue, and `open` means something for a
    claim that is not a concept.
    """
    open_claims = [c for c in entities["claims"] if c["status"] == "open"]
    types = {c["type"] for c in open_claims}
    assert "question" in types, "no open questions: `open` is a dead signal again"
    assert types == {"concept", "question"}, (
        f"unexpected types carrying status `open`: {sorted(types)}"
    )
    work = [c for c in open_claims if c["type"] == "question"]
    assert len(work) == len(EXPECTED_QUESTIONS)


def test_every_question_quotes_the_record(entities):
    """The description must carry the sentence that leaves the matter open, so a
    reader can check the node against the document without leaving the graph."""
    for claim in _questions(entities):
        description = claim["description"]
        assert '"' in description, (
            f"{claim['id']}: description quotes nothing from the record"
        )
        assert len(description) > 120, f"{claim['id']}: description is too thin to audit"


def test_every_question_points_at_the_passage_it_came_from(entities):
    for claim in _questions(entities):
        doc_ref = claim.get("doc_ref", "")
        assert doc_ref.startswith(("docs/FINDINGS.md#", "docs/JOURNEY_MAP.md#")), (
            f"{claim['id']}: doc_ref {doc_ref!r} does not point into the record"
        )


def test_every_question_is_wired_to_something(entities):
    """A question with no edges is a note, not a graph node: it must relate to
    the claim that raised it, or gate something, or be gated."""
    for claim in _questions(entities):
        wired = [
            rel
            for rel in entities["relationships"]
            if rel["type"] in DEPENDENCY_EDGES | {"relates-to"}
            and claim["id"] in (rel["from"], rel["to"])
        ]
        assert wired, f"{claim['id']} has no relates-to or dependency edge"


def test_the_prompt_library_is_a_shared_blocker(entities):
    """FINDINGS F10 and F15 sit in different sections, never mention each other,
    and are blocked on the same artefact:

        F10  "Open: whether all 34 Divine prompts share this flip axis
              (blocked on the prompt-library restoration, issue #9)."
        F15  "The other 33 period-2 prompts remain blocked on the prompt
              library (issue #9)."

    Read linearly you would never pair them. In the graph they are two edges
    into one node, so unblocking issue #9 visibly frees two threads.
    """
    blocked = {
        rel["from"]
        for rel in entities["relationships"]
        if rel["type"] == "blocked-by" and rel["to"] == "q-prompt-library"
    }
    assert blocked == {"q-flip-axis-generality", "q-lag2-regate-33"}, (
        f"issue #9 should gate exactly the two threads the record blocks on it, "
        f"got {sorted(blocked)}"
    )


def test_the_jlens_full_build_blocks_h_j1(entities):
    """H-J1's disposition ends "Full build still pending (issue #8)", so the
    hypothesis cannot move past its pilot-confidence null until that is built."""
    blocked = {
        rel["to"]
        for rel in entities["relationships"]
        if rel["type"] == "blocks" and rel["from"] == "q-jlens-full-build"
    }
    assert "h-j1" in blocked
    assert _claim(entities, "h-j1")["status"] == "not-supported"


def test_the_fractal_question_is_gated_on_t_mix(entities):
    """JOURNEY_MAP section 7 states the order outright: "Requires T_mix first"."""
    gated = {
        rel["to"]
        for rel in entities["relationships"]
        if rel["type"] == "blocked-by" and rel["from"] == "q-fractal-dimension"
    }
    assert gated == {"q-tmix-llm"}


def test_gate_cadence_gates_nothing(entities):
    """DELIBERATE: caveat 5's finer gate cadence has no blocks edge.

    FINDINGS section 6 files it as the series' one declared debt and bounds it in
    the same breath: "It cannot overturn a principal finding: basin identities
    stand on the gate regardless of cadence." Open work that gates nothing is a
    real category, and inventing a dependant for it would misreport the record.
    """
    gates = [
        rel
        for rel in _dependency_edges(entities)
        if (rel["type"] == "blocks" and rel["from"] == "q-gate-cadence")
        or (rel["type"] == "blocked-by" and rel["to"] == "q-gate-cadence")
    ]
    assert not gates, f"q-gate-cadence should block nothing, got {gates}"


def test_no_dependency_is_drawn_from_both_ends(entities):
    """`blocks` and `blocked-by` are one relation read two ways. Drawing both for
    a pair would double the dependency count and make one stated blocker look
    like two."""
    pairs = set()
    for rel in _dependency_edges(entities):
        if rel["type"] == "blocks":
            pairs.add((rel["from"], rel["to"]))
        else:
            pairs.add((rel["to"], rel["from"]))
    assert len(pairs) == len(_dependency_edges(entities)), (
        "a dependency is drawn from both ends"
    )


def test_dependency_edges_only_ever_involve_open_work(entities):
    """Something has to be open at one end, or it is not a blocker."""
    by_id = {claim["id"]: claim for claim in entities["claims"]}
    for rel in _dependency_edges(entities):
        blocker = rel["from"] if rel["type"] == "blocks" else rel["to"]
        node = by_id.get(blocker)
        assert node is not None, f"{blocker} is not a claim"
        assert node["type"] == "question" and node["status"] == "open", (
            f"{blocker} blocks something but is not open work"
        )


def test_visual_config_styles_the_question_type_and_its_edges():
    config = _load("visual_config.json")
    assert config["colors"]["question"] == "#A8477A"
    assert config["shapes"]["question"] == "circle"
    for edge in sorted(DEPENDENCY_EDGES):
        style = config["edge_styles"][edge]
        assert style["color"] == config["colors"]["question"], (
            f"{edge} should read as part of the question family"
        )
        assert style["arrow"] is True, f"{edge} must be directed"
    assert config["edge_styles"]["blocks"] == config["edge_styles"]["blocked-by"]


# ===========================================================================
# Status vocabulary: the builder, the config it emits and the viewer must agree
# ===========================================================================


def test_visual_config_status_colours_match_the_canonical_map():
    """visual_config.json is what all three graphs read at load time."""
    config = _load("visual_config.json")
    assert config.get("status_colors") == STATUS_COLOURS


def test_visual_config_status_colours_are_in_canonical_order():
    """Dict order is preserved through JSON and drives the emitted legend."""
    config = _load("visual_config.json")
    assert list(config["status_colors"]) == STATUS_ORDER + ["default"]


def test_viewer_status_order_matches_the_canonical_order():
    """viewer.html hardcodes STATUS_ORDER as its legend and filter-chip order."""
    viewer = (REPO_ROOT / "docs" / "graph" / "viewer.html").read_text(encoding="utf-8")
    match = re.search(r"const STATUS_ORDER = \[(.*?)\];", viewer, re.S)
    assert match, "viewer.html must declare a STATUS_ORDER array"
    order = re.findall(r"'([^']+)'", match.group(1))
    assert order == STATUS_ORDER


def test_viewer_status_colours_match_the_canonical_map():
    """The viewer's built-in defaults must not disagree with visual_config.json;
    a graph that fails to load the config should still colour correctly."""
    viewer = (REPO_ROOT / "docs" / "graph" / "viewer.html").read_text(encoding="utf-8")
    match = re.search(r"const STATUS_COLORS = \{(.*?)\};", viewer, re.S)
    assert match, "viewer.html must declare a STATUS_COLORS object"
    pairs = re.findall(r'"([^"]+)":\s*"(#[0-9A-Fa-f]{6})"', match.group(1))
    assert dict(pairs) == STATUS_COLOURS
    assert [name for name, _ in pairs] == STATUS_ORDER + ["default"]


def test_not_supported_colour_is_perceptually_distinct():
    """#8F5A57 must not be confusable at a glance with any other status swatch.
    Nearest neighbours are refuted (dE76 ~28.8) and retired (~30.4)."""
    def _lab(hex_colour):
        rgb = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        r, g, b = lin
        xyz = (
            (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047,
            (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000,
            (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883,
        )
        f = [t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29
             for t in xyz]
        return (116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2]))

    target = _lab(STATUS_COLOURS["not-supported"])
    for name, colour in STATUS_COLOURS.items():
        if name == "not-supported":
            continue
        other = _lab(colour)
        delta = sum((a - b) ** 2 for a, b in zip(target, other)) ** 0.5
        assert delta >= 20.0, (
            f"not-supported {STATUS_COLOURS['not-supported']} is only dE76 "
            f"{delta:.1f} from {name} {colour}; pick a more distinct value"
        )
