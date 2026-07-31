#!/usr/bin/env python3
"""Build the ATR threads-and-opportunities report.

The evidence graph in ``docs/graph/_data/entities.json`` is built by parsing
``docs/FINDINGS.md`` and ``docs/JOURNEY_MAP.md``.  It is therefore, by
construction, exactly as current as those two documents and no more.  This
script reads the graph and then goes and looks at the filesystem, which is
where the two can be caught disagreeing.

It emits two files:

    docs/graph/_data/threads.json   machine-readable, one record per thread
    docs/graph/THREADS.md           the human-readable report

Six detectors, each of which must state its evidence rather than its verdict:

    A  answered-but-unrecorded  a claim carrying a `tests` edge but no verdict
                                edge, cross-checked against the run's script,
                                its executed outputs and its artefacts on disk
    B  blocked, grouped by blocker, with a disk check on whether each blocker
       still holds
    C  needs-compute vs answerable-from-disk, for every open thread
    D  the frontier: findings with nothing downstream of them yet
    E  undeveloped: concepts of degree <= 1, claims with no epistemic edges
    F  unreferenced artefacts: files on disk the graph has never heard of

Takes no arguments.  Idempotent: the outputs carry no wall-clock stamp, so a
second run over unchanged inputs rewrites byte-identical files.  Exits non-zero
only on a real error (a missing or unreadable entities.json), never merely
because it found something.

Usage:  python docs/graph/build_threads_report.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
ENTITIES = os.path.join(HERE, "_data", "entities.json")
OUT_JSON = os.path.join(HERE, "_data", "threads.json")
OUT_MD = os.path.join(HERE, "THREADS.md")

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

VERDICT_EDGES = {"supports", "refutes", "qualifies", "corrects"}
DOWNSTREAM_EDGES = {"corrects", "supersedes", "qualifies", "builds-on"}
EPISTEMIC_EDGES = {
    "supports", "refutes", "qualifies", "corrects", "retires",
    "supersedes", "tests", "builds-on", "evidenced-by",
}

# Pending-work language, matched against claim descriptions.  Deliberately
# literal: every phrase here is one a human wrote into the record meaning
# "this is not finished".
PENDING_PATTERNS = [
    r"pending", r"not run", r"untested", r"awaits?\b", r"unexecuted",
    r"not attempted", r"still not", r"issue #\d+", r"scaffold",
    r"not measured", r"finer cadence", r"blocked on", r"never run",
    r"not yet run", r"remains open", r"is open\b",
]
PENDING_RE = re.compile("|".join(PENDING_PATTERNS), re.IGNORECASE)

# Blocker phrasing.  Each pattern captures the thing being waited on.
ISSUE_RE = re.compile(r"issue #(\d+)", re.IGNORECASE)
BLOCKER_PATTERNS = [
    ("blocked-on", re.compile(r"blocked on (?:the )?([^.,;()]+)", re.IGNORECASE)),
    ("awaits", re.compile(r"awaits? (?:the )?([^.,;()]+)", re.IGNORECASE)),
    ("pending", re.compile(r"(?:still )?pending \(?(?:the )?([^.,;()]+)",
                           re.IGNORECASE)),
]

# The record says in as many words that a piece of work has never been done.
NEVER_RUN_RE = re.compile(
    r"never run|not (?:yet )?run|not attempted|still not|unexecuted|"
    r"not measured|not been run|untested|unresolved|unbuilt|"
    r"new experimental stage|requires? (?:a )?new (?:run|stage|experiment)|"
    r"never comput|not (?:yet )?comput|has comput\w* yet|computed yet",
    re.IGNORECASE)

# The record says in as many words that the inputs are already committed.
ON_DISK_RE = re.compile(
    r"already on disk|from existing data|measurable from existing|"
    r"compute from [`']?\.?pt|data (?:is |are )?already (?:on disk|committed)|"
    r"inputs are (?:already )?committed", re.IGNORECASE)

# Text in a notebook's own output that looks like it is announcing a result.
VERDICT_TEXT_RE = re.compile(
    r"\b(NOT SUPPORTED|SUPPORTED|NOT CONFIRMED|CONFIRMED|REFUTED|"
    r"INCONCLUSIVE|VALIDATION SUMMARY|VERDICT|RESULT:|CONCLUSION)\b"
)

# Prose that asserts a script has never been executed.
NOT_RUN_RE = re.compile(
    r"not (?:yet )?(?:been )?run|never (?:been )?(?:executed|run)|"
    r"not executed|scaffold only|scaffolded[^.]{0,40}not run|"
    r"unexecuted|code only",
    re.IGNORECASE,
)
# Prose that asserts a script has been executed.
WAS_RUN_RE = re.compile(
    r"executed end to end|executed end\b|was executed|ran end to end|"
    r"re-execution|regenerat",
    re.IGNORECASE,
)
# Prose that reports an obstacle RESOLVED rather than standing. A sentence
# carrying one of these does not register the issue numbers it cites as live
# blockers: "restored (issue #24)" is a receipt, not an obstacle. The words
# are broad on purpose; the cost of skipping a live blocker whose sentence
# happens to contain "executed" is far below the cost of a resolved issue
# haunting the still-blocked list forever.
RESOLVED_RE = re.compile(
    r"\bAnswered\b|\bexecuted\b|\brestored\b|\bsuperseded\b|\bresolved\b|"
    r"\bruled\b|\bmerged\b|\bdelivered\b|\bwithdrawn\b",
    re.IGNORECASE,
)

DATA_EXT = (".pt", ".pth", ".json", ".npy", ".npz", ".csv", ".md", ".png")
ARTEFACT_EXT = (".pt", ".pth", ".json", ".npy", ".npz", ".csv")
SCRIPT_EXT = (".py", ".ipynb")

STOPWORDS = {"the", "a", "an", "its", "full", "restoration", "of", "for", "on"}

# --------------------------------------------------------------------------
# The ranking rule.  Printed into the report so the ordering is auditable.
# --------------------------------------------------------------------------

RANK_RULE = [
    ("answered-on-disk",
     "The question has already been answered on disk and only the record is "
     "missing: the run's script exists, carries executed outputs or committed "
     "artefacts, and the graph still shows no verdict. Cost: read the numbers "
     "and write them down."),
    ("newly-unblocked",
     "The claim names a blocker, and the blocker no longer appears to hold on "
     "disk. Cost: resume work that was correctly parked."),
    ("answerable-from-disk",
     "The inputs are committed but the answer has not been extracted: either "
     "the script exists with every declared input present and has simply not "
     "been executed, or the record itself says the data is already on disk and "
     "the artefacts are where it says they are. Cost: analysis, not a model "
     "run."),
    ("needs-compute",
     "A fresh model run is required: a script exists but its inputs do not, or "
     "no script exists at all. Cost: new compute, possibly new code."),
    ("still-blocked",
     "The named blocker still holds on disk. Cost: unblock first."),
    ("cannot-determine",
     "The graph names no run, script or artefact for this thread, so no honest "
     "disk verdict is available. Cost: unknown, and that is the finding."),
]
RANK_ORDER = {name: i for i, (name, _) in enumerate(RANK_RULE)}


# --------------------------------------------------------------------------
# Small filesystem helpers
# --------------------------------------------------------------------------

def rel(path: str) -> str:
    """Repo-relative, forward-slashed."""
    return os.path.relpath(path, REPO).replace(os.sep, "/")


def abspath(path: str) -> str:
    return os.path.join(REPO, path.replace("/", os.sep))


def exists(path: str) -> bool:
    return os.path.exists(abspath(path))


def walk_repo(subdir: str, exts):
    """Every file under ``subdir`` with one of ``exts``, repo-relative, sorted."""
    out = []
    root_dir = abspath(subdir)
    if not os.path.isdir(root_dir):
        return out
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__",
                                                      ".ipynb_checkpoints"})
        for name in sorted(files):
            if name.endswith(tuple(exts)):
                out.append(rel(os.path.join(root, name)))
    return sorted(out)


def read_text(path: str) -> str:
    try:
        with open(abspath(path), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def file_size(path: str):
    try:
        return os.path.getsize(abspath(path))
    except OSError:
        return None


# --------------------------------------------------------------------------
# Notebook inspection
# --------------------------------------------------------------------------

def output_text(out: dict) -> str:
    """Flatten one notebook output to text."""
    chunks = []
    for key in ("text",):
        val = out.get(key)
        if isinstance(val, list):
            chunks.append("".join(val))
        elif isinstance(val, str):
            chunks.append(val)
    data = out.get("data") or {}
    val = data.get("text/plain")
    if isinstance(val, list):
        chunks.append("".join(val))
    elif isinstance(val, str):
        chunks.append(val)
    if out.get("output_type") == "error":
        chunks.append(out.get("ename", "") + ": " + out.get("evalue", ""))
    return "\n".join(c for c in chunks if c)


def inspect_notebook(path: str) -> dict:
    """Execution state of a .ipynb, and any result-looking text in its outputs.

    'Executed' means: the cell carries a non-empty `outputs` array.
    """
    info = {
        "kind": "notebook",
        "readable": False,
        "cells_total": None,
        "code_cells": None,
        "cells_with_outputs": None,
        "executed": None,
        "verdict_quotes": [],
        "banner": None,
        "saves": [],
        "loads": [],
    }
    try:
        with open(abspath(path), "r", encoding="utf-8") as fh:
            nb = json.load(fh)
    except (OSError, ValueError):
        return info
    cells = nb.get("cells") or []
    info["readable"] = True
    info["cells_total"] = len(cells)
    info["code_cells"] = sum(1 for c in cells if c.get("cell_type") == "code")
    with_out = [i for i, c in enumerate(cells) if c.get("outputs")]
    info["cells_with_outputs"] = len(with_out)
    info["executed"] = len(with_out) > 0

    source_all = []
    for idx, cell in enumerate(cells):
        src = cell.get("source")
        src = "".join(src) if isinstance(src, list) else (src or "")
        if cell.get("cell_type") == "code":
            source_all.append(src)
        if idx == 0 and cell.get("cell_type") == "markdown":
            head = " ".join(src.split())
            info["banner"] = head[:400]
        for out in cell.get("outputs") or []:
            text = output_text(out).strip()
            if not text or not VERDICT_TEXT_RE.search(text):
                continue
            lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
            info["verdict_quotes"].append({"cell": idx, "lines": lines[:14]})

    joined = "\n".join(source_all)
    info["saves"] = extract_paths(joined, save=True)
    info["loads"] = extract_paths(joined, save=False)
    return info


def inspect_script(path: str) -> dict:
    src = read_text(path)
    return {
        "kind": "script",
        "readable": bool(src),
        "cells_total": None,
        "code_cells": None,
        "cells_with_outputs": None,
        "executed": None,          # a .py leaves no trace of having been run
        "verdict_quotes": [],
        "banner": None,
        "saves": extract_paths(src, save=True),
        "loads": extract_paths(src, save=False),
    }


LOAD_CALL_RE = re.compile(
    r"(?:torch\.load|np\.load|numpy\.load|json\.load|pd\.read_csv|"
    r"pd\.read_json|open)\s*\(", re.IGNORECASE)
SAVE_CALL_RE = re.compile(
    r"(?:torch\.save|np\.save|numpy\.save|json\.dump|savefig|write_image|"
    r"to_csv)\s*\(", re.IGNORECASE)
STRING_RE = re.compile(r"""['"]([^'"\n]+)['"]""")

# Three things a string ending in ".json" can be without being a filename.
NOT_A_FILENAME_RE = re.compile(r"[\s{}\\]")


def is_filename_literal(literal: str) -> bool:
    """Is this string literal a path, or merely a string that ends like one?

    The 400-character window after a ``torch.load(`` / ``json.dump(`` catches
    every string literal in the neighbourhood, not just the call's argument, so
    it also catches log lines and unformatted templates. Two got all the way
    into the published report as *declared inputs* whose absence then drove
    needs-compute verdicts: ``"\\nsaved bell_anatomy.json"`` (a print, not a
    path) and ``"state_{key}.pt"`` (an f-string template whose real filename is
    not knowable without running the code).

    Three disqualifiers, each of which means "not a path as written":

      whitespace   prose around the name, or a whole sentence containing one
      a backslash  an escape sequence, so the literal is display text
      braces       an unexpanded ``{}`` placeholder

    Rejecting is the conservative direction. A file we decline to call a
    declared input is at worst a thread this report says less about; a
    non-existent one we do call a declared input is this report telling the
    operator that committed work cannot proceed, wrongly.
    """
    return bool(literal) and not NOT_A_FILENAME_RE.search(literal)


def extract_paths(source: str, save: bool):
    """String literals that look like data files, inside load or save calls.

    A blunt instrument, and honest about it: it reports basenames, and the
    caller resolves them against the repository rather than trusting the
    literal path (the notebooks use relative paths such as ``../_DATA``).
    """
    call_re = SAVE_CALL_RE if save else LOAD_CALL_RE
    found = []
    for match in call_re.finditer(source):
        window = source[match.end(): match.end() + 400]
        for lit in STRING_RE.findall(window):
            if lit.endswith(DATA_EXT) and is_filename_literal(lit):
                found.append(os.path.basename(lit))
    # os.path.join(dir, "name.pt") splits the basename off; the loop above
    # already catches the literal, since it scans the whole window.
    return sorted(set(found))


LOCAL_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", re.M)


def missing_local_imports(path: str, repo_modules) -> list:
    """Modules imported by a script that look local to this repo but are absent.

    A module counts as 'local' if the repository record names it: it is
    mentioned by another committed file. This is what catches
    ``import prompt_library`` in a tree where prompt_library.py is gone.
    """
    if path.endswith(".ipynb"):
        try:
            with open(abspath(path), "r", encoding="utf-8") as fh:
                nb = json.load(fh)
        except (OSError, ValueError):
            return []
        src = "\n".join(
            "".join(c.get("source") or []) for c in nb.get("cells") or []
            if c.get("cell_type") == "code")
    else:
        src = read_text(path)
    missing = []
    for mod in sorted(set(LOCAL_IMPORT_RE.findall(src))):
        if mod not in repo_modules:
            continue
        if not (exists(mod + ".py")
                or exists(os.path.join(os.path.dirname(path), mod + ".py"))):
            missing.append(mod + ".py")
    return sorted(set(missing))


# --------------------------------------------------------------------------
# Graph loading
# --------------------------------------------------------------------------

def load_graph():
    if not os.path.exists(ENTITIES):
        sys.stderr.write(
            "build_threads_report: cannot find the evidence graph at %s\n"
            "Run docs/graph/build_evidence_graph.py first.\n" % rel(ENTITIES))
        raise SystemExit(1)
    try:
        with open(ENTITIES, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except ValueError as exc:
        sys.stderr.write("build_threads_report: %s is not valid JSON: %s\n"
                         % (rel(ENTITIES), exc))
        raise SystemExit(1)
    for key in ("claims", "runs", "relationships"):
        if key not in data:
            sys.stderr.write("build_threads_report: %s has no '%s' array\n"
                             % (rel(ENTITIES), key))
            raise SystemExit(1)
    return data


class Graph:
    def __init__(self, data):
        self.data = data
        self.meta = data.get("metadata", {})
        self.claims = {c["id"]: c for c in data["claims"]}
        self.runs = {r["id"]: r for r in data["runs"]}
        self.sources = {s["id"]: s for s in data.get("sources", [])}
        self.rels = data["relationships"]
        self.nodes = {}
        self.nodes.update(self.claims)
        self.nodes.update(self.runs)
        self.nodes.update(self.sources)
        self.incoming = defaultdict(list)
        self.outgoing = defaultdict(list)
        self.degree = Counter()
        for rel_ in self.rels:
            self.incoming[rel_["to"]].append(rel_)
            self.outgoing[rel_["from"]].append(rel_)
            self.degree[rel_["to"]] += 1
            self.degree[rel_["from"]] += 1
        self.blob = json.dumps(data)

    def label(self, node_id):
        node = self.nodes.get(node_id)
        if not node:
            return node_id
        return node.get("label") or node.get("title") or node_id

    def source_doc(self, claim):
        ref = claim.get("doc_ref") or ""
        return ref.split("#", 1)[0] or "unknown"


# --------------------------------------------------------------------------
# Run-on-disk evidence, shared by several detectors
# --------------------------------------------------------------------------

def run_disk_evidence(graph: Graph, run_id: str, repo_modules) -> dict:
    run = graph.runs.get(run_id) or {}
    script = run.get("script")
    out_dir = run.get("output_dir") or run.get("output_path")
    ev = {
        "run": run_id,
        "run_label": run.get("label", run_id),
        "graph_says": run.get("n") or run.get("description", "")[:160],
        "date": run.get("date"),
        "script": script,
        "script_exists": exists(script) if script else None,
        "script_state": None,
        "output_dir": out_dir,
        "output_dir_exists": exists(out_dir) if out_dir else None,
        "output_dir_files": [],
        "declared_inputs": [],
        "missing_inputs": [],
        "saved_artefacts": [],
        "missing_local_modules": [],
    }

    if script and ev["script_exists"]:
        ev["script_state"] = (inspect_notebook(script) if script.endswith(".ipynb")
                              else inspect_script(script))
        ev["missing_local_modules"] = missing_local_imports(script, repo_modules)

        index = basename_index()
        for name in ev["script_state"]["loads"]:
            hits = index.get(name, [])
            ev["declared_inputs"].append({"name": name, "found": hits[:3]})
            if not hits:
                ev["missing_inputs"].append(name)
        for name in ev["script_state"]["saves"]:
            hits = index.get(name, [])
            ev["saved_artefacts"].append({
                "name": name,
                "found": hits[:3],
                "size": file_size(hits[0]) if hits else None,
            })

    if out_dir and ev["output_dir_exists"]:
        full = abspath(out_dir)
        if os.path.isdir(full):
            ev["output_dir_files"] = sorted(
                rel(os.path.join(full, f)) for f in os.listdir(full)
                if os.path.isfile(os.path.join(full, f)))

    # Artefacts the graph itself attributes to this run.
    ev["graph_artefacts"] = []
    for edge in graph.incoming.get(run_id, []):
        if edge["type"] != "produced-by":
            continue
        src = graph.sources.get(edge["from"])
        if not src:
            continue
        ev["graph_artefacts"].append({
            "id": src["id"], "path": src.get("path"),
            "exists": exists(src["path"]) if src.get("path") else None,
        })
    return ev


_BASENAME_INDEX = None


def basename_index():
    """basename -> [repo-relative paths].  Built once, over the whole tree."""
    global _BASENAME_INDEX
    if _BASENAME_INDEX is not None:
        return _BASENAME_INDEX
    index = defaultdict(list)
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__",
                                                ".ipynb_checkpoints", ".pytest_cache"}]
        for name in files:
            index[name].append(rel(os.path.join(root, name)))
    for key in index:
        index[key].sort()
    _BASENAME_INDEX = dict(index)
    return _BASENAME_INDEX


_MD_INDEX = None


def markdown_index():
    """Every markdown line in the repo, outside docs/graph, as (path, no, text)."""
    global _MD_INDEX
    if _MD_INDEX is not None:
        return _MD_INDEX
    lines = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__",
                                                ".ipynb_checkpoints", ".pytest_cache"}]
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            path = rel(os.path.join(root, name))
            if path.startswith("docs/graph/"):
                continue          # our own output; not evidence about the work
            for no, text in enumerate(read_text(path).splitlines(), start=1):
                if text.strip():
                    lines.append((path, no, text.strip()))
    _MD_INDEX = lines
    return _MD_INDEX


def prose_mentions(needle: str, matcher: re.Pattern, limit=6):
    """Markdown lines mentioning `needle` that also match `matcher`."""
    hits = []
    for path, no, text in markdown_index():
        if path == "docs/PODCAST_SOURCES.md":
            continue  # dated snapshot bundle; its mirrored prose is not the record
        if needle in text and matcher.search(text):
            hits.append({"file": path, "line": no,
                         "text": " ".join(text.split())[:300]})
            if len(hits) >= limit:
                break
    return hits


def repo_modules_set():
    """Module names the repository's own files import or mention.

    Any top-level .py in the repository root is unambiguously a local module
    name, and so is any module name the README's structure block advertises,
    including ones the tree no longer contains.
    """
    mods = set()
    for name in os.listdir(REPO):
        if name.endswith(".py"):
            mods.add(name[:-3])
    for match in re.finditer(r"`?([a-z_][a-z0-9_]*)\.py`?", read_text("README.md")):
        mods.add(match.group(1))
    return mods


# --------------------------------------------------------------------------
# DETECTOR A: answered but unrecorded
# --------------------------------------------------------------------------

def detect_answered_but_unrecorded(graph: Graph, repo_modules):
    results = []
    for claim_id, claim in sorted(graph.claims.items()):
        testers = [e for e in graph.incoming.get(claim_id, []) if e["type"] == "tests"]
        if not testers:
            continue
        verdicts = [e for e in graph.incoming.get(claim_id, [])
                    if e["type"] in VERDICT_EDGES]
        if verdicts:
            continue
        # A recorded disposition IS the record's verdict for a hypothesis.
        # This detector exists for claims the record has fallen behind on,
        # not for ones the operator has ruled on: once the status leaves
        # "untested", the gap this detector reports is closed (H4 / issue
        # #54 was the motivating case in both directions). Accepted trade:
        # a ruling that a LATER run quietly contradicts will not resurface
        # here; that case is the drift check's to catch, since it compares
        # the record's prose against the disk without consulting statuses.
        if claim.get("type") == "hypothesis" and claim.get("status") not in (
                None, "untested", "open"):
            continue

        record = {
            "claim": claim_id,
            "label": claim.get("label"),
            "type": claim.get("type"),
            "status": claim.get("status"),
            "description": claim.get("description"),
            "doc_ref": claim.get("doc_ref"),
            "tested_by": [],
            "verdict_edges": [],
            "disk_evidence": [],
            "quoted_result": [],
            "prose_says_not_run": [],
            "prose_says_was_run": [],
            "assessment": None,
        }
        for edge in testers:
            record["tested_by"].append({
                "id": edge["from"], "label": graph.label(edge["from"]),
                "kind": "run" if edge["from"] in graph.runs else "claim",
                "edge_note": edge.get("description"),
            })

        executed_anywhere = False
        for edge in testers:
            if edge["from"] not in graph.runs:
                continue
            ev = run_disk_evidence(graph, edge["from"], repo_modules)
            record["disk_evidence"].append(ev)
            state = ev.get("script_state") or {}
            if state.get("executed"):
                executed_anywhere = True
                for quote in state.get("verdict_quotes", []):
                    record["quoted_result"].append({
                        "script": ev["script"], "cell": quote["cell"],
                        "lines": quote["lines"],
                    })
            if ev["script"]:
                base = os.path.basename(ev["script"])
                record["prose_says_not_run"] += prose_mentions(base, NOT_RUN_RE)
                record["prose_says_was_run"] += prose_mentions(base, WAS_RUN_RE)
            for art in ev.get("saved_artefacts", []):
                if art["found"]:
                    executed_anywhere = True

        if executed_anywhere and record["quoted_result"]:
            record["assessment"] = "answered-on-disk-unrecorded-in-graph"
        elif executed_anywhere:
            record["assessment"] = "artefacts-on-disk-verdict-not-legible"
        elif record["disk_evidence"]:
            record["assessment"] = "genuinely-unrun"
        else:
            record["assessment"] = "no-run-node-cannot-determine"
        results.append(record)

    order = {"answered-on-disk-unrecorded-in-graph": 0,
             "artefacts-on-disk-verdict-not-legible": 1,
             "genuinely-unrun": 2, "no-run-node-cannot-determine": 3}
    results.sort(key=lambda r: (order.get(r["assessment"], 9), r["claim"]))
    return results


# --------------------------------------------------------------------------
# DETECTOR B: blockers, grouped by blocker
# --------------------------------------------------------------------------

def normalise_phrase(captured: str):
    """A stable key and a human label for one parsed blocker phrase."""
    text = " ".join(captured.split()).strip().strip("`'\"")
    if ISSUE_RE.fullmatch(text.strip()):
        return None, None            # an issue number, handled as its own key
    text = re.sub(r"\s+(restoration|rebuild|build)$", "", text, flags=re.I)
    words = [w for w in re.split(r"[^A-Za-z0-9_.#]+", text.lower()) if w]
    words = [w for w in words if w not in STOPWORDS]
    if not words:
        return None, None
    return "phrase-" + "-".join(words[:4]), text


def blocker_candidates(label: str):
    """File basenames a blocker phrase might name, each with where it came from.

    Returns ``[(basename, origin)]`` with ``origin`` one of:

      "written"  the phrase contains this filename verbatim -- "blocked on
                 `prompt_library.py`" -- so the record itself named the file.
      "guessed"  this tool synthesised the name out of prose words: "the prompt
                 library" -> ``prompt_library.py``. A useful lead and nothing
                 more; the record never wrote it.

    The two are kept apart because they are not the same quality of evidence,
    and ``resolve_blocker`` refuses to overturn the record on the second alone.
    """
    cands = []
    for match in re.finditer(r"[\w./-]+\.(?:py|ipynb|pt|json|md|csv)", label):
        cands.append((os.path.basename(match.group(0)), "written"))
    words = [w for w in re.split(r"[^A-Za-z0-9]+", label.lower())
             if w and w not in STOPWORDS]
    if words:
        stem = "_".join(words[:3])
        for ext in (".py", ".json", ".pt", ".md"):
            cands.append((stem + ext, "guessed"))
        if len(words) > 1:
            stem2 = "_".join(words[:2])
            for ext in (".py", ".json", ".pt", ".md"):
                cands.append((stem2 + ext, "guessed"))
    seen, out = set(), []
    for name, origin in cands:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, origin))
    return out


# Prose asserting that a named thing is not available here: the sentences a
# guessed filename must not be allowed to overrule on its own.
ABSENCE_PROSE_RE = re.compile(
    r"exists only on|only on [A-Z]\w+'s|home machine|local machine|"
    r"not committed|never committed|uncommitted|"
    r"temporarily absent|is absent|absent from|missing from|is missing|"
    r"not present|not in the repo|awaiting restoration|to be restored|"
    r"will be restored|yet to be restored|blocked on|blocked by",
    re.IGNORECASE,
)


def literal_mentions(name: str, limit: int = 3):
    """Committed markdown lines that write ``name`` out as a filename."""
    return [{"file": p, "line": n, "text": " ".join(t.split())[:300]}
            for p, n, t in markdown_index() if name in t][:limit]


def contrary_prose(name: str, limit: int = 3):
    """Committed lines that name ``name`` *and* assert it is not available."""
    return [{"file": p, "line": n, "text": " ".join(t.split())[:300]}
            for p, n, t in markdown_index()
            if name in t and ABSENCE_PROSE_RE.search(t)][:limit]


_TEXT_CORPUS = None


def text_corpus():
    """Every committed markdown and script, as one searchable blob.

    Used to decide whether a name the record treats as a file is a file this
    repository has ever known about, as opposed to a phrase this tool has
    guessed a filename from.
    """
    global _TEXT_CORPUS
    if _TEXT_CORPUS is not None:
        return _TEXT_CORPUS
    parts = []
    for path, _no, text in markdown_index():
        parts.append(text)
    for path in walk_repo("experiments", SCRIPT_EXT) + walk_repo("viz", (".py",)):
        parts.append(read_text(path))
    for name in sorted(os.listdir(REPO)):
        if name.endswith(".py"):
            parts.append(read_text(name))
    _TEXT_CORPUS = "\n".join(parts)
    return _TEXT_CORPUS


def resolve_blocker(group) -> dict:
    """Does this blocker still appear to hold?  Evidence first, verdict after.

    Three outcomes, and the third is not a failure:
      False  a file the *record* names as the blocker is present in the tree
      True   the artefact is named by the repository but absent from it
      None   nothing checkable: the blocker is an issue number, a phrase this
             tool cannot responsibly turn into a filename, or a filename it
             only guessed at

    HOW STRONG THE EVIDENCE HAS TO BE
    ---------------------------------

    Turning ``holds`` to False contradicts the record in the operator's own
    document, so the bar is set where the evidence is actually load-bearing.
    Every resolution therefore carries an ``evidence`` grade, and only the first
    two can carry a verdict:

      "written"   the blocker phrase names the file verbatim
                  ("blocked on `prompt_library.py`")
      "in-record" the tool synthesised the name from prose, but the repository
                  writes that exact filename somewhere -- README.md's file tree
                  and its note on `prompt_library.py`, say -- so the name is the
                  record's, not this tool's
      "guessed"   the name exists only because this tool built it out of the
                  words in a phrase, and nothing committed ever writes it

    A "guessed" hit alone is reported as a lead and leaves ``holds`` at None.
    Previously it was enough on its own: a phrase like "blocked on the prompt
    library" became ``prompt_library.py``, any file of that basename anywhere in
    the tree matched, and the blocker was declared lifted. That reasoning would
    have out-argued the record on a coincidence of naming, which is not a thing
    a report about the record should be able to do.

    Prose asserting the opposite ("exists only on Thom's home machine") is
    collected either way and printed next to the verdict, so a resolution that
    overrides the record does so visibly rather than silently.
    """
    res = {"holds": None, "resolution_evidence": [], "checked": [],
           "found": [], "issue_prose": [], "evidence": None,
           "contrary_prose": [], "guessed_only": []}
    index = basename_index()
    corpus = None

    for label in group["artefact_phrases"]:
        for cand, origin in blocker_candidates(label):
            if cand in res["checked"]:
                continue
            res["checked"].append(cand)
            hits = index.get(cand, [])
            if hits:
                mentions = literal_mentions(cand)
                if origin == "written":
                    grade = "written"
                else:
                    if corpus is None:
                        corpus = text_corpus()
                    grade = "in-record" if (mentions or cand in corpus) else "guessed"
                res["found"].append({"name": cand, "paths": hits[:3],
                                     "evidence": grade, "mentions": mentions})
                res["contrary_prose"].extend(contrary_prose(cand))
                continue
            # Absent.  Does the repository nonetheless talk about it as a file?
            mentions = literal_mentions(cand)
            if corpus is None:
                corpus = text_corpus()
            if mentions or cand in corpus:
                res.setdefault("named_but_absent", []).append(
                    {"name": cand, "mentions": mentions})

    named = [f for f in res["found"] if f["evidence"] in ("written", "in-record")]
    res["guessed_only"] = [f for f in res["found"] if f["evidence"] == "guessed"]

    if named:
        res["holds"] = False
        res["evidence"] = ("written" if any(f["evidence"] == "written" for f in named)
                           else "in-record")
        for found in named:
            path = found["paths"][0]
            first = found["mentions"][0] if found["mentions"] else None
            if found["evidence"] == "written":
                because = "the record names this file directly"
            else:
                because = ("the record writes this filename%s, so the name is the "
                           "record's and not this tool's"
                           % (" (%s:%d)" % (first["file"], first["line"])
                              if first else ""))
            res["resolution_evidence"].append({
                "file": path, "line": None, "kind": "verdict",
                "text": "`%s` is present in the working tree (%s bytes); %s"
                        % (path, file_size(path) or 0, because),
            })
        for hit in res["contrary_prose"]:
            res["resolution_evidence"].append({
                "file": hit["file"], "line": hit["line"], "kind": "contrary",
                "text": "`%s`:%d &mdash; “%s” (stale: the file it calls "
                        "absent is in the tree)"
                        % (hit["file"], hit["line"], hit["text"]),
            })
    elif res["guessed_only"]:
        # A name this tool invented matched a file. Worth reading, never a
        # verdict: the record is not overturned by a coincidence of naming.
        res["evidence"] = "guessed"
        for found in res["guessed_only"]:
            res["resolution_evidence"].append({
                "file": found["paths"][0], "line": None, "kind": "verdict",
                "text": "`%s` was guessed from the phrase, not written in the "
                        "record, and a file of that name exists (`%s`). Nothing "
                        "committed writes that filename, so this is a lead to "
                        "check by hand, not grounds to call the blocker lifted"
                        % (found["name"], found["paths"][0]),
            })
        for hit in res["contrary_prose"]:
            res["resolution_evidence"].append({
                "file": hit["file"], "line": hit["line"], "kind": "contrary",
                "text": "`%s`:%d &mdash; “%s” (which a guessed filename "
                        "is not evidence enough to overturn)"
                        % (hit["file"], hit["line"], hit["text"]),
            })
    elif res.get("named_but_absent"):
        res["holds"] = True
        res["evidence"] = "in-record"
        for item in res["named_but_absent"]:
            first = item["mentions"][0] if item["mentions"] else None
            res["resolution_evidence"].append({
                "file": first["file"] if first else None,
                "line": first["line"] if first else None, "kind": "verdict",
                "text": "`%s` is named by the repository%s but no such file "
                        "exists anywhere in the working tree"
                        % (item["name"],
                           " (%s:%d)" % (first["file"], first["line"])
                           if first else ""),
            })
    elif group["issues"]:
        res["resolution_evidence"] = [{
            "file": None, "line": None, "kind": "verdict",
            "text": "the blocker is recorded only as an issue number, which "
                    "names no file; the working tree cannot settle it and this "
                    "report does not guess",
        }]
    else:
        res["resolution_evidence"] = [{
            "file": None, "line": None, "kind": "verdict",
            "text": "the blocker is a phrase with no filename this tool can "
                    "responsibly look for; not checked",
        }]

    seen, unique = set(), []
    for item in res["resolution_evidence"]:
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        unique.append(item)
    res["resolution_evidence"] = unique

    # Supplementary only, never a verdict: committed prose that mentions the
    # issue number at all, so a reader can go and look.
    for number in group["issues"]:
        hits = [{"file": p, "line": n, "text": " ".join(t.split())[:260]}
                for p, n, t in markdown_index()
                if ("issue #%s" % number) in t.lower()][:4]
        if hits:
            res["issue_prose"].append({"issue": number, "mentions": hits})
    return res


def cell_count(row) -> str:
    """Render a notebook's executed share against its *code*-cell count.

    Markdown cells can never carry an `outputs` array, so quoting "8 of 14"
    against the total invites the reader to think six cells failed to run when
    six of them were prose. Code cells are the denominator that means anything;
    the total is kept in parentheses because it is what a reader sees when they
    open the file.
    """
    code = row.get("code_cells")
    total = row.get("cells_total")
    got = row.get("cells_with_outputs")
    if not code:
        return "&mdash;"
    text = "%s of %s" % (got, code)
    if total and total != code:
        text += " (%s cells total)" % total
    return text


def executed_share(state) -> str:
    """"8 of its 8 code cells (14 cells total)" -- the ledger's denominator.

    The narrative sections used to quote the executed count against
    ``cells_total`` ("8 of its 14 cells") while the notebook ledger quoted it
    against code cells ("8 of 8, 14 cells total") for the same file, so the
    document contradicted itself about whether six cells had failed to run.
    They had not: markdown cells cannot carry an `outputs` array at all, and six
    of those fourteen were prose. Code cells are the only denominator that means
    anything, and both places now use it -- with the total kept in parentheses,
    because it is what a reader sees when they open the file.
    """
    got = state.get("cells_with_outputs")
    code = state.get("code_cells")
    total = state.get("cells_total")
    if not code:
        return "%s of its cells" % got
    text = "%s of its %s code cells" % (got, code)
    if total and total != code:
        text += " (%s cells total)" % total
    return text


def sentences_of(text: str):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def quote_for(sentence: str, pattern, limit: int) -> str:
    """Quote `sentence` in at most `limit` characters, keeping the match visible.

    Truncating a long sentence from its head can cut off the very words the
    verdict is citing. F1's description is one 400-character sentence whose
    "finer cadence was never run" sits at the end, so a head-truncated quote
    read as a bare statement of results with no pending language in it at all --
    a reader checking the justification found it did not support the verdict.

    So: centre the window on the match and mark the elision. If the whole
    sentence fits, or nothing matches, this is a plain trim and behaves exactly
    as before.
    """
    text = " ".join((sentence or "").split())
    if not text:
        return ""
    if len(text) <= limit:
        return text
    match = pattern.search(text) if pattern is not None else None
    if match is None:
        return text[:limit]
    # Centre on the match, then clamp to the sentence's ends.
    span = match.end() - match.start()
    slack = max(limit - span, 0)
    start = max(match.start() - slack // 2, 0)
    end = min(start + limit, len(text))
    start = max(min(start, end - limit), 0)
    # Prefer whole words at either seam.
    if start > 0:
        space = text.find(" ", start)
        if space != -1 and space < match.start():
            start = space + 1
    if end < len(text):
        space = text.rfind(" ", start, end)
        if space != -1 and space > match.end():
            end = space
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")


def detect_blockers(graph: Graph):
    """Parse blockers out of claim descriptions and group by the blocker.

    One sentence naming both a phrase and an issue number ("blocked on the
    prompt library (issue #9)") is one obstacle, not two: the issue number
    becomes the group key and the phrase becomes the thing to look for on disk.
    """
    groups = {}
    for claim_id, claim in sorted(graph.claims.items()):
        if claim.get("status") == "retired":
            continue  # a retired claim's prose is history, not an obstacle
        desc = claim.get("description") or ""
        for sentence in sentences_of(desc):
            if RESOLVED_RE.search(sentence):
                continue  # a receipt for a cleared obstacle, not a live one
            issues = sorted(set(ISSUE_RE.findall(sentence)), key=int)
            phrases = []
            for _kind, pattern in BLOCKER_PATTERNS:
                for match in pattern.finditer(sentence):
                    key, label = normalise_phrase(match.group(1))
                    if key:
                        phrases.append((key, label))
            if not issues and not phrases:
                continue
            if issues:
                group_key = "issue-%s" % issues[0]
                group_label = (phrases[0][1] + " (issue #%s)" % issues[0]
                               if phrases else "issue #%s" % issues[0])
            else:
                group_key, group_label = phrases[0][0], phrases[0][1]

            grp = groups.setdefault(group_key, {
                "key": group_key, "label": group_label, "issues": set(),
                "artefact_phrases": [], "claims": {}})
            grp["issues"].update(issues)
            for _key, label in phrases:
                if label not in grp["artefact_phrases"]:
                    grp["artefact_phrases"].append(label)
            if phrases and "(issue #" not in grp["label"] and grp["issues"]:
                grp["label"] = "%s (issue #%s)" % (
                    phrases[0][1], sorted(grp["issues"], key=int)[0])

            entry = grp["claims"].setdefault(claim_id, {
                "claim": claim_id, "label": claim.get("label"),
                "type": claim.get("type"), "status": claim.get("status"),
                "doc_ref": claim.get("doc_ref"), "quotes": []})
            quote = " ".join(sentence.split())
            if quote not in entry["quotes"]:
                entry["quotes"].append(quote[:400])

    # Some graph versions state the same obstacle structurally, as a `blocks`
    # or `blocked-by` edge. Fold those in: same groups, extra evidence.
    for edge in graph.rels:
        if edge["type"] == "blocks":
            blocker_id, blocked_id = edge["from"], edge["to"]
        elif edge["type"] == "blocked-by":
            blocker_id, blocked_id = edge["to"], edge["from"]
        else:
            continue
        blocker = graph.nodes.get(blocker_id, {})
        blocked = graph.claims.get(blocked_id)
        if blocked is None:
            continue
        blocker_desc = blocker.get("description", "") or ""
        issues = sorted(set(ISSUE_RE.findall(blocker_desc)), key=int)
        group_key = ("issue-%s" % issues[0]) if issues else ("node-%s" % blocker_id)
        grp = groups.setdefault(group_key, {
            "key": group_key,
            "label": blocker.get("label", blocker_id),
            "issues": set(), "artefact_phrases": [], "claims": {}})
        grp["issues"].update(issues)
        for match in re.finditer(r"[\w./-]+\.(?:py|ipynb|pt|json)", blocker_desc):
            phrase = match.group(0)
            if phrase not in grp["artefact_phrases"]:
                grp["artefact_phrases"].append(phrase)
        for _kind, pattern in BLOCKER_PATTERNS:
            for match in pattern.finditer(blocker_desc):
                _key, label = normalise_phrase(match.group(1))
                if label and label not in grp["artefact_phrases"]:
                    grp["artefact_phrases"].append(label)
        entry = grp["claims"].setdefault(blocked_id, {
            "claim": blocked_id, "label": blocked.get("label"),
            "type": blocked.get("type"), "status": blocked.get("status"),
            "doc_ref": blocked.get("doc_ref"), "quotes": []})
        quote = "graph edge `%s` %s `%s`: %s" % (
            edge["from"], edge["type"], edge["to"],
            " ".join((edge.get("description") or "").split()))
        if quote not in entry["quotes"]:
            entry["quotes"].append(quote[:400])

    out = []
    for key, grp in groups.items():
        grp["issues"] = sorted(grp["issues"], key=int)
        claims = sorted(grp["claims"].values(), key=lambda c: c["claim"])
        resolution = resolve_blocker(grp)
        if grp["issues"] or resolution["holds"] is not None:
            kind = "artefact"
        elif key.startswith("node-"):
            kind = "prerequisite"
        else:
            kind = "unresolvable-phrase"
        out.append({
            "key": key,
            "label": grp["label"],
            "kind": kind,
            "issues": grp["issues"],
            "artefact_phrases": grp["artefact_phrases"],
            "gates": claims,
            "gate_count": len(claims),
            "still_holds": resolution["holds"],
            "resolution_evidence_kind": resolution["evidence"],
            "resolution": resolution,
        })
    out.sort(key=lambda g: (-g["gate_count"], g["key"]))
    return out


# --------------------------------------------------------------------------
# DETECTOR C: needs-compute vs answerable-from-disk
# --------------------------------------------------------------------------

SCRIPT_IN_TEXT_RE = re.compile(r"(experiments/[\w./-]+\.(?:py|ipynb))")


def runs_for_claim(graph: Graph, claim_id: str):
    """Every run node the graph associates with a claim, however loosely."""
    found = []
    for edge in graph.incoming.get(claim_id, []):
        if edge["from"] in graph.runs and edge["type"] in EPISTEMIC_EDGES:
            found.append(edge["from"])
    for edge in graph.outgoing.get(claim_id, []):
        if edge["to"] in graph.runs:
            found.append(edge["to"])
    for ev in graph.claims.get(claim_id, {}).get("evidence") or []:
        if ev in graph.runs:
            found.append(ev)
    return sorted(set(found))


def detect_open_threads(graph: Graph, answered, blockers, repo_modules):
    """Every claim that reads as unfinished, with a disk verdict for each."""
    claim_ids = set()
    pending_hits = {}
    for claim_id, claim in graph.claims.items():
        if claim.get("status") == "retired":
            continue  # answered and retired: nothing here is unfinished
        desc = claim.get("description") or ""
        hits = set()
        # Sentence by sentence, so a resolution receipt ("Answered
        # 2026-07-31 ... issue #24", "executed 2026-07-25 in the issue #25
        # artifact regeneration") does not read as pending work merely
        # because it cites the issue it closed.
        for sentence in sentences_of(desc):
            if RESOLVED_RE.search(sentence):
                continue
            hits.update(m.group(0).lower()
                        for m in PENDING_RE.finditer(sentence))
        hits = sorted(hits)
        if hits:
            claim_ids.add(claim_id)
            pending_hits[claim_id] = hits
    for rec in answered:
        claim_ids.add(rec["claim"])
    blocked_by = defaultdict(list)
    for grp in blockers:
        for gate in grp["gates"]:
            claim_ids.add(gate["claim"])
            blocked_by[gate["claim"]].append(grp)

    threads = []
    for claim_id in sorted(claim_ids):
        claim = graph.claims[claim_id]
        desc = claim.get("description") or ""
        run_ids = runs_for_claim(graph, claim_id)
        named_scripts = sorted(set(SCRIPT_IN_TEXT_RE.findall(desc)))

        evidence = [run_disk_evidence(graph, r, repo_modules) for r in run_ids]
        # A script named in the prose but attached to no run node still counts.
        known = {e["script"] for e in evidence}
        for script in named_scripts:
            if script in known:
                continue
            evidence.append({
                "run": None, "run_label": "(named in the record, no run node)",
                "graph_says": None, "date": None, "script": script,
                "script_exists": exists(script),
                "script_state": (inspect_notebook(script)
                                 if exists(script) and script.endswith(".ipynb")
                                 else (inspect_script(script) if exists(script)
                                       else None)),
                "output_dir": None, "output_dir_exists": None,
                "output_dir_files": [], "declared_inputs": [],
                "missing_inputs": [], "saved_artefacts": [],
                "missing_local_modules": (missing_local_imports(script, repo_modules)
                                          if exists(script) else []),
                "graph_artefacts": [],
            })

        groups = blocked_by.get(claim_id, [])
        answered_record = next((r for r in answered if r["claim"] == claim_id), None)
        verdict, reason = classify_thread(
            evidence, groups, named_scripts, answered_record,
            pending_hits.get(claim_id, []), desc)

        threads.append({
            "claim": claim_id,
            "label": claim.get("label"),
            "type": claim.get("type"),
            "status": claim.get("status"),
            "asserted": claim.get("asserted"),
            "doc_ref": claim.get("doc_ref"),
            "description": desc,
            "pending_language": pending_hits.get(claim_id, []),
            "blockers": [{"key": g["key"], "label": g["label"],
                          "still_holds": g["still_holds"]} for g in groups],
            "runs": run_ids,
            "evidence": evidence,
            "verdict": verdict,
            "reason": reason,
            "rank_tier": RANK_ORDER.get(verdict, 99),
        })

    threads.sort(key=lambda t: (t["rank_tier"], t["claim"]))
    return threads


def _judge_scripts(evidence, scripts):
    """Disk state of a specific set of scripts: (executed, artefacts, missing)."""
    executed, artefacts, missing_inputs, missing_scripts = [], [], [], []
    for ev in evidence:
        if ev.get("script") not in scripts:
            continue
        if not ev.get("script_exists"):
            missing_scripts.append(ev["script"])
            continue
        state = ev.get("script_state") or {}
        if state.get("executed"):
            executed.append("%s carries executed outputs in %s"
                            % (ev["script"], executed_share(state)))
        got = [a["found"][0] for a in ev.get("saved_artefacts", []) if a["found"]]
        if got:
            artefacts.append("%s has written %s"
                             % (ev["script"], ", ".join(got[:3])))
        if ev.get("output_dir_files"):
            artefacts.append("%s holds %d file(s)"
                             % (ev["output_dir"], len(ev["output_dir_files"])))
        missing_inputs += ["%s (declared input of %s)" % (m, ev["script"])
                           for m in ev.get("missing_inputs", [])]
        missing_inputs += ["%s (imported by %s, absent from the tree)"
                           % (m, ev["script"])
                           for m in ev.get("missing_local_modules", [])]
    return executed, artefacts, missing_inputs, missing_scripts


def classify_thread(evidence, blocker_groups, named_scripts, answered_record,
                    pending_language, description):
    """Could this thread be settled from what is already committed?

    The order of the checks is the point, and it is deliberately conservative.
    A run that produced a finding having left artefacts says nothing about the
    unfinished part of that finding, so artefacts alone never buy a verdict.
    Only two things do: a run designated to test the claim that has already
    executed, or a named script whose state can be read off disk.

    Returns (verdict, human-readable reason). 'cannot-determine' is a valid
    and useful answer, and much better than a confident wrong one.
    """
    # 1. The designated test has already run, and the graph has no verdict.
    if answered_record and answered_record["assessment"] in (
            "answered-on-disk-unrecorded-in-graph",
            "artefacts-on-disk-verdict-not-legible"):
        scripts = {e["script"] for e in answered_record["disk_evidence"]}
        executed, artefacts, _, _ = _judge_scripts(
            answered_record["disk_evidence"], scripts)
        return ("answered-on-disk",
                "the run the graph designates as this claim's test has already "
                "executed and the graph still records no verdict: "
                + "; ".join(executed + artefacts))

    # 2. Blockers that name a missing artefact, or an external issue, govern
    #    the open part of a claim whatever else exists. A blocker that is
    #    simply another open question is a prerequisite, not an obstacle: it
    #    is recorded and carried into the reason, but it does not decide.
    external = [g for g in blocker_groups if g["kind"] == "artefact"]
    prereq = [g for g in blocker_groups if g["kind"] == "prerequisite"]
    suffix = ("" if not prereq else
              " (prerequisite in the graph: %s)"
              % ", ".join(g["label"] for g in prereq))
    holding = [g for g in external if g["still_holds"] is True]
    unblocked = [g for g in external if g["still_holds"] is False]
    unknown_issue = [g for g in external
                     if g["still_holds"] is None and g["issues"]]
    if holding:
        return ("still-blocked",
                "blocked on %s, and %s%s"
                % (", ".join(g["label"] for g in holding),
                   "; ".join(dict.fromkeys(
                       e["text"] for g in holding
                       for e in g["resolution"]["resolution_evidence"]
                       if e.get("kind") != "contrary")),
                   suffix))
    if unblocked and not unknown_issue:
        # Only the verdict-bearing evidence goes in the one-line reason. The
        # stale prose the resolution overrides is quoted in full in the blockers
        # section, where there is room for it; repeating it in a table cell for
        # every gated claim buries the verdict under its own footnotes.
        return ("newly-unblocked",
                "the record calls this blocked on %s, and %s%s"
                % (", ".join(g["label"] for g in unblocked),
                   "; ".join(dict.fromkeys(
                       e["text"] for g in unblocked
                       for e in g["resolution"]["resolution_evidence"]
                       if e.get("kind") != "contrary")),
                   suffix))
    if unknown_issue:
        return ("still-blocked",
                "the record calls this blocked on %s; an issue number names no "
                "file, so the working tree cannot say whether it has been "
                "resolved and this report does not guess%s"
                % (", ".join(g["label"] for g in unknown_issue), suffix))

    # 3. A script named in the record, judged directly.
    if named_scripts:
        executed, artefacts, missing_inputs, missing_scripts = _judge_scripts(
            evidence, set(named_scripts))
        if missing_scripts:
            return ("needs-compute",
                    "the script the record names does not exist in the tree: "
                    + ", ".join(sorted(set(missing_scripts))))
        if executed or artefacts:
            return ("answered-on-disk",
                    "the script the record names has already produced results: "
                    + "; ".join(executed + artefacts))
        if missing_inputs:
            return ("needs-compute",
                    "the script exists but its declared inputs do not: "
                    + ", ".join(sorted(set(missing_inputs))[:4]))
        return ("answerable-from-disk",
                "%s exists, every input it declares is present in the tree, and "
                "it has not been executed%s"
                % (", ".join(named_scripts), suffix))

    # 5. The record states that the inputs for this are already committed.
    #    No script is named, so this is the record's own claim, corroborated
    #    only by the artefacts being where it says they are. Reported as such.
    if ON_DISK_RE.search(description):
        sentence = next((s for s in sentences_of(description)
                         if ON_DISK_RE.search(s)), "")
        pts = walk_repo("experiments", (".pt",))
        return ("answerable-from-disk",
                "the record itself says the inputs are already committed "
                "(\"%s\"), and %d .pt artefact(s) exist under experiments/; "
                "no script is named for the computation, so this rests on the "
                "record's word plus the presence of the data%s"
                % (quote_for(sentence, ON_DISK_RE, 200), len(pts), suffix))

    # 6. The record states in as many words that the work was never done, and
    #    names no script for it: that is new compute by definition.
    never = NEVER_RUN_RE.search(description)
    if never:
        sentence = next((s for s in sentences_of(description)
                         if NEVER_RUN_RE.search(s)), "")
        return ("needs-compute",
                "the record states the work has not been done (\"%s\") and "
                "names no script for it, so answering it requires a fresh run%s"
                % (quote_for(sentence, NEVER_RUN_RE, 220), suffix))

    # 7. Nothing checkable.
    return ("cannot-determine",
            "the pending language here (%s) is tied to no script, run or "
            "artefact this tool can find, so no disk verdict is available"
            % ", ".join(pending_language) if pending_language else
            "nothing in the graph or the tree ties this thread to a script, "
            "run or artefact")


# --------------------------------------------------------------------------
# DETECTOR D: the frontier
# --------------------------------------------------------------------------

def detect_frontier(graph: Graph):
    rows = []
    for claim_id, claim in graph.claims.items():
        if claim.get("type") != "finding":
            continue
        downstream = [e for e in graph.incoming.get(claim_id, [])
                      if e["type"] in DOWNSTREAM_EDGES]
        if downstream:
            continue
        rows.append({
            "claim": claim_id,
            "label": claim.get("label"),
            "status": claim.get("status"),
            "asserted": claim.get("asserted"),
            "source_doc": graph.source_doc(claim),
            "doc_ref": claim.get("doc_ref"),
            "outgoing_edges": sorted({e["type"] for e in graph.outgoing.get(claim_id, [])}),
            "incoming_edges": sorted({e["type"] for e in graph.incoming.get(claim_id, [])}),
        })
    rows.sort(key=lambda r: (r["asserted"] or "", r["claim"]), reverse=True)
    return rows


# --------------------------------------------------------------------------
# DETECTOR E: undeveloped
# --------------------------------------------------------------------------

def detect_undeveloped(graph: Graph):
    thin_concepts, silent_claims = [], []
    for claim_id, claim in sorted(graph.claims.items()):
        degree = graph.degree[claim_id]
        edges = graph.incoming.get(claim_id, []) + graph.outgoing.get(claim_id, [])
        epistemic = sorted({e["type"] for e in edges if e["type"] in EPISTEMIC_EDGES})
        row = {
            "claim": claim_id, "label": claim.get("label"),
            "type": claim.get("type"), "status": claim.get("status"),
            "asserted": claim.get("asserted"), "degree": degree,
            "edge_types": sorted({e["type"] for e in edges}),
            "doc_ref": claim.get("doc_ref"),
        }
        if claim.get("type") == "concept" and degree <= 1:
            thin_concepts.append(row)
        if not epistemic:
            silent_claims.append(row)
    thin_concepts.sort(key=lambda r: (r["degree"], r["claim"]))
    silent_claims.sort(key=lambda r: (r["type"], r["claim"]))
    return {"thin_concepts": thin_concepts, "silent_claims": silent_claims}


# --------------------------------------------------------------------------
# DETECTOR F: unreferenced artefacts
# --------------------------------------------------------------------------

def detect_unreferenced(graph: Graph):
    known_dirs = set()
    for run in graph.runs.values():
        for key in ("output_dir", "output_path"):
            val = run.get(key)
            if val:
                known_dirs.add(val.rstrip("/"))

    scripts = walk_repo("experiments", SCRIPT_EXT)
    script_text = {s: read_text(s) for s in scripts}

    run_pairs = set()
    for run in graph.runs.values():
        script = run.get("script")
        outdir = (run.get("output_dir") or "").rstrip("/")
        if script and outdir:
            run_pairs.add((script, outdir))

    def producer_of(basename, parent):
        """Scripts that plausibly WRITE this artefact.

        A bare basename match is not enough: half the experiment scripts
        save a file called results.json, and matching on the name alone
        attributed each one's artefact to all of the others. So the script
        must also be TIED to the artefact's directory, one of three ways:
        a run node pairs the script with that output directory; the script
        sits in the directory's parent and names the directory in its own
        text (output paths are built from string literals); or the artefact
        sits directly beside the script.
        """
        parent_norm = parent.rstrip("/")
        dir_base = os.path.basename(parent_norm)
        parent_of_parent = os.path.dirname(parent_norm)
        hits = []
        for path, text in script_text.items():
            low = text.lower()
            writes = ("save" in low or "json.dump" in low
                      or "to_csv" in low or '"w"' in low or "'w'" in low)
            if basename not in text or not writes:
                continue
            script_dir = os.path.dirname(path)
            tied = (
                (path, parent_norm) in run_pairs
                or (script_dir == parent_of_parent and dir_base in text)
                or script_dir == parent_norm
            )
            if tied:
                hits.append(path)
        return sorted(hits)

    artefacts, orphan_scripts = [], []
    for path in walk_repo("experiments", ARTEFACT_EXT + (".md",)):
        base = os.path.basename(path)
        if path in graph.blob or base in graph.blob:
            continue
        parent = os.path.dirname(path)
        in_known_dir = parent.rstrip("/") in known_dirs or parent in graph.blob
        producers = producer_of(base, parent)
        producer_runs = []
        for prod in producers:
            for run_id, run in graph.runs.items():
                if run.get("script") == prod:
                    producer_runs.append({
                        "run": run_id, "graph_says": run.get("n"),
                        "date": run.get("date"),
                        "label": run.get("label")})
        artefacts.append({
            "path": path,
            "size": file_size(path),
            "kind": "report" if path.endswith(".md") else "data",
            "classification": "sidecar-of-a-known-output-dir" if in_known_dir
                              else "orphan-directory",
            "directory": parent,
            "produced_by_script": producers,
            # An empty producers list is a statement, not a blank: no script
            # could be tied to this artefact's directory, so the attribution
            # is unresolved rather than silently guessed from a basename.
            "producer_resolution": "path-tied" if producers else "unresolved",
            "producer_run_nodes": producer_runs,
        })

    for path in scripts:
        base = os.path.basename(path)
        if path in graph.blob or base in graph.blob:
            continue
        state = (inspect_notebook(path) if path.endswith(".ipynb")
                 else inspect_script(path))
        orphan_scripts.append({
            "path": path,
            "kind": state["kind"],
            "cells_total": state.get("cells_total"),
            "code_cells": state.get("code_cells"),
            "cells_with_outputs": state.get("cells_with_outputs"),
            "executed": state.get("executed"),
        })

    artefacts.sort(key=lambda a: (a["classification"] != "orphan-directory",
                                  a["kind"] != "data", a["path"]))
    orphan_scripts.sort(key=lambda s: s["path"])
    return {"artefacts": artefacts, "orphan_scripts": orphan_scripts}


# --------------------------------------------------------------------------
# Supplementary: the execution ledger
# --------------------------------------------------------------------------

def execution_ledger(graph: Graph):
    """For every run whose script is a notebook: how much of it has been run.

    Cheap, and it catches drift in both directions: a notebook the record calls
    a scaffold that carries outputs, and a notebook credited with findings that
    carries almost none.
    """
    rows = []
    for run_id, run in sorted(graph.runs.items()):
        script = run.get("script")
        if not script:
            continue
        row = {"run": run_id, "label": run.get("label"), "script": script,
               "exists": exists(script), "cells_total": None,
               "code_cells": None, "cells_with_outputs": None, "note": None}
        if not row["exists"]:
            row["note"] = "script named by the graph is absent from the tree"
            rows.append(row)
            continue
        if script.endswith(".ipynb"):
            state = inspect_notebook(script)
            row["cells_total"] = state["cells_total"]
            row["code_cells"] = state["code_cells"]
            row["cells_with_outputs"] = state["cells_with_outputs"]
            label = (run.get("label") or "") + " " + (run.get("description") or "")
            if state["executed"] and NOT_RUN_RE.search(label):
                row["note"] = ("the graph calls this run unexecuted; the "
                               "notebook carries outputs")
            elif not state["executed"]:
                row["note"] = "no cell carries outputs"
            elif state["cells_with_outputs"] <= 2 and state["code_cells"] > 4:
                row["note"] = ("only %d of %d code cells carry outputs; the "
                               "rest have been cleared or never ran"
                               % (state["cells_with_outputs"], state["code_cells"]))
        else:
            row["note"] = "plain script: execution leaves no trace in the file"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Ranking: low-hanging fruit
# --------------------------------------------------------------------------

def rank_fruit(threads, answered):
    """Rank every open thread by the printed rule.

    Tiers 1 to 3 are the fruit proper: work that is cheap because the evidence
    is already committed. Tiers 4 to 6 are ranked too, so the list doubles as
    the full opportunity queue and the cut line is visible rather than
    implicit.
    """
    answered_claims = {r["claim"]: r for r in answered}
    fruit = []
    for thread in threads:
        tier = thread["rank_tier"]
        record = answered_claims.get(thread["claim"])
        quote = None
        if record and record["quoted_result"]:
            quote = record["quoted_result"][0]
        fruit.append({
            "rank_tier": tier,
            "tier_name": thread["verdict"],
            "claim": thread["claim"],
            "label": thread["label"],
            "status": thread["status"],
            "reason": thread["reason"],
            "quoted_result": quote,
            "doc_ref": thread["doc_ref"],
            "cheap": thread["rank_tier"] <= RANK_ORDER["answerable-from-disk"],
            "scripts": sorted({e["script"] for e in thread["evidence"]
                               if e.get("script")}),
        })
    fruit.sort(key=lambda f: (f["rank_tier"], f["claim"]))
    for i, item in enumerate(fruit, start=1):
        item["rank"] = i
    return fruit


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def md_escape(text):
    return (text or "").replace("|", "\\|")


def quote_block(lines, indent="> "):
    return "\n".join(indent + "`" + line.strip() + "`" for line in lines)


def render_markdown(graph, answered, blockers, threads, frontier, undeveloped,
                    unreferenced, ledger, fruit):
    meta = graph.meta
    out = []
    add = out.append

    add("# Threads, Blockers and Loose Ends")
    add("")
    add("*What the evidence graph knows that the prose has not got round to "
        "saying, and what the filesystem knows that neither of them does.*")
    add("")
    add("> Generated by [`build_threads_report.py`](build_threads_report.py) "
        "from [`_data/entities.json`](_data/entities.json) and the working "
        "tree. Nothing here is asserted; everything here is quoted. The graph "
        "is built by parsing `docs/FINDINGS.md` and `docs/JOURNEY_MAP.md`, so "
        "wherever it and the filesystem disagree, this report shows both and "
        "leaves the ruling to the operator.")
    add("")
    add("| | |")
    add("|:---|:---|")
    add("| Graph | %s |" % md_escape(meta.get("title", "")))
    add("| Sources parsed | %s |" % ", ".join(
        "`%s`" % s for s in meta.get("sources_parsed", [])))
    add("| Record last updated | %s |" % meta.get("source_last_updated", "unknown"))
    add("| Nodes read | %d claims, %d runs, %d sources, %d edges |" % (
        len(graph.claims), len(graph.runs), len(graph.sources), len(graph.rels)))
    add("")

    # ---- headline -------------------------------------------------------
    headline = [r for r in answered
                if r["assessment"] == "answered-on-disk-unrecorded-in-graph"]
    add("---")
    add("")
    add("## The headline")
    add("")
    if not headline:
        add("No claim in the graph carries a `tests` edge without a verdict "
            "edge while also carrying executed results on disk. Nothing is "
            "sitting answered and unrecorded.")
        add("")
    for record in headline:
        add("### %s" % md_escape(record["label"] or record["claim"]))
        add("")
        add("**The graph says:** `%s`, status **%s**, tested by %s, and no "
            "`supports` / `refutes` / `qualifies` / `corrects` edge points at "
            "it. On the graph alone this is an open question."
            % (record["claim"], record["status"],
               ", ".join("`%s`" % t["id"] for t in record["tested_by"])))
        add("")
        for ev in record["disk_evidence"]:
            state = ev.get("script_state") or {}
            add("**The disk says otherwise.** `%s` exists and %s carry a "
                "non-empty `outputs` array."
                % (ev["script"], executed_share(state)))
            add("")
            saved = [a for a in ev.get("saved_artefacts", []) if a["found"]]
            if saved:
                add("Artefacts the script writes, and where they now are:")
                add("")
                for art in saved:
                    add("- `%s` &rarr; `%s`%s" % (
                        art["name"], art["found"][0],
                        " (%s bytes)" % art["size"] if art["size"] else ""))
                add("")
        for quote in record["quoted_result"]:
            add("From cell %d of `%s`:" % (quote["cell"], quote["script"]))
            add("")
            add(quote_block(quote["lines"]))
            add("")
        if record["prose_says_not_run"]:
            add("Meanwhile the committed prose still says it was never run:")
            add("")
            for hit in record["prose_says_not_run"]:
                add("- [`%s`:%d](../../%s) &mdash; %s" % (
                    hit["file"], hit["line"], hit["file"], md_escape(hit["text"])))
            add("")
        if record["disk_evidence"]:
            state = record["disk_evidence"][0].get("script_state") or {}
            if state.get("banner") and NOT_RUN_RE.search(state["banner"]):
                add("The notebook's own opening cell agrees with the prose and "
                    "not with its own outputs:")
                add("")
                add("> %s" % md_escape(state["banner"]))
                add("")
        if record["prose_says_was_run"]:
            add("One committed document does record the execution &mdash; "
                "which is how this can be checked rather than merely suspected:")
            add("")
            for hit in record["prose_says_was_run"]:
                add("- [`%s`:%d](../../%s) &mdash; %s" % (
                    hit["file"], hit["line"], hit["file"], md_escape(hit["text"])))
            add("")
        add("**What to do with it:** nothing automatic. `docs/FINDINGS.md` and "
            "`docs/JOURNEY_MAP.md` are the canonical scientific record and "
            "amending a hypothesis disposition is the operator's call. This "
            "report exists so the call can be made knowingly.")
        add("")

    others = [r for r in answered if r not in headline]
    if others:
        add("### Also tested, also unresolved")
        add("")
        add("| Claim | Type | Status | Tested by | On disk |")
        add("|:---|:---|:---|:---|:---|")
        for record in others:
            add("| `%s` | %s | %s | %s | %s |" % (
                record["claim"], record["type"], record["status"],
                ", ".join("`%s`" % t["id"] for t in record["tested_by"]),
                record["assessment"]))
        add("")
        for record in others:
            if record["assessment"] == "no-run-node-cannot-determine":
                add("`%s` is tested by claims rather than runs (%s), so there is "
                    "no script to check. The graph's `tests` edge here means "
                    "\"bears on\", not \"was executed against\"; no disk verdict "
                    "is available and none is offered." % (
                        record["claim"],
                        ", ".join("`%s`" % t["id"] for t in record["tested_by"])))
                add("")

    # ---- low-hanging fruit ---------------------------------------------
    add("---")
    add("")
    add("## Low-hanging fruit, and the rule that ranks it")
    add("")
    add("The ordering below is mechanical. It is printed here so that it can "
        "be argued with rather than merely trusted.")
    add("")
    for i, (name, description) in enumerate(RANK_RULE, start=1):
        add("%d. **%s** &mdash; %s" % (i, name, description))
    add("")
    cheap = [f for f in fruit if f["cheap"]]
    add("Tiers 1 to 3 are the fruit proper &mdash; %d of %d open threads. The "
        "rest are ranked in the same list so that the cut line is visible."
        % (len(cheap), len(fruit)))
    add("")
    if not fruit:
        add("No open threads were found.")
        add("")
    else:
        add("| # | Cheap? | Tier | Claim | Evidence, in brief |")
        add("|---:|:---:|:---|:---|:---|")
        for item in fruit:
            brief = item["reason"]
            if len(brief) > 190:
                brief = brief[:187].rsplit(" ", 1)[0] + "..."
            add("| %d | %s | %s | `%s` | %s |" % (
                item["rank"], "yes" if item["cheap"] else "no",
                item["tier_name"], item["claim"], md_escape(brief)))
        add("")
        if cheap:
            add("The cheap ones in full:")
            add("")
        for item in cheap:
            add("**%d. %s**" % (item["rank"], md_escape(item["label"] or item["claim"])))
            add("")
            add("- Tier: *%s*" % item["tier_name"])
            add("- Record: `%s`" % (item["doc_ref"] or "unrecorded"))
            if item["scripts"]:
                add("- Script(s): %s" % ", ".join("`%s`" % s for s in item["scripts"]))
            add("- Evidence: %s" % item["reason"])
            if item["quoted_result"]:
                add("- Already printed, in cell %d of `%s`:"
                    % (item["quoted_result"]["cell"], item["quoted_result"]["script"]))
                add("")
                add(quote_block(item["quoted_result"]["lines"][:8], indent="  > "))
            add("")

    # ---- blockers -------------------------------------------------------
    add("---")
    add("")
    add("## Blocked, grouped by blocker")
    add("")
    add("Grouped by the obstacle rather than by the claim, because that is the "
        "thing prose cannot show you: read `FINDINGS.md` top to bottom and two "
        "claims waiting on the same missing file are simply two paragraphs.")
    add("")
    sections = [
        ("artefact",
         "Missing artefacts and open issues",
         "Something the record names is not in the working tree, or the record "
         "defers to an issue number. These are the blockers that gate work."),
        ("prerequisite",
         "Prerequisites recorded in the graph",
         "Not obstacles but orderings: one open question that has to be "
         "answered before another can be. Carried through from the graph's own "
         "`blocks` and `blocked-by` edges."),
        ("unresolvable-phrase",
         "Phrases this tool declined to resolve",
         "The record uses blocking language, but the thing being waited on is "
         "not named as a file. Listed rather than guessed at."),
    ]
    for kind, title, blurb in sections:
        groups = [g for g in blockers if g["kind"] == kind]
        if not groups:
            continue
        add("### %s" % title)
        add("")
        add(blurb)
        add("")
        for group in sorted(groups, key=lambda g: (-g["gate_count"], g["key"])):
            holds = group["still_holds"]
            state = {True: "still holds on disk",
                     False: "no longer holds on disk",
                     None: "the working tree cannot settle it"}[holds]
            # What kind of evidence the verdict rests on, said out loud: a
            # filename the record wrote is not the same thing as a filename this
            # tool built out of a phrase, and only the first can move a verdict.
            grade = {
                "written": "the blocker phrase names the file itself",
                "in-record": "the filename was inferred from the phrase, then "
                             "confirmed against the record's own wording of it",
                "guessed": "on a filename this tool guessed from the phrase and "
                           "the record never writes &mdash; a lead, not a verdict",
            }.get(group.get("resolution_evidence_kind"))
            add("#### %s &mdash; gates %d claim%s; %s%s" % (
                md_escape(group["label"]), group["gate_count"],
                "" if group["gate_count"] == 1 else "s", state,
                " (%s)" % grade if grade else ""))
            add("")
            for gate in group["gates"]:
                add("- **`%s`** (%s, %s) &mdash; %s" % (
                    gate["claim"], gate["type"], gate["status"],
                    gate["doc_ref"] or ""))
                for quote in gate["quotes"]:
                    add("  > %s" % md_escape(quote))
            add("")
            evidence = group["resolution"]["resolution_evidence"]
            add("*Disk check:* %s" % (
                "; ".join(e["text"] for e in evidence
                          if e.get("kind") != "contrary")
                or "nothing committed speaks to this either way"))
            # The prose the verdict argues against, quoted rather than
            # summarised: a resolution that contradicts the record should show
            # the reader exactly what it is contradicting.
            contrary = [e for e in evidence if e.get("kind") == "contrary"]
            if contrary:
                add("")
                add("*What the record still says, and this disagrees with:*")
                for item in contrary:
                    add("- %s" % item["text"])
            if group["resolution"]["checked"]:
                add("")
                add("*Filenames tried:* %s" % ", ".join(
                    "`%s`" % c for c in group["resolution"]["checked"][:8]))
            for entry in group["resolution"]["issue_prose"]:
                add("")
                add("*Committed prose mentioning issue #%s:*" % entry["issue"])
                for hit in entry["mentions"]:
                    add("- [`%s`:%d](../../%s) &mdash; %s" % (
                        hit["file"], hit["line"], hit["file"],
                        md_escape(hit["text"])))
            add("")

    # ---- open threads ---------------------------------------------------
    add("---")
    add("")
    add("## Every open thread, and whether it needs compute")
    add("")
    add("A thread is open here if its description carries pending-work "
        "language, or it is tested without a verdict, or it names a blocker. "
        "The verdict column answers one question only: could this be settled "
        "from what is already committed?")
    add("")
    add("| Claim | Status | Verdict | Evidence |")
    add("|:---|:---|:---|:---|")
    for thread in threads:
        add("| `%s` | %s | **%s** | %s |" % (
            thread["claim"], thread["status"], thread["verdict"],
            md_escape(thread["reason"])))
    add("")
    counts = Counter(t["verdict"] for t in threads)
    add("Totals: %s." % ", ".join("%d %s" % (n, k) for k, n in
                                  sorted(counts.items(), key=lambda kv: -kv[1])))
    add("")
    add("Pending-work language was matched literally against claim "
        "descriptions. The phrases that fired, claim by claim:")
    add("")
    for thread in threads:
        if thread["pending_language"]:
            add("- `%s`: %s" % (thread["claim"], ", ".join(
                "*%s*" % p for p in thread["pending_language"])))
    add("")

    # ---- frontier -------------------------------------------------------
    add("---")
    add("")
    add("## The frontier")
    add("")
    add("Findings with nothing yet pointing back at them: no incoming "
        "`corrects`, `supersedes`, `qualifies` or `builds-on`. The most recent "
        "work that nothing has built on, which is either the edge of the "
        "study or the place it stopped.")
    add("")
    by_doc = defaultdict(list)
    for row in frontier:
        by_doc[row["source_doc"]].append(row)
    for doc in sorted(by_doc):
        rows = by_doc[doc]
        add("### From `%s` &mdash; %d finding%s" % (
            doc, len(rows), "" if len(rows) == 1 else "s"))
        add("")
        add("| Asserted | Claim | Status | Edges out |")
        add("|:---|:---|:---|:---|")
        for row in rows:
            add("| %s | `%s` | %s | %s |" % (
                row["asserted"] or "undated", row["claim"], row["status"],
                ", ".join(row["outgoing_edges"]) or "none"))
        add("")
        dates = Counter(r["asserted"] for r in rows if r["asserted"])
        if dates:
            add("Clustered on: %s." % ", ".join(
                "%s (%d)" % (d, n) for d, n in sorted(dates.items())))
            add("")

    # ---- undeveloped ----------------------------------------------------
    add("---")
    add("")
    add("## Introduced and then dropped")
    add("")
    add("### Concepts of degree one or less")
    add("")
    add("Named once, wired to at most one other node, and never taken up.")
    add("")
    if undeveloped["thin_concepts"]:
        add("| Concept | Degree | Only edge | Record |")
        add("|:---|---:|:---|:---|")
        for row in undeveloped["thin_concepts"]:
            add("| `%s` | %d | %s | %s |" % (
                row["claim"], row["degree"],
                ", ".join(row["edge_types"]) or "none", row["doc_ref"] or ""))
        add("")
    else:
        add("None.")
        add("")
    add("### Claims with no epistemic edge at all")
    add("")
    add("Nothing supports, refutes, qualifies, corrects, tests or builds on "
        "these. They are wired into the graph only by `documented-in`, "
        "`cites`, `relates-to` or the work edges `blocks` / `blocked-by`: "
        "vocabulary and scheduling rather than argument. Question nodes belong "
        "here by construction &mdash; an unanswered question has no evidence "
        "yet &mdash; so read the concepts as the finding.")
    add("")
    add("| Claim | Type | Degree | Edge types |")
    add("|:---|:---|---:|:---|")
    for row in undeveloped["silent_claims"]:
        add("| `%s` | %s | %d | %s |" % (
            row["claim"], row["type"], row["degree"],
            ", ".join(row["edge_types"]) or "none"))
    add("")
    kinds = Counter(row["type"] for row in undeveloped["silent_claims"])
    add("By type: %s." % ", ".join("%d %s" % (n, k) for k, n in sorted(kinds.items())))
    add("")
    open_kinds = Counter(c.get("type") for c in graph.claims.values()
                         if c.get("status") == "open")
    add("### Is `open` a work signal?")
    add("")
    add("Every claim the graph records at status `open`, by type: %s."
        % ", ".join("%d %s" % (n, k) for k, n in sorted(open_kinds.items())))
    add("")
    if set(open_kinds) <= {"concept"}:
        add("All of them are concepts, and `open` is simply the default for a "
            "concept carrying no epistemic status. On this vocabulary `open` "
            "cannot be used as a work signal: it does not distinguish an "
            "unanswered question from a term that was merely defined. That is "
            "a gap in the schema, not in the science.")
    else:
        add("The vocabulary now carries a `question` type, so `open` no longer "
            "means only \"a concept nobody gave a status\". Read the two apart: "
            "an `open` concept is a term that was defined and left alone; an "
            "`open` question is unfinished work. The tables above and the "
            "open-thread table earlier keep them separate for that reason.")
    add("")

    # ---- unreferenced ---------------------------------------------------
    add("---")
    add("")
    add("## Files the graph has never heard of")
    add("")
    add("Everything under `experiments/` that looks like a run output and "
        "appears nowhere in `entities.json`, by path or by basename. Split by "
        "whether it sits inside a directory the graph already knows as a run's "
        "`output_dir` &mdash; a sidecar of a recorded report is a much smaller "
        "matter than a file in a directory the record does not mention at all.")
    add("")
    orphans = [a for a in unreferenced["artefacts"]
               if a["classification"] == "orphan-directory"]
    sidecars = [a for a in unreferenced["artefacts"]
                if a["classification"] != "orphan-directory"]
    add("### In directories the graph does not know (%d)" % len(orphans))
    add("")
    if orphans:
        add("| File | Bytes | Written by | The graph says of that run |")
        add("|:---|---:|:---|:---|")
        for art in orphans:
            producers = ", ".join("`%s`" % p for p in art["produced_by_script"]) or "&mdash;"
            says = "; ".join("`%s`: %s" % (r["run"], r["graph_says"] or "&mdash;")
                             for r in art["producer_run_nodes"]) or "no run node"
            add("| `%s` | %s | %s | %s |" % (
                art["path"], art["size"] if art["size"] is not None else "?",
                producers, md_escape(says)))
        add("")
    else:
        add("None.")
        add("")
    add("### Sidecars of recorded output directories (%d)" % len(sidecars))
    add("")
    add("Mostly the machine-readable twin of a report the graph does carry as "
        "an artefact node. Listed for completeness, not as a problem.")
    add("")
    add("<details><summary>Show all %d</summary>" % len(sidecars))
    add("")
    for art in sidecars:
        add("- `%s` (%s bytes)" % (art["path"],
                                   art["size"] if art["size"] is not None else "?"))
    add("")
    add("</details>")
    add("")
    if unreferenced["orphan_scripts"]:
        add("### Scripts in `experiments/` with no run node (%d)"
            % len(unreferenced["orphan_scripts"]))
        add("")
        add("| Script | Executed | Code cells with outputs |")
        add("|:---|:---|:---|")
        for row in unreferenced["orphan_scripts"]:
            add("| `%s` | %s | %s |" % (
                row["path"],
                {True: "yes", False: "no", None: "not knowable"}[row["executed"]],
                cell_count(row)))
        add("")

    # ---- ledger ---------------------------------------------------------
    add("---")
    add("")
    add("## Execution ledger")
    add("")
    add("Every run whose script the graph names, and how much of that script "
        "carries executed output. This is the check that catches drift in "
        "both directions.")
    add("")
    add("| Run | Script | Code cells with outputs | Note |")
    add("|:---|:---|:---|:---|")
    for row in ledger:
        cells = cell_count(row)
        add("| `%s` | `%s` | %s | %s |" % (
            row["run"], row["script"], cells, md_escape(row["note"] or "")))
    add("")

    # ---- method ---------------------------------------------------------
    add("---")
    add("")
    add("## How this was derived, and what it cannot see")
    add("")
    add("- **Answered-but-unrecorded** takes every claim with an incoming "
        "`tests` edge and no incoming `supports` / `refutes` / `qualifies` / "
        "`corrects`, then follows each testing run to its script. For a "
        "notebook, *executed* means one or more cells carry a non-empty "
        "`outputs` array; result text is lifted from those outputs by matching "
        "a small verdict vocabulary.")
    add("- **Blockers** are parsed out of claim descriptions by literal "
        "phrase: `issue #N`, `blocked on X`, `awaits X`, `pending X`. They are "
        "grouped by the obstacle. Whether a blocker still holds is decided by "
        "looking for a file of that name in the working tree; an issue number "
        "names no file, so for those the answer is honestly \"cannot be "
        "determined here\" unless something committed says otherwise.")
    add("- **Structural blockers.** Where the graph carries `blocks` or "
        "`blocked-by` edges, those are folded into the same groups. An edge "
        "between two open questions is treated as a prerequisite &mdash; an "
        "ordering, not an obstacle &mdash; and never on its own makes a thread "
        "blocked.")
    add("- **The execution ledger** is the cheapest check here and catches "
        "drift in both directions: a notebook the record calls a scaffold that "
        "carries outputs, and a notebook credited with a finding that carries "
        "almost none.")
    add("- **Needs-compute versus answerable-from-disk** rests on three "
        "checks: does the script exist, do the inputs it declares (via "
        "`torch.load`, `json.load`, `open`, and local imports) exist, and has "
        "anything been written to its output directory.")
    add("- **This report cannot see GitHub.** Issue state, pull requests and "
        "review threads are outside the working tree. Where an issue number is "
        "the blocker, treat the verdict here as a filesystem observation only.")
    add("- **This report never edits the record.** `docs/FINDINGS.md` and "
        "`docs/JOURNEY_MAP.md` are read-only to it, by design.")
    add("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    data = load_graph()
    graph = Graph(data)
    repo_modules = repo_modules_set()

    answered = detect_answered_but_unrecorded(graph, repo_modules)
    blockers = detect_blockers(graph)
    threads = detect_open_threads(graph, answered, blockers, repo_modules)
    frontier = detect_frontier(graph)
    undeveloped = detect_undeveloped(graph)
    unreferenced = detect_unreferenced(graph)
    ledger = execution_ledger(graph)
    fruit = rank_fruit(threads, answered)

    payload = {
        "metadata": {
            "title": "ATR threads, blockers and opportunities",
            "description": (
                "Open threads derived from the ATR evidence graph and "
                "cross-checked against the working tree. Generated; do not "
                "edit by hand."),
            "generator": "docs/graph/build_threads_report.py",
            "reads": [rel(ENTITIES), "the repository working tree"],
            "graph_source_last_updated": graph.meta.get("source_last_updated"),
            "ranking_rule": [{"tier": i + 1, "name": name, "rule": text}
                             for i, (name, text) in enumerate(RANK_RULE)],
            "detectors": ["answered-but-unrecorded", "blocked-grouped-by-blocker",
                          "needs-compute-vs-answerable-from-disk", "frontier",
                          "undeveloped", "unreferenced-artefacts",
                          "execution-ledger"],
        },
        "counts": {
            "answered_but_unrecorded": len(answered),
            "answered_on_disk_unrecorded": sum(
                1 for r in answered
                if r["assessment"] == "answered-on-disk-unrecorded-in-graph"),
            "blockers": len(blockers),
            "shared_blockers": sum(1 for b in blockers if b["gate_count"] > 1),
            "open_threads": len(threads),
            "low_hanging_fruit": len(fruit),
            "frontier": len(frontier),
            "thin_concepts": len(undeveloped["thin_concepts"]),
            "silent_claims": len(undeveloped["silent_claims"]),
            "unreferenced_artefacts": len(unreferenced["artefacts"]),
            "orphan_scripts": len(unreferenced["orphan_scripts"]),
        },
        "open_status_audit": dict(Counter(
            c.get("type") for c in graph.claims.values()
            if c.get("status") == "open")),
        "answered_but_unrecorded": answered,
        "blockers": blockers,
        "open_threads": threads,
        "low_hanging_fruit": fruit,
        "frontier": frontier,
        "undeveloped": undeveloped,
        "unreferenced": unreferenced,
        "execution_ledger": ledger,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(graph, answered, blockers, threads, frontier,
                                 undeveloped, unreferenced, ledger, fruit))

    counts = payload["counts"]
    print("ATR threads report")
    print("  read      %s (%d claims, %d runs, %d edges)"
          % (rel(ENTITIES), len(graph.claims), len(graph.runs), len(graph.rels)))
    print("  tested without a verdict     %d (%d answered on disk already)"
          % (counts["answered_but_unrecorded"],
             counts["answered_on_disk_unrecorded"]))
    print("  blockers                     %d (%d gate more than one claim)"
          % (counts["blockers"], counts["shared_blockers"]))
    print("  open threads                 %d" % counts["open_threads"])
    print("  low-hanging fruit            %d" % counts["low_hanging_fruit"])
    print("  frontier findings            %d" % counts["frontier"])
    print("  thin concepts / silent claims %d / %d"
          % (counts["thin_concepts"], counts["silent_claims"]))
    print("  unreferenced artefacts       %d (%d orphan scripts)"
          % (counts["unreferenced_artefacts"], counts["orphan_scripts"]))
    print("  wrote     %s" % rel(OUT_JSON))
    print("  wrote     %s" % rel(OUT_MD))
    for record in answered:
        if record["assessment"] == "answered-on-disk-unrecorded-in-graph":
            print("  NOTE: %s is recorded as '%s' but its run has executed "
                  "results on disk" % (record["claim"], record["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
