#!/usr/bin/env python3
"""Does the canonical record still match the repository?

The knowledge graph under ``docs/graph/`` is built by *parsing* the record --
``docs/FINDINGS.md`` and ``docs/JOURNEY_MAP.md``.  That makes it exactly as
current as those documents and no more: if the record goes stale, the graph
reproduces the staleness faithfully and cannot see it.  Every check here
therefore has one shape, and only one:

        what the record CLAIMS   vs   what the repository CONTAINS

The motivating case is H4.  FINDINGS.md, JOURNEY_MAP.md, README.md and the
notebook's own status banner all say `spectral_resonance.ipynb` was never run.
The notebook on disk carries executed outputs in every one of its code cells,
and cell 9 prints a verdict ("NOT SUPPORTED ... Mean |cos sim| 0.2387").  No
amount of reading the record catches that.  Looking at the disk does, in about
a second.  This script is that second, run on every push.

THE CHECKS
----------

  A  record says not run, disk says executed
     A run the record calls not-run / scaffolded / untested, whose notebook
     carries executed output cells, or whose declared output_dir holds
     artefacts.  Three limbs: the graph's own run nodes; not-run sentences in
     the record that name a script; and one record document calling a named
     test not-run while another records it resolved with an artefact on disk.

  B  record says executed, disk is missing it
     A run the record lists as performed whose script, output_dir, output_path
     is absent -- or whose declared output directory is empty, or whose
     notebook holds no executed cell and declares no outputs anywhere.

  C  the graph points at something that is not there
     Any doc_ref, script, output_dir, output_path or artefact path in
     entities.json that does not resolve on disk, plus doc_ref fragments that
     no longer match a heading in the document they cite.

  D  a notebook contradicts itself
     A status banner (the markdown above the first code cell) declaring the
     notebook unrun, over cells that carry executed output.

  E  a stale blocker
     A blocker named in the record -- "blocked on X (issue #N)", "X is
     temporarily absent" -- whose named file now exists on disk.  Work that is
     unblocked and nobody noticed.

TRUSTWORTHINESS
---------------

A guard that cries wolf gets muted, and a muted guard is worse than none: it
converts a real signal into background noise.  Two rules follow.

1. Nothing is reported as a divergence unless both sides are established --
   a record statement that can be quoted with its location, and a disk fact
   that can be shown with its path.  Anything that cannot be settled
   confidently goes to the ADVISORY section, which never affects the exit
   code.  "I could not tell" is printed as "I could not tell".

2. Known-accepted divergences live in an allowlist, ``docs/graph/.drift-allow``
   (optional; absent means empty).  One key per line, then whitespace, then
   ``#``, then the reason:

       A/experiments/gpt2_small/spectral_resonance.ipynb  # tracked in #54
       C/f4-null-model-regime/doc_ref#fragment  # heading renamed, see #61

   The comment delimiter is *whitespace followed by* ``#``, not the first ``#``
   on the line.  That matters: check C mints keys that contain a literal ``#``
   ("...#fragment"), and splitting on the first one would parse such an entry as
   a different key with a mangled reason -- so the divergence it was written to
   silence would keep failing while the entry itself was reported stale.  Keys
   themselves never contain whitespace, so the rule is unambiguous.

   The reason is required -- an allowlist entry without one is a configuration
   error and fails the run, because an unexplained silence is how a guard rots.
   Every report prints the exact key to add, so silencing is a copy-paste and
   the record of *why* is forced into the repository.

Exit codes:  0 = no divergence.  1 = at least one un-allowlisted divergence.
             2 = the check could not run (missing graph, unreadable allowlist).

Usage:  python3 docs/graph/check_record_drift.py [--root DIR] [--json]
                                                 [--allowlist PATH]

No arguments, no dependencies beyond the standard library, no network.
``--root`` points the whole check at another copy of the tree, which is how the
detectors are exercised against fault-injected fixtures without touching the
working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

ENTITIES_REL = os.path.join("docs", "graph", "_data", "entities.json")
ALLOWLIST_REL = os.path.join("docs", "graph", ".drift-allow")

# Documents that constitute the record.  FINDINGS.md and JOURNEY_MAP.md are the
# canonical scientific record and are never rewritten by this tooling; README.md
# is checked too because it makes run-status claims of its own (and does, today,
# disagree with FINDINGS.md about one of them).  The list is taken from the
# graph's metadata when present so this file does not drift from the builder.
FALLBACK_RECORD_DOCS = ["docs/FINDINGS.md", "docs/JOURNEY_MAP.md", "README.md"]

# Directories that are not part of the record or the evidence.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    ".venv", "venv", ".mypy_cache", ".ruff_cache",
}

SCRIPT_EXTS = (".ipynb", ".py")
FILE_EXTS = (
    "py", "ipynb", "md", "json", "pt", "png", "csv", "tsv", "txt",
    "yml", "yaml", "npy", "npz", "html", "mjs", "js",
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# "The record calls this not run."  Deliberately narrow: each alternative is a
# phrase that only means unexecuted.  "pending" and "open" are NOT here -- they
# describe work not yet scheduled, which is not a claim about a file on disk.
NOT_RUN = re.compile(
    r"""\b(?:
          not\ yet\ run
        | not\ (?:been\ )?run
        | never\ (?:been\ )?run
        | never\ (?:been\ )?executed
        | not\ executed
        | unexecuted
        | not\ attempted
        | designed\ but\ not\ run
        | scaffold(?:ed|\ only|s)?
        | untested
        | no\ executed\ results
        | code\ only
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# "The record says this exists / happened."  Used only to pair against a
# not-run statement about the same named subject in another document.
RESOLVED = re.compile(
    r"\bRESOLVED\b|\bwas (?:run|paid|completed)\b|\bhas been (?:run|completed)\b"
    r"|\bpaid at close\b|\bRecord: `",
    re.IGNORECASE,
)

# "This thing is absent from the repository."  Again narrow, and deliberately
# distinct from NOT_RUN: a file that is missing is a different claim from an
# experiment that was not performed.
ABSENT = re.compile(
    r"\b(?:temporarily absent|absent from|is absent|missing from|is missing"
    r"|not present|will be restored|awaiting restoration|yet to be restored"
    r"|blocked on|blocked by|awaiting|awaits)\b",
    re.IGNORECASE,
)

# A path inside backticks, inside a markdown link target, or bare. The fourth
# alternative catches a backticked *directory*: the record routinely cites a
# run's evidence as "Record: `experiments/gpt2_small/output_permutation/`", and
# a directory full of reports is exactly the kind of disk fact that settles an
# argument about whether something ran.
PATH_TOKEN = re.compile(
    r"`([^`\n]*?\.(?:" + "|".join(FILE_EXTS) + r"))`"
    r"|\(([^()\s]*?\.(?:" + "|".join(FILE_EXTS) + r"))\)"
    r"|(?<![`(\w/])((?:experiments|docs|archive|viz)/[\w./-]*?\.(?:"
    + "|".join(FILE_EXTS) + r"))(?![\w/])"
    r"|`((?:experiments|docs|archive|viz)/[\w./-]*/)`"
)

ISSUE = re.compile(r"issue\s*#(\d+)", re.IGNORECASE)

# Words that carry no identity when matching a named subject across documents.
LEAD_STOP = {
    "the", "a", "an", "and", "or", "of", "this", "that", "its", "their", "other",
    "one", "is", "was", "but", "with", "has", "have", "only", "declared", "item",
    "in", "to", "for", "on", "at", "by", "as", "not", "no", "still", "now",
}

# The tail of a named piece of work: "the W_E permutation TEST", "the depth
# CONTROL".  Used to lift a subject out of a not-run sentence that names no file.
TOPIC = (
    r"(?:test|run|probe|build|sweep|audit|control|analysis|experiment"
    r"|notebook|gate|null|check|protocol|replication|re-probe|re-gate)"
)
NAMED_WORK = re.compile(
    r"([A-Za-z0-9_][\w\-]*(?:\s+[\w\-]+){0,3}\s+" + TOPIC + r")\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def quote(text: str, limit: int = 190) -> str:
    """One-line, length-capped quotation of a record line."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} GB"


def gh_anchor(heading: str) -> str:
    """GitHub's heading anchor.

    Punctuation is deleted and every remaining space becomes one hyphen -- runs
    of spaces are NOT collapsed, which is why "Science & Mathematics" anchors as
    `science--mathematics`.  Both the collapsed and uncollapsed forms are
    accepted downstream so a stricter or looser generator is never called drift.
    """
    h = re.sub(r"\{#[^}]*\}", "", heading).strip().replace("`", "").replace("*", "")
    h = re.sub(r"[^\w\s-]", "", h.lower())
    return re.sub(r"\s", "-", h.strip())


def heading_anchors(markdown: str) -> set:
    """Every anchor the headings of a markdown document can be linked by.

    Both GitHub's literal slug and its space-collapsed form are returned, plus
    any explicit ``{#custom-id}``, so that neither a stricter nor a looser
    generator downstream is mistaken for drift.

    "## 4. Adjacent Science & Mathematics" therefore yields *both*
    `4-adjacent-science--mathematics` -- GitHub's own answer, because '&' is
    deleted rather than expanded to "and" and the two spaces it leaves behind
    become two hyphens -- and the collapsed `4-adjacent-science-mathematics`.

    This is the single definition of the rule for the whole of docs/graph/:
    build_evidence_graph.py and build_isomorphism_graph.py import it for their
    own doc_ref validation rather than each carrying a private copy, because two
    anchor resolvers that disagree is exactly the failure this file exists to
    catch, and the one place it could not catch it is inside itself.
    """
    found = set()
    for line in (markdown or "").split("\n"):
        match = re.match(r"^#{1,6}\s+(.*)$", line)
        if not match:
            continue
        heading = match.group(1).strip()
        explicit = re.search(r"\{#([^}]+)\}", heading)
        if explicit:
            found.add(explicit.group(1))
        slug = gh_anchor(heading)
        if slug:
            found.add(slug)
            found.add(re.sub(r"-+", "-", slug))
    return found


def normalise_name(text: str) -> str:
    """Lowercase, with separators flattened, for name-to-name comparison.

    Underscores count as separators, which is the whole point: it is what lets
    "blocked on the prompt library" meet `prompt_library.py`.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def inside_root(raw: str):
    """A path as written in a document -> a repo-relative path, or None.

    None means "this does not name a place inside the tree": a URL, an absolute
    path, or a path that climbs out of the root with ``..``.

    The care here is not pedantry.  The obvious version, ``raw.lstrip("./")``,
    strips a character *set* rather than a prefix, so "../experiments/foo.py"
    comes back as "experiments/foo.py" -- a citation pointing at a sibling
    checkout silently resolves to a different, root-relative file that does
    exist.  For a check whose only value is never crying wolf, that is the worst
    possible failure: it invents a divergence out of a misresolution and fails
    CI on it.  A path that leaves the root is not resolvable here, and saying so
    is the honest answer.
    """
    text = (raw or "").split("#", 1)[0].strip().replace(os.sep, "/")
    if not text or text.startswith(("http://", "https://", "mailto:")):
        return None
    if text.startswith("/") or posixpath.isabs(text):
        return None
    while text.startswith("./"):
        text = text[2:]
    if not text:
        return None
    trailing = text.endswith("/")
    normalised = posixpath.normpath(text)
    if normalised == "." or normalised == ".." or normalised.startswith("../"):
        return None
    return normalised + "/" if trailing else normalised


def paths_in(text: str) -> list:
    """Every path-looking token in a string, in order, de-duplicated."""
    out = []
    for match in PATH_TOKEN.finditer(text or ""):
        raw = match.group(1) or match.group(2) or match.group(3) or match.group(4)
        if raw and raw not in out:
            out.append(raw)
    return out


class Repo:
    """The repository as this check sees it: a root, and an index of filenames."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.files = []            # repo-relative paths
        self.by_basename = {}      # basename -> [repo-relative paths]
        self.by_stem = {}          # normalised stem -> [repo-relative paths]
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), self.root)
                rel = rel.replace(os.sep, "/")
                self.files.append(rel)
                self.by_basename.setdefault(name, []).append(rel)
                stem = normalise_name(os.path.splitext(name)[0])
                if stem:
                    self.by_stem.setdefault(stem, []).append(rel)

    # -- filesystem ---------------------------------------------------------

    def abs(self, rel: str) -> str:
        return os.path.join(self.root, rel.replace("/", os.sep))

    def exists(self, rel: str) -> bool:
        return bool(rel) and os.path.exists(self.abs(rel))

    def is_file(self, rel: str) -> bool:
        return bool(rel) and os.path.isfile(self.abs(rel))

    def size(self, rel: str) -> int:
        try:
            return os.path.getsize(self.abs(rel))
        except OSError:
            return 0

    def dir_entries(self, rel: str) -> list:
        try:
            return sorted(
                e for e in os.listdir(self.abs(rel)) if not e.startswith(".")
            )
        except OSError:
            return []

    def read(self, rel: str) -> str:
        with open(self.abs(rel), encoding="utf-8", errors="replace") as handle:
            return handle.read()

    def lines(self, rel: str) -> list:
        return self.read(rel).split("\n")

    # -- resolution ---------------------------------------------------------

    def resolve(self, raw: str):
        """A path as written in a document -> a repo-relative path, or None.

        Tried in order: as written; relative to a document's own directory is
        NOT guessed (that produces false pairs); finally a unique basename match
        anywhere in the repository.  Ambiguous basenames resolve to nothing --
        a guess here would become a false positive later.  A path that escapes
        the root resolves to nothing at all: see ``inside_root``.
        """
        rel = inside_root(raw)
        if rel is None:
            return None
        if self.exists(rel):
            return rel
        hits = self.by_basename.get(os.path.basename(rel), [])
        if len(hits) == 1:
            return hits[0]
        return None


def _joined(value) -> str:
    """A notebook `source`/`text` field as a string, whatever shape it is in."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part for part in value if isinstance(part, str))
    return ""


def notebook_state(repo: Repo, rel: str):
    """What a .ipynb says about its own execution.  None if unreadable.

    "Unreadable" includes structurally wrong, not merely unparseable as JSON.
    check_d walks every .ipynb in the tree, so a single malformed notebook --
    a `cells` array holding a string, an `outputs` entry that is not an object
    -- used to abort the whole check with an AttributeError instead of emitting
    the unreadable-notebook advisory this function exists to feed.  A drift
    check that dies on one bad file reports nothing about the other eighty.
    """
    try:
        notebook = json.loads(repo.read(rel))
    except (OSError, ValueError):
        return None
    if not isinstance(notebook, dict):
        return None
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not all(isinstance(c, dict) for c in cells):
        return None

    code = [c for c in cells if c.get("cell_type") == "code"]
    executed, counts, texts = [], [], []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs")
        outputs = outputs if isinstance(outputs, list) else []
        count = cell.get("execution_count")
        if not isinstance(count, int):
            count = None
        if count or outputs:
            executed.append(index)
            if count:
                counts.append(count)
        for out in outputs:
            if not isinstance(out, dict):
                continue
            chunk = out.get("text")
            if chunk is None:
                data = out.get("data")
                chunk = data.get("text/plain") if isinstance(data, dict) else None
            if chunk is None:
                chunk = out.get("traceback")
            if chunk:
                texts.append((index, _joined(chunk) or str(chunk)))

    banner = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            break
        if cell.get("cell_type") == "markdown":
            banner.append(_joined(cell.get("source")))

    # Files the executed outputs say were written, resolved against the
    # notebook's own directory (a notebook saving to "../_DATA/x.pt" is talking
    # about a real place in the repository).
    written = []
    home = os.path.dirname(rel)
    for _, text in texts:
        for token in re.findall(r"[\w./-]*\.(?:pt|json|png|csv|npz|npy|md)", text):
            candidate = os.path.normpath(os.path.join(home, token)).replace(os.sep, "/")
            if candidate.startswith(".."):
                continue
            if repo.is_file(candidate) and candidate not in written:
                written.append(candidate)

    return {
        "total_cells": len(cells),
        "code_cells": len(code),
        "executed_cells": executed,
        "exec_counts": counts,
        "outputs": texts,
        "banner": "\n".join(banner),
        "written": written,
    }


# A verdict a notebook printed about itself, and the numbers behind it.
VERDICT_WORD = re.compile(r"NOT SUPPORTED|SUPPORTED|REFUTED|CONFIRMED|INCONCLUSIVE")
VERDICT_NUMBER = re.compile(r"cos sim|\bp\s*[=<]|Mean |Median |Heads [<>]|\[SAVED\]")


def verdict_lines(state, limit: int = 4) -> list:
    """The report-worthy lines of a notebook's executed output.

    Verdicts first, then the numbers behind them. A number a notebook printed is
    the least deniable evidence there is that it ran, so the report carries the
    numbers themselves rather than an assertion about them.
    """
    strong, weak, seen = [], [], set()
    for index, text in state["outputs"]:
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.lower() in seen:
                continue
            if VERDICT_WORD.search(stripped):
                bucket = strong
            elif VERDICT_NUMBER.search(stripped):
                bucket = weak
            else:
                continue
            seen.add(stripped.lower())
            bucket.append("cell %d: %s" % (index, quote(stripped, 130)))
    return (strong + weak)[:limit]


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Divergence:
    """One established disagreement between the record and the repository."""

    def __init__(self, key, check, title, record, disk, resolution):
        self.key = key                 # stable allowlist key
        self.check = check             # "A".."E"
        self.title = title
        self.record = record           # [(citation, quotation)]
        self.disk = disk               # [str]
        self.resolution = resolution   # str
        self.allowed_because = None

    def as_dict(self):
        return {
            "key": self.key,
            "check": self.check,
            "title": self.title,
            "record": [{"citation": c, "says": s} for c, s in self.record],
            "disk": self.disk,
            "resolution": self.resolution,
            "allowlisted": self.allowed_because,
        }


class Advisory:
    """Something this check could not settle.  Never affects the exit code."""

    def __init__(self, kind, subject, note):
        self.kind = kind
        self.subject = subject
        self.note = note

    def as_dict(self):
        return {"kind": self.kind, "subject": self.subject, "note": self.note}


# ---------------------------------------------------------------------------
# The graph, read as claims about the world
# ---------------------------------------------------------------------------


class Record:
    """entities.json plus the record documents, indexed for questioning."""

    def __init__(self, repo: Repo):
        self.repo = repo
        self.data = json.loads(repo.read(ENTITIES_REL))
        self.claims = self.data.get("claims", []) or []
        self.runs = self.data.get("runs", []) or []
        self.sources = self.data.get("sources", []) or []
        self.relationships = self.data.get("relationships", []) or []
        self.by_id = {n["id"]: n for n in self.claims + self.runs + self.sources if "id" in n}

        metadata = self.data.get("metadata", {}) or {}
        docs = list(metadata.get("sources_parsed") or []) + list(
            metadata.get("sources_checked_not_parsed") or []
        )
        self.docs = [d for d in (docs or FALLBACK_RECORD_DOCS) if repo.is_file(d)]

        self.out_edges = {}
        self.in_edges = {}
        for edge in self.relationships:
            self.out_edges.setdefault(edge.get("from"), []).append(edge)
            self.in_edges.setdefault(edge.get("to"), []).append(edge)

    # -- record documents ---------------------------------------------------

    def doc_lines(self):
        """(doc, line_number, text) for every line of every record document."""
        for doc in self.docs:
            for number, text in enumerate(self.repo.lines(doc), 1):
                yield doc, number, text

    # -- run status as the record states it ---------------------------------

    @staticmethod
    def _cite(node) -> str:
        return "%s (entities.json: %s)" % (
            node.get("doc_ref") or "entities.json", node.get("id", "?"),
        )

    @staticmethod
    def _disposition(node) -> str:
        """A claim's disposition, which is the part that states the status."""
        text = node.get("description") or ""
        _, marker, tail = text.partition("Disposition:")
        return quote(tail if marker else text, 150)

    def run_not_run_reasons(self, run) -> list:
        """Why the record says this run did not happen.  Empty list = it did."""
        reasons = []
        description = run.get("description") or ""
        if NOT_RUN.search(description):
            reasons.append((self._cite(run), quote(description)))
        for edge in self.out_edges.get(run.get("id"), []):
            if edge.get("type") != "tests":
                continue
            target = self.by_id.get(edge.get("to")) or {}
            if target.get("status") == "untested":
                reasons.append(
                    (
                        self._cite(target),
                        "%s -- the claim this run tests, recorded status 'untested': %s"
                        % (target.get("label") or target.get("id"), self._disposition(target)),
                    )
                )
        for claim in self.claims:
            if claim.get("status") != "untested":
                continue
            if run.get("id") in (claim.get("evidence") or []):
                reasons.append(
                    (
                        self._cite(claim),
                        "%s -- recorded status 'untested', and cites this run as its evidence"
                        % (claim.get("label") or claim.get("id")),
                    )
                )
        # de-duplicate, preserving order
        seen, unique = set(), []
        for citation, says in reasons:
            if (citation, says) not in seen:
                seen.add((citation, says))
                unique.append((citation, says))
        return unique


# ---------------------------------------------------------------------------
# Check A -- record says not run, disk says executed
# ---------------------------------------------------------------------------


def disk_evidence_of_execution(repo: Repo, run) -> list:
    """Everything on disk that says this run happened."""
    evidence = []
    script = run.get("script")
    resolved = repo.resolve(script) if script else None
    if resolved and resolved.endswith(".ipynb"):
        state = notebook_state(repo, resolved)
        if state and state["executed_cells"]:
            counts = state["exec_counts"]
            span = (
                " (execution_count %d-%d)" % (min(counts), max(counts))
                if len(counts) > 1
                else (" (execution_count %d)" % counts[0] if counts else "")
            )
            evidence.append(
                "%s -- %d of %d code cells carry executed output%s; %d of %d cells total"
                % (
                    resolved,
                    len(state["executed_cells"]),
                    state["code_cells"],
                    span,
                    len(state["executed_cells"]),
                    state["total_cells"],
                )
            )
            for line in verdict_lines(state):
                evidence.append("  its own output reads -- %s" % line)
            for written in state["written"]:
                evidence.append(
                    "  %s (%s) exists -- named by the executed output"
                    % (written, human_size(repo.size(written)))
                )
    for key in ("output_dir", "output_path"):
        declared = run.get(key)
        if not declared:
            continue
        target = repo.resolve(declared) or declared.rstrip("/")
        if repo.exists(target):
            if os.path.isdir(repo.abs(target)):
                entries = repo.dir_entries(target)
                if entries:
                    evidence.append(
                        "%s holds %d artefact(s): %s"
                        % (target, len(entries), ", ".join(entries[:6]))
                    )
            else:
                evidence.append(
                    "%s exists (%s)" % (target, human_size(repo.size(target)))
                )
    return evidence


def check_a(repo: Repo, record: Record):
    """Three limbs, merged on the artefact they are talking about."""
    findings = {}   # key -> Divergence
    advisories = []

    def bucket(key, title, resolution):
        if key not in findings:
            findings[key] = Divergence(key, "A", title, [], [], resolution)
        return findings[key]

    # -- limb 1: the graph's own run nodes ---------------------------------
    for run in record.runs:
        if run.get("type") != "run":
            continue
        reasons = record.run_not_run_reasons(run)
        if not reasons:
            continue
        evidence = disk_evidence_of_execution(repo, run)
        script = repo.resolve(run.get("script")) if run.get("script") else None
        if not evidence:
            if not script and not run.get("output_dir") and not run.get("output_path"):
                advisories.append(
                    Advisory(
                        "unverifiable-run",
                        run.get("id"),
                        "the record calls it not run and names no script, output_dir or "
                        "output_path, so there is nothing on disk to check it against",
                    )
                )
            continue
        key = "A/" + (script or ("run:" + run.get("id", "?")))
        item = bucket(
            key,
            "the record says this was never run; the repository says it ran",
            "either run it for real and update the record (status, disposition and the "
            "run inventory), or -- if these outputs are a stray local execution -- clear "
            "them so the file matches the record. Do not edit FINDINGS.md or "
            "JOURNEY_MAP.md as a side effect of this check: the record is the operator's "
            "call. Open an issue and cite this key.",
        )
        for citation, says in reasons:
            item.record.append((citation, says))
        for line in evidence:
            if line not in item.disk:
                item.disk.append(line)

    # -- limb 2: not-run sentences in the record that name a script --------
    for doc, number, text in record.doc_lines():
        match = NOT_RUN.search(text)
        if not match:
            continue
        for raw in paths_in(text):
            if not raw.endswith(SCRIPT_EXTS):
                continue
            # The claim and the filename must be in the same breath.
            position = text.find(raw)
            if position >= 0 and min(
                abs(position - match.start()), abs(position + len(raw) - match.end())
            ) > 200:
                continue
            resolved = repo.resolve(raw)
            if not resolved:
                advisories.append(
                    Advisory(
                        "unresolved-path",
                        "%s:%d" % (doc, number),
                        "calls %r not run, but no such file resolves in the repository "
                        "(and no unique basename match), so its execution state is unknown"
                        % raw,
                    )
                )
                continue
            if not resolved.endswith(".ipynb"):
                continue  # a .py file carries no execution state to compare
            state = notebook_state(repo, resolved)
            if not state or not state["executed_cells"]:
                continue
            key = "A/" + resolved
            item = bucket(
                key,
                "the record says this was never run; the repository says it ran",
                "either run it for real and update the record (status, disposition and "
                "the run inventory), or -- if these outputs are a stray local execution "
                "-- clear them so the file matches the record. Do not edit FINDINGS.md "
                "or JOURNEY_MAP.md as a side effect of this check: the record is the "
                "operator's call. Open an issue and cite this key.",
            )
            citation = "%s:%d" % (doc, number)
            if citation not in [c for c, _ in item.record]:
                item.record.append((citation, quote(text)))
            for line in disk_evidence_of_execution(
                repo, {"script": resolved, "id": resolved}
            ):
                if line not in item.disk:
                    item.disk.append(line)

    # -- limb 3: one document calls it not run, another records it done ----
    doc_text = {doc: record.repo.lines(doc) for doc in record.docs}
    for doc, number, text in record.doc_lines():
        if not NOT_RUN.search(text):
            continue
        for sentence in re.split(r"(?<=[.;])\s+", text):
            if not NOT_RUN.search(sentence):
                continue
            if paths_in(sentence):
                continue  # limb 2 owns the cases that name a file
            for phrase_match in NAMED_WORK.finditer(sentence):
                phrase = phrase_match.group(1).split()
                while phrase and phrase[0].lower().strip(",") in LEAD_STOP:
                    phrase.pop(0)
                subject = " ".join(phrase).strip(" ,")
                tokens = [t for t in re.split(r"[\s\-_]+", subject.lower()) if t and t not in LEAD_STOP]
                if len(tokens) < 3:
                    continue  # too generic to identify anything
                for other_doc, lines in doc_text.items():
                    for other_number, other_text in enumerate(lines, 1):
                        if (other_doc, other_number) == (doc, number):
                            continue
                        if subject.lower() not in other_text.lower():
                            continue
                        if not RESOLVED.search(other_text):
                            continue
                        # The counter-statement must be backed by something on
                        # disk, or this is prose against prose and not our job.
                        window = lines[max(0, other_number - 6): other_number + 8]
                        artefacts = []
                        for candidate_line in window:
                            for raw in paths_in(candidate_line):
                                resolved = repo.resolve(raw)
                                if resolved and resolved not in artefacts:
                                    artefacts.append(resolved)
                        if not artefacts:
                            advisories.append(
                                Advisory(
                                    "unbacked-contradiction",
                                    "%s:%d vs %s:%d" % (doc, number, other_doc, other_number),
                                    "%r is called not run in one document and settled in "
                                    "another, but the settled statement names no file that "
                                    "exists, so the disk cannot arbitrate" % subject,
                                )
                            )
                            continue
                        key = "A/" + normalise_name(subject).replace(" ", "-")
                        item = bucket(
                            key,
                            "one record document calls %r not run; another records it "
                            "done, and the disk agrees with the second" % subject,
                            "correct the stale document. FINDINGS.md is canonical here; "
                            "the divergent statement is the one to fix, by the operator, "
                            "not by this tooling.",
                        )
                        citation = "%s:%d" % (doc, number)
                        if citation not in [c for c, _ in item.record]:
                            item.record.append((citation, quote(sentence)))
                        counter = "%s:%d" % (other_doc, other_number)
                        if counter not in [c for c, _ in item.record]:
                            item.record.append(
                                (counter, "contradicted here: " + quote(other_text))
                            )
                        for artefact in artefacts:
                            if os.path.isdir(repo.abs(artefact)):
                                entries = repo.dir_entries(artefact)
                                line = "%s holds %d artefact(s): %s" % (
                                    artefact, len(entries), ", ".join(entries[:6]),
                                )
                            else:
                                line = "%s exists (%s)" % (
                                    artefact, human_size(repo.size(artefact)),
                                )
                            if line not in item.disk:
                                item.disk.append(line)

    return list(findings.values()), advisories


# ---------------------------------------------------------------------------
# Check B -- record says executed, disk is missing it
# ---------------------------------------------------------------------------


def check_b(repo: Repo, record: Record):
    findings, advisories = [], []
    for run in record.runs:
        if run.get("type") != "run":
            continue
        if record.run_not_run_reasons(run):
            continue  # check A's territory
        citation = record._cite(run)
        says = "listed as a performed run: %s" % quote(run.get("description"), 150)
        missing = []

        for key in ("script", "output_dir", "output_path"):
            declared = run.get(key)
            if not declared:
                continue
            target = declared.split("#", 1)[0].rstrip("/")
            if repo.exists(target):
                if key == "output_dir" and not repo.dir_entries(target):
                    missing.append("%s exists but is empty (declared %s)" % (target, key))
                continue
            elsewhere = repo.resolve(declared)
            if elsewhere:
                missing.append(
                    "%s is not at the declared %s path; the only match is %s"
                    % (os.path.basename(target), key, elsewhere)
                )
            else:
                missing.append("declared %s %s does not exist" % (key, target))

        script = run.get("script")
        resolved = repo.resolve(script) if script else None
        if resolved and resolved.endswith(".ipynb"):
            state = notebook_state(repo, resolved)
            if (
                state
                and not state["executed_cells"]
                and not run.get("output_dir")
                and not run.get("output_path")
            ):
                missing.append(
                    "%s holds no executed cell (0 of %d code cells) and the run declares "
                    "no output_dir or output_path, so nothing on disk shows it ran"
                    % (resolved, state["code_cells"])
                )

        if not missing:
            if not script and not run.get("output_dir") and not run.get("output_path"):
                advisories.append(
                    Advisory(
                        "unverifiable-run",
                        run.get("id"),
                        "recorded as performed but names no script, output_dir or "
                        "output_path -- nothing on disk to check it against",
                    )
                )
            continue

        findings.append(
            Divergence(
                "B/" + (run.get("id") or "?"),
                "B",
                "the record lists this run as performed; the repository is missing it",
                [(citation, says)],
                missing,
                "restore the missing file or directory, or correct the run inventory in "
                "the record so it describes what the repository actually holds. If the "
                "artefacts were deliberately not committed, say so in the record and "
                "allowlist this key with that reason.",
            )
        )
    return findings, advisories


# ---------------------------------------------------------------------------
# Check C -- the graph points at something that is not there
# ---------------------------------------------------------------------------


def check_c(repo: Repo, record: Record):
    findings, advisories = [], []
    anchor_cache = {}

    def anchors_of(doc):
        if doc not in anchor_cache:
            anchor_cache[doc] = heading_anchors(repo.read(doc))
        return anchor_cache[doc]

    fields = [("sources", ("path",)), ("claims", ("doc_ref",)),
              ("runs", ("script", "output_dir", "output_path", "doc_ref"))]
    for collection, keys in fields:
        for node in record.data.get(collection, []) or []:
            for key in keys:
                raw = node.get(key)
                if not raw or raw.startswith(("http://", "https://")):
                    continue
                path, _, fragment = raw.partition("#")
                path = path.rstrip("/")
                if not path:
                    continue
                if not repo.exists(path):
                    elsewhere = repo.resolve(raw)
                    findings.append(
                        Divergence(
                            "C/%s/%s" % (node.get("id", "?"), key),
                            "C",
                            "the graph cites a path that does not exist",
                            [
                                (
                                    "entities.json %s[%s].%s" % (collection, node.get("id"), key),
                                    "%s -- %s" % (raw, quote(node.get("label") or node.get("title") or "", 90)),
                                )
                            ],
                            [
                                "%s is not in the repository" % path
                                + (" (nearest match: %s)" % elsewhere if elsewhere else "")
                            ],
                            "either restore the file or fix the reference in the document "
                            "the builder parses, then regenerate the graph "
                            "(python3 docs/graph/build_evidence_graph.py). The graph is a "
                            "build product: never hand-edit entities.json.",
                        )
                    )
                    continue
                if fragment and path.endswith(".md"):
                    if fragment not in anchors_of(path):
                        findings.append(
                            Divergence(
                                "C/%s/%s#fragment" % (node.get("id", "?"), key),
                                "C",
                                "the graph cites a heading that no longer exists",
                                [
                                    (
                                        "entities.json %s[%s].%s"
                                        % (collection, node.get("id"), key),
                                        raw,
                                    )
                                ],
                                [
                                    "%s exists, but no heading in it anchors as #%s"
                                    % (path, fragment)
                                ],
                                "the section was renamed or removed. Re-point the "
                                "reference in the source document and regenerate the "
                                "graph; every provenance link to this anchor is dead in "
                                "the viewer and on the published site.",
                            )
                        )
    return findings, advisories


# ---------------------------------------------------------------------------
# Check D -- a notebook contradicts itself
# ---------------------------------------------------------------------------


def check_d(repo: Repo, record: Record):
    findings, advisories = [], []
    for rel in sorted(p for p in repo.files if p.endswith(".ipynb")):
        state = notebook_state(repo, rel)
        if state is None:
            advisories.append(
                Advisory("unreadable-notebook", rel, "could not be parsed as a notebook")
            )
            continue
        banner = state["banner"]
        match = NOT_RUN.search(banner)
        if match and state["executed_cells"]:
            line = ""
            for candidate in banner.split("\n"):
                if NOT_RUN.search(candidate):
                    line = quote(candidate, 160)
                    break
            disk = [
                "%s -- %d of %d code cells carry executed output; %d of %d cells total"
                % (
                    rel,
                    len(state["executed_cells"]),
                    state["code_cells"],
                    len(state["executed_cells"]),
                    state["total_cells"],
                )
            ]
            for verdict in verdict_lines(state):
                disk.append("  its own output reads -- %s" % verdict)
            for written in state["written"]:
                disk.append(
                    "  %s (%s) exists -- named by the executed output"
                    % (written, human_size(repo.size(written)))
                )
            findings.append(
                Divergence(
                    "D/" + rel,
                    "D",
                    "a notebook's status banner contradicts its own execution state",
                    [("%s (status banner, above the first code cell)" % rel, line)],
                    disk,
                    "the banner is the first thing a reader sees and it is wrong. Either "
                    "rewrite the banner to describe the run that happened (date, verdict, "
                    "where the outputs went), or clear the outputs. This is the notebook's "
                    "own file, not the canonical record, so it can be fixed directly.",
                )
            )
        elif not state["executed_cells"] and not match and state["code_cells"]:
            advisories.append(
                Advisory(
                    "unlabelled-unrun-notebook",
                    rel,
                    "carries no executed output in any of its %d code cells and no status "
                    "banner saying so; a reader cannot tell whether it was never run or "
                    "was stripped" % state["code_cells"],
                )
            )
    return findings, advisories


# ---------------------------------------------------------------------------
# Check E -- a stale blocker
# ---------------------------------------------------------------------------


def blocker_nodes(record: Record):
    """Nodes the graph names as blockers, with what they block."""
    blockers = {}
    for edge in record.relationships:
        if edge.get("type") == "blocked-by":
            blockers.setdefault(edge.get("to"), []).append(edge.get("from"))
        elif edge.get("type") == "blocks":
            blockers.setdefault(edge.get("from"), []).append(edge.get("to"))
    return blockers


def named_files_of(repo: Repo, text: str, name_text: str, exclude=()):
    """Files a blocker names: explicit paths, then its own name read as a file.

    Two limbs with deliberately different reach.

    Explicit paths are taken from anywhere in ``text`` -- a backticked path is
    unambiguous, so the surrounding prose cannot mislead.

    The name limb reads only ``name_text``, the blocker's own label, and asks
    whether a file exists whose stem is that name: "blocked on the prompt
    library" meets `prompt_library.py`.  It is restricted to the label because
    the earlier version, which read the whole node including the prose of its
    edges, matched "JOURNEY_MAP" in a sentence that merely *cited*
    JOURNEY_MAP.md and reported three blockers as resolved that were not. A
    check that does that once gets muted forever. One-word stems are refused
    ("output", "results" would match half the record), the record documents
    themselves are excluded (the record cannot be its own blocker), and the file
    must exist for anything to be reported -- so a wrong guess stays silent.
    """
    found = []
    for raw in paths_in(text):
        resolved = repo.resolve(raw)
        if resolved in exclude:
            continue
        found.append(("named path %r" % raw, resolved, raw))
    haystack = normalise_name(name_text)
    for stem, paths in repo.by_stem.items():
        if len(stem.split()) < 2:
            continue
        if not re.search(r"(?<![\w])%s(?![\w])" % re.escape(stem), haystack):
            continue
        for path in paths:
            if path in exclude:
                continue
            found.append(("named %r" % stem, path, stem))
    unique, seen = [], set()
    for how, resolved, raw in found:
        if (resolved, raw) in seen:
            continue
        seen.add((resolved, raw))
        unique.append((how, resolved, raw))
    return unique


def check_e(repo: Repo, record: Record):
    findings, advisories = [], []
    blockers = blocker_nodes(record)

    # -- limb 1: blocker nodes in the graph --------------------------------
    for blocker_id, blocked_ids in sorted(blockers.items()):
        node = record.by_id.get(blocker_id)
        if not node:
            continue
        text = " ".join(
            [node.get("label") or "", node.get("description") or ""]
            + [
                edge.get("description") or ""
                for edge in record.relationships
                if edge.get("type") in ("blocks", "blocked-by")
                and blocker_id in (edge.get("from"), edge.get("to"))
            ]
        )
        named = named_files_of(
            repo, text, node.get("label") or "", exclude=set(record.docs)
        )
        issues = sorted(set(ISSUE.findall(text)))
        present = [(how, path) for how, path, _ in named if path]
        if present:
            blocked_labels = [
                "%s -- %s"
                % (
                    other,
                    quote((record.by_id.get(other) or {}).get("label") or "", 110),
                )
                for other in blocked_ids
            ]
            findings.append(
                Divergence(
                    "E/" + blocker_id,
                    "E",
                    "a blocker the record still treats as standing has been resolved on disk",
                    [
                        (
                            record._cite(node),
                            "%s%s -- and the record still lists %d thing(s) as blocked on it: %s"
                            % (
                                quote(node.get("label") or blocker_id, 120),
                                (" (issue %s)" % ", ".join("#" + i for i in issues))
                                if issues and "#" not in (node.get("label") or "")
                                else "",
                                len(blocked_ids),
                                "; ".join(blocked_labels),
                            ),
                        )
                    ],
                    ["%s: %s exists (%s)" % (how, path, human_size(repo.size(path)))
                     for how, path in present],
                    "the blocker is gone, so the work behind it is available now. Close "
                    "the issue, lift the 'blocked on' language from the record, and take "
                    "the unblocked threads off the shelf%s."
                    % (" (issue %s)" % ", ".join("#" + i for i in issues) if issues else ""),
                )
            )
        elif named:
            advisories.append(
                Advisory(
                    "blocker-standing",
                    blocker_id,
                    "names %s, which is still absent -- blocker confirmed standing"
                    % ", ".join(sorted({raw for _, path, raw in named if not path})),
                )
            )
        else:
            advisories.append(
                Advisory(
                    "blocker-unnamed",
                    blocker_id,
                    "%s names no file this check can resolve%s, so whether it still holds "
                    "cannot be decided from disk"
                    % (
                        quote(node.get("label") or blocker_id, 100),
                        " (only issue %s)" % ", ".join("#" + i for i in issues) if issues else "",
                    ),
                )
            )

    # -- limb 2: absence statements in the record --------------------------
    for doc, number, text in record.doc_lines():
        for sentence in re.split(r"(?<=[.;])\s+", text):
            match = ABSENT.search(sentence)
            if not match:
                continue
            for raw in paths_in(sentence):
                position = sentence.find(raw)
                # The filename must sit next to the absence claim, not merely
                # somewhere in the same sentence: "notebooks A and B import C,
                # which is temporarily absent" is a statement about C.
                distance = min(
                    abs(position + len(raw) - match.start()),
                    abs(match.end() - position),
                )
                if distance > 60:
                    continue
                resolved = repo.resolve(raw)
                if not resolved:
                    advisories.append(
                        Advisory(
                            "blocker-standing",
                            "%s:%d" % (doc, number),
                            "says %r is absent, and it is: no such file in the repository "
                            "-- statement confirmed" % raw,
                        )
                    )
                    continue
                issues = sorted(set(ISSUE.findall(sentence)))
                findings.append(
                    Divergence(
                        "E/" + resolved,
                        "E",
                        "the record says this file is missing; it is in the repository",
                        [("%s:%d" % (doc, number), quote(sentence))],
                        ["%s exists (%s)" % (resolved, human_size(repo.size(resolved)))],
                        "delete the absence note%s and re-check everything the record "
                        "parks behind it."
                        % (" and close issue %s" % ", ".join("#" + i for i in issues) if issues else ""),
                    )
                )
    return findings, advisories


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


# The comment delimiter in .drift-allow: whitespace, then '#'.  NOT the first
# '#' on the line -- check C's keys end in a literal "#fragment", and splitting
# there would make the very keys this report tells the operator to copy in
# unsilenceable (parsed as the wrong key, with the rest of the real key eaten as
# the reason, so the divergence keeps failing and the entry is simultaneously
# reported stale).  Allowlist keys never contain whitespace.
ALLOW_COMMENT = re.compile(r"\s#")


def load_allowlist(path: str):
    """key -> reason.  Returns (allow, problems); a reason is mandatory."""
    allow, problems = {}, []
    if not os.path.isfile(path):
        return allow, problems
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            delimiter = ALLOW_COMMENT.search(line)
            if delimiter:
                key = line[: delimiter.start()].strip()
                reason = line[delimiter.end():].strip()
            else:
                key, reason = line, ""
            if not key:
                problems.append("%s:%d has no key" % (path, number))
                continue
            if not reason:
                problems.append(
                    "%s:%d entry %r has no reason. Every allowlisted divergence needs "
                    "one, written after whitespace and a '#'." % (path, number, key)
                )
                continue
            allow[key] = reason
    return allow, problems


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CHECK_TITLES = {
    "A": "record says not run, disk says executed",
    "B": "record says executed, disk is missing it",
    "C": "the graph points at something that is not there",
    "D": "a notebook contradicts itself",
    "E": "a stale blocker",
}


def report(divergences, allowed, advisories, unused_allow, root):
    out = []
    write = out.append
    write("=" * 78)
    write("RECORD DRIFT CHECK -- what the record claims vs what the repository holds")
    write("root: %s" % root)
    write("=" * 78)
    write("")

    if divergences:
        write("%d DIVERGENCE(S)" % len(divergences))
        write("")
        for index, item in enumerate(divergences, 1):
            write("-" * 78)
            write(
                "DRIFT %d of %d  [%s] %s"
                % (index, len(divergences), item.check, CHECK_TITLES[item.check])
            )
            write("  %s" % item.title)
            write("")
            write("  THE RECORD SAYS")
            for citation, says in item.record:
                write("    %s" % citation)
                write("        %s" % says)
            write("")
            write("  THE DISK SAYS")
            for line in item.disk:
                write("    %s" % line)
            write("")
            write("  WHAT WOULD RESOLVE IT")
            for line in _wrap(item.resolution, 72):
                write("    %s" % line)
            write("")
            write("  key: %s" % item.key)
            write("")
    else:
        write("No divergence. Every run the record calls not-run is unexecuted on disk,")
        write("every run it calls performed has its script and outputs, every path and")
        write("heading the graph cites resolves, no notebook contradicts its own banner,")
        write("and every standing blocker still stands.")
        write("")

    if allowed:
        write("-" * 78)
        write("ALLOWLISTED (%d) -- divergences accepted in docs/graph/.drift-allow" % len(allowed))
        for item in allowed:
            write("  [%s] %s" % (item.check, item.key))
            write("        because: %s" % item.allowed_because)
        write("")

    if advisories:
        write("-" * 78)
        write("ADVISORY (%d) -- unknowns, not divergences. These do not fail the check."
              % len(advisories))
        for item in advisories:
            write("  [%s] %s" % (item.kind, item.subject))
            for line in _wrap(item.note, 68):
                write("        %s" % line)
        write("")

    if unused_allow:
        write("-" * 78)
        write("STALE ALLOWLIST (%d) -- these keys no longer match anything." % len(unused_allow))
        for key, reason in sorted(unused_allow.items()):
            write("  %s  # %s" % (key, reason))
        write("  Delete them: an allowlist nobody prunes is how the next H4 hides.")
        write("")

    write("=" * 78)
    if divergences:
        write("FAIL: %d divergence(s) between the record and the repository." % len(divergences))
        write("")
        write("Each block above gives the record's own words with their location, the")
        write("contradicting fact with its path, and what would settle it. Fix the side")
        write("that is wrong -- and note that docs/FINDINGS.md and docs/JOURNEY_MAP.md are")
        write("the canonical scientific record: correcting them is the operator's call,")
        write("never an automated edit and never a side effect of making CI green.")
        write("")
        write("If a divergence is known and accepted, record it -- with the reason -- in")
        write("docs/graph/.drift-allow:")
        write("")
        for item in divergences:
            write("    %s  # why this is accepted" % item.key)
        write("")
        write("The reason is whatever follows the whitespace-and-'#' after the key, so a")
        write("key containing a '#' can be pasted in verbatim. An entry with no reason at")
        write("all is a configuration error and fails.")
    elif allowed:
        write(
            "PASS: no un-allowlisted divergence. %d accepted in docs/graph/.drift-allow, "
            "listed above with the reason each was accepted." % len(allowed)
        )
    else:
        write("PASS: the record and the repository agree on everything this check can see.")
    write("=" * 78)
    return "\n".join(out)


def _wrap(text, width):
    words, lines, current = (text or "").split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines or [""]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the canonical record against the repository it describes.",
    )
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="repository root to check (default: this file's repository)")
    parser.add_argument("--allowlist", default=None,
                        help="allowlist file (default: <root>/docs/graph/.drift-allow)")
    parser.add_argument("--json", action="store_true",
                        help="emit the findings as JSON instead of a report")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print("cannot check %s: not a directory" % root, file=sys.stderr)
        return 2

    repo = Repo(root)
    if not repo.is_file(ENTITIES_REL):
        print(
            "cannot check the record: %s is missing.\n"
            "It is a build product -- generate it first:\n"
            "    python3 docs/graph/build_evidence_graph.py" % ENTITIES_REL,
            file=sys.stderr,
        )
        return 2

    allowlist_path = args.allowlist or os.path.join(root, ALLOWLIST_REL)
    allow, problems = load_allowlist(allowlist_path)
    if problems:
        print("allowlist is malformed, so the check cannot be trusted:", file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        print(
            "\nFormat: one key per line, then whitespace, then '#', then why it is\n"
            "accepted. The reason begins at the first '#' that follows whitespace, so a\n"
            "key may itself contain a '#':\n"
            "    A/experiments/gpt2_small/spectral_resonance.ipynb  # tracked in issue #54\n"
            "    C/f4-null-model-regime/doc_ref#fragment  # heading renamed, tracked in #61",
            file=sys.stderr,
        )
        return 2

    try:
        record = Record(repo)
    except (OSError, ValueError) as error:
        print("cannot read %s: %s" % (ENTITIES_REL, error), file=sys.stderr)
        return 2

    divergences, advisories = [], []
    for check in (check_a, check_b, check_c, check_d, check_e):
        found, notes = check(repo, record)
        divergences.extend(found)
        advisories.extend(notes)

    divergences.sort(key=lambda d: (d.check, d.key))
    advisories.sort(key=lambda a: (a.kind, a.subject))

    failing, allowed = [], []
    for item in divergences:
        if item.key in allow:
            item.allowed_because = allow[item.key]
            allowed.append(item)
        else:
            failing.append(item)
    unused = {k: v for k, v in allow.items() if k not in {d.key for d in divergences}}

    if args.json:
        print(json.dumps(
            {
                "root": root,
                "divergences": [d.as_dict() for d in failing],
                "allowlisted": [d.as_dict() for d in allowed],
                "advisories": [a.as_dict() for a in advisories],
                "stale_allowlist_keys": sorted(unused),
                "exit": 1 if failing else 0,
            },
            indent=2,
        ))
    else:
        print(report(failing, allowed, advisories, unused, root))

    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
