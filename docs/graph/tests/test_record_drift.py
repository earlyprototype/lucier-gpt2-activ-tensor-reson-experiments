"""Tests for docs/graph/check_record_drift.py.

The drift check's only value is that it never cries wolf. These tests pin the
three ways it could: an allowlist entry that cannot silence the divergence it
names, a citation that resolves to the wrong file, and one malformed notebook
taking the whole run down with it.

Run from the repo root:

    python3 -m pytest docs/graph/tests -q
"""

import importlib.util
import json
from pathlib import Path

import pytest

# docs/graph/tests/test_record_drift.py -> docs/graph/tests -> docs/graph
GRAPH_DIR = Path(__file__).resolve().parents[1]
DRIFT_PY = GRAPH_DIR / "check_record_drift.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_record_drift_under_test",
                                                  DRIFT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drift = _load_module()


# --------------------------------------------------------------------------
# The allowlist parser
# --------------------------------------------------------------------------

# The exact shape check C mints for a dead heading anchor, and the exact shape
# report() prints as the copy-paste line for it.
FRAGMENT_KEY = "C/f4-null-model-regime/doc_ref#fragment"


def write_allow(tmp_path, text):
    path = tmp_path / ".drift-allow"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_key_containing_a_hash_can_be_allowlisted(tmp_path):
    """The regression this file exists for.

    check_c mints keys ending in a literal '#fragment'. Splitting the allowlist
    line on the FIRST '#' parsed such an entry as the key 'C/<id>/doc_ref' with
    the reason 'fragment  # ...', so the divergence kept failing while the entry
    was simultaneously reported stale -- unsilenceable by construction.
    """
    path = write_allow(tmp_path, "%s  # heading renamed, tracked in #61\n" % FRAGMENT_KEY)
    allow, problems = drift.load_allowlist(path)
    assert problems == []
    assert FRAGMENT_KEY in allow
    assert allow[FRAGMENT_KEY] == "heading renamed, tracked in #61"


def test_report_copy_paste_line_round_trips(tmp_path):
    """What report() tells you to paste is what load_allowlist() accepts.

    This is the contract that actually matters: the tool prints a line, the
    operator pastes it, the divergence goes quiet. Asserted against report()'s
    own output rather than a hand-written imitation of it.
    """
    item = drift.Divergence(
        FRAGMENT_KEY, "C", "the graph cites a heading that no longer exists",
        [("entities.json claims[f4].doc_ref", "docs/FINDINGS.md#gone")],
        ["docs/FINDINGS.md exists, but no heading in it anchors as #gone"],
        "re-point the reference.",
    )
    text = drift.report([item], [], [], {}, "/repo")
    pasted = [ln.strip() for ln in text.splitlines()
              if ln.strip().startswith(FRAGMENT_KEY) and "#" in ln.strip()[len(FRAGMENT_KEY):]]
    assert pasted, "report() printed no copy-paste allowlist line for the key"

    allow, problems = drift.load_allowlist(write_allow(tmp_path, pasted[-1] + "\n"))
    assert problems == []
    assert FRAGMENT_KEY in allow

    # And it silences: the same key now matches, and is not reported stale.
    assert item.key in allow
    unused = {k: v for k, v in allow.items() if k not in {item.key}}
    assert unused == {}


def test_plain_key_still_parses(tmp_path):
    key = "A/experiments/gpt2_small/spectral_resonance.ipynb"
    allow, problems = drift.load_allowlist(
        write_allow(tmp_path, "%s  # tracked in issue #54\n" % key))
    assert problems == []
    assert allow == {key: "tracked in issue #54"}


def test_entry_without_a_reason_is_a_configuration_error(tmp_path):
    allow, problems = drift.load_allowlist(write_allow(tmp_path, "A/some/thing.ipynb\n"))
    assert allow == {}
    assert len(problems) == 1
    assert "has no reason" in problems[0]


def test_hash_key_without_a_reason_is_a_configuration_error(tmp_path):
    """A '#' inside the key must not be mistaken for the reason delimiter."""
    allow, problems = drift.load_allowlist(write_allow(tmp_path, FRAGMENT_KEY + "\n"))
    assert allow == {}
    assert len(problems) == 1
    assert FRAGMENT_KEY in problems[0]


def test_comments_and_blank_lines_are_skipped(tmp_path):
    allow, problems = drift.load_allowlist(write_allow(
        tmp_path, "# a header comment\n\n   \nB/run-x  # accepted\n"))
    assert problems == []
    assert allow == {"B/run-x": "accepted"}


def test_the_repositorys_own_allowlist_parses(tmp_path):
    """The committed .drift-allow is itself well formed.

    An empty allowlist is a legitimate, healthy state: it means every known
    divergence has been resolved (the last two entries, both tied to issue
    #54, were deleted 2026-07-31 when the operator's ruling landed and the
    record was updated). What this test guards is that whatever entries do
    exist parse cleanly, never that entries exist.
    """
    allow, problems = drift.load_allowlist(str(GRAPH_DIR / ".drift-allow"))
    assert problems == []


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------

def test_parent_relative_path_does_not_resolve_to_a_root_relative_file():
    """`lstrip("./")` strips a character set, not a prefix.

    "../experiments/foo.py" must not become "experiments/foo.py": a citation
    pointing outside the tree silently resolving to a real file inside it is how
    this check invents a divergence and fails CI on it.
    """
    assert drift.inside_root("../experiments/foo.py") is None
    assert drift.inside_root("../../etc/passwd") is None
    assert drift.inside_root("docs/../../escape.md") is None


def test_dot_slash_prefix_is_stripped_as_a_prefix():
    assert drift.inside_root("./docs/FINDINGS.md") == "docs/FINDINGS.md"
    assert drift.inside_root("././docs/FINDINGS.md") == "docs/FINDINGS.md"
    # ...and the leading characters of an ordinary path are left alone.
    assert drift.inside_root("docs/FINDINGS.md") == "docs/FINDINGS.md"
    assert drift.inside_root(".drift-allow") == ".drift-allow"


def test_absolute_and_remote_paths_do_not_resolve():
    assert drift.inside_root("/etc/passwd") is None
    assert drift.inside_root("https://example.com/x.py") is None
    assert drift.inside_root("mailto:someone@example.com") is None


def test_fragment_and_trailing_slash_handling():
    assert drift.inside_root("docs/FINDINGS.md#f4-null-model") == "docs/FINDINGS.md"
    assert drift.inside_root("experiments/gpt2_small/output_lagk/") == \
        "experiments/gpt2_small/output_lagk/"


def test_repo_resolve_refuses_an_escaping_path(tmp_path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "foo.py").write_text("x = 1\n", encoding="utf-8")
    repo = drift.Repo(str(tmp_path))
    assert repo.resolve("experiments/foo.py") == "experiments/foo.py"
    assert repo.resolve("./experiments/foo.py") == "experiments/foo.py"
    assert repo.resolve("../experiments/foo.py") is None


# --------------------------------------------------------------------------
# Malformed notebooks degrade to the advisory path
# --------------------------------------------------------------------------

MALFORMED = [
    ("cells-holds-a-string", {"cells": ["not a cell"]}),
    ("cells-holds-a-number", {"cells": [3]}),
    ("cells-is-not-a-list", {"cells": {"0": {"cell_type": "code"}}}),
    ("top-level-is-a-list", [{"cell_type": "code"}]),
]


@pytest.mark.parametrize("name,payload", MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_notebook_reads_as_unreadable(tmp_path, name, payload):
    rel = "%s.ipynb" % name
    (tmp_path / rel).write_text(json.dumps(payload), encoding="utf-8")
    repo = drift.Repo(str(tmp_path))
    assert drift.notebook_state(repo, rel) is None


def test_check_d_emits_an_advisory_instead_of_raising(tmp_path):
    """One corrupt notebook must not abort the walk over every other one."""
    (tmp_path / "corrupt.ipynb").write_text(json.dumps({"cells": ["oops"]}),
                                            encoding="utf-8")
    (tmp_path / "fine.ipynb").write_text(json.dumps({
        "cells": [
            {"cell_type": "markdown", "source": ["# Not yet run\n"]},
            {"cell_type": "code", "source": ["print(1)\n"],
             "execution_count": 1,
             "outputs": [{"output_type": "stream", "text": ["SUPPORTED\n"]}]},
        ]
    }), encoding="utf-8")
    repo = drift.Repo(str(tmp_path))

    findings, advisories = drift.check_d(repo, None)

    assert [a.subject for a in advisories if a.kind == "unreadable-notebook"] == \
        ["corrupt.ipynb"]
    # The healthy notebook next to it was still examined and still reported.
    assert [f.key for f in findings] == ["D/fine.ipynb"]


def test_notebook_with_odd_output_shapes_still_parses(tmp_path):
    """Non-dict outputs and a non-integer execution_count are tolerated."""
    (tmp_path / "odd.ipynb").write_text(json.dumps({
        "cells": [
            {"cell_type": "code", "source": "print(1)",
             "execution_count": "3",
             "outputs": ["not an output object",
                         {"output_type": "stream", "text": "NOT SUPPORTED\n"}]},
        ]
    }), encoding="utf-8")
    repo = drift.Repo(str(tmp_path))
    state = drift.notebook_state(repo, "odd.ipynb")
    assert state is not None
    assert state["code_cells"] == 1
    assert state["executed_cells"] == [0]
    assert state["exec_counts"] == []          # "3" is not an execution count
    assert any("NOT SUPPORTED" in text for _, text in state["outputs"])
