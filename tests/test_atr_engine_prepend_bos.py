"""Contract tests for the engine's BOS passthrough (issue #75).

The claim this file has to defend is not "prepend_bos works" -- that needs a
real HookedTransformer and a forward pass, and neither exists here. It is the
narrower and more dangerous one: **the default path is unchanged**. Every
result in the record was produced by an engine that called
``run_with_cache(prompt, names_filter=...)`` and nothing else. If #75 quietly
altered that call, every future run would be comparable only to itself, and the
change would be invisible in exactly the way the BOS itself was invisible.

So the tests come in two halves, split by what they need:

  * standard library only -- an AST reading of ``atr_engine.py`` that pins the
    call shape and the parameter order. Runs in any checkout, including CI,
    which does not install the model stack (see .github/workflows/graph.yml on
    why requirements.txt is not installed there).
  * torch -- the behaviour of ``_bos_kwargs`` itself. Skipped where torch is
    absent rather than faked, because a stubbed ``torch.is_tensor`` would be
    testing the stub.

Run from the repo root:

    python3 -m pytest tests -q
"""

import ast
from pathlib import Path

import pytest

# tests/test_atr_engine_prepend_bos.py -> tests -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "atr_engine.py"

ENGINE_TREE = ast.parse(ENGINE_PATH.read_text(), filename=str(ENGINE_PATH))

# The parameter lists as they stood before #75, transcribed from the engine at
# 504243c. #75 is a strict addition: these names, this order, these defaults.
# A test that only checked "prepend_bos exists" would pass just as happily on a
# signature that had reordered everything else underneath it, and a reordered
# signature silently breaks every positional caller in experiments/.
HISTORICAL_SIGNATURES = {
    "run_atr_loop": [
        ("model", None), ("prompt", None), ("layer_start", None),
        ("layer_end", None), ("max_iter", None), ("schedule", None),
        ("verbose", True),
    ],
    "run_atr_gated": [
        ("model", None), ("prompt", None), ("layer_start", None),
        ("layer_end", None), ("max_iter", 1000), ("threshold", 0.999),
        ("patience", 3), ("check_every", 10), ("check_start", 100),
        ("verbose", False), ("gate_lag", 1), ("capture_terminal", False),
        ("inject_hook_name", None), ("renorm", "seed_j"),
    ],
}

# Two per function: the seed pass and the one inside the re-injection loop.
EXPECTED_RUN_WITH_CACHE_CALLS = {"run_atr_loop": 2, "run_atr_gated": 2}


def _function(name):
    for node in ENGINE_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {ENGINE_PATH}")


def _params(func):
    """(name, default) for every positional-or-keyword parameter, in order."""
    args = func.args.posonlyargs + func.args.args
    defaults = [None] * (len(args) - len(func.args.defaults)) + [
        ast.literal_eval(d) for d in func.args.defaults
    ]
    # strict=True pins the padding above, not the AST. The two lists are equal
    # by construction -- `defaults` is padded to len(args) on the line before --
    # so this can never fire on a real signature. It fires if someone edits that
    # padding wrong, at which point a bare zip() would silently truncate and this
    # helper would go on comparing a SHORTER parameter list, quietly passing the
    # drift check it exists to fail. Free assertion on an invariant that is
    # otherwise only true by inspection.
    return list(zip([a.arg for a in args], defaults, strict=True))


def _run_with_cache_calls(node):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run_with_cache"
    ]


@pytest.mark.parametrize("name", sorted(HISTORICAL_SIGNATURES))
def test_prepend_bos_is_appended_and_defaults_to_none(name):
    params = _params(_function(name))
    assert params[-1] == ("prepend_bos", None), (
        f"{name} must take prepend_bos as its LAST parameter, defaulting to "
        "None ('use the model's own cfg.default_prepend_bos')"
    )


@pytest.mark.parametrize("name", sorted(HISTORICAL_SIGNATURES))
def test_nothing_existing_moved(name):
    params = _params(_function(name))
    assert params[:-1] == HISTORICAL_SIGNATURES[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_RUN_WITH_CACHE_CALLS))
def test_every_forward_pass_carries_the_bos_decision(name):
    calls = _run_with_cache_calls(_function(name))
    assert len(calls) == EXPECTED_RUN_WITH_CACHE_CALLS[name], (
        f"{name} has {len(calls)} run_with_cache calls; if a forward pass was "
        "added or removed, this test needs updating and so does the new call"
    )
    for call in calls:
        # A `**bos_kwargs` unpack parses as a keyword whose arg is None.
        unpacked = [k.value.id for k in call.keywords
                    if k.arg is None and isinstance(k.value, ast.Name)]
        assert "bos_kwargs" in unpacked, (
            f"a run_with_cache call in {name} does not forward **bos_kwargs, "
            "so that forward pass would use the model's default BOS regardless "
            "of what the caller asked for -- the single-position H-pos0 seed "
            "would silently gain a BOS on every iteration but the first"
        )
        assert not any(k.arg == "prepend_bos" for k in call.keywords), (
            "pass prepend_bos through bos_kwargs, never directly: on the "
            "default path the kwarg must be absent from the call, not present "
            "as None"
        )


@pytest.mark.parametrize("name", sorted(EXPECTED_RUN_WITH_CACHE_CALLS))
def test_bos_kwargs_is_derived_from_the_parameters(name):
    """bos_kwargs must be built by _bos_kwargs(prompt, prepend_bos).

    Without this the previous test is satisfiable by ``bos_kwargs = {}``.
    """
    sources = [
        n.value for n in ast.walk(_function(name))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "bos_kwargs" for t in n.targets)
    ]
    assert len(sources) == 1, f"{name} should assign bos_kwargs exactly once"
    call = sources[0]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == "_bos_kwargs"
    assert [a.id for a in call.args] == ["prompt", "prepend_bos"]


def test_no_forward_pass_elsewhere_escapes_the_passthrough():
    """Four call sites, all inside the two loops. A fifth would need wiring."""
    assert len(_run_with_cache_calls(ENGINE_TREE)) == sum(
        EXPECTED_RUN_WITH_CACHE_CALLS.values()
    )


# -- the half that needs torch ------------------------------------------------

def _engine():
    pytest.importorskip(
        "torch", reason="atr_engine imports torch; the AST tests above cover "
        "the call shape without it")
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    import atr_engine
    return atr_engine


def test_default_adds_nothing_to_the_call():
    """The whole strict-addition claim, in one assertion."""
    assert _engine()._bos_kwargs("a prompt", None) == {}


@pytest.mark.parametrize("flag", [True, False])
def test_explicit_flag_is_forwarded(flag):
    assert _engine()._bos_kwargs("a prompt", flag) == {"prepend_bos": flag}


def test_token_ids_plus_prepend_bos_is_refused():
    """TransformerLens would ignore the flag; the engine must not pretend."""
    engine = _engine()
    import torch
    with pytest.raises(ValueError, match="prepend_bos"):
        engine._bos_kwargs(torch.tensor([[50256]]), False)


def test_token_ids_alone_are_fine():
    engine = _engine()
    import torch
    assert engine._bos_kwargs(torch.tensor([[50256]]), None) == {}
