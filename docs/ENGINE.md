# The ATR engine: one source of truth

`atr_engine.py` at the root of this repository is the **canonical** ATR engine.
Every repository in the programme (this one, `atr_research`, `atr_plasticity`)
runs the same loop, and running the same loop means running the same code, not
three copies that drift.

## Why this matters

The plasticity branch states the risk exactly: if a repo reimplements the loop,
a bug in the reimplementation becomes indistinguishable from a finding. Three
subtly different gated loops would make cross-repo results incomparable without
anyone noticing. A single engine removes that whole class of false result.

## What the engine contains

| Function | Role |
|---|---|
| `run_atr_loop` | Fixed-schedule full-tensor re-injection (Act I / snapshot sweeps) |
| `run_atr_gated` | Convergence-gated loop; classifies the terminal basin at lock-in |
| `lag_scan` | Periodic-attractor census: mean cosine at lags 1..k (the F9 anti-aliasing instrument) |
| `get_top_tokens`, `get_readout_detail`, `position_argmax_ids` | Readout helpers (`ln_final -> W_U`) |

`run_atr_gated` carries the parameters the whole programme needs, all with
defaults that reproduce the historical single-window path bit-for-bit:

- `gate_lag` (default 1): the lag of the convergence gate. A period-p cycle can
  only pass a gate whose lag is a multiple of p. The default matches every
  pre-F9 run; `gate_lag=2` is what classifies the Divine bell as converged.
- `capture_terminal` (default False): also return `terminal_mean_vec`,
  `terminal_last_vec`, and a `lag_scan` over the last nine iterates.
- `inject_hook_name` (default None) and `renorm` (`"seed_j"` default, or
  `"natural_i"`): the injection-site and rescale-target controls developed for
  the Stage 2 layer-window experiments (EXP_010c).
- `prepend_bos` (default None): whether TransformerLens prepends the BOS token
  when it tokenises a string prompt. None defers to the model's own
  `cfg.default_prepend_bos` — `True` for GPT-2 by the library's global default,
  explicitly `False` for GPT-NeoX/Pythia — which is what every run in the
  record used. `True`/`False` overrides it. `run_atr_loop` takes it too, and in
  both the parameter is threaded to *every* forward pass in the loop, not just
  the seed pass. Added for issue #75.

## The BOS, and the two ways to control the input sequence

Until #75 the engine handed a bare string to `run_with_cache` at all four of
its call sites, so whether a BOS token was prepended was decided by the model
config, invisibly, at a call site that could not see it. Nobody chose it. The
engine simply could not express "run GPT-2 without a BOS", which is what the
H-pos0 single-position test (a sequence whose one token *is* the BOS) and the
caveat-17 tokenisation control both need.

There are now two ways to say what the model actually sees, and they are not
interchangeable:

- **`prepend_bos=True|False`** — an override on the tokenisation of a string
  prompt. Convenient, and the right tool for a BOS-free arm of an existing
  prompt set, because it leaves the prompts themselves alone.
- **a token-ID `prompt`** — pass a `[pos]` or `[1, pos]` integer tensor instead
  of a string and TransformerLens takes it verbatim, without a tokeniser in the
  path at all. This is the exact-sequence route: it states the sequence rather
  than asking the tokeniser to produce it. For a single-token run the
  difference is load-bearing — `prepend_bos=False` still leaves the string
  `<|endoftext|>` to be tokenised into exactly `[50256]` and nothing else,
  while `torch.tensor([[50256]])` removes the question. No new parameter was
  needed for this: the engine has always passed `prompt` straight through and
  never tokenised it itself.

Combining the two raises `ValueError`. TransformerLens applies `prepend_bos`
inside `to_tokens`, which a token-ID tensor never reaches, so the flag would be
silently ignored — and a run whose entire point is the absence of a BOS must
not be able to lie about it.

## The vendoring contract

Sessions run network-restricted, so a git submodule is avoided. Downstream repos
**vendor** this file instead:

1. Copy `atr_engine.py` verbatim.
2. Prepend a header recording the source: repo, path, and the **exact commit
   hash** it was copied from.
3. Do not diverge. If a genuine downstream-only need arises, it is a *recorded
   diff* in that header, reviewed, and the standing goal is always to upstream
   it here so the diff collapses back to zero.
4. An equivalence check in the downstream repo asserts the vendored copy's
   shared surface matches the pinned canonical version, so drift fails loudly
   in CI rather than silently in a result.

## Current state (2026-07-25)

`atr_research/_STAGE2_JSPACE/experiments/atr_engine2.py` was a recorded-diff
superset: the canonical engine plus the `capture_terminal` / `inject_hook_name`
/ `renorm` extensions. Those extensions are now upstreamed here (this commit),
so the canonical engine already carries them. `atr_research` is intentionally
untouched by this PR, so its `atr_engine2.py` still holds the extensions inline;
its recorded diff will collapse to the vendor header alone only *after* the
follow-up re-vendors it. That follow-up is mechanical: re-vendor `atr_engine2.py`
from this file at the merge commit, keep only the vendor header, and add the
equivalence check. It is tracked as its own issue and PR in that repo, not here.
Note that the canonical engine may briefly lead the fork (for example the
`natural_i` injection-site fix in this PR); the re-vendor is what re-syncs them.

`atr_plasticity` imports the engine rather than reimplementing it (see its
README), which is the same contract by a different mechanism.
