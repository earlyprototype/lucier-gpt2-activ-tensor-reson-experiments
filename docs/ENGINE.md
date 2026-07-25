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
so the recorded diff collapses to the vendor header alone. The follow-up in
`atr_research` is mechanical: re-vendor `atr_engine2.py` from this file at the
merge commit, keep only the vendor header, and add the equivalence check. That
work is tracked as its own issue and PR in that repo, not here.

`atr_plasticity` imports the engine rather than reimplementing it (see its
README), which is the same contract by a different mechanism.
