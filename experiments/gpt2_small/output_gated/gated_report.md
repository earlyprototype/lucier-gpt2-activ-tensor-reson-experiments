# Section 5 — GPT-2 Small Convergence-Gated Re-sweep

- Prompts: 125  |  gate: cos_sim_mean > 0.999 x3 (every 10 iters past 100), max_iter=1000
- Locked in (converged): 91/125  |  ran to 1000: 34/125
- Lock-in iteration: min=120, median=120, max=120

## Basin shares: iter 100 (published) vs at lock-in

| Basin | @100 | @lock-in | delta |
|:---|---:|---:|---:|
| `prolet` | 44 (35.2%) | 54 (43.2%) | +10 |
| `Divine` | 34 (27.2%) | 34 (27.2%) | +0 |
| `Anarch` | 26 (20.8%) | 17 (13.6%) | -9 |
| `till` | 19 (15.2%) | 19 (15.2%) | +0 |
| `solidarity` | 2 (1.6%) | 1 (0.8%) | -1 |

## `till` hypothesis (was 19 prompts @100)

Where the iter-100 `till` prompts end at lock-in:
- `till`: 19

**`till` retention: 19/19** (largely stable).