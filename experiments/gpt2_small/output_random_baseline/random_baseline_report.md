# EXP_009d3: Random Baseline — Results Summary

## Experimental Parameters
- **Trials:** 125 random Gaussian tensors
- **Seed:** 42
- **Norm calibration:** 397.18 ± 43.86 (from Stage 1)
- **Iteration schedule:** [0, 2, 3, 5, 10, 20, 50, 100]
- **Model:** GPT-2 Small (12L, 768d)

## Key Results

| Metric | Real (Stage 1) | Random Baseline |
|:---|:---|:---|
| Terminal basins | 5 | 18 |
| Position collapse at iter 100 | ~1.000 | 1.0000 |
| Cosine convergence at iter 100 | ~1.000 | 0.9256 |
| Basin overlap | — | 1/5 (20%) |

## Interpretation

### Does random input converge?
Position collapse: YES ✅
Cosine convergence: NO ❌

### Basin identity overlap
Shared basins: {'prolet'}
Real-only basins: {'till', 'Anarch', 'solidarity', 'Divine'}
Random-only basins: {'exchanged', 'arbit', 'vp', 'justified', 'relat', '―', 'ei', 'Kobe', 'instant', 'NP', 'Brah', 'Hindu', 'strike', 'Nero', 'abs', 'ision', 'cond'}

### Bootstrap significance
Random basin count: 14.1 (95% CI: [11, 17])
Real basin count: 5

## Outcome Classification

Based on the results above, this experiment falls into one of three categories:

1. **Same basins** → Eigenvoice is intrinsic to weight geometry
2. **Different basins** → Eigenvoice is manifold-specific
3. **No convergence** → ATR requires structured input

---
*Generated automatically by EXP_009d3*
