# ATR in Context: Method Comparison & Scaling Programme

**Date:** 2026-03-20
**Purpose:** Orientate ATR within the mechanistic interpretability landscape. Inform resource planning.

---

## 1. Comparative Method Matrix

### 1a. What Each Method Reveals

| Method | What It Probes | Resolution | Output |
|---|---|---|---|
| **Logit Lens / Tuned Lens** | What the residual stream "predicts" at each layer | Per-layer, per-token | Token probability distributions per layer |
| **Activation Patching** | Causal contribution of specific components to outputs | Per-head, per-layer, per-position | Causal effect scores |
| **Linear Probes** | Whether a representation linearly encodes a concept | Per-layer | Probe accuracy (% classification) |
| **Sparse Autoencoders (SAEs)** | Decompose activations into interpretable features | Per-feature (thousands per layer) | Feature dictionaries + activation patterns |
| **Ablation Studies** | What breaks when a component is removed | Per-head, per-layer | Performance delta |
| **Representation Engineering** | Steer model behaviour via activation vectors | Per-layer, per-concept | Control vectors + behavioural change |
| **ATR (Ours)** | Weight geometry's dominant modes / attractor landscape | Per-model (global) | Basin tokens, convergence trajectories, bias profile |

### 1b. Resource Requirements

| Method | Compute | VRAM | Time per Run | Training Required? | Training Data Access? |
|---|---|---|---|---|---|
| **Logit Lens** | 1 forward pass | Model size | Seconds | No | No |
| **Activation Patching** | O(components × prompts) forward passes | Model size | Minutes–hours | No | No |
| **Linear Probes** | Forward passes + probe training | Model + probe | Hours | Yes (probe) | Yes (labelled data) |
| **SAEs** | Training run (millions of tokens) | Model + SAE (2–4×) | Days–weeks | Yes (SAE training) | Yes (activation dataset) |
| **Ablation** | O(components) forward passes | Model size | Minutes–hours | No | No |
| **Rep. Engineering** | Forward passes + PCA/mean diff | Model size | Minutes | No | Yes (contrast pairs) |
| **ATR** | N forward passes per prompt (N ≈ 50–200) | Model size | **Seconds–minutes** | **No** | **No** |

### 1c. Efficiency Analysis

| Method | Setup Complexity | Marginal Cost per New Model | Automation Potential |
|---|---|---|---|
| **Logit Lens** | Low (standard) | Very low | High |
| **Activation Patching** | Medium (requires prompt design) | Medium | Medium |
| **Linear Probes** | High (requires labelled data) | High (retrain probes) | Low |
| **SAEs** | Very high (training pipeline) | Very high (retrain SAE) | Low |
| **Ablation** | Medium | Medium | Medium |
| **Rep. Engineering** | Medium (requires contrast pairs) | Medium | Medium |
| **ATR** | **Very low** (load model, iterate) | **Very low** | **Very high** |

### 1d. What Each Method Cannot Do

| Method | Key Limitation |
|---|---|
| **Logit Lens** | Only shows prediction, not mechanism. Inaccurate in early layers. |
| **Activation Patching** | Requires known input→output pairs. Cannot discover unknown structure. |
| **Linear Probes** | Only finds what you look for. Cannot discover unknown concepts. |
| **SAEs** | Expensive. Feature interpretation is manual. Scale uncertain beyond GPT-2. |
| **Ablation** | Destructive. Cannot distinguish redundant from essential components. |
| **Rep. Engineering** | Requires concept pairs. Doesn't reveal model's "native" organisation. |
| **ATR** | **Reveals static weight geometry, not dynamic computation. Doesn't explain HOW the model processes input — only what it converges to WITHOUT input.** |

### 1e. Unique ATR Capabilities

| Capability | ATR | Nearest Alternative |
|---|---|---|
| Reveal training-data thematic structure without training-data access | Proposed (single-model evidence; cross-model programme would test it) | Rep. Engineering (partial, needs contrast pairs) |
| No labelled data required | Yes | Logit Lens, Ablation, Patching |
| No training/fine-tuning step | Yes | Logit Lens, Ablation, Patching |
| Global model characterisation (not per-prompt) | Yes | SAEs (but vastly more expensive) |
| Cross-model comparison via basin profiles | Proposed (untested) | None established |
| Bias auditing of proprietary models (API-only) | No (needs weights) | Behavioural testing |
| Seconds per run on consumer GPU | Yes | Logit Lens only |

---

## 2. ATR's Position in the Field

### What ATR Adds

ATR is **complementary** to existing methods, not a replacement. It answers a question no other method asks:

> **"What are the dominant modes of the model's weight geometry, independent of any input?"**

This is the bias question. Other methods tell you what the model does with a specific input. ATR tells you what the model does when input influence is exhausted — revealing the training data's thematic fingerprint.

### Closest Parallels

- **Power iteration** in linear algebra (find dominant eigenvector by repeated multiplication) — ATR is the nonlinear analogue
- **Room impulse response** measurement in acoustics (inject impulse, observe system response) — ATR is this for neural networks
- **Lyapunov exponent analysis** in dynamical systems (characterise stability of trajectories) — ATR maps the attractor landscape

### State-of-the-Art Impact Potential

| Claim | Evidence Level | Impact if Validated |
|---|---|---|
| Attractor basins exist with the shape and dominance shares observed | Supported (GPT-2 Small, single-model) | Confirms iterated forward map produces a discrete attractor landscape |
| Basin tokens cluster semantically rather than by BPE substring | Supported (W_E neighbourhood test on GPT-2 Small; statistical validation pending) | The attractors carry meaning, not artefact |
| Basin profiles vary predictably by model / training data | Untested | If supported, ATR becomes a model-characterisation tool |
| ATR-derived basins can stand in for training-data inspection in bias characterisation | Untested at scale | If supported on cross-model evidence, potential safety relevance |
| Basin topology correlates with model capabilities | Speculative | Transformative if true, but years from testable |

---

## 3. Compute Programme

### 3a. Research Programme: Cross-Model ATR Landscape Mapping

**Objective:** Run ATR across a systematic sweep of open-weight language models to determine whether attractor basin profiles are model-specific, architecture-specific, or scale-dependent.

### 3b. Experimental Design

| Phase | Models | Count | VRAM per Model | Estimated GPU-Hours |
|---|---|---|---|---|
| **Phase 1: GPT-2 Family** | GPT-2 Small/Medium/Large/XL | 4 | 0.5–6 GB | ~2 hours |
| **Phase 2: Pythia Suite** | Pythia 70M → 12B (8 checkpoints × 143 training snapshots) | ~1,144 | 0.3–48 GB | ~50 hours |
| **Phase 3: Architecture Diversity** | OLMo-7B, Llama-3-8B, Mistral-7B, Gemma-2B/7B | 5 | 4–32 GB | ~10 hours |
| **Phase 4: Aligned vs Base** | Llama-3-8B vs Llama-3-8B-Instruct (same weights, different alignment) | 2 | 16 GB | ~4 hours |
| **Phase 5: Scale Sweep** | Llama family 1B → 70B (if resources allow) | 4–6 | 2–140 GB | ~100 hours |

**Total estimated:** ~170 GPU-hours (A100-class), expandable to ~500 with full scale sweep.

### 3c. Resource Requirements

| Resource | Minimum | Ideal |
|---|---|---|
| GPU | 1× A100 40GB | 4× A100 80GB |
| Storage | 500 GB (model weights + `.pt` outputs) | 2 TB |
| RAM | 64 GB | 128 GB |
| Duration | 1 week (batch jobs) | 3 days (parallel) |

### 3d. What the Programme Produces

| Deliverable | Format | Value |
|---|---|---|
| Basin profiles for 10+ models | JSON + visualisations | First cross-model attractor comparison |
| Scaling laws for basin count vs model size | Statistical analysis | Novel finding |
| Bias profiles for base vs aligned models | Comparative matrices | Safety-relevant |
| Pythia training dynamics | Basin evolution across checkpoints | Developmental biology of LLMs |
| Open-source ATR toolkit | Python package + notebooks | Community contribution |

### 3e. Why HPC

- **Low total compute** — 170–500 GPU-hours is modest by HPC standards
- **High ratio of output to compute** — each ATR run produces rich, interpretable data per unit of GPU time
- **Novel research territory** — cross-model attractor-landscape comparison has not been published; results either way (basins generalise predictably, basins generalise but unpredictably, or basins fail to generalise) are publishable findings
- **Methodological cleanness** — fully deterministic at the same-machine level, no training required, reproducible in principle
- **Conditional safety relevance** — *if* the central hypothesis (basin profiles reflect training-corpus thematic structure) generalises across models, ATR could offer a route to bias characterisation without training-data access. The cross-model programme is what would test this; the relevance is conditional on that test, not established by the GPT-2 Small results alone.

### 3f. Timeline

| Week | Activity |
|---|---|
| 1 | Phase 1 + 2 (GPT-2 family + Pythia subset) — validate ATR scales |
| 2 | Phase 3 + 4 (architecture diversity + alignment comparison) |
| 3 | Phase 5 (scale sweep, if resource permits) |
| 4 | Analysis, visualisation, draft findings |

---

## 4. Development Roadmap

### Immediate (Current Setup — Local GPU)
- [x] ATR on GPT-2 Small (done)
- [ ] T_mix_LLM measurement
- [ ] Basin-sorted convergence matrix
- [ ] Automated ATR pipeline (parameterisable model, prompt set, output directory)

### Short-Term (Pre-HPC)
- [ ] ATR on GPT-2 Medium/Large (validate scaling locally)
- [ ] Standardised output format (JSON schema for basin profiles)
- [ ] Comparison metrics (basin count, depth, distribution entropy)
- [ ] Prior art survey + position paper draft

### Medium-Term (HPC Programme)
- [ ] Phases 1–5 as described above
- [ ] Cross-model analysis pipeline
- [ ] Publication-ready visualisations

### Long-Term (Post-Publication)
- [ ] Open-source ATR toolkit
- [ ] Community replication
- [ ] Extension to multimodal models
- [ ] Formal connection to mixing time mathematics

---

*Document created 2026-03-20. Living document — update as programme develops.*
