"""
ATR Engine: Activation Tensor Resonance
=========================================
Core engine for iterative re-injection of the residual stream through 
a TransformerLens HookedTransformer model.

Extracted from 01_attractor_dominance.ipynb (EXP_009d1) for cross-model reuse.

Method:
    x₀ = f(embed(prompt))
    xₙ₊₁ = f(normalise(xₙ))
    normalise(x) = x · (‖x₀‖₂ / ‖x‖₂)

See docs/TECHNICAL.md for the full formal specification.
"""

import torch
import torch.nn.functional as F


def get_top_tokens(model, resid_vector, k=5):
    """Decode a residual stream vector into top-k token predictions.
    
    Applies the Final LayerNorm before unembedding for correct decoding.
    
    Args:
        model: HookedTransformer model instance
        resid_vector: [d_model] tensor, a single position's residual stream
        k: number of top predictions to return
    
    Returns:
        List of (token_string, probability) tuples
    """
    normalized = model.ln_final(resid_vector)
    logits = normalized @ model.W_U + model.b_U
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    tokens = [model.tokenizer.decode([idx]) for idx in top_indices]
    return list(zip(tokens, top_probs.tolist()))


def get_readout_detail(model, resid_vector, k=5):
    """Confidence-aware readout for one residual vector (ATR-R1).

    Complements get_top_tokens with the confidence metrics the concordance
    audit (ATR-R3) needs: token IDs, the top-1 vs top-2 *logit* margin, and
    the full-vocabulary softmax entropy (nats).

    Args:
        model: HookedTransformer model instance
        resid_vector: [d_model] tensor, a single position's residual stream
        k: number of top predictions to return

    Returns:
        dict with top_token_ids, top_token_strings, top_token_probs,
        top_logit_margin (logit[top1] - logit[top2]), entropy.
    """
    normalized = model.ln_final(resid_vector)
    logits = normalized @ model.W_U + model.b_U
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    top_logits = logits[top_indices]
    ids = [int(idx) for idx in top_indices]
    margin = float(top_logits[0] - top_logits[1]) if top_logits.numel() > 1 else 0.0
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    return {
        "top_token_ids": ids,
        "top_token_strings": [model.tokenizer.decode([idx]) for idx in ids],
        "top_token_probs": [float(p) for p in top_probs],
        "top_logit_margin": margin,
        "entropy": entropy,
    }


def position_argmax_ids(model, tensor):
    """Argmax token id + decoded string for every position (ID-first trace, ATR-R2).

    Vectorised over positions; argmax of logits equals argmax of softmax, so
    the decoded strings match a per-position top-1 readout exactly.

    Args:
        model: HookedTransformer model instance
        tensor: [seq_len, d_model] residual stream tensor

    Returns:
        (id_list, string_list): token ids and their decoded strings per position.
    """
    normalized = model.ln_final(tensor)
    logits = normalized @ model.W_U + model.b_U
    id_list = [int(idx) for idx in logits.argmax(dim=-1)]
    string_list = [model.tokenizer.decode([idx]) for idx in id_list]
    return id_list, string_list


def run_atr_loop(model, prompt, layer_start, layer_end, max_iter, schedule, verbose=True):
    """
    Activation Tensor Resonance loop: iteratively re-inject the ENTIRE 
    residual stream tensor (all token positions) through the layer slice.
    
    This is a nonlinear analogue of power iteration: it converges to fixed
    points of the full transformer forward map.
    
    Args:
        model: HookedTransformer model instance
        prompt: string prompt text
        layer_start: first layer index (typically 0)
        layer_end: last layer index (typically model.cfg.n_layers - 1)
        max_iter: maximum number of iterations (typically 100)
        schedule: list of iteration numbers at which to record snapshots
        verbose: if True, print progress at each snapshot
    
    Returns:
        List of snapshot dicts at each scheduled iteration, containing:
        - iteration, tensor, last_vector, mean_vector
        - last_norm, mean_norm, tensor_norm
        - top_tokens, all_position_tokens
        - cosine_sim_last, cosine_sim_mean, position_similarity
    """
    snapshots = []
    hook_point_read = f"blocks.{layer_end}.hook_resid_post"
    hook_point_write = f"blocks.{layer_start}.hook_resid_pre"
    
    # Initial forward pass
    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt,
            names_filter=lambda n: n == hook_point_read
        )
    
    current_tensor = cache[hook_point_read][0].clone()
    seq_len = current_tensor.shape[0]
    initial_norm = current_tensor.norm().item()
    
    last_vec = current_tensor[-1, :].clone()
    mean_vec = current_tensor.mean(dim=0).clone()
    
    # Record iteration 0 snapshot
    if 0 in schedule:
        top_tokens_last = get_top_tokens(model, last_vec)
        readout = get_readout_detail(model, last_vec)
        all_pos_ids, all_pos_tokens = position_argmax_ids(model, current_tensor)
        snapshots.append({
            "iteration": 0,
            "tensor": current_tensor.clone().cpu(),
            "last_vector": last_vec.clone().cpu(),
            "mean_vector": mean_vec.clone().cpu(),
            "last_norm": last_vec.norm().item(),
            "mean_norm": mean_vec.norm().item(),
            "tensor_norm": current_tensor.norm().item(),
            "top_tokens": top_tokens_last,
            "all_position_tokens": all_pos_tokens,
            "top_token_ids_last": readout["top_token_ids"],
            "top_token_strings_last": readout["top_token_strings"],
            "top_token_probs_last": readout["top_token_probs"],
            "top_logit_margin_last": readout["top_logit_margin"],
            "entropy_last": readout["entropy"],
            "all_position_token_ids": all_pos_ids,
            "all_position_token_strings": all_pos_tokens,
            "cosine_sim_last": 1.0,
            "cosine_sim_mean": 1.0,
            "position_similarity": 1.0,
        })
    
    prev_last = last_vec.clone()
    prev_mean = mean_vec.clone()
    
    # Iterative re-injection loop
    for i in range(1, max_iter + 1):
        # L2 normalise to maintain energy level: ‖xₙ‖₂ = ‖x₀‖₂
        current_norm = current_tensor.norm().item()
        if current_norm > 0:
            current_tensor = current_tensor * (initial_norm / current_norm)
        
        inject_tensor = current_tensor.clone()
        
        def injection_hook(resid, hook, tensor=inject_tensor):
            resid[0, :, :] = tensor
            return resid
        
        model.add_hook(hook_point_write, injection_hook)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prompt,
                    names_filter=lambda n: n == hook_point_read
                )
        finally:
            model.reset_hooks()
        
        current_tensor = cache[hook_point_read][0].clone()
        last_vec = current_tensor[-1, :].clone()
        mean_vec = current_tensor.mean(dim=0).clone()
        
        if i in schedule:
            cos_sim_last = F.cosine_similarity(
                last_vec.unsqueeze(0), prev_last.unsqueeze(0)
            ).item()
            cos_sim_mean = F.cosine_similarity(
                mean_vec.unsqueeze(0), prev_mean.unsqueeze(0)
            ).item()
            
            # Position collapse metric
            pos_norms = current_tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normalized_positions = current_tensor / pos_norms
            pos_sim_matrix = normalized_positions @ normalized_positions.T
            mask = ~torch.eye(seq_len, dtype=torch.bool, device=pos_sim_matrix.device)
            position_similarity = pos_sim_matrix[mask].mean().item()
            
            top_tokens_last = get_top_tokens(model, last_vec)
            readout = get_readout_detail(model, last_vec)
            all_pos_ids, all_pos_tokens = position_argmax_ids(model, current_tensor)

            snapshots.append({
                "iteration": i,
                "tensor": current_tensor.clone().cpu(),
                "last_vector": last_vec.clone().cpu(),
                "mean_vector": mean_vec.clone().cpu(),
                "last_norm": last_vec.norm().item(),
                "mean_norm": mean_vec.norm().item(),
                "tensor_norm": current_tensor.norm().item(),
                "top_tokens": top_tokens_last,
                "all_position_tokens": all_pos_tokens,
                "top_token_ids_last": readout["top_token_ids"],
                "top_token_strings_last": readout["top_token_strings"],
                "top_token_probs_last": readout["top_token_probs"],
                "top_logit_margin_last": readout["top_logit_margin"],
                "entropy_last": readout["entropy"],
                "all_position_token_ids": all_pos_ids,
                "all_position_token_strings": all_pos_tokens,
                "cosine_sim_last": cos_sim_last,
                "cosine_sim_mean": cos_sim_mean,
                "position_similarity": position_similarity,
            })
            
            if verbose:
                print(f"  iter {i:>3}: top='{top_tokens_last[0][0].strip()}', "
                      f"cos_mean={cos_sim_mean:.4f}, pos_collapse={position_similarity:.4f}")
        
        prev_last = last_vec.clone()
        prev_mean = mean_vec.clone()

    return snapshots


def run_atr_gated(model, prompt, layer_start, layer_end, max_iter=1000,
                  threshold=0.999, patience=3, check_every=10, check_start=100,
                  verbose=False, gate_lag=1):
    """Convergence-gated ATR loop (early-stop variant of run_atr_loop).

    Iterates the full-tensor re-injection until the tensor stops moving, then
    classifies the terminal basin *at lock-in* rather than at a fixed horizon.
    Lock-in = ``cos_sim_mean`` (mean-vector cosine between iterate t and
    iterate t - ``gate_lag``) stays above ``threshold`` for ``patience``
    consecutive checks, checked every ``check_every`` iterations once
    ``check_start`` is passed. If lock-in never occurs, runs to ``max_iter``
    and classifies there.

    ``gate_lag`` (int >= 1, default 1) generalises the gate. The default
    reproduces the historical consecutive-iteration comparison exactly, so
    existing callers behave as before. A period-p limit cycle can only pass
    a gate whose lag is a multiple of p: the Divine period-2 bell holds its
    lag-1 cosine at 0.685 forever (never passes) but reads 1.0 at
    ``gate_lag=2``. Assumes ``check_start`` >= ``gate_lag`` (true for any
    realistic setting).

    Lean by design (one readout decode, at the end) so a 125-prompt × 1000-iter
    sweep stays forward-pass-bound.

    Returns a dict:
        terminal_token, terminal_token_id, terminal_prob,
        lock_in_iter (None if never locked), converged (bool),
        n_iters (iterations actually run), final_cos_sim_mean,
        top_logit_margin, entropy.
    """
    hook_point_read = f"blocks.{layer_end}.hook_resid_post"
    hook_point_write = f"blocks.{layer_start}.hook_resid_pre"

    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt, names_filter=lambda n: n == hook_point_read
        )
    current_tensor = cache[hook_point_read][0].clone()
    initial_norm = current_tensor.norm().item()
    # Oldest-first buffer of the last gate_lag mean vectors: entry 0 is the
    # iterate gate_lag steps back once i >= gate_lag.
    mean_history = [current_tensor.mean(dim=0).clone()]

    consecutive = 0
    lock_in_iter = None
    final_cos = 1.0
    i = 0

    for i in range(1, max_iter + 1):
        current_norm = current_tensor.norm().item()
        if current_norm > 0:
            current_tensor = current_tensor * (initial_norm / current_norm)

        inject_tensor = current_tensor.clone()

        def injection_hook(resid, hook, tensor=inject_tensor):
            resid[0, :, :] = tensor
            return resid

        model.add_hook(hook_point_write, injection_hook)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prompt, names_filter=lambda n: n == hook_point_read
                )
        finally:
            model.reset_hooks()

        current_tensor = cache[hook_point_read][0].clone()
        mean_vec = current_tensor.mean(dim=0).clone()

        if i >= check_start and i % check_every == 0:
            cos = F.cosine_similarity(
                mean_vec.unsqueeze(0), mean_history[0].unsqueeze(0)
            ).item()
            final_cos = cos
            consecutive = consecutive + 1 if cos > threshold else 0
            if verbose:
                print(f"    iter {i:>4}: cos_mean={cos:.5f} streak={consecutive}")
            if consecutive >= patience:
                lock_in_iter = i
                break

        mean_history.append(mean_vec)
        if len(mean_history) > gate_lag:
            mean_history.pop(0)

    last_vec = current_tensor[-1, :].clone()
    top = get_top_tokens(model, last_vec, k=1)[0]
    detail = get_readout_detail(model, last_vec)
    return {
        "terminal_token": top[0],
        "terminal_token_id": detail["top_token_ids"][0],
        "terminal_prob": top[1],
        "lock_in_iter": lock_in_iter,
        "converged": lock_in_iter is not None,
        "n_iters": i,
        "final_cos_sim_mean": final_cos,
        "top_logit_margin": detail["top_logit_margin"],
        "entropy": detail["entropy"],
    }


def lag_scan(iterates, max_lag=8):
    """Mean cosine similarity between iterates k apart, for k = 1..max_lag.

    Census instrument for periodic attractors: a lag-1 gate can never pass a
    period-2 limit cycle, and any single-lag gate aliases every period that
    does not divide its lag. Scanning k = 1..max_lag makes the period readable
    from the pattern: a fixed point scores ~1.0 at every lag, a period-p cycle
    only at multiples of p, a drifting state at none (cosine decays with lag).

    Pure tensor arithmetic, no model needed.

    Args:
        iterates: consecutive iterates in order, with no gaps (a list of [d]
            tensors, or a stacked [n, d] tensor); e.g. the mean_vector or
            last_vector at iterations t, t+1, t+2, ...
        max_lag: largest lag to scan.

    Returns:
        dict {k: mean cosine over the n - k available pairs} for each lag
        k in 1..max_lag that has at least one pair.
    """
    if not torch.is_tensor(iterates):
        iterates = torch.stack(list(iterates))
    result = {}
    for k in range(1, max_lag + 1):
        if k >= iterates.shape[0]:
            break
        cos = F.cosine_similarity(iterates[k:], iterates[:-k], dim=-1)
        result[k] = cos.mean().item()
    return result
