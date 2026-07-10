"""
ATR Engine — Activation Tensor Resonance
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
        resid_vector: [d_model] tensor — a single position's residual stream
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
        resid_vector: [d_model] tensor — a single position's residual stream
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
        (id_list, string_list) — token ids and their decoded strings per position.
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
    
    This is a nonlinear analogue of power iteration — converges to fixed 
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
