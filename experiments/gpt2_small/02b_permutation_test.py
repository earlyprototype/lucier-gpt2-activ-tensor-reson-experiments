"""
Permutation test for the "all-warm" cross-similarity finding (FINDINGS.md caveat 4).

Observed claim (Session 02): the 14x14 pairwise cosine matrix of the 5 basin +
9 waypoint tokens in GPT-2 Small's embedding matrix W_E has ALL 91 off-diagonal
pairs positive, range ~0.18-0.47.

Pre-registered design (RESEARCH_NOTE / caveat 4): sample N random 14-token sets
from the vocabulary, compute their pairwise cosine matrices, and ask how often
randomness reproduces the observation. Three statistics, increasing strictness:

  S1  all-positive: every off-diagonal pair > 0
  S2  min off-diagonal cosine >= observed min
  S3  mean off-diagonal cosine >= observed mean

Context statistic: the global mean pairwise cosine of random token pairs —
GPT-2's embedding space is known to be anisotropic (a shared mean direction
makes most cosines positive), so S1 alone may be weak; S2/S3 carry the load.

Run:  py -3.11 -X utf8 experiments/gpt2_small/02b_permutation_test.py
"""
import json
import pathlib

import numpy as np
import torch
from transformer_lens import HookedTransformer

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "output_permutation"
OUT.mkdir(exist_ok=True)

N_PERM = 10_000
SET_SIZE = 14
SEED = 1969  # the year of the room

# Canonical token IDs from Session 01 (docs/sessions/SESSION_01.md §8)
TOKENS = {
    # basins
    "prolet": 22758, "Divine": 13009, "Anarch": 32229,
    "solidarity": 17803, "till": 10597,
    # waypoints
    "capit": 46964, "injustice": 21942, "Fem": 31149, "Rousse": 42849,
    "Ag": 10262, "FT": 9792, "ash": 1077, "Canad": 2294, "Zero": 28667,
}


def offdiag(mat):
    return mat[~np.eye(mat.shape[0], dtype=bool)]


def main():
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    W_E = model.W_E.detach().cpu().numpy()  # [50257, 768]

    # verify the recorded IDs still decode to the expected strings
    for name, tid in TOKENS.items():
        dec = model.tokenizer.decode([tid]).strip()
        assert dec == name, f"token id drift: {tid} decodes to {dec!r}, expected {name!r}"

    W = W_E / np.linalg.norm(W_E, axis=1, keepdims=True)
    ids = np.array(list(TOKENS.values()))

    obs = W[ids] @ W[ids].T
    obs_off = offdiag(obs)
    observed = {
        "n_pairs": int(obs_off.size) // 2,
        "all_positive": bool((obs_off > 0).all()),
        "min": float(obs_off.min()),
        "mean": float(obs_off.mean()),
        "max": float(obs_off.max()),
    }
    print(f"OBSERVED: 91 pairs all-positive={observed['all_positive']} "
          f"min={observed['min']:.3f} mean={observed['mean']:.3f} max={observed['max']:.3f}")

    rng = np.random.default_rng(SEED)
    vocab = W.shape[0]
    s1 = s2 = s3 = 0
    rand_mins, rand_means = np.empty(N_PERM), np.empty(N_PERM)
    for k in range(N_PERM):
        sample = rng.choice(vocab, size=SET_SIZE, replace=False)
        m = W[sample] @ W[sample].T
        od = offdiag(m)
        rand_mins[k] = od.min()
        rand_means[k] = od.mean()
        s1 += od.min() > 0
        s2 += od.min() >= observed["min"]
        s3 += od.mean() >= observed["mean"]

    # global anisotropy context: mean cosine over many random pairs
    pairs = rng.choice(vocab, size=(200_000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    global_mean_cos = float((W[pairs[:, 0]] * W[pairs[:, 1]]).sum(1).mean())

    results = {
        "design": {"n_permutations": N_PERM, "set_size": SET_SIZE, "seed": SEED,
                   "sampling": "uniform over full vocab, without replacement per set"},
        "observed": observed,
        "null": {
            "S1_all_positive_count": int(s1), "S1_p": (s1 + 1) / (N_PERM + 1),
            "S2_min_geq_observed_count": int(s2), "S2_p": (s2 + 1) / (N_PERM + 1),
            "S3_mean_geq_observed_count": int(s3), "S3_p": (s3 + 1) / (N_PERM + 1),
            "random_min_median": float(np.median(rand_mins)),
            "random_mean_median": float(np.median(rand_means)),
        },
        "context": {"global_mean_pairwise_cosine": global_mean_cos},
    }
    (OUT / "permutation_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["null"], indent=2))
    print(f"context: global mean pairwise cosine = {global_mean_cos:.3f}")
    print(f"[SAVED] {OUT / 'permutation_results.json'}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
