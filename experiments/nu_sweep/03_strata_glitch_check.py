"""Do the nu-sweep strata attractors point into the anomalous-token cluster?

Operator-ordered in session (2026-08-02, recorded on PR #114 / issue #113)
as an archive-only analysis: no new ATR runs, no forward passes. Inputs are
the committed Stage A archive (stage_a_results.pt) and the GPT-2 Small
embedding matrix; the cluster definitions and null models replicate
07_glitch_alignment.py exactly, so the numbers are comparable with the F13
record (the flip axis vs the same clusters).

Question. The sub-band strata lock onto tokens ('arbit', the horizontal
bar, the m032 fragments) that look like the model's rarely-trained token
family. Experiment 07 established the Divine flip axis points into that
cluster (cos -0.596 vs the geometric core, null mean about 0, p < 0.001).
This script asks the same geometric question of each Stage A level's
terminal attractor states.

Pre-stated interpretation rule (before execution, mirroring 07's
convention): a level's attractor "points into" a cluster if its top-50
vocabulary alignment fraction exceeds the 99.9th percentile of the
matched-size random null, i.e. empirical p < 0.001; the raw cosine against
the cluster's centroid-offset direction is reported alongside with its own
random null. The low-norm cluster (function words) is the contrast set: 07
found the flip axis NOT aligned with it.

Statistics per (level, cluster):
  - cos(s_hat, u): s_hat is the level's mean terminal-state direction
    (per-trial terminal_last_vec, the readout position's residual vector,
    normalised, averaged, renormalised); u = normalise(centroid(cluster) -
    global mean embedding), exactly 07's u. Null: 1000 random token sets of
    matched size.
  - top-50 fraction: of the 50 vocab rows most cosine-aligned with s_hat,
    the fraction inside the cluster. Null: 1000 random 50-token draws
    (expectation is cluster_size / vocab_size).
  - per-trial mean and range of cos(state, u), so a level whose trials
    disagree is visible.

Run from the repo root (analysis only, single-threaded, minutes):

    ATR_GPT2_LOCAL=... python3 experiments/nu_sweep/03_strata_glitch_check.py

Outputs (in experiments/nu_sweep/output/):
    strata_alignment.json      every computed number
    strata_alignment_report.md the readable table, from the data only
"""

import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
torch.set_num_threads(1)

OUT_DIR = Path(__file__).resolve().parent / "output"
ARCHIVE = OUT_DIR / "stage_a_results.pt"
N_NULL = 1000
TOP_K = 50
SEED = 42

# Cluster definitions transcribed verbatim from 07_glitch_alignment.py so
# the two records measure against identical yardsticks.
GROUP_REQUIRED = [
    " SolidGoldMagikarp", " petertodd", " attRot", " Adinida", " ertodd",
    " quickShipAvailable", " externalToEVA", " TheNitrome", " RandomRedditor",
    " StreamerBot", " davidjl", "ertodd",
]
GROUP_FAMILY = [
    " RandomRedditorWithNo", " TheNitromeFan", " TPPStreamerBot",
    " externalToEVAOnly", " externalTo", " guiActiveUn", " guiActiveUnfocused",
    " guiActive", " guiIcon", " guiName", " unfocusedRange",
    " SmartyHeaderCode", " Mechdragon", " Dragonbound", " Skydragon",
    " Leilan", " PsyNetMessage", " partName", " UCHIJ", " SetFontSize",
    " strutConnector", " oreAndOnline", " InstoreAndOnline",
    " BuyableInstoreAndOnline", " isSpecialOrderable", " inventoryQuantity",
    " channelAvailability", " soType", " soDeliveryDate", " Smartstocks",
    " natureconservancy", " largeDownload", " srfN", " srfAttach",
    "GoldMagikarp", "EStreamFrame", "reportprint", "embedreportprint",
    "cloneembedreportprint", "rawdownload", "rawdownloadcloneembedreportprint",
    "externalActionCode",
]
GROUP_ODDITIES = ["【", " 「", "……", "―"]


def load_embeddings():
    """GPT-2 Small W_E and tokenizer, offline via ATR_GPT2_LOCAL when set.
    Only the embedding matrix is needed; no forward passes run."""
    local = os.environ.get("ATR_GPT2_LOCAL")
    if local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        hf = GPT2LMHeadModel.from_pretrained(local)
        tok = GPT2TokenizerFast.from_pretrained(local)
        W_E = hf.transformer.wte.weight.detach().float()
        return W_E, tok
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    return model.W_E.detach().float(), model.tokenizer


def match_single(tok, strings):
    """Exact single-token matches, probing each candidate with and without a
    leading space (07's rule, verbatim in behaviour)."""
    hits = []
    for s in strings:
        alt = s[1:] if s.startswith(" ") else " " + s
        for v in (s, alt):
            ids = tok.encode(v)
            if len(ids) == 1:
                hits.append((v, int(ids[0])))
    seen, out = set(), []
    for s, i in hits:
        if i not in seen:
            seen.add(i)
            out.append((s, i))
    return out


def build_clusters(W_E, tok):
    """07's six clusters: geometric core, low-norm contrast, curated family."""
    V = W_E.shape[0]
    mu = W_E.mean(dim=0)
    dist_order = torch.argsort((W_E - mu).norm(dim=1))
    norm_order = torch.argsort(W_E.norm(dim=1))
    k01, k05 = round(V * 0.001), round(V * 0.005)
    core = match_single(tok, GROUP_REQUIRED + GROUP_FAMILY)
    core_ids = {i for _, i in core}
    odd = [h for h in match_single(tok, GROUP_ODDITIES)
           if h[1] not in core_ids]
    return {
        "geom_0p1": dist_order[:k01],
        "geom_0p5": dist_order[:k05],
        "lownorm_0p1": norm_order[:k01],
        "lownorm_0p5": norm_order[:k05],
        "curated_core": torch.tensor([i for _, i in core], dtype=torch.long),
        "curated_all": torch.tensor([i for _, i in core + odd],
                                    dtype=torch.long),
    }


def load_all_levels():
    """Every trial on disk, grouped by level and ordered by mean pin.

    Originally this script read Stage A's archive alone. Stage C added ten
    fine levels and the shared-pin control, so it now reads the shared
    checkpoint directory, which is a strict superset: the Stage A levels
    keep the same trials and the same numbers, and the new levels join the
    table. The instruments and nulls are unchanged."""
    ckpt_dir = ARCHIVE.parent / "checkpoints"
    results = {}
    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("*.pt")):
            r = torch.load(ckpt, map_location="cpu", weights_only=True)
            results.setdefault(r["level"], {})[r["pid"]] = r
    if not results:  # no checkpoints in this checkout: fall back to Stage A
        saved = torch.load(ARCHIVE, map_location="cpu", weights_only=True)
        results = saved["results"]
    levels = sorted(
        (lv for lv in results if results[lv]),
        key=lambda lv: sum(r["target_norm"] for r in results[lv].values())
        / len(results[lv]))
    return results, levels


def main():
    """Compute the alignment table and write the report."""
    results, levels = load_all_levels()
    W_E, tok = load_embeddings()
    V = W_E.shape[0]
    mu = W_E.mean(dim=0)
    rows_hat = W_E / W_E.norm(dim=1, keepdim=True)
    gen = torch.Generator().manual_seed(SEED)
    clusters = build_clusters(W_E, tok)
    cluster_sets = {k: set(v.tolist()) for k, v in clusters.items()}

    out = {"config": {"n_null": N_NULL, "top_k": TOP_K, "seed": SEED,
                      "rule": "points into cluster if top-50 fraction "
                              "empirical p < 0.001 vs matched random null",
                      "chance_cos_768": 0.0288},
           "clusters": {k: len(v) for k, v in clusters.items()},
           "levels": {}}

    # The top-50 null depends only on the cluster and the vocabulary, not on
    # the level's state direction, so draw it once per cluster (review round,
    # PR #114). This changes the RNG draw sequence relative to the first
    # committed run; the archived outputs are regenerated in the same commit.
    cluster_null_frac = {}
    for cname in clusters:
        draws = []
        for _ in range(N_NULL):
            ridx = set(torch.randperm(V, generator=gen)[:TOP_K].tolist())
            draws.append(len(ridx & cluster_sets[cname]) / TOP_K)
        cluster_null_frac[cname] = torch.tensor(draws)

    for level in levels:
        trials = results[level]
        vecs = torch.stack([r["terminal_last_vec"].float()
                            for r in trials.values()])
        vhat = vecs / vecs.norm(dim=1, keepdim=True)
        s_hat = vhat.mean(dim=0)
        s_hat = s_hat / s_hat.norm()
        top_ids = set(torch.topk(rows_hat @ s_hat, TOP_K).indices.tolist())
        level_out = {"n_trials": len(trials), "clusters": {}}
        for cname, ids in clusters.items():
            cen = W_E[ids].mean(dim=0)
            u = cen - mu
            u = u / u.norm()
            cos_mean_dir = float(s_hat @ u)
            per_trial = vhat @ u
            null_cos = []
            for _ in range(N_NULL):
                ridx = torch.randperm(V, generator=gen)[:len(ids)]
                ru = W_E[ridx].mean(dim=0) - mu
                null_cos.append(float(s_hat @ (ru / ru.norm())))
            null_cos_t = torch.tensor(null_cos)
            p_cos = float(((null_cos_t.abs() >= abs(cos_mean_dir))
                           .float().mean()))
            frac = len(top_ids & cluster_sets[cname]) / TOP_K
            null_frac_t = cluster_null_frac[cname]
            p_frac = float((null_frac_t >= frac).float().mean())
            level_out["clusters"][cname] = {
                "cos_mean_dir": cos_mean_dir,
                "cos_null_mean": float(null_cos_t.mean()),
                "cos_null_std": float(null_cos_t.std()),
                "p_cos_two_sided": p_cos,
                "per_trial_cos_mean": float(per_trial.mean()),
                "per_trial_cos_min": float(per_trial.min()),
                "per_trial_cos_max": float(per_trial.max()),
                "top50_fraction": frac,
                "top50_null_mean": float(null_frac_t.mean()),
                "p_top50": p_frac,
                "points_into": bool(p_frac < 0.001),
            }
        out["levels"][level] = level_out

    with open(OUT_DIR / "strata_alignment.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = [
        "# Strata alignment vs the anomalous-token cluster",
        "",
        "Archive-only analysis of the Stage A terminal states against the",
        "six cluster definitions of 07_glitch_alignment.py (F13's",
        "instrument). Pre-stated rule: a level points into a cluster if its",
        "top-50 alignment fraction clears empirical p < 0.001 against a",
        "matched random null. A random direction in 768 dimensions scores",
        "about 0.0288 absolute cosine; each cosine below carries its own",
        "matched null. Raw numbers: `strata_alignment.json`.",
        "",
        "| level | cos vs geom core (null mean, p) | top-50 in geom 0.5% "
        "(null, p) | top-50 in curated family (null, p) | top-50 in "
        "low-norm 0.5% (null, p) |",
        "|:--|:--|:--|:--|:--|",
    ]
    for level in levels:
        lo = out["levels"][level]
        g = lo["clusters"]["geom_0p1"]
        g5 = lo["clusters"]["geom_0p5"]
        c = lo["clusters"]["curated_all"]
        ln = lo["clusters"]["lownorm_0p5"]
        lines.append(
            f"| {level} | {g['cos_mean_dir']:+.3f} "
            f"({g['cos_null_mean']:+.3f}, p={g['p_cos_two_sided']:.3f}) | "
            f"{g5['top50_fraction']:.2f} ({g5['top50_null_mean']:.3f}, "
            f"p={g5['p_top50']:.3f}) | "
            f"{c['top50_fraction']:.2f} ({c['top50_null_mean']:.3f}, "
            f"p={c['p_top50']:.3f}) | "
            f"{ln['top50_fraction']:.2f} ({ln['top50_null_mean']:.3f}, "
            f"p={ln['p_top50']:.3f}) |")
    lines += [
        "",
        "Levels whose top-50 fraction clears the pre-stated rule for the",
        "geometric 0.5 percent cluster: "
        + (", ".join(lv for lv in levels
                     if out["levels"][lv]["clusters"]["geom_0p5"]
                     ["points_into"]) or "none") + ".",
        "For the curated family: "
        + (", ".join(lv for lv in levels
                     if out["levels"][lv]["clusters"]["curated_all"]
                     ["points_into"]) or "none") + ".",
        "For the low-norm contrast set: "
        + (", ".join(lv for lv in levels
                     if out["levels"][lv]["clusters"]["lownorm_0p5"]
                     ["points_into"]) or "none") + ".",
        "",
        "Every number above is regenerated by re-running this script;",
        "nothing is hand-computed. Interpretation lands in issue #113 and",
        "the findings record, not here.",
    ]
    with open(OUT_DIR / "strata_alignment_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {OUT_DIR / 'strata_alignment_report.md'}")


if __name__ == "__main__":
    main()
