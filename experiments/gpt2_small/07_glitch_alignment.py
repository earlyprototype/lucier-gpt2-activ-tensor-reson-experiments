"""
EXP: Is the Divine hinge aligned with the anomalous-token cluster? (issue #14, thread 4)

The bell anatomy (06_bell_anatomy.py) showed the Divine period-2 cycle is a
rank-1 see-saw: A = M + d, B = M - d, one global hinge direction d shared by
all positions, 95 percent mute to the readout. The phase-B riders included
tokens from the published GPT-2 anomalous-token cluster (the SolidGoldMagikarp
family, Rumbelow and Watkins 2023). This script asks the geometric question
directly: is d ALIGNED with that cluster in embedding space, or merely near it?

Method:
  1. Recover phase A from output_divine_motion/state_divine.pt and apply one
     ATR step for phase B (exact replication of 06_bell_anatomy.py). Sanity
     gate before any measurement: reproduce cos(A,B) ~ 0.6849 and
     cos(A, f(f(A))) = 1.0000 from bell_anatomy.json, else abort.
     Hinge d = (A - B)/2 at the last position, normalised. Pole convention:
     +d is the A side (A = M + d), -d is the B side (B = M - d).
  2. Identify the anomalous cluster three ways: (a) geometric: tokens whose
     W_E rows sit closest to the mean-embedding centroid (closest 0.1 and
     0.5 percent), (b) low-norm variant: bottom 0.1 and 0.5 percent by W_E
     row norm, (c) curated: known SolidGoldMagikarp-family strings matched
     against the vocab as exact single tokens (each candidate probed with and
     without leading space), plus the ideographic oddities from this repo's
     own bell readout (topA/topB of bell_anatomy.json).
  3. Measure, everything in W_E row space with cosine geometry (d lives in
     the same 768-dim residual coordinate space that W_E rows write into, so
     these cosines are well-defined; TransformerLens processed basis
     throughout, the same basis 04/06 used for chordness):
       - cos(d, u), u = normalise(cluster centroid - global mean embedding)
       - cos(d, PC1 of the cluster's centered embeddings)
       - top-50 vocab tokens by cos(row, +d) and cos(row, -d); fraction of
         each top-50 inside each cluster definition; top-15 listed per pole
       - empirical nulls for cos(d, u): 1000 random token sets of matched
         size, and 1000 norm-matched sets (members resampled within 100
         norm-rank bins)
       - the same cosines for the pivot M and for the per-position hinge
         directions (pos_alignment = 1.0 in bell_anatomy.json says the hinge
         is global, not last-position-specific; recomputed here)

Run:  cd experiments/gpt2_small
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python 07_glitch_alignment.py
      (expects output_divine_motion/state_divine.pt from 05_divine_motion.py;
       writes output_glitch/glitch_alignment.json)

If huggingface.co is unreachable, set ATR_GPT2_LOCAL to a directory
containing the standard gpt2 files (config.json, pytorch_model.bin,
vocab.json, merges.txt) and the script will load offline.
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "output_glitch")
DIVINE_OUT = os.path.join(HERE, "output_divine_motion")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

LOCAL = os.environ.get("ATR_GPT2_LOCAL")
if LOCAL:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
    hf_model = GPT2LMHeadModel.from_pretrained(LOCAL)
    tokenizer = GPT2TokenizerFast.from_pretrained(LOCAL)
    import transformer_lens.loading_from_pretrained as lfp
    _cfg = GPT2Config.from_pretrained(LOCAL)
    class _Shim:
        @staticmethod
        def from_pretrained(name, *a, **k):
            return _cfg
    lfp.AutoConfig = _Shim
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf_model,
                                              tokenizer=tokenizer, device="cpu")
else:
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()

SEED = 20260719
N_NULL = 1000
N_BINS = 100
TOPK_POLE = 50
TOPK_LIST = 15
EXP_COSAB = 0.6849116683006287    # bell_anatomy.json
GATE_TOL_AB = 2e-3
GATE_MIN_AA2 = 0.99995

# ------------------------------------------------------------- the hinge ----
# Exact replication of 06_bell_anatomy.py: load the committed phase-A state,
# one ATR step gives B, a second step verifies period-2.
st = torch.load(os.path.join(DIVINE_OUT, "state_divine.pt"), weights_only=True)
A_full = st["current_tensor"]
initial_norm = st["initial_norm"]
prompt = "The cat sat on the mat and then the"
hook_read = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
hook_write = "blocks.0.hook_resid_pre"


def step(x):
    cur = x * (initial_norm / x.norm())
    inject = cur.clone()
    def h(resid, hook, tensor=inject):
        resid[0, :, :] = tensor
        return resid
    model.add_hook(hook_write, h)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, names_filter=lambda n: n == hook_read)
    finally:
        model.reset_hooks()
    return cache[hook_read][0].clone()


B_full = step(A_full)
A2_full = step(B_full)


def norm_to(x, n):
    return x * (n / x.norm())


Bn = norm_to(B_full, initial_norm)
A2n = norm_to(A2_full, initial_norm)
A, B, A2 = A_full[-1], Bn[-1], A2n[-1]

cosAB = float(F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0)))
cosAA2 = float(F.cosine_similarity(A.unsqueeze(0), A2.unsqueeze(0)))
gate_ok = (abs(cosAB - EXP_COSAB) < GATE_TOL_AB) and (cosAA2 >= GATE_MIN_AA2)
print(f"SANITY GATE: cos(A,B)={cosAB:.6f} (expect {EXP_COSAB:.4f})  "
      f"cos(A,f(f(A)))={cosAA2:.6f} (expect 1.0000)  -> "
      f"{'PASS' if gate_ok else 'FAIL'}", flush=True)
if not gate_ok:
    print("Gate FAILED: map replication is wrong; refusing to measure a "
          "wrong hinge. Nothing written.")
    sys.exit(1)

M = (A + B) / 2
d_raw = (A - B) / 2          # as in 06: A = M + d, B = M - d
dhat = d_raw / d_raw.norm()  # +d = A pole, -d = B pole
Mhat = M / M.norm()

# -------------------------------------------------------- embedding space ----
W_E = model.W_E.detach().float()
V, D_MODEL = W_E.shape
mu = W_E.mean(dim=0)
dist_to_centroid = (W_E - mu).norm(dim=1)
row_norms = W_E.norm(dim=1)
dist_order = torch.argsort(dist_to_centroid)
norm_order = torch.argsort(row_norms)
dist_sorted, _ = torch.sort(dist_to_centroid)
norms_sorted, _ = torch.sort(row_norms)

k01 = round(V * 0.001)   # 50
k05 = round(V * 0.005)   # 251

cos_all = (W_E / row_norms.unsqueeze(1)) @ dhat   # cos(row, +d) for every row
cos_sorted, _ = torch.sort(cos_all)


def esc(s):
    """ASCII-safe rendering of a token string (unicode escapes)."""
    return s.encode("unicode_escape").decode("ascii")


def decode_ids(ids):
    return [esc(model.tokenizer.decode([int(i)])) for i in ids]


# ---- (a) geometric and (b) low-norm clusters --------------------------------
geom_01 = dist_order[:k01]
geom_05 = dist_order[:k05]
lownorm_01 = norm_order[:k01]
lownorm_05 = norm_order[:k05]

# ---- (c) curated cluster ----------------------------------------------------
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
# Ideographic / fullwidth oddities from this repo's own bell readout
# (topA/topB of bell_anatomy.json): U+3010, space+U+300C, U+2026 x2, U+2015.
GROUP_ODDITIES = ["\u3010", " \u300c", "\u2026\u2026", "\u2015"]


def match_single(strings):
    """Exact single-token matches against the actual vocab.

    Published family lists mix leading-space and spaceless forms (this repo's
    own bell riders decode spaceless, e.g. 'ertodd', 'quickShipAvailable'),
    so each candidate is probed both ways; every variant that is an exact
    single token is kept. A candidate counts as unmatched only if neither
    variant is a single token."""
    hits, misses = [], []
    for s in strings:
        alt = s[1:] if s.startswith(" ") else " " + s
        matched = []
        for v in (s, alt):
            ids = model.tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                matched.append((v, int(ids[0])))
        if matched:
            hits.extend(matched)
        else:
            misses.append(s)
    return hits, misses


def dedupe_hits(hits):
    seen, out = set(), []
    for s, i in hits:
        if i not in seen:
            seen.add(i)
            out.append((s, i))
    return out


core_hits, core_misses = match_single(GROUP_REQUIRED + GROUP_FAMILY)
odd_hits, odd_misses = match_single(GROUP_ODDITIES)
core_hits = dedupe_hits(core_hits)
odd_hits = [h for h in dedupe_hits(odd_hits)
            if h[1] not in {i for _, i in core_hits}]

curated_core = torch.tensor([i for _, i in core_hits], dtype=torch.long)
curated_all = torch.tensor([i for _, i in core_hits + odd_hits],
                           dtype=torch.long)

n_family_cand = len(set(GROUP_REQUIRED + GROUP_FAMILY))
print(f"curated: {len(core_hits)} distinct family tokens from "
      f"{n_family_cand} candidate strings "
      f"({len(core_misses)} candidates unmatched either way), "
      f"{len(odd_hits)} readout-oddity tokens; "
      f"unmatched family: {[esc(s) for s in core_misses]}", flush=True)

CLUSTERS = {
    "geom_0p1": geom_01,
    "geom_0p5": geom_05,
    "lownorm_0p1": lownorm_01,
    "lownorm_0p5": lownorm_05,
    "curated_core": curated_core,
    "curated_all": curated_all,
}
CLUSTER_SETS = {k: set(v.tolist()) for k, v in CLUSTERS.items()}


def overlap(a, b):
    return len(CLUSTER_SETS[a] & CLUSTER_SETS[b])


overlaps = {
    "geom_0p1_and_lownorm_0p1": overlap("geom_0p1", "lownorm_0p1"),
    "geom_0p5_and_lownorm_0p5": overlap("geom_0p5", "lownorm_0p5"),
    "geom_0p1_and_lownorm_0p5": overlap("geom_0p1", "lownorm_0p5"),
    "geom_0p5_and_lownorm_0p1": overlap("geom_0p5", "lownorm_0p1"),
    "curated_core_and_geom_0p1": overlap("curated_core", "geom_0p1"),
    "curated_core_and_geom_0p5": overlap("curated_core", "geom_0p5"),
    "curated_core_and_lownorm_0p1": overlap("curated_core", "lownorm_0p1"),
    "curated_core_and_lownorm_0p5": overlap("curated_core", "lownorm_0p5"),
    "curated_all_and_geom_0p5": overlap("curated_all", "geom_0p5"),
    "curated_all_and_lownorm_0p5": overlap("curated_all", "lownorm_0p5"),
}
print("cluster overlaps:", overlaps, flush=True)

# ------------------------------------------------------------ null models ----
gen = torch.Generator().manual_seed(SEED)


def null_stats(vals, cos_act):
    vals = vals.float()
    n = len(vals)
    p_abs = (1 + int((vals.abs() >= abs(cos_act)).sum())) / (n + 1)
    if cos_act < 0:
        p_signed = (1 + int((vals <= cos_act).sum())) / (n + 1)
    else:
        p_signed = (1 + int((vals >= cos_act).sum())) / (n + 1)
    return {"n": n, "mean_cos": float(vals.mean()), "std_cos": float(vals.std()),
            "mean_abs_cos": float(vals.abs().mean()),
            "max_abs_cos": float(vals.abs().max()),
            "p_abs": p_abs, "p_signed_same_direction": p_signed}


def null_random(k, cos_act):
    """1000 uniform random token sets of size k (without replacement)."""
    vals = torch.empty(N_NULL)
    for i in range(N_NULL):
        idx = torch.randperm(V, generator=gen)[:k]
        u = W_E[idx].mean(dim=0) - mu
        vals[i] = (u / u.norm()) @ dhat
    return null_stats(vals, cos_act)


norm_rank = torch.empty(V, dtype=torch.long)
norm_rank[norm_order] = torch.arange(V)
bin_of = (norm_rank * N_BINS) // V
bin_members = [torch.nonzero(bin_of == b, as_tuple=True)[0]
               for b in range(N_BINS)]


def null_norm_matched(ids, cos_act):
    """1000 sets that resample each member within its norm-rank bin."""
    S = torch.zeros(N_NULL, D_MODEL)
    for m in ids.tolist():
        pool = bin_members[int(bin_of[m])]
        pick = pool[torch.randint(len(pool), (N_NULL,), generator=gen)]
        S += W_E[pick]
    U = S / len(ids) - mu
    U = U / U.norm(dim=1, keepdim=True)
    return null_stats(U @ dhat, cos_act)


# ------------------------------------------------------ per-cluster maths ----
def cluster_report(name, ids):
    k = len(ids)
    cen = W_E[ids].mean(dim=0)
    u = cen - mu
    u = u / u.norm()
    cos_du = float(u @ dhat)
    cos_Mu = float(u @ Mhat)
    X = W_E[ids] - cen
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    pc1 = Vh[0]
    cos_dpc1 = float(pc1 @ dhat)
    member_cos = cos_all[ids]
    ranks_plus = torch.searchsorted(cos_sorted.contiguous(),
                                    member_cos.contiguous())
    pct_toward_minus_d = 1.0 - ranks_plus.float() / V
    rep = {
        "k": k,
        "cos_d_u": cos_du,
        "cos_M_u": cos_Mu,
        "cos_d_pc1": cos_dpc1,
        "abs_cos_d_pc1": abs(cos_dpc1),
        "member_cos_with_plus_d_median": float(member_cos.median()),
        "member_percentile_toward_minus_d_median":
            float(pct_toward_minus_d.median()),
        "null_random": null_random(k, cos_du),
        "null_norm_matched": null_norm_matched(ids, cos_du),
    }
    print(f"[{name}] k={k}  cos(d,u)={cos_du:+.4f}  cos(M,u)={cos_Mu:+.4f}  "
          f"cos(d,PC1)={cos_dpc1:+.4f}  "
          f"p_abs(rand)={rep['null_random']['p_abs']:.4g}  "
          f"p_abs(norm-matched)={rep['null_norm_matched']['p_abs']:.4g}",
          flush=True)
    return rep, u


alignment = {}
u_vectors = {}
for name, ids in CLUSTERS.items():
    alignment[name], u_vectors[name] = cluster_report(name, ids)

# ------------------------------------------------------------- pole scans ----
def pole_scan(sign_label, sign):
    vals, idx = torch.topk(sign * cos_all, TOPK_POLE)
    idx_list = idx.tolist()
    fractions = {name: sum(1 for i in idx_list if i in CLUSTER_SETS[name])
                 / TOPK_POLE for name in CLUSTERS}
    curated_hits = [decode_ids([i])[0] for i in idx_list
                    if i in CLUSTER_SETS["curated_all"]]
    top_list = [[decode_ids([i])[0], round(float(v), 4)]
                for i, v in zip(idx_list[:TOPK_LIST], vals[:TOPK_LIST])]
    print(f"pole {sign_label}: top-{TOPK_POLE} cluster fractions "
          f"{ {n: round(f, 3) for n, f in fractions.items()} }", flush=True)
    return {
        "top50_token_ids": idx_list,
        "top50_tokens": decode_ids(idx_list),
        "top50_cos_with_pole": [round(float(v), 4) for v in vals],
        "top50_fraction_in_cluster": fractions,
        "curated_members_in_top50": curated_hits,
        f"top{TOPK_LIST}": top_list,
    }


poles = {
    "plus_d_A_pole": pole_scan("+d (A pole)", 1.0),
    "minus_d_B_pole": pole_scan("-d (B pole)", -1.0),
}

# -------------------------------------------------------- per-position d ----
D_pos = (A_full - Bn) / 2
Dh = D_pos / D_pos.norm(dim=1, keepdim=True)
T = D_pos.shape[0]
pw = Dh @ Dh.T
pos_alignment = float((pw.sum() - T) / (T * (T - 1)))
per_position = {
    "note": ("pos_alignment = 1.0 means one global hinge shared by every "
             "position, i.e. the hinge is NOT last-position-specific; the "
             "last-position d stands for all positions"),
    "pos_alignment_recomputed": pos_alignment,
    "cos_dpos_dlast": [round(float(c), 6) for c in (Dh @ dhat)],
    "cos_dpos_u_geom_0p1": [round(float(c), 6)
                            for c in (Dh @ u_vectors["geom_0p1"])],
    "cos_dpos_u_curated_core": [round(float(c), 6)
                                for c in (Dh @ u_vectors["curated_core"])],
}
print(f"per-position: pos_alignment={pos_alignment:.4f}  "
      f"cos(d_pos, d_last) min={min(per_position['cos_dpos_dlast']):.4f}",
      flush=True)

# ------------------------------------------------- direction diagnostics ----
# Where do these centroid directions sit relative to the global-mean
# embedding direction and to each other? (Needed to interpret the signs.)
muhat = mu / mu.norm()
direction_relations = {
    "mu_norm": float(mu.norm()),
    "cos_d_muhat": float(dhat @ muhat),
    "cos_M_muhat": float(Mhat @ muhat),
    "cos_u_geom_0p1_muhat": float(u_vectors["geom_0p1"] @ muhat),
    "cos_u_lownorm_0p1_muhat": float(u_vectors["lownorm_0p1"] @ muhat),
    "cos_u_lownorm_0p5_muhat": float(u_vectors["lownorm_0p5"] @ muhat),
    "cos_u_curated_core_muhat": float(u_vectors["curated_core"] @ muhat),
    "cos_u_geom_0p1_u_curated_core": float(u_vectors["geom_0p1"]
                                           @ u_vectors["curated_core"]),
    "cos_u_geom_0p1_u_lownorm_0p1": float(u_vectors["geom_0p1"]
                                          @ u_vectors["lownorm_0p1"]),
    "cos_u_curated_core_u_curated_all": float(u_vectors["curated_core"]
                                              @ u_vectors["curated_all"]),
}
print("direction relations:",
      {k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in direction_relations.items()}, flush=True)

# -------------------------------------- raw-basis robustness (if available) ----
raw_check = {"available": False,
             "note": ("membership sets recomputed in the raw HF embedding "
                      "basis (before TransformerLens center_writing_weights); "
                      "requires ATR_GPT2_LOCAL")}
if LOCAL:
    RAW = hf_model.transformer.wte.weight.detach().float()
    if RAW.shape == W_E.shape and not torch.allclose(RAW, W_E):
        mu_r = RAW.mean(dim=0)
        dist_r = (RAW - mu_r).norm(dim=1)
        norms_r = RAW.norm(dim=1)
        g01r = set(torch.argsort(dist_r)[:k01].tolist())
        g05r = set(torch.argsort(dist_r)[:k05].tolist())
        n01r = set(torch.argsort(norms_r)[:k01].tolist())
        n05r = set(torch.argsort(norms_r)[:k05].tolist())

        def jac(a, b):
            return len(a & b) / len(a | b)

        raw_check = {
            "available": True,
            "note": raw_check["note"],
            "mean_abs_row_mean_processed": float(W_E.mean(dim=1).abs().mean()),
            "mean_abs_row_mean_raw": float(RAW.mean(dim=1).abs().mean()),
            "jaccard_geom_0p1": jac(CLUSTER_SETS["geom_0p1"], g01r),
            "jaccard_geom_0p5": jac(CLUSTER_SETS["geom_0p5"], g05r),
            "jaccard_lownorm_0p1": jac(CLUSTER_SETS["lownorm_0p1"], n01r),
            "jaccard_lownorm_0p5": jac(CLUSTER_SETS["lownorm_0p5"], n05r),
        }
        print(f"raw-basis check: jaccard geom_0p1={raw_check['jaccard_geom_0p1']:.3f} "
              f"geom_0p5={raw_check['jaccard_geom_0p5']:.3f}", flush=True)

# ---------------------------------------------------- curated token detail ----
dist_rank = torch.empty(V, dtype=torch.long)
dist_rank[dist_order] = torch.arange(V)
REQUIRED_FORMS = set(GROUP_REQUIRED) | {
    (s[1:] if s.startswith(" ") else " " + s) for s in GROUP_REQUIRED}
curated_detail = []
for (s, i), grp in ([(h, "family") for h in core_hits]
                    + [(h, "oddity") for h in odd_hits]):
    if grp == "family" and s in REQUIRED_FORMS:
        grp = "required"
    curated_detail.append({
        "string": esc(s),
        "token_id": i,
        "group": grp,
        "dist_to_centroid_percentile": round(float(dist_rank[i]) / V, 5),
        "norm_percentile": round(float(norm_rank[i]) / V, 5),
        "cos_with_plus_d": round(float(cos_all[i]), 4),
    })

# ----------------------------------------------------------------- output ----
results = {
    "meta": {
        "issue": 14,
        "thread": 4,
        "script": "07_glitch_alignment.py",
        "seed": SEED,
        "n_null": N_NULL,
        "n_norm_bins": N_BINS,
        "torch_version": torch.__version__,
        "space_note": ("d is a residual-stream direction; W_E rows write "
                       "directly into the same 768-dim residual coordinate "
                       "space, so cos(d, W_E row) is well-defined. All "
                       "geometry uses the TransformerLens processed basis "
                       "(fold_ln, center_writing_weights, center_unembed), "
                       "the same basis 04/06 used for chordness."),
        "pole_note": ("d = (A - B)/2 at the last position, as in "
                      "06_bell_anatomy.py: +d is the phase-A pole, "
                      "-d is the phase-B pole"),
        "cluster_definitions": {
            "geom_0p1": "closest 0.1 percent of vocab by L2 distance to the "
                        "mean-embedding centroid (k=50)",
            "geom_0p5": "closest 0.5 percent by the same distance (k=251)",
            "lownorm_0p1": "bottom 0.1 percent by W_E row norm (k=50)",
            "lownorm_0p5": "bottom 0.5 percent by W_E row norm (k=251)",
            "curated_core": "published SolidGoldMagikarp-family strings "
                            "matched as exact single tokens",
            "curated_all": "curated_core plus the ideographic/fullwidth "
                           "oddities from this repo's bell readout",
        },
    },
    "sanity_gate": {
        "cos_AB": cosAB, "expected_cos_AB": EXP_COSAB,
        "cos_A_ffA": cosAA2, "passed": gate_ok,
    },
    "hinge": {
        "d_norm_raw": float(d_raw.norm()),
        "A_last_norm": float(A.norm()),
        "B_last_norm": float(B.norm()),
        "M_norm": float(M.norm()),
        "cos_d_M": float(dhat @ Mhat),
    },
    "clusters": {
        "sizes": {n: len(ids) for n, ids in CLUSTERS.items()},
        "overlaps": overlaps,
        "geom_0p1_tokens": decode_ids(geom_01),
        "geom_0p5_tokens": decode_ids(geom_05),
        "lownorm_0p1_tokens": decode_ids(lownorm_01),
        "lownorm_0p5_token_ids": lownorm_05.tolist(),
        "curated_matched": curated_detail,
        "curated_unmatched": [esc(s) for s in core_misses + odd_misses],
        "curated_match_counts": {
            "family_candidate_strings": n_family_cand,
            "family_candidates_unmatched": len(core_misses),
            "family_distinct_tokens": len(core_hits),
            "oddity_candidate_strings": len(GROUP_ODDITIES),
            "oddity_distinct_tokens": len(odd_hits),
            "note": "each candidate probed with and without leading space; "
                    "all exact single-token variants kept, deduped by id",
        },
    },
    "alignment": alignment,
    "poles": poles,
    "direction_relations": direction_relations,
    "per_position": per_position,
    "raw_basis_check": raw_check,
}

with open(os.path.join(OUT, "glitch_alignment.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print("saved output_glitch/glitch_alignment.json", flush=True)
