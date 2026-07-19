"""
EXP: J-lens pilot (restricted Jacobian lens) + J-space membership probe (issue #8, PILOT)

Builds a restricted, pilot-scale version of the Jacobian lens from
docs/JSPACE_PRIMER.md Part 3 for GPT-2 Small, then asks where the saved
converged ATR attractor states sit relative to the span of the pilot
J-lens vectors, per layer.

Construction (pilot restrictions relative to the paper):
  - Corpus: ~30 hand-written prompts across registers (paper: 1000
    pretraining-sampled prompts). This is the pilot's main limitation.
  - Token set: union of (a) top-20 readout tokens of every converged
    state, (b) ~100 common English word tokens, (c) ~50 random vocab
    control tokens. (Paper: full vocabulary.)
  - J-lens vector v_{t,l}: for scalar s_t = sum over positions t' of the
    layer-final logit of token t, one backward pass gives ds_t/dh_{l,pos}
    for all layers and positions at once (causal masking restricts the
    sum to t' >= pos automatically). Averaged over positions, then over
    prompts. One vector per (token, layer).

Membership probe: for each converged state's last-position vector (5
prompt states + 3 fresh converged-noise states), per layer, project onto
the span of the layer's J-lens dictionary (least squares) and onto a
nonnegative sparse k=25 combination (projected gradient), recording
variance share ||proj||^2 / ||state||^2. Control: equal-sized random
dictionary with matched per-vector norms.

Outputs: output_jlens_pilot/{jlens_pilot_report.md, jlens_pilot_results.json,
jlens_vectors.pt}

Run:  ATR_GPT2_LOCAL=<dir> python 05_jlens_pilot.py [--probe]
      (--probe: time 10 backward passes and exit)
"""
import os, sys, json, math, time, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "output_jlens_pilot")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

import torch
# This container reports 4 CPUs but is cgroup-throttled: multi-threaded BLAS
# is ~5x SLOWER than single-threaded here (measured). Pin to one thread.
torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--probe", action="store_true", help="timing probe only")
parser.add_argument("--n-prompts", type=int, default=None)
parser.add_argument("--max-tokens", type=int, default=None)
args = parser.parse_args()

# ---- Model load (offline shim, copied from 04_readout_confidence.py) ----
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
for p in model.parameters():
    p.requires_grad_(False)  # backward must compute activation grads only

N_LAYERS = model.cfg.n_layers
D_MODEL = model.cfg.d_model
V = model.cfg.d_vocab
RESID_HOOKS = [f"blocks.{l}.hook_resid_post" for l in range(N_LAYERS)]

# ---- 1. Pilot corpus: ~30 hand-written prompts across registers ----
CORPUS = [
    # factual
    "The capital of France is Paris, a city on the Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The moon orbits the Earth roughly once every month.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    # narrative
    "She opened the door slowly and stepped into the dark hallway.",
    "The old fisherman pulled his boat onto the cold grey sand.",
    "Once upon a time there lived a king with three daughters.",
    "He ran through the rain, clutching the letter to his chest.",
    # question
    "What is the fastest way to travel between two cities?",
    "Why does the sky appear blue during the day?",
    "How many people live in the largest city on Earth?",
    "Where did you leave the keys to the front door?",
    # instruction
    "Preheat the oven to 350 degrees and grease the baking pan.",
    "Please review the attached report and send me your comments.",
    "Turn left at the second light, then continue straight ahead.",
    "Mix the flour and sugar together before adding the eggs.",
    # casual chat
    "Hey, are you coming to the party on Saturday night?",
    "I honestly can't believe how good that movie was.",
    "Sorry I'm late, the traffic on the bridge was terrible.",
    "Yeah, let's just grab coffee and talk about it tomorrow.",
    # code-like
    "def add(a, b): return a + b",
    "for i in range(10): print(i * 2)",
    "import os; path = os.path.join(base, name)",
    "if x is None: raise ValueError('missing input')",
    # list-like
    "Shopping list: eggs, milk, bread, butter, apples, coffee.",
    "Steps: open the file, read the data, close the file.",
    "Top three colors: blue, green, and deep red.",
    "Ingredients: two cups flour, one egg, half cup sugar.",
    # nonsense
    "Flurb glex morp wintly skade brimf ozzle quent.",
    "The borogoves did gyre and gimble in the wabe.",
]

# ---- 2. Restricted token set ----
def build_token_set(seed=1234):
    groups = {}
    # (a) top-20 readout tokens of every converged state (recomputed from
    # the saved tensors; the archived JSON predates the id-first format)
    conv = torch.load(os.path.join(HERE, "output_confidence", "converged_tensors.pt"),
                  weights_only=True)
    readout_ids = []
    with torch.no_grad():
        for label, tens in conv.items():
            vec = tens[-1, :]
            logits = model.ln_final(vec) @ model.W_U + model.b_U
            top = torch.topk(logits, 20).indices.tolist()
            readout_ids.extend(int(i) for i in top)
    groups["readout"] = sorted(set(readout_ids))
    # (b) ~100 common English word tokens (leading-space single tokens)
    common_words = [
        "the", "of", "and", "to", "in", "is", "was", "for", "on", "with",
        "as", "his", "her", "that", "this", "it", "he", "she", "they", "we",
        "you", "not", "have", "had", "but", "from", "are", "were", "be", "by",
        "time", "people", "world", "life", "man", "woman", "child", "day",
        "year", "water", "house", "city", "country", "school", "work", "money",
        "book", "word", "name", "place", "thing", "hand", "eye", "head",
        "night", "light", "door", "road", "tree", "fire", "food", "friend",
        "story", "music", "game", "war", "love", "death", "good", "bad",
        "new", "old", "great", "small", "long", "high", "right", "left",
        "first", "last", "next", "other", "same", "different", "big", "little",
        "go", "come", "see", "know", "think", "say", "make", "take",
        "get", "give", "find", "tell", "ask", "run", "walk", "look",
    ]
    common_ids = []
    for w in common_words:
        ids = model.tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            common_ids.append(ids[0])
    groups["common"] = sorted(set(common_ids))
    # (c) ~50 random vocabulary tokens as controls
    rng = random.Random(seed)
    groups["random_control"] = sorted(rng.sample(range(V), 50))
    token_ids = sorted(set(groups["readout"]) | set(groups["common"])
                       | set(groups["random_control"]))
    return token_ids, groups

TOKEN_IDS, TOKEN_GROUPS = build_token_set()
if args.max_tokens and len(TOKEN_IDS) > args.max_tokens:
    keep = set(TOKEN_GROUPS["readout"])
    rest = [t for t in TOKEN_IDS if t not in keep]
    random.Random(7).shuffle(rest)
    n_rest = max(0, args.max_tokens - len(keep))
    TOKEN_IDS = sorted(keep | set(rest[:n_rest]))
print(f"token set: {len(TOKEN_IDS)} tokens "
      f"(readout {len(TOKEN_GROUPS['readout'])}, common {len(TOKEN_GROUPS['common'])}, "
      f"random {len(TOKEN_GROUPS['random_control'])}, before union)", flush=True)

# ---- 3. J-lens vectors via VJPs ----
# Efficiency route: parameters are frozen, so backward computes activation
# gradients only. Tokens are batched by replicating the prompt across the
# batch dimension: batch element b backpropagates scalar
# s_{t_b} = sum_{t'} logit_{t_b}(t'), so one backward pass over a batch of
# B replicas yields ds_t/dh_{l,pos} for B different tokens at every layer
# and position at once (batch elements are independent in the forward).
TOKEN_BATCH = 25

def prompt_gradients(prompt, token_ids, token_batch=TOKEN_BATCH):
    """VJP gradients for one prompt over the restricted token set.

    Returns per-token, per-layer position-averaged gradients
    [n_tokens, n_layers, d_model] and per-token per-layer mean raw
    per-position gradient norms [n_tokens, n_layers].
    """
    tokens = model.to_tokens(prompt)  # [1, seq]
    n_tok = len(token_ids)
    grads = torch.zeros(n_tok, N_LAYERS, D_MODEL)
    rawnorm = torch.zeros(n_tok, N_LAYERS)
    for start in range(0, n_tok, token_batch):
        chunk = token_ids[start:start + token_batch]
        B = len(chunk)
        stash = {}
        def root(t, hook):
            # graph root: params are frozen and inputs are token ids, so
            # autograd needs a leaf requiring grad at the earliest resid
            return t.detach().requires_grad_(True)
        def keep(t, hook):
            t.retain_grad()
            stash[hook.name] = t
            return t
        model.add_hook("blocks.0.hook_resid_pre", root)
        for name in RESID_HOOKS:
            model.add_hook(name, keep)
        model.add_hook("ln_final.hook_normalized", keep)
        try:
            # grads needed: no no_grad. return_type=None skips the full
            # [B, seq, 50257] unembed matmul; each batch element only needs
            # one W_U column, taken below.
            model(tokens.expand(B, -1), return_type=None)
        finally:
            model.reset_hooks()
        idx = torch.tensor(chunk)
        normed = stash["ln_final.hook_normalized"]     # [B, seq, d_model]
        W = model.W_U[:, idx]                          # [d_model, B]
        # s = sum_b sum_{t'} logit_{t_b}(b, t'), b_U constant dropped
        s = (normed.sum(dim=1) * W.T).sum()
        s.backward()
        for l, name in enumerate(RESID_HOOKS):
            g = stash[name].grad                       # [B, seq, d_model]
            grads[start:start + B, l] = g.mean(dim=1)  # average over positions
            rawnorm[start:start + B, l] = g.norm(dim=-1).mean(dim=1)
        del stash
    return grads, rawnorm

if args.probe:
    t0 = time.time()
    g, _ = prompt_gradients(CORPUS[0], TOKEN_IDS[:2 * TOKEN_BATCH])
    dt = time.time() - t0
    per_tok = dt / (2 * TOKEN_BATCH)
    total = per_tok * len(TOKEN_IDS) * len(CORPUS) / 60
    print(f"probe: {dt:.1f}s for {2*TOKEN_BATCH} tokens -> ~{per_tok:.3f}s/token, "
          f"projected total ~{total:.0f} min for {len(TOKEN_IDS)} tokens x {len(CORPUS)} prompts")
    sys.exit(0)

if args.n_prompts:
    CORPUS = CORPUS[:args.n_prompts]

n_tok = len(TOKEN_IDS)
acc = torch.zeros(n_tok, N_LAYERS, D_MODEL)   # running sum of per-prompt means
raw_norms_log = []                            # per prompt: per-layer mean raw norm
snapshot_half = None
t_start = time.time()
for pi, prompt in enumerate(CORPUS):
    g, rawnorm = prompt_gradients(prompt, TOKEN_IDS)
    acc += g
    raw_norms_log.append(rawnorm.mean(dim=0).tolist())  # [n_layers]
    if pi + 1 == max(1, len(CORPUS) // 2):
        snapshot_half = acc.clone() / (pi + 1)
    print(f"prompt {pi+1}/{len(CORPUS)} done "
          f"({(time.time()-t_start)/60:.1f} min elapsed)", flush=True)
jlens = acc / len(CORPUS)                     # [n_tokens, n_layers, d_model]
compute_minutes = (time.time() - t_start) / 60

# Stability of the averaged Jacobian: cosine(running mean at N/2, at N) per layer
stability = {}
for l in range(N_LAYERS):
    a = snapshot_half[:, l, :]
    b = jlens[:, l, :]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    stability[f"layer_{l}"] = {
        "mean_cos": float(cos.mean()), "min_cos": float(cos.min()),
        "median_cos": float(cos.median()),
    }
half_n = len(CORPUS) // 2

torch.save({
    "token_ids": TOKEN_IDS,
    "token_strings": [model.tokenizer.decode([t]) for t in TOKEN_IDS],
    "token_groups": TOKEN_GROUPS,
    "layers": list(range(N_LAYERS)),
    "hooks": RESID_HOOKS,
    "vectors": jlens,             # [n_tokens, n_layers, d_model]
    "n_prompts": len(CORPUS),
    "corpus": CORPUS,
}, os.path.join(OUT, "jlens_vectors.pt"))
print("J-lens vectors saved.", flush=True)

# ---- 4. States to probe ----
conv = torch.load(os.path.join(HERE, "output_confidence", "converged_tensors.pt"),
                  weights_only=True)
states = {label: tens[-1, :].clone() for label, tens in conv.items()}

# 3 fresh converged-noise states (noise loop copied from 04_readout_confidence.py)
NOISE_NORM, NOISE_SEQ, NOISE_ITERS = 397.0, 10, 100
L0, L1 = 0, N_LAYERS - 1
hook_read = f"blocks.{L1}.hook_resid_post"
hook_write = f"blocks.{L0}.hook_resid_pre"
scaffold_tokens = torch.full((1, NOISE_SEQ), 262)
torch.manual_seed(2026)
for trial in range(3):
    x = torch.randn(NOISE_SEQ, D_MODEL)
    x = x * (NOISE_NORM / x.norm())
    initial_norm = x.norm().item()
    current = x.clone()
    for i in range(NOISE_ITERS):
        cn = current.norm().item()
        if cn > 0:
            current = current * (initial_norm / cn)
        inject = current.clone()
        def hookfn(resid, hook, tensor=inject):
            resid[0, :, :] = tensor
            return resid
        model.add_hook(hook_write, hookfn)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    scaffold_tokens, names_filter=lambda n: n == hook_read)
        finally:
            model.reset_hooks()
        current = cache[hook_read][0].clone()
    states[f"Noise_{trial}"] = current[-1, :].clone()
print("noise states converged.", flush=True)

FAMILIES = {
    "prolet": ["Lucier", "Semantic", "Nonsense", "Imperative"],
    "divine": ["Syntactic"],
    "noise": ["Noise_0", "Noise_1", "Noise_2"],
}

# ---- 5. Membership probe ----
def lstsq_share(D, h):
    """Variance share of h captured by span of dictionary rows D [n, d]."""
    sol = torch.linalg.lstsq(D.T, h.unsqueeze(1))
    proj = D.T @ sol.solution
    return float(proj.squeeze(1).norm() ** 2 / h.norm() ** 2)

def nn_sparse_share(D, h, k=25, steps=300):
    """Nonnegative sparse approximation, at most k atoms, projected gradient.

    min_c ||D^T c - h||^2 s.t. c >= 0, ||c||_0 <= k. Simple hard-threshold
    projected gradient; pilot quality only.
    """
    Dt = D.T  # [d, n]
    G = Dt.T @ Dt
    lip = float(torch.linalg.eigvalsh(G).max())
    step = 1.0 / max(lip, 1e-8)
    c = torch.zeros(D.shape[0])
    b = Dt.T @ h
    for _ in range(steps):
        grad = G @ c - b
        c = c - step * grad
        c = c.clamp_min(0.0)
        if int((c > 0).sum()) > k:
            thresh = torch.topk(c, k).values.min()
            c[c < thresh] = 0.0
    approx = Dt @ c
    return float(approx.norm() ** 2 / h.norm() ** 2)

torch.manual_seed(99)
results = {"per_state": {}, "families": FAMILIES,
           "token_set": {k: len(v) for k, v in TOKEN_GROUPS.items()},
           "n_tokens_union": n_tok, "n_prompts": len(CORPUS),
           "compute_minutes": compute_minutes,
           "stability_half_vs_full": stability,
           "stability_half_n": half_n,
           "raw_grad_norms_per_prompt_layer_mean": raw_norms_log,
           "corpus": CORPUS}

# random control dictionaries: same size, matched per-vector norms, 3 seeds
def random_dict_like(D, gen):
    R = torch.randn(D.shape, generator=gen)
    R = R / R.norm(dim=-1, keepdim=True) * D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return R

gen = torch.Generator().manual_seed(4242)
for label, h in states.items():
    per_layer = []
    for l in range(N_LAYERS):
        D = jlens[:, l, :]
        entry = {
            "layer": l,
            "lstsq_share": lstsq_share(D, h),
            "nn_sparse_k25_share": nn_sparse_share(D, h),
        }
        rand_shares, rand_nn = [], []
        for _ in range(3):
            R = random_dict_like(D, gen)
            rand_shares.append(lstsq_share(R, h))
            rand_nn.append(nn_sparse_share(R, h))
        entry["random_lstsq_share_mean"] = sum(rand_shares) / len(rand_shares)
        entry["random_nn_sparse_k25_share"] = sum(rand_nn) / len(rand_nn)
        per_layer.append(entry)
    results["per_state"][label] = {"per_layer": per_layer,
                                   "state_norm": float(h.norm())}
    print(f"{label}: L6 lstsq={per_layer[6]['lstsq_share']:.3f} "
          f"rand={per_layer[6]['random_lstsq_share_mean']:.3f}", flush=True)

with open(os.path.join(OUT, "jlens_pilot_results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print("DONE. Results in", OUT, flush=True)
