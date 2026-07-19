import os, sys, json, math
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
SCRATCH=os.path.dirname(os.path.abspath(__file__)); LOCAL=os.path.join(SCRATCH,"gpt2local")
REPO="/home/user/lucier-gpt2-activ-tensor-reson-experiments"
sys.path.insert(0, REPO)
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
hf=GPT2LMHeadModel.from_pretrained(LOCAL); tok=GPT2TokenizerFast.from_pretrained(LOCAL)
import transformer_lens.loading_from_pretrained as lfp
cfg=GPT2Config.from_pretrained(LOCAL)
class S:
    @staticmethod
    def from_pretrained(n,*a,**k): return cfg
lfp.AutoConfig=S
from transformer_lens import HookedTransformer
m=HookedTransformer.from_pretrained("gpt2",hf_model=hf,tokenizer=tok,device="cpu"); m.eval()

st=torch.load(os.path.join(REPO,"experiments/gpt2_small/output_divine_motion/state_divine.pt"), weights_only=True)
A_full=st["current_tensor"]; initial_norm=st["initial_norm"]
prompt="The cat sat on the mat and then the"
hook_read=f"blocks.{m.cfg.n_layers-1}.hook_resid_post"; hook_write="blocks.0.hook_resid_pre"

def step(x):
    cur=x*(initial_norm/x.norm())
    inject=cur.clone()
    def h(resid,hook,tensor=inject):
        resid[0,:,:]=tensor; return resid
    m.add_hook(hook_write,h)
    try:
        with torch.no_grad():
            _,cache=m.run_with_cache(prompt,names_filter=lambda n:n==hook_read)
    finally:
        m.reset_hooks()
    return cache[hook_read][0].clone()

B_full=step(A_full)
A2_full=step(B_full)  # should return ~A (verify period-2)
# rescale for comparison (loop normalises before injecting; compare normalized states)
def norm_to(x, n): return x*(n/x.norm())
Bn=norm_to(B_full, initial_norm); A2n=norm_to(A2_full, initial_norm)
A=A_full[-1]; B=Bn[-1]; A2=A2n[-1]

def readout(v,k=10):
    with torch.no_grad():
        logits=m.ln_final(v)@m.W_U+m.b_U
        p=torch.softmax(logits,dim=-1)
        tp,ti=torch.topk(p,k)
        H=float(-(p*torch.log(p.clamp_min(1e-12))).sum())
        return logits,[(m.tokenizer.decode([int(i)]),float(x)) for i,x in zip(ti,tp)],H,[int(i) for i in ti]

def chord(ids):
    E=m.W_E[ids]; E=E/E.norm(dim=-1,keepdim=True); s=E@E.T; n=len(ids)
    return float((s.sum()-n)/(n*(n-1)))

logA,topA,HA,idsA=readout(A); logB,topB,HB,idsB=readout(B)
M=(A+B)/2; d=(A-B)/2
logM,topM,HM,idsM=readout(M)
cosAB=float(torch.nn.functional.cosine_similarity(A.unsqueeze(0),B.unsqueeze(0)))
cosAA2=float(torch.nn.functional.cosine_similarity(A.unsqueeze(0),A2.unsqueeze(0)))
print(f"period-2 verify: cos(A, f(f(A)))={cosAA2:.6f}  cos(A,B)={cosAB:.4f}")
print(f"norms: |A|={A.norm():.0f} |B|={B.norm():.0f} |M|={M.norm():.0f} |d|={d.norm():.0f}")
print(f"\nPHASE A (H={HA:.2f}): {topA}")
print(f"\nPHASE B (H={HB:.2f}): {topB}")
print(f"\nPIVOT M (H={HM:.2f}): {topM}")
print(f"\nchordness: A_top10={chord(idsA):.3f}  B_top10={chord(idsB):.3f}  M_top10={chord(idsM):.3f}")

# see-saw riders: tokens whose logits swing most between phases
dl=logA-logB
up,ui=torch.topk(dl,10); dn,di=torch.topk(-dl,10)
print("\nRIDERS toward A:", [(m.tokenizer.decode([int(i)]),f"+{float(x):.1f}") for i,x in zip(ui,up)])
print("RIDERS toward B:", [(m.tokenizer.decode([int(i)]),f"+{float(x):.1f}") for i,x in zip(di,dn)])

# invisibility of the axis d vs random, and of M
def logit_response(v):
    with torch.no_grad():
        base=m.ln_final(M)@m.W_U
        pert=m.ln_final(M+v)@m.W_U
        return float((pert-base).norm())
torch.manual_seed(0)
rd=[logit_response(torch.randn(768)*(d.norm()/math.sqrt(768))* (1.0)) for _ in range(20)]
rd=[x for x in rd]
resp_d=logit_response(d); resp_r=sum(rd)/len(rd)
print(f"\naxis invisibility: logit response of d = {resp_d:.0f}, equal-ish random mean = {resp_r:.0f}, ratio={resp_d/resp_r:.3f}")

# is the see-saw one global axis? per-position d alignment
D=(A_full - Bn)/2  # [T,768]
Dn_=D/D.norm(dim=-1,keepdim=True)
pw=Dn_@Dn_.T; T=D.shape[0]
off=float((pw.sum()-T)/(T*(T-1)))
print(f"per-position flip-axis alignment (mean pairwise cos over {T} positions): {off:.4f}")

# loud/quiet decomposition of d via W_U singular directions
with torch.no_grad():
    U,Sv,Vh=torch.linalg.svd(m.W_U, full_matrices=False)  # W_U [768,50257]; U [768,768]
    coords=U.T@ (d/d.norm())
    top100=float((coords[:100]**2).sum()); bot100=float((coords[-100:]**2).sum())
    coordsM=U.T@(M/M.norm())
    top100M=float((coordsM[:100]**2).sum()); bot100M=float((coordsM[-100:]**2).sum())
print(f"axis d energy in top-100 vs bottom-100 W_U singular dirs: {top100:.3f} / {bot100:.3f}")
print(f"pivot M energy in top-100 vs bottom-100:                  {top100M:.3f} / {bot100M:.3f}")
json.dump({"cosAB":cosAB,"cosAA2":cosAA2,"HA":HA,"HB":HB,"HM":HM,
           "topA":topA,"topB":topB,"topM":topM,
           "chordA":chord(idsA),"chordB":chord(idsB),"chordM":chord(idsM),
           "axis_ratio":resp_d/resp_r,"pos_alignment":off,
           "d_top100":top100,"d_bot100":bot100,"M_top100":top100M,"M_bot100":bot100M},
          open("bell_anatomy.json","w"), indent=1)
print("\nsaved bell_anatomy.json")
