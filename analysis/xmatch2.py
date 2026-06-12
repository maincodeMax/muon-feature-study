# Basis-free SAE feature matching: correlate feature activation patterns over a shared token stream.
# Usage: python analysis/xmatch2.py --run_a muon_s0 --run_b adamw_s0 [--tokens 1000000]
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import GPT, GPTConfig, DistributedDataLoader

CKPTS = {
    'muon_s0': 'ckpts/muon_s0/L3.35_step004375.pt',
    'muon_s1': 'ckpts/muon_s1/L3.35_step004375.pt',
    'muon_s2': 'ckpts/muon_s2/L3.35_step004375.pt',
    'adamw_s0': 'ckpts/adamw_s0/L3.35_step005100.pt',
    'adamw_s1': 'ckpts/adamw_s1/step005100.pt',
    'adamw_s2': 'ckpts/adamw_s2/L3.35_step005100.pt',
}
K = 32

p = argparse.ArgumentParser()
p.add_argument('--run_a', required=True)
p.add_argument('--run_b', required=True)
p.add_argument('--layer', type=int, default=6)
p.add_argument('--tokens', type=int, default=1_000_000)
p.add_argument('--n_layer', type=int, default=12)
p.add_argument('--n_head', type=int, default=6)
p.add_argument('--n_embd', type=int, default=768)
p.add_argument('--out', default='analysis/saes/xmatch2')
p.add_argument('--ckpt_a', default=None)
p.add_argument('--ckpt_b', default=None)
p.add_argument('--name_suffix', default='')
p.add_argument('--gelu', action='store_true')
a = p.parse_args()
device = 'cuda'
LAYER = a.layer

def resolve_ckpt(tag):
    if tag in CKPTS:
        return CKPTS[tag]
    import glob
    c = sorted(glob.glob(f'ckpts/{tag}/L3.35_*.pt')) or sorted(glob.glob(f'ckpts/{tag}/step005100.pt'))
    return c[0]

overrides = {}
if a.ckpt_a: overrides[a.run_a] = a.ckpt_a
if a.ckpt_b: overrides[a.run_b] = a.ckpt_b
def ck_for(tag):
    return overrides.get(tag) or resolve_ckpt(tag)
SAES = {tag: (ck_for(tag), f'analysis/saes/{tag}__{os.path.basename(ck_for(tag))[:-3]}__L{a.layer}.pt')
        for tag in set([a.run_a, a.run_b])}

def load_model(ckpt_path):
    d = torch.load(ckpt_path, map_location=device)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in d['model'].items()}
    m = GPT(GPTConfig(vocab_size=50304, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd))
    m.load_state_dict(sd)
    if a.gelu:
        for blk in m.transformer.h:
            blk.mlp.use_gelu = True
    return m.cuda().bfloat16().eval()

def load_sae(sae_path):
    d = torch.load(sae_path, map_location=device)
    return {k: d[k].cuda().float() for k in ('W_enc', 'b_enc', 'W_dec', 'b_dec')}

def encode(sae, x):
    z = (x - sae['b_dec']) @ sae['W_enc'] + sae['b_enc']
    topv, topi = z.topk(K, dim=-1)
    topv = torch.relu(topv)
    out = torch.zeros_like(z)
    out.scatter_(-1, topi, topv)
    return out

pair = {}
for tag in (a.run_a, a.run_b):
    ckpt, sae_path = SAES[tag]
    model = load_model(ckpt)
    cap = []
    model.transformer.h[LAYER].register_forward_hook(lambda m, i, o, c=cap: c.append(o.detach()))
    pair[tag] = (model, load_sae(sae_path), cap)

F_DIM = pair[a.run_a][1]['W_enc'].shape[1]
B, T = 8, 1024
loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin', B, T, 0, 1)

n = 0
SA = torch.zeros(F_DIM, device=device); SB = torch.zeros(F_DIM, device=device)
SAA = torch.zeros(F_DIM, device=device); SBB = torch.zeros(F_DIM, device=device)
C = torch.zeros(F_DIM, F_DIM, device=device)
with torch.no_grad():
    while n < a.tokens:
        x, _ = loader.next_batch()
        feats = {}
        for tag in (a.run_a, a.run_b):
            model, sae, cap = pair[tag]
            model(x, return_logits=False)
            acts = cap.pop().reshape(-1, a.n_embd).float()
            cap.clear()
            feats[tag] = encode(sae, acts)
        fa, fb = feats[a.run_a], feats[a.run_b]
        SA += fa.sum(0); SB += fb.sum(0)
        SAA += (fa * fa).sum(0); SBB += (fb * fb).sum(0)
        C += fa.t() @ fb
        n += fa.shape[0]

muA, muB = SA / n, SB / n
varA = (SAA / n - muA ** 2).clamp_min(1e-12)
varB = (SBB / n - muB ** 2).clamp_min(1e-12)
cov = C / n - torch.outer(muA, muB)
corr = cov / torch.outer(varA.sqrt(), varB.sqrt())

aliveA = SA > 0
aliveB = SB > 0

sub = corr[aliveA][:, aliveB]
best_ab = sub.max(dim=1).values
best_ba = sub.max(dim=0).values

def stats(t):
    return dict(mean=t.mean().item(), p50=t.median().item(),
                frac_gt03=(t > 0.3).float().mean().item(),
                frac_gt05=(t > 0.5).float().mean().item(),
                frac_gt07=(t > 0.7).float().mean().item())

os.makedirs(a.out, exist_ok=True)
res = dict(run_a=a.run_a, run_b=a.run_b, layer=a.layer, tokens=n,
           n_alive_a=int(aliveA.sum()), n_alive_b=int(aliveB.sum()),
           a_to_b=stats(best_ab), b_to_a=stats(best_ba))
json.dump(res, open(f'{a.out}/{a.run_a}__{a.run_b}_L{a.layer}{a.name_suffix}.json', 'w'), indent=1)
torch.save(dict(best_ab=best_ab.cpu(), best_ba=best_ba.cpu()),
           f'{a.out}/{a.run_a}__{a.run_b}_L{a.layer}{a.name_suffix}_best.pt')
print(json.dumps(res, indent=1))
