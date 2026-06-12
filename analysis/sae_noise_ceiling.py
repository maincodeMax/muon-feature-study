# Instrument-noise ceiling: match two SAEs (different SAE seeds) trained on the SAME model.
import argparse, glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import GPT, GPTConfig, DistributedDataLoader

K, LAYER, D = 32, 6, 768

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--sae_a', required=True)
p.add_argument('--sae_b', required=True)
p.add_argument('--tokens', type=int, default=1_000_000)
a = p.parse_args()
device = 'cuda'

def resolve(tag):
    c = sorted(glob.glob(f'ckpts/{tag}/L3.35_*.pt')) or sorted(glob.glob(f'ckpts/{tag}/step005100.pt'))
    return c[0]

d = torch.load(resolve(a.run), map_location=device)
sd = {k.removeprefix('_orig_mod.'): v for k, v in d['model'].items()}
model = GPT(GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768))
model.load_state_dict(sd)
model = model.cuda().bfloat16().eval()
cap = []
model.transformer.h[LAYER].register_forward_hook(lambda m, i, o: cap.append(o.detach()))

saes = {}
for name, path in (('A', a.sae_a), ('B', a.sae_b)):
    x = torch.load(path, map_location=device)
    saes[name] = {k: x[k].cuda().float() for k in ('W_enc', 'b_enc', 'W_dec', 'b_dec')}

def encode(sae, x):
    z = (x - sae['b_dec']) @ sae['W_enc'] + sae['b_enc']
    topv, topi = z.topk(K, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, topi, torch.relu(topv))
    return out

F = saes['A']['W_enc'].shape[1]
n = 0
SA = torch.zeros(F, device=device); SB = torch.zeros(F, device=device)
SAA = torch.zeros(F, device=device); SBB = torch.zeros(F, device=device)
C = torch.zeros(F, F, device=device)
loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin', 8, 1024, 0, 1)
with torch.no_grad():
    while n < a.tokens:
        x, _ = loader.next_batch()
        model(x, return_logits=False)
        acts = cap.pop().reshape(-1, D).float()
        cap.clear()
        fa, fb = encode(saes['A'], acts), encode(saes['B'], acts)
        SA += fa.sum(0); SB += fb.sum(0)
        SAA += (fa * fa).sum(0); SBB += (fb * fb).sum(0)
        C += fa.t() @ fb
        n += acts.shape[0]
muA, muB = SA / n, SB / n
varA = (SAA / n - muA ** 2).clamp_min(1e-12)
varB = (SBB / n - muB ** 2).clamp_min(1e-12)
corr = (C / n - torch.outer(muA, muB)) / torch.outer(varA.sqrt(), varB.sqrt())
sub = corr[SA > 0][:, SB > 0]
best = sub.max(dim=1).values
res = dict(run=a.run, mean=round(best.mean().item(), 4),
           frac_gt07=round((best > 0.7).float().mean().item(), 4), tokens=n)
os.makedirs('analysis/figs/ceiling', exist_ok=True)
json.dump(res, open(f'analysis/figs/ceiling/{a.run}.json', 'w'))
print('CEILING', json.dumps(res))
