# Loss landscape over the plane through three trained models.
# Plane: origin = model A, u = B - A, v = (C - A) orthogonalized against u.
# Usage: python analysis/loss_surface.py --rows 0:25 --out analysis/figs/surface_rows_0.json
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import GPT, GPTConfig, DistributedDataLoader

MODELS = ['ckpts/muon_s0/step005100.pt', 'ckpts/adamw_s0/step005100.pt', 'ckpts/muon_s1/step005100.pt']
GRID = 25
ALPHA = (-0.35, 1.35)   # along u (muon_s0 -> adamw_s0)
BETA = (-0.35, 1.35)    # along v_orth (toward muon_s1)
EVAL_TOKENS = 524288
LOSS_CAP = 9.0

p = argparse.ArgumentParser()
p.add_argument('--rows', required=True)  # e.g. 0:13
p.add_argument('--out', required=True)
a = p.parse_args()
r0, r1 = map(int, a.rows.split(':'))

device = 'cuda'

def load_flat(path):
    d = torch.load(path, map_location='cpu')
    sd = {k.removeprefix('_orig_mod.'): v.float() for k, v in d['model'].items()}
    return sd

sds = [load_flat(m) for m in MODELS]
keys = list(sds[0].keys())
theta0 = sds[0]
u = {k: sds[1][k] - theta0[k] for k in keys}
w = {k: sds[2][k] - theta0[k] for k in keys}

dot_uw = sum((u[k] * w[k]).sum().item() for k in keys)
dot_uu = sum((u[k] * u[k]).sum().item() for k in keys)
proj = dot_uw / dot_uu
v = {k: w[k] - proj * u[k] for k in keys}
dot_vv = sum((v[k] * v[k]).sum().item() for k in keys)
# model C sits at alpha=proj, beta=1 in (u, v) coords by construction
positions = dict(A=[0.0, 0.0], B=[1.0, 0.0], C=[proj, 1.0],
                 norm_u=dot_uu ** 0.5, norm_v=dot_vv ** 0.5)

model = GPT(GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768)).cuda().bfloat16().eval()

B, T = 8, 1024
n_batches = EVAL_TOKENS // (B * T)

def eval_loss(alpha, beta):
    sd = {k: (theta0[k] + alpha * u[k] + beta * v[k]).bfloat16() for k in keys}
    model.load_state_dict(sd)
    loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin', B, T, 0, 1)
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = loader.next_batch()
            _, loss = model(x, y, return_logits=False)
            total += loss.item()
    return min(total / n_batches, LOSS_CAP)

alphas = [ALPHA[0] + (ALPHA[1] - ALPHA[0]) * i / (GRID - 1) for i in range(GRID)]
betas = [BETA[0] + (BETA[1] - BETA[0]) * j / (GRID - 1) for j in range(GRID)]

rows = []
for i in range(r0, min(r1, GRID)):
    row = []
    for j in range(GRID):
        row.append(round(eval_loss(alphas[i], betas[j]), 4))
    rows.append(dict(i=i, losses=row))
    print(f'row {i} done', flush=True)

json.dump(dict(rows=rows, alphas=alphas, betas=betas, positions=positions,
               grid=GRID, eval_tokens=EVAL_TOKENS, models=MODELS), open(a.out, 'w'))
print('SURFACE_ROWS_DONE', a.rows)
