# Linear CKA between two models' layer-6 residual activations over a shared token stream.
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
p = argparse.ArgumentParser()
p.add_argument('--run_a', required=True)
p.add_argument('--run_b', required=True)
p.add_argument('--layer', type=int, default=6)
p.add_argument('--tokens', type=int, default=1_000_000)
p.add_argument('--component', choices=['block', 'attn', 'mlp'], default='block')
a = p.parse_args()
device = 'cuda'

def load_model(ck):
    d = torch.load(ck, map_location=device)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in d['model'].items()}
    m = GPT(GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768))
    m.load_state_dict(sd)
    return m.cuda().bfloat16().eval()

def resolve(tag):
    if tag in CKPTS:
        return CKPTS[tag]
    import glob
    c = sorted(glob.glob(f'ckpts/{tag}/L3.35_*.pt')) or sorted(glob.glob(f'ckpts/{tag}/step005100.pt'))
    return c[0]

pair = {}
for tag in (a.run_a, a.run_b):
    m = load_model(resolve(tag))
    cap = []
    target = m.transformer.h[a.layer] if a.component == 'block' else getattr(m.transformer.h[a.layer], a.component)
    target.register_forward_hook(lambda mod, i, o, c=cap: c.append(o.detach()))
    pair[tag] = (m, cap)

D = 768
n = 0
sA = torch.zeros(D, device=device); sB = torch.zeros(D, device=device)
AA = torch.zeros(D, D, device=device); BB = torch.zeros(D, D, device=device); AB = torch.zeros(D, D, device=device)
loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin', 8, 1024, 0, 1)
with torch.no_grad():
    while n < a.tokens:
        x, _ = loader.next_batch()
        acts = {}
        for tag in (a.run_a, a.run_b):
            m, cap = pair[tag]
            m(x, return_logits=False)
            acts[tag] = cap.pop().reshape(-1, D).float()
            cap.clear()
        A, B = acts[a.run_a], acts[a.run_b]
        sA += A.sum(0); sB += B.sum(0)
        AA += A.t() @ A; BB += B.t() @ B; AB += A.t() @ B
        n += A.shape[0]
muA, muB = sA / n, sB / n
cAB = AB - n * torch.outer(muA, muB)
cAA = AA - n * torch.outer(muA, muA)
cBB = BB - n * torch.outer(muB, muB)
cka = (cAB.norm() ** 2 / (cAA.norm() * cBB.norm())).item()
os.makedirs('analysis/figs/cka', exist_ok=True)
json.dump(dict(run_a=a.run_a, run_b=a.run_b, layer=a.layer, component=a.component, tokens=n, cka=cka),
          open(f'analysis/figs/cka/{a.run_a}__{a.run_b}_L{a.layer}_{a.component}.json', 'w'))
print('CKA', a.run_a, a.run_b, round(cka, 4))
