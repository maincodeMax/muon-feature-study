# Cross-seed SAE transfer: evaluate the SAE trained on model A against model B's activations.
# Measures whether interpretability artifacts transfer across retrains, per optimizer family.
import argparse, glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import GPT, GPTConfig, DistributedDataLoader

K, LAYER, D, TOKENS = 32, 6, 768, 500_000

p = argparse.ArgumentParser()
p.add_argument('--sae_run', required=True)   # SAE trained on this run
p.add_argument('--model_run', required=True) # evaluated on this run's activations
a = p.parse_args()
device = 'cuda'

def resolve(tag):
    c = sorted(glob.glob(f'ckpts/{tag}/L3.35_*.pt')) or sorted(glob.glob(f'ckpts/{tag}/step005100.pt'))
    return c[0]

ck = resolve(a.model_run)
d = torch.load(ck, map_location=device)
sd = {k.removeprefix('_orig_mod.'): v for k, v in d['model'].items()}
model = GPT(GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768))
model.load_state_dict(sd)
model = model.cuda().bfloat16().eval()
cap = []
model.transformer.h[LAYER].register_forward_hook(lambda m, i, o: cap.append(o.detach()))

sae_ck = resolve(a.sae_run)
sae_path = f'analysis/saes/{a.sae_run}__{os.path.basename(sae_ck)[:-3]}__L{LAYER}.pt'
sd_sae = torch.load(sae_path, map_location=device)
W_enc, b_enc = sd_sae['W_enc'].cuda().float(), sd_sae['b_enc'].cuda().float()
W_dec, b_dec = sd_sae['W_dec'].cuda().float(), sd_sae['b_dec'].cuda().float()

loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin', 8, 1024, 0, 1)
sse, svar, n = 0.0, 0.0, 0
mu_acc = torch.zeros(D, device=device)
chunks = []
with torch.no_grad():
    while n < TOKENS:
        x, _ = loader.next_batch()
        model(x, return_logits=False)
        acts = cap.pop().reshape(-1, D).float()
        cap.clear()
        chunks.append(acts)
        mu_acc += acts.sum(0)
        n += acts.shape[0]
    mu = mu_acc / n
    for acts in chunks:
        z = (acts - b_dec) @ W_enc + b_enc
        topv, topi = z.topk(K, dim=-1)
        zs = torch.zeros_like(z)
        zs.scatter_(-1, topi, torch.relu(topv))
        recon = b_dec + zs @ W_dec
        sse += ((recon - acts) ** 2).sum().item()
        svar += ((acts - mu) ** 2).sum().item()

fvu = sse / svar
os.makedirs('analysis/figs/transfer', exist_ok=True)
json.dump(dict(sae_run=a.sae_run, model_run=a.model_run, fvu=round(fvu, 4), tokens=n),
          open(f'analysis/figs/transfer/{a.sae_run}__on__{a.model_run}.json', 'w'))
print('TRANSFER', a.sae_run, '->', a.model_run, 'fvu', round(fvu, 4))
