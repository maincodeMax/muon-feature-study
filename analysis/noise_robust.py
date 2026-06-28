# Residual-stream noise robustness: degrade x with Gaussian noise scaled to each model's
# stream RMS, measure FVU and per-feature firing stability of the (clean-trained) SAE.
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
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4]
K, LAYER, D = 32, 6, 768
TOKENS = 500_000

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
a = p.parse_args()
device = 'cuda'
torch.manual_seed(0)

d = torch.load(CKPTS[a.run], map_location=device)
sd = {k.removeprefix('_orig_mod.'): v for k, v in d['model'].items()}
model = GPT(GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768))
model.load_state_dict(sd)
model = model.cuda().bfloat16().eval()
cap = []
model.transformer.h[LAYER].register_forward_hook(lambda m, i, o: cap.append(o.detach()))

sae_path = f'analysis/saes/{a.run}__{os.path.basename(CKPTS[a.run])[:-3]}__L{LAYER}.pt'
sd_sae = torch.load(sae_path, map_location=device)
W_enc, b_enc = sd_sae['W_enc'].cuda().float(), sd_sae['b_enc'].cuda().float()
W_dec, b_dec = sd_sae['W_dec'].cuda().float(), sd_sae['b_dec'].cuda().float()

def encode(x):
    z = (x - b_dec) @ W_enc + b_enc
    topv, topi = z.topk(K, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, topi, torch.relu(topv))
    return out

loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin', 8, 1024, 0, 1)
acts = []
n = 0
with torch.no_grad():
    while n < TOKENS:
        x, _ = loader.next_batch()
        model(x, return_logits=False)
        acts.append(cap.pop().reshape(-1, D).float())
        cap.clear()
        n += acts[-1].shape[0]
X = torch.cat(acts)
rms = X.norm(dim=1).mean() / D ** 0.5

clean_z = encode(X)
results = {}
with torch.no_grad():
    for sig in SIGMAS:
        Xn = X + torch.randn_like(X) * sig * rms
        zn = encode(Xn)
        recon = b_dec + zn @ W_dec
        fvu = ((recon - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
        # per-feature firing stability: fraction of clean firings preserved under noise
        both = ((clean_z > 0) & (zn > 0)).sum()
        clean_fires = (clean_z > 0).sum()
        overlap = (both / clean_fires).item()
        # firing-value correlation over all (token, feature) cells that fire cleanly
        mask = clean_z > 0
        corr = torch.corrcoef(torch.stack([clean_z[mask], zn[mask]]))[0, 1].item()
        results[str(sig)] = dict(fvu=round(fvu.item(), 4), fire_overlap=round(overlap, 4),
                                 value_corr=round(corr, 4))
os.makedirs('analysis/figs/noise', exist_ok=True)
json.dump(dict(run=a.run, rms=rms.item(), results=results),
          open(f'analysis/figs/noise/{a.run}.json', 'w'), indent=1)
print(a.run, json.dumps(results))
