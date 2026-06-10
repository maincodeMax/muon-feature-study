# Train a top-k SAE on the residual stream of a bench checkpoint, streaming activations.
# Usage: python analysis/sae_v1.py --ckpt ckpts/muon_s0/L3.35_step004375.pt [--layer 6] [--tokens 50000000]
import argparse, json, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import GPT, GPTConfig, DistributedDataLoader

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--layer', type=int, default=6)
p.add_argument('--expansion', type=int, default=16)
p.add_argument('--k', type=int, default=32)
p.add_argument('--tokens', type=int, default=50_000_000)
p.add_argument('--val_tokens', type=int, default=2_000_000)
p.add_argument('--sae_batch', type=int, default=4096)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--out', default='analysis/saes')
p.add_argument('--normalize', action='store_true')  # rescale acts to mean norm sqrt(d_in) before SAE
p.add_argument('--tag', default='')
p.add_argument('--n_layer', type=int, default=12)
p.add_argument('--n_head', type=int, default=6)
p.add_argument('--n_embd', type=int, default=768)
a = p.parse_args()

device = 'cuda'
torch.manual_seed(0)  # identical SAE init + data order for every model; only the model differs

# ---- load bench checkpoint ----
d = torch.load(a.ckpt, map_location=device)
sd = {k.removeprefix('_orig_mod.'): v for k, v in d['model'].items()}
model = GPT(GPTConfig(vocab_size=50304, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd))
model.load_state_dict(sd)
model = model.cuda().bfloat16().eval()
run_tag = os.path.basename(os.path.dirname(a.ckpt))
ckpt_tag = os.path.basename(a.ckpt)[:-3]

captured = []
model.transformer.h[a.layer].register_forward_hook(lambda m, i, o: captured.append(o.detach()))

# ---- SAE ----
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / d_in ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(self.W_enc.t().clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.renorm()

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        topv, topi = z.topk(self.k, dim=-1)
        topv = F.relu(topv)
        recon = self.b_dec + (topv.unsqueeze(-1) * self.W_dec[topi]).sum(-2)
        return recon, topv, topi

d_in = a.n_embd
sae = TopKSAE(d_in, a.expansion * d_in, a.k).cuda()
opt = torch.optim.Adam(sae.parameters(), lr=a.lr)

B, T = 32, 1024
loader = DistributedDataLoader('data/fineweb10B/fineweb_train_*.bin', B, T, 0, 1)
val_loader = DistributedDataLoader('data/fineweb10B/fineweb_val_*.bin', B, T, 0, 1)

act_scale = None  # set from first harvest; applied only with --normalize
raw_norm_mean = None

def harvest(ldr):
    global act_scale, raw_norm_mean
    x, _ = ldr.next_batch()
    with torch.no_grad():
        model(x, return_logits=False)
    acts = captured.pop().reshape(-1, d_in).float()
    captured.clear()
    if act_scale is None:
        raw_norm_mean = acts.norm(dim=1).mean().item()
        act_scale = d_in ** 0.5 / raw_norm_mean if a.normalize else 1.0
        print(f'raw act norm mean {raw_norm_mean:.3f} scale {act_scale:.4f}', flush=True)
    return acts * act_scale

# ---- train, streaming with a shuffle buffer ----
BUF_ROWS = 262144
fire_counts = torch.zeros(a.expansion * d_in, device=device)
tokens_done, step, loss_ema = 0, 0, None
buf = []
buf_rows = 0
while tokens_done < a.tokens:
    while buf_rows < BUF_ROWS and tokens_done < a.tokens:
        acts = harvest(loader)
        buf.append(acts)
        buf_rows += acts.shape[0]
        tokens_done += B * T
    data = torch.cat(buf)[torch.randperm(buf_rows, device=device)]
    buf, buf_rows = [], 0
    for i in range(0, data.shape[0] - a.sae_batch + 1, a.sae_batch):
        xb = data[i:i + a.sae_batch]
        recon, topv, topi = sae(xb)
        loss = (recon - xb).pow(2).sum() / (xb - xb.mean(0)).pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.renorm()
        with torch.no_grad():
            fire_counts.scatter_add_(0, topi.flatten(), torch.ones_like(topi.flatten(), dtype=torch.float))
        loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()
        step += 1
        if step % 500 == 0:
            print(f'step {step} tokens {tokens_done} fvu_ema {loss_ema:.4f}', flush=True)
    del data

# ---- eval on held-out val tokens ----
sae.eval()
sse, svar, n_val = 0.0, 0.0, 0
val_fires = torch.zeros(a.expansion * d_in, device=device)
mean_acc = torch.zeros(d_in, device=device)
with torch.no_grad():
    val_batches = []
    while n_val < a.val_tokens:
        acts = harvest(val_loader)
        val_batches.append(acts)
        mean_acc += acts.sum(0)
        n_val += acts.shape[0]
    mu = mean_acc / n_val
    for acts in val_batches:
        recon, topv, topi = sae(acts)
        sse += (recon - acts).pow(2).sum().item()
        svar += (acts - mu).pow(2).sum().item()
        val_fires.scatter_add_(0, topi.flatten(), torch.ones_like(topi.flatten(), dtype=torch.float))

fvu = sse / svar
dead_train = (fire_counts == 0).float().mean().item()
dead_val = (val_fires == 0).float().mean().item()
density = (val_fires / n_val)
with torch.no_grad():
    Wd = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = Wd @ Wd.t()
    cos.fill_diagonal_(-1)
    max_cos = cos.max(dim=1).values

os.makedirs(a.out, exist_ok=True)
suffix = ('_norm' if a.normalize else '') + (f'_{a.tag}' if a.tag else '')
name = f'{run_tag}__{ckpt_tag}__L{a.layer}{suffix}'
metrics = dict(
    ckpt=a.ckpt, layer=a.layer, expansion=a.expansion, k=a.k,
    normalize=a.normalize, raw_act_norm_mean=raw_norm_mean,
    tokens=tokens_done, val_tokens=n_val, fvu=fvu,
    dead_frac_train=dead_train, dead_frac_val=dead_val,
    density_mean=density.mean().item(),
    density_p50=density.median().item(),
    density_p99=density.quantile(0.99).item(),
    maxcos_mean=max_cos.mean().item(),
    maxcos_p50=max_cos.median().item(),
    maxcos_p95=max_cos.quantile(0.95).item(),
)
json.dump(metrics, open(f'{a.out}/{name}.json', 'w'), indent=1)
torch.save(dict(W_dec=sae.W_dec.detach().cpu(), W_enc=sae.W_enc.detach().cpu(),
                b_enc=sae.b_enc.detach().cpu(), b_dec=sae.b_dec.detach().cpu(),
                fire_counts=fire_counts.cpu(), val_fires=val_fires.cpu(), metrics=metrics),
           f'{a.out}/{name}.pt')
print('SAVED', name, json.dumps(metrics, indent=1))
