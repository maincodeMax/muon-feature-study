# Tilde-style bilinear-circuit analysis: stable rank of QK and OV circuits per arm.
import argparse, glob, json, os, sys
import torch

p = argparse.ArgumentParser()
p.add_argument('--tags', nargs='+', required=True)
p.add_argument('--per_head', action='store_true')
p.add_argument('--n_head', type=int, default=6)
a = p.parse_args()

def resolve(tag):
    c = sorted(glob.glob(f'ckpts/{tag}/L3.35_*.pt')) or sorted(glob.glob(f'ckpts/{tag}/L3.50_*.pt')) or sorted(glob.glob(f'ckpts/{tag}/step005100.pt'))
    return c[0] if c else None

def srank(M):
    s = torch.linalg.svdvals(M.float())
    return (s.pow(2).sum() / s[0].pow(2)).item()

out = {}
for tag in a.tags:
    ck = resolve(tag)
    if ck is None:
        continue
    sd = {k.removeprefix('_orig_mod.'): v.float() for k, v in torch.load(ck, map_location='cpu')['model'].items()}
    qk, ov = [], []
    L = max(int(k.split('.')[2]) for k in sd if k.startswith('transformer.h.')) + 1
    for i in range(L):
        Wq = sd[f'transformer.h.{i}.attn.c_q.weight']
        Wk = sd[f'transformer.h.{i}.attn.c_k.weight']
        Wv = sd[f'transformer.h.{i}.attn.c_v.weight']
        Wo = sd[f'transformer.h.{i}.attn.c_proj.weight']
        if a.per_head:
            hd = Wq.shape[0] // a.n_head
            for h in range(a.n_head):
                qkh = Wq[h*hd:(h+1)*hd].T @ Wk[h*hd:(h+1)*hd]
                qk.append(srank(qkh))
        else:
            qk.append(srank(Wq.T @ Wk))
        ov.append(srank(Wo @ Wv))
    import statistics
    out[tag] = dict(qk=round(sum(qk)/len(qk), 2), ov=round(sum(ov)/len(ov), 1),
                    qk_max=round(max(qk),2), qk_min=round(min(qk),2),
                    qk_cv=round(statistics.pstdev(qk)/(sum(qk)/len(qk)), 3))
    print(f"{tag:26s} QK {out[tag]['qk']:6.2f} (min {out[tag]['qk_min']:.2f} max {out[tag]['qk_max']:.2f} cv {out[tag]['qk_cv']:.2f})  OV {out[tag]['ov']:6.1f}", flush=True)
os.makedirs('analysis/figs/circuits', exist_ok=True)
json.dump(out, open('analysis/figs/circuits/circuit_sranks.json', 'w'), indent=1)
