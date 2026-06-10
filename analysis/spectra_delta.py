# Singular-value spectra of W - W_init: pure accumulated-update geometry, init confound removed.
# Init is reconstructed exactly: bench_v1 calls torch.manual_seed(seed) immediately before GPT(...).
import glob, json, os, re, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import GPT, GPTConfig

out = {}
for run_dir in sorted(glob.glob('ckpts/*_s[0-9]')):
    run = os.path.basename(run_dir)
    seed = int(run.rsplit('_s', 1)[1])
    torch.manual_seed(seed)
    init_sd = {k: v.cuda().float() for k, v in
               GPT(GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768)).state_dict().items()}
    ckpts = sorted(glob.glob(f'{run_dir}/L3.35_*.pt')) or sorted(glob.glob(f'{run_dir}/step005100.pt'))
    d = torch.load(ckpts[0], map_location='cuda')
    out[run] = dict(ckpt=ckpts[0], val_loss=d['val_loss'], params={})
    for name, w in d['model'].items():
        name = name.removeprefix('_orig_mod.')
        if w.ndim != 2 or not name.startswith('transformer.h.'):
            continue
        delta = w.float() - init_sd[name]
        s = torch.linalg.svdvals(delta)
        sn = s[0].item()
        fro2 = (s ** 2).sum().item()
        out[run]['params'][name] = dict(spectral_norm=sn, stable_rank=fro2 / sn ** 2)
    print('done', run)

with open('analysis/spectra/delta_summary.json', 'w') as f:
    json.dump(out, f, indent=1)

# per-class averages
from collections import defaultdict
acc = defaultdict(list)
for run, info in out.items():
    opt = run.split('_')[0]
    for name, st in info['params'].items():
        cls = '.'.join(name.split('.')[3:5])
        acc[(opt, cls)].append(st['stable_rank'])
print('%-12s %10s %10s %7s' % ('class', 'muon dSR', 'adamw dSR', 'ratio'))
for cls in ['attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
    mu = sum(acc[('muon', cls)]) / len(acc[('muon', cls)])
    ad = sum(acc[('adamw', cls)]) / len(acc[('adamw', cls)])
    print('%-12s %10.1f %10.1f %7.2f' % (cls, mu, ad, mu / ad))
