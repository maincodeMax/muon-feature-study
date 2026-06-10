# Weight singular-value spectra across training checkpoints.
# Usage: python analysis/spectra.py [--ckpt_root ckpts] [--out analysis/spectra]
import argparse, glob, json, os
import torch

p = argparse.ArgumentParser()
p.add_argument('--ckpt_root', default='ckpts')
p.add_argument('--out', default='analysis/spectra')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

for run_dir in sorted(glob.glob(f'{a.ckpt_root}/*')):
    run = os.path.basename(run_dir)
    if os.path.exists(f'{a.out}/{run}_summary.json'):
        print('skip (done)', run)
        continue
    svals, summary = {}, {}
    for ckpt_path in sorted(glob.glob(f'{run_dir}/*.pt')):
        ckpt = os.path.basename(ckpt_path)[:-3]
        d = torch.load(ckpt_path, map_location='cuda')
        summary[ckpt] = dict(step=d['step'], val_loss=d['val_loss'], params={})
        for name, w in d['model'].items():
            name = name.removeprefix('_orig_mod.')  # torch.compile wrapper prefix
            if w.ndim != 2 or not name.startswith('transformer.h.'):
                continue
            s = torch.linalg.svdvals(w.float())
            svals[f'{ckpt}|{name}'] = s.cpu()
            sn = s[0].item()
            fro2 = (s ** 2).sum().item()
            pmass = (s ** 2) / fro2
            ent = -(pmass * (pmass + 1e-12).log()).sum().item()
            summary[ckpt]['params'][name] = dict(
                spectral_norm=sn,
                frob=fro2 ** 0.5,
                stable_rank=fro2 / sn ** 2,
                eff_rank=float(torch.exp(torch.tensor(ent))))
        del d
    torch.save(svals, f'{a.out}/{run}_svals.pt')
    with open(f'{a.out}/{run}_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print('done', run, f'({len(summary)} ckpts)')
