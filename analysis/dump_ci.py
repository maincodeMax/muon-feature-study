import glob, json, os
import torch
torch.manual_seed(0)
out = []
for f in sorted(glob.glob('analysis/saes/xmatch2/*_L6_best.pt')):
    name = os.path.basename(f).replace('_L6_best.pt', '')
    a, b = name.split('__')
    g = 'muon' if a[:4] == b[:4] == 'muon' else 'adamw' if a[:5] == b[:5] == 'adamw' else 'cross'
    v = torch.load(f)['best_ab']
    boots = torch.stack([v[torch.randint(len(v), (len(v),))].mean() for _ in range(1000)])
    out.append(dict(pair=name, group=g, m=round(v.mean().item(), 4),
                    lo=round(boots.quantile(0.025).item(), 4), hi=round(boots.quantile(0.975).item(), 4)))
json.dump(out, open('analysis/figs/pair_cis_L6.json', 'w'))
print('PAIRS', len(out))
