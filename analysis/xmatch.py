# Cross-match SAE decoder dictionaries between runs.
# For each alive feature in A, find its best cosine match among alive features in B.
# Within-optimizer seed pairs = reproducibility ceiling; cross-optimizer pairs = the question.
import itertools, json
import torch

RUNS = {
    'muon_s0': 'analysis/saes/muon_s0__L3.35_step004375__L6.pt',
    'muon_s1': 'analysis/saes/muon_s1__L3.35_step004375__L6.pt',
    'muon_s2': 'analysis/saes/muon_s2__L3.35_step004375__L6.pt',
    'adamw_s0': 'analysis/saes/adamw_s0__L3.35_step005100__L6.pt',
    'adamw_s1': 'analysis/saes/adamw_s1__step005100__L6.pt',
    'adamw_s2': 'analysis/saes/adamw_s2__L3.35_step005100__L6.pt',
}

def load(path):
    d = torch.load(path, map_location='cuda')
    W = d['W_dec'].cuda().float()
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    alive = d['val_fires'].cuda() > 0
    return W[alive]

dicts = {k: load(v) for k, v in RUNS.items()}
for k, v in dicts.items():
    print(k, 'alive features:', v.shape[0])

def match(A, B):
    best = (A @ B.t()).max(dim=1).values
    return dict(mean=best.mean().item(), p50=best.median().item(),
                p25=best.quantile(0.25).item(),
                frac_gt07=(best > 0.7).float().mean().item(),
                frac_gt05=(best > 0.5).float().mean().item())

results = {}
for a, b in itertools.combinations(RUNS, 2):
    kind = 'within_muon' if a.startswith('muon') and b.startswith('muon') else \
           'within_adamw' if a.startswith('adamw') and b.startswith('adamw') else 'cross'
    m = match(dicts[a], dicts[b])
    results[f'{a}|{b}'] = dict(kind=kind, **m)
    print(f'{kind:13} {a}->{b}: mean {m["mean"]:.3f} p50 {m["p50"]:.3f} '
          f'frac>0.7 {m["frac_gt07"]:.3f} frac>0.5 {m["frac_gt05"]:.3f}')

for kind in ['within_muon', 'within_adamw', 'cross']:
    rs = [r for r in results.values() if r['kind'] == kind]
    print(f'AVG {kind:13} mean {sum(r["mean"] for r in rs)/len(rs):.3f} '
          f'frac>0.7 {sum(r["frac_gt07"] for r in rs)/len(rs):.3f}')
json.dump(results, open('analysis/saes/xmatch_L6.json', 'w'), indent=1)
