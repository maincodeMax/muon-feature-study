import glob, json
rows = {}
meta = None
for f in glob.glob('analysis/figs/surface_rows_*.json'):
    d = json.load(open(f))
    meta = d
    for r in d['rows']:
        rows[r['i']] = r['losses']
grid = [rows[i] for i in range(meta['grid'])]
out = dict(grid=grid, alphas=meta['alphas'], betas=meta['betas'],
           positions=meta['positions'], eval_tokens=meta['eval_tokens'], models=meta['models'])
json.dump(out, open('analysis/figs/surface.json', 'w'))

A = meta['alphas']; B = meta['betas']; P = meta['positions']
def at(a, b):
    i = min(range(len(A)), key=lambda i: abs(A[i] - a))
    j = min(range(len(B)), key=lambda j: abs(B[j] - b))
    return grid[i][j]

print('models: A=muon_s0 (0,0)  B=adamw_s0 (1,0)  C=muon_s1 (%.3f,1)' % P['C'][0])
print('loss at A: %.4f  B: %.4f  C: %.4f' % (at(0, 0), at(1, 0), at(P['C'][0], 1)))
print('direct path A->B (muon -> adamw):')
for t in [0.25, 0.5, 0.75]:
    print('  t=%.2f loss %.4f' % (t, at(t, 0)))
print('direct path A->C (muon seed -> muon seed):')
for t in [0.25, 0.5, 0.75]:
    print('  t=%.2f loss %.4f' % (t, at(t * P['C'][0], t)))
print('direct path B->C (adamw -> muon seed):')
for t in [0.25, 0.5, 0.75]:
    a = 1 + t * (P['C'][0] - 1)
    print('  t=%.2f loss %.4f' % (t, at(a, t)))
flat = [v for row in grid for v in row]
print('grid min %.3f max %.3f  capped@9: %d/%d' % (min(flat), max(flat), sum(1 for v in flat if v >= 9), len(flat)))
print('norm_u %.1f norm_v %.1f' % (P['norm_u'], P['norm_v']))
