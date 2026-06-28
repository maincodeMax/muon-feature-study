# Publication-grade matching figure for the MOSS paper.
import glob, os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13, 'axes.linewidth': 0.8, 'font.family': 'serif'})
C = {0: '#0e7c7b', 1: '#5b6e8c', 2: '#c2622a'}
LBL = ['Muon–Muon', 'Muon–AdamW', 'AdamW–AdamW']

def group_of(name):
    a, b = name.split('__')[0], name.split('__')[1].split('_L')[0]
    if a[:4] == b[:4] == 'muon': return 0
    if a[:5] == b[:5] == 'adamw': return 2
    return 1

torch.manual_seed(0)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
for ax, layer in zip(axes, [6, 9]):
    pts = {0: [], 1: [], 2: []}
    for f in sorted(glob.glob(f'analysis/saes/xmatch2/*_L{layer}_best.pt')):
        if 'soap' in f: continue
        g = group_of(os.path.basename(f))
        v = torch.load(f)['best_ab']
        boots = torch.stack([v[torch.randint(len(v), (len(v),))].mean() for _ in range(1000)])
        pts[g].append((v.mean().item(), boots.quantile(0.025).item(), boots.quantile(0.975).item()))
    for g, rows in pts.items():
        n = len(rows)
        for k, (m, lo, hi) in enumerate(rows):
            x = g + (k - (n - 1) / 2) * (0.55 / max(n - 1, 1))
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=7,
                        color=C[g], elinewidth=1.6, capsize=4, capthick=1.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(LBL, fontsize=12)
    ax.set_title(f'Layer {layer}', fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, 2.55)
axes[0].set_ylabel('Mean best feature match')
fig.tight_layout()
fig.savefig('analysis/figs/fig_match_paper.pdf')
fig.savefig('analysis/figs/fig_match_paper.png', dpi=300)
print('PAPER_FIG_DONE')
