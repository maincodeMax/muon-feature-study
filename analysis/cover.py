# Cover image generator for the muon feature study.
import glob, os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG, SURFACE, FG, STRONG, DIM, ACCENT = '#060a08', '#0f1412', '#8a9490', '#cfd9d4', '#525c57', '#00C47C'

def group_of(name):
    a, b = name.split('__')[0], name.split('__')[1].split('_L')[0]
    if a[:4] == b[:4] == 'muon': return 0
    if a[:5] == b[:5] == 'adamw': return 2
    return 1

fig, ax = plt.subplots(figsize=(12, 6.3), dpi=110)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
torch.manual_seed(0)
colors = [ACCENT, '#5a8a7d', '#8a9490']
for f in glob.glob('analysis/saes/xmatch2/*_L6_best.pt'):
    d = torch.load(f)
    g = group_of(os.path.basename(f))
    v = d['best_ab']
    boots = torch.stack([v[torch.randint(len(v), (len(v),))].mean() for _ in range(500)])
    m = v.mean().item()
    lo, hi = boots.quantile(0.025).item(), boots.quantile(0.975).item()
    x = g + (torch.rand(1).item() * 0.36 - 0.18)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=11, color=colors[g],
                ecolor=colors[g], elinewidth=2.2, capsize=5, capthick=2.2, alpha=0.95)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['MUON ↔ MUON', 'MUON ↔ ADAMW', 'ADAMW ↔ ADAMW'],
                   color=STRONG, fontsize=15, family='monospace')
ax.tick_params(axis='y', colors=FG, labelsize=12)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['bottom', 'left']:
    ax.spines[s].set_color('#1c2320')
ax.set_ylabel('feature match (activation corr.)', color=FG, fontsize=13, family='monospace')
ax.grid(axis='y', color='#1c2320', lw=0.6)
ax.set_title('SAME DICTIONARY, DIFFERENT GEOMETRY', color=ACCENT, fontsize=19,
             family='monospace', fontweight='bold', pad=20, loc='left')
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(0.578, 0.632)
ax.margins(y=0.15)
fig.subplots_adjust(left=0.105, right=0.975, top=0.86, bottom=0.17)
fig.text(0.105, 0.03, 'muon vs adamw . 15 model pairs . 95% bootstrap CIs . MC-1',
         color=DIM, fontsize=11, family='monospace')
fig.savefig('analysis/figs/muon-features-cover.png', facecolor=BG)
print('COVER_DONE')
