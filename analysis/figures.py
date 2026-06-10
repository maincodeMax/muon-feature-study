# Produce the three blog figures + bootstrap CIs for the matching result.
import glob, json, os, re
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('analysis/figs', exist_ok=True)
C = {'muon': '#0e7c7b', 'adamw': '#d4622a'}

# ---------- fig 1: loss curves ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
for f in glob.glob('logs/muon_s[0-9]_*.txt') + glob.glob('logs/adamw_s[0-9]_*.txt'):
    if '350m' in f:
        continue
    opt = 'muon' if 'muon' in os.path.basename(f) else 'adamw'
    steps, losses = [], []
    for line in open(f, errors='ignore'):
        m = re.match(r'step:(\d+)/5100 val_loss:([\d.]+)', line)
        if m:
            steps.append(int(m.group(1))); losses.append(float(m.group(2)))
    if steps:
        ax.plot(steps, losses, color=C[opt], alpha=0.7, lw=1.3)
for opt, lbl in [('muon', 'Muon'), ('adamw', 'AdamW')]:
    ax.plot([], [], color=C[opt], label=lbl)
for wp in [3.8, 3.6, 3.5, 3.4, 3.35]:
    ax.axhline(wp, color='#999', lw=0.5, ls=':')
ax.set_xlim(250, 5100); ax.set_ylim(3.25, 4.2)
ax.set_xlabel('step'); ax.set_ylabel('FineWeb val loss')
ax.set_title('124M, 3 seeds per optimizer; dotted lines = matched-loss waypoints')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('analysis/figs/fig1_loss.png', dpi=200); plt.close(fig)

# ---------- fig 2: stable rank vs waypoint ----------
from collections import defaultdict
acc = defaultdict(list)
for f in glob.glob('analysis/spectra/*_summary.json'):
    run = os.path.basename(f).replace('_summary.json', '')
    if '350m' in run: continue
    opt = run.split('_')[0]
    d = json.load(open(f))
    for ckpt, info in d.items():
        m = re.match(r'L(\d\.\d\d)_', ckpt)
        if not m: continue
        for name, st in info['params'].items():
            cls = '.'.join(name.split('.')[3:5])
            acc[(opt, float(m.group(1)), cls)].append(st['stable_rank'])
classes = ['attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
all_wps = {k[1] for k in list(acc)}
wps = sorted({w for w in all_wps
              if all(acc.get((o, w, c)) for o in ('muon', 'adamw') for c in classes)}, reverse=True)
for ax, cls in zip(axes.flat, classes):
    for opt in ['muon', 'adamw']:
        ys = [sum(acc[(opt, w, cls)]) / len(acc[(opt, w, cls)]) for w in wps]
        ax.plot(wps, ys, 'o-', color=C[opt], label=opt)
    ax.set_title(cls, fontsize=10)
    ax.set_ylim(0, 165)
axes[0, 0].invert_xaxis()  # shared x: invert once so training progresses left to right
axes[0, 0].legend(frameon=False, fontsize=9)
for ax in axes[1]: ax.set_xlabel('val loss waypoint')
for ax in axes[:, 0]: ax.set_ylabel('stable rank')
fig.suptitle('Weight stable rank at matched validation loss (mean over seeds and layers)')
fig.tight_layout(); fig.savefig('analysis/figs/fig2_stablerank.png', dpi=200); plt.close(fig)

# ---------- fig 3: match distributions + bootstrap CIs ----------
def group_of(name):
    a, b = name.split('__')[0], name.split('__')[1].split('_L')[0]
    if a[:4] == b[:4] == 'muon': return 'Muon-Muon'
    if a[:5] == b[:5] == 'adamw': return 'AdamW-AdamW'
    return 'Muon-AdamW'

torch.manual_seed(0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, layer in zip(axes, [6, 9]):
    groups = {'Muon-Muon': [], 'Muon-AdamW': [], 'AdamW-AdamW': []}
    for f in glob.glob(f'analysis/saes/xmatch2/*_L{layer}_best.pt'):
        d = torch.load(f)
        groups[group_of(os.path.basename(f))].append(d['best_ab'])
    for i, (g, vecs) in enumerate(groups.items()):
        means, cis = [], []
        for v in vecs:
            boots = torch.stack([v[torch.randint(len(v), (len(v),))].mean() for _ in range(1000)])
            means.append(v.mean().item())
            cis.append((boots.quantile(0.025).item(), boots.quantile(0.975).item()))
        for m, (lo, hi) in zip(means, cis):
            ax.errorbar(i + torch.rand(1).item() * 0.3 - 0.15, m, yerr=[[m - lo], [hi - m]],
                        fmt='o', ms=5, color=['#0e7c7b', '#5a5a8a', '#d4622a'][i], capsize=3)
        print(f'L{layer} {g}: pair means {[round(m,3) for m in means]}')
    ax.set_xticks(range(3)); ax.set_xticklabels(groups.keys(), fontsize=9)
    ax.set_title(f'layer {layer}')
axes[0].set_ylabel('mean best feature match (activation corr.)')
fig.suptitle('Cross-model feature matching: per-pair means with 95% bootstrap CIs over features')
fig.tight_layout(); fig.savefig('analysis/figs/fig3_match.png', dpi=200); plt.close(fig)
print('FIGS_DONE')
