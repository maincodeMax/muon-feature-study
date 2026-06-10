import json, glob, re
from collections import defaultdict

acc = defaultdict(list)
eff = defaultdict(list)
for f in glob.glob("analysis/spectra/*_summary.json"):
    run = f.split("/")[-1].replace("_summary.json", "")
    opt = run.split("_")[0]
    d = json.load(open(f))
    for ckpt, info in d.items():
        m = re.match(r"L(\d\.\d\d)_", ckpt)
        if not m:
            continue
        wp = m.group(1)
        for name, st in info["params"].items():
            cls = ".".join(name.split(".")[3:5])
            acc[(opt, wp, cls)].append(st["stable_rank"])
            eff[(opt, wp, cls)].append(st["eff_rank"])

wps = sorted(set(k[1] for k in acc))
classes = ["attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
hdr = ("waypoint", "class", "muonSR", "adamwSR", "ratio", "muonEff", "adamwEff")
print("%-9s %-12s %9s %9s %6s %9s %9s" % hdr)
for wp in wps:
    for cls in classes:
        mu = acc.get(("muon", wp, cls))
        ad = acc.get(("adamw", wp, cls))
        if not mu or not ad:
            continue
        mum, adm = sum(mu) / len(mu), sum(ad) / len(ad)
        mue = sum(eff[("muon", wp, cls)]) / len(eff[("muon", wp, cls)])
        ade = sum(eff[("adamw", wp, cls)]) / len(eff[("adamw", wp, cls)])
        print("L%-8s %-12s %9.1f %9.1f %6.2f %9.1f %9.1f" % (wp, cls, mum, adm, mum / adm, mue, ade))
