import json, glob
groups = {"within_muon": [], "within_adamw": [], "cross": []}
for f in sorted(glob.glob("analysis/saes/xmatch2/*.json")):
    d = json.load(open(f))
    a, b = d["run_a"], d["run_b"]
    kind = "within_muon" if a[:4] == b[:4] == "muon" else \
           "within_adamw" if a[:5] == b[:5] == "adamw" else "cross"
    m = d["a_to_b"]
    groups[kind].append((f.split("/")[-1][:-5], d["tokens"], m["mean"], m["p50"], m["frac_gt05"], m["frac_gt07"]))
    print("%-13s %-22s tok %7d  mean %.3f  p50 %.3f  >0.5 %.3f  >0.7 %.3f" %
          (kind, a + "|" + b, d["tokens"], m["mean"], m["p50"], m["frac_gt05"], m["frac_gt07"]))
print()
for k, rows in groups.items():
    n = len(rows)
    print("AVG %-13s (n=%d)  mean %.3f  p50 %.3f  >0.5 %.3f  >0.7 %.3f" %
          (k, n, sum(r[2] for r in rows) / n, sum(r[3] for r in rows) / n,
           sum(r[4] for r in rows) / n, sum(r[5] for r in rows) / n))
