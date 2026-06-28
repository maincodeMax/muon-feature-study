import json, glob
rows = []
for f in sorted(glob.glob("analysis/saes/*.json")):
    d = json.load(open(f))
    run = d["ckpt"].split("/")[1]
    ck = d["ckpt"].split("/")[2].replace(".pt", "")
    rows.append((run, ck, d["layer"], d.get("normalize", False),
                 d.get("raw_act_norm_mean"), d["fvu"], d["dead_frac_val"],
                 d["maxcos_mean"], d["density_p50"]))
print("%-9s %-20s %2s %-5s %8s %7s %7s %7s %9s" %
      ("run", "ckpt", "L", "norm", "actnorm", "fvu", "dead", "maxcos", "dens_p50"))
for r in sorted(rows, key=lambda x: (x[2], x[3], x[0], x[1])):
    an = ("%8.0f" % r[4]) if r[4] else "      --"
    print("%-9s %-20s %2d %-5s %s %7.4f %7.4f %7.4f %9.6f" %
          (r[0], r[1], r[2], str(r[3]), an, r[5], r[6], r[7], r[8]))
