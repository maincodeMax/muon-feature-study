import json, glob
files = glob.glob("analysis/spectra/*_summary.json")
print("files:", files)
if files:
    d = json.load(open(files[0]))
    print("ckpt keys:", list(d.keys())[:6])
    k = list(d.keys())[0]
    print("param keys sample:", list(d[k]["params"].keys())[:3])
