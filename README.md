# Same Dictionary, Different Geometry

Reproducible logs, code, and data for a study of what Muon actually changes inside
a language model. Recent work shows Muon representations are more robust and transferable
(arXiv:2606.09658). We ask what that means anatomically: are the sparse features different?
At 124M scale, firing-pattern matching says no. Muon recovers essentially the AdamW feature set
up to seed noise, but packages it into a hotter, sparser, more crowded, and more
seed-reproducible residual-stream geometry.

Blog post: https://maincode.com/blog/same-dictionary-different-geometry

## Headline numbers

- Final FineWeb val loss, 5,100 steps, 3 seeds: Muon 3.292 ± 0.001, AdamW 3.350 ± 0.002.
- Weight stable rank at matched loss: Muon 1.7–10× AdamW, every weight class; survives W−W_init.
- SAE feature matching over 1M shared tokens (mean best activation correlation):
  Muon↔Muon **0.618** > Muon↔AdamW **0.599** ≈ AdamW↔AdamW **0.594**.
  Crossing the optimizer costs no more than crossing the seed; Muon is the more reproducible recipe.
- Packaging at matched loss: residual stream 4–7× hotter, 4–5× more dead SAE features,
  rarer firing, tighter decoder crowding. Replicates at layer 9 (26×) and, except for the
  config-sensitive dead-feature metric, at a 350M single-seed spot check.

## Reproducible logs (`records/`)

Following the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) convention:
each log embeds the complete training source, environment info, and full loss curve.
To reproduce a run, cut the code block out of the log and launch it with the `run.sh`
pattern from modded-nanogpt (8 GPUs):

```
torchrun --standalone --nproc_per_node=8 <extracted>.py --optimizer muon --seed 0
```

| record | what |
|---|---|
| `records/{muon,adamw}_s{0,1,2}/` | the six 124M science runs (3 seeds per optimizer) |
| `records/{muon,adamw}_s{0,1,2}_350m/` | the 350M scale check (24 layers, d=1024, 3 seeds per optimizer) |
| `records/{muon,adamw}_s{3,4}/` | two extra 124M seeds per pure arm (10 within-arm pairs) |
| `records/muonattn_s{0,1,2}{,_w250}/` | hybrid: Muon on attention, AdamW on MLPs (zero-warmup and warmup-250 variants) |
| `records/muonmlp_s*{,_w250}/` | hybrid: Muon on MLPs, AdamW on attention (both warmup variants, seeds 0-4) |
| `records/muonmlp{3,6,9}_s{0,1,2}/` | dose-response: Muon on the MLPs of the first k layers only |
| `records/muon_s{0,1,2}_wd/` | full Muon with decoupled weight decay 0.1 (norm-inflation dissociation) |
| `records/adamw_s{0,1,2}_switch/` | optimizer switch: Muon for the first 500 steps, AdamW thereafter |
| `records/{muon,adamw}_s{0,1,2}_gelu/` | activation variant: GeLU in place of squared-ReLU |

Both arms use Keller Jordan's tuned hyperparameters from the
[2024-10-29 optimizer comparison record](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-10-29_Optimizers):
AdamW lr 0.0018 β=(0.9, 0.95) warmup 250, Muon lr 0.02 momentum 0.95 warmup 0,
zero weight decay, trapezoidal schedule, embed/head on AdamW in both arms.
Data: FineWeb (first 900M tokens), identical order across all runs; seeds vary init only.

## Code (`analysis/`)

| script | role |
|---|---|
| `bench_v1.py` | trainer (10/18/24 record + optimizer flag, seeds, waypoint checkpoints) |
| `spectra.py`, `spectra_delta.py`, `compare_sr.py` | weight SVD spectra, W−W_init variant |
| `sae_v1.py` | streaming top-k SAE (16×, k=32, 50M tokens), identical across models |
| `xmatch2.py`, `xmatch_verdict.py` | basis-free feature matching via firing-pattern correlation |
| `figures.py`, `cover.py`, `render_draft.py` | figures, bootstrap CIs, draft rendering |
| `model_def.py` | model/dataloader classes shared by the instruments |

Component-intervention arms inherit each side's tuned configuration unchanged;
`--warmup_override`, `--muon_wd`, `--muon_mlp_layers`, `--init_from/--start_step`, and `--gelu`
in `analysis/bench_v1.py` reproduce every arm. `data/transfer/` holds cross-seed SAE transfer
evaluations and `data/ceiling/` the same-model different-SAE-seed instrument-noise controls.

## Data (`data/`)

Per-SAE metrics (22 SAEs across layers 3/6/9, normalization control, 350M), all 45+ pairwise
matching results (layers 6 and 9), and per-run spectra summaries. Every figure and table in
the post regenerates from these files; see `analysis/figures.py`.

## Hardware

All runs on a single 8×NVIDIA B200 node. One 124M run ≈ 8 minutes; the full
six-run science sweep ≈ 65 minutes; every SAE and matching batch parallelizes one-per-GPU.
