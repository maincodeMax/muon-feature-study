# Where Muon's Advantage Lives

Small-scale causal localization of an optimizer's effect on learned features.

This repository contains the training code, analysis scripts, and figure data for a study of
Muon vs. AdamW on a 124M-parameter language model. The study uses sparse-autoencoder (SAE)
feature matching by firing-pattern correlation, component-targeted optimizer reassignment
(value circuit vs. scoring circuit), and stable-rank analysis to localize where Muon's
advantage acts.

## Headline results

- **Feature identity is seed-ceiling limited.** Cross-optimizer SAE feature agreement (0.599)
  meets the within-AdamW seed ceiling (0.594), while within-Muon (0.618) exceeds both.
- **Muon is more reproducible, set early.** The within-Muon reproducibility advantage is fully
  formed by step 500 (one tenth of training); a 500-step Muon warm-up followed by AdamW
  retains it.
- **Muon on the value circuit (W_V W_O) is largely sufficient.** muon-VO recovers ~86% of the
  loss gap and full reproducibility; muon-QK recovers neither despite raising scoring-circuit
  stable rank to near-full-Muon levels.
- **Endpoint rank is not a sufficient knob.** Promoting W_OV stable rank under AdamW reaches
  Muon-level rank but recovers none of the advantage.

## Repository structure

```
├── bench_v1.py              # Main trainer (GPT-2 124M, Muon/AdamW/hybrid arms)
├── run_muon.py              # Muon training entry point
├── run_adam.py              # AdamW training entry point
├── soap.py                  # SOAP optimizer implementation
├── train_gpt.py             # Original modded-nanogpt trainer
├── triton_kernels.py        # Triton kernel implementations
├── analysis/
│   ├── circuits.py          # Attention circuit stable-rank analysis (W_QK, W_OV)
│   ├── cka.py               # CKA (centered kernel alignment) comparison
│   ├── xmatch.py             # Feature matching via firing-pattern correlation
│   ├── xmatch2.py           # Pairwise feature matching (v2, clustered CIs)
│   ├── xmatch_verdict.py    # Matching result aggregation
│   ├── sae_v1.py            # Streaming top-k SAE training
│   ├── sae_table.py         # SAE dictionary demographics table
│   ├── sae_transfer.py      # Cross-seed SAE transfer evaluation
│   ├── sae_noise_ceiling.py # Same-model different-SAE-seed noise ceiling
│   ├── spectra.py           # Weight SVD spectra analysis
│   ├── spectra_delta.py     # W − W_init spectral delta
│   ├── compare_sr.py        # Stable-rank comparison across arms
│   ├── loss_surface.py      # Loss surface visualization
│   ├── figures.py           # Figure generation + bootstrap CIs
│   ├── fig_paper.py         # Paper figure generation
│   ├── cover.py             # Cover image generation
│   ├── dump_ci.py           # CI dump for layer 6
│   ├── dump_ci9.py          # CI dump for layer 9
│   ├── debug_sr.py          # Stable-rank debugging
│   ├── noise_robust.py      # SAE noise robustness analysis
│   ├── surface_merge.py    # Loss surface merge utility
│   ├── model_def.py        # Shared model/dataloader definitions
│   ├── figs/                # Figure data (JSON) + rendered figures (PNG/PDF)
│   │   ├── pair_cis_L6.json  # Layer 6 feature match CIs (Fig 1a)
│   │   ├── pair_cis_L9.json  # Layer 9 feature match CIs (Appendix B)
│   │   ├── surface_rows_*.json # Loss surface data
│   │   ├── circuits/          # Circuit stable-rank data (Fig 2)
│   │   ├── cka/               # CKA comparison results
│   │   ├── ceiling/           # SAE noise ceiling results
│   │   ├── noise/             # SAE noise robustness results
│   │   └── transfer/          # Cross-seed SAE transfer results
│   ├── saes/                # SAE configs and pairwise matching results (JSON)
│   │   └── xmatch2/          # Pairwise matching results (JSON)
│   └── spectra/             # Singular value summaries (JSON)
├── scripts/                # Run scripts for all experiment arms
│   ├── run_sweep1.sh        # Main 3-seed muon+adamw runs
│   ├── run_hybrid.sh        # muonattn + muonmlp arms + SAEs + matchings
│   ├── run_overnight.sh     # Warmup controls, extra seeds, dose response
│   ├── run_ov.sh            # OV rank-reg sweep + muon-QK LR sweep
│   ├── run_vo_split.sh      # W_V-only vs W_O-only split
│   ├── run_writegate.sh    # MLP write-gate vs read-gate split
│   ├── run_qknorm.sh        # QK-norm ablation (no QK-norm, 4 arms × 2 seeds)
│   ├── run_review.sh        # Weight-decay, optimizer-switch, SAE ceilings
│   ├── run_filler.sh        # Loss-matched SAEs, gelu arms, 350M seeds
│   ├── run_filler2.sh       # Remaining gelu + 350M runs
│   ├── run_wave1.sh         # EMu composite, NS budget, Lion probe
│   ├── run_wave2.sh         # EMu-v2, switch+ramp, NS dose seeds
│   ├── run_350m.sh          # 350M scale check
│   ├── run_remain.sh        # 350M muonqk + muonvo
│   ├── run_timing.sh        # Timing benchmark (ms/step + peak mem)
│   └── ...                  # Other run scripts
├── records/                # Reproducible training logs (modded-nanogpt format)
├── data/                    # Structured analysis data (matching, CKA, spectra)
├── figures/                 # Rendered figures
├── Dockerfile               # Training environment
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Model configuration

- **Architecture:** GPT-2 class (12 layers, d=768, 6 heads, head_dim=128, vocab=50304)
- **Attention:** softmax with rotary embeddings (RoPE) and QK-norm
- **Training:** 5,100 steps, batch_size=512 (8×64), sequence_length=1024
- **Data:** FineWeb (first 900M training tokens, 100M validation), identical order across seeds
- **Hardware:** 8× NVIDIA B200 (single node)

## Optimizer configurations

Both recipes independently tuned (no under-tuned-baseline confound), following the
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) optimizer comparison:

| Optimizer | LR | Momentum | Warmup | Weight decay |
|---|---|---|---|---|
| Muon | 0.02 | 0.95 (Nesterov) | 0 steps | 0 |
| AdamW | 0.0018 | (0.9, 0.95) | 250 steps | 0 |

Embeddings and LM head use AdamW in all arms. Component-targeted arms (muon-VO, muon-QK,
muon-attn) apply Muon to the specified circuit's weights and AdamW elsewhere, with matched
250-step warmup.

## Reproducing a run

```bash
# Setup
pip install -r requirements.txt

# Train Muon seed 0 (8 GPUs)
torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muon --seed 0

# Train AdamW seed 0
torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer adamw --seed 0

# Component-targeted: Muon on value circuit only
torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muonvo --seed 0 --warmup_override 250
```

## Analysis pipeline

```bash
# SAE training (layer 6, 50M tokens, top-k=32)
python analysis/sae_v1.py --ckpt ckpts/muon_s0/L3.35_step004375.pt --tokens 50000000

# Feature matching (pairwise, 1M shared tokens)
python analysis/xmatch2.py --run_a muon_s0 --run_b muon_s1 --tokens 1000000

# Circuit stable ranks
python analysis/circuits.py --tags muon_s0 adamw_s0 muonvo_s0 muonqk_s0

# CKA comparison
python analysis/cka.py --run_a muon_s0 --run_b adamw_s0

# Weight spectra
python analysis/spectra.py --run muon_s0
```

## Compute budget

- Main model: 124M parameters, 5,100 steps, ~2.67B tokens per run
- Per-run FLOPs: ~6 × 124M × 2.67B ≈ 1.99 × 10^18
- Total experimental FLOPs (all arms): ~2.2 × 10^20
- Wall-clock: ~8 min per 124M run on 8× B200

## License

MIT
