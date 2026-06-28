#!/bin/bash
exec > ~/nanogpt_sae4.log 2>&1
cd ~/research/modded-nanogpt && source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
rm -f logs/*/state_step*.pt
echo "=== #4 SAE k-robustness: train k=16/64 SAEs: $(date) ==="
for K in 16 64; do gpu=0; pids=()
  for t in muon_s0 muon_s1 muon_s2 adamw_s0 adamw_s1 adamw_s2; do
    ck=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null|head -1); [ -z "$ck" ] && ck=ckpts/$t/step005100.pt
    CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ck" --k $K --tag k$K --tokens 50000000 & pids+=($!); gpu=$((gpu+1))
  done; for p in "${pids[@]}"; do wait $p; done
done
echo "=== matchings (using k-tagged SAEs via --sae_suffix): $(date) ==="
for K in 16 64; do gpu=0; pids=()
  for fam in muon adamw; do for pr in "0:1" "0:2" "1:2"; do
    ck=$(ls ckpts/${fam}_s${pr%%:*}/L3.35_*.pt 2>/dev/null|head -1)
    CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a ${fam}_s${pr%%:*} --run_b ${fam}_s${pr#*:} --sae_suffix _k$K --name_suffix _k$K --tokens 1000000 & pids+=($!); gpu=$((gpu+1))
  done; done; for p in "${pids[@]}"; do wait $p; done
done
echo "=== SAE4 COMPLETE: $(date) ==="
