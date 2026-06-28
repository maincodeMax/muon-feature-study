#!/bin/bash
exec > ~/nanogpt_saes.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
gpu=0
pids=()
for run in muon_s0 muon_s1 muon_s2 adamw_s0 adamw_s1 adamw_s2; do
  ckpt=$(ls ckpts/$run/L3.35_*.pt | head -1)
  echo "gpu $gpu -> $ckpt"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!)
  gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; echo "job $pid done rc=$?"; done
echo "=== sae batch complete: $(date) ==="
