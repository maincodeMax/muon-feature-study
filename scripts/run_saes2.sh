#!/bin/bash
exec > ~/nanogpt_saes2.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
T=50000000
declare -a JOBS=(
  "0|--ckpt ckpts/muon_s0/step005100.pt"
  "1|--ckpt ckpts/muon_s1/step005100.pt"
  "2|--ckpt ckpts/muon_s2/step005100.pt"
  "3|--ckpt ckpts/muon_s0/L3.35_step004375.pt --layer 3"
  "4|--ckpt ckpts/adamw_s0/L3.35_step005100.pt --layer 3"
  "5|--ckpt ckpts/muon_s0/L3.35_step004375.pt --layer 9"
  "6|--ckpt ckpts/adamw_s0/L3.35_step005100.pt --layer 9"
  "7|--ckpt ckpts/muon_s0/L3.35_step004375.pt --normalize"
)
pids=()
for j in "${JOBS[@]}"; do
  gpu="${j%%|*}"; args="${j#*|}"
  echo "gpu $gpu: $args"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py $args --tokens $T &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; echo "job $pid rc=$?"; done
echo "=== wave2a done, starting wave2b: $(date) ==="
CUDA_VISIBLE_DEVICES=0 python analysis/sae_v1.py --ckpt ckpts/adamw_s0/L3.35_step005100.pt --normalize --tokens $T
echo "=== sae wave2 complete: $(date) ==="
