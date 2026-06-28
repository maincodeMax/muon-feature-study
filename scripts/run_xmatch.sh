#!/bin/bash
exec > ~/nanogpt_xmatch.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
PAIRS="muon_s0:muon_s1 muon_s0:muon_s2 muon_s1:muon_s2 adamw_s0:adamw_s1 adamw_s0:adamw_s2 adamw_s1:adamw_s2 muon_s0:adamw_s0 muon_s0:adamw_s1 muon_s0:adamw_s2 muon_s1:adamw_s0 muon_s1:adamw_s1 muon_s1:adamw_s2 muon_s2:adamw_s0 muon_s2:adamw_s1 muon_s2:adamw_s2"
gpu=0
pids=()
for pr in $PAIRS; do
  A="${pr%%:*}"; B="${pr#*:}"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
  pids+=($!)
  gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then
    for pid in "${pids[@]}"; do wait $pid; done
    gpu=0; pids=()
  fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== xmatch complete: $(date) ==="
