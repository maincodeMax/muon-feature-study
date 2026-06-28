#!/bin/bash
exec > ~/nanogpt_cka.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
PAIRS="muon_s0:muon_s1 muon_s0:muon_s2 muon_s1:muon_s2 adamw_s0:adamw_s1 adamw_s0:adamw_s2 adamw_s1:adamw_s2 muon_s0:adamw_s0 muon_s0:adamw_s1 muon_s0:adamw_s2 muon_s1:adamw_s0 muon_s1:adamw_s1 muon_s1:adamw_s2 muon_s2:adamw_s0 muon_s2:adamw_s1 muon_s2:adamw_s2"
gpu=1; pids=()
for pr in $PAIRS; do
  A="${pr%%:*}"; B="${pr#*:}"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/cka.py --run_a $A --run_b $B &
  pids+=($!); gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== cka complete: $(date) ==="
