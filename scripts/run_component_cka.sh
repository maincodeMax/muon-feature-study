#!/bin/bash
exec > ~/nanogpt_compcka.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
while ! grep -q "tiny pipeline complete" ~/nanogpt_tiny.log 2>/dev/null; do sleep 120; done
echo "=== tiny done, starting component CKA: $(date) ==="
PAIRS="muon_s0:muon_s1 muon_s0:muon_s2 muon_s1:muon_s2 adamw_s0:adamw_s1 adamw_s0:adamw_s2 adamw_s1:adamw_s2 muon_s0:adamw_s0 muon_s0:adamw_s1 muon_s0:adamw_s2 muon_s1:adamw_s0 muon_s1:adamw_s1 muon_s1:adamw_s2 muon_s2:adamw_s0 muon_s2:adamw_s1 muon_s2:adamw_s2"
for comp in attn mlp; do
  gpu=1; pids=()
  for pr in $PAIRS; do
    A="${pr%%:*}"; B="${pr#*:}"
    CUDA_VISIBLE_DEVICES=$gpu python analysis/cka.py --run_a $A --run_b $B --component $comp &
    pids+=($!); gpu=$((gpu+1))
    if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
  done
  for pid in "${pids[@]}"; do wait $pid; done
  echo "=== component $comp done: $(date) ==="
done
echo "=== component cka complete: $(date) ==="
