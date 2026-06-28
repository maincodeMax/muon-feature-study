#!/bin/bash
exec > ~/nanogpt_surface.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
# wait for the soap arm to release the GPUs
while ! grep -q "soap arm complete" ~/nanogpt_soap.log 2>/dev/null; do sleep 120; done
echo "=== soap done, starting surface evals: $(date) ==="
pids=()
gpu=1
for shard in 0:4 4:8 8:12 12:16 16:20 20:23 23:25; do
  CUDA_VISIBLE_DEVICES=$gpu python analysis/loss_surface.py --rows $shard --out analysis/figs/surface_rows_${gpu}.json &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; echo "shard rc=$?"; done
echo "=== surface complete: $(date) ==="
