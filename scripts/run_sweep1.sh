#!/bin/bash
set -uo pipefail
exec > ~/nanogpt_sweep1.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
for seed in 0 1 2; do
  for opt in muon adamw; do
    echo "=== run ${opt}_s${seed} start: $(date) ==="
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed
    echo "=== run ${opt}_s${seed} exit: $? ==="
  done
done
echo "=== sweep1 complete: $(date) ==="
