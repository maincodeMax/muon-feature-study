#!/bin/bash
set -uo pipefail
exec > ~/nanogpt_pair.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
echo "=== muon run start: $(date) ==="
torchrun --standalone --nproc_per_node=8 run_muon.py
echo "=== muon exit code: $? ==="
echo "=== adam run start: $(date) ==="
torchrun --standalone --nproc_per_node=8 run_adam.py
echo "=== adam exit code: $? ==="
echo "=== pair complete: $(date) ==="
