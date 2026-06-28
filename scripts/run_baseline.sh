#!/bin/bash
set -euo pipefail
exec > ~/nanogpt_baseline.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
echo "=== baseline run start: $(date) ==="
./run.sh
echo "=== baseline done: $(date) ==="
