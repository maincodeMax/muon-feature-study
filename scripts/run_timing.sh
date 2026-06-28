#!/bin/bash
exec > ~/nanogpt_timing.log 2>&1
cd ~/research/modded-nanogpt && source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
echo "=== TIMING BENCH (ms/step + peak mem), 3 arms x 200 steps: $(date) ==="
for A in adamw muon muonvo; do
  echo "######## ARM: $A ########"
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $A --seed 0 --max_steps 200 --run_suffix _timing 2>&1 | tail -4
  echo "######## $A done ########"
  rm -f logs/*/state_step*.pt
done
echo "=== TIMING COMPLETE: $(date) ==="
