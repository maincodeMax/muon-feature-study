#!/bin/bash
exec > ~/nanogpt_vosplit.log 2>&1
cd ~/research/modded-nanogpt && source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
echo "=== WV-only vs WO-only split: 2 arms x 3 seeds: $(date) ==="
for seed in 0 1 2; do
  for A in muonv muono; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $A --seed $seed --warmup_override 250 --run_suffix _split
    echo "=== ${A}_s${seed} exit: $? ==="
    rm -f logs/*/state_step*.pt
  done
done
echo "=== circuit ranks (which projection lifts OV rank?):"
CUDA_VISIBLE_DEVICES="" python analysis/circuits.py --tags muonv_s0_split muono_s0_split 2>/dev/null
echo "=== VOSPLIT COMPLETE: $(date) ==="
