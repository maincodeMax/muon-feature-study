#!/bin/bash
exec > ~/nanogpt_writegate.log 2>&1
cd ~/research/modded-nanogpt && source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
echo "=== MLP WRITE-GATE: muon-down (mlp write) vs muon-up (mlp read), 2 arms x 3 seeds: $(date) ==="
for seed in 0 1 2; do
  for A in muondown muonup; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $A --seed $seed --warmup_override 250 --run_suffix _wgate
    echo "=== ${A}_s${seed} exit: $? ==="
    rm -f logs/*/state_step*.pt
  done
done
echo "=== circuit ranks (MLP write vs read):"
CUDA_VISIBLE_DEVICES="" python analysis/circuits.py --tags muondown_s0_wgate muonup_s0_wgate 2>/dev/null
echo "=== WRITEGATE COMPLETE: $(date) ==="
