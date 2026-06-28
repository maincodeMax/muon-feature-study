#!/bin/bash
exec > ~/nanogpt_qknorm.log 2>&1
cd ~/research/modded-nanogpt && source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
echo "=== QK-NORM ABLATION (confound test): 4 arms x 2 seeds, NO QK-norm: $(date) ==="
for seed in 0 1; do
  for spec in "adamw" "muon" "muonqk --warmup_override 250" "muonvo --warmup_override 250"; do
    set -- $spec
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $1 ${@:2} --seed $seed --no_qknorm --run_suffix _noqkn
    echo "=== ${1}_s${seed}_noqkn exit: $? ==="
    rm -f logs/*/state_step*.pt
  done
done
echo "=== QKNORM ABLATION COMPLETE: $(date) ==="
