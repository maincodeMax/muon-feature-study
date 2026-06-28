#!/bin/bash
exec > ~/nanogpt_ov.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
rm -f logs/*/state_step*.pt

echo "=== OV SUFFICIENCY: AdamW + OV-rank promotion (lambda sweep): $(date) ==="
for L in 0.0001 0.001 0.01 0.03; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer adamw --seed 0 --ov_rank_reg $L --run_suffix _ovreg${L}
  echo "=== ovreg${L} exit: $? ==="
done

echo "=== missing muon-QK LR points: $(date) ==="
for LR in 0.005 0.01; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muonqk --seed 0 --warmup_override 250 --muon_lr $LR --run_suffix _lr${LR}
  echo "=== muonqk lr${LR} exit: $? ==="
done
echo "=== OV+LR COMPLETE: $(date) ==="
