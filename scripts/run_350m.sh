#!/bin/bash
exec > ~/nanogpt_350m.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
export PYTORCH_ALLOC_CONF=expandable_segments:True
for opt in muon adamw; do
  echo "=== 350m $opt start: $(date) ==="
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed 0 --n_layer 24 --n_head 8 --n_embd 1024 --run_suffix _350m --device_batch_size 16
  echo "=== 350m $opt exit: $? ==="
done
CUDA_VISIBLE_DEVICES=1 python analysis/sae_v1.py --ckpt ckpts/muon_s0_350m/step005100.pt --layer 12 --n_layer 24 --n_head 8 --n_embd 1024 --tokens 50000000 &
CUDA_VISIBLE_DEVICES=2 python analysis/sae_v1.py --ckpt ckpts/adamw_s0_350m/step005100.pt --layer 12 --n_layer 24 --n_head 8 --n_embd 1024 --tokens 50000000 &
wait
echo "=== 350m complete: $(date) ==="
