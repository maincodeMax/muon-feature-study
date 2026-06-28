#!/bin/bash
exec > ~/nanogpt_harden.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime

echo "=== PHASE A: L9 SAEs + delta spectra: $(date) ==="
CUDA_VISIBLE_DEVICES=0 python analysis/sae_v1.py --ckpt ckpts/muon_s1/L3.35_step004375.pt --layer 9 --tokens 50000000 &
CUDA_VISIBLE_DEVICES=1 python analysis/sae_v1.py --ckpt ckpts/muon_s2/L3.35_step004375.pt --layer 9 --tokens 50000000 &
CUDA_VISIBLE_DEVICES=2 python analysis/sae_v1.py --ckpt ckpts/adamw_s1/step005100.pt --layer 9 --tokens 50000000 &
CUDA_VISIBLE_DEVICES=3 python analysis/sae_v1.py --ckpt ckpts/adamw_s2/L3.35_step005100.pt --layer 9 --tokens 50000000 &
CUDA_VISIBLE_DEVICES=4 python analysis/spectra_delta.py &
wait
echo "=== PHASE B: L9 xmatch: $(date) ==="
PAIRS="muon_s0:muon_s1 muon_s0:muon_s2 muon_s1:muon_s2 adamw_s0:adamw_s1 adamw_s0:adamw_s2 adamw_s1:adamw_s2 muon_s0:adamw_s0 muon_s0:adamw_s1 muon_s0:adamw_s2 muon_s1:adamw_s0 muon_s1:adamw_s1 muon_s1:adamw_s2 muon_s2:adamw_s0 muon_s2:adamw_s1 muon_s2:adamw_s2"
gpu=0; pids=()
for pr in $PAIRS; do
  A="${pr%%:*}"; B="${pr#*:}"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --layer 9 --tokens 1000000 &
  pids+=($!)
  gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=0; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== PHASE C: 350M pair: $(date) ==="
torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muon --seed 0 --n_layer 24 --n_head 8 --n_embd 1024 --run_suffix _350m
echo "=== 350m muon exit: $? ==="
torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer adamw --seed 0 --n_layer 24 --n_head 8 --n_embd 1024 --run_suffix _350m
echo "=== 350m adamw exit: $? ==="
CUDA_VISIBLE_DEVICES=0 python analysis/sae_v1.py --ckpt ckpts/muon_s0_350m/step005100.pt --layer 12 --n_layer 24 --n_head 8 --n_embd 1024 --tokens 50000000 &
CUDA_VISIBLE_DEVICES=1 python analysis/sae_v1.py --ckpt ckpts/adamw_s0_350m/step005100.pt --layer 12 --n_layer 24 --n_head 8 --n_embd 1024 --tokens 50000000 &
wait
echo "=== harden complete: $(date) ==="
