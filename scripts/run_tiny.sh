#!/bin/bash
exec > ~/nanogpt_tiny.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
DIMS="--n_layer 6 --n_head 6 --n_embd 384"
for seed in 0 1 2; do
  for opt in muon adamw; do
    echo "=== tiny ${opt}_s${seed} start: $(date) ==="
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed $DIMS --run_suffix _tiny
    echo "=== tiny ${opt}_s${seed} exit: $? ==="
  done
done
gpu=1; pids=()
for seed in 0 1 2; do
  for opt in muon adamw; do
    ckpt=$(ls ckpts/${opt}_s${seed}_tiny/L3.35_*.pt 2>/dev/null | head -1)
    [ -z "$ckpt" ] && ckpt=ckpts/${opt}_s${seed}_tiny/step005100.pt
    CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --layer 3 $DIMS --tokens 50000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== tiny SAEs done: $(date) ==="
PAIRS="muon_s0_tiny:muon_s1_tiny muon_s0_tiny:muon_s2_tiny muon_s1_tiny:muon_s2_tiny adamw_s0_tiny:adamw_s1_tiny adamw_s0_tiny:adamw_s2_tiny adamw_s1_tiny:adamw_s2_tiny muon_s0_tiny:adamw_s0_tiny muon_s0_tiny:adamw_s1_tiny muon_s0_tiny:adamw_s2_tiny muon_s1_tiny:adamw_s0_tiny muon_s1_tiny:adamw_s1_tiny muon_s1_tiny:adamw_s2_tiny muon_s2_tiny:adamw_s0_tiny muon_s2_tiny:adamw_s1_tiny muon_s2_tiny:adamw_s2_tiny"
gpu=1; pids=()
for pr in $PAIRS; do
  A="${pr%%:*}"; B="${pr#*:}"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --layer 3 $DIMS --tokens 1000000 &
  pids+=($!); gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== tiny pipeline complete: $(date) ==="
