#!/bin/bash
exec > ~/nanogpt_soap.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
for seed in 0 1 2; do
  echo "=== soap_s${seed} start: $(date) ==="
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer soap --seed $seed
  echo "=== soap_s${seed} exit: $? ==="
done
gpu=1; pids=()
for seed in 0 1 2; do
  ckpt=$(ls ckpts/soap_s${seed}/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/soap_s${seed}/step005100.pt
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== soap SAEs done: $(date) ==="
PAIRS="soap_s0:soap_s1 soap_s0:soap_s2 soap_s1:soap_s2 soap_s0:muon_s0 soap_s0:muon_s1 soap_s0:muon_s2 soap_s1:muon_s0 soap_s1:muon_s1 soap_s1:muon_s2 soap_s2:muon_s0 soap_s2:muon_s1 soap_s2:muon_s2 soap_s0:adamw_s0 soap_s0:adamw_s1 soap_s0:adamw_s2 soap_s1:adamw_s0 soap_s1:adamw_s1 soap_s1:adamw_s2 soap_s2:adamw_s0 soap_s2:adamw_s1 soap_s2:adamw_s2"
gpu=1; pids=()
for pr in $PAIRS; do
  A="${pr%%:*}"; B="${pr#*:}"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
  pids+=($!); gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== soap arm complete: $(date) ==="
