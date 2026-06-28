#!/bin/bash
exec > ~/nanogpt_hybrid.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
for seed in 0 1 2; do
  for opt in muonattn muonmlp; do
    echo "=== ${opt}_s${seed} start: $(date) ==="
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed
    echo "=== ${opt}_s${seed} exit: $? ==="
  done
done
gpu=1; pids=()
for seed in 0 1 2; do
  for opt in muonattn muonmlp; do
    ckpt=$(ls ckpts/${opt}_s${seed}/L3.35_*.pt 2>/dev/null | head -1)
    [ -z "$ckpt" ] && ckpt=ckpts/${opt}_s${seed}/step005100.pt
    CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== hybrid SAEs done: $(date) ==="
PAIRS=""
for h in muonattn muonmlp; do
  PAIRS="$PAIRS ${h}_s0:${h}_s1 ${h}_s0:${h}_s2 ${h}_s1:${h}_s2"
  for hs in 0 1 2; do for os in 0 1 2; do
    PAIRS="$PAIRS ${h}_s${hs}:muon_s${os} ${h}_s${hs}:adamw_s${os}"
  done; done
done
gpu=1; pids=()
for pr in $PAIRS; do
  A="${pr%%:*}"; B="${pr#*:}"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
  pids+=($!); gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== hybrid pipeline complete: $(date) ==="
