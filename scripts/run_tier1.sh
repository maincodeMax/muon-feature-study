#!/bin/bash
exec > ~/nanogpt_tier1.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc 2>/dev/null
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
while ! grep -q "OVERNIGHT COMPLETE" ~/nanogpt_overnight.log 2>/dev/null; do sleep 180; done
echo "=== overnight done, TIER1 starting: $(date) ==="

echo "=== T1A: trajectory SAEs: $(date) ==="
STEPS="step000500 step001000 step002000 step003000 step004000"
gpu=1; pids=()
for run in muon_s0 muon_s1 muon_s2 adamw_s0 adamw_s1 adamw_s2; do
  for st in $STEPS; do
    [ -f "ckpts/$run/$st.pt" ] || continue
    [ -f "analysis/saes/${run}__${st}__L6.pt" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt ckpts/$run/$st.pt --tokens 50000000 &
    pids+=($!); gpu=$((gpu+1))
    if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== T1A SAEs done: $(date) ==="

echo "=== T1B: trajectory matchings: $(date) ==="
gpu=1; pids=()
for st in $STEPS; do
  for fam in muon adamw; do
    for pr in "0:1" "0:2" "1:2"; do
      A="${fam}_s${pr%%:*}"; B="${fam}_s${pr#*:}"
      CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B \
        --ckpt_a ckpts/$A/$st.pt --ckpt_b ckpts/$B/$st.pt --name_suffix _$st --tokens 1000000 &
      pids+=($!); gpu=$((gpu+1))
      if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
    done
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== T1B done: $(date) ==="

echo "=== T1C: cross-seed SAE transfer: $(date) ==="
gpu=1; pids=()
for fam in muon adamw; do
  for i in 0 1 2 3 4; do
    for j in 0 1 2 3 4; do
      [ $i -eq $j ] && continue
      [ -d ckpts/${fam}_s${i} ] && [ -d ckpts/${fam}_s${j} ] || continue
      CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_transfer.py --sae_run ${fam}_s${i} --model_run ${fam}_s${j} &
      pids+=($!); gpu=$((gpu+1))
      if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
    done
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== TIER1 COMPLETE: $(date) ==="
