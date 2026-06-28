#!/bin/bash
exec > ~/nanogpt_filler2.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime

echo "=== F3b: remaining gelu runs: $(date) ==="
for spec in "adamw 1" "muon 2" "adamw 2"; do
  set -- $spec
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $1 --seed $2 --gelu --run_suffix _gelu
  echo "=== ${1}_s${2}_gelu exit: $? ==="
done

echo "=== F3c: gelu SAEs + matchings: $(date) ==="
gpu=1; pids=()
for t in muon_s0_gelu muon_s1_gelu muon_s2_gelu adamw_s0_gelu adamw_s1_gelu adamw_s2_gelu; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  base="analysis/saes/${t}__$(basename ${ckpt%.pt})__L6"
  [ -f "${base}.pt" ] && continue
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --gelu --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for fam in muon adamw; do
  for pr in "0:1" "0:2" "1:2"; do
    CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a ${fam}_s${pr%%:*}_gelu --run_b ${fam}_s${pr#*:}_gelu --gelu --tokens 1000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done

echo "=== F4: 350M extra seeds: $(date) ==="
for seed in 1 2; do
  for opt in muon adamw; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed --n_layer 24 --n_head 8 --n_embd 1024 --run_suffix _350m --device_batch_size 16
    echo "=== ${opt}_s${seed}_350m exit: $? ==="
    rm -f ckpts/${opt}_s${seed}_350m/L3.30_*.pt ckpts/${opt}_s${seed}_350m/L3.40_*.pt ckpts/${opt}_s${seed}_350m/L3.60_*.pt ckpts/${opt}_s${seed}_350m/L3.80_*.pt ckpts/${opt}_s${seed}_350m/step000125.pt ckpts/${opt}_s${seed}_350m/step000250.pt
    rm -f logs/*/state_step*.pt
  done
done
gpu=1; pids=()
for t in muon_s1_350m muon_s2_350m adamw_s1_350m adamw_s2_350m; do
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt ckpts/$t/step005100.pt --layer 12 --n_layer 24 --n_head 8 --n_embd 1024 --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for fam in muon adamw; do
  for pr in "0:1" "0:2" "1:2"; do
    A="${fam}_s${pr%%:*}_350m"; B="${fam}_s${pr#*:}_350m"
    CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --layer 12 --n_layer 24 --n_head 8 --n_embd 1024 --ckpt_a ckpts/$A/step005100.pt --ckpt_b ckpts/$B/step005100.pt --tokens 1000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== FILLER2 COMPLETE: $(date) ==="
