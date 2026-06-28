#!/bin/bash
exec > ~/nanogpt_overnight.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime

echo "=== PHASE A: warmup controls: $(date) ==="
for seed in 0 1 2; do
  for opt in muonattn muonmlp; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed --warmup_override 250 --run_suffix _w250
    echo "=== ${opt}_s${seed}_w250 exit: $? ==="
  done
done

echo "=== PHASE B: extra seeds: $(date) ==="
for seed in 3 4; do
  for opt in muon adamw muonmlp; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed
    echo "=== ${opt}_s${seed} exit: $? ==="
  done
done

echo "=== PHASE C: dose response: $(date) ==="
for k in 3 6 9; do
  for seed in 0 1 2; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muonmlp --muon_mlp_layers $k --seed $seed
    echo "=== muonmlp${k}_s${seed} exit: $? ==="
  done
done

echo "=== PHASE D: SAEs: $(date) ==="
TAGS="muonattn_s0_w250 muonattn_s1_w250 muonattn_s2_w250 muonmlp_s0_w250 muonmlp_s1_w250 muonmlp_s2_w250 muon_s3 muon_s4 adamw_s3 adamw_s4 muonmlp_s3 muonmlp_s4 muonmlp3_s0 muonmlp3_s1 muonmlp3_s2 muonmlp6_s0 muonmlp6_s1 muonmlp6_s2 muonmlp9_s0 muonmlp9_s1 muonmlp9_s2"
gpu=1; pids=()
for t in $TAGS; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== SAEs done: $(date) ==="

echo "=== PHASE E: matchings: $(date) ==="
PAIRS=""
for fam in muonattn_w250 muonmlp_w250; do
  base="${fam%_w250}"
  PAIRS="$PAIRS ${base}_s0_w250:${base}_s1_w250 ${base}_s0_w250:${base}_s2_w250 ${base}_s1_w250:${base}_s2_w250"
done
for k in 3 6 9; do
  PAIRS="$PAIRS muonmlp${k}_s0:muonmlp${k}_s1 muonmlp${k}_s0:muonmlp${k}_s2 muonmlp${k}_s1:muonmlp${k}_s2"
done
for fam in muon adamw muonmlp; do
  for pr in "0:3" "0:4" "1:3" "1:4" "2:3" "2:4" "3:4"; do
    a="${pr%%:*}"; b="${pr#*:}"
    PAIRS="$PAIRS ${fam}_s${a}:${fam}_s${b}"
  done
done
gpu=1; pids=()
for pr in $PAIRS; do
  A="${pr%%:*}"; B="${pr#*:}"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
  pids+=($!); gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== matchings done: $(date) ==="

echo "=== PHASE F: hybrid L9 + CKA: $(date) ==="
gpu=1; pids=()
for t in muonattn_s0 muonattn_s1 muonattn_s2 muonmlp_s0 muonmlp_s1 muonmlp_s2; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --layer 9 --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for fam in muonattn muonmlp; do
  for pr in "0:1" "0:2" "1:2"; do
    a="${pr%%:*}"; b="${pr#*:}"
    CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a ${fam}_s${a} --run_b ${fam}_s${b} --layer 9 --tokens 1000000 &
    pids+=($!); gpu=$((gpu+1))
    CUDA_VISIBLE_DEVICES=$gpu python analysis/cka.py --run_a ${fam}_s${a} --run_b ${fam}_s${b} &
    pids+=($!); gpu=$((gpu+1))
    if [ $gpu -ge 7 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== OVERNIGHT COMPLETE: $(date) ==="
