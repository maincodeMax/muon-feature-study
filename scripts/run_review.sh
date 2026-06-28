#!/bin/bash
exec > ~/nanogpt_review.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime

echo "=== R1: muon+wd arm: $(date) ==="
for seed in 0 1 2; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muon --seed $seed --muon_wd 0.1 --run_suffix _wd
  echo "=== muon_s${seed}_wd exit: $? ==="
done

echo "=== R2: switch arm (muon step500 -> adamw): $(date) ==="
for seed in 0 1 2; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer adamw --seed $seed \
    --init_from ckpts/muon_s${seed}/step000500.pt --start_step 500 --run_suffix _switch
  echo "=== switch_s${seed} exit: $? ==="
done

echo "=== R3: SAEs (wd, switch) + ceiling SAEs: $(date) ==="
gpu=1; pids=()
for t in muon_s0_wd muon_s1_wd muon_s2_wd adamw_s0_switch adamw_s1_switch adamw_s2_switch; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for run in muon_s0 adamw_s0; do
  for ss in 1 2; do
    ckpt=$(ls ckpts/$run/L3.35_*.pt 2>/dev/null | head -1)
    [ -z "$ckpt" ] && ckpt=ckpts/$run/step005100.pt
    CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --sae_seed $ss --tokens 50000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== R3 done: $(date) ==="

echo "=== R4: matchings + ceiling: $(date) ==="
gpu=1; pids=()
for fam in muon_wd adamw_switch; do
  base="${fam%%_*}"; suf="${fam#*_}"
  for pr in "0:1" "0:2" "1:2"; do
    A="${base}_s${pr%%:*}_${suf}"; B="${base}_s${pr#*:}_${suf}"
    CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for run in muon_s0 adamw_s0; do
  ckpt=$(ls ckpts/$run/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$run/step005100.pt
  base="analysis/saes/${run}__$(basename ${ckpt%.pt})__L6"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_noise_ceiling.py --run $run --sae_a ${base}.pt --sae_b ${base}_saeseed1.pt &
  pids+=($!); gpu=$((gpu+1))
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_noise_ceiling.py --run $run --sae_a ${base}.pt --sae_b ${base}_saeseed2.pt &
  pids+=($!); gpu=$((gpu+1))
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_noise_ceiling.py --run $run --sae_a ${base}_saeseed1.pt --sae_b ${base}_saeseed2.pt &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== norms for wd arm:"
for t in muon_s0_wd muon_s1_wd muon_s2_wd; do
  f=$(ls analysis/saes/${t}__*__L6.json 2>/dev/null | head -1)
  [ -n "$f" ] && python -c "import json; d=json.load(open(\"$f\")); print(\"$t\", \"norm\", round(d[\"raw_act_norm_mean\"]))"
done
echo "=== REVIEW BATCH COMPLETE: $(date) ==="
