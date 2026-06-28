#!/bin/bash
exec > ~/nanogpt_filler.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime

echo "=== F1: loss-matched SAEs at L3.50: $(date) ==="
TAGS="muon_s0 muon_s1 muon_s2 muon_s3 muon_s4 adamw_s0 adamw_s1 adamw_s2 adamw_s3 adamw_s4 muonmlp_s0 muonmlp_s1 muonmlp_s2 muonmlp_s3 muonmlp_s4 muonattn_s0 muonattn_s1 muonattn_s2 muonattn_s0_w250 muonattn_s1_w250 muonattn_s2_w250 muonmlp_s0_w250 muonmlp_s1_w250 muonmlp_s2_w250 muonmlp3_s0 muonmlp3_s1 muonmlp3_s2 muonmlp6_s0 muonmlp6_s1 muonmlp6_s2 muonmlp9_s0 muonmlp9_s1 muonmlp9_s2 muon_s0_wd muon_s1_wd muon_s2_wd adamw_s0_switch adamw_s1_switch adamw_s2_switch"
gpu=1; pids=()
for t in $TAGS; do
  ckpt=$(ls ckpts/$t/L3.50_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && continue
  base="analysis/saes/${t}__$(basename ${ckpt%.pt})__L6"
  [ -f "${base}.pt" ] && continue
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
  if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done

echo "=== F2: loss-matched within-arm matchings: $(date) ==="
match_fam () {
  local fam=$1; shift
  local seeds=("$@")
  for ((i=0; i<${#seeds[@]}; i++)); do
    for ((j=i+1; j<${#seeds[@]}; j++)); do
      A="${fam/SEED/${seeds[i]}}"; B="${fam/SEED/${seeds[j]}}"
      cka=$(ls ckpts/$A/L3.50_*.pt 2>/dev/null | head -1); ckb=$(ls ckpts/$B/L3.50_*.pt 2>/dev/null | head -1)
      [ -z "$cka" ] || [ -z "$ckb" ] && continue
      CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --ckpt_a "$cka" --ckpt_b "$ckb" --name_suffix _Lmatch35 --tokens 1000000 &
      pids+=($!); gpu=$((gpu+1))
      if [ $gpu -eq 8 ]; then for pid in "${pids[@]}"; do wait $pid; done; gpu=1; pids=(); fi
    done
  done
}
gpu=1; pids=()
match_fam "muon_sSEED" 0 1 2 3 4
match_fam "adamw_sSEED" 0 1 2 3 4
match_fam "muonmlp_sSEED" 0 1 2 3 4
match_fam "muonattn_sSEED" 0 1 2
match_fam "muonattn_sSEED_w250" 0 1 2
match_fam "muonmlp_sSEED_w250" 0 1 2
match_fam "muonmlp3_sSEED" 0 1 2
match_fam "muonmlp6_sSEED" 0 1 2
match_fam "muonmlp9_sSEED" 0 1 2
match_fam "muon_sSEED_wd" 0 1 2
match_fam "adamw_sSEED_switch" 0 1 2
for pid in "${pids[@]}"; do wait $pid; done

echo "=== F3: gelu architecture arm: $(date) ==="
for seed in 0 1 2; do
  for opt in muon adamw; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed --gelu --run_suffix _gelu
    echo "=== ${opt}_s${seed}_gelu exit: $? ==="
  done
done
gpu=1; pids=()
for t in muon_s0_gelu muon_s1_gelu muon_s2_gelu adamw_s0_gelu adamw_s1_gelu adamw_s2_gelu; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --gelu --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for fam in muon adamw; do
  for pr in "0:1" "0:2" "1:2"; do
    A="${fam}_s${pr%%:*}_gelu"; B="${fam}_s${pr#*:}_gelu"
    CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --gelu --tokens 1000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done

echo "=== F4: 350M extra seeds: $(date) ==="
export PYTORCH_ALLOC_CONF=expandable_segments:True
for seed in 1 2; do
  for opt in muon adamw; do
    torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer $opt --seed $seed --n_layer 24 --n_head 8 --n_embd 1024 --run_suffix _350m --device_batch_size 16
    echo "=== ${opt}_s${seed}_350m exit: $? ==="
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
echo "=== FILLER COMPLETE: $(date) ==="
