#!/bin/bash
exec > ~/nanogpt_wave2b.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
while ! grep -q "WAVE2 COMPLETE" ~/nanogpt_wave2.log 2>/dev/null; do sleep 120; done
rm -f logs/*/state_step*.pt

echo "=== W2b: circuit-targeted arms: $(date) ==="
for seed in 0 1 2; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muonqk --seed $seed --warmup_override 250 --run_suffix _w250
  echo "=== muonqk_s${seed}_w250 exit: $? ==="
done
for seed in 0 1; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muonvo --seed $seed --warmup_override 250 --run_suffix _w250
  echo "=== muonvo_s${seed}_w250 exit: $? ==="
done
rm -f logs/*/state_step*.pt

echo "=== W2b: circuits + SAEs + matchings: $(date) ==="
python analysis/circuits.py --tags muonqk_s0_w250 muonqk_s1_w250 muonqk_s2_w250 muonvo_s0_w250 muonvo_s1_w250 muonattn_s0_emu2 muon_s0_ns2 2>/dev/null
gpu=1; pids=()
for t in muonqk_s0_w250 muonqk_s1_w250 muonqk_s2_w250; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for pr in "0:1" "0:2" "1:2"; do
  A="muonqk_s${pr%%:*}_w250"; B="muonqk_s${pr#*:}_w250"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== WAVE2B COMPLETE: $(date) ==="
