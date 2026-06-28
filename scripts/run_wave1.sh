#!/bin/bash
exec > ~/nanogpt_wave1.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
rm -f logs/*/state_step*.pt

echo "=== W1-H1: EMu composite (3 seeds): $(date) ==="
for seed in 0 1 2; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muonattn --seed $seed \
    --init_from ckpts/muon_s${seed}/step000500.pt --start_step 500 --muon_wd 0.1 \
    --warmup_override 250 --run_suffix _emu
  echo "=== emu_s${seed} exit: $? ==="
done

echo "=== W1-H2: NS budget probe: $(date) ==="
for ns in 1 3; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muon --seed 0 --ns_steps $ns --run_suffix _ns${ns}
  echo "=== muon_s0_ns${ns} exit: $? ==="
done

echo "=== W1-H3: Lion lr probe: $(date) ==="
for lr in 0.0001 0.0003; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer lion --seed 0 --lion_lr $lr --run_suffix _lr${lr}
  echo "=== lion_s0_lr${lr} exit: $? ==="
done
rm -f logs/*/state_step*.pt

echo "=== W1-H4: SOAP SAE rescue + EMu SAEs: $(date) ==="
gpu=1; pids=()
for t in soap_s0 soap_s1 soap_s2; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=$(ls ckpts/$t/step005100.pt 2>/dev/null)
  [ -z "$ckpt" ] && continue
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --normalize --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for t in muonattn_s0_emu muonattn_s1_emu muonattn_s2_emu; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done

echo "=== W1: matchings: $(date) ==="
gpu=1; pids=()
for pr in "0:1" "0:2" "1:2"; do
  A="muonattn_s${pr%%:*}_emu"; B="muonattn_s${pr#*:}_emu"
  CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== WAVE1 COMPLETE: $(date) ==="
