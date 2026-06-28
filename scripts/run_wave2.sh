#!/bin/bash
exec > ~/nanogpt_wave2.log 2>&1
cd ~/research/modded-nanogpt
source .venv/bin/activate
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime
rm -f logs/*/state_step*.pt

echo "=== W2: EMu-v2 with post-switch ramp (3 seeds): $(date) ==="
for seed in 0 1 2; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muonattn --seed $seed \
    --init_from ckpts/muon_s${seed}/step000500.pt --start_step 500 --muon_wd 0.1 \
    --post_switch_warmup 250 --run_suffix _emu2
  echo "=== emu2_s${seed} exit: $? ==="
done

echo "=== W2: switch+ramp control: $(date) ==="
torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer adamw --seed 0 \
  --init_from ckpts/muon_s0/step000500.pt --start_step 500 --post_switch_warmup 250 --run_suffix _switchramp
echo "=== switchramp_s0 exit: $? ==="

echo "=== W2: NS dose seeds: $(date) ==="
for seed in 1 2; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muon --seed $seed --ns_steps 3 --run_suffix _ns3
  echo "=== muon_s${seed}_ns3 exit: $? ==="
done
torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer muon --seed 0 --ns_steps 2 --run_suffix _ns2
echo "=== muon_s0_ns2 exit: $? ==="

echo "=== W2: Lion retune: $(date) ==="
for lr in 0.0006 0.001; do
  torchrun --standalone --nproc_per_node=8 bench_v1.py --optimizer lion --seed 0 --lion_lr $lr --run_suffix _lr${lr}
  echo "=== lion_s0_lr${lr} exit: $? ==="
done
rm -f logs/*/state_step*.pt

echo "=== W2: SAEs + matchings: $(date) ==="
gpu=1; pids=()
for t in muonattn_s0_emu2 muonattn_s1_emu2 muonattn_s2_emu2 muon_s0_ns3 muon_s1_ns3 muon_s2_ns3; do
  ckpt=$(ls ckpts/$t/L3.35_*.pt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && ckpt=ckpts/$t/step005100.pt
  base="analysis/saes/${t}__$(basename ${ckpt%.pt})__L6"
  [ -f "${base}.pt" ] && continue
  CUDA_VISIBLE_DEVICES=$gpu python analysis/sae_v1.py --ckpt "$ckpt" --tokens 50000000 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait $pid; done
gpu=1; pids=()
for fam in emu2 ns3; do
  pre="muonattn"; suf="_emu2"
  if [ "$fam" = "ns3" ]; then pre="muon"; suf="_ns3"; fi
  for pr in "0:1" "0:2" "1:2"; do
    A="${pre}_s${pr%%:*}${suf}"; B="${pre}_s${pr#*:}${suf}"
    CUDA_VISIBLE_DEVICES=$gpu python analysis/xmatch2.py --run_a $A --run_b $B --tokens 1000000 &
    pids+=($!); gpu=$((gpu+1))
  done
done
for pid in "${pids[@]}"; do wait $pid; done
echo "=== WAVE2 COMPLETE: $(date) ==="
