#!/bin/bash
#SBATCH --job-name=weight_delta
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/weight_delta/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/weight_delta/slurm_%j.err
#SBATCH --export=ALL

# ||DeltaW||/||W|| for ALL image+text adapters. CPU-only (matrix norms, no
# inference): runs on a COMPUTE node to dodge login-node throttling/network.
# The ADAPTERS env string is built here, so just:  sbatch janus/eval/18_run_weight_delta.sh
# If 'genoa' is rejected, override at submit time, e.g.:
#   sbatch --partition=gpu_h100 --gres=gpu:h100:1 janus/eval/18_run_weight_delta.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/weight_delta

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
JANUS_PY=$HOME/.conda/envs/janus/bin/python

O=$HOME/multimodal-peft/janus/train/output
A=""
for r in 2 4 8 16 32 64; do A="$A;img_r$r=$O/gen_7b_rank_r$r/adapter_best"; done
for kl in 0 025 1 15; do A="$A;img_kl$kl=$O/gen_7b_kl${kl}_r32/adapter_best"; done
A="$A;img_v5=$O/gen_7b_naruto_v5/adapter_best"
# Regularizer ablation (r32): L2-SP and logit-MSE alternatives to KL (skipped if absent)
A="$A;img_l2=$O/gen_7b_l2_r32/adapter_best;img_mse=$O/gen_7b_mse_r32/adapter_best"
# Anole-style reconstruction-loss competitor (r32)
A="$A;img_anole=$O/gen_7b_anole_r32/adapter_best"
for r in 4 8 16 32 64; do A="$A;text_r$r=$O/text_poetry_r${r}_kl0.5/adapter_best"; done
for kl in 0 0.25 1 1.5; do A="$A;text_kl$(echo $kl|tr -d '.')=$O/text_poetry_r16_kl${kl}/adapter_best"; done
export ADAPTERS="${A#;}"

echo "================================================================"
echo "  weight-delta | Started: $(date)"
echo "================================================================"
$JANUS_PY ~/multimodal-peft/janus/eval/18_weight_delta_norm.py
echo "Done: $(date)"
