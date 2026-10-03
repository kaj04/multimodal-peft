#!/bin/bash
#SBATCH --job-name=weight_delta
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=janus/tests/output/weight_delta/slurm_%j.out
#SBATCH --error=janus/tests/output/weight_delta/slurm_%j.err
#SBATCH --export=ALL

# ||DeltaW||/||W|| for ALL image+text adapters. CPU-only (matrix norms, no
# inference): runs on a COMPUTE node to dodge login-node throttling/network.
# The ADAPTERS env string is built here, so just:  sbatch janus/eval/18_run_weight_delta.sh
# If 'genoa' is rejected, override at submit time, e.g.:
#   sbatch --partition=gpu_h100 --gres=gpu:h100:1 janus/eval/18_run_weight_delta.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/weight_delta

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
JANUS_PY=$HOME/.conda/envs/janus/bin/python

O=$HOME/multimodal-peft/janus/train/output

# An ADAPTERS string passed in from outside wins over the default list built
# below. Without this the runner silently overwrote whatever the caller asked
# for and evaluated the default sweep instead, which is easy to miss because
# the job still succeeds and produces a full-looking table.
if [ -n "${ADAPTERS:-}" ]; then
  echo "  ADAPTERS supplied by caller; skipping the default list."
else

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
# Strong text targets at r32 for a same-rank comparison: Welsh (language) vs
# medicine (domain) vs poetry r32 above (style). KL differs (0.1/0.25/0.5).
# text_welsh pointed at text_7b_welsh_r32_kl0.1, the adapter whose training was
# cut short by the wall clock and which was replaced everywhere else in the
# paper. It gives 0.598% here against the 1.9% the paper reports, so anyone
# re-running this script would contradict the text.
A="$A;text_welsh=$O/chainA_s1_welsh/adapter_best"
A="$A;text_med=$O/text_med_r32_kl025/adapter_best"
export ADAPTERS="${A#;}"
fi

echo "================================================================"
echo "  weight-delta | Started: $(date)"
echo "================================================================"
$JANUS_PY ~/multimodal-peft/janus/eval/18_weight_delta_norm.py
echo "Done: $(date)"
