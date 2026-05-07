#!/bin/bash
# =============================================================================
# 05_overfit_test.sh — LoRA overfit diagnostic (1 GPU, ~1h)
#
# Trains LoRA on 50 fixed samples for 200 epochs, then evaluates AbsRel
# on those same samples vs baseline. Diagnoses whether the issue is:
#   (A) LoRA architectural incompatibility / bug
#   (B) CE / metric structural mismatch
#   (C) Generalisation problem (fixable with hyperparam tuning)
#
# Usage:
#   sbatch scripts/snellius/05_overfit_test.sh
# =============================================================================
#SBATCH --job-name=4m_overfit_test
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/overfit_test_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"
cd "$REPO_ROOT"
mkdir -p logs

echo "=== LoRA Overfit Diagnostic ==="
echo "    Samples : 50"
echo "    Epochs  : 200"
echo "    LoRA    : rank=8, attention only"
echo "    GPU     : $SLURM_JOB_GPUS"

python run_overfit_test.py \
    --data_path  /scratch-shared/tesr125338/nyu_tokens/train \
    --n_samples  50 \
    --epochs     200 \
    --lora_rank  8 \
    --lora_modules attention \
    --lr         2e-4 \
    --output_dir output/overfit_test

echo "=== Done. Results in output/overfit_test/overfit_results.json ==="
