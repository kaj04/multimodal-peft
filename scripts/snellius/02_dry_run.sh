#!/bin/bash
# =============================================================================
# 02_dry_run.sh  —  Validate LoRA injection on Snellius (no data needed)
#
# Run this FIRST to confirm:
#   1. The model loads from HuggingFace
#   2. LoRA injection works correctly
#   3. Gradients flow only through LoRA parameters
#   4. Memory usage is within budget
#
# Usage:
#   sbatch scripts/snellius/02_dry_run.sh
# =============================================================================
#SBATCH --job-name=4m_lora_dryrun
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dry_run_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== Dry run 1/2: 4M-21_B (smallest, ~198M params — fastest to validate) ==="
python run_finetuning_lora.py \
    --dry_run \
    --pretrained_model EPFL-VILAB/4M-21_B \
    --lora_rank 8 \
    --lora_modules attention \
    --task semseg \
    --dtype bfloat16

echo ""
echo "=== Dry run 2/2: 4M-21_XL (~2.8B params — paper baseline model) ==="
python run_finetuning_lora.py \
    --dry_run \
    --pretrained_model EPFL-VILAB/4M-21_XL \
    --lora_rank 8 \
    --lora_modules attention \
    --task semseg \
    --dtype bfloat16

echo ""
echo "Dry run complete. Check logs above for:"
echo "  - Total params / Trainable (LoRA) params / %"
echo "  - Loss value after forward pass"
echo "  - 'OK — gradients flow only through LoRA parameters'"
echo "  - Peak GPU memory (MiB)"
