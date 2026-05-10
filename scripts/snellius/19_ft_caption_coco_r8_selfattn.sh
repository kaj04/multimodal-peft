#!/bin/bash
# =============================================================================
# 19_ft_caption_coco_r8_selfattn.sh
# LoRA fine-tuning: COCO captioning — rank=8, selfattn, dropout=0.05
#
# Mirrors depth run 15 (r8/selfattn/drop005) for clean cross-task comparison:
# same LoRA config, same LR schedule, different task (caption vs depth).
# Train subset: 10k images from COCO train2017 (seed=42).
#
# Prerequisites:
#   coco_tokens/{train,val}/{tok_rgb,caption}/ must exist.
#   Run first: sbatch scripts/snellius/09_tokenize_coco.sh
#
# Usage:
#   sbatch scripts/snellius/19_ft_caption_coco_r8_selfattn.sh
# =============================================================================
#SBATCH --job-name=4m_cap_r8_sa
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_caption_r8_selfattn_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs
cd "$REPO_ROOT"

echo "================================================================"
echo "  4M-21 LoRA FT — COCO Captioning, r8/selfattn/drop005"
echo "  Config : cfgs/finetune/caption/coco_rgb_r8_selfattn.yaml"
echo "  Output : /scratch-shared/tesr125338/output/caption/coco_rgb_r8_selfattn_drop005"
echo "  Train  : 10k images (seed=42 subset of COCO train2017)"
echo "  Val    : 5k images (full COCO val2017)"
echo "  Started: $(date)"
echo "================================================================"

python run_finetuning_lora.py \
    --config cfgs/finetune/caption/coco_rgb_r8_selfattn.yaml

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
