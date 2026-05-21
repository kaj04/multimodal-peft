#!/bin/bash
# =============================================================================
# 39_ft_caption_coco_r8_attention_v2.sh
# LoRA fine-tuning: COCO captioning — rank=8, attention, dropout=0.1
#
# FIXED vs run 23: padding tokens (id=0) excluded from the loss
# (build_mod_dict mask_padding=True). With ~80% padding in caption sequences,
# run 23 learned to predict zeros in 2 epochs and early-stopped at epoch 13
# with val/token_acc=0.787 ≈ all-zeros baseline (0.806). This run should
# yield a meaningful loss curve and properly converge on actual captioning.
# Patience raised 10→20.
#
# Usage:
#   sbatch scripts/snellius/39_ft_caption_coco_r8_attention_v2.sh
# =============================================================================
#SBATCH --job-name=ft_cap_v2
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_caption_r8_attention_v2_%j.out

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
echo "  Caption LoRA fine-tuning (run 39 — padding fix)"
echo "  Task   : caption (COCO, 10k train / 5k val)"
echo "  LoRA   : rank=8, attention, dropout=0.1"
echo "  Fix    : mask_padding=True (excludes pad tokens from loss)"
echo "  Output : /scratch-shared/tesr125338/output/caption/coco_rgb_r8_attention_v2"
echo "  Started: $(date)"
echo "================================================================"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=29510 \
    run_finetuning_lora.py \
    --config cfgs/finetune/caption/coco_rgb_r8_attention.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Weights → /scratch-shared/tesr125338/output/caption/coco_rgb_r8_attention_v2/"
echo "    lora_weights-best.pth  ← use for eval_all.py"
echo "================================================================"
