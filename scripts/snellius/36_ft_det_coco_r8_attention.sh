#!/bin/bash
# =============================================================================
# 36_ft_det_coco_r8_attention.sh
# LoRA fine-tuning: COCO detection, rank=8, attention, dropout=0.1
#
# Run 36: det-only baseline for comparison with run 34 (5-task multitask).
# Same LoRA config as run 34 — only single-task training on det.
# Tokenized data from run 31 (31_tokenize_coco_det.sh).
#
# Usage:
#   sbatch scripts/snellius/36_ft_det_coco_r8_attention.sh
# =============================================================================
#SBATCH --job-name=ft_det_r8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_det_coco_r8_attention_%j.out

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
echo "  Det-only LoRA fine-tuning (run 36b — padding fix)"
echo "  Task   : det (COCO val2017, 10k train / 5k val)"
echo "  LoRA   : rank=8, attention, dropout=0.1"
echo "  Fix    : mask_padding=True (excludes pad tokens from loss)"
echo "  Output : /scratch-shared/tesr125338/output/det/coco_rgb_r8_attention_v2"
echo "  Started: $(date)"
echo "================================================================"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=29507 \
    run_finetuning_lora.py \
    --config cfgs/finetune/det/coco_rgb_r8_attention.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Weights → /scratch-shared/tesr125338/output/det/coco_rgb_r8_attention/"
echo "    lora_weights-best.pth  ← use for eval_all.py (v2 with padding fix)"
echo "================================================================"
