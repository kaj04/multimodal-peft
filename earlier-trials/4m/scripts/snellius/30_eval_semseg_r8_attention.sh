#!/bin/bash
# =============================================================================
# 30_eval_semseg_r8_attention.sh
# Semseg mIoU evaluation for run 25 (semseg-only LoRA, r8/attention).
#
# Compare against:
#   eval_semseg/multitask_caption_semseg_r8/  <- run 26 mIoU=25.32%
#
# Usage:
#   sbatch scripts/snellius/30_eval_semseg_r8_attention.sh
# =============================================================================
#SBATCH --job-name=eval_sem_r8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_semseg_r8_attention_%j.out

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
echo "  Semseg eval -- run 25 (r8/attention, semseg-only LoRA)"
echo "  Weights : semseg/coco_rgb_r8_attention/lora_weights-best.pth"
echo "  Dataset : COCO val2017 (5000 images)"
echo "  Started : $(date)"
echo "================================================================"

python eval_semseg_coco.py \
    --lora_weights /scratch-shared/tesr125338/output/semseg/coco_rgb_r8_attention/lora_weights-best.pth \
    --lora_rank    8 \
    --lora_modules attention \
    --coco_dir     /scratch-shared/tesr125338/coco \
    --tokens_dir   /scratch-shared/tesr125338/coco_tokens/val \
    --output_dir   /scratch-shared/tesr125338/output/eval_semseg/coco_r8_attention \
    --run_name     semseg_r8_attention \
    --device       cuda \
    --dtype        bfloat16

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Results --> output/eval_semseg/coco_r8_attention/"
echo "  Compare: multitask mIoU=25.32% vs semseg-only above"
echo "================================================================"
