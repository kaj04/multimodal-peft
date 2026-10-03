#!/bin/bash
# =============================================================================
# 27_eval_semseg_multitask.sh
# Pixel-space semseg eval of multi-task LoRA (run 26) on COCO val2017.
# Metric: mIoU (mean Intersection-over-Union, 133 classes).
#
# Usage:
#   sbatch scripts/snellius/27_eval_semseg_multitask.sh
# =============================================================================
#SBATCH --job-name=eval_seg_mt
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_semseg_multitask_%j.out

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
echo "  Semseg eval — multi-task LoRA (run 26)"
echo "  Weights : coco_rgb_caption_semseg_r8_attention/lora_weights-best.pth"
echo "  Dataset : COCO val2017 (5000 images)"
echo "  Metric  : mIoU (133 COCO classes)"
echo "  Started : $(date)"
echo "================================================================"

python eval_semseg_coco.py \
    --lora_weights /scratch-shared/tesr125338/output/multitask/coco_rgb_caption_semseg_r8_attention/lora_weights-best.pth \
    --lora_rank    8 \
    --lora_modules attention \
    --coco_dir     /scratch-shared/tesr125338/coco \
    --tokens_dir   /scratch-shared/tesr125338/coco_tokens/val \
    --output_dir   /scratch-shared/tesr125338/output/eval_semseg/multitask_caption_semseg_r8 \
    --run_name     multitask_caption_semseg_r8 \
    --hf_cache     "$HF_HOME" \
    --device       cuda \
    --dtype        bfloat16

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Results → output/eval_semseg/multitask_caption_semseg_r8/summary.txt"
echo "================================================================"
