#!/bin/bash
# =============================================================================
# 29_eval_caption_r8_attention.sh
# Caption quality evaluation for run 23 (r8/attention, caption-only LoRA).
#
# This is the fair baseline to compare against run 26 multitask:
#   same rank (8), same modules (attention), different training (caption-only)
#
# Compare results against:
#   output/eval_caption/multitask_caption_semseg_r8/  <- run 26 (multitask)
#
# Usage:
#   sbatch scripts/snellius/29_eval_caption_r8_attention.sh
# =============================================================================
#SBATCH --job-name=eval_cap_r8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_caption_r8_attention_%j.out

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
echo "  Caption eval -- run 23 (r8/attention, caption-only LoRA)"
echo "  Weights : coco_rgb_r8_attention_drop01/lora_weights-best.pth"
echo "  Dataset : COCO val2017 (5000 images)"
echo "  Started : $(date)"
echo "================================================================"

python eval_caption_coco.py \
    --lora_weights /scratch-shared/tesr125338/output/caption/coco_rgb_r8_attention_drop01/lora_weights-best.pth \
    --lora_rank    8 \
    --lora_modules attention \
    --coco_root    /scratch-shared/tesr125338/coco/val2017 \
    --coco_ann     /scratch-shared/tesr125338/coco/annotations/captions_val2017.json \
    --max_images   0 \
    --output_dir   /scratch-shared/tesr125338/output/eval_caption/coco_r8_attention_drop01 \
    --device       cuda \
    --dtype        bfloat16

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Results --> output/eval_caption/coco_r8_attention_drop01/"
echo "  Compare with run 26: output/eval_caption/multitask_caption_semseg_r8/"
echo "================================================================"
