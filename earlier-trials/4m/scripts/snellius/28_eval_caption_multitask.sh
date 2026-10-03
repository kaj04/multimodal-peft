#!/bin/bash
# =============================================================================
# 28_eval_caption_multitask.sh
# Eval caption quality of multi-task LoRA (run 26) on COCO val2017.
# Reuses eval_caption_coco.py — only the weights path changes vs run 21.
#
# Compare results against:
#   output/eval_caption/coco_r8_attention_drop01/  ← run 23 (caption-only)
#
# Usage:
#   sbatch scripts/snellius/28_eval_caption_multitask.sh
# =============================================================================
#SBATCH --job-name=eval_cap_mt
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_caption_multitask_%j.out

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
echo "  Caption eval — multi-task LoRA (run 26)"
echo "  Weights : coco_rgb_caption_semseg_r8_attention/lora_weights-best.pth"
echo "  Dataset : COCO val2017 (5000 images)"
echo "  Started : $(date)"
echo "================================================================"

python eval_caption_coco.py \
    --lora_weights /scratch-shared/tesr125338/output/multitask/coco_rgb_caption_semseg_r8_attention/lora_weights-best.pth \
    --lora_rank    8 \
    --lora_modules attention \
    --coco_root    /scratch-shared/tesr125338/coco/val2017 \
    --coco_ann     /scratch-shared/tesr125338/coco/annotations/captions_val2017.json \
    --max_images   0 \
    --output_dir   /scratch-shared/tesr125338/output/eval_caption/multitask_caption_semseg_r8 \
    --device       cuda \
    --dtype        bfloat16

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Results → output/eval_caption/multitask_caption_semseg_r8/"
echo "  Compare with run 23: output/eval_caption/coco_r8_attention_drop01/"
echo "================================================================"
