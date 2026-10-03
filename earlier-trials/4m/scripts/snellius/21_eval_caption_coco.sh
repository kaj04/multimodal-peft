#!/bin/bash
# =============================================================================
# 21_eval_caption_coco.sh
# Official caption quality evaluation on full COCO val2017 (5000 images).
#
# Compares:
#   - Baseline: 4M-21 XL (no LoRA)
#   - Caption LoRA: run 20 (r4/selfattn/drop01, trained on 10k COCO train)
#
# Output: CIDEr / BLEU@1-4 / ROUGE_L / METEOR in metrics.json + summary.txt
#
# Note on dataset sizes:
#   train2017 official: ~118k images → we sampled 10k (seed=42)
#   val2017 official  :   5000 images → always full, never subsampled
#
# Runtime: ~4-5h on 1× A100 (2× 5000 images at ~1.8s/img)
#
# Usage:
#   sbatch scripts/snellius/21_eval_caption_coco.sh
# =============================================================================
#SBATCH --job-name=eval_cap_coco
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_caption_coco_%j.out

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
echo "  4M-21 Caption Quality Eval — COCO val2017 (full, 5000 images)"
echo "  Baseline : 4M-21 XL (no LoRA)"
echo "  LoRA     : run 20 — r4/selfattn/drop01 (COCO 10k train)"
echo "  Output   : /scratch-shared/tesr125338/output/eval_caption/coco_r4_selfattn_drop01"
echo "  Started  : $(date)"
echo "================================================================"

python eval_caption_coco.py \
    --coco_root    /scratch-shared/tesr125338/coco/val2017 \
    --coco_ann     /scratch-shared/tesr125338/coco/annotations/captions_val2017.json \
    --lora_weights /scratch-shared/tesr125338/output/caption/coco_rgb_r4_selfattn_drop01/lora_weights-best.pth \
    --lora_rank    4 \
    --lora_modules selfattn \
    --max_images   0 \
    --output_dir   /scratch-shared/tesr125338/output/eval_caption/coco_r4_selfattn_drop01 \
    --device       cuda \
    --dtype        bfloat16

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Results → /scratch-shared/tesr125338/output/eval_caption/coco_r4_selfattn_drop01/"
echo "    metrics.json   ← CIDEr / BLEU / ROUGE / METEOR"
echo "    summary.txt    ← human-readable table"
echo "================================================================"
