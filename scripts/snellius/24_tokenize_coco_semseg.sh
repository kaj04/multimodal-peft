#!/bin/bash
# =============================================================================
# 24_tokenize_coco_semseg.sh
# Tokenize COCO 2017 panoptic segmentation → tok_semseg VQ codes.
#
# Reads PNG masks directly from panoptic_*.zip (no GPFS extraction).
# Only processes the seed=42 10k train subset (same images as tok_rgb/caption).
#
# Output:
#   /scratch-shared/tesr125338/coco_tokens/train/tok_semseg/  (10k .npy files)
#   /scratch-shared/tesr125338/coco_tokens/val/tok_semseg/    (5k .npy files)
#
# Runtime: ~30-60 min on 1× A100
#
# Usage:
#   sbatch scripts/snellius/24_tokenize_coco_semseg.sh
# =============================================================================
#SBATCH --job-name=tok_coco_semseg
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tokenize_coco_semseg_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs
cd "$REPO_ROOT"

COCO_DIR=/scratch-shared/tesr125338/coco
TOKENS_DIR=/scratch-shared/tesr125338/coco_tokens

echo "================================================================"
echo "  COCO semseg tokenisation"
echo "  Source  : $COCO_DIR/annotations/panoptic_*.zip"
echo "  Output  : $TOKENS_DIR/{train,val}/tok_semseg/"
echo "  Started : $(date)"
echo "================================================================"

# Verify panoptic zips exist
for f in "$COCO_DIR/annotations/panoptic_train2017.zip" \
         "$COCO_DIR/annotations/panoptic_val2017.zip" \
         "$COCO_DIR/annotations/panoptic_train2017.json" \
         "$COCO_DIR/annotations/panoptic_val2017.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        echo "Run: sbatch scripts/snellius/06b_download_coco_panoptic.sh"
        exit 1
    fi
done

echo "  Panoptic files found ✓"
echo ""

python scripts/prepare_coco_semseg.py \
    --coco_dir   "$COCO_DIR" \
    --tokens_dir "$TOKENS_DIR" \
    --hf_cache   "$HF_HOME" \
    --batch_size 32 \
    --device     cuda \
    --splits     both

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Token check:"
echo "    train tok_semseg: $(ls $TOKENS_DIR/train/tok_semseg/ 2>/dev/null | wc -l) files (expected ~10000)"
echo "    val   tok_semseg: $(ls $TOKENS_DIR/val/tok_semseg/   2>/dev/null | wc -l) files (expected 5000)"
echo "  Next: sbatch scripts/snellius/26_ft_multitask_coco_r8_attention.sh"
echo "================================================================"
