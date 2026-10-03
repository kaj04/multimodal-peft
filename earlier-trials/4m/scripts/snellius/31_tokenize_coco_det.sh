#!/bin/bash
# =============================================================================
# 31_tokenize_coco_det.sh
# Tokenize COCO detection annotations → det tokens (text sequence).
# CPU-only, ~5-10 min for 10k train + 5k val.
#
# Usage:
#   sbatch scripts/snellius/31_tokenize_coco_det.sh
# =============================================================================
#SBATCH --job-name=tok_det
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tokenize_coco_det_%j.out

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
echo "  COCO detection tokenization → det tokens"
echo "  Input  : /scratch-shared/tesr125338/coco/annotations/instances_*.json"
echo "  Output : /scratch-shared/tesr125338/coco_tokens/{train,val}/det/"
echo "  Started: $(date)"
echo "================================================================"

# Dry run first to verify format
echo "--- Dry run (8 images) ---"
python scripts/prepare_coco_det.py \
    --coco_dir   /scratch-shared/tesr125338/coco \
    --tokens_dir /scratch-shared/tesr125338/coco_tokens \
    --splits both \
    --dry_run

echo ""
echo "--- Full run ---"
python scripts/prepare_coco_det.py \
    --coco_dir   /scratch-shared/tesr125338/coco \
    --tokens_dir /scratch-shared/tesr125338/coco_tokens \
    --splits both

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  train/det/ and val/det/ ready"
echo "================================================================"
