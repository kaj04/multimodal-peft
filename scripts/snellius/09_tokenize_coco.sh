#!/bin/bash
# =============================================================================
# 09_tokenize_coco.sh
# Tokenize COCO 2017 for captioning experiment (Phase 1: RGB on GPU, Phase 2: captions on CPU)
#
# Usage:
#   sbatch scripts/snellius/09_tokenize_coco.sh
# =============================================================================
#SBATCH --job-name=tokenize_coco
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tokenize_coco_%j.out

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
OUTPUT_DIR=/scratch-shared/tesr125338/coco_tokens

echo "=== COCO tokenization ==="
echo "    Input  : $COCO_DIR"
echo "    Output : $OUTPUT_DIR"

python scripts/prepare_coco.py \
    --coco_dir   "$COCO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --hf_cache   "$HF_HOME" \
    --batch_size 64 \
    --device     cuda \
    --splits     both \
    --phase      both

echo "=== Done. Next step: sbatch scripts/snellius/10_ft_caption_coco.sh ==="
