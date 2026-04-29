#!/bin/bash
# =============================================================================
# 01_tokenize_nyu.sh  —  Pre-tokenize NYUv2 dataset on Snellius
#
# Converts raw NYUv2 RGB images and depth maps into discrete token sequences
# using the pretrained 4M VQ-VAE tokenizers. Run once before fine-tuning.
#
# Prereq: NYUv2 downloaded to $NYU_RAW_DIR with structure:
#   $NYU_RAW_DIR/train/rgb/   (*.png)
#   $NYU_RAW_DIR/train/depth/ (*.png, uint16, metric depth * 1000)
#   $NYU_RAW_DIR/val/rgb/
#   $NYU_RAW_DIR/val/depth/
#
# Usage:
#   sbatch scripts/snellius/01_tokenize_nyu.sh
# =============================================================================
#SBATCH --job-name=tokenize_nyu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tokenize_nyu_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
source activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"   # adjust if cloned elsewhere

NYU_RAW_DIR="/scratch-shared/tesr125338/nyu_raw"
NYU_TOK_DIR="/scratch-shared/tesr125338/nyu_tokens"

mkdir -p logs "$NYU_TOK_DIR"

cd "$REPO_ROOT/ml-4m"

for SPLIT in train val; do
    echo "=== Tokenizing $SPLIT split ==="

    # RGB tokens  (DiVAE, 16k vocab, 196 tokens @ 224px)
    python save_vq_tokens.py \
        --root "$NYU_RAW_DIR/$SPLIT/rgb" \
        --output_dir "$NYU_TOK_DIR/$SPLIT/tok_rgb" \
        --tokenizer EPFL-VILAB/4M_tokenizers_rgb_16k_224-448 \
        --task tok_rgb \
        --input_size 224 \
        --n_crops 1

    # Depth tokens  (DiVAE, 8k vocab, 196 tokens @ 224px)
    python save_vq_tokens.py \
        --root "$NYU_RAW_DIR/$SPLIT/depth" \
        --output_dir "$NYU_TOK_DIR/$SPLIT/tok_depth" \
        --tokenizer EPFL-VILAB/4M_tokenizers_depth_8k_224-448 \
        --task tok_depth \
        --input_size 224 \
        --n_crops 1

done

echo "Tokenization complete. Output: $NYU_TOK_DIR"
echo "Expected structure:"
echo "  $NYU_TOK_DIR/train/tok_rgb/  (*.npy, shape 196)"
echo "  $NYU_TOK_DIR/train/tok_depth/(*.npy, shape 196)"
echo "  $NYU_TOK_DIR/val/tok_rgb/"
echo "  $NYU_TOK_DIR/val/tok_depth/"
