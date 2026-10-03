#!/bin/bash
# =============================================================================
# 13_tokenize_nyu_labeled.sh
# Tokenize NYUv2 LABELED split (795 train + 654 test) from nyu_raw/
#
# Why: the existing nyu_tokens/ was built from 47k raw WebDataset video frames
# (raw Kinect depth). Training on those and validating on the labeled split
# creates a depth quality mismatch. This script tokenizes the correct data:
# post-processed, in-painted depth matching the MultiMAE/4M-21 paper recipe.
#
# Usage:
#   sbatch scripts/snellius/13_tokenize_nyu_labeled.sh
# =============================================================================
#SBATCH --job-name=tok_nyu_labeled
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tokenize_nyu_labeled_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

NYU_RAW=/scratch-shared/tesr125338/nyu_raw
OUTPUT_DIR=/scratch-shared/tesr125338/nyu_tokens_labeled

echo "=== Tokenize NYUv2 labeled split ==="
echo "    Input  : $NYU_RAW  (train: 795, test: 654)"
echo "    Output : $OUTPUT_DIR"

python scripts/prepare_nyu_labeled.py \
    --nyu_raw    "$NYU_RAW" \
    --output_dir "$OUTPUT_DIR" \
    --hf_cache   "$HF_HOME" \
    --batch_size 32 \
    --device     cuda \
    --splits     both

echo "=== Done. Next: sbatch scripts/snellius/14_ft_depth_nyu_labeled_r4_selfattn_drop01.sh ==="
