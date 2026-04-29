#!/bin/bash
# =============================================================================
# 01_prepare_nyu.sh  —  Download NYUv2 + tokenize for 4M-PEFT
#
# Runs scripts/prepare_nyu.py which:
#   1. Downloads tokenizer .pth checkpoints from HuggingFace Hub
#   2. Downloads NYUv2 RGB+depth from HuggingFace (sayakpaul/nyu_depth_v2)
#   3. Tokenizes both modalities and saves .npy files to scratch
#
# Note: only tok_rgb + tok_depth are prepared here (sufficient for the
# depth-estimation fine-tuning baseline). Semseg will be added separately.
#
# Usage:
#   sbatch scripts/snellius/01_prepare_nyu.sh
# =============================================================================
#SBATCH --job-name=prepare_nyu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/prepare_nyu_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
source activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== Preparing NYUv2 tokens for 4M-PEFT ==="
echo "Output dir:      /scratch-shared/tesr125338/nyu_tokens"
echo "Tokenizers dir:  /scratch-shared/tesr125338/tokenizer_ckpts"
echo "HF cache:        $HF_HOME"
echo ""

# Dry-run first (8 images) to validate the pipeline before full tokenization
echo "--- Dry run (8 images) ---"
python scripts/prepare_nyu.py \
    --output_dir /scratch-shared/tesr125338/nyu_tokens_dryrun \
    --hf_cache   "$HF_HOME" \
    --batch_size 8 \
    --dry_run

echo ""
echo "--- Dry run passed. Running full tokenization (795 + 654 images) ---"

python scripts/prepare_nyu.py \
    --output_dir /scratch-shared/tesr125338/nyu_tokens \
    --hf_cache   "$HF_HOME" \
    --batch_size 32

echo ""
echo "=== Tokenization complete ==="
echo "Files ready at: /scratch-shared/tesr125338/nyu_tokens/"
echo ""
echo "Next step: sbatch scripts/snellius/04_ft_depth_nyu.sh"
