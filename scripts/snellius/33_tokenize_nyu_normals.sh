#!/bin/bash
# =============================================================================
# 33_tokenize_nyu_normals.sh
# Compute surface normals from NYUv2 depth maps and tokenize them.
# GPU needed for VQVAE encoder. ~10 min for 795 train + 654 val.
#
# Usage:
#   sbatch scripts/snellius/33_tokenize_nyu_normals.sh
# =============================================================================
#SBATCH --job-name=tok_normals
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tokenize_nyu_normals_%j.out

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
echo "  NYUv2 surface normals tokenization"
echo "  Input  : /scratch-shared/tesr125338/nyu_raw/{train,test}/depth/"
echo "  Output : /scratch-shared/tesr125338/nyu_tokens_labeled/{train,val}/tok_normal/"
echo "  Started: $(date)"
echo "================================================================"

echo "--- Dry run (8 images) ---"
python scripts/prepare_nyu_normals.py \
    --nyu_raw    /scratch-shared/tesr125338/nyu_raw \
    --output_dir /scratch-shared/tesr125338/nyu_tokens_labeled \
    --hf_cache   /scratch-shared/tesr125338/hf_cache \
    --splits both \
    --dry_run

echo ""
echo "--- Full run ---"
python scripts/prepare_nyu_normals.py \
    --nyu_raw    /scratch-shared/tesr125338/nyu_raw \
    --output_dir /scratch-shared/tesr125338/nyu_tokens_labeled \
    --hf_cache   /scratch-shared/tesr125338/hf_cache \
    --splits both

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  tok_normal/ ready alongside tok_rgb/ and tok_depth/"
echo "================================================================"
