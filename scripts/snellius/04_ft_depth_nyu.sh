#!/bin/bash
# =============================================================================
# 04_ft_depth_nyu.sh  —  LoRA fine-tuning: NYUv2 depth estimation (4 x A100)
#
# Usage:
#   sbatch scripts/snellius/04_ft_depth_nyu.sh
# =============================================================================
#SBATCH --job-name=4m_lora_depth_nyu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_depth_nyu_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== Starting LoRA fine-tuning: NYUv2 depth, rank=8, XL ==="

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    run_finetuning_lora.py \
    --config cfgs/finetune/depth/nyu_rgb_r8.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: output/depth/nyu_rgb_r8/"
