#!/bin/bash
# =============================================================================
# 03_ft_semseg_nyu.sh  —  LoRA fine-tuning: NYUv2 semseg (4 x A100)
#
# Reproduces the 4M-21 paper setup (rank=8, attention, XL model).
# Run AFTER 01_tokenize_nyu.sh has completed.
#
# Usage:
#   sbatch scripts/snellius/03_ft_semseg_nyu.sh
# =============================================================================
#SBATCH --job-name=4m_lora_semseg_nyu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ft_semseg_nyu_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
source activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== Starting LoRA fine-tuning: NYUv2 semseg, rank=8, XL ==="
echo "Config: cfgs/finetune/semseg/nyu_rgb_r8.yaml"
echo "GPUs: $SLURM_NTASKS"
echo "Job ID: $SLURM_JOB_ID"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    run_finetuning_lora.py \
    --config cfgs/finetune/semseg/nyu_rgb_r8.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: output/semseg/nyu_rgb_r8/"
