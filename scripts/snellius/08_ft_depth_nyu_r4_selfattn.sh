#!/bin/bash
# =============================================================================
# 08_ft_depth_nyu_r4_selfattn.sh
# LoRA fine-tuning: NYUv2 depth — rank=4, self-attention only (4 x A100)
#
# Motivation: r8/attention run showed overfitting (train CE ↓, val CE flat).
# This run reduces capacity (rank 4 + selfattn only) keeping all else equal,
# for a clean single-variable comparison.
#
# Usage:
#   sbatch scripts/snellius/08_ft_depth_nyu_r4_selfattn.sh
# =============================================================================
#SBATCH --job-name=4m_lora_r4_selfattn
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_depth_r4_selfattn_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== LoRA fine-tuning: NYUv2 depth, rank=4, selfattn only ==="
echo "    Config : cfgs/finetune/depth/nyu_rgb_r4_selfattn.yaml"
echo "    Output : output/depth/nyu_rgb_r4_selfattn/"
echo "    WandB  : lora_depth_nyu_r4_selfattn"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29502 \
    run_finetuning_lora.py \
    --config cfgs/finetune/depth/nyu_rgb_r4_selfattn.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: output/depth/nyu_rgb_r4_selfattn/"
