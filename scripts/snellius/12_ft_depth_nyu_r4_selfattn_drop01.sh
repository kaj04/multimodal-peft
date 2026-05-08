#!/bin/bash
# =============================================================================
# 12_ft_depth_nyu_r4_selfattn_drop01.sh
# LoRA fine-tuning: NYUv2 depth — rank=4, self-attention only, dropout=0.1
#
# Motivation: r4/selfattn run overfits (train CE ↓, val CE ↑).
# Single change: add lora_dropout=0.1 to regularize the adapter weights.
# Early stopping patience=5 to avoid stopping during LR warmup.
#
# Usage:
#   sbatch scripts/snellius/12_ft_depth_nyu_r4_selfattn_drop01.sh
# =============================================================================
#SBATCH --job-name=4m_lora_drop01
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_depth_r4_selfattn_drop01_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== LoRA fine-tuning: NYUv2 depth, rank=4, selfattn, dropout=0.1 ==="
echo "    Config : cfgs/finetune/depth/nyu_rgb_r4_selfattn_drop01.yaml"
echo "    Output : output/depth/nyu_rgb_r4_selfattn_drop01/"
echo "    WandB  : lora_depth_nyu_r4_selfattn_drop01"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29504 \
    run_finetuning_lora.py \
    --config cfgs/finetune/depth/nyu_rgb_r4_selfattn_drop01.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: output/depth/nyu_rgb_r4_selfattn_drop01/"
