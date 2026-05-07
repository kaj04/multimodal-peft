#!/bin/bash
# =============================================================================
# 09_ft_depth_nyu_r2_selfattn.sh
# LoRA fine-tuning: NYUv2 depth — rank=2, self-attention only (4 x A100)
#
# Motivation: r4/selfattn run showed val loss increasing from epoch 0 → 1,
# suggesting overfitting even with reduced capacity. This run applies more
# aggressive regularisation:
#   - rank 2        (was 4)
#   - blr 1e-5      (was 1e-4)
#   - weight_decay 0.1 (was 0.05)
#   - warmup_epochs 10 (was 5)
#   - early_stop_patience 3
#
# Usage:
#   sbatch scripts/snellius/09_ft_depth_nyu_r2_selfattn.sh
# =============================================================================
#SBATCH --job-name=4m_lora_r2_selfattn
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_depth_r2_selfattn_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== LoRA fine-tuning: NYUv2 depth, rank=2, selfattn only ==="
echo "    Config : cfgs/finetune/depth/nyu_rgb_r2_selfattn.yaml"
echo "    Output : output/depth/nyu_rgb_r2_selfattn/"
echo "    WandB  : lora_depth_nyu_r2_selfattn"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29503 \
    run_finetuning_lora.py \
    --config cfgs/finetune/depth/nyu_rgb_r2_selfattn.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: output/depth/nyu_rgb_r2_selfattn/"
