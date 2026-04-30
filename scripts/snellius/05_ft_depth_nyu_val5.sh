#!/bin/bash
# =============================================================================
# 05_ft_depth_nyu_val5.sh  —  LoRA validation run: NYUv2 depth, 5 epochs
#
# Purpose: sanity-check that LoRA training on 4M-21_XL works end-to-end.
# Expected outcome:
#   - Loss decreases each epoch (model is learning)
#   - No NaN, OOM, or crash
#   - Completes within 3-4h on 4×A100
#
# Run AFTER 01_prepare_nyu.sh has completed (nyu_tokens/ must exist).
#
# Usage:
#   sbatch scripts/snellius/05_ft_depth_nyu_val5.sh
# =============================================================================
#SBATCH --job-name=4m_lora_val5
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/ft_depth_nyu_val5_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

# Ensure wandb is installed
pip install -q wandb

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== LoRA validation run: NYUv2 depth, 5 epochs, XL ==="
echo "Config: cfgs/finetune/depth/nyu_rgb_r8_e5.yaml"
echo "GPUs: $SLURM_NTASKS  |  Job ID: $SLURM_JOB_ID"
echo ""
echo "What to look for in the log:"
echo "  - train_loss should decrease each epoch"
echo "  - val_loss should decrease (or at least not diverge)"
echo "  - No NaN in loss"
echo "  - No OOM errors"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    run_finetuning_lora.py \
    --config cfgs/finetune/depth/nyu_rgb_r8_e5.yaml \
    --dist_eval

echo ""
echo "=== Validation run complete ==="
echo "Checkpoints in: output/depth/nyu_rgb_r8_e5/"
echo ""
echo "Check results:"
echo "  cat logs/ft_depth_nyu_val5_${SLURM_JOB_ID}.out | grep -E 'epoch|loss|lr'"
echo ""
echo "If loss decreased over 5 epochs → launch full 50-epoch run:"
echo "  sbatch scripts/snellius/04_ft_depth_nyu.sh"
