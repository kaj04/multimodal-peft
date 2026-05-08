#!/bin/bash
# =============================================================================
# 15_ft_depth_nyu_labeled_r8_selfattn_drop005.sh
# LoRA fine-tuning: NYUv2 labeled, rank=8, selfattn, dropout=0.05, blr=5e-5
#
# Changes vs run 14 (r4/drop01):
#   - rank 8 (paper standard for 4M-21 XL)
#   - dropout 0.05 (less restrictive)
#   - blr 5e-5 (lower LR to stabilize oscillating grad norm)
#
# Usage:
#   sbatch scripts/snellius/15_ft_depth_nyu_labeled_r8_selfattn_drop005.sh
# =============================================================================
#SBATCH --job-name=4m_lab_r8_d005
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_depth_labeled_r8_selfattn_drop005_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== LoRA fine-tuning: NYUv2 labeled, rank=8, selfattn, dropout=0.05, blr=5e-5 ==="
echo "    Config : cfgs/finetune/depth/nyu_labeled_r8_selfattn_drop005.yaml"
echo "    Output : /scratch-shared/tesr125338/output/depth/nyu_labeled_r8_selfattn_drop005/"
echo "    WandB  : lora_depth_nyu_labeled_r8_selfattn_drop005"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29506 \
    run_finetuning_lora.py \
    --config cfgs/finetune/depth/nyu_labeled_r8_selfattn_drop005.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: /scratch-shared/tesr125338/output/depth/nyu_labeled_r8_selfattn_drop005/"
