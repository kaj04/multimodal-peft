#!/bin/bash
# =============================================================================
# 14_ft_depth_nyu_labeled_r4_selfattn_drop01.sh
# LoRA fine-tuning: NYUv2 depth — labeled split, rank=4, selfattn, dropout=0.1
#
# Uses nyu_tokens_labeled/ (795 train / 654 val, post-processed depth).
# This matches the MultiMAE / 4M-21 paper recipe and eliminates the
# train/val depth quality mismatch present in nyu_tokens/ (raw video frames).
#
# Prerequisites:
#   sbatch scripts/snellius/13_tokenize_nyu_labeled.sh  (must complete first)
#
# Usage:
#   sbatch scripts/snellius/14_ft_depth_nyu_labeled_r4_selfattn_drop01.sh
# =============================================================================
#SBATCH --job-name=4m_labeled_r4
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_depth_labeled_r4_selfattn_drop01_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== LoRA fine-tuning: NYUv2 labeled, rank=4, selfattn, dropout=0.1 ==="
echo "    Config : cfgs/finetune/depth/nyu_labeled_r4_selfattn_drop01.yaml"
echo "    Data   : nyu_tokens_labeled/ (795 train / 654 val, post-processed depth)"
echo "    Output : output/depth/nyu_labeled_r4_selfattn_drop01/"
echo "    WandB  : lora_depth_nyu_labeled_r4_selfattn_drop01"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29505 \
    run_finetuning_lora.py \
    --config cfgs/finetune/depth/nyu_labeled_r4_selfattn_drop01.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: output/depth/nyu_labeled_r4_selfattn_drop01/"
