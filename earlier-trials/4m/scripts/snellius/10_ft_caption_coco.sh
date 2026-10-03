#!/bin/bash
# =============================================================================
# 10_ft_caption_coco.sh
# LoRA fine-tuning: COCO captioning — rank=8, attention (cross-task baseline)
#
# Config mirrors depth r8/attention for a clean cross-task comparison:
# same LoRA rank, same LR, same effective batch size.
# Task: tok_rgb@224 → caption (wordpiece, 30k vocab, 64 tokens)
#
# Prerequisites:
#   - coco_tokens/{train,val}/{tok_rgb,caption}/ must exist
#     (run sbatch scripts/snellius/09_tokenize_coco.sh first)
#
# Usage:
#   sbatch scripts/snellius/10_ft_caption_coco.sh
# =============================================================================
#SBATCH --job-name=4m_caption_r8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_caption_r8_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

echo "=== LoRA fine-tuning: COCO captioning, rank=8, attention ==="
echo "    Config : cfgs/finetune/caption/coco_rgb_r8_attention.yaml"
echo "    Output : output/caption/coco_rgb_r8_attention/"
echo "    WandB  : lora_caption_coco_r8_attention"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29503 \
    run_finetuning_lora.py \
    --config cfgs/finetune/caption/coco_rgb_r8_attention.yaml \
    --dist_eval

echo "Training complete. Checkpoints in: output/caption/coco_rgb_r8_attention/"
echo "Next step: sbatch scripts/snellius/11_eval_caption.sh"
