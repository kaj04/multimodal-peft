#!/bin/bash
# =============================================================================
# 34_ft_5task_r8_attention.sh
# Multi-task LoRA fine-tuning: COCO (caption+semseg+det) + NYUv2 (depth+normal)
#
# Run 34: 5 tasks, rank=8, attention modules, dropout=0.1
# Sampling: proportional random shuffle (not round-robin)
# Early stopping: mean val loss across 5 tasks, patience=15
#
# Usage:
#   sbatch scripts/snellius/34_ft_5task_r8_attention.sh
# =============================================================================
#SBATCH --job-name=ft_5task
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_5task_r8_attention_%j.out

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
echo "  5-task LoRA fine-tuning (run 34)"
echo "  Tasks  : caption + semseg + det (COCO) + depth + normal (NYUv2)"
echo "  LoRA   : rank=8, attention, dropout=0.1"
echo "  Output : /scratch-shared/tesr125338/output/multitask/coco_nyu_5task_r8_attention"
echo "  Started: $(date)"
echo "================================================================"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    run_finetuning_lora.py \
    --config cfgs/finetune/multitask/coco_nyu_5task_r8_attention.yaml

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Weights → /scratch-shared/tesr125338/output/multitask/coco_nyu_5task_r8_attention/"
echo "    lora_weights-best.pth  ← use this for eval"
echo "================================================================"
