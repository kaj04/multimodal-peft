#!/bin/bash
# =============================================================================
# 40_ft_5task_r8_attention_v2.sh
# Multi-task LoRA fine-tuning: COCO (caption+semseg+det) + NYUv2 (depth+normal)
#
# Run 40 (v2): FIXED vs run 34 — padding tokens excluded from loss.
# Identical LoRA config to run 34; only fix is mask_padding=True in
# build_mod_dict and patience raised 15→25.
#
# Usage:
#   sbatch scripts/snellius/40_ft_5task_r8_attention_v2.sh
# =============================================================================
#SBATCH --job-name=ft_5task_v2
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_5task_r8_attention_v2_%j.out

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
echo "  5-task LoRA fine-tuning (run 40 — padding fix)"
echo "  Tasks  : caption + semseg + det (COCO) + depth + normal (NYUv2)"
echo "  LoRA   : rank=8, attention, dropout=0.1"
echo "  Fix    : mask_padding=True (excludes pad tokens from caption+det loss)"
echo "  Output : /scratch-shared/tesr125338/output/multitask/coco_nyu_5task_r8_attention_v2"
echo "  Started: $(date)"
echo "================================================================"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=29511 \
    run_finetuning_lora.py \
    --config cfgs/finetune/multitask/coco_nyu_5task_r8_attention_v2.yaml

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Weights → /scratch-shared/tesr125338/output/multitask/coco_nyu_5task_r8_attention_v2/"
echo "    lora_weights-best.pth  ← use for eval_all.py"
echo "================================================================"
