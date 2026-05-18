#!/bin/bash
# =============================================================================
# 38_ft_normal_nyu_r8_attention.sh
# LoRA fine-tuning: NYUv2 surface normals, rank=8, attention, dropout=0.1
#
# Run 38: normal-only baseline for comparison with run 34 (5-task multitask).
# First normal-estimation LoRA run. Tokenized data from run 33
# (33_tokenize_nyu_normals.sh, stored in nyu_tokens_labeled/*/tok_normal/).
#
# Usage:
#   sbatch scripts/snellius/38_ft_normal_nyu_r8_attention.sh
# =============================================================================
#SBATCH --job-name=ft_normal_r8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ft_normal_nyu_r8_attention_%j.out

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
echo "  Normal-only LoRA fine-tuning (run 38)"
echo "  Task   : normal (NYUv2 labeled, 795 train / 654 val)"
echo "  LoRA   : rank=8, attention, dropout=0.1"
echo "  Output : /scratch-shared/tesr125338/output/normal/nyu_labeled_r8_attention"
echo "  Started: $(date)"
echo "================================================================"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=29509 \
    run_finetuning_lora.py \
    --config cfgs/finetune/normal/nyu_labeled_r8_attention.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Weights → /scratch-shared/tesr125338/output/normal/nyu_labeled_r8_attention/"
echo "    lora_weights-best.pth  ← use for eval_all.py"
echo "================================================================"
