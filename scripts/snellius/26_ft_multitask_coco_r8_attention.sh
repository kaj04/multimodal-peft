#!/bin/bash
# =============================================================================
# 26_ft_multitask_coco_r8_attention.sh
# Multi-task LoRA FT: COCO caption + semseg simultaneously
#
# Run 26: single LoRA (rank=8, attention, dropout=0.1) trained on batches
# alternating between caption and semseg tasks. Same LoRA weights updated
# by both tasks → tests whether joint training improves captioning vs
# caption-only run 23.
#
# Requires:
#   - coco_tokens/train/tok_rgb/     (from run 09)
#   - coco_tokens/train/caption/     (from run 09)
#   - coco_tokens/train/tok_semseg/  (from run 24)  ← must run first!
#
# Usage:
#   sbatch scripts/snellius/26_ft_multitask_coco_r8_attention.sh
# =============================================================================
#SBATCH --job-name=4m_mt_cap_seg
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_multitask_caption_semseg_%j.out

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
echo "  4M-21 Multi-task LoRA FT — Caption + Semseg (run 26)"
echo "  Config : cfgs/finetune/multitask/coco_rgb_caption_semseg_r8_attention.yaml"
echo "  Output : /scratch-shared/tesr125338/output/multitask/coco_rgb_caption_semseg_r8_attention"
echo "  Train  : 10k images (caption + semseg, same seed=42 subset)"
echo "  Val    : 5k images (COCO val2017)"
echo "  WandB  : lora_multitask_caption_semseg_r8_attention @ 4M-PEFT"
echo "  Started: $(date)"
echo "================================================================"

# Pre-flight: check all three modalities are tokenized
TOKENS_TRAIN=/scratch-shared/tesr125338/coco_tokens/train

n_rgb=$(ls    "${TOKENS_TRAIN}/tok_rgb/"   2>/dev/null | wc -l)
n_cap=$(ls    "${TOKENS_TRAIN}/caption/"   2>/dev/null | wc -l)
n_seg=$(ls    "${TOKENS_TRAIN}/tok_semseg/" 2>/dev/null | wc -l)

echo "  Token check (train):"
echo "    tok_rgb    : ${n_rgb}  (expected ~10000)"
echo "    caption    : ${n_cap}  (expected ~10000)"
echo "    tok_semseg : ${n_seg}  (expected ~10000)"

if [ "${n_rgb}" -lt 100 ] || [ "${n_cap}" -lt 100 ]; then
    echo "ERROR: tok_rgb or caption tokens missing — run 09_tokenize_coco.sh first"
    exit 1
fi
if [ "${n_seg}" -lt 100 ]; then
    echo "ERROR: tok_semseg tokens missing — run 24_tokenize_coco_semseg.sh first"
    exit 1
fi

echo ""

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29510 \
    run_finetuning_lora.py \
    --config cfgs/finetune/multitask/coco_rgb_caption_semseg_r8_attention.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Checkpoint → /scratch-shared/tesr125338/output/multitask/coco_rgb_caption_semseg_r8_attention/"
echo "    lora_weights-best.pth  ← best mean(val_caption, val_semseg)"
echo "  Next eval:"
echo "    sbatch scripts/snellius/28_eval_caption_multitask.sh"
echo "    sbatch scripts/snellius/27_eval_semseg_multitask.sh"
echo "================================================================"
