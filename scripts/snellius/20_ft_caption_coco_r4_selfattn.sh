#!/bin/bash
# =============================================================================
# 20_ft_caption_coco_r4_selfattn.sh
# LoRA fine-tuning: COCO captioning — rank=4, selfattn, dropout=0.1, blr=3e-5
#
# Follows run 19 (r8/drop005) with reduced rank and stronger dropout to
# address overfitting on the 10k-image train subset. eval_freq=1 gives
# per-epoch val loss for fine-grained early stopping (patience=5 epochs).
#
# Train subset: 10k images from COCO train2017 (seed=42, fixed).
# Val:          full COCO val2017 (5000 images).
#
# Prerequisites — tokenized data must exist (run 09 first if not):
#   sbatch scripts/snellius/09_tokenize_coco.sh
#   sbatch scripts/snellius/20_ft_caption_coco_r4_selfattn.sh
#
# Usage:
#   sbatch scripts/snellius/20_ft_caption_coco_r4_selfattn.sh
# =============================================================================
#SBATCH --job-name=4m_cap_r4_sa
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_caption_r4_selfattn_%j.out

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
echo "  4M-21 LoRA FT — COCO Captioning, r4/selfattn/drop01"
echo "  Config : cfgs/finetune/caption/coco_rgb_r4_selfattn.yaml"
echo "  Output : /scratch-shared/tesr125338/output/caption/coco_rgb_r4_selfattn_drop01"
echo "  Train  : 10k images (seed=42 subset of COCO train2017)"
echo "  Val    : 5k images (full COCO val2017)"
echo "  WandB  : lora_caption_coco_r4_selfattn_drop01 @ 4M-PEFT"
echo "  Started: $(date)"
echo "================================================================"

# Verify tokenized data exists before launching
TOKENS_TRAIN=/scratch-shared/tesr125338/coco_tokens/train
TOKENS_VAL=/scratch-shared/tesr125338/coco_tokens/val

n_rgb_train=$(ls "${TOKENS_TRAIN}/tok_rgb/"  2>/dev/null | wc -l)
n_cap_train=$(ls "${TOKENS_TRAIN}/caption/"  2>/dev/null | wc -l)
n_rgb_val=$(ls   "${TOKENS_VAL}/tok_rgb/"    2>/dev/null | wc -l)
n_cap_val=$(ls   "${TOKENS_VAL}/caption/"    2>/dev/null | wc -l)

echo "  Token check:"
echo "    train tok_rgb : ${n_rgb_train}  (expected ~10000)"
echo "    train caption : ${n_cap_train}  (expected ~10000)"
echo "    val   tok_rgb : ${n_rgb_val}    (expected 5000)"
echo "    val   caption : ${n_cap_val}    (expected 5000)"

if [ "${n_rgb_train}" -lt 100 ] || [ "${n_cap_train}" -lt 100 ]; then
    echo "ERROR: tokenized data missing — run 09_tokenize_coco.sh first"
    exit 1
fi

echo ""

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29508 \
    run_finetuning_lora.py \
    --config cfgs/finetune/caption/coco_rgb_r4_selfattn.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Checkpoints → /scratch-shared/tesr125338/output/caption/coco_rgb_r4_selfattn_drop01/"
echo "    lora_weights-best.pth     ← best val loss (use this for eval)"
echo "    lora_weights-epochN.pth   ← final epoch"
echo "    log.txt                   ← per-epoch train/val loss"
echo "    config.yaml               ← exact config"
echo "================================================================"
