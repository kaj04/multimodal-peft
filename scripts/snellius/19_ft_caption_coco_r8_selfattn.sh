#!/bin/bash
# =============================================================================
# 19_ft_caption_coco_r8_selfattn.sh
# LoRA fine-tuning: COCO captioning — rank=8, selfattn, dropout=0.05
#
# Mirrors depth run 15 (r8/selfattn/drop005) for clean cross-task comparison:
#   - same LoRA config  : rank=8, selfattn, dropout=0.05
#   - same LR           : blr=5e-5
#   - same effective bs : 4 GPUs × bs=2 × accum=4 = 32
#   - same epochs / schedule / early stopping
#
# Train subset: 10k images from COCO train2017 (seed=42, fixed).
# Val:          full COCO val2017 (5000 images).
#
# Prerequisites — run in this order:
#   sbatch scripts/snellius/09_tokenize_coco.sh   (tokenise, ~2h)
#   sbatch scripts/snellius/19_ft_caption_coco_r8_selfattn.sh
#
# Usage:
#   sbatch scripts/snellius/19_ft_caption_coco_r8_selfattn.sh
# =============================================================================
#SBATCH --job-name=4m_cap_r8_sa
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_caption_r8_selfattn_%j.out

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
echo "  4M-21 LoRA FT — COCO Captioning, r8/selfattn/drop005"
echo "  Config : cfgs/finetune/caption/coco_rgb_r8_selfattn.yaml"
echo "  Output : /scratch-shared/tesr125338/output/caption/coco_rgb_r8_selfattn_drop005"
echo "  Train  : 10k images (seed=42 subset of COCO train2017)"
echo "  Val    : 5k images (full COCO val2017)"
echo "  WandB  : lora_caption_coco_r8_selfattn_drop005 @ 4M-PEFT"
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
    --master_port=29507 \
    run_finetuning_lora.py \
    --config cfgs/finetune/caption/coco_rgb_r8_selfattn.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Checkpoints → /scratch-shared/tesr125338/output/caption/coco_rgb_r8_selfattn_drop005/"
echo "    lora_weights-best.pth     ← best val loss (use this for eval)"
echo "    lora_weights-epochN.pth   ← final epoch"
echo "    log.txt                   ← per-epoch train/val loss"
echo "    config.yaml               ← exact config"
echo "================================================================"
