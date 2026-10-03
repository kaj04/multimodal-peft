#!/bin/bash
# =============================================================================
# 23_ft_caption_coco_r8_attention.sh
# LoRA fine-tuning: COCO captioning — rank=8, attention (selfattn+crossattn), dropout=0.1
#
# Follows run 20 (r4/selfattn/drop01) which showed minimal improvement.
# Hypothesis: cross-attention is the key visual→text channel for captioning;
# adding crossattn LoRA and increasing rank should improve performance.
#
# Changes vs run 20:
#   lora_rank    : 4 → 8
#   lora_modules : selfattn → attention (selfattn + crossattn)
#   patience     : 5 → 10  (more time to find better optimum)
#   (dropout=0.1, blr=3e-5, eval_freq=1 unchanged)
#
# Train: 10k images from COCO train2017 (seed=42, same tokenized data as runs 19/20).
# Val:   full COCO val2017 (5000 images).
#
# Usage:
#   sbatch scripts/snellius/23_ft_caption_coco_r8_attention.sh
# =============================================================================
#SBATCH --job-name=4m_cap_r8_attn
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_caption_r8_attention_%j.out

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
echo "  4M-21 LoRA FT -- COCO Captioning, r8/attention/drop01"
echo "  Config : cfgs/finetune/caption/coco_rgb_r8_attention.yaml"
echo "  Output : /scratch-shared/tesr125338/output/caption/coco_rgb_r8_attention_drop01"
echo "  Train  : 10k images (seed=42 subset of COCO train2017)"
echo "  Val    : 5k images (full COCO val2017)"
echo "  WandB  : lora_caption_coco_r8_attention_drop01 @ 4M-PEFT"
echo "  Started: $(date)"
echo "================================================================"

# Verify tokenized data exists
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
    echo "ERROR: tokenized data missing -- run 09_tokenize_coco.sh first"
    exit 1
fi

echo ""

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29509 \
    run_finetuning_lora.py \
    --config cfgs/finetune/caption/coco_rgb_r8_attention.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Checkpoints --> /scratch-shared/tesr125338/output/caption/coco_rgb_r8_attention_drop01/"
echo "    lora_weights-best.pth     <- best val loss"
echo "    log.txt                   <- per-epoch train/val loss"
echo "  Next: update LORA_WEIGHTS in notebooks/viz_caption_and_depth_forgetting.py"
echo "        and paths in scripts/snellius/21_eval_caption_coco.sh (run_name=r8_attention)"
echo "================================================================"
