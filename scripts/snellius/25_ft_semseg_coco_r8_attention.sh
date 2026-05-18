#!/bin/bash
# =============================================================================
# 25_ft_semseg_coco_r8_attention.sh
# LoRA fine-tuning: COCO semseg-only — rank=8, attention, dropout=0.1
#
# Semseg-only baseline for comparison with run 26 (multitask caption+semseg).
# Same LoRA config (r8/attention/drop01), same data, single task.
#
# Usage:
#   sbatch scripts/snellius/25_ft_semseg_coco_r8_attention.sh
# =============================================================================
#SBATCH --job-name=4m_sem_r8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_semseg_r8_attention_%j.out

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
echo "  4M-21 LoRA FT -- COCO Semseg-only, r8/attention/drop01"
echo "  Config : cfgs/finetune/semseg/coco_rgb_r8_attention.yaml"
echo "  Output : /scratch-shared/tesr125338/output/semseg/coco_rgb_r8_attention"
echo "  Train  : 10k images (seed=42 subset of COCO train2017)"
echo "  Val    : 5k images (full COCO val2017)"
echo "  WandB  : lora_semseg_coco_r8_attention @ 4M-PEFT"
echo "  Started: $(date)"
echo "================================================================"

# Verify tokenized data exists
TOKENS_TRAIN=/scratch-shared/tesr125338/coco_tokens/train
TOKENS_VAL=/scratch-shared/tesr125338/coco_tokens/val

n_rgb_train=$(ls "${TOKENS_TRAIN}/tok_rgb/"     2>/dev/null | wc -l)
n_sem_train=$(ls "${TOKENS_TRAIN}/tok_semseg/"  2>/dev/null | wc -l)
n_rgb_val=$(ls   "${TOKENS_VAL}/tok_rgb/"       2>/dev/null | wc -l)
n_sem_val=$(ls   "${TOKENS_VAL}/tok_semseg/"    2>/dev/null | wc -l)

echo "  Token check:"
echo "    train tok_rgb    : ${n_rgb_train}  (expected ~10000)"
echo "    train tok_semseg : ${n_sem_train}  (expected ~10000)"
echo "    val   tok_rgb    : ${n_rgb_val}    (expected 5000)"
echo "    val   tok_semseg : ${n_sem_val}    (expected 5000)"

if [ "${n_sem_train}" -lt 100 ]; then
    echo "ERROR: tok_semseg missing -- run 24_tokenize_coco_semseg.sh first"
    exit 1
fi

echo ""

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=${SLURM_NTASKS} \
    --master_port=29511 \
    run_finetuning_lora.py \
    --config cfgs/finetune/semseg/coco_rgb_r8_attention.yaml \
    --dist_eval

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Checkpoints --> /scratch-shared/tesr125338/output/semseg/coco_rgb_r8_attention/"
echo "  Next: sbatch scripts/snellius/30_eval_semseg_r8_attention.sh"
echo "================================================================"
