#!/bin/bash
# =============================================================================
# 17_eval_depth_labeled_r8.sh
# Depth evaluation: baseline vs LoRA (run 15: r8/selfattn/drop005)
# on BOTH train and val splits of NYUv2 labeled.
#
# 4 runs:
#   [1] NYU val  — baseline (pretrained 4M-21 XL, no LoRA)
#   [2] NYU val  — LoRA r8/selfattn (lora_weights-best.pth from run 15)
#   [3] NYU train — baseline
#   [4] NYU train — LoRA r8/selfattn
#
# Usage:
#   sbatch scripts/snellius/17_eval_depth_labeled_r8.sh
# =============================================================================
#SBATCH --job-name=4m_eval_r8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_depth_labeled_r8_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs
cd "$REPO_ROOT"

# ── Paths ────────────────────────────────────────────────────────────────────
NYU_VAL="/scratch-shared/tesr125338/nyu_raw/test"
NYU_TRAIN="/scratch-shared/tesr125338/nyu_raw/train"
LORA_WEIGHTS="/scratch-shared/tesr125338/output/depth/nyu_labeled_r8_selfattn_drop005/lora_weights-best.pth"
OUTPUT_DIR="/scratch-shared/tesr125338/output/eval_depth"

FM_MODEL="EPFL-VILAB/4M-21_XL"
TOK_DEPTH="EPFL-VILAB/4M_tokenizers_depth_8k_224-448"
TOK_TEXT="ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"

COMMON_ARGS="--fm ${FM_MODEL} --tok_depth ${TOK_DEPTH} --tok_text ${TOK_TEXT}
             --output_dir ${OUTPUT_DIR} --max_depth_clip 10.0
             --divae_steps 50 --device cuda --dtype bfloat16"

LORA_ARGS="--lora_weights ${LORA_WEIGHTS} --lora_rank 8 --lora_modules selfattn"

echo "================================================================"
echo "  4M-21 Depth Eval — Baseline vs LoRA r8/selfattn/drop005"
echo "  NYU labeled: val (654) + train (795)"
echo "  LoRA: ${LORA_WEIGHTS}"
echo "  Started: $(date)"
echo "================================================================"

# ── [1] Val — Baseline ───────────────────────────────────────────────────────
echo ">>> [1/4] NYU val — Baseline"
python eval_depth.py \
    --dataset nyu --dataset_root "${NYU_VAL}" \
    --run_name nyu_val_baseline \
    ${COMMON_ARGS}

# ── [2] Val — LoRA ───────────────────────────────────────────────────────────
echo ">>> [2/4] NYU val — LoRA r8/selfattn"
python eval_depth.py \
    --dataset nyu --dataset_root "${NYU_VAL}" \
    --run_name nyu_val_lora_r8_selfattn \
    ${COMMON_ARGS} ${LORA_ARGS}

# ── [3] Train — Baseline ─────────────────────────────────────────────────────
echo ">>> [3/4] NYU train — Baseline"
python eval_depth.py \
    --dataset nyu --dataset_root "${NYU_TRAIN}" \
    --run_name nyu_train_baseline \
    ${COMMON_ARGS}

# ── [4] Train — LoRA ─────────────────────────────────────────────────────────
echo ">>> [4/4] NYU train — LoRA r8/selfattn"
python eval_depth.py \
    --dataset nyu --dataset_root "${NYU_TRAIN}" \
    --run_name nyu_train_lora_r8_selfattn \
    ${COMMON_ARGS} ${LORA_ARGS}

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  All evaluations complete: $(date)"
echo "================================================================"

for split in val train; do
    for model in baseline lora_r8_selfattn; do
        f="${OUTPUT_DIR}/nyu_${split}_${model}/summary.txt"
        if [ -f "$f" ]; then
            echo ""
            echo "--- NYU ${split} / ${model} ---"
            cat "$f"
        fi
    done
done
