#!/bin/bash
# =============================================================================
# 09_eval_depth.sh  —  Depth evaluation: baseline vs LoRA on NYU / DIODE
#
# Runs eval_depth.py with affine-invariant metrics (AbsRel, RMSE, δ<1.25).
#
# Usage:
#   # All four combinations (baseline + LoRA × NYU + DIODE):
#   sbatch scripts/snellius/09_eval_depth.sh
#
#   # Smoke-test (20 images):
#   sbatch --export=ALL,MAX_IMAGES=20 scripts/snellius/09_eval_depth.sh
#
#   # Only baseline:
#   sbatch --export=ALL,SKIP_LORA=1 scripts/snellius/09_eval_depth.sh
#
# Output:
#   output/eval_depth/nyu_baseline_<timestamp>/
#   output/eval_depth/nyu_lora_<timestamp>/
#   output/eval_depth/diode_baseline_<timestamp>/
#   output/eval_depth/diode_lora_<timestamp>/
#
#   Each contains: results.json  summary.txt  config.json
#
# Monitor:
#   tail -f logs/eval_depth_<jobid>.out
# =============================================================================
#SBATCH --job-name=4m_eval_depth
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_depth_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs
cd "$REPO_ROOT"

# ── Paths ─────────────────────────────────────────────────────────────────────
NYU_ROOT="/scratch-shared/tesr125338/nyu_raw/test"
DIODE_ROOT="/scratch-shared/tesr125338/diode/val"
LORA_WEIGHTS="output/depth/nyu_rgb_r8/lora_weights-best.pth"

FM_MODEL="EPFL-VILAB/4M-21_XL"
TOK_DEPTH="EPFL-VILAB/4M_tokenizers_depth_8k_224-448"
TOK_TEXT="ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"

# Optional cap for smoke-test
MAX_IMAGES_ARG=""
if [ -n "${MAX_IMAGES:-}" ]; then
    MAX_IMAGES_ARG="--max_images ${MAX_IMAGES}"
    echo "  → Smoke-test mode: first ${MAX_IMAGES} images per run"
fi

echo "================================================================"
echo "  4M-21 Depth Evaluation — Baseline vs LoRA"
echo "  NYU Depth v2 test + DIODE val"
echo "  Started: $(date)"
echo "================================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. NYU — Baseline
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [1/4] NYU — Baseline"
python eval_depth.py \
    --dataset      nyu \
    --dataset_root "${NYU_ROOT}" \
    --fm           "${FM_MODEL}" \
    --tok_depth    "${TOK_DEPTH}" \
    --tok_text     "${TOK_TEXT}" \
    --output_dir   output/eval_depth \
    --run_name     nyu_baseline \
    --max_depth_clip 10.0 \
    --divae_steps  50 \
    --device       cuda \
    --dtype        bfloat16 \
    ${MAX_IMAGES_ARG}
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. NYU — LoRA
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "${SKIP_LORA:-}" ] && [ -f "${LORA_WEIGHTS}" ]; then
    echo ">>> [2/4] NYU — LoRA  (${LORA_WEIGHTS})"
    python eval_depth.py \
        --dataset      nyu \
        --dataset_root "${NYU_ROOT}" \
        --fm           "${FM_MODEL}" \
        --tok_depth    "${TOK_DEPTH}" \
        --tok_text     "${TOK_TEXT}" \
        --lora_weights "${LORA_WEIGHTS}" \
        --output_dir   output/eval_depth \
        --run_name     nyu_lora \
        --max_depth_clip 10.0 \
        --divae_steps  50 \
        --device       cuda \
        --dtype        bfloat16 \
        ${MAX_IMAGES_ARG}
    echo ""
else
    echo ">>> [2/4] NYU — LoRA  SKIPPED (SKIP_LORA set or weights not found)"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. DIODE — Baseline
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [3/4] DIODE — Baseline"
python eval_depth.py \
    --dataset      diode \
    --dataset_root "${DIODE_ROOT}" \
    --fm           "${FM_MODEL}" \
    --tok_depth    "${TOK_DEPTH}" \
    --tok_text     "${TOK_TEXT}" \
    --output_dir   output/eval_depth \
    --run_name     diode_baseline \
    --max_depth_clip 300.0 \
    --divae_steps  50 \
    --device       cuda \
    --dtype        bfloat16 \
    ${MAX_IMAGES_ARG}
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 4. DIODE — LoRA
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "${SKIP_LORA:-}" ] && [ -f "${LORA_WEIGHTS}" ]; then
    echo ">>> [4/4] DIODE — LoRA  (${LORA_WEIGHTS})"
    python eval_depth.py \
        --dataset      diode \
        --dataset_root "${DIODE_ROOT}" \
        --fm           "${FM_MODEL}" \
        --tok_depth    "${TOK_DEPTH}" \
        --tok_text     "${TOK_TEXT}" \
        --lora_weights "${LORA_WEIGHTS}" \
        --output_dir   output/eval_depth \
        --run_name     diode_lora \
        --max_depth_clip 300.0 \
        --divae_steps  50 \
        --device       cuda \
        --dtype        bfloat16 \
        ${MAX_IMAGES_ARG}
    echo ""
else
    echo ">>> [4/4] DIODE — LoRA  SKIPPED"
    echo ""
fi

echo "================================================================"
echo "  All evaluations complete: $(date)"
echo "================================================================"
echo ""
echo "--- Results summary ---"
for d in output/eval_depth/nyu_baseline output/eval_depth/nyu_lora \
          output/eval_depth/diode_baseline output/eval_depth/diode_lora; do
    if [ -f "$d/summary.txt" ]; then
        echo ""
        cat "$d/summary.txt"
    fi
done
