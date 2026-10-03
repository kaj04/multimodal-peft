#!/bin/bash
# =============================================================================
# 35_eval_all.sh
# Evaluate ONE LoRA checkpoint on all 5 tasks → row.json
#
# Edit RUN_NAME and LORA_WEIGHTS before submitting.
# Submit one job per LoRA (baseline, run23, run25, run_det, run_depth,
# run_normal, run34) to fill all rows of the table in parallel.
#
# Usage:
#   RUN_NAME=run34_5task LORA=/path/to/lora_weights-best.pth \
#       sbatch scripts/snellius/35_eval_all.sh
#
# Or edit the variables below and sbatch directly.
# =============================================================================
#SBATCH --job-name=eval_all
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_all_%x_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs
cd "$REPO_ROOT"

# ── Edit these two variables before submitting ────────────────────────────────
RUN_NAME="${RUN_NAME:-run34_5task}"
LORA_WEIGHTS="${LORA}"   # empty string = baseline (no LoRA); set via --export=ALL,LORA=...
OUTPUT_DIR="/scratch-shared/tesr125338/output/eval_all/${RUN_NAME}"
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  eval_all — ${RUN_NAME}"
echo "  LoRA    : ${LORA_WEIGHTS:-<baseline, no LoRA>}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Tasks   : caption semseg det depth normal"
echo "  Started : $(date)"
echo "================================================================"

# Build python command — omit --lora_weights entirely for baseline
PYTHON_ARGS=(
    --run_name      "${RUN_NAME}"
    --output_dir    "${OUTPUT_DIR}"
    --tasks         caption semseg det depth normal
    --device        cuda
    --dtype         bfloat16
)
if [ -n "${LORA_WEIGHTS}" ]; then
    PYTHON_ARGS+=(
        --lora_weights  "${LORA_WEIGHTS}"
        --lora_rank     8
        --lora_modules  attention
    )
fi

python eval_all.py "${PYTHON_ARGS[@]}"

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  row.json → ${OUTPUT_DIR}/row.json"
echo "================================================================"
