#!/bin/bash
# =============================================================================
# 06_eval_depth.sh  —  Metric depth evaluation on NYUv2 val set (1 x A100)
#
# Runs eval_depth_pixel.py to compute δ₁/RMSE/AbsRel/SILog in metres
# after per-image median-scale alignment.
#
# Usage:
#   # Baseline only (zero-shot 4M-21):
#   sbatch scripts/snellius/06_eval_depth.sh baseline
#
#   # LoRA checkpoint (replace path as needed):
#   sbatch scripts/snellius/06_eval_depth.sh lora output/depth/nyu_rgb_r8_e5/lora_weights-best.pth
#
#   # Both (comparison table in one job):
#   sbatch scripts/snellius/06_eval_depth.sh both output/depth/nyu_rgb_r8/lora_weights-best.pth
#
#   Default (no args): runs baseline.
# =============================================================================
#SBATCH --job-name=4m_eval_depth
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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

MODE="${1:-baseline}"
LORA_CKPT="${2:-}"

echo "=== NYUv2 metric depth evaluation ==="
echo "Mode:      $MODE"
echo "LoRA ckpt: ${LORA_CKPT:-N/A}"
echo "Job ID:    $SLURM_JOB_ID"
echo ""

if [ "$MODE" = "baseline" ]; then
    python eval_depth_pixel.py \
        --mode baseline \
        --batch_size 4 \
        --output_json "logs/eval_depth_baseline_${SLURM_JOB_ID}.json"

elif [ "$MODE" = "lora" ]; then
    if [ -z "$LORA_CKPT" ]; then
        echo "ERROR: LoRA checkpoint path required as \$2 for mode 'lora'"
        exit 1
    fi
    python eval_depth_pixel.py \
        --mode lora \
        --lora_ckpt "$LORA_CKPT" \
        --batch_size 4 \
        --output_json "logs/eval_depth_lora_${SLURM_JOB_ID}.json"

elif [ "$MODE" = "both" ]; then
    if [ -z "$LORA_CKPT" ]; then
        echo "ERROR: LoRA checkpoint path required as \$2 for mode 'both'"
        exit 1
    fi
    python eval_depth_pixel.py \
        --mode both \
        --lora_ckpt "$LORA_CKPT" \
        --batch_size 4 \
        --output_json "logs/eval_depth_comparison_${SLURM_JOB_ID}.json"

else
    echo "ERROR: Unknown mode '$MODE'. Use: baseline | lora | both"
    exit 1
fi

echo ""
echo "=== Evaluation complete ==="
echo "Results in: logs/eval_depth_*_${SLURM_JOB_ID}.json"
