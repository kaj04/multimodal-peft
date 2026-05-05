#!/bin/bash
# =============================================================================
# 10_eval_forgetting.sh  —  Catastrophic forgetting eval: surface normals
#
# Evaluates baseline vs LoRA (depth fine-tuned, rank=8) on DIODE val normals.
# Requires 2 GPUs: cuda:0 = baseline, cuda:1 = LoRA model.
#
# Usage:
#   sbatch scripts/snellius/10_eval_forgetting.sh
#   sbatch scripts/snellius/10_eval_forgetting.sh --export=EPOCH=19
# =============================================================================
#SBATCH --job-name=4m_eval_forgetting
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_forgetting_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

# ── configurable epoch (override with --export=EPOCH=XX) ──────────────────────
EPOCH=${EPOCH:-49}
LORA_WEIGHTS="output/depth/nyu_rgb_r8/lora_weights-epoch${EPOCH}.pth"
OUTPUT_DIR="output/eval_forgetting/ep${EPOCH}"

echo "=== Catastrophic Forgetting Evaluation ==="
echo "    LoRA weights : $LORA_WEIGHTS"
echo "    Output dir   : $OUTPUT_DIR"
echo "    GPUs         : $SLURM_JOB_GPUS"

python eval_forgetting.py \
    --lora_weights "$LORA_WEIGHTS" \
    --output_dir   "$OUTPUT_DIR"

echo "=== Done. Results in: $OUTPUT_DIR/normals_results.json ==="
