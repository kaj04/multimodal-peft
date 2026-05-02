#!/bin/bash
# =============================================================================
# 07_rank_ablation_depth.sh  —  LoRA rank ablation: NYUv2 depth (job array)
#
# Launches one job per rank: r4, r8, r16, r32, r64.
# Each job runs 50 epochs on 4×A100.
#
# Usage:
#   sbatch scripts/snellius/07_rank_ablation_depth.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/rank_depth_<jobid>_<arrayid>.out
#
# After all jobs complete, run eval for each checkpoint:
#   for r in 4 8 16 32 64; do
#     sbatch scripts/snellius/06_eval_depth.sh lora \
#       output/depth/nyu_rgb_r${r}/lora_weights-best.pth
#   done
# =============================================================================
#SBATCH --job-name=4m_rank_abl
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/rank_depth_%A_%a.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

# Map SLURM_ARRAY_TASK_ID → rank
RANKS=(4 8 16 32 64)
RANK=${RANKS[$SLURM_ARRAY_TASK_ID]}

CONFIG="cfgs/finetune/depth/nyu_rgb_r${RANK}.yaml"

echo "=== LoRA rank ablation: depth, rank=${RANK} ==="
echo "Config: ${CONFIG}"
echo "Array ID: $SLURM_ARRAY_TASK_ID  |  Job: $SLURM_ARRAY_JOB_ID"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=$((29510 + SLURM_ARRAY_TASK_ID)) \
    run_finetuning_lora.py \
    --config "$CONFIG" \
    --dist_eval

echo "Done: rank=${RANK}. Checkpoint in output/depth/nyu_rgb_r${RANK}/"
