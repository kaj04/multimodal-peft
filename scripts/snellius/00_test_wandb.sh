#!/bin/bash
# =============================================================================
# 00_test_wandb.sh — WandB offline workflow test (no GPU, no model)
#
# Validates the full WANDB_MODE=offline → wandb sync pipeline.
# After the job finishes, run from the login node:
#   wandb sync ~/4m-peft/wandb/run-*/
#
# Usage:
#   sbatch scripts/snellius/00_test_wandb.sh
# =============================================================================
#SBATCH --job-name=wandb_test
#SBATCH --partition=thin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/wandb_test_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

REPO_ROOT="$HOME/4m-peft"
cd "$REPO_ROOT"
mkdir -p logs

# ── WandB offline mode (compute nodes have no internet) ───────────────────────
export WANDB_MODE=offline
export WANDB_API_KEY=$(cat ~/.wandb_api_key)   # see setup instructions below
export WANDB_PROJECT="4m-peft"

echo "=== WandB connectivity test ==="
echo "WANDB_MODE: $WANDB_MODE"
echo "WANDB_PROJECT: $WANDB_PROJECT"

python test_wandb.py --project 4m-peft --n_epochs 10

echo ""
echo "=== Job done. To sync to wandb.ai, run from the LOGIN NODE: ==="
echo "    cd ~/4m-peft && wandb sync wandb/run-*/"
