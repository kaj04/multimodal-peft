#!/bin/bash
#SBATCH --job-name=test_wandb
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=logs/test_wandb_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')
export WANDB_PROJECT="4m-peft"

cd "$HOME/4m-peft"
mkdir -p logs

echo "=== WandB connectivity test ==="
echo "API key length: ${#WANDB_API_KEY} chars"

# Explicit login before init — fixes 'user not logged in' on Snellius
wandb login "$WANDB_API_KEY"

python - << 'EOF'
import wandb
run = wandb.init(project="4m-peft", name="connectivity-test", tags=["test"])
wandb.log({"test_metric": 1.0})
run.finish()
print("WandB test SUCCESS")
EOF

echo "=== Done ==="
