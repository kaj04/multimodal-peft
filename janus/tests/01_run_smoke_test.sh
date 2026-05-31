#!/bin/bash
#SBATCH --job-name=janus_smoke
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/slurm_%j.err

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output

echo "================================================================"
echo "  Janus-Pro-1B smoke test"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

# Shared HF cache on scratch (model must be pre-downloaded on login node)
export HF_HOME=/scratch-shared/tesr125338/hf_cache
# Compute nodes have no internet -> force offline so it loads from cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python ~/multimodal-peft/janus/tests/01_smoke_test.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Outputs: ~/multimodal-peft/janus/tests/output/"
echo "================================================================"
