#!/bin/bash
#SBATCH --job-name=anole_inference_test
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/tesr125338/anole_outputs/tests/slurm_%j.out
#SBATCH --error=/scratch-shared/tesr125338/anole_outputs/tests/slurm_%j.err

echo "================================================================"
echo "  Anole-7b-v0.1 inference test"
echo "  Started: $(date)"
echo "================================================================"

mkdir -p /scratch-shared/tesr125338/anole_outputs/tests

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate anole

export CKPT_PATH=/scratch-shared/tesr125338/anole_ckpt
export ANOLE_REPO=~/anole

python ~/4m-peft/anole/tests/01_test_inference.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Images saved in: /scratch-shared/tesr125338/anole_outputs/tests/"
echo "================================================================"
