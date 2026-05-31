#!/bin/bash
#SBATCH --job-name=janus_baseline
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/baseline/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/baseline/slurm_%j.err

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/baseline

echo "================================================================"
echo "  Janus-Pro-1B baseline metric validation"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Optional overrides: N_CAPTION / N_GEN (defaults 100 / 30)
python ~/multimodal-peft/janus/eval/01_baseline.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Report: ~/multimodal-peft/janus/tests/output/baseline/baseline_results.json"
echo "================================================================"
