#!/bin/bash
#SBATCH --job-name=janus_forgetting
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting/slurm_%j.err

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting

echo "================================================================"
echo "  Cross-modal forgetting eval (adapter OFF vs ON)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# ADAPTER defaults to the captioning run's epoch-1 adapter; override if needed.
python ~/multimodal-peft/janus/eval/02_eval_adapter.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Report: ~/multimodal-peft/janus/tests/output/forgetting/forgetting_captioning.json"
echo "================================================================"
