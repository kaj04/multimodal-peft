#!/bin/bash
#SBATCH --job-name=janus_rank_sweep
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/sweep_slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/sweep_slurm_%j.err

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output
mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting

echo "================================================================"
echo "  Janus-Pro-1B LoRA captioning rank sweep"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Default ranks; override with: RANKS=2,4,8 sbatch ...
# Other overrides: TRAIN_N, EVAL_N, GEN_EVAL_N, EPOCHS, LR, SEED, LORA_ALPHA_RULE
python ~/multimodal-peft/janus/train/03_rank_sweep.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Summary: ~/multimodal-peft/janus/tests/output/forgetting/rank_sweep_summary.json"
echo "================================================================"
