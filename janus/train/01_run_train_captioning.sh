#!/bin/bash
#SBATCH --job-name=janus_train_cap
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-1B LoRA captioning fine-tune"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Quick plumbing test first? Override with: TRAIN_N=200 EPOCHS=1 sbatch ...
python ~/multimodal-peft/janus/train/01_train_captioning.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/multimodal-peft/janus/train/output/"
echo "================================================================"
