#!/bin/bash
#SBATCH --job-name=janus_train_gen
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=janus/train/output/slurm_%j.out
#SBATCH --error=janus/train/output/slurm_%j.err

mkdir -p $HOME/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-1B LoRA GENERATION fine-tune (custom loss)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Use the absolute python of the janus env (safe against PATH leaks).
JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# Quick plumbing test first: TRAIN_N=200 sbatch ...
$JANUS_PY ~/multimodal-peft/janus/train/02_train_generation.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/multimodal-peft/janus/train/output/"
echo "================================================================"
