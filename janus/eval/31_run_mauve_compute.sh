#!/bin/bash
#SBATCH --job-name=mauve_cmp
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/mauve/slurm_cmp_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/mauve/slurm_cmp_%j.err
#SBATCH --export=ALL

# Phase 2: compute MAUVE in the mauve venv (gpt2-large featurizer). Needs TAG (+
# KINDS). Pre-export KINDS (commas) then --export=ALL.
#   export KINDS=english,welsh; sbatch --export=ALL,TAG=text_7b_welsh_r32_kl0.1 janus/eval/31_run_mauve_compute.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/mauve
module load 2023
module load Anaconda3/2023.07-2
export HF_HOME=/scratch-shared/tesr125338/hf_cache
MAUVE_PY=$HOME/multimodal-peft/mauve_venv/bin/python

echo "  mauve-compute | TAG=${TAG} KINDS=${KINDS:-english} | $(date)"
$MAUVE_PY ~/multimodal-peft/janus/eval/31_mauve_compute.py
echo "  Done: $(date)"
