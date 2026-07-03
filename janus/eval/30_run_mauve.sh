#!/bin/bash
#SBATCH --job-name=mauve_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/mauve/slurm_gen_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/mauve/slurm_gen_%j.err
#SBATCH --export=ALL

# Phase 1: free-form text generation for MAUVE (janus env). Needs TAG (+ ADAPTER,
# TARGETS). Pre-export TARGETS (it contains commas) then --export=ALL.
#   smoke: export TARGETS=welsh; sbatch --export=ALL,SMOKE=1,TAG=welsh_smoke,ADAPTER=... janus/eval/30_run_mauve.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/mauve
module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "  mauve-gen | TAG=${TAG} ADAPTER=${ADAPTER:-（base)} TARGETS=${TARGETS:-} SMOKE=${SMOKE:-0} | $(date)"
$JANUS_PY ~/multimodal-peft/janus/eval/30_mauve_generate.py
echo "  Done: $(date)"
