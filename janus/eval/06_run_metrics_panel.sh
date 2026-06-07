#!/bin/bash
#SBATCH --job-name=metrics_panel
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/metrics_panel/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/metrics_panel/slurm_%j.err
#SBATCH --export=ALL

# Required env vars:
#   ADAPTER     -- path to the LoRA adapter dir (epoch checkpoint)
# Optional:
#   MODEL_ID, N_GEN, CAPTION_PREFIX

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/metrics_panel

echo "================================================================"
echo "  Metrics panel | $(date)"
echo "  ADAPTER       = ${ADAPTER}"
echo "  MODEL_ID      = ${MODEL_ID:-deepseek-community/Janus-Pro-7B}"
echo "  N_GEN         = ${N_GEN:-200}"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python

# Defensive deps install (idempotent; safe to run every time).
# If you've already installed these once, pip skips. ~5 sec when cached.
$JANUS_PY -m pip install --quiet clean-fid pyiqa scipy 2>&1 | tail -3
# art-fid is optional: if it fails to install (it has tricky deps), the
# metrics script falls back gracefully and just skips ArtFID.
$JANUS_PY -m pip install --quiet art-fid 2>&1 | tail -3 || \
    echo "  WARN: art-fid not installed; ArtFID will be skipped."

$JANUS_PY ~/multimodal-peft/janus/eval/06_metrics_panel.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
