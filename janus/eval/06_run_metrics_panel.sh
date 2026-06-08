#!/bin/bash
#SBATCH --job-name=metrics_panel
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
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
# NOTE: art-fid is intentionally NOT installed here -- it depends on numpy<2,
# which conflicts with opencv-python>=4.13 and breaks the env. The metrics
# script falls back gracefully and just skips ArtFID. We can compute ArtFID
# in a SEPARATE follow-up job with a pinned numpy<2, on the already-generated
# images saved to disk by this run.
$JANUS_PY -m pip install --quiet clean-fid pyiqa scipy 2>&1 | tail -3

$JANUS_PY ~/multimodal-peft/janus/eval/06_metrics_panel.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
