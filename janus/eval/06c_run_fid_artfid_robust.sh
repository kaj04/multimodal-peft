#!/bin/bash
#SBATCH --job-name=fid_robust
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=janus/tests/output/metrics_panel/slurm_%j.out
#SBATCH --error=janus/tests/output/metrics_panel/slurm_%j.err
#SBATCH --export=ALL

# Robust FID + ArtFID for all adapters under metrics_panel/ (reuses the
# on-disk generated images; builds the full 1221-image Naruto reference
# once). Single job covers all 6 adapters, ~10-15 min each.

echo "================================================================"
echo "  Robust FID + ArtFID | $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/06c_fid_artfid_robust.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
