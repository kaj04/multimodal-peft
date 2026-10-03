#!/bin/bash
#SBATCH --job-name=artfid_only
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=janus/tests/output/metrics_panel/slurm_%j.out
#SBATCH --error=janus/tests/output/metrics_panel/slurm_%j.err
#SBATCH --export=ALL

# Compute ONLY ArtFID for all adapters under metrics_panel/, reusing the
# BEFORE/AFTER images already on disk. Updates each metrics_panel.json
# in-place with the new delta.ArtFID value.
#
# Expected runtime: ~2-3 min per adapter on H100 (Inception fwd + LPIPS),
# so ~15-20 min total for all 6 (v2/v3/v4/v5 + 2 joint).

echo "================================================================"
echo "  ArtFID-only update job | $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python

# art-fid should already be installed from previous metrics_panel runs; if
# not, this falls back to the non-destructive check_and_install.
if ! $JANUS_PY -c "from art_fid.art_fid import compute_art_fid" 2>/dev/null; then
    echo "  [deps] art-fid missing; installing"
    $JANUS_PY -m pip install --quiet "numpy<2" 2>&1 | tail -3
    $JANUS_PY -m pip install --quiet "art-fid" 2>&1 | tail -3
fi

$JANUS_PY ~/multimodal-peft/janus/eval/06b_artfid_only.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
