#!/bin/bash
#SBATCH --job-name=kl_post
#SBATCH --partition=rome
#SBATCH --time=00:15:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/kl_post_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/kl_post_%j.err
#SBATCH --export=ALL

# Post-processing for the KL sweep (CPU only): build the visual grid +
# the metric curves, both into overleaf/. Chained after all KL qualitatives.

module load 2023
module load Anaconda3/2023.07-2
source activate janus
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "== organize visual grid =="
bash $HOME/multimodal-peft/scripts/snellius_organize_kl_comparison.sh

echo "== plot metric curves =="
$JANUS_PY $HOME/multimodal-peft/janus/eval/09b_plot_kl_sweep.py

echo "Done: $(date)"
echo "Artifacts in: $HOME/multimodal-peft/janus/tests/output/overleaf/{kl_comparison,kl_sweep}"
