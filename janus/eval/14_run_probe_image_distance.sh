#!/bin/bash
#SBATCH --job-name=img_dist_probe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/image_distance_probe/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/image_distance_probe/slurm_%j.err
#SBATCH --export=ALL

# Rank candidate IMAGE styles by base-Janus VQ-token CE to pick the most
# distant one for the generation-direction PEFT (kills the "base already
# good -> no real intervention" objection).
#
#   base probe:
#     sbatch janus/eval/14_run_probe_image_distance.sh
#   plumbing check (4 images/style, ~2 min):
#     sbatch --export=ALL,SMOKE=1 janus/eval/14_run_probe_image_distance.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/image_distance_probe

echo "================================================================"
echo "  Image-style distance probe   SMOKE=${SMOKE:-0}"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/14_probe_image_distance.py
echo "Done: $(date)"
