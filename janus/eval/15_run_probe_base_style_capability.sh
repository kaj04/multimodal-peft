#!/bin/bash
#SBATCH --job-name=base_style_probe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=janus/tests/output/base_style_capability/slurm_%j.out
#SBATCH --error=janus/tests/output/base_style_capability/slurm_%j.err
#SBATCH --export=ALL

# Generate a styles x contents grid with the BASE model to see which styles it
# already does well (close) vs fails at (distant = good PEFT target).
#
#   default styles/contents:
#     sbatch janus/eval/15_run_probe_base_style_capability.sh
#   custom styles (label=suffix;...) and/or contents (a;b;c):
#     sbatch --export=ALL,STYLES="thermal=thermal camera image;low-poly=low poly 3D render" janus/eval/15_run_probe_base_style_capability.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/base_style_capability

echo "================================================================"
echo "  Base style-capability probe"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/15_probe_base_style_capability.py
echo "Done: $(date)"
