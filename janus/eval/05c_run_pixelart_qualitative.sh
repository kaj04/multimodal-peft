#!/bin/bash
#SBATCH --job-name=pixelart_qual
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/pixelart_qual/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/pixelart_qual/slurm_%j.err
#SBATCH --export=ALL

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/pixelart_qual

echo "================================================================"
echo "  Pixel-art qualitative (5 prompts x 4 conditions)"
echo "  ADAPTER = ${ADAPTER}"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/05c_pixelart_qualitative.py

echo "  Done: $(date)"
