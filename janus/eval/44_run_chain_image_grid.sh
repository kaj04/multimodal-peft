#!/bin/bash
#SBATCH --job-name=chain_grid
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=janus/tests/output/chain_image_grid/slurm_%j.out
#SBATCH --error=janus/tests/output/chain_image_grid/slurm_%j.err
#SBATCH --export=ALL
#
# One image grid across every chain stage: does the image step learn Naruto, and
# does it survive the text step that follows?
#   sbatch janus/eval/44_run_chain_image_grid.sh
#
# 4 rows x 5 prompts = 20 generations, roughly 2-3 minutes each on one H100.
# Cells are cached to disk, so a re-run only fills what is missing.

mkdir -p $HOME/multimodal-peft/janus/tests/output/chain_image_grid

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "================================================================"
echo "  chain image grid | Started: $(date)"
echo "================================================================"

$JANUS_PY $HOME/multimodal-peft/janus/eval/44_chain_image_grid.py

echo "Done: $(date)"
