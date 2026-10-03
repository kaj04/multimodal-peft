#!/bin/bash
#SBATCH --job-name=janus_smoke
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=janus/tests/output/slurm_%j.out
#SBATCH --error=janus/tests/output/slurm_%j.err

mkdir -p $HOME/multimodal-peft/janus/tests/output

echo "================================================================"
echo "  Janus-Pro-1B smoke test"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

# Shared HF cache on scratch. Snellius GPU nodes have internet, so the model
# downloads here automatically on first run (and is reused from cache after).
export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Use the absolute python of the janus env (safe against PATH leaks).
JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

$JANUS_PY ~/multimodal-peft/janus/tests/01_smoke_test.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Outputs: ~/multimodal-peft/janus/tests/output/"
echo "================================================================"
