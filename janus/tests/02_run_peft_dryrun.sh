#!/bin/bash
#SBATCH --job-name=janus_peft_dryrun
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/dryrun/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/dryrun/slurm_%j.err

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/dryrun

echo "================================================================"
echo "  Janus-Pro-1B PEFT dry-run"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Use the absolute python of the janus env (safe against PATH leaks).
JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

$JANUS_PY ~/multimodal-peft/janus/tests/02_peft_dryrun.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Report: ~/multimodal-peft/janus/tests/output/dryrun/results.json"
echo "================================================================"
