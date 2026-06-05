#!/bin/bash
#SBATCH --job-name=pokemon_qual
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/pokemon_qual/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/pokemon_qual/slurm_%j.err
#SBATCH --export=ALL

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/pokemon_qual

echo "================================================================"
echo "  Pokemon qualitative comparison"
echo "  Started: $(date)"
echo "  ADAPTER     = ${ADAPTER:-<default=last pokemon adapter>}"
echo "  N_PROMPTS   = ${N_PROMPTS:-<default=20>}"
echo "  N_SAVE      = ${N_SAVE:-<default=10>}"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

$JANUS_PY ~/multimodal-peft/janus/eval/05_pokemon_qualitative.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/multimodal-peft/janus/tests/output/pokemon_qual/"
echo "================================================================"
