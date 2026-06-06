#!/bin/bash
#SBATCH --job-name=naruto_qual
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/naruto_qual_7b/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/naruto_qual_7b/slurm_%j.err
#SBATCH --export=ALL

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/naruto_qual_7b

echo "================================================================"
echo "  Naruto qualitative (2x2 prefix x adapter) comparison"
echo "  Started: $(date)"
echo "  MODEL_ID       = ${MODEL_ID:-<default=Janus-Pro-1B>}"
echo "  ADAPTER        = ${ADAPTER:-<default>}"
echo "  CAPTION_PREFIX = ${CAPTION_PREFIX:-<default>}"
echo "  N_EVAL_BASE    = ${N_EVAL_BASE:-5}"
echo "  N_HUMAN_BASE   = ${N_HUMAN_BASE:-5}"
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
echo "  Output: ~/multimodal-peft/janus/tests/output/naruto_qual_7b/"
echo "================================================================"
