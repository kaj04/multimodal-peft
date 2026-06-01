#!/bin/bash
#SBATCH --job-name=janus_rank_sweep
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/sweep_slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/sweep_slurm_%j.err
#SBATCH --export=ALL

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output
mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting

echo "================================================================"
echo "  Janus-Pro-1B LoRA captioning rank sweep"
echo "  Started: $(date)"
echo "  RANKS  = ${RANKS:-<default=2,4,8,16,32,64>}"
echo "  TRAIN_N=${TRAIN_N:-<default=4000>}  EVAL_N=${EVAL_N:-<default=200>}"
echo "  GEN_EVAL_N=${GEN_EVAL_N:-<default=30>}  SEED=${SEED:-<default=42>}"
echo "  LORA_ALPHA_RULE=${LORA_ALPHA_RULE:-<default=2x>}"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Use the absolute python of the janus env, otherwise --export=ALL can leak the
# system Anaconda (python 3.11) and break transformers imports.
JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# Default ranks; override with: RANKS=2,4,8 sbatch ...
# Other overrides: TRAIN_N, EVAL_N, GEN_EVAL_N, EPOCHS, LR, SEED, LORA_ALPHA_RULE
$JANUS_PY ~/multimodal-peft/janus/train/03_rank_sweep.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Summary: ~/multimodal-peft/janus/tests/output/forgetting/rank_sweep_summary.json"
echo "================================================================"
