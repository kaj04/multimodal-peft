#!/bin/bash
#SBATCH --job-name=janus_forgetting
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting/slurm_%j.err
#SBATCH --export=ALL

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting

echo "================================================================"
echo "  Cross-modal forgetting eval (adapter OFF vs ON)"
echo "  Started: $(date)"
echo "  ADAPTER     = ${ADAPTER:-<default>}"
echo "  TRAINED_ON  = ${TRAINED_ON:-<default=captioning>}"
echo "  OUTPUT_TAG  = ${OUTPUT_TAG:-<none>}"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Use the absolute python of the janus env. The shell-relative `python` can
# resolve to the SYSTEM Anaconda (python 3.11) instead of the env (3.10) when
# --export=ALL propagates a broken PATH/CONDA_PREFIX from the launching shell.
JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

$JANUS_PY ~/multimodal-peft/janus/eval/02_eval_adapter.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Report dir: ~/multimodal-peft/janus/tests/output/forgetting/"
echo "================================================================"
