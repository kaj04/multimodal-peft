#!/bin/bash
#SBATCH --job-name=metric_debug
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=janus/tests/output/metric_debug/slurm_%j.out
#SBATCH --error=janus/tests/output/metric_debug/slurm_%j.err
#SBATCH --export=ALL

# Metric-debugging suite for the KL sweep (train-vs-test consistency, memorization,
# distributional DINO, content-preservation). Re-scores the already-generated panel
# images; no retraining, no regeneration.
#   sbatch janus/eval/23_run_metric_debug.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/metric_debug

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "  metric-debug | $(date)"
$JANUS_PY ~/multimodal-peft/janus/eval/23_metric_debug.py
echo "  Done: $(date)"
