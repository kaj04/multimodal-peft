#!/bin/bash
#SBATCH --job-name=domain_suit
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/domain_suitability/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/domain_suitability/slurm_%j.err
#SBATCH --export=ALL

# Base-only suitability of candidate text-PEFT domains: PPL xControl + MMLU
# subject accuracy per domain, both comparable.
#   sbatch janus/eval/19_run_domain_suitability.sh
# Knobs (optional): MMLU_PER_SUBJECT=50  MAX_WINDOWS=40  SMOKE=1 (tiny dry-run)

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/domain_suitability

echo "================================================================"
echo "  Domain suitability | Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/19_domain_suitability.py
echo "Done: $(date)"
