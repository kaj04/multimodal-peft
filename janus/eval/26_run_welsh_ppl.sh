#!/bin/bash
#SBATCH --job-name=welsh_ppl
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=janus/tests/output/welsh_ppl/slurm_%j.out
#SBATCH --error=janus/tests/output/welsh_ppl/slurm_%j.err
#SBATCH --export=ALL

# EVAL-ONLY Welsh acquisition: held-out Welsh PPL base vs adapter, no retraining.
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/text_7b_welsh_r32_kl0.1/adapter_best janus/eval/26_run_welsh_ppl.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/welsh_ppl

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "  welsh PPL | ADAPTER=${ADAPTER} | $(date)"
$JANUS_PY ~/multimodal-peft/janus/eval/26_welsh_ppl.py
echo "  Done: $(date)"
