#!/bin/bash
#SBATCH --job-name=welsh_probe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=janus/tests/output/welsh_probe_%j.out
#SBATCH --error=janus/tests/output/welsh_probe_%j.err
#SBATCH --export=ALL
#
# Does the Welsh prompt reach the image generator? Needs ADAPTER.
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/chainA_s1_welsh/adapter_best janus/eval/41_run_welsh_probe.sh
#
# Twelve generations at a fixed seed, so any difference between two images is
# attributable to the prompt and nothing else.

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "================================================================"
echo "  welsh prompt probe | ADAPTER=${ADAPTER}"
echo "  Started: $(date)"
echo "================================================================"

$JANUS_PY $HOME/multimodal-peft/janus/eval/41_welsh_prompt_probe.py

echo "Done: $(date)"
