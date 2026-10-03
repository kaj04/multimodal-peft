#!/bin/bash
#SBATCH --job-name=first_tok
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=janus/tests/output/first_token_collapse/slurm_%j.out
#SBATCH --error=janus/tests/output/first_token_collapse/slurm_%j.err
#SBATCH --export=ALL
#
# Why the damaged chain models return an empty continuation. Needs ADAPTER.
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/chainB_s2_naruto/adapter_best janus/eval/43_run_first_token_collapse.sh
#
# Six prompts, one forward pass each, base and adapter in the same process.

mkdir -p $HOME/multimodal-peft/janus/tests/output/first_token_collapse

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "================================================================"
echo "  first-token collapse | ADAPTER=${ADAPTER}"
echo "  Started: $(date)"
echo "================================================================"

$JANUS_PY $HOME/multimodal-peft/janus/eval/43_first_token_collapse.py

echo "Done: $(date)"
