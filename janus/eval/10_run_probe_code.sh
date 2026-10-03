#!/bin/bash
#SBATCH --job-name=code_probe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=janus/tests/output/code_probe/slurm_%j.out
#SBATCH --error=janus/tests/output/code_probe/slurm_%j.err
#SBATCH --export=ALL

# Probe base Janus code ability (no adapter):
#   sbatch janus/eval/10_run_probe_code.sh
# Or with an adapter later:
#   sbatch --export=ALL,ADAPTER=/path/to/adapter_best janus/eval/10_run_probe_code.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/code_probe

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/10_probe_code_capability.py
echo "Done: $(date)"
