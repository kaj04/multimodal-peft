#!/bin/bash
#SBATCH --job-name=t2i_xmodal
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/t2i_crossmodal_qual/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/t2i_crossmodal_qual/slurm_%j.err
#SBATCH --export=ALL

# Cross-modal image test: base vs adapter, same prompts/seed. Needs ADAPTER.
#   sbatch --export=ALL,ADAPTER=/path/to/adapter_best janus/eval/13_run_t2i_crossmodal_qual.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/t2i_crossmodal_qual

echo "================================================================"
echo "  T2I cross-modal qualitative | ADAPTER=${ADAPTER}"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/13_t2i_crossmodal_qual.py
echo "Done: $(date)"
