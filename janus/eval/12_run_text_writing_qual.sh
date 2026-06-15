#!/bin/bash
#SBATCH --job-name=writing_qual
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/text_writing_qual/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/text_writing_qual/slurm_%j.err
#SBATCH --export=ALL

# Base-vs-adapter writing test. Needs ADAPTER.
#   sbatch --export=ALL,ADAPTER=/path/to/adapter_best janus/eval/12_run_text_writing_qual.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/text_writing_qual

echo "================================================================"
echo "  Writing qualitative | ADAPTER=${ADAPTER}"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/12_text_writing_qual.py
echo "Done: $(date)"
