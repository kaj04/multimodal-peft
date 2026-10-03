#!/bin/bash
#SBATCH --job-name=xm_welsh_txt
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=janus/tests/output/xmodal_welsh_textinimg/slurm_%j.out
#SBATCH --error=janus/tests/output/xmodal_welsh_textinimg/slurm_%j.err
#SBATCH --export=ALL

# Exp B: Welsh text-in-image (feasibility gate, generation + grid). Needs ADAPTER.
#   smoke: sbatch --export=ALL,SMOKE=1,ADAPTER=... janus/eval/29_run_xmodal_welsh_textinimg.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/xmodal_welsh_textinimg
module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "  expB textinimg | ADAPTER=${ADAPTER} | SMOKE=${SMOKE:-0} | $(date)"
$JANUS_PY ~/multimodal-peft/janus/eval/29_xmodal_welsh_textinimg.py
echo "  Done: $(date)"
