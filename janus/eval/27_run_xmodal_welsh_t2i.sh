#!/bin/bash
#SBATCH --job-name=xm_welsh_t2i
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
#SBATCH --output=janus/tests/output/xmodal_welsh_t2i/slurm_%j.out
#SBATCH --error=janus/tests/output/xmodal_welsh_t2i/slurm_%j.err
#SBATCH --export=ALL

# Exp A: Welsh prompt -> image faithfulness (CLIPScore vs English ref). Needs ADAPTER.
#   smoke: sbatch --export=ALL,SMOKE=1,ADAPTER=... janus/eval/27_run_xmodal_welsh_t2i.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/xmodal_welsh_t2i
module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "  expA t2i | ADAPTER=${ADAPTER} | SMOKE=${SMOKE:-0} | $(date)"
$JANUS_PY ~/multimodal-peft/janus/eval/27_xmodal_welsh_t2i.py
echo "  Done: $(date)"
