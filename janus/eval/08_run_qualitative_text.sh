#!/bin/bash
#SBATCH --job-name=qual_text
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/qualitative_text/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/qualitative_text/slurm_%j.err
#SBATCH --export=ALL

# Qualitative text examples (MMLU + free-gen + POPE/MME), one model per run:
#   sbatch --export=ALL,ADAPTER= janus/eval/08_run_qualitative_text.sh
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/gen_7b_naruto_v5/adapter_epoch6 janus/eval/08_run_qualitative_text.sh
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/joint_7b_v5_matched/adapter_epoch6 janus/eval/08_run_qualitative_text.sh
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/joint_7b_v5_far/adapter_epoch6 janus/eval/08_run_qualitative_text.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/qualitative_text

echo "================================================================"
echo "  Qualitative text examples | $(date)"
echo "  ADAPTER = ${ADAPTER:-<none, base model>}"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/08_qualitative_text_examples.py

echo "  Done: $(date)"
