#!/bin/bash
#SBATCH --job-name=welsh_text_ctrl
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/welsh_text_ctrl_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/welsh_text_ctrl_%j.err
#SBATCH --export=ALL
#
# Text-only control for the Welsh VQA probe. Needs ADAPTER.
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/chainA_s1_welsh/adapter_best janus/eval/42_run_welsh_text_control.sh
#
# The same 28 questions as the VQA probe, with the image replaced by one English
# sentence. No image generation, so this is 112 forward passes and finishes in
# minutes.

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "================================================================"
echo "  welsh text-only control | ADAPTER=${ADAPTER}"
echo "  Started: $(date)"
echo "================================================================"

$JANUS_PY $HOME/multimodal-peft/janus/eval/42_welsh_text_vqa_control.py

echo "Done: $(date)"
