#!/bin/bash
#SBATCH --job-name=forget_battery
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting_battery/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting_battery/slurm_%j.err
#SBATCH --export=ALL

# Cross-modal forgetting battery (PPL + MMLU + POPE + MME).
# Run once per condition:
#
#   # base (no adapter)
#   sbatch --export=ALL,ADAPTER= janus/eval/07_run_forgetting_battery.sh
#
#   # image-only v5
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/gen_7b_naruto_v5/adapter_epoch6 janus/eval/07_run_forgetting_battery.sh
#
#   # joint matched / far
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/joint_7b_v5_matched/adapter_epoch6 janus/eval/07_run_forgetting_battery.sh
#   sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/joint_7b_v5_far/adapter_epoch6 janus/eval/07_run_forgetting_battery.sh
#
# SMOKE TEST FIRST (plumbing check, ~10 min):
#   sbatch --export=ALL,SMOKE=1,ADAPTER= janus/eval/07_run_forgetting_battery.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/forgetting_battery

echo "================================================================"
echo "  Forgetting battery | $(date)"
echo "  ADAPTER = ${ADAPTER:-<none, base model>}"
echo "  SMOKE   = ${SMOKE:-0}"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

$JANUS_PY ~/multimodal-peft/janus/eval/07_forgetting_battery.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
