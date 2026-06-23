#!/bin/bash
#SBATCH --job-name=cmmd_ci
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/confidence_intervals/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/confidence_intervals/slurm_%j.err
#SBATCH --export=ALL

# Bootstrap 95% CI for CMMD / Gen-CMMD. Pass GEN_DIR (+ REF_DIR or HF_DATASET_NAME)
# and LABEL via the environment (pre-export space/odd values, then --export=ALL).
#   GEN_DIR=... HF_DATASET_NAME=... LABEL=... sbatch janus/eval/21_run_bootstrap_cmmd.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/confidence_intervals

echo "================================================================"
echo "  CMMD bootstrap CI | LABEL=${LABEL}  GEN_DIR=${GEN_DIR}"
echo "  REF=${REF_DIR:-HF:$HF_DATASET_NAME}  B=${B:-2000}  | $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/21_bootstrap_cmmd.py
echo "Done: $(date)"
