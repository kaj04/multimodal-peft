#!/bin/bash
#SBATCH --job-name=writing_rank_cmp
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/writing_rank_compare/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/writing_rank_compare/slurm_%j.err
#SBATCH --export=ALL

# Side-by-side poetry continuations base + r4/r8/r16/r32/r64 -> pick the rank.
#   sbatch janus/eval/16_run_compare_writing_ranks.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/writing_rank_compare

echo "================================================================"
echo "  Writing rank compare"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/16_compare_writing_ranks.py
echo "Done: $(date)"
