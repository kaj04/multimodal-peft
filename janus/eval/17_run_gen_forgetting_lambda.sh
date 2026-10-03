#!/bin/bash
#SBATCH --job-name=gen_forget_lambda
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=janus/tests/output/gen_forgetting_lambda/slurm_%j.out
#SBATCH --error=janus/tests/output/gen_forgetting_lambda/slurm_%j.err
#SBATCH --export=ALL

# Question-A image-gen forgetting over the r16 lambda sweep: CMMD(adapter-gen
# vs base-gen) on shared COCO prompts. Base generated once, reused for all
# lambda. Run AFTER the lambda trainings finish (skips any adapter not yet ready).
#   sbatch janus/eval/17_run_gen_forgetting_lambda.sh
#   cheaper/pricier: --export=ALL,N_GEN=96   /  N_GEN=256

mkdir -p $HOME/multimodal-peft/janus/tests/output/gen_forgetting_lambda

echo "================================================================"
echo "  gen-forgetting lambda (Question A) | N_GEN=${N_GEN:-128}"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/17_gen_forgetting_lambda.py
echo "Done: $(date)"
