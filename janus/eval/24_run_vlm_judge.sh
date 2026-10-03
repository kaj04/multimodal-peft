#!/bin/bash
#SBATCH --job-name=vlm_judge
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=janus/tests/output/vlm_judge/slurm_%j.out
#SBATCH --error=janus/tests/output/vlm_judge/slurm_%j.err
#SBATCH --export=ALL

# LLM-as-judge with an INDEPENDENT open VLM (Qwen2-VL-7B). Pairwise, counterbalanced.
#   SMOKE first:   sbatch --export=ALL,SMOKE=1 janus/eval/24_run_vlm_judge.sh
#   full:          sbatch janus/eval/24_run_vlm_judge.sh
#   other pair:    sbatch --export=ALL,PAIR=kl0.5,kl1 janus/eval/24_run_vlm_judge.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/vlm_judge

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "  vlm-judge | $(date)"
# NO pip upgrades here: the janus env already has a recent transformers (it
# supports Janus), so Qwen2-VL should load as-is. If the SMOKE test fails on the
# transformers version, we use a SEPARATE venv -- never upgrade the janus env.
$JANUS_PY -c "import transformers; print('  transformers', transformers.__version__)"

$JANUS_PY ~/multimodal-peft/janus/eval/24_vlm_judge.py
echo "  Done: $(date)"
