#!/bin/bash
#SBATCH --job-name=lang_probe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/language_probe/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/language_probe/slurm_%j.err
#SBATCH --export=ALL

# Distance probe over candidate TEXT targets (low-resource languages + a synthetic
# cipher): base PPL xControl + 3 example snippets each, to pick the strongest target.
#   sbatch janus/eval/25_run_probe_language_targets.sh
# Knobs: WIKI_DATE=20231101  MAX_DOCS=300

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/language_probe

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

echo "  language probe | $(date)"
$JANUS_PY ~/multimodal-peft/janus/eval/25_probe_language_targets.py
echo "  Done: $(date)"
