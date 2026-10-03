#!/bin/bash
#SBATCH --job-name=text_ppl_probe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=janus/tests/output/text_ppl_probe/slurm_%j.out
#SBATCH --error=janus/tests/output/text_ppl_probe/slurm_%j.err
#SBATCH --export=ALL

# Rank candidate text STYLES by base-Janus perplexity to pick the most
# distant one for the text-PEFT asymmetry experiment.
#
#   base model (no adapter):
#     sbatch janus/eval/11_run_probe_text_ppl.sh
#   plumbing check (4 windows/style, ~1 min):
#     sbatch --export=ALL,SMOKE=1 janus/eval/11_run_probe_text_ppl.sh
#   re-measure the SAME styles after a text LoRA (symmetry check):
#     sbatch --export=ALL,ADAPTER=/path/to/adapter_best janus/eval/11_run_probe_text_ppl.sh

mkdir -p $HOME/multimodal-peft/janus/tests/output/text_ppl_probe

echo "================================================================"
echo "  Text-style PPL probe"
echo "  ADAPTER = ${ADAPTER:-<base>}   SMOKE = ${SMOKE:-0}"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

$JANUS_PY ~/multimodal-peft/janus/eval/11_probe_text_ppl.py
echo "Done: $(date)"
