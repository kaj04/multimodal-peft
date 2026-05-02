#!/bin/bash
# =============================================================================
# 08_eval_diode.sh  —  Zero-shot DIODE depth evaluation (4M-21 pretrained)
#
# Reproduces Table 1 of the 4M paper:
#   4M-21 XL  depth L1 (normalized) = 0.68
#
# No LoRA, no fine-tuning — original 4M-21 XL weights only.
#
# Usage:
#   # Full DIODE val (771 images, ~4 h on one A100):
#   sbatch scripts/snellius/08_eval_diode.sh
#
#   # Quick smoke-test (50 images, ~15 min):
#   sbatch --export=ALL,MAX_IMAGES=50 scripts/snellius/08_eval_diode.sh
#
# Output:
#   output/eval_diode/results.json   — per-image L1 + aggregate stats
#   output/eval_diode/summary.txt    — human-readable table vs paper
#
# Monitor:
#   tail -f logs/eval_diode_<jobid>.out
# =============================================================================
#SBATCH --job-name=4m_diode_eval
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=5:00:00
#SBATCH --output=logs/eval_diode_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs

cd "$REPO_ROOT"

# DIODE val root — adjust if stored elsewhere on scratch
DIODE_ROOT="/scratch-shared/tesr125338/diode/val"

# Optional: cap images for a quick test (override via --export MAX_IMAGES=50)
MAX_IMAGES_ARG=""
if [ -n "${MAX_IMAGES:-}" ]; then
    MAX_IMAGES_ARG="--max_images ${MAX_IMAGES}"
    echo "  → Smoke-test mode: evaluating first ${MAX_IMAGES} images"
fi

echo "================================================================"
echo "  4M-21 Zero-Shot DIODE Depth Evaluation"
echo "================================================================"
echo "  DIODE root  : ${DIODE_ROOT}"
echo "  FM model    : EPFL-VILAB/4M-21_XL"
echo "  DiVAE steps : 50"
echo "  Started     : $(date)"
echo "================================================================"

python eval_diode.py \
    --diode_root  "${DIODE_ROOT}" \
    --fm          EPFL-VILAB/4M-21_XL \
    --tok_depth   EPFL-VILAB/4M_tokenizers_depth_8k_224-448 \
    --tok_text    ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json \
    --output_dir  output/eval_diode \
    --dtype       bfloat16 \
    --divae_steps 50 \
    --device      cuda \
    --seed        0 \
    ${MAX_IMAGES_ARG}

echo ""
echo "================================================================"
echo "  Job finished: $(date)"
echo "================================================================"
echo ""
echo "--- summary.txt ---"
cat output/eval_diode/summary.txt
