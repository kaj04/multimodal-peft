#!/bin/bash
# =============================================================================
# 18_eval_caption_forgetting_r8.sh
# Captioning forgetting evaluation: baseline vs LoRA (run 15: r8/selfattn/drop005)
# on COCO val2017 (500 images by default).
#
# Metric: CIDEr, BLEU@4, ROUGE_L, METEOR
# A significant CIDEr drop indicates catastrophic forgetting after depth FT.
#
# Usage:
#   sbatch scripts/snellius/18_eval_caption_forgetting_r8.sh
#
#   # All 5000 COCO val images (slower, ~4-5h):
#   sbatch --export=ALL,MAX_IMAGES=0 scripts/snellius/18_eval_caption_forgetting_r8.sh
# =============================================================================
#SBATCH --job-name=4m_cap_forg
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_caption_forgetting_r8_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

# ── Extra packages installed on scratch (pycocoevalcap, pycocotools, nltk) ──
export PYTHONPATH=/scratch-shared/tesr125338/pip_packages:${PYTHONPATH}
export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs
cd "$REPO_ROOT"

# ── Paths ────────────────────────────────────────────────────────────────────
COCO_ROOT="/scratch-shared/tesr125338/coco/val2017"
COCO_ANN="/scratch-shared/tesr125338/coco/annotations/captions_val2017.json"
LORA_WEIGHTS="/scratch-shared/tesr125338/output/depth/nyu_labeled_r8_selfattn_drop005/lora_weights-best.pth"
OUTPUT_DIR="/scratch-shared/tesr125338/output/eval_forgetting/caption_r8"

FM_MODEL="EPFL-VILAB/4M-21_XL"
TOK_TEXT="ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"

# Max images: default 500, override with MAX_IMAGES env var (0 = all 5000)
MAX_IMAGES="${MAX_IMAGES:-500}"

echo "================================================================"
echo "  4M-21 Caption Forgetting Eval — r8/selfattn/drop005"
echo "  Dataset  : COCO val2017 (${MAX_IMAGES} images)"
echo "  LoRA     : ${LORA_WEIGHTS}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Started  : $(date)"
echo "================================================================"

python eval_caption_forgetting.py \
    --coco_root     "${COCO_ROOT}" \
    --coco_ann      "${COCO_ANN}" \
    --fm            "${FM_MODEL}" \
    --tok_text      "${TOK_TEXT}" \
    --lora_weights  "${LORA_WEIGHTS}" \
    --lora_rank     8 \
    --lora_modules  selfattn \
    --max_images    "${MAX_IMAGES}" \
    --output_dir    "${OUTPUT_DIR}" \
    --device        cuda \
    --dtype         bfloat16

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"

if [ -f "${OUTPUT_DIR}/summary.txt" ]; then
    cat "${OUTPUT_DIR}/summary.txt"
fi
