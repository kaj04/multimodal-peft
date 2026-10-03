#!/bin/bash
# =============================================================================
# 06b_download_coco_panoptic.sh  —  Download COCO 2017 panoptic annotations
#
# Downloads the panoptic segmentation annotations (not included in 06_download_coco.sh).
# Required for multi-task LoRA experiment (caption + semseg).
#
# Downloads (~2 GB total):
#   - panoptic_annotations_trainval2017.zip  (~821 MB) — JSON + PNG masks
#
# Output structure (appended to existing COCO dir):
#   /scratch-shared/tesr125338/coco/annotations/
#     panoptic_train2017.json       ← segment_id → category_id mappings
#     panoptic_val2017.json
#     panoptic_train2017/           ← per-image panoptic PNG masks
#       000000123456.png
#       ...
#     panoptic_val2017/             ← per-image panoptic PNG masks
#
# Usage:
#   sbatch scripts/snellius/06b_download_coco_panoptic.sh
# =============================================================================
#SBATCH --job-name=download_panoptic
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_coco_panoptic_%j.out

set -e
mkdir -p logs

COCO_DIR="/scratch-shared/tesr125338/coco"
cd "$COCO_DIR"

echo "=== Downloading COCO 2017 panoptic annotations ==="
echo "    Target dir : $COCO_DIR/annotations/"
echo "    Start time : $(date)"

echo ""
echo "[1/1] Downloading panoptic annotations (~821 MB) ..."
wget -q --show-progress \
    http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip \
    -O panoptic_annotations_trainval2017.zip

echo "    Extracting ..."
unzip -q panoptic_annotations_trainval2017.zip
rm panoptic_annotations_trainval2017.zip

echo "    panoptic_train2017/ : $(ls annotations/panoptic_train2017/ | wc -l) PNG files"
echo "    panoptic_val2017/   : $(ls annotations/panoptic_val2017/   | wc -l) PNG files"
echo ""
echo "=== Download complete ==="
echo "    End time : $(date)"
echo "    Next step: sbatch scripts/snellius/24_tokenize_coco_semseg.sh"
