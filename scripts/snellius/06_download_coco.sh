#!/bin/bash
# =============================================================================
# 06_download_coco.sh  —  Download COCO 2017 (images + captions)
#
# Downloads:
#   - train2017 images   (~18 GB)
#   - val2017 images     (~1 GB)
#   - captions annotations (~241 MB)
#
# Output structure:
#   /scratch-shared/tesr125338/coco/
#     train2017/          (118,287 images)
#     val2017/            (5,000 images)
#     annotations/
#       captions_train2017.json
#       captions_val2017.json
#
# Usage:
#   sbatch scripts/snellius/06_download_coco.sh
# =============================================================================
#SBATCH --job-name=download_coco
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_coco_%j.out

set -e
mkdir -p logs

COCO_DIR="/scratch-shared/tesr125338/coco"
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

echo "=== Downloading COCO 2017 ==="
echo "    Target dir : $COCO_DIR"
echo "    Start time : $(date)"

# ── Annotations (~241 MB, fast) ───────────────────────────────────────────────
echo ""
echo "[1/3] Downloading annotations ..."
wget -q --show-progress \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
    -O annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
rm annotations_trainval2017.zip
echo "    Annotations extracted: $(ls annotations/ | wc -l) files"

# ── Val images (~1 GB, fast) ──────────────────────────────────────────────────
echo ""
echo "[2/3] Downloading val2017 images (~1 GB) ..."
wget -q --show-progress \
    http://images.cocodataset.org/zips/val2017.zip \
    -O val2017.zip
unzip -q val2017.zip
rm val2017.zip
echo "    val2017: $(ls val2017/ | wc -l) images"

# ── Train images (~18 GB, slow) ───────────────────────────────────────────────
echo ""
echo "[3/3] Downloading train2017 images (~18 GB) ..."
wget -q --show-progress \
    http://images.cocodataset.org/zips/train2017.zip \
    -O train2017.zip
unzip -q train2017.zip
rm train2017.zip
echo "    train2017: $(ls train2017/ | wc -l) images"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Download complete ==="
echo "    End time : $(date)"
echo "    Disk usage:"
du -sh "$COCO_DIR"/*/
