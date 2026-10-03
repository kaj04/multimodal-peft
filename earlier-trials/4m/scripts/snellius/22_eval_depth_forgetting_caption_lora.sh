#!/bin/bash
# =============================================================================
# 22_eval_depth_forgetting_caption_lora.sh
# Depth forgetting evaluation: does caption FT (run 20) forget depth estimation?
#
# Runs eval_depth.py (NYU val, 654 images) with the caption LoRA weights.
# Compares depth metrics to the baseline — a significant drop in AbsRel / RMSE
# indicates catastrophic forgetting of depth.
#
# Run order for a complete caption FT evaluation:
#   sbatch scripts/snellius/21_eval_caption_coco.sh              (caption quality)
#   sbatch scripts/snellius/22_eval_depth_forgetting_caption_lora.sh  (depth forgetting metrics)
#   # Viz notebooks:
#   jupytext --to notebook notebooks/viz_caption_coco.py
#   jupytext --to notebook notebooks/viz_depth_forgetting_caption_lora.py
#
# Runtime: ~1h on 1× A100 (654 images, ~5s/img including DiVAE decode)
#
# Usage:
#   sbatch scripts/snellius/22_eval_depth_forgetting_caption_lora.sh
# =============================================================================
#SBATCH --job-name=depth_forget_cap
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/depth_forgetting_caption_lora_%j.out

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

export HF_HOME=/scratch-shared/tesr125338/hf_cache
REPO_ROOT="$HOME/4m-peft"

mkdir -p logs
cd "$REPO_ROOT"

echo "================================================================"
echo "  Depth Forgetting Eval — Caption LoRA r4/selfattn (run 20)"
echo "  Dataset  : NYUv2 labeled val (654 images)"
echo "  LoRA     : coco_rgb_r4_selfattn_drop01/lora_weights-best.pth"
echo "  Output   : /scratch-shared/tesr125338/output/eval_depth/depth_forgetting_caption_lora_r4"
echo "  Started  : $(date)"
echo "================================================================"

python eval_depth.py \
    --dataset        nyu \
    --dataset_root   /scratch-shared/tesr125338/nyu_raw/test \
    --lora_weights   /scratch-shared/tesr125338/output/caption/coco_rgb_r4_selfattn_drop01/lora_weights-best.pth \
    --lora_rank      4 \
    --lora_modules   selfattn \
    --output_dir     /scratch-shared/tesr125338/output/eval_depth \
    --run_name       depth_forgetting_caption_lora_r4 \
    --divae_steps    50 \
    --device         cuda \
    --dtype          bfloat16

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Results → /scratch-shared/tesr125338/output/eval_depth/depth_forgetting_caption_lora_r4/"
echo "    results.json   ← per-image + aggregate AbsRel/RMSE/δ1/δ2/δ3"
echo "    summary.txt    ← human-readable table"
echo "  Compare against baseline NYU results to quantify forgetting."
echo "================================================================"
