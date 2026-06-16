#!/bin/bash
#SBATCH --job-name=pixelart_coco
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Pixel-art LoRA on SYNTHESIZED data: COCO photos -> crisp pixel art.
#
#  Why this beats the DiffusionDB-pixelart dataset (duplicates + adversarial
#  captions + mild "pixelated photo" look):
#    - COCO captions are clean CONTENT ("a cat sitting on a couch") -> no style
#      words fighting the target -> the style is learned UNCONDITIONALLY.
#    - COCO content is diverse (80 categories) -> the adapter generalises
#      (style over content), not "pixel character" only.
#    - the pixelation is crisp + strong (palette-reduced) -> bigger weight
#      shift, and PIXELART_SIZE/COLORS are a controllable distance dial.
#    - no duplicates; we have COCO on scratch already.
#  We train on train2017 so val2017 stays clean for the POPE understanding
#  probe in the forgetting battery.
#
#  Config = Naruto winning recipe (r=48, KL=0.5, no prefix), which learned an
#  unconditional style from clean captions. SUCCESS: 05c "Adapter | plain" is
#  pixel art. If weak: KL_LAMBDA=0.1; if pixelation too mild: PIXELART_SIZE=32
#  PIXELART_COLORS=16.
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Pixel-art on COCO (synthesized) | r=48 KL=0.5 train2017"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python
$JANUS_PY --version

export MODEL_ID="${MODEL_ID:-deepseek-community/Janus-Pro-7B}"
export RUN_NAME="${RUN_NAME:-gen_7b_pixelart_coco}"

# Data: COCO train2017 + on-the-fly pixel-art transform of the target image.
export DATASET_SOURCE="${DATASET_SOURCE:-coco}"
export COCO_SPLIT="${COCO_SPLIT:-train2017}"
export IMAGE_TRANSFORM="${IMAGE_TRANSFORM:-pixelart}"
export PIXELART_SIZE="${PIXELART_SIZE:-48}"
export PIXELART_COLORS="${PIXELART_COLORS:-24}"
export CLEAN_CAPTIONS="${CLEAN_CAPTIONS:-0}"     # COCO captions already clean
export CAPTION_PREFIX="${CAPTION_PREFIX:-}"       # empty -> unconditional

# LoRA = Naruto winning recipe
export LORA_RANK="${LORA_RANK:-48}"
export LORA_ALPHA="${LORA_ALPHA:-96}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
export TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
export KL_LAMBDA="${KL_LAMBDA:-0.5}"
export LR="${LR:-1e-4}"
export EPOCHS="${EPOCHS:-6}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export WARMUP_RATIO="0.05"
export ADAM_BETA1="0.9"
export ADAM_BETA2="0.95"
export WEIGHT_DECAY="0.0"
export GRAD_ACCUM="8"
export GRAD_CLIP="1.0"
export RECON_LOSS_WEIGHT="0.0"

# Eval / sampling: neutral prefix-free prompts -> live test of UNCONDITIONAL style
export EVAL_EVERY_STEPS="50"
export EVAL_MID_N="10"
export GEN_EVAL_N="20"
export EVAL_N="80"
export TRAIN_N="4000"
export SAMPLE_EVERY_STEPS="100"
export N_SAMPLE_PROMPTS="4"
export SAMPLE_PROMPTS="a cat;a castle on a hill;a portrait of a man;a sailboat on the sea"

export RESUME="${RESUME:-1}"
export SAVE_EVERY_STEPS="200"
export WANDB_PROJECT="janus-peft"

$JANUS_PY ~/multimodal-peft/janus/train/02_train_generation.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/multimodal-peft/janus/train/output/$RUN_NAME/"
echo "  Check: $RUN_NAME/transform_preview/ (pixelation strength)"
echo "  Then 05c -> 'Adapter | plain' column must be pixel art."
echo "================================================================"
