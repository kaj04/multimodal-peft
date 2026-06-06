#!/bin/bash
#SBATCH --job-name=janus_train_7b
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B LoRA generation fine-tune on Naruto BLIP captions
#  - hybrid prompt templating with trigger phrase ("anime sprite in
#    Naruto style, ")
#  - LR 2e-5 cosine with 5% warmup (lower than 1B because 7B is more
#    sensitive)
#  - AdamW beta2=0.95 (Janus paper recipe)
#  - r=8 alpha=16, 15 epochs on ~1100 train pairs
#  - resume-capable: re-sbatch the same script if it hits the time limit
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B LoRA GENERATION fine-tune (Naruto, hybrid prefix)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# --- Run config ---
export MODEL_ID="deepseek-community/Janus-Pro-7B"
export RUN_NAME="gen_7b_naruto_r8_prefix_cosine"
export DATASET_SOURCE="hf"
export HF_DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_IMAGE_COL="image"
export HF_TEXT_COL="text"

# Hybrid prefix templating - trigger phrase prepended to every training caption
export CAPTION_PREFIX="anime sprite in Naruto style, "

# Optimisation
export EPOCHS="15"
export LR="2e-5"
export LR_SCHEDULE="cosine"
export WARMUP_RATIO="0.05"
export ADAM_BETA1="0.9"
export ADAM_BETA2="0.95"
export WEIGHT_DECAY="0.0"
export GRAD_ACCUM="8"
export GRAD_CLIP="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16"
export RECON_LOSS_WEIGHT="0.0"   # plain CE (Janus native objective)

# Eval cadence
export EVAL_EVERY_STEPS="50"
export EVAL_MID_N="10"
export GEN_EVAL_N="20"
export EVAL_N="80"
export TRAIN_N="4000"            # capped by dataset size internally

# Resume / checkpoint
export RESUME="1"
export SAVE_EVERY_STEPS="200"

export WANDB_PROJECT="janus-peft"

$JANUS_PY ~/multimodal-peft/janus/train/02_train_generation.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/multimodal-peft/janus/train/output/$RUN_NAME/"
echo "================================================================"
