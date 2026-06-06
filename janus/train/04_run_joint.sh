#!/bin/bash
#SBATCH --job-name=janus_joint
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B JOINT image+text LoRA fine-tune
#  Submit one of these for each C0..C3:
#
#    sbatch --export=ALL,TEXT_BUCKET=none    janus/train/04_run_joint.sh
#    sbatch --export=ALL,TEXT_BUCKET=matched janus/train/04_run_joint.sh
#    sbatch --export=ALL,TEXT_BUCKET=near    janus/train/04_run_joint.sh
#    sbatch --export=ALL,TEXT_BUCKET=far     janus/train/04_run_joint.sh
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B JOINT fine-tune (TEXT_BUCKET=${TEXT_BUCKET:-none})"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# --- Defaults that the joint trainer expects (can be overridden via --export) ---
export MODEL_ID="${MODEL_ID:-deepseek-community/Janus-Pro-7B}"
export TEXT_BUCKET="${TEXT_BUCKET:-none}"
export TEXT_RATIO="${TEXT_RATIO:-0.5}"
export RUN_NAME="${RUN_NAME:-joint_7b_${TEXT_BUCKET}_r8}"

export DATASET_SOURCE="hf"
export HF_DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_IMAGE_COL="image"
export HF_TEXT_COL="text"
export CAPTION_PREFIX="anime sprite in Naruto style, "

# Training knobs (mirrors the 7B baseline run)
export EPOCHS="${EPOCHS:-8}"
export LR="${LR:-2e-5}"
export LR_SCHEDULE="cosine"
export WARMUP_RATIO="0.05"
export ADAM_BETA1="0.9"
export ADAM_BETA2="0.95"
export WEIGHT_DECAY="0.0"
export GRAD_ACCUM="8"
export GRAD_CLIP="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16"

export EVAL_EVERY_STEPS="50"
export EVAL_MID_N="10"
export GEN_EVAL_N="20"
export EVAL_N="80"
export TRAIN_N="4000"

export RESUME="1"
export SAVE_EVERY_STEPS="200"

export WANDB_PROJECT="janus-peft"

$JANUS_PY ~/multimodal-peft/janus/train/04_train_joint.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/multimodal-peft/janus/train/output/$RUN_NAME/"
echo "================================================================"
