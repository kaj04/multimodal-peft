#!/bin/bash
#SBATCH --job-name=janus_joint
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=janus/train/output/slurm_%j.out
#SBATCH --error=janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B JOINT image+text LoRA fine-tune
#  Config inherits from the v5 image-only winning baseline. The ONLY
#  manipulated factor across runs is TEXT_BUCKET:
#
#    sbatch --export=ALL,TEXT_BUCKET=matched janus/train/04_run_joint.sh   # C1
#    sbatch --export=ALL,TEXT_BUCKET=near    janus/train/04_run_joint.sh   # C2
#    sbatch --export=ALL,TEXT_BUCKET=far     janus/train/04_run_joint.sh   # C3
#
#  C0 (image-only baseline) is already covered by gen_7b_naruto_v5,
#  no separate joint run needed for that condition.
#
#  Wall-time budget bumped to 20h: KL_LAMBDA>0 doubles the transformer
#  forward; alternating text+image steps adds another full forward.
# =====================================================================

mkdir -p $HOME/multimodal-peft/janus/train/output

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

# --- Defaults (overridable via --export) ---
export MODEL_ID="${MODEL_ID:-deepseek-community/Janus-Pro-7B}"
export TEXT_BUCKET="${TEXT_BUCKET:-matched}"
export TEXT_RATIO="${TEXT_RATIO:-0.5}"
export RUN_NAME="${RUN_NAME:-joint_7b_v5_${TEXT_BUCKET}}"

export DATASET_SOURCE="hf"
export HF_DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_IMAGE_COL="image"
export HF_TEXT_COL="text"

# v5 winning config -- mirrors gen_7b_naruto_v5 exactly except for the
# text-bucket addition (the only manipulated factor across C0..C3).
export CAPTION_PREFIX=""           # gate problem solved, no prefix needed
export LORA_RANK="48"
export LORA_ALPHA="96"             # 2 * rank
export LORA_DROPOUT="0.1"
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
export KL_LAMBDA="0.5"             # anti-degradation regularizer
export EPOCHS="6"
export LR="1e-4"
export LR_SCHEDULE="cosine"
export WARMUP_RATIO="0.05"
export ADAM_BETA1="0.9"
export ADAM_BETA2="0.95"
export WEIGHT_DECAY="0.0"
export GRAD_ACCUM="8"
export GRAD_CLIP="1.0"

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
