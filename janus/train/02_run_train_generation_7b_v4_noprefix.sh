#!/bin/bash
#SBATCH --job-name=janus_7b_v4np
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B LoRA generation v4 = v2-CAPACITY but WITHOUT prefix
#
#  Diagnostic from v2 qualitative pairs:
#    - With prefix:    base 7B already strong in Naruto-style, adapter
#                      sometimes degrades it
#    - Without prefix: adapter doesn't fire at all (img 1 unchanged)
#                      => adapter learned the prefix as a HARD TRIGGER, not
#                      as a style transfer signal
#
#  Hypothesis v4: removing the prefix from training forces the adapter to
#    learn UNCONDITIONAL style transfer (every input -> Naruto style),
#    instead of "gated by the trigger phrase". Lose-side: model loses
#    ability to NOT-do-Naruto style (use base 7B for that). Win-side:
#    consistent activation, no prefix-dependency confusion, cleaner
#    interpretation of "is the adapter learning style?".
#
#  Variable isolation: this run differs from v2-CAPACITY ONLY in
#  CAPTION_PREFIX="" (empty). Same rank, MLP modules, LR, epochs.
#  -> a v2 vs v4 comparison cleanly attributes any difference to the
#  prefix factor.
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B LoRA v4-NOPREFIX  (v2 config minus prefix)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# --- Run config (v4 = v2 minus prefix) ---
export MODEL_ID="deepseek-community/Janus-Pro-7B"
export RUN_NAME="gen_7b_naruto_v4_noprefix"
export DATASET_SOURCE="hf"
export HF_DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_IMAGE_COL="image"
export HF_TEXT_COL="text"

# THE key change: no prefix during training
export CAPTION_PREFIX=""

# Same capacity as v2 (isolate the prefix variable)
export LORA_RANK="16"
export LORA_ALPHA="32"
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# No KL reg (we want a clean v2-without-prefix comparison; KL is v3's thing)
export KL_LAMBDA="0.0"

# Optimisation (same as v2-CAPACITY)
export EPOCHS="8"
export LR="5e-5"
export LR_SCHEDULE="cosine"
export WARMUP_RATIO="0.05"
export ADAM_BETA1="0.9"
export ADAM_BETA2="0.95"
export WEIGHT_DECAY="0.0"
export GRAD_ACCUM="8"
export GRAD_CLIP="1.0"
export RECON_LOSS_WEIGHT="0.0"

# Eval / sampling
export EVAL_EVERY_STEPS="50"
export EVAL_MID_N="10"
export GEN_EVAL_N="20"
export EVAL_N="80"
export TRAIN_N="4000"
export SAMPLE_EVERY_STEPS="200"
export N_SAMPLE_PROMPTS="4"

# Resume / checkpoint
export RESUME="1"
export SAVE_EVERY_STEPS="200"

export WANDB_PROJECT="janus-peft"

$JANUS_PY ~/multimodal-peft/janus/train/02_train_generation.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/multimodal-peft/janus/train/output/$RUN_NAME/"
echo "  Samples: ~/multimodal-peft/janus/train/output/$RUN_NAME/samples/"
echo "================================================================"
