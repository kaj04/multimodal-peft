#!/bin/bash
#SBATCH --job-name=janus_7b_v2cap
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B LoRA generation v2-CAPACITY
#
#  Diagnosis of v1 (gen_7b_naruto_r8_prefix_cosine):
#    - trainable ratio was 0.13% (vs 0.44% on 1B baseline that worked)
#    - qualitative output worse than the base 7B itself on Naruto prompts
#    - prefix dependency: adapter only "fires" when CAPTION_PREFIX present
#
#  v2-CAPACITY changes (Ivan's hypothesis: not enough trainable capacity):
#    - LORA_RANK 8 -> 16
#    - TARGET_MODULES adds the MLP block (gate_proj, up_proj, down_proj)
#      on top of the attention block (q/k/v/o_proj). Standard modern LoRA
#      recipe on Llama-family models (Janus inherits the architecture).
#    - LR 2e-5 -> 5e-5 (more capacity tolerates a higher LR; we also need
#      a bigger step size to actually move the larger Delta-W subspace
#      within the same number of training steps)
#    - EPOCHS 15 -> 8 (with more capacity convergence is faster; val_CE
#      curve from this run will tell us if we need more or fewer)
#    - SAMPLE_EVERY_STEPS 200 -> save 4 fixed-prompt images every 200 steps
#      under output/RUN/samples/step_XXXXXX/ for an eye-trace of progress
#
#  Hypothesis we're testing:
#    "the v1 7B was capacity-starved; bringing the trainable ratio back to
#    ~0.7-1% restores the same kind of style-transfer behaviour we saw on 1B"
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B LoRA v2-CAPACITY  (Naruto, hybrid prefix)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# --- Run config (v2-CAPACITY) ---
export MODEL_ID="deepseek-community/Janus-Pro-7B"
export RUN_NAME="gen_7b_naruto_v2_capacity"
export DATASET_SOURCE="hf"
export HF_DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_IMAGE_COL="image"
export HF_TEXT_COL="text"
export CAPTION_PREFIX="anime sprite in Naruto style, "

# Capacity: r=16 + MLP modules. Should bring trainable ratio to ~0.7-1%.
export LORA_RANK="16"
export LORA_ALPHA="32"
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Optimisation: higher LR, fewer epochs (more capacity = faster convergence)
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
