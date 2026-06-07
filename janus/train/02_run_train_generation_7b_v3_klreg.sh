#!/bin/bash
#SBATCH --job-name=janus_7b_v3kl
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B LoRA generation v3 = v2-CAPACITY + KL-to-base regularizer
#
#  Hypothesis stack:
#    v1 (r=8, q/k/v/o)             -> failed: ratio 0.13% too low, degrades base
#    v2 (r=16, +MLP, LR 5e-5)      -> tests "more capacity fixes it"
#    v3 (v2 + KL(p_a || p_base))   -> tests "the adapter overwrites good base
#                                      capability; force it to STAY close to
#                                      the base distribution unless CE really
#                                      moves it"
#
#  Mechanism (inspired by RLHF KL penalty / knowledge distillation):
#    L = CE(p_adapter, target) + lambda * mean_pos KL(p_adapter || p_base)
#    where p_base = adapter-disabled forward on the SAME input, no_grad
#
#  Cost: ~2x transformer forward per step (we replay the base path under
#  disable_adapter() + no_grad()). Wall time ~16-18h on H100 for 8 epochs
#  on the 1141-image Naruto BLIP dataset; hence the 20h SBATCH time.
#
#  KL_LAMBDA = 0.5  (mid-range start; sweepable in a follow-up)
#    - too low -> degenerates to v2-CAPACITY
#    - too high -> adapter is too constrained, doesn't fit anything
#    - 0.1 to 1.0 is the sensible operating window
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B LoRA v3-KLREG  (v2-CAPACITY + KL-to-base reg)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# --- Run config (v3 = v2 + KL) ---
export MODEL_ID="deepseek-community/Janus-Pro-7B"
export RUN_NAME="gen_7b_naruto_v3_klreg"
export DATASET_SOURCE="hf"
export HF_DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_IMAGE_COL="image"
export HF_TEXT_COL="text"
export CAPTION_PREFIX="anime sprite in Naruto style, "

# Same capacity as v2 (the KL is ORTHOGONAL to the capacity fix; we want
# both effects in v3, so the comparison v2 vs v3 isolates the KL contribution).
export LORA_RANK="16"
export LORA_ALPHA="32"
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# KL-to-base
export KL_LAMBDA="0.5"

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
