#!/bin/bash
#SBATCH --job-name=janus_7b_v4ag
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B LoRA generation v4-AGGRESSIVE
#
#  Synthesis of three diagnostics from v1/v2:
#    1) base 7B is already strong on Naruto-style; a "polite" adapter
#       (low rank, low LR) gets out-competed by the base
#    2) the prefix acted as a hard ON/OFF trigger -> adapter doesn't fire
#       on naturally-phrased prompts at inference
#    3) v2's moderate setup (r=16, LR 5e-5) avoided catastrophic outputs
#       (unlike v1) but did not clearly beat the base
#
#  v4-AGGRESSIVE pushes harder on three axes at once:
#    - LORA_RANK     16 -> 32           (doubled; ratio ~1.1%, 2.5x the 1B baseline)
#    - LORA_ALPHA    32 -> 64           (keep alpha = 2*rank convention)
#    - LR            5e-5 -> 8e-5       (more force per step to overpower base)
#    - LORA_DROPOUT  0.05 -> 0.10       (counter overfit risk from higher capacity)
#    - EPOCHS        8 -> 6             (with stronger steps, fewer epochs avoid late-stage overfit)
#    - CAPTION_PREFIX "" (empty)        (no hard-trigger gate; unconditional style transfer)
#
#  W&B samples logging is now ON in the trainer -> you can scrub through
#  the 4 fixed-prompt samples per checkpoint in the W&B UI without rsync.
#  If you see degradation creeping in around e.g. step 600, you can scancel
#  and use the latest good adapter (adapter_epochN with the lowest val_CE).
#
#  Honest hedge: if A still doesn't clearly beat the base 7B, the result
#  itself is informative -- documents the LoRA + small-data + strong-base
#  limitation, which is a legitimate finding for the thesis.
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B LoRA v4-AGGRESSIVE  (r=32, LR 8e-5, no prefix, dropout 0.1)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# --- Run config (v4-AGGRESSIVE) ---
export MODEL_ID="deepseek-community/Janus-Pro-7B"
export RUN_NAME="gen_7b_naruto_v4_aggressive"
export DATASET_SOURCE="hf"
export HF_DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_IMAGE_COL="image"
export HF_TEXT_COL="text"

# No prefix during training (key change vs v2/v3)
export CAPTION_PREFIX=""

# Aggressive capacity
export LORA_RANK="32"
export LORA_ALPHA="64"
export LORA_DROPOUT="0.1"
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# No KL reg (clean comparison vs v2; v3 already tests the KL axis)
export KL_LAMBDA="0.0"

# Optimisation
export EPOCHS="6"
export LR="8e-5"
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
