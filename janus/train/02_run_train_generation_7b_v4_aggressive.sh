#!/bin/bash
#SBATCH --job-name=janus_7b_v4ag
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B LoRA generation v4 = AGGRESSIVE + KL-reg + no prefix
#
#  Synthesis of diagnostics from v1, v2, v3 + supervisor meeting:
#    v1 (r=8, prefix)              -> catastrophic outputs (blob faces)
#    v2 (r=16, prefix)             -> coherent but didn't clearly beat base;
#                                     gate problem on natural prompts
#    v3 (v2 + KL_LAMBDA=0.5)       -> KL RESOLVED the degradation on prefix
#                                     prompts (Sakura no longer broken;
#                                     glasses-Naruto preserved). Gate problem
#                                     persists on natural prompts.
#
#  v4 combines the WINS from each axis simultaneously:
#    - LORA_RANK     16 -> 32          (more capacity to actually learn style)
#    - LORA_ALPHA    32 -> 64          (preserve alpha = 2*rank)
#    - LORA_DROPOUT  0.05 -> 0.10      (mitigate overfit at higher rank/LR)
#    - LR            5e-5 -> 8e-5      (more force per step)
#    - EPOCHS        8 -> 6            (compensate stronger steps -> fewer epochs)
#    - CAPTION_PREFIX  ""              (fix the gate problem: no hard trigger)
#    - KL_LAMBDA     0.5 (KEPT)        (proven anti-degradation regularizer)
#
#  Conceptual bet: aggressive capacity + push gives the adapter the AUTHORITY
#  to learn unconditional style transfer (since no prefix gates it), while
#  the KL regularizer anchors it so the new capacity doesn't overwrite
#  good base behaviour (the failure mode of v1/v2). The two forces are in
#  productive tension -- CE pulls toward dataset style, KL pulls back to base.
#  Sweet spot: adapter learns when CE signal is strong, defers when not.
#
#  W&B sample logging ON: scroll-through visual trace per checkpoint in the
#  UI (no rsync needed). If degradation creeps in at e.g. step 800, scancel
#  and keep the best adapter_epochN by val_CE + visual inspection.
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B LoRA v4  (r=32, LR 8e-5, dropout 0.1, no prefix, KL_LAMBDA=0.5)"
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

# KL-to-base regularizer (preserved from v3, which empirically RESOLVED the
# v2 degradation problem on prefix prompts -- AFTER no longer noticeably
# worse than BEFORE on the Sakura/red-panda-style canonical outputs).
# Hypothesis: KL + aggressive setup work together -- the aggressive capacity
# gives the adapter enough power to actually learn style transfer; the KL
# regularizer prevents that power from overwriting good base capability.
export KL_LAMBDA="0.5"

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
