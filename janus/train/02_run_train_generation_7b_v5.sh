#!/bin/bash
#SBATCH --job-name=janus_7b_v5
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Janus-Pro-7B LoRA generation v5 = v4 pushed further on TWO axes only
#
#  v4 was the first run with positive CLIP DirSim (+0.157) and strong
#  KID/CMMD/DINO improvements -> the aggressive direction is working.
#  v5 tests whether there is still margin on the same axis, isolating
#  to the TWO variables that drove the v4 win (capacity and step size).
#
#  Changes vs v4 (EVERYTHING else identical):
#    - LORA_RANK   32 -> 48     (+50% capacity; trainable ratio ~1.6%)
#    - LR          8e-5 -> 1e-4 (+25% force per step)
#
#  Held constant from v4:
#    - LORA_ALPHA          = 2 * rank = 96
#    - LORA_DROPOUT        = 0.10
#    - KL_LAMBDA           = 0.5
#    - TARGET_MODULES      = q,k,v,o + gate,up,down (attention + MLP)
#    - CAPTION_PREFIX      = "" (gate problem stayed solved in v4)
#    - EPOCHS              = 6
#    - LR_SCHEDULE         = cosine, 5% warmup
#    - ADAM betas          = (0.9, 0.95)
#
#  Decision tree on result:
#    v5 >= v4 metrics -> use v5 config for the rank sweep
#    v5 < v4 metrics  -> use v4 config for the rank sweep
#  Either outcome is informative (validates concavity of the response
#  surface in the (rank, LR) plane).
# =====================================================================

mkdir -p /home/fcolasurdo/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B LoRA v5  (r=48, LR 1e-4; KL=0.5 dropout=0.10 kept from v4)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python
echo "  python = $JANUS_PY"
$JANUS_PY --version

# --- Run config (v5 = v4 + (rank, LR) bump) ---
# Variables the rank sweep overrides use ${VAR:-default} so a --export on the
# sbatch line takes precedence; a bare launch keeps the v5 defaults.
# (Previously these were hard exports that silently overrode --export, so the
# rank-sweep for-loop had no effect and every run was rank=48 / same RUN_NAME.)
export MODEL_ID="deepseek-community/Janus-Pro-7B"
export RUN_NAME="${RUN_NAME:-gen_7b_naruto_v5}"
export DATASET_SOURCE="${DATASET_SOURCE:-hf}"
export HF_DATASET_NAME="${HF_DATASET_NAME:-lambdalabs/naruto-blip-captions}"
export HF_IMAGE_COL="${HF_IMAGE_COL:-image}"
export HF_TEXT_COL="${HF_TEXT_COL:-text}"

export CAPTION_PREFIX="${CAPTION_PREFIX:-}"

# Rank-sweep variables (overridable). Default = v5 (rank 48).
export LORA_RANK="${LORA_RANK:-48}"
export LORA_ALPHA="${LORA_ALPHA:-96}"   # keep = 2 * rank when overriding
export LR="${LR:-1e-4}"

# Inherited unchanged from v4
export LORA_DROPOUT="0.1"
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
# KL_LAMBDA and EPOCHS overridable (KL sweep uses ${KL_LAMBDA}; default = v5).
export KL_LAMBDA="${KL_LAMBDA:-0.5}"
export EPOCHS="${EPOCHS:-6}"
export LR_SCHEDULE="cosine"
export WARMUP_RATIO="0.05"
export ADAM_BETA1="0.9"
export ADAM_BETA2="0.95"
export WEIGHT_DECAY="0.0"
export GRAD_ACCUM="8"
export GRAD_CLIP="1.0"
# Overridable so a --export can add the Anole-style reconstruction term
# (RECON_LOSS_WEIGHT=0.5) as a competitor in the regularizer ablation.
export RECON_LOSS_WEIGHT="${RECON_LOSS_WEIGHT:-0.0}"

# Eval / sampling
export EVAL_EVERY_STEPS="50"
export EVAL_MID_N="10"
export GEN_EVAL_N="20"
export EVAL_N="80"
export TRAIN_N="4000"
export SAMPLE_EVERY_STEPS="100"
export N_SAMPLE_PROMPTS="6"

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
