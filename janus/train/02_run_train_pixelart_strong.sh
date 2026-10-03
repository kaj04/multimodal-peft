#!/bin/bash
#SBATCH --job-name=pixelart_strong
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=janus/train/output/slurm_%j.out
#SBATCH --error=janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# =====================================================================
#  Pixel-art LoRA, STRONG + UNCONDITIONAL (hyperparameter-driven).
#
#  The first pixel-art run learned the style only as a prompt trigger:
#  "Adapter | plain" produced no pixel art. The DiffusionDB captions fight the
#  target ("oil on canvas, photoreal, 8k, by <artist>"), AND the KL=0.5 term
#  pulls the adapter's output toward what the base does for those captions
#  (= photoreal). Two forces push to photoreal; only the target images push to
#  pixel -> weak compromise.
#
#  Key insight: pixelation is CONSTANT across every training image, regardless
#  of caption. So we don't need to clean the (chaotic, hard-to-clean) captions
#  -- we remove what FIGHTS the constant signal and add capacity/time:
#    - KL_LAMBDA = 0    (the main lever: stop regularising toward the base's
#                        photoreal default; also halves runtime = single fwd)
#    - LORA_RANK = 64   (was 32; more capacity to shift the unconditional prior)
#    - EPOCHS    = 8     no prefix, raw captions (CLEAN_CAPTIONS=0)
#
#  SUCCESS CRITERION: in 05c, the "Adapter | plain" column (no style word) must
#  show pixel art. Neutral SAMPLE_PROMPTS let us watch this emerge live in W&B.
#
#  If still weak: bump LORA_RANK=96; if it degrades content, add KL_LAMBDA=0.1.
# =====================================================================

mkdir -p $HOME/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Pixel-art STRONG (raw captions, KL=0, r=64, no prefix, 8 epochs)"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python
$JANUS_PY --version

export MODEL_ID="${MODEL_ID:-deepseek-community/Janus-Pro-7B}"
export RUN_NAME="${RUN_NAME:-gen_7b_pixelart_strong}"

# Dataset: DiffusionDB pixel-art via the parquet branch (the script load is
# deprecated). Same source as the failed run, but with cleaned captions.
export DATASET_SOURCE="${DATASET_SOURCE:-hf}"
export HF_DATASET_NAME="${HF_DATASET_NAME:-jainr3/diffusiondb-pixelart}"
export HF_REVISION="${HF_REVISION:-refs/convert/parquet}"
export HF_DATA_FILES="${HF_DATA_FILES:-2k_random_2k/train/0000.parquet}"
export HF_IMAGE_COL="${HF_IMAGE_COL:-image}"
export HF_TEXT_COL="${HF_TEXT_COL:-text}"

# THE levers (raw captions; let the constant pixelation signal win)
export CLEAN_CAPTIONS="${CLEAN_CAPTIONS:-0}"
export CAPTION_PREFIX="${CAPTION_PREFIX:-}"          # empty -> force unconditional

export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"               # keep = 2 * rank
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
export TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
export KL_LAMBDA="${KL_LAMBDA:-0}"                   # 0 = stop pulling toward photoreal default
export LR="${LR:-1e-4}"
export EPOCHS="${EPOCHS:-8}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export WARMUP_RATIO="0.05"
export ADAM_BETA1="0.9"
export ADAM_BETA2="0.95"
export WEIGHT_DECAY="0.0"
export GRAD_ACCUM="8"
export GRAD_CLIP="1.0"
export RECON_LOSS_WEIGHT="0.0"

# Eval / sampling. Neutral, prefix-free prompts so the live samples test the
# UNCONDITIONAL behaviour (does a plain prompt already come out pixelated?).
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
echo "  Next: 05c grid -> check the 'Adapter | plain' column is pixel art."
echo "================================================================"
