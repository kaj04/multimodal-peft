#!/bin/bash
#SBATCH --job-name=text_poetry
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=janus/train/output/slurm_%j.out
#SBATCH --error=janus/train/output/slurm_%j.err
#SBATCH --export=ALL

# Text (poetry) LoRA on Janus-Pro-7B. The wall-clock budget is enforced by
# MAX_TRAIN_MINUTES (clean stop + final eval/save), NOT the SLURM --time.
#   sbatch janus/train/03_run_train_text_poetry.sh
#
# All knobs are ${VAR:-default} so an sbatch --export overrides them WITHOUT a
# hard export silently winning (the bug that wrecked the rank/KL sweeps).

mkdir -p $HOME/multimodal-peft/janus/train/output

echo "================================================================"
echo "  Janus-Pro-7B TEXT LoRA (poetry)"
echo "  RUN_NAME=${RUN_NAME:-text_7b_poetry_v1}  budget=${MAX_TRAIN_MINUTES:-150} min"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python
$JANUS_PY --version

export MODEL_ID="${MODEL_ID:-deepseek-community/Janus-Pro-7B}"
export RUN_NAME="${RUN_NAME:-text_7b_poetry_v1}"
export HF_DATASET_NAME="${HF_DATASET_NAME:-biglam/gutenberg-poetry-corpus}"

# LoRA config = user's choice (lighter than v5: r=16/alpha=32) + KL=0.5 (symmetric)
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
export KL_LAMBDA="${KL_LAMBDA:-0.5}"

export LR="${LR:-1e-4}"
export EPOCHS="${EPOCHS:-3}"
export SEQ_LEN="${SEQ_LEN:-512}"
export TRAIN_WINDOWS="${TRAIN_WINDOWS:-6000}"
export VAL_WINDOWS="${VAL_WINDOWS:-200}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export MAX_TRAIN_MINUTES="${MAX_TRAIN_MINUTES:-150}"   # ~2.5h, well inside --time

export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"
export SAMPLE_EVERY_STEPS="${SAMPLE_EVERY_STEPS:-150}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"
export RESUME="${RESUME:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-janus-peft}"

$JANUS_PY ~/multimodal-peft/janus/train/03_train_text.py
TRAIN_RC=$?

echo ""
echo "================================================================"
echo "  Done: $(date)  (train exit=$TRAIN_RC)"
echo "  Output: ~/multimodal-peft/janus/train/output/$RUN_NAME/"
echo "  adapter_best -> $(readlink -f ~/multimodal-peft/janus/train/output/$RUN_NAME/adapter_best 2>/dev/null)"
echo "================================================================"

# Propagate the real exit code so a crashed training marks the SLURM job FAILED
# and the chained afterok evals do NOT run on a missing adapter.
exit $TRAIN_RC
