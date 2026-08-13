#!/bin/bash
#SBATCH --job-name=chain_img
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/chain/slurm_img_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/chain/slurm_img_%j.err
#SBATCH --export=ALL
#
# The IMAGE step of a sequential chain (anime style).
#
# Required env: RUN_NAME
# Optional env: INIT_ADAPTER (previous adapter_best; empty would mean the image
#               task is first, which neither of our two chains uses)
#
# CAPTION_PREFIX is forced empty by chain_config.sh. The Final recipe applies
# the style unconditionally, and a stray trigger prefix would silently make the
# chain's CMMD incomparable with every image number already in the paper.

set -uo pipefail
source "$HOME/multimodal-peft/janus/chain/chain_config.sh"
mkdir -p "$EVAL_OUT/chain"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
PY="$HOME/.conda/envs/janus/bin/python"
export HF_HOME=/scratch-shared/tesr125338/hf_cache

: "${RUN_NAME:?set RUN_NAME}"
INIT_ADAPTER="${INIT_ADAPTER:-}"

export HF_DATASET_NAME="$IMAGE_DATASET"
export EPOCHS="$IMAGE_EPOCHS"
export LORA_DROPOUT="$IMAGE_DROPOUT"
export INIT_ADAPTER

echo "=================================================================="
echo "  CHAIN IMAGE STEP | run=$RUN_NAME"
echo "  init_adapter=${INIT_ADAPTER:-<none, fresh from base>}"
echo "  r=$LORA_RANK alpha=$LORA_ALPHA lambda=$KL_LAMBDA epochs=$EPOCHS"
echo "  caption_prefix='${CAPTION_PREFIX}'  (must be empty)"
echo "  started $(date)"
echo "=================================================================="

if [ -n "$INIT_ADAPTER" ] && [ ! -e "$INIT_ADAPTER" ]; then
  echo "  FATAL: INIT_ADAPTER does not exist: $INIT_ADAPTER"; exit 2
fi

"$PY" "$REPO/janus/train/02_train_generation.py"
TRAIN_RC=$?

HIST="$TRAIN_OUT/$RUN_NAME/history.json"
if [ "$TRAIN_RC" -eq 0 ] && [ -f "$HIST" ]; then
  "$PY" - "$HIST" <<'PYCHK'
import json, sys
h = json.load(open(sys.argv[1]))
b, best = h.get("baseline_val_CE"), h.get("best_val_CE")
if b is None or best is None:
    print("  [check] history.json has no baseline/best val_CE; cannot verify.")
else:
    print(f"  [check] val_CE baseline={b:.4f} -> best={best:.4f} (gain {b-best:+.4f})")
    if (b - best) <= 0.005:
        print("  [check] *** WARNING: image step barely acquired the style. ***")
        print("  [check] Compare CMMD against the r=32/lambda=0.5 standalone")
        print("  [check] reference (.108) before reading any later step.")
PYCHK
fi

echo "  CHAIN IMAGE STEP done | rc=$TRAIN_RC | $(date)"
echo "  adapter_best -> $(readlink -f "$TRAIN_OUT/$RUN_NAME/adapter_best" 2>/dev/null)"
exit $TRAIN_RC
