#!/bin/bash
#SBATCH --job-name=chain_eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=janus/tests/output/chain/slurm_eval_%j.out
#SBATCH --error=janus/tests/output/chain/slurm_eval_%j.err
#SBATCH --export=ALL
#
# One evaluation point of a sequential chain. Everything a single step needs is
# done inside ONE job, so the chain is 6 jobs per chain instead of ~18.
#
# Required env (set by run_chain.sh):
#   TAG        - e.g. chainA_s2_naruto
#   ADAPTER    - .../adapter_best of the step just trained
#   STEP       - 1, 2 or 3
#
# What is measured, and why each one:
#   * held-out PPL on BOTH text targets, every step. These are the rows of the
#     retention matrix: a target trained at step 1 must keep being scored after
#     steps 2 and 3, otherwise there is no forgetting measurement at all.
#   * the four public benchmarks (subsampled config), for the capabilities we
#     never train.
#   * image acquisition (CMMD / DirSim vs REAL style images). This is the
#     headline image metric of the paper. Note it is NOT interchangeable with
#     Gen-CMMD: after the image step, distance from the base means success, so
#     only distance from the target can tell us whether the style survives.
#   * Gen-CMMD vs base, STEP 1 ONLY. Before the image step it measures
#     cross-modal leakage and is interpretable; afterwards it conflates
#     acquisition with drift, so running it would produce a number we could
#     not read.

set -uo pipefail
source "$HOME/multimodal-peft/janus/chain/chain_config.sh"
mkdir -p "$EVAL_OUT/chain"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
PY="$HOME/.conda/envs/janus/bin/python"
export HF_HOME=/scratch-shared/tesr125338/hf_cache

: "${TAG:?set TAG}"; : "${ADAPTER:?set ADAPTER}"; : "${STEP:?set STEP}"
echo "=================================================================="
echo "  CHAIN EVAL | TAG=$TAG  STEP=$STEP"
echo "  ADAPTER=$ADAPTER"
echo "  started $(date)"
echo "=================================================================="

if [ ! -e "$ADAPTER" ]; then
  echo "  FATAL: adapter not found: $ADAPTER"; exit 2
fi

RC=0

# ---- 1. held-out PPL on every text target -------------------------------
echo; echo "---- [1/4] held-out PPL: Welsh ----"
env TAG="$TAG" ADAPTER="$ADAPTER" CORPUS=welsh \
    HF_DATASET_NAME="$WELSH_DATASET" HF_DATASET_CONFIG="$WELSH_CONFIG" \
    HF_TEXT_COL="$WELSH_TEXT_COL" MAX_DOCS="$WELSH_MAX_DOCS" \
    SEQ_LEN="$SEQ_LEN" TRAIN_WINDOWS="$TRAIN_WINDOWS" \
    VAL_WINDOWS="$VAL_WINDOWS" SEED="$SEED" \
    "$PY" "$REPO/janus/eval/35_target_ppl.py" || RC=1

echo; echo "---- [2/4] held-out PPL: medicine ----"
env TAG="$TAG" ADAPTER="$ADAPTER" CORPUS=medicine \
    HF_DATASET_NAME="$MED_DATASET" HF_DATASET_CONFIG="$MED_CONFIG" \
    HF_TEXT_COL="$MED_TEXT_COL" MAX_DOCS="$MED_MAX_DOCS" \
    SEQ_LEN="$SEQ_LEN" TRAIN_WINDOWS="$TRAIN_WINDOWS" \
    VAL_WINDOWS="$VAL_WINDOWS" SEED="$SEED" \
    "$PY" "$REPO/janus/eval/35_target_ppl.py" || RC=1

# ---- 2. untrained capabilities ------------------------------------------
echo; echo "---- [3/4] benchmarks (subsampled config) ----"
env TAG="$TAG" ADAPTER="$ADAPTER" \
    MMLU_N="$MMLU_N" POPE_N="$POPE_N" MME_N="$MME_N" \
    "$PY" "$REPO/janus/eval/07_forgetting_battery.py" || RC=1

# ---- 3. image acquisition vs REAL style ---------------------------------
# REUSE_IMAGES: the adapter-off reference is the same base model at every
# step, so generating it once and reusing saves ~80 generations per step.
# NOTE both 06 and 17 take JANUS_OUTPUT_DIR (not OUTPUT_DIR) and append their
# own subdirectory: 06 writes <root>/<adapterparent__adaptername>/metrics_panel.json,
# 17 writes <root>/gen_forgetting_lambda/gen_forgetting.json. 36_chain_matrix.py
# resolves those exact paths, so do not "simplify" them here.
echo; echo "---- [4/4] image acquisition (CMMD/DirSim vs real style) ----"
env ADAPTER="$ADAPTER" CAPTION_PREFIX="$CAPTION_PREFIX" \
    HF_DATASET_NAME="$IMAGE_DATASET" N_GEN="$N_GEN" EVAL_N="$N_GEN" \
    SEED="$SEED" REUSE_IMAGES=1 \
    JANUS_OUTPUT_DIR="$EVAL_OUT/chain/$TAG/image_metrics" \
    "$PY" "$REPO/janus/eval/06_metrics_panel.py" || RC=1

# ---- 4. cross-modal leakage, step 1 only --------------------------------
if [ "$STEP" = "1" ]; then
  echo; echo "---- [extra] Gen-CMMD vs base (leakage, interpretable only pre-image) ----"
  env ADAPTER="$ADAPTER" TAG="$TAG" \
      JANUS_OUTPUT_DIR="$EVAL_OUT/chain/$TAG" \
      "$PY" "$REPO/janus/eval/17_gen_forgetting_lambda.py" || RC=1
else
  echo; echo "---- [extra] Gen-CMMD skipped at step $STEP (not interpretable) ----"
fi

echo
echo "=================================================================="
echo "  CHAIN EVAL done | TAG=$TAG  rc=$RC  $(date)"
echo "=================================================================="
exit $RC
