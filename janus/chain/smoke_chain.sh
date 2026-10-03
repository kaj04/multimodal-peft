#!/bin/bash
#SBATCH --job-name=chain_smoke
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=janus/tests/output/chain/slurm_smoke_%j.out
#SBATCH --error=janus/tests/output/chain/slurm_smoke_%j.err
#SBATCH --export=ALL
#
# SMOKE TEST -- run this before the real chains. Two tiny text steps, ~300
# windows and one epoch each. It is not a scientific run; it exists to answer
# three questions that would otherwise only surface after ~24 GPU-hours:
#
#   1. Does INIT_ADAPTER actually load the weights? The load helper aborts if
#      mean|lora_B| is still zero afterwards, so a silent fresh start is
#      impossible. Look for the "INIT_ADAPTER <-" block in the log.
#
#   2. Is the second step's starting point really the first step's model?
#      Step 2 prints its own baseline val_CE on the NEW corpus before training.
#      If chaining worked, that baseline belongs to the step-1 model, not to
#      the base. We print both so the difference is visible.
#
#   3. Can each corpus actually supply the windows we ask for? TRAIN_WINDOWS
#      is 6000 in the real config. If a corpus yields fewer, later steps would
#      silently see less data than earlier ones, and "forgetting" would be
#      confounded with exposure -- the corpus-size artefact this project has
#      already documented once.

set -uo pipefail
source "$HOME/multimodal-peft/janus/chain/chain_config.sh"
mkdir -p "$EVAL_OUT/chain"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
PY="$HOME/.conda/envs/janus/bin/python"
export HF_HOME=/scratch-shared/tesr125338/hf_cache

# tiny overrides -- everything structural stays as in chain_config.sh
export TRAIN_WINDOWS=300
export VAL_WINDOWS=50
export EPOCHS=1
export LORA_DROPOUT="$TEXT_DROPOUT"
export DOC_MODE=rows
export MAX_TRAIN_MINUTES=25

S1="smoke_s1_welsh"
S2="smoke_s2_medicine"
A1="$TRAIN_OUT/$S1/adapter_best"

echo "=================================================================="
echo "  SMOKE 1/2 : welsh, fresh from base"
echo "=================================================================="
env RUN_NAME="$S1" INIT_ADAPTER= \
    HF_DATASET_NAME="$WELSH_DATASET" HF_DATASET_CONFIG="$WELSH_CONFIG" \
    HF_TEXT_COL="$WELSH_TEXT_COL" MAX_DOCS=4000 \
    "$PY" "$REPO/janus/train/03_train_text.py"
RC1=$?
[ "$RC1" -ne 0 ] && { echo "  smoke step 1 FAILED rc=$RC1"; exit $RC1; }

echo
echo "=================================================================="
echo "  SMOKE 2/2 : medicine, INIT_ADAPTER from step 1"
echo "  watch for 'INIT_ADAPTER <-' and a non-zero mean|lora_B| after"
echo "=================================================================="
env RUN_NAME="$S2" INIT_ADAPTER="$A1" \
    HF_DATASET_NAME="$MED_DATASET" HF_DATASET_CONFIG="$MED_CONFIG" \
    HF_TEXT_COL="$MED_TEXT_COL" MAX_DOCS=4000 \
    "$PY" "$REPO/janus/train/03_train_text.py"
RC2=$?

echo
echo "=================================================================="
echo "  SMOKE SUMMARY"
echo "=================================================================="
"$PY" - "$TRAIN_OUT/$S1/history.json" "$TRAIN_OUT/$S2/history.json" <<'PYSUM'
import json, sys, os
for label, path in (("step1 welsh", sys.argv[1]), ("step2 medicine", sys.argv[2])):
    if not os.path.isfile(path):
        print(f"  {label:16} history.json MISSING"); continue
    h = json.load(open(path))
    b, best = h.get("baseline_val_CE"), h.get("best_val_CE")
    print(f"  {label:16} baseline_val_CE={b}  best_val_CE={best}")
print()
print("  What to check:")
print("   * step 2 log shows 'INIT_ADAPTER <-' with after-load mean|lora_B| > 0")
print("   * step 2 baseline_val_CE is the step-1 model on medicine, not the base")
print("   * neither step warns about too few windows")
PYSUM

echo "  smoke done | rc1=$RC1 rc2=$RC2 | $(date)"
exit $RC2
