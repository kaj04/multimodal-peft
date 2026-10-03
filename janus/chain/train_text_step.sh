#!/bin/bash
#SBATCH --job-name=chain_txt
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=janus/tests/output/chain/slurm_txt_%j.out
#SBATCH --error=janus/tests/output/chain/slurm_txt_%j.err
#SBATCH --export=ALL
#
# One TEXT step of a sequential chain.
#
# Required env: RUN_NAME, CORPUS (welsh|medicine)
# Optional env: INIT_ADAPTER (empty for step 1, previous adapter_best otherwise)
#
# The time limit is deliberately generous: EPOCHS governs, and a job that
# finishes early is billed for the time actually used, not the time reserved.

set -uo pipefail
source "$HOME/multimodal-peft/janus/chain/chain_config.sh"
mkdir -p "$EVAL_OUT/chain"

module load 2023
module load Anaconda3/2023.07-2
source activate janus
PY="$HOME/.conda/envs/janus/bin/python"
export HF_HOME=/scratch-shared/tesr125338/hf_cache

: "${RUN_NAME:?set RUN_NAME}"; : "${CORPUS:?set CORPUS (welsh|medicine)}"
INIT_ADAPTER="${INIT_ADAPTER:-}"

case "$CORPUS" in
  welsh)
    export HF_DATASET_NAME="$WELSH_DATASET" HF_DATASET_CONFIG="$WELSH_CONFIG"
    export HF_TEXT_COL="$WELSH_TEXT_COL" MAX_DOCS="$WELSH_MAX_DOCS" ;;
  medicine)
    export HF_DATASET_NAME="$MED_DATASET" HF_DATASET_CONFIG="$MED_CONFIG"
    export HF_TEXT_COL="$MED_TEXT_COL" MAX_DOCS="$MED_MAX_DOCS" ;;
  *) echo "  FATAL: unknown CORPUS=$CORPUS"; exit 2 ;;
esac

# One document per row for both corpora (Welsh articles, PubMed articles).
export DOC_MODE=rows
export EPOCHS="$TEXT_EPOCHS"
export LORA_DROPOUT="$TEXT_DROPOUT"
export INIT_ADAPTER

# Same guard as the image step. 03 happens to default to the 7B and 02 to the
# 1B, which is why only the image direction broke, but a chain whose steps load
# different models cannot work at all and the failure is invisible until an
# adapter load rejects the width. Check it on both sides.
case "$MODEL_ID" in
  *7B*) : ;;
  *) echo "  FATAL: MODEL_ID='$MODEL_ID' is not the 7B model. Every step of a"
     echo "         chain must load the same model."
     exit 2 ;;
esac

echo "=================================================================="
echo "  CHAIN TEXT STEP | run=$RUN_NAME  corpus=$CORPUS"
echo "  MODEL_ID=$MODEL_ID"
echo "  init_adapter=${INIT_ADAPTER:-<none, fresh from base>}"
echo "  r=$LORA_RANK alpha=$LORA_ALPHA lambda=$KL_LAMBDA epochs=$EPOCHS lr=$LR"
echo "  windows=$TRAIN_WINDOWS val=$VAL_WINDOWS  resume=$RESUME"
echo "  started $(date)"
echo "=================================================================="

if [ -n "$INIT_ADAPTER" ] && [ ! -e "$INIT_ADAPTER" ]; then
  echo "  FATAL: INIT_ADAPTER does not exist: $INIT_ADAPTER"; exit 2
fi

"$PY" "$REPO/janus/train/03_train_text.py"
TRAIN_RC=$?

# Acquisition sanity check. A step that does not learn its target leaves the
# diagonal of the retention matrix empty, and without acquisition there is
# nothing whose loss could be called forgetting. We warn loudly rather than
# abort: a weak step is itself informative, and killing the chain here would
# throw away the later steps too.
HIST="$TRAIN_OUT/$RUN_NAME/history.json"
if [ "$TRAIN_RC" -eq 0 ] && [ -f "$HIST" ]; then
  "$PY" - "$HIST" <<'PYCHK'
import json, sys
h = json.load(open(sys.argv[1]))
b = h.get("baseline_val_CE")
best = h.get("best_val_CE")
if b is None or best is None:
    print("  [check] history.json has no baseline/best val_CE; cannot verify.")
else:
    gain = b - best
    print(f"  [check] val_CE baseline={b:.4f} -> best={best:.4f} (gain {gain:+.4f})")
    if gain <= 0.005:
        print("  [check] *** WARNING: this step barely acquired its target. ***")
        print("  [check] The diagonal entry is near-empty, so 'forgetting' of")
        print("  [check] this task is not measurable. Chain continues; flag it")
        print("  [check] in the analysis.")
PYCHK
fi

echo "  CHAIN TEXT STEP done | rc=$TRAIN_RC | $(date)"
echo "  adapter_best -> $(readlink -f "$TRAIN_OUT/$RUN_NAME/adapter_best" 2>/dev/null)"
exit $TRAIN_RC
