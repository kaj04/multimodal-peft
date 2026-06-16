#!/bin/bash
# janus/run_text_sweep.sh
#
# Text (poetry) sweep launcher for the reverse-direction modularity evidence.
# For each value of the swept variable, chain:  train poetry  ->  light
# forgetting battery (afterok on the training).
#
# Strategy (sequential, not quadratic): run the RANK sweep first to find the
# best rank, THEN the LAMBDA sweep at that rank.
#
#   tonight  (rank sweep, fixed KL):
#     bash janus/run_text_sweep.sh
#   later    (lambda sweep at the chosen rank, e.g. 16):
#     SWEEP=lambda FIXED_RANK=16 bash janus/run_text_sweep.sh
#
# Each point's TASK GAIN comes from its training history.json (poetry val_CE);
# each point's FORGETTING comes from its light battery vs the base_light
# reference. Run from the LOGIN node.

set -euo pipefail
REPO="$HOME/multimodal-peft"

SWEEP="${SWEEP:-rank}"                 # rank | lambda
EPOCHS="${EPOCHS:-3}"
MAXMIN="${MAXMIN:-120}"
BASE_BATTERY="${BASE_BATTERY:-1}"      # 1 = also run the base (no-adapter) reference

# Light battery sizes (for the curve; full battery later on the winner)
BMMLU="${BMMLU:-570}"; BPOPE="${BPOPE:-1000}"; BMME="${BMME:-600}"

# Image-gen reference: the battery covers text-LM/knowledge/UNDERSTANDING but
# NOT generation. So per point we also generate a few fixed prompts base-vs-
# adapter (13_t2i) -> generation-degradation check + an illustrative grid.
GEN_REF="${GEN_REF:-1}"
GEN_PROMPTS="${GEN_PROMPTS:-a cat sitting on a chair;a medieval castle on a hill;a bowl of fruit on a table;an astronaut riding a horse}"

if [ "$SWEEP" = "rank" ]; then
    VALUES="${VALUES:-4 8 16 32 64}"
    FIXED_KL="${FIXED_KL:-0.5}"
    echo "  RANK sweep: ranks={$VALUES}  KL=$FIXED_KL (fixed)"
else
    VALUES="${VALUES:-0 0.1 0.5 1.0 2.0}"
    FIXED_RANK="${FIXED_RANK:-16}"
    echo "  LAMBDA sweep: lambdas={$VALUES}  rank=$FIXED_RANK (fixed)"
fi
echo "  epochs=$EPOCHS  light-battery: MMLU=$BMMLU POPE=$BPOPE MME=$BMME"
echo "------------------------------------------------------------------"

# Base reference battery (same light settings), no dependency.
if [ "$BASE_BATTERY" = "1" ]; then
    bj=$(sbatch --parsable \
        --export=ALL,ADAPTER=,TAG=base_light,MMLU_N=$BMMLU,POPE_N=$BPOPE,MME_N=$BMME \
        "$REPO/janus/eval/07_run_forgetting_battery.sh")
    echo "  [ref] base_light battery        job $bj"
fi

for v in $VALUES; do
    if [ "$SWEEP" = "rank" ]; then
        R=$v; A=$((2 * v)); KL=$FIXED_KL
    else
        R=$FIXED_RANK; A=$((2 * FIXED_RANK)); KL=$v
    fi
    NAME="text_poetry_r${R}_kl${KL}"

    JT=$(sbatch --parsable \
        --export=ALL,RUN_NAME=$NAME,LORA_RANK=$R,LORA_ALPHA=$A,KL_LAMBDA=$KL,EPOCHS=$EPOCHS,MAX_TRAIN_MINUTES=$MAXMIN \
        "$REPO/janus/train/03_run_train_text_poetry.sh")
    ADAPTER="$REPO/janus/train/output/$NAME/adapter_best"
    JB=$(sbatch --parsable --dependency=afterok:"$JT" \
        --export=ALL,ADAPTER=$ADAPTER,TAG=$NAME,MMLU_N=$BMMLU,POPE_N=$BPOPE,MME_N=$BMME \
        "$REPO/janus/eval/07_run_forgetting_battery.sh")
    LINE="  $NAME : train $JT -> battery $JB"

    if [ "$GEN_REF" = "1" ]; then
        JI=$(sbatch --parsable --dependency=afterok:"$JT" \
            --export=ALL,ADAPTER=$ADAPTER,PROMPTS="$GEN_PROMPTS" \
            "$REPO/janus/eval/13_run_t2i_crossmodal_qual.sh")
        LINE="$LINE + t2i $JI"
    fi
    echo "$LINE  (afterok:$JT)"
done

echo "------------------------------------------------------------------"
echo "  queued. monitor: squeue -u \$USER -o \"%i %j %T %M %R\""
echo "  morning read:"
echo "    task gain : ~/multimodal-peft/janus/train/output/text_poetry_r*/history.json"
echo "    forgetting: ~/multimodal-peft/janus/tests/output/forgetting_battery/text_poetry_r*/forgetting_results.json"
echo "    reference : .../forgetting_battery/base_light/forgetting_results.json"
echo "    gen grids : ~/multimodal-peft/janus/tests/output/t2i_crossmodal_qual/text_poetry_r*/t2i_crossmodal_grid.png"
echo "=================================================================="
