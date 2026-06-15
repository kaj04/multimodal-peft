#!/bin/bash
# janus/run_text_peft_night.sh
#
# One-shot night pipeline for the text-PEFT (poetry) modularity experiment:
#   1. train   text LoRA on poetry            (03_run_train_text_poetry.sh)
#   2. writing qualitative  base vs adapter   (12_run_text_writing_qual.sh)  [afterok 1]
#   3. t2i cross-modal      base vs adapter   (13_run_t2i_crossmodal_qual.sh)[afterok 1]
#
# Steps 2-3 run only if training EXITS CLEANLY (afterok). Training stops itself
# at MAX_TRAIN_MINUTES (clean exit 0) well before the SLURM wall, so afterok
# fires. Run from the repo root on the LOGIN node:
#   bash janus/run_text_peft_night.sh
#
# Override the run name / budget inline, e.g.:
#   RUN_NAME=text_7b_poetry_v1 MAX_TRAIN_MINUTES=150 bash janus/run_text_peft_night.sh

set -euo pipefail

REPO="$HOME/multimodal-peft"
RUN_NAME="${RUN_NAME:-text_7b_poetry_v1}"
MAX_TRAIN_MINUTES="${MAX_TRAIN_MINUTES:-150}"
ADAPTER="$REPO/janus/train/output/$RUN_NAME/adapter_best"

echo "=================================================================="
echo "  Night pipeline: text-PEFT (poetry) + qualitative evals"
echo "  RUN_NAME=$RUN_NAME  MAX_TRAIN_MINUTES=$MAX_TRAIN_MINUTES"
echo "  adapter (post-train) -> $ADAPTER"
echo "=================================================================="

# 1) training  (--parsable -> just the job id)
JID_TRAIN=$(sbatch --parsable \
    --export=ALL,RUN_NAME="$RUN_NAME",MAX_TRAIN_MINUTES="$MAX_TRAIN_MINUTES" \
    "$REPO/janus/train/03_run_train_text_poetry.sh")
echo "  [1] training            job $JID_TRAIN"

# 2) writing qual  (after training succeeds)
JID_WRITE=$(sbatch --parsable --dependency=afterok:"$JID_TRAIN" \
    --export=ALL,ADAPTER="$ADAPTER" \
    "$REPO/janus/eval/12_run_text_writing_qual.sh")
echo "  [2] writing qual        job $JID_WRITE   (afterok:$JID_TRAIN)"

# 3) t2i cross-modal  (after training succeeds)
JID_T2I=$(sbatch --parsable --dependency=afterok:"$JID_TRAIN" \
    --export=ALL,ADAPTER="$ADAPTER" \
    "$REPO/janus/eval/13_run_t2i_crossmodal_qual.sh")
echo "  [3] t2i cross-modal     job $JID_T2I   (afterok:$JID_TRAIN)"

echo "------------------------------------------------------------------"
echo "  queued. monitor with:  squeue -u \$USER"
echo "  results in the morning:"
echo "    train   : $REPO/janus/train/output/$RUN_NAME/ (history.json, samples/)"
echo "    writing : $REPO/janus/tests/output/text_writing_qual/$RUN_NAME/writing_qual.txt"
echo "    images  : $REPO/janus/tests/output/t2i_crossmodal_qual/$RUN_NAME/t2i_crossmodal_grid.png"
echo "=================================================================="
