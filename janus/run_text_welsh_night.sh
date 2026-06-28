#!/bin/bash
# janus/run_text_welsh_night.sh
#
# Red-flag #2 experiment: a STRONG, genuinely-unseen TEXT adaptation. We picked
# WELSH (probe 25: low-resource natural language, abundant data, xControl 2.34,
# not a cipher-style shortcut) and train an UNCONDITIONAL next-token LoRA on
# Welsh Wikipedia -- exactly the same recipe as poetry/medicine, just a far more
# distant target, so the forgetting test stops being trivial.
#
# One launch chains everything (all evals afterok on the training):
#   1. train      Welsh text LoRA                (03_run_train_text_poetry.sh + Welsh env)
#   2. writing    base-vs-adapter continuations  (12)  EN + CY incipits -> acquisition + "bleed"
#   3. t2i        cross-modal image gen          (13)  does the TEXT LoRA break image gen?
#   4. battery    PPL + MMLU + POPE + MME         (07)  cross-modal forgetting (knowledge + understanding)
#   5. gen-CMMD   adapter-gen vs base-gen         (17)  image-generation distribution shift
#
# Acquisition itself is FREE: the trainer logs held-out Welsh val_CE/PPL
# (baseline vs best) into history.json. MMLU is logit-scored on the A/B/C/D
# answer, so the "answers in Welsh" bleed does NOT corrupt the knowledge number.
#
# Run from the LOGIN node:   bash janus/run_text_welsh_night.sh
# Override inline, e.g.:      KL_LAMBDA=0.5 LORA_RANK=64 bash janus/run_text_welsh_night.sh

set -euo pipefail
REPO="$HOME/multimodal-peft"

# ---- target / config ------------------------------------------------------
HF_DATASET_NAME="${HF_DATASET_NAME:-wikimedia/wikipedia}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-20231101.cy}"   # cy = Welsh
DOC_MODE="${DOC_MODE:-rows}"                            # one article per row
HF_TEXT_COL="${HF_TEXT_COL:-text}"
MAX_DOCS="${MAX_DOCS:-40000}"                           # enough articles to build ~8k windows

# ---- LoRA / optimisation --------------------------------------------------
LORA_RANK="${LORA_RANK:-32}"                            # strong capacity (like medicine)
LORA_ALPHA="${LORA_ALPHA:-$((2 * LORA_RANK))}"
KL_LAMBDA="${KL_LAMBDA:-0.1}"                           # LOW anchor -> strong adaptation (poetry was 0.5)
EPOCHS="${EPOCHS:-3}"
TRAIN_WINDOWS="${TRAIN_WINDOWS:-6000}"                  # ~3M tokens
# Cap HIGH on purpose: EPOCHS should govern (training finishes naturally in
# ~3-4h), the cap is only a safety backstop. --time is set above it for margin.
MAX_TRAIN_MINUTES="${MAX_TRAIN_MINUTES:-300}"
TRAIN_TIME="${TRAIN_TIME:-06:00:00}"                    # SLURM wall (overrides runner's 4h)
RUN_NAME="${RUN_NAME:-text_7b_welsh_r${LORA_RANK}_kl${KL_LAMBDA}}"

# ---- forgetting battery sizes --------------------------------------------
BMMLU="${BMMLU:-1000}"; BPOPE="${BPOPE:-1000}"; BMME="${BMME:-600}"
NGEN="${NGEN:-128}"
BASE_BATTERY="${BASE_BATTERY:-0}"   # 1 = (re)run the base reference battery too

ADAPTER="$REPO/janus/train/output/$RUN_NAME/adapter_best"

# ---- string env with spaces / ; / | : EXPORT first, ride on --export=ALL ---
# (inline VAR=val in --export breaks on spaces/commas -- the lesson from the
#  caption-prefix bug; these get inherited via ALL instead.)
# Training periodic samples (trainer splits on ';'): mix Welsh seeds (acquisition)
# and English seeds (does the default voice bleed into Welsh?).
export SAMPLE_PROMPTS="Mae Cymru yn;Ar fore heulog,;Yr oedd unwaith;The history of Wales;In the morning, the city;Here is how to make bread:"
# Writing-qual incipits (script 12 splits on '|').
export INCIPITS="Mae Cymru yn wlad|Ar fore heulog, aeth|Yr oedd unwaith|The history of Wales|In the morning, the city woke up|Here is how you make bread:"
export NEUTRAL="The weather today is|To make a cup of tea,|The meeting will start at"
export POEM_PRIMER=""   # CRITICAL: no poetry priming for a language target

echo "=================================================================="
echo "  WELSH unconditional text-PEFT  | RUN_NAME=$RUN_NAME"
echo "  dataset=$HF_DATASET_NAME/$HF_DATASET_CONFIG  mode=$DOC_MODE  max_docs=$MAX_DOCS"
echo "  rank=$LORA_RANK alpha=$LORA_ALPHA  KL_LAMBDA=$KL_LAMBDA  budget=${MAX_TRAIN_MINUTES}min"
echo "  adapter (post-train) -> $ADAPTER"
echo "=================================================================="

# optional base-reference battery (only if you don't already have one)
if [ "$BASE_BATTERY" = "1" ]; then
    bj=$(sbatch --parsable \
        --export=ALL,ADAPTER=,TAG=base,MMLU_N=$BMMLU,POPE_N=$BPOPE,MME_N=$BMME \
        "$REPO/janus/eval/07_run_forgetting_battery.sh")
    echo "  [ref] base battery        job $bj"
fi

# 1) TRAIN  (--time overrides the runner's 4h header so the high cap fits)
JT=$(sbatch --parsable --time="$TRAIN_TIME" \
    --export=ALL,RUN_NAME=$RUN_NAME,HF_DATASET_NAME=$HF_DATASET_NAME,HF_DATASET_CONFIG=$HF_DATASET_CONFIG,DOC_MODE=$DOC_MODE,HF_TEXT_COL=$HF_TEXT_COL,MAX_DOCS=$MAX_DOCS,LORA_RANK=$LORA_RANK,LORA_ALPHA=$LORA_ALPHA,KL_LAMBDA=$KL_LAMBDA,EPOCHS=$EPOCHS,TRAIN_WINDOWS=$TRAIN_WINDOWS,MAX_TRAIN_MINUTES=$MAX_TRAIN_MINUTES \
    "$REPO/janus/train/03_run_train_text_poetry.sh")
echo "  [1] train                 job $JT"

# 2) writing qualitative (EN + CY) -- acquisition + bleed, by eye
JW=$(sbatch --parsable --dependency=afterok:"$JT" \
    --export=ALL,ADAPTER=$ADAPTER \
    "$REPO/janus/eval/12_run_text_writing_qual.sh")
echo "  [2] writing qual          job $JW   (afterok:$JT)"

# 3) cross-modal image generation (base vs adapter)
JI=$(sbatch --parsable --dependency=afterok:"$JT" \
    --export=ALL,ADAPTER=$ADAPTER \
    "$REPO/janus/eval/13_run_t2i_crossmodal_qual.sh")
echo "  [3] t2i cross-modal       job $JI   (afterok:$JT)"

# 4) forgetting battery (PPL + MMLU + POPE + MME)
JB=$(sbatch --parsable --dependency=afterok:"$JT" \
    --export=ALL,ADAPTER=$ADAPTER,TAG=$RUN_NAME,MMLU_N=$BMMLU,POPE_N=$BPOPE,MME_N=$BMME \
    "$REPO/janus/eval/07_run_forgetting_battery.sh")
echo "  [4] forgetting battery    job $JB   (afterok:$JT)"

# 5) image-gen distribution shift (Gen-CMMD adapter vs base)
JG=$(sbatch --parsable --dependency=afterok:"$JT" \
    --export=ALL,ADAPTERS="welsh=$ADAPTER",N_GEN=$NGEN \
    "$REPO/janus/eval/17_run_gen_forgetting_lambda.sh")
echo "  [5] gen-CMMD              job $JG   (afterok:$JT)"

echo "------------------------------------------------------------------"
echo "  queued. monitor:  squeue -u \$USER -o \"%i %j %T %M %R\""
echo "  EYEBALL on the filesystem:"
echo "    acquisition (Welsh PPL): $REPO/janus/train/output/$RUN_NAME/history.json"
echo "    live EN/CY samples     : $REPO/janus/train/output/$RUN_NAME/samples/*.txt"
echo "    writing (base|adapter) : $REPO/janus/tests/output/text_writing_qual/$RUN_NAME/writing_qual.txt"
echo "    images (base|adapter)  : $REPO/janus/tests/output/t2i_crossmodal_qual/$RUN_NAME/t2i_crossmodal_grid.png"
echo "    forgetting             : $REPO/janus/tests/output/forgetting_battery/$RUN_NAME/forgetting_results.json"
echo "    base reference         : $REPO/janus/tests/output/forgetting_battery/base/forgetting_results.json"
echo "    gen-CMMD               : $REPO/janus/tests/output/gen_forgetting_lambda/summary*.json (welsh)"
echo "=================================================================="
