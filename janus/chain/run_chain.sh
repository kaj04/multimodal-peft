#!/bin/bash
# =====================================================================
#  janus/chain/run_chain.sh   --   submit ONE sequential chain
#
#    bash janus/chain/run_chain.sh A      # welsh -> anime -> medicine
#    bash janus/chain/run_chain.sh B      # medicine -> anime -> welsh
#
#  B is the exact reverse of A, so the two chains differ ONLY in order. The
#  image step sits in the middle of both: it is a control, not a variable.
#  Welsh and medicine each appear once first and once last, which is what
#  makes "does what you learn first survive?" answerable.
#
#  Six jobs per chain, wired train -> eval -> train -> eval -> train -> eval
#  with afterok, so a failed training stops the chain instead of evaluating an
#  adapter that was never written. Submit both chains and come back later.
#
#  Step 1 of each chain is also, by construction, a matched-hyperparameter
#  standalone adapter for that target, which the paper does not currently have.
# =====================================================================
set -euo pipefail
source "$HOME/multimodal-peft/janus/chain/chain_config.sh"

CHAIN="${1:-}"
case "$CHAIN" in
  A) T1=welsh;    M2=image; T3=medicine ;;
  B) T1=medicine; M2=image; T3=welsh    ;;
  *) echo "usage: bash janus/chain/run_chain.sh [A|B]"; exit 1 ;;
esac

CH="$REPO/janus/chain"
mkdir -p "$EVAL_OUT/chain"

R1="chain${CHAIN}_s1_${T1}"
R2="chain${CHAIN}_s2_naruto"
R3="chain${CHAIN}_s3_${T3}"
A1="$TRAIN_OUT/$R1/adapter_best"
A2="$TRAIN_OUT/$R2/adapter_best"
A3="$TRAIN_OUT/$R3/adapter_best"

echo "=================================================================="
echo "  CHAIN $CHAIN :  $T1 (text)  ->  anime (image)  ->  $T3 (text)"
echo "  r=$LORA_RANK  lambda=$KL_LAMBDA  epochs text=$TEXT_EPOCHS image=$IMAGE_EPOCHS"
echo "=================================================================="

# --- step 1: text, from the frozen base -------------------------------
J1=$(sbatch --parsable \
     --export=ALL,RUN_NAME=$R1,CORPUS=$T1,INIT_ADAPTER= \
     "$CH/train_text_step.sh")
E1=$(sbatch --parsable --dependency=afterok:"$J1" \
     --export=ALL,TAG=$R1,ADAPTER=$A1,STEP=1 \
     "$CH/eval_step.sh")
echo "  step1  train $J1  ->  eval $E1   ($T1, from base)"

# --- step 2: image, starting from step 1 ------------------------------
J2=$(sbatch --parsable --dependency=afterok:"$J1" \
     --export=ALL,RUN_NAME=$R2,INIT_ADAPTER=$A1 \
     "$CH/train_image_step.sh")
E2=$(sbatch --parsable --dependency=afterok:"$J2" \
     --export=ALL,TAG=$R2,ADAPTER=$A2,STEP=2 \
     "$CH/eval_step.sh")
echo "  step2  train $J2  ->  eval $E2   (anime, on top of $T1)"

# --- step 3: text, starting from step 2 -------------------------------
J3=$(sbatch --parsable --dependency=afterok:"$J2" \
     --export=ALL,RUN_NAME=$R3,CORPUS=$T3,INIT_ADAPTER=$A2 \
     "$CH/train_text_step.sh")
E3=$(sbatch --parsable --dependency=afterok:"$J3" \
     --export=ALL,TAG=$R3,ADAPTER=$A3,STEP=3 \
     "$CH/eval_step.sh")
echo "  step3  train $J3  ->  eval $E3   ($T3, on top of anime)"

# The final eval id is recorded so run_both.sh can hang the matrix job off
# both chains without parsing this output.
echo "$E3" > "$EVAL_OUT/chain/.last_job_${CHAIN}"

echo "------------------------------------------------------------------"
echo "  trainings run back to back; each eval starts as soon as its own"
echo "  training succeeds, so evaluation overlaps the next training."
echo "  monitor : squeue -u \$USER -o \"%.10i %.12j %.8T %.10M %R\""
echo "  logs    : $EVAL_OUT/chain/"
echo "  last eval job for chain $CHAIN : $E3"
echo "=================================================================="
