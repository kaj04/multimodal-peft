#!/bin/bash
# scripts/launch_all_kl_chained.sh
#
# One-shot launcher for the whole KL sweep with SLURM dependencies, so the
# full pipeline (train -> metrics + qualitative -> organize + plot) runs
# unattended and the Overleaf assets are ready by morning.
#
# KL=0.5 is NOT retrained (reuses gen_7b_rank_r32 from the rank sweep); its
# qualitative (qual_rank_r32) and metrics already exist.
#
# Run on the LOGIN NODE:  bash scripts/launch_all_kl_chained.sh

set -eu
cd $HOME/multimodal-peft

MODEL="deepseek-community/Janus-Pro-7B"
OUTQ="$HOME/multimodal-peft/janus/tests/output"

# KL value : run name
RUNS=(
  "0.0:gen_7b_kl0_r32"
  "0.25:gen_7b_kl025_r32"
  "1.0:gen_7b_kl1_r32"
  "1.5:gen_7b_kl15_r32"
)

QUAL_JOBS=()

for entry in "${RUNS[@]}"; do
  KL="${entry%%:*}"; NAME="${entry##*:}"
  ADAPTER="$HOME/multimodal-peft/janus/train/output/$NAME/adapter_best"

  # 1. training
  TJ=$(sbatch --parsable \
       --export=ALL,KL_LAMBDA=$KL,LORA_RANK=32,LORA_ALPHA=64,RUN_NAME=$NAME \
       janus/train/02_run_train_generation_7b_v5.sh)
  echo "train  $NAME (KL=$KL) -> $TJ"

  # 2a. metrics panel (chained on training)
  sbatch --parsable --dependency=afterok:$TJ \
    --export=ALL,MODEL_ID=$MODEL,ADAPTER=$ADAPTER,CAPTION_PREFIX="" \
    janus/eval/06_run_metrics_panel.sh >/dev/null
  echo "  metrics chained on $TJ"

  # 2b. qualitative (chained on training); capture jobid for the final step
  QJ=$(sbatch --parsable --dependency=afterok:$TJ \
       --export=ALL,MODEL_ID=$MODEL,ADAPTER=$ADAPTER,CAPTION_PREFIX="",JANUS_OUTPUT_DIR=$OUTQ/qual_$NAME \
       janus/eval/05_run_pokemon_qualitative.sh)
  echo "  qualitative -> $QJ"
  QUAL_JOBS+=("$QJ")
done

# 3. organize + plot, chained after ALL new qualitatives complete.
DEP=$(IFS=:; echo "${QUAL_JOBS[*]}")
PJ=$(sbatch --parsable --dependency=afterok:$DEP scripts/kl_post.sh)
echo ""
echo "post-processing (organize + plot) -> $PJ  (after quals: $DEP)"
echo ""
echo "All submitted. By morning, check:"
echo "  $OUTQ/overleaf/kl_comparison/   (visual grid: base + 5 KL columns)"
echo "  $OUTQ/overleaf/kl_sweep/        (metric table + curve)"
