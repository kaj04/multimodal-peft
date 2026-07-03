#!/bin/bash
# janus/run_mauve_all.sh
#
# Full MAUVE experiment: free-form generation quality for base + every headline
# adapter (image + text), and target acquisition for the text targets.
#   * english MAUVE (all)   -> free-form English preserved = forgetting
#   * target  MAUVE (text)  -> acquisition at the generation-distribution level
#
# Requires the mauve venv first (once):  bash janus/eval/setup_mauve_venv.sh
# Then:                                  bash janus/run_mauve_all.sh
#   smoke:                               SMOKE=1 bash janus/run_mauve_all.sh
#
# The base job builds the SHARED human references; every other gen job depends on
# it (afterok) so the refs exist before they run (no write race). Each gen -> its
# MAUVE compute (mauve venv) -> a final consolidated table.

set -euo pipefail
REPO="$HOME/multimodal-peft"
O="$REPO/janus/train/output"
E="$REPO/janus/eval"
SMOKE="${SMOKE:-0}"
N="${N:-2000}"

# name | adapter_path (empty=base) | targets (comma, "" = english only)
specs=(
  "base||poetry,pubmed,welsh"
  "gen_7b_naruto_v5|$O/gen_7b_naruto_v5/adapter_best|"
  "gen_7b_sketch_v5|$O/gen_7b_sketch_v5/adapter_best|"
  "text_poetry_r16_kl0.5|$O/text_poetry_r16_kl0.5/adapter_best|poetry"
  "text_med_r32_kl025|$O/text_med_r32_kl025/adapter_best|pubmed"
  "text_7b_welsh_r32_kl0.1|$O/text_7b_welsh_r32_kl0.1/adapter_best|welsh"
)

echo "=================================================================="
echo "  MAUVE experiment | SMOKE=$SMOKE  N=$N"
echo "=================================================================="

cmp_ids=""
base_gen=""
for spec in "${specs[@]}"; do
  IFS='|' read -r name adapter tgt <<< "$spec"
  if [ -n "$tgt" ]; then KINDS="english,$tgt"; else KINDS="english"; fi
  export TARGETS="$tgt" KINDS="$KINDS"

  if [ "$name" = "base" ]; then
    jG=$(sbatch --parsable \
        --export=ALL,TAG=$name,ADAPTER=,SMOKE=$SMOKE,N=$N \
        "$E/30_run_mauve.sh")
    base_gen="$jG"
  else
    jG=$(sbatch --parsable --dependency=afterok:"$base_gen" \
        --export=ALL,TAG=$name,ADAPTER=$adapter,SMOKE=$SMOKE,N=$N \
        "$E/30_run_mauve.sh")
  fi
  jC=$(sbatch --parsable --dependency=afterok:"$jG" \
      --export=ALL,TAG=$name \
      "$E/31_run_mauve_compute.sh")
  cmp_ids="${cmp_ids:+$cmp_ids:}$jC"
  echo "  $name : gen $jG -> mauve $jC   (kinds=$KINDS)"
done

# final consolidated table, after all compute jobs
jT=$(sbatch --parsable --dependency=afterok:"$cmp_ids" \
    --partition=genoa --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    --output="$REPO/janus/tests/output/mauve/slurm_table_%j.out" \
    --wrap="module load 2023; module load Anaconda3/2023.07-2; source activate janus; \
            \$HOME/.conda/envs/janus/bin/python $E/32_mauve_table.py")
echo "------------------------------------------------------------------"
echo "  table job $jT  (afterok all compute)"
echo "  result: $REPO/janus/tests/output/mauve/mauve_table.txt"
echo "  monitor: squeue -u \$USER -o \"%i %j %T %M %R\""
echo "=================================================================="
