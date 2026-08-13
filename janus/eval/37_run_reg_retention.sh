#!/bin/bash
# =====================================================================
#  janus/eval/37_run_reg_retention.sh   --   E3: retention for the
#  regularizer ablation
#
#    bash janus/eval/37_run_reg_retention.sh
#
#  The ablation in the appendix compares five training objectives at rank 32
#  in the image direction, but scores them only on how well they ACQUIRE the
#  style. In a paper about forgetting that is the wrong axis, and the text
#  currently asserts that the Anole reconstruction loss "offers no forgetting
#  protection" without ever having measured forgetting for it. This fills the
#  missing columns.
#
#  Only three jobs are needed. Two conditions already have their retention
#  measured, because their adapters are shared with sweeps we ran earlier:
#    None (lambda=0)      -> gen_7b_kl0_r32     (the lambda=0 column)
#    KL-to-base, ours     -> gen_7b_rank_r32    (the rank sweep ran at lambda=0.5)
#  Both already have a forgetting_results.json, so re-running them would only
#  reproduce numbers we have.
#
#  Full evaluation configuration on purpose: the acquisition half of this
#  ablation lives in the same regime as the image rank sweep, whose base row
#  is the full config (MME 1785). Using the subsampled config here would make
#  the two halves of the table incomparable.
#
#  Read the result with 38_reg_retention_table.py.
# =====================================================================
set -euo pipefail
REPO="$HOME/multimodal-peft"
OUT="$REPO/janus/train/output"

# full configuration, as used by the image tables
export MMLU_N=1425
export POPE_N=3000
export MME_N=0          # 0 = the whole MME set

# name | run directory
specs=(
  "logit-MSE|gen_7b_mse_r32"
  "L2-SP|gen_7b_l2_r32"
  "Anole-recon|gen_7b_anole_r32"
)

echo "=================================================================="
echo "  E3: retention for the regularizer ablation (3 jobs)"
echo "  full config: MMLU=$MMLU_N POPE=$POPE_N MME=full"
echo "=================================================================="

ids=""
for spec in "${specs[@]}"; do
  IFS='|' read -r label run <<< "$spec"
  adapter="$OUT/$run/adapter_best"
  if [ ! -e "$adapter" ]; then
    echo "  SKIP $label: no adapter at $adapter"
    continue
  fi
  jid=$(sbatch --parsable \
        --export=ALL,TAG=$run,ADAPTER=$adapter,MMLU_N=$MMLU_N,POPE_N=$POPE_N,MME_N=$MME_N \
        "$REPO/janus/eval/07_run_forgetting_battery.sh")
  ids="${ids:+$ids:}$jid"
  echo "  $label  ->  job $jid   ($run)"
done

if [ -n "$ids" ]; then
  jt=$(sbatch --parsable --dependency=afterany:"$ids" \
       --job-name=reg_table --partition=genoa \
       --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
       --output="$REPO/janus/tests/output/forgetting_battery/slurm_regtable_%j.out" \
       --wrap="module load 2023; module load Anaconda3/2023.07-2; \
               source activate janus; \
               \$HOME/.conda/envs/janus/bin/python $REPO/janus/eval/38_reg_retention_table.py")
  echo "------------------------------------------------------------------"
  echo "  table job $jt  (afterany, so a partial table is still assembled)"
  echo "  result: janus/tests/output/forgetting_battery/reg_retention.txt"
fi
echo "=================================================================="
