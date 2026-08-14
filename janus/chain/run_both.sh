#!/bin/bash
# =====================================================================
#  janus/chain/run_both.sh   --   submit both chains AND the final matrix
#
#    bash janus/chain/run_both.sh
#
#  Submits chain A and chain B (12 jobs) and then one small CPU job that
#  builds the retention matrix once both chains have finished. Everything is
#  queued in one go, so the results are assembled without anyone waiting up
#  for them.
#
#  The matrix job depends on the two final evaluations with afterany, not
#  afterok, and that is deliberate. 36_chain_matrix.py is written to tolerate
#  missing cells and prints "--" for them, so a partial matrix is still worth
#  having. With afterok a single failed evaluation would leave the matrix job
#  pending forever and we would end up with nothing assembled at all.
#
#  It runs on a CPU partition: the script is pure stdlib and reads JSON, so
#  putting it on a GPU would waste budget.
# =====================================================================
set -euo pipefail
source "$HOME/multimodal-peft/janus/chain/chain_config.sh"
CH="$REPO/janus/chain"
mkdir -p "$EVAL_OUT/chain"

rm -f "$EVAL_OUT/chain/.last_job_A" "$EVAL_OUT/chain/.last_job_B"

FROM_STEP="${FROM_STEP:-1}" bash "$CH/run_chain.sh" A
echo
FROM_STEP="${FROM_STEP:-1}" bash "$CH/run_chain.sh" B
echo

LA=$(cat "$EVAL_OUT/chain/.last_job_A" 2>/dev/null || true)
LB=$(cat "$EVAL_OUT/chain/.last_job_B" 2>/dev/null || true)
if [ -z "$LA" ] || [ -z "$LB" ]; then
  echo "  WARNING: could not read both final job ids (A='$LA' B='$LB')."
  echo "  The chains are running; build the matrix by hand when they finish:"
  echo "     python janus/eval/36_chain_matrix.py"
  exit 0
fi

JM=$(sbatch --parsable --dependency=afterany:"$LA":"$LB" \
     --job-name=chain_matrix --partition=genoa \
     --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:15:00 \
     --output="$EVAL_OUT/chain/slurm_matrix_%j.out" \
     --wrap="module load 2023; module load Anaconda3/2023.07-2; \
             source activate janus; \
             \$HOME/.conda/envs/janus/bin/python $REPO/janus/eval/36_chain_matrix.py")

echo "=================================================================="
echo "  matrix job $JM   (afterany on $LA and $LB)"
echo "  result : $EVAL_OUT/chain/chain_matrix.txt  (+ .json)"
echo ""
echo "  Everything is queued. Tomorrow, read the matrix with:"
echo "     cat $EVAL_OUT/chain/chain_matrix.txt"
echo "  and check nothing died with:"
echo "     sacct -S now-2days -X -u \$USER --format=JobID,JobName%14,State,Elapsed"
echo "=================================================================="
