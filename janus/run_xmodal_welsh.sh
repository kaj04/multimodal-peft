#!/bin/bash
# janus/run_xmodal_welsh.sh
#
# Cross-modal collaboration experiments for the Welsh text-LoRA. Launches all
# three as INDEPENDENT jobs (run in parallel, each grabs its own GPU):
#   A (27) Welsh prompt -> image faithfulness   (CLIPScore vs English ref)
#   C (28) Welsh VQA understanding              (Yes/No accuracy)
#   B (29) Welsh text-in-image feasibility      (generation + grid, OCR later)
#
# Smoke first (tiny, ~10 min each):  SMOKE=1 bash janus/run_xmodal_welsh.sh
# Full:                              bash janus/run_xmodal_welsh.sh
# Override adapter:                  ADAPTER=/path/... bash janus/run_xmodal_welsh.sh

set -euo pipefail
REPO="$HOME/multimodal-peft"
ADAPTER="${ADAPTER:-$REPO/janus/train/output/text_7b_welsh_r32_kl0.1/adapter_best}"
SMOKE="${SMOKE:-0}"

echo "=================================================================="
echo "  cross-modal Welsh experiments | SMOKE=$SMOKE"
echo "  ADAPTER=$ADAPTER"
echo "=================================================================="

jA=$(sbatch --parsable --export=ALL,ADAPTER="$ADAPTER",SMOKE="$SMOKE" \
    "$REPO/janus/eval/27_run_xmodal_welsh_t2i.sh")
echo "  [A] welsh prompt -> image     job $jA"
jC=$(sbatch --parsable --export=ALL,ADAPTER="$ADAPTER",SMOKE="$SMOKE" \
    "$REPO/janus/eval/28_run_xmodal_welsh_vqa.sh")
echo "  [C] welsh VQA understanding   job $jC"
jB=$(sbatch --parsable --export=ALL,ADAPTER="$ADAPTER",SMOKE="$SMOKE" \
    "$REPO/janus/eval/29_run_xmodal_welsh_textinimg.sh")
echo "  [B] welsh text-in-image       job $jB"

echo "------------------------------------------------------------------"
echo "  monitor: squeue -u \$USER -o \"%i %j %T %M %R\""
echo "  results:"
echo "    A: $REPO/janus/tests/output/xmodal_welsh_t2i/expA_results.txt"
echo "    C: $REPO/janus/tests/output/xmodal_welsh_vqa/expC_results.txt"
echo "    B: $REPO/janus/tests/output/xmodal_welsh_textinimg/textinimg_grid.png"
echo "=================================================================="
