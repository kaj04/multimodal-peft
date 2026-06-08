#!/bin/bash
#SBATCH --job-name=metrics_panel
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/metrics_panel/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/metrics_panel/slurm_%j.err
#SBATCH --export=ALL

# Required env vars:
#   ADAPTER     -- path to the LoRA adapter dir (epoch checkpoint)
# Optional:
#   MODEL_ID, N_GEN, CAPTION_PREFIX

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/metrics_panel

echo "================================================================"
echo "  Metrics panel | $(date)"
echo "  ADAPTER       = ${ADAPTER}"
echo "  MODEL_ID      = ${MODEL_ID:-deepseek-community/Janus-Pro-7B}"
echo "  N_GEN         = ${N_GEN:-200}"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
source activate janus

export HF_HOME=/scratch-shared/tesr125338/hf_cache

JANUS_PY=$HOME/.conda/envs/janus/bin/python

# Dependency installs are NON-DESTRUCTIVE by design:
#   - check first via the import name; install only if missing
#   - --no-deps to avoid pip pulling/upgrading numpy/scipy/torch as transitive
#     dependencies. This is what BROKE the conda env on Jun-7: pip install
#     scipy upgraded numpy from 1.26 to 2.x, and a follow-up art-fid install
#     downgraded numpy back to 1.x while scipy was already at the 2.x-built
#     version -> Janus model imports started crashing across the env for hours.
#   - art-fid is intentionally NOT installed here (its numpy<2 pin conflicts
#     with opencv>=4.13 in this env). Run a separate job in an isolated venv
#     if ArtFID is needed; the script's fallback path tolerates its absence.
check_and_install () {
    local mod="$1"; local pkg="${2:-$1}"
    if ! $JANUS_PY -c "import ${mod}" 2>/dev/null; then
        echo "  [deps] installing ${pkg} (--no-deps)"
        $JANUS_PY -m pip install --quiet --no-deps "${pkg}" 2>&1 | tail -3
    else
        echo "  [deps] ${mod} already present, skipping"
    fi
}
check_and_install cleanfid clean-fid
check_and_install pyiqa
# setuptools provides pkg_resources, required by CLIP-IQA inside pyiqa.
# Some clean condas don't ship it. --no-deps keeps numpy/scipy untouched.
check_and_install pkg_resources setuptools
# ArtFID (paper 2207.12280): style-aware FID with Inception fine-tuned on
# WikiArt. Requires numpy<2, which our env already satisfies after the
# Jun-7 repair (numpy 1.26.4 + scipy<1.13). --no-deps prevents it from
# trying to pin its own numpy version.
check_and_install art_fid art-fid

$JANUS_PY ~/multimodal-peft/janus/eval/06_metrics_panel.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
