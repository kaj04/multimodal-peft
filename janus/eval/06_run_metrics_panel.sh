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

# setuptools provides pkg_resources (a namespace package required by CLIP-IQA
# inside pyiqa). --no-deps does NOT work here: pkg_resources is an internal
# part of setuptools that only materialises with a proper full install.
# We do --upgrade WITHOUT --no-deps; this can pull a newer setuptools but
# does NOT touch numpy/scipy.
if ! $JANUS_PY -c "import pkg_resources" 2>/dev/null; then
    echo "  [deps] installing setuptools (--upgrade, full deps)"
    $JANUS_PY -m pip install --quiet --upgrade setuptools 2>&1 | tail -3
else
    echo "  [deps] pkg_resources already present, skipping setuptools install"
fi

# ArtFID (paper 2207.12280): style-aware FID with Inception fine-tuned on
# WikiArt, plus LPIPS for content preservation. Needs full deps (lpips, etc.)
# to work -- --no-deps would leave lpips missing. We pre-pin numpy<2 so
# art-fid's transitive deps don't try to install numpy>=2.
if ! $JANUS_PY -c "import art_fid" 2>/dev/null; then
    echo "  [deps] installing art-fid (full deps, numpy<2 pre-pinned)"
    $JANUS_PY -m pip install --quiet "numpy<2" 2>&1 | tail -3
    $JANUS_PY -m pip install --quiet "art-fid" 2>&1 | tail -3
else
    echo "  [deps] art_fid already present, skipping"
fi

$JANUS_PY ~/multimodal-peft/janus/eval/06_metrics_panel.py

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
