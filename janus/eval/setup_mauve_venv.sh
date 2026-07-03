#!/bin/bash
# janus/eval/setup_mauve_venv.sh
#
# Create a SEPARATE venv for MAUVE (phase 2), layered on top of the janus
# site-packages via --system-site-packages, so we reuse janus' torch/transformers
# without re-downloading and WITHOUT modifying the janus environment. Run ONCE on
# the login node (has internet):
#   bash janus/eval/setup_mauve_venv.sh

set -euo pipefail
module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache

VENV="$HOME/multimodal-peft/mauve_venv"
JANUS_PY="$HOME/.conda/envs/janus/bin/python"

if [ ! -d "$VENV" ]; then
  echo "  creating venv (system-site-packages) at $VENV"
  "$JANUS_PY" -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# mauve-text + scikit-learn; torch/transformers/numpy come from janus (system site)
pip install mauve-text scikit-learn
python -c "import mauve, sklearn; print('  OK mauve', mauve.__version__ if hasattr(mauve,'__version__') else '', 'sklearn', sklearn.__version__)"
echo "  done. venv python: $VENV/bin/python"
