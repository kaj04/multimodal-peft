#!/bin/bash
# =============================================================================
# 00_setup_env.sh  —  One-time environment setup on Snellius
# Run once after cloning the repo:
#   bash scripts/snellius/00_setup_env.sh
# =============================================================================

set -e  # exit on first error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Repo root: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Load modules
# ---------------------------------------------------------------------------
module load 2023
module load Anaconda3/2023.07-2

# ---------------------------------------------------------------------------
# 2. Create conda environment (skip if exists)
# ---------------------------------------------------------------------------
ENV_NAME="4m-peft"

if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Conda env '$ENV_NAME' already exists, skipping creation."
else
    echo "Creating conda env '$ENV_NAME' with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate
source activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 3. Install PyTorch (CUDA 12.1 — matches Snellius A100/H100 driver)
# ---------------------------------------------------------------------------
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 4. Extra dependencies not pulled in by ml-4m
# ---------------------------------------------------------------------------
pip install h5py pyyaml>=6.0

# ---------------------------------------------------------------------------
# 5. Install the fourm package from the submodule
# ---------------------------------------------------------------------------
echo "Installing fourm package from ml-4m submodule..."
pip install -e "$REPO_ROOT/ml-4m"

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
python -c "
import torch, fourm
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
from fourm.models.lora_utils import inject_trainable_LoRA
print('fourm.models.lora_utils: OK')
"

echo ""
echo "Setup complete. To activate: source activate $ENV_NAME"
echo "HF cache tip: export HF_HOME=/scratch-shared/tesr125338/hf_cache"
