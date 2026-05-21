#!/bin/bash
# chameleon/tests/00_check_env.sh
#
# Run on Snellius LOGIN NODE (no GPU needed) to verify the environment
# has everything needed for Chameleon before requesting GPU time.
#
# Usage:
#   bash chameleon/tests/00_check_env.sh

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate 4m-peft

echo "========================================"
echo "  Chameleon environment check"
echo "========================================"

python3 - <<'EOF'
import sys
print(f"Python: {sys.version.split()[0]}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# transformers
try:
    import transformers
    print(f"transformers: {transformers.__version__}")
    # Check Chameleon support
    try:
        from transformers import ChameleonForConditionalGeneration
        print("  ✓ ChameleonForConditionalGeneration available")
    except ImportError:
        print("  ✗ Chameleon not in this transformers version")
        print("    Fix: pip install -U transformers")
except ImportError:
    print("transformers: NOT INSTALLED")

# peft
try:
    import peft
    print(f"peft: {peft.__version__}")
    print("  ✓ PEFT available")
except ImportError:
    print("peft: NOT INSTALLED")
    print("  Fix: pip install peft")

# accelerate
try:
    import accelerate
    print(f"accelerate: {accelerate.__version__}")
except ImportError:
    print("accelerate: NOT INSTALLED")
    print("  Fix: pip install accelerate")

# pillow
try:
    from PIL import Image
    import PIL
    print(f"Pillow: {PIL.__version__}  ✓")
except ImportError:
    print("Pillow: NOT INSTALLED — pip install Pillow")

# HuggingFace login
print()
import os
token = os.environ.get('HF_TOKEN') or os.path.expanduser('~/.cache/huggingface/token')
import pathlib
if pathlib.Path(token).exists():
    print("HF token: found in cache  ✓")
else:
    print("HF token: NOT FOUND")
    print("  Chameleon-7B is a gated model — you need to:")
    print("  1. Accept license at: https://huggingface.co/facebook/chameleon-7b")
    print("  2. Run: huggingface-cli login")
    print("  Or set HF_TOKEN env variable in your SLURM scripts")
EOF

echo ""
echo "========================================"
echo "  HF cache disk usage"
echo "========================================"
du -sh /scratch-shared/tesr125338/hf_cache 2>/dev/null || echo "  Cache not found"

echo ""
echo "If everything looks good, request a GPU node and run:"
echo "  salloc --partition=gpu_a100 --gres=gpu:1 --mem=40G --time=01:00:00"
echo "  python chameleon/tests/01_test_inference.py"
