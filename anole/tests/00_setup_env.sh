#!/bin/bash
# anole/tests/00_setup_env.sh
#
# Run ONCE on Snellius login node to set up the Anole environment.
# This creates a new conda env 'anole' separate from '4m-peft'.
#
# Usage:
#   bash anole/tests/00_setup_env.sh
#
# After this completes, download the model:
#   huggingface-cli download GAIR/Anole-7b-v0.1 \
#       --local-dir /scratch-shared/tesr125338/anole_ckpt \
#       --token $HF_TOKEN

set -e

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"

echo "========================================"
echo "  Creating conda env: anole"
echo "========================================"

conda create -n anole python=3.10 -y
conda activate anole

echo ""
echo "========================================"
echo "  Installing PyTorch 2.1 + CUDA 11.8"
echo "========================================"

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "========================================"
echo "  Cloning Anole repo"
echo "========================================"

cd ~
if [ ! -d "anole" ]; then
    git clone https://github.com/GAIR-NLP/anole.git
else
    echo "  ~/anole already exists, skipping clone"
fi

echo ""
echo "========================================"
echo "  Installing Anole dependencies"
echo "========================================"

cd ~/anole
# Install essentials (skip xformers — optional, may conflict)
pip install accelerate==0.31.0 jsonlines pandas Pillow numpy click python-dotenv

echo ""
echo "========================================"
echo "  Installing Anole custom transformers fork"
echo "========================================"

cd ~/anole/transformers
pip install -e .
cd ~/anole

echo ""
echo "========================================"
echo "  Env setup complete. Verify with:"
echo "    conda activate anole"
echo "    python -c \"from chameleon.inference.chameleon import ChameleonInferenceModel; print('OK')\""
echo ""
echo "  Then download the model (takes ~15 GB, run from login node):"
echo "    huggingface-cli download GAIR/Anole-7b-v0.1 \\"
echo "        --local-dir /scratch-shared/tesr125338/anole_ckpt \\"
echo "        --token \$HF_TOKEN"
echo "========================================"
