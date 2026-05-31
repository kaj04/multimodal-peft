# janus/env.sh
#
# Daily startup for Janus experiments on Snellius.
# Source it (do NOT execute) so the env changes apply to your shell:
#
#     source ~/multimodal-peft/janus/env.sh
#
# After this you are in the `janus` conda env, with the HF cache pointed
# to scratch and the working directory set to the repo.

module load 2023
module load Anaconda3/2023.07-2

# Enter the Janus conda env (separate from 4m-peft and anole envs)
source activate janus

# Download/load HF models into the shared scratch cache, not home (low quota)
export HF_HOME=/scratch-shared/tesr125338/hf_cache

# Move into the repo
cd ~/multimodal-peft

echo "================================================"
echo "  Janus environment ready"
echo "  conda env : $(basename $CONDA_PREFIX)"
echo "  HF_HOME   : $HF_HOME"
echo "  pwd       : $(pwd)"
echo "================================================"
echo "For GPU work, request a node with:"
echo "  salloc --partition=gpu_a100 --gres=gpu:1 --mem=40G --time=01:00:00"
