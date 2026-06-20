#!/bin/bash
#SBATCH --job-name=threeway_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/fcolasurdo/multimodal-peft/janus/tests/output/threeway/slurm_%j.out
#SBATCH --error=/home/fcolasurdo/multimodal-peft/janus/tests/output/threeway/slurm_%j.err
#SBATCH --export=ALL

# Fair 3-way image generation (base | poetry | naruto) on a fixed prompt set.
#   sbatch janus/tests/04_run_three_way_gen.sh
# Override the prompts (semicolon-separated) or the adapters via --export, e.g.:
#   sbatch --export=ALL,PROMPTS="a cat;a dog" janus/tests/04_run_three_way_gen.sh

mkdir -p /home/fcolasurdo/multimodal-peft/janus/tests/output/threeway

module load 2023
module load Anaconda3/2023.07-2
source activate janus
export HF_HOME=/scratch-shared/tesr125338/hf_cache
JANUS_PY=$HOME/.conda/envs/janus/bin/python

# Default prompt set (Francesco's 9 validation prompts). Overridable via --export.
export PROMPTS="${PROMPTS:-a clock on a desk between a red pen on the right and a black pen on the left;an Italian grandmother in traditional clothes cooking sushi in her home;a photo of a child at home sitting on a couch while watching a Japanese anime;a fine-dining dish of salmon on a bed of vegetables;a happy girl in a park with peacocks chasing each other in the background;a photo for an advertisement of a canned energy drink called Bullit with the iconic image of Leonardo da Vinci's Vitruvian Man on it;a super detailed portrait photo of an elderly man biting into a bruschetta with fresh tomato, all messy with tomato around his mouth;a famous hyperrealistic painting like the Mona Lisa but with a mustache;an image of a cartoon where two characters fighting in a ring}"

echo "================================================================"
echo "  3-way generation | Started: $(date)"
echo "================================================================"
$JANUS_PY ~/multimodal-peft/janus/tests/04_three_way_gen.py
echo "Done: $(date)"
