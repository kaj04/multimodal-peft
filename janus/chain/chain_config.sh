# =====================================================================
#  janus/chain/chain_config.sh   --   E2 frozen configuration
#
#  Sourced by every chain script. Nothing here may differ between chain A
#  and chain B: the two chains must differ ONLY in the order of the tasks,
#  otherwise order is confounded with hyperparameters and the retention
#  matrix cannot be read.
#
#  Why these values (all justified by tables already in the paper):
#    rank 32   - the rank Welsh and medicine already use; the image rank
#                sweep puts the acquisition knee at 16-32, so the image
#                direction loses very little relative to its own r=48 recipe.
#    lambda .5 - the Final recipe in BOTH directions. The image KL sweep at
#                r=32 gives CMMD .108 at this value under the trigger-free
#                protocol, which is the standalone reference the chain's
#                image step is compared against.
#    modules   - attention + MLP, already identical across all four runs.
#    LR 1e-4   - already identical across all four runs.
#    epochs    - per modality, as in the standalone runs (text 3, image 6).
#                Equalising epochs ACROSS modalities would be meaningless:
#                a text epoch is 512-token windows, an image epoch is
#                caption-image pairs.
#  The KL anchors to the FROZEN BASE, not to the previous step. That is
#  deliberate and discussed in the paper: a rolling anchor is
#  Learning-without-Forgetting, and we are diagnosing, not treating.
# =====================================================================

# ---- structural: identical for every step of every chain ----
export LORA_RANK=32
export LORA_ALPHA=64
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
export KL_LAMBDA=0.5
export LR=1e-4
export LR_SCHEDULE=cosine
export SEED=42

# ---- never resume: chaining is INIT_ADAPTER, not RESUME ----
# Reusing a RUN_NAME with RESUME=1 has contaminated a baseline in this
# project before. Every chain step gets a fresh RUN_NAME and RESUME=0.
export RESUME=0

# ---- text steps ----
export TEXT_EPOCHS=3
export TEXT_DROPOUT=0.05
export SEQ_LEN=512
export TRAIN_WINDOWS=6000        # verify each corpus can supply this (smoke)
export VAL_WINDOWS=200

# ---- image step ----
export IMAGE_EPOCHS=6
export IMAGE_DROPOUT=0.10
# The Final recipe applies the style unconditionally. An accidental prefix
# would make the chain's CMMD incomparable with the paper's numbers.
export CAPTION_PREFIX=""

# ---- corpora ----
# Welsh
export WELSH_DATASET="wikimedia/wikipedia"
export WELSH_CONFIG="20231101.cy"
export WELSH_TEXT_COL="text"
export WELSH_MAX_DOCS=40000
# Medicine
export MED_DATASET="ccdv/pubmed-summarization"
export MED_CONFIG=""
export MED_TEXT_COL="article"
export MED_MAX_DOCS=40000
# Image style
export IMAGE_DATASET="lambdalabs/naruto-blip-captions"

# ---- evaluation sizes (subsampled config, as used by the text tables) ----
export MMLU_N=570
export POPE_N=1000
export MME_N=600
export N_GEN=80                  # held-out prompts for image acquisition

# ---- paths ----
export REPO="$HOME/multimodal-peft"
export TRAIN_OUT="$REPO/janus/train/output"
export EVAL_OUT="$REPO/janus/tests/output"
