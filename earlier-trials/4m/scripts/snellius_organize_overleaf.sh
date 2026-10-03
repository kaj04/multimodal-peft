#!/bin/bash
# scripts/snellius_organize_overleaf.sh
#
# Runs ON SNELLIUS (login node, no GPU). Builds the unified comparison folder
# for the Overleaf write-up:
#   ~/multimodal-peft/janus/tests/output/overleaf/janus_comparison/
#
# ONE sub-folder per METHOD (= column of the eventual figure). Each sub-folder
# gets the SAME 5 prompt PNGs (= rows), all in 'natural' mode (no inference
# prefix), so the comparison isolates the training-time differences.
#
# Methods covered (8 total -- 5 image-only experiments + 3 joint training):
#   no_adapter            : base 7B, no LoRA          (BEFORE from qual_v5)
#   adapter_v2            : v2-CAPACITY               (AFTER from qual_v2_capacity)
#   adapter_v3            : v3-KL-REG                 (AFTER from qual_v3_klreg)
#   adapter_v4            : v4-AGGRESSIVE             (AFTER from qual_v4_aggressive)
#   adapter_solo_gen      : v5 (current best image-only) (AFTER from qual_v5)
#   adapter_text_matched  : joint v5 + Naruto wiki    (AFTER from qual_joint_matched)
#   adapter_text_far      : joint v5 + physics wiki   (AFTER from qual_joint_far)
#   adapter_text_near     : joint v5 + anime wiki     (AFTER from qual_joint_near; pending C2)
#
# Index mapping (qualitative script flat-enumerates eval + human into one list):
#   0 -> eval_00_natural
#   1 -> eval_01_natural
#   5 -> human_05_natural   (NOT human_00 -- eval takes 00-04, human takes 05-09)
#   7 -> human_07_natural
#   8 -> human_08_natural
#
# Idempotent: re-run after qual_joint_near completes to fill in the missing
# column without disturbing the others.

set -u

BASE="$HOME/multimodal-peft/janus/tests/output"
DEST="$BASE/overleaf/janus_comparison"

# Indices 0,1,2,5,6,7,8,9 -> qualitative script tags
# (eval block 00-04, human block 05-09; we skip eval_03/04 to keep the
# eval/human balance at 3 + 5 = 8 rows in the Overleaf figure)
TAGS=(
    "eval_00_natural"
    "eval_01_natural"
    "eval_02_natural"
    "human_05_natural"
    "human_06_natural"
    "human_07_natural"
    "human_08_natural"
    "human_09_natural"
)

mkdir -p "$DEST"

# copy_method <method_name> <source_qual_folder> <phase: before|after>
copy_method () {
    local method="$1"
    local src_folder="$2"
    local phase="$3"
    local dest_dir="$DEST/$method"
    local src_dir="$BASE/$src_folder/naruto_qual_7b"

    mkdir -p "$dest_dir"
    echo "[$method]  <-  $src_folder  (phase=$phase)"

    if [ ! -d "$src_dir" ]; then
        echo "  WARN: source folder missing: $src_dir  (skipping, empty $method/)"
        return
    fi

    local n=0
    for tag in "${TAGS[@]}"; do
        local src="$src_dir/${tag}_${phase}.png"
        local dst="$dest_dir/${tag}.png"
        if [ -f "$src" ]; then
            cp -f "$src" "$dst"
            n=$((n + 1))
        else
            echo "  MISSING: $src"
        fi
    done
    echo "  copied $n / ${#TAGS[@]} images"
}

# Image-only experiments (5 columns)
copy_method "no_adapter"           "qual_v5"               "before"
copy_method "adapter_v2"           "qual_v2_capacity"      "after"
copy_method "adapter_v3"           "qual_v3_klreg"         "after"
copy_method "adapter_v4"           "qual_v4_aggressive"    "after"
copy_method "adapter_solo_gen"     "qual_v5"               "after"

# Joint training experiments (3 columns)
copy_method "adapter_text_matched" "qual_joint_matched"    "after"
copy_method "adapter_text_far"     "qual_joint_far"        "after"
copy_method "adapter_text_near"    "qual_joint_near"       "after"

echo ""
echo "================================================================"
echo "Final layout (per-method counts):"
for m in no_adapter adapter_v2 adapter_v3 adapter_v4 adapter_solo_gen \
         adapter_text_matched adapter_text_far adapter_text_near; do
    cnt=$(find "$DEST/$m" -name "*.png" 2>/dev/null | wc -l)
    printf "  %-22s  %d images\n" "$m" "$cnt"
done

echo ""
echo "Destination: $DEST"
echo "rsync to local with:"
echo "  rsync -avz snellius:$DEST ./"
echo "================================================================"
