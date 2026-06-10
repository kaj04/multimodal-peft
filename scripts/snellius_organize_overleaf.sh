#!/bin/bash
# scripts/snellius_organize_overleaf.sh
#
# Runs ON SNELLIUS. Builds the side-by-side comparison folder for the
# Overleaf write-up.
#
# Layout produced under
#   ~/multimodal-peft/janus/tests/output/overleaf/janus_comparison/
# is a 5-METHOD x 5-IMAGE grid -- ONE sub-folder per method (= column of
# the Overleaf figure) containing the same 5 prompt PNGs (= rows).
#
# Methods (columns):
#   no_adapter           : base Janus-Pro-7B, no LoRA (BEFORE PNG from qual_v5)
#   adapter_solo_gen     : v5 image-only adapter      (AFTER PNG from qual_v5)
#   adapter_text_matched : joint with Naruto wiki     (AFTER from qual_joint_matched)
#   adapter_text_far     : joint with physics wiki    (AFTER from qual_joint_far)
#   adapter_text_near    : joint with anime wiki      (AFTER from qual_joint_near; C2 run TBD)
#
# Selected indices 0,1,5,7,8 within the qualitative script's flat ordering
# (5 eval + 5 human prompts -> 10 total per mode):
#   0 -> eval_00_natural
#   1 -> eval_01_natural
#   5 -> human_00_natural
#   7 -> human_02_natural
#   8 -> human_03_natural
#
# All in 'natural' mode (no prefix at inference) to test the gate-solved
# behaviour. Edit TAGS below if you want different prompts or 'prefix' mode.

set -u

BASE="$HOME/multimodal-peft/janus/tests/output"
DEST="$BASE/overleaf/janus_comparison"

# Indices 0,1,5,7,8 -> these prompt tags
TAGS=(
    "eval_00_natural"
    "eval_01_natural"
    "human_00_natural"
    "human_02_natural"
    "human_03_natural"
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
    echo "[$method]  <-  $src_folder/${phase}_*.png"

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

copy_method "no_adapter"           "qual_v5"            "before"
copy_method "adapter_solo_gen"     "qual_v5"            "after"
copy_method "adapter_text_matched" "qual_joint_matched" "after"
copy_method "adapter_text_far"     "qual_joint_far"     "after"
copy_method "adapter_text_near"    "qual_joint_near"    "after"

echo ""
echo "================================================================"
echo "Final layout (per-method counts):"
for m in no_adapter adapter_solo_gen adapter_text_matched adapter_text_far adapter_text_near; do
    cnt=$(find "$DEST/$m" -name "*.png" 2>/dev/null | wc -l)
    printf "  %-22s  %d images\n" "$m" "$cnt"
done

echo ""
echo "Destination: $DEST"
echo "rsync to local with:"
echo "  rsync -avz snellius:$DEST ./"
echo "================================================================"
