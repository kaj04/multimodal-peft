#!/bin/bash
# scripts/snellius_organize_rank_comparison.sh
#
# Runs ON SNELLIUS (login node). Builds the rank-comparison image grid for the
# Overleaf figure under
#   ~/multimodal-peft/janus/tests/output/overleaf/rank_comparison/
#
# ONE sub-folder per COLUMN (= base + each rank). Each holds the SAME 5 prompt
# PNGs (= rows). The base column is the adapter-OFF output (identical across
# rank runs, taken from r2); each rank column is that rank's adapter-ON output.
#
# Columns:
#   base : adapter disabled (base 7B)        <- 'before' PNG from qual_rank_r2
#   r2   : rank-2 adapter                     <- 'after'  PNG from qual_rank_r2
#   r4 r8 r16 r32 r64 : same, per rank
#
# Indices 0,1,5,7,8 -> tags (eval 00-04, human 05-09), 'natural' mode (no prefix,
# matching the rank-sweep training).

set -u
BASE="$HOME/multimodal-peft/janus/tests/output"
DEST="$BASE/overleaf/rank_comparison"

TAGS=(
    "eval_00_natural"
    "eval_01_natural"
    "human_05_natural"
    "human_07_natural"
    "human_08_natural"
)
RANKS=(2 4 8 16 32 64)

mkdir -p "$DEST"

# Base column: adapter-OFF output (before). Identical across rank runs (same
# base model, same seed), so we take it from the r2 run.
base_src="$BASE/qual_rank_r2/naruto_qual_7b"
mkdir -p "$DEST/base"
echo "[base]  <-  qual_rank_r2 (before)"
nb=0
for tag in "${TAGS[@]}"; do
    s="$base_src/${tag}_before.png"
    if [ -f "$s" ]; then cp -f "$s" "$DEST/base/${tag}.png"; nb=$((nb+1));
    else echo "  MISSING $s"; fi
done
echo "  copied $nb / ${#TAGS[@]}"

# Rank columns: adapter-ON output (after) from each rank's run.
for R in "${RANKS[@]}"; do
    src="$BASE/qual_rank_r${R}/naruto_qual_7b"
    dst="$DEST/r${R}"
    mkdir -p "$dst"
    echo "[r${R}]  <-  qual_rank_r${R} (after)"
    if [ ! -d "$src" ]; then echo "  WARN: $src missing"; continue; fi
    n=0
    for tag in "${TAGS[@]}"; do
        s="$src/${tag}_after.png"
        if [ -f "$s" ]; then cp -f "$s" "$dst/${tag}.png"; n=$((n+1));
        else echo "  MISSING $s"; fi
    done
    echo "  copied $n / ${#TAGS[@]}"
done

echo ""
echo "================================================================"
echo "Final layout (per-column counts):"
for m in base r2 r4 r8 r16 r32 r64; do
    cnt=$(find "$DEST/$m" -name "*.png" 2>/dev/null | wc -l)
    printf "  %-6s  %d images\n" "$m" "$cnt"
done
echo ""
echo "Destination: $DEST"
echo "Download with:  rsync -avz snellius:$DEST ./"
echo "================================================================"
