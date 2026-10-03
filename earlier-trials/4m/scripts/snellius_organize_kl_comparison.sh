#!/bin/bash
# scripts/snellius_organize_kl_comparison.sh
#
# Builds overleaf/kl_comparison/ : one sub-folder per column (base + each KL),
# each with the same 5 prompt PNGs (indices 0,1,5,7,8, natural mode).
# Base = adapter-OFF output; each KL column = that KL run's adapter-ON output.
# KL=0.5 reuses the rank-sweep r=32 qualitative (qual_rank_r32).

set -u
BASE="$HOME/multimodal-peft/janus/tests/output"
DEST="$BASE/overleaf/kl_comparison"

TAGS=(eval_00_natural eval_01_natural human_05_natural human_07_natural human_08_natural)

# column label : qualitative source folder (under tests/output/<src>/naruto_qual_7b)
# KL=0.5 points at qual_rank_r32 (reused); the others at their qual_<run> dirs.
COLS=(
  "kl0:qual_gen_7b_kl0_r32"
  "kl025:qual_gen_7b_kl025_r32"
  "kl05:qual_rank_r32"
  "kl1:qual_gen_7b_kl1_r32"
  "kl15:qual_gen_7b_kl15_r32"
)

mkdir -p "$DEST"

# Base column from the kl0 run's 'before' (adapter off; same across runs)
base_src="$BASE/qual_gen_7b_kl0_r32/naruto_qual_7b"
mkdir -p "$DEST/base"
echo "[base] <- qual_gen_7b_kl0_r32 (before)"
for tag in "${TAGS[@]}"; do
  s="$base_src/${tag}_before.png"
  [ -f "$s" ] && cp -f "$s" "$DEST/base/${tag}.png" || echo "  MISSING $s"
done

for entry in "${COLS[@]}"; do
  col="${entry%%:*}"; srcname="${entry##*:}"
  src="$BASE/$srcname/naruto_qual_7b"
  dst="$DEST/$col"; mkdir -p "$dst"
  echo "[$col] <- $srcname (after)"
  if [ ! -d "$src" ]; then echo "  WARN missing $src"; continue; fi
  for tag in "${TAGS[@]}"; do
    s="$src/${tag}_after.png"
    [ -f "$s" ] && cp -f "$s" "$dst/${tag}.png" || echo "  MISSING $s"
  done
done

echo ""
echo "Per-column counts:"
for m in base kl0 kl025 kl05 kl1 kl15; do
  cnt=$(find "$DEST/$m" -name "*.png" 2>/dev/null | wc -l)
  printf "  %-6s %d\n" "$m" "$cnt"
done
echo "Destination: $DEST"
