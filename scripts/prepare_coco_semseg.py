#!/usr/bin/env python3
# prepare_coco_semseg.py
#
# Tokenizes COCO 2017 panoptic segmentation → tok_semseg tokens (VQ codes).
# Reads PNG masks directly from ZIP — no full extraction to GPFS needed.
#
# Pipeline per image:
#   panoptic PNG (from zip) → segment_id map → category_id map (0-133)
#   → center-crop 224×224 (nearest-neighbour) → shift_idx_by_one (+1)
#   → VQVAE encoder → 196 tokens → int16 .npy
#
# Only processes images whose tok_rgb already exists (preserves the seed=42
# 10k train subset sampled by prepare_coco.py).
#
# Usage:
#   python scripts/prepare_coco_semseg.py \
#       --coco_dir   /scratch-shared/tesr125338/coco \
#       --tokens_dir /scratch-shared/tesr125338/coco_tokens \
#       --hf_cache   /scratch-shared/tesr125338/hf_cache \
#       --device     cuda
#
# Or via SLURM:
#   sbatch scripts/snellius/24_tokenize_coco_semseg.sh

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

SEMSEG_TOKENIZER_REPO = 'EPFL-VILAB/4M_tokenizers_semseg_4k_224-448'


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Tokenize COCO panoptic semseg → tok_semseg')
    p.add_argument('--coco_dir',   required=True,
                   help='Root of downloaded COCO (contains annotations/panoptic_*.zip)')
    p.add_argument('--tokens_dir', required=True,
                   help='Root of existing coco_tokens/ (contains train/tok_rgb/, val/tok_rgb/)')
    p.add_argument('--hf_cache',   default=None,
                   help='HuggingFace cache dir')
    p.add_argument('--batch_size', default=32, type=int,
                   help='Images per GPU batch')
    p.add_argument('--device',     default='cuda')
    p.add_argument('--splits',     default='both', choices=['train', 'val', 'both'])
    p.add_argument('--dry_run',    action='store_true',
                   help='Process only the first 8 images per split (sanity check)')
    return p.parse_args()


# ── Semantic map helpers ──────────────────────────────────────────────────────

def build_stem_to_ann(pan_json_path):
    """Load panoptic JSON → dict: image_stem → annotation dict."""
    with open(pan_json_path) as f:
        pan_data = json.load(f)
    stem_to_ann = {}
    for ann in pan_data['annotations']:
        stem = Path(ann['file_name']).stem   # e.g. '000000123456'
        stem_to_ann[stem] = ann
    return stem_to_ann


def load_semantic_map(zip_file, stem_to_ann, pan_dir_in_zip, stem):
    """Read one panoptic PNG from ZIP → (H, W) int32 array of category_ids.

    Panoptic PNG encoding:
        segment_id = R + G * 256 + B * 65536
    The JSON maps segment_id → category_id (1-133 for COCO, 0 = background).
    """
    png_name = f'{pan_dir_in_zip}/{stem}.png'
    try:
        raw = zip_file.read(png_name)
    except KeyError:
        return None

    img = Image.open(io.BytesIO(raw)).convert('RGB')
    pan_arr = np.array(img, dtype=np.int32)
    seg_id_map = (pan_arr[:, :, 0]
                  + pan_arr[:, :, 1] * 256
                  + pan_arr[:, :, 2] * 65536)

    ann = stem_to_ann.get(stem)
    if ann is None:
        return None

    cat_map = np.zeros(seg_id_map.shape, dtype=np.int32)
    for seg in ann['segments_info']:
        cat_map[seg_id_map == seg['id']] = seg['category_id']

    return cat_map   # values in [0, 133]


def center_crop_resize_nearest(cat_map, size=224):
    """Center-crop to square then resize to size×size with nearest-neighbour.

    Must use nearest-neighbour (not bilinear) for label maps — interpolating
    class IDs would create spurious class values.
    """
    h, w = cat_map.shape
    s = min(h, w)
    top  = (h - s) // 2
    left = (w - s) // 2
    cat_map = cat_map[top:top + s, left:left + s]

    img = Image.fromarray(cat_map.astype(np.int32), mode='I')
    img = img.resize((size, size), Image.NEAREST)
    return np.array(img, dtype=np.int32)


# ── Tokenisation ─────────────────────────────────────────────────────────────

def flush_batch(stems, maps, out_dir, semseg_tok, device):
    """Tokenise a batch of semantic maps and save each as int16 .npy."""
    # Stack → (B, 224, 224) long; shift_idx_by_one to match SemsegTransform
    # used during VQVAE training: 0 (bg) → 1, cat 1-133 → 2-134
    batch = torch.stack(
        [torch.from_numpy(m).long() for m in maps]
    ).to(device) + 1                     # shift_idx_by_one

    with torch.no_grad():
        tokens = semseg_tok.tokenize(batch)        # (B, 14, 14)
        tokens = tokens.reshape(tokens.shape[0], -1)  # (B, 196)

    tokens_np = tokens.cpu().numpy().astype(np.int16)
    for i, stem in enumerate(stems):
        np.save(out_dir / f'{stem}.npy', tokens_np[i])

    del batch, tokens
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def tokenize_split(split_name, coco_dir, tokens_dir, semseg_tok,
                   batch_size, device, dry_run=False):
    ann_dir    = Path(coco_dir) / 'annotations'
    zip_path   = ann_dir / f'panoptic_{split_name}2017.zip'
    json_path  = ann_dir / f'panoptic_{split_name}2017.json'
    pan_dir    = f'panoptic_{split_name}2017'   # folder name inside the ZIP

    out_dir = Path(tokens_dir) / split_name / 'tok_semseg'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine which image stems to process ────────────────────────────────
    # Use existing tok_rgb stems → guarantees same seed=42 subset for train
    rgb_dir = Path(tokens_dir) / split_name / 'tok_rgb'
    if not rgb_dir.exists():
        raise FileNotFoundError(f'tok_rgb dir not found: {rgb_dir}')
    existing_stems = {p.stem for p in rgb_dir.glob('*.npy')}
    print(f'\n[{split_name}] tok_rgb stems  : {len(existing_stems)}')

    # Build JSON lookup
    stem_to_ann = build_stem_to_ann(json_path)
    valid_stems = sorted(existing_stems & set(stem_to_ann.keys()))
    print(f'[{split_name}] valid for semseg: {len(valid_stems)}')

    # Skip already tokenized
    done = {p.stem for p in out_dir.glob('*.npy')}
    stems = [s for s in valid_stems if s not in done]
    if dry_run:
        stems = stems[:8]
    print(f'[{split_name}] to tokenize     : {len(stems)}  (already done: {len(done)})')

    if len(stems) == 0:
        print(f'[{split_name}] Nothing to do — all files present.')
        return

    # ── Tokenise ──────────────────────────────────────────────────────────────
    batch_stems, batch_maps = [], []
    n_skipped = 0

    with zipfile.ZipFile(zip_path) as zf:
        for stem in tqdm(stems, desc=f'  {split_name} semseg'):
            cat_map = load_semantic_map(zf, stem_to_ann, pan_dir, stem)
            if cat_map is None:
                n_skipped += 1
                continue

            cat_map = center_crop_resize_nearest(cat_map, size=224)
            batch_stems.append(stem)
            batch_maps.append(cat_map)

            if len(batch_maps) == batch_size:
                flush_batch(batch_stems, batch_maps, out_dir, semseg_tok, device)
                batch_stems, batch_maps = [], []

        if batch_maps:   # flush remainder
            flush_batch(batch_stems, batch_maps, out_dir, semseg_tok, device)

    n_saved = len(list(out_dir.glob('*.npy')))
    print(f'[{split_name}] Saved {n_saved} tok_semseg files → {out_dir}')
    if n_skipped:
        print(f'[{split_name}] Skipped {n_skipped} images (missing in JSON or ZIP)')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('  COCO panoptic semseg tokenisation')
    print(f'  COCO dir   : {args.coco_dir}')
    print(f'  Tokens dir : {args.tokens_dir}')
    print(f'  Device     : {device}')
    if args.dry_run:
        print('  DRY RUN — 8 images per split only')
    print('=' * 60)

    # Load semseg VQ-VAE tokeniser
    print(f'\nLoading semseg tokenizer from {SEMSEG_TOKENIZER_REPO} ...')
    from fourm.vq.vqvae import VQVAE
    semseg_tok = VQVAE.from_pretrained(SEMSEG_TOKENIZER_REPO,
                                       cache_dir=args.hf_cache)
    semseg_tok = semseg_tok.to(device).eval()
    print('Tokenizer ready.\n')

    splits = []
    if args.splits in ('train', 'both'):
        splits.append('train')
    if args.splits in ('val', 'both'):
        splits.append('val')

    for split in splits:
        tokenize_split(
            split_name=split,
            coco_dir=args.coco_dir,
            tokens_dir=args.tokens_dir,
            semseg_tok=semseg_tok,
            batch_size=args.batch_size,
            device=device,
            dry_run=args.dry_run,
        )

    print('\n' + '=' * 60)
    print('  Tokenisation complete.')
    print(f'  Output: {args.tokens_dir}/{{train,val}}/tok_semseg/')
    print('  Next step: sbatch scripts/snellius/25_ft_semseg_coco_r8_attention.sh')
    print('             sbatch scripts/snellius/26_ft_multitask_coco_r8_attention.sh')
    print('=' * 60)


if __name__ == '__main__':
    main()
