#!/usr/bin/env python3
# prepare_coco_det.py
#
# Tokenizes COCO 2017 detection annotations → det tokens (text sequence).
#
# Pipeline per image:
#   instances JSON → bbox [x,y,w,h] → normalize by img size → [xmin,ymin,xmax,ymax] in [0,1]
#   → sort by dist-to-origin → serialize:
#       "v0=<xmin_q> v1=<ymin_q> v2=<xmax_q> v3=<ymax_q> <class_name> ..."
#   → tokenize with 30k wordpiece tokenizer → int16 .npy (shape 256,)
#
# Only processes images whose tok_rgb already exists (preserves seed=42 10k train subset).
#
# Usage:
#   python scripts/prepare_coco_det.py \
#       --coco_dir   /scratch-shared/tesr125338/coco \
#       --tokens_dir /scratch-shared/tesr125338/coco_tokens \
#   Or via SLURM: sbatch scripts/snellius/31_tokenize_coco_det.sh

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

TOKENIZER_PATH = str(
    REPO_ROOT / 'ml-4m' / 'fourm' / 'utils' / 'tokenizer' /
    'trained' / 'text_tokenizer_4m_wordpiece_30k.json'
)
MAX_DET_TOKENS = 256   # max sequence length for det modality
COORD_BINS     = 1000  # quantization bins (0-999)


def get_args():
    p = argparse.ArgumentParser(description='Tokenize COCO detection → det tokens')
    p.add_argument('--coco_dir',   required=True,
                   help='Root of COCO (contains annotations/instances_*.json)')
    p.add_argument('--tokens_dir', required=True,
                   help='Root of coco_tokens/ (contains train/tok_rgb/, val/tok_rgb/)')
    p.add_argument('--splits',     default='both', choices=['train', 'val', 'both'])
    p.add_argument('--dry_run',    action='store_true',
                   help='Process only first 8 images per split')
    return p.parse_args()


def quantize(norm_val):
    """Normalize float in [0,1] → integer in [0, COORD_BINS-1]."""
    return int(round(max(0.0, min(1.0, norm_val)) * (COORD_BINS - 1)))


def bboxes_to_det_string(bboxes_norm):
    """Convert list of (xmin,ymin,xmax,ymax,class_name) → det string.

    Sorted by distance-to-origin (top-left first), matching DetectionTransform
    default bbox_order='dist_to_orig' used in 4M-21.
    """
    # Sort by Euclidean distance of top-left corner from image origin
    bboxes_norm = sorted(bboxes_norm, key=lambda b: math.sqrt(b[0]**2 + b[1]**2))

    parts = []
    for xmin, ymin, xmax, ymax, cls in bboxes_norm:
        parts.append(
            f'v0={quantize(xmin)} v1={quantize(ymin)} '
            f'v2={quantize(xmax)} v3={quantize(ymax)} {cls}'
        )
    return ' '.join(parts)


def tokenize_det_split(split_name, coco_dir, tokens_dir, tok, dry_run=False):
    ann_path = Path(coco_dir) / 'annotations' / f'instances_{split_name}2017.json'
    out_dir  = Path(tokens_dir) / split_name / 'det'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only process images that already have tok_rgb (preserves seed=42 subset)
    rgb_dir = Path(tokens_dir) / split_name / 'tok_rgb'
    if not rgb_dir.exists():
        raise FileNotFoundError(f'tok_rgb dir not found: {rgb_dir}')
    existing_stems = {p.stem for p in rgb_dir.glob('*.npy')}
    print(f'\n[{split_name}] tok_rgb stems: {len(existing_stems)}')

    print(f'[{split_name}] Loading {ann_path} ...')
    with open(ann_path) as f:
        data = json.load(f)

    # Build lookups
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    img_id_to_info = {img['id']: img for img in data['images']}

    # Group annotations by image_id, skip crowd annotations
    img_id_to_anns = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            continue
        img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    # Filter to existing stems
    images = [img for img in data['images']
               if Path(img['file_name']).stem in existing_stems]
    if dry_run:
        images = images[:8]

    done    = {p.stem for p in out_dir.glob('*.npy')}
    images  = [img for img in images if Path(img['file_name']).stem not in done]
    print(f'[{split_name}] to tokenize: {len(images)}  (already done: {len(done)})')

    eos_id = tok.token_to_id('[EOS]')
    pad_id = 0
    n_skip = 0

    for img_info in tqdm(images, desc=f'  {split_name} det'):
        stem   = Path(img_info['file_name']).stem
        img_id = img_info['id']
        W, H   = img_info['width'], img_info['height']

        anns = img_id_to_anns.get(img_id, [])
        if not anns:
            # Save empty sequence (just EOS) for images with no detections
            arr = np.array([eos_id] + [pad_id] * (MAX_DET_TOKENS - 1), dtype=np.int16)
            np.save(out_dir / f'{stem}.npy', arr)
            continue

        # Convert COCO [x,y,w,h] → normalized [xmin,ymin,xmax,ymax]
        bboxes_norm = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            xmin, ymin = x / W,       y / H
            xmax, ymax = (x + w) / W, (y + h) / H
            # Clamp to [0,1] (some COCO annotations slightly exceed image bounds)
            xmin, ymin = max(0., xmin), max(0., ymin)
            xmax, ymax = min(1., xmax), min(1., ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            cls = cat_id_to_name[ann['category_id']]
            bboxes_norm.append((xmin, ymin, xmax, ymax, cls))

        if not bboxes_norm:
            n_skip += 1
            continue

        det_str = bboxes_to_det_string(bboxes_norm)
        encoded = tok.encode(det_str)
        ids     = encoded.ids

        # Truncate, append EOS, pad to MAX_DET_TOKENS
        ids = ids[: MAX_DET_TOKENS - 1]
        ids.append(eos_id)
        ids += [pad_id] * (MAX_DET_TOKENS - len(ids))

        arr = np.array(ids, dtype=np.int16)
        np.save(out_dir / f'{stem}.npy', arr)

    n_saved = len(list(out_dir.glob('*.npy')))
    print(f'[{split_name}] Saved {n_saved} det token files → {out_dir}')
    if n_skip:
        print(f'[{split_name}] Skipped {n_skip} images (invalid boxes)')


def main():
    args = get_args()

    from tokenizers import Tokenizer
    print(f'Loading tokenizer from {TOKENIZER_PATH} ...')
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    print('Tokenizer ready.\n')

    splits = []
    if args.splits in ('train', 'both'):
        splits.append('train')
    if args.splits in ('val', 'both'):
        splits.append('val')

    for split in splits:
        tokenize_det_split(split, args.coco_dir, args.tokens_dir, tok, args.dry_run)

    print('\n=== Done ===')
    for split in splits:
        n = len(list((Path(args.tokens_dir) / split / 'det').glob('*.npy')))
        print(f'  [{split}] det tokens: {n}')
    print('Next: sbatch scripts/snellius/32_ft_det_coco_r8_attention.sh')


if __name__ == '__main__':
    main()
