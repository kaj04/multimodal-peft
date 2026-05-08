#!/usr/bin/env python3
# prepare_coco.py
#
# Tokenizes COCO 2017 for 4M-PEFT captioning experiment.
#
# Phase 1 — RGB tokens (GPU, ~2h for 118k images):
#   coco/train2017/*.jpg  →  coco_tokens/train/tok_rgb/000000XXXXXX.npy
#   coco/val2017/*.jpg    →  coco_tokens/val/tok_rgb/000000XXXXXX.npy
#
# Phase 2 — Caption tokens (CPU, ~5 min):
#   annotations/captions_train2017.json  →  coco_tokens/train/caption/000000XXXXXX.npy
#   annotations/captions_val2017.json    →  coco_tokens/val/caption/000000XXXXXX.npy
#
# Token format (identical to prepare_nyu.py):
#   - RGB:     int16 .npy, shape (196,)  — 14×14 VQ codes, vocab 16384
#   - Caption: int16 .npy, shape (64,)   — wordpiece IDs, padded with 0
#              first caption per image selected deterministically
#
# File naming: COCO image ID as stem (e.g. 000000123456.npy) so tok_rgb
# and tok_caption can be paired by filename — same pattern as tok_rgb/tok_depth in NYU.
#
# Usage:
#   python scripts/prepare_coco.py \
#       --coco_dir   /scratch-shared/tesr125338/coco \
#       --output_dir /scratch-shared/tesr125338/coco_tokens \
#       --hf_cache   /scratch-shared/tesr125338/hf_cache
#
# Or via SLURM:
#   sbatch scripts/snellius/09_tokenize_coco.sh

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

RGB_TOKENIZER_REPO = 'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448'
CAPTION_TOKENIZER_PATH = str(
    REPO_ROOT / 'ml-4m' / 'fourm' / 'utils' / 'tokenizer' /
    'trained' / 'text_tokenizer_4m_wordpiece_30k.json'
)
MAX_CAPTION_LEN = 64   # tokens; COCO captions avg ~15 wordpiece tokens


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Prepare COCO 2017 tokens for 4M-PEFT')
    p.add_argument('--coco_dir',   required=True,
                   help='Root of downloaded COCO (contains train2017/, val2017/, annotations/)')
    p.add_argument('--output_dir', required=True,
                   help='Where to save tokenized .npy files')
    p.add_argument('--hf_cache',   default=None,
                   help='HuggingFace cache dir')
    p.add_argument('--batch_size', default=64, type=int,
                   help='Images per GPU batch for RGB tokenization')
    p.add_argument('--device',     default='cuda')
    p.add_argument('--splits',     default='both',
                   choices=['train', 'val', 'both'])
    p.add_argument('--phase',      default='both',
                   choices=['rgb', 'caption', 'both'],
                   help='Which tokenization phase to run')
    p.add_argument('--dry_run',    action='store_true',
                   help='Process only the first 8 images per split')
    return p.parse_args()


# ── Preprocessing — identical to prepare_nyu.py ───────────────────────────

def preprocess_rgb_batch(pil_images, size: int = 224) -> torch.Tensor:
    """Resize + normalise RGB images to [-1, 1] tensor (B, 3, H, W).

    Matches RGBTransform(imagenet_default_mean_and_std=False) used by
    the 4M RGB tokenizer (same as prepare_nyu.py).
    """
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return torch.stack([tf(img.convert('RGB')) for img in pil_images])


# ── Phase 1: RGB tokenization ─────────────────────────────────────────────

def tokenize_rgb_split(
        split_name: str,
        img_dir: Path,
        out_dir: Path,
        rgb_tok,
        batch_size: int,
        device: torch.device,
        dry_run: bool = False,
):
    """Tokenize all JPEGs in img_dir → out_dir/<stem>.npy (int16, shape 196)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob('*.jpg'))
    if not img_paths:
        raise RuntimeError(f'No .jpg files found in {img_dir}')
    if dry_run:
        img_paths = img_paths[:8]

    print(f'\n[RGB {split_name}] {len(img_paths)} images → {out_dir}')

    for i in tqdm(range(0, len(img_paths), batch_size), desc=f'  {split_name} RGB'):
        batch_paths = img_paths[i: i + batch_size]

        # Load and preprocess
        pil_imgs = [Image.open(p) for p in batch_paths]
        rgb_t = preprocess_rgb_batch(pil_imgs).to(device)

        with torch.no_grad():
            tokens = rgb_tok.tokenize(rgb_t)            # (B, 14, 14)
            tokens = tokens.view(tokens.shape[0], -1)   # (B, 196)

        tokens_np = tokens.cpu().numpy().astype(np.int16)

        for j, path in enumerate(batch_paths):
            np.save(out_dir / f'{path.stem}.npy', tokens_np[j])

        del rgb_t, tokens
        torch.cuda.empty_cache()

    print(f'  Saved {len(img_paths)} RGB token files → {out_dir}')


# ── Phase 2: Caption tokenization ────────────────────────────────────────

def tokenize_captions_split(
        split_name: str,
        ann_path: Path,
        out_dir: Path,
        tok,
        max_len: int = MAX_CAPTION_LEN,
        dry_run: bool = False,
):
    """Tokenize COCO captions → out_dir/<image_id_12digit>.npy (int16, shape max_len).

    Selects the FIRST caption per image deterministically.
    Pads with 0 (padding_idx=0 in SequenceDecoderEmbedding).
    Appends [EOS] before padding.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n[Caption {split_name}] Reading {ann_path} ...')
    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Map image_id → first caption (annotations are not sorted, use min ann_id)
    id_to_caption = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_caption:
            id_to_caption[img_id] = ann
        else:
            # Keep annotation with smallest id → deterministic "first" caption
            if ann['id'] < id_to_caption[img_id]['id']:
                id_to_caption[img_id] = ann

    # COCO image filenames: 000000123456.jpg → stem = 000000123456
    images = data['images']
    if dry_run:
        images = images[:8]

    eos_id = tok.token_to_id('[EOS]')
    if eos_id is None:
        # Fallback: some tokenizers use [SEP] as sequence terminator
        eos_id = tok.token_to_id('[SEP]')
    pad_id = 0

    skipped = 0
    for img_info in tqdm(images, desc=f'  {split_name} captions'):
        img_id   = img_info['id']
        stem     = Path(img_info['file_name']).stem   # e.g. '000000123456'

        if img_id not in id_to_caption:
            skipped += 1
            continue

        caption = id_to_caption[img_id]['caption']
        encoded = tok.encode(caption)
        ids     = encoded.ids                          # list[int]

        # Truncate to max_len-1, append EOS, pad to max_len
        ids = ids[: max_len - 1]
        ids.append(eos_id)
        ids += [pad_id] * (max_len - len(ids))

        arr = np.array(ids, dtype=np.int16)            # (max_len,)
        np.save(out_dir / f'{stem}.npy', arr)

    n_saved = len(images) - skipped
    print(f'  Saved {n_saved} caption token files → {out_dir}')
    if skipped:
        print(f'  Skipped {skipped} images with no caption annotation')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    coco_dir = Path(args.coco_dir)
    out_dir  = Path(args.output_dir)

    print(f'Device     : {device}')
    print(f'COCO dir   : {coco_dir}')
    print(f'Output dir : {out_dir}')
    if args.dry_run:
        print('DRY RUN — processing 8 images per split only')

    splits = []
    if args.splits in ('train', 'both'):
        splits.append('train')
    if args.splits in ('val', 'both'):
        splits.append('val')

    # ── Phase 1: RGB tokenization (GPU) ──────────────────────────────────────
    if args.phase in ('rgb', 'both'):
        print('\n=== Phase 1: RGB tokenization ===')
        from fourm.vq.vqvae import VQ
        print(f'Loading RGB tokenizer from {RGB_TOKENIZER_REPO} ...')
        rgb_tok = VQ.from_pretrained(RGB_TOKENIZER_REPO, cache_dir=args.hf_cache)
        rgb_tok.eval().to(device)
        print(f'  Codebook size: {rgb_tok.quantize.codebook_size}')

        for split in splits:
            img_dir = coco_dir / f'{split}2017'
            tok_out = out_dir  / split / 'tok_rgb'
            tokenize_rgb_split(split, img_dir, tok_out, rgb_tok,
                               args.batch_size, device, args.dry_run)

        del rgb_tok
        torch.cuda.empty_cache()

    # ── Phase 2: Caption tokenization (CPU) ──────────────────────────────────
    if args.phase in ('caption', 'both'):
        print('\n=== Phase 2: Caption tokenization ===')
        from tokenizers import Tokenizer
        print(f'Loading caption tokenizer from {CAPTION_TOKENIZER_PATH} ...')
        cap_tok = Tokenizer.from_file(CAPTION_TOKENIZER_PATH)

        ann_map = {
            'train': coco_dir / 'annotations' / 'captions_train2017.json',
            'val':   coco_dir / 'annotations' / 'captions_val2017.json',
        }
        for split in splits:
            # Folder name must match target_mod key used by run_finetuning_lora.py:
            #   target_mod='caption' → target_folder='caption' (split on '@', take [0])
            cap_out = out_dir / split / 'caption'
            tokenize_captions_split(split, ann_map[split], cap_out,
                                    cap_tok, MAX_CAPTION_LEN, args.dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n=== Done ===')
    for split in splits:
        rgb_n = len(list((out_dir / split / 'tok_rgb').glob('*.npy'))) if (out_dir / split / 'tok_rgb').exists() else 0
        cap_n = len(list((out_dir / split / 'caption').glob('*.npy')))  if (out_dir / split / 'caption').exists()  else 0
        print(f'  [{split}]  tok_rgb: {rgb_n}  tok_caption: {cap_n}')
    print(f'\nNext step: sbatch scripts/snellius/10_ft_caption_coco.sh')


if __name__ == '__main__':
    main()
