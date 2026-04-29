#!/usr/bin/env python3
# Copyright 2024 Francesco Colasurdo / AMORe Lab, TU/e
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# -----------------------------------------------------------------------
# prepare_nyu.py
#
# One-shot data preparation script for 4M-PEFT fine-tuning on NYUv2.
#
# What it does:
#   1. Downloads the 4M tokenizer checkpoints from HuggingFace Hub
#      (RGB 16k, depth 8k) into --tokenizers_dir
#   2. Downloads NYUv2 RGB+depth from HuggingFace datasets
#      (sayakpaul/nyu_depth_v2, 795 train + 654 val)
#   3. Tokenizes every image with the pretrained 4M VQ-VAE encoders
#      and saves one .npy file (shape 196,) per image per modality
#
# Output layout (ready for TokenizedDenseDataset in run_finetuning_lora.py):
#
#   <output_dir>/
#     train/
#       tok_rgb/    00000.npy … 00794.npy   (int16, shape 196)
#       tok_depth/  00000.npy … 00794.npy   (int16, shape 196)
#     val/
#       tok_rgb/    00000.npy … 00653.npy
#       tok_depth/  00000.npy … 00653.npy
#
# Semseg note:
#   NYUv2 semseg labels (40 classes) are NOT in the HuggingFace split used
#   here. They require the original nyu_depth_v2_labeled.mat from NYU.
#   This script only prepares tok_rgb and tok_depth (sufficient for the
#   depth-estimation fine-tuning baseline).
#   A future script (prepare_nyu_semseg.py) will add tok_semseg once the
#   .mat file is available.
#
# Usage (on Snellius, inside the 4m-peft conda env):
#
#   python scripts/prepare_nyu.py \
#       --output_dir /scratch-shared/tesr125338/nyu_tokens \
#       --tokenizers_dir /scratch-shared/tesr125338/tokenizer_ckpts \
#       --hf_cache  /scratch-shared/tesr125338/hf_cache
#
# Or via SLURM:
#   sbatch scripts/snellius/01_prepare_nyu.sh
# -----------------------------------------------------------------------

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Prepare NYUv2 tokens for 4M-PEFT')
    p.add_argument('--output_dir', required=True,
                   help='Where to save tokenized .npy files')
    p.add_argument('--tokenizers_dir', required=True,
                   help='Local directory to store/load tokenizer .pth checkpoints')
    p.add_argument('--hf_cache', default=None,
                   help='HuggingFace cache dir (set to $HF_HOME on Snellius)')
    p.add_argument('--batch_size', default=32, type=int,
                   help='Images per GPU batch during tokenization')
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--device', default='cuda',
                   help='Device for tokenizer inference')
    p.add_argument('--splits', default='both',
                   choices=['train', 'val', 'both'],
                   help='Which NYUv2 splits to process')
    p.add_argument('--dry_run', action='store_true',
                   help='Process only the first 8 images per split (sanity check)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Download tokenizer checkpoints from HuggingFace Hub
# ---------------------------------------------------------------------------

# Mapping: (HF repo id, local filename stem, modality)
TOKENIZER_REPOS = [
    ('EPFL-VILAB/4M_tokenizers_rgb_16k_224-448',   'rgb_16k_224'),
    ('EPFL-VILAB/4M_tokenizers_depth_8k_224-448',  'depth_8k_224'),
]


def download_tokenizer(repo_id: str, local_stem: str,
                       tokenizers_dir: str, hf_cache: str) -> str:
    """Download a tokenizer .pth from HuggingFace Hub if not already present.

    Returns the local path to the .pth file.
    """
    target = os.path.join(tokenizers_dir, f'{local_stem}.pth')
    if os.path.exists(target):
        print(f'  [cache] {local_stem}.pth already exists, skipping download.')
        return target

    from huggingface_hub import snapshot_download
    print(f'  Downloading {repo_id} ...')
    repo_dir = snapshot_download(repo_id, cache_dir=hf_cache)

    # Find the single .pth file inside the downloaded repo
    pth_candidates = glob.glob(os.path.join(repo_dir, '*.pth'))
    if not pth_candidates:
        raise RuntimeError(
            f'No .pth file found in downloaded repo {repo_dir}. '
            'Check the HuggingFace repo structure.')
    if len(pth_candidates) > 1:
        print(f'  Warning: multiple .pth files found, using: {pth_candidates[0]}')

    import shutil
    os.makedirs(tokenizers_dir, exist_ok=True)
    shutil.copy(pth_candidates[0], target)
    print(f'  Saved → {target}')
    return target


# ---------------------------------------------------------------------------
# Step 2: Load tokenizers using fourm's get_image_tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(local_stem: str, tokenizers_dir: str, device: str):
    """Load a VQ tokenizer (encoder only) from a local .pth checkpoint."""
    # Add ml-4m to path so fourm is importable
    repo_root = Path(__file__).resolve().parent.parent
    ml4m_path = str(repo_root / 'ml-4m')
    if ml4m_path not in sys.path:
        sys.path.insert(0, ml4m_path)

    from fourm.vq import get_image_tokenizer
    model, _ = get_image_tokenizer(
        local_stem,
        tokenizers_root=tokenizers_dir,
        encoder_only=True,
        device=device,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Step 3: Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_rgb_batch(pil_images, size: int = 224) -> torch.Tensor:
    """Centre-crop + resize RGB images → tensor in [-1, 1].

    Matches RGBTransform(imagenet_default_mean_and_std=False) used by
    the 4M RGB tokenizer: normalise with mean=0.5 std=0.5 → [-1, 1].
    """
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),                        # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),    # → [-1, 1]
    ])
    return torch.stack([tf(img.convert('RGB')) for img in pil_images])


def pil_depth_to_uint16_array(pil_img) -> np.ndarray:
    """Convert a PIL depth image to a uint16 numpy array in [0, 65535].

    NYUv2 depth in sayakpaul/nyu_depth_v2 is stored as a PIL image in
    mode 'I' (32-bit int) or 'I;16' (16-bit). The raw values represent
    depth in millimetres * (65535 / max_depth_mm).  We normalise to the
    full uint16 range so DepthTransform.depth_to_tensor can work with it
    unchanged (divide by 65535).
    """
    arr = np.array(pil_img)  # could be int32 or float
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # Depth in metres (common in HF dataset) → map to [0, 65535]
        arr_clipped = np.clip(arr, 0, None)
        max_val = arr_clipped.max()
        if max_val > 0:
            arr = (arr_clipped / max_val * 65535).astype(np.uint16)
        else:
            arr = arr.astype(np.uint16)
    else:
        arr = arr.astype(np.uint16)
    return arr


def truncated_depth_standardization(depth: torch.Tensor,
                                    thresh: float = 0.1) -> torch.Tensor:
    """Truncated depth standardization (mirrors DepthTransform.truncated_depth_standardization)."""
    trunc = torch.sort(depth.reshape(-1))[0]
    trunc = trunc[int(thresh * len(trunc)): int((1 - thresh) * len(trunc))]
    return (depth - trunc.mean()) / torch.sqrt(trunc.var() + 1e-6)


def preprocess_depth_batch(pil_images, size: int = 224) -> torch.Tensor:
    """Resize depth maps and apply DepthTransform.postprocess logic.

    Output: (B, 1, H, W) float32 tensor, standardized.
    """
    tensors = []
    for img in pil_images:
        arr = pil_depth_to_uint16_array(img)
        pil_resized = Image.fromarray(arr).resize((size, size), Image.NEAREST)
        arr_resized = np.array(pil_resized, dtype=np.float32)

        depth = torch.tensor(arr_resized / (2**16 - 1.0)).unsqueeze(0)  # [1, H, W]
        depth = truncated_depth_standardization(depth)
        tensors.append(depth)
    return torch.stack(tensors)  # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Step 4: Tokenize one split and save .npy files
# ---------------------------------------------------------------------------

def tokenize_split(
        split_name: str,
        hf_split_key: str,
        dataset,
        rgb_tokenizer,
        depth_tokenizer,
        output_dir: str,
        batch_size: int,
        device: torch.device,
        dry_run: bool = False,
):
    """Tokenize all images in one split and save as .npy files."""
    rgb_out = Path(output_dir) / split_name / 'tok_rgb'
    depth_out = Path(output_dir) / split_name / 'tok_depth'
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    data = dataset[hf_split_key]
    n = min(8, len(data)) if dry_run else len(data)
    print(f'\n[{split_name}] Tokenizing {n} images'
          f'{" (DRY RUN)" if dry_run else ""} ...')

    for i in tqdm(range(0, n, batch_size)):
        batch = data[i: i + batch_size]
        rgb_pil   = batch['image']        # list of PIL images
        depth_pil = batch['depth_map']    # list of PIL images

        # ── RGB ──────────────────────────────────────────────────────────
        rgb_tensor = preprocess_rgb_batch(rgb_pil).to(device)
        with torch.no_grad():
            rgb_tokens = rgb_tokenizer.tokenize(rgb_tensor)   # (B, 14, 14)
            rgb_tokens = rgb_tokens.view(rgb_tokens.shape[0], -1)  # (B, 196)
        rgb_tokens = rgb_tokens.cpu().numpy().astype(np.int16)

        # ── Depth ────────────────────────────────────────────────────────
        depth_tensor = preprocess_depth_batch(depth_pil).to(device)
        with torch.no_grad():
            depth_tokens = depth_tokenizer.tokenize(depth_tensor)  # (B, 14, 14)
            depth_tokens = depth_tokens.view(depth_tokens.shape[0], -1)
        depth_tokens = depth_tokens.cpu().numpy().astype(np.int16)

        # ── Save ─────────────────────────────────────────────────────────
        for j, global_idx in enumerate(range(i, min(i + batch_size, n))):
            name = f'{global_idx:05d}'
            np.save(rgb_out   / f'{name}.npy', rgb_tokens[j])
            np.save(depth_out / f'{name}.npy', depth_tokens[j])

    print(f'  → {rgb_out}:   {n} files saved')
    print(f'  → {depth_out}: {n} files saved')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 1. Download tokenizer checkpoints ──────────────────────────────────
    print('\n=== Step 1: Download tokenizer checkpoints ===')
    os.makedirs(args.tokenizers_dir, exist_ok=True)
    for repo_id, local_stem in TOKENIZER_REPOS:
        download_tokenizer(repo_id, local_stem, args.tokenizers_dir, args.hf_cache)

    # ── 2. Load tokenizers ─────────────────────────────────────────────────
    print('\n=== Step 2: Load tokenizers ===')
    rgb_tokenizer   = load_tokenizer('rgb_16k_224',  args.tokenizers_dir, str(device))
    depth_tokenizer = load_tokenizer('depth_8k_224', args.tokenizers_dir, str(device))
    print('  Both tokenizers loaded OK.')

    # ── 3. Download NYUv2 from HuggingFace ─────────────────────────────────
    print('\n=== Step 3: Download NYUv2 dataset ===')
    from datasets import load_dataset
    print('  Loading sayakpaul/nyu_depth_v2 ...')
    dataset = load_dataset('sayakpaul/nyu_depth_v2',
                           cache_dir=args.hf_cache)
    print(f'  Dataset: {dataset}')

    # ── 4. Tokenize ────────────────────────────────────────────────────────
    print('\n=== Step 4: Tokenize ===')
    splits_to_run = {
        'train': 'train',
        'val':   'validation',
    }
    if args.splits == 'train':
        splits_to_run = {'train': 'train'}
    elif args.splits == 'val':
        splits_to_run = {'val': 'validation'}

    for out_split, hf_key in splits_to_run.items():
        tokenize_split(
            split_name=out_split,
            hf_split_key=hf_key,
            dataset=dataset,
            rgb_tokenizer=rgb_tokenizer,
            depth_tokenizer=depth_tokenizer,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=device,
            dry_run=args.dry_run,
        )

    print('\n=== Done ===')
    print(f'Tokens saved to: {args.output_dir}')
    print('Expected structure:')
    for split in splits_to_run:
        print(f'  {args.output_dir}/{split}/tok_rgb/    (*.npy, shape 196)')
        print(f'  {args.output_dir}/{split}/tok_depth/  (*.npy, shape 196)')
    print('\nNext step: sbatch scripts/snellius/03_ft_depth_nyu.sh')


if __name__ == '__main__':
    main()
