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
#   1. Loads the 4M VQ tokenizers directly from HuggingFace Hub via
#      VQ.from_pretrained() (PyTorchModelHubMixin — no .pth files needed)
#   2. Downloads NYUv2 RGB+depth from HuggingFace datasets
#      (sayakpaul/nyu_depth_v2, 795 train + 654 val)
#   3. Tokenizes every image and saves one .npy file (shape 196,) per
#      image per modality
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
#   NYUv2 semseg labels (40 classes) are NOT in the HuggingFace dataset
#   used here. They require the original nyu_depth_v2_labeled.mat.
#   This script prepares tok_rgb + tok_depth only — sufficient for the
#   depth-estimation fine-tuning baseline (L1).
#
# Usage (on Snellius, inside the 4m-peft conda env):
#
#   python scripts/prepare_nyu.py \
#       --output_dir /scratch-shared/tesr125338/nyu_tokens \
#       --hf_cache   /scratch-shared/tesr125338/hf_cache
#
# Or via SLURM:
#   sbatch scripts/snellius/01_prepare_nyu.sh
# -----------------------------------------------------------------------

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# HuggingFace Hub repo IDs for the 4M tokenizers
RGB_TOKENIZER_REPO   = 'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448'
DEPTH_TOKENIZER_REPO = 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448'


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Prepare NYUv2 tokens for 4M-PEFT')
    p.add_argument('--output_dir', required=True,
                   help='Where to save tokenized .npy files')
    p.add_argument('--hf_cache', default=None,
                   help='HuggingFace cache dir (set to $HF_HOME on Snellius)')
    p.add_argument('--batch_size', default=32, type=int,
                   help='Images per GPU batch during tokenization')
    p.add_argument('--device', default='cuda',
                   help='Device for tokenizer inference')
    p.add_argument('--splits', default='both',
                   choices=['train', 'val', 'both'],
                   help='Which NYUv2 splits to process')
    p.add_argument('--dry_run', action='store_true',
                   help='Process only the first 8 images per split (sanity check)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tokenizer loading — uses VQ.from_pretrained (PyTorchModelHubMixin)
# ---------------------------------------------------------------------------

def load_tokenizer(repo_id: str, hf_cache: str, device: str):
    """Load a 4M VQ encoder from HuggingFace Hub.

    VQ has PyTorchModelHubMixin so from_pretrained reads config.json +
    model.safetensors directly — no manual .pth download needed.
    """
    # Add ml-4m to sys.path so fourm is importable when called as a script
    repo_root = Path(__file__).resolve().parent.parent
    ml4m_path = str(repo_root / 'ml-4m')
    if ml4m_path not in sys.path:
        sys.path.insert(0, ml4m_path)

    from fourm.vq.vqvae import VQ

    print(f'  Loading tokenizer from {repo_id} ...')
    model = VQ.from_pretrained(repo_id, cache_dir=hf_cache)
    model.eval()
    model.to(device)
    print(f'  OK — codebook size: {model.quantize.codebook_size}')
    return model


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_rgb_batch(pil_images, size: int = 224) -> torch.Tensor:
    """Resize + normalise RGB images to [-1, 1] tensor (B, 3, H, W).

    Matches RGBTransform(imagenet_default_mean_and_std=False) used by
    the 4M RGB DiVAE tokenizer.
    """
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),                          # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),      # → [-1, 1]
    ])
    return torch.stack([tf(img.convert('RGB')) for img in pil_images])


def _depth_pil_to_float_array(pil_img) -> np.ndarray:
    """Convert a PIL depth image to a float array normalised to [0, 65535].

    sayakpaul/nyu_depth_v2 stores depth as float32 in metres.
    DepthTransform.depth_to_tensor() divides by (2^16 - 1), so we scale
    metric depth to fill the uint16 range before handing off.
    """
    arr = np.array(pil_img, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 65535.0
    return arr


def _truncated_depth_standardization(depth: torch.Tensor,
                                     thresh: float = 0.1) -> torch.Tensor:
    """Mirror of DepthTransform.truncated_depth_standardization."""
    flat = torch.sort(depth.reshape(-1))[0]
    lo, hi = int(thresh * len(flat)), int((1 - thresh) * len(flat))
    trunc = flat[lo:hi]
    return (depth - trunc.mean()) / torch.sqrt(trunc.var() + 1e-6)


def preprocess_depth_batch(pil_images, size: int = 224) -> torch.Tensor:
    """Resize + standardise depth maps to (B, 1, H, W) float tensor.

    Mirrors DepthTransform(standardize_depth=True).postprocess().
    """
    tensors = []
    for img in pil_images:
        arr = _depth_pil_to_float_array(img)
        pil_resized = Image.fromarray(arr).resize((size, size), Image.NEAREST)
        arr_r = np.array(pil_resized, dtype=np.float32)
        depth = torch.tensor(arr_r / (2**16 - 1.0)).unsqueeze(0)  # (1, H, W)
        depth = _truncated_depth_standardization(depth)
        tensors.append(depth)
    return torch.stack(tensors)  # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Tokenize a split from HDF5/WebDataset tars — STREAMING (no OOM)
# ---------------------------------------------------------------------------

def _iter_h5_samples(tar_path: str):
    """Generator: yield (rgb_pil, depth_pil) for every .h5 in a tar archive.

    Memory stays bounded: one sample at a time is in RAM. The generator is
    exhausted sample-by-sample, so the tarfile stays open only for the
    duration of one tar.
    """
    import tarfile, h5py

    with tarfile.open(tar_path, 'r') as tf:
        members = [m for m in tf.getmembers() if m.name.endswith('.h5')]
        members.sort(key=lambda m: m.name)   # deterministic order
        for member in members:
            raw = tf.extractfile(member).read()
            with h5py.File(io.BytesIO(raw), 'r') as f:
                rgb   = f['rgb'][:]    # (3, H, W) uint8
                depth = f['depth'][:]  # (H, W)   float32 metres
            rgb_img = Image.fromarray(rgb.transpose(1, 2, 0))
            dep_img = Image.fromarray(depth, mode='F')
            yield rgb_img, dep_img


def tokenize_split_from_tars(
        split_name: str,
        tar_list: list,
        rgb_tok,
        depth_tok,
        output_dir: str,
        batch_size: int,
        device: torch.device,
        dry_run: bool = False,
):
    """Stream-tokenize one dataset split from a list of .tar files.

    Memory footprint:
      - GPU: one batch of tokenizer activations at a time
      - CPU: at most `batch_size` PIL images at a time
      - Disk: .npy written immediately after each batch

    Never loads the full split into RAM.
    """
    rgb_out   = Path(output_dir) / split_name / 'tok_rgb'
    depth_out = Path(output_dir) / split_name / 'tok_depth'
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    max_samples = 8 if dry_run else None   # None = unlimited
    global_idx  = 0                        # .npy file counter
    rgb_buf, dep_buf = [], []              # accumulate one batch

    def flush_batch(rgb_list, dep_list, start_idx):
        """Tokenize a full batch and save .npy files. Frees GPU memory."""
        rgb_t = preprocess_rgb_batch(rgb_list).to(device)
        dep_t = preprocess_depth_batch(dep_list).to(device)

        with torch.no_grad():
            rgb_tok_out = rgb_tok.tokenize(rgb_t)
            rgb_tok_out = rgb_tok_out.view(rgb_tok_out.shape[0], -1)   # (B, 196)
            dep_tok_out = depth_tok.tokenize(dep_t)
            dep_tok_out = dep_tok_out.view(dep_tok_out.shape[0], -1)

        rgb_np = rgb_tok_out.cpu().numpy().astype(np.int16)
        dep_np = dep_tok_out.cpu().numpy().astype(np.int16)

        for j in range(len(rgb_list)):
            name = f'{start_idx + j:05d}'
            np.save(rgb_out   / f'{name}.npy', rgb_np[j])
            np.save(depth_out / f'{name}.npy', dep_np[j])

        # Explicitly free GPU tensors
        del rgb_t, dep_t, rgb_tok_out, dep_tok_out
        torch.cuda.empty_cache()

    n_tars = len(tar_list)
    print(f'\n[{split_name}] {n_tars} tars{" (DRY RUN — max 8 samples)" if dry_run else ""}')

    done = False
    for tar_idx, tar_path in enumerate(tqdm(tar_list, desc=f'  {split_name} tars')):
        if done:
            break
        for rgb_img, dep_img in _iter_h5_samples(tar_path):
            rgb_buf.append(rgb_img)
            dep_buf.append(dep_img)

            if len(rgb_buf) == batch_size:
                flush_batch(rgb_buf, dep_buf, global_idx)
                global_idx += len(rgb_buf)
                rgb_buf, dep_buf = [], []

            if max_samples is not None and global_idx + len(rgb_buf) >= max_samples:
                done = True
                break   # stop reading this tar

    # Flush any remaining partial batch
    if rgb_buf:
        if max_samples is not None:
            keep = max_samples - global_idx
            rgb_buf, dep_buf = rgb_buf[:keep], dep_buf[:keep]
        if rgb_buf:
            flush_batch(rgb_buf, dep_buf, global_idx)
            global_idx += len(rgb_buf)

    print(f'  Saved {global_idx} samples')
    print(f'  tok_rgb   → {rgb_out}')
    print(f'  tok_depth → {depth_out}')
    return global_idx


# ---------------------------------------------------------------------------
# Tokenize one split — parquet / HuggingFace datasets path (legacy)
# ---------------------------------------------------------------------------

def tokenize_split(
        split_name: str,
        hf_split_key: str,
        dataset,
        rgb_tok,
        depth_tok,
        output_dir: str,
        batch_size: int,
        device: torch.device,
        dry_run: bool = False,
):
    rgb_out   = Path(output_dir) / split_name / 'tok_rgb'
    depth_out = Path(output_dir) / split_name / 'tok_depth'
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    data = dataset[hf_split_key]
    n = min(8, len(data)) if dry_run else len(data)
    print(f'\n[{split_name}] {n} images{" (DRY RUN)" if dry_run else ""}')

    for i in tqdm(range(0, n, batch_size)):
        batch      = data[i: i + batch_size]
        rgb_pil    = batch['image']        # list[PIL]
        depth_pil  = batch['depth_map']    # list[PIL]

        # RGB
        rgb_t = preprocess_rgb_batch(rgb_pil).to(device)
        with torch.no_grad():
            rgb_tokens = rgb_tok.tokenize(rgb_t)            # (B, 14, 14)
            rgb_tokens = rgb_tokens.view(rgb_tokens.shape[0], -1)  # (B, 196)
        rgb_np = rgb_tokens.cpu().numpy().astype(np.int16)

        # Depth
        dep_t = preprocess_depth_batch(depth_pil).to(device)
        with torch.no_grad():
            dep_tokens = depth_tok.tokenize(dep_t)
            dep_tokens = dep_tokens.view(dep_tokens.shape[0], -1)
        dep_np = dep_tokens.cpu().numpy().astype(np.int16)

        # Save
        for j, idx in enumerate(range(i, min(i + batch_size, n))):
            name = f'{idx:05d}'
            np.save(rgb_out   / f'{name}.npy', rgb_np[j])
            np.save(depth_out / f'{name}.npy', dep_np[j])

    print(f'  tok_rgb   → {rgb_out}   ({n} files)')
    print(f'  tok_depth → {depth_out} ({n} files)')


# ---------------------------------------------------------------------------
# NYUv2 dataset discovery — returns tar paths OR HF dataset
# ---------------------------------------------------------------------------

def get_nyu_data_source(hf_cache: str):
    """Locate or download NYUv2 and return a data source descriptor.

    Returns one of:
      {'format': 'tars',  'train': [path,...], 'val': [path,...]}
      {'format': 'hf',    'dataset': <HF DatasetDict>}

    The tar format (WebDataset) is the actual storage format used by
    sayakpaul/nyu_depth_v2 on HuggingFace Hub.
    """
    import glob
    from huggingface_hub import snapshot_download

    # ── Try direct HF load first (works on datasets<4.0) ────────────────
    try:
        from datasets import load_dataset
        ds = load_dataset('sayakpaul/nyu_depth_v2',
                          cache_dir=hf_cache,
                          trust_remote_code=True)
        print(f'  HF dataset loaded directly: {ds}')
        return {'format': 'hf', 'dataset': ds}
    except Exception:
        pass

    # ── Robust path: snapshot_download ──────────────────────────────────
    print('  datasets>=4.0 detected — using snapshot_download ...')
    repo_dir = snapshot_download(
        repo_id='sayakpaul/nyu_depth_v2',
        repo_type='dataset',
        cache_dir=hf_cache,
        ignore_patterns=['*.py'],
    )
    print(f'  Snapshot at: {repo_dir}')

    # Debug: list files
    all_files = []
    for root, _, files in os.walk(repo_dir):
        for f in files:
            all_files.append(os.path.relpath(os.path.join(root, f), repo_dir))
    print(f'  Files in repo ({len(all_files)} total): {all_files[:15]}')

    parquet_files = glob.glob(os.path.join(repo_dir, '**', '*.parquet'), recursive=True)
    tar_files     = glob.glob(os.path.join(repo_dir, '**', '*.tar'),     recursive=True)

    if parquet_files:
        from datasets import load_dataset
        train_files = sorted(f for f in parquet_files
                             if 'train' in os.path.basename(f).lower())
        val_files   = sorted(f for f in parquet_files
                             if any(s in os.path.basename(f).lower()
                                    for s in ('val', 'test')))
        print(f'  Parquet — train: {len(train_files)}, val: {len(val_files)}')
        ds = load_dataset('parquet', data_files={
            'train': train_files, 'validation': val_files})
        print(f'  {ds}')
        return {'format': 'hf', 'dataset': ds}

    elif tar_files:
        train_tars = sorted(f for f in tar_files
                            if 'train' in os.path.basename(f).lower())
        val_tars   = sorted(f for f in tar_files
                            if any(s in os.path.basename(f).lower()
                                   for s in ('val', 'test')))
        print(f'  WebDataset/HDF5 — train: {len(train_tars)} tars, val: {len(val_tars)} tars')
        return {'format': 'tars', 'train': train_tars, 'val': val_tars}

    else:
        raise RuntimeError(
            f'Unknown dataset format in {repo_dir}.\nAll files: {all_files}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.dry_run:
        print('DRY RUN mode — processing 8 images per split only')

    # ── 1. Load tokenizers from HuggingFace Hub ────────────────────────────
    print('\n=== Step 1: Load tokenizers ===')
    rgb_tok   = load_tokenizer(RGB_TOKENIZER_REPO,   args.hf_cache, str(device))
    depth_tok = load_tokenizer(DEPTH_TOKENIZER_REPO, args.hf_cache, str(device))

    # ── 2. Locate NYUv2 ───────────────────────────────────────────────────
    print('\n=== Step 2: Locate NYUv2 dataset ===')
    source = get_nyu_data_source(args.hf_cache)

    # ── 3. Tokenize ────────────────────────────────────────────────────────
    print('\n=== Step 3: Tokenize ===')

    split_names = []
    if args.splits in ('train', 'both'):
        split_names.append('train')
    if args.splits in ('val', 'both'):
        split_names.append('val')

    for split_name in split_names:
        if source['format'] == 'tars':
            hf_key = 'val' if split_name == 'val' else 'train'
            tokenize_split_from_tars(
                split_name=split_name,
                tar_list=source[hf_key],
                rgb_tok=rgb_tok,
                depth_tok=depth_tok,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                device=device,
                dry_run=args.dry_run,
            )
        else:
            # HF DatasetDict path (parquet or datasets<4.0)
            hf_key = 'validation' if split_name == 'val' else 'train'
            tokenize_split(
                split_name=split_name,
                hf_split_key=hf_key,
                dataset=source['dataset'],
                rgb_tok=rgb_tok,
                depth_tok=depth_tok,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                device=device,
                dry_run=args.dry_run,
            )

    print('\n=== Done ===')
    print(f'Tokens saved to: {args.output_dir}')
    if not args.dry_run:
        print('\nNext step:  sbatch scripts/snellius/04_ft_depth_nyu.sh')


if __name__ == '__main__':
    main()
