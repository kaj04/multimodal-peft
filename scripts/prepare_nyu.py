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
# Tokenize one split
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
# NYUv2 dataset loading — robust to datasets library version
# ---------------------------------------------------------------------------

def load_nyu_dataset(hf_cache: str):
    """Load NYUv2 from HuggingFace, bypassing the datasets loading-script issue.

    datasets>=4.0 removed support for custom loading scripts (.py files).
    sayakpaul/nyu_depth_v2 uses one, so we bypass the datasets library:
      1. Download the raw repo files via huggingface_hub.snapshot_download
         (this ignores .py scripts — only gets data files)
      2. Find the parquet files in the downloaded snapshot
      3. Load them directly with datasets.load_dataset('parquet', ...)

    Falls back to direct load_dataset() call (works on datasets<4.0).
    """
    from datasets import load_dataset

    # ── Fast path: datasets<4.0 with trust_remote_code (works on 2.x) ──
    try:
        ds = load_dataset('sayakpaul/nyu_depth_v2',
                          cache_dir=hf_cache,
                          trust_remote_code=True)
        print(f'  {ds}')
        return ds
    except Exception:
        pass  # datasets>=4.0 — fall through to snapshot approach

    # ── Robust path: snapshot_download → load parquet directly ──────────
    print('  datasets>=4.0 detected — using snapshot_download fallback ...')
    from huggingface_hub import snapshot_download
    import glob

    repo_dir = snapshot_download(
        repo_id='sayakpaul/nyu_depth_v2',
        repo_type='dataset',
        cache_dir=hf_cache,
        ignore_patterns=['*.py'],   # skip the loading script
    )
    print(f'  Snapshot at: {repo_dir}')

    # List everything so we can debug if parquet not found
    all_files = []
    for root, _, files in os.walk(repo_dir):
        for f in files:
            all_files.append(os.path.relpath(os.path.join(root, f), repo_dir))
    print(f'  Files in repo ({len(all_files)}): {all_files[:10]}')

    # ── Detect format from file list ────────────────────────────────────
    parquet_files = glob.glob(os.path.join(repo_dir, '**', '*.parquet'),
                              recursive=True)
    tar_files = glob.glob(os.path.join(repo_dir, '**', '*.tar'),
                          recursive=True)

    if parquet_files:
        # Parquet format (newer HF datasets)
        train_files = sorted(f for f in parquet_files
                             if 'train' in os.path.basename(f).lower())
        val_files   = sorted(f for f in parquet_files
                             if any(s in os.path.basename(f).lower()
                                    for s in ('val', 'test')))
        print(f'  Parquet — train: {len(train_files)}, val: {len(val_files)}')
        ds = load_dataset('parquet', data_files={
            'train': train_files, 'validation': val_files})
        print(f'  {ds}')
        return ds

    elif tar_files:
        # WebDataset format — peek inside first tar to find key names
        return _load_nyu_webdataset(repo_dir, tar_files)

    else:
        raise RuntimeError(
            f'Unknown dataset format in {repo_dir}.\nAll files: {all_files}')


def _peek_tar_keys(tar_path: str) -> list:
    """Return the file extensions (keys) of the first sample in a tar archive."""
    import tarfile
    keys = []
    with tarfile.open(tar_path, 'r') as tf:
        members = tf.getmembers()
        if not members:
            return keys
        # Get the stem of the first file (e.g. '000000' from '000000.jpg')
        first_stem = os.path.splitext(members[0].name)[0]
        for m in members:
            stem = os.path.splitext(m.name)[0]
            if stem == first_stem:
                keys.append(os.path.splitext(m.name)[1].lstrip('.'))
            else:
                break   # moved to next sample
    return keys


def _load_nyu_webdataset(repo_dir: str, tar_files: list) -> dict:
    """Load NYUv2 WebDataset tars into a plain dict of lists.

    Returns {'train': [{'image': PIL, 'depth_map': PIL}, ...],
             'validation': [...]}
    compatible with the rest of the pipeline.
    """
    import tarfile
    import io

    train_tars = sorted(f for f in tar_files
                        if 'train' in os.path.basename(f).lower())
    val_tars   = sorted(f for f in tar_files
                        if any(s in os.path.basename(f).lower()
                               for s in ('val', 'test')))

    print(f'  WebDataset — train tars: {len(train_tars)}, val tars: {len(val_tars)}')

    # Peek to discover key names
    keys = _peek_tar_keys(train_tars[0])
    print(f'  Keys per sample in first tar: {keys}')

    # Heuristic: find the rgb key and depth key
    rgb_key   = next((k for k in keys if k in ('jpg', 'jpeg', 'png')
                      and 'depth' not in k), None)
    depth_key = next((k for k in keys if 'depth' in k or k in ('npy',)), None)

    # Fallback if naming is ambiguous — print and let user check
    if rgb_key is None or depth_key is None:
        print(f'  WARNING: could not auto-detect rgb/depth keys from {keys}')
        print(f'  Assuming first image key = rgb, second = depth')
        img_keys = [k for k in keys if k in
                    ('jpg', 'jpeg', 'png', 'npy', 'npz')]
        rgb_key   = img_keys[0] if len(img_keys) > 0 else keys[0]
        depth_key = img_keys[1] if len(img_keys) > 1 else keys[1]

    print(f'  rgb_key={rgb_key!r}, depth_key={depth_key!r}')

    def read_split(tar_list):
        samples = []
        for tar_path in tqdm(tar_list, desc='  reading tars'):
            with tarfile.open(tar_path, 'r') as tf:
                members = {m.name: m for m in tf.getmembers()}
                # Group members by stem
                stems = {}
                for name in members:
                    stem, ext = os.path.splitext(name)
                    stems.setdefault(stem, {})[ext.lstrip('.')] = name

                for stem, exts in stems.items():
                    if rgb_key not in exts or depth_key not in exts:
                        continue
                    # Load RGB
                    rgb_bytes = tf.extractfile(members[exts[rgb_key]]).read()
                    rgb_img   = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
                    # Load depth
                    dep_bytes = tf.extractfile(members[exts[depth_key]]).read()
                    if depth_key == 'npy':
                        dep_arr = np.load(io.BytesIO(dep_bytes))
                        dep_img = Image.fromarray(dep_arr.astype(np.float32))
                    else:
                        dep_img = Image.open(io.BytesIO(dep_bytes))
                    samples.append({'image': rgb_img, 'depth_map': dep_img})
        return samples

    print('  Loading train split ...')
    train_samples = read_split(train_tars)
    print(f'  train: {len(train_samples)} samples')

    print('  Loading val split ...')
    val_samples = read_split(val_tars)
    print(f'  val:   {len(val_samples)} samples')

    # Wrap as a simple indexable object matching the expected interface
    class SimpleDataset:
        def __init__(self, data): self._data = data
        def __getitem__(self, key):
            if isinstance(key, str):
                return _SimpleSliceable(self._data, key)
            return self._data[key]
        def __len__(self): return len(self._data)

    class _SimpleSliceable:
        """Enables dataset['train'][i:i+bs]['image'] access pattern."""
        def __init__(self, data, split): self._data = data
        def __getitem__(self, idx):
            chunk = self._data[idx] if isinstance(idx, slice) else [self._data[idx]]
            return {'image':     [s['image']     for s in chunk],
                    'depth_map': [s['depth_map'] for s in chunk]}
        def __len__(self): return len(self._data)

    class NYUDataset:
        def __init__(self, train, val):
            self._train = _SimpleSliceable(train, 'train')
            self._val   = _SimpleSliceable(val,   'val')
        def __getitem__(self, key):
            return self._train if key == 'train' else self._val

    return NYUDataset(train_samples, val_samples)


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

    # ── 2. Download NYUv2 ──────────────────────────────────────────────────
    print('\n=== Step 2: Load NYUv2 dataset ===')
    dataset = load_nyu_dataset(args.hf_cache)

    # ── 3. Tokenize ────────────────────────────────────────────────────────
    print('\n=== Step 3: Tokenize ===')
    split_map = {'train': 'train', 'val': 'validation'}
    if args.splits == 'train':
        split_map = {'train': 'train'}
    elif args.splits == 'val':
        split_map = {'val': 'validation'}

    for out_name, hf_key in split_map.items():
        tokenize_split(
            split_name=out_name,
            hf_split_key=hf_key,
            dataset=dataset,
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
