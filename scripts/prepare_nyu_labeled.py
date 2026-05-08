#!/usr/bin/env python3
# prepare_nyu_labeled.py
#
# Tokenize the NYUv2 LABELED split (795 train + 654 test) from nyu_raw/.
# This is the correct training data for matching MultiMAE / 4M-21 paper recipes:
#   - post-processed, in-painted depth (not raw Kinect video frames)
#   - same distribution as validation → no train/val depth quality mismatch
#   - matches the 795/654 split used in all 4M-21 depth benchmarks
#
# Input layout (nyu_raw/):
#   nyu_raw/train/rgb/00000.png   ...  (795 images)
#   nyu_raw/train/depth/00000.npy ...  (795 float32 depth maps, metres)
#   nyu_raw/test/rgb/00000.png    ...  (654 images)
#   nyu_raw/test/depth/00000.npy  ...  (654 float32 depth maps, metres)
#
# Output layout (nyu_tokens_labeled/):
#   nyu_tokens_labeled/train/tok_rgb/00000.npy    int16 (196,)
#   nyu_tokens_labeled/train/tok_depth/00000.npy  int16 (196,)
#   nyu_tokens_labeled/val/tok_rgb/00000.npy
#   nyu_tokens_labeled/val/tok_depth/00000.npy
#
# Usage:
#   python scripts/prepare_nyu_labeled.py \
#       --nyu_raw    /scratch-shared/tesr125338/nyu_raw \
#       --output_dir /scratch-shared/tesr125338/nyu_tokens_labeled \
#       --hf_cache   /scratch-shared/tesr125338/hf_cache
#
# Or via SLURM:
#   sbatch scripts/snellius/13_tokenize_nyu_labeled.sh

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

RGB_TOKENIZER_REPO   = 'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448'
DEPTH_TOKENIZER_REPO = 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448'


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Tokenize NYUv2 labeled split (795 train + 654 test) from nyu_raw/')
    p.add_argument('--nyu_raw',    required=True,
                   help='Root of nyu_raw (contains train/ and test/)')
    p.add_argument('--output_dir', required=True,
                   help='Where to write nyu_tokens_labeled/')
    p.add_argument('--hf_cache',   default=None)
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--splits',     default='both',
                   choices=['train', 'val', 'both'])
    p.add_argument('--dry_run',    action='store_true',
                   help='Process only the first 8 images per split')
    return p.parse_args()


# ── Preprocessing — reuse exact functions from prepare_nyu.py ────────────────

def preprocess_rgb_batch(pil_images, size: int = 224) -> torch.Tensor:
    """Resize + normalise RGB to [-1, 1] tensor (B, 3, H, W)."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return torch.stack([tf(img.convert('RGB')) for img in pil_images])


def _truncated_depth_standardization(depth: torch.Tensor,
                                     thresh: float = 0.1) -> torch.Tensor:
    """Mirror of DepthTransform.truncated_depth_standardization."""
    flat = torch.sort(depth.reshape(-1))[0]
    lo = int(thresh * len(flat))
    hi = int((1 - thresh) * len(flat))
    trunc = flat[lo:hi]
    return (depth - trunc.mean()) / torch.sqrt(trunc.var() + 1e-6)


def preprocess_depth_from_numpy(depth_arrays, size: int = 224) -> torch.Tensor:
    """Preprocess depth float32 numpy arrays (H, W) in metres → (B, 1, H, W).

    Replicates the exact pipeline from prepare_nyu.py:
      1. Normalise to [0, 65535]  (as _depth_pil_to_float_array does)
      2. Resize to size×size (NEAREST)
      3. Divide by 65535
      4. Truncated depth standardisation
    """
    tensors = []
    for arr in depth_arrays:
        # Step 1: normalise to [0, 65535]
        if arr.max() > 0:
            arr = arr / arr.max() * 65535.0
        # Step 2: resize via PIL (NEAREST, same as prepare_nyu.py)
        pil_resized = Image.fromarray(arr).resize((size, size), Image.NEAREST)
        arr_r = np.array(pil_resized, dtype=np.float32)
        # Step 3: divide by 65535
        depth = torch.tensor(arr_r / (2**16 - 1.0)).unsqueeze(0)  # (1, H, W)
        # Step 4: truncated standardisation
        depth = _truncated_depth_standardization(depth)
        tensors.append(depth)
    return torch.stack(tensors)  # (B, 1, H, W)


# ── Tokenize one split ────────────────────────────────────────────────────────

def tokenize_split(
        split_name: str,
        rgb_dir: Path,
        depth_dir: Path,
        out_dir: Path,
        rgb_tok,
        depth_tok,
        batch_size: int,
        device: torch.device,
        dry_run: bool = False,
):
    """Tokenize all PNG+npy pairs in rgb_dir/depth_dir → out_dir/{tok_rgb,tok_depth}/."""
    tok_rgb_dir   = out_dir / 'tok_rgb'
    tok_depth_dir = out_dir / 'tok_depth'
    tok_rgb_dir.mkdir(parents=True, exist_ok=True)
    tok_depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_dir.glob('*.png'))
    if not rgb_files:
        raise RuntimeError(f'No .png files found in {rgb_dir}')
    if dry_run:
        rgb_files = rgb_files[:8]

    print(f'\n[{split_name}] {len(rgb_files)} images → {out_dir}')

    skipped = 0
    for i in tqdm(range(0, len(rgb_files), batch_size), desc=f'  {split_name}'):
        batch_rgb_paths = rgb_files[i: i + batch_size]

        # Load RGB
        rgb_pil = [Image.open(p) for p in batch_rgb_paths]

        # Load depth (.npy, same stem as RGB)
        depth_arrays = []
        valid_indices = []
        for j, rp in enumerate(batch_rgb_paths):
            dp = depth_dir / (rp.stem + '.npy')
            if not dp.exists():
                print(f'  [skip] missing depth for {rp.name}')
                skipped += 1
                continue
            depth_arrays.append(np.load(dp).astype(np.float32))
            valid_indices.append(j)

        if not depth_arrays:
            continue

        rgb_pil_valid = [rgb_pil[j] for j in valid_indices]
        batch_paths_valid = [batch_rgb_paths[j] for j in valid_indices]

        # Preprocess
        rgb_t   = preprocess_rgb_batch(rgb_pil_valid).to(device)
        depth_t = preprocess_depth_from_numpy(depth_arrays).to(device)

        # Tokenize
        with torch.no_grad():
            rgb_tokens   = rgb_tok.tokenize(rgb_t)
            rgb_tokens   = rgb_tokens.view(rgb_tokens.shape[0], -1)       # (B, 196)
            depth_tokens = depth_tok.tokenize(depth_t)
            depth_tokens = depth_tokens.view(depth_tokens.shape[0], -1)   # (B, 196)

        rgb_np   = rgb_tokens.cpu().numpy().astype(np.int16)
        depth_np = depth_tokens.cpu().numpy().astype(np.int16)

        for k, path in enumerate(batch_paths_valid):
            np.save(tok_rgb_dir   / f'{path.stem}.npy', rgb_np[k])
            np.save(tok_depth_dir / f'{path.stem}.npy', depth_np[k])

        del rgb_t, depth_t, rgb_tokens, depth_tokens
        torch.cuda.empty_cache()

    n_saved = len(rgb_files) - skipped
    print(f'  Saved {n_saved} pairs → {out_dir}')
    if skipped:
        print(f'  Skipped {skipped} (missing depth)')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nyu_raw = Path(args.nyu_raw)
    out_dir = Path(args.output_dir)

    print(f'Device     : {device}')
    print(f'nyu_raw    : {nyu_raw}')
    print(f'Output dir : {out_dir}')
    if args.dry_run:
        print('DRY RUN — processing 8 images per split')

    # Load tokenizers
    from fourm.vq.vqvae import VQ
    print(f'\nLoading RGB tokenizer ...')
    rgb_tok = VQ.from_pretrained(RGB_TOKENIZER_REPO, cache_dir=args.hf_cache)
    rgb_tok.eval().to(device)

    print(f'Loading depth tokenizer ...')
    depth_tok = VQ.from_pretrained(DEPTH_TOKENIZER_REPO, cache_dir=args.hf_cache)
    depth_tok.eval().to(device)

    # Split mapping: our "val" = nyu_raw "test"
    split_map = {
        'train': ('train', 'train'),  # (nyu_raw subdir, output subdir)
        'val':   ('test',  'val'),
    }

    splits = []
    if args.splits in ('train', 'both'):
        splits.append('train')
    if args.splits in ('val', 'both'):
        splits.append('val')

    for split in splits:
        src_subdir, out_subdir = split_map[split]
        rgb_dir   = nyu_raw / src_subdir / 'rgb'
        depth_dir = nyu_raw / src_subdir / 'depth'
        split_out = out_dir / out_subdir

        tokenize_split(
            split_name=split,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            out_dir=split_out,
            rgb_tok=rgb_tok,
            depth_tok=depth_tok,
            batch_size=args.batch_size,
            device=device,
            dry_run=args.dry_run,
        )

    # Summary
    print('\n=== Done ===')
    for split in splits:
        _, out_subdir = split_map[split]
        rgb_n   = len(list((out_dir / out_subdir / 'tok_rgb').glob('*.npy')))
        depth_n = len(list((out_dir / out_subdir / 'tok_depth').glob('*.npy')))
        print(f'  [{split}]  tok_rgb: {rgb_n}  tok_depth: {depth_n}')
    print(f'\nNext: sbatch scripts/snellius/14_ft_depth_nyu_labeled_r4_selfattn_drop01.sh')


if __name__ == '__main__':
    main()
