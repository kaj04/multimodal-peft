#!/usr/bin/env python3
# prepare_nyu_normals.py
#
# Compute and tokenize surface normals for the NYUv2 LABELED split.
# Normals are estimated from depth maps via cross-product of 3-D gradient vectors.
#
# Input layout (same nyu_raw/ as prepare_nyu_labeled.py):
#   nyu_raw/train/depth/00000.npy  ...  (795 float32 depth maps, metres)
#   nyu_raw/test/depth/00000.npy   ...  (654 float32 depth maps, metres)
#
# Output layout (appended to nyu_tokens_labeled/):
#   nyu_tokens_labeled/train/tok_normal/00000.npy   int16 (196,)
#   nyu_tokens_labeled/val/tok_normal/00000.npy
#
# Usage:
#   python scripts/prepare_nyu_normals.py \
#       --nyu_raw    /scratch-shared/tesr125338/nyu_raw \
#       --output_dir /scratch-shared/tesr125338/nyu_tokens_labeled \
#       --hf_cache   /scratch-shared/tesr125338/hf_cache
#
# Or via SLURM:
#   sbatch scripts/snellius/32_tokenize_nyu_normals.sh

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

NORMAL_TOKENIZER_REPO = 'EPFL-VILAB/4M_tokenizers_normal_8k_224-448'

# NYUv2 camera intrinsics (640×480 Kinect)
NYU_FX = 525.0
NYU_FY = 525.0
NYU_CX = 319.5
NYU_CY = 239.5


# ── Normal estimation ─────────────────────────────────────────────────────────

def depth_to_normal(depth: np.ndarray) -> np.ndarray:
    """Estimate surface normals from a depth map (H, W) in metres.

    Uses NYUv2 camera intrinsics. Returns (H, W, 3) float32 in [-1, 1]
    with the convention that nz points toward the camera (negative Z).
    Invalid / zero-depth pixels are set to (0, 0, 1).
    """
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Back-project to camera space
    X = (uu - NYU_CX) * depth / NYU_FX
    Y = (vv - NYU_CY) * depth / NYU_FY
    Z = depth.astype(np.float32)

    # Compute tangent vectors via finite differences
    dX_du = np.gradient(X, axis=1)
    dY_du = np.gradient(Y, axis=1)
    dZ_du = np.gradient(Z, axis=1)

    dX_dv = np.gradient(X, axis=0)
    dY_dv = np.gradient(Y, axis=0)
    dZ_dv = np.gradient(Z, axis=0)

    tu = np.stack([dX_du, dY_du, dZ_du], axis=-1)  # (H, W, 3)
    tv = np.stack([dX_dv, dY_dv, dZ_dv], axis=-1)  # (H, W, 3)

    normal = np.cross(tu, tv)  # (H, W, 3)

    # Normalize to unit length
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = (norm[..., 0] > 1e-8) & (depth > 0)
    normal = np.where(valid[..., None], normal / (norm + 1e-8), 0.0)

    # Default direction for invalid pixels: point toward camera
    normal[~valid] = np.array([0.0, 0.0, 1.0])

    return normal.astype(np.float32)


def normal_to_rgb_uint8(normal: np.ndarray) -> np.ndarray:
    """Map normal (H, W, 3) in [-1, 1] → RGB uint8 in [0, 255].

    Convention: R=(nx+1)/2, G=(ny+1)/2, B=(nz+1)/2 — same as 4M pretraining.
    """
    rgb = ((normal + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_normal_batch(normal_arrays, size: int = 224) -> torch.Tensor:
    """Convert list of (H, W, 3) float32 normals → (B, 3, H, W) tensor in [-1, 1].

    Mirrors NormalTransform in ml-4m/fourm/data/modality_transforms.py:
      1. Convert to RGB uint8
      2. Resize to 224×224 (BILINEAR)
      3. ToTensor → [0, 1]
      4. Normalize mean=0.5, std=0.5 → [-1, 1]
    """
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensors = []
    for n in normal_arrays:
        rgb = normal_to_rgb_uint8(n)
        pil = Image.fromarray(rgb, mode='RGB')
        tensors.append(tf(pil))
    return torch.stack(tensors)   # (B, 3, H, W)


# ── Tokenize one split ────────────────────────────────────────────────────────

def tokenize_split(
        split_name: str,
        depth_dir: Path,
        out_dir: Path,
        normal_tok,
        batch_size: int,
        device: torch.device,
        dry_run: bool = False,
):
    tok_normal_dir = out_dir / 'tok_normal'
    tok_normal_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(depth_dir.glob('*.npy'))
    if not depth_files:
        raise RuntimeError(f'No .npy files found in {depth_dir}')
    if dry_run:
        depth_files = depth_files[:8]

    # Skip already-done files
    already_done = {p.stem for p in tok_normal_dir.glob('*.npy')}
    todo = [p for p in depth_files if p.stem not in already_done]
    print(f'\n[{split_name}] depth files: {len(depth_files)} | '
          f'to tokenize: {len(todo)}  (already done: {len(already_done)})')

    skipped = 0
    for i in tqdm(range(0, len(todo), batch_size), desc=f'  {split_name} normals'):
        batch_paths = todo[i: i + batch_size]

        normal_arrays = []
        valid_paths   = []
        for dp in batch_paths:
            depth = np.load(dp).astype(np.float32)
            if depth.max() == 0:
                skipped += 1
                continue
            normal_arrays.append(depth_to_normal(depth))
            valid_paths.append(dp)

        if not normal_arrays:
            continue

        normal_t = preprocess_normal_batch(normal_arrays).to(device)

        with torch.no_grad():
            tokens = normal_tok.tokenize(normal_t)
            tokens = tokens.view(tokens.shape[0], -1)   # (B, 196)

        tokens_np = tokens.cpu().numpy().astype(np.int16)

        for k, dp in enumerate(valid_paths):
            np.save(tok_normal_dir / f'{dp.stem}.npy', tokens_np[k])

        del normal_t, tokens
        torch.cuda.empty_cache()

    n_saved = len(todo) - skipped
    print(f'  Saved {n_saved} normal token files → {tok_normal_dir}')
    if skipped:
        print(f'  Skipped {skipped} (zero-depth / invalid)')


# ── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Compute and tokenize NYUv2 surface normals from depth maps')
    p.add_argument('--nyu_raw',    required=True,
                   help='Root of nyu_raw (contains train/ and test/)')
    p.add_argument('--output_dir', required=True,
                   help='nyu_tokens_labeled/ dir (tok_normal/ will be added)')
    p.add_argument('--hf_cache',   default=None)
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--splits',     default='both',
                   choices=['train', 'val', 'both'])
    p.add_argument('--dry_run',    action='store_true',
                   help='Process only the first 8 images per split')
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nyu_raw = Path(args.nyu_raw)
    out_dir = Path(args.output_dir)

    print(f'Device     : {device}')
    print(f'nyu_raw    : {nyu_raw}')
    print(f'Output dir : {out_dir}')
    if args.dry_run:
        print('DRY RUN — processing 8 images per split')

    from fourm.vq.vqvae import VQ
    print(f'\nLoading normal tokenizer ({NORMAL_TOKENIZER_REPO}) ...')
    normal_tok = VQ.from_pretrained(NORMAL_TOKENIZER_REPO, cache_dir=args.hf_cache)
    normal_tok.eval().to(device)

    # nyu_raw uses "test" subdir; our split calls it "val"
    split_map = {
        'train': ('train', 'train'),
        'val':   ('test',  'val'),
    }

    splits = []
    if args.splits in ('train', 'both'):
        splits.append('train')
    if args.splits in ('val', 'both'):
        splits.append('val')

    for split in splits:
        src_subdir, out_subdir = split_map[split]
        depth_dir = nyu_raw / src_subdir / 'depth'
        split_out = out_dir / out_subdir

        tokenize_split(
            split_name=split,
            depth_dir=depth_dir,
            out_dir=split_out,
            normal_tok=normal_tok,
            batch_size=args.batch_size,
            device=device,
            dry_run=args.dry_run,
        )

    print('\n=== Done ===')
    for split in splits:
        _, out_subdir = split_map[split]
        n = len(list((out_dir / out_subdir / 'tok_normal').glob('*.npy')))
        print(f'  [{split}] tok_normal: {n}')
    print('\nNext: sbatch scripts/snellius/33_tokenize_nyu_normals.sh')


if __name__ == '__main__':
    main()
