#!/usr/bin/env python3
"""
eval_diode.py
=============
Zero-shot depth evaluation on the DIODE val set using the **pretrained**
4M-21 model (original weights, no LoRA, no fine-tuning).

Reproduces Table 1 of the 4M-21 paper:
  4M-21 XL  depth L1 (normalized) = 0.68

Metric
------
Mean L1 in the per-image [0,1] min-max-normalized depth space, computed
only over valid pixels (depth_mask > 0 & depth > 0).

Both prediction and ground truth are independently min-max-normalized to
[0,1] using their respective valid-pixel distributions before the L1 is
computed.  This is the standard relative-depth evaluation protocol used
when a model cannot recover absolute metric scale.

DIODE spatial handling
----------------------
DIODE images are 1024×768 (landscape).  The 4M model expects 224×224 RGB.
We center-crop both the RGB and the GT depth map to the square of side
min(H,W) = 768, then resize to 224×224.  This keeps RGB and GT spatially
aligned.

Usage
-----
  python eval_diode.py \\
      --diode_root /scratch-shared/tesr125338/diode/val \\
      [--fm         EPFL-VILAB/4M-21_XL] \\
      [--tok_depth  EPFL-VILAB/4M_tokenizers_depth_8k_224-448] \\
      [--output_dir output/eval_diode] \\
      [--max_images 50]          # quick smoke-test: ~50 images
      [--divae_steps 50]         # DDIM steps for DiVAE decoding (paper: 50)

Output
------
  output/eval_diode/results.json   — per-image L1 + aggregate stats
  output/eval_diode/summary.txt    — human-readable comparison vs paper
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms.functional import center_crop

# ── Add ml-4m to PYTHONPATH ───────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'ml-4m'))

from fourm.models.fm import FM
from fourm.vq.vqvae import DiVAE
from fourm.models.generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_empty_target_modality,
    init_full_input_modality,
)
from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import RGBTransform
from fourm.utils.plotting_utils import decode_tok_depth


# ── Generation hyper-parameters (mirrors DEFAULTS_RGB2X in demo_4M_sampler.py)
DEPTH_GEN_PARAMS = {
    'tokens_per_target':      196,
    'autoregression_scheme':  'roar',
    'decoding_steps':         1,
    'token_decoding_schedule':'linear',
    'temp':                   0.01,
    'temp_schedule':          'constant',
    'cfg_scale':              2.0,
    'cfg_schedule':           'constant',
}

# Paper Table 1 reference values (4M-21 XL, zero-shot)
PAPER_DEPTH_L1  = 0.68
PAPER_NORMAL_L1 = 20.8


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Zero-shot DIODE depth evaluation with 4M-21 pretrained weights'
    )
    p.add_argument('--diode_root', required=True,
                   help='Path to DIODE val root (contains indoors/ and outdoor/)')
    p.add_argument('--fm', default='EPFL-VILAB/4M-21_XL',
                   help='4M model HuggingFace hub ID or local .safetensors path')
    p.add_argument('--tok_depth',
                   default='EPFL-VILAB/4M_tokenizers_depth_8k_224-448',
                   help='Depth DiVAE hub ID or local .safetensors path')
    p.add_argument('--tok_text',
                   default='ml-4m/fourm/utils/tokenizer/trained/'
                           'text_tokenizer_4m_wordpiece_30k.json',
                   help='Path to text tokenizer JSON (relative to repo root or absolute)')
    p.add_argument('--output_dir', default='output/eval_diode',
                   help='Directory for results.json and summary.txt')
    p.add_argument('--max_images', type=int, default=None,
                   help='Process only the first N images (smoke-test)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='bfloat16',
                   choices=['bfloat16', 'float32'])
    p.add_argument('--divae_steps', type=int, default=50,
                   help='DiVAE DDIM decoding steps (paper uses 50)')
    p.add_argument('--seed', type=int, default=0,
                   help='Base random seed (incremented per image)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def collect_diode_samples(diode_root):
    """Walk the DIODE val tree and collect (rgb, depth, mask) triples.

    Expected structure:
      <diode_root>/
        indoors/  scene_XXXXX/  scan_XXXXX/  <stem>.png
                                             <stem>_depth.npy
                                             <stem>_depth_mask.npy
        outdoor/  ...
    """
    diode_root = Path(diode_root)
    samples = []
    for split in ['indoors', 'outdoor']:
        split_dir = diode_root / split
        if not split_dir.exists():
            print(f'  Warning: {split_dir} not found — skipping.')
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for scan_dir in sorted(scene_dir.iterdir()):
                if not scan_dir.is_dir():
                    continue
                for rgb_p in sorted(scan_dir.glob('*.png')):
                    stem    = rgb_p.stem
                    depth_p = scan_dir / f'{stem}_depth.npy'
                    mask_p  = scan_dir / f'{stem}_depth_mask.npy'
                    if depth_p.exists() and mask_p.exists():
                        samples.append(dict(
                            rgb   = str(rgb_p),
                            depth = str(depth_p),
                            mask  = str(mask_p),
                            split = split,
                            name  = f'{split}/{scene_dir.name}/{scan_dir.name}/{stem}',
                        ))
    return samples


def _center_crop_np(arr, size):
    """Center-crop a numpy HxW (or HxWxC) array to size×size."""
    h, w  = arr.shape[:2]
    top   = (h - size) // 2
    left  = (w - size) // 2
    return arr[top:top + size, left:left + size]


def load_rgb_tensor(rgb_path, device, dtype_torch):
    """Load RGB from disk → normalized (1,3,224,224) tensor.

    Matches the demo_4M_sampler.py preprocessing exactly:
      load → preprocess (convert to RGB) → center-crop to square → resize 224
      → postprocess (ImageNet normalise) → unsqueeze(0)
    """
    rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
    img_pil = rgb_transform.load(rgb_path)
    img_pil = rgb_transform.preprocess(img_pil)           # ensures RGB mode
    s       = min(img_pil.size)                            # min(W, H)
    img_pil = center_crop(img_pil, (s, s)).resize((224, 224), Image.BILINEAR)
    tensor  = rgb_transform.postprocess(img_pil).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype_torch)


def load_gt_depth_224(depth_path, mask_path):
    """Load DIODE GT depth (metres) + valid mask, center-cropped & resized to 224×224.

    The same center-crop geometry used for RGB ensures spatial alignment.
    """
    depth = np.load(depth_path).squeeze()    # (H, W) float32 metres
    mask  = np.load(mask_path).squeeze()     # (H, W) uint8

    h, w  = depth.shape
    s     = min(h, w)
    depth = _center_crop_np(depth, s)
    mask  = _center_crop_np(mask,  s)

    depth_224 = cv2.resize(depth.astype(np.float32), (224, 224),
                           interpolation=cv2.INTER_LINEAR)
    mask_224  = cv2.resize(mask.astype(np.float32),  (224, 224),
                           interpolation=cv2.INTER_NEAREST)

    valid = (mask_224 > 0) & (depth_224 > 0)
    return depth_224, valid


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────
def generate_depth_tokens(sampler_fm, tok_text, rgb_tensor, device, seed):
    """Run FM generation: rgb@224 → tok_depth@224 tokens.

    Returns the raw out_dict from GenerationSampler.generate(); depth tokens
    live at out_dict['tok_depth@224']['tensor'] with shape (1, 196).
    """
    ntoks = DEPTH_GEN_PARAMS['tokens_per_target']

    schedule = build_chained_generation_schedules(
        cond_domains          = ['rgb@224'],
        target_domains        = ['tok_depth@224'],
        tokens_per_target     = [ntoks],
        autoregression_schemes= [DEPTH_GEN_PARAMS['autoregression_scheme']],
        decoding_steps        = [DEPTH_GEN_PARAMS['decoding_steps']],
        token_decoding_schedules=[DEPTH_GEN_PARAMS['token_decoding_schedule']],
        temps                 = [DEPTH_GEN_PARAMS['temp']],
        temp_schedules        = [DEPTH_GEN_PARAMS['temp_schedule']],
        cfg_scales            = [DEPTH_GEN_PARAMS['cfg_scale']],
        cfg_schedules         = [DEPTH_GEN_PARAMS['cfg_schedule']],
        cfg_grow_conditioning = True,
    )

    eos_id      = tok_text.token_to_id('[EOS]')
    sample_dict = {'rgb@224': {'tensor': rgb_tensor}}
    sample_dict = init_full_input_modality(
        sample_dict, MODALITY_INFO, 'rgb@224', device, eos_id=eos_id)
    sample_dict = init_empty_target_modality(
        sample_dict, MODALITY_INFO, 'tok_depth@224', 1, ntoks, device)

    out_dict = sampler_fm.generate(
        sample_dict, schedule,
        text_tokenizer=tok_text,
        verbose=False,
        seed=seed,
    )
    return out_dict


def decode_depth_raw(out_dict, tok_depth_model, divae_steps):
    """Decode tok_depth@224 tokens → raw float depth map (224, 224).

    Uses decode_tok_depth with cmap=None to return raw DiVAE output values
    instead of a colourised image.
    """
    # decode_tok_depth looks up tokenizers['tok_depth'] (key without @res suffix)
    toks_proxy = {'tok_depth': tok_depth_model}
    raw = decode_tok_depth(
        out_dict, toks_proxy,
        key       = 'tok_depth@224',
        image_size= 224,
        patch_size= 16,
        t         = divae_steps,
        verbose   = False,
        cmap      = None,   # ← return raw floats, not colourised RGB
    )
    # Shape: (B, H, W) — squeeze batch dim
    if raw.ndim == 3:
        raw = raw[0]
    return raw.astype(np.float32)   # (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────
def _minmax_norm(arr, valid):
    """Normalize arr to [0,1] using the min/max of valid pixels."""
    vv = arr[valid]
    if len(vv) < 2:
        return arr
    lo, hi = float(vv.min()), float(vv.max())
    denom  = max(hi - lo, 1e-8)
    return (arr - lo) / denom


def depth_l1_normalized(pred_raw, gt_depth, valid):
    """Mean L1 in [0,1] normalized depth space (both sides normalized independently)."""
    pred_norm = _minmax_norm(pred_raw, valid)
    gt_norm   = _minmax_norm(gt_depth, valid)
    return float(np.abs(pred_norm[valid] - gt_norm[valid]).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device      = torch.device(args.device)
    dtype_torch = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    # ── Header ──────────────────────────────────────────────────────────────
    print('=' * 64)
    print('  DIODE Zero-Shot Depth Eval — 4M-21 Pretrained (no LoRA)')
    print('=' * 64)
    print(f'  FM model   : {args.fm}')
    print(f'  Depth tok  : {args.tok_depth}')
    print(f'  DIODE root : {args.diode_root}')
    print(f'  DiVAE steps: {args.divae_steps}')
    print(f'  dtype      : {args.dtype}   device: {args.device}')
    print()

    # ── Collect samples ─────────────────────────────────────────────────────
    print('Collecting DIODE val samples ...', end='', flush=True)
    samples = collect_diode_samples(args.diode_root)
    if not samples:
        print('\nERROR: No samples found. Check --diode_root.')
        sys.exit(1)
    n_indoor  = sum(s['split'] == 'indoors' for s in samples)
    n_outdoor = sum(s['split'] == 'outdoor' for s in samples)
    print(f' {len(samples)} found  (indoor={n_indoor}, outdoor={n_outdoor})')

    if args.max_images is not None:
        samples = samples[:args.max_images]
        print(f'  → Capped at {args.max_images} images  (smoke-test mode)')
    n_total = len(samples)
    print()

    # ── Load FM ─────────────────────────────────────────────────────────────
    print('Loading FM model (this downloads ~7 GB on first run) ...')
    fm = FM.from_pretrained(args.fm)
    fm = fm.to(device=device, dtype=dtype_torch).eval()
    n_params = sum(p.numel() for p in fm.parameters()) / 1e6
    print(f'  FM loaded  ({n_params:.0f}M params, dtype={args.dtype})')
    sampler_fm = GenerationSampler(fm)

    # ── Load depth DiVAE ────────────────────────────────────────────────────
    print('Loading depth DiVAE tokenizer ...')
    tok_depth = DiVAE.from_pretrained(args.tok_depth)
    tok_depth = tok_depth.to(device=device).eval()
    print('  DiVAE loaded')

    # ── Load text tokenizer ─────────────────────────────────────────────────
    tok_text_path = args.tok_text
    if not os.path.isabs(tok_text_path):
        tok_text_path = os.path.join(_REPO_ROOT, tok_text_path)
    print(f'Loading text tokenizer from {tok_text_path} ...')
    tok_text = Tokenizer.from_file(tok_text_path)
    print()

    # ── Evaluation loop ─────────────────────────────────────────────────────
    results    = []
    l1_all     = []
    l1_indoor  = []
    l1_outdoor = []
    t0         = time.time()

    for idx, s in enumerate(samples):

        # Progress every 50 images (or at start / last image)
        if idx == 0 or (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            rate    = elapsed / max(idx, 1)
            eta     = rate * (n_total - idx - 1)
            print(f'  [{idx + 1:4d}/{n_total}]  '
                  f'elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s', flush=True)

        try:
            # 1 — Load & preprocess RGB
            rgb_tensor = load_rgb_tensor(s['rgb'], device, dtype_torch)

            # 2 — FM inference: rgb@224 → tok_depth@224
            with torch.no_grad():
                out_dict = generate_depth_tokens(
                    sampler_fm, tok_text, rgb_tensor, device,
                    seed=args.seed + idx,
                )

            # 3 — DiVAE decode: tokens → raw depth map (224, 224)
            with torch.no_grad():
                pred_raw = decode_depth_raw(out_dict, tok_depth, args.divae_steps)

            # 4 — Load GT depth, center-cropped & resized to 224×224
            gt_depth, valid = load_gt_depth_224(s['depth'], s['mask'])

            if valid.sum() < 100:
                print(f'    Skip {s["name"]}: only {valid.sum()} valid pixels')
                results.append({'name': s['name'], 'split': s['split'],
                                 'l1': None, 'error': 'too_few_valid'})
                continue

            # 5 — Compute L1 in normalized [0,1] space
            l1 = depth_l1_normalized(pred_raw, gt_depth, valid)

            results.append({'name': s['name'], 'split': s['split'], 'l1': l1})
            l1_all.append(l1)
            if s['split'] == 'indoors':
                l1_indoor.append(l1)
            else:
                l1_outdoor.append(l1)

        except Exception as e:
            import traceback
            print(f'  ERROR on {s["name"]}: {e}')
            traceback.print_exc()
            results.append({'name': s['name'], 'split': s['split'],
                             'l1': None, 'error': str(e)})

    elapsed_total = time.time() - t0
    n_valid = len(l1_all)

    # ── Aggregate ────────────────────────────────────────────────────────────
    mean_l1         = float(np.mean(l1_all))     if l1_all     else float('nan')
    mean_l1_indoor  = float(np.mean(l1_indoor))  if l1_indoor  else float('nan')
    mean_l1_outdoor = float(np.mean(l1_outdoor)) if l1_outdoor else float('nan')

    summary = {
        'model':                    args.fm,
        'tok_depth':                args.tok_depth,
        'n_images_total':           n_total,
        'n_images_valid':           n_valid,
        'n_errors':                 n_total - n_valid,
        'divae_steps':              args.divae_steps,
        'depth_L1_all':             mean_l1,
        'depth_L1_indoor':          mean_l1_indoor,
        'depth_L1_outdoor':         mean_l1_outdoor,
        'paper_4M21_XL_depth_L1':   PAPER_DEPTH_L1,
        'paper_4M21_XL_normal_L1':  PAPER_NORMAL_L1,
        'elapsed_seconds':          elapsed_total,
        'seconds_per_image':        elapsed_total / max(n_valid, 1),
    }

    # ── Save JSON ────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'per_image': results}, f, indent=2)

    # ── Human-readable summary ───────────────────────────────────────────────
    diff   = mean_l1 - PAPER_DEPTH_L1
    if   abs(diff) < 0.03: verdict = '✓  matches paper  (|diff| < 0.03)'
    elif diff > 0:          verdict = '⚠  HIGHER than paper (worse by {:.4f})'.format(diff)
    else:                   verdict = '✓  LOWER than paper  (better by {:.4f})'.format(-diff)

    lines = [
        '',
        '=' * 64,
        '  DIODE Depth Evaluation Summary',
        '=' * 64,
        f'  Model         : {args.fm}',
        f'  Depth tok     : {args.tok_depth}',
        f'  DiVAE steps   : {args.divae_steps}',
        f'  Images valid  : {n_valid} / {n_total}  '
            f'(errors: {n_total - n_valid})',
        f'  Time          : {elapsed_total:.1f}s  '
            f'({elapsed_total / max(n_valid, 1):.1f}s / image)',
        '',
        '  DEPTH L1  ↓  (lower = better)',
        '  Metric: mean |pred_norm - gt_norm| in [0,1] min-max space',
        '  ' + '-' * 44,
        f'  All     : {mean_l1:.4f}  (n={n_valid})',
        f'  Indoor  : {mean_l1_indoor:.4f}  (n={len(l1_indoor)})',
        f'  Outdoor : {mean_l1_outdoor:.4f}  (n={len(l1_outdoor)})',
        '',
        '  COMPARISON — paper Table 1  (4M-21 XL, zero-shot)',
        '  ' + '-' * 44,
        f'  Paper  depth L1  : {PAPER_DEPTH_L1:.2f}',
        f'  Ours   depth L1  : {mean_l1:.4f}',
        f'  Difference       : {diff:+.4f}   {verdict}',
        '',
        f'  JSON : {json_path}',
        '=' * 64,
        '',
    ]

    txt_path = os.path.join(args.output_dir, 'summary.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
