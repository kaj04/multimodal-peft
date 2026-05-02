#!/usr/bin/env python3
"""
eval_depth.py
=============
Modular depth evaluation for 4M-21 models (baseline or LoRA fine-tuned).

Supports:
  --dataset nyu    NYU Depth v2 test set (654 images, indoor, GT in metres)
  --dataset diode  DIODE val set (771 images, indoor+outdoor, GT in metres)

Supports:
  No --lora_weights  →  baseline (pretrained 4M-21 XL, no LoRA)
  --lora_weights     →  LoRA fine-tuned model (loads adapter on top of base)

Metrics (affine-invariant evaluation — standard for relative depth models):
  AbsRel   mean( |pred - gt| / gt )                        lower = better
  RMSE     sqrt( mean( (pred - gt)^2 ) )  [metres]         lower = better
  delta1   % pixels where max(p/g, g/p) < 1.25             higher = better
  delta2   % pixels where max(p/g, g/p) < 1.25^2           higher = better
  delta3   % pixels where max(p/g, g/p) < 1.25^3           higher = better
  L1_norm  mean |pred_norm - gt_norm| in [0,1] min-max      lower = better
           (kept for internal backward-compatibility with eval_diode.py runs)

Alignment: per-image least-squares scale-and-shift (affine-invariant protocol,
as used in MiDaS, ZoeDepth, Depth Anything). Converts z-score pred to metres.

NYU data format expected:
  <dataset_root>/
    rgb/        *.png or *.jpg
    depth/      *.npy   (float32, metres, same stems as rgb/)
  Valid mask: depth > 0 & depth <= max_depth_clip (default 10m for NYU)

DIODE data format expected (unchanged from eval_diode.py):
  <dataset_root>/
    indoors/  scene_*/  scan_*/  <stem>.png
                                  <stem>_depth.npy
                                  <stem>_depth_mask.npy
    outdoor/  ...

Usage
-----
  # Baseline on NYU:
  python eval_depth.py --dataset nyu --dataset_root /path/to/nyu_raw/test

  # LoRA on NYU:
  python eval_depth.py --dataset nyu --dataset_root /path/to/nyu_raw/test \\
      --lora_weights output/depth/nyu_rgb_r8/lora_weights-best.pth

  # Baseline on DIODE:
  python eval_depth.py --dataset diode \\
      --dataset_root /scratch-shared/tesr125338/diode/val

  # Smoke-test (first 20 images):
  python eval_depth.py --dataset nyu --dataset_root /path/... --max_images 20

Output
------
  output/eval_depth/<run_name>/
    results.json   per-image metrics + aggregate summary
    summary.txt    human-readable table (copy-paste to report)
    config.json    exact config used (for reproducibility)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms.functional import center_crop

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

# Generation hyper-parameters — mirrors DEFAULTS_RGB2X in demo_4M_sampler.py
DEPTH_GEN_PARAMS = {
    'tokens_per_target':       196,
    'autoregression_scheme':   'roar',
    'decoding_steps':          1,
    'token_decoding_schedule': 'linear',
    'temp':                    0.01,
    'temp_schedule':           'constant',
    'cfg_scale':               2.0,
    'cfg_schedule':            'constant',
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Modular depth evaluation: baseline vs LoRA on NYU / DIODE'
    )

    # Dataset
    p.add_argument('--dataset', required=True, choices=['nyu', 'diode'],
                   help='Which dataset to evaluate on')
    p.add_argument('--dataset_root', required=True,
                   help='Path to dataset root (see module docstring for structure)')

    # Model
    p.add_argument('--fm', default='EPFL-VILAB/4M-21_XL',
                   help='4M model HuggingFace hub ID or local path')
    p.add_argument('--tok_depth',
                   default='EPFL-VILAB/4M_tokenizers_depth_8k_224-448',
                   help='Depth DiVAE hub ID or local path')
    p.add_argument('--tok_text',
                   default='ml-4m/fourm/utils/tokenizer/trained/'
                           'text_tokenizer_4m_wordpiece_30k.json',
                   help='Path to text tokenizer JSON')
    p.add_argument('--lora_weights', default=None,
                   help='Path to lora_weights-*.pth — if omitted, runs baseline')

    # Evaluation
    p.add_argument('--max_depth_clip', type=float, default=10.0,
                   help='GT depth cap in metres (10m for NYU, 300m for DIODE)')
    p.add_argument('--max_images', type=int, default=None,
                   help='Cap number of images (smoke-test)')
    p.add_argument('--divae_steps', type=int, default=50,
                   help='DiVAE DDIM decoding steps (paper uses 50)')
    p.add_argument('--seed', type=int, default=0)

    # Output
    p.add_argument('--output_dir', default='output/eval_depth',
                   help='Parent output directory')
    p.add_argument('--run_name', default=None,
                   help='Sub-directory name (auto-generated if omitted)')

    # Hardware
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'])

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def collect_nyu_samples(root):
    """Collect (rgb_path, depth_path) pairs from a simple NYU hierarchy.

    Expected layout:
      root/rgb/   *.png or *.jpg
      root/depth/ *.npy   (float32, metres, matching stems)
    """
    root = Path(root)
    rgb_dir   = root / 'rgb'
    depth_dir = root / 'depth'
    if not rgb_dir.exists():
        raise FileNotFoundError(f'NYU rgb dir not found: {rgb_dir}')
    if not depth_dir.exists():
        raise FileNotFoundError(f'NYU depth dir not found: {depth_dir}')

    samples = []
    for rgb_p in sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg')):
        depth_p = depth_dir / f'{rgb_p.stem}.npy'
        if depth_p.exists():
            samples.append(dict(rgb=str(rgb_p), depth=str(depth_p),
                                mask=None, split='nyu', name=rgb_p.stem))
    return samples


def collect_diode_samples(root):
    """Walk DIODE val tree and collect (rgb, depth, mask) triples."""
    root = Path(root)
    samples = []
    for split in ['indoors', 'outdoor']:
        split_dir = root / split
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
                            rgb=str(rgb_p), depth=str(depth_p),
                            mask=str(mask_p), split=split,
                            name=f'{split}/{scene_dir.name}/{scan_dir.name}/{stem}',
                        ))
    return samples


def _center_crop_np(arr, size):
    h, w = arr.shape[:2]
    top  = (h - size) // 2
    left = (w - size) // 2
    return arr[top:top + size, left:left + size]


def load_rgb_tensor(rgb_path, device, dtype_torch):
    """RGB → (1,3,224,224) ImageNet-normalised tensor."""
    rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
    img_pil = rgb_transform.load(rgb_path)
    img_pil = rgb_transform.preprocess(img_pil)
    s       = min(img_pil.size)
    img_pil = center_crop(img_pil, (s, s)).resize((224, 224), Image.BILINEAR)
    tensor  = rgb_transform.postprocess(img_pil).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype_torch)


def load_gt_nyu(depth_path, max_depth_clip):
    """Load NYU GT depth .npy → (224,224) float32 metres + valid mask."""
    depth = np.load(depth_path).squeeze().astype(np.float32)
    h, w  = depth.shape
    s     = min(h, w)
    depth = _center_crop_np(depth, s)
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
    valid = (depth > 0) & (depth <= max_depth_clip)
    return depth, valid


def load_gt_diode(depth_path, mask_path, max_depth_clip):
    """Load DIODE GT depth + mask → (224,224) float32 metres + valid mask."""
    depth = np.load(depth_path).squeeze().astype(np.float32)
    mask  = np.load(mask_path).squeeze()
    h, w  = depth.shape
    s     = min(h, w)
    depth = _center_crop_np(depth, s)
    mask  = _center_crop_np(mask,  s)
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask.astype(np.float32), (224, 224),
                       interpolation=cv2.INTER_NEAREST)
    valid = (mask > 0) & (depth > 0) & (depth <= max_depth_clip)
    return depth, valid


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

def generate_depth_tokens(sampler_fm, tok_text, rgb_tensor, device, seed):
    """RGB tensor → raw depth map (224,224) in DiVAE z-score space."""
    ntoks    = DEPTH_GEN_PARAMS['tokens_per_target']
    schedule = build_chained_generation_schedules(
        cond_domains           = ['rgb@224'],
        target_domains         = ['tok_depth@224'],
        tokens_per_target      = [ntoks],
        autoregression_schemes = [DEPTH_GEN_PARAMS['autoregression_scheme']],
        decoding_steps         = [DEPTH_GEN_PARAMS['decoding_steps']],
        token_decoding_schedules=[DEPTH_GEN_PARAMS['token_decoding_schedule']],
        temps                  = [DEPTH_GEN_PARAMS['temp']],
        temp_schedules         = [DEPTH_GEN_PARAMS['temp_schedule']],
        cfg_scales             = [DEPTH_GEN_PARAMS['cfg_scale']],
        cfg_schedules          = [DEPTH_GEN_PARAMS['cfg_schedule']],
        cfg_grow_conditioning  = True,
    )
    eos_id      = tok_text.token_to_id('[EOS]')
    sample_dict = {'rgb@224': {'tensor': rgb_tensor}}
    sample_dict = init_full_input_modality(
        sample_dict, MODALITY_INFO, 'rgb@224', device, eos_id=eos_id)
    sample_dict = init_empty_target_modality(
        sample_dict, MODALITY_INFO, 'tok_depth@224', 1, ntoks, device)
    return sampler_fm.generate(sample_dict, schedule,
                                text_tokenizer=tok_text, verbose=False, seed=seed)


def decode_depth_raw(out_dict, tok_depth_model, divae_steps):
    """Depth tokens → raw float map (224,224) in z-score space."""
    raw = decode_tok_depth(
        out_dict, {'tok_depth': tok_depth_model},
        key='tok_depth@224', image_size=224, patch_size=16,
        t=divae_steps, verbose=False, cmap=None,
    )
    if raw.ndim == 3:
        raw = raw[0]
    return raw.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def scale_shift_align(pred, gt, valid):
    """Least-squares affine alignment: find scale, shift minimising MSE(scale*pred+shift, gt).

    This is the standard affine-invariant evaluation protocol (MiDaS, ZoeDepth,
    Depth Anything) for models that produce relative depth (no absolute scale).
    Converts pred from z-score space to the same metric scale as gt.
    """
    p = pred[valid].reshape(-1, 1).astype(np.float64)
    g = gt[valid].reshape(-1).astype(np.float64)
    A = np.hstack([p, np.ones_like(p)])
    result = np.linalg.lstsq(A, g, rcond=None)
    scale, shift = result[0]
    pred_aligned = scale * pred + shift
    # Clip to positive depth (alignment can overshoot to negatives)
    pred_aligned = np.clip(pred_aligned, 1e-3, None)
    return pred_aligned.astype(np.float32), float(scale), float(shift)


def compute_metrics(pred_m, gt_m, valid):
    """Compute all depth metrics in metric space (metres).

    Args:
        pred_m: (H,W) float32 — scale-shift-aligned prediction in metres
        gt_m:   (H,W) float32 — GT depth in metres
        valid:  (H,W) bool    — valid pixel mask

    Returns:
        dict with abs_rel, rmse, delta1, delta2, delta3, l1_norm
    """
    p = pred_m[valid].astype(np.float64)
    g = gt_m[valid].astype(np.float64)

    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse    = float(np.sqrt(np.mean((p - g) ** 2)))

    ratio   = np.maximum(p / g, g / p)
    delta1  = float(np.mean(ratio < 1.25))
    delta2  = float(np.mean(ratio < 1.25 ** 2))
    delta3  = float(np.mean(ratio < 1.25 ** 3))

    # L1 in [0,1] min-max space — kept for backward compatibility
    def minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / max(hi - lo, 1e-8)
    l1_norm = float(np.abs(minmax(pred_m[valid]) - minmax(gt_m[valid])).mean())

    return dict(abs_rel=abs_rel, rmse=rmse,
                delta1=delta1, delta2=delta2, delta3=delta3,
                l1_norm=l1_norm)


def aggregate_metrics(records):
    """Mean of each metric across a list of per-image metric dicts."""
    if not records:
        return {}
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Summary formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_row(label, m, n):
    if not m:
        return f'  {label:<20}  N/A'
    return (f'  {label:<20}  '
            f'AbsRel={m["abs_rel"]:.4f}  '
            f'RMSE={m["rmse"]:.4f}m  '
            f'd1={m["delta1"]:.4f}  '
            f'd2={m["delta2"]:.4f}  '
            f'd3={m["delta3"]:.4f}  '
            f'L1={m["l1_norm"]:.4f}  '
            f'(n={n})')


def build_summary_txt(args, run_dir, agg_by_split, counts, elapsed):
    model_tag = 'LoRA' if args.lora_weights else 'Baseline'
    lines = [
        '',
        '=' * 80,
        f'  Depth Evaluation Summary',
        '=' * 80,
        f'  Dataset      : {args.dataset.upper()}',
        f'  Model        : {args.fm}',
        f'  Model type   : {model_tag}',
    ]
    if args.lora_weights:
        lines.append(f'  LoRA weights : {args.lora_weights}')
    lines += [
        f'  DiVAE steps  : {args.divae_steps}',
        f'  Max depth    : {args.max_depth_clip}m',
        f'  Alignment    : scale-and-shift (affine-invariant, per-image)',
        f'  Time         : {elapsed:.1f}s  ({elapsed/max(sum(counts.values()),1):.1f}s/img)',
        '',
        '  METRICS  (AbsRel↓  RMSE↓  δ<1.25↑  δ<1.25²↑  δ<1.25³↑  L1↓)',
        '  ' + '-' * 74,
    ]
    for split, m in agg_by_split.items():
        lines.append(_fmt_row(split, m, counts.get(split, 0)))
    lines += [
        '',
        '  Note: Affine-invariant evaluation (MiDaS/ZoeDepth/DepthAnything protocol).',
        '        L1_norm kept for backward-compatibility with eval_diode.py runs.',
        '',
        f'  Output dir: {run_dir}',
        '=' * 80,
        '',
    ]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    if args.run_name is None:
        model_tag  = 'lora' if args.lora_weights else 'baseline'
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'{args.dataset}_{model_tag}_{timestamp}'

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ── Save config ───────────────────────────────────────────────────────────
    config = vars(args)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    device      = torch.device(args.device)
    dtype_torch = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    model_tag   = 'LoRA' if args.lora_weights else 'Baseline'

    print('=' * 80)
    print(f'  Depth Evaluation — {args.dataset.upper()} — {model_tag}')
    print('=' * 80)
    print(f'  FM model    : {args.fm}')
    print(f'  LoRA weights: {args.lora_weights or "None (baseline)"}')
    print(f'  Dataset     : {args.dataset_root}')
    print(f'  Output      : {run_dir}')
    print()

    # ── Collect samples ───────────────────────────────────────────────────────
    print('Collecting samples ...', end='', flush=True)
    if args.dataset == 'nyu':
        samples = collect_nyu_samples(args.dataset_root)
    else:
        samples = collect_diode_samples(args.dataset_root)

    if not samples:
        print('\nERROR: No samples found. Check --dataset_root.')
        sys.exit(1)
    print(f' {len(samples)} found')

    if args.max_images is not None:
        samples = samples[:args.max_images]
        print(f'  → Capped at {args.max_images} (smoke-test mode)')
    n_total = len(samples)
    print()

    # ── Load FM model ─────────────────────────────────────────────────────────
    print('Loading FM model ...')
    fm = FM.from_pretrained(args.fm)

    if args.lora_weights:
        print(f'  Injecting LoRA weights from {args.lora_weights} ...')
        lora_sd = torch.load(args.lora_weights, map_location='cpu')
        missing, unexpected = fm.load_state_dict(lora_sd, strict=False)
        n_loaded = len(lora_sd)
        print(f'  Loaded {n_loaded} LoRA tensors  '
              f'(missing={len(missing)}, unexpected={len(unexpected)})')
        if unexpected:
            print(f'  WARNING unexpected keys: {unexpected[:3]}')

    fm = fm.to(device=device, dtype=dtype_torch).eval()
    print(f'  FM ready  ({sum(p.numel() for p in fm.parameters())/1e6:.0f}M params)')
    sampler_fm = GenerationSampler(fm)

    # ── Load DiVAE ────────────────────────────────────────────────────────────
    print('Loading depth DiVAE ...')
    tok_depth = DiVAE.from_pretrained(args.tok_depth).to(device=device).eval()
    print('  DiVAE ready')

    # ── Load text tokenizer ───────────────────────────────────────────────────
    tok_text_path = args.tok_text
    if not os.path.isabs(tok_text_path):
        tok_text_path = os.path.join(_REPO_ROOT, tok_text_path)
    tok_text = Tokenizer.from_file(tok_text_path)
    print(f'  Text tokenizer ready\n')

    # ── Evaluation loop ───────────────────────────────────────────────────────
    per_image_results = []
    metrics_by_split  = {}   # split → list of metric dicts
    t0 = time.time()

    for idx, s in enumerate(samples):
        if idx == 0 or (idx + 1) % 25 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            rate    = elapsed / max(idx, 1)
            eta     = rate * (n_total - idx - 1)
            print(f'  [{idx+1:4d}/{n_total}]  '
                  f'elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s', flush=True)

        try:
            # 1. RGB → tensor
            rgb_tensor = load_rgb_tensor(s['rgb'], device, dtype_torch)

            # 2. FM inference → depth tokens → z-score map
            with torch.no_grad():
                out_dict = generate_depth_tokens(
                    sampler_fm, tok_text, rgb_tensor, device,
                    seed=args.seed + idx)
            with torch.no_grad():
                pred_raw = decode_depth_raw(out_dict, tok_depth, args.divae_steps)

            # 3. Load GT depth + valid mask
            if args.dataset == 'nyu':
                gt_depth, valid = load_gt_nyu(s['depth'], args.max_depth_clip)
            else:
                gt_depth, valid = load_gt_diode(
                    s['depth'], s['mask'], args.max_depth_clip)

            if valid.sum() < 100:
                per_image_results.append(
                    {'name': s['name'], 'split': s['split'],
                     'metrics': None, 'error': 'too_few_valid_pixels'})
                continue

            # 4. Affine-invariant alignment: z-score → metres
            pred_aligned, scale, shift = scale_shift_align(pred_raw, gt_depth, valid)

            # 5. Compute metrics
            m = compute_metrics(pred_aligned, gt_depth, valid)

            per_image_results.append({
                'name':    s['name'],
                'split':   s['split'],
                'metrics': m,
                'align':   {'scale': scale, 'shift': shift},
            })
            metrics_by_split.setdefault(s['split'], []).append(m)

        except Exception as e:
            import traceback
            print(f'  ERROR {s["name"]}: {e}')
            traceback.print_exc()
            per_image_results.append(
                {'name': s['name'], 'split': s['split'],
                 'metrics': None, 'error': str(e)})

    elapsed_total = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_by_split = {split: aggregate_metrics(ms)
                    for split, ms in metrics_by_split.items()}
    counts = {split: len(ms) for split, ms in metrics_by_split.items()}

    # Overall = mean over all valid images regardless of split
    all_metrics = [r['metrics'] for r in per_image_results if r['metrics'] is not None]
    agg_by_split['all'] = aggregate_metrics(all_metrics)
    counts['all'] = len(all_metrics)

    # ── Save results.json ─────────────────────────────────────────────────────
    output = {
        'config':       config,
        'summary':      {
            'dataset':      args.dataset,
            'model':        args.fm,
            'lora_weights': args.lora_weights,
            'model_type':   model_tag,
            'n_total':      n_total,
            'n_valid':      len(all_metrics),
            'n_errors':     n_total - len(all_metrics),
            'elapsed_s':    elapsed_total,
            's_per_image':  elapsed_total / max(len(all_metrics), 1),
            'metrics_all':  agg_by_split,
            'counts':       counts,
        },
        'per_image':    per_image_results,
    }
    json_path = os.path.join(run_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    # ── Save & print summary.txt ──────────────────────────────────────────────
    summary_txt = build_summary_txt(args, run_dir, agg_by_split, counts, elapsed_total)
    txt_path = os.path.join(run_dir, 'summary.txt')
    with open(txt_path, 'w') as f:
        f.write(summary_txt)

    print(summary_txt)
    print(f'  results.json → {json_path}')
    print(f'  summary.txt  → {txt_path}')
    print(f'  config.json  → {os.path.join(run_dir, "config.json")}')


if __name__ == '__main__':
    main()
