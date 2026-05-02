#!/usr/bin/env python3
"""
debug_diode_metric.py
=====================
Runs the DIODE depth eval pipeline on 3 images and prints the value
range at every step so we can understand:
  1. What range does the DiVAE output (pred_raw)?
  2. What range does GT depth have (metres)?
  3. What range does GT have after truncated_depth_standardization?
  4. How does each normalization strategy affect the L1?

Usage:
  python debug_diode_metric.py --diode_root /scratch-shared/tesr125338/diode/val
"""
import os, sys, argparse
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms.functional import center_crop

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml-4m'))

from fourm.models.fm import FM
from fourm.vq.vqvae import DiVAE
from fourm.models.generate import (
    GenerationSampler, build_chained_generation_schedules,
    init_empty_target_modality, init_full_input_modality,
)
from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import RGBTransform
from fourm.utils.plotting_utils import decode_tok_depth

import cv2
from pathlib import Path

DEPTH_GEN_PARAMS = {
    'tokens_per_target': 196, 'autoregression_scheme': 'roar',
    'decoding_steps': 1, 'token_decoding_schedule': 'linear',
    'temp': 0.01, 'temp_schedule': 'constant',
    'cfg_scale': 2.0, 'cfg_schedule': 'constant',
}

def collect_samples(diode_root, n=3):
    diode_root = Path(diode_root)
    samples = []
    for split in ['indoors', 'outdoor']:
        split_dir = diode_root / split
        if not split_dir.exists():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir(): continue
            for scan_dir in sorted(scene_dir.iterdir()):
                if not scan_dir.is_dir(): continue
                for rgb_p in sorted(scan_dir.glob('*.png')):
                    stem = rgb_p.stem
                    depth_p = scan_dir / f'{stem}_depth.npy'
                    mask_p  = scan_dir / f'{stem}_depth_mask.npy'
                    if depth_p.exists() and mask_p.exists():
                        samples.append({'rgb': str(rgb_p), 'depth': str(depth_p),
                                        'mask': str(mask_p), 'split': split, 'name': stem})
                        if len(samples) >= n:
                            return samples
    return samples

def _center_crop_np(arr, size):
    h, w = arr.shape[:2]
    return arr[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]

def truncated_depth_std_np(depth_arr, valid_mask, thresh=0.1):
    """NumPy version of fourm's truncated_depth_standardization."""
    vals = depth_arr[valid_mask].flatten()
    vals_sorted = np.sort(vals)
    lo_idx = int(thresh * len(vals_sorted))
    hi_idx = int((1 - thresh) * len(vals_sorted))
    trunc = vals_sorted[lo_idx:hi_idx]
    mean_ = trunc.mean()
    std_  = np.sqrt(trunc.var() + 1e-6)
    return (depth_arr - mean_) / std_, mean_, std_

def l1_independent_minmax(pred, gt, valid):
    """Our current (broken) metric."""
    def mm(x):
        v = x[valid]; lo, hi = v.min(), v.max()
        return (x - lo) / max(hi - lo, 1e-8)
    return float(np.abs(mm(pred)[valid] - mm(gt)[valid]).mean())

def l1_gt_zscored_gt_range(pred_raw, gt_depth, valid):
    """
    Hypothesis A:
    - GT  → truncated z-score  → normalize to [0,1] using GT z-score range
    - Pred (already in z-score space from DiVAE) → same [0,1] normalization using GT's range
    """
    gt_z, gt_mean, gt_std = truncated_depth_std_np(gt_depth, valid)
    gt_z_valid = gt_z[valid]
    z_min, z_max = gt_z_valid.min(), gt_z_valid.max()
    denom = max(z_max - z_min, 1e-8)

    gt_01   = np.clip((gt_z   - z_min) / denom, 0, 1)
    pred_01 = np.clip((pred_raw - z_min) / denom, 0, 1)
    return float(np.abs(pred_01[valid] - gt_01[valid]).mean()), gt_mean, gt_std

def l1_pred_zscored_independently(pred_raw, gt_depth, valid):
    """
    Hypothesis B:
    - GT  → truncated z-score (using GT stats)
    - Pred → truncated z-score (using Pred stats)
    - Normalize both to [0,1] using GT z-score range
    - Same as A but pred is also z-scored first
    """
    gt_z, _, _ = truncated_depth_std_np(gt_depth, valid)
    pred_z, _, _ = truncated_depth_std_np(pred_raw, valid)
    gt_z_valid = gt_z[valid]
    z_min, z_max = gt_z_valid.min(), gt_z_valid.max()
    denom = max(z_max - z_min, 1e-8)

    gt_01   = np.clip((gt_z   - z_min) / denom, 0, 1)
    pred_01 = np.clip((pred_z - z_min) / denom, 0, 1)
    return float(np.abs(pred_01[valid] - gt_01[valid]).mean())

def l1_direct_zscored(pred_raw, gt_depth, valid):
    """
    Hypothesis C:
    - GT  → truncated z-score (using GT stats)
    - Pred: use as-is from DiVAE (already in z-score space)
    - L1 directly in z-score space (NOT [0,1])
    """
    gt_z, _, _ = truncated_depth_std_np(gt_depth, valid)
    return float(np.abs(pred_raw[valid] - gt_z[valid]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diode_root', required=True)
    p.add_argument('--fm',         default='EPFL-VILAB/4M-21_XL')
    p.add_argument('--tok_depth',  default='EPFL-VILAB/4M_tokenizers_depth_8k_224-448')
    p.add_argument('--tok_text',   default='ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json')
    p.add_argument('--device',     default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    dtype  = torch.bfloat16

    print('Loading models...')
    fm = FM.from_pretrained(args.fm).to(device=device, dtype=dtype).eval()
    sampler_fm = GenerationSampler(fm)
    tok_depth  = DiVAE.from_pretrained(args.tok_depth).to(device=device).eval()
    tok_text_path = args.tok_text
    if not os.path.isabs(tok_text_path):
        tok_text_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), tok_text_path)
    tok_text = Tokenizer.from_file(tok_text_path)
    print('Models loaded.\n')

    samples = collect_samples(args.diode_root, n=3)

    for idx, s in enumerate(samples):
        print(f'{"="*60}')
        print(f'Image {idx+1}: {s["name"]}  [{s["split"]}]')
        print(f'{"="*60}')

        # --- RGB preprocessing ---
        rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
        img_pil = rgb_transform.load(s['rgb'])
        img_pil = rgb_transform.preprocess(img_pil)
        sq = min(img_pil.size)
        img_pil = center_crop(img_pil, (sq, sq)).resize((224, 224), Image.BILINEAR)
        rgb_tensor = rgb_transform.postprocess(img_pil).unsqueeze(0).to(device=device, dtype=dtype)

        # --- FM generation ---
        ntoks = DEPTH_GEN_PARAMS['tokens_per_target']
        schedule = build_chained_generation_schedules(
            cond_domains=['rgb@224'], target_domains=['tok_depth@224'],
            tokens_per_target=[ntoks],
            autoregression_schemes=[DEPTH_GEN_PARAMS['autoregression_scheme']],
            decoding_steps=[DEPTH_GEN_PARAMS['decoding_steps']],
            token_decoding_schedules=[DEPTH_GEN_PARAMS['token_decoding_schedule']],
            temps=[DEPTH_GEN_PARAMS['temp']], temp_schedules=[DEPTH_GEN_PARAMS['temp_schedule']],
            cfg_scales=[DEPTH_GEN_PARAMS['cfg_scale']], cfg_schedules=[DEPTH_GEN_PARAMS['cfg_schedule']],
            cfg_grow_conditioning=True,
        )
        eos_id = tok_text.token_to_id('[EOS]')
        sample_dict = {'rgb@224': {'tensor': rgb_tensor}}
        sample_dict = init_full_input_modality(sample_dict, MODALITY_INFO, 'rgb@224', device, eos_id=eos_id)
        sample_dict = init_empty_target_modality(sample_dict, MODALITY_INFO, 'tok_depth@224', 1, ntoks, device)

        with torch.no_grad():
            out_dict = sampler_fm.generate(sample_dict, schedule, text_tokenizer=tok_text,
                                           verbose=False, seed=idx)

        # --- DiVAE decode ---
        with torch.no_grad():
            raw = decode_tok_depth(out_dict, {'tok_depth': tok_depth},
                                   key='tok_depth@224', image_size=224, patch_size=16,
                                   t=50, verbose=False, cmap=None)
        pred_raw = raw[0] if raw.ndim == 3 else raw  # (224,224)

        # --- GT loading ---
        gt_depth = np.load(s['depth']).squeeze()
        gt_mask  = np.load(s['mask']).squeeze()
        h, w = gt_depth.shape
        sq_gt = min(h, w)
        gt_depth = _center_crop_np(gt_depth, sq_gt)
        gt_mask  = _center_crop_np(gt_mask,  sq_gt)
        gt_depth = cv2.resize(gt_depth.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
        gt_mask  = cv2.resize(gt_mask.astype(np.float32),  (224, 224), interpolation=cv2.INTER_NEAREST)
        valid    = (gt_mask > 0) & (gt_depth > 0)

        # ── Print ranges ─────────────────────────────────────────────
        print(f'\n  pred_raw (DiVAE output, raw):')
        print(f'    range  : [{pred_raw.min():.3f}, {pred_raw.max():.3f}]')
        print(f'    mean   : {pred_raw.mean():.3f}')
        print(f'    std    : {pred_raw.std():.3f}')

        print(f'\n  gt_depth (metres, valid pixels):')
        gv = gt_depth[valid]
        print(f'    range  : [{gv.min():.3f}, {gv.max():.3f}]')
        print(f'    mean   : {gv.mean():.3f}')
        print(f'    std    : {gv.std():.3f}')

        gt_z, gt_mean, gt_std = truncated_depth_std_np(gt_depth, valid)
        print(f'\n  gt_depth after truncated z-score (using GT stats):')
        print(f'    z-score params: mean={gt_mean:.3f}m, std={gt_std:.3f}m')
        gzv = gt_z[valid]
        print(f'    range  : [{gzv.min():.3f}, {gzv.max():.3f}]')
        print(f'    mean   : {gzv.mean():.3f}')
        print(f'    std    : {gzv.std():.3f}')

        # ── Compare metrics ──────────────────────────────────────────
        l1_a = l1_independent_minmax(pred_raw, gt_depth, valid)
        l1_b, _, _ = l1_gt_zscored_gt_range(pred_raw, gt_depth, valid)
        l1_c = l1_pred_zscored_independently(pred_raw, gt_depth, valid)
        l1_d = l1_direct_zscored(pred_raw, gt_depth, valid)

        print(f'\n  L1 scores (paper target = 0.68):')
        print(f'    [A] Our current  (independent min-max)          : {l1_a:.4f}')
        print(f'    [B] GT z-score → GT range norm → L1 in [0,1]   : {l1_b:.4f}')
        print(f'    [C] Both z-scored → GT range norm → L1 in [0,1]: {l1_c:.4f}')
        print(f'    [D] Pred raw vs GT z-score (direct z-score L1)  : {l1_d:.4f}')
        print()

if __name__ == '__main__':
    main()
