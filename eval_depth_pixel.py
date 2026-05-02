#!/usr/bin/env python3
# Copyright 2024 Francesco Colasurdo / AMORe Lab, TU/e
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# -----------------------------------------------------------------------
# eval_depth_pixel.py
#
# Full NYUv2 validation set evaluation for 4M-21 depth estimation.
# Evaluates in metric depth (metres) using per-image median-scale alignment.
#
# Motivation
# ----------
# During tokenization, depth is per-image z-scored (truncated standardisation)
# before encoding, so decoded tokens live in a per-image normalised space that
# has no direct absolute scale.  Standard metric evaluation (Eigen et al., 2014)
# recovers the scale per image by aligning the predicted median to the GT median.
# This is the same protocol used by Monodepth2, MiDaS, ZoeDepth, etc.
#
# Metrics (standard NYUv2 Eigen-split, cap 10 m)
# -----------------------------------------------
#   δ₁       higher is better  % pixels where max(pred/gt, gt/pred) < 1.25
#   δ₂       higher is better  threshold 1.25²
#   δ₃       higher is better  threshold 1.25³
#   RMSE     lower is better   √mean((pred − gt)²)                 [m]
#   AbsRel   lower is better   mean(|pred − gt| / gt)
#   SqRel    lower is better   mean((pred − gt)² / gt)             [m]
#   log RMSE lower is better   √mean((log pred − log gt)²)
#   SILog    lower is better   scale-invariant log error (Eigen14) [%]
#
# Usage
# -----
#   # Baseline (no LoRA):
#   python eval_depth_pixel.py --mode baseline
#
#   # LoRA checkpoint:
#   python eval_depth_pixel.py --mode lora \
#       --lora_ckpt output/depth/nyu_rgb_r8_e5/lora_weights-best.pth
#
#   # Compare both in a single run:
#   python eval_depth_pixel.py --mode both \
#       --lora_ckpt output/depth/nyu_rgb_r8_e5/lora_weights-best.pth
#
# Run on Snellius (single A100):
#   sbatch scripts/snellius/06_eval_depth.sh
# -----------------------------------------------------------------------

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path.home() / '4m-peft'
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

HF_CACHE          = '/scratch-shared/tesr125338/hf_cache'
NYU_VAL_TOKENS    = '/scratch-shared/tesr125338/nyu_tokens/val'
NYU_DEPTH_CAP_M   = 10.0           # standard NYUv2 eval cap (metres)
PRETRAINED        = 'EPFL-VILAB/4M-21_XL'
DEPTH_TOK_REPO    = 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448'
DECODE_TIMESTEPS  = 25             # DDIM steps for DiVAE decoder
IMAGE_SIZE        = 224
NUM_TOKENS        = 196


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='NYUv2 metric depth evaluation')
    p.add_argument('--mode', default='baseline',
                   choices=['baseline', 'lora', 'both'],
                   help='baseline: zero-shot 4M-21;  lora: load adapter weights; '
                        'both: evaluate baseline then lora and print comparison table')
    p.add_argument('--lora_ckpt', default='',
                   help='Path to lora_weights-*.pth (required for mode lora/both)')
    p.add_argument('--lora_rank', default=8, type=int)
    p.add_argument('--lora_scale', default=1.0, type=float)
    p.add_argument('--lora_modules', default='attention',
                   choices=['selfattn', 'crossattn', 'attention', 'mlp', 'all'])
    p.add_argument('--pretrained_model', default=PRETRAINED)
    p.add_argument('--tok_val_dir', default=NYU_VAL_TOKENS,
                   help='Root of pre-tokenized val data (must contain tok_rgb/ + tok_depth/)')
    p.add_argument('--hf_cache', default=HF_CACHE)
    p.add_argument('--batch_size', default=4, type=int,
                   help='Images per batch during DiVAE decoding (A100: 4-8)')
    p.add_argument('--decode_timesteps', default=DECODE_TIMESTEPS, type=int,
                   help='DDIM steps for the DiVAE diffusion decoder')
    p.add_argument('--max_samples', default=None, type=int,
                   help='Evaluate on the first N samples only (None = full val set)')
    p.add_argument('--output_json', default='',
                   help='If set, write results dict to this JSON file')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='bfloat16',
                   choices=['float32', 'bfloat16'])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ground-truth depth loader — streams raw depth (metres) from HDF5 tar files
# ---------------------------------------------------------------------------

def _iter_gt_depth_from_tars(hf_cache: str):
    """Yield (idx, depth_np) for every sample in the val split.

    depth_np: float32 ndarray, shape (H, W), values in metres.
    Samples arrive in the same deterministic order used by prepare_nyu.py
    (tars sorted alphabetically, h5 members sorted within each tar).
    """
    snap_dir = Path(hf_cache) / 'datasets--sayakpaul--nyu_depth_v2'
    data_dir = sorted((snap_dir / 'snapshots').iterdir())[-1] / 'data'
    val_tars = sorted(data_dir.glob('val-*.tar'))
    if not val_tars:
        raise FileNotFoundError(
            f'No val-*.tar files found under {data_dir}\n'
            'Make sure prepare_nyu.py has been run and HF_CACHE is set correctly.')

    global_idx = 0
    for tar_path in sorted(val_tars):
        with tarfile.open(str(tar_path), 'r') as tf:
            members = sorted(
                [m for m in tf.getmembers() if m.name.endswith('.h5')],
                key=lambda m: m.name)
            for member in members:
                raw = tf.extractfile(member).read()
                with h5py.File(io.BytesIO(raw), 'r') as f:
                    depth = f['depth'][:]  # (H, W) float32, metres
                yield global_idx, depth
                global_idx += 1


# ---------------------------------------------------------------------------
# Token dataset — loads all pre-tokenized val token files into memory
# ---------------------------------------------------------------------------

def load_val_tokens(tok_val_dir: str, max_samples=None):
    """Load sorted val token arrays.

    Returns:
        rgb_tokens:   (N, 196) int64 tensor
        depth_tokens: (N, 196) int64 tensor
        stems:        list of str, e.g. ['00000', '00001', ...]
    """
    val_root  = Path(tok_val_dir)
    rgb_dir   = val_root / 'tok_rgb'
    depth_dir = val_root / 'tok_depth'

    for d in (rgb_dir, depth_dir):
        if not d.exists():
            raise FileNotFoundError(f'Token directory not found: {d}')

    rgb_stems   = sorted(p.stem for p in rgb_dir.rglob('*.npy'))
    depth_stems = sorted(p.stem for p in depth_dir.rglob('*.npy'))
    common      = sorted(set(rgb_stems) & set(depth_stems))

    if max_samples is not None:
        common = common[:max_samples]

    print(f'Loading {len(common)} val samples from {tok_val_dir} ...')

    def _load(folder, stem):
        matches = list(folder.rglob(f'{stem}.npy'))
        assert matches, f'{stem}.npy not found under {folder}'
        return np.load(matches[0]).astype(np.int64)

    rgb_list   = [_load(rgb_dir,   s) for s in common]
    depth_list = [_load(depth_dir, s) for s in common]

    rgb_tokens   = torch.from_numpy(np.stack(rgb_list))
    depth_tokens = torch.from_numpy(np.stack(depth_list))

    print(f'  RGB tokens:   {rgb_tokens.shape}')
    print(f'  Depth tokens: {depth_tokens.shape}')
    return rgb_tokens, depth_tokens, common


# ---------------------------------------------------------------------------
# Build 4M mod_dict for inference (same logic as run_finetuning_lora.py)
# ---------------------------------------------------------------------------

def build_mod_dict(input_tokens, target_tokens, device):
    """All 196 RGB tokens visible; depth tokens masked (to be predicted)."""
    B = input_tokens.shape[0]
    return {
        'tok_rgb@224': {
            'tensor':                 input_tokens.to(device),
            'input_mask':             torch.zeros((B, NUM_TOKENS), dtype=torch.bool, device=device),
            'target_mask':            torch.ones( (B, NUM_TOKENS), dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.zeros((B, NUM_TOKENS), dtype=torch.int,  device=device),
        },
        'tok_depth@224': {
            'tensor':                 target_tokens.to(device),
            'input_mask':             torch.ones( (B, NUM_TOKENS), dtype=torch.bool, device=device),
            'target_mask':            torch.zeros((B, NUM_TOKENS), dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.ones( (B, NUM_TOKENS), dtype=torch.int,  device=device),
        },
    }


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_4m_model(pretrained_model: str, hf_cache: str):
    """Load 4M-21_XL with no LoRA (pure baseline)."""
    from fourm.models.fm import FM
    print(f'Loading {pretrained_model} ...  (may take ~2 min)')
    model = FM.from_pretrained(pretrained_model, cache_dir=hf_cache)
    return model


def inject_lora(model, lora_rank: int, lora_scale: float, lora_modules: str):
    """Inject LoRA adapters (zeroed — act as identity until weights loaded)."""
    from fourm.models.lora_utils import get_LoRA_module_names, inject_trainable_LoRA
    target_modules = get_LoRA_module_names(lora_modules)
    inject_trainable_LoRA(model, rank=lora_rank, scale=lora_scale,
                          target_replace_modules=target_modules)


def load_lora_weights(model, ckpt_path: str, device):
    """Load pre-trained LoRA adapter weights into an LoRA-injected model."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'LoRA checkpoint not found: {ckpt_path}')
    sd = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(sd, strict=False)
    print(f'  Loaded {len(sd)} LoRA tensors from {ckpt_path}')


def load_depth_tokenizer(hf_cache: str, device):
    """Load the depth DiVAE tokenizer (for decoding tokens → pixel depth)."""
    from fourm.vq.vqvae import DiVAE
    print(f'Loading depth DiVAE tokenizer ({DEPTH_TOK_REPO}) ...')
    tok = DiVAE.from_pretrained(DEPTH_TOK_REPO, cache_dir=hf_cache)
    tok.eval()
    tok.to(device)
    print('  Depth tokenizer ready.')
    return tok


# ---------------------------------------------------------------------------
# Inference: predict depth token indices for all val samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_depth_tokens(model, rgb_tokens, depth_tokens, device, dtype,
                          batch_size: int = 4):
    """Run forward pass over all samples, returning predicted depth token ids.

    Args:
        model:        4M FM model (with or without LoRA)
        rgb_tokens:   (N, 196) int64 CPU tensor — all val RGB tokens
        depth_tokens: (N, 196) int64 CPU tensor — GT depth tokens (used as
                      decoder input; model is trained to predict these)
        device, dtype: obvious
        batch_size:   images per forward pass

    Returns:
        pred_tokens: (N, 196) int64 CPU tensor
        token_acc:   scalar float — fraction of tokens matching GT
    """
    N = rgb_tokens.shape[0]
    all_pred = []

    model.eval()
    for i in range(0, N, batch_size):
        rgb_b   = rgb_tokens[i: i + batch_size]
        depth_b = depth_tokens[i: i + batch_size]

        mod_dict = build_mod_dict(rgb_b, depth_b, device)
        with torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == 'cuda')):
            logits = model(mod_dict,
                           num_encoder_tokens=NUM_TOKENS,
                           num_decoder_tokens=NUM_TOKENS,
                           return_logits=True)

        pred = logits['tok_depth@224'].float().argmax(dim=-1).cpu()  # (B, 196)
        all_pred.append(pred)

        if (i // batch_size) % 20 == 0:
            print(f'  Inference: {min(i + batch_size, N)}/{N}')

    pred_tokens = torch.cat(all_pred, dim=0)   # (N, 196)
    token_acc   = (pred_tokens == depth_tokens).float().mean().item()
    return pred_tokens, token_acc


# ---------------------------------------------------------------------------
# Decode depth tokens → pixel depth maps via DiVAE diffusion decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_depth_tokens(depth_tok, token_ids, device, dtype,
                         batch_size: int = 4, timesteps: int = 25):
    """Decode (N, 196) token indices → (N, H, W) float depth maps.

    The decoded values are in the DiVAE's output space (per-image
    standardised during tokenization — not in metres).

    Returns:
        depth_maps: (N, 224, 224) float32 CPU tensor
    """
    N = token_ids.shape[0]
    all_depths = []

    for i in range(0, N, batch_size):
        toks = token_ids[i: i + batch_size].to(device).long()  # (B, 196)
        toks_2d = toks.reshape(toks.shape[0], 14, 14)

        with torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == 'cuda')):
            dec = depth_tok.decode_tokens(
                toks_2d, timesteps=timesteps,
                image_size=IMAGE_SIZE, verbose=False)   # (B, 1, H, W)

        depth = dec[:, 0].float().cpu()   # (B, H, W)
        all_depths.append(depth)

        if (i // batch_size) % 20 == 0:
            print(f'  Decoding: {min(i + batch_size, N)}/{N}')

    return torch.cat(all_depths, dim=0)   # (N, 224, 224)


# ---------------------------------------------------------------------------
# Load raw GT depth (metres) aligned to the token file ordering
# ---------------------------------------------------------------------------

def load_gt_depth_metres(hf_cache: str, stems,
                          target_size: int = IMAGE_SIZE):
    """Load GT depth maps in metres for the given sample indices.

    Args:
        stems: list of token file stems, e.g. ['00000', '00012', ...]
               The stem integer is the global index in the val split.

    Returns:
        gt_depths: (N, H, W) float32 ndarray in metres
    """
    needed   = {int(s) for s in stems}
    ordered  = sorted(int(s) for s in stems)
    idx_to_s = {int(s): s for s in stems}

    results  = {}   # int_idx → float32 ndarray

    print(f'Loading raw GT depth for {len(stems)} samples from HDF5 tars ...')
    for global_idx, depth_m in _iter_gt_depth_from_tars(hf_cache):
        if global_idx in needed:
            # Resize to match decoded depth size
            pil = Image.fromarray(depth_m, mode='F')
            pil = pil.resize((target_size, target_size), Image.NEAREST)
            results[global_idx] = np.array(pil, dtype=np.float32)

        if len(results) == len(needed):
            break   # got everything we need

    if len(results) < len(needed):
        missing = needed - set(results.keys())
        raise RuntimeError(f'Could not find GT depth for indices: {sorted(missing)[:10]}')

    print(f'  GT depth loaded: {len(results)} samples')
    gt_list = [results[i] for i in ordered]
    return np.stack(gt_list)   # (N, H, W) float32 metres


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_depth_metrics(pred_np: np.ndarray,
                           gt_np:   np.ndarray,
                           cap_m:   float = NYU_DEPTH_CAP_M,
                           eps:     float = 1e-6) -> dict:
    """Compute standard monocular depth metrics for a SINGLE image pair.

    Both arrays must be float32, shape (H, W).
    pred_np is already scale-aligned to gt_np (caller's responsibility).
    """
    # Valid mask: GT in (0, cap_m] and prediction positive
    mask = (gt_np > eps) & (gt_np <= cap_m) & (pred_np > eps)
    if mask.sum() < 10:
        return None

    p = pred_np[mask].astype(np.float64)
    g = gt_np[mask].astype(np.float64)

    ratio = np.maximum(p / g, g / p)
    delta1 = (ratio < 1.25   ).mean()
    delta2 = (ratio < 1.25**2).mean()
    delta3 = (ratio < 1.25**3).mean()

    rmse   = np.sqrt(((p - g) ** 2).mean())
    absrel = (np.abs(p - g) / g).mean()
    sqrel  = (((p - g) ** 2) / g).mean()
    log_rmse = np.sqrt(((np.log(p) - np.log(g)) ** 2).mean())

    # Scale-invariant log error (Eigen 2014): SILog in %
    d = np.log(p) - np.log(g)
    silog = np.sqrt((d ** 2).mean() - d.mean() ** 2) * 100.0

    return {
        'delta1':   delta1,
        'delta2':   delta2,
        'delta3':   delta3,
        'rmse':     rmse,
        'absrel':   absrel,
        'sqrel':    sqrel,
        'log_rmse': log_rmse,
        'silog':    silog,
    }


def per_image_median_align(pred_map: np.ndarray,
                            gt_map:  np.ndarray,
                            cap_m:   float = NYU_DEPTH_CAP_M,
                            eps:     float = 1e-6) -> np.ndarray:
    """Scale pred so that median(pred_valid) = median(gt_valid).

    This is the standard scale-only alignment used by Monodepth2 / Eigen.
    Both maps are (H, W) float32.  gt_map is in metres.
    pred_map is in DiVAE normalised space (arbitrary units).
    """
    mask_gt = (gt_map > eps) & (gt_map <= cap_m)
    if mask_gt.sum() < 10 or pred_map[mask_gt].max() <= eps:
        return pred_map   # fallback: no alignment

    # Shift pred to non-negative before scaling (in case of negative values)
    pred_pos = pred_map - pred_map[mask_gt].min() + eps

    scale = np.median(gt_map[mask_gt]) / np.median(pred_pos[mask_gt])
    return scale * pred_pos


# ---------------------------------------------------------------------------
# Evaluate one configuration: baseline or lora
# ---------------------------------------------------------------------------

def evaluate_config(
        model,
        depth_tok,
        rgb_tokens,
        depth_tokens,
        gt_depth_np,
        stems,
        device,
        dtype,
        args,
        label: str = 'baseline',
):
    """Run full evaluation for one model configuration.

    Returns dict of averaged metric values.
    """
    print(f'\n{"="*60}')
    print(f'Evaluating: {label}')
    print(f'{"="*60}')

    # Step 1: predict depth token ids
    print('Step 1/3 — Predicting depth tokens ...')
    pred_tokens, token_acc = predict_depth_tokens(
        model, rgb_tokens, depth_tokens, device, dtype,
        batch_size=args.batch_size)
    print(f'  Token accuracy: {token_acc * 100:.2f}%')

    # Step 2: decode predicted tokens → pixel depth (normalised space)
    print('Step 2/3 — Decoding tokens → depth maps via DiVAE ...')
    pred_depth = decode_depth_tokens(
        depth_tok, pred_tokens, device, dtype,
        batch_size=args.batch_size,
        timesteps=args.decode_timesteps)
    pred_depth_np = pred_depth.numpy()   # (N, H, W) in normalised space

    # Step 3: per-image median-scale alignment + metric computation
    print('Step 3/3 — Aligning to GT and computing metrics ...')
    N = len(stems)
    metric_accum = {k: [] for k in
                    ['delta1', 'delta2', 'delta3', 'rmse',
                     'absrel', 'sqrel', 'log_rmse', 'silog']}

    for i in range(N):
        pred_img = pred_depth_np[i]            # (H, W) normalised
        gt_img   = gt_depth_np[i]              # (H, W) metres

        # Align scale
        pred_aligned = per_image_median_align(pred_img, gt_img)

        # Compute metrics
        m = compute_depth_metrics(pred_aligned, gt_img)
        if m is None:
            continue   # skip degenerate samples

        for k, v in m.items():
            metric_accum[k].append(v)

    # Average over all valid images
    results = {
        'label':      label,
        'n_samples':  N,
        'n_valid':    len(metric_accum['delta1']),
        'token_acc':  token_acc,
    }
    for k, vals in metric_accum.items():
        results[k] = float(np.mean(vals)) if vals else float('nan')

    return results


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results_table(results_list: list):
    """Print a comparison table for one or more configurations."""
    cols = ['delta1', 'delta2', 'delta3', 'rmse', 'absrel', 'sqrel',
            'log_rmse', 'silog']
    col_display = ['δ₁↑', 'δ₂↑', 'δ₃↑', 'RMSE↓', 'AbsRel↓', 'SqRel↓',
                   'logRMSE↓', 'SILog↓']

    header_w = 14
    col_w    = 10

    print('\n' + '═' * (header_w + col_w * len(cols) + 4))
    print(f'NYUv2 Depth Evaluation  (median-scale alignment, cap={NYU_DEPTH_CAP_M}m)')
    print('═' * (header_w + col_w * len(cols) + 4))

    # Column headers
    header = f'{"Method":<{header_w}} ' + '  '.join(f'{c:>{col_w}}' for c in col_display)
    print(header)
    print('─' * (header_w + col_w * len(cols) + 4))

    for r in results_list:
        row = f'{r["label"]:<{header_w}} '
        for k in cols:
            v = r.get(k, float('nan'))
            row += f'  {v:>{col_w}.4f}'
        print(row)

    print('─' * (header_w + col_w * len(cols) + 4))

    # Token accuracy footnote
    for r in results_list:
        print(f'  {r["label"]}: token accuracy = {r["token_acc"] * 100:.2f}%  '
              f'({r["n_valid"]}/{r["n_samples"]} valid samples)')

    print('\nNote: δ₁/δ₂/δ₃ higher is better. RMSE/AbsRel/SqRel/logRMSE/SILog lower is better.')
    print('      Metrics in metres (after per-image median-scale alignment).')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    print(f'Device : {device}')
    print(f'Dtype  : {dtype}')
    print(f'Mode   : {args.mode}')
    if args.mode in ('lora', 'both'):
        if not args.lora_ckpt:
            raise ValueError('--lora_ckpt is required for mode lora/both')
        print(f'LoRA   : {args.lora_ckpt}')

    # ── Load pre-tokenized val data ──────────────────────────────────────────
    rgb_tokens, depth_tokens, stems = load_val_tokens(
        args.tok_val_dir, max_samples=args.max_samples)

    # ── Load GT depth in metres ──────────────────────────────────────────────
    gt_depth_np = load_gt_depth_metres(args.hf_cache, stems)
    print(f'GT depth shape: {gt_depth_np.shape}  '
          f'range [{gt_depth_np.min():.2f}, {gt_depth_np.max():.2f}] m')

    # ── Load depth tokenizer (decoder) ──────────────────────────────────────
    depth_tok = load_depth_tokenizer(args.hf_cache, device)

    # ── Load 4M model ────────────────────────────────────────────────────────
    model = load_4m_model(args.pretrained_model, args.hf_cache)

    if args.mode in ('lora', 'both'):
        inject_lora(model, args.lora_rank, args.lora_scale, args.lora_modules)

    model = model.to(dtype).to(device).eval()

    results_list = []

    # ── Baseline evaluation ──────────────────────────────────────────────────
    if args.mode in ('baseline', 'both'):
        # For 'both': LoRA was injected but weights are still zero → pure baseline
        r = evaluate_config(
            model, depth_tok, rgb_tokens, depth_tokens,
            gt_depth_np, stems, device, dtype, args,
            label='4M-21 (baseline)')
        results_list.append(r)

    # ── LoRA evaluation ──────────────────────────────────────────────────────
    if args.mode in ('lora', 'both'):
        # LoRA adapters were injected before model.to(device) above.
        # All we need to do here is load the trained weights on top.
        load_lora_weights(model, args.lora_ckpt, device)
        model.eval()

        r = evaluate_config(
            model, depth_tok, rgb_tokens, depth_tokens,
            gt_depth_np, stems, device, dtype, args,
            label=f'4M-21 + LoRA r{args.lora_rank}')
        results_list.append(r)

    # ── Print table ──────────────────────────────────────────────────────────
    print_results_table(results_list)

    # ── Optionally save to JSON ───────────────────────────────────────────────
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f'\nResults saved → {args.output_json}')


if __name__ == '__main__':
    main()
