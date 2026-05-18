#!/usr/bin/env python3
# eval_normal_nyu.py
#
# Surface-normal estimation evaluation on NYUv2 val (Eigen split, 654 images).
# Metric: mean angle error (degrees) + % within 11.25° / 22.5° / 30°.
#
# GT normals are computed on-the-fly from the depth maps in nyu_raw/test/depth/
# using the same gradient-based method as prepare_nyu_normals.py.
#
# Usage:
#   python eval_normal_nyu.py \
#       --lora_weights /path/to/lora_weights-best.pth \
#       --lora_rank 8 --lora_modules attention \
#       --nyu_raw   /scratch-shared/tesr125338/nyu_raw \
#       --tokens_dir /scratch-shared/tesr125338/nyu_tokens_labeled/val \
#       --output_dir /scratch-shared/tesr125338/output/eval_normal/run34 \
#       --device cuda --dtype bfloat16

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

NORMAL_TOKENIZER_REPO = 'EPFL-VILAB/4M_tokenizers_normal_8k_224-448'
FM_MODEL              = 'EPFL-VILAB/4M-21_XL'

# NYUv2 Kinect intrinsics (640×480)
NYU_FX, NYU_FY = 525.0, 525.0
NYU_CX, NYU_CY = 319.5, 239.5


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='NYUv2 normal estimation eval')
    p.add_argument('--lora_weights', default=None)
    p.add_argument('--lora_rank',    default=8, type=int)
    p.add_argument('--lora_modules', default='attention', type=str)
    p.add_argument('--nyu_raw',      required=True,
                   help='nyu_raw root (contains test/depth/*.npy)')
    p.add_argument('--tokens_dir',   required=True,
                   help='nyu_tokens_labeled/val dir (contains tok_rgb/)')
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--run_name',     default='normal_eval')
    p.add_argument('--hf_cache',     default=None)
    p.add_argument('--max_images',   default=0, type=int,
                   help='Max images (0 = all 654)')
    p.add_argument('--device',       default='cuda')
    p.add_argument('--dtype',        default='bfloat16',
                   choices=['float32', 'bfloat16'])
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(args, device, dtype):
    from fourm.models.fm import FM
    from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names

    fm = FM.from_pretrained(FM_MODEL, cache_dir=args.hf_cache)

    if args.lora_weights:
        target_mods = get_LoRA_module_names(args.lora_modules)
        inject_trainable_LoRA(fm, rank=args.lora_rank, scale=1.0,
                              target_replace_modules=target_mods)
        lora_sd = torch.load(args.lora_weights, map_location='cpu')
        missing, unexpected = fm.load_state_dict(lora_sd, strict=False)
        print(f'  Loaded {len(lora_sd)} LoRA tensors '
              f'(missing={len(missing)}, unexpected={len(unexpected)})')

    return fm.to(device=device, dtype=dtype).eval()


# ── Generation ────────────────────────────────────────────────────────────────

def generate_normal_tokens(fm, rgb_tokens, device, dtype):
    """RGB tokens (1, 196) → predicted normal tokens (1, 196)."""
    from fourm.models.generate import (
        GenerationSampler, build_chained_generation_schedules,
        init_empty_target_modality, init_full_input_modality,
    )
    from fourm.data.modality_info import MODALITY_INFO
    from tokenizers import Tokenizer

    tok_path = str(REPO_ROOT / 'ml-4m' / 'fourm' / 'utils' / 'tokenizer' /
                   'trained' / 'text_tokenizer_4m_wordpiece_30k.json')
    tok_text = Tokenizer.from_file(tok_path)
    eos_id   = tok_text.token_to_id('[EOS]')

    sampler  = GenerationSampler(fm)
    schedule = build_chained_generation_schedules(
        cond_domains            = ['tok_rgb@224'],
        target_domains          = ['tok_normal@224'],
        tokens_per_target       = [196],
        autoregression_schemes  = ['roar'],
        decoding_steps          = [1],
        token_decoding_schedules= ['linear'],
        temps                   = [0.01],
        temp_schedules          = ['constant'],
        cfg_scales              = [2.0],
        cfg_schedules           = ['constant'],
        cfg_grow_conditioning   = True,
    )

    sd = {'tok_rgb@224': {'tensor': rgb_tokens}}
    sd = init_full_input_modality(sd, MODALITY_INFO, 'tok_rgb@224', device, eos_id=eos_id)
    sd = init_empty_target_modality(sd, MODALITY_INFO, 'tok_normal@224', 1, 196, device)

    with torch.no_grad():
        out = sampler.generate(sd, schedule, text_tokenizer=tok_text,
                               verbose=False, seed=42)

    return out['tok_normal@224']['tensor']   # (1, 196) long


def decode_normal_tokens(normal_tok, tokens, size=224):
    """Normal tokens (1, 196) → unit-normal map (H, W, 3) float32.

    The VQVAE decoder returns a reconstructed image in approximately [-1, 1]
    (same space as the normalized input to tokenize()). We renormalize to
    unit-length normal vectors after decoding.
    """
    from einops import rearrange
    img_tok = rearrange(tokens, 'b (nh nw) -> b nh nw', nh=14, nw=14)
    with torch.no_grad():
        out = normal_tok.decode_tokens(img_tok)   # (1, C, H, W) or (1, 3, H, W)

    # Handle possible class-logit outputs (shouldn't happen for normal, but safe)
    if out.shape[1] != 3:
        raise ValueError(f'Unexpected normal decoder output shape: {out.shape}')

    # Resize to target size if needed
    if out.shape[-1] != size:
        out = F.interpolate(out.float(), size=(size, size), mode='bilinear', align_corners=False)

    # Convert from image-space [-1, 1] to unit normal vectors
    normal = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()   # (H, W, 3)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (norm + 1e-8)
    return normal.astype(np.float32)


# ── GT normals from depth ─────────────────────────────────────────────────────

def depth_to_normal(depth: np.ndarray, size: int = 224) -> np.ndarray:
    """Compute surface normal from depth map (H, W) in metres → (H, W, 3).

    Uses NYUv2 intrinsics. Resizes to (size, size) using NEAREST on depth
    first (matches tokenization), then computes normals.
    """
    # Resize depth to eval resolution (nearest, same as tokenization)
    pil = Image.fromarray(depth)
    pil = pil.resize((size, size), Image.NEAREST)
    d   = np.array(pil, dtype=np.float32)

    H, W = d.shape
    # Scale intrinsics to match resized image
    scale_x = size / 640.0
    scale_y = size / 480.0
    fx = NYU_FX * scale_x
    fy = NYU_FY * scale_y
    cx = NYU_CX * scale_x
    cy = NYU_CY * scale_y

    u  = np.arange(W, dtype=np.float32)
    v  = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d

    dX_du = np.gradient(X, axis=1)
    dY_du = np.gradient(Y, axis=1)
    dZ_du = np.gradient(Z, axis=1)
    dX_dv = np.gradient(X, axis=0)
    dY_dv = np.gradient(Y, axis=0)
    dZ_dv = np.gradient(Z, axis=0)

    tu = np.stack([dX_du, dY_du, dZ_du], axis=-1)
    tv = np.stack([dX_dv, dY_dv, dZ_dv], axis=-1)

    normal = np.cross(tu, tv)
    norm   = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid  = (norm[..., 0] > 1e-8) & (d > 0)
    normal = np.where(valid[..., None], normal / (norm + 1e-8), 0.0)
    normal[~valid] = np.array([0.0, 0.0, 1.0])

    return normal.astype(np.float32), valid


# ── Metrics ───────────────────────────────────────────────────────────────────

def angle_error_deg(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel angle error in degrees for valid pixels."""
    dot = (pred * gt).sum(axis=-1)                     # (H, W)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))                 # (H, W)
    return angle[valid]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  Normal estimation eval — NYUv2 val (654 images)')
    print(f'  LoRA     : {args.lora_weights}')
    print(f'  Output   : {args.output_dir}')
    print('=' * 60)

    # ── Load normal VQVAE decoder ─────────────────────────────────────────────
    from fourm.vq.vqvae import VQVAE
    print('\nLoading normal VQVAE ...')
    normal_tok = VQVAE.from_pretrained(NORMAL_TOKENIZER_REPO, cache_dir=args.hf_cache)
    normal_tok = normal_tok.to(device).eval()

    # ── Load 4M + LoRA ────────────────────────────────────────────────────────
    fm = load_model(args, device, dtype)

    # ── Sample list ───────────────────────────────────────────────────────────
    rgb_dir   = Path(args.tokens_dir) / 'tok_rgb'
    depth_dir = Path(args.nyu_raw) / 'test' / 'depth'
    stems     = sorted(p.stem for p in rgb_dir.glob('*.npy'))
    if args.max_images > 0:
        stems = stems[:args.max_images]
    print(f'\n{len(stems)} val images to evaluate')

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_errors = []
    n_skip     = 0

    for stem in tqdm(stems, desc='Normal eval'):
        rgb_tok_path  = rgb_dir   / f'{stem}.npy'
        depth_path    = depth_dir / f'{stem}.npy'

        if not rgb_tok_path.exists() or not depth_path.exists():
            n_skip += 1
            continue

        # GT normal from depth
        depth_m = np.load(depth_path).astype(np.float32)
        if depth_m.max() == 0:
            n_skip += 1
            continue
        gt_normal, valid = depth_to_normal(depth_m, size=224)

        # Predict normal tokens
        rgb_tokens = torch.from_numpy(
            np.load(rgb_tok_path).astype(np.int64)
        ).unsqueeze(0).to(device)

        try:
            pred_tokens = generate_normal_tokens(fm, rgb_tokens, device, dtype)
            pred_normal = decode_normal_tokens(normal_tok, pred_tokens, size=224)
        except Exception as e:
            print(f'  [skip] {stem}: {e}')
            n_skip += 1
            continue

        errors = angle_error_deg(pred_normal, gt_normal, valid)
        if len(errors) > 0:
            all_errors.append(errors)

    if not all_errors:
        print('No valid predictions — check data paths.')
        return

    all_errors = np.concatenate(all_errors)
    mean_err   = float(np.mean(all_errors))
    within     = {
        'within_11_25': float(np.mean(all_errors < 11.25)),
        'within_22_5':  float(np.mean(all_errors < 22.5)),
        'within_30':    float(np.mean(all_errors < 30.0)),
    }

    print(f'\nMean angle error : {mean_err:.2f}°')
    print(f'Within 11.25°    : {100*within["within_11_25"]:.1f}%')
    print(f'Within 22.5°     : {100*within["within_22_5"]:.1f}%')
    print(f'Within 30°       : {100*within["within_30"]:.1f}%')
    print(f'Images evaluated : {len(stems) - n_skip}  |  Skipped: {n_skip}')

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        'run_name':        args.run_name,
        'lora_weights':    args.lora_weights,
        'n_images':        len(stems) - n_skip,
        'mean_angle_err':  mean_err,
        **within,
    }
    import json as json_mod
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json_mod.dump(summary, f, indent=2)

    txt = (
        f'Run              : {args.run_name}\n'
        f'Weights          : {args.lora_weights}\n'
        f'Images           : {len(stems) - n_skip}\n'
        f'Mean angle error : {mean_err:.2f}°\n'
        f'Within 11.25°    : {100*within["within_11_25"]:.1f}%\n'
        f'Within 22.5°     : {100*within["within_22_5"]:.1f}%\n'
        f'Within 30°       : {100*within["within_30"]:.1f}%\n'
    )
    with open(Path(args.output_dir) / 'summary.txt', 'w') as f:
        f.write(txt)
    print(txt)
    print(f'Results saved → {args.output_dir}')


if __name__ == '__main__':
    main()
