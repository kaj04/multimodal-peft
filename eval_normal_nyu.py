#!/usr/bin/env python3
# eval_normal_nyu.py
#
# Surface-normal evaluation on NYUv2 val (Eigen split, 654 images).
#
# NOTE: the normal VQVAE tokenizer (EPFL-VILAB/4M_tokenizers_normal_8k_224-448)
# uses a 'unet_patched' decoder type that is not implemented in the public ml-4m
# codebase.  We therefore evaluate at the **token level**:
#
#   token_acc  — fraction of the 196 spatial tokens where the model's prediction
#                exactly matches the GT token from the pre-tokenized val set.
#                Range [0, 1], higher is better.
#                Random baseline ≈ 1/8192 ≈ 0.00012.
#
# GT tokens are loaded directly from nyu_tokens_labeled/val/tok_normal/ (created
# by prepare_nyu_normals.py / 33_tokenize_nyu_normals.sh).
# No VQVAE decoder is needed.
#
# Usage:
#   python eval_normal_nyu.py \
#       --lora_weights /path/to/lora_weights-best.pth \
#       --lora_rank 8 --lora_modules attention \
#       --tokens_dir /scratch-shared/tesr125338/nyu_tokens_labeled/val \
#       --output_dir /scratch-shared/tesr125338/output/eval_normal/run34 \
#       --device cuda --dtype bfloat16

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

FM_MODEL = 'EPFL-VILAB/4M-21_XL'


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='NYUv2 normal token-accuracy eval')
    p.add_argument('--lora_weights', default=None)
    p.add_argument('--lora_rank',    default=8, type=int)
    p.add_argument('--lora_modules', default='attention', type=str)
    p.add_argument('--nyu_raw',      default=None,
                   help='Unused (kept for CLI compatibility with eval_all.py)')
    p.add_argument('--tokens_dir',   required=True,
                   help='nyu_tokens_labeled/val (contains tok_rgb/ and tok_normal/)')
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  Normal eval — NYUv2 val (token accuracy, 654 images)')
    print(f'  LoRA     : {args.lora_weights}')
    print(f'  Output   : {args.output_dir}')
    print('  Metric   : token_acc (exact match vs GT tok_normal, range [0,1])')
    print('=' * 60)

    # ── Load 4M + LoRA ────────────────────────────────────────────────────────
    fm = load_model(args, device, dtype)

    # ── Sample list ───────────────────────────────────────────────────────────
    rgb_dir    = Path(args.tokens_dir) / 'tok_rgb'
    normal_dir = Path(args.tokens_dir) / 'tok_normal'

    stems = sorted(p.stem for p in rgb_dir.glob('*.npy'))
    if args.max_images > 0:
        stems = stems[:args.max_images]
    print(f'\n{len(stems)} val images to evaluate')

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_matches = []   # list of per-image token accuracy floats
    n_skip      = 0

    for stem in tqdm(stems, desc='Normal token eval'):
        rgb_tok_path    = rgb_dir    / f'{stem}.npy'
        normal_tok_path = normal_dir / f'{stem}.npy'

        if not rgb_tok_path.exists() or not normal_tok_path.exists():
            n_skip += 1
            continue

        # GT tokens (int16 on disk → int64)
        gt_tokens = torch.from_numpy(
            np.load(normal_tok_path).astype(np.int64)
        ).to(device)   # (196,)

        # Predicted tokens
        rgb_tokens = torch.from_numpy(
            np.load(rgb_tok_path).astype(np.int64)
        ).unsqueeze(0).to(device)   # (1, 196)

        try:
            pred_tokens = generate_normal_tokens(fm, rgb_tokens, device, dtype)
            pred_tokens = pred_tokens.squeeze(0)   # (196,)
        except Exception as e:
            print(f'  [skip] {stem}: {e}')
            n_skip += 1
            continue

        # Token accuracy: fraction of 196 tokens that match exactly
        acc = float((pred_tokens == gt_tokens).float().mean().item())
        all_matches.append(acc)

    if not all_matches:
        print('No valid predictions — check data paths.')
        return

    token_acc = float(np.mean(all_matches))

    print(f'\nToken accuracy : {token_acc:.4f}  ({100*token_acc:.2f}%)')
    print(f'Images evaluated : {len(all_matches)}  |  Skipped: {n_skip}')
    print(f'(random baseline ≈ {1/8192:.5f}  for 8k codebook)')

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        'run_name':     args.run_name,
        'lora_weights': args.lora_weights,
        'n_images':     len(all_matches),
        'n_skip':       n_skip,
        'token_acc':    round(token_acc, 6),
        'metric_note':  'Exact token match vs GT tok_normal (VQVAE decoder unavailable)',
    }
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    txt = (
        f'Run            : {args.run_name}\n'
        f'Weights        : {args.lora_weights}\n'
        f'Images         : {len(all_matches)}\n'
        f'Token accuracy : {token_acc:.4f}  ({100*token_acc:.2f}%)\n'
        f'(random ≈ {1/8192:.5f})\n'
    )
    with open(Path(args.output_dir) / 'summary.txt', 'w') as f:
        f.write(txt)
    print(txt)
    print(f'Results saved → {args.output_dir}')


if __name__ == '__main__':
    main()
