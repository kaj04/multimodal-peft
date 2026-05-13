#!/usr/bin/env python3
# eval_semseg_coco.py
#
# Pixel-space semantic segmentation evaluation on COCO val2017.
# Metric: mean Intersection-over-Union (mIoU) over 133 COCO panoptic classes.
#
# The model generates tok_semseg tokens from RGB input, decodes them through
# the VQVAE to get per-pixel class logits, then computes mIoU against the
# ground-truth panoptic annotation.
#
# Usage:
#   python eval_semseg_coco.py \
#       --lora_weights /path/to/lora_weights-best.pth \
#       --lora_rank 8 --lora_modules attention \
#       --coco_dir /scratch-shared/tesr125338/coco \
#       --tokens_dir /scratch-shared/tesr125338/coco_tokens/val \
#       --output_dir /scratch-shared/tesr125338/output/eval_semseg/run26 \
#       --device cuda --dtype bfloat16

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

SEMSEG_TOKENIZER_REPO = 'EPFL-VILAB/4M_tokenizers_semseg_4k_224-448'
FM_MODEL              = 'EPFL-VILAB/4M-21_XL'
COCO_NUM_CLASSES      = 134   # 0 = background/ignored, 1-133 = semantic classes


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Evaluate semseg LoRA on COCO val2017')
    p.add_argument('--lora_weights', required=True,
                   help='Path to lora_weights-best.pth')
    p.add_argument('--lora_rank',    default=8, type=int)
    p.add_argument('--lora_modules', default='attention', type=str)
    p.add_argument('--coco_dir',     required=True,
                   help='Root of COCO (contains annotations/panoptic_val2017.{zip,json})')
    p.add_argument('--tokens_dir',   required=True,
                   help='Root of tokenized val data (contains tok_rgb/)')
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--run_name',     default='semseg_eval')
    p.add_argument('--hf_cache',     default=None)
    p.add_argument('--max_images',   default=0, type=int,
                   help='Max images to evaluate (0 = all 5000)')
    p.add_argument('--device',       default='cuda')
    p.add_argument('--dtype',        default='bfloat16',
                   choices=['float32', 'bfloat16'])
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(args, device, dtype):
    from fourm.models.fm import FM
    from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names
    from lora_dropout import add_lora_dropout

    print(f'Loading FM: {FM_MODEL}')
    fm = FM.from_pretrained(FM_MODEL, cache_dir=args.hf_cache)

    target_mods = get_LoRA_module_names(args.lora_modules)
    inject_trainable_LoRA(fm, rank=args.lora_rank, scale=1.0,
                          target_replace_modules=target_mods)

    lora_sd = torch.load(args.lora_weights, map_location='cpu')
    missing, unexpected = fm.load_state_dict(lora_sd, strict=False)
    print(f'  Loaded {len(lora_sd)} LoRA tensors '
          f'(missing={len(missing)}, unexpected={len(unexpected)})')

    fm = fm.to(device=device, dtype=dtype).eval()
    return fm


# ── Generation helpers ────────────────────────────────────────────────────────

def generate_semseg_tokens(fm, rgb_tokens, device, dtype):
    """RGB tokens (B, 196) → predicted semseg tokens (B, 196)."""
    from fourm.models.generate import (
        GenerationSampler, build_chained_generation_schedules,
        init_empty_target_modality, init_full_input_modality,
    )
    from fourm.data.modality_info import MODALITY_INFO
    from tokenizers import Tokenizer

    tok_text_path = str(REPO_ROOT / 'ml-4m' / 'fourm' / 'utils' / 'tokenizer' /
                        'trained' / 'text_tokenizer_4m_wordpiece_30k.json')
    tok_text = Tokenizer.from_file(tok_text_path)
    eos_id   = tok_text.token_to_id('[EOS]')

    sampler  = GenerationSampler(fm)
    schedule = build_chained_generation_schedules(
        cond_domains            = ['tok_rgb@224'],
        target_domains          = ['tok_semseg@224'],
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
    sd = init_empty_target_modality(sd, MODALITY_INFO, 'tok_semseg@224', 1, 196, device)

    with torch.no_grad():
        out = sampler.generate(sd, schedule, text_tokenizer=tok_text,
                               verbose=False, seed=42)

    return out['tok_semseg@224']['tensor']   # (1, 196) long


def decode_semseg_tokens(semseg_tok, tokens):
    """Semseg tokens (1, 196) → class prediction map (224, 224) int."""
    from einops import rearrange
    img_tok = rearrange(tokens, 'b (nh nw) -> b nh nw', nh=14, nw=14)
    with torch.no_grad():
        logits = semseg_tok.decode_tokens(img_tok)   # (1, C, H, W)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)
    return pred.astype(np.int32)


# ── Ground-truth loading ──────────────────────────────────────────────────────

def load_gt_semseg(zip_file, stem_to_ann, pan_dir, stem, size=224):
    """Load GT panoptic PNG from zip → (224, 224) int32 category_id map."""
    png_name = f'{pan_dir}/{stem}.png'
    try:
        raw = zip_file.read(png_name)
    except KeyError:
        return None

    img = Image.open(io.BytesIO(raw)).convert('RGB')
    pan_arr = np.array(img, dtype=np.int32)
    seg_id_map = (pan_arr[:, :, 0]
                  + pan_arr[:, :, 1] * 256
                  + pan_arr[:, :, 2] * 65536)

    ann = stem_to_ann.get(stem)
    if ann is None:
        return None

    cat_map = np.zeros(seg_id_map.shape, dtype=np.int32)
    for seg in ann['segments_info']:
        cat_map[seg_id_map == seg['id']] = seg['category_id']

    # Center-crop + resize (nearest) to 224×224 — same as tokenisation
    h, w = cat_map.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    cat_map = cat_map[top:top + s, left:left + s]
    img_p = Image.fromarray(cat_map.astype(np.int32), mode='I')
    img_p = img_p.resize((size, size), Image.NEAREST)
    return np.array(img_p, dtype=np.int32)


# ── mIoU computation ──────────────────────────────────────────────────────────

class IoUMeter:
    """Accumulates per-class intersection and union for mIoU."""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = np.zeros(num_classes, dtype=np.int64)
        self.union        = np.zeros(num_classes, dtype=np.int64)

    def update(self, pred, gt):
        """pred, gt: (H, W) int arrays with values in [0, num_classes-1]."""
        for c in range(self.num_classes):
            p = (pred == c)
            g = (gt   == c)
            self.intersection[c] += (p & g).sum()
            self.union[c]        += (p | g).sum()

    def mean_iou(self, ignore_background=True):
        """Compute mIoU over classes that appear in the GT."""
        iou_per_class = []
        start = 1 if ignore_background else 0   # skip class 0 (background)
        for c in range(start, self.num_classes):
            if self.union[c] > 0:
                iou_per_class.append(self.intersection[c] / self.union[c])
        return float(np.mean(iou_per_class)) if iou_per_class else 0.0

    def per_class_iou(self):
        iou = {}
        for c in range(1, self.num_classes):
            if self.union[c] > 0:
                iou[c] = float(self.intersection[c] / self.union[c])
        return iou


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  Semseg eval — COCO val2017')
    print(f'  LoRA     : {args.lora_weights}')
    print(f'  COCO dir : {args.coco_dir}')
    print(f'  Output   : {args.output_dir}')
    print('=' * 60)

    # ── Load semseg VQVAE decoder ─────────────────────────────────────────────
    from fourm.vq.vqvae import VQVAE
    print('\nLoading semseg VQVAE ...')
    semseg_tok = VQVAE.from_pretrained(SEMSEG_TOKENIZER_REPO,
                                       cache_dir=args.hf_cache)
    semseg_tok = semseg_tok.to(device).eval()

    # ── Load 4M + LoRA ────────────────────────────────────────────────────────
    fm = load_model(args, device, dtype)

    # ── Build sample list from existing tok_rgb stems ─────────────────────────
    rgb_dir = Path(args.tokens_dir) / 'tok_rgb'
    stems   = sorted(p.stem for p in rgb_dir.glob('*.npy'))
    if args.max_images > 0:
        stems = stems[:args.max_images]
    print(f'\n{len(stems)} val images to evaluate')

    # ── Load panoptic GT (from zip) ───────────────────────────────────────────
    ann_dir  = Path(args.coco_dir) / 'annotations'
    zip_path = ann_dir / 'panoptic_val2017.zip'
    json_path = ann_dir / 'panoptic_val2017.json'
    with open(json_path) as f:
        pan_data = json.load(f)
    stem_to_ann = {Path(a['file_name']).stem: a for a in pan_data['annotations']}

    # ── Evaluation loop ───────────────────────────────────────────────────────
    meter    = IoUMeter(COCO_NUM_CLASSES)
    results  = []
    n_skip   = 0

    with zipfile.ZipFile(zip_path) as zf:
        for stem in tqdm(stems, desc='Eval semseg'):
            # Load RGB tokens
            rgb_tok_path = rgb_dir / f'{stem}.npy'
            if not rgb_tok_path.exists():
                n_skip += 1
                continue
            rgb_tokens = torch.from_numpy(
                np.load(rgb_tok_path).astype(np.int64)
            ).unsqueeze(0).to(device)

            # Load GT semseg
            gt = load_gt_semseg(zf, stem_to_ann, 'panoptic_val2017', stem)
            if gt is None:
                n_skip += 1
                continue

            # Generate semseg prediction
            pred_tokens = generate_semseg_tokens(fm, rgb_tokens, device, dtype)
            pred = decode_semseg_tokens(semseg_tok, pred_tokens)

            # After shift_idx_by_one during tokenisation, the decoded class IDs
            # are shifted: pred value k corresponds to category_id k-1.
            # Undo the shift to align with GT (which uses category_ids 1-133).
            # Background (pred=0 after undo) is ignored in mIoU.
            # Note: for evaluation we compare pred (shifted) vs gt (1-133) directly
            # since the tokenizer was trained with shifted values and decodes them back
            # in shifted space. We simply ignore the boundary:
            #   gt class c → pred should output c+1 (shifted) → undo → c
            pred_unshifted = np.clip(pred - 1, 0, COCO_NUM_CLASSES - 1)

            meter.update(pred_unshifted, gt)
            results.append({'stem': stem})

    miou = meter.mean_iou(ignore_background=True)
    per_class = meter.per_class_iou()

    print(f'\nmIoU = {miou:.4f}  ({100*miou:.2f}%)')
    print(f'Images evaluated: {len(results)}  |  Skipped: {n_skip}')

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        'run_name':        args.run_name,
        'lora_weights':    args.lora_weights,
        'lora_rank':       args.lora_rank,
        'lora_modules':    args.lora_modules,
        'n_images':        len(results),
        'mIoU':            miou,
        'per_class_iou':   per_class,
    }

    import json as json_mod
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json_mod.dump(summary, f, indent=2)

    summary_txt = (
        f'Run      : {args.run_name}\n'
        f'Weights  : {args.lora_weights}\n'
        f'Images   : {len(results)}\n'
        f'mIoU     : {miou:.4f} ({100*miou:.2f}%)\n'
    )
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(summary_txt)
    print(summary_txt)
    print(f'Results saved → {args.output_dir}')


if __name__ == '__main__':
    main()
