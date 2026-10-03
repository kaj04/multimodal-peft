#!/usr/bin/env python3
# eval_det_coco.py
#
# Object-detection evaluation on COCO val2017.
# Metric: AP@IoU=0.50 (AP50) and AP@IoU=0.50:0.95 (AP) via pycocotools.
#
# The model generates a det token sequence from RGB input, the sequence is
# decoded to a string like "v0=45 v1=120 v2=300 v3=400 dog v0=10 ...",
# bounding boxes are parsed, and COCO-standard evaluation is run.
#
# Confidence score: rank-based (1/(rank+1)) matching 4M's dist-to-origin
# sorting — earlier bboxes in the sequence are closer to the image origin
# and are treated as higher-confidence predictions.
#
# Usage:
#   python eval_det_coco.py \
#       --lora_weights /path/to/lora_weights-best.pth \
#       --lora_rank 8 --lora_modules attention \
#       --coco_dir /scratch-shared/tesr125338/coco \
#       --tokens_dir /scratch-shared/tesr125338/coco_tokens/val \
#       --output_dir /scratch-shared/tesr125338/output/eval_det/run34 \
#       --device cuda --dtype bfloat16

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

FM_MODEL = 'EPFL-VILAB/4M-21_XL'
COORD_BINS = 1000


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='COCO detection eval — AP50 / AP')
    p.add_argument('--lora_weights', default=None,
                   help='Path to lora_weights-best.pth (None = baseline, no LoRA)')
    p.add_argument('--lora_rank',    default=8, type=int)
    p.add_argument('--lora_modules', default='attention', type=str)
    p.add_argument('--coco_dir',     required=True,
                   help='COCO root (contains annotations/instances_val2017.json)')
    p.add_argument('--tokens_dir',   required=True,
                   help='Tokenized val dir (contains tok_rgb/)')
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--run_name',     default='det_eval')
    p.add_argument('--hf_cache',     default=None)
    p.add_argument('--max_images',   default=0, type=int,
                   help='Max images (0 = all 5000)')
    p.add_argument('--device',       default='cuda')
    p.add_argument('--dtype',        default='bfloat16',
                   choices=['float32', 'bfloat16'])
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(args, device, dtype):
    from fourm.models.fm import FM
    from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names
    from lora_dropout import add_lora_dropout

    fm = FM.from_pretrained(FM_MODEL, cache_dir=args.hf_cache)

    if args.lora_weights:
        target_mods = get_LoRA_module_names(args.lora_modules)
        inject_trainable_LoRA(fm, rank=args.lora_rank, scale=1.0,
                              target_replace_modules=target_mods)
        if args.lora_dropout > 0 if hasattr(args, 'lora_dropout') else False:
            add_lora_dropout(fm, p=args.lora_dropout)
        lora_sd = torch.load(args.lora_weights, map_location='cpu')
        missing, unexpected = fm.load_state_dict(lora_sd, strict=False)
        print(f'  Loaded {len(lora_sd)} LoRA tensors '
              f'(missing={len(missing)}, unexpected={len(unexpected)})')

    return fm.to(device=device, dtype=dtype).eval()


# ── Generation ────────────────────────────────────────────────────────────────

def generate_det_tokens(fm, rgb_tokens, tok_text, device, dtype):
    """RGB tokens (1, 196) → det token sequence (1, 256)."""
    from fourm.models.generate import (
        GenerationSampler, build_chained_generation_schedules,
        init_empty_target_modality, init_full_input_modality,
    )
    from fourm.data.modality_info import MODALITY_INFO

    eos_id = tok_text.token_to_id('[EOS]')
    sampler = GenerationSampler(fm)
    schedule = build_chained_generation_schedules(
        cond_domains            = ['tok_rgb@224'],
        target_domains          = ['det'],
        tokens_per_target       = [256],
        autoregression_schemes  = ['autoregressive'],
        decoding_steps          = [256],
        token_decoding_schedules= ['linear'],
        temps                   = [0.01],
        temp_schedules          = ['constant'],
        cfg_scales              = [2.0],
        cfg_schedules           = ['constant'],
        cfg_grow_conditioning   = True,
    )

    sd = {'tok_rgb@224': {'tensor': rgb_tokens}}
    sd = init_full_input_modality(sd, MODALITY_INFO, 'tok_rgb@224', device, eos_id=eos_id)
    sd = init_empty_target_modality(sd, MODALITY_INFO, 'det', 1, 256, device)

    with torch.no_grad():
        out = sampler.generate(sd, schedule, text_tokenizer=tok_text,
                               verbose=False, seed=42)

    return out['det']['tensor']   # (1, 256) long


# ── Parsing ───────────────────────────────────────────────────────────────────

# Matches one detection: "v0=Q v1=Q v2=Q v3=Q class_name [class continues]"
_DET_RE = re.compile(
    r'v0=(\d+)\s+v1=(\d+)\s+v2=(\d+)\s+v3=(\d+)\s+(.*?)(?=\s*v0=\d|\s*\[EOS\]|$)',
    re.DOTALL,
)


def parse_det_string(text, name_to_id, img_w, img_h):
    """Parse a 4M det string → list of COCO prediction dicts (without image_id)."""
    detections = []
    for rank, m in enumerate(_DET_RE.finditer(text)):
        xmin_q, ymin_q, xmax_q, ymax_q = int(m.group(1)), int(m.group(2)), \
                                           int(m.group(3)), int(m.group(4))
        cls = m.group(5).strip()
        if cls not in name_to_id:
            continue
        xmin = xmin_q / (COORD_BINS - 1) * img_w
        ymin = ymin_q / (COORD_BINS - 1) * img_h
        xmax = xmax_q / (COORD_BINS - 1) * img_w
        ymax = ymax_q / (COORD_BINS - 1) * img_h
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        if w < 1 or h < 1:
            continue
        detections.append({
            'bbox':        [xmin, ymin, w, h],
            'category_id': name_to_id[cls],
            'score':       1.0 / (rank + 1),   # rank-based confidence
        })
    return detections


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  Detection eval — COCO val2017')
    print(f'  LoRA     : {args.lora_weights}')
    print(f'  Output   : {args.output_dir}')
    print('=' * 60)

    # ── Load annotation ───────────────────────────────────────────────────────
    ann_path = Path(args.coco_dir) / 'annotations' / 'instances_val2017.json'
    with open(ann_path) as f:
        instances = json.load(f)

    # name → COCO category_id  (e.g. 'dog' → 18)
    name_to_id = {c['name']: c['id'] for c in instances['categories']}
    # image_id → (width, height)
    id_to_size = {img['id']: (img['width'], img['height'])
                  for img in instances['images']}

    # ── Load text tokenizer ───────────────────────────────────────────────────
    from tokenizers import Tokenizer
    tok_path = str(REPO_ROOT / 'ml-4m' / 'fourm' / 'utils' / 'tokenizer' /
                   'trained' / 'text_tokenizer_4m_wordpiece_30k.json')
    tok_text = Tokenizer.from_file(tok_path)

    # ── Load model ────────────────────────────────────────────────────────────
    fm = load_model(args, device, dtype)

    # ── Sample list ───────────────────────────────────────────────────────────
    rgb_dir = Path(args.tokens_dir) / 'tok_rgb'
    stems   = sorted(p.stem for p in rgb_dir.glob('*.npy'))
    if args.max_images > 0:
        stems = stems[:args.max_images]
    print(f'\n{len(stems)} val images to evaluate')

    # ── Generation + parsing loop ─────────────────────────────────────────────
    all_preds = []
    n_skip    = 0

    for stem in tqdm(stems, desc='Det eval'):
        rgb_tok_path = rgb_dir / f'{stem}.npy'
        if not rgb_tok_path.exists():
            n_skip += 1
            continue

        image_id = int(stem)
        if image_id not in id_to_size:
            n_skip += 1
            continue

        img_w, img_h = id_to_size[image_id]

        rgb_tokens = torch.from_numpy(
            np.load(rgb_tok_path).astype(np.int64)
        ).unsqueeze(0).to(device)

        try:
            det_tokens = generate_det_tokens(fm, rgb_tokens, tok_text, device, dtype)
        except Exception as e:
            print(f'  [skip] {stem}: generation failed — {e}')
            n_skip += 1
            continue

        # Decode token IDs → string using id_to_token (tok_text.decode() returns
        # empty for detection tokens because they are not standard wordpiece subwords)
        ids = det_tokens.squeeze(0).cpu().tolist()
        parts = [tok_text.id_to_token(i) for i in ids
                 if i > 0 and tok_text.id_to_token(i) is not None]
        text = ' '.join(parts)

        dets = parse_det_string(text, name_to_id, img_w, img_h)
        for d in dets:
            all_preds.append({'image_id': image_id, **d})

    print(f'\nTotal predictions: {len(all_preds)}  |  Skipped images: {n_skip}')

    # ── COCO eval ─────────────────────────────────────────────────────────────
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import io, contextlib

        coco_gt = COCO(str(ann_path))

        if all_preds:
            coco_dt = coco_gt.loadRes(all_preds)
        else:
            print('  Warning: no predictions — returning zeros')
            metrics = {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                       'APs': 0.0, 'APm': 0.0, 'APl': 0.0}
            _save_results(args, metrics, len(stems) - n_skip, n_skip)
            return

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        print(buf.getvalue())

        stats = coco_eval.stats
        metrics = {
            'AP':   float(stats[0]),
            'AP50': float(stats[1]),
            'AP75': float(stats[2]),
            'APs':  float(stats[3]),
            'APm':  float(stats[4]),
            'APl':  float(stats[5]),
        }

    except ImportError:
        print('  pycocotools not found — saving raw predictions only')
        metrics = {'AP': -1, 'AP50': -1}

    _save_results(args, metrics, len(stems) - n_skip, n_skip)


def _save_results(args, metrics, n_images, n_skip):
    summary = {
        'run_name':     args.run_name,
        'lora_weights': args.lora_weights,
        'n_images':     n_images,
        'n_skip':       n_skip,
        **metrics,
    }
    out = Path(args.output_dir)
    with open(out / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    txt = (
        f'Run      : {args.run_name}\n'
        f'Weights  : {args.lora_weights}\n'
        f'Images   : {n_images}\n'
        f'AP50     : {metrics.get("AP50", -1):.4f}\n'
        f'AP       : {metrics.get("AP",   -1):.4f}\n'
    )
    with open(out / 'summary.txt', 'w') as f:
        f.write(txt)
    print(txt)
    print(f'Results saved → {args.output_dir}')


if __name__ == '__main__':
    main()
