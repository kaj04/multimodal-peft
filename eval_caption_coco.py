#!/usr/bin/env python3
"""
eval_caption_coco.py
====================
Caption quality evaluation on COCO val2017 — official NLP metrics.

Runs RGB → caption generation with:
  - Baseline: 4M-21 XL (no LoRA)
  - LoRA:     caption fine-tuned model (run 20: r4/selfattn/drop01)

Metrics (standard COCO captioning evaluation):
  CIDEr    Consensus-based Image Description Evaluation    higher = better
  BLEU@4   4-gram BLEU                                     higher = better
  ROUGE_L  Longest common subsequence recall               higher = better
  METEOR   Unigram F-score with stemming/synonym matching  higher = better

Evaluated on the full COCO val2017 set (5000 images, never subsampled).
COCO train2017 was sampled to 10k for training (seed=42), val is always full.

Usage
-----
  # Full eval — baseline + caption LoRA (5000 images, ~4-5h on 1× A100):
  python eval_caption_coco.py \\
      --lora_weights /scratch-shared/tesr125338/output/caption/coco_rgb_r4_selfattn_drop01/lora_weights-best.pth \\
      --output_dir   /scratch-shared/tesr125338/output/eval_caption/coco_r4_selfattn_drop01

  # Smoke-test (20 images, ~3 min):
  python eval_caption_coco.py --max_images 20 ...

Output
------
  <output_dir>/
    results_baseline.json   COCO-format predictions (baseline)
    results_lora.json       COCO-format predictions (caption LoRA)
    metrics.json            CIDEr/BLEU/ROUGE/METEOR for both models
    summary.txt             human-readable comparison table
    config.json             exact config used
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── pip packages on scratch (pycocoevalcap, pycocotools, nltk) ───────────────
PIP_PACKAGES = '/scratch-shared/tesr125338/pip_packages'
NLTK_DATA    = '/scratch-shared/tesr125338/nltk_data'
if PIP_PACKAGES not in sys.path:
    sys.path.append(PIP_PACKAGES)

import nltk
nltk.data.path.append(NLTK_DATA)

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'ml-4m'))

from fourm.models.fm import FM
from fourm.models.generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_empty_target_modality,
    init_full_input_modality,
)
from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import RGBTransform
from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names

# ── Caption generation hyper-parameters ──────────────────────────────────────
CAPTION_GEN_PARAMS = {
    'tokens_per_target':       64,
    'autoregression_scheme':   'autoregressive',
    'decoding_steps':          64,
    'token_decoding_schedule': 'linear',
    'temp':                    0.01,   # near-greedy → deterministic eval
    'temp_schedule':           'constant',
    'cfg_scale':               2.0,
    'cfg_schedule':            'constant',
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Caption quality eval: baseline vs caption LoRA on full COCO val2017'
    )
    # Data — full val2017, never subsampled
    p.add_argument('--coco_root',
                   default='/scratch-shared/tesr125338/coco/val2017',
                   help='Path to COCO val2017 image directory (5000 images)')
    p.add_argument('--coco_ann',
                   default='/scratch-shared/tesr125338/coco/annotations/captions_val2017.json',
                   help='Path to captions_val2017.json')
    # Model
    p.add_argument('--fm', default='EPFL-VILAB/4M-21_XL')
    p.add_argument('--hf_cache', default='/scratch-shared/tesr125338/hf_cache')
    p.add_argument('--tok_text',
                   default='ml-4m/fourm/utils/tokenizer/trained/'
                           'text_tokenizer_4m_wordpiece_30k.json')
    p.add_argument('--lora_weights',
                   default='/scratch-shared/tesr125338/output/caption/'
                           'coco_rgb_r4_selfattn_drop01/lora_weights-best.pth',
                   help='Caption LoRA weights .pth — if omitted, only baseline is run')
    p.add_argument('--lora_rank',    type=int, default=4,
                   help='LoRA rank used during training (run 20 = 4)')
    p.add_argument('--lora_modules', default='selfattn',
                   choices=['selfattn', 'crossattn', 'attention', 'mlp', 'all'])
    # Evaluation
    p.add_argument('--max_images', type=int, default=0,
                   help='Cap number of images (0 = all 5000, standard COCO eval)')
    p.add_argument('--seed', type=int, default=0)
    # Output
    p.add_argument('--output_dir',
                   default='/scratch-shared/tesr125338/output/eval_caption/'
                           'coco_r4_selfattn_drop01')
    # Hardware
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype',  default='bfloat16', choices=['bfloat16', 'float32'])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def collect_coco_samples(coco_root, coco_ann, max_images):
    """Return list of dicts {image_id, image_path}."""
    with open(coco_ann) as f:
        ann = json.load(f)

    id2fname = {img['id']: img['file_name'] for img in ann['images']}
    samples = []
    for image_id, fname in sorted(id2fname.items()):
        path = os.path.join(coco_root, fname)
        if os.path.exists(path):
            samples.append({'image_id': image_id, 'image_path': path})

    if max_images and max_images > 0:
        samples = samples[:max_images]
    return samples


def load_rgb_tensor(image_path, device, dtype_torch):
    """RGB image → (1,3,224,224) ImageNet-normalised tensor."""
    rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
    img_pil = rgb_transform.load(image_path)
    img_pil = rgb_transform.preprocess(img_pil)
    s       = min(img_pil.size)
    img_pil = center_crop(img_pil, (s, s)).resize((224, 224), Image.BILINEAR)
    tensor  = rgb_transform.postprocess(img_pil).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype_torch)


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

def generate_caption_tokens(sampler, tok_text, rgb_tensor, device, seed):
    ntoks    = CAPTION_GEN_PARAMS['tokens_per_target']
    schedule = build_chained_generation_schedules(
        cond_domains            = ['rgb@224'],
        target_domains          = ['caption'],
        tokens_per_target       = [ntoks],
        autoregression_schemes  = [CAPTION_GEN_PARAMS['autoregression_scheme']],
        decoding_steps          = [CAPTION_GEN_PARAMS['decoding_steps']],
        token_decoding_schedules= [CAPTION_GEN_PARAMS['token_decoding_schedule']],
        temps                   = [CAPTION_GEN_PARAMS['temp']],
        temp_schedules          = [CAPTION_GEN_PARAMS['temp_schedule']],
        cfg_scales              = [CAPTION_GEN_PARAMS['cfg_scale']],
        cfg_schedules           = [CAPTION_GEN_PARAMS['cfg_schedule']],
        cfg_grow_conditioning   = True,
    )
    eos_id      = tok_text.token_to_id('[EOS]')
    sample_dict = {'rgb@224': {'tensor': rgb_tensor}}
    sample_dict = init_full_input_modality(
        sample_dict, MODALITY_INFO, 'rgb@224', device, eos_id=eos_id)
    sample_dict = init_empty_target_modality(
        sample_dict, MODALITY_INFO, 'caption', 1,
        CAPTION_GEN_PARAMS['tokens_per_target'], device)
    return sampler.generate(
        sample_dict, schedule,
        text_tokenizer=tok_text, verbose=False, seed=seed)


def decode_caption_string(out_dict, tok_text):
    tokens = out_dict['caption']['tensor']
    if tokens.ndim > 1:
        tokens = tokens[0]
    token_ids = tokens.cpu().tolist()
    eos_id = tok_text.token_to_id('[EOS]')
    if eos_id in token_ids:
        token_ids = token_ids[:token_ids.index(eos_id)]
    token_ids = [t for t in token_ids if t > 0]
    if not token_ids:
        return ''
    return tok_text.decode(token_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation with pycocoevalcap
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_captions(ann_file, predictions, image_ids):
    """
    Run official COCO captioning evaluation.

    Returns dict with CIDEr, Bleu_1-4, ROUGE_L, METEOR.
    SPICE is monkey-patched out (requires Java, not available on Snellius).
    """
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    try:
        from pycocoevalcap.spice.spice import Spice
        Spice.compute_score = lambda self, gts, res: (0.0, [0.0] * len(res))
        print('  [SPICE] monkey-patched — skipping Java dependency')
    except Exception as _e:
        print(f'  [SPICE] not found / already absent ({_e})')

    coco_gt   = COCO(ann_file)
    coco_dt   = coco_gt.loadRes(predictions)
    evaluator = COCOEvalCap(coco_gt, coco_dt)
    evaluator.params['image_id'] = list(image_ids)
    evaluator.evaluate()
    evaluator.eval.pop('SPICE', None)
    return evaluator.eval


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args, use_lora, device, dtype_torch):
    label = 'caption LoRA' if use_lora else 'baseline'
    print(f'  Loading FM ({label}) ...')
    fm = FM.from_pretrained(args.fm, cache_dir=args.hf_cache)

    if use_lora:
        target_mods = get_LoRA_module_names(args.lora_modules)
        print(f'    Injecting LoRA (rank={args.lora_rank}, modules={target_mods}) ...')
        inject_trainable_LoRA(fm, rank=args.lora_rank, scale=1.0,
                              target_replace_modules=target_mods)
        lora_sd = torch.load(args.lora_weights, map_location='cpu')
        missing, unexpected = fm.load_state_dict(lora_sd, strict=False)
        print(f'    Loaded {len(lora_sd)} LoRA tensors '
              f'(missing={len(missing)}, unexpected={len(unexpected)})')

    fm = fm.to(device=device, dtype=dtype_torch).eval()
    n_params = sum(p.numel() for p in fm.parameters()) / 1e6
    print(f'    FM ready ({n_params:.0f}M params)')
    return fm, GenerationSampler(fm)


# ─────────────────────────────────────────────────────────────────────────────
# Run inference on all samples
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(samples, sampler, tok_text, device, dtype_torch, args, tag):
    predictions = []
    n  = len(samples)
    t0 = time.time()

    for idx, s in enumerate(tqdm(samples, desc=f'  Generating [{tag}]')):
        if idx == 0 or (idx + 1) % 100 == 0 or (idx + 1) == n:
            elapsed = time.time() - t0
            rate    = elapsed / max(idx, 1)
            eta     = rate * (n - idx - 1)
            print(f'    [{idx+1:5d}/{n}]  elapsed={elapsed:6.0f}s  ETA={eta:6.0f}s',
                  flush=True)
        try:
            rgb_tensor = load_rgb_tensor(s['image_path'], device, dtype_torch)
            with torch.no_grad():
                out = generate_caption_tokens(sampler, tok_text, rgb_tensor, device,
                                              seed=args.seed + idx)
            caption = decode_caption_string(out, tok_text)
            predictions.append({'image_id': s['image_id'], 'caption': caption})
        except Exception as e:
            import traceback
            print(f'\n  ERROR image_id={s["image_id"]}: {e}')
            traceback.print_exc()
            predictions.append({'image_id': s['image_id'], 'caption': ''})

    elapsed_total = time.time() - t0
    print(f'  Done in {elapsed_total:.0f}s  ({elapsed_total/n:.1f}s/img)')
    return predictions, elapsed_total


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(args, metrics_base, metrics_lora, n_images, elapsed_base, elapsed_lora):
    metric_keys = ['CIDEr', 'Bleu_4', 'ROUGE_L', 'METEOR', 'Bleu_1']

    lines = [
        '',
        '=' * 75,
        '  Caption Quality Evaluation — COCO val2017 (full, 5000 images)',
        '=' * 75,
        f'  FM model     : {args.fm}',
        f'  LoRA weights : {args.lora_weights}',
        f'  LoRA config  : rank={args.lora_rank}, modules={args.lora_modules}',
        f'  N images     : {n_images}  (full COCO val2017 = 5000)',
        f'  Temp         : {CAPTION_GEN_PARAMS["temp"]}  (near-greedy)',
        f'  Max tokens   : {CAPTION_GEN_PARAMS["tokens_per_target"]}',
        '',
        '  METRICS  (↑ = higher is better)',
        '  ' + '-' * 62,
        f'  {"Metric":<12}  {"Baseline":>12}  {"Caption LoRA":>12}  {"Δ":>10}  {"Δ%":>8}',
        '  ' + '-' * 62,
    ]
    for k in metric_keys:
        b  = metrics_base.get(k, float('nan'))
        lo = metrics_lora.get(k, float('nan')) if metrics_lora else float('nan')
        d  = lo - b
        dp = 100 * d / max(abs(b), 1e-8)
        arrow = '↑' if d >= 0 else '↓'
        lines.append(f'  {k:<12}  {b:>12.4f}  {lo:>12.4f}  {d:>+9.4f}  {dp:>+6.1f}%{arrow}')
    lines += [
        '  ' + '-' * 62,
        '',
        f'  Baseline inference     : {elapsed_base:.0f}s  ({elapsed_base/n_images:.1f}s/img)',
    ]
    if metrics_lora:
        lines.append(
            f'  Caption LoRA inference : {elapsed_lora:.0f}s  ({elapsed_lora/n_images:.1f}s/img)')
    lines += [
        '',
        '  Note: CIDEr is the primary COCO captioning metric.',
        '        Training subset was 10k images (seed=42 sample of train2017).',
        '        Val is the full COCO val2017 — 5000 images, never subsampled.',
        '',
        '=' * 75,
        '',
    ]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-run validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_environment(args):
    errors = []

    if not os.path.isdir(args.coco_root):
        errors.append(f'COCO image dir not found: {args.coco_root}')
    if not os.path.isfile(args.coco_ann):
        errors.append(f'COCO annotation file not found: {args.coco_ann}')

    if args.lora_weights and not os.path.isfile(args.lora_weights):
        errors.append(f'LoRA weights not found: {args.lora_weights}')

    tok_path = args.tok_text
    if not os.path.isabs(tok_path):
        tok_path = os.path.join(_REPO_ROOT, tok_path)
    if not os.path.isfile(tok_path):
        errors.append(f'Text tokenizer not found: {tok_path}')

    missing_pkgs = []
    for pkg in ['pycocotools.coco', 'pycocoevalcap.eval', 'nltk']:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        errors.append(f'Missing Python packages: {missing_pkgs}')

    try:
        from pycocoevalcap.spice.spice import Spice
        Spice.compute_score = lambda self, gts, res: (0.0, [0.0] * len(res))
    except Exception:
        pass

    try:
        import nltk
        nltk.tokenize.word_tokenize('hello world')
    except Exception as e:
        errors.append(f'NLTK punkt not available: {e}')

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        test_file = os.path.join(args.output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.remove(test_file)
    except Exception as e:
        errors.append(f'Output dir not writeable ({args.output_dir}): {e}')

    if args.device == 'cuda' and not torch.cuda.is_available():
        errors.append('CUDA requested but torch.cuda.is_available() == False')

    print('\n── Pre-run validation ──────────────────────────────────────────────')
    if errors:
        print('  FAILED:')
        for e in errors:
            print(f'    ✗ {e}')
        print('───────────────────────────────────────────────────────────────────\n')
        sys.exit(1)
    else:
        print('  All checks passed ✓')
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'  GPU: {gpu_name}  ({gpu_mem:.1f} GB VRAM)')
        print('───────────────────────────────────────────────────────────────────\n')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    validate_environment(args)

    device      = torch.device(args.device)
    dtype_torch = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    config = {**vars(args), 'timestamp': datetime.now().isoformat()}
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print('=' * 75)
    print('  Caption Quality Eval — COCO val2017 (full, 5000 images)')
    print(f'  Caption LoRA: {args.lora_weights or "None (baseline only)"}')
    print(f'  Output: {args.output_dir}')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 75)

    # ── Text tokenizer ────────────────────────────────────────────────────────
    tok_text_path = args.tok_text
    if not os.path.isabs(tok_text_path):
        tok_text_path = os.path.join(_REPO_ROOT, tok_text_path)
    tok_text = Tokenizer.from_file(tok_text_path)
    print(f'  Text tokenizer ready  (vocab={tok_text.get_vocab_size()})')

    # ── Collect samples ───────────────────────────────────────────────────────
    max_img = args.max_images if args.max_images > 0 else None
    samples = collect_coco_samples(args.coco_root, args.coco_ann, max_img)
    print(f'  COCO val2017 samples: {len(samples)}  '
          f'({"full set" if len(samples) == 5000 else f"subset of 5000"})')
    image_ids = {s['image_id'] for s in samples}

    # ─────────────────────────────────────────────────────────────────────────
    # [1] BASELINE
    # ─────────────────────────────────────────────────────────────────────────
    print('\n>>> [1/2] Baseline (no LoRA)')
    fm_base, sampler_base = load_model(args, use_lora=False, device=device,
                                        dtype_torch=dtype_torch)
    preds_base, elapsed_base = run_inference(
        samples, sampler_base, tok_text, device, dtype_torch, args, tag='baseline')

    pred_base_path = os.path.join(args.output_dir, 'results_baseline.json')
    with open(pred_base_path, 'w') as f:
        json.dump(preds_base, f, indent=2)
    print(f'  Predictions saved → {pred_base_path}')

    print('  Running COCO captioning evaluation ...')
    metrics_base = evaluate_captions(args.coco_ann, preds_base, image_ids)
    print(f'  CIDEr={metrics_base.get("CIDEr", "?"):.4f}  '
          f'BLEU@4={metrics_base.get("Bleu_4", "?"):.4f}  '
          f'METEOR={metrics_base.get("METEOR", "?"):.4f}')

    del fm_base, sampler_base
    torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # [2] Caption LoRA
    # ─────────────────────────────────────────────────────────────────────────
    metrics_lora = None
    elapsed_lora = 0.0
    preds_lora   = []

    if args.lora_weights:
        print('\n>>> [2/2] Caption LoRA (run 20: r4/selfattn/drop01)')
        fm_lora, sampler_lora = load_model(args, use_lora=True, device=device,
                                            dtype_torch=dtype_torch)
        preds_lora, elapsed_lora = run_inference(
            samples, sampler_lora, tok_text, device, dtype_torch, args, tag='caption_lora')

        pred_lora_path = os.path.join(args.output_dir, 'results_lora.json')
        with open(pred_lora_path, 'w') as f:
            json.dump(preds_lora, f, indent=2)
        print(f'  Predictions saved → {pred_lora_path}')

        print('  Running COCO captioning evaluation ...')
        metrics_lora = evaluate_captions(args.coco_ann, preds_lora, image_ids)
        print(f'  CIDEr={metrics_lora.get("CIDEr", "?"):.4f}  '
              f'BLEU@4={metrics_lora.get("Bleu_4", "?"):.4f}  '
              f'METEOR={metrics_lora.get("METEOR", "?"):.4f}')

        del fm_lora, sampler_lora
        torch.cuda.empty_cache()
    else:
        print('\n>>> [2/2] Caption LoRA — SKIPPED (no --lora_weights provided)')

    # ─────────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────────
    metrics_out = {
        'n_images':            len(samples),
        'baseline':            metrics_base,
        'caption_lora':        metrics_lora,
        'lora_weights':        args.lora_weights,
        'lora_rank':           args.lora_rank,
        'lora_modules':        args.lora_modules,
        'elapsed_baseline_s':  elapsed_base,
        'elapsed_lora_s':      elapsed_lora,
        'timestamp':           datetime.now().isoformat(),
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)

    summary = build_summary(args, metrics_base, metrics_lora,
                             len(samples), elapsed_base, elapsed_lora)
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(summary)
    print(f'  metrics.json → {metrics_path}')
    print(f'  summary.txt  → {summary_path}')


if __name__ == '__main__':
    main()
