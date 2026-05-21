#!/usr/bin/env python3
"""
inspect_tokens.py
Quick diagnostic script to check token file statistics and identify
the padding issue in det/caption training data.

Usage (run on Snellius, no GPU needed):
    python scripts/inspect_tokens.py
"""
import glob
import json
import numpy as np
from pathlib import Path

TOKENS_ROOT = '/scratch-shared/tesr125338/coco_tokens'
TRAIN_LOG_DET = '/scratch-shared/tesr125338/output/det/coco_rgb_r8_attention/log.txt'
TRAIN_LOG_CAP = '/scratch-shared/tesr125338/output/caption/coco_rgb_r8_attention_drop01/log.txt'


def inspect_folder(folder, n_samples=200, label=''):
    files = sorted(glob.glob(f'{folder}/*.npy'))
    if not files:
        print(f'  [WARN] No .npy files found in {folder}')
        return
    files = files[:n_samples]

    arrays = [np.load(f).astype(np.int32) for f in files]
    all_tokens = np.concatenate(arrays)
    seq_len = arrays[0].shape[0]

    padding_fracs = [(a == 0).mean() for a in arrays]
    eos_fracs     = [(a == 2).mean() for a in arrays]   # [EOS] is typically id=2

    # Non-zero token range
    nonzero = all_tokens[all_tokens != 0]

    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    print(f'  Files inspected       : {len(files)}')
    print(f'  Sequence length       : {seq_len}')
    print(f'  Avg padding (id=0)    : {np.mean(padding_fracs)*100:.1f}%  (range {np.min(padding_fracs)*100:.0f}–{np.max(padding_fracs)*100:.0f}%)')
    print(f'  Avg EOS fraction      : {np.mean(eos_fracs)*100:.1f}%')
    print(f'  Non-zero token stats  : min={nonzero.min() if len(nonzero) else "N/A"}  max={nonzero.max() if len(nonzero) else "N/A"}  unique={len(np.unique(nonzero)) if len(nonzero) else 0}')
    print(f'  Pure-padding images   : {sum(1 for a in arrays if (a == 0).sum() == seq_len - 1)} / {len(files)}  (only EOS + pad)')

    # Distribution of actual content tokens per sequence
    content_lens = [int((a != 0).sum()) for a in arrays]
    print(f'  Content tokens/seq    : mean={np.mean(content_lens):.1f}  median={np.median(content_lens):.0f}  min={min(content_lens)}  max={max(content_lens)}')


def inspect_training_log(log_path, label=''):
    print(f'\n{"="*60}')
    print(f'  Training log: {label}')
    print(f'{"="*60}')
    p = Path(log_path)
    if not p.exists():
        print(f'  [WARN] Log not found: {log_path}')
        return

    lines = [json.loads(l) for l in p.read_text().strip().splitlines() if l.strip()]

    print(f'  {'Epoch':>6}  {'train/loss':>12}  {'val/loss':>12}  {'val/token_acc':>14}')
    print(f'  {"-"*50}')
    for log in lines[:30]:   # first 30 epochs
        epoch      = log.get('epoch', '?')
        train_loss = log.get('train/loss', log.get('train/loss_caption', log.get('train/loss_det', '—')))
        val_loss   = log.get('val/loss', '—')
        val_acc    = log.get('val/token_acc', '—')
        tl = f'{train_loss:.4f}' if isinstance(train_loss, float) else str(train_loss)
        vl = f'{val_loss:.4f}'   if isinstance(val_loss,   float) else str(val_loss)
        va = f'{val_acc:.4f}'    if isinstance(val_acc,    float) else str(val_acc)
        print(f'  {epoch:>6}  {tl:>12}  {vl:>12}  {va:>14}')


def main():
    print('4M-PEFT Token Inspection — det and caption padding analysis')

    # Det tokens
    inspect_folder(f'{TOKENS_ROOT}/train/det', label='DETECTION — train/det/')
    inspect_folder(f'{TOKENS_ROOT}/val/det',   label='DETECTION — val/det/')

    # Caption tokens
    inspect_folder(f'{TOKENS_ROOT}/train/caption', label='CAPTION — train/caption/')
    inspect_folder(f'{TOKENS_ROOT}/val/caption',   label='CAPTION — val/caption/')

    # Depth for comparison (should have ~0% padding — it's a dense VQ map)
    inspect_folder(f'{TOKENS_ROOT}/val/tok_depth', label='DEPTH — val/tok_depth/ (control, expect ~0% pad)')

    # Training curves
    inspect_training_log(TRAIN_LOG_DET, label='det (run 36)')
    inspect_training_log(TRAIN_LOG_CAP, label='caption (run 23)')


if __name__ == '__main__':
    main()
