#!/usr/bin/env python3
"""
anole/tests/01_test_inference.py

Tests all 4 capabilities of Anole-7b-v0.1, saving every output to disk:
  1. Text generation          → text_generation.txt
  2. Image captioning / VQA   → image_understanding.txt
  3. Text-to-image            → text2image/*.png
  4. Interleaved generation   → interleaved/*.txt + interleaved/*.png

Usage (on Snellius GPU node, from ~/anole/):
    module load 2023 && module load Anaconda3/2023.07-2
    eval "$(conda shell.bash hook)" && conda activate anole
    export CKPT_PATH=/scratch-shared/tesr125338/anole_ckpt
    cd ~/anole
    python ~/4m-peft/anole/tests/01_test_inference.py
"""

import os
import sys
import torch
from pathlib import Path

CKPT_PATH  = os.environ.get('CKPT_PATH', '/scratch-shared/tesr125338/anole_ckpt')
OUTPUT_DIR = Path('/scratch-shared/tesr125338/anole_outputs/tests')
COCO_IMG   = Path('/scratch-shared/tesr125338/coco/val2017/000000000139.jpg')

SEP = '=' * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_token_sequence(tokens, boi=8197, eoi=8196):
    """Split a flat token sequence into alternating text/image segments."""
    segments = []
    current, mode = [], 'text'
    for tok in tokens.tolist():
        if tok == boi:
            if current:
                segments.append({'type': 'text', 'tokens': torch.tensor(current)})
            current, mode = [], 'image'
        elif tok == eoi:
            if current:
                segments.append({'type': 'image', 'tokens': torch.tensor(current)})
            current, mode = [], 'text'
        else:
            current.append(tok)
    if current:
        segments.append({'type': mode, 'tokens': torch.tensor(current)})
    return segments


def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    print(f'  Saved → {path}')


def save_image(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    print(f'  Saved → {path}')


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def check_env():
    print(SEP)
    print('  Environment check')
    print(SEP)
    print(f'  Python    : {sys.version.split()[0]}')
    print(f'  PyTorch   : {torch.__version__}')
    print(f'  CUDA      : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU       : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  CKPT_PATH : {CKPT_PATH}')
    print(f'  Output    : {OUTPUT_DIR}')

    required = [
        Path(CKPT_PATH) / 'models' / '7b',
        Path(CKPT_PATH) / 'tokenizer' / 'text_tokenizer.json',
        Path(CKPT_PATH) / 'tokenizer' / 'vqgan.ckpt',
        Path(CKPT_PATH) / 'tokenizer' / 'vqgan.yaml',
    ]
    ok = True
    for p in required:
        mark = '✓' if p.exists() else '✗ MISSING'
        print(f'  {mark}  {p.relative_to(CKPT_PATH)}')
        if not p.exists():
            ok = False
    if not ok:
        print('\n  ERROR: checkpoint incomplete. Re-download with:')
        print(f'    huggingface-cli download GAIR/Anole-7b-v0.1 --local-dir {CKPT_PATH}')
        sys.exit(1)

    try:
        from constants import MODEL_7B_PATH  # noqa
        print('  ✓ Anole constants importable (running from ~/anole/)')
    except ImportError:
        print('  ERROR: run this script from ~/anole/  (cd ~/anole)')
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    from chameleon.inference.chameleon import ChameleonInferenceModel
    from constants import (MODEL_7B_PATH, TOKENIZER_TEXT_PATH,
                           TOKENIZER_IMAGE_CFG_PATH, TOKENIZER_IMAGE_PATH)
    print(SEP)
    print('  Loading Anole-7b-v0.1')
    print(SEP)
    model = ChameleonInferenceModel(
        str(MODEL_7B_PATH),
        str(TOKENIZER_TEXT_PATH),
        str(TOKENIZER_IMAGE_CFG_PATH),
        str(TOKENIZER_IMAGE_PATH),
    )
    print('  ✓ Model loaded\n')
    return model


# ---------------------------------------------------------------------------
# Test 1 — Text generation
# ---------------------------------------------------------------------------

def test_text_generation(model):
    from chameleon.inference.chameleon import ChameleonInferenceModel
    print(SEP)
    print('  Test 1: text generation')
    print(SEP)

    prompts = [
        'The capital of France is',
        'The main difference between supervised and unsupervised learning is',
        'To fine-tune a large language model efficiently, one common technique is',
    ]

    lines = []
    for prompt in prompts:
        tokens = model.generate(
            prompt_ui=[{'type': 'text', 'value': prompt}],
            options=ChameleonInferenceModel.Options(max_seq_len=80),
        )
        text = model.decode_text(tokens).strip()
        print(f'  P: {prompt}')
        print(f'  A: {text}\n')
        lines.append(f'PROMPT: {prompt}\nOUTPUT: {text}\n')

    save_text(OUTPUT_DIR / 'text_generation.txt', '\n'.join(lines))
    print('  ✓ Text generation OK\n')


# ---------------------------------------------------------------------------
# Test 2 — Image understanding (captioning + VQA)
# ---------------------------------------------------------------------------

def test_image_understanding(model):
    from chameleon.inference.chameleon import ChameleonInferenceModel
    print(SEP)
    print('  Test 2: image understanding (captioning + VQA)')
    print(SEP)

    if not COCO_IMG.exists():
        print(f'  SKIP: COCO image not found at {COCO_IMG}\n')
        return

    questions = [
        'Describe this image in detail.',
        'How many people are visible in this image?',
        'What objects can you identify in this image?',
        'What is the overall mood or atmosphere of this image?',
    ]

    lines = [f'Image: {COCO_IMG}\n']
    for q in questions:
        tokens = model.generate(
            prompt_ui=[
                {'type': 'text',  'value': q + ' '},
                {'type': 'image', 'value': f'file:{COCO_IMG}'},
            ],
            options=ChameleonInferenceModel.Options(max_seq_len=150),
        )
        answer = model.decode_text(tokens).strip()
        print(f'  Q: {q}')
        print(f'  A: {answer}\n')
        lines.append(f'Q: {q}\nA: {answer}\n')

    save_text(OUTPUT_DIR / 'image_understanding.txt', '\n'.join(lines))
    print('  ✓ Image understanding OK\n')


# ---------------------------------------------------------------------------
# Test 3 — Text-to-image generation
# ---------------------------------------------------------------------------

def test_text_to_image(model):
    from chameleon.inference.chameleon import ChameleonInferenceModel
    print(SEP)
    print('  Test 3: text-to-image generation')
    print(SEP)

    prompts = [
        'a golden retriever playing on a sunny beach',
        'a futuristic city skyline at night with neon lights',
        'a bowl of fresh fruit on a wooden table',
        'a snowy mountain landscape at sunrise',
        'a close-up portrait of a cat with blue eyes',
    ]

    out_dir = OUTPUT_DIR / 'text2image'
    for i, prompt in enumerate(prompts):
        print(f'  [{i+1}/{len(prompts)}] {prompt}')
        tokens = model.generate(
            prompt_ui=[{'type': 'text', 'value': prompt}],
            options=ChameleonInferenceModel.Options(
                max_seq_len=1200,
                mode='image',
            ),
        )
        images = model.decode_image(tokens)
        if images:
            fname = f'{i+1:02d}_{prompt[:30].replace(" ", "_")}.png'
            save_image(out_dir / fname, images[0])
        else:
            print(f'  WARNING: no image decoded for prompt {i+1}')

    print('  ✓ Text-to-image OK\n')


# ---------------------------------------------------------------------------
# Test 4 — Interleaved text + image generation
# ---------------------------------------------------------------------------

def test_interleaved(model):
    from chameleon.inference.chameleon import ChameleonInferenceModel
    print(SEP)
    print('  Test 4: interleaved text + image generation')
    print(SEP)

    prompts = [
        ('paris',    'Please introduce the city of Paris with pictures.'),
        ('animals',  'Show me three different animals with a brief description of each.'),
        ('food',     'Describe two Italian dishes and show what they look like.'),
    ]

    for slug, prompt in prompts:
        print(f'  Prompt: {prompt}')
        out_dir = OUTPUT_DIR / 'interleaved' / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        tokens = model.generate(
            prompt_ui=[{'type': 'text', 'value': prompt}],
            options=ChameleonInferenceModel.Options(max_seq_len=3000),
        )

        segments = split_token_sequence(tokens[0])
        text_log = [f'PROMPT: {prompt}\n']
        img_count = 0

        for seg in segments:
            if seg['type'] == 'text':
                text = model.decode_text(seg['tokens']).strip()
                if text:
                    print(f'  TEXT: {text[:80]}...' if len(text) > 80 else f'  TEXT: {text}')
                    text_log.append(f'[TEXT]\n{text}\n')
            else:
                img_count += 1
                images = model.decode_image(seg['tokens'])
                if images:
                    fname = f'image_{img_count:02d}.png'
                    save_image(out_dir / fname, images[0])
                    text_log.append(f'[IMAGE] → {fname}\n')

        save_text(out_dir / 'output.txt', '\n'.join(text_log))
        print(f'  Generated: {img_count} images\n')

    print('  ✓ Interleaved generation OK\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_env()

    if not torch.cuda.is_available():
        print('WARNING: no GPU — will be very slow.')
        if input('Continue? [y/N] ').lower() != 'y':
            sys.exit(0)

    model = load_model()

    test_text_generation(model)
    test_image_understanding(model)
    test_text_to_image(model)
    test_interleaved(model)

    print(SEP)
    print('  All tests done!')
    print(f'  Outputs in: {OUTPUT_DIR}')
    print('  Structure:')
    print('    text_generation.txt      ← test 1')
    print('    image_understanding.txt  ← test 2')
    print('    text2image/*.png         ← test 3 (5 images)')
    print('    interleaved/*/           ← test 4 (text + images per prompt)')
    print(SEP)


if __name__ == '__main__':
    main()
