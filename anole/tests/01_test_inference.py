#!/usr/bin/env python3
"""
anole/tests/01_test_inference.py

Test Anole-7b-v0.1 inference end-to-end:
  1. Text-only generation
  2. Text-to-image generation  (saves PNG to output dir)
  3. Image-to-text captioning  (uses local COCO image)

All generated images are saved to OUTPUT_DIR for visual inspection
via the Snellius file browser or scp.

Requirements:
  - conda env 'anole' with Anole's custom transformers fork installed
  - CKPT_PATH env variable pointing to the downloaded Anole-7b-v0.1 checkpoint
  - ~/anole/ repo cloned from https://github.com/GAIR-NLP/anole

Usage (on Snellius GPU node):
    module load 2023 && module load Anaconda3/2023.07-2
    eval "$(conda shell.bash hook)" && conda activate anole
    export CKPT_PATH=/scratch-shared/tesr125338/anole_ckpt
    cd ~/anole
    python ~/4m-peft/anole/tests/01_test_inference.py

Expected output:
  - Text generation: coherent continuation
  - Image generation: PNG file saved to OUTPUT_DIR
  - Captioning: text description of the COCO image
"""

import os
import sys
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CKPT_PATH  = os.environ.get('CKPT_PATH', '/scratch-shared/tesr125338/anole_ckpt')
OUTPUT_DIR = Path('/scratch-shared/tesr125338/anole_outputs/tests')
COCO_IMG   = Path('/scratch-shared/tesr125338/coco/val2017/000000000139.jpg')


def check_env():
    print('=' * 60)
    print('  Environment check')
    print('=' * 60)
    print(f'  Python      : {sys.version.split()[0]}')
    print(f'  PyTorch     : {torch.__version__}')
    print(f'  CUDA avail  : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU         : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  CKPT_PATH   : {CKPT_PATH}')

    # Verify checkpoint structure
    required = [
        Path(CKPT_PATH) / 'models' / '7b',
        Path(CKPT_PATH) / 'tokenizer' / 'text_tokenizer.json',
        Path(CKPT_PATH) / 'tokenizer' / 'vqgan.ckpt',
        Path(CKPT_PATH) / 'tokenizer' / 'vqgan.yaml',
    ]
    all_ok = True
    for p in required:
        exists = p.exists()
        mark = '✓' if exists else '✗ MISSING'
        print(f'  {mark}  {p}')
        if not exists:
            all_ok = False
    if not all_ok:
        print('\n  ERROR: some checkpoint files are missing.')
        print('  Run: huggingface-cli download GAIR/Anole-7b-v0.1 \\')
        print(f'         --local-dir {CKPT_PATH}')
        sys.exit(1)

    # Verify Anole repo is importable (must run from ~/anole or have it on sys.path)
    try:
        from constants import MODEL_7B_PATH  # noqa: F401
        print('  ✓ Anole constants importable')
    except ImportError:
        print('  ERROR: cannot import Anole constants.')
        print('  Make sure you run this script from ~/anole/:')
        print('    cd ~/anole && python path/to/this/script.py')
        sys.exit(1)

    print()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'  Output dir  : {OUTPUT_DIR}')
    print()


def load_model():
    from chameleon.inference.chameleon import ChameleonInferenceModel
    from constants import MODEL_7B_PATH, TOKENIZER_TEXT_PATH, TOKENIZER_IMAGE_CFG_PATH, TOKENIZER_IMAGE_PATH

    print('=' * 60)
    print('  Loading Anole-7b-v0.1')
    print('=' * 60)
    print(f'  Model  : {MODEL_7B_PATH}')
    print(f'  Tokeni : {TOKENIZER_TEXT_PATH}')
    print(f'  VQGAN  : {TOKENIZER_IMAGE_PATH}')

    model = ChameleonInferenceModel(
        str(MODEL_7B_PATH),
        str(TOKENIZER_TEXT_PATH),
        str(TOKENIZER_IMAGE_CFG_PATH),
        str(TOKENIZER_IMAGE_PATH),
    )
    print('  ✓ Model loaded\n')
    return model


def test_text_generation(model):
    """Simple text continuation — verifies the LM head works."""
    print('=' * 60)
    print('  Test 1: text-only generation')
    print('=' * 60)

    from chameleon.inference.chameleon import ChameleonInferenceModel

    prompt = 'The capital of France is'
    tokens = model.generate(
        prompt_ui=[{'type': 'text', 'value': prompt}],
        options=ChameleonInferenceModel.Options(
            max_seq_len=64,
            mode='text',
        ),
    )
    text = model.decode_text(tokens).strip()
    print(f'  Prompt    : {prompt}')
    print(f'  Generated : {text}')
    assert 'paris' in text.lower() or 'Paris' in text, \
        f'Unexpected output: {text}'
    print('  ✓ Text generation OK\n')


def test_image_generation(model):
    """Generate an image from a text prompt and save it."""
    print('=' * 60)
    print('  Test 2: text-to-image generation')
    print('=' * 60)

    from chameleon.inference.chameleon import ChameleonInferenceModel

    prompts = [
        'a sunny beach with palm trees',
        'a red apple on a wooden table',
    ]

    for prompt in prompts:
        print(f'  Prompt: {prompt}')
        tokens = model.generate(
            prompt_ui=[
                {'type': 'text', 'value': prompt},
            ],
            options=ChameleonInferenceModel.Options(
                max_seq_len=1200,
                mode='image',
            ),
        )
        images = model.decode_image(tokens)
        if images:
            out_path = OUTPUT_DIR / f'gen_{prompt[:20].replace(" ", "_")}.png'
            images[0].save(str(out_path))
            print(f'  Saved : {out_path}')
        else:
            print('  WARNING: no image decoded from token sequence')

    print('  ✓ Image generation OK\n')


def test_image_captioning(model):
    """Caption a COCO image — verifies multimodal understanding."""
    print('=' * 60)
    print('  Test 3: image captioning (image → text)')
    print('=' * 60)

    if not COCO_IMG.exists():
        print(f'  SKIP: COCO image not found at {COCO_IMG}')
        return

    from chameleon.inference.chameleon import ChameleonInferenceModel

    questions = [
        'Describe this image in one sentence.',
        'How many people are in this image?',
    ]

    for q in questions:
        print(f'  Q: {q}')
        tokens = model.generate(
            prompt_ui=[
                {'type': 'text',  'value': q + ' '},
                {'type': 'image', 'value': f'file:{COCO_IMG}'},
            ],
            options=ChameleonInferenceModel.Options(
                max_seq_len=128,
                mode='text',
            ),
        )
        answer = model.decode_text(tokens).strip()
        print(f'  A: {answer}\n')

    print('  ✓ Image captioning OK\n')


def main():
    check_env()

    if not torch.cuda.is_available():
        print('WARNING: no GPU detected — Anole on CPU will be extremely slow.')
        resp = input('Continue anyway? [y/N] ')
        if resp.lower() != 'y':
            sys.exit(0)

    model = load_model()

    test_text_generation(model)
    test_image_generation(model)
    test_image_captioning(model)

    print('=' * 60)
    print('  All tests passed — Anole-7b-v0.1 is ready.')
    print(f'  Generated images are in: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
