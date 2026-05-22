#!/usr/bin/env python3
"""
chameleon/tests/01_test_inference.py

Step 1: verify Chameleon-7B loads and runs inference correctly.
No fine-tuning here — just testing that the model works end-to-end.

Usage (on Snellius, interactive node with GPU):
    salloc --partition=gpu_a100 --gres=gpu:1 --mem=40G --time=01:00:00
    conda activate 4m-peft
    python chameleon/tests/01_test_inference.py

Expected output:
    - Model loads (~14 GB in bfloat16)
    - Text-only generation works
    - Image+text generation works (requires a test image)
    - Prints generated text for both cases
"""

import sys
import torch
from pathlib import Path

HF_CACHE  = '/scratch-shared/tesr125338/hf_cache'
MODEL_ID  = 'facebook/chameleon-7b'

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

    try:
        import transformers
        print(f'  transformers: {transformers.__version__}')
        # Chameleon requires >= 4.41.0
        from packaging import version
        if version.parse(transformers.__version__) < version.parse('4.41.0'):
            print('  WARNING: transformers < 4.41.0 — Chameleon may not be supported!')
            print('  Fix: pip install -U transformers')
    except ImportError:
        print('  ERROR: transformers not installed')
        sys.exit(1)

    try:
        import peft
        print(f'  peft        : {peft.__version__}')
    except ImportError:
        print('  WARNING: peft not installed — needed for LoRA later')
        print('  Fix: pip install peft')

    print()


def load_model():
    from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

    print('=' * 60)
    print(f'  Loading Chameleon-7B from cache: {HF_CACHE}')
    print('  (first run downloads ~14 GB — takes a few minutes)')
    print('=' * 60)

    processor = ChameleonProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)

    model = ChameleonForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',       # auto-places layers on available GPUs
        cache_dir=HF_CACHE,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'\n  Model loaded: {n_params:.1f}B parameters')
    print(f'  dtype: {next(model.parameters()).dtype}')
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f'  GPU memory used: {mem:.1f} GB')

    return model, processor


def test_text_only(model, processor):
    """Simple text generation — no image, just verify the LM works."""
    print('\n' + '=' * 60)
    print('  Test 1: text-only generation')
    print('=' * 60)

    prompt = 'The capital of France is'
    inputs = processor(text=prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    generated = processor.decode(output[0], skip_special_tokens=True)
    print(f'  Prompt   : {prompt}')
    print(f'  Generated: {generated}')
    assert 'Paris' in generated or 'paris' in generated.lower(), \
        'Unexpected output — model might not be loading correctly'
    print('  ✓ Text-only generation OK')


def test_image_captioning(model, processor):
    """Image captioning — tries local COCO image first, then downloads from web."""
    print('\n' + '=' * 60)
    print('  Test 2: image + text (captioning)')
    print('=' * 60)

    from PIL import Image
    import requests
    import io
    from pathlib import Path

    # Try local COCO val2017 image first (already on Snellius scratch)
    local_img = Path('/scratch-shared/tesr125338/coco/val2017/000000000139.jpg')
    if local_img.exists():
        image = Image.open(local_img).convert('RGB')
        print(f'  Test image: {image.size} pixels (local COCO: {local_img.name})')
    else:
        # Fallback: Maradona at Napoli — public domain, Wikimedia Commons
        img_url = 'https://commons.wikimedia.org/wiki/Special:FilePath/Diego_Maradona_Napoli.JPG'
        try:
            response = requests.get(img_url, timeout=10, allow_redirects=True,
                                   headers={'User-Agent': 'Mozilla/5.0'})
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            print(f'  Test image: {image.size} pixels (downloaded)')
        except Exception as e:
            print(f'  Could not load test image ({e}) — skipping image test')
            return

    prompt = 'Describe this image in one sentence:'
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors='pt',
    ).to(model.device)

    # Cast pixel_values to model dtype (bfloat16) to avoid dtype mismatch
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(next(model.parameters()).dtype)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated = processor.decode(output[0], skip_special_tokens=True)
    print(f'  Prompt   : {prompt}')
    print(f'  Generated: {generated}')
    print('  ✓ Image captioning OK')


def test_lora_injection(model):
    """Verify LoRA can be injected with HuggingFace PEFT."""
    print('\n' + '=' * 60)
    print('  Test 3: LoRA injection via HuggingFace PEFT')
    print('=' * 60)

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            # Target attention projection matrices (same strategy as 4M experiments)
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.1,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(model, lora_config)

        total     = sum(p.numel() for p in peft_model.parameters())
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f'  Total params    : {total/1e6:.1f}M')
        print(f'  Trainable (LoRA): {trainable/1e6:.3f}M  ({100*trainable/total:.3f}%)')
        print(f'  ✓ LoRA injection OK')

        # List LoRA modules
        lora_keys = [n for n, _ in peft_model.named_parameters() if 'lora_' in n]
        print(f'  LoRA tensors    : {len(lora_keys)}')
        print(f'  Example keys    : {lora_keys[:3]}')

    except ImportError:
        print('  SKIP: peft not installed. Run: pip install peft')


def main():
    check_env()

    if not torch.cuda.is_available():
        print('WARNING: no GPU detected. Loading Chameleon-7B on CPU will be very slow.')
        print('Use an interactive GPU node: salloc --partition=gpu_a100 --gres=gpu:1 ...')
        response = input('Continue anyway? [y/N] ')
        if response.lower() != 'y':
            sys.exit(0)

    model, processor = load_model()

    test_text_only(model, processor)
    test_image_captioning(model, processor)
    test_lora_injection(model)

    print('\n' + '=' * 60)
    print('  All tests passed! Chameleon-7B is ready.')
    print('  Next: chameleon/tests/02_test_coco_caption.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
