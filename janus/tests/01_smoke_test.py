#!/usr/bin/env python3
"""
janus/tests/01_smoke_test.py

Phase 1 smoke test for Janus-Pro-1B. Verifies the model loads and that the
two capabilities we care about actually work, saving every output to disk
for human inspection:

  1. Understanding (image -> text): caption + count on a few local COCO images
  2. Generation (text -> image): a few prompts -> PNG files

Outputs go to ~/multimodal-peft/janus/tests/output/ (visible in Open OnDemand).

Run via the SLURM script 01_run_smoke_test.sh (needs a GPU). The model must be
pre-downloaded on the login node first (compute nodes have no internet).
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID   = "deepseek-community/Janus-Pro-1B"
COCO_DIR   = Path("/scratch-shared/tesr125338/coco/val2017")
OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
))

SEP = "=" * 60


def load_model():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    print(SEP)
    print(f"  Loading {MODEL_ID}")
    print(SEP)
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    model = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n:.2f}B params, dtype {next(model.parameters()).dtype}")
    if torch.cuda.is_available():
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print()
    return model, processor


def pick_coco_images(n=3):
    if not COCO_DIR.exists():
        print(f"  WARNING: COCO dir not found at {COCO_DIR}")
        return []
    imgs = sorted(COCO_DIR.glob("*.jpg"))[:n]
    return imgs


def test_understanding(model, processor):
    """image -> text: caption + simple VQA on a few local COCO images."""
    print(SEP)
    print("  Test 1: understanding (image -> text)")
    print(SEP)

    images = pick_coco_images(n=3)
    if not images:
        print("  SKIP: no COCO images available\n")
        return

    questions = [
        "Describe this image in one sentence.",
        "How many people are in this image?",
    ]

    # Save the tested images alongside the answers so they can be inspected
    # in the Open OnDemand file browser (scratch is not visible there).
    img_out = OUTPUT_DIR / "understanding_images"
    img_out.mkdir(parents=True, exist_ok=True)

    lines = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        image.save(img_out / img_path.name)
        lines.append(f"=== {img_path.name} ===")
        print(f"  Image: {img_path.name}")
        for q in questions:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": q}],
            }]
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, generation_mode="text",
            )
            inputs = processor(
                text=prompt, images=[image],
                generation_mode="text", return_tensors="pt",
            ).to(model.device, torch.bfloat16)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=64,
                    generation_mode="text", do_sample=False,
                )
            # Decode ONLY the newly generated tokens (strip the prompt/system text)
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            answer = processor.decode(gen_tokens, skip_special_tokens=True).strip()
            print(f"    Q: {q}")
            print(f"    A: {answer}")
            lines.append(f"Q: {q}\nA: {answer}\n")
        lines.append("")
        print()

    (OUTPUT_DIR / "understanding.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved -> {OUTPUT_DIR / 'understanding.txt'}")
    print("  Test 1 done\n")


def test_generation(model, processor):
    """text -> image: a few prompts -> PNG files."""
    print(SEP)
    print("  Test 2: generation (text -> image)")
    print(SEP)

    prompts = [
        "a photograph of a red apple on a wooden table",
        "a cute cat sitting on a sofa",
        "a snowy mountain landscape at sunrise",
    ]

    out_dir = OUTPUT_DIR / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {text}")
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, generation_mode="image",
        )
        inputs = processor(
            text=prompt, generation_mode="image", return_tensors="pt",
        ).to(model.device)

        model.generation_config.num_return_sequences = 1
        with torch.no_grad():
            outputs = model.generate(
                **inputs, generation_mode="image",
                # An image is 576 VQ tokens (24x24). Without this, generate uses
                # the default max_length=20 -> truncated -> all-black image.
                max_new_tokens=576,
                do_sample=True, use_cache=True,
            )
            # Decode INSIDE no_grad so the tensor never requires grad (root fix).
            decoded = model.decode_image_tokens(outputs)

        decoded = decoded.detach().float().cpu()
        # Diagnostics: if images are still off, these tell us whether the
        # problem is generation (decoded near-constant) or conversion.
        print(f"    decoded shape={tuple(decoded.shape)} "
              f"min={float(decoded.min()):.3f} max={float(decoded.max()):.3f} "
              f"mean={float(decoded.mean()):.3f}")

        # Try the processor's postprocess (proper denormalization). If this
        # transformers version chokes on it, fall back to a manual conversion
        # so we always get a viewable image instead of crashing the test.
        try:
            result = processor.postprocess(list(decoded), return_tensors="np")
            arr = np.asarray(result["pixel_values"][0])
        except Exception as e:
            print(f"    postprocess failed ({e}); manual conversion")
            arr = decoded[0].numpy()

        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):   # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if float(arr.min()) < 0.0:          # roughly [-1, 1] -> [0, 1]
                arr = (arr + 1.0) / 2.0
            if float(arr.max()) <= 1.0 + 1e-3:  # [0, 1] -> [0, 255]
                arr = (arr.clip(0, 1) * 255).round().astype("uint8")
            else:
                arr = arr.clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)
        fname = f"{i+1:02d}_{text[:30].replace(' ', '_')}.png"
        img.save(str(out_dir / fname))
        print(f"    Saved -> {out_dir / fname}")

    print("  Test 2 done\n")


def main():
    print(SEP)
    print("  Janus-Pro-1B smoke test")
    print(f"  CUDA: {torch.cuda.is_available()}  |  torch {torch.__version__}")
    print(f"  Output: {OUTPUT_DIR}")
    print(SEP + "\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()

    # Run each test independently so one failure doesn't hide the other.
    try:
        test_understanding(model, processor)
    except Exception:
        print("  Test 1 (understanding) FAILED:")
        traceback.print_exc()
        print()

    try:
        test_generation(model, processor)
    except Exception:
        print("  Test 2 (generation) FAILED:")
        traceback.print_exc()
        print()

    print(SEP)
    print("  Smoke test finished.")
    print(f"  Inspect outputs in: {OUTPUT_DIR}")
    print("    understanding.txt   <- captions / VQA answers")
    print("    generation/*.png    <- generated images")
    print(SEP)


if __name__ == "__main__":
    main()
