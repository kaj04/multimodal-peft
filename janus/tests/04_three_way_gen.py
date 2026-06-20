#!/usr/bin/env python3
"""
janus/tests/04_three_way_gen.py

Fair 3-way image generation for a by-hand check of the modularity claim. The
SAME prompts at the SAME seeds are rendered by three models loaded together
(PeftModel multi-adapter), so the adapter is the only variable:

  base    : Janus-Pro-7B, no adapter
  poetry  : the text/poetry LoRA   -> cross-modal check, should look ~= base
  naruto  : the image/Naruto LoRA  -> style acquisition, should be restyled

If the poetry (text) adapter leaves generation essentially unchanged from base
while Naruto restyles it, the claim holds; if poetry images are broken/different,
it does not. We use the configurations the paper calls final.

Run on a GPU (interactive srun, or a short sbatch). ~30 images, a few minutes.

  module load 2023 Anaconda3/2023.07-2 ; source activate janus
  export HF_HOME=/scratch-shared/tesr125338/hf_cache
  PROMPTS="a cat sitting on a chair;a red sports car;a bowl of strawberries" \
    python ~/multimodal-peft/janus/tests/04_three_way_gen.py

Outputs (view in Open OnDemand):
  tests/output/threeway/NN_<slug>/{base,poetry,naruto}.png
  tests/output/threeway/NN_<slug>_montage.png   (base | poetry | naruto)
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
SEED     = int(os.environ.get("SEED", "42"))
TRAIN    = Path(os.path.expanduser("~/multimodal-peft/janus/train/output"))
POETRY   = os.environ.get("POETRY", str(TRAIN / "text_7b_poetry_v1" / "adapter_best"))
NARUTO   = os.environ.get("NARUTO", str(TRAIN / "gen_7b_naruto_v5"   / "adapter_best"))
OUT_DIR  = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"))) / "threeway"

# Default prompts (override with PROMPTS="...;...;..."). Mix of neutral subjects
# so any Naruto restyling / poetry-side breakage is obvious.
_DEFAULT = ("a cat sitting on a chair;a red sports car on a city street;"
            "a bowl of strawberries on a table;a lighthouse on a rocky coast;"
            "a cup of coffee next to a laptop")
PROMPTS = [p.strip() for p in os.environ.get("PROMPTS", _DEFAULT).split(";") if p.strip()]

SEP = "=" * 64


@torch.no_grad()
def generate(model, processor, prompt):
    p = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        add_generation_prompt=True, generation_mode="image")
    inputs = processor(text=p, generation_mode="image",
                       return_tensors="pt").to(model.device)
    model.generation_config.num_return_sequences = 1
    out = model.generate(**inputs, generation_mode="image",
                         max_new_tokens=576, do_sample=True, use_cache=True)
    dec = model.decode_image_tokens(out).detach().float().cpu()
    try:
        arr = np.asarray(processor.postprocess(
            list(dec), return_tensors="np")["pixel_values"][0])
    except Exception:
        arr = dec[0].numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if float(arr.min()) < -0.01:
        arr = (arr + 1.0) / 2.0
    elif float(arr.max()) > 1.01:
        arr = arr / 255.0
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def montage(items, out_path):
    pad = 24
    imgs = [(lbl, im) for lbl, im in items]
    w = sum(im.width for _, im in imgs)
    h = max(im.height for _, im in imgs) + pad
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    x = 0
    for lbl, im in imgs:
        canvas.paste(im, (x, pad))
        d.text((x + 5, 6), lbl, fill="black")
        x += im.width
    canvas.save(str(out_path))


def slug(text, n=28):
    return "".join(c if c.isalnum() else "_" for c in text)[:n].strip("_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  3-way generation | {len(PROMPTS)} prompts | seed {SEED}")
    print(f"  poetry = {POETRY}")
    print(f"  naruto = {NARUTO}")
    print(SEP, flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    base.eval()
    # Load both adapters; names must be dot-free.
    model = PeftModel.from_pretrained(base, POETRY, adapter_name="poetry")
    model.load_adapter(NARUTO, adapter_name="naruto")
    model.eval()

    def render(kind, prompt, seed):
        torch.manual_seed(seed)          # same seed across the three -> fair
        if kind == "base":
            with model.disable_adapter():
                return generate(model, processor, prompt)
        model.set_adapter(kind)
        return generate(model, processor, prompt)

    for i, prompt in enumerate(PROMPTS):
        d = OUT_DIR / f"{i:02d}_{slug(prompt)}"
        d.mkdir(parents=True, exist_ok=True)
        seed = SEED + i
        print(f"  [{i+1}/{len(PROMPTS)}] {prompt!r}", flush=True)
        imgs = []
        for kind in ("base", "poetry", "naruto"):
            img = render(kind, prompt, seed)
            img.save(str(d / f"{kind}.png"))
            imgs.append((kind, img))
        montage(imgs, OUT_DIR / f"{i:02d}_{slug(prompt)}_montage.png")
        print(f"      saved -> {d}  (+ montage)", flush=True)

    print(f"\n  done -> {OUT_DIR}")
    print("  look for: poetry ~= base (modularity holds) and naruto restyled.")


if __name__ == "__main__":
    main()
