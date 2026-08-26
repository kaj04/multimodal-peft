#!/usr/bin/env python3
"""
janus/eval/44_chain_image_grid.py  --  one image grid across every chain stage.

The text side of the chain is now understood: the image step drives the language
head into a whitespace mode and the next text step repairs it. The image side is
not. Two questions are open, and both are visual:

  does the image step actually acquire the Naruto style?
  does it survive the text step that follows it?

If the style is still there at s3 while the text has recovered, the two
modalities are dissociated along the chain, which is a much stronger statement
than either measurement alone.

One row per stage, one column per prompt, base at the top and Naruto-from-base
at the bottom as the reference for what "learned it properly" looks like. The
seed is fixed per column and identical across rows, so any difference down a
column is the adapter and nothing else. Two of the prompts are people (does the
style appear) and two are neutral scenes (does it bleed where it should not).

All adapters are loaded as named adapters on ONE base model, so no row can
differ because of a reload.

  python janus/eval/44_chain_image_grid.py
  ROWS="base,chainA_s2_naruto" python janus/eval/44_chain_image_grid.py   # subset
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import generate_image  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
TRAIN_OUT = Path(os.path.expanduser("~/multimodal-peft/janus/train/output"))
OUT = Path(os.path.expanduser(
    "~/multimodal-peft/janus/tests/output")) / "chain_image_grid"
SEED = int(os.environ.get("SEED", "42"))
THUMB = int(os.environ.get("THUMB", "320"))

# base is generated with the adapter disabled; the rest are adapter directories.
# Chain A only by default: four rows is enough to see whether the style is
# acquired at s2 and whether it survives s3. Widen with ROWS once that is known.
DEFAULT_ROWS = [
    "base",
    "chainA_s1_welsh", "chainA_s2_naruto", "chainA_s3_medicine",
]
ROWS = [r.strip() for r in os.environ.get("ROWS", ",".join(DEFAULT_ROWS)).split(",")
        if r.strip()]

# Two characters (is the style acquired), one in-domain scene, two neutral
# subjects (does the style bleed where it does not belong). Pipe-separated so a
# prompt may contain commas: PROMPTS="a cat|a dog".
DEFAULT_PROMPTS = [
    "a young man with spiky blond hair",
    "a woman with long dark hair",
    "a ninja running through a forest",
    "a red fox sitting in a snowy forest",
    "a cup of coffee on a wooden table",
]
PROMPTS = [p.strip() for p in os.environ.get(
    "PROMPTS", "|".join(DEFAULT_PROMPTS)).split("|") if p.strip()]

SEP = "=" * 72


def adapter_path(tag):
    p = TRAIN_OUT / tag / "adapter_best"
    if not p.exists():
        raise SystemExit(f"  missing adapter: {p}")
    return str(p)


def build_grid(cells, rows, prompts):
    """cells[(row, col)] -> PIL image. Returns the labelled grid."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()

    pad, label_w, header_h = 6, 210, 34
    w = label_w + len(prompts) * (THUMB + pad) + pad
    h = header_h + len(rows) * (THUMB + pad) + pad
    grid = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(grid)

    for j, p in enumerate(prompts):
        x = label_w + j * (THUMB + pad)
        d.text((x + 4, 9), p[:38], fill="black", font=small)

    for i, r in enumerate(rows):
        y = header_h + i * (THUMB + pad)
        d.text((6, y + THUMB // 2 - 8), r, fill="black", font=font)
        for j in range(len(prompts)):
            img = cells.get((r, j))
            if img is None:
                continue
            grid.paste(img.resize((THUMB, THUMB)),
                       (label_w + j * (THUMB + pad), y))
    return grid


def main():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel

    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    named = [r for r in ROWS if r != "base"]
    if not named:
        raise SystemExit("  need at least one adapter row besides base")

    print(f"{SEP}\n  chain image grid\n  rows   : {ROWS}\n"
          f"  prompts: {len(PROMPTS)}  seed={SEED} (fixed per column)\n{SEP}",
          flush=True)

    proc = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    # One model, every adapter loaded onto it under its own name.
    model = PeftModel.from_pretrained(base, adapter_path(named[0]),
                                      adapter_name=named[0])
    for tag in named[1:]:
        model.load_adapter(adapter_path(tag), adapter_name=tag)
    model.eval()
    print(f"  loaded adapters: {list(model.peft_config.keys())}\n", flush=True)

    cells = {}
    for row in ROWS:
        for j, prompt in enumerate(PROMPTS):
            path = OUT / "cells" / f"{row}__p{j}.png"
            if path.exists():
                from PIL import Image
                cells[(row, j)] = Image.open(path).convert("RGB")
                print(f"  cached {row} p{j}", flush=True)
                continue
            if row == "base":
                with model.disable_adapter():
                    torch.manual_seed(SEED + j)
                    img = generate_image(model, proc, prompt)
            else:
                model.set_adapter(row)
                torch.manual_seed(SEED + j)
                img = generate_image(model, proc, prompt)
            img.save(path)
            cells[(row, j)] = img
            print(f"  done {row} p{j}  {prompt[:34]!r}", flush=True)

    grid = build_grid(cells, ROWS, PROMPTS)
    out_png = OUT / "chain_image_grid.png"
    grid.save(out_png)
    print(f"\n{SEP}\n  saved -> {out_png}\n  cells  -> {OUT/'cells'}\n{SEP}")


if __name__ == "__main__":
    main()
