#!/usr/bin/env python3
"""
janus/eval/15_probe_base_style_capability.py

Find a DISTANT image style by asking the base what it CANNOT do, instead of
hunting for fragile style datasets. For a small set of content subjects x a set
of candidate styles, we let the BASE Janus-Pro-7B generate (style is put in the
prompt) and lay everything out in one grid:

  rows    = styles  (plain photoreal reference + candidate styles)
  columns = content subjects (cat, portrait, castle, ...)

A human then reads the grid: styles the base renders convincingly are CLOSE
(bad PEFT targets -> small weight shift -> trivial no-forgetting). Styles it
fails at are DISTANT (good PEFT targets -> forced large weight shift -> the
strongest forgetting test). This directly probes "is the base already good at
this style?" -- the central validity question -- with zero dataset dependency.

Once a promising style is chosen, we quantify its distance with the VQ-CE probe
(14_probe_image_distance.py) on a real dataset of that style, and BLIP-caption
that dataset (content-only) for training.

Base model only (no adapter). Saves the grid PNG + individual frames.
Run via 15_run_probe_base_style_capability.sh (GPU, ~5 min).
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import generate_image  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
SEED     = int(os.environ.get("SEED", "42"))

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "base_style_capability"

# Styles = (short label, prompt suffix). "" suffix = plain photoreal reference.
# Override with STYLES env: "label=suffix;label=suffix;..." (empty suffix ok).
_DEFAULT_STYLES = [
    ("plain", ""),
    ("pixel art", "pixel art, 8-bit"),
    ("pencil sketch", "pencil sketch, line drawing, black and white"),
    ("van Gogh", "in the style of Vincent van Gogh, post-impressionist oil painting"),
    ("ukiyo-e", "ukiyo-e Japanese woodblock print"),
    ("X-ray", "X-ray radiograph, medical imaging, grayscale"),
    ("cross-stitch", "cross-stitch embroidery pattern"),
]
# Content subjects (columns). Override with CONTENTS env: "a;b;c".
_DEFAULT_CONTENTS = ["a cat", "a portrait of a man", "a castle on a hill"]

SEP = "=" * 64


def _parse_styles():
    raw = os.environ.get("STYLES", "")
    if not raw:
        return _DEFAULT_STYLES
    out = []
    for part in raw.split(";"):
        if not part.strip():
            continue
        label, _, suffix = part.partition("=")
        out.append((label.strip(), suffix.strip()))
    return out


def _parse_contents():
    raw = os.environ.get("CONTENTS", "")
    if not raw:
        return _DEFAULT_CONTENTS
    return [c.strip() for c in raw.split(";") if c.strip()]


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = _parse_styles()
    contents = _parse_contents()
    print(SEP)
    print(f"  Base style-capability probe | MODEL={MODEL_ID}")
    print(f"  {len(styles)} styles x {len(contents)} contents")
    print(SEP, flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    base.eval()

    frames = [[None] * len(contents) for _ in range(len(styles))]
    for r, (label, suffix) in enumerate(styles):
        for c, content in enumerate(contents):
            prompt = content if not suffix else f"{content}, {suffix}"
            torch.manual_seed(SEED + r * 100 + c)
            img = generate_image(base, processor, prompt)
            frames[r][c] = img
            safe = label.replace(" ", "_").replace("/", "-")
            img.save(OUTPUT_DIR / f"{r:02d}_{safe}__{c}.png")
        print(f"  done style '{label}'", flush=True)

    # ---- compose grid: rows = styles (left-labelled), cols = contents ----
    cell, pad, top_h, gutter = 256, 6, 28, 160
    grid_w = gutter + len(contents) * cell + (len(contents) + 1) * pad
    grid_h = top_h + len(styles) * (cell + pad) + pad
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    fcol, frow = _font(15), _font(15)

    for c, content in enumerate(contents):
        x = gutter + c * (cell + pad) + 4
        d.text((x, 6), content, fill=(0, 0, 0), font=fcol)

    y = top_h
    for r, (label, _suffix) in enumerate(styles):
        d.text((6, y + cell // 2 - 8), label, fill=(20, 20, 20), font=frow)
        for c in range(len(contents)):
            x = gutter + c * (cell + pad)
            canvas.paste(frames[r][c].resize((cell, cell)), (x, y))
        y += cell + pad

    grid_path = OUTPUT_DIR / "base_style_grid.png"
    canvas.save(grid_path)
    print(f"\n  saved grid -> {grid_path}")
    print(f"  individual frames in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
