#!/usr/bin/env python3
"""
janus/eval/05c_pixelart_qualitative.py

Quick VISUAL-ONLY test for the pixel-art generalisation run. For five fixed
content prompts, generates a 5x4 grid:

  col 1: base model, plain prompt              ("a cat on a windowsill")
  col 2: base model, prompt + style request    (... ", pixel art style")
  col 3: adapter,    plain prompt
  col 4: adapter,    prompt + style request

This answers two questions at a glance:
  - does the BASE already know pixel art when explicitly asked? (col 1 vs 2)
  - does the ADAPTER (trained WITHOUT a prefix) produce pixel art even when
    NOT asked? (col 3 -- the key generalisation finding)

Same seed for base vs adapter on a given (prompt, variant), so within a
column-pair only the adapter changes. Saves the composed grid as a single PNG
plus the 20 individual frames.

Run via 05c_run_pixelart_qualitative.sh (needs a GPU).
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import generate_image  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
SEED     = int(os.environ.get("SEED", "42"))
STYLE_SUFFIX = os.environ.get("STYLE_SUFFIX", ", pixel art style")

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "pixelart_qual"

# Five fixed content prompts (edit freely). First is the user's example.
PROMPTS = [
    "a group of Dutch students cycling on bikes",
    "a cat sitting on a windowsill",
    "a medieval castle on a hill",
    "a cup of coffee on a wooden table",
    "an astronaut floating in space",
]

COLS = ["Base | plain", "Base | +style", "Adapter | plain", "Adapter | +style"]
SEP = "=" * 64


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
    print(SEP)
    print(f"  Pixel-art qualitative | MODEL={MODEL_ID}")
    print(f"  ADAPTER={ADAPTER}")
    print(f"  STYLE_SUFFIX={STYLE_SUFFIX!r}")
    print(SEP, flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    # frames[r][c] = PIL image
    frames = [[None] * 4 for _ in range(len(PROMPTS))]

    for r, prompt in enumerate(PROMPTS):
        plain = prompt
        styled = prompt + STYLE_SUFFIX
        seed_plain = SEED + r
        seed_styled = SEED + r + 1000

        # col 1: base, plain
        with model.disable_adapter():
            torch.manual_seed(seed_plain)
            frames[r][0] = generate_image(model, processor, plain)
            torch.manual_seed(seed_styled)
            frames[r][1] = generate_image(model, processor, styled)
        # col 3/4: adapter
        torch.manual_seed(seed_plain)
        frames[r][2] = generate_image(model, processor, plain)
        torch.manual_seed(seed_styled)
        frames[r][3] = generate_image(model, processor, styled)

        # save individual frames
        tags = ["base_plain", "base_style", "adapter_plain", "adapter_style"]
        for c, tag in enumerate(tags):
            frames[r][c].save(OUTPUT_DIR / f"prompt{r}_{tag}.png")
        print(f"  done prompt {r}: {prompt}", flush=True)

    # ---- compose the grid PNG ----
    cell = 256
    pad = 6
    label_h = 28          # column header
    row_label_h = 22      # prompt caption under each row
    grid_w = 4 * cell + 5 * pad
    grid_h = label_h + len(PROMPTS) * (cell + row_label_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    fcol = _font(15)
    frow = _font(14)

    # column headers
    for c in range(4):
        x = pad + c * (cell + pad)
        d.text((x + 4, 6), COLS[c], fill=(0, 0, 0), font=fcol)

    y = label_h
    for r, prompt in enumerate(PROMPTS):
        for c in range(4):
            x = pad + c * (cell + pad)
            canvas.paste(frames[r][c].resize((cell, cell)), (x, y))
        # prompt caption under the row
        d.text((pad + 4, y + cell + 3), f"{r}: {prompt}", fill=(40, 40, 40),
               font=frow)
        y += cell + row_label_h + pad

    grid_path = OUTPUT_DIR / "pixelart_grid.png"
    canvas.save(grid_path)
    print(f"\n  saved grid -> {grid_path}")
    print(f"  individual frames in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
