#!/usr/bin/env python3
"""
janus/eval/29_xmodal_welsh_textinimg.py

Experiment B (cross-modal, EXPLORATORY + feasibility-gated): can the Welsh
text-LoRA render requested Welsh text INSIDE a generated image more correctly
than the base? Visual text rendering is a known VQ-VAE weakness and a monolingual
LM adapter may not transfer to glyph rendering, so this is the riskiest of the
three -- we first GATE on legibility.

This script generates base-vs-adapter images for short Welsh-text sign prompts,
plus a few ENGLISH-text controls to gauge whether the base can render ANY legible
text at all. We save individual frames + a base|adapter grid for human inspection
(the go/no-go gate). The automatic OCR metric (CER/WER of the rendered string vs
the target, via Qwen2-VL or Tesseract-cym) is added only if the gate passes.

SMOKE=1 -> first 4 prompts, 1 seed. Run via 29_run_xmodal_welsh_textinimg.sh.
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
K_SEEDS  = int(os.environ.get("K_SEEDS", "3"))
SMOKE    = os.environ.get("SMOKE", "0") == "1"
OUT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output")) / "xmodal_welsh_textinimg"

# (target_text, lang, prompt). The English controls test the legibility gate.
PROMPTS = [
    ("CROESO", "cy", 'a wooden sign that reads "CROESO"'),
    ("BARA", "cy", 'a bakery shop sign that says "BARA"'),
    ("ARAF", "cy", 'a road sign that reads "ARAF"'),
    ("CYMRU", "cy", 'a banner that reads "CYMRU"'),
    ("HEDDWCH", "cy", 'a poster with the word "HEDDWCH"'),
    ("CARIAD", "cy", 'a greeting card with the word "CARIAD"'),
    ("DŴR", "cy", 'a bottle label that reads "DŴR"'),
    ("CROESO I GYMRU", "cy", 'a road sign that reads "CROESO I GYMRU"'),
    ("WELCOME", "en", 'a wooden sign that reads "WELCOME"'),     # control
    ("OPEN", "en", 'a shop sign that reads "OPEN"'),            # control
]
if SMOKE:
    PROMPTS = PROMPTS[:4]
    K_SEEDS = 1

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
    if not ADAPTER:
        raise SystemExit("set ADAPTER=/path/to/adapter_best")
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    print(f"{SEP}\n  Exp B: Welsh text-in-image (feasibility) | ADAPTER={ADAPTER}\n"
          f"  prompts={len(PROMPTS)} x {K_SEEDS} seeds\n{SEP}", flush=True)
    proc = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    # frames[i] = (base seed0, adapter seed0) for the grid; all seeds saved to disk
    grid_frames = []
    for i, (target, lang, prompt) in enumerate(PROMPTS):
        first = [None, None]
        for k in range(K_SEEDS):
            seed = SEED + 100 * i + k
            with model.disable_adapter():
                torch.manual_seed(seed); img_b = generate_image(model, proc, prompt)
            torch.manual_seed(seed); img_a = generate_image(model, proc, prompt)
            img_b.save(OUT / "images" / f"p{i:02d}_{lang}_s{k}_base.png")
            img_a.save(OUT / "images" / f"p{i:02d}_{lang}_s{k}_adapter.png")
            if k == 0:
                first = [img_b, img_a]
        grid_frames.append((target, lang, first))
        print(f"  [{i+1}/{len(PROMPTS)}] {lang} target={target!r}", flush=True)

    # base|adapter grid (seed 0), with the target string labelled
    cell, pad, head, rlab = 256, 6, 26, 22
    gw = 2 * cell + 3 * pad
    gh = head + len(grid_frames) * (cell + rlab + pad) + pad
    canvas = Image.new("RGB", (gw, gh), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    fh, fr = _font(15), _font(14)
    d.text((pad + 4, 6), "Base", fill=(0, 0, 0), font=fh)
    d.text((pad + cell + pad + 4, 6), "Welsh LoRA", fill=(0, 0, 0), font=fh)
    y = head
    for target, lang, (ib, ia) in grid_frames:
        if ib is not None:
            canvas.paste(ib.resize((cell, cell)), (pad, y))
        if ia is not None:
            canvas.paste(ia.resize((cell, cell)), (pad + cell + pad, y))
        d.text((pad + 4, y + cell + 3), f"[{lang}] target: {target}",
               fill=(40, 40, 40), font=fr)
        y += cell + rlab + pad
    gp = OUT / "textinimg_grid.png"
    canvas.save(gp)
    print(f"\n  saved grid -> {gp}\n  frames in {OUT}/images")
    print("  GATE: inspect by eye -- (1) does the BASE render any legible text "
          "(English controls)? (2) is the adapter's Welsh more correct? If yes, "
          "we add the OCR metric (CER/WER via Qwen2-VL).")


if __name__ == "__main__":
    main()
