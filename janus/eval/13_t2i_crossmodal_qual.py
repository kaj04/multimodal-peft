#!/usr/bin/env python3
"""
janus/eval/13_t2i_crossmodal_qual.py

CROSS-MODAL qualitative test for the poetry text-LoRA: does specialising the
TEXT stream change image generation? This is the reverse-direction half of the
modularity experiment. The text LoRA only touches the shared backbone
(q,k,v,o + gate,up,down), never the VQ / generation heads, so whatever change
we see in the images flows purely through the shared transformer.

For a fixed set of detailed, varied prompts (written here -- no official
captions needed) we generate the SAME prompt with the adapter OFF (base) and
ON (adapter), at the SAME seed, so the only variable is the LoRA. Output is a
2-column grid (base | adapter) plus the individual frames, for an at-a-glance
better / worse / neutral judgement.

Base = the same model with disable_adapter(), identical to 05c's convention.

Run via 13_run_t2i_crossmodal_qual.sh (GPU, ~10-15 min). Needs ADAPTER=...
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

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "t2i_crossmodal_qual" / (Path(ADAPTER).parent.name if ADAPTER else "base")

# Detailed, varied prompts spanning content types (animal, landscape, portrait,
# still-life, architecture, vehicle, food, fantasy/action). Edit freely.
PROMPTS = [
    "a red fox sitting in a snowy forest at dawn, soft morning light",
    "a wide mountain landscape with a turquoise lake and pine trees under a clear sky",
    "a close-up portrait of an elderly fisherman with a weathered face and a wool cap",
    "a still life of a bowl of lemons and a glass of water on a wooden table",
    "a futuristic city skyline at night with neon lights and flying cars",
    "a vintage red sports car parked on a cobblestone street in an old town",
    "a plate of spaghetti with tomato sauce and fresh basil, top-down view",
    "an astronaut riding a horse on the surface of Mars, cinematic lighting",
]

COLS = ["Base (adapter off)", "Adapter (poetry text-LoRA)"]
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  T2I cross-modal qualitative | MODEL={MODEL_ID}")
    print(f"  ADAPTER={ADAPTER}")
    print(SEP, flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    frames = [[None, None] for _ in range(len(PROMPTS))]
    for r, prompt in enumerate(PROMPTS):
        seed = SEED + r
        # col 0: base (adapter disabled)
        with model.disable_adapter():
            torch.manual_seed(seed)
            frames[r][0] = generate_image(model, processor, prompt)
        # col 1: adapter on (same seed)
        torch.manual_seed(seed)
        frames[r][1] = generate_image(model, processor, prompt)

        frames[r][0].save(OUTPUT_DIR / f"prompt{r}_base.png")
        frames[r][1].save(OUTPUT_DIR / f"prompt{r}_adapter.png")
        print(f"  done prompt {r}: {prompt}", flush=True)

    # ---- compose 2-column grid ----
    cell, pad, label_h, row_label_h = 256, 6, 28, 22
    grid_w = 2 * cell + 3 * pad
    grid_h = label_h + len(PROMPTS) * (cell + row_label_h + pad) + pad
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    fcol, frow = _font(15), _font(14)
    for c in range(2):
        d.text((pad + c * (cell + pad) + 4, 6), COLS[c], fill=(0, 0, 0), font=fcol)
    y = label_h
    for r, prompt in enumerate(PROMPTS):
        for c in range(2):
            canvas.paste(frames[r][c].resize((cell, cell)),
                         (pad + c * (cell + pad), y))
        d.text((pad + 4, y + cell + 3), f"{r}: {prompt[:80]}",
               fill=(40, 40, 40), font=frow)
        y += cell + row_label_h + pad
    grid_path = OUTPUT_DIR / "t2i_crossmodal_grid.png"
    canvas.save(grid_path)
    print(f"\n  saved grid -> {grid_path}")
    print(f"  individual frames in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
