#!/usr/bin/env python3
"""
janus/eval/05_pokemon_qualitative.py

Qualitative side-by-side comparison of a Janus generation adapter against the
base model, on prompts held-out from the training split.

DESIGN (Naruto / Janus-Pro-7B baseline):
  - 10 BASE prompts:
       * 5 sampled from the held-out eval split of HF_DATASET_NAME (BLIP-style)
       * 5 hand-written by us (free-form, partially out-of-domain)
  - Each base prompt is generated TWICE:
       * 'natural'  -> the raw base prompt
       * 'prefix'   -> CAPTION_PREFIX + base prompt    (same prefix used at training)
  - For each (base, mode) we generate one image with the ADAPTER OFF (before)
    and one with the ADAPTER ON (after). Same seed -> identical sampling noise
    -> only the adapter changes within a pair, only the prefix changes across
    'natural' vs 'prefix' for the same base.
  - This gives a clean 2x2 experiment per base prompt:
         (prefix off, adapter off)  (prefix off, adapter on)
         (prefix on,  adapter off)  (prefix on,  adapter on)
    The 'natural' column tests whether the adapter learned to activate the
    domain WITHOUT the trigger phrase. The 'prefix' column tests whether the
    trigger phrase is the right way to query the adapter at inference.

OUTPUT (per pair):
  - PNG with a text header (category, source, prompt) and the BEFORE | AFTER
    images side by side.
  - JSON with mean CLIPScore by (source x mode) and per-pair scores.

Total: 10 base prompts x 2 modes = 20 pair PNGs.

Run via SLURM (needs a GPU). Typical chain:
  sbatch --export=ALL,ADAPTER=/.../adapter_epoch15,MODEL_ID=deepseek-community/Janus-Pro-7B \
         --dependency=afterok:<train_jobid> \
         ~/multimodal-peft/janus/eval/05_run_pokemon_qualitative.sh
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import CLIPScorer, generate_image

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-1B")
ADAPTER  = os.environ.get(
    "ADAPTER",
    os.path.expanduser(
        "~/multimodal-peft/janus/train/output/gen_1b_pokemon_r8_ce_constant/adapter_epoch5"
    ),
)

# Caption prefix used at training (must match the trainer's CAPTION_PREFIX).
# We test BOTH with and without it: each base prompt is generated twice.
CAPTION_PREFIX = os.environ.get("CAPTION_PREFIX",
                                "anime sprite in Naruto style, ")

# Same SEED as the trainer so the eval split is the same set of held-out items
SEED       = int(os.environ.get("SEED", "42"))
TRAIN_N    = int(os.environ.get("TRAIN_N", "4000"))   # matches the trainer cap
EVAL_N     = int(os.environ.get("EVAL_N", "200"))     # matches the trainer cap
N_EVAL_BASE  = int(os.environ.get("N_EVAL_BASE",  "5"))   # prompts from eval split
N_HUMAN_BASE = int(os.environ.get("N_HUMAN_BASE", "5"))   # hand-written prompts

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "naruto_qual_7b"

SEP = "=" * 64

HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME",
                                 "lambdalabs/naruto-blip-captions")
HF_TEXT_COL     = os.environ.get("HF_TEXT_COL", "text")


# ---------------------------------------------------------------------------
# Hand-written prompts (kept identical across prefix on/off; the prefix is
# the only thing that varies between the two PNGs for a given base index).
# Mix of in-domain ("ninja", "shinobi") and out-of-domain ("samurai", "wizard")
# to test how strongly the adapter generalises beyond the literal training data.
# ---------------------------------------------------------------------------
HUMAN_PROMPTS = [
    "a young man with blonde spiky hair",
    "a samurai warrior holding a katana",
    "an old wizard with a long white beard",
    "a teenage girl with pink hair",
    "a ninja jumping between rooftops at night",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_prompts(n):
    """Re-derive the SAME held-out eval split the trainer used and return the
    first n captions from it."""
    from datasets import load_dataset
    print(f"  loading {HF_DATASET_NAME} (text_col={HF_TEXT_COL!r}) ...", flush=True)
    ds = load_dataset(HF_DATASET_NAME, split="train")
    ids = list(range(len(ds)))
    random.Random(SEED).shuffle(ids)
    n_eval  = min(EVAL_N, 80)
    n_train = min(TRAIN_N, len(ids) - n_eval)
    eval_ids = ids[n_train:n_train + n_eval]
    return [ds[i][HF_TEXT_COL] for i in eval_ids[:n]]


# ---------------------------------------------------------------------------
# Image composition: side-by-side with header text
# ---------------------------------------------------------------------------

def _get_font(size=18):
    """Best-effort font lookup. Falls back to the PIL default if no TTF found."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_w):
    """Greedy word wrap to fit max_w pixels."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        trial = cur + " " + w
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def compose_pair(before_img, after_img, header_lines):
    """Build the final PNG: a text header on top, then BEFORE | AFTER images.
    header_lines is a list of (text, bold?) tuples for layout flexibility."""
    target_h = max(before_img.height, after_img.height)
    target_w = max(before_img.width,  after_img.width)
    gap = 8

    # Header sizing
    font_title = _get_font(22)
    font_body  = _get_font(18)
    sub_labels = _get_font(16)

    canvas_w = target_w * 2 + gap

    # Draft a temp image to measure header height (depends on wrap)
    temp = Image.new("RGB", (canvas_w, 4000), (255, 255, 255))
    d = ImageDraw.Draw(temp)
    y = 12
    for text, bold in header_lines:
        font = font_title if bold else font_body
        for ln in _wrap_text(d, text, font, canvas_w - 24):
            bbox = d.textbbox((12, y), ln, font=font)
            y = bbox[3] + 4
        y += 4
    # Reserve space for the BEFORE/AFTER sub-labels under the header
    y_sub = y + 6
    bbox = d.textbbox((12, y_sub), "BEFORE (base)", font=sub_labels)
    y_images = bbox[3] + 8
    header_h = y_images

    # Real canvas
    canvas = Image.new("RGB", (canvas_w, header_h + target_h),
                       (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    y = 12
    for text, bold in header_lines:
        font = font_title if bold else font_body
        for ln in _wrap_text(d, text, font, canvas_w - 24):
            d.text((12, y), ln, fill=(0, 0, 0), font=font)
            bbox = d.textbbox((12, y), ln, font=font)
            y = bbox[3] + 4
        y += 4
    # BEFORE / AFTER sub-labels right above each image
    d.text((12, y + 6), "BEFORE (base)",
           fill=(80, 80, 80), font=sub_labels)
    d.text((target_w + gap + 12, y + 6), "AFTER (LoRA)",
           fill=(80, 80, 80), font=sub_labels)

    # Paste images
    canvas.paste(before_img.resize((target_w, target_h)), (0, header_h))
    canvas.paste(after_img.resize((target_w, target_h)),
                 (target_w + gap, header_h))
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  Janus qualitative comparison (4-category, 2x2 design)")
    print(f"  MODEL_ID:       {MODEL_ID}")
    print(f"  ADAPTER:        {ADAPTER}")
    print(f"  CAPTION_PREFIX: {CAPTION_PREFIX!r}")
    print(f"  N_EVAL_BASE={N_EVAL_BASE}  N_HUMAN_BASE={N_HUMAN_BASE}")
    print(f"  OUTPUT_DIR:     {OUTPUT_DIR}")
    print(SEP + "\n")

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    scorer = CLIPScorer()

    # Build the 10 base prompts: 5 from eval split + 5 hand-written.
    eval_prompts  = load_eval_prompts(N_EVAL_BASE)
    human_prompts = HUMAN_PROMPTS[:N_HUMAN_BASE]
    base_prompts = [("eval",  p) for p in eval_prompts] \
                 + [("human", p) for p in human_prompts]

    print(f"  base prompts ({len(base_prompts)}):")
    for k, (src, p) in enumerate(base_prompts):
        print(f"    [{k:02d}] ({src:5s}) {p}")
    print()

    # For each (base prompt, mode) we generate before+after with the same seed.
    # The seed is base-index-only (not mode-dependent) so prefix vs no-prefix
    # share noise too. -> the 2x2 of {prefix x adapter} for one base is comparable.
    results = []
    t0 = time.time()

    for idx, (source, base_prompt) in enumerate(base_prompts):
        for mode in ("natural", "prefix"):
            full_prompt = (CAPTION_PREFIX + base_prompt) if mode == "prefix" else base_prompt
            tag = f"{source}_{idx:02d}_{mode}"
            print(f"  [{tag}]  prompt={full_prompt!r}", flush=True)

            # BEFORE: adapter off
            torch.manual_seed(SEED + idx)
            with model.disable_adapter():
                img_before = generate_image(model, processor, full_prompt)
            clip_before = scorer.score(img_before, base_prompt)

            # AFTER: adapter on
            torch.manual_seed(SEED + idx)
            img_after = generate_image(model, processor, full_prompt)
            clip_after = scorer.score(img_after, base_prompt)

            # Compose PNG with header
            header = [
                (f"[{source.upper()}]  mode={mode}", True),
                (f"prompt: {full_prompt}", False),
            ]
            pair = compose_pair(img_before, img_after, header)
            pair.save(OUTPUT_DIR / f"{tag}_pair.png")
            img_before.save(OUTPUT_DIR / f"{tag}_before.png")
            img_after.save(OUTPUT_DIR / f"{tag}_after.png")

            results.append({
                "idx":          idx,
                "source":       source,
                "mode":         mode,
                "base_prompt":  base_prompt,
                "full_prompt":  full_prompt,
                "clip_before":  float(clip_before),
                "clip_after":   float(clip_after),
                "delta_clip":   float(clip_after - clip_before),
            })
            print(f"      CLIPScore before={clip_before:.2f}  "
                  f"after={clip_after:.2f}  delta={clip_after - clip_before:+.2f}",
                  flush=True)

    # Aggregate
    def agg(filter_fn):
        sel = [r for r in results if filter_fn(r)]
        if not sel:
            return None
        return {
            "n":            len(sel),
            "before_mean":  float(np.mean([r["clip_before"] for r in sel])),
            "after_mean":   float(np.mean([r["clip_after"]  for r in sel])),
            "delta_mean":   float(np.mean([r["delta_clip"]  for r in sel])),
        }

    summary = {
        "model":       MODEL_ID,
        "adapter":     ADAPTER,
        "caption_prefix": CAPTION_PREFIX,
        "by_category": {
            "eval_natural":  agg(lambda r: r["source"] == "eval"  and r["mode"] == "natural"),
            "eval_prefix":   agg(lambda r: r["source"] == "eval"  and r["mode"] == "prefix"),
            "human_natural": agg(lambda r: r["source"] == "human" and r["mode"] == "natural"),
            "human_prefix":  agg(lambda r: r["source"] == "human" and r["mode"] == "prefix"),
        },
        "by_source": {
            "eval":  agg(lambda r: r["source"] == "eval"),
            "human": agg(lambda r: r["source"] == "human"),
        },
        "by_mode": {
            "natural": agg(lambda r: r["mode"] == "natural"),
            "prefix":  agg(lambda r: r["mode"] == "prefix"),
        },
        "overall": agg(lambda r: True),
        "per_pair": results,
        "minutes":  round((time.time() - t0) / 60, 1),
    }

    out_json = OUTPUT_DIR / "naruto_qual_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + SEP)
    print("  NARUTO QUALITATIVE SUMMARY (CLIPScore: before -> after, delta)")
    print(SEP)
    for k, v in summary["by_category"].items():
        if v is None:
            continue
        print(f"  {k:>14s} (n={v['n']}): "
              f"{v['before_mean']:.2f} -> {v['after_mean']:.2f}  "
              f"({v['delta_mean']:+.2f})")
    print(f"  {'OVERALL':>14s} (n={summary['overall']['n']}): "
          f"{summary['overall']['before_mean']:.2f} -> "
          f"{summary['overall']['after_mean']:.2f}  "
          f"({summary['overall']['delta_mean']:+.2f})")
    print(SEP)
    print(f"  pairs saved : {len(results)} PNGs under {OUTPUT_DIR}")
    print(f"  results json: {out_json}")
    print(SEP)


if __name__ == "__main__":
    main()
