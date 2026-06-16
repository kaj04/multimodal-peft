#!/usr/bin/env python3
"""
janus/eval/14_probe_image_distance.py

Pick the IMAGE style for the generation-direction PEFT the same principled way
we picked poetry for the text direction: by distance from the base. This is
the VQ-token analogue of the text PPL probe (11_probe_text_ppl.py), and it
exists to kill the central validity threat -- "maybe the base is already good
at this style, so PEFT barely moves the weights and of course nothing forgets".

For each candidate style we compute the BASE Janus-Pro-7B mean cross-entropy on
the image's VQ tokens, teacher-forced, using the EXACT training objective path
(generation_loss in 02_train_generation.py): encode image -> VQ tokens; predict
them with [text embeds | image embeds[:-1]] -> generation_head -> CE. A high CE
means the base is "surprised" by these images = the style/domain is distant =
PEFT will be FORCED to move the weights a lot = the strongest, least-trivial
forgetting test. (We don't try to disentangle style from content: "distance"
here is exactly how badly the base models the target, which is what determines
the intervention strength.)

Controls for confounds the same way the text probe does:
  - FIXED neutral caption for every style (constant conditioning), and
  - EQUAL number of images per style.

`coco_photo` is the FAMILIAR floor (natural photos = what Janus generates well)
so every style above it is "more distant" by that margin.

Output: a ranked table (low -> high CE) + json. The highest-CE style is the
recommended image-PEFT target. Run via 14_run_probe_image_distance.sh.
"""

import json
import os
import random
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
SEED     = int(os.environ.get("SEED", "42"))
N_IMAGES = int(os.environ.get("N_IMAGES", "64"))      # equal budget per style
CAPTION  = os.environ.get("CAPTION", "a picture")     # fixed neutral conditioning
SMOKE    = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    N_IMAGES = 4

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "image_distance_probe"

SEP = "=" * 70

# Candidate image styles. local_glob -> read PNG/JPG from disk; otherwise an HF
# dataset (image column auto-detected). A row that fails to load is skipped with
# a warning; the probe still ranks the rest. Edit freely.
STYLES = [
    {"name": "coco_photo",
     "local_glob": "/scratch-shared/tesr125338/coco/val2017/*.jpg",
     "note": "natural photos (FAMILIAR control / floor)"},

    {"name": "pixelart", "hf": "jainr3/diffusiondb-pixelart",
     "config": "2k_random_2k", "split": "train", "image_col": "image",
     "note": "DiffusionDB images pixelated post-hoc"},

    {"name": "naruto", "hf": "lambdalabs/naruto-blip-captions",
     "split": "train", "image_col": "image",
     "note": "anime (Naruto) frames"},

    {"name": "ukiyoe", "hf": "KappaNeuro/ukiyo-e-art", "split": "train",
     "note": "Japanese woodblock prints"},

    {"name": "xray_medical", "hf": "keremberke/chest-xray-classification",
     "config": "full", "split": "train",
     "note": "chest X-rays (alien domain)"},
]


# ---------------------------------------------------------------------------
# Image loading (robust): local glob or HF dataset, equal budget, skip-on-fail
# ---------------------------------------------------------------------------

def _detect_image_col(ds):
    try:
        from datasets import Image as HFImage
        for c, f in ds.features.items():
            if isinstance(f, HFImage):
                return c
    except Exception:
        pass
    row = ds[0]
    for c, v in row.items():
        if isinstance(v, Image.Image):
            return c
        if isinstance(v, dict) and ("bytes" in v or "path" in v):
            return c
    raise ValueError("no image column found")


def load_style_images(spec, n, rng):
    if spec.get("local_glob"):
        paths = glob(spec["local_glob"])
        rng.shuffle(paths)
        imgs = []
        for p in paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                continue
            if len(imgs) >= n:
                break
        return imgs, "local"

    from datasets import load_dataset
    load_args = [spec["hf"]] + ([spec["config"]] if spec.get("config") else [])
    kwargs = {"split": spec.get("split", "train")}
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]
    ds = load_dataset(*load_args, **kwargs)
    col = spec.get("image_col")
    if col is None or col not in ds.column_names:
        col = _detect_image_col(ds)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    imgs = []
    for i in idx:
        try:
            imgs.append(ds[i][col].convert("RGB"))
        except Exception:
            continue
        if len(imgs) >= n:
            break
    return imgs, col


# ---------------------------------------------------------------------------
# Base VQ-token CE  (identical path to generation_loss in 02_train_generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def image_ce(base, processor, image, caption):
    inner = base.model
    device = base.device
    dtype = next(base.parameters()).dtype

    messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, generation_mode="image")
    text_ids = processor(text=prompt, generation_mode="image",
                         return_tensors="pt").to(device)["input_ids"]
    pixel_values = processor.image_processor(
        images=[image], return_tensors="pt")["pixel_values"].to(device, dtype)
    image_tokens = inner.vqmodel.encode(pixel_values).image_tokens.reshape(1, -1)

    text_embeds = base.get_input_embeddings()(text_ids)
    image_embeds = base.prepare_embeddings_for_image_generation(image_tokens)
    inputs_embeds = torch.cat([text_embeds, image_embeds[:, :-1, :]], dim=1)
    Lt = text_embeds.shape[1]

    hidden = inner(inputs_embeds=inputs_embeds).last_hidden_state
    img_hidden = hidden[:, Lt - 1:, :]
    logits = inner.generation_head(img_hidden).float()
    targets = image_tokens.reshape(-1).long()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets).item()


def style_distance(base, processor, images):
    losses = [image_ce(base, processor, im, CAPTION) for im in images]
    mean_ce = float(np.mean(losses))
    return {"mean_CE": mean_ce, "PPL": float(np.exp(mean_ce)), "n_images": len(losses)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    print(SEP)
    print(f"  Image-style distance probe (VQ-token CE) | base {MODEL_ID}")
    print(f"  budget = {N_IMAGES} images/style, caption={CAPTION!r} (constant)")
    print(SEP + "\n", flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    base.eval()

    rows = []
    for spec in STYLES:
        name = spec["name"]
        src = spec.get("local_glob") or spec["hf"]
        print(f"  [{name}] loading {src} ...", flush=True)
        try:
            images, col = load_style_images(spec, N_IMAGES, rng)
            if not images:
                raise ValueError("no images loaded")
        except Exception as e:
            print(f"    SKIP {name}: {type(e).__name__}: {e}\n", flush=True)
            continue
        t0 = time.time()
        res = style_distance(base, processor, images)
        res.update({"name": name, "source": src, "image_col": col,
                    "note": spec.get("note", "")})
        rows.append(res)
        print(f"    img_col={col!r}  n={res['n_images']}  "
              f"mean_CE={res['mean_CE']:.4f}  PPL={res['PPL']:.2f}  "
              f"({time.time()-t0:.0f}s)\n", flush=True)

    rows.sort(key=lambda r: r["PPL"])
    ctrl = next((r for r in rows if r["name"] == "coco_photo"),
                rows[0] if rows else None)

    lines = [SEP, f"IMAGE-STYLE DISTANCE PROBE   base {MODEL_ID}",
             f"budget = {N_IMAGES} images/style, caption={CAPTION!r}", SEP,
             f"{'rank':>4}  {'style':<16}{'PPL':>10}{'mean_CE':>10}"
             f"{'xControl':>10}  note", "-" * len(SEP)]
    for k, r in enumerate(rows):
        xctrl = r["PPL"] / ctrl["PPL"] if ctrl else float("nan")
        lines.append(f"{k+1:>4}  {r['name']:<16}{r['PPL']:>10.2f}"
                     f"{r['mean_CE']:>10.4f}{xctrl:>9.2f}x  {r['note']}")
    lines.append("-" * len(SEP))
    if rows:
        far = rows[-1]
        lines.append(f"  MOST DISTANT (recommended image-PEFT target): "
                     f"{far['name']}  PPL={far['PPL']:.2f}  "
                     f"({far['PPL']/ctrl['PPL']:.2f}x control)")
        lines.append(f"  -> source={far['source']}  image_col={far['image_col']}")
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    (OUTPUT_DIR / "image_distance.txt").write_text(report + "\n", encoding="utf-8")
    (OUTPUT_DIR / "image_distance.json").write_text(
        json.dumps({"model": MODEL_ID, "n_images": N_IMAGES, "caption": CAPTION,
                    "styles": rows}, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUTPUT_DIR}/image_distance.txt (+ .json)")


if __name__ == "__main__":
    main()
