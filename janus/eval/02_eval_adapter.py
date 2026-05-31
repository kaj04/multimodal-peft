#!/usr/bin/env python3
"""
janus/eval/02_eval_adapter.py

Measure CROSS-MODAL FORGETTING (RQ1) from a LoRA adapter.

Loads base Janus-Pro-1B + a trained adapter, then evaluates BOTH capabilities
on the SAME held-out split, with the adapter OFF (before) and ON (after):
  - captioning (image -> text): CIDEr, BLEU-4
  - generation (text -> image): CLIPScore

The before/after comparison is perfectly controlled: identical model, data and
seed, the ONLY difference is whether the LoRA adapter is active (peft's
`disable_adapter()` toggles it). The deltas are one row of the transfer matrix:
"trained on captioning -> what happened to captioning AND to generation".

Saves a few before/after samples (captions and images) for eye inspection.

Run via 02_run_eval_adapter.sh (needs a GPU).
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (captioning_scores, CLIPScorer,  # noqa: E402
                     generate_caption, generate_image)

MODEL_ID   = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-1B")
ADAPTER    = os.environ.get(
    "ADAPTER",
    os.path.expanduser("~/multimodal-peft/janus/train/output/cap_1b_r8_valsplit/adapter_epoch1"))
COCO_DIR   = Path("/scratch-shared/tesr125338/coco")
VAL_IMAGES = COCO_DIR / "val2017"
VAL_CAPS   = COCO_DIR / "annotations" / "captions_val2017.json"

# Must match the training split so the eval set is the SAME held-out images.
TRAIN_N    = int(os.environ.get("TRAIN_N", "4000"))
EVAL_N     = int(os.environ.get("EVAL_N", "200"))
SEED       = int(os.environ.get("SEED", "42"))
GEN_EVAL_N = int(os.environ.get("GEN_EVAL_N", "30"))   # images are slow
N_SAVE     = 5

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "forgetting"

SEP = "=" * 64


def load_eval_split():
    data = json.loads(VAL_CAPS.read_text())
    id2file = {im["id"]: im["file_name"] for im in data["images"]}
    refs = defaultdict(list)
    for ann in data["annotations"]:
        refs[ann["image_id"]].append(ann["caption"])
    ids = [i for i in refs if (VAL_IMAGES / id2file[i]).exists()]
    random.Random(SEED).shuffle(ids)
    eval_ids = ids[TRAIN_N:TRAIN_N + EVAL_N]   # disjoint from training
    return eval_ids, id2file, refs


def run_eval(model, processor, eval_ids, id2file, refs, scorer, tag, save_dir):
    """Captioning (all eval_ids) + generation CLIPScore (first GEN_EVAL_N)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    # Captioning
    hyps, cap_samples = {}, []
    for k, img_id in enumerate(eval_ids):
        image = Image.open(VAL_IMAGES / id2file[img_id]).convert("RGB")
        hyps[img_id] = generate_caption(model, processor, image)
        if k < N_SAVE:
            cap_samples.append({"image": id2file[img_id], "caption": hyps[img_id]})
    cap = captioning_scores(refs, hyps)

    # Generation (fix seed so before/after see the same sampling noise)
    gen_ids = eval_ids[:GEN_EVAL_N]
    vals = []
    for k, img_id in enumerate(gen_ids):
        torch.manual_seed(SEED + k)
        prompt = refs[img_id][0]
        img = generate_image(model, processor, prompt)
        vals.append(scorer.score(img, prompt))
        if k < N_SAVE:
            img.save(save_dir / f"{k:02d}_{img_id}_{tag}.png")
    clip_mean = float(np.mean(vals))

    print(f"  [{tag}] CIDEr={cap['CIDEr']:.4f} BLEU_4={cap['BLEU_4']:.4f} "
          f"CLIPScore={clip_mean:.2f}", flush=True)
    return {"CIDEr": cap["CIDEr"], "BLEU_4": cap["BLEU_4"],
            "CLIPScore": clip_mean, "n_cap": cap["n"], "n_gen": len(vals),
            "caption_samples": cap_samples}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print("  Cross-modal forgetting eval (adapter OFF vs ON)")
    print(f"  adapter: {ADAPTER}")
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

    eval_ids, id2file, refs = load_eval_split()
    print(f"  eval images: {len(eval_ids)} (captioning), {GEN_EVAL_N} (generation)\n")

    t0 = time.time()
    # BEFORE: adapter disabled -> base model behavior
    print("  evaluating BEFORE (adapter disabled)...", flush=True)
    with model.disable_adapter():
        before = run_eval(model, processor, eval_ids, id2file, refs, scorer,
                          "before", OUTPUT_DIR / "samples")
    # AFTER: adapter active
    print("  evaluating AFTER (adapter active)...", flush=True)
    after = run_eval(model, processor, eval_ids, id2file, refs, scorer,
                     "after", OUTPUT_DIR / "samples")

    deltas = {
        "CIDEr":     after["CIDEr"] - before["CIDEr"],
        "BLEU_4":    after["BLEU_4"] - before["BLEU_4"],
        "CLIPScore": after["CLIPScore"] - before["CLIPScore"],
    }
    results = {"model": MODEL_ID, "adapter": ADAPTER,
               "trained_on": "captioning",
               "before": before, "after": after, "delta": deltas,
               "minutes": round((time.time() - t0) / 60, 1)}

    print("\n" + SEP)
    print("  FORGETTING SUMMARY  (trained on captioning)")
    print(SEP)
    print(f"  Captioning CIDEr : {before['CIDEr']:.4f} -> {after['CIDEr']:.4f}  "
          f"(Δ {deltas['CIDEr']:+.4f})   [target task, expect UP]")
    print(f"  Generation CLIP  : {before['CLIPScore']:.2f} -> {after['CLIPScore']:.2f}  "
          f"(Δ {deltas['CLIPScore']:+.2f})   [other modality, DOWN = forgetting]")
    print(SEP)

    out = OUTPUT_DIR / "forgetting_captioning.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results -> {out}")
    print(f"  Before/after samples -> {OUTPUT_DIR / 'samples'}")


if __name__ == "__main__":
    main()
