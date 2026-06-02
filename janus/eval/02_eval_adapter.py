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
# Which direction this adapter was trained on ("captioning" or "generation").
# Determines the target task (expected to improve) vs the probe (forgetting).
TRAINED_ON = os.environ.get("TRAINED_ON", "captioning")
# Optional suffix on output filenames + samples dir, so sweeps don't overwrite
# each other's JSON. Example: OUTPUT_TAG="r4" -> forgetting_captioning_r4.json
OUTPUT_TAG = os.environ.get("OUTPUT_TAG", "")

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
    """Captioning (all eval_ids) + generation CLIPScore (first GEN_EVAL_N).

    For captioning we report:
      - CIDEr, BLEU-4 : n-gram match against COCO references (style + content)
      - CLIPScore_cap : CLIP cosine between the image and the GENERATED caption
                        -- a content-aware probe insensitive to wording style.
                        Useful because CIDEr can rise from purely stylistic
                        adaptation toward terse COCO references.

    For generation we report:
      - CLIPScore_gen : CLIP cosine between the GENERATED image and its prompt
                        (the cross-modal probe used for forgetting).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Captioning: generate captions on all eval ids
    hyps, cap_samples, cap_clip_vals = {}, [], []
    for k, img_id in enumerate(eval_ids):
        image = Image.open(VAL_IMAGES / id2file[img_id]).convert("RGB")
        gen_cap = generate_caption(model, processor, image)
        hyps[img_id] = gen_cap
        # CLIPScore on (image, generated caption): content-aware probe
        cap_clip_vals.append(scorer.score(image, gen_cap))
        if k < N_SAVE:
            cap_samples.append({"image": id2file[img_id], "caption": gen_cap})
    cap = captioning_scores(refs, hyps)
    clip_cap = float(np.mean(cap_clip_vals))

    # --- Generation: CLIPScore on (generated image, prompt)
    gen_ids = eval_ids[:GEN_EVAL_N]
    gen_clip_vals = []
    for k, img_id in enumerate(gen_ids):
        torch.manual_seed(SEED + k)
        prompt = refs[img_id][0]
        img = generate_image(model, processor, prompt)
        gen_clip_vals.append(scorer.score(img, prompt))
        if k < N_SAVE:
            img.save(save_dir / f"{k:02d}_{img_id}_{tag}.png")
    clip_gen = float(np.mean(gen_clip_vals))

    print(f"  [{tag}] CIDEr={cap['CIDEr']:.4f} BLEU_4={cap['BLEU_4']:.4f} "
          f"CLIPScore_cap={clip_cap:.2f} CLIPScore_gen={clip_gen:.2f}",
          flush=True)
    return {"CIDEr":          cap["CIDEr"],
            "BLEU_4":         cap["BLEU_4"],
            "CLIPScore":      clip_gen,     # kept for backward-compatibility
            "CLIPScore_gen":  clip_gen,
            "CLIPScore_cap":  clip_cap,
            "n_cap": cap["n"], "n_gen": len(gen_clip_vals),
            "caption_samples": cap_samples}


def reference_clipscore(processor, scorer, eval_ids, id2file, refs):
    """CLIPScore of (image, FIRST COCO reference caption) on the same eval set.
    A model-independent ceiling: how well do humans' captions align with the
    images under CLIP? Computed once per run."""
    vals = []
    for img_id in eval_ids:
        image = Image.open(VAL_IMAGES / id2file[img_id]).convert("RGB")
        vals.append(scorer.score(image, refs[img_id][0]))
    return float(np.mean(vals))


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
    tag_suffix = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
    samples_dir = OUTPUT_DIR / f"samples_{TRAINED_ON}{tag_suffix}"
    print("  evaluating BEFORE (adapter disabled)...", flush=True)
    with model.disable_adapter():
        before = run_eval(model, processor, eval_ids, id2file, refs, scorer,
                          "before", samples_dir)
    # AFTER: adapter active
    print("  evaluating AFTER (adapter active)...", flush=True)
    after = run_eval(model, processor, eval_ids, id2file, refs, scorer,
                     "after", samples_dir)

    # Model-independent ceiling: how aligned are the human COCO references?
    print("  measuring reference (human) CLIPScore ceiling...", flush=True)
    ref_clip_cap = reference_clipscore(processor, scorer, eval_ids, id2file, refs)

    deltas = {
        "CIDEr":          after["CIDEr"]          - before["CIDEr"],
        "BLEU_4":         after["BLEU_4"]         - before["BLEU_4"],
        "CLIPScore":      after["CLIPScore"]      - before["CLIPScore"],       # back-compat
        "CLIPScore_gen":  after["CLIPScore_gen"]  - before["CLIPScore_gen"],
        "CLIPScore_cap":  after["CLIPScore_cap"]  - before["CLIPScore_cap"],
    }
    results = {"model": MODEL_ID, "adapter": ADAPTER,
               "trained_on": TRAINED_ON,
               "before": before, "after": after, "delta": deltas,
               "reference_CLIPScore_cap": ref_clip_cap,
               "minutes": round((time.time() - t0) / 60, 1)}

    # Label which metric is the trained target vs the cross-modal probe
    cap_role = "target task, expect UP" if TRAINED_ON == "captioning" \
        else "probe modality, DOWN = forgetting"
    gen_role = "target task, expect UP" if TRAINED_ON == "generation" \
        else "probe modality, DOWN = forgetting"

    print("\n" + SEP)
    print(f"  FORGETTING SUMMARY  (trained on {TRAINED_ON})")
    print(SEP)
    print(f"  Captioning CIDEr     : {before['CIDEr']:.4f} -> {after['CIDEr']:.4f}  "
          f"(Δ {deltas['CIDEr']:+.4f})   [{cap_role}, style+content]")
    print(f"  Captioning CLIP(img,gen-cap): {before['CLIPScore_cap']:.2f} -> {after['CLIPScore_cap']:.2f}  "
          f"(Δ {deltas['CLIPScore_cap']:+.2f})   [{cap_role}, content-aware]")
    print(f"  Generation CLIP(gen-img,prompt): {before['CLIPScore_gen']:.2f} -> {after['CLIPScore_gen']:.2f}  "
          f"(Δ {deltas['CLIPScore_gen']:+.2f})   [{gen_role}]")
    print(f"  Reference CLIP ceiling (human caps): {ref_clip_cap:.2f}")
    print(SEP)

    out = OUTPUT_DIR / f"forgetting_{TRAINED_ON}{tag_suffix}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results -> {out}")
    print(f"  Before/after samples -> {OUTPUT_DIR / 'samples'}")


if __name__ == "__main__":
    main()
