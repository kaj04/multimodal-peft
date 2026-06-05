#!/usr/bin/env python3
"""
janus/eval/05_pokemon_qualitative.py

Qualitative side-by-side comparison of a Pokemon-fine-tuned adapter against the
base Janus-Pro-1B, on PROMPTS held-out from the training split.

For each prompt:
  - generate one image with the adapter OFF (base model)
  - generate one image with the adapter ON  (fine-tuned)
  - same seed -> identical sampling noise -> only the adapter changes
  - save both PNGs side-by-side under output/pokemon_qual/

Also computes mean CLIPScore(image, prompt) for before vs after on the same
prompts (N>=15 by default), so we get a small numerical claim to back the
visual inspection.

Run via SLURM (needs a GPU):

  sbatch --export=ALL,ADAPTER=$HOME/multimodal-peft/janus/train/output/gen_1b_pokemon_r8_ce_constant/adapter_epoch5 \
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import CLIPScorer, generate_image

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-1B")
ADAPTER  = os.environ.get(
    "ADAPTER",
    os.path.expanduser(
        "~/multimodal-peft/janus/train/output/gen_1b_pokemon_r8_ce_constant/adapter_epoch5"
    ),
)

# Same SEED as the trainer so the eval split is the same set of held-out items
SEED       = int(os.environ.get("SEED", "42"))
TRAIN_N    = int(os.environ.get("TRAIN_N", "4000"))   # matches the trainer cap
EVAL_N     = int(os.environ.get("EVAL_N", "200"))     # matches the trainer cap
N_PROMPTS  = int(os.environ.get("N_PROMPTS", "20"))   # how many to evaluate
N_SAVE     = int(os.environ.get("N_SAVE", "10"))      # how many to save as PNGs

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "pokemon_qual"

SEP = "=" * 64


HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME",
                                 "lambdalabs/naruto-blip-captions")
HF_TEXT_COL     = os.environ.get("HF_TEXT_COL", "text")


def load_pokemon_eval_split():
    """Re-derive the SAME held-out eval split the trainer used. Defaults to the
    naruto fallback; override HF_DATASET_NAME / HF_TEXT_COL to match the
    trainer's choices."""
    from datasets import load_dataset
    print(f"  loading {HF_DATASET_NAME} (text_col={HF_TEXT_COL!r}) ...", flush=True)
    ds = load_dataset(HF_DATASET_NAME, split="train")
    ids = list(range(len(ds)))
    random.Random(SEED).shuffle(ids)
    n_eval  = min(EVAL_N, 80)
    n_train = min(TRAIN_N, len(ids) - n_eval)
    eval_ids = ids[n_train:n_train + n_eval]
    prompts = [ds[i][HF_TEXT_COL] for i in eval_ids]
    return prompts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  Pokemon qualitative comparison")
    print(f"  adapter: {ADAPTER}")
    print(f"  N_PROMPTS={N_PROMPTS}  N_SAVE={N_SAVE}  output: {OUTPUT_DIR}")
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

    prompts = load_pokemon_eval_split()[:N_PROMPTS]
    print(f"  using {len(prompts)} held-out prompts\n")

    before_scores, after_scores = [], []
    samples = []
    t0 = time.time()

    # -- BEFORE: adapter disabled (base model behaviour)
    print("  generating BEFORE (adapter disabled)...", flush=True)
    before_imgs = []
    with model.disable_adapter():
        for k, prompt in enumerate(prompts):
            torch.manual_seed(SEED + k)
            img = generate_image(model, processor, prompt)
            before_imgs.append(img)
            before_scores.append(scorer.score(img, prompt))
            if (k + 1) % 5 == 0:
                print(f"    before {k+1}/{len(prompts)} "
                      f"running CLIPScore={np.mean(before_scores):.2f}", flush=True)

    # -- AFTER: adapter active
    print("\n  generating AFTER (adapter active)...", flush=True)
    after_imgs = []
    for k, prompt in enumerate(prompts):
        torch.manual_seed(SEED + k)   # same noise as the matched 'before' sample
        img = generate_image(model, processor, prompt)
        after_imgs.append(img)
        after_scores.append(scorer.score(img, prompt))
        if (k + 1) % 5 == 0:
            print(f"    after {k+1}/{len(prompts)} "
                  f"running CLIPScore={np.mean(after_scores):.2f}", flush=True)

    # -- Save the first N_SAVE pairs as side-by-side PNGs for eye inspection
    print(f"\n  saving the first {N_SAVE} before/after pairs...", flush=True)
    for k in range(min(N_SAVE, len(prompts))):
        b, a = before_imgs[k], after_imgs[k]
        # Side-by-side canvas: same height, doubled width
        w = max(b.width, a.width)
        h = max(b.height, a.height)
        canvas = Image.new("RGB", (w * 2 + 8, h), (255, 255, 255))
        canvas.paste(b.resize((w, h)), (0, 0))
        canvas.paste(a.resize((w, h)), (w + 8, 0))
        canvas.save(OUTPUT_DIR / f"{k:02d}_pair.png")
        b.save(OUTPUT_DIR / f"{k:02d}_before.png")
        a.save(OUTPUT_DIR / f"{k:02d}_after.png")
        samples.append({
            "k": k, "prompt": prompts[k],
            "clip_before": before_scores[k],
            "clip_after":  after_scores[k],
        })

    mean_before = float(np.mean(before_scores))
    mean_after  = float(np.mean(after_scores))
    delta = mean_after - mean_before

    results = {
        "model": MODEL_ID, "adapter": ADAPTER, "n_prompts": len(prompts),
        "CLIPScore_before_mean": mean_before,
        "CLIPScore_after_mean":  mean_after,
        "delta_CLIPScore":       delta,
        "CLIPScore_before_std":  float(np.std(before_scores)),
        "CLIPScore_after_std":   float(np.std(after_scores)),
        "samples": samples,
        "minutes": round((time.time() - t0) / 60, 1),
    }

    out_json = OUTPUT_DIR / "pokemon_qual_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + SEP)
    print("  POKEMON QUALITATIVE SUMMARY")
    print(SEP)
    print(f"  CLIPScore before : {mean_before:.2f}")
    print(f"  CLIPScore after  : {mean_after:.2f}")
    print(f"  Delta            : {delta:+.2f}   (target task; UP = adapter helps)")
    print(f"  Pairs saved      : {min(N_SAVE, len(prompts))} under {OUTPUT_DIR}")
    print(f"  Results json     : {out_json}")
    print(SEP)


if __name__ == "__main__":
    main()
