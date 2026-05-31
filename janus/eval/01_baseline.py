#!/usr/bin/env python3
"""
janus/eval/01_baseline.py

Phase 4 baseline: validate that our evaluation metrics actually work, by
running them on the pretrained Janus-Pro-1B (NO fine-tuning) over a small
subset of COCO val2017. The point is to prove the metric pipeline is sound
before we trust any post-LoRA number.

Metrics:
  - Captioning (image -> text): CIDEr, BLEU-4   (pycocoevalcap)
  - Generation (text -> image): CLIPScore        (CLIP)

Each block is isolated and reported independently. A few sample captions and
generated images are saved for human sanity-checking alongside the numbers.

Run via 01_run_baseline.sh (needs a GPU).
"""

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import captioning_scores, CLIPScorer  # noqa: E402

MODEL_ID    = "deepseek-community/Janus-Pro-1B"
COCO_DIR    = Path("/scratch-shared/tesr125338/coco")
VAL_IMAGES  = COCO_DIR / "val2017"
VAL_CAPS    = COCO_DIR / "annotations" / "captions_val2017.json"
OUTPUT_DIR  = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "baseline"

# Small by default so this is a fast metric-validation pass; scale up later.
N_CAPTION = int(os.environ.get("N_CAPTION", "100"))
N_GEN     = int(os.environ.get("N_GEN", "30"))
N_SAVE    = 5   # how many sample outputs to save for eyeballing

SEP = "=" * 64


# ---------------------------------------------------------------------------
# Generation helpers (validated in the smoke test)
# ---------------------------------------------------------------------------

def generate_caption(model, processor, image, max_new_tokens=64):
    messages = [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text", "text": "Describe this image in one sentence."}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, generation_mode="text")
    inputs = processor(text=prompt, images=[image], generation_mode="text",
                       return_tensors="pt").to(model.device, torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             generation_mode="text", do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def generate_image(model, processor, prompt_text):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, generation_mode="image")
    inputs = processor(text=prompt, generation_mode="image",
                       return_tensors="pt").to(model.device)
    model.generation_config.num_return_sequences = 1
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_mode="image",
                                 max_new_tokens=576, do_sample=True, use_cache=True)
        decoded = model.decode_image_tokens(outputs)
    decoded = decoded.detach().float().cpu()
    try:
        arr = np.asarray(processor.postprocess(list(decoded), return_tensors="np")["pixel_values"][0])
    except Exception:
        arr = decoded[0].numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if float(arr.min()) < 0.0:
            arr = (arr + 1.0) / 2.0
        if float(arr.max()) <= 1.0 + 1e-3:
            arr = (arr.clip(0, 1) * 255).round().astype("uint8")
        else:
            arr = arr.clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_val_references():
    data = json.loads(VAL_CAPS.read_text())
    id2file = {im["id"]: im["file_name"] for im in data["images"]}
    refs = defaultdict(list)
    for ann in data["annotations"]:
        refs[ann["image_id"]].append(ann["caption"])
    # keep only ids whose image file is actually on disk
    ids = [i for i in sorted(refs) if (VAL_IMAGES / id2file[i]).exists()]
    return ids, id2file, refs


# ---------------------------------------------------------------------------
# Eval blocks
# ---------------------------------------------------------------------------

def eval_captioning(model, processor, ids, id2file, refs):
    print(SEP); print(f"  Captioning baseline (N={N_CAPTION})"); print(SEP)
    sample_ids = ids[:N_CAPTION]
    hyps, samples = {}, []
    for k, img_id in enumerate(sample_ids):
        image = Image.open(VAL_IMAGES / id2file[img_id]).convert("RGB")
        cap = generate_caption(model, processor, image)
        hyps[img_id] = cap
        if k < N_SAVE:
            samples.append({"image": id2file[img_id], "generated": cap,
                            "reference": refs[img_id][0]})
        if (k + 1) % 25 == 0:
            print(f"    captioned {k+1}/{len(sample_ids)}")
    scores = captioning_scores(refs, hyps)
    print(f"  CIDEr={scores['CIDEr']:.4f}  BLEU_4={scores['BLEU_4']:.4f}  n={scores['n']}")
    return {"scores": scores, "samples": samples}


def eval_generation(model, processor, ids, id2file, refs):
    print(SEP); print(f"  Generation baseline (N={N_GEN}, CLIPScore)"); print(SEP)
    scorer = CLIPScorer()
    img_dir = OUTPUT_DIR / "gen_samples"
    img_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = ids[:N_GEN]
    vals = []
    for k, img_id in enumerate(sample_ids):
        prompt = refs[img_id][0]                     # use a real caption as prompt
        image = generate_image(model, processor, prompt)
        cs = scorer.score(image, prompt)
        vals.append(cs)
        if k < N_SAVE:
            image.save(img_dir / f"{k:02d}_{img_id}.png")
            (img_dir / f"{k:02d}_{img_id}.txt").write_text(prompt, encoding="utf-8")
        if (k + 1) % 10 == 0:
            print(f"    generated {k+1}/{len(sample_ids)}  (running mean CLIPScore={np.mean(vals):.2f})")
    return {"clipscore_mean": float(np.mean(vals)),
            "clipscore_std": float(np.std(vals)), "n": len(vals)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print("  Janus-Pro-1B baseline metric validation")
    print(f"  torch {torch.__version__}  CUDA {torch.cuda.is_available()}")
    print(f"  output: {OUTPUT_DIR}")
    print(SEP + "\n")

    from transformers import JanusForConditionalGeneration, JanusProcessor
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    model = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu").eval()

    ids, id2file, refs = load_val_references()
    print(f"  val images with captions available: {len(ids)}\n")

    results = {"model": MODEL_ID, "n_caption": N_CAPTION, "n_gen": N_GEN,
               "metrics": {}}

    t0 = time.time()
    try:
        cap = eval_captioning(model, processor, ids, id2file, refs)
        results["metrics"]["captioning"] = {"status": "PASS", **cap["scores"]}
        results["caption_samples"] = cap["samples"]
    except Exception as e:
        print("  captioning FAILED:"); traceback.print_exc()
        results["metrics"]["captioning"] = {"status": "FAIL", "error": str(e)}
    print(f"  (captioning block: {time.time()-t0:.0f}s)\n")

    t0 = time.time()
    try:
        gen = eval_generation(model, processor, ids, id2file, refs)
        results["metrics"]["generation"] = {"status": "PASS", **gen}
        print(f"  CLIPScore mean={gen['clipscore_mean']:.2f} "
              f"std={gen['clipscore_std']:.2f}  n={gen['n']}")
    except Exception as e:
        print("  generation FAILED:"); traceback.print_exc()
        results["metrics"]["generation"] = {"status": "FAIL", "error": str(e)}
    print(f"  (generation block: {time.time()-t0:.0f}s)\n")

    print(SEP); print("  BASELINE SUMMARY"); print(SEP)
    cap_m = results["metrics"].get("captioning", {})
    gen_m = results["metrics"].get("generation", {})
    if cap_m.get("status") == "PASS":
        print(f"  Captioning : CIDEr={cap_m['CIDEr']:.4f}  BLEU_4={cap_m['BLEU_4']:.4f}")
    else:
        print(f"  Captioning : FAIL ({cap_m.get('error','')})")
    if gen_m.get("status") == "PASS":
        print(f"  Generation : CLIPScore={gen_m['clipscore_mean']:.2f}")
    else:
        print(f"  Generation : FAIL ({gen_m.get('error','')})")
    print(SEP)

    out = OUTPUT_DIR / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results written to {out}")
    print(f"  Sample captions + generated images under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
