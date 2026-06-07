#!/usr/bin/env python3
"""
janus/eval/06_metrics_panel.py

Quantitative metrics panel for evaluating an image-generation LoRA adapter
on the Naruto BLIP captions held-out split. We compute SEVERAL metrics
(not just CLIPScore) so we can correlate them with the eye-judgement on
the qualitative pairs and identify which numbers are actually faithful
to our domain (stylised anime art).

Pipeline:
  1. Load base + LoRA adapter
  2. Generate N images on held-out prompts WITH ADAPTER OFF (base reference)
  3. Generate N images on the SAME prompts WITH ADAPTER ON
  4. Compute each metric BEFORE and AFTER -> delta tells us which direction
     the adapter is moving the model

Metrics implemented (see paper refs in the docstring of each):
  - FID                 : Inception features, standard generative-quality metric
  - ArtFID              : like FID but with style-aware Inception (paper 2207.12280);
                          fallback to "FID on CLIP features" if art-fid lib not present
  - CMMD                : Jayasumana 2024, CLIP-based MMD; robust on small N
  - CLIP-IQA            : per-image quality from CLIP (no reference needed)
  - CLIP DirSim         : Gal et al. 2022, measures whether (after-before) direction
                          aligns with target text direction ("photo"->"anime sprite
                          in Naruto style"). Probably the most domain-relevant.
  - DINO self-similarity: average cosine in DINO feature space between generated
                          and reference Naruto images. Less Inception-biased.

Lazy-imports each library so a missing optional dep only disables that one
metric instead of crashing the whole run.

Run via SLURM (needs GPU + the metric libraries):
    sbatch --export=ALL,ADAPTER=/path,MODEL_ID=... 06_run_metrics_panel.sh
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
from metrics import generate_image  # noqa: E402  (reuse the shared generator)

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
CAPTION_PREFIX = os.environ.get("CAPTION_PREFIX", "anime sprite in Naruto style, ")

SEED      = int(os.environ.get("SEED", "42"))
TRAIN_N   = int(os.environ.get("TRAIN_N", "4000"))
EVAL_N    = int(os.environ.get("EVAL_N",  "80"))
N_GEN     = int(os.environ.get("N_GEN",   "200"))   # how many to generate per condition

HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME",
                                 "lambdalabs/naruto-blip-captions")
HF_IMAGE_COL    = os.environ.get("HF_IMAGE_COL", "image")
HF_TEXT_COL     = os.environ.get("HF_TEXT_COL",  "text")

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "metrics_panel"

SEP = "=" * 64


# ---------------------------------------------------------------------------
# Data: re-derive the SAME held-out eval split the trainer used, AND keep
# the corresponding reference real images for distribution-distance metrics.
# ---------------------------------------------------------------------------

def load_eval_split():
    from datasets import load_dataset
    print(f"  loading {HF_DATASET_NAME} ...", flush=True)
    ds = load_dataset(HF_DATASET_NAME, split="train")
    ids = list(range(len(ds)))
    random.Random(SEED).shuffle(ids)
    n_eval  = min(EVAL_N, 80)
    n_train = min(TRAIN_N, len(ids) - n_eval)
    eval_ids = ids[n_train:n_train + n_eval]
    prompts = [ds[i][HF_TEXT_COL]  for i in eval_ids]
    refs    = [ds[i][HF_IMAGE_COL].convert("RGB") for i in eval_ids]
    # If N_GEN > len(eval_ids), recycle prompts (so we can hit N=200 even
    # with only 80 unique eval items by oversampling 2.5x with different seeds).
    extended_prompts = (prompts * ((N_GEN // len(prompts)) + 1))[:N_GEN]
    return extended_prompts, prompts, refs


# ---------------------------------------------------------------------------
# Metric implementations (each is a self-contained function returning a float
# or dict; lazy-imports keep the import surface small).
# ---------------------------------------------------------------------------

def metric_fid(gen_imgs, ref_imgs):
    """Standard FID via cleanfid (Parmar et al. 2022, robust impl)."""
    try:
        from cleanfid import fid
    except ImportError:
        return {"value": None, "error": "cleanfid not installed (pip install clean-fid)"}
    import tempfile
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for i, img in enumerate(gen_imgs): img.save(f"{a}/{i:04d}.png")
        for i, img in enumerate(ref_imgs): img.save(f"{b}/{i:04d}.png")
        score = fid.compute_fid(a, b, mode="clean", num_workers=2)
    return {"value": float(score)}


def metric_artfid(gen_imgs, ref_imgs):
    """ArtFID (Wright & Ommer 2022, arxiv 2207.12280): FID using an Inception
    net fine-tuned on WikiArt. More faithful to stylised-art quality than the
    ImageNet-pretrained vanilla Inception. Falls back to a NOTE if the
    optional art-fid package isn't installed."""
    try:
        from art_fid import compute_art_fid
    except ImportError:
        return {"value": None,
                "error": "art-fid not installed (pip install art-fid). "
                         "Fallback recommendation: report FID + CMMD."}
    import tempfile
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for i, img in enumerate(gen_imgs): img.save(f"{a}/{i:04d}.png")
        for i, img in enumerate(ref_imgs): img.save(f"{b}/{i:04d}.png")
        score = compute_art_fid(a, b)
    return {"value": float(score)}


def metric_cmmd(gen_imgs, ref_imgs):
    """CMMD (Jayasumana et al. 2024, CVPR): MMD distance in CLIP feature
    space. Designed to fix FID's known issues on small N and on stylised
    domains. Implementation here uses CLIP ViT-L/14 features and the
    Gaussian RBF kernel (sigma chosen by median heuristic)."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        return {"value": None, "error": "transformers not installed"}

    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()

    @torch.no_grad()
    def feats(imgs):
        out = []
        for i in range(0, len(imgs), 16):
            batch = imgs[i:i + 16]
            x = proc(images=batch, return_tensors="pt").to("cuda")
            f = clip.get_image_features(**x)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu())
        return torch.cat(out, 0)

    x = feats(gen_imgs).numpy()
    y = feats(ref_imgs).numpy()
    # Median-heuristic sigma
    from scipy.spatial.distance import pdist
    sigma = float(np.median(pdist(np.vstack([x, y])))) or 1.0
    def k(a, b):
        sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-sq / (2 * sigma ** 2))
    cmmd = k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()
    return {"value": float(cmmd), "sigma": sigma}


def metric_clip_iqa(gen_imgs):
    """CLIP-IQA (Wang et al. 2022): per-image quality score from a CLIP
    prompt-based probe ("good photo" vs "bad photo"). No reference needed.
    Returns mean and std across the generated set."""
    try:
        import pyiqa
    except ImportError:
        return {"value": None, "error": "pyiqa not installed (pip install pyiqa)"}
    iqa = pyiqa.create_metric("clipiqa", device="cuda")
    scores = []
    for img in gen_imgs:
        # pyiqa wants tensor input; convert PIL -> tensor
        import torchvision.transforms.functional as TF
        t = TF.to_tensor(img).unsqueeze(0).to("cuda")
        scores.append(float(iqa(t).item()))
    return {"value": float(np.mean(scores)), "std": float(np.std(scores))}


def metric_clip_dirsim(before_imgs, after_imgs,
                       text_source="a photograph of a person",
                       text_target="an anime sprite in Naruto style"):
    """CLIP Directional Similarity (Gal et al. 2022, StyleGAN-NADA paper):
    cosine similarity between the image-edit direction and the text-edit
    direction. High value = the adapter is moving images in the SEMANTIC
    direction we wanted ('photo' -> 'anime sprite Naruto'). Per-image, then
    averaged."""
    from transformers import CLIPModel, CLIPProcessor
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()

    @torch.no_grad()
    def txt_feat(text):
        x = proc(text=[text], return_tensors="pt", padding=True).to("cuda")
        f = clip.get_text_features(**x)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def img_feat(img):
        x = proc(images=[img], return_tensors="pt").to("cuda")
        f = clip.get_image_features(**x)
        return f / f.norm(dim=-1, keepdim=True)

    t_src = txt_feat(text_source)
    t_tgt = txt_feat(text_target)
    text_dir = (t_tgt - t_src)
    text_dir = text_dir / text_dir.norm(dim=-1, keepdim=True)

    sims = []
    for b_img, a_img in zip(before_imgs, after_imgs):
        i_src = img_feat(b_img)
        i_tgt = img_feat(a_img)
        img_dir = (i_tgt - i_src)
        n = img_dir.norm(dim=-1, keepdim=True)
        if n.item() < 1e-8:
            continue   # identical images -> direction undefined
        img_dir = img_dir / n
        sims.append(float((img_dir * text_dir).sum().item()))
    return {"value": float(np.mean(sims)) if sims else None,
            "std":   float(np.std(sims))  if sims else None,
            "n":     len(sims)}


def metric_dino_sim(gen_imgs, ref_imgs):
    """Mean cosine similarity in DINOv2 feature space, between each generated
    image and its NEAREST reference. Less Inception-biased than FID; good for
    'how close are my generations to the real distribution' on small N."""
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        return {"value": None, "error": "transformers required"}
    try:
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
    except Exception as e:
        return {"value": None, "error": f"could not load dinov2: {e}"}

    @torch.no_grad()
    def feats(imgs):
        out = []
        for i in range(0, len(imgs), 8):
            batch = imgs[i:i + 8]
            x = proc(images=batch, return_tensors="pt").to("cuda")
            f = dino(**x).last_hidden_state[:, 0]   # CLS token
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu())
        return torch.cat(out, 0)

    g = feats(gen_imgs)
    r = feats(ref_imgs)
    sim_matrix = g @ r.T            # (N_gen, N_ref)
    best = sim_matrix.max(dim=1).values
    return {"value": float(best.mean().item()),
            "std":   float(best.std().item())}


# ---------------------------------------------------------------------------
# Main: generate before/after sets, compute all metrics, write JSON.
# ---------------------------------------------------------------------------

def generate_set(model, processor, prompts, with_adapter: bool):
    """Generate len(prompts) images; same seed schedule so before/after pairs
    have matched sampling noise (only adapter differs)."""
    imgs = []
    if with_adapter:
        for k, p in enumerate(prompts):
            torch.manual_seed(SEED + k)
            imgs.append(generate_image(model, processor, p))
    else:
        with model.disable_adapter():
            for k, p in enumerate(prompts):
                torch.manual_seed(SEED + k)
                imgs.append(generate_image(model, processor, p))
    return imgs


def main():
    if not ADAPTER:
        raise SystemExit("ADAPTER env var required (path to adapter dir)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = Path(ADAPTER).parent.name + "__" + Path(ADAPTER).name
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print(f"  metrics panel | MODEL={MODEL_ID}")
    print(f"  ADAPTER       : {ADAPTER}")
    print(f"  CAPTION_PREFIX: {CAPTION_PREFIX!r}")
    print(f"  N_GEN         : {N_GEN}")
    print(f"  OUTPUT_DIR    : {out_dir}")
    print(SEP + "\n")

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    gen_prompts, unique_prompts, real_imgs = load_eval_split()
    # Prepend the trigger phrase to the generation prompts (we generate IN-DISTRIBUTION
    # with respect to training; the comparison is then "does the adapter make
    # in-distribution outputs better than the base?").
    gen_prompts_pref = [CAPTION_PREFIX + p for p in gen_prompts]
    print(f"  generating {N_GEN} images BEFORE (adapter off)...", flush=True)
    t0 = time.time()
    before = generate_set(model, processor, gen_prompts_pref, with_adapter=False)
    print(f"  {len(before)} generated in {(time.time()-t0)/60:.1f} min")
    print(f"  generating {N_GEN} images AFTER  (adapter on)...", flush=True)
    t0 = time.time()
    after = generate_set(model, processor, gen_prompts_pref, with_adapter=True)
    print(f"  {len(after)} generated in {(time.time()-t0)/60:.1f} min\n")

    # Save a sample grid for sanity
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    for k in range(min(10, len(before))):
        before[k].save(sample_dir / f"{k:02d}_before.png")
        after[k].save(sample_dir / f"{k:02d}_after.png")

    # Use the held-out REAL images as the reference distribution. We need at
    # least len(gen) reals to compare; recycle if short.
    real_extended = (real_imgs * ((N_GEN // len(real_imgs)) + 1))[:N_GEN]

    results = {"model": MODEL_ID, "adapter": ADAPTER,
               "caption_prefix": CAPTION_PREFIX, "n_gen": N_GEN,
               "before": {}, "after": {}, "delta": {}}

    def run_metric(name, fn, *args, **kwargs):
        print(f"  [{name}] computing ...", flush=True)
        t = time.time()
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            out = {"value": None, "error": f"{type(e).__name__}: {e}"}
        print(f"    -> {out}  ({time.time()-t:.1f}s)", flush=True)
        return out

    # --- Distribution-level (compare before/after vs real) ---
    results["before"]["FID"]     = run_metric("FID before",     metric_fid,    before, real_extended)
    results["after"] ["FID"]     = run_metric("FID after",      metric_fid,    after,  real_extended)
    results["before"]["ArtFID"]  = run_metric("ArtFID before",  metric_artfid, before, real_extended)
    results["after"] ["ArtFID"]  = run_metric("ArtFID after",   metric_artfid, after,  real_extended)
    results["before"]["CMMD"]    = run_metric("CMMD before",    metric_cmmd,   before, real_extended)
    results["after"] ["CMMD"]    = run_metric("CMMD after",     metric_cmmd,   after,  real_extended)
    results["before"]["DINO"]    = run_metric("DINO before",    metric_dino_sim, before, real_extended)
    results["after"] ["DINO"]    = run_metric("DINO after",     metric_dino_sim, after,  real_extended)

    # --- Per-image (single set) ---
    results["before"]["CLIP_IQA"] = run_metric("CLIP-IQA before", metric_clip_iqa, before)
    results["after"] ["CLIP_IQA"] = run_metric("CLIP-IQA after",  metric_clip_iqa, after)

    # --- Pairwise (before <-> after) ---
    results["delta"]["CLIP_DirSim"] = run_metric("CLIP DirSim",  metric_clip_dirsim,
                                                  before, after)

    # Delta summary: for each metric where both before/after are defined as a
    # single number, compute after - before (sign convention noted per metric).
    for m in ["FID", "ArtFID", "CMMD", "DINO", "CLIP_IQA"]:
        b = results["before"][m].get("value")
        a = results["after"][m].get("value")
        if b is not None and a is not None:
            results["delta"][m] = a - b

    out_json = out_dir / "metrics_panel.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + SEP)
    print("  METRICS PANEL SUMMARY  (sign: + means INCREASE)")
    print(SEP)
    print(f"  FID            : {results['before']['FID'].get('value')}"
          f"  -> {results['after']['FID'].get('value')}  (lower is better)")
    print(f"  ArtFID         : {results['before']['ArtFID'].get('value')}"
          f"  -> {results['after']['ArtFID'].get('value')}  (lower is better)")
    print(f"  CMMD           : {results['before']['CMMD'].get('value')}"
          f"  -> {results['after']['CMMD'].get('value')}  (lower is better)")
    print(f"  DINO sim       : {results['before']['DINO'].get('value')}"
          f"  -> {results['after']['DINO'].get('value')}  (higher is better)")
    print(f"  CLIP-IQA       : {results['before']['CLIP_IQA'].get('value')}"
          f"  -> {results['after']['CLIP_IQA'].get('value')}  (higher is better)")
    print(f"  CLIP DirSim    : {results['delta']['CLIP_DirSim'].get('value')}  "
          f"(higher = adapter moves in target text direction)")
    print(SEP)
    print(f"  json: {out_json}")


if __name__ == "__main__":
    main()
