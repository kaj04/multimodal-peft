#!/usr/bin/env python3
"""
janus/eval/21_bootstrap_cmmd.py

Bootstrap 95% CI for CMMD (and, by the same machinery, Gen-CMMD) -- the set-level
distance metric in most of the result tables. CMMD has no closed-form variance,
so we use the nonparametric bootstrap [Efron & Tibshirani 1993] over the generated
and reference image sets.

It reuses the EXACT CMMD definition of the metrics panel (06_metrics_panel.py):
CLIP ViT-L/14 features (L2-normalised), Gaussian-RBF kernel with a median-heuristic
bandwidth, biased MMD^2 V-statistic. The point estimate therefore matches the panel
(up to the reference sample); the bootstrap fixes the bandwidth at the full-set
median heuristic (a fixed kernel hyper-parameter, not re-estimated per resample)
and recomputes MMD^2 on resampled indices via precomputed kernel matrices, so
B=2000 resamples cost milliseconds.

Inputs (env):
  GEN_DIR    directory of generated images (e.g. .../metrics_panel/<tag>/generated/after)
  REF_DIR    directory of reference images; OR leave empty and set HF_DATASET_NAME
  HF_DATASET_NAME / HF_IMAGE_COL / N_REF   build the reference set from HF if no REF_DIR
  B=2000  SEED=0  LABEL=<name>

Usage:
  # panel CMMD (adapter generations vs real style images from HF)
  GEN_DIR=$HOME/.../metrics_panel/gen_7b_sketch_r64_kl025__adapter_best/generated/after \
  HF_DATASET_NAME=zoheb/sketch-scene LABEL=sketch_r64 \
      sbatch janus/eval/21_run_bootstrap_cmmd.sh

  # Gen-CMMD (text adapter image gen vs base image gen): two saved dirs
  GEN_DIR=$HOME/.../gen_forgetting_lambda/med  REF_DIR=$HOME/.../gen_forgetting_lambda/base \
  LABEL=gen_cmmd_med  sbatch janus/eval/21_run_bootstrap_cmmd.sh
"""
import json
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist, pdist

GEN_DIR = os.environ.get("GEN_DIR", "")
REF_DIR = os.environ.get("REF_DIR", "")
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME", "")
HF_IMAGE_COL = os.environ.get("HF_IMAGE_COL", "image")
N_REF   = int(os.environ.get("N_REF", "80"))
B       = int(os.environ.get("B", "2000"))
SEED    = int(os.environ.get("SEED", "0"))
LABEL   = os.environ.get("LABEL", "cmmd")
OUT     = Path(os.environ.get(
    "JANUS_OUTPUT_DIR", os.path.expanduser("~/multimodal-peft/janus/tests/output"))
) / "confidence_intervals"

CLIP_ID = "openai/clip-vit-large-patch14"
_EXT = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def load_dir(d):
    files = sorted(p for ext in _EXT for p in glob(os.path.join(d, ext)))
    return [Image.open(p).convert("RGB") for p in files]


def load_hf_refs(name, col, n, seed):
    from datasets import load_dataset
    ds = load_dataset(name, split="train", streaming=True)
    rng = random.Random(seed)
    pool = []
    for i, row in enumerate(ds):
        img = row.get(col)
        if img is not None:
            pool.append(img.convert("RGB"))
        if len(pool) >= n * 4 or i > n * 12:
            break
    rng.shuffle(pool)
    return pool[:n]


@torch.no_grad()
def clip_feats(imgs, device="cuda", batch=16):
    from transformers import CLIPModel, CLIPProcessor
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    out = []
    for i in range(0, len(imgs), batch):
        x = proc(images=imgs[i:i + batch], return_tensors="pt").to(device)
        f = clip.get_image_features(**x)
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.cpu().numpy())
    return np.concatenate(out, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert GEN_DIR, "set GEN_DIR"
    gen = load_dir(GEN_DIR)
    if REF_DIR:
        ref = load_dir(REF_DIR)
    else:
        assert HF_DATASET_NAME, "set REF_DIR or HF_DATASET_NAME"
        ref = load_hf_refs(HF_DATASET_NAME, HF_IMAGE_COL, N_REF, SEED)
    print(f"  LABEL={LABEL}  n_gen={len(gen)}  n_ref={len(ref)}  B={B}", flush=True)

    x = clip_feats(gen)
    y = clip_feats(ref)

    # median-heuristic bandwidth over the pooled features (exactly as the panel)
    sigma = float(np.median(pdist(np.vstack([x, y])))) or 1.0
    Kxx = np.exp(-cdist(x, x, "sqeuclidean") / (2 * sigma ** 2))
    Kyy = np.exp(-cdist(y, y, "sqeuclidean") / (2 * sigma ** 2))
    Kxy = np.exp(-cdist(x, y, "sqeuclidean") / (2 * sigma ** 2))

    n, m = len(x), len(y)
    point = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

    rng = np.random.default_rng(SEED)
    boots = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, n, n)
        iy = rng.integers(0, m, m)
        boots[b] = (Kxx[np.ix_(ix, ix)].mean()
                    + Kyy[np.ix_(iy, iy)].mean()
                    - 2 * Kxy[np.ix_(ix, iy)].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])

    res = {"label": LABEL, "metric": "CMMD", "gen_dir": GEN_DIR,
           "ref_dir": REF_DIR or f"HF:{HF_DATASET_NAME}",
           "n_gen": n, "n_ref": m, "B": B, "sigma": sigma,
           "value": float(point), "ci95": [float(lo), float(hi)],
           "boot_std": float(boots.std())}
    print(f"  CMMD = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (boot sd {boots.std():.4f})")
    (OUT / f"cmmd_ci_{LABEL}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  saved -> {OUT}/cmmd_ci_{LABEL}.json")


if __name__ == "__main__":
    main()
