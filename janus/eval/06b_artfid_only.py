#!/usr/bin/env python3
"""
janus/eval/06b_artfid_only.py

Compute ONLY ArtFID for already-evaluated adapters, reusing the BEFORE/AFTER
images saved on disk by 06_metrics_panel.py. Updates the existing
metrics_panel.json in-place (only the delta.ArtFID field).

Why: the full metrics panel takes ~10 min/adapter even with REUSE_IMAGES,
mostly to reload CLIP and DINO for the other 4 metrics. After the recent
art-fid bug fixes (monkeypatch, size normalisation, mode='art_fid') we
just want to populate ArtFID without redoing the others.

Discovers all metrics_panel/<run>__<epoch>/ subfolders that have a
generated/ subdir, and processes each one. Run via 06b_run_artfid_only.sh
(needs GPU for Inception + LPIPS).

Reference set (style images): held-out eval split of HF_DATASET_NAME,
re-derived from SEED with the same convention as 06_metrics_panel.py so
the comparison is consistent.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SEED = int(os.environ.get("SEED", "42"))
TRAIN_N = int(os.environ.get("TRAIN_N", "4000"))
EVAL_N  = int(os.environ.get("EVAL_N", "80"))

HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME",
                                 "lambdalabs/naruto-blip-captions")
HF_IMAGE_COL    = os.environ.get("HF_IMAGE_COL", "image")

ROOT = Path(os.environ.get(
    "METRICS_ROOT",
    os.path.expanduser("~/multimodal-peft/janus/tests/output/metrics_panel")))

SEP = "=" * 64


def load_ref_images(n):
    """Re-derive the same held-out eval split + return n reference real
    Naruto images. Mirrors 06_metrics_panel.py's load_eval_split."""
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET_NAME, split="train")
    ids = list(range(len(ds)))
    random.Random(SEED).shuffle(ids)
    n_eval  = min(EVAL_N, 80)
    n_train = min(TRAIN_N, len(ids) - n_eval)
    eval_ids = ids[n_train:n_train + n_eval]
    real_imgs = [ds[i][HF_IMAGE_COL].convert("RGB") for i in eval_ids]
    # Pad to N if needed (recycle, same as the full panel)
    extended = (real_imgs * ((n // len(real_imgs)) + 1))[:n]
    return extended


def compute_artfid_one(before_dir, after_dir, ref_imgs):
    """Run ArtFID on a single (before, after, ref) triplet. Returns
    {'value': float} on success or {'value': None, 'error': str}."""
    from art_fid.art_fid import compute_art_fid

    # Load before/after PILs from disk (they were saved at 384x384 native)
    before_paths = sorted(before_dir.glob("*.png"))
    after_paths  = sorted(after_dir.glob("*.png"))
    before_imgs = [Image.open(p).convert("RGB") for p in before_paths]
    after_imgs  = [Image.open(p).convert("RGB") for p in after_paths]
    if not before_imgs or not after_imgs:
        return {"value": None, "error": "no images on disk"}
    # Trim ref_imgs to match
    n = min(len(before_imgs), len(after_imgs), len(ref_imgs))
    before_imgs = before_imgs[:n]
    after_imgs  = after_imgs[:n]
    refs        = ref_imgs[:n]

    import tempfile
    with tempfile.TemporaryDirectory() as a, \
         tempfile.TemporaryDirectory() as b, \
         tempfile.TemporaryDirectory() as c:
        # Normalise all to 256x256 PNG to avoid art-fid's stack-size crash
        SIZE = 256
        def _save(img, path):
            img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(path)
        for i, img in enumerate(after_imgs):  _save(img, f"{a}/{i:04d}.png")
        for i, img in enumerate(refs):        _save(img, f"{b}/{i:04d}.png")
        for i, img in enumerate(before_imgs): _save(img, f"{c}/{i:04d}.png")

        # Monkeypatch torch DataLoader to force single-process load
        # (art-fid's internal LPIPS loader has hardcoded workers that
        # crash on the collate stage in our torch version).
        import torch.utils.data as _td
        _orig = _td.DataLoader.__init__
        def _patched(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)
            return _orig(self, *args, **kwargs)
        _td.DataLoader.__init__ = _patched

        try:
            score = compute_art_fid(
                path_to_stylized=a,
                path_to_style=b,
                path_to_content=c,
                batch_size=32,
                device="cuda",
                mode="art_fid",
                content_metric="lpips",
                num_workers=0,
            )
            return {"value": float(score), "mode": "art_fid", "n": n}
        except Exception as e:
            return {"value": None, "error": f"{type(e).__name__}: {e}"}
        finally:
            _td.DataLoader.__init__ = _orig


def main():
    print(SEP)
    print(f"  ArtFID-only update across {ROOT}")
    print(SEP)
    if not ROOT.exists():
        raise SystemExit(f"metrics root not found: {ROOT}")

    # Load reference Naruto eval split once
    print("  loading reference Naruto images ...", flush=True)
    ref_imgs = load_ref_images(n=EVAL_N)
    print(f"  loaded {len(ref_imgs)} reference images\n", flush=True)

    # Iterate over all subfolders with a generated/ directory
    candidates = sorted([p for p in ROOT.iterdir()
                         if p.is_dir() and (p / "generated" / "before").exists()
                                       and (p / "generated" / "after").exists()])
    if not candidates:
        raise SystemExit(f"no candidate folders under {ROOT}")

    for run_dir in candidates:
        print(f"  [{run_dir.name}]", flush=True)
        before_dir = run_dir / "generated" / "before"
        after_dir  = run_dir / "generated" / "after"
        n_before = len(list(before_dir.glob("*.png")))
        n_after  = len(list(after_dir.glob("*.png")))
        print(f"    found before={n_before} after={n_after}", flush=True)

        t0 = time.time()
        result = compute_artfid_one(before_dir, after_dir, ref_imgs)
        elapsed = time.time() - t0
        print(f"    result: {result}   ({elapsed:.1f}s)", flush=True)

        # Update the existing metrics_panel.json in-place
        json_path = run_dir / "metrics_panel.json"
        if not json_path.exists():
            print(f"    WARN: no metrics_panel.json under {run_dir}; skipping update")
            continue
        d = json.loads(json_path.read_text())
        d.setdefault("delta", {})["ArtFID"] = result
        json_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"    -> updated {json_path.name}\n", flush=True)

    print(SEP)
    print("  DONE")
    print(SEP)


if __name__ == "__main__":
    main()
