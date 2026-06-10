#!/usr/bin/env python3
"""
janus/eval/06c_fid_artfid_robust.py

Robust FID + ArtFID for all already-evaluated adapters, fixing the two
root causes of the previous failures:

1. RANK-DEFICIENT COVARIANCE. FID needs sqrtm of 2048x2048 covariance
   products; with 80 samples the sample covariance has rank <= 79, so
   scipy.linalg.sqrtm produces huge imaginary components (the
   'Imaginary component 1e+123' errors). Fix: compute
   tr(sqrt(S1*S2)) via the eigenvalues of S1@S2, clamping
   negative/complex eigenvalues (numerical noise around zero) to 0.
   Mathematically equivalent for PSD matrices, numerically robust.
   This replaces ONLY the fragile sqrtm step; the art-fid package's
   official feature extractor (Inception fine-tuned on art) and the
   official ArtFID formula are used unchanged.

2. TINY REFERENCE SET. The previous reference was the 80 held-out
   Naruto images. For distribution distances the reference should
   represent the TARGET STYLE distribution; using the full Naruto
   dataset (1221 images) is standard practice in style-transfer
   evaluation and triples the stability of the reference statistics.
   (No leakage concern: the reference defines the style target, it is
   not a test set.)

Caveat we report alongside the numbers: the GENERATED side remains
N=80, so FID/ArtFID values are directional; KID stays the primary
distribution metric. Updates each metrics_panel.json in-place with
delta.FID_robust and delta.ArtFID.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get(
    "METRICS_ROOT",
    os.path.expanduser("~/multimodal-peft/janus/tests/output/metrics_panel")))
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME",
                                 "lambdalabs/naruto-blip-captions")
HF_IMAGE_COL = os.environ.get("HF_IMAGE_COL", "image")
REF_DIR = Path(os.environ.get(
    "REF_DIR", os.path.expanduser(
        "~/multimodal-peft/janus/tests/output/naruto_ref_full_256")))
SIZE = 256
SEP = "=" * 64


# ---------------------------------------------------------------------------
# Robust Frechet distance (eigenvalue clamping instead of sqrtm)
# ---------------------------------------------------------------------------

def robust_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """d^2 = |mu1-mu2|^2 + tr(S1) + tr(S2) - 2*tr(sqrt(S1*S2)).
    tr(sqrt(S1*S2)) = sum_i sqrt(lambda_i) where lambda_i are the
    eigenvalues of S1@S2 (real, >=0 for PSD inputs up to numerical noise;
    we clamp Re(lambda) at 0 and ignore the spurious imaginary parts)."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    eigvals = np.linalg.eigvals(sigma1 @ sigma2)
    tr_covmean = float(np.sum(np.sqrt(np.maximum(eigvals.real, 0.0))))
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2)
                 - 2.0 * tr_covmean)


def _norm_save(img, path):
    img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(path)


def build_reference_dir():
    """Materialise the FULL Naruto dataset (1221 imgs) as 256x256 PNGs,
    once; reused across all adapters and reruns."""
    if REF_DIR.exists() and len(list(REF_DIR.glob("*.png"))) > 1000:
        print(f"  reference dir already built: {REF_DIR}")
        return
    from datasets import load_dataset
    print(f"  building full reference dir at {REF_DIR} ...", flush=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET_NAME, split="train")
    for i in range(len(ds)):
        _norm_save(ds[i][HF_IMAGE_COL], REF_DIR / f"{i:05d}.png")
    print(f"  saved {len(ds)} reference images", flush=True)


# ---------------------------------------------------------------------------
# Patched art-fid calls
# ---------------------------------------------------------------------------

def patch_artfid_frechet():
    """Replace art_fid's fragile sqrtm-based frechet with the robust one.
    Module-level patch: compute_fid resolves the name at call time from
    the module globals, so this propagates everywhere inside art_fid."""
    import art_fid.art_fid as af
    af.compute_frechet_distance = robust_frechet_distance
    return af


def patch_dataloader_single_process():
    import torch.utils.data as _td
    orig = _td.DataLoader.__init__
    def patched(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
        return orig(self, *args, **kwargs)
    _td.DataLoader.__init__ = patched
    return orig


def compute_for_adapter(run_dir, af):
    """Returns dict with FID_robust (vanilla-Inception, cleanfid features,
    before+after vs full reference) and ArtFID (art-Inception features,
    official formula, robust frechet)."""
    import tempfile
    before_dir = run_dir / "generated" / "before"
    after_dir = run_dir / "generated" / "after"

    # --- Normalised copies of generated sets (256x256, art-fid collate-safe)
    with tempfile.TemporaryDirectory() as tb, tempfile.TemporaryDirectory() as ta:
        for i, p in enumerate(sorted(before_dir.glob("*.png"))):
            _norm_save(Image.open(p), f"{tb}/{i:04d}.png")
        for i, p in enumerate(sorted(after_dir.glob("*.png"))):
            _norm_save(Image.open(p), f"{ta}/{i:04d}.png")

        results = {}

        # --- FID_robust on art_fid's standard pipeline is actually the
        # art-Inception; for a VANILLA FID we reuse cleanfid's extractor.
        try:
            from cleanfid import fid as cf
            feat_model = cf.build_feature_extractor(mode="clean", device="cuda")
            def folder_stats(folder):
                feats = cf.get_folder_features(
                    str(folder), model=feat_model, num_workers=0,
                    batch_size=32, device="cuda", mode="clean")
                return feats.mean(axis=0), np.cov(feats, rowvar=False)
            mu_r, s_r = folder_stats(REF_DIR)
            mu_b, s_b = folder_stats(tb)
            mu_a, s_a = folder_stats(ta)
            results["FID_robust"] = {
                "before": robust_frechet_distance(mu_b, s_b, mu_r, s_r),
                "after":  robust_frechet_distance(mu_a, s_a, mu_r, s_r),
                "note": "cleanfid Inception features, eigval-clamped frechet, "
                        "full-dataset reference (1221), N_gen=80 (directional)",
            }
            results["FID_robust"]["delta"] = (results["FID_robust"]["after"]
                                              - results["FID_robust"]["before"])
        except Exception as e:
            results["FID_robust"] = {"value": None,
                                     "error": f"{type(e).__name__}: {e}"}

        # --- ArtFID: official package, patched frechet ---
        try:
            score = af.compute_art_fid(
                path_to_stylized=ta,
                path_to_style=str(REF_DIR),
                path_to_content=tb,
                batch_size=32,
                device="cuda",
                mode="art_fid",
                content_metric="lpips",
                num_workers=0,
            )
            results["ArtFID"] = {
                "value": float(score), "mode": "art_fid",
                "note": "art-Inception features (official), eigval-clamped "
                        "frechet, full-dataset style ref (1221), N_gen=80",
            }
        except Exception as e:
            results["ArtFID"] = {"value": None,
                                 "error": f"{type(e).__name__}: {e}"}
    return results


def main():
    print(SEP)
    print("  Robust FID + ArtFID update")
    print(SEP, flush=True)
    build_reference_dir()
    af = patch_artfid_frechet()
    patch_dataloader_single_process()

    candidates = sorted([p for p in ROOT.iterdir()
                         if p.is_dir() and (p / "generated" / "after").exists()])
    if not candidates:
        raise SystemExit(f"no candidate folders under {ROOT}")

    for run_dir in candidates:
        print(f"\n  [{run_dir.name}]", flush=True)
        t0 = time.time()
        res = compute_for_adapter(run_dir, af)
        for k, v in res.items():
            print(f"    {k}: {v}", flush=True)
        json_path = run_dir / "metrics_panel.json"
        if json_path.exists():
            d = json.loads(json_path.read_text())
            d.setdefault("delta", {})["ArtFID"] = res.get("ArtFID")
            d.setdefault("delta", {})["FID_robust"] = res.get("FID_robust")
            json_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
            print(f"    -> updated {json_path.name}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n" + SEP)
    print("  DONE")
    print(SEP)


if __name__ == "__main__":
    main()
