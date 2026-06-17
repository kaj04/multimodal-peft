#!/usr/bin/env python3
"""
janus/eval/17_gen_forgetting_lambda.py

Question-A image-generation forgetting for the poetry text-LoRA LAMBDA sweep:
does specializing the TEXT stream shift IMAGE generation away from the base?
We generate the SAME COCO prompts (same seeds) with the adapter OFF (base) and
ON, and measure CMMD between the two GENERATED sets. Low = generation unchanged
(no forgetting); rising as KL->0 = forgetting. This is the distribution-SHIFT-
from-base measure -> NO real-image reference needed (that would be Question B,
absolute quality vs real photos -- not run here).

Cheap by design: the base-gen set is IDENTICAL for every lambda, so we generate
it ONCE and reuse it for all adapters (base loaded once + PeftModel multi-
adapter via load_adapter/set_adapter). All images are saved to disk (REUSE_IMAGES
to resume; also lets FID/KID/Question-B be added later WITHOUT regenerating).

CMMD = Jayasumana et al., CVPR 2024 (CLIP ViT-L/14 + Gaussian-RBF MMD, median-
heuristic sigma) -- robust at small N, unlike FID. Implementation copied from
06_metrics_panel.py. KID (clean-fid, folder-based) added best-effort if present.

Run via 17_run_gen_forgetting_lambda.sh (GPU). Default N_GEN=128 ~ 2h / ~400 SBU.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import generate_image  # noqa: E402

MODEL_ID  = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
SEED      = int(os.environ.get("SEED", "42"))
N_GEN     = int(os.environ.get("N_GEN", "128"))
REUSE     = os.environ.get("REUSE_IMAGES", "1") == "1"
COCO_CAPS = Path(os.environ.get(
    "COCO_CAPS", "/scratch-shared/tesr125338/coco/annotations/captions_val2017.json"))
OUT_ROOT  = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"))) / "gen_forgetting_lambda"
TRAIN_OUT = Path(os.path.expanduser("~/multimodal-peft/janus/train/output"))

# Adapters to evaluate: "name=path;..." or default = the r16 lambda sweep.
_env = os.environ.get("ADAPTERS", "")
if _env:
    ADAPTERS = [(s.split("=")[0].strip(), s.split("=")[1].strip())
                for s in _env.split(";") if "=" in s]
else:
    ADAPTERS = [(f"kl{kl}", str(TRAIN_OUT / f"text_poetry_r16_kl{kl}" / "adapter_best"))
                for kl in ("0", "0.25", "0.5", "1", "1.5")]

SEP = "=" * 64


def load_coco_prompts(n, seed):
    data = json.loads(COCO_CAPS.read_text())
    caps = [a["caption"].strip() for a in data["annotations"]
            if a.get("caption", "").strip()]
    rng = random.Random(seed)
    rng.shuffle(caps)
    seen, out = set(), []
    for c in caps:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= n:
            break
    return out


# --- CMMD (copied verbatim in spirit from 06_metrics_panel.py) ---------------

def _clip_feats(clip, proc, imgs, device="cuda", batch=16):
    @torch.no_grad()
    def _to_tensor(x):
        if torch.is_tensor(x):
            return x
        for attr in ("image_features", "pooler_output", "last_hidden_state"):
            v = getattr(x, attr, None)
            if torch.is_tensor(v):
                return v if v.dim() == 2 else v[:, 0]
        raise RuntimeError("cannot extract image feats")
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = proc(images=imgs[i:i + batch], return_tensors="pt").to(device)
            f = _to_tensor(clip.get_image_features(**x))
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu())
    return torch.cat(out, 0)


def cmmd(clip, proc, gen_imgs, ref_imgs):
    from scipy.spatial.distance import pdist
    x = _clip_feats(clip, proc, gen_imgs).numpy()
    y = _clip_feats(clip, proc, ref_imgs).numpy()
    sigma = float(np.median(pdist(np.vstack([x, y])))) or 1.0

    def k(a, b):
        sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-sq / (2 * sigma ** 2))
    val = k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()
    return float(val), sigma


def kid_folders(dir_a, dir_b):
    """Standard KID (Inception + polynomial kernel) via clean-fid, folder-based.
    Best-effort: returns None if clean-fid is absent or errors (e.g. small N)."""
    try:
        from cleanfid import fid
        return float(fid.compute_kid(str(dir_a), str(dir_b), mode="clean",
                                     num_workers=2))
    except Exception as e:
        print(f"    (KID skipped: {type(e).__name__}: {e})", flush=True)
        return None


@torch.no_grad()
def gen_set(model, processor, prompts, out_dir, with_adapter, adapter_name=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.png"))
    if REUSE and len(existing) >= len(prompts):
        print(f"    REUSE {out_dir.name}: {len(prompts)} images from disk", flush=True)
        return [Image.open(p).convert("RGB") for p in existing[:len(prompts)]]

    imgs = []

    def _gen():
        for k, p in enumerate(prompts):
            torch.manual_seed(SEED + k)
            img = generate_image(model, processor, p)
            img.save(out_dir / f"{k:04d}.png")
            imgs.append(img)

    if with_adapter:
        model.set_adapter(adapter_name)
        _gen()
    else:
        with model.disable_adapter():
            _gen()
    return imgs


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    avail = [(n, p) for n, p in ADAPTERS
             if Path(p).exists() or Path(p).is_symlink()]
    if not avail:
        raise SystemExit("no lambda adapters found (are the trainings done?)")
    print(SEP)
    print(f"  gen-forgetting (Question A: shift from base) | N_GEN={N_GEN}")
    print(f"  adapters: {[n for n, _ in avail]}")
    print(SEP, flush=True)

    prompts = load_coco_prompts(N_GEN, SEED)
    print(f"  loaded {len(prompts)} COCO prompts\n", flush=True)

    from transformers import (JanusForConditionalGeneration, JanusProcessor,
                              CLIPModel, CLIPProcessor)
    from peft import PeftModel
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    base.eval()
    n0, p0 = avail[0]
    model = PeftModel.from_pretrained(base, p0, adapter_name=n0)
    for n, p in avail[1:]:
        model.load_adapter(p, adapter_name=n)
    model.eval()

    base_dir = OUT_ROOT / "base"
    print("  generating BASE set (once, reused for all lambda)...", flush=True)
    base_imgs = gen_set(model, processor, prompts, base_dir, with_adapter=False)

    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()

    rows = []
    for n, _ in avail:
        adir = OUT_ROOT / n
        print(f"  generating adapter set {n}...", flush=True)
        a_imgs = gen_set(model, processor, prompts, adir,
                         with_adapter=True, adapter_name=n)
        val, sigma = cmmd(clip, cproc, a_imgs, base_imgs)
        kid = kid_folders(adir, base_dir)
        rows.append({"name": n, "cmmd_vs_base": val, "kid_vs_base": kid,
                     "sigma": sigma, "n": len(a_imgs)})
        print(f"    {n}: CMMD(vs base)={val:.6f}  KID={kid}\n", flush=True)

    lines = [SEP, f"GEN-FORGETTING (Question A: distribution shift from base)  N={N_GEN}",
             "lower CMMD = generation closer to base = less forgetting", SEP,
             f"{'lambda':<10}{'CMMD_vs_base':>16}{'KID_vs_base':>16}", "-" * 42]
    for r in rows:
        kids = f"{r['kid_vs_base']:.6f}" if r['kid_vs_base'] is not None else "n/a"
        lines.append(f"{r['name']:<10}{r['cmmd_vs_base']:>16.6f}{kids:>16}")
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    (OUT_ROOT / "gen_forgetting.txt").write_text(report + "\n", encoding="utf-8")
    (OUT_ROOT / "gen_forgetting.json").write_text(
        json.dumps({"model": MODEL_ID, "n_gen": N_GEN, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\n  saved -> {OUT_ROOT}/gen_forgetting.txt (+ .json); images in {OUT_ROOT}/<name>/")


if __name__ == "__main__":
    main()
