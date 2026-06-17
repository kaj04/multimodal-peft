#!/usr/bin/env python3
"""
janus/eval/18_weight_delta_norm.py

Intervention-strength of each LoRA adapter, to pre-empt the reviewer objection
"maybe the adapter barely changed the model, so 'no forgetting' is trivial".

Each LoRA adapter induces a weight update on every backbone projection,
    Delta_W = (alpha / r) * B @ A          (a "task vector", Ilharco et al. 2023)
and we summarize it by the relative Frobenius norm  ||Delta_W|| / ||W||, where W
is the corresponding BASE weight. We report this per module (for a layer x module
heatmap) and aggregated per adapter (one headline number). Forgetting magnitude
is known to track the size/geometry of this update.

CPU only, no inference: reads each adapter's adapter_model.safetensors and the
base model's cached safetensors (lazily, only the needed tensors). ||BA||_F is
computed via the r x r trace identity, so the full Delta_W is never formed:
    ||BA||_F^2 = trace((A A^T)(B^T B)) = sum( (A A^T) * (B^T B) ).

Run (login node is fine):
  python janus/eval/18_weight_delta_norm.py
Override the adapter set:
  ADAPTERS="name=/path/to/adapter_best;..." python janus/eval/18_weight_delta_norm.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
TRAIN_OUT = Path(os.path.expanduser("~/multimodal-peft/janus/train/output"))
OUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"))) / "weight_delta"

# Default adapter set: the text sweeps (paths we know). Add the image adapters
# via the ADAPTERS env, e.g.
#   ADAPTERS="v5=/.../gen_7b_..._r48/adapter_best;r2=/.../..._r2/adapter_best;..."
_env = os.environ.get("ADAPTERS", "")
if _env:
    ADAPTERS = [(s.split("=", 1)[0].strip(), s.split("=", 1)[1].strip())
                for s in _env.split(";") if "=" in s]
else:
    ADAPTERS = (
        [(f"text_r{r}",  str(TRAIN_OUT / f"text_poetry_r{r}_kl0.5" / "adapter_best"))
         for r in (4, 8, 16, 32, 64)]
        + [(f"text_kl{kl}", str(TRAIN_OUT / f"text_poetry_r16_kl{kl}" / "adapter_best"))
           for kl in ("0", "0.25", "0.5", "1", "1.5")]
    )

SEP = "=" * 64


# --- adapter Delta_W norms (no full matrix formed) --------------------------

def adapter_deltas(adapter_dir):
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scaling = alpha / ((r ** 0.5) if cfg.get("use_rslora", False) else r)
    A, B = {}, {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.endswith("lora_A.weight"):
                A[k[: -len(".lora_A.weight")]] = f.get_tensor(k).float()
            elif k.endswith("lora_B.weight"):
                B[k[: -len(".lora_B.weight")]] = f.get_tensor(k).float()
    deltas = {}
    for mod, a in A.items():           # a: (r, k)
        b = B[mod]                     # b: (d, r)
        aat = a @ a.t()                # (r, r)
        btb = b.t() @ b                # (r, r)
        fro = torch.sqrt(torch.clamp((aat * btb).sum(), min=0.0)).item()
        deltas[mod] = scaling * fro
    return deltas, r, alpha


# --- base ||W|| per module (lazy read from cached safetensors) ---------------

def _base_key(mod):
    bk = mod
    for pre in ("base_model.model.", "base_model."):
        if bk.startswith(pre):
            bk = bk[len(pre):]
            break
    return bk + ".weight"


def base_norms(mods):
    from huggingface_hub import hf_hub_download
    want = {mod: _base_key(mod) for mod in mods}
    norms = {}
    try:
        idx = json.loads(Path(hf_hub_download(
            MODEL_ID, "model.safetensors.index.json",
            local_files_only=True)).read_text())["weight_map"]
        by_shard = {}
        for mod, bk in want.items():
            if bk in idx:
                by_shard.setdefault(idx[bk], {})[bk] = mod
        for shard, keys in by_shard.items():
            p = hf_hub_download(MODEL_ID, shard, local_files_only=True)
            with safe_open(p, framework="pt") as f:
                for bk, mod in keys.items():
                    norms[mod] = float(f.get_tensor(bk).float().norm())
    except Exception as e:
        print(f"  (sharded base read failed: {type(e).__name__}: {e}; "
              f"trying single file)", flush=True)
        try:
            p = hf_hub_download(MODEL_ID, "model.safetensors", local_files_only=True)
            with safe_open(p, framework="pt") as f:
                avail = set(f.keys())
                for mod, bk in want.items():
                    if bk in avail:
                        norms[mod] = float(f.get_tensor(bk).float().norm())
        except Exception as e2:
            print(f"  (base norms unavailable: {e2}); reporting absolute only",
                  flush=True)
    return norms


# --- per-module (layer, type) parsing for the heatmap -----------------------

_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_mod(mod):
    m = re.search(r"layers\.(\d+)", mod)
    layer = int(m.group(1)) if m else -1
    ptype = next((p for p in _PROJ if mod.endswith(p)), mod.split(".")[-1])
    return layer, ptype


def heatmap(rel, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = sorted({parse_mod(m)[0] for m in rel})
    grid = np.full((len(layers), len(_PROJ)), np.nan)
    li = {l: i for i, l in enumerate(layers)}
    for m, v in rel.items():
        l, p = parse_mod(m)
        if p in _PROJ and l in li:
            grid[li[l], _PROJ.index(p)] = v
    fig, ax = plt.subplots(figsize=(4.5, max(3, len(layers) * 0.14)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(_PROJ)))
    ax.set_xticklabels([p.replace("_proj", "") for p in _PROJ], rotation=45, ha="right")
    ax.set_yticks(range(0, len(layers), max(1, len(layers) // 12)))
    ax.set_yticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 12))])
    ax.set_ylabel("layer"); ax.set_title(f"||$\\Delta$W||/||W||  {name}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"dw_heatmap_{name}.png", dpi=130)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    avail = [(n, Path(p)) for n, p in ADAPTERS
             if (Path(p) / "adapter_config.json").exists()
             or (Path(p) / "adapter_config.json").is_symlink()]
    if not avail:
        raise SystemExit("no adapters found (check paths / ADAPTERS env)")
    print(SEP)
    print(f"  weight-delta norms | base {MODEL_ID}")
    print(f"  adapters: {[n for n, _ in avail]}")
    print(SEP, flush=True)

    # base norms once (union of all modules)
    all_mods = set()
    per_adapter = {}
    for n, p in avail:
        d, r, alpha = adapter_deltas(p)
        per_adapter[n] = (d, r, alpha)
        all_mods |= set(d.keys())
    bn = base_norms(all_mods)
    print(f"  base ||W|| matched for {len(bn)}/{len(all_mods)} modules\n", flush=True)

    rows, summary = [], {}
    for n, (d, r, alpha) in per_adapter.items():
        # aggregate Frobenius: ||DW||_total = sqrt(sum dw_mod^2)
        tot_dw = float(np.sqrt(sum(v * v for v in d.values())))
        rel = {}
        if bn:
            tot_w = float(np.sqrt(sum(bn[m] ** 2 for m in d if m in bn)))
            rel = {m: d[m] / bn[m] for m in d if m in bn and bn[m] > 0}
            rel_tot = tot_dw / tot_w if tot_w > 0 else float("nan")
            mean_rel = float(np.mean(list(rel.values()))) if rel else float("nan")
            try:
                heatmap(rel, n)
            except Exception as e:
                print(f"    (heatmap skipped: {type(e).__name__}: {e})", flush=True)
        else:
            rel_tot, mean_rel = float("nan"), float("nan")
        summary[n] = {"r": r, "alpha": alpha, "abs_dw": tot_dw,
                      "rel_dw_total": rel_tot, "rel_dw_mean_module": mean_rel,
                      "n_modules": len(d)}
        print(f"  {n:<14} r={r:<3} ||DW||={tot_dw:8.2f}  "
              f"||DW||/||W||(total)={rel_tot*100:6.3f}%  "
              f"mean/module={mean_rel*100:6.3f}%", flush=True)
        rows.append((n, r, alpha, tot_dw, rel_tot, mean_rel))

    (OUT_DIR / "weight_delta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT_DIR}/weight_delta.json ; heatmaps dw_heatmap_*.png")


if __name__ == "__main__":
    main()
