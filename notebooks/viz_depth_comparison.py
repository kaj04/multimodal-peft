# %% [markdown]
# # 4M-21 Depth Estimation: Baseline vs LoRA Visual Comparison
#
# Shows 5 validation samples as a 5×4 grid:
#   Col 1 – Original RGB (decoded from VQ-VAE RGB tokens)
#   Col 2 – Ground-truth depth (decoded from saved tok_depth tokens)
#   Col 3 – 4M-21 baseline prediction (LoRA injected but weights = 0)
#   Col 4 – 4M-21 + LoRA fine-tuned prediction
#
# Run on Snellius (single GPU node is enough):
#   cd ~/4m-peft
#   conda activate 4m-peft
#   python notebooks/viz_depth_comparison.py
#
# Output saved to: figures/depth_comparison.png
# (also displayed inline when running as a Jupyter notebook)

# %% ─── Cell 1: paths & imports ────────────────────────────────────────────
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # headless on Snellius; change to 'TkAgg' locally
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── repo root ──────────────────────────────────────────────────────────────
REPO_ROOT = Path.home() / '4m-peft'
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

# ── paths ──────────────────────────────────────────────────────────────────
HF_CACHE      = '/scratch-shared/tesr125338/hf_cache'
NYU_VAL       = '/scratch-shared/tesr125338/nyu_tokens/val'
LORA_CKPT     = str(REPO_ROOT / 'output/depth/nyu_rgb_r8_e5/lora_weights-best.pth')
FIGURES_DIR   = REPO_ROOT / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

NUM_SAMPLES   = 5          # how many val images to visualize
LORA_RANK     = 8
LORA_SCALE    = 1.0
PRETRAINED    = 'EPFL-VILAB/4M-21_XL'
RGB_TOK_REPO  = 'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448'
DEPTH_TOK_REPO= 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16     # matches training precision

print(f'Device : {DEVICE}')
print(f'LoRA checkpoint : {LORA_CKPT}')
assert Path(LORA_CKPT).exists(), (
    f'\n[!] LoRA checkpoint not found: {LORA_CKPT}\n'
    f'    Run the 5-epoch validation job first:\n'
    f'    sbatch scripts/snellius/05_ft_depth_nyu_val5.sh'
)

# %% ─── Cell 2: load VQ-VAE tokenizers ────────────────────────────────────
from fourm.vq.vqvae import VQ

print('Loading RGB tokenizer …')
rgb_tok = VQ.from_pretrained(RGB_TOK_REPO, cache_dir=HF_CACHE).eval().to(DEVICE)
print('Loading Depth tokenizer …')
depth_tok = VQ.from_pretrained(DEPTH_TOK_REPO, cache_dir=HF_CACHE).eval().to(DEVICE)
print('Tokenizers ready.\n')

# %% ─── Cell 3: load 5 pre-tokenized val samples ──────────────────────────
val_root  = Path(NYU_VAL)
rgb_dir   = val_root / 'tok_rgb'
depth_dir = val_root / 'tok_depth'

# Collect stems common to both folders, take first NUM_SAMPLES
rgb_stems   = sorted(p.stem for p in rgb_dir.glob('**/*.npy'))
depth_stems = sorted(p.stem for p in depth_dir.glob('**/*.npy'))
common = sorted(set(rgb_stems) & set(depth_stems))[:NUM_SAMPLES]
print(f'Selected {len(common)} samples: {common}\n')

def _load_npy(folder: Path, stem: str) -> np.ndarray:
    matches = list(folder.rglob(f'{stem}.npy'))
    assert matches, f'{stem}.npy not found under {folder}'
    return np.load(matches[0]).astype(np.int64)

rgb_tokens   = torch.from_numpy(
    np.stack([_load_npy(rgb_dir,   s) for s in common])).to(DEVICE)   # (N, 196)
depth_tokens = torch.from_numpy(
    np.stack([_load_npy(depth_dir, s) for s in common])).to(DEVICE)   # (N, 196)

print(f'RGB tokens shape   : {rgb_tokens.shape}')
print(f'Depth tokens shape : {depth_tokens.shape}')

# %% ─── Cell 4: decode ground-truth images for display ────────────────────

@torch.no_grad()
def decode_tokens(vqvae: VQ, tokens_flat: torch.Tensor) -> torch.Tensor:
    """Decode flat (B, 196) tokens → pixel tensor (B, C, H, W).

    VQ.decode_tokens expects shape (B, H_Q, W_Q) = (B, 14, 14).
    """
    B = tokens_flat.shape[0]
    tokens_2d = tokens_flat.reshape(B, 14, 14).long()
    return vqvae.decode_tokens(tokens_2d)   # (B, C, H, W)


with torch.no_grad():
    rgb_pixels   = decode_tokens(rgb_tok,   rgb_tokens)    # (N, 3, H, W) ~ [-1, 1]
    gt_depth_pix = decode_tokens(depth_tok, depth_tokens)  # (N, 1, H, W)

print(f'RGB decoded  : {rgb_pixels.shape}   range [{rgb_pixels.min():.2f}, {rgb_pixels.max():.2f}]')
print(f'GT depth dec : {gt_depth_pix.shape}  range [{gt_depth_pix.min():.2f}, {gt_depth_pix.max():.2f}]')

# %% ─── Cell 5: helpers for building 4M mod_dict ──────────────────────────

def build_mod_dict(input_tokens, target_tokens,
                   input_mod='tok_rgb@224', target_mod='tok_depth@224',
                   num_input=196, num_target=196):
    """Build a mod_dict for 4M single-input→single-target forward pass.

    input_tokens  : (B, num_input)  long  — fully visible to encoder
    target_tokens : (B, num_target) long  — fully hidden; all predicted
    """
    B, device = input_tokens.shape[0], input_tokens.device
    return {
        input_mod: {
            'tensor'                : input_tokens,
            'input_mask'            : torch.zeros( (B, num_input),  dtype=torch.bool, device=device),
            'target_mask'           : torch.ones(  (B, num_input),  dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.zeros( (B, num_input),  dtype=torch.int,  device=device),
        },
        target_mod: {
            'tensor'                : target_tokens,
            'input_mask'            : torch.ones(  (B, num_target), dtype=torch.bool, device=device),
            'target_mask'           : torch.zeros( (B, num_target), dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.ones(  (B, num_target), dtype=torch.int,  device=device),
        },
    }

# %% ─── Cell 6: load 4M-21_XL + inject LoRA (weights = 0 at init) ─────────
from fourm.models.fm import FM
from fourm.models.lora_utils import (
    get_LoRA_module_names, inject_trainable_LoRA
)

print(f'Loading {PRETRAINED} …  (this may take ~2 min)')
model = FM.from_pretrained(PRETRAINED, cache_dir=HF_CACHE)
model = model.to(DTYPE).to(DEVICE).eval()
print('Base model loaded.\n')

# Inject LoRA adapters — at this point lora_up weights are all zeros,
# so model behaves identically to the pre-trained 4M-21_XL baseline.
target_modules = get_LoRA_module_names('attention')
inject_trainable_LoRA(model, rank=LORA_RANK, scale=LORA_SCALE,
                      target_replace_modules=target_modules)
print('LoRA injected (weights = 0 → pure baseline).\n')

# %% ─── Cell 7: baseline inference (LoRA = 0, pure 4M-21) ─────────────────

@torch.no_grad()
def predict_depth_tokens(model, rgb_toks, depth_toks):
    """Run a forward pass and return predicted depth tokens.

    Returns: (B, 196) long tensor of predicted depth codes.
    """
    # Build mod_dict — depth tokens are the "target", fully masked from encoder
    mod_dict = build_mod_dict(rgb_toks, depth_toks)

    # Cast to model dtype for safety
    for mod, d in mod_dict.items():
        if 'tensor' in d:
            d['tensor'] = d['tensor'].to(torch.long)   # indices stay long

    with torch.cuda.amp.autocast(dtype=DTYPE, enabled=(DEVICE.type == 'cuda')):
        mod_logits = model(mod_dict,
                           num_encoder_tokens=196,
                           num_decoder_tokens=196,
                           return_logits=True)

    logits = mod_logits['tok_depth@224']        # (B, 196, vocab_size)
    pred_tokens = logits.float().argmax(dim=-1) # (B, 196)
    return pred_tokens


print('Running baseline inference …')
baseline_pred_tokens = predict_depth_tokens(model, rgb_tokens, depth_tokens)
print(f'Baseline pred tokens: {baseline_pred_tokens.shape}')

# Decode to pixel space
with torch.no_grad():
    baseline_depth_pix = decode_tokens(depth_tok, baseline_pred_tokens)  # (N, 1, H, W)
print(f'Baseline depth decoded: range [{baseline_depth_pix.min():.2f}, {baseline_depth_pix.max():.2f}]\n')

# %% ─── Cell 8: load LoRA weights and run fine-tuned inference ─────────────

print(f'Loading LoRA weights from {LORA_CKPT} …')
lora_sd = torch.load(LORA_CKPT, map_location=DEVICE)
# strict=False because lora_sd only contains lora_ keys; all other params are already loaded
missing, unexpected = model.load_state_dict(lora_sd, strict=False)
lora_keys = [k for k in lora_sd]
print(f'  {len(lora_keys)} LoRA tensors loaded')
print(f'  Missing (expected — non-LoRA params): {len(missing)}')
print(f'  Unexpected: {len(unexpected)}')
if unexpected:
    print(f'  Unexpected keys: {unexpected[:5]}')

print('\nRunning LoRA inference …')
lora_pred_tokens = predict_depth_tokens(model, rgb_tokens, depth_tokens)
print(f'LoRA pred tokens: {lora_pred_tokens.shape}')

with torch.no_grad():
    lora_depth_pix = decode_tokens(depth_tok, lora_pred_tokens)  # (N, 1, H, W)
print(f'LoRA depth decoded: range [{lora_depth_pix.min():.2f}, {lora_depth_pix.max():.2f}]\n')

# Token accuracy vs GT
correct_base = (baseline_pred_tokens == depth_tokens).float().mean().item()
correct_lora = (lora_pred_tokens      == depth_tokens).float().mean().item()
print(f'Token accuracy on these {NUM_SAMPLES} samples:')
print(f'  Baseline : {correct_base*100:.2f}%')
print(f'  LoRA     : {correct_lora*100:.2f}%')

# %% ─── Cell 9: tensor → numpy helpers ────────────────────────────────────

def to_rgb_np(tensor):
    """(B, 3, H, W) bfloat16 in [-1,1] → list of (H, W, 3) uint8 numpy arrays."""
    imgs = []
    for t in tensor.float().cpu():
        t = (t.clamp(-1, 1) + 1) / 2      # → [0, 1]
        t = t.permute(1, 2, 0).numpy()     # (H, W, 3)
        imgs.append((t * 255).astype(np.uint8))
    return imgs


def to_depth_np(tensor, colormap='magma'):
    """(B, 1, H, W) → list of (H, W, 3) uint8 colormapped depth images.

    Values are normalised per-sample so dynamic range is always fully used.
    """
    cmap = cm.get_cmap(colormap)
    imgs = []
    for t in tensor.float().cpu():
        d = t.squeeze(0).numpy()            # (H, W)
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
        rgba = cmap(d_norm)                 # (H, W, 4) in [0, 1]
        imgs.append((rgba[:, :, :3] * 255).astype(np.uint8))
    return imgs

# %% ─── Cell 10: assemble and save the comparison figure ──────────────────

rgb_imgs      = to_rgb_np(rgb_pixels)
gt_imgs       = to_depth_np(gt_depth_pix)
base_imgs     = to_depth_np(baseline_depth_pix)
lora_imgs     = to_depth_np(lora_depth_pix)

N_COLS = 4
fig, axes = plt.subplots(NUM_SAMPLES, N_COLS,
                         figsize=(N_COLS * 3.5, NUM_SAMPLES * 3.2))

col_titles = [
    'RGB (decoded)',
    'GT Depth (decoded)',
    '4M-21 baseline',
    f'4M-21 + LoRA r={LORA_RANK}',
]

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=6)

for row in range(NUM_SAMPLES):
    imgs = [rgb_imgs[row], gt_imgs[row], base_imgs[row], lora_imgs[row]]
    for col, img in enumerate(imgs):
        ax = axes[row, col]
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(f'Sample {common[row]}', fontsize=8, rotation=90,
                          labelpad=4, va='center')

# Add token accuracy as a figure suptitle
fig.suptitle(
    f'NYUv2 Depth  |  Token accuracy on {NUM_SAMPLES} val samples:\n'
    f'Baseline {correct_base*100:.2f}%   →   LoRA {correct_lora*100:.2f}%',
    fontsize=11, y=1.01
)

plt.tight_layout()
out_path = FIGURES_DIR / 'depth_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {out_path}')

# ── also show inline if running in JupyterLab ──────────────────────────────
try:
    from IPython.display import display, Image as IPImage
    display(IPImage(str(out_path)))
except ImportError:
    pass   # running as plain script — just open the file

# %% [markdown]
# ## Next steps
#
# 1. Run the 50-epoch full training:
#    ```bash
#    sbatch scripts/snellius/04_ft_depth_nyu.sh
#    ```
# 2. Repeat this notebook with `LORA_CKPT = 'output/depth/nyu_rgb_r8/lora_weights-best.pth'`
# 3. Compute pixel-space metrics (δ₁, RMSE, AbsRel) with `eval_depth_pixel.py`
