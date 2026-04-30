# %% [markdown]
# # 4M-21 Depth Estimation: Baseline vs LoRA Comparison
#
# Visualises 5 validation samples as a 5×4 grid:
#   Col 1 – Original RGB (from NYUv2 HF dataset)
#   Col 2 – Ground-truth depth (from NYUv2 HF dataset, magma colormap)
#   Col 3 – Baseline token correctness map (4M-21 zero-shot, no LoRA)
#   Col 4 – LoRA token correctness map (4M-21 + fine-tuned LoRA)
#
# Correctness maps: each cell in the 14×14 grid is GREEN if the model
# predicted the correct depth token, RED otherwise.
# Overall accuracy is shown as a % above each column.
#
# Run on Snellius (single GPU node):
#   python notebooks/viz_depth_comparison.py
# Output: figures/depth_comparison.png

# %% --- Cell 1: imports & paths ----------------------------------------------
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # headless on Snellius; use 'TkAgg' if running locally
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- repo root ----------------------------------------------------------------
REPO_ROOT = Path.home() / '4m-peft'
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

# --- paths -------------------------------------------------------------------
HF_CACHE     = '/scratch-shared/tesr125338/hf_cache'
NYU_VAL      = '/scratch-shared/tesr125338/nyu_tokens/val'
LORA_CKPT    = str(REPO_ROOT / 'output/depth/nyu_rgb_r8_e5/lora_weights-best.pth')
FIGURES_DIR  = REPO_ROOT / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

NUM_SAMPLES  = 5
LORA_RANK    = 8
LORA_SCALE   = 1.0
PRETRAINED   = 'EPFL-VILAB/4M-21_XL'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16

print(f'Device : {DEVICE}')
assert Path(LORA_CKPT).exists(), (
    f'\nLoRA checkpoint not found: {LORA_CKPT}\n'
    f'Run the 5-epoch validation job first:\n'
    f'  sbatch scripts/snellius/05_ft_depth_nyu_val5.sh'
)
print(f'LoRA checkpoint : {LORA_CKPT}')

# %% --- Cell 2: load 5 pre-tokenised val samples ----------------------------
val_root  = Path(NYU_VAL)
rgb_dir   = val_root / 'tok_rgb'
depth_dir = val_root / 'tok_depth'

rgb_stems   = sorted(p.stem for p in rgb_dir.glob('**/*.npy'))
depth_stems = sorted(p.stem for p in depth_dir.glob('**/*.npy'))
common = sorted(set(rgb_stems) & set(depth_stems))[:NUM_SAMPLES]
print(f'Selected {len(common)} samples: {common}')

def _load_npy(folder: Path, stem: str) -> np.ndarray:
    matches = list(folder.rglob(f'{stem}.npy'))
    assert matches, f'{stem}.npy not found under {folder}'
    return np.load(matches[0]).astype(np.int64)

rgb_tokens   = torch.from_numpy(
    np.stack([_load_npy(rgb_dir,   s) for s in common])).to(DEVICE)   # (N, 196)
depth_tokens = torch.from_numpy(
    np.stack([_load_npy(depth_dir, s) for s in common])).to(DEVICE)   # (N, 196)

print(f'RGB tokens   : {rgb_tokens.shape}')
print(f'Depth tokens : {depth_tokens.shape}')

# %% --- Cell 3: load original RGB and depth images from HF dataset ----------
from datasets import load_dataset
from PIL import Image

print('Loading HF dataset from local cache (no internet needed)...')
ds_val = load_dataset('sayakpaul/nyu_depth_v2',
                      cache_dir=HF_CACHE,
                      split='validation')
print(f'Validation split: {len(ds_val)} samples')

# .npy stems are zero-padded indices → direct dataset indices
sample_indices = [int(s) for s in common]
print(f'Loading indices: {sample_indices}')

SIZE = 224  # match tokenisation resolution (14×14 patches × 16px = 224)

rgb_imgs   = []
depth_imgs = []
for idx in sample_indices:
    sample = ds_val[idx]
    rgb_imgs.append(
        np.array(sample['image'].convert('RGB').resize((SIZE, SIZE), Image.BILINEAR),
                 dtype=np.uint8))                                         # (H, W, 3)
    depth_np = np.array(sample['depth_map'], dtype=np.float32)           # metres
    depth_np = np.array(
        Image.fromarray(depth_np).resize((SIZE, SIZE), Image.NEAREST))
    depth_imgs.append(depth_np)                                          # (H, W)

print(f'Loaded {len(rgb_imgs)} RGB images and {len(depth_imgs)} depth maps.')

# %% --- Cell 4: mod_dict helper ---------------------------------------------

def build_mod_dict(input_tokens, target_tokens,
                   input_mod='tok_rgb@224', target_mod='tok_depth@224',
                   num_input=196, num_target=196):
    """Build a mod_dict for a single-input → single-target 4M forward pass.

    input_tokens  : (B, num_input)  long  — fully visible to the encoder
    target_tokens : (B, num_target) long  — fully hidden; all predicted
    """
    B, device = input_tokens.shape[0], input_tokens.device
    return {
        input_mod: {
            'tensor'                : input_tokens,
            'input_mask'            : torch.zeros((B, num_input),  dtype=torch.bool, device=device),
            'target_mask'           : torch.ones( (B, num_input),  dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.zeros((B, num_input),  dtype=torch.int,  device=device),
        },
        target_mod: {
            'tensor'                : target_tokens,
            'input_mask'            : torch.ones( (B, num_target), dtype=torch.bool, device=device),
            'target_mask'           : torch.zeros((B, num_target), dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.ones( (B, num_target), dtype=torch.int,  device=device),
        },
    }

# %% --- Cell 5: load 4M-21_XL + inject LoRA (weights = 0 at init) ----------
from fourm.models.fm import FM
from fourm.models.lora_utils import get_LoRA_module_names, inject_trainable_LoRA

print(f'Loading {PRETRAINED} ...  (this may take ~2 min)')
model = FM.from_pretrained(PRETRAINED, cache_dir=HF_CACHE)
model = model.to(DTYPE).to(DEVICE).eval()
print('Base model loaded.')

# Inject LoRA — at this point lora_up weights are all zeros,
# so the model behaves identically to the pre-trained 4M-21_XL baseline.
target_modules = get_LoRA_module_names('attention')
inject_trainable_LoRA(model, rank=LORA_RANK, scale=LORA_SCALE,
                      target_replace_modules=target_modules)
print('LoRA injected (weights = 0 → pure baseline).\n')

# %% --- Cell 6: baseline inference (LoRA weights = 0) -----------------------

@torch.no_grad()
def predict_depth_tokens(model, rgb_toks, depth_toks):
    """Run one forward pass and return predicted depth token indices.

    Returns: (B, 196) long tensor.
    """
    mod_dict = build_mod_dict(rgb_toks, depth_toks)
    with torch.cuda.amp.autocast(dtype=DTYPE, enabled=(DEVICE.type == 'cuda')):
        mod_logits = model(mod_dict,
                           num_encoder_tokens=196,
                           num_decoder_tokens=196,
                           return_logits=True)
    logits = mod_logits['tok_depth@224']         # (B, 196, vocab_size)
    return logits.float().argmax(dim=-1)         # (B, 196)


print('Running baseline inference (LoRA = 0)...')
baseline_tokens = predict_depth_tokens(model, rgb_tokens, depth_tokens)

baseline_acc = (baseline_tokens == depth_tokens).float().mean().item()
print(f'Baseline token accuracy: {baseline_acc*100:.2f}%\n')

# %% --- Cell 7: load LoRA weights and run fine-tuned inference --------------
print(f'Loading LoRA weights from {LORA_CKPT}...')
lora_sd = torch.load(LORA_CKPT, map_location=DEVICE)
missing, unexpected = model.load_state_dict(lora_sd, strict=False)
print(f'  {len(lora_sd)} LoRA tensors loaded')
print(f'  Missing (non-LoRA params, expected): {len(missing)}')
print(f'  Unexpected: {len(unexpected)}')

print('\nRunning LoRA inference...')
lora_tokens = predict_depth_tokens(model, rgb_tokens, depth_tokens)

lora_acc = (lora_tokens == depth_tokens).float().mean().item()
print(f'LoRA token accuracy : {lora_acc*100:.2f}%')
print(f'Improvement         : +{(lora_acc - baseline_acc)*100:.2f}pp\n')

# %% --- Cell 8: build correctness maps (14×14 grids) -----------------------

def correctness_map(pred_tokens, gt_tokens):
    """Return a (N, 14, 14) boolean array: True where prediction == GT."""
    correct = (pred_tokens == gt_tokens).cpu().numpy()   # (N, 196)
    return correct.reshape(-1, 14, 14)                   # (N, 14, 14)


baseline_correct = correctness_map(baseline_tokens, depth_tokens)  # (N,14,14)
lora_correct     = correctness_map(lora_tokens,     depth_tokens)  # (N,14,14)

# Per-sample accuracy
baseline_acc_per = baseline_correct.mean(axis=(1, 2))   # (N,)
lora_acc_per     = lora_correct.mean(axis=(1, 2))        # (N,)

print('Per-sample token accuracy:')
for i, s in enumerate(common):
    print(f'  Sample {s}:  baseline {baseline_acc_per[i]*100:.2f}%  '
          f'→  LoRA {lora_acc_per[i]*100:.2f}%  '
          f'({(lora_acc_per[i]-baseline_acc_per[i])*100:+.2f}pp)')

# %% --- Cell 9: depth colormap helper ---------------------------------------
import matplotlib.cm as cm

def colorise_depth(depth_np):
    """Normalise a (H, W) float depth array and apply magma colormap.

    Returns (H, W, 3) uint8.
    """
    d = depth_np.astype(np.float32)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
    rgba = cm.magma(d_norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)

# %% --- Cell 10: assemble and save the figure --------------------------------

fig, axes = plt.subplots(NUM_SAMPLES, 4,
                         figsize=(4 * 3.8, NUM_SAMPLES * 3.5))

col_titles = [
    'RGB',
    'GT Depth',
    f'Baseline (acc {baseline_acc*100:.2f}%)',
    f'LoRA r={LORA_RANK} (acc {lora_acc*100:.2f}%)',
]

# Colour map for correctness: 0=wrong (red), 1=correct (green)
CMAP_CORRECT = matplotlib.colors.ListedColormap(['#d62728', '#2ca02c'])

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=8)

for row in range(NUM_SAMPLES):
    ax_rgb, ax_depth, ax_base, ax_lora = axes[row]

    # Col 1 — RGB
    ax_rgb.imshow(rgb_imgs[row])
    ax_rgb.set_ylabel(f'#{common[row]}', fontsize=9, rotation=0,
                      labelpad=28, va='center')

    # Col 2 — GT depth
    ax_depth.imshow(colorise_depth(depth_imgs[row]))

    # Col 3 — Baseline correctness map
    ax_base.imshow(baseline_correct[row], cmap=CMAP_CORRECT,
                   vmin=0, vmax=1, interpolation='nearest')
    ax_base.set_xlabel(f'{baseline_acc_per[row]*100:.1f}% correct',
                       fontsize=8)

    # Col 4 — LoRA correctness map
    ax_lora.imshow(lora_correct[row], cmap=CMAP_CORRECT,
                   vmin=0, vmax=1, interpolation='nearest')
    ax_lora.set_xlabel(f'{lora_acc_per[row]*100:.1f}% correct  '
                       f'({(lora_acc_per[row]-baseline_acc_per[row])*100:+.1f}pp)',
                       fontsize=8)

    for ax in [ax_rgb, ax_depth, ax_base, ax_lora]:
        ax.set_xticks([])
        ax.set_yticks([])

# Legend for correctness maps
legend_handles = [
    mpatches.Patch(color='#2ca02c', label='Correct token'),
    mpatches.Patch(color='#d62728', label='Wrong token'),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=2, fontsize=10, bbox_to_anchor=(0.72, -0.01))

fig.suptitle(
    f'NYUv2 depth  |  5 val samples  |  '
    f'Baseline {baseline_acc*100:.2f}%  →  LoRA {lora_acc*100:.2f}%  '
    f'({(lora_acc-baseline_acc)*100:+.2f}pp)',
    fontsize=12, y=1.01
)

plt.tight_layout()
out_path = FIGURES_DIR / 'depth_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {out_path}')

try:
    from IPython.display import display, Image as IPImage
    display(IPImage(str(out_path)))
except ImportError:
    pass

# %% [markdown]
# ## What the correctness maps show
#
# Each cell in the 14×14 grid corresponds to one depth token (covers a 16×16 px
# patch of the 224×224 image). GREEN = the model predicted the exact correct
# VQ-VAE token index; RED = wrong prediction.
#
# A random baseline would be ~0.012% correct (1/8192 vocab).
# Any meaningful structure in the green pattern (e.g. flat walls, floors)
# suggests the model has learned scene geometry even at this token level.
#
# ## Next steps
# 1. Launch the 50-epoch full run:
#    sbatch scripts/snellius/04_ft_depth_nyu.sh
# 2. Once a decoder is available, replace correctness maps with pixel-space
#    depth images (eval_depth_pixel.py with delta1, RMSE, AbsRel)
