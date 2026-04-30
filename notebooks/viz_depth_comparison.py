# %% [markdown]
# # 4M-21 Depth Estimation: Baseline vs LoRA Visual Comparison
#
# Visualises 5 validation samples as a 5×4 grid:
#   Col 1 – Original RGB  (from NYUv2 HF dataset)
#   Col 2 – GT depth      (decoded from saved tok_depth tokens via DiVAE)
#   Col 3 – Baseline pred (4M-21 zero-shot → decoded via DiVAE)
#   Col 4 – LoRA pred     (4M-21 + fine-tuned LoRA → decoded via DiVAE)
#
# Pixel-space metrics (δ₁, RMSE, AbsRel) computed on these 5 samples.
# NOTE: metrics are in normalised depth space (not metres) — for proper
#       metric-depth evaluation use eval_depth_pixel.py on the full val set.
#
# Run on Snellius (single GPU):
#   python notebooks/viz_depth_comparison.py
# Output: figures/depth_comparison.png

# %% --- Cell 1: imports & paths ----------------------------------------------
import sys, io
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO_ROOT    = Path.home() / '4m-peft'
sys.path.insert(0, str(REPO_ROOT / 'ml-4m'))

HF_CACHE     = '/scratch-shared/tesr125338/hf_cache'
NYU_VAL      = '/scratch-shared/tesr125338/nyu_tokens/val'
LORA_CKPT    = str(REPO_ROOT / 'output/depth/nyu_rgb_r8_e5/lora_weights-best.pth')
FIGURES_DIR  = REPO_ROOT / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

NUM_SAMPLES  = 5
LORA_RANK    = 8
LORA_SCALE   = 1.0
PRETRAINED   = 'EPFL-VILAB/4M-21_XL'
DEPTH_TOK_REPO = 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16

print(f'Device : {DEVICE}')
assert Path(LORA_CKPT).exists(), f'LoRA checkpoint not found: {LORA_CKPT}'
print(f'LoRA   : {LORA_CKPT}')

# %% --- Cell 2: load depth DiVAE tokenizer (encoder + diffusion decoder) ----
from fourm.vq.vqvae import DiVAE

print('Loading Depth DiVAE tokenizer (encoder + decoder)...')
depth_tok = DiVAE.from_pretrained(DEPTH_TOK_REPO, cache_dir=HF_CACHE).eval().to(DEVICE)
print(f'  has decoder: {hasattr(depth_tok, "decoder")}')
print('Depth tokenizer ready.\n')

# %% --- Cell 3: load 5 pre-tokenised val samples ----------------------------
val_root  = Path(NYU_VAL)
rgb_dir   = val_root / 'tok_rgb'
depth_dir = val_root / 'tok_depth'

rgb_stems   = sorted(p.stem for p in rgb_dir.glob('**/*.npy'))
depth_stems = sorted(p.stem for p in depth_dir.glob('**/*.npy'))
common = sorted(set(rgb_stems) & set(depth_stems))[:NUM_SAMPLES]
print(f'Selected {len(common)} samples: {common}')

def _load_npy(folder, stem):
    matches = list(folder.rglob(f'{stem}.npy'))
    assert matches, f'{stem}.npy not found under {folder}'
    return np.load(matches[0]).astype(np.int64)

rgb_tokens   = torch.from_numpy(
    np.stack([_load_npy(rgb_dir,   s) for s in common])).to(DEVICE)  # (N,196)
depth_tokens = torch.from_numpy(
    np.stack([_load_npy(depth_dir, s) for s in common])).to(DEVICE)  # (N,196)

print(f'RGB tokens   : {rgb_tokens.shape}')
print(f'Depth tokens : {depth_tokens.shape}')

# %% --- Cell 4: load original RGB images from cached tar files --------------
import tarfile, h5py
from PIL import Image

snap_dir = Path(HF_CACHE) / 'datasets--sayakpaul--nyu_depth_v2'
data_dir = sorted((snap_dir / 'snapshots').iterdir())[-1] / 'data'
val_tars = sorted(data_dir.glob('val-*.tar'))
print(f'Found {len(val_tars)} val tar files')

def load_rgb_from_tars(tar_paths, indices):
    indices_set = set(indices)
    results = {}
    global_idx = 0
    for tar_path in sorted(tar_paths):
        with tarfile.open(tar_path) as tar:
            h5_members = sorted(
                [m for m in tar.getmembers() if m.name.endswith('.h5')],
                key=lambda m: m.name)
            for member in h5_members:
                if global_idx in indices_set:
                    fobj = tar.extractfile(member)
                    with h5py.File(io.BytesIO(fobj.read()), 'r') as f:
                        rgb = f['rgb'][...]   # (3, H, W) uint8
                    results[global_idx] = rgb
                    if len(results) == len(indices_set):
                        return results
                global_idx += 1
    return results

sample_indices = [int(s) for s in common]
raw = load_rgb_from_tars(val_tars, sample_indices)

SIZE = 224
rgb_imgs = []
for idx in sample_indices:
    rgb = raw[idx]   # (3, H, W)
    pil = Image.fromarray(rgb.transpose(1, 2, 0))
    rgb_imgs.append(np.array(pil.resize((SIZE, SIZE), Image.BILINEAR)))

print(f'Loaded {len(rgb_imgs)} RGB images, shape {rgb_imgs[0].shape}')

# %% --- Cell 5: decode GT depth tokens → pixel space -----------------------

@torch.no_grad()
def decode_depth(tokens_flat, timesteps=25):
    """Decode (B, 196) depth token indices → (B, 224, 224) float tensor.

    Uses the DiVAE diffusion decoder (25 DDIM steps).
    Returns values in the tokenizer's normalised space (roughly [-1, 1]).
    """
    B = tokens_flat.shape[0]
    tokens_2d = tokens_flat.reshape(B, 14, 14).long()
    dec = depth_tok.decode_tokens(tokens_2d, timesteps=timesteps,
                                  image_size=SIZE, verbose=False)
    return dec[:, 0].float()   # (B, H, W)


print('Decoding GT depth tokens...')
gt_depth = decode_depth(depth_tokens)   # (N, 224, 224)
print(f'GT depth decoded: {gt_depth.shape}, '
      f'range [{gt_depth.min():.3f}, {gt_depth.max():.3f}]')

# %% --- Cell 6: mod_dict helper ---------------------------------------------

def build_mod_dict(input_tokens, target_tokens,
                   input_mod='tok_rgb@224', target_mod='tok_depth@224',
                   num_input=196, num_target=196):
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

# %% --- Cell 7: load 4M-21_XL + inject LoRA --------------------------------
from fourm.models.fm import FM
from fourm.models.lora_utils import get_LoRA_module_names, inject_trainable_LoRA

print(f'Loading {PRETRAINED} ...  (this may take ~2 min)')
model = FM.from_pretrained(PRETRAINED, cache_dir=HF_CACHE)
model = model.to(DTYPE).to(DEVICE).eval()

target_modules = get_LoRA_module_names('attention')
inject_trainable_LoRA(model, rank=LORA_RANK, scale=LORA_SCALE,
                      target_replace_modules=target_modules)
model = model.to(DEVICE)   # move newly created LoRA layers to GPU
print('Model ready (LoRA injected, weights = 0 → pure baseline).\n')

# %% --- Cell 8: inference helper --------------------------------------------

@torch.no_grad()
def predict_depth_tokens(model, rgb_toks, depth_toks):
    """Forward pass → predicted depth token indices (B, 196)."""
    mod_dict = build_mod_dict(rgb_toks, depth_toks)
    with torch.cuda.amp.autocast(dtype=DTYPE, enabled=(DEVICE.type == 'cuda')):
        mod_logits = model(mod_dict,
                           num_encoder_tokens=196,
                           num_decoder_tokens=196,
                           return_logits=True)
    return mod_logits['tok_depth@224'].float().argmax(dim=-1)   # (B, 196)

# %% --- Cell 9: baseline inference + decode ---------------------------------
print('Baseline inference (LoRA = 0)...')
baseline_tokens = predict_depth_tokens(model, rgb_tokens, depth_tokens)
baseline_acc = (baseline_tokens == depth_tokens).float().mean().item()
print(f'Baseline token accuracy: {baseline_acc*100:.2f}%')

print('Decoding baseline depth predictions...')
baseline_depth = decode_depth(baseline_tokens)   # (N, 224, 224)

# %% --- Cell 10: load LoRA weights + inference + decode ---------------------
print(f'Loading LoRA weights from {LORA_CKPT}...')
lora_sd = torch.load(LORA_CKPT, map_location=DEVICE)
model.load_state_dict(lora_sd, strict=False)
print(f'  {len(lora_sd)} LoRA tensors loaded')

print('LoRA inference...')
lora_tokens = predict_depth_tokens(model, rgb_tokens, depth_tokens)
lora_acc = (lora_tokens == depth_tokens).float().mean().item()
print(f'LoRA token accuracy: {lora_acc*100:.2f}%  '
      f'(improvement: {(lora_acc-baseline_acc)*100:+.2f}pp)')

print('Decoding LoRA depth predictions...')
lora_depth = decode_depth(lora_tokens)   # (N, 224, 224)

# %% --- Cell 11: pixel-space metrics ----------------------------------------

def compute_metrics(pred, gt, eps=1e-6):
    """Compute δ₁, RMSE, AbsRel between predicted and GT depth tensors.

    Both pred and gt are (B, H, W) float in the same normalised space.
    NOTE: values are in tokenizer-normalised units, not metres.
          For metric-depth evaluation in metres use eval_depth_pixel.py.
    """
    # Mask out near-zero GT (invalid / background)
    mask = gt > eps

    pred_m = pred[mask]
    gt_m   = gt[mask]

    # δ₁: % of pixels where max(pred/gt, gt/pred) < 1.25
    ratio  = torch.max(pred_m / gt_m, gt_m / pred_m)
    delta1 = (ratio < 1.25).float().mean().item()

    # RMSE
    rmse = torch.sqrt(((pred_m - gt_m) ** 2).mean()).item()

    # AbsRel
    absrel = ((pred_m - gt_m).abs() / gt_m).mean().item()

    return {'δ₁': delta1, 'RMSE': rmse, 'AbsRel': absrel}


# Clamp to positive range for ratio-based metrics
gt_pos       = gt_depth.clamp(min=1e-3)
baseline_pos = baseline_depth.clamp(min=1e-3)
lora_pos     = lora_depth.clamp(min=1e-3)

metrics_base = compute_metrics(baseline_pos, gt_pos)
metrics_lora = compute_metrics(lora_pos,     gt_pos)

print('\n── Pixel-space metrics on 5 val samples (normalised space) ──')
print(f'{"Metric":<10} {"Baseline":>12} {"LoRA":>12} {"Δ":>10}')
print('─' * 46)
for k in ['δ₁', 'RMSE', 'AbsRel']:
    b, l = metrics_base[k], metrics_lora[k]
    arrow = '↑' if k == 'δ₁' else '↓'   # higher δ₁ better, lower RMSE/AbsRel better
    print(f'{k:<10} {b:>12.4f} {l:>12.4f} {l-b:>+10.4f} {arrow}')

# %% --- Cell 12: colormap helper --------------------------------------------

def colorise(depth_tensor):
    """(H, W) float tensor → (H, W, 3) uint8, magma colormap, per-image norm."""
    d = depth_tensor.cpu().numpy()
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
    rgba = cm.magma(d_norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)

# %% --- Cell 13: assemble and save figure -----------------------------------

fig, axes = plt.subplots(NUM_SAMPLES, 4,
                         figsize=(4 * 3.8, NUM_SAMPLES * 3.5))

col_titles = [
    'RGB',
    'GT Depth (decoded)',
    f'Baseline  δ₁={metrics_base["δ₁"]:.3f}',
    f'LoRA r={LORA_RANK}  δ₁={metrics_lora["δ₁"]:.3f}',
]

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=8)

for row in range(NUM_SAMPLES):
    ax_rgb, ax_gt, ax_base, ax_lora = axes[row]

    ax_rgb.imshow(rgb_imgs[row])
    ax_rgb.set_ylabel(f'#{common[row]}', fontsize=9, rotation=0,
                      labelpad=30, va='center')

    ax_gt.imshow(colorise(gt_depth[row]))
    ax_base.imshow(colorise(baseline_depth[row]))
    ax_lora.imshow(colorise(lora_depth[row]))

    for ax in [ax_rgb, ax_gt, ax_base, ax_lora]:
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle(
    f'NYUv2 depth  |  5 val samples  (normalised space, not metres)\n'
    f'Baseline  δ₁={metrics_base["δ₁"]:.3f}  RMSE={metrics_base["RMSE"]:.3f}  AbsRel={metrics_base["AbsRel"]:.3f}\n'
    f'LoRA      δ₁={metrics_lora["δ₁"]:.3f}  RMSE={metrics_lora["RMSE"]:.3f}  AbsRel={metrics_lora["AbsRel"]:.3f}',
    fontsize=11, y=1.02
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
# ## Notes on metrics
#
# δ₁ (higher is better): fraction of pixels where max(pred/gt, gt/pred) < 1.25
# RMSE (lower is better): root mean square error
# AbsRel (lower is better): mean absolute relative error
#
# These metrics are computed in the DiVAE-decoded normalised space.
# For paper-ready results in metres → run eval_depth_pixel.py on
# the full val set after the 50-epoch training completes.
#
# ## Next steps
# 1. Launch 50-epoch full run:
#    sbatch scripts/snellius/04_ft_depth_nyu.sh
# 2. After 50 epochs: run eval_depth_pixel.py for metric-depth evaluation
