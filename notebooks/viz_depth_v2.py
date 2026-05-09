# %% [markdown]
# # 4M-21 Depth Estimation — Baseline vs LoRA (run 15: r8/selfattn/labeled)
#
# **Grid**: N rows × 4 columns
#   - Col 1: RGB image
#   - Col 2: GT depth (from `nyu_raw/test/depth/*.npy`, in metres, turbo colormap)
#   - Col 3: Baseline prediction (4M-21 XL, no LoRA)
#   - Col 4: LoRA prediction (run 15: rank=8, selfattn, trained on NYUv2 labeled)
#
# **Dataset**: NYUv2 labeled val split — 654 images in `nyu_raw/test/`
#
# **Run as script** (headless, saves figure):
#   python notebooks/viz_depth_v2.py
#
# **Run in Jupyter** (on Snellius):
#   Open this file in JupyterLab → Run All Cells

# %% [markdown]
# ## Config — edit paths here

# %%
import os, sys

REPO_ROOT    = os.path.expanduser('~/4m-peft')
os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'ml-4m'))

# ── Paths ─────────────────────────────────────────────────────────────────────
NYU_VAL      = '/scratch-shared/tesr125338/nyu_raw/test'        # 654 labeled val images
LORA_WEIGHTS = '/scratch-shared/tesr125338/output/depth/nyu_labeled_r8_selfattn_drop005/lora_weights-best.pth'
HF_CACHE     = '/scratch-shared/tesr125338/hf_cache'
FIGURES_DIR  = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── LoRA config (must match training run 15) ──────────────────────────────────
LORA_RANK    = 8
LORA_MODULES = 'selfattn'

# ── Visualisation ─────────────────────────────────────────────────────────────
N_VIZ        = 12      # rows in the visual grid
MAX_DEPTH    = 10.0    # metres — NYU standard clip
DIVAE_STEPS  = 50      # DiVAE DDIM steps (4M paper uses 50)
SEED         = 42

# ── 4M model IDs ──────────────────────────────────────────────────────────────
FM_MODEL      = 'EPFL-VILAB/4M-21_XL'
TOK_DEPTH_ID  = 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448'
TOK_TEXT_PATH = os.path.join(REPO_ROOT,
    'ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json')

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16
print(f'Device: {DEVICE}  |  dtype: {DTYPE}')
assert os.path.exists(LORA_WEIGHTS), f'LoRA weights not found: {LORA_WEIGHTS}'
print(f'LoRA : {LORA_WEIGHTS}  ✓')

# %% [markdown]
# ## Imports

# %%
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless on Snellius — remove for interactive Jupyter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

from fourm.models.fm import FM
from fourm.vq.vqvae import DiVAE
from fourm.models.generate import (
    GenerationSampler, build_chained_generation_schedules,
    init_empty_target_modality, init_full_input_modality,
)
from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import RGBTransform
from fourm.utils.plotting_utils import decode_tok_depth
from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names

print('Imports OK')

# %% [markdown]
# ## Helper functions

# %%
# ── Generation params (same as eval_depth.py) ─────────────────────────────────
DEPTH_GEN_PARAMS = {
    'tokens_per_target':        196,
    'autoregression_scheme':    'roar',
    'decoding_steps':           1,
    'token_decoding_schedule':  'linear',
    'temp':                     0.01,
    'temp_schedule':            'constant',
    'cfg_scale':                2.0,
    'cfg_schedule':             'constant',
}


def load_rgb_tensor(rgb_path):
    """RGB image → (1,3,224,224) ImageNet-normalised tensor."""
    rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
    img_pil = rgb_transform.load(rgb_path)
    img_pil = rgb_transform.preprocess(img_pil)
    s       = min(img_pil.size)
    img_pil = center_crop(img_pil, (s, s)).resize((224, 224), Image.BILINEAR)
    return rgb_transform.postprocess(img_pil).unsqueeze(0).to(device=DEVICE, dtype=DTYPE)


def load_rgb_display(rgb_path):
    """RGB → (224,224,3) uint8 numpy for display."""
    img = Image.open(rgb_path).convert('RGB')
    s   = min(img.size)
    img = center_crop(img, (s, s)).resize((224, 224), Image.BILINEAR)
    return np.array(img)


def load_gt_depth(depth_path):
    """GT depth .npy (float32, metres) → (224,224) + valid mask."""
    depth = np.load(depth_path).squeeze().astype(np.float32)
    h, w  = depth.shape
    s     = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    depth = depth[top:top+s, left:left+s]
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
    valid = (depth > 0) & (depth <= MAX_DEPTH)
    return depth, valid


def generate_depth(sampler, rgb_tensor, seed):
    """RGB tensor → depth prediction dict."""
    ntoks    = DEPTH_GEN_PARAMS['tokens_per_target']
    schedule = build_chained_generation_schedules(
        cond_domains            = ['rgb@224'],
        target_domains          = ['tok_depth@224'],
        tokens_per_target       = [ntoks],
        autoregression_schemes  = [DEPTH_GEN_PARAMS['autoregression_scheme']],
        decoding_steps          = [DEPTH_GEN_PARAMS['decoding_steps']],
        token_decoding_schedules= [DEPTH_GEN_PARAMS['token_decoding_schedule']],
        temps                   = [DEPTH_GEN_PARAMS['temp']],
        temp_schedules          = [DEPTH_GEN_PARAMS['temp_schedule']],
        cfg_scales              = [DEPTH_GEN_PARAMS['cfg_scale']],
        cfg_schedules           = [DEPTH_GEN_PARAMS['cfg_schedule']],
        cfg_grow_conditioning   = True,
    )
    eos_id = tok_text.token_to_id('[EOS]')
    sd = {'rgb@224': {'tensor': rgb_tensor}}
    sd = init_full_input_modality(sd, MODALITY_INFO, 'rgb@224', DEVICE, eos_id=eos_id)
    sd = init_empty_target_modality(sd, MODALITY_INFO, 'tok_depth@224', 1, ntoks, DEVICE)
    return sampler.generate(sd, schedule, text_tokenizer=tok_text, verbose=False, seed=seed)


def decode_depth_raw(out_dict):
    """Depth tokens → raw z-score map (224,224) float32."""
    raw = decode_tok_depth(
        out_dict, {'tok_depth': tok_depth},
        key='tok_depth@224', image_size=224, patch_size=16,
        t=DIVAE_STEPS, verbose=False, cmap=None,
    )
    return (raw[0] if raw.ndim == 3 else raw).astype(np.float32)


def scale_shift_align(pred, gt, valid):
    """Least-squares affine alignment (per-image scale + shift)."""
    p = pred[valid].reshape(-1, 1).astype(np.float64)
    g = gt[valid].reshape(-1).astype(np.float64)
    A = np.hstack([p, np.ones_like(p)])
    scale, shift = np.linalg.lstsq(A, g, rcond=None)[0]
    aligned = np.clip(scale * pred + shift, 1e-3, None).astype(np.float32)
    return aligned, float(scale), float(shift)


def colorise_depth(depth_m, vmin=0.0, vmax=MAX_DEPTH):
    """Depth map in metres → (224,224,3) uint8, turbo colormap, fixed [vmin, vmax]."""
    d = np.clip(depth_m, vmin, vmax)
    d_norm = (d - vmin) / (vmax - vmin + 1e-8)
    rgba = cm.turbo(d_norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


print('Helper functions defined.')

# %% [markdown]
# ## Load samples and text tokenizer

# %%
tok_text = Tokenizer.from_file(TOK_TEXT_PATH)
print(f'Text tokenizer ready (vocab={tok_text.get_vocab_size()})')

rgb_dir   = Path(NYU_VAL) / 'rgb'
depth_dir = Path(NYU_VAL) / 'depth'

rgb_files   = sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg'))
all_samples = []
for rgb_p in rgb_files:
    depth_p = depth_dir / f'{rgb_p.stem}.npy'
    if depth_p.exists():
        all_samples.append({'rgb': str(rgb_p), 'depth': str(depth_p), 'name': rgb_p.stem})

print(f'NYUv2 labeled val samples: {len(all_samples)}')
assert len(all_samples) > 0, 'No samples found — check NYU_VAL path'

# Fixed subset for reproducibility
rng = np.random.default_rng(SEED)
indices = rng.choice(len(all_samples), size=N_VIZ, replace=False)
indices.sort()
viz_samples = [all_samples[i] for i in indices]
print(f'Viz samples: {N_VIZ}')

# %% [markdown]
# ## Load DiVAE (depth tokenizer)

# %%
print('Loading depth DiVAE ...')
tok_depth = DiVAE.from_pretrained(TOK_DEPTH_ID, cache_dir=HF_CACHE).to(DEVICE).eval()
print('DiVAE ready.')

# %% [markdown]
# ## Phase 1 — Baseline inference
# Load 4M-21 XL (no LoRA), generate depth for all samples, free GPU memory.

# %%
print('Loading baseline FM ...')
fm = FM.from_pretrained(FM_MODEL, cache_dir=HF_CACHE)
fm = fm.to(device=DEVICE, dtype=DTYPE).eval()
sampler = GenerationSampler(fm)
print(f'FM ready ({sum(p.numel() for p in fm.parameters())/1e6:.0f}M params)')

preds_baseline = {}    # name → raw z-score (224,224) float32
for s in tqdm(viz_samples, desc='Baseline'):
    with torch.no_grad():
        rgb_t   = load_rgb_tensor(s['rgb'])
        out     = generate_depth(sampler, rgb_t, seed=SEED)
        raw     = decode_depth_raw(out)
    preds_baseline[s['name']] = raw

del fm, sampler
torch.cuda.empty_cache()
print(f'Baseline done. {len(preds_baseline)} predictions cached.')

# %% [markdown]
# ## Phase 2 — LoRA inference
# Load 4M-21 XL + inject LoRA architecture + load weights, generate depth.

# %%
print('Loading LoRA FM ...')
fm = FM.from_pretrained(FM_MODEL, cache_dir=HF_CACHE)
target_mods = get_LoRA_module_names(LORA_MODULES)
inject_trainable_LoRA(fm, rank=LORA_RANK, scale=1.0, target_replace_modules=target_mods)
lora_sd = torch.load(LORA_WEIGHTS, map_location='cpu')
missing, unexpected = fm.load_state_dict(lora_sd, strict=False)
print(f'  Loaded {len(lora_sd)} LoRA tensors (missing={len(missing)}, unexpected={len(unexpected)})')
fm = fm.to(device=DEVICE, dtype=DTYPE).eval()
sampler = GenerationSampler(fm)

preds_lora = {}        # name → raw z-score (224,224) float32
for s in tqdm(viz_samples, desc='LoRA r8/selfattn'):
    with torch.no_grad():
        rgb_t   = load_rgb_tensor(s['rgb'])
        out     = generate_depth(sampler, rgb_t, seed=SEED)
        raw     = decode_depth_raw(out)
    preds_lora[s['name']] = raw

del fm, sampler
torch.cuda.empty_cache()
print(f'LoRA done. {len(preds_lora)} predictions cached.')

# %% [markdown]
# ## Visual grid
# RGB | GT depth | Baseline | LoRA — turbo colormap, fixed [0, MAX_DEPTH] range.

# %%
n_rows = N_VIZ

# Extra width on the right for the colorbar (last column wider)
fig = plt.figure(figsize=(18, n_rows * 3.8))
gs  = fig.add_gridspec(n_rows, 5,
                        width_ratios=[4, 4, 4, 4, 0.25],
                        hspace=0.05, wspace=0.05)

axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n_rows)]
cbar_ax = fig.add_subplot(gs[:, 4])   # tall, narrow axes for colorbar

col_titles = [
    'RGB\n(NYUv2 labeled val)',
    'GT Depth\n(metres)',
    'Baseline\n(4M-21 XL, no LoRA)',
    'LoRA r8/selfattn\n(depth FT, run 15)',
]
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=9, fontweight='bold', pad=6)

for row, s in enumerate(viz_samples):
    gt_depth, valid = load_gt_depth(s['depth'])
    pred_b_aligned, *_ = scale_shift_align(preds_baseline[s['name']], gt_depth, valid)
    pred_l_aligned, *_ = scale_shift_align(preds_lora[s['name']],     gt_depth, valid)

    axes[row][0].imshow(load_rgb_display(s['rgb']))
    axes[row][1].imshow(colorise_depth(gt_depth))
    axes[row][2].imshow(colorise_depth(pred_b_aligned))
    axes[row][3].imshow(colorise_depth(pred_l_aligned))

    axes[row][0].set_ylabel(s['name'], fontsize=7, rotation=0, labelpad=28, va='center')

    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])

# Colorbar in the dedicated axes — no overlap with images
sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(0, MAX_DEPTH))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Depth (metres)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    'NYUv2 labeled val — 4M-21 XL  |  Baseline vs LoRA r8/selfattn (depth FT)',
    fontsize=11, fontweight='bold', y=1.005,
)

out_path = os.path.join(FIGURES_DIR, 'depth_comparison_r8_labeled.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {out_path}')

try:
    from IPython.display import display, Image as IPImage
    display(IPImage(out_path))
except Exception:
    pass
