# %% [markdown]
# # 4M-21 Caption Quality + Depth Forgetting — Run 20 (r4/selfattn/drop01)
#
# **Two evaluations in one notebook** — the FM is loaded only twice (baseline,
# then caption LoRA), and both tasks are run per model to save GPU load time.
#
# ## Grid A — Caption quality (COCO val2017)
# | RGB | GT Caption | Baseline | Caption LoRA |
#
# ## Grid B — Depth forgetting (NYUv2 labeled val)
# | RGB | GT Depth | Baseline | Caption LoRA |
# (Caption LoRA was NOT trained on depth — visual diff vs baseline = forgetting signal)
#
# **Datasets**:
#   COCO val2017: 5000 images (never subsampled; train used 10k/~118k, seed=42)
#   NYUv2 val   :  654 images (labeled test split)
#
# **Run as script** (headless):
#   python notebooks/viz_caption_and_depth_forgetting.py
#
# **Run in Jupyter** on Snellius:
#   jupytext --to notebook notebooks/viz_caption_and_depth_forgetting.py
#   Open the .ipynb → Run All Cells

# %% [markdown]
# ## Config — edit paths here

# %%
import os, sys

REPO_ROOT = os.path.expanduser('~/4m-peft')
os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'ml-4m'))

# ── COCO paths ────────────────────────────────────────────────────────────────
COCO_ROOT = '/scratch-shared/tesr125338/coco/val2017'
COCO_ANN  = '/scratch-shared/tesr125338/coco/annotations/captions_val2017.json'

# ── NYU paths ─────────────────────────────────────────────────────────────────
NYU_VAL   = '/scratch-shared/tesr125338/nyu_raw/test'

# ── Caption LoRA checkpoint (run 20: r4/selfattn/drop01) ──────────────────────
LORA_WEIGHTS = '/scratch-shared/tesr125338/output/caption/coco_rgb_r4_selfattn_drop01/lora_weights-best.pth'
LORA_RANK    = 4
LORA_MODULES = 'selfattn'

# ── Output ────────────────────────────────────────────────────────────────────
HF_CACHE    = '/scratch-shared/tesr125338/hf_cache'
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Visualisation ─────────────────────────────────────────────────────────────
N_VIZ_CAP   = 10     # rows in caption grid
N_VIZ_DEPTH = 12     # rows in depth grid
MAX_DEPTH   = 10.0   # metres — NYU standard clip
DIVAE_STEPS = 50
SEED        = 42

# ── 4M model IDs ──────────────────────────────────────────────────────────────
FM_MODEL      = 'EPFL-VILAB/4M-21_XL'
TOK_DEPTH_ID  = 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448'
TOK_TEXT_PATH = os.path.join(REPO_ROOT,
    'ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json')

# ── Caption generation params ─────────────────────────────────────────────────
CAPTION_GEN_PARAMS = {
    'tokens_per_target':       64,
    'autoregression_scheme':   'autoregressive',
    'decoding_steps':          64,
    'token_decoding_schedule': 'linear',
    'temp':                    0.01,
    'temp_schedule':           'constant',
    'cfg_scale':               2.0,
    'cfg_schedule':            'constant',
}

# ── Depth generation params ───────────────────────────────────────────────────
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

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16
print(f'Device : {DEVICE}  |  dtype: {DTYPE}')
assert os.path.exists(LORA_WEIGHTS), f'LoRA weights not found: {LORA_WEIGHTS}'
assert os.path.exists(COCO_ANN),     f'COCO annotations not found: {COCO_ANN}'
print(f'LoRA   : {LORA_WEIGHTS}  ✓')
print(f'COCO   : {COCO_ANN}  ✓')
print(f'NYU    : {NYU_VAL}  ✓' if os.path.isdir(NYU_VAL) else f'NYU    : {NYU_VAL}  ✗ NOT FOUND')

# %% [markdown]
# ## Imports

# %%
import json
import textwrap
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
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
# ── Common ────────────────────────────────────────────────────────────────────

def load_rgb_tensor(image_path):
    """RGB image → (1,3,224,224) ImageNet-normalised tensor."""
    rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
    img_pil = rgb_transform.load(str(image_path))
    img_pil = rgb_transform.preprocess(img_pil)
    s       = min(img_pil.size)
    img_pil = center_crop(img_pil, (s, s)).resize((224, 224), Image.BILINEAR)
    return rgb_transform.postprocess(img_pil).unsqueeze(0).to(device=DEVICE, dtype=DTYPE)


def load_rgb_display(image_path):
    """RGB → (224,224,3) uint8 numpy for display."""
    img = Image.open(image_path).convert('RGB')
    s   = min(img.size)
    img = center_crop(img, (s, s)).resize((224, 224), Image.BILINEAR)
    return np.array(img)


# ── Caption helpers ───────────────────────────────────────────────────────────

def generate_caption(sampler, tok_text, rgb_tensor, seed):
    ntoks    = CAPTION_GEN_PARAMS['tokens_per_target']
    schedule = build_chained_generation_schedules(
        cond_domains            = ['rgb@224'],
        target_domains          = ['caption'],
        tokens_per_target       = [ntoks],
        autoregression_schemes  = [CAPTION_GEN_PARAMS['autoregression_scheme']],
        decoding_steps          = [CAPTION_GEN_PARAMS['decoding_steps']],
        token_decoding_schedules= [CAPTION_GEN_PARAMS['token_decoding_schedule']],
        temps                   = [CAPTION_GEN_PARAMS['temp']],
        temp_schedules          = [CAPTION_GEN_PARAMS['temp_schedule']],
        cfg_scales              = [CAPTION_GEN_PARAMS['cfg_scale']],
        cfg_schedules           = [CAPTION_GEN_PARAMS['cfg_schedule']],
        cfg_grow_conditioning   = True,
    )
    eos_id = tok_text.token_to_id('[EOS]')
    sd = {'rgb@224': {'tensor': rgb_tensor}}
    sd = init_full_input_modality(sd, MODALITY_INFO, 'rgb@224', DEVICE, eos_id=eos_id)
    sd = init_empty_target_modality(sd, MODALITY_INFO, 'caption', 1, ntoks, DEVICE)
    return sampler.generate(sd, schedule, text_tokenizer=tok_text, verbose=False, seed=seed)


def decode_caption_string(out_dict, tok_text):
    tokens = out_dict['caption']['tensor']
    if tokens.ndim > 1:
        tokens = tokens[0]
    token_ids = tokens.cpu().tolist()
    eos_id = tok_text.token_to_id('[EOS]')
    if eos_id in token_ids:
        token_ids = token_ids[:token_ids.index(eos_id)]
    token_ids = [t for t in token_ids if t > 0]
    if not token_ids:
        return ''
    return tok_text.decode(token_ids, skip_special_tokens=True).strip()


# ── Depth helpers ─────────────────────────────────────────────────────────────

def generate_depth(sampler, tok_text, rgb_tensor, seed):
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
    raw = decode_tok_depth(
        out_dict, {'tok_depth': tok_depth},
        key='tok_depth@224', image_size=224, patch_size=16,
        t=DIVAE_STEPS, verbose=False, cmap=None,
    )
    return (raw[0] if raw.ndim == 3 else raw).astype(np.float32)


def load_gt_depth(depth_path):
    depth = np.load(depth_path).squeeze().astype(np.float32)
    h, w  = depth.shape
    s     = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    depth = depth[top:top+s, left:left+s]
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
    valid = (depth > 0) & (depth <= MAX_DEPTH)
    return depth, valid


def scale_shift_align(pred, gt, valid):
    p = pred[valid].reshape(-1, 1).astype(np.float64)
    g = gt[valid].reshape(-1).astype(np.float64)
    A = np.hstack([p, np.ones_like(p)])
    scale, shift = np.linalg.lstsq(A, g, rcond=None)[0]
    return np.clip(scale * pred + shift, 1e-3, None).astype(np.float32)


def colorise_depth(depth_m, vmin=0.0, vmax=MAX_DEPTH):
    d      = np.clip(depth_m, vmin, vmax)
    d_norm = (d - vmin) / (vmax - vmin + 1e-8)
    return (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)


def wrap(text, width=34):
    if not text:
        return '(empty)'
    return '\n'.join(textwrap.wrap(str(text), width=width))


print('Helper functions defined.')

# %% [markdown]
# ## Load samples and tokenizers

# %%
tok_text = Tokenizer.from_file(TOK_TEXT_PATH)
print(f'Text tokenizer: vocab={tok_text.get_vocab_size()}')

# ── COCO val samples ──────────────────────────────────────────────────────────
with open(COCO_ANN) as f:
    coco_data = json.load(f)

id2path = {img['id']: os.path.join(COCO_ROOT, img['file_name'])
           for img in coco_data['images']}
id2captions = {}
for ann in coco_data['annotations']:
    id2captions.setdefault(ann['image_id'], []).append(ann['caption'])

coco_all = [
    {'image_id': iid, 'path': path}
    for iid, path in sorted(id2path.items())
    if os.path.exists(path)
]
print(f'COCO val2017 on disk : {len(coco_all)}  (expected 5000)')

rng = np.random.default_rng(SEED)
cap_idx = rng.choice(len(coco_all), size=N_VIZ_CAP, replace=False)
cap_idx.sort()
cap_samples = [coco_all[i] for i in cap_idx]

# ── NYU val samples ───────────────────────────────────────────────────────────
rgb_dir   = Path(NYU_VAL) / 'rgb'
depth_dir = Path(NYU_VAL) / 'depth'
nyu_all   = []
for rgb_p in sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg')):
    depth_p = depth_dir / f'{rgb_p.stem}.npy'
    if depth_p.exists():
        nyu_all.append({'rgb': str(rgb_p), 'depth': str(depth_p), 'name': rgb_p.stem})

print(f'NYUv2 labeled val    : {len(nyu_all)}  (expected 654)')
assert len(nyu_all) > 0, 'No NYU samples found — check NYU_VAL path'

depth_idx = rng.choice(len(nyu_all), size=N_VIZ_DEPTH, replace=False)
depth_idx.sort()
depth_samples = [nyu_all[i] for i in depth_idx]

print(f'Caption viz samples  : {N_VIZ_CAP}')
print(f'Depth viz samples    : {N_VIZ_DEPTH}')

# %% [markdown]
# ## Load DiVAE (depth tokenizer)

# %%
print('Loading depth DiVAE ...')
tok_depth = DiVAE.from_pretrained(TOK_DEPTH_ID, cache_dir=HF_CACHE).to(DEVICE).eval()
print('DiVAE ready.')

# %% [markdown]
# ## Phase 1 — Baseline inference (caption + depth in one model load)

# %%
print('=' * 60)
print('Loading baseline FM (no LoRA) ...')
fm = FM.from_pretrained(FM_MODEL, cache_dir=HF_CACHE)
fm = fm.to(device=DEVICE, dtype=DTYPE).eval()
sampler = GenerationSampler(fm)
print(f'FM ready ({sum(p.numel() for p in fm.parameters())/1e6:.0f}M params)')

# ── 1A: Caption on COCO ───────────────────────────────────────────────────────
print(f'\n[1A] Baseline captions on {N_VIZ_CAP} COCO samples ...')
captions_baseline = {}
for s in tqdm(cap_samples, desc='  Baseline captions'):
    try:
        rgb_t = load_rgb_tensor(s['path'])
        with torch.no_grad():
            out = generate_caption(sampler, tok_text, rgb_t, seed=SEED)
        captions_baseline[s['image_id']] = decode_caption_string(out, tok_text)
    except Exception as e:
        captions_baseline[s['image_id']] = ''
        print(f'  skip {s["image_id"]}: {e}')

# ── 1B: Depth on NYU ─────────────────────────────────────────────────────────
print(f'\n[1B] Baseline depth on {N_VIZ_DEPTH} NYU samples ...')
preds_depth_baseline = {}
for s in tqdm(depth_samples, desc='  Baseline depth'):
    try:
        rgb_t = load_rgb_tensor(s['rgb'])
        with torch.no_grad():
            out = generate_depth(sampler, tok_text, rgb_t, seed=SEED)
            raw = decode_depth_raw(out)
        preds_depth_baseline[s['name']] = raw
    except Exception as e:
        preds_depth_baseline[s['name']] = None
        print(f'  skip {s["name"]}: {e}')

del fm, sampler
torch.cuda.empty_cache()
print(f'\nBaseline done: {len(captions_baseline)} captions, {len(preds_depth_baseline)} depth maps.')

# %% [markdown]
# ## Phase 2 — Caption LoRA inference (caption + depth in one model load)

# %%
print('=' * 60)
print(f'Loading caption LoRA FM (rank={LORA_RANK}, modules={LORA_MODULES}) ...')
fm = FM.from_pretrained(FM_MODEL, cache_dir=HF_CACHE)
target_mods = get_LoRA_module_names(LORA_MODULES)
inject_trainable_LoRA(fm, rank=LORA_RANK, scale=1.0, target_replace_modules=target_mods)
lora_sd = torch.load(LORA_WEIGHTS, map_location='cpu')
missing, unexpected = fm.load_state_dict(lora_sd, strict=False)
print(f'  Loaded {len(lora_sd)} LoRA tensors '
      f'(missing={len(missing)}, unexpected={len(unexpected)})')
fm = fm.to(device=DEVICE, dtype=DTYPE).eval()
sampler = GenerationSampler(fm)

# ── 2A: Caption on COCO ───────────────────────────────────────────────────────
print(f'\n[2A] Caption LoRA captions on {N_VIZ_CAP} COCO samples ...')
captions_lora = {}
for s in tqdm(cap_samples, desc='  LoRA captions'):
    try:
        rgb_t = load_rgb_tensor(s['path'])
        with torch.no_grad():
            out = generate_caption(sampler, tok_text, rgb_t, seed=SEED)
        captions_lora[s['image_id']] = decode_caption_string(out, tok_text)
    except Exception as e:
        captions_lora[s['image_id']] = ''
        print(f'  skip {s["image_id"]}: {e}')

# ── 2B: Depth on NYU (forgetting check) ──────────────────────────────────────
print(f'\n[2B] Caption LoRA depth on {N_VIZ_DEPTH} NYU samples (forgetting check) ...')
preds_depth_lora = {}
for s in tqdm(depth_samples, desc='  LoRA depth (forgetting)'):
    try:
        rgb_t = load_rgb_tensor(s['rgb'])
        with torch.no_grad():
            out = generate_depth(sampler, tok_text, rgb_t, seed=SEED)
            raw = decode_depth_raw(out)
        preds_depth_lora[s['name']] = raw
    except Exception as e:
        preds_depth_lora[s['name']] = None
        print(f'  skip {s["name"]}: {e}')

del fm, sampler
torch.cuda.empty_cache()
print(f'\nCaption LoRA done: {len(captions_lora)} captions, {len(preds_depth_lora)} depth maps.')

# %% [markdown]
# ## Grid A — Caption quality (COCO val2017)
# RGB image | GT caption | Baseline caption | Caption LoRA caption

# %%
n_rows = N_VIZ_CAP
fig_cap, axes = plt.subplots(n_rows, 4, figsize=(22, n_rows * 3.0))

col_titles = [
    'RGB\n(COCO val2017)',
    'GT Caption\n(captions_val2017.json)',
    'Baseline\n(4M-21 XL, no LoRA)',
    'Caption LoRA r4/selfattn\n(run 20, COCO 10k train)',
]
col_colors = ['#000000', '#2c3e50', '#1a6bb5', '#c0392b']

for col, (title, color) in enumerate(zip(col_titles, col_colors)):
    axes[0, col].set_title(title, fontsize=9, fontweight='bold', color=color, pad=7)

for row, s in enumerate(cap_samples):
    iid     = s['image_id']
    rgb_np  = load_rgb_display(s['path'])
    gt_show = id2captions.get(iid, ['N/A'])[0]
    cap_b   = captions_baseline.get(iid, '')
    cap_l   = captions_lora.get(iid, '')

    axes[row, 0].imshow(rgb_np)
    axes[row, 0].set_ylabel(f'id {iid}', fontsize=7, rotation=0, labelpad=38, va='center')
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

    for col, (text, color) in enumerate(
        [(gt_show, '#2c3e50'), (cap_b, '#1a6bb5'), (cap_l, '#c0392b')], start=1
    ):
        ax = axes[row, col]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.05, 0.95, wrap(text), va='top', ha='left', fontsize=8, color=color,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#f8f9fa',
                          edgecolor=color, linewidth=1.3))
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)

legend_patches = [
    mpatches.Patch(color='#2c3e50', label='GT caption (COCO)'),
    mpatches.Patch(color='#1a6bb5', label='Baseline (4M-21 XL, no LoRA)'),
    mpatches.Patch(color='#c0392b', label='Caption LoRA r4/selfattn (run 20)'),
]
fig_cap.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
fig_cap.suptitle(
    'COCO val2017 — Caption Quality: 4M-21 XL Baseline vs Caption LoRA r4/selfattn (run 20)',
    fontsize=11, fontweight='bold', y=1.005,
)
plt.tight_layout()

cap_path = os.path.join(FIGURES_DIR, 'caption_coco_r4.png')
fig_cap.savefig(cap_path, dpi=150, bbox_inches='tight')
print(f'Caption grid saved → {cap_path}')

try:
    from IPython.display import display, Image as IPImage
    display(IPImage(cap_path))
except Exception:
    pass

# %% [markdown]
# ## Grid B — Depth forgetting (NYUv2 labeled val)
# RGB | GT depth | Baseline depth | Caption LoRA depth

# %%
n_rows = N_VIZ_DEPTH

fig_dep = plt.figure(figsize=(18, n_rows * 3.8))
gs      = fig_dep.add_gridspec(n_rows, 5,
                                width_ratios=[4, 4, 4, 4, 0.25],
                                hspace=0.05, wspace=0.05)

axes_dep = [[fig_dep.add_subplot(gs[r, c]) for c in range(4)] for r in range(n_rows)]
cbar_ax  = fig_dep.add_subplot(gs[:, 4])

col_titles_dep = [
    'RGB\n(NYUv2 labeled val)',
    'GT Depth\n(metres)',
    'Baseline\n(4M-21 XL, no LoRA)',
    'Caption LoRA r4/selfattn\n(run 20 — depth forgetting?)',
]
for col, title in enumerate(col_titles_dep):
    axes_dep[0][col].set_title(title, fontsize=9, fontweight='bold', pad=6)

for row, s in enumerate(depth_samples):
    gt_depth, valid = load_gt_depth(s['depth'])

    pred_b_raw = preds_depth_baseline.get(s['name'])
    pred_l_raw = preds_depth_lora.get(s['name'])

    pred_b = scale_shift_align(pred_b_raw, gt_depth, valid) if pred_b_raw is not None else np.zeros((224, 224))
    pred_l = scale_shift_align(pred_l_raw, gt_depth, valid) if pred_l_raw is not None else np.zeros((224, 224))

    axes_dep[row][0].imshow(load_rgb_display(s['rgb']))
    axes_dep[row][1].imshow(colorise_depth(gt_depth))
    axes_dep[row][2].imshow(colorise_depth(pred_b))
    axes_dep[row][3].imshow(colorise_depth(pred_l))

    axes_dep[row][0].set_ylabel(s['name'], fontsize=7, rotation=0, labelpad=28, va='center')
    for ax in axes_dep[row]:
        ax.set_xticks([]); ax.set_yticks([])

sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(0, MAX_DEPTH))
sm.set_array([])
cbar = fig_dep.colorbar(sm, cax=cbar_ax)
cbar.set_label('Depth (metres)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig_dep.suptitle(
    'NYUv2 val — Depth Forgetting: 4M-21 XL Baseline vs Caption LoRA r4/selfattn (run 20)',
    fontsize=11, fontweight='bold', y=1.005,
)

dep_path = os.path.join(FIGURES_DIR, 'depth_forgetting_caption_lora_r4.png')
fig_dep.savefig(dep_path, dpi=150, bbox_inches='tight')
print(f'Depth grid saved → {dep_path}')

try:
    from IPython.display import display, Image as IPImage
    display(IPImage(dep_path))
except Exception:
    pass
