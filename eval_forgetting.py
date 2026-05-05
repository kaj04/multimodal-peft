"""
eval_forgetting.py — Catastrophic Forgetting Evaluation
Task: Surface Normals on DIODE val
Models: 4M-21 XL baseline vs LoRA (depth fine-tuning, rank=8)
Metrics: MAE (degrees), delta < 11.25, 22.5, 30 degrees

Usage:
    python eval_forgetting.py \
        --lora_weights output/depth/nyu_rgb_r8/lora_weights-epoch49.pth \
        --output_dir   output/eval_forgetting/ep49
"""

import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# ── repo path ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'ml-4m'))

from fourm.models.fm import FM
from fourm.vq.vqvae import DiVAE
from fourm.models.generate import (
    GenerationSampler, build_chained_generation_schedules,
    init_full_input_modality, init_empty_target_modality,
)
from fourm.data.modality_info import MODALITY_INFO
from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names
from tokenizers import Tokenizer

# ── constants ─────────────────────────────────────────────────────────────────
HF_CACHE    = '/scratch-shared/tesr125338/hf_cache'
FM_MODEL    = 'EPFL-VILAB/4M-21_XL'
TOK_NORMAL  = 'EPFL-VILAB/4M_tokenizers_normal_8k_224-448'
TOK_TEXT    = os.path.join(REPO_ROOT, 'ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json')
DIODE_ROOT  = '/scratch-shared/tesr125338/diode/val'
LORA_RANK   = 8
LORA_MODS   = 'attention'
DIVAE_STEPS = 25

GEN_PARAMS = dict(
    tokens_per_target       = 196,
    autoregression_scheme   = 'roar',
    decoding_steps          = 25,
    token_decoding_schedule = 'linear',
    temp                    = 1.0,
    temp_schedule           = 'constant',
    cfg_scale               = 2.0,
    cfg_schedule            = 'constant',
)

RGB_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── data ──────────────────────────────────────────────────────────────────────
def load_diode_samples(root):
    """Collect all DIODE val samples that have both RGB and normal maps."""
    samples = []
    for rgb_p in sorted(Path(root).rglob('*.png')):
        normal_p = rgb_p.parent / (rgb_p.stem + '_normal.npy')
        mask_p   = rgb_p.parent / (rgb_p.stem + '_depth_mask.npy')
        if normal_p.exists() and mask_p.exists():
            split = 'indoors' if 'indoors' in str(rgb_p) else 'outdoors'
            samples.append(dict(rgb=rgb_p, normal=normal_p, mask=mask_p, split=split))
    return samples


def load_rgb_tensor(path, device):
    img = Image.open(path).convert('RGB')
    return RGB_TRANSFORM(img).unsqueeze(0).to(device)


def load_gt_normal(path, mask_path):
    """Load and resize DIODE normal map to 224×224. Returns (224,224,3) float32 and valid mask."""
    gt = np.load(path).astype(np.float32)     # (H, W, 3), unit vectors
    mask = np.load(mask_path).astype(bool)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)

    # resize to 224×224 using nearest (preserve vectors)
    gt_pil  = Image.fromarray(((gt + 1) / 2 * 255).clip(0, 255).astype(np.uint8))
    gt_224  = np.array(gt_pil.resize((224, 224), Image.NEAREST)).astype(np.float32)
    gt_224  = gt_224 / 127.5 - 1.0            # back to [-1, 1]
    norm    = np.linalg.norm(gt_224, axis=-1, keepdims=True)
    gt_224  = gt_224 / (norm + 1e-8)

    mask_224 = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((224, 224), Image.NEAREST)
    ) > 0
    # invalid normals are (0,0,0)
    valid = mask_224 & (norm.squeeze(-1) > 0.5)
    return gt_224, valid


# ── inference ─────────────────────────────────────────────────────────────────
def run_normal_inference(sampler, rgb_tensor, tok_text_obj, device, seed=42):
    """RGB → normal tokens → decoded normal map (224,224,3)."""
    ntoks    = GEN_PARAMS['tokens_per_target']
    tok_mod  = 'tok_normal@224'
    schedule = build_chained_generation_schedules(
        cond_domains=['rgb@224'], target_domains=[tok_mod],
        tokens_per_target=[ntoks],
        autoregression_schemes=[GEN_PARAMS['autoregression_scheme']],
        decoding_steps=[GEN_PARAMS['decoding_steps']],
        token_decoding_schedules=[GEN_PARAMS['token_decoding_schedule']],
        temps=[GEN_PARAMS['temp']], temp_schedules=[GEN_PARAMS['temp_schedule']],
        cfg_scales=[GEN_PARAMS['cfg_scale']], cfg_schedules=[GEN_PARAMS['cfg_schedule']],
        cfg_grow_conditioning=True,
    )
    eos_id = tok_text_obj.token_to_id('[EOS]')
    sd = {'rgb@224': {'tensor': rgb_tensor}}
    sd = init_full_input_modality(sd, MODALITY_INFO, 'rgb@224', device, eos_id=eos_id)
    sd = init_empty_target_modality(sd, MODALITY_INFO, tok_mod, 1, ntoks, device)
    out = sampler.generate(sd, schedule, text_tokenizer=tok_text_obj, verbose=False, seed=seed)
    return out, tok_mod


def decode_normals(out, tok_mod, tok_normal_obj):
    tokens = out[tok_mod]['tokens'].squeeze(0).reshape(1, 14, 14)
    with torch.no_grad():
        dec = tok_normal_obj.decode_tokens(
            tokens, timesteps=DIVAE_STEPS, image_size=224, verbose=False
        )
    n = dec.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / (norm + 1e-8)


# ── metrics ───────────────────────────────────────────────────────────────────
def angular_errors(pred_n, gt_n, valid):
    """Returns per-pixel angular errors in degrees for valid pixels."""
    dot    = np.clip((pred_n[valid] * gt_n[valid]).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def summarise(errors_all):
    e = np.concatenate(errors_all) if errors_all else np.array([])
    if len(e) == 0:
        return {}
    return dict(
        mae        = float(e.mean()),
        rmse       = float(np.sqrt((e**2).mean())),
        delta_11_25 = float(np.mean(e < 11.25)),
        delta_22_5  = float(np.mean(e < 22.5)),
        delta_30    = float(np.mean(e < 30.0)),
        n_pixels    = int(len(e)),
        n_images    = None,  # filled in later
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── devices ───────────────────────────────────────────────────────────────
    device_base = torch.device('cuda:0')
    device_lora = torch.device('cuda:1')
    print(f'Baseline → {device_base} | LoRA → {device_lora}')

    # ── load models ───────────────────────────────────────────────────────────
    print('Loading baseline FM ...')
    fm_base = FM.from_pretrained(FM_MODEL, cache_dir=HF_CACHE,
                                  torch_dtype=torch.bfloat16).to(device_base).eval()

    print('Loading LoRA FM ...')
    fm_lora = FM.from_pretrained(FM_MODEL, cache_dir=HF_CACHE,
                                  torch_dtype=torch.bfloat16)
    target_mods = get_LoRA_module_names(LORA_MODS)
    inject_trainable_LoRA(fm_lora, rank=LORA_RANK, scale=1.0,
                          target_replace_modules=target_mods)
    lora_sd = torch.load(args.lora_weights, map_location='cpu')
    missing, _ = fm_lora.load_state_dict(lora_sd, strict=False)
    print(f'  Loaded {len(lora_sd)} LoRA tensors | missing={len(missing)}')
    fm_lora = fm_lora.to(device_lora).eval()

    print('Loading normal tokenizer ...')
    tok_norm_base = DiVAE.from_pretrained(TOK_NORMAL, cache_dir=HF_CACHE).to(device_base).eval()
    tok_norm_lora = DiVAE.from_pretrained(TOK_NORMAL, cache_dir=HF_CACHE).to(device_lora).eval()
    tok_text = Tokenizer.from_file(TOK_TEXT)

    sampler_base = GenerationSampler(fm_base)
    sampler_lora = GenerationSampler(fm_lora)

    # ── load data ─────────────────────────────────────────────────────────────
    samples = load_diode_samples(DIODE_ROOT)
    print(f'DIODE samples: {len(samples)} ({sum(1 for s in samples if s["split"]=="indoors")} indoor, '
          f'{sum(1 for s in samples if s["split"]=="outdoors")} outdoor)')

    # ── evaluation loop ───────────────────────────────────────────────────────
    errors_base, errors_lora = {'all': [], 'indoors': [], 'outdoors': []}, \
                               {'all': [], 'indoors': [], 'outdoors': []}
    n_valid = 0

    for i, s in enumerate(tqdm(samples, desc='Evaluating')):
        try:
            gt_n, valid = load_gt_normal(s['normal'], s['mask'])
            if valid.sum() < 100:
                continue

            rgb_t_base = load_rgb_tensor(s['rgb'], device_base)
            rgb_t_lora = load_rgb_tensor(s['rgb'], device_lora)

            with torch.no_grad():
                out_b, tm_b = run_normal_inference(sampler_base, rgb_t_base, tok_text, device_base, seed=42+i)
                out_l, tm_l = run_normal_inference(sampler_lora, rgb_t_lora, tok_text, device_lora, seed=42+i)

            pred_b = decode_normals(out_b, tm_b, tok_norm_base)
            pred_l = decode_normals(out_l, tm_l, tok_norm_lora)

            err_b = angular_errors(pred_b, gt_n, valid)
            err_l = angular_errors(pred_l, gt_n, valid)

            split = s['split']
            errors_base['all'].append(err_b);      errors_lora['all'].append(err_l)
            errors_base[split].append(err_b);      errors_lora[split].append(err_l)
            n_valid += 1

        except Exception as e:
            print(f'  [skip] {s["rgb"].name}: {e}')

    # ── compute metrics ───────────────────────────────────────────────────────
    results = dict(
        lora_weights = args.lora_weights,
        n_images     = n_valid,
        baseline     = {k: summarise(v) for k, v in errors_base.items()},
        lora         = {k: summarise(v) for k, v in errors_lora.items()},
    )
    for split in ['all', 'indoors', 'outdoors']:
        n = len(errors_base[split])
        if results['baseline'][split]:
            results['baseline'][split]['n_images'] = n
            results['lora'][split]['n_images']     = n

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, 'normals_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {out_path}')

    # ── print summary ─────────────────────────────────────────────────────────
    print('\n' + '='*65)
    print('SURFACE NORMALS — DIODE val')
    print(f'LoRA: {args.lora_weights}')
    print('='*65)
    print('{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        'Split', 'Model', 'MAE↓', 'δ<11.25↑', 'δ<22.5↑', 'δ<30↑'))
    print('-'*65)
    for split in ['all', 'indoors', 'outdoors']:
        for tag, res in [('baseline', results['baseline']), ('lora', results['lora'])]:
            m = res[split]
            if m:
                print('{:<12} {:>10} {:>10.3f} {:>10.4f} {:>10.4f} {:>10.4f}'.format(
                    split, tag, m['mae'], m['delta_11_25'], m['delta_22_5'], m['delta_30']))
        print()
    print('='*65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_weights', default='output/depth/nyu_rgb_r8/lora_weights-epoch49.pth')
    parser.add_argument('--output_dir',   default='output/eval_forgetting/ep49')
    args = parser.parse_args()
    main(args)
