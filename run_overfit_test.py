# run_overfit_test.py — LoRA Overfit Diagnostic
#
# Trains LoRA (same config as main run) on N=50 fixed training samples
# for many epochs until CE loss saturates, then evaluates AbsRel on
# those same samples.
#
# Diagnostic outcomes:
#   CE → 0  AND  AbsRel improves  → generalisation problem (fix hyperparms)
#   CE → 0  AND  AbsRel stays bad → CE/metric structural mismatch (key finding)
#   CE stays high                  → LoRA architectural incompatibility / bug
#
# Usage (single GPU):
#   python run_overfit_test.py
#
# Usage (SLURM):
#   sbatch scripts/snellius/05_overfit_test.sh

import os, sys, json, argparse, datetime, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'ml-4m'))

import fourm.utils as utils
from fourm.models.fm import FM
from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names
from fourm.data.modality_info import MODALITY_INFO
from fourm.models.generate import (
    GenerationSampler, build_chained_generation_schedules,
    init_full_input_modality, init_empty_target_modality,
)
from tokenizers import Tokenizer
from fourm.vq.vqvae import DiVAE

# ── Config ────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',    default='/scratch-shared/tesr125338/nyu_tokens/train')
    p.add_argument('--fm_model',     default='EPFL-VILAB/4M-21_XL')
    p.add_argument('--hf_cache',     default='/scratch-shared/tesr125338/hf_cache')
    p.add_argument('--tok_depth',    default='EPFL-VILAB/4M_tokenizers_depth_8k_224-448')
    p.add_argument('--tok_text',     default='ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json')
    p.add_argument('--output_dir',   default='output/overfit_test')

    # LoRA — identical to main run
    p.add_argument('--lora_rank',    type=int,   default=8)
    p.add_argument('--lora_modules', type=str,   default='attention')
    p.add_argument('--lora_scale',   type=float, default=1.0)

    # Overfit settings
    p.add_argument('--n_samples',    type=int,   default=50)
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--lr',           type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--seed',         type=int,   default=42)

    # WandB
    p.add_argument('--wandb_project', default='4M-PEFT')
    p.add_argument('--wandb_entity',  default='francesco-colasurdo04-eindhoven-university-of-technology')

    # AbsRel eval
    p.add_argument('--divae_steps',  type=int,   default=25)
    return p.parse_args()


# ── Dataset — tokenised NYU samples ──────────────────────────────────────────
class NyuTokenDataset(torch.utils.data.Dataset):
    """Loads pre-tokenised NYU samples (one .pth file per sample)."""
    def __init__(self, root, n_samples, seed=42):
        files = sorted(Path(root).glob('**/*.pth'))
        rng   = np.random.default_rng(seed)
        idxs  = rng.choice(len(files), size=min(n_samples, len(files)), replace=False)
        self.files = [files[i] for i in sorted(idxs)]
        print(f'  Overfit dataset: {len(self.files)} samples from {root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location='cpu')


# ── CE loss on one batch ──────────────────────────────────────────────────────
def compute_ce_loss(model, batch, device):
    """Forward pass → CE loss on depth tokens only."""
    # batch is a dict of modality tensors from the tokenised dataset
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss_dict = model(batch)
    # sum only depth token loss; fall back to total loss if key absent
    depth_key = 'tok_depth@224'
    if depth_key in loss_dict:
        return loss_dict[depth_key]
    return sum(loss_dict.values())


# ── AbsRel evaluation (scale-shift aligned) ──────────────────────────────────
def scale_shift_align(pred, gt, valid):
    """Least-squares affine alignment: find s, t s.t. pred*s+t ≈ gt."""
    p, g = pred[valid].float(), gt[valid].float()
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    sol = torch.linalg.lstsq(A, g.unsqueeze(1)).solution
    s, t = sol[0, 0], sol[1, 0]
    return pred * s + t


def absrel(pred, gt, valid):
    pred_aligned = scale_shift_align(pred, gt, valid)
    return ((pred_aligned[valid] - gt[valid]).abs() / gt[valid].abs()).mean().item()


@torch.no_grad()
def eval_absrel_on_samples(model, dataset, tok_depth, tok_text_obj, device, args):
    """Run autoregressive depth prediction on all overfit samples → AbsRel."""
    model.eval()
    sampler  = GenerationSampler(model)
    abs_rels = []

    import torchvision.transforms as T
    from PIL import Image

    rgb_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for idx in range(len(dataset)):
        sample = dataset[idx]

        # ── get RGB tensor ────────────────────────────────────────────────────
        if 'rgb@224' in sample and 'tensor' in sample['rgb@224']:
            rgb_t = sample['rgb@224']['tensor'].unsqueeze(0).to(device, dtype=torch.bfloat16)
        else:
            print(f'  [skip] sample {idx}: no rgb@224 tensor')
            continue

        # ── get GT depth ──────────────────────────────────────────────────────
        if 'depth_zbuffer@224' in sample and 'tensor' in sample['depth_zbuffer@224']:
            gt_depth = sample['depth_zbuffer@224']['tensor'].squeeze()   # (224,224)
        else:
            print(f'  [skip] sample {idx}: no depth GT')
            continue

        valid = gt_depth > 0

        try:
            # ── autoregressive depth prediction ───────────────────────────────
            ntoks    = 196
            tok_mod  = 'tok_depth@224'
            schedule = build_chained_generation_schedules(
                cond_domains=['rgb@224'], target_domains=[tok_mod],
                tokens_per_target=[ntoks],
                autoregression_schemes=['roar'],
                decoding_steps=[25],
                token_decoding_schedules=['linear'],
                temps=[1.0], temp_schedules=['constant'],
                cfg_scales=[2.0], cfg_schedules=['constant'],
                cfg_grow_conditioning=True,
            )
            eos_id = tok_text_obj.token_to_id('[EOS]')
            sd = {'rgb@224': {'tensor': rgb_t}}
            sd = init_full_input_modality(sd, MODALITY_INFO, 'rgb@224', device, eos_id=eos_id)
            sd = init_empty_target_modality(sd, MODALITY_INFO, tok_mod, 1, ntoks, device)
            out = sampler.generate(sd, schedule, text_tokenizer=tok_text_obj,
                                   verbose=False, seed=42)

            tokens = out[tok_mod]['ids'].squeeze(0).reshape(1, 14, 14)
            dec    = tok_depth.decode_tokens(tokens, timesteps=args.divae_steps,
                                             image_size=224, verbose=False)
            pred   = dec.squeeze(0).squeeze(0).cpu()   # (224,224)

            ar = absrel(pred, gt_depth, valid)
            abs_rels.append(ar)

        except Exception as e:
            print(f'  [skip] sample {idx}: {e}')

    return float(np.mean(abs_rels)) if abs_rels else float('nan'), len(abs_rels)


# ── Training loop ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── WandB ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        name    = f'overfit_n{args.n_samples}_r{args.lora_rank}',
        config  = vars(args),
        tags    = ['overfit', 'diagnostic'],
    )
    wandb.define_metric('epoch')
    wandb.define_metric('loss/*',    step_metric='epoch')
    wandb.define_metric('lr',        step_metric='epoch')
    wandb.define_metric('absrel/*',  step_metric='epoch')
    print(f'WandB run: {run.url}')

    # ── Dataset & loader ──────────────────────────────────────────────────────
    print(f'\nLoading {args.n_samples} overfit samples ...')
    dataset = NyuTokenDataset(args.data_path, args.n_samples, seed=args.seed)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.n_samples,   # single batch = all samples
        shuffle=True, num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print('Loading FM ...')
    model = FM.from_pretrained(args.fm_model, cache_dir=args.hf_cache)
    target_mods = get_LoRA_module_names(args.lora_modules)
    inject_trainable_LoRA(model, rank=args.lora_rank, scale=args.lora_scale,
                          target_replace_modules=target_mods)
    model = model.to(torch.bfloat16).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable params: {n_params/1e6:.2f}M  (LoRA only)')

    # ── Tokenizers for AbsRel eval ────────────────────────────────────────────
    print('Loading depth tokenizer for AbsRel eval ...')
    tok_depth = DiVAE.from_pretrained(args.tok_depth, cache_dir=args.hf_cache
                                      ).to(torch.bfloat16).to(device).eval()
    tok_text  = Tokenizer.from_file(args.tok_text)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=args.lr,
                                     weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # ── Training ──────────────────────────────────────────────────────────────
    print(f'\nStarting overfit training: {args.epochs} epochs, {args.n_samples} samples\n')
    baseline_absrel = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in loader:
            optimizer.zero_grad()
            loss = compute_ce_loss(model, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        wandb.log({'epoch': epoch, 'loss/train': avg_loss,
                   'lr': args.lr})

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f'  epoch {epoch:3d}/{args.epochs} | CE loss = {avg_loss:.4f}')

    # ── AbsRel evaluation on the SAME training samples ────────────────────────
    print('\n' + '='*55)
    print('AbsRel evaluation on overfit samples ...')

    # Baseline (fresh model, no training)
    print('  Running baseline (no LoRA weights loaded) ...')
    fm_base = FM.from_pretrained(args.fm_model, cache_dir=args.hf_cache
                                  ).to(torch.bfloat16).to(device).eval()
    base_absrel, n_eval = eval_absrel_on_samples(
        fm_base, dataset, tok_depth, tok_text, device, args)
    del fm_base
    torch.cuda.empty_cache()
    print(f'  Baseline AbsRel: {base_absrel:.4f}  (n={n_eval})')

    # LoRA fine-tuned
    print('  Running LoRA fine-tuned model ...')
    lora_absrel, n_eval = eval_absrel_on_samples(
        model, dataset, tok_depth, tok_text, device, args)
    print(f'  LoRA AbsRel:     {lora_absrel:.4f}  (n={n_eval})')

    delta = lora_absrel - base_absrel

    # ── Log final results ─────────────────────────────────────────────────────
    wandb.log({
        'epoch':                 args.epochs,
        'absrel/baseline':       base_absrel,
        'absrel/lora':           lora_absrel,
        'absrel/delta':          delta,
        'absrel/improved':       int(delta < 0),
    })

    # ── Print diagnostic summary ──────────────────────────────────────────────
    print('\n' + '='*55)
    print('OVERFIT TEST — DIAGNOSTIC SUMMARY')
    print('='*55)
    print(f'  Samples:          {args.n_samples}')
    print(f'  Epochs:           {args.epochs}')
    print(f'  LoRA rank:        {args.lora_rank}')
    print(f'  Final CE loss:    {avg_loss:.4f}')
    print(f'  Baseline AbsRel:  {base_absrel:.4f}')
    print(f'  LoRA AbsRel:      {lora_absrel:.4f}')
    print(f'  Δ AbsRel:         {delta:+.4f}  ({"↓ IMPROVED" if delta < 0 else "↑ DEGRADED"})')
    print('-'*55)
    if avg_loss > 1.0:
        verdict = 'CE did not converge → LoRA architectural incompatibility or bug'
    elif delta < -0.005:
        verdict = 'CE converged + AbsRel improved → generalisation problem (fix hyperparms)'
    else:
        verdict = 'CE converged + AbsRel unchanged → CE/metric structural mismatch'
    print(f'  VERDICT: {verdict}')
    print('='*55)

    # ── Save results ──────────────────────────────────────────────────────────
    results = dict(n_samples=args.n_samples, epochs=args.epochs,
                   lora_rank=args.lora_rank, final_ce_loss=avg_loss,
                   absrel_baseline=base_absrel, absrel_lora=lora_absrel,
                   delta_absrel=delta, verdict=verdict)
    out_path = os.path.join(args.output_dir, 'overfit_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {out_path}')
    wandb.finish()


if __name__ == '__main__':
    main()
