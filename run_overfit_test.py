# run_overfit_test.py — LoRA Overfit Diagnostic
#
# Trains LoRA (same config as main run) on N=50 fixed training samples
# for many epochs until CE loss saturates, then evaluates AbsRel on
# those same samples via teacher-forced argmax prediction.
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

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_MOD      = 'tok_rgb@224'
TARGET_MOD     = 'tok_depth@224'
NUM_TOKENS     = 196   # 14×14 grid

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',    default='/scratch-shared/tesr125338/nyu_tokens/train')
    p.add_argument('--fm_model',     default='EPFL-VILAB/4M-21_XL')
    p.add_argument('--hf_cache',     default='/scratch-shared/tesr125338/hf_cache')
    p.add_argument('--tok_depth',    default='EPFL-VILAB/4M_tokenizers_depth_8k_224-448')
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

    # DiVAE decoding
    p.add_argument('--divae_steps',  type=int,   default=25)
    return p.parse_args()


# ── Dataset — tokenised NYU samples ──────────────────────────────────────────
class NyuTokenDataset(torch.utils.data.Dataset):
    """Loads N random samples from the pre-tokenised NYU directory.

    Structure expected:
        root/tok_rgb/00000.npy   ← RGB token indices   (196,)
        root/tok_depth/00000.npy ← depth token indices (196,)
    """
    def __init__(self, root, n_samples, seed=42):
        root = Path(root)
        rgb_dir   = root / 'tok_rgb'
        depth_dir = root / 'tok_depth'

        stems = sorted(p.stem for p in rgb_dir.glob('*.npy')
                       if (depth_dir / (p.stem + '.npy')).exists())
        if len(stems) == 0:
            raise RuntimeError(f'No matching tok_rgb / tok_depth pairs found in {root}')

        rng  = np.random.default_rng(seed)
        idxs = rng.choice(len(stems), size=min(n_samples, len(stems)), replace=False)
        self.stems     = [stems[i] for i in sorted(idxs)]
        self.rgb_dir   = rgb_dir
        self.depth_dir = depth_dir
        print(f'  Overfit dataset: {len(self.stems)} samples from {root}')

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem  = self.stems[idx]
        rgb   = torch.from_numpy(np.load(self.rgb_dir   / f'{stem}.npy').astype(np.int64))
        depth = torch.from_numpy(np.load(self.depth_dir / f'{stem}.npy').astype(np.int64))
        return rgb, depth   # (196,), (196,)


# ── Build 4M-compatible mod_dict from token tensors ──────────────────────────
def build_mod_dict(input_tokens: torch.Tensor,
                   target_tokens: torch.Tensor) -> dict:
    """Build mod_dict for RGB→depth fine-tuning forward pass.

    Input (RGB) tokens are fully visible to the encoder.
    Target (depth) tokens are hidden from encoder and predicted by decoder.
    """
    B      = input_tokens.shape[0]
    device = input_tokens.device

    dec_attn = torch.ones((B, NUM_TOKENS), dtype=torch.int, device=device)
    dec_attn[:, 0] = NUM_TOKENS   # ROAR: first slot encodes full sequence length

    return {
        INPUT_MOD: {
            'tensor':                   input_tokens,
            'input_mask':               torch.zeros((B, NUM_TOKENS), dtype=torch.bool, device=device),
            'target_mask':              torch.ones( (B, NUM_TOKENS), dtype=torch.bool, device=device),
            'decoder_attention_mask':   torch.zeros((B, NUM_TOKENS), dtype=torch.int,  device=device),
        },
        TARGET_MOD: {
            'tensor':                   target_tokens,
            'input_mask':               torch.ones( (B, NUM_TOKENS), dtype=torch.bool, device=device),
            'target_mask':              torch.zeros((B, NUM_TOKENS), dtype=torch.bool, device=device),
            'decoder_attention_mask':   dec_attn,
        },
    }


# ── CE loss on one batch ──────────────────────────────────────────────────────
def compute_ce_loss(model, rgb_tokens, depth_tokens, device):
    """Forward pass → CE loss on depth tokens only."""
    rgb_tokens   = rgb_tokens.to(device)
    depth_tokens = depth_tokens.to(device)
    mod_dict = build_mod_dict(rgb_tokens, depth_tokens)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss, mod_loss = model(mod_dict,
                               num_encoder_tokens=NUM_TOKENS,
                               num_decoder_tokens=NUM_TOKENS,
                               loss_type='mod')
    return loss


# ── AbsRel evaluation (scale-shift aligned) ──────────────────────────────────
def scale_shift_align(pred, gt, valid):
    """Least-squares affine alignment: find s, t s.t. pred*s+t ≈ gt."""
    p, g = pred[valid].float(), gt[valid].float()
    A    = torch.stack([p, torch.ones_like(p)], dim=1)
    sol  = torch.linalg.lstsq(A, g.unsqueeze(1)).solution
    s, t = sol[0, 0], sol[1, 0]
    return pred * s + t


def absrel(pred, gt, valid):
    pred_aligned = scale_shift_align(pred, gt, valid)
    return ((pred_aligned[valid] - gt[valid]).abs() / gt[valid].abs()).mean().item()


@torch.no_grad()
def eval_absrel_on_samples(model, dataset, tok_depth, device, args):
    """Teacher-forced argmax depth prediction on all overfit samples → AbsRel.

    For each sample:
      1. Build mod_dict from RGB + GT depth tokens.
      2. Call model(return_logits=True) → depth token logits.
      3. Argmax → predicted depth token IDs.
      4. Decode BOTH predicted and GT tokens with DiVAE.
      5. Compute AbsRel with scale-shift alignment.

    Using token-level GT (decoded from GT tokens) as reference means
    AbsRel → 0 iff the model predicts the exact correct token IDs.
    """
    model.eval()
    abs_rels = []

    for idx in range(len(dataset)):
        rgb_tokens, gt_tokens = dataset[idx]

        # Add batch dimension
        rgb_b = rgb_tokens.unsqueeze(0).to(device)   # (1, 196)
        gt_b  = gt_tokens.unsqueeze(0).to(device)    # (1, 196)

        mod_dict = build_mod_dict(rgb_b, gt_b)

        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                mod_logits = model(mod_dict,
                                   num_encoder_tokens=NUM_TOKENS,
                                   num_decoder_tokens=NUM_TOKENS,
                                   return_logits=True)

            # mod_logits[TARGET_MOD]: shape (num_target_tokens, vocab_size)
            # = (196, 8192) for a single sample
            depth_logits = mod_logits[TARGET_MOD].float()        # (196, V)
            pred_ids     = depth_logits.argmax(dim=-1)           # (196,)
            pred_ids_2d  = pred_ids.reshape(1, 14, 14)           # (1, 14, 14)
            gt_ids_2d    = gt_tokens.reshape(1, 14, 14).to(device)

            # Decode both with DiVAE
            pred_depth = tok_depth.decode_tokens(
                pred_ids_2d, timesteps=args.divae_steps, image_size=224, verbose=False
            ).squeeze().cpu()   # (224, 224)

            gt_depth = tok_depth.decode_tokens(
                gt_ids_2d, timesteps=args.divae_steps, image_size=224, verbose=False
            ).squeeze().cpu()   # (224, 224)

            valid = gt_depth > 0
            if valid.sum() < 10:
                print(f'  [skip] sample {idx}: too few valid GT pixels')
                continue

            ar = absrel(pred_depth, gt_depth, valid)
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
    wandb.define_metric('loss/*',   step_metric='epoch')
    wandb.define_metric('lr',       step_metric='epoch')
    wandb.define_metric('absrel/*', step_metric='epoch')
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

    # ── Depth tokenizer for AbsRel eval ───────────────────────────────────────
    print('Loading depth tokenizer for AbsRel eval ...')
    from fourm.vq.vqvae import DiVAE
    tok_depth = DiVAE.from_pretrained(args.tok_depth, cache_dir=args.hf_cache
                                      ).to(torch.bfloat16).to(device).eval()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=args.lr,
                                    weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # ── Training ──────────────────────────────────────────────────────────────
    print(f'\nStarting overfit training: {args.epochs} epochs, {args.n_samples} samples\n')
    avg_loss = float('nan')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for rgb_tokens, depth_tokens in loader:
            optimizer.zero_grad()
            loss = compute_ce_loss(model, rgb_tokens, depth_tokens, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        wandb.log({'epoch': epoch, 'loss/train': avg_loss, 'lr': args.lr})

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f'  epoch {epoch:3d}/{args.epochs} | CE loss = {avg_loss:.4f}')

    # ── AbsRel evaluation on the SAME training samples ────────────────────────
    print('\n' + '='*55)
    print('AbsRel evaluation on overfit samples ...')

    # Baseline: fresh model with no LoRA training
    print('  Running baseline (fresh FM, no LoRA weights) ...')
    fm_base = FM.from_pretrained(args.fm_model, cache_dir=args.hf_cache)
    inject_trainable_LoRA(fm_base, rank=args.lora_rank, scale=args.lora_scale,
                          target_replace_modules=target_mods)
    fm_base = fm_base.to(torch.bfloat16).to(device).eval()

    base_absrel, n_eval = eval_absrel_on_samples(
        fm_base, dataset, tok_depth, device, args)
    del fm_base
    torch.cuda.empty_cache()
    print(f'  Baseline AbsRel: {base_absrel:.4f}  (n={n_eval})')

    # LoRA fine-tuned model
    print('  Running LoRA fine-tuned model ...')
    lora_absrel, n_eval = eval_absrel_on_samples(
        model, dataset, tok_depth, device, args)
    print(f'  LoRA AbsRel:     {lora_absrel:.4f}  (n={n_eval})')

    delta = lora_absrel - base_absrel

    # ── Log final results ─────────────────────────────────────────────────────
    wandb.log({
        'epoch':            args.epochs,
        'absrel/baseline':  base_absrel,
        'absrel/lora':      lora_absrel,
        'absrel/delta':     delta,
        'absrel/improved':  int(delta < 0),
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
