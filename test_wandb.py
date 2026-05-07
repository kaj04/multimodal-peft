"""
test_wandb.py — WandB logging structure test (no GPU, no model)

Validates metric grouping and chart layout for the 4M-PEFT training runs.
Metric naming convention:
  loss/*      → all losses on ONE chart (train + val + depth together)
  lr          → learning rate schedule
  grad_norm   → gradient norm (training stability)
  val/*       → validation-only metrics (token_acc)

Usage:
    python test_wandb.py
"""

import wandb
import math
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50)
args = parser.parse_args()

run = wandb.init(
    project = '4M-PEFT',
    entity  = 'francesco-colasurdo04-eindhoven-university-of-technology',
    name    = 'wandb-layout-test',
    config  = dict(
        model        = '4M-21_XL',
        lora_rank    = 8,
        lora_modules = 'attention',
        lr_peak      = 2e-4,
        batch_size   = 16,
        n_epochs     = args.n_epochs,
        dataset      = 'NYUv2',
        task         = 'depth',
    ),
    tags = ['test', 'lora', 'depth'],
)

# ── Define epoch as x-axis for all metrics ────────────────────────────────────
# This makes every chart use epoch (not step) on the x-axis
wandb.define_metric('epoch')
wandb.define_metric('loss/*',    step_metric='epoch')
wandb.define_metric('lr',        step_metric='epoch')
wandb.define_metric('grad_norm', step_metric='epoch')
wandb.define_metric('val/*',     step_metric='epoch')

print(f'WandB run: {run.url}')
print(f'Logging {args.n_epochs} epochs ...')

for epoch in range(args.n_epochs):
    # simulate realistic 4M-style curves
    train_loss  = 3.6  * math.exp(-0.06 * epoch) + 0.15
    depth_loss  = train_loss * 2                           # depth-specific token loss
    val_loss    = 9.4  + 0.09 * epoch                     # diverging val (as observed)
    lr          = 2e-4 * 0.5 * (1 + math.cos(math.pi * epoch / args.n_epochs))
    grad_norm   = 3.5  * math.exp(-0.04 * epoch) + 0.3 + 0.2 * (epoch % 5 == 0)
    token_acc   = max(0.001, 0.003 - 0.00003 * epoch)     # nearly flat, very low

    wandb.log({
        'epoch':            epoch,

        # ── losses: all on ONE chart automatically ─────────────────────────
        'loss/train':       train_loss,
        'loss/val':         val_loss,
        'loss/depth_train': depth_loss,

        # ── optimizer ─────────────────────────────────────────────────────
        'lr':               lr,
        'grad_norm':        grad_norm,

        # ── validation only ────────────────────────────────────────────────
        'val/token_acc':    token_acc,
    })

    print(f'  ep {epoch:2d} | train={train_loss:.3f} | val={val_loss:.3f} '
          f'| depth={depth_loss:.3f} | lr={lr:.2e} | gnorm={grad_norm:.2f}')
    time.sleep(0.05)

wandb.finish()
print('Done.')
