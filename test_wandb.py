"""
test_wandb.py — Minimal WandB connectivity test (no GPU, no model)

Logs fake training curves for 10 epochs to validate the offline
WANDB_MODE=offline → wandb sync workflow on Snellius.

Usage:
    python test_wandb.py
    # after job finishes, from login node:
    wandb sync wandb/run-*/
"""

import wandb
import math
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project', default='4m-peft')
parser.add_argument('--entity', default=None)   # your wandb username/team
parser.add_argument('--n_epochs', type=int, default=10)
args = parser.parse_args()

run = wandb.init(
    project = args.project,
    entity  = args.entity,
    name    = 'wandb-connectivity-test',
    config  = dict(
        model        = '4M-21_XL',
        lora_rank    = 8,
        lora_modules = 'attention',
        lr           = 2e-4,
        batch_size   = 16,
        n_epochs     = args.n_epochs,
        test_run     = True,
    ),
    tags = ['test', 'connectivity'],
)

print(f'WandB run: {run.url}')
print(f'Mode: {wandb.run.settings.mode}')
print(f'Run dir: {wandb.run.dir}')
print(f'Logging {args.n_epochs} fake epochs ...')

for epoch in range(args.n_epochs):
    # simulate realistic-looking curves
    train_loss = 3.6 * math.exp(-0.15 * epoch) + 0.1
    val_loss   = 3.6 * math.exp(-0.05 * epoch) + 0.5 + 0.3 * epoch / args.n_epochs
    lr         = 2e-4 * 0.5 * (1 + math.cos(math.pi * epoch / args.n_epochs))

    wandb.log({
        'epoch':            epoch,
        'train/loss':       train_loss,
        'val/loss':         val_loss,
        'train/lr':         lr,
        'train/grad_norm':  0.5 + 0.1 * (epoch % 3),
    })

    print(f'  epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.2e}')
    time.sleep(0.2)

wandb.finish()
print('\nDone. If WANDB_MODE=offline, run from login node:')
print('  wandb sync wandb/run-*/')
