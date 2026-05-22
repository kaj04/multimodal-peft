# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --------------------------------------------------------------------
# run_finetuning_lora.py
#
# Fine-tuning script for 4M-21 models with LoRA on dense downstream tasks.
# Supports: semantic segmentation (tok_semseg) and depth estimation (tok_depth).
#
# Usage (single GPU, dry-run to validate LoRA injection):
#   python run_finetuning_lora.py --dry_run
#
# Usage (multi-GPU, ADE20K semseg, rank=64):
#   OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_finetuning_lora.py \
#       --pretrained_model EPFL-VILAB/4M-21_XL \
#       --lora_rank 64 --lora_modules attention \
#       --task semseg \
#       --data_path /path/to/nyu_tokens/train \
#       --eval_data_path /path/to/nyu_tokens/val \
#       --output_dir output/lora_semseg_r64
#
# Adapted from:
#   ml-4m/run_training_4m.py  (EPFL / Apple)
#   MultiMAE/run_finetuning_semseg.py  (EPFL-VILAB)
# --------------------------------------------------------------------

import argparse
import datetime
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import wandb
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn

import fourm.utils as utils
from fourm.models.fm import FM
from fourm.models.lora_utils import (
    ATTENTION_MODULES,
    MLP_MODULES,
    TRANSFORMER_MODULES,
    get_LoRA_module_names,
    inject_trainable_LoRA,
    unfreeze_all_LoRA_layers,
)
from fourm.utils import NativeScalerWithGradNormCount as NativeScaler
from lora_dropout import add_lora_dropout


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    config_parser = argparse.ArgumentParser(description='Config file', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str,
                               help='Path to YAML config file')

    parser = argparse.ArgumentParser('4M-21 LoRA fine-tuning', add_help=False)

    # --- LoRA ---
    parser.add_argument('--lora_rank', default=64, type=int,
                        help='LoRA rank (paper uses 64 for ADE20K semseg, 8 for NYUv2 depth)')
    parser.add_argument('--lora_scale', default=1.0, type=float,
                        help='LoRA scale factor (x = W_0 x + scale * B A x)')
    parser.add_argument('--lora_modules', default='attention', type=str,
                        choices=['selfattn', 'crossattn', 'attention', 'mlp', 'all'],
                        help='Which transformer sub-modules to apply LoRA to')
    parser.add_argument('--lora_dropout', default=0.0, type=float,
                        help='Dropout probability between lora_down and lora_up (0.0 = disabled)')

    # --- Model ---
    parser.add_argument('--pretrained_model', default='EPFL-VILAB/4M-21_XL', type=str,
                        help='HuggingFace model ID or local path (e.g. EPFL-VILAB/4M-21_B)')
    parser.add_argument('--dtype', default='bfloat16', type=str,
                        choices=['float32', 'fp32', 'float16', 'fp16', 'bfloat16', 'bf16'],
                        help='Model dtype for training')

    # --- Task (single-task mode) ---
    parser.add_argument('--task', default=None, type=str,
                        choices=['semseg', 'depth', 'caption', 'normal', 'det'],
                        help='Single downstream task. Mutually exclusive with --tasks.')
    parser.add_argument('--input_mod', default='tok_rgb@224', type=str,
                        help='Input modality key (must match MODALITY_INFO)')
    parser.add_argument('--num_input_tokens', default=196, type=int,
                        help='Number of input tokens (196 = 14x14 patches at 224px)')
    parser.add_argument('--num_target_tokens', default=196, type=int,
                        help='Number of target tokens to predict')
    parser.add_argument('--loss_type', default='mod', type=str, choices=['mod', 'token'],
                        help='mod = mean of per-modality loss; token = mean per-token loss')

    # --- Multi-task mode ---
    # Set via YAML as a list of dicts, e.g.:
    #   tasks:
    #     - name: caption
    #       target_mod: caption
    #       num_target_tokens: 64
    #     - name: semseg
    #       target_mod: tok_semseg@224
    #       num_target_tokens: 196
    # When 'tasks' is set, single-task 'task' is ignored.
    parser.add_argument('--tasks', default=None, type=list,
                        help='Multi-task config (list of dicts). See YAML example above.')

    # --- Training ---
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Gradient accumulation steps (effective_bs = batch_size * accum_iter * n_gpus)')
    parser.add_argument('--blr', default=1e-4, type=float,
                        help='Base LR (absolute_lr = blr * total_batch_size / 256)')
    parser.add_argument('--min_blr', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=None, type=float,
                        help='Max gradient norm for clipping (None = no clipping)')
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_betas', default=[0.9, 0.95], nargs='+', type=float)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='Evaluate every N epochs')
    parser.add_argument('--early_stop_patience', default=0, type=int,
                        help='Stop training if val loss does not improve for N consecutive evals. 0 = disabled.')
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    # --- Data ---
    parser.add_argument('--data_path', default='', type=str,
                        help='Root of tokenized training data (simple hierarchical format)')
    parser.add_argument('--eval_data_path', default='', type=str,
                        help='Root of tokenized validation data')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)

    # --- Checkpointing ---
    parser.add_argument('--output_dir', default='output/lora_ft', type=str)
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint path')
    parser.add_argument('--auto_resume', action='store_true', default=True)

    # --- Misc ---
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--eval', action='store_true', help='Eval only (no training)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run one synthetic forward+backward pass to validate setup (no data needed)')
    parser.add_argument('--print_all', action='store_true', default=False)
    parser.add_argument('--show_user_warnings', action='store_true', default=False)

    # --- Logging ---
    parser.add_argument('--log_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', default='4m-lora-ft', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_run_name', default='auto', type=str)

    # --- Distributed ---
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--dist_eval', action='store_true', default=False)

    # Parse config file first, then CLI overrides
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        import yaml
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args.config_path = args_config.config

    # Derive target_mod from task (single-task mode only)
    _TASK_TO_MOD = {
        'semseg':   ('tok_semseg@224', 196),
        'depth':    ('tok_depth@224',  196),
        'caption':  ('caption',         64),
        'normal':   ('tok_normal@224', 196),
        'det':      ('det',            256),
    }
    if args.tasks:
        # Multi-task: target_mod / num_target_tokens are per-task (in args.tasks list)
        args.target_mod = None
    elif args.task:
        if args.task not in _TASK_TO_MOD:
            raise ValueError(f'Unknown task: {args.task}')
        args.target_mod, args.num_target_tokens = _TASK_TO_MOD[args.task]
    else:
        raise ValueError('Must specify either --task (single) or --tasks (multi-task via YAML).')

    return args


# ---------------------------------------------------------------------------
# Dataset: loads pre-tokenized numpy files in the simple hierarchical format
#
#   data_root/
#     tok_rgb/        ← subfolder name (no @224 suffix on disk)
#       img001.npy    ← shape (196,) int16/int32
#       img002.npy
#     tok_semseg/     or tok_depth/
#       img001.npy
#       img002.npy
# ---------------------------------------------------------------------------

class TokenizedDenseDataset(torch.utils.data.Dataset):
    """Loads paired (input_tokens, target_tokens) from pre-tokenized numpy files.

    Each sample is a pair of .npy files with the same stem under two subfolders.
    File names must be identical across modalities (same stem, same order).

    Args:
        root: Path to the data root containing modality subfolders.
        input_mod_folder: Subfolder name for input tokens (e.g. 'tok_rgb').
        target_mod_folder: Subfolder name for target tokens (e.g. 'tok_semseg').
    """

    def __init__(self, root: str, input_mod_folder: str = 'tok_rgb',
                 target_mod_folder: str = 'tok_semseg'):
        self.root = Path(root)
        self.input_dir = self.root / input_mod_folder
        self.target_dir = self.root / target_mod_folder

        if not self.input_dir.exists():
            raise FileNotFoundError(f'Input token folder not found: {self.input_dir}')
        if not self.target_dir.exists():
            raise FileNotFoundError(f'Target token folder not found: {self.target_dir}')

        # Collect file stems — sorted for reproducibility
        input_stems = sorted(p.stem for p in self.input_dir.rglob('*.npy'))
        target_stems = sorted(p.stem for p in self.target_dir.rglob('*.npy'))

        # Only keep samples that exist in both modalities
        common = sorted(set(input_stems) & set(target_stems))
        if len(common) == 0:
            raise RuntimeError(
                f'No matching sample stems found between {self.input_dir} and {self.target_dir}. '
                'Make sure file names are identical across modality folders.')
        if len(common) < len(input_stems):
            print(f'Warning: {len(input_stems) - len(common)} input files have no matching target.')

        self.samples = common
        print(f'[Dataset] {root}: {len(self.samples)} samples '
              f'({input_mod_folder} → {target_mod_folder})')

    def _find_file(self, folder: Path, stem: str) -> Path:
        """Find .npy file with given stem, possibly inside subdirectories."""
        matches = list(folder.rglob(f'{stem}.npy'))
        if not matches:
            raise FileNotFoundError(f'{stem}.npy not found under {folder}')
        return matches[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem = self.samples[idx]
        input_tokens = np.load(self._find_file(self.input_dir, stem)).astype(np.int64)
        target_tokens = np.load(self._find_file(self.target_dir, stem)).astype(np.int64)
        return torch.from_numpy(input_tokens), torch.from_numpy(target_tokens)


# ---------------------------------------------------------------------------
# Multi-task dataset: interleaves N TokenizedDenseDatasets
# ---------------------------------------------------------------------------

class MultiTaskDataset(torch.utils.data.Dataset):
    """Wraps N TokenizedDenseDatasets and exposes them as a single dataset.

    Each item is (input_tokens, target_tokens, task_idx) where task_idx
    identifies which task/dataset this sample belongs to.

    The dataset is ordered [all task-0 samples, all task-1 samples, ...]
    but the DataLoader's DistributedSampler+shuffle will interleave them.

    Note: all tasks share the same input modality (tok_rgb@224) but have
    different target modalities and token counts. Batches must be
    homogeneous (same task_idx) — enforced via the per-task DataLoaders
    in train_one_epoch_multitask().
    """

    def __init__(self, datasets):
        self.datasets = datasets
        # Build a flat index: (dataset_idx, local_sample_idx)
        self.index = []
        for d_idx, ds in enumerate(datasets):
            for s_idx in range(len(ds)):
                self.index.append((d_idx, s_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        d_idx, s_idx = self.index[idx]
        input_tokens, target_tokens = self.datasets[d_idx][s_idx]
        return input_tokens, target_tokens, torch.tensor(d_idx, dtype=torch.long)


# ---------------------------------------------------------------------------
# Build mod_dict for 4M forward pass
# Adapted from fourm/data/transfer_utils.py::convert_samples_to_mod_dict
# ---------------------------------------------------------------------------

def build_mod_dict(input_tokens: torch.Tensor,
                   target_tokens: torch.Tensor,
                   input_mod: str,
                   target_mod: str,
                   num_input_tokens: int,
                   num_target_tokens: int,
                   mask_padding: bool = True) -> Dict[str, Dict[str, torch.Tensor]]:
    """Build a 4M-compatible mod_dict for single-input → single-target fine-tuning.

    The input modality tokens are all visible to the encoder (input_mask=False).
    The target modality tokens are all hidden from the encoder (input_mask=True)
    and all predicted by the decoder (target_mask=False).

    Args:
        input_tokens:  (B, num_input_tokens)  long, pre-tokenized input (e.g. RGB tokens)
        target_tokens: (B, num_target_tokens) long, pre-tokenized target (e.g. semseg tokens)
        input_mod:  modality key string (e.g. 'tok_rgb@224')
        target_mod: modality key string (e.g. 'tok_semseg@224')
        num_input_tokens:  number of input token positions
        num_target_tokens: number of target token positions
        mask_padding: if True (default), exclude pad tokens (id=0) from the loss.
            For sparse sequence modalities (det, caption) ~80-85% of token positions
            are padding. Without this flag the loss is dominated by padding and the
            model learns to predict zeros long before learning the real task, causing
            spurious fast convergence and early-stopping too early.
            For dense modalities (depth, semseg, normal) every position is a real VQ
            code so this has no effect. Safe to leave True for all tasks.

    Returns:
        mod_dict compatible with FourM.forward()
    """
    B = input_tokens.shape[0]
    device = input_tokens.device

    # target_mask=True  → 4M excludes that position from the cross-entropy loss.
    # Mask out padding (id=0) positions so the loss is computed only on real tokens.
    if mask_padding:
        target_mask = (target_tokens == 0)   # True where pad → not predicted
        # Safety: if ALL tokens in a sample are padding (e.g. image with no detections),
        # target_mask would be all-True → loss=0 with no grad_fn → backward crash.
        # In that case fall back to predicting all tokens (including padding) for that sample.
        all_masked = target_mask.all(dim=-1, keepdim=True)  # (B, 1) bool
        target_mask = target_mask & ~all_masked              # unmask fully-masked samples
    else:
        target_mask = torch.zeros((B, num_target_tokens), dtype=torch.bool, device=device)

    mod_dict = {
        # Input modality: fully visible, not predicted
        input_mod: {
            'tensor': input_tokens,
            'input_mask': torch.zeros((B, num_input_tokens), dtype=torch.bool, device=device),
            'target_mask': torch.ones((B, num_input_tokens), dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.zeros((B, num_input_tokens), dtype=torch.int, device=device),
        },
        # Target modality: fully hidden; only non-padding positions predicted
        target_mod: {
            'tensor': target_tokens,
            'input_mask': torch.ones((B, num_target_tokens), dtype=torch.bool, device=device),
            'target_mask': target_mask.to(device=device),
            'decoder_attention_mask': torch.ones((B, num_target_tokens), dtype=torch.int, device=device),
        },
    }
    # Set the first decoder_attention_mask entry to num_target_tokens so all
    # target tokens can attend to each other (ROAR-style parallel decoding)
    mod_dict[target_mod]['decoder_attention_mask'][:, 0] = num_target_tokens

    return mod_dict


# ---------------------------------------------------------------------------
# Model setup: load FM + inject LoRA + freeze
# ---------------------------------------------------------------------------

def get_model(args) -> nn.Module:
    """Load a pretrained 4M model and inject LoRA adapters."""
    print(f'Loading pretrained model: {args.pretrained_model}')
    model = FM.from_pretrained(args.pretrained_model)

    # Determine which transformer sub-modules to inject LoRA into
    target_modules = get_LoRA_module_names(args.lora_modules)
    print(f'Injecting LoRA (rank={args.lora_rank}, scale={args.lora_scale}, '
          f'dropout={args.lora_dropout}) into modules: {target_modules}')
    inject_trainable_LoRA(
        model,
        rank=args.lora_rank,
        scale=args.lora_scale,
        target_replace_modules=target_modules,
    )
    if args.lora_dropout > 0.0:
        n_patched = add_lora_dropout(model, p=args.lora_dropout)
        print(f'  Added dropout={args.lora_dropout} to {n_patched} LoRAWrapper layers')

    # Freeze everything; then unfreeze LoRA params
    for param in model.parameters():
        param.requires_grad = False
    unfreeze_all_LoRA_layers(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params:     {total_params / 1e6:.1f} M')
    print(f'Trainable (LoRA): {trainable_params / 1e6:.3f} M '
          f'({100 * trainable_params / total_params:.3f}%)')

    return model


# ---------------------------------------------------------------------------
# LoRA-only checkpoint: save/load just the adapter weights (~60 MB for XL)
# ---------------------------------------------------------------------------

def save_lora_weights(model_without_ddp: nn.Module, output_dir: str, tag: str = 'best'):
    """Save only the LoRA adapter weights — much smaller than a full checkpoint.

    The full model state_dict for 4M-21_XL is ~9 GB. The LoRA adapters are
    ~60 MB (14.9M params at bfloat16). Saving them separately lets you:
      - Share the adapters without distributing the full model
      - Load them on top of a fresh FM.from_pretrained() for evaluation
      - Iterate quickly on ablations (one adapter per run, not one 9GB file)

    Args:
        model_without_ddp: the model (unwrapped from DDP)
        output_dir: directory where lora_weights-{tag}.pth is written
        tag: string suffix, e.g. 'best', '5', 'final'
    """
    lora_sd = {k: v.cpu() for k, v in model_without_ddp.state_dict().items()
               if 'lora_' in k}
    if not lora_sd:
        print('  [save_lora_weights] Warning: no lora_ keys found in state_dict — skipping.')
        return None

    path = os.path.join(output_dir, f'lora_weights-{tag}.pth')
    torch.save(lora_sd, path)

    n_tensors = len(lora_sd)
    n_params  = sum(v.numel() for v in lora_sd.values())
    size_mb   = sum(v.nbytes  for v in lora_sd.values()) / 1e6
    print(f'  LoRA weights saved → {path}')
    print(f'    {n_tensors} tensors | {n_params / 1e6:.2f}M params | {size_mb:.1f} MB')
    return path


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler: NativeScaler,
        args,
        lr_schedule_values: Optional[np.ndarray] = None,
        wd_schedule_values: Optional[np.ndarray] = None,
        log_writer=None,
        start_steps: int = 0,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = f'Epoch [{epoch}]'
    print_freq = 50

    optimizer.zero_grad()

    for step, (input_tokens, target_tokens) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header=header)):

        it = start_steps + step  # global step index

        # Update LR and WD per step
        if lr_schedule_values is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule_values[it]
        if wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if param_group.get('weight_decay', 0) > 0:
                    param_group['weight_decay'] = wd_schedule_values[it]

        input_tokens = input_tokens.to(device, non_blocking=True)
        target_tokens = target_tokens.to(device, non_blocking=True)

        mod_dict = build_mod_dict(
            input_tokens, target_tokens,
            input_mod=args.input_mod,
            target_mod=args.target_mod,
            num_input_tokens=args.num_input_tokens,
            num_target_tokens=args.num_target_tokens,
        )

        with torch.cuda.amp.autocast(enabled=(args.dtype in ['float16', 'fp16']),
                                      dtype=torch.bfloat16 if args.dtype in ['bfloat16', 'bf16'] else torch.float16):
            loss, mod_loss = model(
                mod_dict,
                num_encoder_tokens=args.num_input_tokens,
                num_decoder_tokens=args.num_target_tokens,
                loss_type=args.loss_type,
            )

        # loss       = total loss averaged over ALL token positions (encoder + decoder)
        #              → equals target_mod_loss / 2 when num_encoder == num_decoder tokens
        # target_loss = loss on target modality tokens only → comparable with val loss
        target_loss_value = mod_loss[args.target_mod].item()
        loss_value        = loss.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training.')
            sys.exit(1)

        # Guard: 4M returns torch.zeros(1) (no grad_fn) when a batch has no predictable
        # tokens (e.g., all-padding det/caption samples). Backward would crash.
        # Skip this batch's gradient contribution — it carries no learning signal anyway.
        if not loss.requires_grad:
            if (step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()
            continue

        # Gradient accumulation (use total loss for backward — correct signal)
        loss = loss / args.accum_iter
        grad_norm = loss_scaler(
            loss, optimizer,
            clip_grad=args.clip_grad,
            parameters=[p for p in model.parameters() if p.requires_grad],
            update_grad=((step + 1) % args.accum_iter == 0),
        )
        if (step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Logging — use target_loss for 'loss' so train/val are on the same scale
        metric_logger.update(loss=target_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)
        for mod_name, mod_l in mod_loss.items():
            metric_logger.update(**{f'loss_{mod_name}': mod_l.item()})

        if log_writer is not None and (step + 1) % args.accum_iter == 0:
            log_writer.update({
                'train/loss':       target_loss_value,   # target-mod CE, matches val scale
                'train/loss_total': loss_value,          # total loss (encoder+decoder avg)
                'train/lr':         optimizer.param_groups[0]['lr'],
                'train/grad_norm':  grad_norm if grad_norm is not None else 0.0,
                'train/step':       it,
            })

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'[Train] Averaged stats: {metric_logger}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Multi-task training loop (one epoch)
# Alternates batches between tasks: step 0 = task 0, step 1 = task 1, ...
# All tasks update the SAME LoRA weights → shared adapter learns both tasks.
# ---------------------------------------------------------------------------

def train_one_epoch_multitask(
        model: nn.Module,
        task_loaders,          # list of DataLoaders, one per task
        task_cfgs,             # list of dicts: [{target_mod, num_target_tokens, name}, ...]
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler: NativeScaler,
        args,
        lr_schedule_values=None,
        wd_schedule_values=None,
        log_writer=None,
        start_steps: int = 0,
):
    """Multi-task training with proportional random scheduling.

    Each task contributes len(its DataLoader) steps per epoch — proportional to
    dataset size. Steps are globally shuffled (not round-robin) so the model
    sees tasks in random order, eliminating periodic gradient artifacts.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = f'Epoch [{epoch}] (multi-task)'
    print_freq = 100

    optimizer.zero_grad()

    # Build proportional schedule: task_idx repeated len(loader) times per task,
    # then shuffled globally so the order is random each epoch.
    schedule = []
    for task_idx, dl in enumerate(task_loaders):
        schedule.extend([task_idx] * len(dl))
    rng = np.random.default_rng(epoch)
    rng.shuffle(schedule)

    iterators = [iter(dl) for dl in task_loaders]

    for step, task_idx in enumerate(
            metric_logger.log_every(schedule, print_freq, header=header)):

        it = start_steps + step

        if lr_schedule_values is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule_values[min(it, len(lr_schedule_values) - 1)]
        if wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if param_group.get('weight_decay', 0) > 0:
                    param_group['weight_decay'] = wd_schedule_values[min(it, len(wd_schedule_values) - 1)]

        try:
            input_tokens, target_tokens = next(iterators[task_idx])
        except StopIteration:
            # Reset iterator if exhausted (shouldn't happen, but defensive)
            iterators[task_idx] = iter(task_loaders[task_idx])
            input_tokens, target_tokens = next(iterators[task_idx])

        tcfg = task_cfgs[task_idx]
        target_mod        = tcfg['target_mod']
        num_target_tokens = tcfg['num_target_tokens']
        task_name         = tcfg.get('name', f'task{task_idx}')

        input_tokens  = input_tokens.to(device, non_blocking=True)
        target_tokens = target_tokens.to(device, non_blocking=True)

        mod_dict = build_mod_dict(
            input_tokens, target_tokens,
            input_mod=args.input_mod,
            target_mod=target_mod,
            num_input_tokens=args.num_input_tokens,
            num_target_tokens=num_target_tokens,
        )

        with torch.cuda.amp.autocast(enabled=(args.dtype in ['float16', 'fp16']),
                                      dtype=torch.bfloat16 if args.dtype in ['bfloat16', 'bf16'] else torch.float16):
            loss, mod_loss = model(
                mod_dict,
                num_encoder_tokens=args.num_input_tokens,
                num_decoder_tokens=num_target_tokens,
                loss_type=args.loss_type,
            )

        target_loss_value = mod_loss[target_mod].item()
        loss_value        = loss.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training.')
            sys.exit(1)

        # Guard: 4M returns torch.zeros(1) (no grad_fn) when a batch has no predictable
        # tokens (e.g., all-padding det/caption samples). Backward would crash.
        # Skip this batch's gradient contribution — it carries no learning signal anyway.
        if not loss.requires_grad:
            if (step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()
            continue

        loss = loss / args.accum_iter
        grad_norm = loss_scaler(
            loss, optimizer,
            clip_grad=args.clip_grad,
            parameters=[p for p in model.parameters() if p.requires_grad],
            update_grad=((step + 1) % args.accum_iter == 0),
        )
        if (step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{f'loss_{task_name}': target_loss_value})
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None and (step + 1) % args.accum_iter == 0:
            log_writer.update({
                f'train/loss_{task_name}': target_loss_value,
                'train/lr':               optimizer.param_groups[0]['lr'],
                'train/grad_norm':         grad_norm if grad_norm is not None else 0.0,
                'train/step':             it,
            })

    metric_logger.synchronize_between_processes()
    print(f'[Train MT] Averaged stats: {metric_logger}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
        model: nn.Module,
        data_loader: Iterable,
        device: torch.device,
        args,
        epoch: int = 0,
):
    """Evaluate token-level loss and top-1 accuracy on the target modality.

    What we are predicting
    ----------------------
    The model is a token-to-token sequence predictor. Given 196 RGB tokens
    (14×14 patch grid, each an index into a 16,384-entry VQ-VAE codebook),
    it must predict 196 depth tokens (indices into an 8,192-entry codebook).

    At each of the 196 positions the decoder produces a probability distribution
    over 8,192 possible depth codes — exactly like a 196-way classification with
    8,192 classes per position.

    Metrics
    -------
    - loss:     cross-entropy over all 196 × batch token predictions (lower = better)
    - token_acc: fraction of the 196 positions where argmax == ground truth
                 (higher = better; random baseline ≈ 1/8192 ≈ 0.012%)

    Note: pixel-space metrics (δ₁, RMSE in metres) require passing the
    predicted tokens through the VQ-VAE depth decoder. That is implemented
    in a separate evaluation script (eval_depth_pixel.py — TODO Phase 2).
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = f'[Eval] Epoch [{epoch}]'
    print_freq = 20

    for input_tokens, target_tokens in metric_logger.log_every(data_loader, print_freq, header=header):
        input_tokens  = input_tokens.to(device, non_blocking=True)   # (B, 196)
        target_tokens = target_tokens.to(device, non_blocking=True)  # (B, 196)

        mod_dict = build_mod_dict(
            input_tokens, target_tokens,
            input_mod=args.input_mod,
            target_mod=args.target_mod,
            num_input_tokens=args.num_input_tokens,
            num_target_tokens=args.num_target_tokens,
        )

        with torch.cuda.amp.autocast(enabled=(args.dtype in ['float16', 'fp16']),
                                      dtype=torch.bfloat16 if args.dtype in ['bfloat16', 'bf16'] else torch.float16):
            # Single forward pass: return logits to compute both loss and accuracy.
            # mod_logits[target_mod] shape: (B, num_target_tokens, vocab_size)
            #   e.g. (B, 196, 8192) for depth estimation
            mod_logits = model(
                mod_dict,
                num_encoder_tokens=args.num_input_tokens,
                num_decoder_tokens=args.num_target_tokens,
                return_logits=True,
            )

        logits = mod_logits[args.target_mod]              # (B, T, vocab_size)
        B, T, V = logits.shape

        # Cross-entropy loss — match training objective exactly:
        # exclude padding (id=0) positions, same as mask_padding=True in build_mod_dict.
        logits_flat  = logits.reshape(-1, V).float()      # (B*T, vocab_size)
        targets_flat = target_tokens.reshape(-1).long()   # (B*T,)
        valid        = targets_flat != 0                  # True on non-padding positions
        if valid.any():
            loss = F.cross_entropy(logits_flat[valid], targets_flat[valid], reduction='mean')
        else:
            loss = torch.zeros(1, device=logits.device).squeeze()

        # Top-1 token accuracy on non-padding positions only
        preds      = logits.argmax(dim=-1)                # (B, T)
        preds_flat = preds.reshape(-1)
        if valid.any():
            token_acc = (preds_flat[valid] == targets_flat[valid]).float().mean()
        else:
            token_acc = torch.zeros(1, device=logits.device).squeeze()

        metric_logger.update(loss=loss.item())
        metric_logger.update(token_acc=token_acc.item())

    metric_logger.synchronize_between_processes()
    print(f'[Eval]  Averaged stats: {metric_logger}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_multitask(model, task_loaders, task_cfgs, device, epoch=0):
    """Evaluate each task separately; return per-task stats + mean loss."""
    all_stats = {}
    losses = []

    for tcfg, data_loader in zip(task_cfgs, task_loaders):
        task_name         = tcfg.get('name', tcfg['target_mod'])
        target_mod        = tcfg['target_mod']
        num_target_tokens = tcfg['num_target_tokens']

        model.eval()
        metric_logger = utils.MetricLogger(delimiter='  ')

        for input_tokens, target_tokens in data_loader:
            input_tokens  = input_tokens.to(device, non_blocking=True)
            target_tokens = target_tokens.to(device, non_blocking=True)

            mod_dict = build_mod_dict(
                input_tokens, target_tokens,
                input_mod='tok_rgb@224',
                target_mod=target_mod,
                num_input_tokens=196,
                num_target_tokens=num_target_tokens,
            )

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                mod_logits = model(
                    mod_dict,
                    num_encoder_tokens=196,
                    num_decoder_tokens=num_target_tokens,
                    return_logits=True,
                )

            logits = mod_logits[target_mod]
            B, T, V = logits.shape
            logits_flat  = logits.reshape(-1, V).float()
            targets_flat = target_tokens.reshape(-1).long()
            valid        = targets_flat != 0
            if valid.any():
                loss = F.cross_entropy(logits_flat[valid], targets_flat[valid], reduction='mean')
            else:
                loss = torch.zeros(1, device=logits.device).squeeze()
            preds      = logits.argmax(dim=-1).reshape(-1)
            token_acc  = (preds[valid] == targets_flat[valid]).float().mean() if valid.any() \
                         else torch.zeros(1, device=logits.device).squeeze()

            metric_logger.update(loss=loss.item())
            metric_logger.update(token_acc=token_acc.item())

        metric_logger.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        all_stats[task_name] = stats
        losses.append(stats['loss'])
        print(f'[Eval MT] {task_name:12s}  loss={stats["loss"]:.4f}  '
              f'token_acc={stats["token_acc"]:.4f}')

    all_stats['mean_loss'] = float(np.mean(losses))
    print(f'[Eval MT] mean_loss={all_stats["mean_loss"]:.4f}')
    return all_stats


# ---------------------------------------------------------------------------
# Dry-run: validate LoRA injection without real data
# ---------------------------------------------------------------------------

def dry_run(model: nn.Module, args, device: torch.device):
    """Run a single synthetic forward+backward pass to validate the setup."""
    print('\n--- DRY RUN: validating LoRA setup with synthetic data ---')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    B = 2

    input_tokens = torch.randint(0, 16384, (B, args.num_input_tokens), device=device)
    target_tokens = torch.randint(0, 4096, (B, args.num_target_tokens), device=device)

    mod_dict = build_mod_dict(
        input_tokens, target_tokens,
        input_mod=args.input_mod,
        target_mod=args.target_mod,
        num_input_tokens=args.num_input_tokens,
        num_target_tokens=args.num_target_tokens,
    )

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss, mod_loss = model(
            mod_dict,
            num_encoder_tokens=args.num_input_tokens,
            num_decoder_tokens=args.num_target_tokens,
            loss_type=args.loss_type,
        )

    print(f'  Forward pass OK. Loss = {loss.item():.4f}')
    print(f'  Per-modality losses: {", ".join(f"{k}={v.item():.4f}" for k, v in mod_loss.items())}')

    loss.backward()
    print('  Backward pass OK.')

    # Verify only LoRA parameters received gradients
    grad_names = [n for n, p in model.named_parameters() if p.grad is not None]
    non_lora_grads = [n for n in grad_names if 'lora' not in n]
    print(f'  Parameters with gradients:     {len(grad_names)}')
    print(f'  Non-LoRA params with gradients: {len(non_lora_grads)}')
    if non_lora_grads:
        print(f'  WARNING: unexpected gradients in: {non_lora_grads[:5]}')
    else:
        print('  OK — gradients flow only through LoRA parameters.')

    if device.type == 'cuda':
        peak_mem_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved_mib = torch.cuda.max_memory_reserved(device) / 1024**2
        print(f'  Peak GPU memory allocated: {peak_mem_mib:.0f} MiB')
        print(f'  Peak GPU memory reserved:  {reserved_mib:.0f} MiB')

    print('--- DRY RUN COMPLETE ---\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings('ignore', category=UserWarning)

    # dtype
    dtype_map = {
        'float32': torch.float32, 'fp32': torch.float32,
        'float16': torch.float16, 'fp16': torch.float16,
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = get_model(args)
    model.to(device)
    model_without_ddp = model

    # -----------------------------------------------------------------------
    # Dry-run: validate and exit
    # -----------------------------------------------------------------------
    if args.dry_run:
        dry_run(model, args, device)
        return

    # -----------------------------------------------------------------------
    # Data  (single-task OR multi-task branch)
    # -----------------------------------------------------------------------
    is_multitask = bool(getattr(args, 'tasks', None))
    input_folder = args.input_mod.split('@')[0]

    if is_multitask:
        # ── Multi-task: one DataLoader per task ──────────────────────────────
        task_cfgs = args.tasks   # list of dicts: [{name, target_mod, num_target_tokens}]
        print(f'Multi-task mode: {[t["name"] for t in task_cfgs]}')

        def _make_loader(root, target_mod, shuffle, drop_last):
            target_folder = target_mod.split('@')[0]
            ds = TokenizedDenseDataset(root=root,
                                       input_mod_folder=input_folder,
                                       target_mod_folder=target_folder)
            sampler = (
                torch.utils.data.DistributedSampler(
                    ds, num_replicas=num_tasks, rank=global_rank,
                    shuffle=shuffle, drop_last=drop_last)
                if args.distributed
                else (torch.utils.data.RandomSampler(ds) if shuffle
                      else torch.utils.data.SequentialSampler(ds))
            )
            return ds, torch.utils.data.DataLoader(
                ds, sampler=sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=drop_last,
            )

        # Per-task data paths: each task cfg can override args.data_path /
        # args.eval_data_path so tasks from different datasets (e.g. COCO vs
        # NYUv2) can live in separate directories.
        train_datasets, data_loaders_train = zip(*[
            _make_loader(
                t.get('data_path', args.data_path),
                t['target_mod'], shuffle=True, drop_last=True,
            )
            for t in task_cfgs
        ])
        data_loaders_train = list(data_loaders_train)

        val_datasets, data_loaders_val = [], []
        if args.eval_data_path or any('eval_data_path' in t for t in task_cfgs):
            val_datasets, data_loaders_val = zip(*[
                _make_loader(
                    t.get('eval_data_path', args.eval_data_path),
                    t['target_mod'], shuffle=False, drop_last=False,
                )
                for t in task_cfgs
            ])
            data_loaders_val = list(data_loaders_val)

        # Steps/epoch = sum of all loader lengths (proportional to dataset size).
        # With proportional random scheduling each task contributes len(loader) steps.
        num_training_steps_per_epoch = sum(len(dl) for dl in data_loaders_train)

        # Placeholders so the rest of main() compiles in single-task path
        data_loader_train = None
        data_loader_val   = None if not val_datasets else True   # truthy flag only

    else:
        # ── Single-task (original code path) ─────────────────────────────────
        target_folder = args.target_mod.split('@')[0]

        dataset_train = TokenizedDenseDataset(
            root=args.data_path,
            input_mod_folder=input_folder,
            target_mod_folder=target_folder,
        )

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True,
        ) if args.distributed else torch.utils.data.RandomSampler(dataset_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = None
        if args.eval_data_path:
            dataset_val = TokenizedDenseDataset(
                root=args.eval_data_path,
                input_mod_folder=input_folder,
                target_mod_folder=target_folder,
            )
            sampler_val = (
                torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
                if (args.distributed and args.dist_eval)
                else torch.utils.data.SequentialSampler(dataset_val)
            )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )

        num_training_steps_per_epoch = len(dataset_train) // (args.batch_size * num_tasks)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    # Only pass trainable (LoRA) params to the optimizer
    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]

    total_batch_size = args.batch_size * args.accum_iter * num_tasks
    args.lr = args.blr * total_batch_size / 256
    args.min_lr = args.min_blr * total_batch_size / 256

    print(f'Base LR: {args.blr:.2e}  →  Actual LR: {args.lr:.2e}')
    print(f'Total batch size: {total_batch_size}  (bs={args.batch_size} × accum={args.accum_iter} × GPUs={num_tasks})')
    print(f'Steps per epoch: {num_training_steps_per_epoch}')

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.weight_decay,
    )
    loss_scaler = NativeScaler(enabled=(dtype == torch.float16))

    # LR schedule: cosine with warmup
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch,
    )

    # -----------------------------------------------------------------------
    # DDP
    # -----------------------------------------------------------------------
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # Copy the config YAML into the output dir for full reproducibility
        if utils.is_main_process() and getattr(args, 'config_path', ''):
            dest = os.path.join(args.output_dir, 'config.yaml')
            shutil.copy(args.config_path, dest)
            print(f'Config saved → {dest}')

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler,
    )

    # -----------------------------------------------------------------------
    # WandB logger
    # -----------------------------------------------------------------------
    log_writer = utils.WandbLogger(args) if (global_rank == 0 and args.log_wandb) else None
    if log_writer is not None:
        # epoch as x-axis → all charts use epoch not internal step
        wandb.define_metric('epoch')
        wandb.define_metric('loss/*',    step_metric='epoch')
        wandb.define_metric('lr',        step_metric='epoch')
        wandb.define_metric('grad_norm', step_metric='epoch')
        wandb.define_metric('val/*',     step_metric='epoch')

    # -----------------------------------------------------------------------
    # Eval only
    # -----------------------------------------------------------------------
    if args.eval:
        assert data_loader_val is not None, 'Provide --eval_data_path for evaluation.'
        eval_stats = evaluate(model, data_loader_val, device, args, epoch=0)
        print(f'Eval stats: {eval_stats}')
        return

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print(f'\nStarting LoRA fine-tuning for {args.epochs} epochs')
    if args.early_stop_patience > 0:
        print(f'Early stopping: patience={args.early_stop_patience} epochs')
    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.start_epoch if hasattr(args, 'start_epoch') else 0, args.epochs):

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        # ── Training step ────────────────────────────────────────────────────
        if is_multitask:
            # Shuffle each task's sampler independently
            if args.distributed:
                for dl in data_loaders_train:
                    dl.sampler.set_epoch(epoch)
            train_stats = train_one_epoch_multitask(
                model=model,
                task_loaders=data_loaders_train,
                task_cfgs=task_cfgs,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
            )
        else:
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
            )

        # Periodic LoRA-only checkpoint (replaces 18 GB full-model saves)
        if args.output_dir and utils.is_main_process() and (epoch + 1) % args.save_ckpt_freq == 0:
            save_lora_weights(model_without_ddp, args.output_dir, tag=f'epoch{epoch}')

        # ── Evaluation ───────────────────────────────────────────────────────
        eval_stats = {}
        should_eval = epoch % args.eval_freq == 0 or epoch == args.epochs - 1

        if should_eval:
            if is_multitask and data_loaders_val:
                eval_stats = evaluate_multitask(
                    model, data_loaders_val, task_cfgs, device, epoch=epoch)
                current_val_loss = eval_stats['mean_loss']
            elif not is_multitask and data_loader_val is not None:
                eval_stats = evaluate(model, data_loader_val, device, args, epoch=epoch)
                current_val_loss = eval_stats.get('loss', float('inf'))
            else:
                current_val_loss = float('inf')

            if current_val_loss < best_loss:
                best_loss = current_val_loss
                patience_counter = 0
                if args.output_dir and utils.is_main_process():
                    save_lora_weights(model_without_ddp, args.output_dir, tag='best')
                print(f'  New best val loss: {best_loss:.4f}')
            else:
                patience_counter += 1
                print(f'  Val loss did not improve ({patience_counter}/{args.early_stop_patience})')
                if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                    print(f'  Early stopping triggered at epoch {epoch}.')
                    break

        # ── Logging ──────────────────────────────────────────────────────────
        log_stats = {
            **{f'train/{k}': v for k, v in train_stats.items()},
            **{f'val/{k}': v for k, v in eval_stats.items()},
            'epoch': epoch,
            'lora_rank': args.lora_rank,
            'lora_modules': args.lora_modules,
        }
        if log_writer is not None:
            if is_multitask:
                wandb_stats = {'epoch': epoch, 'lr': train_stats.get('lr', 0),
                               'grad_norm': train_stats.get('grad_norm', 0)}
                for tcfg in task_cfgs:
                    n = tcfg['name']
                    wandb_stats[f'loss/train_{n}'] = train_stats.get(f'loss_{n}', 0)
                    if eval_stats:
                        wandb_stats[f'loss/val_{n}'] = eval_stats.get(n, {}).get('loss', 0)
                        wandb_stats[f'val/token_acc_{n}'] = eval_stats.get(n, {}).get('token_acc', 0)
                if eval_stats:
                    wandb_stats['loss/val_mean'] = eval_stats.get('mean_loss', 0)
            else:
                target_key = args.target_mod.replace('@', '_').replace('/', '_')
                wandb_stats = {
                    'epoch':                    epoch,
                    'loss/train':               train_stats.get('loss', 0),
                    f'loss/{target_key}_train': train_stats.get(f'loss_{args.target_mod}', 0),
                    'lr':                       train_stats.get('lr', 0),
                    'grad_norm':                train_stats.get('grad_norm', 0),
                }
                if eval_stats:
                    wandb_stats['loss/val']      = eval_stats.get('loss', 0)
                    wandb_stats['val/token_acc'] = eval_stats.get('token_acc', 0)
            log_writer.update(wandb_stats)

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training completed in {total_time}')
    print(f'Best val loss: {best_loss:.4f}')
    print(f'Checkpoints saved to: {args.output_dir}')

    # Save LoRA-only weights at end of training (compact, shareable artifact)
    # 'epoch' retains the last value of the loop (actual final epoch, even with early stopping)
    if args.output_dir and utils.is_main_process():
        save_lora_weights(model_without_ddp, args.output_dir, tag=f'epoch{epoch}')
        print(f'\nArtifacts in {args.output_dir}:')
        print(f'  lora_weights-best.pth    — LoRA adapter weights at best val loss (~60 MB) ← use this')
        print(f'  lora_weights-epoch{epoch}.pth — LoRA adapter weights at actual final epoch ({epoch})')
        print(f'  log.txt                  — per-epoch train/val loss (JSON lines)')
        print(f'  config.yaml              — exact config used for this run')


if __name__ == '__main__':
    args = get_args()
    main(args)
