#!/usr/bin/env python3
# eval_all.py
#
# Evaluate one LoRA checkpoint on ALL 5 tasks and produce a single row.json
# that can be aggregated across runs with make_table.py.
#
# Each task is evaluated by calling its dedicated script as a subprocess so
# that model loading failures are isolated and the remaining tasks continue.
#
# Output layout:
#   output_dir/
#     caption/   ← metrics.json from eval_caption_coco.py
#     semseg/    ← results.json from eval_semseg_coco.py
#     det/       ← results.json from eval_det_coco.py
#     depth/     ← results.json from eval_depth.py
#     normal/    ← results.json from eval_normal_nyu.py
#     row.json   ← merged metrics (one table row)
#     summary.txt
#
# Usage:
#   python eval_all.py \
#       --lora_weights /path/to/lora_weights-best.pth \
#       --lora_rank 8 --lora_modules attention \
#       --run_name "5task_r8" \
#       --output_dir /scratch-shared/tesr125338/output/eval/run34 \
#       --tasks caption semseg det depth normal
#
# Or via SLURM:
#   sbatch scripts/snellius/35_eval_all.sh

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── Default paths (Snellius) ──────────────────────────────────────────────────
COCO_DIR        = '/scratch-shared/tesr125338/coco'
COCO_TOKENS_VAL = '/scratch-shared/tesr125338/coco_tokens/val'
NYU_RAW         = '/scratch-shared/tesr125338/nyu_raw'
NYU_TOKENS_VAL  = '/scratch-shared/tesr125338/nyu_tokens_labeled/val'
HF_CACHE        = '/scratch-shared/tesr125338/hf_cache'
ALL_TASKS       = ['caption', 'semseg', 'det', 'depth', 'normal']


# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Evaluate one LoRA on all 5 tasks → row.json')
    p.add_argument('--lora_weights', default=None,
                   help='Path to lora_weights-best.pth (None = baseline)')
    p.add_argument('--lora_rank',    default=8, type=int)
    p.add_argument('--lora_modules', default='attention')
    p.add_argument('--run_name',     required=True,
                   help='Short identifier for this run (used in table)')
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--tasks',        nargs='+', default=ALL_TASKS,
                   choices=ALL_TASKS,
                   help='Which tasks to evaluate (default: all 5)')
    p.add_argument('--max_images',   default=0, type=int,
                   help='Max images per task (0 = full val set)')
    p.add_argument('--device',       default='cuda')
    p.add_argument('--dtype',        default='bfloat16')
    # Per-dataset paths (override defaults if needed)
    p.add_argument('--coco_dir',        default=COCO_DIR)
    p.add_argument('--coco_tokens_val', default=COCO_TOKENS_VAL)
    p.add_argument('--nyu_raw',         default=NYU_RAW)
    p.add_argument('--nyu_tokens_val',  default=NYU_TOKENS_VAL)
    p.add_argument('--hf_cache',        default=HF_CACHE)
    return p.parse_args()


# ── Sub-eval runners ──────────────────────────────────────────────────────────

def _run(cmd, task_name):
    """Run a subprocess and return True if successful."""
    print(f'\n{"="*60}')
    print(f'  Running: {task_name}')
    print(f'  Cmd: {" ".join(cmd)}')
    print('=' * 60)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f'  [ERROR] {task_name} eval failed with exit code {e.returncode}')
        return False


def run_caption(args, out_root):
    out = str(out_root / 'caption')
    cmd = [
        sys.executable, 'eval_caption_coco.py',
        '--coco_root',    f'{args.coco_dir}/val2017',
        '--coco_ann',     f'{args.coco_dir}/annotations/captions_val2017.json',
        '--max_images',   str(args.max_images),
        '--output_dir',   out,
        '--device',       args.device,
        '--dtype',        args.dtype,
    ]
    if args.lora_weights:
        cmd += ['--lora_weights', args.lora_weights,
                '--lora_rank',    str(args.lora_rank),
                '--lora_modules', args.lora_modules]
    if args.hf_cache:
        cmd += ['--hf_cache', args.hf_cache]
    return _run(cmd, 'caption'), out


def run_semseg(args, out_root):
    out = str(out_root / 'semseg')
    cmd = [
        sys.executable, 'eval_semseg_coco.py',
        '--coco_dir',     args.coco_dir,
        '--tokens_dir',   args.coco_tokens_val,
        '--max_images',   str(args.max_images),
        '--output_dir',   out,
        '--device',       args.device,
        '--dtype',        args.dtype,
    ]
    if args.lora_weights:
        cmd += ['--lora_weights', args.lora_weights,
                '--lora_rank',    str(args.lora_rank),
                '--lora_modules', args.lora_modules]
    if args.hf_cache:
        cmd += ['--hf_cache', args.hf_cache]
    return _run(cmd, 'semseg'), out


def run_det(args, out_root):
    out = str(out_root / 'det')
    cmd = [
        sys.executable, 'eval_det_coco.py',
        '--coco_dir',     args.coco_dir,
        '--tokens_dir',   args.coco_tokens_val,
        '--run_name',     args.run_name,
        '--max_images',   str(args.max_images),
        '--output_dir',   out,
        '--device',       args.device,
        '--dtype',        args.dtype,
    ]
    if args.lora_weights:
        cmd += ['--lora_weights', args.lora_weights,
                '--lora_rank',    str(args.lora_rank),
                '--lora_modules', args.lora_modules]
    if args.hf_cache:
        cmd += ['--hf_cache', args.hf_cache]
    return _run(cmd, 'det'), out


def run_depth(args, out_root):
    # eval_depth.py writes to output_dir/run_name/results.json
    # → set output_dir=out_root, run_name='depth' so results land at out_root/depth/results.json
    out = str(out_root / 'depth')
    cmd = [
        sys.executable, 'eval_depth.py',
        '--dataset',      'nyu',
        '--dataset_root', f'{args.nyu_raw}/test',
        '--run_name',     'depth',
        '--output_dir',   str(out_root),
        '--device',       args.device,
        '--dtype',        args.dtype,
    ]
    if args.max_images > 0:
        cmd += ['--max_images', str(args.max_images)]
    if args.lora_weights:
        cmd += ['--lora_weights', args.lora_weights,
                '--lora_rank',    str(args.lora_rank),
                '--lora_modules', args.lora_modules]
    # eval_depth.py has no --hf_cache arg; relies on HF_HOME env var (set in SLURM)
    return _run(cmd, 'depth'), out


def run_normal(args, out_root):
    out = str(out_root / 'normal')
    cmd = [
        sys.executable, 'eval_normal_nyu.py',
        '--nyu_raw',      args.nyu_raw,
        '--tokens_dir',   args.nyu_tokens_val,
        '--run_name',     args.run_name,
        '--max_images',   str(args.max_images),
        '--output_dir',   out,
        '--device',       args.device,
        '--dtype',        args.dtype,
    ]
    if args.lora_weights:
        cmd += ['--lora_weights', args.lora_weights,
                '--lora_rank',    str(args.lora_rank),
                '--lora_modules', args.lora_modules]
    if args.hf_cache:
        cmd += ['--hf_cache', args.hf_cache]
    return _run(cmd, 'normal'), out


# ── Result collection ─────────────────────────────────────────────────────────

def collect_caption(out_dir):
    p = Path(out_dir) / 'metrics.json'
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    # eval_caption_coco.py writes metrics nested under 'caption_lora' (LoRA run)
    # or 'baseline' (no LoRA). Prefer 'caption_lora', fall back to 'baseline', then flat.
    m = d.get('caption_lora') or d.get('baseline') or d
    return {
        'CIDEr':  round(m.get('CIDEr',  -1), 4),
        'BLEU4':  round(m.get('Bleu_4', -1), 4),
        'METEOR': round(m.get('METEOR', -1), 4),
        'ROUGE_L':round(m.get('ROUGE_L',-1), 4),
    }


def collect_semseg(out_dir):
    p = Path(out_dir) / 'results.json'
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return {'mIoU': round(d.get('mIoU', -1), 4)}


def collect_det(out_dir):
    p = Path(out_dir) / 'results.json'
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        'AP50': round(d.get('AP50', -1), 4),
        'AP':   round(d.get('AP',   -1), 4),
    }


def collect_depth(out_dir):
    p = Path(out_dir) / 'results.json'
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    # eval_depth.py writes nested: d['summary']['metrics_all']['all'] with lowercase keys
    m = d.get('summary', {}).get('metrics_all', {}).get('all', {})
    return {
        'AbsRel': round(m.get('abs_rel', -1), 4),
        'delta1': round(m.get('delta1',  -1), 4),
        'RMSE':   round(m.get('rmse',    -1), 4),
    }


def collect_normal(out_dir):
    p = Path(out_dir) / 'results.json'
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    # eval_normal_nyu.py now reports token_acc (VQVAE decoder unavailable)
    return {
        'token_acc': round(d.get('token_acc', -1), 6),
    }


TASK_RUNNERS = {
    'caption': (run_caption, collect_caption),
    'semseg':  (run_semseg,  collect_semseg),
    'det':     (run_det,     collect_det),
    'depth':   (run_depth,   collect_depth),
    'normal':  (run_normal,  collect_normal),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = get_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'  eval_all — run: {args.run_name}')
    print(f'  LoRA    : {args.lora_weights}')
    print(f'  Tasks   : {args.tasks}')
    print(f'  Output  : {args.output_dir}')
    print('=' * 60)

    row = {
        'run_name':     args.run_name,
        'lora_weights': args.lora_weights,
        'lora_rank':    args.lora_rank,
        'lora_modules': args.lora_modules,
    }

    status = {}
    for task in args.tasks:
        runner, collector = TASK_RUNNERS[task]
        ok, task_out = runner(args, out_root)
        status[task] = 'ok' if ok else 'FAILED'
        metrics = collector(task_out)
        row[task] = metrics if metrics else {'error': 'eval failed or output missing'}

    # ── Save row.json ─────────────────────────────────────────────────────────
    with open(out_root / 'row.json', 'w') as f:
        json.dump(row, f, indent=2)

    # ── Print summary table row ───────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'  RESULTS — {args.run_name}')
    print('=' * 60)
    for task in ALL_TASKS:
        if task not in row:
            print(f'  {task:8s}: (skipped)')
            continue
        m = row[task]
        if isinstance(m, dict) and 'error' not in m:
            vals = '  '.join(f'{k}={v}' for k, v in m.items())
            print(f'  {task:8s}: {vals}')
        else:
            print(f'  {task:8s}: {m}')

    print(f'\nStatus: {status}')
    print(f'row.json → {out_root / "row.json"}')

    # ── Save summary.txt ──────────────────────────────────────────────────────
    lines = [f'run_name : {args.run_name}', f'weights  : {args.lora_weights}', '']
    for task in ALL_TASKS:
        if task not in row:
            lines.append(f'{task:8s}: skipped')
        else:
            m = row[task]
            vals = '  '.join(f'{k}={v}' for k, v in m.items()) if isinstance(m, dict) else str(m)
            lines.append(f'{task:8s}: {vals}')
    with open(out_root / 'summary.txt', 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
