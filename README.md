# 4M-PEFT

Parameter-Efficient Fine-Tuning of native multimodal models for dense prediction tasks.

This repository investigates LoRA-based fine-tuning of [4M-21](https://github.com/apple/ml-4m) on dense downstream tasks (semantic segmentation, depth estimation), working toward modality-aware and multi-task PEFT on native multimodal architectures.

## Structure

```
4m-peft/
├── ml-4m/                        # git submodule (apple/ml-4m) — core fourm library
├── run_finetuning_lora.py        # Main fine-tuning script (LoRA on 4M-21)
├── cfgs/finetune/
│   ├── semseg/
│   │   ├── nyu_rgb_r8.yaml       # NYUv2 semseg, rank=8  (paper baseline)
│   │   └── ade20k_rgb_r64.yaml   # ADE20K semseg, rank=64 (paper baseline)
│   └── depth/
│       └── nyu_rgb_r8.yaml       # NYUv2 depth, rank=8
└── scripts/snellius/
    ├── 00_setup_env.sh           # One-time env setup on Snellius
    ├── 01_tokenize_nyu.sh        # Pre-tokenize NYUv2 dataset
    ├── 02_dry_run.sh             # Validate LoRA injection (no data needed)
    ├── 03_ft_semseg_nyu.sh       # Fine-tune semseg on NYUv2
    └── 04_ft_depth_nyu.sh        # Fine-tune depth on NYUv2
```

## Quick Start

### 1. Clone (with submodule)
```bash
git clone --recursive https://github.com/<your-user>/4m-peft.git
cd 4m-peft
```

### 2. Install
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ml-4m/
```

### 3. Dry-run (validates LoRA, no data needed)
```bash
python run_finetuning_lora.py --dry_run --pretrained_model EPFL-VILAB/4M-21_B
```

Expected output:
```
Total params:     198.0 M
Trainable (LoRA): 0.xxx M (x.xxx%)
Forward pass OK. Loss = x.xxxx
Backward pass OK.
OK — gradients flow only through LoRA parameters.
```

### 4. Fine-tune (with config file)
```bash
# Single GPU
python run_finetuning_lora.py --config cfgs/finetune/semseg/nyu_rgb_r8.yaml

# Multi-GPU (4 GPUs)
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_finetuning_lora.py \
    --config cfgs/finetune/semseg/nyu_rgb_r8.yaml
```

## Data Preparation

NYUv2 and ADE20K need to be pre-tokenized before training. See `scripts/snellius/01_tokenize_nyu.sh`.

Expected layout after tokenization:
```
nyu_tokens/
├── train/
│   ├── tok_rgb/    (*.npy, shape (196,))
│   └── tok_depth/  (*.npy, shape (196,))
└── val/
    ├── tok_rgb/
    └── tok_depth/
```

## Snellius (HPC)

```bash
# Setup (once)
bash scripts/snellius/00_setup_env.sh

# Validate LoRA injection
sbatch scripts/snellius/02_dry_run.sh

# Fine-tune
sbatch scripts/snellius/03_ft_semseg_nyu.sh
```

## Research Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| L1 | Reproduce 4M-21 LoRA baselines (NYUv2 semseg + depth) | 🔄 In progress |
| L2 | Modality-Aware LoRA (separate A,B per modality type) | 📋 Planned |
| L3 | Multi-Task LoRA (joint semseg + depth fine-tuning) | 📋 Planned |
| L4 | Multi-Modal Input LoRA (RGB+depth → semseg) | 📋 Planned |

## Citation

If you use this work, please also cite the original 4M-21 paper:
```
@article{4m21,
  title={4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities},
  author={Bachmann, Roman and others},
  year={2024}
}
```
