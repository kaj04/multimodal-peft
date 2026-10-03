# Chameleon — Early Fusion PEFT Experiments

This directory contains all code for LoRA fine-tuning experiments on
[Meta's Chameleon-7B](https://huggingface.co/facebook/chameleon-7b),
a truly early-fusion multimodal model where text and image tokens share
a single transformer.

Chameleon experiments are the second pillar of the project alongside 4M-21
(see the root-level code). Together they form a comparative study of PEFT
behavior on early fusion multimodal models.

## Model overview

| Property | Chameleon-7B | 4M-21-XL |
|----------|-------------|---------|
| Architecture | Single transformer (early fusion) | Encoder-decoder (early fusion) |
| Token space | Interleaved text+image tokens | Modality-specific tokens |
| Image tokenizer | VQ-VAE (4096 codes) | Per-modality VQ-VAEs |
| Parameters | 7B | ~600M |
| Tasks | Caption, VQA, image gen | Caption, semseg, det, depth, normal |
| LoRA via | HuggingFace PEFT | Custom inject_trainable_LoRA |

## Setup

### 1. HuggingFace access (gated model)
Chameleon-7B requires accepting a license:
1. Go to https://huggingface.co/facebook/chameleon-7b
2. Accept the Chameleon Community License
3. On Snellius: `huggingface-cli login` (paste your token)

### 2. Install dependencies
```bash
# In your 4m-peft conda environment
pip install -U transformers>=4.41.0 peft accelerate
```

### 3. Check environment
```bash
bash chameleon/tests/00_check_env.sh
```

### 4. Run inference test
```bash
# On a GPU node
salloc --partition=gpu_a100 --gres=gpu:1 --mem=40G --time=01:00:00
python chameleon/tests/01_test_inference.py
```

## Directory structure

```
chameleon/
├── tests/
│   ├── 00_check_env.sh          # environment check (CPU, login node)
│   ├── 01_test_inference.py     # basic inference + LoRA injection test
│   └── 02_test_coco_caption.py  # captioning on COCO val (coming soon)
├── cfgs/                        # training configs (coming soon)
├── scripts/                     # SLURM scripts (coming soon)
└── README.md
```
