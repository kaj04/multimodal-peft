# Multimodal PEFT — Early Fusion Model Study

A systematic study of **Parameter-Efficient Fine-Tuning (LoRA)** on early fusion
multimodal models, where all modalities share a unified token representation.

> **Research question**: How does LoRA fine-tuning behave in models where text, images,
> and other modalities are represented in the same discrete token space? Do cross-task
> transfer patterns differ from late-fusion architectures?

## Models

| Model | Architecture | Token space | Code |
|-------|-------------|-------------|------|
| [4M-21-XL](https://huggingface.co/EPFL-VILAB/4M-21_XL) | Encoder-decoder transformer | Per-modality VQ codes | root level |
| [Chameleon-7B](https://huggingface.co/facebook/chameleon-7b) | Single transformer | Interleaved text+image | `chameleon/` |

## Key findings (4M-21-XL)

- **Multi-task LoRA** outperforms single-task on caption and detection
- **Sparse token sequences** (caption ~80% padding, det ~85%) cause premature
  convergence without padding masking — fixed via `mask_padding=True` in `build_mod_dict`
- **Asymmetric cross-task transfer**: caption LoRA improves semseg (+44% vs baseline),
  while semseg LoRA degrades general visual encoding → semantic tasks transfer better
  than VQ-code-prediction tasks

## Repository structure

```
multimodal-peft/
│
├── ml-4m/                      # 4M codebase (submodule, read-only)
├── cfgs/finetune/              # LoRA training configs (4M)
│   ├── caption/
│   ├── semseg/
│   ├── det/
│   ├── depth/
│   ├── normal/
│   └── multitask/
├── scripts/snellius/           # SLURM job scripts (4M)
├── run_finetuning_lora.py      # LoRA training script (4M)
├── eval_all.py                 # Run all 5 task evals → row.json
├── eval_caption_coco.py        # CIDEr / BLEU-4 on COCO val2017
├── eval_semseg_coco.py         # mIoU on COCO panoptic val2017
├── eval_det_coco.py            # AP50 / AP on COCO val2017
├── eval_depth.py               # AbsRel / delta1 on NYUv2
├── eval_normal_nyu.py          # Token accuracy on NYUv2 normals
├── make_table.py               # Aggregate row.json -> markdown + LaTeX table
│
└── chameleon/                  # Chameleon-7B experiments
    ├── tests/                  # Setup verification + inference tests
    ├── cfgs/                   # Training configs (coming)
    ├── scripts/                # SLURM scripts (coming)
    └── README.md
```

## Reproducing 4M-21 experiments

### Environment
```bash
conda activate 4m-peft
export HF_HOME=/scratch-shared/tesr125338/hf_cache
```

### Training (Snellius)
```bash
sbatch scripts/snellius/39_ft_caption_coco_r8_attention_v2.sh   # caption (v2, pad fix)
sbatch scripts/snellius/36_ft_det_coco_r8_attention.sh           # detection (v2)
sbatch scripts/snellius/25_ft_semseg_coco_r8_attention.sh        # semseg
sbatch scripts/snellius/37_ft_depth_nyu_r8_attention.sh          # depth
sbatch scripts/snellius/38_ft_normal_nyu_r8_attention.sh         # normal
sbatch scripts/snellius/40_ft_5task_r8_attention_v2.sh           # multitask (v2)
```

### Evaluation
```bash
sbatch --export=ALL,RUN_NAME=caption_v2,LORA=/path/to/lora_weights-best.pth \
    scripts/snellius/35_eval_all.sh

python make_table.py \
    --results output/eval_all/baseline/row.json \
               output/eval_all/caption_v2/row.json \
    --output results/table.md
```

## Chameleon experiments

See [`chameleon/README.md`](chameleon/README.md) for setup and instructions.

## Acknowledgements

Built on [ml-4m](https://github.com/apple/ml-4m) (EPFL / Apple) and
[Chameleon](https://huggingface.co/facebook/chameleon-7b) (Meta).
