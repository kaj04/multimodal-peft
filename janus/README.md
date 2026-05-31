# Janus-Pro PEFT experiments

LoRA / parameter-efficient fine-tuning experiments on **Janus-Pro** (DeepSeek),
a natively unified multimodal model that does **both** image understanding
(image → text) **and** image generation (text → image) from a single
`nn.Module`, fully integrated in HuggingFace `transformers`.

This is the active line of the project. Prior model trials (4M-21, Chameleon,
Anole) are kept as archived folders / lessons-learned.

## Why Janus-Pro (vs the dead ends)

| Wall we hit | Chameleon | Anole | **Janus-Pro** |
|-------------|:---:|:---:|:---:|
| Native HF `transformers` | ✓ | ✗ (custom fork) | ✓ `JanusForConditionalGeneration` |
| Weights accessible for PEFT | ✓ | ✗ (subprocess) | ✓ standard `nn.Module` |
| Image generation enabled | ✗ (Meta disabled) | ✓ | ✓ `generation_mode="image"` |
| Decoupled understanding/gen encoders | ✗ (shared → conflict) | ✗ | ✓ SigLIP + VQ-VAE |

## Folder structure

```
janus/
├── env.sh        # daily startup: source it to load env + HF cache + cd repo
├── tests/        # sanity checks (inference smoke test, PEFT dry-run)
├── data_prep/    # dataset prep / loaders (data itself lives on scratch)
├── train/        # LoRA fine-tuning scripts
└── eval/         # metrics: captioning (CIDEr), generation (CLIP/FID), forgetting matrix
```

Data and model weights are **not** in the repo — they live on Snellius scratch
(`/scratch-shared/tesr125338/`). The HF cache auto-downloads model weights there.

## Environment

Separate conda env `janus` (recent official `transformers` — incompatible with
the `4m-peft` and `anole` envs, which pin different transformers versions).

```
source ~/multimodal-peft/janus/env.sh
```

## Roadmap (go slow, verify each link before the next)

1. ✅ **Smoke test (1B)** — `tests/01_smoke_test.py`: model loads, understanding
   (captions accurate) and generation (real 384x384 images) both work.
2. ✅ **PEFT dry-run (1B)** — `tests/02_peft_dryrun.py`: LoRA injected (4.3M params,
   0.21%); one training step PASSES in all three directions (understanding,
   text-only, generation-custom-loss); gradients reach LoRA params in every case.
   Finding: LoRA lands on the LM (192 tensors) + SigLIP understanding encoder
   (144); generation path uses only the 192 LM tensors. *(The de-risk Anole failed.)*
3. ✅ **Dataset + experiment design** — focus on CROSS-MODAL forgetting (the
   understanding↔generation tension, unique to unified models). Dataset = COCO
   val2017 split (train/eval disjoint); OOD domain forgetting kept as optional
   later study. Metrics: CIDEr/BLEU (caption), CLIPScore (generation).
4. ✅ **Baseline** — `eval/01_baseline.py`: metrics validated on pretrained 1B over
   COCO val. Captioning CIDEr=0.30 / BLEU-4=0.14; generation CLIPScore=30.6.
   (CIDEr deflated by the model's verbosity vs terse COCO refs — fine for relative
   forgetting deltas.) Reusable scorers in `eval/metrics.py`.
5. **Experiments** — `train/` + `eval/`:
   - Exp 1: cross-modal forgetting transfer matrix (understanding ↔ generation)
   - Exp 2: LoRA rank sweep (task gain vs forgetting vs trainable params)
   - Exp 3: where to place LoRA (shared backbone vs decoupled pathways)

   All on 1B first; scale winning configs to 7B for final numbers.
```
