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
2. **PEFT dry-run (1B)** — `tests/`: inject LoRA, run ONE training step in both
   directions (caption + generation), confirm `forward → loss → backward` works.
   *(This is exactly where Anole was impossible — the decisive de-risk.)*
3. **Dataset + experiment design** — decide fine-tuning target vs evaluation
   probes (in-distribution vs out-of-distribution), metrics per capability.
4. **Baseline** — `eval/`: measure the pretrained model on every probe before
   any fine-tuning (honest metrics first — the 4M lesson).
5. **Experiments** — `train/` + `eval/`:
   - Exp 1: cross-modal forgetting transfer matrix (understanding ↔ generation)
   - Exp 2: LoRA rank sweep (task gain vs forgetting vs trainable params)
   - Exp 3: where to place LoRA (shared backbone vs decoupled pathways)

   All on 1B first; scale winning configs to 7B for final numbers.
```
