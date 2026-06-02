#!/usr/bin/env python3
"""
janus/train/02_train_generation.py

LoRA fine-tuning of Janus-Pro-1B on the GENERATION direction (text -> image).

This direction is NOT supported by the stock HF forward (which only computes
text loss), so we use a CUSTOM training step — the exact path validated in the
PEFT dry-run (gradients confirmed to reach the LoRA params):

    target image  --vqmodel.encode-->  image token ids        (the labels)
    [text embeds | image embeds[:-1]]  --transformer-->  hidden states
    hidden states  --generation_head-->  logits over the image codebook
    cross_entropy(logits, image tokens)   (teacher forcing)

Setup: batch_size=1 + gradient accumulation, COCO val2017 disjoint split,
W&B logging, adapter checkpoint per epoch. Compared to the captioning trainer
this script ships with optimization choices better suited to the sparse signal
of CE over the 16k-codebook image tokens:
  - lower LR (5e-5 vs 1e-4): the generation loss is sparser than text CE;
  - linear warmup (5%) + cosine decay: avoid LoRA degradation in the first
    few steps and let the schedule decide the late phase;
  - weight_decay = 0 on LoRA params (decay vs an additive update biases toward 0);
  - 3 epochs by default instead of 1 (one epoch over 4k samples is too short
    to see the CLIPScore probe move in our pilot);
  - mid-epoch CLIPScore probe every EVAL_EVERY_STEPS steps with a small set,
    so a degradation is visible while training rather than only at the end.

All knobs are env-overridable. The loss itself is unchanged from the dry-run
(plain CE on image-token IDs); changing the loss is a separate question
(reconstruction term a la Anole, coarse-to-fine, etc.) we tackle only if
the better-tuned config still does not move the target.

Run via 02_run_train_generation.sh (needs a GPU).
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from metrics import CLIPScorer, generate_image  # noqa: E402

# --- CONFIG (env-overridable) ----------------------------------------------
MODEL_ID   = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-1B")
COCO_DIR   = Path("/scratch-shared/tesr125338/coco")
VAL_IMAGES = COCO_DIR / "val2017"
VAL_CAPS   = COCO_DIR / "annotations" / "captions_val2017.json"

TRAIN_N    = int(os.environ.get("TRAIN_N", "4000"))
EVAL_N     = int(os.environ.get("EVAL_N", "200"))
GEN_EVAL_N = int(os.environ.get("GEN_EVAL_N", "30"))   # images are slow to gen
EPOCHS     = int(os.environ.get("EPOCHS", "3"))        # gen needs more than 1
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR         = float(os.environ.get("LR", "5e-5"))       # lower than text LR; gen loss is sparser
LORA_RANK  = int(os.environ.get("LORA_RANK", "4"))     # captioning saturates at low r; try the same regime
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * int(os.environ.get("LORA_RANK", "4")))))
SEED       = int(os.environ.get("SEED", "42"))

# LR schedule + optimisation
WARMUP_RATIO  = float(os.environ.get("WARMUP_RATIO", "0.05"))   # 5% linear warmup
WEIGHT_DECAY  = float(os.environ.get("WEIGHT_DECAY", "0.0"))    # 0 on LoRA is standard
GRAD_CLIP     = float(os.environ.get("GRAD_CLIP", "1.0"))

# Mid-epoch evaluation: every N optimizer steps, do a quick eval with MIN samples
EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "100"))
EVAL_MID_N       = int(os.environ.get("EVAL_MID_N", "10"))      # small for speed

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "janus-peft")
RUN_NAME      = os.environ.get("RUN_NAME", "gen_1b_r8_valsplit")

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/train/output"),
)) / RUN_NAME

SEP = "=" * 64


# ---------------------------------------------------------------------------
# Data (same split convention as the captioning trainer)
# ---------------------------------------------------------------------------

def load_split():
    data = json.loads(VAL_CAPS.read_text())
    id2file = {im["id"]: im["file_name"] for im in data["images"]}
    refs = defaultdict(list)
    for ann in data["annotations"]:
        refs[ann["image_id"]].append(ann["caption"])
    ids = [i for i in refs if (VAL_IMAGES / id2file[i]).exists()]
    random.Random(SEED).shuffle(ids)
    train_ids = ids[:TRAIN_N]
    eval_ids  = ids[TRAIN_N:TRAIN_N + EVAL_N]
    return train_ids, eval_ids, id2file, refs


# ---------------------------------------------------------------------------
# Custom generation training step (validated in the dry-run)
# ---------------------------------------------------------------------------

def generation_loss(base, processor, image, caption):
    """One text->image teacher-forced step; returns the CE loss (with grad)."""
    inner = base.model
    device = base.device
    dtype = next(base.parameters()).dtype

    messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, generation_mode="image")
    text_ids = processor(text=prompt, generation_mode="image",
                         return_tensors="pt").to(device)["input_ids"]

    pixel_values = processor.image_processor(
        images=[image], return_tensors="pt")["pixel_values"].to(device, dtype)
    with torch.no_grad():
        image_tokens = inner.vqmodel.encode(pixel_values).image_tokens.reshape(1, -1)

    text_embeds  = base.get_input_embeddings()(text_ids)
    image_embeds = base.prepare_embeddings_for_image_generation(image_tokens)
    inputs_embeds = torch.cat([text_embeds, image_embeds[:, :-1, :]], dim=1)

    hidden = inner(inputs_embeds=inputs_embeds).last_hidden_state
    Lt = text_embeds.shape[1]
    img_hidden = hidden[:, Lt - 1:, :]
    logits = inner.generation_head(img_hidden)

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        image_tokens.reshape(-1).long(),
    )


@torch.no_grad()
def eval_clipscore(base, processor, scorer, eval_ids, id2file, refs, n):
    base.eval()
    vals = []
    for k, img_id in enumerate(eval_ids[:n]):
        torch.manual_seed(SEED + k)
        img = generate_image(base, processor, refs[img_id][0])
        vals.append(scorer.score(img, refs[img_id][0]))
    base.train()
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import wandb
    from transformers import (JanusForConditionalGeneration, JanusProcessor,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  LoRA GENERATION fine-tune | run={RUN_NAME}")
    print(f"  TRAIN_N={TRAIN_N} EVAL_N={EVAL_N} GEN_EVAL_N={GEN_EVAL_N}")
    print(f"  EPOCHS={EPOCHS} GRAD_ACCUM={GRAD_ACCUM} LR={LR} weight_decay={WEIGHT_DECAY}")
    print(f"  rank={LORA_RANK} alpha={LORA_ALPHA} warmup={WARMUP_RATIO} grad_clip={GRAD_CLIP}")
    print(f"  eval_every_steps={EVAL_EVERY_STEPS} eval_mid_n={EVAL_MID_N}")
    print(SEP + "\n")

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    lora_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    # Inject in-place: keep `base` for the custom forward, `peft_model` to save.
    peft_model = get_peft_model(base, lora_cfg)
    base.train()
    trainable = [p for p in base.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in trainable)/1e6:.3f}M\n")

    train_ids, eval_ids, id2file, refs = load_split()
    print(f"  train images: {len(train_ids)} | eval images: {len(eval_ids)}\n")

    scorer = CLIPScorer()
    wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
        "model": MODEL_ID, "direction": "generation", "train_n": TRAIN_N,
        "epochs": EPOCHS, "grad_accum": GRAD_ACCUM, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
        "grad_clip": GRAD_CLIP, "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA, "seed": SEED,
        "schedule": "linear_warmup_cosine",
    })

    # AdamW + cosine schedule with linear warmup.
    # Total optimizer steps over the whole run determine the cosine length.
    total_steps  = (len(train_ids) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    print(f"  total optimizer steps: {total_steps}  warmup: {warmup_steps}\n", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print("  measuring baseline CLIPScore (pre-training)...", flush=True)
    base_clip = eval_clipscore(base, processor, scorer, eval_ids, id2file, refs, GEN_EVAL_N)
    print(f"  baseline CLIPScore={base_clip:.2f}", flush=True)
    wandb.log({"eval/CLIPScore": base_clip, "epoch": 0})
    history = {"baseline_clipscore": base_clip, "epochs": []}

    global_step = 0
    optimizer.zero_grad()
    rng = random.Random(SEED)
    t_start = time.time()

    for epoch in range(EPOCHS):
        order = train_ids[:]
        rng.shuffle(order)
        running = 0.0
        for i, img_id in enumerate(order):
            image = Image.open(VAL_IMAGES / id2file[img_id]).convert("RGB")
            loss = generation_loss(base, processor, image, refs[img_id][0])
            (loss / GRAD_ACCUM).backward()
            running += loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                avg = running / GRAD_ACCUM
                running = 0.0
                cur_lr = scheduler.get_last_lr()[0]
                wandb.log({"train/loss": avg, "train/lr": cur_lr,
                           "train/grad_norm": float(gnorm), "step": global_step})
                if global_step % 20 == 0:
                    print(f"  epoch {epoch} step {global_step} "
                          f"({i+1}/{len(order)}) loss={avg:.4f} lr={cur_lr:.2e}",
                          flush=True)

                # Mid-epoch eval: cheap CLIPScore probe to see direction live.
                if EVAL_EVERY_STEPS > 0 and global_step % EVAL_EVERY_STEPS == 0:
                    mid_clip = eval_clipscore(base, processor, scorer,
                                              eval_ids, id2file, refs, EVAL_MID_N)
                    print(f"    [mid-eval @ step {global_step}] "
                          f"CLIPScore({EVAL_MID_N})={mid_clip:.2f}", flush=True)
                    wandb.log({"eval/CLIPScore_mid": mid_clip, "step": global_step})

        clip = eval_clipscore(base, processor, scorer, eval_ids, id2file, refs, GEN_EVAL_N)
        print(f"  [epoch {epoch}] CLIPScore={clip:.2f}", flush=True)
        wandb.log({"eval/CLIPScore": clip, "epoch": epoch + 1})
        history["epochs"].append({"epoch": epoch + 1, "CLIPScore": clip})
        ckpt = OUTPUT_DIR / f"adapter_epoch{epoch+1}"
        peft_model.save_pretrained(ckpt)
        print(f"  saved adapter -> {ckpt}", flush=True)

    history["minutes"] = round((time.time() - t_start) / 60, 1)
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2),
                                             encoding="utf-8")
    wandb.finish()

    print("\n" + SEP)
    print("  DONE")
    print(f"  baseline CLIPScore : {base_clip:.2f}")
    if history["epochs"]:
        print(f"  final CLIPScore    : {history['epochs'][-1]['CLIPScore']:.2f}")
    print(f"  adapters + history in {OUTPUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
