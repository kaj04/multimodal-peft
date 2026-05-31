#!/usr/bin/env python3
"""
janus/train/01_train_captioning.py

First REAL LoRA fine-tuning run: Janus-Pro-1B on COCO captioning
(image -> text), logged to Weights & Biases.

Design choices (deliberately minimal-risk for the first real run):
  - batch_size = 1 + gradient accumulation. Reuses the exact single-sample
    label-masking from the validated dry-run, avoiding any batched-padding bugs.
    True batching can come later once this is proven.
  - Data: a disjoint split of COCO val2017 (we don't have train2017 yet).
    TRAIN_N images to fine-tune on, EVAL_N held-out images to measure CIDEr.
  - Eval: CIDEr on the held-out split BEFORE training (baseline) and after each
    epoch, so we can see the captioning gain. (Generation forgetting is measured
    separately later; this run validates the training pipeline end to end.)
  - Saves the LoRA adapter after each epoch.

Everything is configurable via env vars (see CONFIG). Run via the SLURM script.
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

# reuse the validated scorers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from metrics import captioning_scores  # noqa: E402

# --- CONFIG (env-overridable) ----------------------------------------------
MODEL_ID   = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-1B")
COCO_DIR   = Path("/scratch-shared/tesr125338/coco")
VAL_IMAGES = COCO_DIR / "val2017"
VAL_CAPS   = COCO_DIR / "annotations" / "captions_val2017.json"

TRAIN_N    = int(os.environ.get("TRAIN_N", "4000"))   # fine-tune images
EVAL_N     = int(os.environ.get("EVAL_N", "200"))     # held-out images for CIDEr
EPOCHS     = int(os.environ.get("EPOCHS", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR         = float(os.environ.get("LR", "1e-4"))
LORA_RANK  = int(os.environ.get("LORA_RANK", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
SEED       = int(os.environ.get("SEED", "42"))

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "janus-peft")
RUN_NAME      = os.environ.get("RUN_NAME", "cap_1b_r8_valsplit")

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/train/output"),
)) / RUN_NAME

SEP = "=" * 64


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split():
    data = json.loads(VAL_CAPS.read_text())
    id2file = {im["id"]: im["file_name"] for im in data["images"]}
    refs = defaultdict(list)
    for ann in data["annotations"]:
        refs[ann["image_id"]].append(ann["caption"])
    ids = [i for i in refs if (VAL_IMAGES / id2file[i]).exists()]
    rng = random.Random(SEED)
    rng.shuffle(ids)
    train_ids = ids[:TRAIN_N]
    eval_ids  = ids[TRAIN_N:TRAIN_N + EVAL_N]   # disjoint from train
    return train_ids, eval_ids, id2file, refs


# ---------------------------------------------------------------------------
# Single-sample builders (reuse validated dry-run logic)
# ---------------------------------------------------------------------------

def build_caption_inputs(model, processor, image, caption):
    """One supervised (image -> caption) example, labels masked to the answer."""
    user_turn = [{"role": "user",
                  "content": [{"type": "image"},
                              {"type": "text", "text": "Describe this image."}]}]
    full = user_turn + [{"role": "assistant",
                         "content": [{"type": "text", "text": caption}]}]
    prompt_str = processor.apply_chat_template(
        user_turn, add_generation_prompt=True, generation_mode="text")
    full_str = processor.apply_chat_template(full, generation_mode="text")

    prompt_inputs = processor(text=prompt_str, images=[image], return_tensors="pt")
    full_inputs   = processor(text=full_str,   images=[image], return_tensors="pt")

    input_ids = full_inputs["input_ids"]
    labels = input_ids.clone()
    labels[:, :prompt_inputs["input_ids"].shape[1]] = -100
    img_id = getattr(model.config, "image_token_id", None)
    if img_id is not None:
        labels[input_ids == img_id] = -100

    dtype = next(model.parameters()).dtype
    return {
        "input_ids":      input_ids.to(model.device),
        "attention_mask": full_inputs["attention_mask"].to(model.device),
        "pixel_values":   full_inputs["pixel_values"].to(model.device, dtype),
        "labels":         labels.to(model.device),
    }


@torch.no_grad()
def generate_caption(model, processor, image, max_new_tokens=64):
    messages = [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text", "text": "Describe this image in one sentence."}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, generation_mode="text")
    inputs = processor(text=prompt, images=[image], generation_mode="text",
                       return_tensors="pt").to(model.device, next(model.parameters()).dtype)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                        generation_mode="text", do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def eval_cider(model, processor, eval_ids, id2file, refs):
    model.eval()
    hyps = {}
    for img_id in eval_ids:
        image = Image.open(VAL_IMAGES / id2file[img_id]).convert("RGB")
        hyps[img_id] = generate_caption(model, processor, image)
    model.train()
    return captioning_scores(refs, hyps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import wandb
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  LoRA captioning fine-tune | run={RUN_NAME}")
    print(f"  TRAIN_N={TRAIN_N} EVAL_N={EVAL_N} EPOCHS={EPOCHS} "
          f"GRAD_ACCUM={GRAD_ACCUM} LR={LR} rank={LORA_RANK}")
    print(SEP + "\n")

    # Model + LoRA
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    model = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    lora_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"  trainable params: {n_train/1e6:.3f}M\n")

    train_ids, eval_ids, id2file, refs = load_split()
    print(f"  train images: {len(train_ids)} | eval images: {len(eval_ids)}\n")

    wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
        "model": MODEL_ID, "train_n": TRAIN_N, "eval_n": EVAL_N, "epochs": EPOCHS,
        "grad_accum": GRAD_ACCUM, "lr": LR, "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA, "seed": SEED,
    })

    optimizer = torch.optim.AdamW(trainable, lr=LR)

    # Baseline CIDEr (step 0)
    print("  measuring baseline CIDEr (pre-training)...", flush=True)
    base = eval_cider(model, processor, eval_ids, id2file, refs)
    print(f"  baseline CIDEr={base['CIDEr']:.4f} BLEU_4={base['BLEU_4']:.4f}", flush=True)
    wandb.log({"eval/CIDEr": base["CIDEr"], "eval/BLEU_4": base["BLEU_4"], "epoch": 0})
    history = {"baseline": base, "epochs": []}

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
            batch = build_caption_inputs(model, processor, image, refs[img_id][0])
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            running += out.loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg = running / GRAD_ACCUM
                running = 0.0
                wandb.log({"train/loss": avg, "train/lr": LR,
                           "train/grad_norm": float(gnorm), "step": global_step})
                if global_step % 20 == 0:
                    print(f"  epoch {epoch} step {global_step} "
                          f"({i+1}/{len(order)}) loss={avg:.4f}", flush=True)

        # End-of-epoch eval + checkpoint
        ev = eval_cider(model, processor, eval_ids, id2file, refs)
        print(f"  [epoch {epoch}] CIDEr={ev['CIDEr']:.4f} BLEU_4={ev['BLEU_4']:.4f}",
              flush=True)
        wandb.log({"eval/CIDEr": ev["CIDEr"], "eval/BLEU_4": ev["BLEU_4"],
                   "epoch": epoch + 1})
        history["epochs"].append({"epoch": epoch + 1, **ev})
        ckpt = OUTPUT_DIR / f"adapter_epoch{epoch+1}"
        model.save_pretrained(ckpt)
        print(f"  saved adapter -> {ckpt}", flush=True)

    history["minutes"] = round((time.time() - t_start) / 60, 1)
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2),
                                             encoding="utf-8")
    wandb.finish()

    print("\n" + SEP)
    print("  DONE")
    print(f"  baseline CIDEr : {base['CIDEr']:.4f}")
    if history["epochs"]:
        print(f"  final CIDEr    : {history['epochs'][-1]['CIDEr']:.4f}")
    print(f"  adapters + history in {OUTPUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
