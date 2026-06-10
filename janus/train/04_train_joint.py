#!/usr/bin/env python3
"""
janus/train/04_train_joint.py

Joint LoRA fine-tuning of Janus-Pro-7B on (image, text) pairs from Naruto BLIP
PLUS a stream of plain-text Wikipedia chunks from one of three domain buckets:

    TEXT_BUCKET = none    -> C0: image-only baseline (same as 02_train_generation)
    TEXT_BUCKET = matched -> C1: matched-domain text (Naruto wiki ~94 articles)
    TEXT_BUCKET = near    -> C2: near-domain text   (anime/manga wiki ~1700)
    TEXT_BUCKET = far     -> C3: far-domain text    (physics/math wiki ~3800)

Why this design (see the project memo for the full rationale):
  - All three text buckets are drawn from Wikipedia EN, so style/length/form
    are controlled. The only variable across C1/C2/C3 is the semantic domain
    of the text. -> Differences in the trained image adapter can be attributed
    to text DOMAIN, not text style.
  - Image data is identical across C0..C3 (same Naruto BLIP captions). The
    image direction is the dependent variable; text is the manipulated factor.
  - One LoRA adapter learns from BOTH directions in C1/C2/C3 (we are not
    training two separate adapters); the question is whether the text-stream
    domain shifts what the SHARED parameter set memorises.

Training step alternation:
  - Each grad-accum micro-step alternates image and text in a 1:1 ratio
    (TEXT_RATIO=0.5 default). The text step is plain causal-LM CE on the
    next-token; the image step is the same teacher-forced VQ-CE used in
    02_train_generation.py.
  - For TEXT_BUCKET=none the text step is skipped entirely -> C0 behaves
    exactly like the existing image-only trainer.

All other knobs (LR, schedule, warmup, betas, prefix, ranks) inherit the
same conventions as 02_train_generation.py.
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
MODEL_ID   = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")

# Joint-training specific
TEXT_BUCKET = os.environ.get("TEXT_BUCKET", "none").lower()   # none|matched|near|far
TEXT_RATIO  = float(os.environ.get("TEXT_RATIO", "0.5"))      # fraction of micro-steps that are text
TEXT_LR_SCALE = float(os.environ.get("TEXT_LR_SCALE", "1.0")) # multiplier on text loss
# 0 = use the whole bucket; >0 = subsample to this many chunks (size-control
# experiment, e.g. far@320 to match matched's chunk budget).
TEXT_MAX_CHUNKS = int(os.environ.get("TEXT_MAX_CHUNKS", "0"))

DATA_PREP_DIR = Path(os.environ.get(
    "DATA_PREP_DIR",
    os.path.expanduser("~/multimodal-peft/janus/data_prep/output")))
TEXT_FILES = {
    "matched": DATA_PREP_DIR / "wiki_matched_tokens.jsonl",
    "near":    DATA_PREP_DIR / "wiki_near_tokens.jsonl",
    "far":     DATA_PREP_DIR / "wiki_far_tokens.jsonl",
}

# Image side (Naruto BLIP)
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME",
                                 "lambdalabs/naruto-blip-captions")
HF_IMAGE_COL    = os.environ.get("HF_IMAGE_COL", "image")
HF_TEXT_COL     = os.environ.get("HF_TEXT_COL", "text")
CAPTION_PREFIX  = os.environ.get("CAPTION_PREFIX",
                                 "anime sprite in Naruto style, ")

# Generic training knobs
TRAIN_N    = int(os.environ.get("TRAIN_N", "4000"))
EVAL_N     = int(os.environ.get("EVAL_N", "80"))
GEN_EVAL_N = int(os.environ.get("GEN_EVAL_N", "20"))
EPOCHS     = int(os.environ.get("EPOCHS", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR         = float(os.environ.get("LR", "2e-5"))
LORA_RANK  = int(os.environ.get("LORA_RANK", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * int(os.environ.get("LORA_RANK", "8")))))
SEED       = int(os.environ.get("SEED", "42"))

LR_SCHEDULE   = os.environ.get("LR_SCHEDULE", "cosine").lower()
WARMUP_RATIO  = float(os.environ.get("WARMUP_RATIO", "0.05"))
WEIGHT_DECAY  = float(os.environ.get("WEIGHT_DECAY", "0.0"))
GRAD_CLIP     = float(os.environ.get("GRAD_CLIP", "1.0"))
ADAM_BETA1    = float(os.environ.get("ADAM_BETA1", "0.9"))
ADAM_BETA2    = float(os.environ.get("ADAM_BETA2", "0.95"))

# Match the v5 image-only baseline: extended target_modules (attention + MLP),
# higher dropout, KL-to-base regularizer. Without these the joint trainer
# would inherit v1's broken capacity profile.
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
TARGET_MODULES = os.environ.get(
    "TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
KL_LAMBDA = float(os.environ.get("KL_LAMBDA", "0.0"))

EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "100"))
EVAL_MID_N       = int(os.environ.get("EVAL_MID_N", "10"))

SAVE_EVERY_STEPS = int(os.environ.get("SAVE_EVERY_STEPS", "200"))
RESUME           = os.environ.get("RESUME", "1") == "1"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "janus-peft")
RUN_NAME      = os.environ.get("RUN_NAME", f"joint_7b_{TEXT_BUCKET}_r{LORA_RANK}")

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/train/output"),
)) / RUN_NAME

SEP = "=" * 64


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_image_split():
    """Naruto BLIP captions. Re-uses the same split convention as 02_train_*."""
    from datasets import load_dataset
    print(f"  loading {HF_DATASET_NAME} from HF ...", flush=True)
    ds = load_dataset(HF_DATASET_NAME, split="train")
    id2img = {i: ds[i][HF_IMAGE_COL] for i in range(len(ds))}
    refs   = {i: [ds[i][HF_TEXT_COL]] for i in range(len(ds))}
    ids = list(range(len(ds)))
    random.Random(SEED).shuffle(ids)
    n_eval  = min(EVAL_N, 80)
    n_train = min(TRAIN_N, len(ids) - n_eval)
    train_ids = ids[:n_train]
    eval_ids  = ids[n_train:n_train + n_eval]
    print(f"  image split: train={n_train} eval={n_eval}", flush=True)
    return train_ids, eval_ids, id2img, refs


def load_text_chunks(bucket: str):
    """Load pre-tokenised wiki chunks (512 ids per chunk) for the given bucket.
    Returns a list of LongTensors (one per chunk). For TEXT_BUCKET=none returns
    empty list.

    If TEXT_MAX_CHUNKS > 0, the chunk list is randomly subsampled down to that
    many chunks (seed-stable). This disentangles the corpus-SIZE confound from
    the domain-DISTANCE one: e.g. TEXT_BUCKET=far TEXT_MAX_CHUNKS=320 yields a
    'far' corpus with the SAME chunk budget as 'matched' (320), so any
    difference in the resulting text-LM perplexity is attributable to domain
    distance, not to corpus size."""
    if bucket == "none":
        return []
    path = TEXT_FILES.get(bucket)
    if path is None or not path.exists():
        raise SystemExit(f"text bucket file not found: {path}")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(torch.tensor(row["input_ids"], dtype=torch.long))
    n_full = len(chunks)
    if TEXT_MAX_CHUNKS > 0 and n_full > TEXT_MAX_CHUNKS:
        # Dedicated RNG so the subsample is reproducible and independent of
        # the image-split / text-step RNGs.
        idx = list(range(n_full))
        random.Random(SEED + 7).shuffle(idx)
        keep = sorted(idx[:TEXT_MAX_CHUNKS])
        chunks = [chunks[i] for i in keep]
        print(f"  text bucket '{bucket}': subsampled {n_full} -> "
              f"{len(chunks)} chunks (TEXT_MAX_CHUNKS={TEXT_MAX_CHUNKS})",
              flush=True)
    else:
        print(f"  text bucket '{bucket}': {len(chunks)} chunks of "
              f"{len(chunks[0]) if chunks else 0} tokens", flush=True)
    return chunks


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _apply_prefix(caption: str) -> str:
    if CAPTION_PREFIX:
        return CAPTION_PREFIX + caption
    return caption


def image_generation_loss(base, processor, image, caption,
                          kl_weight: float = 0.0, peft_model=None):
    """Same as 02_train_generation.generation_loss but for the joint setting.
    Supports the v5 KL-to-base regularizer (adapter is held close to the
    frozen base distribution; same recipe that solved the v2 degradation
    failure mode in the image-only baseline)."""
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
    Lt = text_embeds.shape[1]

    # Adapter forward
    hidden = inner(inputs_embeds=inputs_embeds).last_hidden_state
    img_hidden = hidden[:, Lt - 1:, :]
    logits = inner.generation_head(img_hidden)

    logits_f = logits.float()
    targets = image_tokens.reshape(-1).long()
    ce = F.cross_entropy(logits_f.reshape(-1, logits_f.shape[-1]), targets)

    if kl_weight > 0.0 and peft_model is not None:
        # Base distribution forward (adapter disabled, no grad)
        with peft_model.disable_adapter():
            with torch.no_grad():
                hidden_b = inner(inputs_embeds=inputs_embeds).last_hidden_state
                img_hidden_b = hidden_b[:, Lt - 1:, :]
                logits_b = inner.generation_head(img_hidden_b).float()
        log_p_a = F.log_softmax(logits_f, dim=-1)
        log_p_b = F.log_softmax(logits_b, dim=-1)
        kl = (log_p_a.exp() * (log_p_a - log_p_b)).sum(dim=-1).mean()
        return ce + kl_weight * kl
    return ce


def text_lm_loss(base, input_ids):
    """Standard causal LM next-token CE on a 1xL token sequence.

    NOTE: Janus is a multimodal model; its base.forward(input_ids=...) without
    image inputs runs the text path and returns standard LM logits. The CE is
    computed manually (logits shifted right vs labels shifted left) instead of
    relying on labels=... in case the HF auto-loss is wired only to a specific
    head."""
    device = base.device
    input_ids = input_ids.unsqueeze(0).to(device)   # (1, L)
    out = base(input_ids=input_ids)
    # Out can be either CausalLMOutput-like (.logits) or a ModelOutput; we
    # access .logits which is the standard attribute.
    logits = out.logits if hasattr(out, "logits") else out[0]
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
    return loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir, peft_model, optimizer, scheduler,
                    epoch, global_step, base_clip, history):
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir / "adapter"))
    torch.save({
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "epoch":       epoch,
        "global_step": global_step,
        "base_clip":   base_clip,
        "history":     history,
        "torch_rng":   torch.get_rng_state(),
        "cuda_rng":    (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None),
    }, out_dir / "state.pt")


def load_checkpoint_if_present(ckpt_dir, base, optimizer, scheduler):
    state_path = ckpt_dir / "state.pt"
    adapter_dir = ckpt_dir / "adapter"
    if not (state_path.exists() and adapter_dir.exists()):
        return 0, 0, None, {"baseline_clipscore": None, "epochs": []}
    from peft import PeftModel
    print(f"  RESUMING from {ckpt_dir}", flush=True)
    PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True,
                              adapter_name="default")
    s = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(s["optimizer"])
    scheduler.load_state_dict(s["scheduler"])
    torch.set_rng_state(s["torch_rng"])
    if s.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda_rng"])
    print(f"  resumed at epoch={s['epoch']} global_step={s['global_step']}",
          flush=True)
    return s["epoch"], s["global_step"], s["base_clip"], s["history"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_clipscore(base, processor, scorer, eval_ids, get_image, refs, n):
    base.eval()
    vals = []
    for k, img_id in enumerate(eval_ids[:n]):
        torch.manual_seed(SEED + k)
        raw = refs[img_id][0]
        img = generate_image(base, processor, _apply_prefix(raw))
        vals.append(scorer.score(img, raw))
    base.train()
    return float(np.mean(vals))


@torch.no_grad()
def eval_image_loss(base, processor, eval_ids, get_image, refs, n):
    base.eval()
    losses = []
    for img_id in eval_ids[:n]:
        image = get_image(img_id)
        ce = image_generation_loss(base, processor, image,
                                   _apply_prefix(refs[img_id][0]))
        losses.append(ce.item())
    base.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import wandb
    from transformers import (JanusForConditionalGeneration, JanusProcessor,
                              get_cosine_schedule_with_warmup,
                              get_constant_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  Janus JOINT fine-tune | run={RUN_NAME}")
    print(f"  MODEL_ID={MODEL_ID}")
    print(f"  TEXT_BUCKET={TEXT_BUCKET}  TEXT_RATIO={TEXT_RATIO}  TEXT_LR_SCALE={TEXT_LR_SCALE}")
    print(f"  TEXT_MAX_CHUNKS={TEXT_MAX_CHUNKS}  (0 = full bucket)")
    print(f"  EPOCHS={EPOCHS} GRAD_ACCUM={GRAD_ACCUM} LR={LR} schedule={LR_SCHEDULE}")
    print(f"  rank={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
          f"betas=({ADAM_BETA1},{ADAM_BETA2})")
    print(f"  target_modules={TARGET_MODULES}")
    print(f"  kl_lambda={KL_LAMBDA}")
    print(f"  caption_prefix={CAPTION_PREFIX!r}")
    print(SEP + "\n")

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    lora_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                          target_modules=TARGET_MODULES,
                          lora_dropout=LORA_DROPOUT, bias="none",
                          task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(base, lora_cfg)
    base.train()
    trainable = [p for p in base.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in trainable)/1e6:.3f}M\n")

    # Data
    train_ids, eval_ids, id2img, refs = load_image_split()
    text_chunks = load_text_chunks(TEXT_BUCKET)

    def get_image(img_id):
        return id2img[img_id].convert("RGB")

    scorer = CLIPScorer()
    wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
        "model": MODEL_ID, "direction": "joint",
        "text_bucket": TEXT_BUCKET, "text_ratio": TEXT_RATIO,
        "text_max_chunks": TEXT_MAX_CHUNKS,
        "text_lr_scale": TEXT_LR_SCALE,
        "epochs": EPOCHS, "grad_accum": GRAD_ACCUM, "lr": LR,
        "warmup_ratio": WARMUP_RATIO, "grad_clip": GRAD_CLIP,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "seed": SEED,
        "caption_prefix": CAPTION_PREFIX,
        "lr_schedule": LR_SCHEDULE,
    })

    # The number of optimiser steps per epoch counts as follows: every
    # GRAD_ACCUM micro-steps -> 1 opt step. Each epoch goes through all image
    # samples once. Text chunks are SAMPLED with replacement -> they do NOT
    # contribute to the per-epoch length; they ride along on the image schedule.
    total_steps  = (len(train_ids) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    print(f"  total optimiser steps: {total_steps}  warmup: {warmup_steps}\n",
          flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=(ADAM_BETA1, ADAM_BETA2))
    if LR_SCHEDULE == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    elif LR_SCHEDULE == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        raise ValueError(f"unknown LR_SCHEDULE={LR_SCHEDULE!r}")

    latest_dir = OUTPUT_DIR / "latest"
    start_epoch, global_step, base_clip, history = (0, 0, None, None)
    if RESUME:
        start_epoch, global_step, base_clip, history = load_checkpoint_if_present(
            latest_dir, base, optimizer, scheduler)

    if base_clip is None:
        print("  measuring baseline CLIPScore (pre-training)...", flush=True)
        base_clip = eval_clipscore(base, processor, scorer, eval_ids,
                                   get_image, refs, GEN_EVAL_N)
        print(f"  baseline CLIPScore={base_clip:.2f}", flush=True)
        wandb.log({"eval/CLIPScore": base_clip, "epoch": 0})
        history = {"baseline_clipscore": base_clip, "epochs": []}
    else:
        print(f"  resumed; baseline CLIPScore={base_clip:.2f}", flush=True)

    optimizer.zero_grad()
    rng = random.Random(SEED)
    txt_rng = random.Random(SEED + 1)
    t_start = time.time()

    # Probabilistic alternation: each micro-step is a text step with prob
    # TEXT_RATIO, else an image step. If TEXT_BUCKET=none, force image.
    # With TEXT_RATIO=0.5 we expect ~50/50 mix; the image side completes 1
    # full epoch when we've consumed all image samples.
    def is_text_step():
        if not text_chunks:
            return False
        return txt_rng.random() < TEXT_RATIO

    for epoch in range(start_epoch, EPOCHS):
        order = train_ids[:]
        rng.shuffle(order)
        img_cursor = 0
        running_img, running_txt = 0.0, 0.0
        running_img_n, running_txt_n = 0, 0

        # We loop until we have CONSUMED all image samples this epoch. Text
        # samples are drawn from the chunks pool with replacement.
        # Total micro-steps per epoch:
        #     n_img_per_epoch / (1 - TEXT_RATIO)    (in expectation)
        if text_chunks:
            n_microsteps = int(len(order) / max(1e-6, (1 - TEXT_RATIO)))
        else:
            n_microsteps = len(order)

        for micro in range(n_microsteps):
            if is_text_step():
                # Text micro-step
                chunk = text_chunks[txt_rng.randrange(len(text_chunks))]
                loss = TEXT_LR_SCALE * text_lm_loss(base, chunk)
                running_txt += loss.item()
                running_txt_n += 1
            else:
                # Image micro-step (advance the image cursor)
                if img_cursor >= len(order):
                    # End of epoch's image budget; stop the inner loop early
                    break
                img_id = order[img_cursor]
                img_cursor += 1
                image = get_image(img_id)
                loss = image_generation_loss(base, processor, image,
                                             _apply_prefix(refs[img_id][0]),
                                             kl_weight=KL_LAMBDA,
                                             peft_model=peft_model)
                running_img += loss.item()
                running_img_n += 1

            (loss / GRAD_ACCUM).backward()

            if ((micro + 1) % GRAD_ACCUM == 0) or (micro + 1 == n_microsteps):
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                cur_lr = scheduler.get_last_lr()[0]
                avg_img = running_img / max(1, running_img_n)
                avg_txt = running_txt / max(1, running_txt_n)
                wandb.log({
                    "train/loss_img":   avg_img,
                    "train/loss_txt":   avg_txt,
                    "train/lr":         cur_lr,
                    "train/grad_norm":  float(gnorm),
                    "train/text_frac":  running_txt_n / max(1, running_txt_n + running_img_n),
                    "step":             global_step,
                })
                running_img, running_txt = 0.0, 0.0
                running_img_n, running_txt_n = 0, 0

                if global_step % 20 == 0:
                    print(f"  epoch {epoch} step {global_step} micro {micro+1}/{n_microsteps}  "
                          f"loss_img={avg_img:.4f} loss_txt={avg_txt:.4f} "
                          f"lr={cur_lr:.2e}", flush=True)

                if EVAL_EVERY_STEPS > 0 and global_step % EVAL_EVERY_STEPS == 0:
                    mid_clip = eval_clipscore(base, processor, scorer, eval_ids,
                                              get_image, refs, EVAL_MID_N)
                    print(f"    [mid-eval @ step {global_step}] "
                          f"CLIPScore({EVAL_MID_N})={mid_clip:.2f}", flush=True)
                    wandb.log({"eval/CLIPScore_mid": mid_clip, "step": global_step})

                if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                    save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                                    epoch, global_step, base_clip, history)

        # End-of-epoch eval: full CLIPScore + full val CE on held-out images
        clip = eval_clipscore(base, processor, scorer, eval_ids,
                              get_image, refs, GEN_EVAL_N)
        val_ce = eval_image_loss(base, processor, eval_ids, get_image, refs, EVAL_N)
        print(f"  [epoch {epoch}] CLIPScore={clip:.2f}  val_CE={val_ce:.4f}",
              flush=True)
        wandb.log({"eval/CLIPScore": clip,
                   "val/CE_loss":   val_ce,
                   "epoch":         epoch + 1})
        history["epochs"].append({"epoch": epoch + 1,
                                  "CLIPScore": clip,
                                  "val_CE": val_ce})

        per_epoch = OUTPUT_DIR / f"adapter_epoch{epoch+1}"
        peft_model.save_pretrained(str(per_epoch))
        print(f"  saved adapter -> {per_epoch}", flush=True)
        save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                        epoch + 1, global_step, base_clip, history)

    history["minutes"] = round((time.time() - t_start) / 60, 1)
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2),
                                             encoding="utf-8")
    wandb.finish()

    print("\n" + SEP)
    print("  DONE")
    print(f"  baseline CLIPScore : {base_clip:.2f}")
    if history["epochs"]:
        last = history["epochs"][-1]
        print(f"  final  CLIPScore   : {last['CLIPScore']:.2f}")
        print(f"  final  val_CE      : {last['val_CE']:.4f}")
    print(f"  adapters + history in {OUTPUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
