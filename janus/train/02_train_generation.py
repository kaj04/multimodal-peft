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

# Loss design. Default = plain CE on VQ tokens (Janus' pretraining objective).
# Setting RECON_LOSS_WEIGHT > 0 adds a reconstruction term on the VQ-VAE
# codebook embedding space, following the recipe in the Anole blog
# (weight ~0.5 reportedly avoids the mode-collapse failure of plain CE).
# Total loss = CE + RECON_LOSS_WEIGHT * MSE(softmax(logits) @ codebook,
#                                            codebook[target_tokens])
RECON_LOSS_WEIGHT = float(os.environ.get("RECON_LOSS_WEIGHT", "0.0"))

# LR schedule + optimisation
WARMUP_RATIO  = float(os.environ.get("WARMUP_RATIO", "0.05"))   # 5% linear warmup
WEIGHT_DECAY  = float(os.environ.get("WEIGHT_DECAY", "0.0"))    # 0 on LoRA is standard
GRAD_CLIP     = float(os.environ.get("GRAD_CLIP", "1.0"))

# Mid-epoch evaluation: every N optimizer steps, do a quick eval with MIN samples
EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "100"))
EVAL_MID_N       = int(os.environ.get("EVAL_MID_N", "10"))      # small for speed

# Resume capability: save full training state (adapter + optimizer + scheduler +
# RNG + counters) so a crashed or time-limited run can pick up where it left off.
SAVE_EVERY_STEPS = int(os.environ.get("SAVE_EVERY_STEPS", "0"))   # 0 = only at epoch end
RESUME           = os.environ.get("RESUME", "1") == "1"           # default: auto-resume if state exists

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

def _find_vq_codebook(vqmodel):
    """Locate the VQ codebook embedding tensor (16384 x embed_dim).

    HF transformer naming has varied across versions: try a few common
    attribute paths and return the first one that yields a 2-D tensor
    of the expected shape.
    """
    candidates = [
        ("quantize", "embedding", "weight"),
        ("quantize", "codebook"),
        ("quantize", "embeddings", "weight"),
        ("quantize", "codebook", "weight"),
        ("vector_quantizer", "embedding", "weight"),
    ]
    for path in candidates:
        obj = vqmodel
        ok = True
        for part in path:
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and torch.is_tensor(obj) and obj.dim() == 2:
            return obj
    # Last-ditch: scan named_parameters for something that looks like a codebook
    n_emb = getattr(vqmodel.config, "num_embeddings", None)
    for name, p in vqmodel.named_parameters():
        if p.dim() == 2 and (n_emb is None or p.shape[0] == n_emb):
            if "embed" in name.lower() or "codebook" in name.lower():
                return p
    raise RuntimeError(
        "Could not locate the VQ codebook embedding tensor; "
        "inspect vqmodel.named_parameters() and update _find_vq_codebook().")


def generation_loss(base, processor, image, caption,
                    recon_weight: float = 0.0):
    """One text->image teacher-forced step.

    Always computes CE on the predicted VQ token IDs (Janus' native objective).
    If recon_weight > 0, additionally computes a reconstruction loss in the
    codebook embedding space (Anole-style: see RECON_LOSS_WEIGHT comment).

    Returns: (total_loss, ce_loss_item, recon_loss_item) where the two scalars
    are detached floats for logging. total_loss has grad.
    """
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
    logits = inner.generation_head(img_hidden)              # (1, n_img, codebook)

    logits_f = logits.float()
    targets = image_tokens.reshape(-1).long()
    ce = F.cross_entropy(logits_f.reshape(-1, logits_f.shape[-1]), targets)

    if recon_weight > 0.0:
        # Codebook lookup: (codebook_size, embed_dim).
        # Cached on the base model so we resolve it only once per process.
        codebook = getattr(base, "_cached_vq_codebook", None)
        if codebook is None:
            codebook = _find_vq_codebook(inner.vqmodel).detach()
            base._cached_vq_codebook = codebook
        codebook_f = codebook.to(logits_f.device, logits_f.dtype)

        # Predicted distribution -> expected codebook embedding (differentiable)
        probs = F.softmax(logits_f, dim=-1)                  # (1, n_img, codebook)
        pred_embed   = probs @ codebook_f                    # (1, n_img, embed_dim)
        target_embed = codebook_f[targets].view(*pred_embed.shape)
        recon = F.mse_loss(pred_embed, target_embed)
        total = ce + recon_weight * recon
        return total, ce.detach().item(), recon.detach().item()

    return ce, ce.detach().item(), 0.0


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume from crash / time-limit)
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir, peft_model, optimizer, scheduler,
                    epoch, global_step, base_clip, history):
    """Save the full training state. Adapter is saved separately by PEFT
    (HF-compatible folder); the rest goes into state.pt so a resume restores
    everything end-to-end."""
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
    """If ckpt_dir/state.pt exists, restore everything and return the resume
    point as (start_epoch, global_step, base_clip, history). Otherwise return
    (0, 0, None, {})."""
    state_path = ckpt_dir / "state.pt"
    adapter_dir = ckpt_dir / "adapter"
    if not (state_path.exists() and adapter_dir.exists()):
        return 0, 0, None, {"baseline_clipscore": None, "epochs": []}

    from peft import PeftModel
    print(f"  RESUMING from {ckpt_dir}", flush=True)
    # Re-load the saved adapter weights on top of the (already-LoRA-injected) model
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
    print(f"  recon_loss_weight={RECON_LOSS_WEIGHT}  (0 = plain CE, Anole uses ~0.5)")
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

    # Early sanity check: if the reconstruction loss is on, locate the VQ
    # codebook NOW (right after model load) instead of at the first training
    # step. This makes a configuration error fail in ~2 minutes instead of ~8.
    if RECON_LOSS_WEIGHT > 0.0:
        print("  Pre-flight: locating VQ codebook for reconstruction loss...",
              flush=True)
        try:
            cb = _find_vq_codebook(base.model.vqmodel)
            print(f"  Codebook OK -> shape={tuple(cb.shape)} dtype={cb.dtype}\n",
                  flush=True)
            base._cached_vq_codebook = cb.detach()
        except Exception:
            print("  Codebook lookup FAILED. Candidates that exist in vqmodel:",
                  flush=True)
            for name, p in base.model.vqmodel.named_parameters():
                if p.dim() == 2 and ("emb" in name.lower()
                                     or "codebook" in name.lower()
                                     or "quant" in name.lower()):
                    print(f"    {name}: shape={tuple(p.shape)} dtype={p.dtype}",
                          flush=True)
            raise

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
        "recon_loss_weight": RECON_LOSS_WEIGHT,
        "loss_type": "ce+recon" if RECON_LOSS_WEIGHT > 0 else "ce",
    })

    # AdamW + cosine schedule with linear warmup.
    # Total optimizer steps over the whole run determine the cosine length.
    total_steps  = (len(train_ids) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    print(f"  total optimizer steps: {total_steps}  warmup: {warmup_steps}\n", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Try to resume from a 'latest/' checkpoint in this RUN's OUTPUT_DIR.
    latest_dir = OUTPUT_DIR / "latest"
    start_epoch, global_step, base_clip, history = (0, 0, None, None)
    if RESUME:
        start_epoch, global_step, base_clip, history = load_checkpoint_if_present(
            latest_dir, base, optimizer, scheduler)

    if base_clip is None:
        print("  measuring baseline CLIPScore (pre-training)...", flush=True)
        base_clip = eval_clipscore(base, processor, scorer, eval_ids, id2file, refs, GEN_EVAL_N)
        print(f"  baseline CLIPScore={base_clip:.2f}", flush=True)
        wandb.log({"eval/CLIPScore": base_clip, "epoch": 0})
        history = {"baseline_clipscore": base_clip, "epochs": []}
    else:
        print(f"  resumed; baseline CLIPScore={base_clip:.2f}", flush=True)

    optimizer.zero_grad()
    rng = random.Random(SEED)
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        order = train_ids[:]
        rng.shuffle(order)
        running, running_ce, running_recon = 0.0, 0.0, 0.0
        for i, img_id in enumerate(order):
            image = Image.open(VAL_IMAGES / id2file[img_id]).convert("RGB")
            loss, ce_item, recon_item = generation_loss(
                base, processor, image, refs[img_id][0],
                recon_weight=RECON_LOSS_WEIGHT)
            (loss / GRAD_ACCUM).backward()
            running       += loss.item()
            running_ce    += ce_item
            running_recon += recon_item

            if (i + 1) % GRAD_ACCUM == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                avg       = running       / GRAD_ACCUM
                avg_ce    = running_ce    / GRAD_ACCUM
                avg_recon = running_recon / GRAD_ACCUM
                running, running_ce, running_recon = 0.0, 0.0, 0.0
                cur_lr = scheduler.get_last_lr()[0]
                wandb.log({"train/loss":       avg,
                           "train/loss_ce":    avg_ce,
                           "train/loss_recon": avg_recon,
                           "train/lr":         cur_lr,
                           "train/grad_norm":  float(gnorm),
                           "step":             global_step})
                if global_step % 20 == 0:
                    extra = (f" ce={avg_ce:.4f} recon={avg_recon:.4f}"
                             if RECON_LOSS_WEIGHT > 0 else "")
                    print(f"  epoch {epoch} step {global_step} "
                          f"({i+1}/{len(order)}) loss={avg:.4f}{extra} "
                          f"lr={cur_lr:.2e}", flush=True)

                # Mid-epoch eval: cheap CLIPScore probe to see direction live.
                if EVAL_EVERY_STEPS > 0 and global_step % EVAL_EVERY_STEPS == 0:
                    mid_clip = eval_clipscore(base, processor, scorer,
                                              eval_ids, id2file, refs, EVAL_MID_N)
                    print(f"    [mid-eval @ step {global_step}] "
                          f"CLIPScore({EVAL_MID_N})={mid_clip:.2f}", flush=True)
                    wandb.log({"eval/CLIPScore_mid": mid_clip, "step": global_step})

                # Periodic full checkpoint (optional, for very long runs).
                if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                    save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                                    epoch, global_step, base_clip, history)
                    print(f"    saved checkpoint @ step {global_step}", flush=True)

        clip = eval_clipscore(base, processor, scorer, eval_ids, id2file, refs, GEN_EVAL_N)
        print(f"  [epoch {epoch}] CLIPScore={clip:.2f}", flush=True)
        wandb.log({"eval/CLIPScore": clip, "epoch": epoch + 1})
        history["epochs"].append({"epoch": epoch + 1, "CLIPScore": clip})

        # Two checkpoints per epoch: a permanent per-epoch adapter (for later
        # forgetting eval / matrix rows), and a 'latest' full state for resume.
        per_epoch = OUTPUT_DIR / f"adapter_epoch{epoch+1}"
        peft_model.save_pretrained(str(per_epoch))
        print(f"  saved adapter -> {per_epoch}", flush=True)
        save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                        epoch + 1, global_step, base_clip, history)
        print(f"  saved resume state -> {latest_dir}", flush=True)

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
