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
COCO_SPLIT = os.environ.get("COCO_SPLIT", "val2017")   # val2017 | train2017
VAL_IMAGES = COCO_DIR / COCO_SPLIT
VAL_CAPS   = COCO_DIR / "annotations" / f"captions_{COCO_SPLIT}.json"

TRAIN_N    = int(os.environ.get("TRAIN_N", "4000"))
EVAL_N     = int(os.environ.get("EVAL_N", "200"))
GEN_EVAL_N = int(os.environ.get("GEN_EVAL_N", "30"))   # images are slow to gen
EPOCHS     = int(os.environ.get("EPOCHS", "3"))        # gen needs more than 1
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR         = float(os.environ.get("LR", "5e-5"))       # lower than text LR; gen loss is sparser
LORA_RANK    = int(os.environ.get("LORA_RANK", "4"))   # captioning saturates at low r; try the same regime
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", str(2 * int(os.environ.get("LORA_RANK", "4")))))
# LoRA dropout. Default 0.05 is the PEFT library default and what we used in
# v1/v2/v3. v4-AGGRESSIVE bumps to 0.1 to mitigate overfit risk when running
# higher rank + higher LR on a small dataset.
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
SEED       = int(os.environ.get("SEED", "42"))

# Dataset source. "coco" (default) keeps the existing val2017 split workflow.
# "hf" loads any image-caption dataset from HuggingFace as a narrow-target
# proof-of-concept. The Pokemon BLIP captions dataset community-standard for
# LoRA-on-Stable-Diffusion has been gated, so we default to the Naruto one
# (same family, same {image, text} schema, public). Override HF_DATASET_NAME
# to use any other dataset with the same column names.
DATASET_SOURCE  = os.environ.get("DATASET_SOURCE", "coco").lower()
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME",
                                 "lambdalabs/naruto-blip-captions")
HF_IMAGE_COL    = os.environ.get("HF_IMAGE_COL", "image")
HF_TEXT_COL     = os.environ.get("HF_TEXT_COL", "text")
# Back-compat: the old "pokemon" value still works as a synonym for hf.
if DATASET_SOURCE == "pokemon":
    DATASET_SOURCE = "hf"

# LR schedule. Default cosine (what we've used until now). "constant" follows
# the Janus paper recipe (constant LR with linear warmup, in all 3 stages).
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine").lower()

# Optional prefix prepended to every caption at training (hybrid prompt
# templating). Idea: if every train caption is e.g. "anime sprite in Naruto
# style, <BLIP caption>", the model learns the style is gated by the trigger
# phrase. At inference, the same prefix re-activates the learned domain; users
# can also drop it to compare. Empty string disables prefixing.
CAPTION_PREFIX = os.environ.get("CAPTION_PREFIX", "")

# Caption cleaning. CLEAN_CAPTIONS=1 strips style/quality "buzz" clauses from
# every caption (oil painting, 8k, trending on artstation, by <artist>,
# pixel/8-bit, ...) so the captions describe only CONTENT. Rationale: on a
# dataset whose captions FIGHT the target style (DiffusionDB prompts demand
# photoreal detail while the images are pixel art), the competing style words
# make the model gate the style on a trigger phrase. Removing them lets the
# adapter attribute the visual style to ITSELF and apply it UNCONDITIONALLY
# (the way the Naruto BLIP content-captions already did). Default off so
# existing runs are unchanged.
CLEAN_CAPTIONS = os.environ.get("CLEAN_CAPTIONS", "0") == "1"

# Target-image transform applied to every training (and eval) image.
# "pixelart" makes a crisp, palette-reduced pixel-art version of any photo, so
# COCO (diverse content + clean content captions) becomes a strong, distinct,
# UNCONDITIONAL style with no adversarial captions and no duplicates. The
# pixelation strength (block grid x palette) doubles as a controllable distance
# dial for the forgetting-vs-intervention sweep. Default "" = no transform.
IMAGE_TRANSFORM = os.environ.get("IMAGE_TRANSFORM", "").lower()
PIXELART_SIZE   = int(os.environ.get("PIXELART_SIZE", "48"))    # downsample grid
PIXELART_COLORS = int(os.environ.get("PIXELART_COLORS", "24"))  # palette size

# AdamW betas. The Janus paper uses beta2=0.95 (default in transformers is 0.999).
# Lower beta2 -> faster adaptation of the second-moment estimate, helps when
# gradient variance is high (sparse CE signal over a 16k codebook).
ADAM_BETA1 = float(os.environ.get("ADAM_BETA1", "0.9"))
ADAM_BETA2 = float(os.environ.get("ADAM_BETA2", "0.95"))

# LoRA target modules. Default is attention-only (q,k,v,o), the recipe used
# until v1. v2-CAPACITY adds the MLP projections (gate/up/down) to give the
# adapter ~3x more trainable params -- needed on 7B to match the trainable-
# parameter RATIO of the 1B baseline (0.44%) which our v1 7B run severely
# underprovisioned at 0.13%.
TARGET_MODULES = os.environ.get(
    "TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")

# Periodic image samples saved to disk during training. Lightweight (a handful
# of fixed prompts) so we can SEE the model evolve epoch-by-epoch in the
# samples/ folder + W&B media panel, instead of only at the final eval.
# Default 100 steps = typically 5-6 times per epoch, ~10 sec each on H100.
SAMPLE_EVERY_STEPS = int(os.environ.get("SAMPLE_EVERY_STEPS", "100"))
N_SAMPLE_PROMPTS   = int(os.environ.get("N_SAMPLE_PROMPTS", "6"))

# Loss design. Default = plain CE on VQ tokens (Janus' pretraining objective).
# Setting RECON_LOSS_WEIGHT > 0 adds a reconstruction term on the VQ-VAE
# codebook embedding space, following the recipe in the Anole blog
# (weight ~0.5 reportedly avoids the mode-collapse failure of plain CE).
# Total loss = CE + RECON_LOSS_WEIGHT * MSE(softmax(logits) @ codebook,
#                                            codebook[target_tokens])
RECON_LOSS_WEIGHT = float(os.environ.get("RECON_LOSS_WEIGHT", "0.0"))

# KL-to-base regularizer weight. KL_LAMBDA > 0 adds
#     lambda * KL(p_adapter || p_base)
# averaged over image-token positions, with p_base from a no-grad forward of
# the model with the adapter disabled. Targets the "the adapter is destroying
# capability the base already had" failure mode (v1 7B diagnostic). Lambda
# in [0.1, 1.0] is a sensible starting range -- start with 0.5.
KL_LAMBDA = float(os.environ.get("KL_LAMBDA", "0.0"))

# Which regularizer the KL_LAMBDA weight applies to (regularizer ablation):
#   "kl"  -> KL(p_adapter || p_base) on the output distribution (default)
#   "mse" -> MSE between adapter and base logits (output space, no softmax)
#   "l2"  -> L2 on the LoRA parameters (weight decay toward the base; no base fwd)
REG_TYPE = os.environ.get("REG_TYPE", "kl").lower()

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
# Sequential (continual) chaining: start from a PREVIOUS run's LoRA weights with
# a fresh optimizer/scheduler/epoch counter. Distinct from RESUME, which
# continues an interrupted run of the same task. See train/_chain_init.py.
INIT_ADAPTER     = os.environ.get("INIT_ADAPTER", "")

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
    """COCO val2017: returns (train_ids, eval_ids, id2file, refs).
    id2file maps int id -> filename string (image opened lazily from disk).
    refs   maps int id -> list of ~5 reference caption strings."""
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


def load_hf_image_caption_split():
    """Load any small image+caption HF dataset (narrow-target proof-of-concept).

    Default is lambdalabs/naruto-blip-captions (~1.2k pairs, public, same
    schema as the gated pokemon variant). Override with HF_DATASET_NAME /
    HF_IMAGE_COL / HF_TEXT_COL env vars.

    Returns the same tuple shape as load_split() so the rest of the trainer
    is unchanged. id2img maps int id -> PIL.Image (already loaded in RAM;
    these datasets are tiny, tens of MB)."""
    from datasets import load_dataset
    # HF_CONFIG: dataset sub-config name (some datasets, e.g.
    # diffusiondb-pixelart, ship multiple configs). HF_REVISION: branch/commit
    # (e.g. 'refs/convert/parquet' to bypass deprecated loading scripts).
    hf_config     = os.environ.get("HF_CONFIG", "") or None
    hf_revision   = os.environ.get("HF_REVISION", "") or None
    # HF_DATA_FILES: point at specific parquet file(s) within the repo, e.g.
    # '2k_random_2k/train/0000.parquet' -- needed for datasets like
    # diffusiondb-pixelart whose parquet branch bundles incompatible
    # sub-configs under a single 'default' config.
    hf_data_files = os.environ.get("HF_DATA_FILES", "") or None
    print(f"  loading {HF_DATASET_NAME} from HF "
          f"(config={hf_config!r}, revision={hf_revision!r}, "
          f"data_files={hf_data_files!r}, "
          f"image_col={HF_IMAGE_COL!r}, text_col={HF_TEXT_COL!r})...", flush=True)
    load_args = [HF_DATASET_NAME] + ([hf_config] if hf_config else [])
    load_kwargs = {"split": "train"}
    if hf_revision:
        load_kwargs["revision"] = hf_revision
    if hf_data_files:
        load_kwargs["data_files"] = hf_data_files
    ds = load_dataset(*load_args, **load_kwargs)
    if HF_IMAGE_COL not in ds.column_names or HF_TEXT_COL not in ds.column_names:
        raise ValueError(
            f"dataset columns {ds.column_names!r} do not contain "
            f"{HF_IMAGE_COL!r}/{HF_TEXT_COL!r}; "
            f"override HF_IMAGE_COL and HF_TEXT_COL")
    id2img = {i: ds[i][HF_IMAGE_COL] for i in range(len(ds))}
    refs   = {i: [ds[i][HF_TEXT_COL]] for i in range(len(ds))}
    if CLEAN_CAPTIONS:
        examples = [(refs[i][0], clean_caption(refs[i][0]))
                    for i in range(min(6, len(ds)))]
        refs = {i: [clean_caption(refs[i][0])] for i in refs}
        print("  CLEAN_CAPTIONS=1 -> stripped style buzzwords. Examples:", flush=True)
        for raw, cl in examples:
            print(f"    RAW  : {raw[:100]}", flush=True)
            print(f"    CLEAN: {cl[:100]}", flush=True)
    ids = list(range(len(ds)))
    random.Random(SEED).shuffle(ids)
    # Small dataset: reserve ~80 for eval, rest for training (capped at TRAIN_N).
    n_eval  = min(EVAL_N, 80)
    n_train = min(TRAIN_N, len(ids) - n_eval)
    train_ids = ids[:n_train]
    eval_ids  = ids[n_train:n_train + n_eval]
    print(f"  hf split: train={n_train} eval={n_eval}  (total in dataset: {len(ds)})",
          flush=True)
    return train_ids, eval_ids, id2img, refs


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
                    recon_weight: float = 0.0,
                    kl_weight: float = 0.0,
                    reg_type: str = "kl",
                    peft_model=None):
    """One text->image teacher-forced step.

    Always computes CE on the predicted VQ token IDs (Janus' native objective).
    Optionally adds two regularizers:

    1) recon_weight > 0  -- Anole-style reconstruction loss on codebook
       embeddings. Softens the CE label (a "wrong" VQ id whose embedding is
       close to the target's contributes less). Architecture-specific to VQ
       models. See the original Anole blog for the recipe.

    2) kl_weight > 0     -- KL(p_adapter || p_base) regularizer toward the
       FROZEN base model's distribution on the same input. Requires
       `peft_model` to call disable_adapter(). Inspired by the PPO/RLHF
       policy-regularization term and by knowledge distillation. Diagnosed
       use case: the base model already has competence we want to preserve
       (so the adapter should *refine* rather than *overwrite*).
       Cost: ~2x transformer forward (one with adapter, one without).

    Returns: (total_loss, ce_item, recon_item, kl_item) -- the last three are
    detached floats for logging. total_loss carries gradient through the
    adapter (LoRA) parameters only.
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

    # Embeddings: the input-embedding lookup and the image-embedding prep
    # are NOT LoRA-targeted (LoRA covers q/k/v/o + optionally MLP), so they
    # produce the SAME tensor with adapter on or off. We compute them ONCE
    # and reuse for both forwards if KL is active.
    text_embeds  = base.get_input_embeddings()(text_ids)
    image_embeds = base.prepare_embeddings_for_image_generation(image_tokens)
    inputs_embeds = torch.cat([text_embeds, image_embeds[:, :-1, :]], dim=1)
    Lt = text_embeds.shape[1]

    # --- Adapter forward (gradient flows here) ---
    hidden = inner(inputs_embeds=inputs_embeds).last_hidden_state
    img_hidden = hidden[:, Lt - 1:, :]
    logits = inner.generation_head(img_hidden)              # (1, n_img, codebook)

    logits_f = logits.float()
    targets = image_tokens.reshape(-1).long()
    ce = F.cross_entropy(logits_f.reshape(-1, logits_f.shape[-1]), targets)

    total = ce
    recon_item, kl_item = 0.0, 0.0

    # --- Optional recon (Anole-style codebook embedding MSE) ---
    if recon_weight > 0.0:
        codebook = getattr(base, "_cached_vq_codebook", None)
        if codebook is None:
            codebook = _find_vq_codebook(inner.vqmodel).detach()
            base._cached_vq_codebook = codebook
        codebook_f = codebook.to(logits_f.device, logits_f.dtype)
        probs = F.softmax(logits_f, dim=-1)                  # (1, n_img, codebook)
        pred_embed   = probs @ codebook_f                    # (1, n_img, embed_dim)
        target_embed = codebook_f[targets].view(*pred_embed.shape)
        recon = F.mse_loss(pred_embed, target_embed)
        total = total + recon_weight * recon
        recon_item = recon.detach().item()

    # --- Optional stay-close-to-base regularizer (weight kl_weight) ---
    # reg_type selects the divergence:
    #   "l2"  -> L2 on the LoRA parameters (weight decay toward the base); no
    #            base forward needed, so it is free.
    #   "mse" -> MSE between adapter and base logits (output space).
    #   "kl"  -> KL(p_adapter || p_base) on the output distribution (default).
    # The "mse"/"kl" variants replay the SAME transformer call with the LoRA
    # adapter disabled (no_grad) to get the base distribution (~2x fwd cost).
    if kl_weight > 0.0 and peft_model is not None:
        if reg_type == "l2":
            reg = sum(p.float().pow(2).sum()
                      for n, p in peft_model.named_parameters()
                      if p.requires_grad and ("lora_A" in n or "lora_B" in n))
        else:
            with peft_model.disable_adapter():
                with torch.no_grad():
                    hidden_b = inner(inputs_embeds=inputs_embeds).last_hidden_state
                    img_hidden_b = hidden_b[:, Lt - 1:, :]
                    logits_b = inner.generation_head(img_hidden_b).float()
            if reg_type == "mse":
                reg = F.mse_loss(logits_f, logits_b)
            else:  # "kl": per-position KL(p_adapter || p_base), mean over positions
                log_p_a = F.log_softmax(logits_f, dim=-1)   # (1, n_img, V)
                log_p_b = F.log_softmax(logits_b, dim=-1)
                reg = (log_p_a.exp() * (log_p_a - log_p_b)).sum(dim=-1).mean()
        total = total + kl_weight * reg
        kl_item = float(reg.detach().item())

    return total, ce.detach().item(), recon_item, kl_item


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


def _apply_prefix(caption: str) -> str:
    """Prepend CAPTION_PREFIX if non-empty. Single place to enforce the rule."""
    if CAPTION_PREFIX:
        return CAPTION_PREFIX + caption
    return caption


# Style/quality "buzz" markers. A comma-clause containing any of these is
# dropped by clean_caption (used when CLEAN_CAPTIONS=1). Tuned for DiffusionDB
# prompts. Extend via EXTRA_STYLE_MARKERS (comma-separated env).
_STYLE_MARKERS = (
    "by ", " style", "artstation", "deviantart", "behance", "pixiv", "trending",
    "render", "octane", "unreal engine", "ray tracing", "raytracing",
    "detailed", "intricate", "lighting", "photoreal", "hyperreal", "realistic",
    "4k", "8k", "16k", "uhd", " hd", "resolution", "concept art",
    "digital painting", "digital art", "masterpiece", "sharp focus", "bokeh",
    "cinematic", "volumetric", "illustration", "painting", "watercolor",
    "oil on", "oil painting", "acrylic", "charcoal", "sketch", "drawing",
    "cgi", "vray", "redshift", "studio", "award", "highly", "ultra",
    "pixel", "8-bit", "16-bit", "voxel", "sprite", "lowres", "low - res",
    "smooth", "wallpaper",
)
_EXTRA_MARKERS = [m.strip().lower()
                  for m in os.environ.get("EXTRA_STYLE_MARKERS", "").split(",")
                  if m.strip()]
if _EXTRA_MARKERS:
    _STYLE_MARKERS = _STYLE_MARKERS + tuple(_EXTRA_MARKERS)


def clean_caption(text: str) -> str:
    """Keep only CONTENT clauses: split on commas (and '!!'), drop any clause
    that looks like style/quality boilerplate, return the rest. Empty result
    -> a neutral fallback so we never train on an empty prompt."""
    if not isinstance(text, str):
        return "an image"
    clauses = [c.strip() for c in text.replace("!!", ",").split(",")]
    kept = [c for c in clauses
            if c and not any(m in c.lower() for m in _STYLE_MARKERS)]
    out = ", ".join(kept).strip(" .,-!")
    return out if len(out) >= 3 else "an image"


def _pixelate(img):
    """Crisp pixel-art transform: nearest-neighbour downsample to a small grid
    + palette quantization, then nearest upscale back. Strong & controllable
    (PIXELART_SIZE = block grid, PIXELART_COLORS = palette)."""
    w, h = img.size
    small = img.resize((PIXELART_SIZE, PIXELART_SIZE), Image.NEAREST)
    small = small.quantize(colors=PIXELART_COLORS).convert("RGB")
    return small.resize((w, h), Image.NEAREST)


def _apply_image_transform(img):
    """Apply the configured target-image transform (identity if unset)."""
    if IMAGE_TRANSFORM == "pixelart":
        return _pixelate(img)
    return img


# Fixed sampling prompts for the periodic sample dumps. Mix of WITH-prefix
# (test in-distribution) and WITHOUT-prefix (test if the adapter "activates"
# the learned style even without the trigger phrase). Hardcoded so the dumps
# are directly comparable across steps and across runs.
# Neutral, style-agnostic sample prompts so the W&B media trace is meaningful
# for ANY dataset/style (Naruto, pixel art, ...). The adapter renders these in
# whatever style it is learning; CAPTION_PREFIX (if set) is applied via
# _apply_prefix at generation time. Override with SAMPLE_PROMPTS env var
# (semicolon-separated) for dataset-specific probes.
_DEFAULT_SAMPLE_PROMPTS = [
    "a cat sitting on a chair",
    "a young man with blonde spiky hair",
    "a house on a hill at sunset",
    "a person riding a bicycle",
    "a bowl of fruit on a table",
    "a robot walking in a city",
]
_env_sp = os.environ.get("SAMPLE_PROMPTS", "")
_FIXED_SAMPLE_PROMPTS = (
    [("sample", p.strip()) for p in _env_sp.split(";") if p.strip()]
    if _env_sp else
    [("sample", p) for p in _DEFAULT_SAMPLE_PROMPTS]
)


@torch.no_grad()
def save_periodic_samples(base, processor, out_root, global_step, n_prompts):
    """Generate a few fixed-prompt samples and save them BOTH as PNGs on disk
    AND as wandb.Image entries in the live W&B run. The on-disk dump gives us
    a deterministic visual trace per checkpoint; the W&B log gives us a
    scrollable in-browser view from anywhere without rsync. Lightweight:
    ~10 seconds per call on H100 for n=4."""
    out_dir = out_root / "samples" / f"step_{global_step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base.eval()
    # Lazy-import wandb so the function works even if W&B isn't initialised
    # (e.g. unit tests). The actual training run always inits it in main().
    try:
        import wandb as _wandb
        wandb_active = _wandb.run is not None
    except Exception:
        _wandb = None
        wandb_active = False
    wandb_images = []
    for k, (tag, prompt) in enumerate(_FIXED_SAMPLE_PROMPTS[:n_prompts]):
        torch.manual_seed(SEED + k)
        # Apply CAPTION_PREFIX (if any) so the samples reflect the training
        # regime: prefixed for Naruto-style runs, plain for prefix-free runs.
        gen_prompt = _apply_prefix(prompt)
        img = generate_image(base, processor, gen_prompt)
        img.save(out_dir / f"{k:02d}_{tag}.png")
        if wandb_active:
            wandb_images.append(_wandb.Image(img, caption=f"{k:02d}_{tag} :: {gen_prompt}"))
    if wandb_active and wandb_images:
        # Logged as a Media panel; the W&B UI lets you scrub across steps
        _wandb.log({"samples": wandb_images, "step": global_step})
    base.train()


@torch.no_grad()
def eval_loss(base, processor, eval_ids, get_image, refs, n):
    """Held-out validation CE on n eval pairs (teacher-forced, no generation).
    Cheap: ~30s for n=80 on H100 because we skip the autoregressive decoding
    step entirely. Logs to W&B as val/CE_loss every epoch -> a real curve we
    can use to spot overfitting (unlike CLIPScore which is OOD-unfaithful on
    stylised domains like Naruto)."""
    base.eval()
    losses = []
    for img_id in eval_ids[:n]:
        image = get_image(img_id)
        # Eval CE only: no KL or recon during validation (we want a pure
        # likelihood-style number to spot overfitting).
        loss, ce, _, _ = generation_loss(base, processor, image,
                                         _apply_prefix(refs[img_id][0]),
                                         recon_weight=0.0, kl_weight=0.0)
        losses.append(ce)
    base.train()
    return float(np.mean(losses))


@torch.no_grad()
def eval_clipscore(base, processor, scorer, eval_ids, id2file, refs, n):
    """Mid-eval CLIPScore probe. Uses the prefixed caption for generation (to
    match the training distribution) but scores against the RAW caption (so the
    numeric metric stays comparable to the baseline)."""
    base.eval()
    vals = []
    for k, img_id in enumerate(eval_ids[:n]):
        torch.manual_seed(SEED + k)
        raw = refs[img_id][0]
        img = generate_image(base, processor, _apply_prefix(raw))
        vals.append(scorer.score(img, raw))
    base.train()
    return float(np.mean(vals))


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
    print(f"  LoRA GENERATION fine-tune | run={RUN_NAME}")
    print(f"  TRAIN_N={TRAIN_N} EVAL_N={EVAL_N} GEN_EVAL_N={GEN_EVAL_N}")
    print(f"  EPOCHS={EPOCHS} GRAD_ACCUM={GRAD_ACCUM} LR={LR} weight_decay={WEIGHT_DECAY}")
    print(f"  rank={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
          f"warmup={WARMUP_RATIO} grad_clip={GRAD_CLIP}")
    print(f"  recon_loss_weight={RECON_LOSS_WEIGHT}  (0 = plain CE, Anole uses ~0.5)")
    print(f"  kl_lambda={KL_LAMBDA}  (0 = no reg; >0 enables the reg_type regularizer)")
    print(f"  reg_type={REG_TYPE}  (kl | mse | l2; kl/mse double the fwd cost, l2 is free)")
    print(f"  dataset_source={DATASET_SOURCE}  lr_schedule={LR_SCHEDULE}")
    print(f"  caption_prefix={CAPTION_PREFIX!r}")
    print(f"  adam_betas=({ADAM_BETA1}, {ADAM_BETA2})")
    print(f"  eval_every_steps={EVAL_EVERY_STEPS} eval_mid_n={EVAL_MID_N}")
    print(SEP + "\n")

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    lora_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                          target_modules=TARGET_MODULES,
                          lora_dropout=LORA_DROPOUT, bias="none",
                          task_type=TaskType.CAUSAL_LM)
    # Inject in-place: keep `base` for the custom forward, `peft_model` to save.
    peft_model = get_peft_model(base, lora_cfg)
    if INIT_ADAPTER:
        from _chain_init import init_from_adapter
        init_from_adapter(peft_model, INIT_ADAPTER,
                          expect_rank=LORA_RANK, expect_modules=TARGET_MODULES)
    base.train()
    trainable = [p for p in base.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in base.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    ratio_pct = 100.0 * trainable_params / total_params
    print(f"  target_modules : {TARGET_MODULES}")
    print(f"  trainable params: {trainable_params/1e6:.3f}M  /  total {total_params/1e6:.1f}M")
    print(f"  trainable ratio: {ratio_pct:.3f}%  "
          f"(1B@r=8 was ~0.44%; aim for >=0.4% on 7B too)\n", flush=True)

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

    # Pick the data source. `id2src` is either a filename mapping (COCO, image
    # loaded from disk per step) or a PIL.Image mapping (HF dataset, pre-loaded).
    if DATASET_SOURCE == "hf":
        train_ids, eval_ids, id2src, refs = load_hf_image_caption_split()
        def get_image(img_id):
            return _apply_image_transform(id2src[img_id].convert("RGB"))
    elif DATASET_SOURCE == "coco":
        train_ids, eval_ids, id2src, refs = load_split()
        def get_image(img_id):
            return _apply_image_transform(
                Image.open(VAL_IMAGES / id2src[img_id]).convert("RGB"))
    else:
        raise ValueError(f"unknown DATASET_SOURCE={DATASET_SOURCE!r}")
    print(f"  train images: {len(train_ids)} | eval images: {len(eval_ids)}\n")

    # Dump a few TRANSFORMED training targets so the pixelation strength can be
    # eyeballed before committing to the full run.
    if IMAGE_TRANSFORM:
        prev = OUTPUT_DIR / "transform_preview"
        prev.mkdir(parents=True, exist_ok=True)
        for k, iid in enumerate(train_ids[:4]):
            get_image(iid).save(prev / f"{k:02d}_{IMAGE_TRANSFORM}.png")
        print(f"  IMAGE_TRANSFORM={IMAGE_TRANSFORM} (size={PIXELART_SIZE}, "
              f"colors={PIXELART_COLORS}) -> previews in {prev}\n", flush=True)

    scorer = CLIPScorer()
    wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
        "model": MODEL_ID, "direction": "generation", "train_n": TRAIN_N,
        "epochs": EPOCHS, "grad_accum": GRAD_ACCUM, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
        "grad_clip": GRAD_CLIP, "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA, "seed": SEED,
        "schedule": f"linear_warmup_{LR_SCHEDULE}",
        "recon_loss_weight": RECON_LOSS_WEIGHT,
        "kl_lambda":         KL_LAMBDA,
        "reg_type":          REG_TYPE,
        "loss_type": "+".join(
            ["ce"]
            + (["recon"]   if RECON_LOSS_WEIGHT > 0 else [])
            + ([REG_TYPE]  if KL_LAMBDA         > 0 else [])
        ),
        "dataset_source": DATASET_SOURCE,
        "lr_schedule": LR_SCHEDULE,
    })

    # AdamW + warmup + (cosine | constant) schedule.
    # Total optimizer steps over the whole run determine the schedule length.
    total_steps  = (len(train_ids) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    print(f"  total optimizer steps: {total_steps}  warmup: {warmup_steps}\n", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=(ADAM_BETA1, ADAM_BETA2))
    if LR_SCHEDULE == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    elif LR_SCHEDULE == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        raise ValueError(f"unknown LR_SCHEDULE={LR_SCHEDULE!r}")

    # Try to resume from a 'latest/' checkpoint in this RUN's OUTPUT_DIR.
    latest_dir = OUTPUT_DIR / "latest"
    start_epoch, global_step, base_clip, history = (0, 0, None, None)
    if RESUME:
        start_epoch, global_step, base_clip, history = load_checkpoint_if_present(
            latest_dir, base, optimizer, scheduler)

    if base_clip is None:
        print("  measuring baseline CLIPScore (pre-training)...", flush=True)
        base_clip = eval_clipscore(base, processor, scorer, eval_ids, id2src, refs, GEN_EVAL_N)
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
        running, running_ce, running_recon, running_kl = 0.0, 0.0, 0.0, 0.0
        for i, img_id in enumerate(order):
            image = get_image(img_id)
            loss, ce_item, recon_item, kl_item = generation_loss(
                base, processor, image, _apply_prefix(refs[img_id][0]),
                recon_weight=RECON_LOSS_WEIGHT,
                kl_weight=KL_LAMBDA,
                reg_type=REG_TYPE,
                peft_model=peft_model)
            (loss / GRAD_ACCUM).backward()
            running       += loss.item()
            running_ce    += ce_item
            running_recon += recon_item
            running_kl    += kl_item

            if (i + 1) % GRAD_ACCUM == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                avg       = running       / GRAD_ACCUM
                avg_ce    = running_ce    / GRAD_ACCUM
                avg_recon = running_recon / GRAD_ACCUM
                avg_kl    = running_kl    / GRAD_ACCUM
                running, running_ce, running_recon, running_kl = 0.0, 0.0, 0.0, 0.0
                cur_lr = scheduler.get_last_lr()[0]
                wandb.log({"train/loss":       avg,
                           "train/loss_ce":    avg_ce,
                           "train/loss_recon": avg_recon,
                           "train/loss_kl":    avg_kl,
                           "train/lr":         cur_lr,
                           "train/grad_norm":  float(gnorm),
                           "step":             global_step})
                if global_step % 20 == 0:
                    extras = []
                    if RECON_LOSS_WEIGHT > 0: extras.append(f"recon={avg_recon:.4f}")
                    if KL_LAMBDA > 0:        extras.append(f"kl={avg_kl:.4f}")
                    extra_str = (" " + " ".join(extras)) if extras else ""
                    print(f"  epoch {epoch} step {global_step} "
                          f"({i+1}/{len(order)}) loss={avg:.4f} ce={avg_ce:.4f}"
                          f"{extra_str} lr={cur_lr:.2e}", flush=True)

                # Mid-epoch eval: cheap CLIPScore probe to see direction live.
                if EVAL_EVERY_STEPS > 0 and global_step % EVAL_EVERY_STEPS == 0:
                    mid_clip = eval_clipscore(base, processor, scorer,
                                              eval_ids, id2src, refs, EVAL_MID_N)
                    print(f"    [mid-eval @ step {global_step}] "
                          f"CLIPScore({EVAL_MID_N})={mid_clip:.2f}", flush=True)
                    wandb.log({"eval/CLIPScore_mid": mid_clip, "step": global_step})

                # Periodic full checkpoint (optional, for very long runs).
                if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                    save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                                    epoch, global_step, base_clip, history)
                    print(f"    saved checkpoint @ step {global_step}", flush=True)

                # Periodic image samples on fixed prompts (visual trace).
                if SAMPLE_EVERY_STEPS > 0 and global_step % SAMPLE_EVERY_STEPS == 0:
                    save_periodic_samples(base, processor, OUTPUT_DIR,
                                          global_step, N_SAMPLE_PROMPTS)
                    print(f"    saved samples @ step {global_step}", flush=True)

        clip = eval_clipscore(base, processor, scorer, eval_ids, id2src, refs, GEN_EVAL_N)
        val_ce = eval_loss(base, processor, eval_ids, get_image, refs, EVAL_N)
        print(f"  [epoch {epoch}] CLIPScore={clip:.2f}  val_CE={val_ce:.4f}",
              flush=True)
        wandb.log({"eval/CLIPScore": clip,
                   "val/CE_loss":   val_ce,
                   "epoch":         epoch + 1})
        history["epochs"].append({"epoch": epoch + 1,
                                  "CLIPScore": clip,
                                  "val_CE": val_ce})

        # Two checkpoints per epoch: a permanent per-epoch adapter (for later
        # forgetting eval / matrix rows), and a 'latest' full state for resume.
        per_epoch = OUTPUT_DIR / f"adapter_epoch{epoch+1}"
        peft_model.save_pretrained(str(per_epoch))
        print(f"  saved adapter -> {per_epoch}", flush=True)
        save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                        epoch + 1, global_step, base_clip, history)
        print(f"  saved resume state -> {latest_dir}", flush=True)

    history["minutes"] = round((time.time() - t_start) / 60, 1)

    # Pick the checkpoint with the LOWEST held-out val_CE as the canonical
    # adapter for downstream eval (a 'adapter_best' symlink). For a rank
    # sweep, different ranks may bottom out at different epochs, so always
    # evaluating the last epoch would unfairly penalise ranks that converge
    # (and start over-fitting) earlier. We record which epoch won.
    best_epoch = None
    if history["epochs"]:
        best = min(history["epochs"], key=lambda e: e["val_CE"])
        best_epoch = best["epoch"]
        history["best_epoch"] = best_epoch
        history["best_val_CE"] = best["val_CE"]
        best_src = OUTPUT_DIR / f"adapter_epoch{best_epoch}"
        best_link = OUTPUT_DIR / "adapter_best"
        try:
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            # Relative target so the link survives if the dir is moved
            best_link.symlink_to(best_src.name)
        except OSError:
            # Filesystems without symlink support: write a pointer file instead
            (OUTPUT_DIR / "adapter_best.txt").write_text(
                best_src.name, encoding="utf-8")

    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2),
                                             encoding="utf-8")
    wandb.finish()

    print("\n" + SEP)
    print("  DONE")
    print(f"  baseline CLIPScore : {base_clip:.2f}")
    if history["epochs"]:
        print(f"  final CLIPScore    : {history['epochs'][-1]['CLIPScore']:.2f}")
        print(f"  best epoch (val_CE): epoch {best_epoch} "
              f"(val_CE={history['best_val_CE']:.4f}) -> adapter_best")
    print(f"  adapters + history in {OUTPUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
