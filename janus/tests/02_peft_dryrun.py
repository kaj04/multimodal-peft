#!/usr/bin/env python3
"""
janus/tests/02_peft_dryrun.py

Phase 2 PEFT dry-run for Janus-Pro-1B.

Goal: prove that LoRA can be injected and that a single training step
(forward -> loss -> backward -> optimizer.step) runs for EVERY training
direction the model supports, with gradients actually reaching the LoRA
parameters. This de-risks the whole project: it is exactly the step that was
impossible on Anole (weights locked in an inference subprocess).

It does NOT learn anything useful; it validates the training plumbing and maps
out which capabilities are trainable, so later experiments build on known-good
foundations.

Directions tested (each isolated; one failure never hides the others):
  A. understanding  (image + text -> text)  -- native forward(labels=...)
  B. text_only      (text -> text)           -- native forward(labels=...)
  C. generation     (text -> image)          -- CUSTOM loss:
        target image --vqmodel.encode--> image tokens (the labels)
        [text embeds | image embeds] --transformer--> hidden states
        hidden states --generation_head--> logits over the image codebook
        cross-entropy(logits, image tokens)

A JSON summary is written to output/dryrun/results.json for the record.
Run via 02_run_peft_dryrun.sh (needs a GPU).
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID   = "deepseek-community/Janus-Pro-1B"
COCO_DIR   = Path("/scratch-shared/tesr125338/coco/val2017")
OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "dryrun"

LORA_RANK           = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE       = 1e-4

SEP = "=" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_grad_norm(model):
    """L2 norm of gradients over trainable (LoRA) params after backward."""
    sq, n = 0.0, 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
            n += 1
    return sq ** 0.5, n


def first_coco_image():
    imgs = sorted(COCO_DIR.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"no COCO images under {COCO_DIR}")
    return Image.open(imgs[0]).convert("RGB")


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------

def build_model_and_lora():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    print(SEP)
    print(f"  Loading {MODEL_ID} and injecting LoRA")
    print(SEP)

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    model = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # get_peft_model injects LoRA in-place into `model` and freezes the base.
    # We keep using `model` directly: the LoRA layers live inside it, so every
    # forward call (native or custom) routes through them.
    get_peft_model(model, lora_config)
    model.train()

    trainable, total = count_params(model)
    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    print(f"  Device          : {device}, dtype {next(model.parameters()).dtype}")
    print(f"  Trainable (LoRA): {trainable/1e6:.3f}M / {total/1e9:.3f}B "
          f"({100*trainable/total:.3f}%)")
    print(f"  LoRA tensors    : {n_lora}\n")
    return model, processor, device, {
        "trainable_params_M": round(trainable / 1e6, 3),
        "total_params_M": round(total / 1e6, 1),
        "trainable_pct": round(100 * trainable / total, 3),
        "lora_tensors": n_lora,
    }


def print_model_overview(model):
    """Map out the submodules later experiments will need."""
    print(SEP)
    print("  Model overview (key submodules / capabilities)")
    print(SEP)
    inner = getattr(model, "model", None)
    checks = {
        "model.model (JanusModel)":          inner is not None,
        "model.model.vqmodel":               hasattr(inner, "vqmodel"),
        "model.model.generation_embeddings": hasattr(inner, "generation_embeddings"),
        "model.model.generation_aligner":    hasattr(inner, "generation_aligner"),
        "model.model.generation_head":       hasattr(inner, "generation_head"),
        "model.lm_head":                     hasattr(model, "lm_head"),
        "prepare_embeddings_for_image_generation":
            hasattr(model, "prepare_embeddings_for_image_generation"),
        "get_input_embeddings":              hasattr(model, "get_input_embeddings"),
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'MISSING'}] {name}")
    if inner is not None:
        print(f"  model.model children: {[n for n, _ in inner.named_children()]}")
        if hasattr(inner, "vqmodel"):
            cb = getattr(inner.vqmodel.config, "num_embeddings", "?")
            print(f"  image codebook size (num_embeddings): {cb}")
    print()


# ---------------------------------------------------------------------------
# Direction A: understanding  (image + text -> text)  [native]
# ---------------------------------------------------------------------------

def step_understanding(model, processor, device, image, optimizer):
    user_turn = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}],
    }]
    full_convo = user_turn + [{
        "role": "assistant",
        "content": [{"type": "text", "text":
                     "A dummy caption used to validate the LoRA training step."}],
    }]
    prompt_str = processor.apply_chat_template(
        user_turn, add_generation_prompt=True, generation_mode="text")
    full_str = processor.apply_chat_template(full_convo, generation_mode="text")

    prompt_inputs = processor(text=prompt_str, images=[image], return_tensors="pt")
    full_inputs   = processor(text=full_str,   images=[image], return_tensors="pt")

    input_ids  = full_inputs["input_ids"]
    labels = input_ids.clone()
    labels[:, :prompt_inputs["input_ids"].shape[1]] = -100   # mask image + prompt
    img_id = getattr(model.config, "image_token_id", None)
    if img_id is not None:
        labels[input_ids == img_id] = -100
    n_sup = int((labels != -100).sum())
    if n_sup == 0:
        raise RuntimeError("no supervised tokens -- label masking is wrong")

    batch = {
        "input_ids":      input_ids.to(device),
        "attention_mask": full_inputs["attention_mask"].to(device),
        "pixel_values":   full_inputs["pixel_values"].to(device, next(model.parameters()).dtype),
        "labels":         labels.to(device),
    }
    optimizer.zero_grad()
    out = model(**batch)
    if not out.loss.requires_grad:
        raise RuntimeError("loss has no grad_fn")
    out.loss.backward()
    gnorm, n = lora_grad_norm(model)
    optimizer.step()
    return {"loss": float(out.loss.item()), "grad_norm": gnorm,
            "params_with_grad": n, "supervised_tokens": n_sup}


# ---------------------------------------------------------------------------
# Direction B: text only  (text -> text)  [native]
# ---------------------------------------------------------------------------

def step_text_only(model, processor, device, optimizer):
    text = "Parameter-efficient fine-tuning adapts only a small set of weights."
    inputs = processor(text=text, generation_mode="text", return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()

    optimizer.zero_grad()
    out = model(input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=labels)
    if not out.loss.requires_grad:
        raise RuntimeError("loss has no grad_fn")
    out.loss.backward()
    gnorm, n = lora_grad_norm(model)
    optimizer.step()
    return {"loss": float(out.loss.item()), "grad_norm": gnorm, "params_with_grad": n}


# ---------------------------------------------------------------------------
# Direction C: generation  (text -> image)  [custom loss]
# ---------------------------------------------------------------------------

def step_generation(model, processor, device, image, optimizer):
    inner = model.model
    dtype = next(model.parameters()).dtype

    # 1. Text prompt (image-generation mode)
    messages = [{"role": "user", "content": [{"type": "text",
                 "text": "a dummy prompt to exercise the generation training step"}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, generation_mode="image")
    text_ids = processor(text=prompt, generation_mode="image",
                         return_tensors="pt").to(device)["input_ids"]

    # 2. Target image -> discrete VQ tokens (the labels)
    pixel_values = processor.image_processor(
        images=[image], return_tensors="pt")["pixel_values"].to(device, dtype)
    image_tokens = inner.vqmodel.encode(pixel_values).image_tokens.reshape(1, -1)
    n_img = image_tokens.shape[1]

    # 3. Input embeddings: [text | image[:-1]] (teacher forcing)
    text_embeds  = model.get_input_embeddings()(text_ids)                       # (1, Lt, D)
    image_embeds = model.prepare_embeddings_for_image_generation(image_tokens)  # (1, n_img, D)
    inputs_embeds = torch.cat([text_embeds, image_embeds[:, :-1, :]], dim=1)

    # 4. Transformer forward -> hidden states
    hidden = inner(inputs_embeds=inputs_embeds).last_hidden_state               # (1, Lt+n_img-1, D)

    # 5. Hidden states that predict each image token, then the generation head
    Lt = text_embeds.shape[1]
    img_hidden = hidden[:, Lt - 1:, :]                                          # (1, n_img, D)
    logits = inner.generation_head(img_hidden)                                  # (1, n_img, codebook)

    # 6. Cross-entropy over the image codebook
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        image_tokens.reshape(-1).long(),
    )
    if not loss.requires_grad:
        raise RuntimeError("loss has no grad_fn")

    optimizer.zero_grad()
    loss.backward()
    gnorm, n = lora_grad_norm(model)
    optimizer.step()
    return {"loss": float(loss.item()), "grad_norm": gnorm,
            "params_with_grad": n, "n_image_tokens": int(n_img)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_step(name, fn):
    """Run one direction in isolation; never let a failure stop the others."""
    print(SEP)
    print(f"  Direction: {name}")
    print(SEP)
    t0 = time.time()
    try:
        info = fn()
        info["status"] = "PASS"
        info["seconds"] = round(time.time() - t0, 1)
        info["gradients_flow"] = info["grad_norm"] > 0.0
        flow = "gradients flow" if info["gradients_flow"] else "NO gradient flow!"
        print(f"  loss={info['loss']:.4f}  grad_norm={info['grad_norm']:.4f}  "
              f"params_with_grad={info['params_with_grad']}  ({flow})")
        print(f"  -> PASS ({info['seconds']}s)\n")
        return info
    except Exception as e:
        print(f"  -> FAIL: {e}")
        traceback.print_exc()
        print()
        return {"status": "FAIL", "error": str(e),
                "seconds": round(time.time() - t0, 1)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print("  Janus-Pro-1B PEFT dry-run")
    print(f"  torch {torch.__version__}  CUDA {torch.cuda.is_available()}")
    print(f"  output: {OUTPUT_DIR}")
    print(SEP + "\n")

    model, processor, device, lora_info = build_model_and_lora()
    print_model_overview(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    image = first_coco_image()

    results = {
        "model": MODEL_ID,
        "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA,
                 "targets": LORA_TARGET_MODULES, **lora_info},
        "directions": {},
    }
    results["directions"]["understanding"] = run_step(
        "understanding (image+text -> text)",
        lambda: step_understanding(model, processor, device, image, optimizer))
    results["directions"]["text_only"] = run_step(
        "text_only (text -> text)",
        lambda: step_text_only(model, processor, device, optimizer))
    results["directions"]["generation"] = run_step(
        "generation (text -> image) [custom loss]",
        lambda: step_generation(model, processor, device, image, optimizer))

    print(SEP)
    print("  SUMMARY")
    print(SEP)
    for name, info in results["directions"].items():
        if info["status"] == "PASS":
            extra = f"loss={info['loss']:.3f} grad_norm={info['grad_norm']:.3f}"
        else:
            extra = info.get("error", "")
        print(f"  {name:16s}: {info['status']:4s}  {extra}")
    print(SEP)

    out_json = OUTPUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results written to {out_json}")

    all_pass = all(d["status"] == "PASS" for d in results["directions"].values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
