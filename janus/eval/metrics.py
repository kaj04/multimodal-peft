#!/usr/bin/env python3
"""
janus/eval/metrics.py

Reusable metric helpers for the Janus PEFT experiments.

Captioning (understanding, image -> text):
    CIDEr and BLEU-4 via pycocoevalcap scorers. We tokenize with a simple
    lowercase/alphanumeric splitter (NOT the Java PTBTokenizer) so the pipeline
    runs without a Java dependency. Numbers are internally consistent and good
    for before/after comparison; for benchmark-comparable absolute numbers,
    swap in PTBTokenizer later.

Generation (text -> image):
    CLIPScore = 100 * cosine(image_embed, text_embed) using a CLIP model.
    Measures how well a generated image matches its prompt.
"""

import re

import torch

_PUNCT = re.compile(r"[^a-z0-9 ]")


# ---------------------------------------------------------------------------
# Generated-image decoding (single source of truth, used everywhere)
# ---------------------------------------------------------------------------

def decoded_to_pil(decoded, processor=None):
    """Convert Janus `decode_image_tokens` output to a PIL image.

    Handles: detach/cpu, optional processor.postprocess (np), CHW/HWC layout,
    and value ranges [-1,1] / [0,1] / [0,255] in a SINGLE clean path. Keep all
    image-saving code routed through here so the denormalization logic never
    diverges (a duplicated, buggy version produced all-black images before).
    """
    import numpy as np
    from PIL import Image

    decoded = decoded.detach().float().cpu()
    arr = None
    if processor is not None:
        try:
            arr = np.asarray(
                processor.postprocess(list(decoded), return_tensors="np")["pixel_values"][0])
        except Exception:
            arr = None
    if arr is None:
        arr = decoded[0].numpy()

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))   # CHW -> HWC

    amin, amax = float(arr.min()), float(arr.max())
    if amin < -0.01:        # [-1, 1] -> [0, 1]
        arr = (arr + 1.0) / 2.0
    elif amax > 1.01:       # [0, 255] -> [0, 1]
        arr = arr / 255.0
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Generation helpers (single source of truth for inference in eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_caption(model, processor, image, max_new_tokens=64):
    """image -> one-sentence caption (understanding)."""
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


@torch.no_grad()
def generate_image(model, processor, prompt_text):
    """text -> PIL image (generation)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, generation_mode="image")
    inputs = processor(text=prompt, generation_mode="image",
                       return_tensors="pt").to(model.device)
    model.generation_config.num_return_sequences = 1
    outputs = model.generate(**inputs, generation_mode="image",
                             max_new_tokens=576, do_sample=True, use_cache=True)
    decoded = model.decode_image_tokens(outputs)
    return decoded_to_pil(decoded, processor)


def simple_tokenize(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Captioning metrics
# ---------------------------------------------------------------------------

def captioning_scores(references: dict, hypotheses: dict) -> dict:
    """
    references: {image_id: [ref caption, ...]}
    hypotheses: {image_id: "generated caption"}
    Returns {"CIDEr": float, "BLEU_4": float, "n": int}.
    """
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu

    ids = [i for i in hypotheses if i in references]
    gts = {i: [simple_tokenize(r) for r in references[i]] for i in ids}
    res = {i: [simple_tokenize(hypotheses[i])] for i in ids}

    cider_score, _ = Cider().compute_score(gts, res)
    bleu_scores, _ = Bleu(4).compute_score(gts, res)  # [B1, B2, B3, B4]
    return {"CIDEr": float(cider_score),
            "BLEU_4": float(bleu_scores[3]),
            "n": len(ids)}


# ---------------------------------------------------------------------------
# Generation metric (CLIPScore)
# ---------------------------------------------------------------------------

class CLIPScorer:
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, image, text) -> float:
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
            padding=True, truncation=True,
        ).to(self.device)
        out = self.model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        cos = float((img * txt).sum(-1).item())
        return 100.0 * cos
