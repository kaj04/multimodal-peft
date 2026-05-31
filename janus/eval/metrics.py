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
