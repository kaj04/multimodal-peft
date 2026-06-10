#!/usr/bin/env python3
"""
janus/eval/08_qualitative_text_examples.py

Extract QUALITATIVE per-example outputs for the cross-modal forgetting
write-up -- the text-side analogue of the side-by-side image figure.
Run once per condition (base / v5 / joint_matched / joint_far / far_small);
each run writes a JSON with this model's answers on a FIXED, seed-stable set
of probes, so 08b_aggregate can line the conditions up column-by-column.

Three probe families, mapped to the two Overleaf tables:

  TABLE A (knowledge + fluency, pure text):
    - MMLU   : fixed questions, the option each model picks (argmax logprob)
    - FREEGEN: greedy continuation of fixed neutral prompts. The matched
               model (PPL 47.9) is expected to degrade/repeat here -> the
               most legible demonstration of text forgetting.

  TABLE B (understanding, image+text junction):
    - POPE/MME: fixed image+question examples, the yes/no each model picks.
               The example images are copied to images/ for the figure.

Greedy decoding is done MANUALLY (forward + argmax loop) rather than via
model.generate(), because model(input_ids).logits is already validated
(used in the forgetting battery) whereas Janus' generate() defaults to
image-generation mode. Deterministic, no sampling, no API surprises.
"""

import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
TAG      = os.environ.get("TAG", "")
SEED     = int(os.environ.get("SEED", "42"))

N_MMLU   = int(os.environ.get("N_MMLU", "6"))
N_POPE   = int(os.environ.get("N_POPE", "4"))
N_MME    = int(os.environ.get("N_MME", "4"))
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "80"))

OUTPUT_ROOT = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "qualitative_text"

SEP = "=" * 64

# Fixed neutral free-generation prompts. Deliberately general (no anime
# content) so that any drift toward the over-fitted matched corpus, or any
# fluency collapse, is obvious by contrast with the base model.
FREEGEN_PROMPTS = [
    "The most important scientific discovery of the twentieth century was",
    "In summary, the main causes of the French Revolution were",
    "A healthy balanced diet should include",
    "The key difference between a virus and a bacterium is that",
    "To explain how a computer works to a child, you could say",
    "The history of the Roman Empire begins with",
]


def load_model():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    base.eval()
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ADAPTER)
        model.eval()
        print(f"  loaded adapter: {ADAPTER}")
        return model, processor
    print("  no adapter -> base model")
    return base, processor


@torch.no_grad()
def text_logits(model, input_ids):
    out = model(input_ids=input_ids.to(model.device))
    return out.logits if hasattr(out, "logits") else out[0]


def first_token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0]


@torch.no_grad()
def greedy_generate(model, tok, prompt, max_new_tokens):
    """Manual greedy decode. Returns the continuation string only."""
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    start_len = ids.shape[1]
    for _ in range(max_new_tokens):
        logits = text_logits(model, ids)[0, -1]
        nxt = int(torch.argmax(logits).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
    cont_ids = ids[0, start_len:].tolist()
    return tok.decode(cont_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Probe families
# ---------------------------------------------------------------------------

MMLU_TEMPLATE = """The following is a multiple choice question about {subject}.

{question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer:"""


def probe_mmlu(model, processor):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    # Seed-stable selection across distinct subjects for variety
    rng = random.Random(SEED)
    by_subj = {}
    for i, row in enumerate(ds):
        by_subj.setdefault(row["subject"], []).append(i)
    subjects = sorted(by_subj)
    rng.shuffle(subjects)
    chosen = [by_subj[s][0] for s in subjects[:N_MMLU]]
    tok = processor.tokenizer
    letters = ["A", "B", "C", "D"]
    out = []
    for i in chosen:
        row = ds[i]
        prompt = MMLU_TEMPLATE.format(
            subject=row["subject"].replace("_", " "), question=row["question"],
            a=row["choices"][0], b=row["choices"][1],
            c=row["choices"][2], d=row["choices"][3])
        lps = [text_logits(model, tok(prompt, return_tensors="pt")["input_ids"])
               [0, -1].float()[first_token_id(tok, " " + L)].item()
               for L in letters]
        pred = int(torch.tensor(lps).argmax().item())
        out.append({
            "subject": row["subject"], "question": row["question"],
            "choices": row["choices"], "gold": int(row["answer"]),
            "pred": pred, "pred_letter": letters[pred],
            "correct": bool(pred == row["answer"]),
        })
        print(f"    [MMLU] {row['subject'][:20]:20s} pred={letters[pred]} "
              f"gold={letters[row['answer']]}", flush=True)
    return out


def probe_freegen(model, processor):
    tok = processor.tokenizer
    out = []
    for p in FREEGEN_PROMPTS:
        cont = greedy_generate(model, tok, p, GEN_TOKENS)
        out.append({"prompt": p, "continuation": cont})
        print(f"    [GEN] {p[:40]}... -> {cont[:60]}...", flush=True)
    return out


@torch.no_grad()
def vqa_yesno(model, processor, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image.convert("RGB")],
                       return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    out = model(**inputs)
    logits = (out.logits if hasattr(out, "logits") else out[0])[0, -1].float()
    tok = processor.tokenizer
    lp_yes = max(logits[first_token_id(tok, t)].item() for t in (" Yes", "Yes"))
    lp_no  = max(logits[first_token_id(tok, t)].item() for t in (" No", "No"))
    return "yes" if lp_yes > lp_no else "no"


def probe_vqa(model, processor, hf_name, n, img_out_dir, prefix):
    from datasets import load_dataset
    ds = load_dataset(hf_name, split="test")
    rng = random.Random(SEED)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    out = []
    for k, i in enumerate(idxs):
        row = ds[i]
        pred = vqa_yesno(model, processor, row["image"], row["question"])
        # Save the example image once (only on base run to avoid clobbering;
        # but harmless if overwritten -- the image is fixed per index/seed).
        img_path = img_out_dir / f"{prefix}_{k:02d}.png"
        if not img_path.exists():
            row["image"].convert("RGB").save(img_path)
        out.append({
            "question": row["question"], "gold": row["answer"].strip().lower(),
            "pred": pred, "correct": pred == row["answer"].strip().lower(),
            "category": row.get("category", ""),
            "image_file": img_path.name,
        })
        print(f"    [{prefix}] {row['question'][:45]:45s} pred={pred} "
              f"gold={row['answer']}", flush=True)
    return out


def main():
    tag = TAG or (Path(ADAPTER).parent.name if ADAPTER else "base_no_adapter")
    out_dir = OUTPUT_ROOT / tag
    img_dir = OUTPUT_ROOT / "_shared_images"   # images are identical across runs
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print(f"  Qualitative text examples | tag={tag}")
    print(f"  ADAPTER: {ADAPTER or '<base>'}")
    print(SEP + "\n")

    model, processor = load_model()
    results = {"model": MODEL_ID, "adapter": ADAPTER, "tag": tag}

    print("  -- MMLU --", flush=True)
    results["mmlu"] = probe_mmlu(model, processor)
    print("  -- FREEGEN --", flush=True)
    results["freegen"] = probe_freegen(model, processor)
    print("  -- POPE --", flush=True)
    results["pope"] = probe_vqa(model, processor, "lmms-lab/POPE",
                                N_POPE, img_dir, "pope")
    print("  -- MME --", flush=True)
    results["mme"] = probe_vqa(model, processor, "lmms-lab/MME",
                               N_MME, img_dir, "mme")

    out_json = out_dir / "qualitative_text.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  wrote {out_json}")
    print(f"  shared images in {img_dir}")


if __name__ == "__main__":
    main()
