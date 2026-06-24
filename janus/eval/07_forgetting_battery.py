#!/usr/bin/env python3
"""
janus/eval/07_forgetting_battery.py

Cross-modal forgetting battery for Janus-Pro-7B adapters. Measures whether
fine-tuning the IMAGE-GENERATION direction (optionally jointly with a text
stream) damaged the model's other capability directions. All benchmarks are
official, public, and scored WITHOUT free-form generation (log-likelihood
only) -> deterministic, no LLM judge, directly comparable across conditions.

Four capability probes:
  1. PPL      : perplexity on WikiText-2 test (512-token windows)   [text LM]
  2. MMLU     : 4-way multiple choice, log-likelihood scoring        [text knowledge]
  3. POPE     : object-hallucination yes/no on COCO images           [understanding]
  4. MME      : perception+cognition yes/no                          [understanding]

POPE and MME are reported in the Janus-Pro paper itself, so the base-model
(no adapter) numbers double as a harness sanity check against published
values.

Conditions to compare (run this script once per condition):
  ADAPTER=""                                   -> base model
  ADAPTER=.../gen_7b_naruto_v5/adapter_epoch6  -> image-only LoRA
  ADAPTER=.../joint_7b_v5_matched/adapter_epoch6
  ADAPTER=.../joint_7b_v5_far/adapter_epoch6

Env knobs:
  SMOKE=1        -> run 5 examples per benchmark, verbose, for plumbing check
  MMLU_N=1425    -> stratified subsample size (25 per subject x 57)
  POPE_N=3000    -> subsample across the 3 POPE splits
  MME_N=0        -> 0 = full (~2.4k)
  SKIP=ppl,mmlu  -> comma-separated benchmarks to skip
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

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")          # empty -> base model
TAG      = os.environ.get("TAG", "")              # output folder name; default derived
SEED     = int(os.environ.get("SEED", "42"))

SMOKE    = os.environ.get("SMOKE", "0") == "1"
MMLU_N   = int(os.environ.get("MMLU_N", "1425"))
POPE_N   = int(os.environ.get("POPE_N", "3000"))
MME_N    = int(os.environ.get("MME_N", "0"))      # 0 = full
SKIP     = set(s.strip() for s in os.environ.get("SKIP", "").split(",") if s.strip())

PPL_CHUNK = 512

OUTPUT_ROOT = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "forgetting_battery"

SEP = "=" * 64

if SMOKE:
    MMLU_N, POPE_N, MME_N = 5, 5, 5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def text_logits(model, input_ids):
    """Forward on text-only input_ids (1, L). Returns logits (1, L, V)."""
    out = model(input_ids=input_ids.to(model.device))
    return out.logits if hasattr(out, "logits") else out[0]


def first_token_id(tokenizer, text):
    """Token id of the first token of `text` (no special tokens). Used for
    single-token answer scoring (' Yes', ' A', ...)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0]


@torch.no_grad()
def choice_logprobs(model, tokenizer, prompt, choices):
    """Log-prob of each single-token choice continuation after `prompt`.
    choices: list of strings whose FIRST TOKEN distinguishes them
    (e.g. [' A', ' B', ' C', ' D'] or [' Yes', ' No'])."""
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    logits = text_logits(model, ids)[0, -1].float()      # (V,)
    logp = F.log_softmax(logits, dim=-1)
    return [logp[first_token_id(tokenizer, c)].item() for c in choices]


@torch.no_grad()
def vqa_yesno_logprobs(model, processor, image, question):
    """Understanding path: image + question -> logprob(' Yes'), logprob(' No')
    at the first answer position. Uses the Janus chat template with an image
    slot. Returns (lp_yes, lp_no)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image.convert("RGB")],
                       return_tensors="pt").to(model.device)
    # bf16 for pixel values if the processor returned float32
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    out = model(**inputs)
    logits = (out.logits if hasattr(out, "logits") else out[0])[0, -1].float()
    logp = F.log_softmax(logits, dim=-1)
    tok = processor.tokenizer
    lp_yes = max(logp[first_token_id(tok, " Yes")].item(),
                 logp[first_token_id(tok, "Yes")].item(),
                 logp[first_token_id(tok, " yes")].item())
    lp_no  = max(logp[first_token_id(tok, " No")].item(),
                 logp[first_token_id(tok, "No")].item(),
                 logp[first_token_id(tok, " no")].item())
    return lp_yes, lp_no


# ---------------------------------------------------------------------------
# 1. Perplexity on WikiText-2 test
# ---------------------------------------------------------------------------

def bench_ppl(model, processor):
    from datasets import load_dataset
    print("  [PPL] loading wikitext-2-raw-v1 test ...", flush=True)
    # Namespaced repo id required by newer huggingface_hub ('wikitext' alone
    # raises HfUriError: Repository id must be 'namespace/name')
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tok = processor.tokenizer
    ids = tok.encode(text, add_special_tokens=False)
    n_chunks = len(ids) // PPL_CHUNK
    if SMOKE:
        n_chunks = min(n_chunks, 5)
    print(f"  [PPL] {len(ids)} tokens -> {n_chunks} chunks of {PPL_CHUNK}", flush=True)

    losses = []
    t0 = time.time()
    for i in range(n_chunks):
        chunk = torch.tensor(ids[i * PPL_CHUNK:(i + 1) * PPL_CHUNK],
                             dtype=torch.long).unsqueeze(0)
        logits = text_logits(model, chunk)
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = chunk[..., 1:].contiguous().to(logits.device)
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
        losses.append(loss.item())
        if (i + 1) % 50 == 0:
            print(f"    chunk {i+1}/{n_chunks}  running PPL="
                  f"{np.exp(np.mean(losses)):.2f}", flush=True)
    mean_ce = float(np.mean(losses))
    ce_std = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
    ppl = float(np.exp(mean_ce))
    print(f"  [PPL] DONE  mean_CE={mean_ce:.4f}  PPL={ppl:.2f}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    # ce_std + per-chunk CE let us bootstrap a CI on the perplexity later
    # (PPL = exp(mean_CE); CI = exp(mean_CE +/- 1.96 * ce_std / sqrt(n_chunks))).
    return {"mean_CE": mean_ce, "ce_std": ce_std, "PPL": ppl,
            "n_chunks": n_chunks, "chunk_ce": [round(float(x), 5) for x in losses]}


# ---------------------------------------------------------------------------
# 2. MMLU (log-likelihood multiple choice, 0-shot)
# ---------------------------------------------------------------------------

MMLU_TEMPLATE = """The following is a multiple choice question about {subject}.

{question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer:"""


def bench_mmlu(model, processor):
    from datasets import load_dataset
    print("  [MMLU] loading cais/mmlu (all, test) ...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(SEED)

    # Stratified subsample: equal per subject
    by_subject = defaultdict(list)
    for i, row in enumerate(ds):
        by_subject[row["subject"]].append(i)
    per_subject = max(1, MMLU_N // max(1, len(by_subject)))
    chosen = []
    for subj, idxs in by_subject.items():
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_subject])
    rng.shuffle(chosen)
    if SMOKE:
        chosen = chosen[:5]
    print(f"  [MMLU] {len(chosen)} questions "
          f"({per_subject}/subject x {len(by_subject)} subjects)", flush=True)

    tok = processor.tokenizer
    correct, total = 0, 0
    per_subject_acc = defaultdict(lambda: [0, 0])
    t0 = time.time()
    for k, i in enumerate(chosen):
        row = ds[i]
        prompt = MMLU_TEMPLATE.format(
            subject=row["subject"].replace("_", " "),
            question=row["question"],
            a=row["choices"][0], b=row["choices"][1],
            c=row["choices"][2], d=row["choices"][3])
        lps = choice_logprobs(model, tok, prompt, [" A", " B", " C", " D"])
        pred = int(np.argmax(lps))
        ok = int(pred == row["answer"])
        correct += ok
        total += 1
        per_subject_acc[row["subject"]][0] += ok
        per_subject_acc[row["subject"]][1] += 1
        if SMOKE:
            print(f"    Q: {row['question'][:80]}...")
            print(f"    logprobs={['%.2f' % x for x in lps]} pred={pred} "
                  f"gold={row['answer']} {'OK' if ok else 'WRONG'}")
        elif (k + 1) % 200 == 0:
            print(f"    {k+1}/{len(chosen)}  running acc={correct/total:.4f}",
                  flush=True)
    acc = correct / max(1, total)
    print(f"  [MMLU] DONE  acc={acc:.4f}  ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    return {"accuracy": acc, "n": total,
            "per_subject": {s: c / max(1, n)
                            for s, (c, n) in sorted(per_subject_acc.items())}}


# ---------------------------------------------------------------------------
# 3. POPE (object hallucination, yes/no)
# ---------------------------------------------------------------------------

def bench_pope(model, processor):
    from datasets import load_dataset
    print("  [POPE] loading lmms-lab/POPE ...", flush=True)
    ds = load_dataset("lmms-lab/POPE", split="test")
    rng = random.Random(SEED)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    if POPE_N > 0:
        idxs = idxs[:POPE_N]
    print(f"  [POPE] {len(idxs)} questions", flush=True)

    tp = fp = tn = fn = 0
    t0 = time.time()
    for k, i in enumerate(idxs):
        row = ds[i]
        lp_yes, lp_no = vqa_yesno_logprobs(model, processor,
                                           row["image"], row["question"])
        pred_yes = lp_yes > lp_no
        gold_yes = row["answer"].strip().lower() == "yes"
        if pred_yes and gold_yes:       tp += 1
        elif pred_yes and not gold_yes: fp += 1
        elif not pred_yes and gold_yes: fn += 1
        else:                           tn += 1
        if SMOKE:
            print(f"    Q: {row['question'][:70]}  lp_yes={lp_yes:.2f} "
                  f"lp_no={lp_no:.2f} pred={'yes' if pred_yes else 'no'} "
                  f"gold={row['answer']}")
        elif (k + 1) % 200 == 0:
            acc = (tp + tn) / max(1, tp + tn + fp + fn)
            print(f"    {k+1}/{len(idxs)}  running acc={acc:.4f}", flush=True)
    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-9, prec + rec)
    yes_ratio = (tp + fp) / total
    print(f"  [POPE] DONE  acc={acc:.4f} f1={f1:.4f} yes_ratio={yes_ratio:.3f}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "yes_ratio": yes_ratio, "n": total}


# ---------------------------------------------------------------------------
# 4. MME (perception + cognition, yes/no with acc and acc+)
# ---------------------------------------------------------------------------

def bench_mme(model, processor):
    from datasets import load_dataset
    print("  [MME] loading lmms-lab/MME ...", flush=True)
    ds = load_dataset("lmms-lab/MME", split="test")
    idxs = list(range(len(ds)))
    if MME_N > 0:
        rng = random.Random(SEED)
        rng.shuffle(idxs)
        idxs = idxs[:MME_N]
    print(f"  [MME] {len(idxs)} questions", flush=True)

    per_cat = defaultdict(lambda: {"correct": 0, "n": 0})
    # acc+ requires both questions of the same image to be correct
    per_image = defaultdict(list)
    t0 = time.time()
    for k, i in enumerate(idxs):
        row = ds[i]
        lp_yes, lp_no = vqa_yesno_logprobs(model, processor,
                                           row["image"], row["question"])
        pred = "yes" if lp_yes > lp_no else "no"
        gold = row["answer"].strip().lower()
        ok = int(pred == gold)
        cat = row["category"]
        per_cat[cat]["correct"] += ok
        per_cat[cat]["n"] += 1
        per_image[(cat, row["question_id"])].append(ok)
        if SMOKE:
            print(f"    [{cat}] {row['question'][:60]}  pred={pred} gold={gold} "
                  f"{'OK' if ok else 'WRONG'}")
        elif (k + 1) % 200 == 0:
            tot_c = sum(v["correct"] for v in per_cat.values())
            tot_n = sum(v["n"] for v in per_cat.values())
            print(f"    {k+1}/{len(idxs)}  running acc={tot_c/max(1,tot_n):.4f}",
                  flush=True)

    # Per-category acc and acc+ (both questions for an image correct)
    result = {}
    for cat, v in sorted(per_cat.items()):
        acc = v["correct"] / max(1, v["n"])
        pairs = [oks for (c, _), oks in per_image.items() if c == cat]
        accplus = (sum(1 for oks in pairs if len(oks) >= 2 and all(oks))
                   / max(1, len(pairs)))
        result[cat] = {"acc": acc, "acc_plus": accplus, "n": v["n"],
                       "mme_score": 100 * (acc + accplus)}
    total_correct = sum(v["correct"] for v in per_cat.values())
    total_n = sum(v["n"] for v in per_cat.values())
    overall_acc = total_correct / max(1, total_n)
    total_score = sum(r["mme_score"] for r in result.values())
    print(f"  [MME] DONE  overall_acc={overall_acc:.4f} "
          f"total_score={total_score:.1f}  ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    return {"overall_accuracy": overall_acc, "total_mme_score": total_score,
            "per_category": result, "n": total_n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tag = TAG or (Path(ADAPTER).parent.name + "__" + Path(ADAPTER).name
                  if ADAPTER else "base_no_adapter")
    out_dir = OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print(f"  Cross-modal forgetting battery")
    print(f"  MODEL_ID : {MODEL_ID}")
    print(f"  ADAPTER  : {ADAPTER or '<none, base model>'}")
    print(f"  TAG      : {tag}")
    print(f"  SMOKE    : {SMOKE}")
    print(f"  SKIP     : {SKIP or '<none>'}")
    print(SEP + "\n")

    model, processor = load_model()

    results = {"model": MODEL_ID, "adapter": ADAPTER, "tag": tag,
               "smoke": SMOKE}
    t_start = time.time()

    if "ppl" not in SKIP:
        results["wikitext2_ppl"] = bench_ppl(model, processor)
    if "mmlu" not in SKIP:
        results["mmlu"] = bench_mmlu(model, processor)
    if "pope" not in SKIP:
        results["pope"] = bench_pope(model, processor)
    if "mme" not in SKIP:
        results["mme"] = bench_mme(model, processor)

    results["minutes"] = round((time.time() - t_start) / 60, 1)
    out_json = out_dir / "forgetting_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + SEP)
    print("  FORGETTING BATTERY SUMMARY")
    print(SEP)
    if "wikitext2_ppl" in results:
        print(f"  WikiText-2 PPL : {results['wikitext2_ppl']['PPL']:.2f}")
    if "mmlu" in results:
        print(f"  MMLU accuracy  : {results['mmlu']['accuracy']:.4f}")
    if "pope" in results:
        print(f"  POPE acc / F1  : {results['pope']['accuracy']:.4f} / "
              f"{results['pope']['f1']:.4f}")
    if "mme" in results:
        print(f"  MME score      : {results['mme']['total_mme_score']:.1f} "
              f"(acc={results['mme']['overall_accuracy']:.4f})")
    print(f"  json: {out_json}")
    print(SEP)


if __name__ == "__main__":
    main()
