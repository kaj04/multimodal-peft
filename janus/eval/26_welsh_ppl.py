#!/usr/bin/env python3
"""
janus/eval/26_welsh_ppl.py

Cheap, EVAL-ONLY acquisition check for the Welsh text-LoRA: held-out Welsh PPL
with the adapter OFF (base) vs ON (adapter_best), on the SAME held-out val
windows the trainer used (same dataset / SEED / MAX_DOCS / split logic). No
retraining -- reuses the already-trained adapter. This replaces the
history.json number, which got contaminated when a run RESUMEd a stale
checkpoint (baseline measured on an already-trained model).

base and adapter are scored on the IDENTICAL windows via disable_adapter(), so
the only thing that changes is the LoRA -> a clean before/after.

Run via 26_run_welsh_ppl.sh (GPU, ~5-10 min). Needs ADAPTER=.../adapter_best.
"""
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
# Defaults MATCH the welsh training run so the held-out val set is reproduced.
HF_DATASET_NAME   = os.environ.get("HF_DATASET_NAME", "wikimedia/wikipedia")
HF_DATASET_CONFIG = os.environ.get("HF_DATASET_CONFIG", "20231101.cy")
HF_TEXT_COL = os.environ.get("HF_TEXT_COL", "text")
MAX_DOCS = int(os.environ.get("MAX_DOCS", "40000"))
SEQ_LEN  = int(os.environ.get("SEQ_LEN", "512"))
TRAIN_WINDOWS = int(os.environ.get("TRAIN_WINDOWS", "6000"))
VAL_WINDOWS   = int(os.environ.get("VAL_WINDOWS", "200"))
SEED = int(os.environ.get("SEED", "42"))
LOAD_RETRIES = int(os.environ.get("LOAD_RETRIES", "6"))


def _load_retry(*a, **k):
    from datasets import load_dataset
    last = None
    for i in range(1, LOAD_RETRIES + 1):
        try:
            return load_dataset(*a, **k)
        except Exception as e:
            last = e
            w = min(60, 8 * i)
            print(f"  [load {i}/{LOAD_RETRIES}] {type(e).__name__}: {e} -> {w}s",
                  flush=True)
            time.sleep(w)
    raise last


def welsh_docs():
    sds = _load_retry(HF_DATASET_NAME, HF_DATASET_CONFIG, split="train",
                      streaming=True)
    docs = []
    for row in sds:
        t = row.get(HF_TEXT_COL)
        if isinstance(t, str) and t.strip():
            docs.append(t.strip())
        if len(docs) >= MAX_DOCS:
            break
    return docs


def build_windows(docs, tok, need):
    w = []
    for d in docs:
        ids = tok.encode(d, add_special_tokens=False)
        for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
            w.append(ids[i:i + SEQ_LEN])
        if need and len(w) >= need:
            break
    return w


@torch.no_grad()
def ppl(model, val):
    losses = []
    for w in val:
        ids = torch.tensor(w, dtype=torch.long).unsqueeze(0).to(model.device)
        out = model(input_ids=ids)
        logits = (out.logits if hasattr(out, "logits") else out[0])[:, :-1, :].float()
        losses.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)).item())
    ce = float(np.mean(losses))
    return ce, float(np.exp(ce))


def main():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    assert ADAPTER, "set ADAPTER=.../adapter_best"
    proc = JanusProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()

    print("  building held-out Welsh val windows ...", flush=True)
    docs = welsh_docs()
    random.Random(SEED).shuffle(docs)
    need = int((TRAIN_WINDOWS + VAL_WINDOWS) * 1.3)
    windows = build_windows(docs, tok, need=need)
    random.Random(SEED + 1).shuffle(windows)
    val = windows[:VAL_WINDOWS]
    print(f"  val windows={len(val)} (of {len(windows)} built)", flush=True)

    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    with model.disable_adapter():
        ce_b, ppl_b = ppl(model, val)
    ce_a, ppl_a = ppl(model, val)

    out = Path(os.path.expanduser(
        "~/multimodal-peft/janus/tests/output")) / "welsh_ppl"
    out.mkdir(parents=True, exist_ok=True)
    line = ("=" * 56 + f"\n  Welsh held-out PPL  (val_windows={len(val)})\n"
            f"  ADAPTER={ADAPTER}\n"
            f"  base    : CE={ce_b:.4f}  PPL={ppl_b:.2f}\n"
            f"  adapter : CE={ce_a:.4f}  PPL={ppl_a:.2f}\n"
            f"  acquisition: PPL {ppl_b:.2f} -> {ppl_a:.2f}  "
            f"({ppl_a/ppl_b:.2f}x, drop {100*(1-ppl_a/ppl_b):.1f}%)\n" + "=" * 56)
    print("\n" + line)
    (out / "welsh_ppl.txt").write_text(line + "\n", encoding="utf-8")
    print(f"\n  saved -> {out}/welsh_ppl.txt")


if __name__ == "__main__":
    main()
