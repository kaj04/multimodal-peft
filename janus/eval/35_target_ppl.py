#!/usr/bin/env python3
"""
janus/eval/35_target_ppl.py

Held-out perplexity of ANY text target, for ANY adapter, on the SAME windows the
trainer held out. Generalises 26_welsh_ppl.py, which is Welsh-only and writes to
a single fixed file; 26 is left untouched so the numbers already in the paper
stay reproducible.

Needed for the sequential-chain experiment, where every step must be scored on
every text target, not just the one it was trained on. That gives the rows of
the retention matrix: PPL of target j after step k.

The window construction mirrors 03_train_text.py exactly (same SEED, the same
1.3 over-collection factor, the same two shuffles, the same VAL_WINDOWS prefix),
so the windows scored here are the ones the trainer never saw. Pass the SAME
corpus settings used for training that target, otherwise a different split is
reproduced and the comparison is meaningless.

base and adapter are scored on identical windows via disable_adapter(), so the
LoRA is the only thing that changes.

Env: ADAPTER (required), CORPUS (label for the output), HF_DATASET_NAME,
HF_DATASET_CONFIG (may be empty), HF_TEXT_COL, MAX_DOCS, SEQ_LEN, TRAIN_WINDOWS,
VAL_WINDOWS, SEED, TAG (subdirectory of the output).
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
TAG      = os.environ.get("TAG", "untagged")
CORPUS   = os.environ.get("CORPUS", "target")

HF_DATASET_NAME   = os.environ.get("HF_DATASET_NAME", "wikimedia/wikipedia")
HF_DATASET_CONFIG = os.environ.get("HF_DATASET_CONFIG", "")
HF_TEXT_COL       = os.environ.get("HF_TEXT_COL", "text")
MAX_DOCS      = int(os.environ.get("MAX_DOCS", "40000"))
SEQ_LEN       = int(os.environ.get("SEQ_LEN", "512"))
TRAIN_WINDOWS = int(os.environ.get("TRAIN_WINDOWS", "6000"))
VAL_WINDOWS   = int(os.environ.get("VAL_WINDOWS", "200"))
SEED          = int(os.environ.get("SEED", "42"))
LOAD_RETRIES  = int(os.environ.get("LOAD_RETRIES", "6"))

OUT = Path(os.path.expanduser(
    "~/multimodal-peft/janus/tests/output")) / "chain_ppl" / TAG
SEP = "=" * 62


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


def corpus_docs():
    # Empty config must not be forwarded: load_dataset(name, "") is an error.
    sds = (_load_retry(HF_DATASET_NAME, HF_DATASET_CONFIG, split="train",
                       streaming=True)
           if HF_DATASET_CONFIG else
           _load_retry(HF_DATASET_NAME, split="train", streaming=True))
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
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{SEP}\n  held-out PPL | corpus={CORPUS} tag={TAG}\n"
          f"  dataset={HF_DATASET_NAME}"
          f"{('/' + HF_DATASET_CONFIG) if HF_DATASET_CONFIG else ''}"
          f" col={HF_TEXT_COL}\n  ADAPTER={ADAPTER}\n{SEP}", flush=True)

    proc = JanusProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()

    print("  rebuilding the trainer's held-out windows ...", flush=True)
    docs = corpus_docs()
    random.Random(SEED).shuffle(docs)
    need = int((TRAIN_WINDOWS + VAL_WINDOWS) * 1.3)
    windows = build_windows(docs, tok, need=need)
    random.Random(SEED + 1).shuffle(windows)
    val = windows[:VAL_WINDOWS]
    print(f"  docs={len(docs)}  windows_built={len(windows)}  "
          f"val_windows={len(val)}", flush=True)
    if len(val) < VAL_WINDOWS:
        print(f"  WARNING: corpus yielded only {len(val)} of {VAL_WINDOWS} "
              f"requested val windows; comparisons across steps stay valid "
              f"(same windows every time) but the set is smaller than planned.",
              flush=True)

    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    with model.disable_adapter():
        ce_b, ppl_b = ppl(model, val)
    ce_a, ppl_a = ppl(model, val)

    rec = {"tag": TAG, "corpus": CORPUS, "adapter": ADAPTER,
           "dataset": HF_DATASET_NAME, "config": HF_DATASET_CONFIG,
           "n_val_windows": len(val), "seq_len": SEQ_LEN, "seed": SEED,
           "base_CE": round(ce_b, 5), "base_PPL": round(ppl_b, 4),
           "adapter_CE": round(ce_a, 5), "adapter_PPL": round(ppl_a, 4),
           "ratio_adapter_over_base": round(ppl_a / ppl_b, 4)}

    line = (f"{SEP}\n  {CORPUS} held-out PPL   (val_windows={len(val)})\n"
            f"  TAG={TAG}\n  ADAPTER={ADAPTER}\n"
            f"  base    : CE={ce_b:.4f}  PPL={ppl_b:.2f}\n"
            f"  adapter : CE={ce_a:.4f}  PPL={ppl_a:.2f}\n"
            f"  ratio   : {ppl_a/ppl_b:.3f}x base "
            f"({100*(1-ppl_a/ppl_b):+.1f}% vs base)\n{SEP}")
    print("\n" + line)

    (OUT / f"ppl_{CORPUS}.json").write_text(json.dumps(rec, indent=2),
                                            encoding="utf-8")
    (OUT / f"ppl_{CORPUS}.txt").write_text(line + "\n", encoding="utf-8")
    print(f"\n  saved -> {OUT}/ppl_{CORPUS}.json")


if __name__ == "__main__":
    main()
