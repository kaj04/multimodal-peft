#!/usr/bin/env python3
"""
janus/data_prep/02_tokenize_corpora.py

Tokenise the three wiki corpora (matched / near / far) into fixed-length
sequences ready for the joint trainer. Uses the Janus processor's tokenizer
(== the LLaMA-style tokenizer Janus inherits from DeepSeek-LLM) so the IDs
match what the model expects at training time.

Output: one .jsonl per bucket. Each line is {"input_ids": [int, int, ...]}
with len(input_ids) == CHUNK_LEN. Articles longer than CHUNK_LEN are split
into multiple non-overlapping chunks; articles shorter are dropped (so we
never train on padding tokens). The default CHUNK_LEN=512 balances long-
range dependency vs throughput on H100.

Run on the login node (CPU only):
    python ~/multimodal-peft/janus/data_prep/02_tokenize_corpora.py
"""

import json
import os
import sys
import time
from pathlib import Path

MODEL_ID  = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
CHUNK_LEN = int(os.environ.get("CHUNK_LEN", "512"))

DATA_DIR = Path(os.environ.get(
    "OUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/data_prep/output")))

# (bucket name, input jsonl, output jsonl)
SPECS = [
    ("matched", DATA_DIR / "wiki_matched_naruto_clean.jsonl",
                DATA_DIR / "wiki_matched_tokens.jsonl"),
    ("near",    DATA_DIR / "wiki_near_anime.jsonl",
                DATA_DIR / "wiki_near_tokens.jsonl"),
    ("far",     DATA_DIR / "wiki_far_physics.jsonl",
                DATA_DIR / "wiki_far_tokens.jsonl"),
]


def main():
    from transformers import JanusProcessor
    print(f"  loading tokenizer from {MODEL_ID} ...", flush=True)
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    for bucket, src, dst in SPECS:
        if not src.exists():
            print(f"  [{bucket}] SKIP: {src} not found")
            continue
        t0 = time.time()
        n_articles = 0
        n_chunks   = 0
        n_dropped_short = 0
        with src.open("r", encoding="utf-8") as fin, \
             dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                row = json.loads(line)
                text = row["text"]
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) < CHUNK_LEN:
                    n_dropped_short += 1
                    continue
                # Non-overlapping chunks of exactly CHUNK_LEN tokens.
                # We drop the trailing remainder if shorter than CHUNK_LEN.
                for i in range(0, len(ids) - CHUNK_LEN + 1, CHUNK_LEN):
                    chunk = ids[i:i + CHUNK_LEN]
                    fout.write(json.dumps({"input_ids": chunk}) + "\n")
                    n_chunks += 1
                n_articles += 1
        elapsed = time.time() - t0
        print(f"  [{bucket}]  articles_used={n_articles:>5d}  "
              f"chunks={n_chunks:>6d}  dropped_short={n_dropped_short:>4d}  "
              f"({elapsed:.1f}s)  -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()
