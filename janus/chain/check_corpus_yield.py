#!/usr/bin/env python3
"""
janus/chain/check_corpus_yield.py

How many training windows can each chain corpus actually supply?

This has to be answered before the chains are launched, and the smoke test
cannot answer it: build_windows() stops as soon as it has collected the number
of windows requested, so a smoke run asking for 350 windows reports ~455 and
tells us nothing about the ceiling. It does not mean the corpus ran out.

It matters because the chain uses TRAIN_WINDOWS=6000 for both text steps. If
one corpus cannot supply that many, that step silently trains on less data than
the other, and any difference in retention would be confounded with exposure
rather than caused by order. This project has already documented exactly that
artefact: a small text corpus inflated held-out perplexity to 55.1 while
knowledge and understanding stayed flat.

Welsh Wikipedia in particular is full of short stubs, and a document shorter
than SEQ_LEN contributes zero windows, so the yield per document can be far
below one.

CPU only: it loads the tokenizer, not the model. Run on a login node.

  python janus/chain/check_corpus_yield.py
  TARGET_WINDOWS=4000 python janus/chain/check_corpus_yield.py
"""
import os
import sys
import time

SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
TRAIN_WINDOWS = int(os.environ.get("TRAIN_WINDOWS", "6000"))
VAL_WINDOWS = int(os.environ.get("VAL_WINDOWS", "200"))
MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")

# what 03_train_text.py will ask for, including its 1.3 over-collection factor
NEED = int((TRAIN_WINDOWS + VAL_WINDOWS) * 1.3)

CORPORA = [
    ("welsh", os.environ.get("WELSH_DATASET", "wikimedia/wikipedia"),
     os.environ.get("WELSH_CONFIG", "20231101.cy"),
     os.environ.get("WELSH_TEXT_COL", "text"),
     int(os.environ.get("WELSH_MAX_DOCS", "40000"))),
    ("medicine", os.environ.get("MED_DATASET", "ccdv/pubmed-summarization"),
     os.environ.get("MED_CONFIG", ""),
     os.environ.get("MED_TEXT_COL", "article"),
     int(os.environ.get("MED_MAX_DOCS", "40000"))),
]
SEP = "=" * 76


def main():
    from datasets import load_dataset
    from transformers import JanusProcessor

    print(f"{SEP}\n  CORPUS YIELD CHECK\n"
          f"  seq_len={SEQ_LEN}  chain asks for {TRAIN_WINDOWS}+{VAL_WINDOWS} "
          f"windows -> need {NEED} (1.3x over-collection)\n{SEP}", flush=True)

    tok = JanusProcessor.from_pretrained(MODEL_ID).tokenizer
    results, ok = [], True

    for name, ds_name, cfg, col, max_docs in CORPORA:
        print(f"\n  [{name}] {ds_name}{('/' + cfg) if cfg else ''} "
              f"col={col} max_docs={max_docs}", flush=True)
        t0 = time.time()
        stream = (load_dataset(ds_name, cfg, split="train", streaming=True)
                  if cfg else
                  load_dataset(ds_name, split="train", streaming=True))

        docs_seen = docs_used = windows = empty_docs = 0
        for row in stream:
            t = row.get(col)
            if not (isinstance(t, str) and t.strip()):
                continue
            docs_seen += 1
            ids = tok.encode(t.strip(), add_special_tokens=False)
            n = max(0, (len(ids) - SEQ_LEN + 1 + SEQ_LEN - 1) // SEQ_LEN) \
                if len(ids) >= SEQ_LEN else 0
            n = len(range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN))
            if n == 0:
                empty_docs += 1
            else:
                docs_used += 1
            windows += n
            if windows >= NEED or docs_seen >= max_docs:
                break
            if docs_seen % 2000 == 0:
                print(f"    ... {docs_seen} docs -> {windows} windows "
                      f"({time.time() - t0:.0f}s)", flush=True)

        reached = windows >= NEED
        ok = ok and reached
        per_doc = windows / max(docs_seen, 1)
        results.append((name, docs_seen, empty_docs, windows, reached, per_doc))
        print(f"    docs read      : {docs_seen}"
              f"{'  (hit MAX_DOCS)' if docs_seen >= max_docs else ''}")
        print(f"    too short      : {empty_docs} "
              f"({100.0 * empty_docs / max(docs_seen, 1):.0f}% yield nothing)")
        print(f"    windows        : {windows}  (need {NEED})")
        print(f"    windows/doc    : {per_doc:.2f}")
        print(f"    -> {'OK' if reached else 'SHORT'}   [{time.time() - t0:.0f}s]",
              flush=True)

    print(f"\n{SEP}\n  VERDICT\n{SEP}")
    for name, seen, empty, w, reached, per_doc in results:
        print(f"  {name:<10} {w:>6} windows from {seen:>6} docs "
              f"({per_doc:.2f}/doc)   {'OK' if reached else '*** SHORT ***'}")

    if ok:
        print(f"\n  Both corpora reach {NEED} windows. TRAIN_WINDOWS="
              f"{TRAIN_WINDOWS} is safe; launch the chains.")
    else:
        worst = min(r[3] for r in results)
        safe = int((worst / 1.3) - VAL_WINDOWS)
        print(f"\n  *** DO NOT LAUNCH AS CONFIGURED ***")
        print(f"  The smaller corpus tops out at {worst} windows, so the two")
        print(f"  text steps would see different amounts of data and any")
        print(f"  retention difference would be confounded with exposure.")
        print(f"  Either raise MAX_DOCS, or set TRAIN_WINDOWS to about "
              f"{max(safe, 0)} in janus/chain/chain_config.sh.")
    print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
