#!/usr/bin/env python3
"""
janus/eval/25_probe_language_targets.py

Pick the TEXT fine-tuning target the same principled way we picked image styles:
by distance from the base. For each candidate we compute the base Janus-Pro-7B
perplexity and express it as xControl (vs.\ English WikiText). A HIGH xControl =
the base is very "surprised" = the target is genuinely unseen = PEFT will be
forced to move the weights a lot = a non-trivial forgetting test (unlike poetry /
medicine, which sit inside the base's competence).

Candidates use REAL, available corpora (constructed GoT/Elvish languages have no
HF dataset, only dictionaries -- confirmed -- so they are excluded):
  - low-resource natural languages via Wikipedia (Welsh, Yoruba, Latin, Esperanto)
  - a synthetic substitution-cipher language built from English (unlimited data,
    provably unseen, exact downstream eval) as a maximally-distant anchor.

For each candidate we also print 3 example snippets so the choice can be judged
qualitatively, not only by the number.

Run via 25_run_probe_language_targets.sh (GPU).
"""
import os
import random
import string
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
WIKI_DATE = os.environ.get("WIKI_DATE", "20231101")
PPL_CHUNK = int(os.environ.get("PPL_CHUNK", "512"))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "25"))
MAX_DOCS = int(os.environ.get("MAX_DOCS", "300"))
SEED = int(os.environ.get("SEED", "42"))
OUT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output")) / "language_probe"

# name -> ("tapaco", lang) | ("wiki", lang_code) | ("cipher", None)
CANDIDATES = [
    ("klingon",   ("tapaco", "tlh")),   # fictional conlang (Star Trek), largest conlang corpus
    ("quenya",    ("tapaco", "qya")),   # Tolkien Elvish
    ("welsh",     ("wiki", "cy")),      # low-resource natural (abundant data) reference
    ("yoruba",    ("wiki", "yo")),      # low-resource natural reference
    ("cipher",    ("cipher", None)),    # synthetic language: unlimited data, max distance
]
SEP = "=" * 70


def _cipher_map():
    rng = random.Random(SEED)
    src = string.ascii_lowercase
    dst = list(src)
    rng.shuffle(dst)
    table = str.maketrans(src + src.upper(), "".join(dst) + "".join(dst).upper())
    return table


def _english_text():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(t for t in ds["text"] if t.strip())


def collect_text(spec):
    kind, arg = spec
    if kind == "cipher":
        return _english_text().translate(_cipher_map())
    from datasets import load_dataset
    if kind == "tapaco":
        # TAPACO = Tatoeba paraphrase corpus (incl. fictional langs Klingon/Quenya).
        ds = load_dataset("community-datasets/tapaco", arg, split="train",
                          trust_remote_code=True)
        cols = ds.column_names
        col = "paraphrase" if "paraphrase" in cols else next(
            (c for c in cols if isinstance(ds[0].get(c), str)), cols[-1])
        return "\n".join(str(r[col]) for r in ds if r.get(col))
    # wiki: stream articles for the language code

    ds = load_dataset("wikimedia/wikipedia", f"{WIKI_DATE}.{arg}", split="train",
                      streaming=True)
    parts, n = [], 0
    for row in ds:
        t = row.get("text", "")
        if t and len(t) > 200:
            parts.append(t)
            n += 1
        if n >= MAX_DOCS:
            break
    return "\n\n".join(parts)


def examples(text, k=3, width=200):
    # take 3 readable snippets spaced through the text
    text = " ".join(text.split())
    if len(text) < width * 3:
        return [text[:width]]
    step = len(text) // (k + 1)
    return [text[step * (i + 1): step * (i + 1) + width].strip() for i in range(k)]


@torch.no_grad()
def ppl(model, tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    n = min(MAX_CHUNKS, len(ids) // PPL_CHUNK)
    if n == 0:
        return float("nan"), 0
    losses = []
    for i in range(n):
        c = torch.tensor(ids[i * PPL_CHUNK:(i + 1) * PPL_CHUNK],
                         dtype=torch.long).unsqueeze(0).to(model.device)
        out = model(input_ids=c)
        logits = (out.logits if hasattr(out, "logits") else out[0])[..., :-1, :].float()
        labels = c[..., 1:]
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      labels.reshape(-1)).item())
    return float(np.exp(np.mean(losses))), n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import JanusForConditionalGeneration, JanusProcessor
    print(f"{SEP}\n  language target probe | MODEL={MODEL_ID}\n{SEP}", flush=True)
    proc = JanusProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer
    model = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()

    print("  [control] English WikiText ...", flush=True)
    ctrl_ppl, _ = ppl(model, tok, _english_text())
    print(f"  control PPL = {ctrl_ppl:.2f}\n", flush=True)

    rows = []
    blocks = []
    for name, spec in CANDIDATES:
        try:
            print(f"  [{name}] loading ...", flush=True)
            text = collect_text(spec)
            ex = examples(text)
            p, n = ppl(model, tok, text)
            xc = p / ctrl_ppl if ctrl_ppl else float("nan")
            rows.append((name, p, xc, n, len(text)))
            blocks.append((name, p, xc, ex))
            print(f"    PPL={p:.1f}  xControl={xc:.2f}  ({n} chunks, {len(text)} chars)",
                  flush=True)
        except Exception as e:
            print(f"    [skip] {name}: {type(e).__name__}: {e}", flush=True)

    rows.sort(key=lambda r: -r[2])  # most distant first
    lines = [SEP, f"  LANGUAGE TARGET PROBE  (control WikiText PPL = {ctrl_ppl:.2f})",
             "  higher xControl = more distant = stronger, less-trivial PEFT target", SEP,
             f"  {'target':<12}{'PPL':>10}{'xControl':>10}{'chunks':>8}"]
    for name, p, xc, n, _ in rows:
        lines.append(f"  {name:<12}{p:>10.1f}{xc:>10.2f}{n:>8}")
    lines.append("")
    for name, p, xc, ex in blocks:
        lines.append(f"\n--- {name}  (PPL {p:.0f}, xControl {xc:.2f}) ---")
        for j, e in enumerate(ex, 1):
            lines.append(f"  [{j}] {e}")
    report = "\n".join(lines)
    print("\n" + report)
    (OUT / "language_probe.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\n  saved -> {OUT}/language_probe.txt")


if __name__ == "__main__":
    main()
