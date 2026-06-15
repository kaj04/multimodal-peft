#!/usr/bin/env python3
"""
janus/eval/11_probe_text_ppl.py

Pick the text *style* for the reverse (text-PEFT) asymmetry experiment the
same way pixel-art was picked for image generation: objectively, by distance
from the base model. Here "distance" = base-model perplexity. The style the
base knows least (highest PPL) is the one whose LoRA must move the weights the
most -> the strongest forgetting test, analogous to pixel-art being far from
Janus' photoreal generation prior.

For each candidate style corpus we compute the BASE Janus-Pro-7B mean
next-token cross-entropy / perplexity on a FIXED token budget (same number of
windows x same window length for every style). The fixed budget is the whole
point: we already found PPL is inflated by corpus SIZE, not only by domain
distance, so a fair comparison must score every style on the same number of
tokens. Length-normalised mean per-token CE on an equal budget removes that
confound.

Output: a ranked table (lowest -> highest PPL) + json. The highest-PPL style
that is still a clean, trainable corpus is the recommended text-PEFT task.

Same forward/CE pattern as 07_forgetting_battery.py (bench_ppl) and the same
model loading as 10_probe_code_capability.py. Optionally pass ADAPTER to
re-measure the SAME styles after a text LoRA (symmetry with the code probe).

Run via 11_run_probe_text_ppl.sh (GPU, ~5 min).
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")            # empty -> base model
SEED     = int(os.environ.get("SEED", "42"))

# Equal token budget per style (the de-confounding knob). 40 x 512 ~= 20k
# tokens/style -> stable PPL estimate, whole probe ~5 min on one H100.
WINDOW_TOKENS = int(os.environ.get("WINDOW_TOKENS", "512"))
MAX_WINDOWS   = int(os.environ.get("MAX_WINDOWS", "40"))
SMOKE         = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    MAX_WINDOWS = 4

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "text_ppl_probe"

SEP = "=" * 70

# ---------------------------------------------------------------------------
# Candidate text styles. Each entry is a self-contained load spec:
#   name        -> short tag used in the report
#   hf          -> HF dataset repo id
#   config      -> optional dataset config name
#   split       -> split to read
#   text_col    -> preferred text column (auto-detected if missing/absent)
#   revision    -> optional (e.g. "refs/convert/parquet" for script-only repos)
#   data_files  -> optional explicit parquet path inside the repo
#   note        -> human description of the style
#
# All verified to exist on the Hub and to be (or auto-convert to) parquet.
# `wikitext_modern` is the FAMILIAR control / floor: modern encyclopedic prose
# is exactly what the base was trained on, so its PPL anchors the low end and
# every style above it is "more distant" by that margin.
#
# Edit freely: add/remove rows, swap datasets. A row that fails to load is
# skipped with a warning, the probe still ranks the rest.
# ---------------------------------------------------------------------------
STYLES = [
    {"name": "wikitext_modern", "hf": "Salesforce/wikitext",
     "config": "wikitext-103-raw-v1", "split": "test", "text_col": "text",
     "note": "modern encyclopedic prose (FAMILIAR control / floor)"},

    {"name": "shakespeare_eme", "hf": "Trelis/tiny-shakespeare",
     "split": "train",
     "note": "Early Modern English drama (verse + dialogue)"},

    {"name": "kjv_bible", "hf": "JDRJ/kjv-bible", "split": "train",
     "note": "King James Bible: archaic religious register"},

    {"name": "gutenberg_poetry", "hf": "biglam/gutenberg-poetry-corpus",
     "split": "train",
     "note": "English poetry lines (Project Gutenberg)"},

    {"name": "legalese", "hf": "mteb/legal_summarization",
     "config": "corpus", "split": "corpus", "text_col": "text",
     "note": "legal contracts / terms-of-service: dense legalese register"},
]


# ---------------------------------------------------------------------------
# Model loading  (same idiom as the battery / code probe)
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
        print(f"  loaded adapter: {ADAPTER}", flush=True)
        return model, processor
    print("  no adapter -> base model", flush=True)
    return base, processor


# ---------------------------------------------------------------------------
# Text collection (robust): non-streaming first, streaming fallback, skip-on-fail
# ---------------------------------------------------------------------------

def _pick_text_col(sample_rows, preferred):
    """Pick the string column with the largest mean length over `sample_rows`.
    If `preferred` is present and holds strings, use it."""
    if not sample_rows:
        return None
    keys = list(sample_rows[0].keys())
    if preferred and preferred in keys and isinstance(sample_rows[0][preferred], str):
        return preferred
    best, best_len = None, -1.0
    for k in keys:
        vals = [r[k] for r in sample_rows if isinstance(r.get(k), str)]
        if not vals:
            continue
        mlen = sum(len(v) for v in vals) / len(vals)
        if mlen > best_len:
            best, best_len = k, mlen
    return best


def collect_text(spec, target_chars, rng):
    """Return (joined_text, text_col, n_docs) for a style spec, or raise."""
    from datasets import load_dataset
    load_args = [spec["hf"]] + ([spec["config"]] if spec.get("config") else [])
    kwargs = {"split": spec.get("split", "train")}
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]

    # --- attempt 1: full (random access). Read a CONTIGUOUS span from a random
    # offset so the NATURAL reading order is preserved regardless of row
    # granularity (line / verse / paragraph). Shuffling rows would scramble
    # line-level corpora (poetry, KJV) into disconnected fragments and inflate
    # their PPL by construction -> that would measure incoherence, not style.
    # A random start also usually lands inside real content, not Gutenberg
    # front-matter (prefaces / licence headers sit at the very top).
    try:
        ds = load_dataset(*load_args, **kwargs)
        n = len(ds)
        sample = [ds[i] for i in range(min(50, n))]
        col = _pick_text_col(sample, spec.get("text_col"))
        if col is None:
            raise ValueError("no string column found")
        start = rng.randrange(n)
        buf, nchars = [], 0
        for j in range(n):
            t = ds[(start + j) % n][col]
            if isinstance(t, str) and t.strip():
                buf.append(t.strip())
                nchars += len(t)
                if nchars >= target_chars:
                    break
        return "\n".join(buf), col, len(buf), start
    except Exception as e:
        print(f"    (full load failed: {type(e).__name__}: {e}; "
              f"trying streaming)", flush=True)

    # --- attempt 2: streaming (for big / script-only repos)
    ds = load_dataset(*load_args, streaming=True, **kwargs)
    it = iter(ds)
    first = next(it)
    col = _pick_text_col([first], spec.get("text_col"))
    if col is None:
        raise ValueError("no string column found (streaming)")
    buf, nchars = [], 0
    for row in [first, *it]:
        t = row.get(col)
        if isinstance(t, str) and t.strip():
            buf.append(t.strip())
            nchars += len(t)
            if nchars >= target_chars:
                break
    return "\n".join(buf), col, len(buf), 0


# ---------------------------------------------------------------------------
# Perplexity on a fixed token budget  (same CE as bench_ppl)
# ---------------------------------------------------------------------------

@torch.no_grad()
def style_ppl(model, tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    n_chunks = min(MAX_WINDOWS, len(ids) // WINDOW_TOKENS)
    if n_chunks == 0:
        return None
    losses = []
    for i in range(n_chunks):
        chunk = torch.tensor(ids[i * WINDOW_TOKENS:(i + 1) * WINDOW_TOKENS],
                             dtype=torch.long).unsqueeze(0).to(model.device)
        out = model(input_ids=chunk)
        logits = (out.logits if hasattr(out, "logits") else out[0])
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = chunk[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
        losses.append(loss.item())
    mean_ce = float(np.mean(losses))
    return {"mean_CE": mean_ce, "PPL": float(np.exp(mean_ce)),
            "n_windows": n_chunks, "tokens_scored": n_chunks * WINDOW_TOKENS,
            "tokens_available": len(ids)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "base" if not ADAPTER else Path(ADAPTER).parent.name
    rng = random.Random(SEED)
    # margin so we always have >= MAX_WINDOWS windows after tokenization
    target_chars = WINDOW_TOKENS * MAX_WINDOWS * 6

    print(SEP)
    print(f"  Text-style PPL probe | {tag}")
    print(f"  MODEL={MODEL_ID}  ADAPTER={ADAPTER or '<base>'}")
    print(f"  budget = {MAX_WINDOWS} windows x {WINDOW_TOKENS} tokens "
          f"= {MAX_WINDOWS * WINDOW_TOKENS} tokens/style (equal across styles)")
    print(SEP + "\n", flush=True)

    model, processor = load_model()
    tok = processor.tokenizer

    rows = []
    for spec in STYLES:
        name = spec["name"]
        print(f"  [{name}] loading {spec['hf']}"
              f"{(' / ' + spec['config']) if spec.get('config') else ''} ...",
              flush=True)
        try:
            text, col, n_docs, start = collect_text(spec, target_chars, rng)
        except Exception as e:
            print(f"    SKIP {name}: {type(e).__name__}: {e}\n", flush=True)
            continue
        t0 = time.time()
        res = style_ppl(model, tok, text)
        if res is None:
            print(f"    SKIP {name}: not enough tokens\n", flush=True)
            continue
        res.update({"name": name, "hf": spec["hf"], "config": spec.get("config"),
                    "text_col": col, "n_docs": n_docs, "start_offset": start,
                    "note": spec.get("note", "")})
        rows.append(res)
        print(f"    col='{col}'  docs={n_docs}  start={start}  "
              f"windows={res['n_windows']}  "
              f"mean_CE={res['mean_CE']:.4f}  PPL={res['PPL']:.2f}  "
              f"({time.time()-t0:.0f}s)\n", flush=True)

    rows.sort(key=lambda r: r["PPL"])

    # control = wikitext if present, else the lowest-PPL style
    ctrl = next((r for r in rows if r["name"] == "wikitext_modern"),
                rows[0] if rows else None)

    lines = [f"{SEP}", f"TEXT-STYLE PPL PROBE   tag={tag}   model={MODEL_ID}",
             f"budget = {MAX_WINDOWS} x {WINDOW_TOKENS} tokens/style", SEP,
             f"{'rank':>4}  {'style':<18}{'PPL':>9}{'mean_CE':>10}"
             f"{'xControl':>10}  note", "-" * len(SEP)]
    for k, r in enumerate(rows):
        xctrl = r["PPL"] / ctrl["PPL"] if ctrl else float("nan")
        lines.append(f"{k+1:>4}  {r['name']:<18}{r['PPL']:>9.2f}"
                     f"{r['mean_CE']:>10.4f}{xctrl:>9.2f}x  {r['note']}")
    lines.append("-" * len(SEP))
    if rows:
        far = rows[-1]
        lines.append(f"  MOST DISTANT (recommended text-PEFT task): "
                     f"{far['name']}  PPL={far['PPL']:.2f}  "
                     f"({far['PPL']/ctrl['PPL']:.2f}x control)")
        lines.append(f"  -> hf={far['hf']}"
                     f"{(' config=' + far['config']) if far['config'] else ''}"
                     f"  text_col={far['text_col']}")
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    (OUTPUT_DIR / f"text_ppl_{tag}.txt").write_text(report + "\n", encoding="utf-8")
    (OUTPUT_DIR / f"text_ppl_{tag}.json").write_text(
        json.dumps({"model": MODEL_ID, "adapter": ADAPTER, "tag": tag,
                    "window_tokens": WINDOW_TOKENS, "max_windows": MAX_WINDOWS,
                    "styles": rows}, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUTPUT_DIR}/text_ppl_{tag}.txt (+ .json)")


if __name__ == "__main__":
    main()
