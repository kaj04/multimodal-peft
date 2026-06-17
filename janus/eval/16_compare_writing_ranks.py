#!/usr/bin/env python3
"""
janus/eval/16_compare_writing_ranks.py

Side-by-side poetry continuations across the rank-sweep adapters (base + r4 r8
r16 r32 r64) on the SAME incipits, so a human can read the style progression
and pick the rank for the next step. Sampled decode (style is clearer than
greedy), same seed per incipit so the only thing changing is the adapter.

Loads the base ONCE and attaches all adapters to one PeftModel (load_adapter +
set_adapter), so it is cheap. Adapters auto-discovered from the rank sweep
(text_poetry_r{4,8,16,32,64}_kl0.5); override with ADAPTERS="name=path;...".

Run via 16_run_compare_writing_ranks.sh (GPU, ~5 min).
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID    = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
SEED        = int(os.environ.get("SEED", "42"))
GEN_TOKENS  = int(os.environ.get("GEN_TOKENS", "70"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.9"))
TOP_P       = float(os.environ.get("TOP_P", "0.95"))

OUT_ROOT  = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"))) / "writing_rank_compare"
TRAIN_OUT = Path(os.path.expanduser("~/multimodal-peft/janus/train/output"))

_env = os.environ.get("ADAPTERS", "")
if _env:
    ADAPTERS = [(s.split("=")[0].strip(), s.split("=")[1].strip())
                for s in _env.split(";") if "=" in s]
else:
    ADAPTERS = [(f"r{r}", str(TRAIN_OUT / f"text_poetry_r{r}_kl0.5" / "adapter_best"))
                for r in (4, 8, 16, 32, 64)]

INCIPITS = [
    "The autumn wind",
    "O gentle heart, what",
    "Upon the silent hill there",
    "Beneath the ancient oak,",
]
SEP = "=" * 72


@torch.no_grad()
def _logits(model, ids):
    out = model(input_ids=ids)
    return out.logits if hasattr(out, "logits") else out[0]


@torch.no_grad()
def sample(model, tok, prompt, n, seed):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    start = ids.shape[1]
    for _ in range(n):
        logits = _logits(model, ids)[0, -1].float() / max(1e-6, TEMPERATURE)
        probs = F.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, dim=-1) <= TOP_P
        keep[0] = True
        sp, si = sp[keep], si[keep]
        nxt = int(si[torch.multinomial(sp / sp.sum(), 1)].item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
    return tok.decode(ids[0, start:].tolist(), skip_special_tokens=True)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    avail = [(n, p) for n, p in ADAPTERS
             if Path(p).exists() or Path(p).is_symlink()]
    if not avail:
        raise SystemExit("no adapters found; check the rank-sweep output folders")
    print(SEP)
    print(f"  writing rank compare | adapters: {[n for n, _ in avail]}")
    print(SEP, flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    base.eval()

    n0, p0 = avail[0]
    model = PeftModel.from_pretrained(base, p0, adapter_name=n0)
    for n, p in avail[1:]:
        model.load_adapter(p, adapter_name=n)
    model.eval()

    header = (f"{SEP}\nWRITING RANK COMPARE  sample(T={TEMPERATURE}, top_p={TOP_P})"
              f"  {GEN_TOKENS} tok\n{SEP}\n")
    lines = [header]
    records = []
    for inc in INCIPITS:
        seed = SEED + (sum(ord(c) for c in inc) % 1000)
        block = [f"\n{'-'*72}\nINCIPIT: {inc!r}\n"]
        with model.disable_adapter():
            b = sample(model, tok, inc, GEN_TOKENS, seed)
        block.append(f"  [base ]  {b}\n")
        rec = {"incipit": inc, "base": b, "ranks": {}}
        for n, _ in avail:
            model.set_adapter(n)
            c = sample(model, tok, inc, GEN_TOKENS, seed)
            block.append(f"  [{n:<5}]  {c}\n")
            rec["ranks"][n] = c
        print("".join(block), flush=True)
        lines += block
        records.append(rec)

    (OUT_ROOT / "writing_rank_compare.txt").write_text("".join(lines), encoding="utf-8")
    (OUT_ROOT / "writing_rank_compare.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT_ROOT}/writing_rank_compare.txt (+ .json)")


if __name__ == "__main__":
    main()
