#!/usr/bin/env python3
"""
janus/eval/10_probe_code_capability.py

Probe the BASE Janus-Pro-7B's out-of-the-box code-writing ability, to set a
baseline BEFORE any code PEFT. Janus-Pro's backbone is DeepSeek-LLM-7B-base
(moderate, non-specialised coding ability; no official code benchmark in the
Janus paper), so we measure empirically what it can do.

For a handful of coding prompts of increasing difficulty, greedy-decode a
completion and save it (txt + json) for manual inspection. Optionally also
runs WITH an adapter (ADAPTER env) so the same script doubles as the
post-PEFT check later.

Greedy decode is manual (forward + argmax), same as the forgetting-battery
free-gen path, to avoid Janus' generate() defaulting to image mode.

Run via 10_run_probe_code.sh (GPU, ~5 min).
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")   # empty -> base model
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "220"))

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "code_probe"

# Increasing difficulty. A light "```python" priming nudges the model into
# code mode so we see its BEST behaviour, not whether it guesses the format.
PROMPTS = [
    "Write a Python function that returns the factorial of a number n.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a Python function that sorts a list of integers using bubble sort.",
    "Write a Python function that counts the frequency of each word in a string.",
    "Write a Python function that merges two sorted lists into one sorted list.",
]
SEP = "=" * 64


@torch.no_grad()
def greedy(model, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    start = ids.shape[1]
    for _ in range(n):
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = int(torch.argmax(logits[0, -1]).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
    return tok.decode(ids[0, start:].tolist(), skip_special_tokens=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "base" if not ADAPTER else Path(ADAPTER).parent.name
    print(SEP)
    print(f"  Code-capability probe | {tag}")
    print(f"  MODEL={MODEL_ID}  ADAPTER={ADAPTER or '<base>'}")
    print(SEP + "\n", flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    base.eval()
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ADAPTER); model.eval()
    else:
        model = base
    tok = processor.tokenizer

    results = []
    lines = [f"{SEP}\nCODE PROBE  tag={tag}  model={MODEL_ID}\n{SEP}\n"]
    for i, p in enumerate(PROMPTS):
        prompt = f"{p}\n\n```python\n"
        cont = greedy(model, tok, prompt, GEN_TOKENS)
        block = f"\n[{i}] PROMPT: {p}\n--- completion ---\n```python\n{cont}\n"
        print(block, flush=True)
        lines.append(block)
        results.append({"i": i, "prompt": p, "completion": cont})

    (OUTPUT_DIR / f"code_probe_{tag}.txt").write_text("".join(lines), encoding="utf-8")
    (OUTPUT_DIR / f"code_probe_{tag}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUTPUT_DIR}/code_probe_{tag}.txt (+ .json)")


if __name__ == "__main__":
    main()
