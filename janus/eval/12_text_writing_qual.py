#!/usr/bin/env python3
"""
janus/eval/12_text_writing_qual.py

QUALITATIVE writing test for the poetry text-LoRA: did the adapter actually
learn the poetic register we tried to transfer? For a fixed set of prompts we
generate base-vs-adapter continuations SIDE BY SIDE, so the stylistic shift is
readable at a glance. No metric, no judge -- pure human inspection.

Two prompt families (both written here, no official benchmark needed):
  - INCIPITS to continue: the LoRA is a pure next-token LM (not instruction-
    tuned), so giving a seed and letting it continue is the faithful probe.
  - NEUTRAL register probes: plain sentences/openings; if the *default* voice
    has drifted poetic even without a poetic seed, that's the strongest signal.

For each prompt we show:
  - GREEDY (deterministic, reproducible) base vs adapter
  - one SAMPLED (temperature) base vs adapter, to surface style that greedy
    flattens.

Base = the SAME model with the adapter disabled (disable_adapter()), so the
only thing that changes between the two columns is the LoRA. Saves a readable
.txt (side by side) + a .json.

Run via 12_run_text_writing_qual.sh (GPU, ~3 min). Needs ADAPTER=...
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
SEED     = int(os.environ.get("SEED", "42"))
GEN_TOKENS  = int(os.environ.get("GEN_TOKENS", "70"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.9"))
TOP_P       = float(os.environ.get("TOP_P", "0.95"))
REP_PENALTY = float(os.environ.get("REP_PENALTY", "1.3"))

# Poetic priming for INCIPITS only (not neutral probes): a generic verse
# couplet biases the unified model into continuing a POEM rather than slipping
# into its chat/assistant persona ("Hello, I am..."), the main artifact in
# run 1. Same priming for base and adapter so the comparison stays fair; it is
# context only -> we decode/show just the continuation after the incipit.
POEM_PRIMER = os.environ.get(
    "POEM_PRIMER",
    "O sing, my soul, and let the verses flow,\n"
    "of things that were, and things we cannot know.\n")

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "text_writing_qual" / (Path(ADAPTER).parent.name if ADAPTER else "base")

# Incipits to continue (poetic seeds).
INCIPITS = [
    "The autumn wind",
    "O gentle heart, what",
    "Upon the silent hill there",
    "I wandered lonely as",
    "Beneath the ancient oak,",
]
# Neutral register probes (plain, non-poetic openings): if the adapter turns
# these poetic, the default voice itself shifted.
NEUTRAL = [
    "The weather today is",
    "To make a cup of tea,",
    "The meeting will start at",
]
SEP = "=" * 72


@torch.no_grad()
def _logits(model, ids):
    out = model(input_ids=ids)
    return out.logits if hasattr(out, "logits") else out[0]


def _rep_penalty(logits, seq_ids, penalty):
    """HF-style repetition penalty on a (V,) logit vector for tokens already in
    seq_ids. Kills the greedy looping seen in run 1."""
    if penalty == 1.0:
        return logits
    uniq = torch.unique(seq_ids)
    vals = logits[uniq]
    logits = logits.clone()
    logits[uniq] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


@torch.no_grad()
def greedy(model, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    start = ids.shape[1]
    for _ in range(n):
        logits = _rep_penalty(_logits(model, ids)[0, -1].float(), ids[0], REP_PENALTY)
        nxt = int(torch.argmax(logits).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
    return tok.decode(ids[0, start:].tolist(), skip_special_tokens=True)


@torch.no_grad()
def sample(model, tok, prompt, n, temperature, top_p, seed):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    start = ids.shape[1]
    for _ in range(n):
        logits = _rep_penalty(_logits(model, ids)[0, -1].float(), ids[0], REP_PENALTY)
        logits = logits / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, dim=-1) <= top_p
        keep[0] = True                       # always keep the top token
        sp, si = sp[keep], si[keep]
        nxt = int(si[torch.multinomial(sp / sp.sum(), 1)].item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
    return tok.decode(ids[0, start:].tolist(), skip_special_tokens=True)


def _both(model, tok, prompt, kind):
    """Return dict with base/adapter greedy + sampled for one prompt. Incipits
    get the poem primer (hidden context); neutral probes stay bare."""
    primer = POEM_PRIMER if kind == "incipit" else ""
    full = primer + prompt
    seed = SEED + (sum(ord(c) for c in prompt) % 1000)   # deterministic
    out = {"prompt": prompt, "kind": kind, "primed": bool(primer)}
    # adapter on
    out["adapter_greedy"] = greedy(model, tok, full, GEN_TOKENS)
    out["adapter_sample"] = sample(model, tok, full, GEN_TOKENS,
                                   TEMPERATURE, TOP_P, seed)
    # adapter off (base)
    with model.disable_adapter():
        out["base_greedy"] = greedy(model, tok, full, GEN_TOKENS)
        out["base_sample"] = sample(model, tok, full, GEN_TOKENS,
                                    TEMPERATURE, TOP_P, seed)
    return out


def _fmt(rec):
    return (
        f"\n{'-'*72}\n[{rec['kind']}] PROMPT: {rec['prompt']!r}\n"
        f"\n  GREEDY  base    | {rec['base_greedy']}\n"
        f"  GREEDY  adapter | {rec['adapter_greedy']}\n"
        f"\n  SAMPLE  base    | {rec['base_sample']}\n"
        f"  SAMPLE  adapter | {rec['adapter_sample']}\n"
    )


def main():
    if not ADAPTER:
        raise SystemExit("set ADAPTER=/path/to/adapter_best")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  Writing qualitative | MODEL={MODEL_ID}")
    print(f"  ADAPTER={ADAPTER}")
    print(SEP + "\n", flush=True)

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    records = []
    for p in INCIPITS:
        rec = _both(model, tok, p, "incipit")
        records.append(rec); print(_fmt(rec), flush=True)
    for p in NEUTRAL:
        rec = _both(model, tok, p, "neutral")
        records.append(rec); print(_fmt(rec), flush=True)

    header = (f"{SEP}\nWRITING QUALITATIVE  adapter={ADAPTER}\n"
              f"greedy + sample(T={TEMPERATURE}, top_p={TOP_P}), "
              f"{GEN_TOKENS} tokens, rep_penalty={REP_PENALTY}\n"
              f"incipits are poem-primed (context hidden); neutral probes bare\n{SEP}\n")
    (OUTPUT_DIR / "writing_qual.txt").write_text(
        header + "".join(_fmt(r) for r in records), encoding="utf-8")
    (OUTPUT_DIR / "writing_qual.json").write_text(
        json.dumps({"model": MODEL_ID, "adapter": ADAPTER,
                    "temperature": TEMPERATURE, "top_p": TOP_P,
                    "gen_tokens": GEN_TOKENS, "rep_penalty": REP_PENALTY,
                    "poem_primer": POEM_PRIMER, "records": records}, indent=2),
        encoding="utf-8")
    print(f"\n  saved -> {OUTPUT_DIR}/writing_qual.txt (+ .json)")


if __name__ == "__main__":
    main()
