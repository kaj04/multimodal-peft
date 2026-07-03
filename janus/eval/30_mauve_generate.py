#!/usr/bin/env python3
"""
janus/eval/30_mauve_generate.py   --   PHASE 1 (janus env, GPU)

Free-form TEXT generation for the MAUVE experiment. MAUVE ("FID for text",
Pillutla et al. 2021) needs two text SETS; here we generate the model set and
save the matching human reference set. MAUVE itself is computed in phase 2
(31_mauve_compute.py) inside a SEPARATE venv, so this script never installs
anything into the janus environment.

For one adapter (ADAPTER empty = base) we produce, on a fixed prompt prefix:
  * english : prompts = 32-token prefixes of held-out WikiText-2 passages. This
    is the free-form counterpart of the likelihood battery: does the adapter
    still generate base-/human-like ENGLISH? (forgetting / cross-modal forgetting)
  * <target>: (text adapters) prompts = prefixes of the target corpus
    (poetry / pubmed / welsh). MAUVE(gen || real target) is the text analogue of
    image CMMD-vs-real-style -> ACQUISITION at the generation-distribution level.

Prompts are fixed length (PROMPT_TOKENS) so generation is batched. References are
written once to a SHARED folder (mauve/_refs) and reused across adapters.

Env: ADAPTER, TAG, TARGETS="poetry,pubmed,welsh" (comma list; "" = english only),
GEN_ENGLISH=1, N, PROMPT_TOKENS, GEN_TOKENS, BATCH, TEMPERATURE, TOP_P, SMOKE.
Run via 30_run_mauve.sh.
"""
import json
import os
import time
from pathlib import Path

import torch

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
TAG      = os.environ.get("TAG") or (Path(ADAPTER).parent.name if ADAPTER else "base")
TARGETS  = [t.strip() for t in os.environ.get("TARGETS", "").split(",") if t.strip()]
GEN_ENGLISH = os.environ.get("GEN_ENGLISH", "1") == "1"
N        = int(os.environ.get("N", "2000"))
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "32"))
GEN_TOKENS    = int(os.environ.get("GEN_TOKENS", "96"))
BATCH    = int(os.environ.get("BATCH", "32"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P       = float(os.environ.get("TOP_P", "0.95"))
SEED     = int(os.environ.get("SEED", "42"))
WIKI_DATE = os.environ.get("WIKI_DATE", "20231101")
LOAD_RETRIES = int(os.environ.get("LOAD_RETRIES", "6"))
SMOKE    = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    N = 48; BATCH = 16

ROOT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output")) / "mauve"
OUT  = ROOT / TAG
REFS = ROOT / "_refs"
SEP = "=" * 64

CORPORA = {  # kind -> (hf spec) handled in corpus_texts()
    "english": None, "poetry": None, "pubmed": None, "welsh": None,
}


def _load_retry(*a, **k):
    from datasets import load_dataset
    last = None
    for i in range(1, LOAD_RETRIES + 1):
        try:
            return load_dataset(*a, **k)
        except Exception as e:
            last = e
            print(f"  [load {i}/{LOAD_RETRIES}] {type(e).__name__}: {e}", flush=True)
            time.sleep(min(60, 8 * i))
    raise last


def corpus_texts(kind):
    """Raw human strings for a corpus (streamed / bounded)."""
    need = N * 3
    if kind == "english":
        ds = _load_retry("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return [t for t in ds["text"] if len(t.split()) > 40]
    if kind == "poetry":
        ds = _load_retry("biglam/gutenberg-poetry-corpus", split="train", streaming=True)
        out, buf = [], []
        for r in ds:
            buf.append(str(r.get("line") or r.get("text") or ""))
            if len(buf) >= 14:
                out.append("\n".join(buf)); buf = []
            if len(out) >= need:
                break
        return out
    if kind == "pubmed":
        ds = _load_retry("ccdv/pubmed-summarization", split="train", streaming=True)
        out = []
        for r in ds:
            t = r.get("article") or r.get("abstract") or ""
            if isinstance(t, str) and len(t.split()) > 60:
                out.append(t)
            if len(out) >= need:
                break
        return out
    if kind == "welsh":
        ds = _load_retry("wikimedia/wikipedia", f"{WIKI_DATE}.cy", split="train",
                         streaming=True)
        out = []
        for r in ds:
            t = r.get("text", "")
            if isinstance(t, str) and len(t.split()) > 60:
                out.append(t)
            if len(out) >= need:
                break
        return out
    raise ValueError(kind)


def build_prompts_and_refs(kind, tok):
    """From the corpus, keep passages with >= L tokens; prompt = first
    PROMPT_TOKENS ids, ref = first L ids decoded. Returns (prompt_ids, ref_texts)."""
    import random
    L = PROMPT_TOKENS + GEN_TOKENS
    texts = corpus_texts(kind)
    random.Random(SEED).shuffle(texts)
    prompts, refs = [], []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) < L:
            continue
        prompts.append(ids[:PROMPT_TOKENS])
        refs.append(tok.decode(ids[:L], skip_special_tokens=True))
        if len(prompts) >= N:
            break
    return prompts, refs


@torch.no_grad()
def generate_set(model, tok, prompt_ids):
    outs = []
    L = PROMPT_TOKENS + GEN_TOKENS
    for i in range(0, len(prompt_ids), BATCH):
        chunk = prompt_ids[i:i + BATCH]
        inp = torch.tensor(chunk, dtype=torch.long, device=model.device)
        o = model.generate(input_ids=inp, generation_mode="text", do_sample=True,
                           top_p=TOP_P, temperature=TEMPERATURE,
                           max_new_tokens=GEN_TOKENS,
                           pad_token_id=(tok.eos_token_id or 0))
        for row in o:
            outs.append(tok.decode(row[:L], skip_special_tokens=True))
        if (i // BATCH) % 10 == 0:
            print(f"    gen {i+len(chunk)}/{len(prompt_ids)}", flush=True)
    return outs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REFS.mkdir(parents=True, exist_ok=True)
    from transformers import JanusForConditionalGeneration, JanusProcessor
    kinds = (["english"] if GEN_ENGLISH else []) + list(TARGETS)
    print(f"{SEP}\n  MAUVE gen | TAG={TAG}  kinds={kinds}  N={N}  "
          f"prompt={PROMPT_TOKENS} gen={GEN_TOKENS}\n  ADAPTER={ADAPTER or '(base)'}\n{SEP}",
          flush=True)
    proc = JanusProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
    model = base
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ADAPTER).eval()

    for kind in kinds:
        # references are model-independent -> build/save once in _refs
        ref_path = REFS / f"ref_{kind}.json"
        prm_path = REFS / f"prompts_{kind}.json"
        if ref_path.exists() and prm_path.exists():
            prompt_ids = json.loads(prm_path.read_text())
            print(f"  [{kind}] reusing shared refs/prompts ({len(prompt_ids)})", flush=True)
        else:
            print(f"  [{kind}] building prompts + refs ...", flush=True)
            prompt_ids, refs = build_prompts_and_refs(kind, tok)
            ref_path.write_text(json.dumps(refs), encoding="utf-8")
            prm_path.write_text(json.dumps(prompt_ids), encoding="utf-8")
            print(f"  [{kind}] {len(prompt_ids)} prompts, refs saved", flush=True)
        gens = generate_set(model, tok, prompt_ids)
        (OUT / f"gen_{kind}.json").write_text(json.dumps(gens), encoding="utf-8")
        print(f"  [{kind}] saved {len(gens)} generations -> {OUT}/gen_{kind}.json", flush=True)

    print(f"\n  done. gens in {OUT}, shared refs in {REFS}")


if __name__ == "__main__":
    main()
