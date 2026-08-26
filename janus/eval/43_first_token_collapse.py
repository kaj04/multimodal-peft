#!/usr/bin/env python3
"""
janus/eval/43_first_token_collapse.py  --  why the damaged chain models emit
nothing.

The qualitative run (08) returns an empty continuation for chainB_s2_naruto on
all six free-generation prompts, and one or two tokens for chainA_s2_naruto.
That is not a decoding bug: 08's greedy loop breaks on the end-of-sequence
token, so an empty continuation means the argmax at the very first step was EOS.
The model does not emit noise, it declares the text finished.

Writing that in the paper needs the number behind it, not the inference. This
prints, for each prompt, the top-5 next-token distribution and the probability
and rank the model assigns to EOS, for the base and the adapter side by side in
one process.

Reading it:
  P(EOS) near 1 at rank 1 for the adapter and near 0 for the base
      -> the collapse is real and specific: sequence-termination has taken over
  EOS nowhere near the top yet 08 returned an empty string
      -> the two scripts disagree and something is wrong; do not report either

  ADAPTER=/path/to/adapter_best python janus/eval/43_first_token_collapse.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER = os.environ.get("ADAPTER", "")
TOPK = int(os.environ.get("TOPK", "5"))
OUT = Path(os.path.expanduser(
    "~/multimodal-peft/janus/tests/output")) / "first_token_collapse"

N_PROMPTS = 6
SEP = "=" * 72


def load_from_08():
    """Reuse 08's prompts and its logits helper, so this cannot drift from the
    run whose output it is explaining."""
    src = HERE / "08_qualitative_text_examples.py"
    spec = importlib.util.spec_from_file_location("qual08", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if len(mod.FREEGEN_PROMPTS) != N_PROMPTS:
        raise SystemExit(
            f"  {src.name} has {len(mod.FREEGEN_PROMPTS)} prompts, expected "
            f"{N_PROMPTS}; update N_PROMPTS here if that change was intended.")
    return mod.FREEGEN_PROMPTS, mod.text_logits


@torch.no_grad()
def first_step(model, tok, text_logits, prompt, eos_id):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    logits = text_logits(model, ids)[0, -1].float()
    probs = F.softmax(logits, dim=-1)
    top_p, top_i = probs.topk(TOPK)
    order = probs.argsort(descending=True)
    eos_rank = int((order == eos_id).nonzero()[0].item()) + 1
    return {
        "top": [(tok.decode([int(i)]), round(float(p), 4))
                for p, i in zip(top_p, top_i)],
        "argmax_is_eos": int(top_i[0].item()) == eos_id,
        "p_eos": round(float(probs[eos_id].item()), 6),
        "eos_rank": eos_rank,
    }


def main():
    if not ADAPTER:
        raise SystemExit("set ADAPTER=/path/to/adapter_best")
    OUT.mkdir(parents=True, exist_ok=True)
    prompts, text_logits = load_from_08()

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel

    proc = JanusProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("  tokenizer has no eos_token_id; 08's break never "
                         "fires and the empty continuations mean something else")

    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    print(f"{SEP}\n  first-token distribution | ADAPTER={ADAPTER}\n"
          f"  eos_token_id={eos_id} ({tok.decode([eos_id])!r})\n{SEP}", flush=True)

    records = []
    for p in prompts:
        with model.disable_adapter():
            b = first_step(model, tok, text_logits, p, eos_id)
        a = first_step(model, tok, text_logits, p, eos_id)
        records.append({"prompt": p, "base": b, "adapter": a})
        print(f"\n  {p!r}")
        for name, r in (("base", b), ("adapter", a)):
            top = "  ".join(f"{t!r}:{v}" for t, v in r["top"])
            print(f"    {name:<8} P(eos)={r['p_eos']:<10} rank={r['eos_rank']:<6} "
                  f"argmax_is_eos={r['argmax_is_eos']}")
            print(f"    {'':<8} top: {top}")
        sys.stdout.flush()

    n_eos = sum(r["adapter"]["argmax_is_eos"] for r in records)
    mean_p = sum(r["adapter"]["p_eos"] for r in records) / len(records)
    base_eos = sum(r["base"]["argmax_is_eos"] for r in records)
    print(f"\n{SEP}")
    print(f"  adapter: EOS is the argmax on {n_eos}/{len(records)} prompts, "
          f"mean P(eos)={mean_p:.4f}")
    print(f"  base   : EOS is the argmax on {base_eos}/{len(records)} prompts")
    print(SEP)

    tag = Path(ADAPTER).parent.name
    (OUT / f"{tag}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"  saved -> {OUT / (tag + '.json')}")


if __name__ == "__main__":
    main()
