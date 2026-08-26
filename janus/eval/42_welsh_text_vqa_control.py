#!/usr/bin/env python3
"""
janus/eval/42_welsh_text_vqa_control.py  --  text-only twin of the Welsh VQA probe.

The VQA probe (28) gives the model an image, which carries no language, plus a
question in Welsh, and finds accuracy at chance. That result is read in the paper
as a failure of cross-modal transfer. It only means that if the model can read a
Welsh question in the first place. If it cannot, the probe measures Welsh
instruction-following, which fails in plain text too, and nothing about the
visual pathway.

This script replaces the image with one English sentence and keeps everything
else identical: the same 28 items imported from 28_xmodal_welsh_vqa.py, the same
questions verbatim, the same answer format, and the same scoring of Yes against
No at the first answer token. The evidence is unambiguous and in English, so the
only thing that decides whether we are asking about the apple or the banana is
the Welsh question itself.

Reading the four conditions:
  base_en      should be near ceiling; if it is not, the probe is broken
  base_cy      whether the base reads Welsh questions at all
  adapter_cy   the number that matters

  adapter improves here but not on images -> the failure is genuinely cross-modal
  adapter improves in neither             -> the failure is Welsh instruction-
                                             following, present already in text

  ADAPTER=/path/to/adapter_best python janus/eval/42_welsh_text_vqa_control.py
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER = os.environ.get("ADAPTER", "")
ANSWER_FMT = os.environ.get("ANSWER_FMT", "\nAnswer with 'Yes' or 'No'.")
# The questions say "in the picture" / "yn y llun", so the sentence has to
# establish a picture for them to stay verbatim.
PASSAGE_FMT = os.environ.get("PASSAGE_FMT", "The picture shows {}.")
OUT = Path(os.path.expanduser(
    "~/multimodal-peft/janus/tests/output")) / "welsh_text_vqa_control"

# The shape of the VQA item list, asserted at load so the two probes cannot
# silently diverge: 14 passages x 2 balanced yes/no questions.
N_ITEMS = 14
N_QUESTIONS = 28

Z = 1.959963984540054
SEP = "=" * 64


def load_items():
    """Import ITEMS from the VQA script, so the two probes cannot drift apart.

    The module name starts with a digit and cannot be imported by name.

    Two hazards, both silent. The VQA script truncates ITEMS to four at module
    level when SMOKE=1, and we submit with --export=ALL, so a stray SMOKE in the
    submitting shell would quietly shrink this control to a fifth of its size
    while still producing a plausible-looking table. And if anyone edits the
    item list on one side only, the two probes stop being comparable and nothing
    says so. So: force SMOKE off across the import, and assert the shape.
    """
    src = HERE / "28_xmodal_welsh_vqa.py"
    saved = os.environ.pop("SMOKE", None)
    try:
        spec = importlib.util.spec_from_file_location("vqa28", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        items = mod.ITEMS
    finally:
        if saved is not None:
            os.environ["SMOKE"] = saved
    n_q = sum(len(qs) for _, qs in items)
    if (len(items), n_q) != (N_ITEMS, N_QUESTIONS):
        raise SystemExit(
            f"  {src.name} has {len(items)} items / {n_q} questions, expected "
            f"{N_ITEMS}/{N_QUESTIONS}. The two probes must run the same set; "
            f"update N_ITEMS/N_QUESTIONS here if the change was intended.")
    return items


def wilson(k, n, z=Z):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h, c + h)


def _first_id(tok, s):
    return tok.encode(s, add_special_tokens=False)[0]


@torch.no_grad()
def yesno_text(model, proc, passage, question):
    """Identical to yesno() in 28, with the image replaced by a sentence."""
    text = passage + "\n" + question
    msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=prompt, return_tensors="pt").to(model.device)
    out = model(**inputs)
    logits = (out.logits if hasattr(out, "logits") else out[0])[0, -1].float()
    logp = F.log_softmax(logits, dim=-1)
    tok = proc.tokenizer
    lp_yes = max(logp[_first_id(tok, t)].item() for t in (" Yes", "Yes", " yes"))
    lp_no = max(logp[_first_id(tok, t)].item() for t in (" No", "No", " no"))
    return "yes" if lp_yes > lp_no else "no"


def main():
    if not ADAPTER:
        raise SystemExit("set ADAPTER=/path/to/adapter_best")
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()

    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel

    n_items = sum(len(qs) for _, qs in items)
    print(f"{SEP}\n  Text-only control for the Welsh VQA probe\n"
          f"  ADAPTER={ADAPTER}\n  {len(items)} passages, {n_items} questions\n{SEP}",
          flush=True)

    proc = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    # Show exactly what the model is asked, once, before the loop. The image
    # branch of the processor is the well-trodden one here; if the text-only
    # branch misbehaves it should be visible in the first seconds and in a form
    # we can paste into the appendix, not inferred from a strange final table.
    p0, qs0 = items[0]
    demo_passage = PASSAGE_FMT.format(p0)
    print(f"{SEP}\n  first item verbatim\n{SEP}")
    print(f"  passage : {demo_passage!r}")
    print(f"  en      : {(qs0[0][0] + ANSWER_FMT)!r}")
    print(f"  cy      : {(qs0[0][1] + ANSWER_FMT)!r}")
    print(f"  gold    : {qs0[0][2]}")
    a_en = yesno_text(model, proc, demo_passage, qs0[0][0] + ANSWER_FMT)
    a_cy = yesno_text(model, proc, demo_passage, qs0[0][1] + ANSWER_FMT)
    print(f"  adapter answers -> en={a_en} cy={a_cy}\n{SEP}", flush=True)

    conds = ["base_en", "base_cy", "adapter_en", "adapter_cy"]
    correct = {c: 0 for c in conds}
    total = 0
    records = []

    for prompt, qs in items:
        passage = PASSAGE_FMT.format(prompt)
        for (q_en, q_cy, gold) in qs:
            total += 1
            rec = {"passage": passage, "q_en": q_en, "q_cy": q_cy, "gold": gold}
            with model.disable_adapter():
                rec["base_en"] = yesno_text(model, proc, passage, q_en + ANSWER_FMT)
                rec["base_cy"] = yesno_text(model, proc, passage, q_cy + ANSWER_FMT)
            rec["adapter_en"] = yesno_text(model, proc, passage, q_en + ANSWER_FMT)
            rec["adapter_cy"] = yesno_text(model, proc, passage, q_cy + ANSWER_FMT)
            for c in conds:
                correct[c] += (rec[c] == gold)
            records.append(rec)
        print(f"  done: {prompt[:46]!r}", flush=True)

    lines = [SEP, "  Text-only control: same questions, sentence instead of image",
             f"  n_questions={total}", SEP]
    for c in conds:
        k = correct[c]
        lo, hi = wilson(k, total)
        lines.append(f"  {c:<12} {100*k/total:6.1f}%  CI[{100*lo:.0f},{100*hi:.0f}]")
    gap = 100 * (correct["base_en"] - correct["base_cy"]) / total
    tr = 100 * (correct["adapter_cy"] - correct["base_cy"]) / total
    nh = 100 * (correct["adapter_en"] - correct["base_en"]) / total
    lines += ["-" * 64,
              f"  Welsh-comprehension gap in text (base_en - base_cy) : {gap:+.1f} pts",
              f"  adapter effect on Welsh questions (cy - cy)         : {tr:+.1f} pts",
              f"  no-harm control (adapter_en - base_en)              : {nh:+.1f} pts",
              SEP]
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "results.txt").write_text(text + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(
        {"n": total, "correct": correct, "records": records}, indent=2),
        encoding="utf-8")
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
