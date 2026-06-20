#!/usr/bin/env python3
"""
janus/tests/05_chat_3way.py

3-way interactive chat: every question is answered by all three models at once,
loaded together via PeftModel multi-adapter, so the only variable is the adapter:

  base    : Janus-Pro-7B, no adapter
  poetry  : text/poetry LoRA   (should answer facts like base; style may shift)
  naruto  : image/Naruto LoRA  (should answer like base -> understanding intact)

This is a by-hand check of the modularity claim: ask knowledge / understanding
questions and see whether the image adapter leaves text intact and whether the
poetry adapter keeps the facts right.

NO conversation memory: each question is an independent prompt (cleaner for a
controlled comparison). Run on an interactive GPU node:

  srun -p gpu_h100 --gres=gpu:h100:1 --time=01:00:00 --mem=80G --cpus-per-task=16 --pty bash
  module load 2023 Anaconda3/2023.07-2 ; source activate janus
  export HF_HOME=/scratch-shared/tesr125338/hf_cache
  python ~/multimodal-peft/janus/tests/05_chat_3way.py

Commands:
  /mode text|ask     text Q->A (no image)  |  image+question (understanding)
  /img <path>        set the image for `ask` mode
  /sample on|off     sampling vs greedy        [default greedy]
  /temp <f> /top_p <f> /tokens <n>
  /raw on|off        raw continuation vs chat template (text mode)
  /help  /quit
Anything else = the question, answered by base / poetry / naruto.
"""

import os
from pathlib import Path

import torch
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
TRAIN    = Path(os.path.expanduser("~/multimodal-peft/janus/train/output"))
POETRY   = os.environ.get("POETRY", str(TRAIN / "text_7b_poetry_v1" / "adapter_best"))
NARUTO   = os.environ.get("NARUTO", str(TRAIN / "gen_7b_naruto_v5"   / "adapter_best"))

SEP = "=" * 64


@torch.no_grad()
def run_text(model, processor, text, image, sample, temp, top_p, n_tok, raw):
    if raw:
        ids = processor.tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        plen = ids.shape[1]
        out = model.generate(input_ids=ids, max_new_tokens=n_tok,
                             generation_mode="text", do_sample=sample,
                             temperature=temp, top_p=top_p)
    else:
        content = ([{"type": "image"}] if image is not None else []) \
            + [{"type": "text", "text": text}]
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, generation_mode="text")
        kw = dict(text=prompt, generation_mode="text", return_tensors="pt")
        if image is not None:
            kw["images"] = [image]
        inputs = processor(**kw).to(model.device, torch.bfloat16)
        plen = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=n_tok,
                             generation_mode="text", do_sample=sample,
                             temperature=temp, top_p=top_p)
    return processor.decode(out[0][plen:], skip_special_tokens=True).strip()


def answer(model, processor, kind, *args):
    """One model's answer; `kind` in {base, poetry, naruto}."""
    if kind == "base":
        with model.disable_adapter():
            return run_text(model, processor, *args)
    model.set_adapter(kind)
    return run_text(model, processor, *args)


HELP = __doc__.split("Commands:")[1].split('"""')[0]


def main():
    print(f"  loading {MODEL_ID} + adapters ...", flush=True)
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    base.eval()
    model = PeftModel.from_pretrained(base, POETRY, adapter_name="poetry")
    model.load_adapter(NARUTO, adapter_name="naruto")
    model.eval()
    print(SEP)
    print("  3-way chat ready | base | poetry | naruto")
    print("  /help for commands, /quit to exit (no memory: each Q independent)")
    print(SEP, flush=True)

    mode = "text"
    image = None
    sample = False
    temp, top_p, n_tok = 0.9, 0.95, 128
    raw = False

    while True:
        try:
            line = input(f"({mode}) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye"); break
        if not line:
            continue
        if line.startswith("/"):
            cmd, *rest = line[1:].split(maxsplit=1)
            arg = rest[0].strip() if rest else ""
            if cmd in ("quit", "q", "exit"):
                print("  bye"); break
            elif cmd == "help":
                print(HELP)
            elif cmd == "mode" and arg in ("text", "ask"):
                mode = arg
            elif cmd == "img":
                try:
                    image = Image.open(os.path.expanduser(arg)).convert("RGB")
                    print(f"  image set: {arg}")
                except Exception as e:
                    print(f"  could not open image: {e}")
            elif cmd == "sample":
                sample = (arg != "off")
            elif cmd == "raw":
                raw = (arg != "off")
            elif cmd == "temp":
                temp = float(arg)
            elif cmd == "top_p":
                top_p = float(arg)
            elif cmd == "tokens":
                n_tok = int(arg)
            else:
                print(f"  ? unknown command '/{cmd}' (try /help)")
            continue

        if mode == "ask" and image is None:
            print("  set an image first with /img <path>")
            continue
        for kind in ("base", "poetry", "naruto"):
            try:
                ans = answer(model, processor, kind, line,
                             image if mode == "ask" else None,
                             sample, temp, top_p, n_tok, raw)
            except Exception as e:
                ans = f"<{type(e).__name__}: {e}>"
            print(f"  [{kind}]\n{ans}\n", flush=True)


if __name__ == "__main__":
    main()
