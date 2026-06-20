#!/usr/bin/env python3
"""
janus/tests/03_interactive.py

Interactive REPL to qualitatively probe Janus-Pro-7B with an optional LoRA
adapter, BASE vs ADAPTER side by side, for sanity-checking the modularity claim
by hand. Three modes:

  text : text prompt  -> text answer/continuation (no image)
  ask  : image + question -> answer            (understanding direction)
  gen  : text prompt  -> image saved to disk   (generation direction)

The point of the modularity claim is cross-direction preservation, so this lets
you, e.g., load the IMAGE adapter and check by hand that `ask` (understanding)
and `text` (knowledge) still work, or load the TEXT adapter and check that `gen`
still produces sane images.

Run on an INTERACTIVE GPU node (no display, so images are saved to disk and
viewed in Open OnDemand):

  salloc -p gpu_h100 --gres=gpu:h100:1 --time=01:00:00 --mem=80G --cpus-per-task=16
  module load 2023 Anaconda3/2023.07-2
  source activate janus
  export HF_HOME=/scratch-shared/tesr125338/hf_cache
  ADAPTER=$HOME/multimodal-peft/janus/train/output/gen_7b_naruto_v5/adapter_best \
    python ~/multimodal-peft/janus/tests/03_interactive.py

Commands inside the REPL:
  /mode text|ask|gen     switch mode
  /img <path>            set the image used by `ask` mode
  /both on|off           compare base vs adapter (default: on if an adapter is loaded)
  /sample on|off         sampling vs greedy (text/ask)        [default greedy]
  /temp <f> /top_p <f>   sampling params
  /raw on|off            text mode: raw continuation vs chat template [default off]
  /tokens <n>            max new tokens for text             [default 128]
  /help   /quit
Anything else = prompt for the current mode.
"""

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# HPC terminals often default to a non-UTF-8 locale (C/ASCII), which makes
# input() crash on accented characters. Force UTF-8 on the I/O streams.
for _stream in (sys.stdin, sys.stdout):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "").strip()
OUT_DIR  = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"))) / "interactive"

SEP = "=" * 64


def load():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    print(f"  loading {MODEL_ID} ...", flush=True)
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    base.eval()
    peft = False
    if ADAPTER:
        from peft import PeftModel
        print(f"  loading adapter {ADAPTER} ...", flush=True)
        base = PeftModel.from_pretrained(base, ADAPTER)
        base.eval()
        peft = True
    return base, processor, peft


def adapter_ctx(model, on, peft):
    """Context where the adapter is active (on) or disabled (base)."""
    if on or not peft:
        return contextlib.nullcontext()
    return model.disable_adapter()


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


@torch.no_grad()
def run_gen(model, processor, text, out_path):
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        add_generation_prompt=True, generation_mode="image")
    inputs = processor(text=prompt, generation_mode="image",
                       return_tensors="pt").to(model.device)
    model.generation_config.num_return_sequences = 1
    outputs = model.generate(**inputs, generation_mode="image",
                             max_new_tokens=576, do_sample=True, use_cache=True)
    decoded = model.decode_image_tokens(outputs).detach().float().cpu()
    try:
        arr = np.asarray(processor.postprocess(
            list(decoded), return_tensors="np")["pixel_values"][0])
    except Exception:
        arr = decoded[0].numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if float(arr.min()) < -0.01:
        arr = (arr + 1.0) / 2.0
    elif float(arr.max()) > 1.01:
        arr = arr / 255.0
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(str(out_path))
    return out_path


HELP = __doc__.split("Commands inside the REPL:")[1].split('"""')[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor, peft = load()
    print(SEP)
    print(f"  ready | adapter={'ON' if peft else 'none'} | "
          f"images -> {OUT_DIR}")
    print(f"  type /help for commands, /quit to exit")
    print(SEP, flush=True)

    mode = "text"
    image = None
    both = peft           # compare base vs adapter when an adapter is loaded
    sample = False
    temp, top_p, n_tok = 0.9, 0.95, 128
    raw = False
    gi = 0                # generated-image counter

    def gen_for(tag, on):
        nonlocal gi
        with adapter_ctx(model, on, peft):
            if mode == "gen":
                p = OUT_DIR / f"{gi:03d}_{tag}.png"
                run_gen(model, processor, prompt, p)
                print(f"  [{tag}] saved -> {p}")
            else:
                ans = run_text(model, processor, prompt, image, sample,
                               temp, top_p, n_tok, raw)
                print(f"  [{tag}]\n{ans}\n")

    while True:
        try:
            line = input(f"({mode}{'/both' if both and peft else ''}) > ").strip()
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
            elif cmd == "mode" and arg in ("text", "ask", "gen"):
                mode = arg
            elif cmd == "img":
                try:
                    image = Image.open(os.path.expanduser(arg)).convert("RGB")
                    print(f"  image set: {arg}")
                except Exception as e:
                    print(f"  could not open image: {e}")
            elif cmd == "both":
                both = (arg != "off")
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

        prompt = line
        if mode == "ask" and image is None:
            print("  set an image first with /img <path>")
            continue
        try:
            if both and peft:
                gen_for("base", on=False)
                gen_for("adapter", on=True)
            else:
                gen_for("adapter" if peft else "base", on=True)
        except Exception as e:
            print(f"  generation failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
