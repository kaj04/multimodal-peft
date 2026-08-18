#!/usr/bin/env python3
"""
janus/eval/41_welsh_prompt_probe.py  --  does the Welsh prompt reach the
generator at all?

The cross-modal probe (27) produces visibly the same motif for every Welsh
prompt, whatever the prompt asks for. Two very different things produce that
picture. Either the model conditions on the Welsh text and gets the content
wrong, which is what the paper claims, or the Welsh contributes nothing and the
image is decided by the seed alone, in which case the claim is far stronger than
the one we currently make and has to be written differently.

The two are separated by holding the seed fixed and varying only the prompt. If
two Welsh prompts with unrelated content give the same image, the text is not
being used. If they differ from each other, it is.

We also print a tokenizer round-trip first, so that a silent encoding failure --
diacritics dropped, everything collapsing to a single unknown token -- cannot be
mistaken for a property of the model.

Twelve generations, roughly half an hour on one H100.

  ADAPTER=/path/to/adapter_best python janus/eval/41_welsh_prompt_probe.py
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import generate_image  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER = os.environ["ADAPTER"]
SEED = int(os.environ.get("SEED", "42"))
OUT = Path(os.path.expanduser(
    "~/multimodal-peft/janus/tests/output/welsh_prompt_probe"))

# Three Welsh prompts whose content is unrelated to one another, the two English
# counterparts as a positive control, and the empty string as the floor: if the
# Welsh cases coincide with `empty`, the text carries no signal at all.
CASES = {
    "en_fox":  "a red fox sitting in a snowy forest",
    "en_cake": "a slice of chocolate cake on a plate",
    "cy_fox":  "llwynog coch yn eistedd mewn coedwig eira",
    "cy_cake": "tafell o gacen siocled ar blât",
    "cy_bus":  "bws ysgol melyn ar stryd y ddinas",
    "empty":   "",
}

SEP = "=" * 64


def main():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel

    OUT.mkdir(parents=True, exist_ok=True)
    proc = JanusProcessor.from_pretrained(MODEL_ID)

    print(SEP)
    print("  tokenizer round-trip: does the Welsh survive encoding?")
    print(SEP)
    for name, text in CASES.items():
        ids = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
        print(f"  {name:<8} n_tok={len(ids):>3}  {proc.tokenizer.decode(ids)!r}",
              flush=True)

    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    print(f"\n{SEP}")
    print("  generating; the seed is identical for every case, so any")
    print("  difference between two images comes from the prompt alone")
    print(SEP, flush=True)
    for name, text in CASES.items():
        with model.disable_adapter():
            torch.manual_seed(SEED)
            generate_image(model, proc, text).save(OUT / f"base_{name}.png")
        torch.manual_seed(SEED)
        generate_image(model, proc, text).save(OUT / f"adapter_{name}.png")
        print(f"    {name} done", flush=True)

    print(f"\n  saved -> {OUT}")
    print("  compare base_cy_fox / base_cy_cake / base_cy_bus against each")
    print("  other and against base_empty.")


if __name__ == "__main__":
    main()
