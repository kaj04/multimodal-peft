"""Did the SigLIP adapters ever receive gradient?

A reviewer of the workshop paper pointed out that SigLIP only runs when an
image is encoded for *understanding*. A text run sees no images at all, and an
image-generation run goes through the VQ tokenizer, so the 72 SigLIP modules
that suffix matching adapts may never be trained. If so, the claim that the
understanding pathway "was not protected by construction" is weaker than it
sounds: it was simply never exercised.

PEFT initialises lora_B to exactly zero, so this needs no model, no data and no
GPU. A module whose lora_B is still all-zero in the saved adapter never took a
step. Reads the adapter file directly; runs on a login node in seconds.

    python 45_siglip_update_check.py <adapter_dir_or_file> [more...]
"""
import argparse
import os
import sys

try:
    from safetensors.torch import load_file
except ImportError:
    sys.exit("safetensors not installed: conda activate janus")

VISION_HINTS = ("vision", "siglip", "vit")


def group_of(key):
    k = key.lower()
    return "SigLIP" if any(h in k for h in VISION_HINTS) else "backbone"


def adapter_file(path):
    if os.path.isfile(path):
        return path
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        cand = os.path.join(path, name)
        if os.path.exists(cand):
            return cand
    return None


def report(path):
    f = adapter_file(path)
    if f is None:
        print("%s\n  no adapter_model.safetensors here" % path)
        return
    state = load_file(f)
    groups = {}
    for key, tensor in state.items():
        if "lora_B" not in key:
            continue
        groups.setdefault(group_of(key), []).append(
            (key, float(tensor.abs().mean())))

    print("\n%s" % path)
    for name in ("backbone", "SigLIP"):
        rows = groups.get(name)
        if not rows:
            print("  %-9s no lora_B tensors matched" % name)
            continue
        values = [v for _, v in rows]
        nonzero = sum(1 for v in values if v > 0)
        print("  %-9s modules=%3d  nonzero=%3d  mean|lora_B|=%.3e"
              % (name, len(rows), nonzero, sum(values) / len(values)))

    if not groups.get("SigLIP"):
        # The vision modules are named something this script did not expect.
        # Print a few keys so the naming can be fixed rather than guessed.
        keys = [k for k in state if "lora_B" in k][:6]
        print("  sample lora_B keys, to check the naming:")
        for k in keys:
            print("    %s" % k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="+",
                    help="adapter directories (or .safetensors files)")
    for p in ap.parse_args().adapters:
        report(p)
