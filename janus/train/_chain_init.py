"""
janus/train/_chain_init.py

Shared helper for SEQUENTIAL (continual) fine-tuning: start a run from the LoRA
weights of a PREVIOUS run instead of from a fresh, zero-initialised adapter.

This is deliberately NOT the same thing as RESUME. RESUME continues an
interrupted run of the *same* task: it restores optimizer, scheduler, RNG and
epoch counter from `<run>/latest`. INIT_ADAPTER starts a *new* task from
previously learned weights, with the optimizer, scheduler and epoch counter
reset. Keeping the two separate matters here, because reusing a RUN_NAME with
RESUME=1 has previously contaminated a baseline in this project.

The failure mode we guard against is silent: if the weights are not actually
loaded, training simply starts from scratch and every log line still looks
normal, so a whole chain would run and produce meaningless retention numbers.
We therefore verify by construction. PEFT initialises `lora_B` to zeros, so a
fresh adapter has an exactly zero B and hence a zero-magnitude update; a trained
adapter does not. Checking that the mean |lora_B| is zero before the load and
non-zero after it is a decisive test that costs nothing.

Both 02_train_generation.py and 03_train_text.py import this, so the two
directions inherit the same behaviour and the same checks.
"""
import json
from pathlib import Path


def lora_b_absmean(model):
    """Mean |lora_B| over the adapter. Exactly 0.0 for a freshly injected
    adapter (PEFT zero-initialises B), > 0 once real weights are loaded."""
    total, n = 0.0, 0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            total += float(p.detach().abs().sum().item())
            n += p.numel()
    return total / max(n, 1)


def _read_adapter_config(init_dir):
    cfg_path = Path(init_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"INIT_ADAPTER={init_dir} has no adapter_config.json. "
            f"Point it at a directory saved by save_pretrained() "
            f"(e.g. <run>/adapter_best or <run>/adapter_epoch3).")
    return json.loads(cfg_path.read_text())


def init_from_adapter(peft_model, init_dir, expect_rank=None,
                      expect_modules=None, verbose=True):
    """Load LoRA weights from `init_dir` into an already-injected peft_model.

    The structural configuration of the current run must match the saved one,
    otherwise the chain would silently compare adapters of different shapes.
    Any mismatch aborts here rather than at analysis time.
    """
    init_dir = str(Path(init_dir).expanduser())
    saved = _read_adapter_config(init_dir)

    saved_r = saved.get("r")
    saved_modules = saved.get("target_modules")
    if isinstance(saved_modules, (list, tuple, set)):
        saved_modules = sorted(str(m) for m in saved_modules)

    if expect_rank is not None and saved_r is not None and int(saved_r) != int(expect_rank):
        raise ValueError(
            f"rank mismatch: this run asks for r={expect_rank} but "
            f"{init_dir} was saved with r={saved_r}. A chain must keep one "
            f"rank throughout; the adapter shape cannot change mid-sequence.")

    if expect_modules is not None and saved_modules is not None:
        want = sorted(str(m) for m in expect_modules)
        if want != saved_modules:
            raise ValueError(
                f"target_modules mismatch:\n  this run : {want}\n"
                f"  saved    : {saved_modules}\n"
                f"The two adapters do not touch the same projections.")

    before = lora_b_absmean(peft_model)

    from peft import load_peft_weights, set_peft_model_state_dict
    state = load_peft_weights(init_dir)
    result = set_peft_model_state_dict(peft_model, state)

    after = lora_b_absmean(peft_model)
    if after <= 0.0:
        raise RuntimeError(
            f"INIT_ADAPTER load produced a still-zero lora_B "
            f"(before={before:.3e}, after={after:.3e}). The weights were NOT "
            f"applied; training would silently start from scratch. Check that "
            f"{init_dir} contains adapter_model.safetensors and that the PEFT "
            f"version matches the one that saved it.")

    if verbose:
        unexpected = getattr(result, "unexpected_keys", None)
        print(f"  INIT_ADAPTER  <- {init_dir}", flush=True)
        print(f"    saved r={saved_r}  modules={len(saved_modules or [])}", flush=True)
        print(f"    mean|lora_B|  before={before:.3e}  after={after:.3e}  "
              f"(non-zero => weights really loaded)", flush=True)
        if unexpected:
            print(f"    NOTE unexpected keys ignored: {len(unexpected)}", flush=True)
    return after
