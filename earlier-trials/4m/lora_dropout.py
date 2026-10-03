"""
lora_dropout.py — post-injection LoRA dropout for 4M-PEFT.

ml-4m's LoRAWrapper does not support dropout.  We avoid modifying the
submodule (apple/ml-4m is read-only) and instead patch every LoRAWrapper
in-place after inject_trainable_LoRA() has already run.

The patch:
  1. Adds  module.lora_dropout = nn.Dropout(p)  as a proper submodule
     (so model.train() / model.eval() toggle it correctly).
  2. Swaps __class__ to a one-shot subclass whose forward() routes the
     low-rank path through the dropout layer:
        out = linear(x) + lora_up( dropout( lora_down(x) ) ) * scale

Usage:
    from lora_dropout import add_lora_dropout
    inject_trainable_LoRA(model, rank=4, ...)
    n = add_lora_dropout(model, p=0.1)
    print(f'Patched {n} LoRAWrapper layers with dropout={p}')
"""

import torch
import torch.nn as nn


def add_lora_dropout(model: nn.Module, p: float) -> int:
    """Patch all LoRAWrapper layers with dropout applied between lora_down and lora_up.

    Args:
        model : model that has already had inject_trainable_LoRA() applied.
        p     : dropout probability.  p <= 0 is a no-op.

    Returns:
        Number of LoRAWrapper layers patched.
    """
    if p <= 0.0:
        return 0

    n_patched = 0
    for module in model.modules():
        if module.__class__.__name__ != 'LoRAWrapper':
            continue

        # Register as a proper nn.Module submodule so it follows train/eval mode.
        module.lora_dropout = nn.Dropout(p=p)

        # Swap __class__ to a minimal subclass that adds dropout to forward().
        # We inherit from the original class so fuse_LoRA_into_linear() and
        # any other existing methods remain available.
        original_class = module.__class__

        class LoRAWrapperWithDropout(original_class):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x) + self.lora_up(self.lora_dropout(self.lora_down(x))) * self.scale

        LoRAWrapperWithDropout.__name__ = 'LoRAWrapperWithDropout'
        LoRAWrapperWithDropout.__qualname__ = 'LoRAWrapperWithDropout'
        module.__class__ = LoRAWrapperWithDropout
        n_patched += 1

    return n_patched
