# Earlier trials

Janus-Pro was the fourth model we tried. The three before it are kept here because the
reasons they failed are what made Janus-Pro the right choice, and because two of those
reasons are properties of the models rather than mistakes we made.

None of this code is needed to reproduce the paper. For that, see
[`../janus/README.md`](../janus/README.md).

## What each attempt was

**`4m/`** — 4M-21-XL (EPFL and Apple), an encoder-decoder transformer with per-modality VQ
codes. We ran LoRA across five tasks: captioning, semantic segmentation, detection, depth
and surface normals, with single-task and multi-task variants. The work produced usable
cross-task transfer measurements, but we could not reproduce the reference numbers the
original paper reports on our evaluation setup, which left every comparison resting on an
unverified baseline. Contains the training script, the evaluation suite, the SLURM jobs,
the configs and the visualisation notebooks, plus `ml-4m` as a submodule.

**`chameleon/`** — Chameleon-7B (Meta), a single transformer over an interleaved text and
image token stream, which is the architecture closest to the question we wanted to ask.
Image generation is disabled in the released weights, so only half the model was reachable
and cross-modal forgetting could not be measured at all.

**`anole/`** — Anole, which re-enables Chameleon's generation path. It is a custom fork
rather than a `transformers` model: generation runs through a subprocess and the weights
are not exposed as a standard `nn.Module`, so LoRA could not be attached to them.

**`drafts/`** — LaTeX working drafts and the running logbook kept during the project. The
submitted paper is not here; it is the version on arXiv.

## Why Janus-Pro

It is the only one of the four that is a native `transformers` model with both directions
enabled and weights exposed for PEFT, and the only one with separate encoders for
understanding and generation over a shared backbone. That last property is what makes the
question in the paper askable: damage can be traced to the shared weights rather than
confounded with a shared encoder.

## Running any of this

The SLURM scripts under `4m/scripts/` set `REPO_ROOT="$HOME/4m-peft"`, the repository's
former name, and resolve `cfgs/` relative to it. They are kept as a record of what was run
and would need their paths updated before they would execute again.
