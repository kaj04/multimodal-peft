# Janus-Pro experiments

Everything the paper reports is produced here. The model is Janus-Pro-7B
(`deepseek-community/Janus-Pro-7B`), a unified model with one shared transformer backbone
and two separate modality encoders: SigLIP for understanding (image to text) and a
vector-quantized tokenizer for generation (text to image).

## Setup

```bash
source janus/env.sh
```

That activates the `janus` conda environment, points the Hugging Face cache at scratch
space rather than at the home quota, and leaves you in the repository root. All commands
below are run from there. Outside the cluster, `requirements.txt` in this folder lists what
the scripts import.

Datasets and model weights are downloaded on first use. Adapters and evaluation results
are written under `janus/train/output/` and `janus/tests/output/`, both ignored by git.

`tests/` holds the smoke tests that check the model loads and that LoRA gradients reach
the adapter in each direction. Run them first if something looks wrong.

## The fastest way in

```bash
python janus/eval/40_manifest.py
```

One row per adapter, every metric the paper quotes, read from the output tree rather than
retyped. If a number is in the paper it is in this table.

## Training

| Script | Direction | Used for |
|---|---|---|
| `train/02_train_generation.py` | text to image | the image adapters: anime style, line sketches |
| `train/03_train_text.py` | text to text | the text adapters: poetry, medicine, Welsh |
| `train/04_train_joint.py` | both at once | the sensitivity control with an alternating text stream |
| `train/03_rank_sweep.py` | — | drives the rank sweeps over an existing recipe |
| `train/_chain_init.py` | — | loads a previous run's adapter into a fresh model; this is what makes a chain a chain |

Each trainer has SLURM runners beside it (`02_run_*.sh`, `03_run_*.sh`). The configuration
that the paper calls *Final* for image generation is `02_run_train_generation_7b_v5.sh`.

Both trainers can anchor the adapter to the frozen base model with a KL penalty. In an
image run that penalty is computed on the generation head; in a text run, on the text head.
`TEXT_KL_LAMBDA` in `02_train_generation.py` adds the mirror term on text logits during an
image run, which is the intervention of the paper's last section. It defaults to zero, so
every earlier run is unaffected.

## Evaluation

| Script | Answers |
|---|---|
| `eval/06_metrics_panel.py` | did the image adapter learn the style? CMMD, KID, DINOv2, CLIP-IQA, CLIP directional similarity |
| `eval/35_target_ppl.py` | did the text adapter learn the target? held-out perplexity, any target, any adapter |
| `eval/07_forgetting_battery.py` | what happened to the capabilities we did not train? WikiText-2, MMLU, POPE, MME |
| `eval/30`, `31`, `32_mauve_*.py` | free-form generation quality, in three phases: generate, score, tabulate |
| `eval/18_weight_delta_norm.py` | how much did the adapter actually move the weights? |
| `eval/25_probe_language_targets.py` | which language target to pick, scored by distance from the base model |
| `eval/33_headroom.py` | how much of a benchmark score is real knowledge that could be lost, rather than luck |
| `eval/38_reg_retention_table.py` | the five-regularizer ablation, assembled into one table |
| `eval/45_siglip_update_check.py` | did the SigLIP adapters ever receive gradient? (they did not) |
| `eval/40_manifest.py` | every number in the paper, reconciled from the output tree |

Cross-modal transfer, the question of whether a language learned through text becomes
usable in the visual directions:

| Script | Probe |
|---|---|
| `eval/27_xmodal_welsh_t2i.py` | image generation from Welsh prompts, scored against the English reference |
| `eval/28_xmodal_welsh_vqa.py` | visual question answering with the question in Welsh |
| `eval/41_welsh_prompt_probe.py` | does the Welsh prompt reach the generator at all? |
| `eval/42_welsh_text_vqa_control.py` | the same questions with the image replaced by a sentence, which is what showed the probe measures instruction following |

## Running a whole block at once

The scripts at the top of `janus/` chain several SLURM jobs, so one experiment block can be
launched with a single command instead of submitting each step by hand.

| Script | Launches |
|---|---|
| `run_text_sweep.sh` | the poetry rank and KL sweeps |
| `run_text_peft_night.sh` | the poetry text adapter, training through evaluation |
| `run_text_welsh_night.sh` | the Welsh target: training, held-out perplexity, the forgetting probes |
| `run_xmodal_welsh.sh` | the cross-modal probes for the Welsh adapter |
| `run_mauve_all.sh` | MAUVE for the base model and every headline adapter |

## The sequential chain

The headline experiment. One adapter is trained three times in a row, each run starting
from the previous run's weights, in two mirrored orders.

```bash
bash janus/chain/run_both.sh
```

`chain/chain_config.sh` holds the frozen configuration. Nothing in it may differ between
the two chains: they must differ only in the order of the tasks, or order is confounded
with hyperparameters and the retention matrix cannot be read. `run_chain.sh` drives one
chain through `train_text_step.sh`, `train_image_step.sh` and `eval_step.sh`.

Reading the result:

| Script | Produces |
|---|---|
| `eval/36_chain_matrix.py` | the retention matrix: performance on every target after every step |
| `eval/44_chain_image_grid.py` | the image row of the chain figure, one prompt across all four stages |
| `eval/08_qualitative_text_examples.py` | the text row of the same figure, greedy continuations at each stage |
| `eval/43_first_token_collapse.py` | the next-token distribution that explains why the damaged model emits only whitespace |

## A note on the numbering

Scripts are numbered in the order they were written, not in the order the paper presents
them, and there are gaps where an approach was abandoned. The numbering is kept because
the SLURM runners, the logs and the output directories all refer to it. Use the tables
above rather than the numbers to find what you need.
