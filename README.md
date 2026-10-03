# Cross-Modal Forgetting and Transfer in Unified Multimodal Post-Training

Code and experiments for the paper, accepted at the **CL4FMAgents** workshop,
NeurIPS 2026.

*Paper: arXiv link to be added.*

A unified multimodal model answers questions about images and generates images with one
shared set of weights, so fine-tuning one capability can damage another that lives in a
different modality. We measure what LoRA post-training of Janus-Pro-7B costs the
capabilities it never trained.

A single fine-tuning run leaves the untrained modality intact, even when the adapter cuts
held-out Welsh perplexity by 76%. That safety belongs to the run and not to the recipe:
chaining three runs destroys what was learned one run earlier and pushes general-text
perplexity far below a model that was never fine-tuned. The cause is in the objective,
which regularizes only the modality being trained and leaves the other head unconstrained.
Adding a KL term on the untrained head removes the damage at no cost to acquisition.

## Reproducing the paper

Everything the paper reports comes from `janus/`.
**[janus/README.md](janus/README.md)** maps each table and figure to the script that
produces it, in the order the pipeline runs.

Model weights, LoRA adapters and evaluation outputs are not in the repository. They are
written to cluster scratch space by the training and evaluation scripts, and the
`.gitignore` keeps them out deliberately, so a fresh clone gives you the code and not the
results.

```bash
git clone https://github.com/kaj04/multimodal-peft
cd multimodal-peft
source janus/env.sh     # conda env, Hugging Face cache on scratch, repo as cwd
```

Once the experiments have run, `janus/eval/40_manifest.py` walks the output tree and prints
one row per adapter with every metric the paper quotes, so any number can be traced back to
the file it came from.

## What you need to run this

The model is Janus-Pro-7B, so the experiments need a single 80 GB GPU; the chain steps were
run on an H100 with 100 GB of host memory. Nothing here fits on a laptop, though all of it
can be read on one.

The SLURM scripts were written for Snellius, the Dutch national supercomputer. On another
cluster expect to change the partition names and the `module load` lines. Paths inside
`#SBATCH` directives are relative to the repository root, so submit from there, and create
the two log directories once before the first job:

```bash
mkdir -p janus/train/output janus/tests/output
```

## Layout

```
janus/            the paper: training, evaluation, and the sequential-chain experiment
earlier-trials/   models tried before Janus-Pro, kept for the record
```

## Earlier trials

Janus-Pro was the fourth model we tried. 4M-21, Chameleon and Anole each failed for a
different reason, and `earlier-trials/README.md` records what each attempt was and why it
was dropped. The code is kept rather than deleted: the reasons those models did not work
are part of why Janus-Pro is the right one for this question.

## Citation

```bibtex
@inproceedings{colasurdo2026crossmodal,
  title     = {Cross-Modal Forgetting and Transfer in Unified Multimodal Post-Training},
  author    = {Colasurdo, Francesco and Slobozhan, Ivan and Vanschoren, Joaquin and Vettoruzzo, Anna},
  booktitle = {NeurIPS 2026 Workshop on Continual Learning in the Era of Foundation Models
               and Embodied Agents (CL4FMAgents)},
  year      = {2026}
}
```

## Acknowledgements

Built on [Janus-Pro](https://huggingface.co/deepseek-community/Janus-Pro-7B) (DeepSeek).
Experiments were run on the Dutch national supercomputer Snellius.
