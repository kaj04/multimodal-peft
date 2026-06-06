# Launch Janus-Pro-7B training + chained qualitative eval

## 1. Training (7B, Naruto, hybrid prefix, cosine LR, 15 epochs)

```
cd ~/multimodal-peft
sbatch janus/train/02_run_train_generation_7b.sh
```

Watch progress:
```
squeue -u $USER
tail -f ~/multimodal-peft/janus/train/output/slurm_<JOBID>.out
```

W&B project: `janus-peft`, run name: `gen_7b_naruto_r8_prefix_cosine`.

Final adapter path:
```
~/multimodal-peft/janus/train/output/gen_7b_naruto_r8_prefix_cosine/adapter_epoch15
```

## 2. Qualitative eval, chained with --dependency=afterok

Grab the training job id (e.g. `12345678`) from `sbatch` output, then:

```
sbatch --dependency=afterok:12345678 --export=ALL,MODEL_ID=deepseek-community/Janus-Pro-7B,ADAPTER=$HOME/multimodal-peft/janus/train/output/gen_7b_naruto_r8_prefix_cosine/adapter_epoch15,CAPTION_PREFIX="anime sprite in Naruto style, " janus/eval/05_run_pokemon_qualitative.sh
```

(The qualitative job starts ONLY if the training job ends with `COMPLETED`.)

Outputs:
```
~/multimodal-peft/janus/tests/output/naruto_qual_7b/
  eval_00_natural_pair.png   eval_00_prefix_pair.png
  eval_01_natural_pair.png   eval_01_prefix_pair.png
  ...
  human_04_natural_pair.png  human_04_prefix_pair.png
  naruto_qual_results.json
```

20 pair PNGs total (10 base prompts x 2 prefix modes), CLIPScore aggregated
by category in `naruto_qual_results.json`.

## 3. If the training job hits the 14h time limit

Just re-sbatch the same script - RESUME=1 picks up from `latest/`:

```
sbatch janus/train/02_run_train_generation_7b.sh
```

## 4. To override hand-written prompts at eval time

Edit `HUMAN_PROMPTS` list inside `janus/eval/05_pokemon_qualitative.py` (top
of file). No env-var override - the prompts are intentionally hard-coded so
the experiment is reproducible from the script alone.
