# Rank sweep — scaling of the v5 configuration

Studies how the v5 recipe scales with LoRA rank. Everything is held fixed
at the v5 winning config; ONLY `LORA_RANK` (and `LORA_ALPHA = 2*rank`) varies.

| Knob | Value |
|---|---|
| Ranks | 2, 4, 8, 16, 32, 64 |
| alpha | 2 * rank (standard scaling) |
| KL_LAMBDA | 0.5 (v5) |
| target_modules | q,k,v,o + gate,up,down |
| LR | 1e-4, cosine, 5% warmup |
| dropout | 0.1, betas (0.9, 0.95) |
| epochs | 6, checkpoint selected by **best val_CE** |
| prefix | "" (none) |
| seed | 42 (single; more seeds can be added later and aggregated) |

Estimated cost: ~700 SBU train + ~870 SBU eval per rank => ~1570/rank
=> ~9400 SBU for the 6-rank sweep (training + image metrics + text battery).

## 0. Prep

```
cd ~/multimodal-peft
git pull
```

## 1. Launch the 6 training runs (parallel)

```
for R in 2 4 8 16 32 64; do
  A=$((2*R))
  sbatch --export=ALL,LORA_RANK=$R,LORA_ALPHA=$A,RUN_NAME=gen_7b_rank_r$R \
    janus/train/02_run_train_generation_7b_v5.sh
done
```

Each writes to `~/multimodal-peft/janus/train/output/gen_7b_rank_r<R>/` and,
on completion, creates an `adapter_best` symlink -> the lowest-val_CE epoch.

Track:
```
squeue -u $USER | grep janus_7b_v5
```
(All six share the SLURM job-name `janus_7b_v5`; tell them apart by RUN_NAME
in each slurm_<JOBID>.out, or by submission order.)

## 2. Chain image-metrics + text-battery per rank (after each finishes)

For each rank, once its training is COMPLETED, grab its training JOBID and:

```
R=2   # repeat for 4 8 16 32 64
TRAIN_JOB=<jobid of gen_7b_rank_r$R training>
ADAPTER=$HOME/multimodal-peft/janus/train/output/gen_7b_rank_r$R/adapter_best

# image-generation metrics panel
sbatch --dependency=afterok:$TRAIN_JOB \
  --export=ALL,MODEL_ID=deepseek-community/Janus-Pro-7B,ADAPTER=$ADAPTER,CAPTION_PREFIX="" \
  janus/eval/06_run_metrics_panel.sh

# cross-modal forgetting battery
sbatch --dependency=afterok:$TRAIN_JOB \
  --export=ALL,ADAPTER=$ADAPTER \
  janus/eval/07_run_forgetting_battery.sh
```

NOTE on `adapter_best`: PeftModel.from_pretrained follows the symlink, so
passing `.../adapter_best` loads the best-val_CE checkpoint automatically.
If your filesystem dropped the symlink, read the target from
`gen_7b_rank_r<R>/adapter_best.txt` (or `history.json` -> "best_epoch") and
point ADAPTER at `adapter_epoch<best>` directly.

## 3. Aggregate into the scaling figure (after all eval done)

```
~/.conda/envs/janus/bin/python janus/eval/09_plot_rank_sweep.py
```
(Script written separately once the JSONs exist. Plots, vs rank on a log
axis: CLIP DirSim (visual acquisition), PPL and POPE (cross-modal
interference). The headline scaling-law figure.)

## Future experiments (annotated, not launched)
- rsLoRA overlap on the 3 highest ranks (alpha/sqrt(r) vs alpha/r)
- KL sweep at fixed rank (KL in {0, 0.1, 0.3, 0.5, 1.0})
- additional seeds on the interesting ranks, aggregated as mean +/- std
