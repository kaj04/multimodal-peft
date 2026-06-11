# KL sweep at fixed rank (r=32) — isolating the KL-to-base regulariser

Tests how the KL-to-base weight $\lambda$ affects the v5 recipe, holding rank
fixed at 32 and everything else at the v5 config. Diagnostic question: was the
KL regulariser the ingredient that made the recipe work, or were the other
v4/v5 changes (no prefix, MLP modules, higher LR) sufficient on their own?

| KL | Status | Run name |
|---|---|---|
| 0    | NEW   | gen_7b_kl0_r32 |
| 0.25 | NEW   | gen_7b_kl025_r32 |
| 0.5  | REUSE | gen_7b_rank_r32 (rank sweep; already has metrics + qualitative) |
| 1.0  | NEW   | gen_7b_kl1_r32 |
| 1.5  | NEW   | gen_7b_kl15_r32 |

NOTE: KL=0 at THIS config is new -- v2 was also KL=0 but with a different
config (prefix on, r=16, LR 5e-5), so it is not a clean comparison. The KL=0
point here (v5 config, no prefix, r=32, LR 1e-4) is the key diagnostic: if it
degrades like v2 did, the KL regulariser is confirmed as the critical
ingredient; if it works, the other v4/v5 changes were sufficient on their own.

4 new training runs (~4.2k SBU); the KL=0.5 point reuses gen_7b_rank_r32.

Config (fixed): r=32, alpha=64, no prefix, MLP+attention LoRA, LR 1e-4,
6 epochs, best-val_CE checkpoint. Battery skipped (cross-modal interference
is already flat; the KL acts on acquisition, not interference).

## 0. Prep
```
cd ~/multimodal-peft
git pull
```

## 1. Four new training runs (parallel)
```
sbatch --export=ALL,KL_LAMBDA=0.0,LORA_RANK=32,LORA_ALPHA=64,RUN_NAME=gen_7b_kl0_r32 janus/train/02_run_train_generation_7b_v5.sh
sbatch --export=ALL,KL_LAMBDA=0.25,LORA_RANK=32,LORA_ALPHA=64,RUN_NAME=gen_7b_kl025_r32 janus/train/02_run_train_generation_7b_v5.sh
sbatch --export=ALL,KL_LAMBDA=1.0,LORA_RANK=32,LORA_ALPHA=64,RUN_NAME=gen_7b_kl1_r32 janus/train/02_run_train_generation_7b_v5.sh
sbatch --export=ALL,KL_LAMBDA=1.5,LORA_RANK=32,LORA_ALPHA=64,RUN_NAME=gen_7b_kl15_r32 janus/train/02_run_train_generation_7b_v5.sh
```

Check (after ~3 min) that each has the right KL and is training from scratch:
```
for f in $(grep -l "gen_7b_kl" ~/multimodal-peft/janus/train/output/slurm_*.out); do
  echo "=== $f ==="; grep -E "run=gen_7b_kl|kl_lambda|rank=|total optimizer steps" "$f" | head -4
done
```
Confirm four DISTINCT kl_lambda values (0.0, 0.25, 1.0, 1.5), rank=32,
total optimizer steps = 852 (not 0).

## 2. Chain metrics + qualitative per new run (after training finishes)
```
for NAME in gen_7b_kl0_r32 gen_7b_kl025_r32 gen_7b_kl1_r32 gen_7b_kl15_r32; do
  ADAPTER=$HOME/multimodal-peft/janus/train/output/$NAME/adapter_best
  sbatch --export=ALL,MODEL_ID=deepseek-community/Janus-Pro-7B,ADAPTER=$ADAPTER,CAPTION_PREFIX="" janus/eval/06_run_metrics_panel.sh
  sbatch --export=ALL,MODEL_ID=deepseek-community/Janus-Pro-7B,ADAPTER=$ADAPTER,CAPTION_PREFIX="",JANUS_OUTPUT_DIR=$HOME/multimodal-peft/janus/tests/output/qual_$NAME janus/eval/05_run_pokemon_qualitative.sh
done
```
(KL=0.5 point already has metrics_panel/gen_7b_rank_r32__adapter_best and
qual_rank_r32 from the rank sweep -- no rerun needed.)

## 3. Aggregate (login node, after all eval)
```
~/.conda/envs/janus/bin/python janus/eval/09b_plot_kl_sweep.py   # written separately
bash scripts/snellius_organize_kl_comparison.sh                  # visual grid
```
Produces the KL-vs-metrics curves + the 5-prompt x 4-KL visual grid for the
meeting with Ivan and Anna.
