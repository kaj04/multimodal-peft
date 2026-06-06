# Joint training C0/C1/C2/C3 on Janus-Pro-7B

Image data is fixed (Naruto BLIP captions). Text bucket is the manipulated factor.

| Run | TEXT_BUCKET | What text we mix in |
|---|---|---|
| C0 | `none`    | none -- pure image baseline (identical to gen_7b_naruto_r8_prefix_cosine) |
| C1 | `matched` | Naruto Wikipedia (~94 cleaned articles) |
| C2 | `near`    | other anime/manga Wikipedia (~1700) |
| C3 | `far`     | physics / mathematics Wikipedia (~3800) |

## 0. Prep (once)

```
cd ~/multimodal-peft
git pull
~/.conda/envs/janus/bin/python janus/data_prep/01b_clean_matched.py
~/.conda/envs/janus/bin/python janus/data_prep/02_tokenize_corpora.py
ls -lh ~/multimodal-peft/janus/data_prep/output/wiki_*_tokens.jsonl
```

The tokenize step takes ~3-5 min on the login node (CPU). You should see
three `.jsonl` files with token chunks ready for the trainer.

## 1. Launch all 4 runs in parallel

```
sbatch --export=ALL,TEXT_BUCKET=none    janus/train/04_run_joint.sh
sbatch --export=ALL,TEXT_BUCKET=matched janus/train/04_run_joint.sh
sbatch --export=ALL,TEXT_BUCKET=near    janus/train/04_run_joint.sh
sbatch --export=ALL,TEXT_BUCKET=far     janus/train/04_run_joint.sh
```

Each job: 1x H100, ~8-10h. Total budget cost: ~40h H100 ≈ 960 SBU (3% of remaining).

You can also stagger them if you want to monitor C0 first before committing
to C1/C2/C3 -- but with the budget headroom there's no real reason to wait.

## 2. Monitor

```
squeue -u $USER
tail -f ~/multimodal-peft/janus/train/output/slurm_<JOBID>.out
```

W&B runs (project `janus-peft`):
- joint_7b_none_r8
- joint_7b_matched_r8
- joint_7b_near_r8
- joint_7b_far_r8

Curves to watch:
- `val/CE_loss` -- THE proper overfitting signal; pick the best epoch from this
- `train/loss_img` and `train/loss_txt` -- both decreasing means joint learning works
- `train/text_frac` -- should hover around 0.5 (sanity check on the alternation)
- `eval/CLIPScore` -- noisy, OOD-unfaithful for stylised anime; informative but not decisive

## 3. Chain a qualitative eval per run (after each finishes)

For each completed run, pick the epoch with lowest `val_CE` (call it `BEST`),
then:

```
sbatch --export=ALL,MODEL_ID=deepseek-community/Janus-Pro-7B,ADAPTER=$HOME/multimodal-peft/janus/train/output/joint_7b_<bucket>_r8/adapter_epoch<BEST>,CAPTION_PREFIX="anime sprite in Naruto style, ",JANUS_OUTPUT_DIR=$HOME/multimodal-peft/janus/tests/output/joint_<bucket> janus/eval/05_run_pokemon_qualitative.sh
```

This produces 20 before/after pair PNGs per run + a JSON with CLIPScore by
category, in `naruto_qual_7b/` (or rename via JANUS_OUTPUT_DIR).

## 4. Comparing across C0..C3

The "deliverable" plot is a 4-column grid: for the same base prompt, show
the after-adapter image from C0, C1, C2, C3 side-by-side. This shows whether
the text domain mixed during training visibly shifts the generated image
distribution.

(Plot script TODO: `janus/eval/06_joint_comparison_grid.py` -- write once we
have the four adapters in hand.)
