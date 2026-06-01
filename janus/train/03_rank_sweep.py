#!/usr/bin/env python3
"""
janus/train/03_rank_sweep.py

Phase 5 / RQ3: LoRA rank sweep on the captioning direction.

For each rank R in RANKS (default: 2, 4, 8, 16, 32, 64) this orchestrator runs,
in sequence, inside a SINGLE SLURM job:

    1. train captioning with LoRA r=R  (reuses 01_train_captioning.py)
    2. evaluate cross-modal forgetting (reuses 02_eval_adapter.py)

Outputs are kept rank-scoped:
    train/output/cap_1b_r{R}_sweep/adapter_epoch1/   (adapter weights)
    train/output/cap_1b_r{R}_sweep/history.json      (training curve)
    tests/output/forgetting/forgetting_captioning_r{R}.json   (deltas)
    tests/output/forgetting/samples_captioning_r{R}/          (before/after)

Resume capability: if the forgetting JSON for a rank already exists with
non-empty content, that rank is skipped. So killing and re-submitting the
job picks up where it left off.

W&B: a separate run per rank (name "cap_1b_r{R}_sweep") so curves overlay
cleanly on the dashboard.

Why subprocess instead of importing: the two existing scripts are already
tested end-to-end; running them as subprocesses keeps a single source of
truth and lets one bad rank fail loudly without taking the whole sweep down.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --- CONFIG ----------------------------------------------------------------

RANKS_STR = os.environ.get("RANKS", "2,4,8,16,32,64")
RANKS = [int(r) for r in RANKS_STR.split(",")]

REPO_ROOT = Path(os.path.expanduser("~/multimodal-peft"))
TRAIN_PY  = REPO_ROOT / "janus" / "train" / "01_train_captioning.py"
EVAL_PY   = REPO_ROOT / "janus" / "eval"  / "02_eval_adapter.py"
TRAIN_OUT = REPO_ROOT / "janus" / "train" / "output"
EVAL_OUT  = REPO_ROOT / "janus" / "tests" / "output" / "forgetting"

# Common training/eval params (overridable via env vars)
TRAIN_N    = os.environ.get("TRAIN_N", "4000")
EVAL_N     = os.environ.get("EVAL_N", "200")
GEN_EVAL_N = os.environ.get("GEN_EVAL_N", "30")
EPOCHS     = os.environ.get("EPOCHS", "1")
GRAD_ACCUM = os.environ.get("GRAD_ACCUM", "8")
LR         = os.environ.get("LR", "1e-4")
SEED       = os.environ.get("SEED", "42")
LORA_ALPHA_RULE = os.environ.get("LORA_ALPHA_RULE", "2x")
# Alpha scaling rule: "2x" -> alpha = 2*r (preserve effective scale),
# "fixed16" -> alpha = 16 for all ranks. Default 2x is the common practice.

PYTHON = sys.executable   # the python running THIS sweep (i.e. the janus env)

SEP = "=" * 64


def alpha_for(rank: int) -> int:
    if LORA_ALPHA_RULE == "fixed16":
        return 16
    return 2 * rank


def rank_json_path(rank: int) -> Path:
    return EVAL_OUT / f"forgetting_captioning_r{rank}.json"


def already_done(rank: int) -> bool:
    """A rank counts as done if its forgetting JSON exists and is non-empty."""
    f = rank_json_path(rank)
    return f.exists() and f.stat().st_size > 0


def row_from_json(rank: int) -> dict:
    """Build a summary row by reading the per-rank forgetting JSON on disk."""
    data = json.loads(rank_json_path(rank).read_text())
    return {
        "rank": rank, "status": "ok",
        "alpha": alpha_for(rank),
        "delta_CIDEr":     data["delta"]["CIDEr"],
        "delta_BLEU_4":    data["delta"]["BLEU_4"],
        "delta_CLIPScore": data["delta"]["CLIPScore"],
        "before_CIDEr":  data["before"]["CIDEr"],
        "after_CIDEr":   data["after"]["CIDEr"],
        "before_CLIP":   data["before"]["CLIPScore"],
        "after_CLIP":    data["after"]["CLIPScore"],
    }


def run_subprocess(label: str, env: dict, script: Path) -> int:
    """Run a child process inheriting the current env + overrides; stream output."""
    full_env = os.environ.copy()
    full_env.update(env)
    full_env["PYTHONUNBUFFERED"] = "1"
    print(f"\n>>> [{label}] starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"    cmd: {PYTHON} {script}", flush=True)
    print(f"    env overrides: {env}", flush=True)
    t0 = time.time()
    proc = subprocess.run([PYTHON, str(script)], env=full_env)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
    print(f"<<< [{label}] {status} in {dt/60:.1f} min", flush=True)
    return proc.returncode


def main():
    print(SEP)
    print(f"  Rank sweep on captioning")
    print(f"  ranks  : {RANKS}")
    print(f"  alpha  : rule={LORA_ALPHA_RULE} (e.g. r={RANKS[0]}->alpha={alpha_for(RANKS[0])})")
    print(f"  splits : TRAIN_N={TRAIN_N} EVAL_N={EVAL_N} GEN_EVAL_N={GEN_EVAL_N}")
    print(f"  opt    : LR={LR} GRAD_ACCUM={GRAD_ACCUM} EPOCHS={EPOCHS} SEED={SEED}")
    print(f"  python : {PYTHON}")
    print(SEP, flush=True)

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)

    summary = []
    t_sweep = time.time()

    for R in RANKS:
        rank_tag = f"r{R}"
        run_name = f"cap_1b_{rank_tag}_sweep"
        adapter_dir = TRAIN_OUT / run_name / "adapter_epoch1"

        if already_done(R):
            print(f"\n=== rank {R}: SKIP (forgetting_captioning_{rank_tag}.json already present) ===", flush=True)
            # Still record this rank in the summary so a SKIP-only re-run
            # rebuilds the aggregated JSON instead of overwriting it with [].
            summary.append(row_from_json(R))
            continue

        print(f"\n=== rank {R} ===", flush=True)

        # 1) Training
        train_env = {
            "LORA_RANK":  str(R),
            "LORA_ALPHA": str(alpha_for(R)),
            "RUN_NAME":   run_name,
            "TRAIN_N":    TRAIN_N,
            "EVAL_N":     EVAL_N,
            "EPOCHS":     EPOCHS,
            "GRAD_ACCUM": GRAD_ACCUM,
            "LR":         LR,
            "SEED":       SEED,
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "janus-peft"),
        }
        if not adapter_dir.exists():
            rc = run_subprocess(f"train r={R}", train_env, TRAIN_PY)
            if rc != 0 or not adapter_dir.exists():
                print(f"  rank {R}: training failed; skipping its eval")
                summary.append({"rank": R, "status": "train_failed"})
                continue
        else:
            print(f"  adapter for r={R} already exists at {adapter_dir}; reusing")

        # 2) Forgetting eval
        eval_env = {
            "ADAPTER":    str(adapter_dir),
            "TRAINED_ON": "captioning",
            "OUTPUT_TAG": rank_tag,
            "EVAL_N":     EVAL_N,
            "GEN_EVAL_N": GEN_EVAL_N,
            "SEED":       SEED,
            "TRAIN_N":    TRAIN_N,   # used to reproduce the same held-out split
        }
        rc = run_subprocess(f"eval r={R}", eval_env, EVAL_PY)
        if rc != 0:
            print(f"  rank {R}: eval failed")
            summary.append({"rank": R, "status": "eval_failed"})
            continue

        # 3) Record success (reads the per-rank JSON we just wrote)
        if already_done(R):
            summary.append(row_from_json(R))
        else:
            summary.append({"rank": R, "status": "json_missing"})

    # --- final summary ----------------------------------------------------
    print("\n" + SEP)
    print("  RANK SWEEP — final summary")
    print(SEP)
    print(f"{'rank':>5} {'status':>12} {'dCIDEr':>10} {'dBLEU4':>10} {'dCLIP':>10}")
    for row in summary:
        if row["status"] == "ok":
            print(f"{row['rank']:>5} {row['status']:>12} "
                  f"{row['delta_CIDEr']:>+10.4f} "
                  f"{row['delta_BLEU_4']:>+10.4f} "
                  f"{row['delta_CLIPScore']:>+10.4f}")
        else:
            print(f"{row['rank']:>5} {row['status']:>12} {'-':>10} {'-':>10} {'-':>10}")
    print(SEP)
    print(f"  total sweep time: {(time.time()-t_sweep)/60:.1f} min")

    out = EVAL_OUT / "rank_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
