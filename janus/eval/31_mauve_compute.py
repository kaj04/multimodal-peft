#!/usr/bin/env python3
"""
janus/eval/31_mauve_compute.py   --   PHASE 2 (mauve venv, CPU/GPU)

Compute MAUVE (Pillutla et al., NeurIPS 2021) between the model generations and
the human references produced by phase 1 (30_mauve_generate.py). Runs in a
separate venv (setup_mauve_venv.sh) that has mauve-text + scikit-learn on top of
the janus site-packages, so the janus environment is never modified.

MAUVE in [0,1], higher = the generated-text distribution is closer to the human
reference distribution. For the ENGLISH set it measures whether free-form English
generation is preserved (forgetting); for a TARGET set it measures acquisition
(generated text close to real target text).

Env: TAG (which adapter's gens), KINDS="english,welsh" (which sets to score),
FEATURIZE_MODEL (default gpt2-large), MAX_LEN, DEVICE_ID. Run via
31_run_mauve_compute.sh.
"""
import json
import os
from pathlib import Path

TAG   = os.environ.get("TAG", "base")
KINDS = [k.strip() for k in os.environ.get("KINDS", "english").split(",") if k.strip()]
FEATURIZE_MODEL = os.environ.get("FEATURIZE_MODEL", "gpt2-large")
MAX_LEN   = int(os.environ.get("MAX_LEN", "256"))
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
SEED      = int(os.environ.get("SEED", "42"))

ROOT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output")) / "mauve"
OUT  = ROOT / TAG
REFS = ROOT / "_refs"
SEP = "=" * 64


def main():
    import mauve  # from the mauve venv
    results = {}
    out_json = OUT / "mauve.json"
    if out_json.exists():
        results = json.loads(out_json.read_text())

    for kind in KINDS:
        gp = OUT / f"gen_{kind}.json"
        rp = REFS / f"ref_{kind}.json"
        if not (gp.exists() and rp.exists()):
            print(f"  [skip] {kind}: missing {gp.name} or {rp.name}", flush=True)
            continue
        gens = json.loads(gp.read_text())
        refs = json.loads(rp.read_text())
        n = min(len(gens), len(refs))
        gens, refs = gens[:n], refs[:n]
        print(f"  [{kind}] MAUVE on n={n} (featurize={FEATURIZE_MODEL}) ...", flush=True)
        out = mauve.compute_mauve(
            p_text=gens, q_text=refs, device_id=DEVICE_ID,
            max_text_length=MAX_LEN, verbose=False, featurize_model_name=FEATURIZE_MODEL,
            seed=SEED)
        results[kind] = {"mauve": float(out.mauve), "n": n}
        print(f"  [{kind}] MAUVE = {out.mauve:.4f}", flush=True)

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  saved -> {out_json}\n  {results}")


if __name__ == "__main__":
    main()
