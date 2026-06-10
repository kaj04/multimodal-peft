#!/usr/bin/env python3
"""
scripts/download_comparison_images.py

Pull qualitative-pair images from Snellius and organise them into per-prompt
folders so each folder is one ROW of the 4-column Overleaf comparison:

  Column 1: No Adapter         = base 7B Janus, no LoRA  (BEFORE from any qual run)
  Column 2: Solo Image Gen     = v5 image-only adapter   (AFTER from qual_v5)
  Column 3: Image + Text MATCH = joint_v5_matched        (AFTER from qual_joint_matched)
  Column 4: Image + Text FAR   = joint_v5_far            (AFTER from qual_joint_far)

Local layout after running:
  comparison_grids/
    eval_00_natural/
      01_no_adapter.png
      02_v5_image_only.png
      03_joint_matched.png
      04_joint_far.png
    eval_01_natural/
      ...

Edit PROMPTS below to pick which 5 (or how many) prompts to include.
Each prompt is a tag like "eval_XX_<mode>" or "human_XX_<mode>" where
  XX in {00..04}, mode in {natural, prefix}.

Cross-platform (scp via Python subprocess). On Windows requires OpenSSH
client (built-in on Windows 10+). On macOS/Linux works out of the box.

Run from anywhere (creates comparison_grids/ in the current directory):
    python download_comparison_images.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG -- edit these if your remote setup differs
# ---------------------------------------------------------------------------
SNELLIUS_USER = "fcolasurdo"
SNELLIUS_HOST = "snellius.surf.nl"
SNELLIUS_BASE = "/home/fcolasurdo/multimodal-peft/janus/tests/output"
LOCAL_BASE    = Path("./comparison_grids")

# Which prompts to include in the comparison. Each tag matches the qualitative
# script's filename convention: <source>_<idx>_<mode>_<phase>.png
# Where:
#   source in {eval, human},  idx in {00..04},  mode in {natural, prefix}
# Pick 5 that are visually informative. Defaults below mix:
#   2 eval-split prompts (in-distribution BLIP captions)
#   3 human prompts (creative, test generalisation)
# all in 'natural' mode (no trigger phrase) to test the gate-solved behaviour.
PROMPTS = [
    "eval_00_natural",
    "eval_01_natural",
    "human_00_natural",
    "human_02_natural",
    "human_03_natural",
]

# Column definitions: (label_filename, remote_run_folder, file_phase)
# 'before' = base 7B (no LoRA), 'after' = with adapter
COLUMNS = [
    ("01_no_adapter",     "qual_v5",            "before"),
    ("02_v5_image_only",  "qual_v5",            "after"),
    ("03_joint_matched",  "qual_joint_matched", "after"),
    ("04_joint_far",      "qual_joint_far",     "after"),
]

# Subfolder appended by the qualitative script
QUAL_SUBFOLDER = "naruto_qual_7b"


def have_scp():
    return shutil.which("scp") is not None


def run_scp(remote_path, local_path):
    """Single scp call; non-fatal on failure (prints + returns False)."""
    cmd = ["scp", "-q",
           f"{SNELLIUS_USER}@{SNELLIUS_HOST}:{remote_path}",
           str(local_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAIL: {remote_path}")
        if r.stderr.strip():
            print(f"      {r.stderr.strip()}")
        return False
    return True


def main():
    if not have_scp():
        sys.exit("scp not found on PATH. Install OpenSSH client (built-in on "
                 "Windows 10+, brew install openssh on macOS).")
    LOCAL_BASE.mkdir(exist_ok=True)
    print(f"Local destination: {LOCAL_BASE.resolve()}")
    print(f"Remote source    : {SNELLIUS_USER}@{SNELLIUS_HOST}:{SNELLIUS_BASE}")
    print(f"Prompts ({len(PROMPTS)}): {PROMPTS}")
    print(f"Columns ({len(COLUMNS)}): {[c[0] for c in COLUMNS]}")
    print()

    failed = []
    for prompt in PROMPTS:
        dest_dir = LOCAL_BASE / prompt
        dest_dir.mkdir(exist_ok=True)
        print(f"[{prompt}]")
        for label, run_folder, phase in COLUMNS:
            remote = (f"{SNELLIUS_BASE}/{run_folder}/{QUAL_SUBFOLDER}/"
                      f"{prompt}_{phase}.png")
            local  = dest_dir / f"{label}.png"
            print(f"  {label:<22s} <- {run_folder}/{prompt}_{phase}.png")
            if not run_scp(remote, local):
                failed.append((prompt, label, remote))
        print()

    if failed:
        print(f"DONE with {len(failed)} failures:")
        for p, l, r in failed:
            print(f"  - {p}/{l} (remote: {r})")
        sys.exit(1)
    else:
        print(f"DONE. {len(PROMPTS)} prompts x {len(COLUMNS)} columns "
              f"= {len(PROMPTS) * len(COLUMNS)} images downloaded.")


if __name__ == "__main__":
    main()
