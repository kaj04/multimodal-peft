#!/usr/bin/env python3
"""
janus/eval/04_plot_rank_sweep.py

Plot the rank-sweep results produced by `03_rank_sweep.py`.

Reads:
    ~/multimodal-peft/janus/tests/output/forgetting/rank_sweep_summary.json
    (path overridable with $SWEEP_JSON or first CLI arg)

Produces, in the same directory:
    rank_sweep.png    -- one figure, two y-axes:
                         * left axis  : task gain on captioning (Delta CIDEr)
                         * right axis : cross-modal forgetting probe (Delta CLIPScore)
                         x axis is rank in log scale (power-of-two values are common).
                         Reference lines at delta=0 mark the "no change" baseline.
    rank_sweep.csv    -- a flat CSV with one row per rank, easy to import elsewhere.

The script works on partial results: a mini-sweep with two ranks plots two
points; the full sweep plots six. Ranks with status != "ok" are skipped.
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SUMMARY = Path(os.path.expanduser(
    "~/multimodal-peft/janus/tests/output/forgetting/rank_sweep_summary.json"))


def load_summary(path: Path):
    data = json.loads(path.read_text())
    rows = [r for r in data if r.get("status") == "ok"]
    rows.sort(key=lambda r: r["rank"])
    if not rows:
        sys.exit(f"no successful rank rows found in {path}")
    return rows


def write_csv(rows, csv_path: Path):
    cols = ["rank", "alpha",
            "before_CIDEr", "after_CIDEr", "delta_CIDEr",
            "delta_BLEU_4",
            "before_CLIP", "after_CLIP", "delta_CLIPScore"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def make_plot(rows, png_path: Path):
    ranks  = [r["rank"]            for r in rows]
    dcider = [r["delta_CIDEr"]     for r in rows]
    dclip  = [r["delta_CLIPScore"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

    color_cap = "tab:blue"
    color_gen = "tab:red"

    l1 = ax1.plot(ranks, dcider, "-o", color=color_cap,
                  label=r"$\Delta$ CIDEr (captioning target)")
    ax1.set_xlabel("LoRA rank")
    ax1.set_ylabel(r"$\Delta$ CIDEr (target task)", color=color_cap)
    ax1.tick_params(axis="y", labelcolor=color_cap)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ranks)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.axhline(0, color=color_cap, linestyle=":", linewidth=0.8, alpha=0.5)
    for x, y in zip(ranks, dcider):
        ax1.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=color_cap)

    ax2 = ax1.twinx()
    l2 = ax2.plot(ranks, dclip, "-s", color=color_gen,
                  label=r"$\Delta$ CLIPScore (generation probe)")
    ax2.set_ylabel(r"$\Delta$ CLIPScore (cross-modal probe)", color=color_gen)
    ax2.tick_params(axis="y", labelcolor=color_gen)
    ax2.axhline(0, color=color_gen, linestyle=":", linewidth=0.8, alpha=0.5)
    for x, y in zip(ranks, dclip):
        ax2.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=color_gen)

    plt.title("LoRA rank sweep on captioning (Janus-Pro-1B)\n"
              "task gain vs cross-modal forgetting", fontsize=10)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SWEEP_JSON")
    summary_path = Path(arg) if arg else DEFAULT_SUMMARY
    if not summary_path.exists():
        sys.exit(f"summary not found: {summary_path}")

    rows = load_summary(summary_path)
    out_dir = summary_path.parent
    csv_path = out_dir / "rank_sweep.csv"
    png_path = out_dir / "rank_sweep.png"

    write_csv(rows, csv_path)
    make_plot(rows, png_path)

    print(f"  ranks plotted : {[r['rank'] for r in rows]}")
    print(f"  CSV  -> {csv_path}")
    print(f"  PNG  -> {png_path}")


if __name__ == "__main__":
    main()
