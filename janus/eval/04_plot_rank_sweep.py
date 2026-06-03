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
            "before_CLIP", "after_CLIP", "delta_CLIPScore",
            "before_CLIPScore_cap", "after_CLIPScore_cap",
            "delta_CLIPScore_cap", "reference_CLIPScore_cap"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def make_plot(rows, png_path: Path):
    """Two stacked panels:
       (top)    delta CIDEr (style+content) vs delta CLIPScore_cap (content-aware)
                -- both on the captioning target. If they diverge -> the CIDEr
                gain is stylistic, not content.
       (bottom) delta CLIPScore_gen (generation probe) -- cross-modal forgetting.
    """
    ranks   = [r["rank"]                       for r in rows]
    dcider  = [r["delta_CIDEr"]                for r in rows]
    dcapclip = [r.get("delta_CLIPScore_cap") for r in rows]
    has_capclip = all(v is not None for v in dcapclip)
    dgenclip = [r["delta_CLIPScore"]           for r in rows]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]})

    # --- Top: CIDEr (style+content) vs CLIPScore_cap (content only)
    color_cider = "tab:blue"
    color_capclip = "tab:green"
    l1 = ax_top.plot(ranks, dcider, "-o", color=color_cider,
                     label=r"$\Delta$ CIDEr (n-gram, style$+$content)")
    ax_top.axhline(0, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    for x, y in zip(ranks, dcider):
        ax_top.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=color_cider)
    ax_top.set_ylabel(r"$\Delta$ CIDEr", color=color_cider)
    ax_top.tick_params(axis="y", labelcolor=color_cider)
    # Pad the y-limits so annotations + legend don't overlap the data points.
    if dcider:
        y_min, y_max = min(dcider), max(dcider)
        pad = max(0.1, 0.2 * (y_max - y_min) if y_max != y_min else 0.2)
        ax_top.set_ylim(min(0, y_min) - pad, y_max + 1.5 * pad)

    lines, labels = list(l1), [l.get_label() for l in l1]
    if has_capclip:
        ax_top_r = ax_top.twinx()
        l2 = ax_top_r.plot(ranks, dcapclip, "-D", color=color_capclip,
                           label=r"$\Delta$ CLIPScore(img, gen-cap) (content-aware)")
        ax_top_r.set_ylabel(r"$\Delta$ CLIPScore (caption)", color=color_capclip)
        ax_top_r.tick_params(axis="y", labelcolor=color_capclip)
        ax_top_r.axhline(0, color=color_capclip, linestyle=":", linewidth=0.6, alpha=0.5)
        for x, y in zip(ranks, dcapclip):
            ax_top_r.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                              xytext=(0, -14), ha="center", fontsize=8,
                              color=color_capclip)
        if dcapclip:
            yr_min, yr_max = min(dcapclip), max(dcapclip)
            padr = max(0.5, 0.3 * (yr_max - yr_min) if yr_max != yr_min else 0.5)
            ax_top_r.set_ylim(yr_min - 1.5 * padr, max(0, yr_max) + padr)
        lines += list(l2)
        labels += [l.get_label() for l in l2]

    # Single combined legend placed BELOW the top panel so it can never overlap
    # the data points (which sit near the top of the panel).
    ax_top.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                  ncol=1 if len(lines) < 2 else 2, fontsize=8, frameon=True)
    ax_top.set_title("Captioning target: n-gram metric vs content-aware metric",
                     fontsize=10, pad=10)

    # --- Bottom: cross-modal forgetting probe
    color_gen = "tab:red"
    ax_bot.plot(ranks, dgenclip, "-s", color=color_gen,
                label=r"$\Delta$ CLIPScore (gen probe, cross-modal forgetting)")
    ax_bot.axhline(0, color=color_gen, linestyle=":", linewidth=0.8, alpha=0.5)
    for x, y in zip(ranks, dgenclip):
        ax_bot.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=color_gen)
    ax_bot.set_ylabel(r"$\Delta$ CLIPScore (generation)", color=color_gen)
    ax_bot.tick_params(axis="y", labelcolor=color_gen)
    if dgenclip:
        y_min, y_max = min(dgenclip), max(dgenclip)
        pad = max(0.1, 0.2 * (y_max - y_min) if y_max != y_min else 0.1)
        ax_bot.set_ylim(min(0, y_min) - pad, y_max + 1.5 * pad)
    ax_bot.set_xlabel("LoRA rank")
    ax_bot.set_xscale("log", base=2)
    ax_bot.set_xticks(ranks)
    ax_bot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_bot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  fontsize=9, frameon=True)
    ax_bot.set_title("Cross-modal forgetting probe", fontsize=10, pad=8)

    fig.suptitle("LoRA rank sweep on captioning (Janus-Pro-1B)", fontsize=11)
    # Reserve extra vertical space between/below panels for the external legends.
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.45)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
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
