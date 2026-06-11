#!/usr/bin/env python3
"""
janus/eval/09_plot_rank_sweep.py

Aggregate the rank-sweep evaluations into:
  1. a complete table  (rank x every metric)  -> printed + saved as .tex/.csv
  2. an overview figure (key metrics vs rank, log-x) -> .pdf/.png

Reads, for each rank R in RANKS:
  metrics_panel/gen_7b_rank_r{R}__adapter_best/metrics_panel.json
  forgetting_battery/gen_7b_rank_r{R}__adapter_best/forgetting_results.json

Missing files are skipped with a warning so the script can be run while
some evals are still in flight. The figure's plotted metrics are chosen in
PLOT_METRICS at the top -- run once, look at the full table, then edit
PLOT_METRICS to the 2-3 most representative signals for the paper.

Runs on the login node (no GPU): JSON -> table + matplotlib figure.
"""

import csv
import json
import os
from pathlib import Path

RANKS = [2, 4, 8, 16, 32, 64]

ROOT = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output")))
METRICS_DIR = ROOT / "metrics_panel"
BATTERY_DIR = ROOT / "forgetting_battery"
OUT = ROOT / "overleaf" / "rank_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# Two-panel normalised figure: which metrics go in each panel, and whether
# lower-is-better (so we can orient every curve as higher=better before
# min-max normalising onto a common 0-1 axis).
#   (metric_name, lower_is_better, legend_label)
IMAGE_PANEL = [
    ("CLIP_DirSim",   False, "CLIP DirSim"),
    ("DINO_after",    False, "DINO sim"),
    ("CMMD_after",    True,  "CMMD"),
    ("KID_after",     True,  "KID"),
    ("CLIP_IQA_after",False, "CLIP-IQA"),
]
TEXT_PANEL = [
    ("PPL",      True,  "WikiText PPL"),
    ("MMLU",     False, "MMLU"),
    ("POPE_acc", False, "POPE acc"),
    ("MME",      False, "MME"),
]

# Every metric we extract, with (label, where it lives, how to pull it,
# arrow for "better"). after/delta refer to the metrics_panel buckets.
def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default

METRIC_SPECS = [
    # name,        arrow, extractor(metrics_json, battery_json)
    ("trainable_ratio", "",  lambda m, b: None),  # filled from history if present
    ("KID_after",     "v", lambda m, b: _g(m, "after", "KID", "value")),
    ("CMMD_after",    "v", lambda m, b: _g(m, "after", "CMMD", "value")),
    ("DINO_after",    "^", lambda m, b: _g(m, "after", "DINO", "value")),
    ("CLIP_IQA_after","^", lambda m, b: _g(m, "after", "CLIP_IQA", "value")),
    ("CLIP_DirSim",   "^", lambda m, b: _g(m, "delta", "CLIP_DirSim", "value")),
    ("ArtFID",        "v", lambda m, b: _g(m, "delta", "ArtFID", "value")),
    ("FID_robust_aft","v", lambda m, b: _g(m, "delta", "FID_robust", "after")),
    ("PPL",           "v", lambda m, b: _g(b, "wikitext2_ppl", "PPL")),
    ("MMLU",          "^", lambda m, b: _g(b, "mmlu", "accuracy")),
    ("POPE_acc",      "^", lambda m, b: _g(b, "pope", "accuracy")),
    ("POPE_f1",       "^", lambda m, b: _g(b, "pope", "f1")),
    ("MME",           "^", lambda m, b: _g(b, "mme", "total_mme_score")),
]

def load_rank(R):
    mp = METRICS_DIR / f"gen_7b_rank_r{R}__adapter_best" / "metrics_panel.json"
    bp = BATTERY_DIR / f"gen_7b_rank_r{R}__adapter_best" / "forgetting_results.json"
    m = json.loads(mp.read_text()) if mp.exists() else None
    b = json.loads(bp.read_text()) if bp.exists() else None
    if m is None:
        print(f"  WARN r{R}: missing metrics_panel.json")
    if b is None:
        print(f"  WARN r{R}: missing forgetting_results.json")
    return m, b


def main():
    rows = {}     # R -> {metric_name: value}
    for R in RANKS:
        m, b = load_rank(R)
        rows[R] = {}
        for name, _arrow, fn in METRIC_SPECS:
            try:
                rows[R][name] = fn(m, b)
            except Exception:
                rows[R][name] = None

    metric_names = [s[0] for s in METRIC_SPECS]

    # ---- 1. Print + save the full table ----
    print("\n" + "=" * 100)
    header = "rank  " + "  ".join(f"{n[:12]:>12s}" for n in metric_names)
    print(header)
    print("-" * len(header))
    for R in RANKS:
        cells = []
        for n in metric_names:
            v = rows[R].get(n)
            cells.append(f"{v:12.4f}" if isinstance(v, (int, float)) else f"{'--':>12s}")
        print(f"r{R:<4d} " + "  ".join(cells))
    print("=" * 100 + "\n")

    # CSV
    with (OUT / "rank_sweep_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank"] + metric_names)
        for R in RANKS:
            w.writerow([R] + [rows[R].get(n) for n in metric_names])

    # LaTeX (transposed: metrics as rows, ranks as columns -- compact for paper)
    arrows = {s[0]: s[1] for s in METRIC_SPECS}
    with (OUT / "rank_sweep_table.tex").open("w") as f:
        f.write("\\begin{table}[h]\\centering\\small\n")
        f.write("\\setlength{\\tabcolsep}{5pt}\\renewcommand{\\arraystretch}{1.15}\n")
        f.write("\\begin{tabular}{l" + "c" * len(RANKS) + "}\n\\toprule\n")
        f.write("\\textbf{Metric} & " +
                " & ".join(f"\\textbf{{r{R}}}" for R in RANKS) + " \\\\\n\\midrule\n")
        for n in metric_names:
            arr = arrows.get(n, "")
            arr_tex = {"^": "$\\uparrow$", "v": "$\\downarrow$", "": ""}[arr]
            label = n.replace("_", "\\_")
            cells = []
            for R in RANKS:
                v = rows[R].get(n)
                cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")
            f.write(f"{label} {arr_tex} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Rank sweep on the v5 configuration "
                "(KL$=0.5$, attention+MLP LoRA, no inference prefix); all "
                "metrics on $N=80$ held-out prompts, best-val\\_CE checkpoint "
                "per rank. $\\uparrow$/$\\downarrow$ indicate the better "
                "direction.}\n")
        f.write("\\label{tab:rank_sweep}\n\\end{table}\n")
    print(f"  wrote {OUT/'rank_sweep_table.csv'}")
    print(f"  wrote {OUT/'rank_sweep_table.tex'}")

    # ---- 2. Two normalised panels (image-gen | text), saved separately ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -> skipping figure")
        return

    def normalise(mname, lower_better):
        """Return (xs, ys_norm): orient higher=better, then min-max to [0,1]
        over the sweep so metrics of different scales share one y-axis.
        NOTE: min-max amplifies flat-but-noisy metrics -- the absolute values
        live in the table; the panel shows relative trend only."""
        xs, vals = [], []
        for R in RANKS:
            v = rows[R].get(mname)
            if isinstance(v, (int, float)):
                xs.append(R); vals.append(v)
        if not vals:
            return [], []
        vmin, vmax = min(vals), max(vals)
        rng = (vmax - vmin) or 1.0
        ys = [(v - vmin) / rng for v in vals]
        if lower_better:                  # flip so up = better
            ys = [1.0 - y for y in ys]
        return xs, ys

    def make_panel(spec, title, fname):
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        for mname, lower_better, label in spec:
            xs, ys = normalise(mname, lower_better)
            if xs:
                ax.plot(xs, ys, "o-", lw=1.8, ms=6, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(R) for R in RANKS])
        ax.set_xlabel("LoRA rank")
        ax.set_ylabel("normalised (higher = better)")
        ax.set_ylim(-0.08, 1.08)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(OUT / f"{fname}.pdf")
        fig.savefig(OUT / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {OUT/(fname+'.pdf')} (+ .png)")

    make_panel(IMAGE_PANEL, "Image generation (acquisition)", "rank_sweep_image")
    make_panel(TEXT_PANEL,  "Text & understanding (interference)", "rank_sweep_text")


if __name__ == "__main__":
    main()
