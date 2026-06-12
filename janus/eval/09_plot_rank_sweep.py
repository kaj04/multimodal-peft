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

# Two-panel ABSOLUTE-value figure (no normalisation -- min-max zooming makes
# the flat interference metrics look like a wild signal). ONE metric per
# panel, on its natural scale, so the real magnitude of variation is honest.
# The interference panel also draws the base-model value as a dashed line.
#   (metric_name, where, key_path, axis_label, y_range_or_None)
ACQ_PLOT  = ("CLIP_DirSim", ("delta", "CLIP_DirSim", "value"),
             "CLIP directional similarity", (-0.10, 0.25))
INT_PLOT  = ("POPE_acc", ("pope", "accuracy"),
             "POPE accuracy", (0.5, 1.0))
# Base-model (no-adapter) values for the dashed reference line in the
# interference panel; read from the battery base run if available.
BASE_TAG = "base_no_adapter"

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

    # Base-model interference value for the dashed reference line
    base_int = None
    bf = BATTERY_DIR / BASE_TAG / "forgetting_results.json"
    if bf.exists():
        base_int = _g(json.loads(bf.read_text()), *INT_PLOT[1])

    def series(key_path_or_name, from_rows=True):
        """Collect (xs, ys) of absolute values vs rank."""
        xs, ys = [], []
        for R in RANKS:
            v = rows[R].get(key_path_or_name)
            if isinstance(v, (int, float)):
                xs.append(R); ys.append(v)
        return xs, ys

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # LEFT: acquisition, single metric, absolute scale (0 = no effect)
    mname, _kp, ylabel, yr = ACQ_PLOT
    xs, ys = series(mname)
    axL.plot(xs, ys, "o-", lw=2, ms=7, color="C0")
    axL.axhline(0.0, ls=":", c="gray", lw=1)          # 0 = adapter has no effect
    axL.set_xscale("log", base=2); axL.set_xticks(RANKS)
    axL.set_xticklabels([str(R) for R in RANKS])
    axL.set_xlabel("LoRA rank"); axL.set_ylabel(ylabel)
    if yr: axL.set_ylim(*yr)
    axL.set_title("Image generation (acquisition)", fontsize=11)
    axL.grid(True, alpha=0.3)

    # RIGHT: interference, single metric, absolute scale + base reference
    mname, _kp, ylabel, yr = INT_PLOT
    xs, ys = series(mname)
    axR.plot(xs, ys, "o-", lw=2, ms=7, color="C3", label="adapter")
    if base_int is not None:
        axR.axhline(base_int, ls="--", c="k", lw=1.2,
                    label=f"base ({base_int:.3f})")
    axR.set_xscale("log", base=2); axR.set_xticks(RANKS)
    axR.set_xticklabels([str(R) for R in RANKS])
    axR.set_xlabel("LoRA rank"); axR.set_ylabel(ylabel)
    if yr: axR.set_ylim(*yr)
    axR.set_title("Understanding (interference)", fontsize=11)
    axR.grid(True, alpha=0.3); axR.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "rank_sweep_figure.pdf")
    fig.savefig(OUT / "rank_sweep_figure.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {OUT/'rank_sweep_figure.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
