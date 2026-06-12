#!/usr/bin/env python3
"""
janus/eval/09b_plot_kl_sweep.py

Aggregate the KL-sweep image-generation metrics (battery skipped: the KL acts
on acquisition, not cross-modal interference) into a table + a curve vs KL.

KL=0.5 reuses the rank-sweep r=32 run (gen_7b_rank_r32). The other points are
the dedicated KL runs. x-axis is KL (linear, since 0 is included).

Outputs to tests/output/overleaf/kl_sweep/:
  kl_sweep_table.{csv,tex}, kl_sweep_image.{pdf,png}

Login node, no GPU.
"""
import csv
import json
import os
from pathlib import Path

# (KL value, run folder tag under metrics_panel/)
KL_POINTS = [
    (0.0,  "gen_7b_kl0_r32__adapter_best"),
    (0.25, "gen_7b_kl025_r32__adapter_best"),
    (0.5,  "gen_7b_rank_r32__adapter_best"),   # reused from the rank sweep
    (1.0,  "gen_7b_kl1_r32__adapter_best"),
    (1.5,  "gen_7b_kl15_r32__adapter_best"),
]

ROOT = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output")))
METRICS_DIR = ROOT / "metrics_panel"
OUT = ROOT / "overleaf" / "kl_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def _g(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k, {})
    return d if d != {} else None

# (name, lower_is_better, label)
IMAGE_METRICS = [
    ("CLIP_DirSim",   False, "CLIP DirSim"),
    ("DINO_after",    False, "DINO sim"),
    ("CMMD_after",    True,  "CMMD"),
    ("KID_after",     True,  "KID"),
    ("CLIP_IQA_after",False, "CLIP-IQA"),
]
EXTRACT = {
    "CLIP_DirSim":    lambda m: _g(m, "delta", "CLIP_DirSim", "value"),
    "DINO_after":     lambda m: _g(m, "after", "DINO", "value"),
    "CMMD_after":     lambda m: _g(m, "after", "CMMD", "value"),
    "KID_after":      lambda m: _g(m, "after", "KID", "value"),
    "CLIP_IQA_after": lambda m: _g(m, "after", "CLIP_IQA", "value"),
}


def main():
    rows = {}
    for kl, tag in KL_POINTS:
        f = METRICS_DIR / tag / "metrics_panel.json"
        m = json.loads(f.read_text()) if f.exists() else None
        if m is None:
            print(f"  WARN: missing {f}")
        rows[kl] = {n: (EXTRACT[n](m) if m else None) for n, _, _ in IMAGE_METRICS}

    names = [n for n, _, _ in IMAGE_METRICS]
    kls = [kl for kl, _ in KL_POINTS]

    # Print + CSV + LaTeX (KL as columns)
    print("\nKL " + "  ".join(f"{n[:12]:>12s}" for n in names))
    for kl in kls:
        cells = [f"{rows[kl][n]:12.4f}" if isinstance(rows[kl][n], (int, float))
                 else f"{'--':>12s}" for n in names]
        print(f"{kl:<4} " + "  ".join(cells))

    with (OUT / "kl_sweep_table.csv").open("w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["KL"] + names)
        for kl in kls:
            w.writerow([kl] + [rows[kl][n] for n in names])

    arrows = {n: ar for n, ar, _ in IMAGE_METRICS}
    with (OUT / "kl_sweep_table.tex").open("w") as fp:
        fp.write("\\begin{table}[h]\\centering\\small\n")
        fp.write("\\setlength{\\tabcolsep}{5pt}\\renewcommand{\\arraystretch}{1.15}\n")
        fp.write("\\begin{tabular}{l" + "c" * len(kls) + "}\n\\toprule\n")
        fp.write("\\textbf{Metric} & " +
                 " & ".join(f"\\textbf{{$\\lambda{{=}}{kl}$}}" for kl in kls) + " \\\\\n\\midrule\n")
        for n, ar, lab in IMAGE_METRICS:
            atex = {True: "$\\downarrow$", False: "$\\uparrow$"}[ar]
            cells = [f"{rows[kl][n]:.3f}" if isinstance(rows[kl][n], (int, float)) else "--"
                     for kl in kls]
            fp.write(f"{lab} {atex} & " + " & ".join(cells) + " \\\\\n")
        fp.write("\\bottomrule\n\\end{tabular}\n")
        fp.write("\\caption{Image-generation metrics across the KL-to-base "
                 "weight $\\lambda$ at fixed rank $r{=}32$ (v5 config). "
                 "$\\lambda{=}0.5$ is the v5 default.}\n")
        fp.write("\\label{tab:kl_sweep}\n\\end{table}\n")
    print(f"  wrote {OUT/'kl_sweep_table.tex'}")

    # Figure: single metric, ABSOLUTE scale vs KL (no normalisation -- here the
    # effect is large and real, so absolute values show the inverted-U clearly,
    # and 0 marks "adapter has no directional effect").
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing -> skip figure"); return

    PLOT_METRIC = "CLIP_DirSim"   # most diagnostic single signal
    xs, ys = [], []
    for kl in kls:
        v = rows[kl][PLOT_METRIC]
        if isinstance(v, (int, float)):
            xs.append(kl); ys.append(v)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(xs, ys, "o-", lw=2, ms=7, color="C0")
    ax.axhline(0.0, ls=":", c="gray", lw=1)   # 0 = no directional effect
    ax.set_xlabel(r"KL weight $\lambda$")
    ax.set_ylabel("CLIP directional similarity")
    ax.set_title("Style acquisition vs KL weight (r=32)", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "kl_sweep_image.pdf")
    fig.savefig(OUT / "kl_sweep_image.png", dpi=150)
    print(f"  wrote {OUT/'kl_sweep_image.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
