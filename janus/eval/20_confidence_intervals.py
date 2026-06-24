#!/usr/bin/env python3
"""
janus/eval/20_confidence_intervals.py

Post-hoc 95% confidence intervals for the reported metrics, WITHOUT retraining.
Reads the JSON outputs already produced by the forgetting battery (07) and the
metrics panel (06) and attaches an estimator-appropriate interval to each number.

This is the FREE pass: pure Python (no GPU, no model load) -> runs in seconds on
a login node, 0 SBU. It covers every metric whose dispersion is already on disk:
  - Proportions (MMLU, POPE, MME accuracy): Wilson score interval
      [Wilson 1927; Brown, Cai & DasGupta 2001]. Needs (accuracy, N).
  - Per-instance means (CLIP DirSim, DINO sim, CLIP-IQA): normal/CLT interval
      x_bar +/- 1.96 * s / sqrt(n)  [Efron & Tibshirani 1993]. Needs (std, n).

Set-level metrics (CMMD, KID) and perplexity need the raw samples (image
embeddings / per-chunk NLL); they are LISTED here as "bootstrap-pending" so the
report is explicit about what still needs the bootstrap pass.

Output (labelled): confidence_intervals/ci_table.txt + ci_table.json
Run:  python janus/eval/20_confidence_intervals.py        # login node, no SBU
"""
import json
import math
import os
from glob import glob
from pathlib import Path

Z = 1.959963984540054  # standard normal 97.5th percentile (95% two-sided)

OUT     = Path(os.environ.get(
    "JANUS_OUTPUT_DIR", os.path.expanduser("~/multimodal-peft/janus/tests/output")))
BATTERY = OUT / "forgetting_battery"
PANEL   = OUT / "metrics_panel"
DEST    = OUT / "confidence_intervals"


def wilson(p, n, z=Z):
    """Wilson score interval for a binomial proportion (Wilson 1927)."""
    if not n:
        return (float("nan"), float("nan"))
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half, centre + half)


def mean_ci(mean, std, n, z=Z):
    """Normal/CLT interval for a sample mean (== nonparametric bootstrap at this n)."""
    if not n or std is None:
        return (float("nan"), float("nan"))
    se = std / math.sqrt(n)
    return (mean - z * se, mean + z * se)


def ppl_ci(mean_ce, ce_std, n, z=Z):
    """CI on perplexity = exp(mean per-chunk NLL): CLT on the mean, then exp the
    endpoints (PPL is a monotone transform). Needs ce_std (battery now dumps it)."""
    if not n or ce_std is None:
        return (float("nan"), float("nan"))
    se = ce_std / math.sqrt(n)
    return (math.exp(mean_ce - z * se), math.exp(mean_ce + z * se))


def mme_score_ci(per_cat, B=2000, seed=0):
    """Parametric bootstrap on the MME total score from the saved per-category
    (acc, acc_plus, n): resample correct counts under the binomial model and
    recompute the score sum_cat (acc + acc_plus) * 100. Free (no raw data)."""
    try:
        import numpy as np
    except Exception:
        return (float("nan"), float("nan"))
    cats = [(c.get("n", 0), c.get("acc", 0.0), c.get("acc_plus", 0.0))
            for c in per_cat.values() if c.get("n")]
    if not cats:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    scores = np.zeros(B)
    for n, acc, accp in cats:
        npair = max(1, n // 2)
        scores += (rng.binomial(n, acc, B) / n
                   + rng.binomial(npair, accp, B) / npair) * 100.0
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return (float(lo), float(hi))


def row(label, value, lo, hi, method, fmt="{:.4f}"):
    v = fmt.format(value) if value is not None else "  --  "
    ci = (f"[{fmt.format(lo)}, {fmt.format(hi)}]"
          if lo == lo and hi == hi else "[bootstrap-pending]")
    return {"label": label, "value": value, "lo": lo, "hi": hi,
            "method": method, "line": f"  {label:<26} {v:>10}  {ci:<26} ({method})"}


def battery_rows(j):
    out, tag = [], j.get("tag", "?")
    mm = j.get("mmlu", {})
    if "accuracy" in mm:
        lo, hi = wilson(mm["accuracy"], mm.get("n", 0))
        out.append(row(f"MMLU acc", mm["accuracy"], lo, hi, "Wilson"))
    pope = j.get("pope", {})
    if "accuracy" in pope:
        lo, hi = wilson(pope["accuracy"], pope.get("n", 0))
        out.append(row("POPE acc", pope["accuracy"], lo, hi, "Wilson"))
    mme = j.get("mme", {})
    if "overall_accuracy" in mme:
        lo, hi = wilson(mme["overall_accuracy"], mme.get("n", 0))
        out.append(row("MME overall-acc", mme["overall_accuracy"], lo, hi, "Wilson"))
    if "total_mme_score" in mme:
        lo, hi = mme_score_ci(mme.get("per_category", {}))
        out.append(row("MME total score", mme["total_mme_score"], lo, hi,
                       "param-bootstrap", fmt="{:.1f}"))
    ppl = j.get("wikitext2_ppl", {})
    if "PPL" in ppl:
        if "ce_std" in ppl:
            lo, hi = ppl_ci(ppl["mean_CE"], ppl["ce_std"], ppl.get("n_chunks", 0))
            out.append(row("WikiText PPL", ppl["PPL"], lo, hi, "chunk-CLT", fmt="{:.3f}"))
        else:
            out.append(row("WikiText PPL", ppl["PPL"], float("nan"), float("nan"),
                           "needs ce_std (re-run battery)", fmt="{:.3f}"))
    return tag, out


def panel_rows(j):
    out = []
    tag = Path(j.get("adapter", "?")).parent.name or "?"
    n = j.get("n_gen", 0)
    after = j.get("after", {})
    delta = j.get("delta", {})
    ds = delta.get("CLIP_DirSim", {})
    if "value" in ds:
        lo, hi = mean_ci(ds["value"], ds.get("std"), ds.get("n", n))
        out.append(row("CLIP DirSim", ds["value"], lo, hi, "normal/bootstrap"))
    for key, name in (("DINO", "DINO sim (after)"), ("CLIP_IQA", "CLIP-IQA (after)")):
        m = after.get(key, {})
        if "value" in m:
            lo, hi = mean_ci(m["value"], m.get("std"), n)
            out.append(row(name, m["value"], lo, hi, "normal/bootstrap"))
    for key in ("CMMD", "KID"):
        m = after.get(key, {})
        if "value" in m and m.get("value") is not None:
            out.append(row(f"{key} (after)", m["value"], float("nan"), float("nan"),
                           "bootstrap (image sets)"))
    return tag, out


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    blocks = []

    for f in sorted(glob(str(BATTERY / "*" / "*.json"))):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not any(k in j for k in ("mmlu", "pope", "wikitext2_ppl")):
            continue
        tag, rows = battery_rows(j)
        blocks.append((f"BATTERY  tag={tag}", rows))

    for f in sorted(glob(str(PANEL / "*" / "metrics_panel.json"))):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        tag, rows = panel_rows(j)
        blocks.append((f"PANEL    tag={tag}  (caption_prefix={j.get('caption_prefix','?')!r})", rows))

    lines = ["=" * 78,
             "  95% confidence intervals (no retraining; evaluation-set sampling noise)",
             "  Wilson [Wilson1927] for proportions; normal/bootstrap [Efron-Tibshirani] for means",
             "=" * 78]
    payload = []
    for header, rows in blocks:
        lines.append("")
        lines.append(header)
        lines.append("-" * 78)
        for r in rows:
            lines.append(r["line"])
        payload.append({"block": header, "rows": [
            {k: r[k] for k in ("label", "value", "lo", "hi", "method")} for r in rows]})

    report = "\n".join(lines)
    print(report)
    (DEST / "ci_table.txt").write_text(report + "\n", encoding="utf-8")
    (DEST / "ci_table.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved -> {DEST}/ci_table.txt (+ .json)")


if __name__ == "__main__":
    main()
