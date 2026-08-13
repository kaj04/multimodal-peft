#!/usr/bin/env python3
"""
janus/eval/33_headroom.py   --   E0: how much was there to forget?

Observed accuracy on a multiple-choice benchmark mixes two things: what the
model actually knows, and what it gets right by guessing. Only the first is
losable -- luck cannot be forgotten. So "MMLU did not drop" is a weak statement
until we say how much of MMLU was real knowledge in the first place.

Assuming a two-state responder (the model either knows an item or guesses
uniformly among k options), observed accuracy decomposes as

    a_obs = a_true + (1 - a_true)/k        =>     a_true = (k*a_obs - 1)/(k - 1)

and a_true is the *headroom*: the largest fraction of items that could in
principle have flipped to wrong. Chance-corrected forgetting metrics of this
form are due to Harmon et al., "Mapping Post-Training Forgetting in Language
Models at Scale" (NeurIPS 2025 CCFM workshop).

Applies to MMLU (k=4) and to the yes/no benchmarks POPE and MME (k=2). It does
NOT apply to the MME *score* (a composite, not a proportion) nor to perplexity;
for MME we therefore use its underlying overall_accuracy and say so.

We also print POPE's yes_ratio, because the correction assumes the guessed
subset is balanced. On a balanced yes/no set a yes-biased guesser still scores
~50% on what it does not know, so the correction survives a bias -- but the
reader should be able to check that, so we report it.

Pure stdlib, runs on a login node in seconds.

  python janus/eval/33_headroom.py
  RESULTS_ROOT=/some/other/dir python janus/eval/33_headroom.py
"""
import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "RESULTS_ROOT",
    os.path.expanduser("~/multimodal-peft/janus/tests/output/forgetting_battery")))
OUT_JSON = ROOT / "headroom.json"
OUT_TXT = ROOT / "headroom.txt"
Z = 1.959963985  # 95%

# benchmark -> (json key, accuracy field, number of options k, note)
BENCHES = [
    ("MMLU", "mmlu", "accuracy", 4, ""),
    ("POPE", "pope", "accuracy", 2, ""),
    ("MME",  "mme",  "overall_accuracy", 2, "accuracy, not the composite score"),
]


def wilson(p, n, z=Z):
    """Wilson score interval for a proportion. Returns (lo, hi)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def true_knowledge(a, k):
    """Chance-corrected accuracy: the fraction the model genuinely knows."""
    return max((k * a - 1.0) / (k - 1.0), 0.0)


def analyse(tag, res):
    """One row per benchmark present in this run's results."""
    rows = []
    for label, key, accfield, k, note in BENCHES:
        blk = res.get(key)
        if not isinstance(blk, dict) or accfield not in blk:
            continue
        a = float(blk[accfield])
        n = int(blk.get("n", 0))
        lo, hi = wilson(a, n)
        rows.append({
            "tag": tag,
            "benchmark": label,
            "k": k,
            "n": n,
            "accuracy": round(a, 4),
            "luck_points": round(a - true_knowledge(a, k), 4),
            "headroom": round(true_knowledge(a, k), 4),
            # the map a -> (k*a-1)/(k-1) is increasing, so transform the endpoints
            "headroom_ci95": [round(true_knowledge(lo, k), 4),
                              round(true_knowledge(hi, k), 4)],
            "pope_yes_ratio": round(float(blk["yes_ratio"]), 4) if key == "pope"
                              and "yes_ratio" in blk else None,
            "note": note,
        })
    return rows


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"  results dir not found: {ROOT}\n"
                         f"  set RESULTS_ROOT to the forgetting_battery output dir.")

    all_rows = []
    for d in sorted(ROOT.iterdir()):
        j = d / "forgetting_results.json"
        if not j.is_file():
            continue
        try:
            res = json.loads(j.read_text())
        except Exception as e:
            print(f"  [skip] {d.name}: {type(e).__name__}: {e}")
            continue
        all_rows.extend(analyse(d.name, res))

    if not all_rows:
        raise SystemExit(f"  no forgetting_results.json found under {ROOT}")

    sep = "=" * 92
    lines = [
        sep,
        "  E0 -- HEADROOM: how much of the base accuracy was real, losable knowledge?",
        "  headroom = (k*acc - 1)/(k - 1)  [two-state responder: knows, or guesses 1-in-k]",
        "  Not defined for the MME composite score or for perplexity.",
        sep,
        f"  {'run':<30}{'bench':<7}{'k':>2}{'n':>7}{'acc':>8}{'luck':>8}"
        f"{'HEADROOM':>10}{'95% CI':>18}",
        "-" * 92,
    ]
    for r in all_rows:
        ci = f"[{r['headroom_ci95'][0]:.3f},{r['headroom_ci95'][1]:.3f}]"
        lines.append(
            f"  {r['tag'][:29]:<30}{r['benchmark']:<7}{r['k']:>2}{r['n']:>7}"
            f"{r['accuracy']:>8.4f}{r['luck_points']:>8.4f}"
            f"{r['headroom']:>10.4f}{ci:>18}")

    yr = [r for r in all_rows if r["pope_yes_ratio"] is not None]
    if yr:
        lines += ["-" * 92,
                  "  POPE yes-ratio (correction assumes the *guessed* subset is balanced;",
                  "  a value near 0.50 means answer bias is not distorting the estimate):"]
        for r in yr:
            flag = "" if abs(r["pope_yes_ratio"] - 0.5) < 0.10 else "   <-- skewed, mention it"
            lines.append(f"    {r['tag'][:40]:<42}yes_ratio = {r['pope_yes_ratio']:.3f}{flag}")

    base = [r for r in all_rows if r["tag"].lower().startswith("base")]
    if base:
        lines += ["-" * 92, "  Paper-ready (base runs only):"]
        for r in base:
            lines.append(
                f"    {r['tag']}/{r['benchmark']}: of the {r['accuracy']*100:.1f}% observed, "
                f"{r['headroom']*100:.1f} points are genuine knowledge and "
                f"{r['luck_points']*100:.1f} are chance.")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT_TXT}\n           {OUT_JSON}")


if __name__ == "__main__":
    main()
