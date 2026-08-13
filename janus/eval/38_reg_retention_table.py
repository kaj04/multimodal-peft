#!/usr/bin/env python3
"""
janus/eval/38_reg_retention_table.py   --   E3: assemble the table

Merges the retention numbers of all five conditions of the regularizer
ablation into one table, so nothing has to be transcribed by hand. Stale
hand-copied values have already caused one inconsistency in this project.

Two of the five conditions share an adapter with sweeps we ran earlier, so
their retention is read from results that already exist rather than recomputed:
  None (lambda=0)  = gen_7b_kl0_r32   (the lambda=0 column of the image sweep)
  KL-to-base, ours = gen_7b_rank_r32  (the rank sweep ran at lambda=0.5)

How to read it. The intervention sizes are not equal, and only three of the
five are comparable: None (1.55%), KL (1.57%) and Anole (1.56%) sit at
essentially the same weight-space displacement, so differences between them
are attributable to the objective. L2-SP moves less than half as much (0.60%),
so preserving well would be trivial for it, and logit-MSE (1.14%) barely
transfers the style at all, which makes it the interesting case of movement
that produced nothing: if it also preserves everything, then displacement per
se is not what costs retention.

Pure stdlib, login node.
"""
import json
import os
from pathlib import Path

ROOT = Path(os.path.expanduser(
    "~/multimodal-peft/janus/tests/output/forgetting_battery"))
OUT_TXT = ROOT / "reg_retention.txt"
OUT_JSON = ROOT / "reg_retention.json"

# label | run dir | ||dW||/||W|| (%) from the published ablation | matched?
CONDITIONS = [
    ("None (lambda=0)",  "gen_7b_kl0_r32",   1.55, True),
    ("L2-SP",            "gen_7b_l2_r32",    0.60, False),
    ("Logit-MSE",        "gen_7b_mse_r32",   1.14, False),
    ("KL-to-base (ours)", "gen_7b_rank_r32", 1.57, True),
    ("Anole (recon)",    "gen_7b_anole_r32", 1.56, True),
]
BASE_RUN = "base_no_adapter"


def _find(run):
    """07_forgetting_battery.py names its output folder TAG if TAG is set, and
    '<run>__<adapter>' otherwise. Both conventions are present in this project
    (gen_7b_kl0_r32 vs gen_7b_rank_r32__adapter_best), so accept either rather
    than silently reporting a run as missing."""
    exact = ROOT / run / "forgetting_results.json"
    if exact.is_file():
        return exact
    for d in sorted(ROOT.glob(f"{run}__*")):
        p = d / "forgetting_results.json"
        if p.is_file():
            return p
    return None


def read(run):
    p = _find(run)
    if p is None:
        return None
    try:
        r = json.loads(p.read_text())
    except Exception:
        return None
    return {
        "wikitext_ppl": (r.get("wikitext2_ppl") or {}).get("PPL"),
        "mmlu": (r.get("mmlu") or {}).get("accuracy"),
        "pope": (r.get("pope") or {}).get("accuracy"),
        "mme": (r.get("mme") or {}).get("total_mme_score"),
    }


def f(v, nd, w):
    return f"{v:>{w}.{nd}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"


def main():
    base = read(BASE_RUN)
    rows, data = [], {}
    for label, run, dw, matched in CONDITIONS:
        r = read(run)
        data[label] = {"run": run, "dW_pct": dw, "matched": matched,
                       "retention": r}
        rows.append((label, run, dw, matched, r))

    sep = "=" * 88
    L = [sep,
         "  E3 -- REGULARIZER ABLATION: what each objective RETAINS",
         "  rank 32, image direction, full evaluation configuration",
         "  ||dW|| is from the published acquisition table (same adapters)",
         sep,
         f"  {'condition':<20}{'||dW||':>8}{'WikiText':>10}{'MMLU':>8}"
         f"{'POPE':>8}{'MME':>8}   comparable?",
         "-" * 88]

    if base:
        L.append(f"  {'base (no adapter)':<20}{'--':>8}"
                 f"{f(base['wikitext_ppl'],2,10)}{f(base['mmlu'],3,8)}"
                 f"{f(base['pope'],3,8)}{f(base['mme'],0,8)}")
        L.append("-" * 88)

    for label, run, dw, matched, r in rows:
        tag = "matched" if matched else "not matched"
        if r is None:
            L.append(f"  {label:<20}{dw:>7.2f}%{'--':>10}{'--':>8}{'--':>8}"
                     f"{'--':>8}   {tag}   [no results yet]")
            continue
        L.append(f"  {label:<20}{dw:>7.2f}%"
                 f"{f(r['wikitext_ppl'],2,10)}{f(r['mmlu'],3,8)}"
                 f"{f(r['pope'],3,8)}{f(r['mme'],0,8)}   {tag}")

    L += ["-" * 88,
          "  'matched' = same weight-space displacement as the unregularized run,",
          "  so the objective is the only thing that differs. The other two moved",
          "  a different amount and must not be ranked against these three.",
          sep]

    have = [r for *_, r in rows if r]
    if base and len(have) >= 3:
        mme = [x["mme"] for x in have if x["mme"]]
        if mme and base["mme"]:
            span = max(mme) - min(mme)
            L += [f"  quick read: MME spans {span:.0f} points across the objectives",
                  f"  (base {base['mme']:.0f}). A small span means the objective",
                  "  governs what is acquired, not what is retained.", sep]

    report = "\n".join(L)
    print(report)
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT_TXT}\n           {OUT_JSON}")


if __name__ == "__main__":
    main()
