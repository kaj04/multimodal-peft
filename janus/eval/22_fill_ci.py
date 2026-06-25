#!/usr/bin/env python3
"""
janus/eval/22_fill_ci.py

Consolidator: join the CI manifest (report/ci_manifest.tsv) with the CI JSONs
(20_confidence_intervals -> ci_table.json ; 21_bootstrap_cmmd -> cmmd_ci_*.json)
and print, per paper-table cell, the value + 95% CI and its source tag, flagging
anything still missing. No number is transcribed by eye: every cell is resolved
from a JSON via the explicit manifest mapping.

Run:  python janus/eval/22_fill_ci.py        # login node, no SBU
"""
import json
import os
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "report" / "ci_manifest.tsv"
CIDIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR", os.path.expanduser("~/multimodal-peft/janus/tests/output"))
) / "confidence_intervals"


def load_ci20():
    """tag -> {row label -> (value, lo, hi, method)} from ci_table.json."""
    f = CIDIR / "ci_table.json"
    if not f.exists():
        return {}
    idx = {}
    for block in json.loads(f.read_text()):
        hdr = block.get("block", "")
        tag = hdr.split("tag=", 1)[1].split()[0] if "tag=" in hdr else hdr
        # battery and panel blocks share the same tag -> MERGE rows, don't overwrite
        idx.setdefault(tag, {}).update(
            {r["label"]: (r["value"], r["lo"], r["hi"], r["method"])
             for r in block.get("rows", [])})
    return idx


def load_cmmd21():
    """key -> json (cmmd_ci_<key>.json with CMMD/KID fields)."""
    out = {}
    for f in glob(str(CIDIR / "cmmd_ci_*.json")):
        key = Path(f).stem[len("cmmd_ci_"):]
        out[key] = json.loads(Path(f).read_text())
    return out


def fmt(v, lo, hi):
    if v is None:
        return "MISSING"
    if lo is None or hi != hi or lo != lo:
        return f"{v:.4f}  (no SE)"
    se = (hi - lo) / 3.9199  # 95% CI half-width / 1.96 = standard error of the estimate
    return f"{v:.4f} +/- {se:.4f}   [{lo:.4f}, {hi:.4f}]"


def main():
    ci20, cmmd21 = load_ci20(), load_cmmd21()
    rows = [l.rstrip("\n").split("\t")
            for l in MANIFEST.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("table\t")]

    lines, missing = [], 0
    cur = None
    for r in rows:
        if len(r) < 6:
            continue
        table, column, metric, src, key, field = r[:6]
        if table != cur:
            lines.append(f"\n=== {table} ===")
            cur = table
        val = lo = hi = None
        note = ""
        if src == "ci20":
            cell = ci20.get(key, {}).get(field)
            if cell:
                val, lo, hi, _ = cell
            else:
                note = f"  <- ci20 tag={key} field='{field}' NOT FOUND (run 20 / re-run battery)"
        elif src == "cmmd21":
            j = cmmd21.get(key)
            if j and field in j and j[field].get("value") is not None:
                val = j[field]["value"]
                ci = j[field].get("ci95", [None, None])
                lo, hi = ci[0], ci[1]
            else:
                note = f"  <- cmmd21 key={key} field='{field}' NOT FOUND (run 21 with LABEL={key})"
        if val is None:
            missing += 1
        lines.append(f"  {column:<10} {metric:<14} {fmt(val, lo, hi):<34} (tag={key}){note}")

    report = "\n".join(lines)
    print(report)
    out = CIDIR / "ci_filled.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n  {missing} cell(s) still missing.  saved -> {out}")


if __name__ == "__main__":
    main()
