#!/usr/bin/env python3
"""
janus/eval/32_mauve_table.py

Consolidate all mauve/<tag>/mauve.json into one readable table + a JSON. Pure
stdlib (login node). Columns: MAUVE-english (free-form English preserved =
forgetting) and MAUVE-<target> (acquisition) where available.
"""
import json
import os
from pathlib import Path

ROOT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output")) / "mauve"
KINDS = ["english", "poetry", "pubmed", "welsh"]


def main():
    rows = {}
    for d in sorted(ROOT.glob("*")):
        j = d / "mauve.json"
        if d.name == "_refs" or not j.exists():
            continue
        rows[d.name] = json.loads(j.read_text())

    hdr = f"  {'tag':<28}" + "".join(f"{k[:7]:>10}" for k in KINDS)
    lines = ["=" * 72, "  MAUVE (higher = closer to human text; [0,1])",
             "  english = free-form English preserved; others = target acquisition",
             "=" * 72, hdr, "-" * 72]
    for tag, r in rows.items():
        cells = ""
        for k in KINDS:
            v = r.get(k, {}).get("mauve")
            cells += f"{v:>10.3f}" if isinstance(v, (int, float)) else f"{'-':>10}"
        lines.append(f"  {tag:<28}{cells}")
    report = "\n".join(lines)
    print(report)
    (ROOT / "mauve_table.txt").write_text(report + "\n", encoding="utf-8")
    (ROOT / "mauve_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  saved -> {ROOT}/mauve_table.txt (+ .json)")


if __name__ == "__main__":
    main()
