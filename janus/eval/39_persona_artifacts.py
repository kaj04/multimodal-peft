#!/usr/bin/env python3
"""
janus/eval/39_persona_artifacts.py   --   E5: is the assistant persona what
moves general-text perplexity?

The paper reports that general-text perplexity falls BELOW the base after text
adaptation, on every target, and admits it has no confirmed mechanism. The
leading hypothesis is that the base model interrupts free continuations with
assistant register -- the "Hello! I'm here to help" openings visible in the
qualitative samples -- and that training on any corpus suppresses that.

This counts the behaviour instead of asserting it. It reads the free-form
English generations already produced for the MAUVE experiment (no model, no
GPU) and reports, per adapter, the fraction of continuations containing
assistant-register phrasing.

What this can and cannot claim. It establishes that the behaviour is real and
quantified rather than anecdotal. It does NOT establish causation: WikiText
perplexity is teacher-forced, so the persona never actually surfaces during
that measurement. The mechanism is indirect -- probability mass held on
assistant-style continuations is wasted at encyclopedic positions, and training
on any corpus moves it elsewhere. Written that way it is defensible; written as
"this explains the improvement" it is not.

The human reference set is scored too, as a control: a measure that fires on
real WikiText prose would be measuring nothing.

Pure stdlib, login node, seconds.
  python janus/eval/39_persona_artifacts.py
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get(
    "MAUVE_ROOT",
    os.path.expanduser("~/multimodal-peft/janus/tests/output/mauve")))
OUT_TXT = ROOT / "persona_artifacts.txt"
OUT_JSON = ROOT / "persona_artifacts.json"

# Deliberately narrow and listed one by one, so the reader can see exactly what
# was counted and nothing is hidden behind an aggregate. Each is second-person
# helper register that has no place in an encyclopedic continuation.
PATTERNS = [
    ("greeting",        r"\bHello\b|\bHi there\b"),
    ("offer_to_help",   r"here to help|happy to help|glad to help|how can I (help|assist)"),
    ("as_an_ai",        r"\bas an AI\b|\bAI language model\b|\bI'm an AI\b"),
    ("hope_helps",      r"hope (this|that) helps|let me know if you"),
    ("sure_certainly",  r"^\s*(Sure|Certainly|Of course)[!,]"),
    ("your_question",   r"your question|you asked|feel free to ask"),
]
COMPILED = [(n, re.compile(p, re.IGNORECASE | re.MULTILINE)) for n, p in PATTERNS]
SEP = "=" * 84


def score(texts):
    per_pat = {n: 0 for n, _ in COMPILED}
    hit = 0
    for t in texts:
        if not isinstance(t, str):
            continue
        any_hit = False
        for n, rx in COMPILED:
            if rx.search(t):
                per_pat[n] += 1
                any_hit = True
        hit += bool(any_hit)
    return hit, per_pat


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"  MAUVE output not found: {ROOT}")

    rows = []

    ref = ROOT / "_refs" / "ref_english.json"
    if ref.is_file():
        texts = json.loads(ref.read_text())
        hit, per = score(texts)
        rows.append(("human WikiText (control)", len(texts), hit, per))

    for d in sorted(ROOT.glob("*")):
        f = d / "gen_english.json"
        if d.name == "_refs" or not f.is_file():
            continue
        texts = json.loads(f.read_text())
        hit, per = score(texts)
        rows.append((d.name, len(texts), hit, per))

    if len(rows) <= 1:
        raise SystemExit(f"  no gen_english.json found under {ROOT}")

    L = [SEP,
         "  E5 -- ASSISTANT-PERSONA ARTIFACTS IN FREE-FORM ENGLISH",
         "  fraction of continuations containing helper register",
         "  (same generations used for MAUVE; no model is loaded)",
         SEP,
         f"  {'set':<28}{'n':>6}{'with artifact':>15}{'rate':>9}",
         "-" * 84]
    for name, n, hit, _ in rows:
        L.append(f"  {name:<28}{n:>6}{hit:>15}{hit / max(n, 1):>9.3f}")

    L += ["-" * 84, "  breakdown by pattern (a generation may match several)",
          f"  {'set':<28}" + "".join(f"{n[:9]:>11}" for n, _ in COMPILED)]
    for name, n, _, per in rows:
        L.append(f"  {name:<28}" + "".join(f"{per[p]:>11}" for p, _ in COMPILED))

    base = next((r for r in rows if r[0].lower().startswith("base")), None)
    if base:
        bn, bhit = base[1], base[2]
        brate = bhit / max(bn, 1)
        L += ["-" * 84, f"  base rate = {brate:.3f}; change per adapter:"]
        for name, n, hit, _ in rows:
            if name is base[0] or "control" in name:
                continue
            r = hit / max(n, 1)
            L.append(f"    {name:<32}{r:>7.3f}   ({100 * (r - brate) / max(brate, 1e-9):+7.1f}% vs base)")

    L.append(SEP)
    report = "\n".join(L)
    print(report)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        [{"set": n, "n": c, "with_artifact": h, "rate": h / max(c, 1),
          "per_pattern": p} for n, c, h, p in rows], indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT_TXT}\n           {OUT_JSON}")


if __name__ == "__main__":
    main()
