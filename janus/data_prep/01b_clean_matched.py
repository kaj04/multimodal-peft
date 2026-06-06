#!/usr/bin/env python3
"""
janus/data_prep/01b_clean_matched.py

Post-hoc cleaning of wiki_matched_naruto.jsonl: the initial keyword-based
filter has ~50% false positives caused by SYSTEMATIC ambiguities --
"Naruto" is both a fictional franchise AND a real Japanese city (with a
railway line, a bridge, a stadium); "Akatsuki" is both the criminal
organisation in the franchise AND the name of multiple Imperial Japanese
Navy destroyer classes; "Shinobi" is a generic Japanese term and the name
of unrelated Sega games. Title-only keyword matching cannot disambiguate
these without context.

This script applies a blacklist of phrases that almost certainly indicate a
NON-franchise article, plus a small whitelist that overrides the blacklist
when the title contains an unambiguous franchise marker (e.g. "Naruto:",
"Naruto Shippuden", "Boruto").

Run on the login node (no GPU, no dataset streaming):
    python ~/multimodal-peft/janus/data_prep/01b_clean_matched.py

Output: wiki_matched_naruto_clean.jsonl in the same directory + a report
of dropped/kept titles for manual audit.
"""

import json
import os
import re
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    "OUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/data_prep/output")))

INPUT  = DATA_DIR / "wiki_matched_naruto.jsonl"
OUTPUT = DATA_DIR / "wiki_matched_naruto_clean.jsonl"
REPORT = DATA_DIR / "wiki_matched_clean_report.txt"

# ---------------------------------------------------------------------------
# WHITELIST: titles that include any of these stay even if blacklist matches.
# Unambiguous franchise markers.
# ---------------------------------------------------------------------------
WHITELIST_TITLE = [
    "Naruto:", "Naruto Shippuden", "Naruto Shippuuden",
    "Naruto the Movie", "Naruto Uzumaki", "List of Naruto",
    "Boruto", "Masashi Kishimoto",
]

# Famous franchise character/jutsu names (exact-word match in title) -> keep.
FRANCHISE_CHARACTERS = {
    "naruto uzumaki", "sasuke uchiha", "sakura haruno", "kakashi hatake",
    "itachi uchiha", "obito uchiha", "madara uchiha", "minato namikaze",
    "jiraiya", "tsunade", "orochimaru", "hashirama", "tobirama",
    "gaara", "hinata", "neji hyuga", "rock lee", "shikamaru", "choji",
    "ino yamanaka", "kiba inuzuka", "shino aburame", "tenten",
    "kabuto", "pain", "nagato", "konan", "deidara", "sasori", "hidan",
    "kakuzu", "kisame", "zabuza", "haku", "danzo", "iruka",
    "rasengan", "chidori", "sharingan", "byakugan", "rinnegan",
    "mangekyo", "amaterasu", "tsukuyomi", "susanoo",
}

# ---------------------------------------------------------------------------
# BLACKLIST: title contains any of these (case-insensitive substring) ->
# drop UNLESS the title also matches the whitelist.
# Each entry's comment explains the false-positive category it kills.
# ---------------------------------------------------------------------------
BLACKLIST_TITLE = [
    # Railway: Naruto Line + stations on it (real city of Naruto, Tokushima)
    " Station",         # "Bando Station", "Konoha Station", ...
    "Naruto Line",
    # Naval: Imperial Japanese Navy ships named Akatsuki
    "destroyer",
    "-class destroyer",
    "Japanese destroyer",
    "Japanese ship",
    "Japanese cruiser",
    # Other Naruto-city geography (round 2: also Kaikyo Bridge, Expressway,
    # Itano Kofun Cluster -- ancient burial mounds, Gokuraku-ji -- temple)
    "Onaruto Bridge",
    "Ōnaruto Bridge",
    "Naruto Bridge",
    "Naruto Strait",
    "Naruto Whirlpools",
    "Naruto, Tokushima",
    "Kaikyo Bridge",
    "Expressway",
    "Kofun Cluster",
    "Gokuraku-ji",
    # Stadiums
    "Stadium",
    # Sumo and unrelated bios
    " stable (",          # sumo stables: "Tagonoura stable (2013)"
    "Takanosato",
    "Kōtetsuyama",        # sumo wrestler whose bio mentions Akatsuki
    "Kotetsuyama",
    # Polish surname Narutowicz (Gabriel/Joanna + Polish presidential pages)
    "Narutowicz",
    "Polish presidential",
    "Plac Narutowicza",
    # Sega 'Shinobi' games unrelated to the franchise
    "Shadow Dancer",
    "G.G. Shinobi",
    "Shinobi III",
    "Shinobi (",
    "Shinobi vs",
    "Akatsuki Blitzkampf",
    # Period drama, songs, train named Akatsuki, etc.
    "Shinobi Tabi",
    "Akatsuki (train)",
    "Akatsuki/",
    "Hichō",              # "Naruto Hichō (1977 TV series)" -- unrelated period drama
    "Hicho",
    # WWII history
    "prisoner-of-war camp",
    # Disambiguation pages: keep franchise-specific ones, drop generic
    "(disambiguation)",
]

# ---------------------------------------------------------------------------
# Anti-bio lead phrases: if the lead matches AND the title is not unambiguous,
# drop it (catches voice actors / wrestlers / singers / etc.).
# ---------------------------------------------------------------------------
BIO_LEAD_PATTERNS = [
    # Voice actors / voice directors / casting directors (round 2: directors too)
    re.compile(r"\bis a(n)? (japanese|american|british) voice actor", re.I),
    re.compile(r"\bis a(n)? (japanese|american|british) voice actress", re.I),
    re.compile(r"\bis a(n)? (japanese|american|british)? ?voice director", re.I),
    re.compile(r"\bis a(n)? (japanese|american|british)? ?casting director", re.I),
    # Singers / musicians / wrestlers / footballers
    re.compile(r"\bis a(n)? japanese (singer|musician|actor|actress|wrestler|sumo|footballer)", re.I),
    re.compile(r"\bis a(n)? american (singer|musician|actor|actress)", re.I),
    re.compile(r"\bis a(n)? (former )?sumo wrestler", re.I),
    re.compile(r"\bwas a (former )?sumo wrestler", re.I),
    # Politicians (Narutowicz Polish president and similar)
    re.compile(r"\bwas a(n)? polish (politician|president|writer)", re.I),
    re.compile(r"\bis a(n)? polish (politician|writer)", re.I),
    # Generic: "X (born YYYY) is a Japanese ..." -- catches many remaining bios
    re.compile(r"\(born [12]\d{3}[^)]*\)\s+is a", re.I),
]


def is_whitelisted(title: str) -> bool:
    tl = title.lower()
    for w in WHITELIST_TITLE:
        if w.lower() in tl:
            return True
    for c in FRANCHISE_CHARACTERS:
        # word-boundary match
        if re.search(r"\b" + re.escape(c) + r"\b", tl):
            return True
    return False


def is_blacklisted_title(title: str) -> bool:
    tl = title.lower()
    return any(b.lower() in tl for b in BLACKLIST_TITLE)


def is_bio_lead(text: str) -> bool:
    lead = text[:400]
    return any(p.search(lead) for p in BIO_LEAD_PATTERNS)


def main():
    if not INPUT.exists():
        raise SystemExit(f"input not found: {INPUT}")

    kept, dropped = [], []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            title = row["title"]
            text  = row["text"]

            if is_whitelisted(title):
                kept.append(row)
                continue

            reason = None
            if is_blacklisted_title(title):
                reason = "blacklist_title"
            elif is_bio_lead(text):
                reason = "bio_lead"

            if reason is None:
                kept.append(row)
            else:
                dropped.append((title, reason))

    # Write cleaned JSONL
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    # Write audit report
    with REPORT.open("w", encoding="utf-8") as f:
        f.write(f"INPUT   : {INPUT.name}  (n={len(kept)+len(dropped)})\n")
        f.write(f"KEPT    : {OUTPUT.name}  (n={len(kept)})\n")
        f.write(f"DROPPED : n={len(dropped)}\n\n")
        f.write("=" * 70 + "\n")
        f.write("DROPPED (with reason):\n")
        f.write("=" * 70 + "\n")
        for t, r in sorted(dropped, key=lambda x: x[1]):
            f.write(f"  [{r:>16s}] {t}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEPT (titles, alphabetical) -- audit manually:\n")
        f.write("=" * 70 + "\n")
        for row in sorted(kept, key=lambda r: r["title"].lower()):
            f.write(f"  {row['title']}\n")

    print(f"  input  : {len(kept) + len(dropped)} articles")
    print(f"  kept   : {len(kept)} -> {OUTPUT}")
    print(f"  dropped: {len(dropped)}")
    print(f"  audit  : {REPORT}")


if __name__ == "__main__":
    main()
