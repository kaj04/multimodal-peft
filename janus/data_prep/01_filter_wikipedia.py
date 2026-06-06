#!/usr/bin/env python3
"""
janus/data_prep/01_filter_wikipedia.py

Build three text corpora for the joint-training experiment by streaming
wikimedia/wikipedia (English) and classifying each article into one of
three buckets based on title+lead keywords:

  - matched : Naruto franchise (target ~100-200 articles)
  - near    : other anime/manga (target ~2000)
  - far     : physics / mathematics (target ~5000)

All three come from the SAME source -> same encyclopedic style, same average
length, controlled "form" variable; the only thing that varies is the
semantic domain. This is what allows the joint-training comparison to
actually attribute observed effects to the *domain*.

Writes JSONL files: each line is {id, url, title, text}.

Run on the login node (no GPU needed), ~20-30 min wall-clock for streaming.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

OUT_DIR = Path(os.environ.get(
    "OUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/data_prep/output")))

# Targets and safety upper bound on how many articles to scan
MATCHED_TARGET = int(os.environ.get("MATCHED_TARGET", "150"))
NEAR_TARGET    = int(os.environ.get("NEAR_TARGET",    "2000"))
FAR_TARGET     = int(os.environ.get("FAR_TARGET",     "5000"))
MAX_CHECKED    = int(os.environ.get("MAX_CHECKED", "7000000"))

# Skip articles shorter than this (too short to be useful for LM)
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "500"))

SEP = "=" * 70


# ---------------------------------------------------------------------------
# Classifiers (keyword-based, case-insensitive, title-first then text head)
# ---------------------------------------------------------------------------

# Boundary-anchored patterns reduce false positives (e.g. "Anaruto" :)
def _make_pattern(words):
    parts = [re.escape(w) for w in words]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


# MATCHED -- Naruto franchise (very narrow set of terms; high precision)
NARUTO_TITLE_RE = _make_pattern([
    "Naruto", "Boruto", "Konoha", "Konohagakure",
    "Akatsuki", "Sharingan", "Rasengan", "Chidori",
    "Jutsu", "Genjutsu", "Shinobi", "Hokage",
    "Uzumaki", "Uchiha", "Hatake", "Hyuga", "Hyuuga",
    "Sasuke", "Sakura", "Kakashi", "Itachi", "Madara",
    "Jiraiya", "Tsunade", "Orochimaru",
    "Masashi Kishimoto",
])
# Also require ANY of these in title or in the first 1500 chars (lead)
NARUTO_BROAD_RE = _make_pattern([
    "Naruto", "Boruto", "Masashi Kishimoto", "Konoha", "shinobi",
    "Akatsuki", "Hidden Leaf",
])

# NEAR -- other anime/manga (broad)
ANIME_TITLE_RE = _make_pattern([
    "anime", "manga", "shōnen", "shoujo", "shonen", "seinen",
    "One Piece", "Dragon Ball", "Bleach", "Attack on Titan",
    "Demon Slayer", "My Hero Academia", "Fullmetal Alchemist",
    "Death Note", "Tokyo Ghoul", "Sailor Moon", "Cowboy Bebop",
    "Neon Genesis Evangelion", "JoJo", "Hunter x Hunter",
    "Studio Ghibli", "Hayao Miyazaki", "MAPPA", "Toei Animation",
    "Shueisha", "Weekly Shōnen Jump", "Weekly Shonen Jump",
    "Pokémon", "Pokemon", "Yu-Gi-Oh",
])

# FAR -- physics / mathematics (broad)
PHYSICS_TITLE_RE = _make_pattern([
    "physics", "quantum", "relativity", "thermodynamics",
    "electromagnetism", "particle", "mechanics",
    "mathematics", "theorem", "equation", "geometry",
    "topology", "algebra", "calculus", "differential",
    "astrophysics", "cosmology", "string theory",
    "statistical mechanics", "field theory",
])
PHYSICS_LEAD_RE = _make_pattern([
    "physics", "quantum mechanics", "relativity", "thermodynamics",
    "particle physics", "mathematical", "theorem",
])


def classify(title: str, text: str):
    """Return one of 'matched', 'near', 'far', or None.
    Articles can only land in ONE bucket -- matched wins over near wins over far.

    The matched classifier is intentionally strict: we want articles ABOUT the
    Naruto franchise (characters, jutsu, locations, the series itself), not
    articles that merely mention Naruto in passing (voice actor bios, cast
    lists, manga publisher histories). False positives there would dilute the
    'matched domain' signal we want to measure."""
    t_lc = title.lower()
    text_lc = text.lower()
    lead = text[:2000]
    lead_lc = lead.lower()

    # Anti-false-positive: occupation/list patterns that mention Naruto but are
    # not about it (e.g. "X is a Japanese voice actor", "List of anime by Y")
    BIO_TERMS = ("voice actor", "voice actress", "mangaka",
                 "japanese actor", "japanese actress", "japanese singer",
                 "is an animator", "is a producer",
                 "list of", "discography")
    is_bio_or_list = any(b in lead_lc[:400] for b in BIO_TERMS)

    # MATCHED rule 1 (strongest): the title itself is Naruto-specific.
    # This catches "Naruto (manga)", "Naruto Uzumaki", "Sasuke Uchiha",
    # "Sharingan", "List of Naruto characters", etc.
    if NARUTO_TITLE_RE.search(title):
        # Even with Naruto title, require some lead corroboration
        if NARUTO_BROAD_RE.search(lead):
            return "matched"

    # MATCHED rule 2: title is not Naruto-named but the article is clearly
    # ABOUT the franchise. Require:
    #   - at least 4 "naruto" mentions in the full text (article subject), AND
    #   - "Naruto" appears in the very first sentence (lead 250 chars), AND
    #   - the article is NOT a bio/list (anti-false-positive)
    if (text_lc.count("naruto") >= 4
        and "naruto" in lead_lc[:250]
        and not is_bio_or_list):
        return "matched"

    # NEAR -- other anime/manga (exclude Naruto-related to keep disjoint)
    if ANIME_TITLE_RE.search(title) and not NARUTO_TITLE_RE.search(title):
        # Also exclude if the article body is mostly about Naruto
        if text_lc.count("naruto") < 4:
            return "near"

    # FAR -- physics / mathematics
    if PHYSICS_TITLE_RE.search(title) and PHYSICS_LEAD_RE.search(lead):
        return "far"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print("  Wikipedia filter for joint-training corpora")
    print(f"  targets: matched={MATCHED_TARGET} near={NEAR_TARGET} far={FAR_TARGET}")
    print(f"  output : {OUT_DIR}")
    print(SEP, flush=True)

    print("  streaming wikimedia/wikipedia 20231101.en...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    paths = {
        "matched": OUT_DIR / "wiki_matched_naruto.jsonl",
        "near":    OUT_DIR / "wiki_near_anime.jsonl",
        "far":     OUT_DIR / "wiki_far_physics.jsonl",
    }
    files = {k: open(p, "w", encoding="utf-8") for k, p in paths.items()}
    counts = {"matched": 0, "near": 0, "far": 0, "skipped_short": 0}
    targets = {"matched": MATCHED_TARGET, "near": NEAR_TARGET, "far": FAR_TARGET}

    t0 = time.time()
    checked = 0
    try:
        for row in ds:
            checked += 1
            text = row["text"]
            if len(text) < MIN_TEXT_CHARS:
                counts["skipped_short"] += 1
                if checked % 20000 == 0:
                    print(f"  checked={checked:>7d}  matched={counts['matched']:>4d}  "
                          f"near={counts['near']:>4d}  far={counts['far']:>4d}  "
                          f"(elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
                continue

            bucket = classify(row["title"], text)
            if bucket is not None and counts[bucket] < targets[bucket]:
                files[bucket].write(json.dumps({
                    "id":    row["id"],
                    "url":   row["url"],
                    "title": row["title"],
                    "text":  text,
                }) + "\n")
                counts[bucket] += 1

            if checked % 20000 == 0:
                print(f"  checked={checked:>7d}  matched={counts['matched']:>4d}  "
                      f"near={counts['near']:>4d}  far={counts['far']:>4d}  "
                      f"(elapsed {(time.time()-t0)/60:.1f} min)", flush=True)

            # All targets met -> stop early
            if (counts["matched"] >= targets["matched"]
                and counts["near"]    >= targets["near"]
                and counts["far"]     >= targets["far"]):
                print("\n  all targets reached, stopping.", flush=True)
                break

            if checked >= MAX_CHECKED:
                print(f"\n  hit MAX_CHECKED={MAX_CHECKED}, stopping.", flush=True)
                break

    finally:
        for f in files.values():
            f.close()

    elapsed = (time.time() - t0) / 60
    print("\n" + SEP)
    print("  DONE")
    print(SEP)
    print(f"  checked total : {checked}")
    print(f"  skipped short : {counts['skipped_short']}")
    print(f"  matched (Naruto)  -> {counts['matched']:>5d} written to {paths['matched']}")
    print(f"  near    (Anime)   -> {counts['near']:>5d} written to {paths['near']}")
    print(f"  far     (Physics) -> {counts['far']:>5d} written to {paths['far']}")
    print(f"  elapsed       : {elapsed:.1f} min")
    print(SEP)

    # Per-bucket stats: avg article length, sample preview
    print("\n  Sample preview from each bucket:")
    for k, p in paths.items():
        if counts[k] == 0:
            print(f"    [{k}] empty")
            continue
        lines = open(p, "r", encoding="utf-8").readlines()
        lengths = [len(json.loads(l)["text"]) for l in lines]
        mean_len = sum(lengths) // len(lengths)
        first = json.loads(lines[0])
        preview = first["text"][:200].replace("\n", " ")
        print(f"\n    [{k}]  n={len(lines)}  avg_chars={mean_len}")
        print(f"      first title: {first['title']}")
        print(f"      preview:     {preview}...")


if __name__ == "__main__":
    main()
