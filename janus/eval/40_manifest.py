
#!/usr/bin/env python3

"""

janus/eval/40_manifest.py  --  one row per adapter, every metric the paper cites.



The paper quoted the same adapter with four different perplexities because each

number was typed into the text independently, from whichever log was open at the

time. This reads the output tree instead and prints a single reconciled table:

if a number is not here, it does not go in the paper.



Defensive on purpose. Result files were written by scripts that evolved over

months, so every accessor falls back rather than raising, and anything that

could not be parsed is listed at the bottom instead of silently omitted.



    python janus/eval/40_manifest.py

"""

import json

import os

from pathlib import Path



OUT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output"))

MANIFEST = OUT / "manifest.json"



# MME ships 14 subtasks; the 1567.1 published for Janus-Pro-7B is perception

# only, so we split the total to make our number comparable to theirs.

PERCEPTION = {"existence", "count", "position", "color", "posters",

              "celebrity", "scene", "landmark", "artwork", "OCR"}



unresolved = []





def load(p):

    try:

        return json.loads(Path(p).read_text())

    except Exception as e:

        unresolved.append(f"{p}: {e}")

        return None





def num(x, *keys, where=""):

    """x may already be a number, or a dict holding one under some key."""

    if isinstance(x, (int, float)):

        return float(x)

    if isinstance(x, dict):

        # Case-insensitive: the battery stores 'PPL'. Matching only 'ppl' sent

        # this through the fallback below, which returned mean_CE (2.618) as

        # though it were the perplexity (13.71). A wrong number returned

        # quietly is worse than no number, so the fallback now announces itself.

        low = {k.lower(): v for k, v in x.items()}

        for k in keys:

            v = low.get(k.lower())

            if isinstance(v, (int, float)):

                return float(v)

        for k, v in x.items():

            if isinstance(v, (int, float)):

                unresolved.append(f"{where}: no key in {keys}; used '{k}'={v}")

                return float(v)

    return None





def mme_split(mme):

    cats = (mme or {}).get("per_category") or {}

    if not cats:

        return None, None

    p = sum(v["mme_score"] for k, v in cats.items() if k in PERCEPTION)

    c = sum(v["mme_score"] for k, v in cats.items() if k not in PERCEPTION)

    return p, c





rows = {}





def row(tag):

    return rows.setdefault(tag, {"tag": tag})





# ---- forgetting battery: WikiText / MMLU / POPE / MME --------------------

for f in sorted((OUT / "forgetting_battery").glob("*/forgetting_results.json")):

    d = load(f)

    if not d:

        continue

    r = row(f.parent.name)

    r["adapter_path"] = d.get("adapter") or "(base)"

    r["wikitext_ppl"] = num(d.get("wikitext2_ppl"), "PPL", "ppl", where=f"{f.parent.name}/wikitext")

    r["mmlu"] = num(d.get("mmlu"), "accuracy")

    r["mmlu_n"] = (d.get("mmlu") or {}).get("n")

    r["pope"] = num(d.get("pope"), "accuracy")

    r["pope_n"] = (d.get("pope") or {}).get("n")

    r["mme"] = num(d.get("mme"), "total_mme_score")

    p, c = mme_split(d.get("mme"))

    r["mme_perception"], r["mme_cognition"] = p, c

    # 1425/3000 is the full harness, 570/1000 the subsampled one. Every

    # comparison in the paper must stay inside one of the two.

    r["eval_config"] = ("full" if (r.get("mmlu_n") or 0) > 1000

                        else "sub" if r.get("mmlu_n") else "?")



# ---- weight displacement -------------------------------------------------

d = load(OUT / "weight_delta" / "weight_delta.json") or {}

for name, v in d.items():

    r = row(name)

    r["rel_dw_pct"] = round(100 * v["rel_dw_total"], 3)

    r["rank"] = v.get("r")

    r["alpha"] = v.get("alpha")



# ---- Welsh held-out perplexity ------------------------------------------

wp = OUT / "welsh_ppl" / "welsh_ppl.txt"

if wp.is_file():

    base = adap = None

    for line in wp.read_text().splitlines():

        if "base" in line and "PPL=" in line:

            base = float(line.split("PPL=")[1].split()[0])

        if "adapter" in line and "PPL=" in line:

            adap = float(line.split("PPL=")[1].split()[0])

    rows.setdefault("_welsh_ppl", {"tag": "_welsh_ppl"})

    rows["_welsh_ppl"].update({"welsh_base_ppl": base, "welsh_adapter_ppl": adap})



# ---- Gen-CMMD ------------------------------------------------------------

d = load(OUT / "gen_forgetting_lambda" / "gen_forgetting.json")

if d:

    for e in d.get("rows", []):

        if isinstance(e, dict) and e.get("name"):

            row(e["name"])["gen_cmmd"] = e.get("cmmd")



# ---- cross-modal VQA -----------------------------------------------------

d = load(OUT / "xmodal_welsh_vqa" / "expC_results.json")

if d:

    rows.setdefault("_xmodal_vqa", {"tag": "_xmodal_vqa"}).update(

        {k: v for k, v in d.items() if not isinstance(v, (list, dict))})



# ---- report --------------------------------------------------------------

COLS = [("tag", 34), ("eval_config", 6), ("wikitext_ppl", 9), ("mmlu", 7),

        ("pope", 7), ("mme", 8), ("mme_perception", 8), ("rel_dw_pct", 8),

        ("gen_cmmd", 9)]

print("=" * 104)

print("  RUN MANIFEST -- every number the paper may cite")

print("=" * 104)

print("  " + "".join(f"{c:<{w}}" for c, w in COLS))

print("-" * 104)

for tag in sorted(rows):

    r = rows[tag]

    cells = []

    for c, w in COLS:

        v = r.get(c)

        s = "--" if v is None else (f"{v:.3f}" if isinstance(v, float) and v < 10

                                    else f"{v:.1f}" if isinstance(v, float) else str(v))

        cells.append(f"{s:<{w}}")

    print("  " + "".join(cells))

print("=" * 104)



wp = rows.get("_welsh_ppl", {})

if wp.get("welsh_base_ppl"):

    b, a = wp["welsh_base_ppl"], wp["welsh_adapter_ppl"]

    print(f"  Welsh held-out PPL : {b} -> {a}   (drop {100*(1-a/b):.1f}%)")

for tag, r in rows.items():

    if r.get("mme_perception"):

        print(f"  MME split [{tag}] : perception {r['mme_perception']:.1f} "

              f"| cognition {r['mme_cognition']:.1f}  (published perception: 1567.1)")



if unresolved:

    print("\n  UNRESOLVED (present but not parsed):")

    for u in unresolved:

        print("   ", u)



MANIFEST.write_text(json.dumps(rows, indent=2), encoding="utf-8")

print(f"\n  saved -> {MANIFEST}")

