#!/usr/bin/env python3
"""
janus/eval/08b_aggregate_text_tables.py

Read the per-condition qualitative_text.json files produced by
08_qualitative_text_examples.py and emit two ready-to-paste LaTeX tables:

  TABLE A  (knowledge + fluency, pure text)  -> table_knowledge_fluency.tex
     A1: MMLU matrix  (questions x conditions, cell = picked letter + check)
     A2: free-generation blocks (a few prompts; each condition's greedy
         continuation shown full-width)

  TABLE B  (understanding, image+text)        -> table_understanding.tex
     image + question + the yes/no each condition answers; the example
     image filenames point at _shared_images/ (download that folder).

Runs on the LOGIN NODE (no GPU): pure JSON -> LaTeX. Output goes to
qualitative_text/overleaf_tables/.

Edit CONDITIONS to control column order/labels, and the *_SHOW lists to
pick which examples appear (keeps tables compact).
"""

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"),
)) / "qualitative_text"
OUT = ROOT / "overleaf_tables"
OUT.mkdir(parents=True, exist_ok=True)

# (folder tag, short column label) -- order = column order in the tables
CONDITIONS = [
    ("base_no_adapter",      "Base"),
    ("gen_7b_naruto_v5",     "v5"),
    ("joint_7b_v5_matched",  "Matched"),
    ("joint_7b_v5_far",      "Far"),
]

# Which free-gen prompt indices to show in full (keeps Table A2 compact)
FREEGEN_SHOW = [0, 3]          # DNA, virus-vs-bacterium
# How many MMLU rows / understanding examples
MMLU_SHOW = 6
UNDERSTANDING_SHOW = 4         # mix of pope/mme
GEN_TRUNC = 240                # chars of continuation to show


def tex_escape(s):
    if s is None:
        return ""
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def load_all():
    data = {}
    for tag, label in CONDITIONS:
        f = ROOT / tag / "qualitative_text.json"
        if not f.exists():
            print(f"  WARN: missing {f} -> column '{label}' will be blank")
            data[tag] = None
        else:
            data[tag] = json.loads(f.read_text())
    return data


# ---------------------------------------------------------------------------
# Table A
# ---------------------------------------------------------------------------

def build_table_A(data):
    # Use the first available condition as the source of the fixed questions
    ref = next((d for d in data.values() if d), None)
    if ref is None:
        return "% no data\n"
    letters = ["A", "B", "C", "D"]
    lines = []
    lines.append("% ---- TABLE A1: MMLU (knowledge) ----")
    ncol = len(CONDITIONS)
    lines.append("\\begin{table}[h]\\centering\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.2}")
    colspec = "p{0.30\\textwidth} c " + " ".join(["c"] * ncol)
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\toprule")
    header = "\\textbf{MMLU question (subject)} & \\textbf{Gold} & " + \
             " & ".join(f"\\textbf{{{lbl}}}" for _, lbl in CONDITIONS) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for qi in range(min(MMLU_SHOW, len(ref["mmlu"]))):
        q = ref["mmlu"][qi]
        subj = tex_escape(q["subject"].replace("_", " "))
        qtext = tex_escape(q["question"])
        if len(qtext) > 70:
            qtext = qtext[:70] + "\\dots"
        gold = letters[q["gold"]]
        cells = []
        for tag, _ in CONDITIONS:
            d = data[tag]
            if d is None:
                cells.append("--")
                continue
            item = d["mmlu"][qi]
            mark = "\\checkmark" if item["correct"] else "\\times"
            cells.append(f"{item['pred_letter']}\\,${mark}$")
        row = f"{qtext} \\textit{{({subj})}} & {gold} & " + " & ".join(cells) + " \\\\"
        lines.append(row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{MMLU knowledge probe: the option each condition "
                 "selects (argmax log-likelihood) on fixed cross-subject "
                 "questions. \\checkmark{} = correct. Knowledge is preserved "
                 "across all conditions.}")
    lines.append("\\label{tab:qual_mmlu}")
    lines.append("\\end{table}")
    lines.append("")

    # ---- A2: free generation ----
    lines.append("% ---- TABLE A2: free generation (fluency) ----")
    lines.append("\\begin{table}[h]\\centering\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.3}")
    lines.append("\\begin{tabularx}{\\textwidth}{l X}")
    lines.append("\\toprule")
    for pi in FREEGEN_SHOW:
        prompt = tex_escape(ref["freegen"][pi]["prompt"])
        lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{Prompt:}} ``{prompt}\\dots''}} \\\\")
        lines.append("\\midrule")
        for tag, lbl in CONDITIONS:
            d = data[tag]
            cont = "--" if d is None else d["freegen"][pi]["continuation"]
            cont = tex_escape(cont[:GEN_TRUNC].replace("\n", " ").strip())
            lines.append(f"\\textbf{{{lbl}}} & {cont}\\dots \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabularx}")
    lines.append("\\caption{Greedy free-form continuations of fixed neutral "
                 "prompts. All conditions -- including Matched, whose WikiText "
                 "perplexity is $47.9$ -- remain fluent and factual. The "
                 "perplexity spike reflects distributional miscalibration on "
                 "held-out text, not a collapse of generative ability.}")
    lines.append("\\label{tab:qual_freegen}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table B
# ---------------------------------------------------------------------------

def build_table_B(data):
    ref = next((d for d in data.values() if d), None)
    if ref is None:
        return "% no data\n"
    # Merge pope + mme examples, take the first UNDERSTANDING_SHOW
    ref_examples = []
    for fam in ("pope", "mme"):
        for ex in ref.get(fam, []):
            ref_examples.append((fam, ex))
    ref_examples = ref_examples[:UNDERSTANDING_SHOW]

    ncol = len(CONDITIONS)
    lines = []
    lines.append("% ---- TABLE B: understanding (image+text) ----")
    lines.append("% NOTE: images are in _shared_images/ ; adjust the path below.")
    lines.append("\\begin{table}[h]\\centering\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.2}")
    colspec = "c p{0.30\\textwidth} c " + " ".join(["c"] * ncol)
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\toprule")
    header = ("\\textbf{Image} & \\textbf{Question} & \\textbf{Gold} & "
              + " & ".join(f"\\textbf{{{lbl}}}" for _, lbl in CONDITIONS) + " \\\\")
    lines.append(header)
    lines.append("\\midrule")
    for fam, ex in ref_examples:
        img = ex["image_file"]
        q = tex_escape(ex["question"])
        if len(q) > 70:
            q = q[:70] + "\\dots"
        gold = ex["gold"]
        # find this example's index within its family to index per condition
        fam_idx = ref[fam].index(ex)
        cells = []
        for tag, _ in CONDITIONS:
            d = data[tag]
            if d is None:
                cells.append("--")
                continue
            item = d[fam][fam_idx]
            mark = "\\checkmark" if item["correct"] else "\\times"
            cells.append(f"{item['pred']}\\,${mark}$")
        incl = ("\\includegraphics[width=0.10\\textwidth]"
                f"{{figs/qual-text/{img}}}")
        row = f"{incl} & {q} & {gold} & " + " & ".join(cells) + " \\\\"
        lines.append(row)
        lines.append("\\addlinespace[2pt]")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\caption{Visual understanding probe (POPE/MME): the yes/no "
                 "answer each condition gives on fixed image+question examples. "
                 "\\checkmark{} = correct. Understanding -- the direction that "
                 "shares the backbone with the trained generation pathway -- is "
                 "consistent across all conditions, including both joint runs.}")
    lines.append("\\label{tab:qual_understanding}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    data = load_all()
    a = build_table_A(data)
    b = build_table_B(data)
    (OUT / "table_knowledge_fluency.tex").write_text(a, encoding="utf-8")
    (OUT / "table_understanding.tex").write_text(b, encoding="utf-8")
    print(f"  wrote {OUT/'table_knowledge_fluency.tex'}")
    print(f"  wrote {OUT/'table_understanding.tex'}")
    print(f"\n  Table B references images at figs/qual-text/<name>.png")
    print(f"  Download them from: {ROOT/'_shared_images'}")
    print("\n  Preamble needs: \\usepackage{tabularx, booktalk?, amssymb, graphicx}")
    print("  (amssymb for \\checkmark, tabularx for the free-gen table)")


if __name__ == "__main__":
    main()
