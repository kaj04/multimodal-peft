#!/usr/bin/env python3
"""
janus/eval/19_domain_suitability.py

Pick a text-PEFT *domain* by two COMPARABLE, base-only signals (no training):

  1. PPL xControl : base perplexity on the domain corpus, divided by base
                    perplexity on a fixed control (WikiText). Same control for
                    every domain -> comparable. Higher = the domain's surface
                    distribution is more distant from what the base models well.
  2. MMLU acc     : base 0-shot accuracy on that domain's MMLU subjects, scored
                    by log-likelihood over {A,B,C,D} (same scorer as the
                    forgetting battery). Always 0-1 -> comparable across domains.
                    Lower = the base is weaker in that domain = more room.

Ideal target = HIGH xControl (distant) AND LOW MMLU acc (room to improve). The
two axes answer different questions: PPL = "is the *form* familiar?", MMLU =
"does it *know* the domain?". PPL alone is not enough (e.g. code can be low-PPL
yet the base can't solve it); that is exactly why we report both.

Comparability rule: every capability number is MCQ accuracy (one scorer); every
distance is xControl (one reference). Never mix in pass@k / raw scores.

Base-only, ~10-15 min on one H100. Run via 19_run_domain_suitability.sh.
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")            # empty -> base; set -> "after"
DOMAINS_FILTER = os.environ.get("DOMAINS_FILTER", "")  # comma list, e.g. "medicine"
SEED     = int(os.environ.get("SEED", "42"))
WINDOW_TOKENS    = int(os.environ.get("WINDOW_TOKENS", "512"))
MAX_WINDOWS      = int(os.environ.get("MAX_WINDOWS", "40"))
MMLU_PER_SUBJECT = int(os.environ.get("MMLU_PER_SUBJECT", "50"))
SMOKE            = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    MAX_WINDOWS, MMLU_PER_SUBJECT = 4, 5

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/tests/output"))) / "domain_suitability"

SEP = "=" * 78

# Fixed control: modern encyclopedic prose = what the base models best -> floor.
CONTROL = {"name": "wikitext", "hf": "Salesforce/wikitext",
           "config": "wikitext-103-raw-v1", "split": "test", "text_col": "text"}

# Each domain: a training corpus (for PPL) + the MMLU subjects that probe its
# knowledge (for accuracy). Edit freely; a corpus that fails to load is skipped
# (PPL shown as n/a) but its MMLU accuracy is still computed.
DOMAINS = [
    {"name": "medicine",
     "corpus": {"hf": "ccdv/pubmed-summarization", "split": "train", "text_col": "article"},
     "subjects": ["clinical_knowledge", "college_medicine", "anatomy",
                  "professional_medicine", "medical_genetics"]},
    {"name": "law",
     "corpus": {"hf": "mteb/legal_summarization", "config": "corpus",
                "split": "corpus", "text_col": "text"},
     "subjects": ["professional_law", "international_law", "jurisprudence"]},
    {"name": "code",
     "corpus": {"hf": "codeparrot/codeparrot-clean-valid", "split": "train",
                "text_col": "content"},
     "subjects": ["college_computer_science", "high_school_computer_science",
                  "machine_learning"]},
    {"name": "finance",
     "corpus": {"hf": "JanosAudran/financial-reports-sec", "config": "small_full",
                "split": "train", "text_col": "sentence"},
     "subjects": ["professional_accounting", "econometrics",
                  "high_school_macroeconomics", "high_school_microeconomics"]},
    {"name": "math",
     "corpus": {"hf": "openai/gsm8k", "config": "main", "split": "train"},
     "subjects": ["college_mathematics", "abstract_algebra",
                  "high_school_mathematics", "elementary_mathematics"]},
]

MMLU_TEMPLATE = """The following is a multiple choice question about {subject}.

{question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer:"""


def load_model():
    from transformers import JanusForConditionalGeneration, JanusProcessor
    processor = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    base.eval()
    if ADAPTER:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, ADAPTER)
        base.eval()
        print(f"  loaded adapter: {ADAPTER}", flush=True)
    return base, processor


# --- PPL on a fixed token budget -------------------------------------------

def _pick_text_col(sample_rows, preferred):
    if not sample_rows:
        return None
    keys = list(sample_rows[0].keys())
    if preferred and preferred in keys and isinstance(sample_rows[0].get(preferred), str):
        return preferred
    best, best_len = None, -1.0
    for k in keys:
        vals = [r[k] for r in sample_rows if isinstance(r.get(k), str)]
        if vals:
            mlen = sum(len(v) for v in vals) / len(vals)
            if mlen > best_len:
                best, best_len = k, mlen
    return best


def collect_text(spec, target_chars, rng):
    """Stream a corpus from the start until target_chars are gathered. Streaming
    avoids downloading multi-GB datasets just to score ~120k chars; for these
    domain corpora there is no Gutenberg-style front-matter to skip, so reading
    from the start is fine."""
    from datasets import load_dataset
    args = [spec["hf"]] + ([spec["config"]] if spec.get("config") else [])
    kw = {"split": spec.get("split", "train"), "streaming": True}
    if spec.get("revision"):
        kw["revision"] = spec["revision"]
    ds = load_dataset(*args, **kw)
    it = iter(ds)
    first = next(it)
    col = _pick_text_col([first], spec.get("text_col"))
    if col is None:
        raise ValueError("no string column found")
    buf, nchars = [], 0
    for row in [first, *it]:
        t = row.get(col)
        if isinstance(t, str) and t.strip():
            buf.append(t.strip())
            nchars += len(t)
            if nchars >= target_chars:
                break
    return "\n".join(buf), col


@torch.no_grad()
def corpus_ppl(model, tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    n = min(MAX_WINDOWS, len(ids) // WINDOW_TOKENS)
    if n == 0:
        return None
    losses = []
    for i in range(n):
        chunk = torch.tensor(ids[i * WINDOW_TOKENS:(i + 1) * WINDOW_TOKENS],
                             dtype=torch.long).unsqueeze(0).to(model.device)
        out = model(input_ids=chunk)
        logits = (out.logits if hasattr(out, "logits") else out[0])
        sl = logits[..., :-1, :].contiguous().float()
        lab = chunk[..., 1:].contiguous()
        losses.append(F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1)).item())
    return float(np.exp(float(np.mean(losses))))


# --- MMLU subject accuracy (same scorer as the battery) --------------------

def _first_tok(tok, text):
    return tok.encode(text, add_special_tokens=False)[0]


@torch.no_grad()
def _choice_logprobs(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    out = model(input_ids=ids)
    logits = (out.logits if hasattr(out, "logits") else out[0])[0, -1].float()
    logp = F.log_softmax(logits, dim=-1)
    return [logp[_first_tok(tok, c)].item() for c in (" A", " B", " C", " D")]


def subject_accuracy(model, tok, mmlu_ds, by_subject, subjects, rng):
    correct, total, per = 0, 0, {}
    for s in subjects:
        idxs = list(by_subject.get(s, []))
        if not idxs:
            per[s] = None
            continue
        rng.shuffle(idxs)
        c, t = 0, 0
        for i in idxs[:MMLU_PER_SUBJECT]:
            row = mmlu_ds[i]
            prompt = MMLU_TEMPLATE.format(
                subject=s.replace("_", " "), question=row["question"],
                a=row["choices"][0], b=row["choices"][1],
                c=row["choices"][2], d=row["choices"][3])
            ok = int(int(np.argmax(_choice_logprobs(model, tok, prompt))) == row["answer"])
            c += ok; t += 1
        per[s] = c / max(1, t)
        correct += c; total += t
    return (correct / total if total else None), total, per


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    target_chars = WINDOW_TOKENS * MAX_WINDOWS * 6
    tag = "base" if not ADAPTER else Path(ADAPTER).parent.name
    domains = [d for d in DOMAINS
               if (not DOMAINS_FILTER or d["name"] in DOMAINS_FILTER.split(","))]
    print(SEP)
    print(f"  Domain suitability | tag={tag} | MODEL={MODEL_ID}")
    print(f"  adapter={ADAPTER or '<base>'} | domains={[d['name'] for d in domains]}")
    print(f"  PPL budget {MAX_WINDOWS}x{WINDOW_TOKENS} tok | MMLU {MMLU_PER_SUBJECT}/subject")
    print(SEP, flush=True)

    model, processor = load_model()
    tok = processor.tokenizer

    # --- control PPL (denominator) ---
    print("  [control] wikitext PPL ...", flush=True)
    ctrl_text, _ = collect_text(CONTROL, target_chars, rng)
    ctrl_ppl = corpus_ppl(model, tok, ctrl_text)
    print(f"    control PPL = {ctrl_ppl:.2f}\n", flush=True)

    # --- MMLU once ---
    from datasets import load_dataset
    print("  loading cais/mmlu (all, test) ...", flush=True)
    mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject = defaultdict(list)
    for i, row in enumerate(mmlu_ds):
        by_subject[row["subject"]].append(i)

    rows = []
    for d in domains:
        name = d["name"]
        xctrl = ppl = acc = per = None
        nq = 0
        try:
            if d.get("corpus"):
                try:
                    txt, _ = collect_text(d["corpus"], target_chars, rng)
                    ppl = corpus_ppl(model, tok, txt)
                    xctrl = ppl / ctrl_ppl if (ppl and ctrl_ppl) else None
                except Exception as e:
                    print(f"  [{name}] corpus PPL SKIP: {type(e).__name__}: {e}",
                          flush=True)
            t0 = time.time()
            acc, nq, per = subject_accuracy(model, tok, mmlu_ds, by_subject,
                                            d["subjects"], rng)
            print(f"  [{name}] xControl={xctrl if xctrl is None else round(xctrl,2)}  "
                  f"MMLU_acc={acc if acc is None else round(acc,3)} (n={nq})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{name}] DOMAIN FAILED: {type(e).__name__}: {e}", flush=True)
        rows.append({"domain": name, "ppl": ppl, "xcontrol": xctrl,
                     "mmlu_acc": acc, "mmlu_n": nq, "per_subject": per,
                     "corpus": (d.get("corpus") or {}).get("hf")})

    # --- report ---
    lines = [SEP, f"DOMAIN SUITABILITY  model={MODEL_ID}  (control PPL={ctrl_ppl:.2f})",
             "  target = HIGH xControl (distant) AND LOW MMLU acc (room)", SEP,
             f"  {'domain':<10}{'xControl':>10}{'PPL':>9}{'MMLU acc':>10}{'n':>6}  corpus",
             "-" * len(SEP)]
    for r in rows:
        xc = "n/a" if r["xcontrol"] is None else f"{r['xcontrol']:.2f}x"
        pp = "n/a" if r["ppl"] is None else f"{r['ppl']:.1f}"
        ac = "n/a" if r["mmlu_acc"] is None else f"{r['mmlu_acc']:.3f}"
        lines.append(f"  {r['domain']:<10}{xc:>10}{pp:>9}{ac:>10}{r['mmlu_n']:>6}  "
                     f"{r['corpus'] or '-'}")
    lines.append("-" * len(SEP))
    cand = [r for r in rows if r["mmlu_acc"] is not None]
    if cand:
        # rank by low acc; note distance separately
        cand_sorted = sorted(cand, key=lambda r: r["mmlu_acc"])
        weak = cand_sorted[0]
        lines.append(f"  weakest capability: {weak['domain']} "
                     f"(MMLU acc={weak['mmlu_acc']:.3f})")
        far = max((r for r in rows if r["xcontrol"] is not None),
                  key=lambda r: r["xcontrol"], default=None)
        if far:
            lines.append(f"  most distant corpus: {far['domain']} "
                         f"(xControl={far['xcontrol']:.2f})")
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    (OUTPUT_DIR / f"domain_suitability_{tag}.txt").write_text(report + "\n", encoding="utf-8")
    (OUTPUT_DIR / f"domain_suitability_{tag}.json").write_text(
        json.dumps({"model": MODEL_ID, "adapter": ADAPTER or None, "tag": tag,
                    "control_ppl": ctrl_ppl, "domains": rows},
                   indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUTPUT_DIR}/domain_suitability_{tag}.txt (+ .json)")


if __name__ == "__main__":
    main()
