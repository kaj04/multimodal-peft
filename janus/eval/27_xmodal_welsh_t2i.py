#!/usr/bin/env python3
"""
janus/eval/27_xmodal_welsh_t2i.py

Experiment A (cross-modal collaboration): does the Welsh text-LoRA improve
text->image faithfulness when the PROMPT is in Welsh? If specializing the text
stream on Welsh helps the model act on Welsh in the generative cross-modal
direction, that is positive transfer, not just robustness.

Design (paired 2x2): a set of parallel English/Welsh prompts (same content); for
each, K seeds, we generate with {base, adapter} x {English, Welsh} prompt.

Metric: CLIPScore(image, ENGLISH reference). We deliberately score against the
ENGLISH prompt with a strong English CLIP, because CLIP text encoders are weak on
Welsh -- so the "meaning" is anchored in English and the only variables are
{model, prompt-language}. The same English reference scores all four cells.

Reads:
  base_en vs base_cy  -> the base's Welsh-comprehension GAP (precondition).
  adapter_cy vs base_cy -> the transfer effect (primary result).
  fraction of the en->cy gap CLOSED by the adapter.
  adapter_en vs base_en -> no-harm control.

SMOKE=1 -> 4 prompts, 2 seeds. Run via 27_run_xmodal_welsh_t2i.sh.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import generate_image, CLIPScorer  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")
ADAPTER  = os.environ.get("ADAPTER", "")
SEED     = int(os.environ.get("SEED", "42"))
K_SEEDS  = int(os.environ.get("K_SEEDS", "4"))
CLIP_ID  = os.environ.get("CLIP_ID", "openai/clip-vit-large-patch14")
SAVE_FIRST = int(os.environ.get("SAVE_FIRST", "6"))   # save PNGs for first N prompts
SMOKE    = os.environ.get("SMOKE", "0") == "1"
OUT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output")) / "xmodal_welsh_t2i"

# Parallel English/Welsh prompts. Concrete, CLIP-scorable, diverse. Welsh is
# hand-authored and should be sanity-checked (or regenerated with NLLB-200) before
# the final paper; the experiment is fair regardless since base and adapter see
# the identical Welsh string.
PAIRS = [
    ("a red fox sitting in a snowy forest", "llwynog coch yn eistedd mewn coedwig eira"),
    ("a black cat sleeping on a wooden chair", "cath ddu yn cysgu ar gadair bren"),
    ("a yellow school bus on a city street", "bws ysgol melyn ar stryd y ddinas"),
    ("a white sailboat on a blue sea", "cwch hwylio gwyn ar fôr glas"),
    ("a slice of chocolate cake on a plate", "tafell o gacen siocled ar blât"),
    ("a brown horse running in a green field", "ceffyl brown yn rhedeg mewn cae gwyrdd"),
    ("an old stone castle on a hill", "castell carreg hen ar fryn"),
    ("a red apple and a green pear on a table", "afal coch a gellygen werdd ar fwrdd"),
    ("a snowy mountain under a clear blue sky", "mynydd eira o dan awyr las glir"),
    ("a small wooden boat on a calm lake", "cwch pren bach ar lyn llonydd"),
    ("two children playing with a ball on the beach", "dau blentyn yn chwarae â phêl ar y traeth"),
    ("a steaming cup of coffee on a desk", "cwpan coffi yn ageru ar ddesg"),
    ("a green frog on a lily pad", "broga gwyrdd ar ddeilen lili"),
    ("a bright red fire truck on the road", "injan dân coch llachar ar y ffordd"),
    ("a bunch of yellow bananas on a market stall", "clwstwr o fananas melyn ar stondin farchnad"),
    ("a brown owl perched on a branch at night", "tylluan frown ar gangen yn y nos"),
    ("a pink rose in a glass vase", "rhosyn pinc mewn fâs wydr"),
    ("a long train crossing a stone bridge", "trên hir yn croesi pont garreg"),
    ("a sheep standing on a grassy hillside", "dafad yn sefyll ar lethr glaswelltog"),
    ("a lighthouse on a rocky coast at sunset", "goleudy ar arfordir creigiog ar fachlud"),
    ("a plate of fish and chips", "plât o bysgod a sglodion"),
    ("a child flying a colourful kite", "plentyn yn hedfan barcud lliwgar"),
    ("a wooden cabin in a pine forest", "caban pren mewn coedwig binwydd"),
    ("a silver bicycle leaning against a brick wall", "beic arian yn pwyso yn erbyn wal frics"),
    ("a basket of red strawberries", "basged o fefus coch"),
    ("a hot air balloon in a blue sky", "balŵn aer poeth mewn awyr las"),
    ("a black umbrella in the rain", "ymbarél du yn y glaw"),
    ("an orange pumpkin on a doorstep", "pwmpen oren ar garreg y drws"),
    ("a grey elephant drinking water", "eliffant llwyd yn yfed dŵr"),
    ("a stack of old books on a shelf", "pentwr o lyfrau hen ar silff"),
]
if SMOKE:
    PAIRS = PAIRS[:4]
    K_SEEDS = 2

SEP = "=" * 64


def mean_se(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def main():
    if not ADAPTER:
        raise SystemExit("set ADAPTER=/path/to/adapter_best")
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    from transformers import JanusForConditionalGeneration, JanusProcessor
    from peft import PeftModel
    print(f"{SEP}\n  Exp A: Welsh prompt -> image faithfulness | CLIP={CLIP_ID}\n"
          f"  ADAPTER={ADAPTER}  prompts={len(PAIRS)} x {K_SEEDS} seeds\n{SEP}", flush=True)
    proc = JanusProcessor.from_pretrained(MODEL_ID)
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    clip = CLIPScorer(CLIP_ID)

    rows = []
    for i, (en, cy) in enumerate(PAIRS):
        for k in range(K_SEEDS):
            seed = SEED + 1000 * i + k
            with model.disable_adapter():
                torch.manual_seed(seed); img_be = generate_image(model, proc, en)
                torch.manual_seed(seed); img_bc = generate_image(model, proc, cy)
            torch.manual_seed(seed); img_ae = generate_image(model, proc, en)
            torch.manual_seed(seed); img_ac = generate_image(model, proc, cy)
            imgs = {"base_en": img_be, "base_cy": img_bc,
                    "adapter_en": img_ae, "adapter_cy": img_ac}
            row = {"i": i, "seed": seed, "en": en, "cy": cy}
            for tag, img in imgs.items():
                row[tag] = clip.score(img, en)   # score vs ENGLISH reference
                if SMOKE or i < SAVE_FIRST:
                    img.save(OUT / "images" / f"p{i:02d}_s{k}_{tag}.png")
            rows.append(row)
        print(f"  [{i+1}/{len(PAIRS)}] {en[:40]!r}", flush=True)

    conds = ["base_en", "base_cy", "adapter_en", "adapter_cy"]
    # per-prompt means (average over seeds) for paired stats over prompts
    per_prompt = {c: [] for c in conds}
    deltas_cy = []   # adapter_cy - base_cy, per prompt
    for i in range(len(PAIRS)):
        pr = [r for r in rows if r["i"] == i]
        pm = {c: float(np.mean([r[c] for r in pr])) for c in conds}
        for c in conds:
            per_prompt[c].append(pm[c])
        deltas_cy.append(pm["adapter_cy"] - pm["base_cy"])

    summ = {c: dict(zip(("mean", "se"), mean_se(per_prompt[c]))) for c in conds}
    d_mean, d_se = mean_se(deltas_cy)
    gap = summ["base_en"]["mean"] - summ["base_cy"]["mean"]
    gap_closed = (d_mean / gap) if abs(gap) > 1e-9 else float("nan")
    n_pos = int(sum(1 for d in deltas_cy if d > 0))

    report = [SEP, "  Exp A: CLIPScore(image, English reference)  [higher = more faithful]",
              f"  n_prompts={len(PAIRS)} x {K_SEEDS} seeds   CLIP={CLIP_ID}", SEP]
    for c in conds:
        report.append(f"  {c:<12} {summ[c]['mean']:6.2f} +/- {summ[c]['se']:.2f}")
    report += [
        "-" * 64,
        f"  base Welsh-comprehension gap (base_en - base_cy) : {gap:+.2f}",
        f"  transfer effect (adapter_cy - base_cy)           : {d_mean:+.2f} +/- {d_se:.2f}"
        f"   ({n_pos}/{len(PAIRS)} prompts improved)",
        f"  fraction of the en->cy gap closed by the adapter : {100*gap_closed:.1f}%",
        f"  no-harm control (adapter_en - base_en)           : "
        f"{summ['adapter_en']['mean']-summ['base_en']['mean']:+.2f}",
        SEP]
    text = "\n".join(report)
    print("\n" + text)
    (OUT / "expA_results.txt").write_text(text + "\n", encoding="utf-8")
    (OUT / "expA_results.json").write_text(json.dumps(
        {"clip": CLIP_ID, "k_seeds": K_SEEDS, "summary": summ,
         "gap": gap, "transfer_mean": d_mean, "transfer_se": d_se,
         "gap_closed": gap_closed, "n_improved": n_pos, "rows": rows},
        indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT}/expA_results.txt (+ .json, images/)")


if __name__ == "__main__":
    main()
