#!/usr/bin/env python3
"""
janus/eval/36_chain_matrix.py   --   E2: build the retention matrix

Collects everything the two chains produced and prints, per chain, the matrix
R[k][j] = performance on target j after step k, plus the untrained capabilities
and the size of the update at each step.

Reading the matrix:
  * the DIAGONAL is acquisition: the target measured right after it was trained.
  * BELOW the diagonal is retention: a target measured after later steps. This
    is where forgetting would appear.
  * ABOVE the diagonal is pre-exposure: a target measured before it was ever
    trained. It comes free and bounds forward transfer.

Forgetting is reported as a relative change against that target's own diagonal
entry, so the sign is unambiguous for perplexity (lower is better):

    forgetting(j after k) = (PPL[k][j] - PPL[j][j]) / PPL[j][j]

positive = the target got worse, i.e. was forgotten.

The image target is scored with CMMD against REAL style images, not with
Gen-CMMD against the base. After the image step, distance from the base means
success, so only distance from the target says whether the style survived. The
standalone reference at this configuration (r=32, lambda=0.5, trigger-free) is
CMMD .108, from the image KL sweep already in the paper: an image step landing
near it acquired normally, one landing well above it was hindered by whatever
came before.

Pure stdlib, login node.
  python janus/eval/36_chain_matrix.py
"""
import json
import os
from pathlib import Path

ROOT = Path(os.path.expanduser("~/multimodal-peft/janus/tests/output"))
CHAIN_DIR = ROOT / "chain"
PPL_DIR = ROOT / "chain_ppl"
BATTERY = ROOT / "forgetting_battery"
OUT_TXT = CHAIN_DIR / "chain_matrix.txt"
OUT_JSON = CHAIN_DIR / "chain_matrix.json"

# standalone image reference at the chain configuration (paper, image KL sweep)
IMAGE_REF_CMMD = 0.108

CHAINS = {
    "A": [("welsh", "text"), ("naruto", "image"), ("medicine", "text")],
    "B": [("medicine", "text"), ("naruto", "image"), ("welsh", "text")],
}


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def step_tag(chain, i, target):
    name = "naruto" if target == "naruto" else target
    return f"chain{chain}_s{i+1}_{name}"


def collect(chain, steps):
    rows = []
    for i, (target, kind) in enumerate(steps):
        tag = step_tag(chain, i, target)
        rec = {"step": i + 1, "tag": tag, "trained": target, "kind": kind}

        # text targets: held-out PPL of every text corpus at this step
        for corpus in ("welsh", "medicine"):
            j = load_json(PPL_DIR / tag / f"ppl_{corpus}.json")
            rec[f"ppl_{corpus}"] = j.get("adapter_PPL") if j else None
            rec[f"ppl_{corpus}_base"] = j.get("base_PPL") if j else None

        # Image target: CMMD / DirSim against real style images.
        # 06_metrics_panel.py writes to <root>/<parent>__<name>/metrics_panel.json
        # where the directory is derived from the ADAPTER path, and nests the
        # values as results["after"]["CMMD"]["value"] / ["delta"]["CLIP_DirSim"].
        # The file lands at
        #   <tag>/image_metrics/metrics_panel/<parent>__<name>/metrics_panel.json
        # because 06 appends "metrics_panel" to JANUS_OUTPUT_DIR and then a
        # directory derived from the adapter path. Search recursively rather
        # than hard-coding that depth.
        rec["cmmd"] = rec["dirsim"] = None
        im_root = CHAIN_DIR / tag / "image_metrics"
        for p in sorted(im_root.rglob("metrics_panel.json")) if im_root.is_dir() else []:
            im = load_json(p)
            if not im:
                continue
            after = im.get("after", {})
            delta = im.get("delta", {})
            cm = after.get("CMMD")
            ds = delta.get("CLIP_DirSim")
            rec["cmmd"] = cm.get("value") if isinstance(cm, dict) else cm
            rec["dirsim"] = ds.get("value") if isinstance(ds, dict) else ds
            break

        # untrained capabilities
        fb = load_json(BATTERY / tag / "forgetting_results.json")
        if fb:
            rec["wikitext_ppl"] = (fb.get("wikitext2_ppl") or {}).get("PPL")
            rec["mmlu"] = (fb.get("mmlu") or {}).get("accuracy")
            rec["pope"] = (fb.get("pope") or {}).get("accuracy")
            rec["mme"] = (fb.get("mme") or {}).get("total_mme_score")
        else:
            for k in ("wikitext_ppl", "mmlu", "pope", "mme"):
                rec[k] = None
        rows.append(rec)
    return rows


def fmt(v, nd=3, width=9):
    return f"{v:>{width}.{nd}f}" if isinstance(v, (int, float)) else f"{'--':>{width}}"


def render(chain, steps, rows):
    L = [f"  CHAIN {chain}:  " + "  ->  ".join(
        f"{t} ({k})" for t, k in steps), "-" * 78]

    L.append("  Text targets, held-out PPL (lower is better)")
    L.append(f"  {'after step':<22}{'welsh':>12}{'medicine':>12}")
    for r in rows:
        mark = {"welsh": "  <-trained", "medicine": "  <-trained"}.get(r["trained"], "")
        L.append(f"  {r['step']}. {r['trained']:<19}"
                 f"{fmt(r['ppl_welsh'], 2, 12)}{fmt(r['ppl_medicine'], 2, 12)}{mark}")

    # forgetting vs each target's own diagonal entry
    diag = {}
    for r in rows:
        if r["trained"] in ("welsh", "medicine"):
            diag[r["trained"]] = r.get(f"ppl_{r['trained']}")
    L += ["", "  Forgetting: relative change vs that target's own diagonal",
          "  (positive = worse than when it was trained)"]
    for tgt, d in diag.items():
        if not d:
            continue
        later = [r for r in rows if r["step"] > next(
            x["step"] for x in rows if x["trained"] == tgt)]
        for r in later:
            v = r.get(f"ppl_{tgt}")
            if v:
                L.append(f"    {tgt:<10} after step {r['step']} ({r['trained']:<9}): "
                         f"{100.0 * (v - d) / d:+6.1f}%")

    L += ["", "  Image target, CMMD vs real style (lower is better; "
              f"standalone reference {IMAGE_REF_CMMD:.3f})",
          f"  {'after step':<22}{'CMMD':>10}{'DirSim':>10}"]
    for r in rows:
        note = ""
        if r["kind"] == "image":
            note = "  <-trained"
        elif r["cmmd"] is not None and r["step"] > 2:
            note = "  (style retained?)"
        L.append(f"  {r['step']}. {r['trained']:<19}"
                 f"{fmt(r['cmmd'], 3, 10)}{fmt(r['dirsim'], 3, 10)}{note}")

    L += ["", "  Untrained capabilities",
          f"  {'after step':<22}{'WikiText':>10}{'MMLU':>9}{'POPE':>9}{'MME':>9}"]
    for r in rows:
        L.append(f"  {r['step']}. {r['trained']:<19}"
                 f"{fmt(r['wikitext_ppl'], 2, 10)}{fmt(r['mmlu'], 3, 9)}"
                 f"{fmt(r['pope'], 3, 9)}{fmt(r['mme'], 0, 9)}")
    return L


def main():
    lines = ["=" * 78,
             "  E2 -- SEQUENTIAL CHAIN: retention matrix",
             "  diagonal = acquisition | below = retention | above = pre-exposure",
             "=" * 78]
    out = {}
    for chain, steps in CHAINS.items():
        rows = collect(chain, steps)
        out[chain] = rows
        lines += render(chain, steps, rows) + ["", "=" * 78]

    missing = sum(1 for c in out.values() for r in c
                  if r["ppl_welsh"] is None and r["cmmd"] is None)
    if missing:
        lines.append(f"  NOTE: {missing} step(s) have no results yet "
                     f"(chain still running, or a job failed).")

    report = "\n".join(lines)
    print(report)
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT_TXT}\n           {OUT_JSON}")


if __name__ == "__main__":
    main()
