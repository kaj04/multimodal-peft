#!/usr/bin/env python3
"""
janus/train/03_train_text.py

LoRA fine-tuning of Janus-Pro-7B on the TEXT direction (pure next-token LM),
the reverse half of the modularity experiment. The image-gen trainer
(02_train_generation.py) specialises text->image; this one specialises the
language stream on a distant *style* (poetry), so we can later ask the
symmetric question: does a TEXT LoRA degrade image generation?

Why poetry: chosen objectively by the PPL probe (11_probe_text_ppl.py) as the
text style most distant from the base (4.04x the wikitext control) -> the
strongest weight shift -> the most informative forgetting test, mirroring how
pixel-art was picked as the distant visual style.

Design (mirrors 02 so the two directions are comparable):
  - LoRA on the SAME shared-backbone modules (q,k,v,o + gate,up,down). The
    text LoRA therefore touches ONLY the shared transformer, never the VQ /
    generation heads -> any image-gen change flows purely through the shared
    trunk, which is exactly the modularity we probe.
  - loss = text CE (+ optional KL-to-base, same regulariser as 02's gen loss
    but on text-token positions).
  - best-val_CE 'adapter_best' symlink, per-epoch adapters, resume state,
    periodic greedy text samples, W&B logging.

Data: biglam/gutenberg-poetry-corpus. Rows are single lines; we reconstruct
whole poems by grouping consecutive lines per gutenberg_id (the corpus is
ordered book-then-line), drop Gutenberg front-matter / boilerplate, and slice
each poem into fixed-length token windows.

Wall-clock budget: MAX_TRAIN_MINUTES stops cleanly mid-epoch (then still runs
final eval + saves), so a "~2-3h" night run lands on budget regardless of
throughput.

Run via 03_run_train_text_poetry.sh (needs a GPU).
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --- CONFIG (env-overridable) ----------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "deepseek-community/Janus-Pro-7B")

# Dataset (poetry). Columns auto-detected if the defaults are absent.
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME", "biglam/gutenberg-poetry-corpus")
# Some HF datasets require a config/subset name (e.g. wikimedia/wikipedia ->
# "20231101.cy" for Welsh). Empty => single-config dataset (poetry/PubMed).
HF_DATASET_CONFIG = os.environ.get("HF_DATASET_CONFIG", "")
HF_SPLIT        = os.environ.get("HF_SPLIT", "train")
LINE_COL        = os.environ.get("LINE_COL", "line")
DOC_ID_COL      = os.environ.get("DOC_ID_COL", "gutenberg_id")
DOC_MODE        = os.environ.get("DOC_MODE", "lines")   # "lines" (poetry) | "rows" (one doc per row)
TEXT_COL        = os.environ.get("HF_TEXT_COL", "")      # rows mode: text column (auto-detect if empty)

# Document construction
SKIP_HEAD_LINES = int(os.environ.get("SKIP_HEAD_LINES", "30"))   # drop per-book front-matter
MIN_DOC_LINES   = int(os.environ.get("MIN_DOC_LINES", "12"))     # ignore tiny fragments
MAX_DOCS        = int(os.environ.get("MAX_DOCS", "150"))         # books to load (diversity, bounded cost)
SEQ_LEN         = int(os.environ.get("SEQ_LEN", "512"))          # tokens per training window

# Train sizing (a wall-clock cap is the real stop; these bound the data set)
TRAIN_WINDOWS = int(os.environ.get("TRAIN_WINDOWS", "6000"))
VAL_WINDOWS   = int(os.environ.get("VAL_WINDOWS", "200"))
EPOCHS        = int(os.environ.get("EPOCHS", "3"))
MAX_TRAIN_MINUTES = float(os.environ.get("MAX_TRAIN_MINUTES", "0"))  # 0 = no cap

# Optimisation (text CE is denser than the gen loss -> a higher LR than 02)
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "8"))
LR           = float(os.environ.get("LR", "1e-4"))
LORA_RANK    = int(os.environ.get("LORA_RANK", "16"))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.environ.get(
    "TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
KL_LAMBDA    = float(os.environ.get("KL_LAMBDA", "0.5"))   # KL(p_adapter||p_base) on text
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.05"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
GRAD_CLIP    = float(os.environ.get("GRAD_CLIP", "1.0"))
ADAM_BETA1   = float(os.environ.get("ADAM_BETA1", "0.9"))
ADAM_BETA2   = float(os.environ.get("ADAM_BETA2", "0.95"))
LR_SCHEDULE  = os.environ.get("LR_SCHEDULE", "cosine").lower()
SEED         = int(os.environ.get("SEED", "42"))

# Eval / sampling cadence
EVAL_EVERY_STEPS   = int(os.environ.get("EVAL_EVERY_STEPS", "100"))
EVAL_MID_N         = int(os.environ.get("EVAL_MID_N", "40"))
SAMPLE_EVERY_STEPS = int(os.environ.get("SAMPLE_EVERY_STEPS", "150"))
SAMPLE_TOKENS      = int(os.environ.get("SAMPLE_TOKENS", "60"))

# Resume / checkpoint
SAVE_EVERY_STEPS = int(os.environ.get("SAVE_EVERY_STEPS", "200"))
RESUME           = os.environ.get("RESUME", "1") == "1"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "janus-peft")
RUN_NAME      = os.environ.get("RUN_NAME", "text_7b_poetry_v1")

OUTPUT_DIR = Path(os.environ.get(
    "JANUS_OUTPUT_DIR",
    os.path.expanduser("~/multimodal-peft/janus/train/output"),
)) / RUN_NAME

SEP = "=" * 64

# Fixed incipits for the periodic generation trace: short seeds to CONTINUE
# (the LoRA is a pure next-token LM, not instruction-tuned), so completion is
# the faithful probe of whether the poetic register was learned. Hardcoded so
# the dumps are comparable across steps/runs. Override with SAMPLE_PROMPTS
# (semicolon-separated).
_DEFAULT_INCIPITS = [
    "The autumn wind",
    "O gentle heart, what",
    "Upon the silent hill there",
    "I wandered lonely",
]
_env_sp = os.environ.get("SAMPLE_PROMPTS", "")
SAMPLE_INCIPITS = ([p.strip() for p in _env_sp.split(";") if p.strip()]
                   if _env_sp else _DEFAULT_INCIPITS)


# ---------------------------------------------------------------------------
# Data: reconstruct poems from the line-level corpus, then window them
# ---------------------------------------------------------------------------

_BOILERPLATE = (
    "project gutenberg", "gutenberg", "produced by", "transcriber",
    "end of the project", "start of the project", "http://", "https://",
    "www.", "copyright", "ebook", "e-book", "etext", "all rights reserved",
    "*** ", "distributed proofread", "produced from", "this etext",
)


def _is_boilerplate(line: str) -> bool:
    low = line.lower()
    return any(b in low for b in _BOILERPLATE)


def _detect_cols(ds):
    """Be robust to column renames: prefer the configured names, else pick the
    longest-string column for text and an int-ish column for the doc id."""
    cols = ds.column_names
    line_col = LINE_COL if LINE_COL in cols else None
    id_col = DOC_ID_COL if DOC_ID_COL in cols else None
    if line_col is None or id_col is None:
        sample = ds[0]
        if line_col is None:
            str_cols = [c for c in cols if isinstance(sample.get(c), str)]
            line_col = max(str_cols, key=lambda c: len(str(sample.get(c))),
                           default=cols[0])
        if id_col is None:
            id_col = next((c for c in cols
                           if isinstance(sample.get(c), (int, float))
                           and c != line_col), cols[0])
    return line_col, id_col


def load_poetry_documents():
    """Group consecutive lines per doc id into whole-poem documents, dropping
    per-book front-matter (first SKIP_HEAD_LINES) and boilerplate lines."""
    from datasets import load_dataset
    print(f"  loading {HF_DATASET_NAME} ({HF_SPLIT}) ...", flush=True)
    if DOC_MODE == "rows":
        # one document per row (PubMed articles, Wikipedia, code files, ...).
        # Stream so we don't download the whole corpus just to read MAX_DOCS docs.
        sds = (load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=HF_SPLIT,
                            streaming=True) if HF_DATASET_CONFIG else
               load_dataset(HF_DATASET_NAME, split=HF_SPLIT, streaming=True))
        it = iter(sds)
        first = next(it)
        col = TEXT_COL if (TEXT_COL and TEXT_COL in first) else max(
            (c for c in first if isinstance(first.get(c), str)),
            key=lambda c: len(first[c]), default=None)
        if col is None:
            raise RuntimeError("DOC_MODE=rows: no string column found")
        print(f"  text_col={col!r}  (doc-per-row, streaming)", flush=True)
        docs = []
        for row in [first, *it]:
            t = row.get(col)
            if isinstance(t, str) and t.strip():
                docs.append(t.strip())
            if len(docs) >= MAX_DOCS:
                break
        print(f"  collected {len(docs)} documents (rows mode)", flush=True)
        return docs
    ds = (load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=HF_SPLIT)
          if HF_DATASET_CONFIG else load_dataset(HF_DATASET_NAME, split=HF_SPLIT))
    line_col, id_col = _detect_cols(ds)
    print(f"  line_col={line_col!r}  doc_id_col={id_col!r}  rows={len(ds)}",
          flush=True)

    docs, cur_id, cur = [], None, []

    def flush():
        if not cur:
            return
        body = [l.strip() for l in cur[SKIP_HEAD_LINES:]
                if l and l.strip() and not _is_boilerplate(l)]
        if len(body) >= MIN_DOC_LINES:
            docs.append("\n".join(body))

    for row in ds:
        gid, ln = row[id_col], row[line_col]
        if gid != cur_id:
            flush()
            cur = []
            cur_id = gid
        if isinstance(ln, str):
            cur.append(ln)
        if len(docs) >= MAX_DOCS:
            break
    flush()
    print(f"  reconstructed {len(docs)} poem-documents", flush=True)
    return docs


def build_windows(docs, tok, need=0):
    """Tokenize each poem-document and slice into non-overlapping SEQ_LEN
    windows. Stops early once `need` windows are collected (docs are shuffled
    by the caller, so an early stop still samples across the corpus)."""
    windows = []
    for d in docs:
        ids = tok.encode(d, add_special_tokens=False)
        for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
            windows.append(ids[i:i + SEQ_LEN])
        if need and len(windows) >= need:
            break
    return windows


# ---------------------------------------------------------------------------
# Text LM loss (+ optional KL-to-base), same regulariser shape as 02
# ---------------------------------------------------------------------------

def _text_logits(base, input_ids):
    out = base(input_ids=input_ids)
    return out.logits if hasattr(out, "logits") else out[0]


def text_lm_loss(base, input_ids, kl_weight=0.0, peft_model=None):
    """Next-token CE on a (1, L) text window. Optional KL(p_adapter || p_base)
    averaged over positions, with p_base from a no-grad adapter-disabled
    forward (same input embeds -> isolates the LoRA effect). Returns
    (total_loss, ce_item, kl_item)."""
    logits = _text_logits(base, input_ids)
    logits_f = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    ce = F.cross_entropy(logits_f.reshape(-1, logits_f.size(-1)),
                         targets.reshape(-1))
    total, kl_item = ce, 0.0

    if kl_weight > 0.0 and peft_model is not None:
        with peft_model.disable_adapter():
            with torch.no_grad():
                logits_b = _text_logits(base, input_ids)[:, :-1, :].float()
        log_p_a = F.log_softmax(logits_f, dim=-1)
        log_p_b = F.log_softmax(logits_b, dim=-1)
        kl = (log_p_a.exp() * (log_p_a - log_p_b)).sum(dim=-1).mean()
        total = total + kl_weight * kl
        kl_item = kl.detach().item()
    return total, ce.detach().item(), kl_item


@torch.no_grad()
def eval_val_ce(base, val_windows, n):
    base.eval()
    losses = []
    for w in val_windows[:n]:
        ids = torch.tensor(w, dtype=torch.long).unsqueeze(0).to(base.device)
        _, ce, _ = text_lm_loss(base, ids, kl_weight=0.0)
        losses.append(ce)
    base.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def greedy(base, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(base.device)
    start = ids.shape[1]
    for _ in range(n):
        nxt = int(torch.argmax(_text_logits(base, ids)[0, -1]).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
    return tok.decode(ids[0, start:].tolist(), skip_special_tokens=True)


@torch.no_grad()
def save_text_samples(base, tok, out_root, global_step):
    out_dir = out_root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    base.eval()
    lines = [f"=== step {global_step} ===\n"]
    for inc in SAMPLE_INCIPITS:
        cont = greedy(base, tok, inc, SAMPLE_TOKENS)
        lines.append(f"[incipit] {inc}\n[cont]    {cont}\n")
    (out_dir / f"samples_step{global_step:06d}.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    try:
        import wandb as _w
        if _w.run is not None:
            _w.log({"samples/text": _w.Html("<pre>" + "\n".join(lines) + "</pre>"),
                    "step": global_step})
    except Exception:
        pass
    base.train()


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume from crash / time-limit)
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir, peft_model, optimizer, scheduler,
                    epoch, global_step, history):
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir / "adapter"))
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch, "global_step": global_step, "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
    }, out_dir / "state.pt")


def load_checkpoint_if_present(ckpt_dir, base, optimizer, scheduler):
    state_path, adapter_dir = ckpt_dir / "state.pt", ckpt_dir / "adapter"
    if not (state_path.exists() and adapter_dir.exists()):
        return 0, 0, {"epochs": []}
    from peft import PeftModel
    print(f"  RESUMING from {ckpt_dir}", flush=True)
    PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True,
                              adapter_name="default")
    s = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(s["optimizer"])
    scheduler.load_state_dict(s["scheduler"])
    torch.set_rng_state(s["torch_rng"])
    if s.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda_rng"])
    print(f"  resumed at epoch={s['epoch']} step={s['global_step']}", flush=True)
    return s["epoch"], s["global_step"], s["history"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import wandb
    from transformers import (JanusForConditionalGeneration, JanusProcessor,
                              get_cosine_schedule_with_warmup,
                              get_constant_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP)
    print(f"  LoRA TEXT fine-tune | run={RUN_NAME}")
    print(f"  dataset={HF_DATASET_NAME}{('/'+HF_DATASET_CONFIG) if HF_DATASET_CONFIG else ''}"
          f"  mode={DOC_MODE}  seq_len={SEQ_LEN}")
    print(f"  rank={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
          f"kl_lambda={KL_LAMBDA}")
    print(f"  LR={LR} epochs={EPOCHS} grad_accum={GRAD_ACCUM} "
          f"schedule={LR_SCHEDULE} max_minutes={MAX_TRAIN_MINUTES}")
    print(f"  target_modules={TARGET_MODULES}")
    print(SEP + "\n", flush=True)

    processor = JanusProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    base = JanusForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    lora_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                          target_modules=TARGET_MODULES,
                          lora_dropout=LORA_DROPOUT, bias="none",
                          task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(base, lora_cfg)
    base.train()
    trainable = [p for p in base.parameters() if p.requires_grad]
    tot = sum(p.numel() for p in base.parameters())
    tr = sum(p.numel() for p in trainable)
    print(f"  trainable params: {tr/1e6:.3f}M / {tot/1e6:.1f}M "
          f"({100.0*tr/tot:.3f}%)\n", flush=True)

    # --- data ---
    docs = load_poetry_documents()
    random.Random(SEED).shuffle(docs)        # diversity across the corpus
    need = TRAIN_WINDOWS + VAL_WINDOWS
    windows = build_windows(docs, tok, need=int(need * 1.3))
    random.Random(SEED + 1).shuffle(windows)
    val_windows = windows[:VAL_WINDOWS]
    train_windows = windows[VAL_WINDOWS:VAL_WINDOWS + TRAIN_WINDOWS]
    print(f"  windows: train={len(train_windows)} val={len(val_windows)} "
          f"(total available {len(windows)})\n", flush=True)
    if not train_windows:
        raise RuntimeError("no training windows built; check dataset / SEQ_LEN")

    wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
        "model": MODEL_ID, "direction": "text_lm", "style": "poetry",
        "dataset": HF_DATASET_NAME, "seq_len": SEQ_LEN,
        "train_windows": len(train_windows), "val_windows": len(val_windows),
        "epochs": EPOCHS, "grad_accum": GRAD_ACCUM, "lr": LR,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT, "kl_lambda": KL_LAMBDA,
        "lr_schedule": LR_SCHEDULE, "seed": SEED,
        "loss_type": "ce" + ("+kl" if KL_LAMBDA > 0 else ""),
    })

    total_steps = (len(train_windows) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=(ADAM_BETA1, ADAM_BETA2))
    if LR_SCHEDULE == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    elif LR_SCHEDULE == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        raise ValueError(f"unknown LR_SCHEDULE={LR_SCHEDULE!r}")
    print(f"  total optimizer steps={total_steps} warmup={warmup_steps}\n",
          flush=True)

    latest_dir = OUTPUT_DIR / "latest"
    start_epoch, global_step, history = 0, 0, {"epochs": []}
    if RESUME:
        start_epoch, global_step, history = load_checkpoint_if_present(
            latest_dir, base, optimizer, scheduler)

    # baseline val_CE (pre-training) for reference
    if not history["epochs"]:
        base_val = eval_val_ce(base, val_windows, VAL_WINDOWS)
        print(f"  baseline val_CE={base_val:.4f} (PPL={np.exp(base_val):.2f})\n",
              flush=True)
        wandb.log({"val/CE": base_val, "val/PPL": float(np.exp(base_val)),
                   "epoch": 0})
        history["baseline_val_CE"] = base_val

    optimizer.zero_grad()
    rng = random.Random(SEED)
    t_start = time.time()
    stop = False

    for epoch in range(start_epoch, EPOCHS):
        order = train_windows[:]
        rng.shuffle(order)
        running, running_ce, running_kl = 0.0, 0.0, 0.0
        for i, w in enumerate(order):
            ids = torch.tensor(w, dtype=torch.long).unsqueeze(0).to(base.device)
            loss, ce_item, kl_item = text_lm_loss(
                base, ids, kl_weight=KL_LAMBDA, peft_model=peft_model)
            (loss / GRAD_ACCUM).backward()
            running += loss.item(); running_ce += ce_item; running_kl += kl_item

            if (i + 1) % GRAD_ACCUM == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
                avg, avg_ce, avg_kl = (running / GRAD_ACCUM,
                                       running_ce / GRAD_ACCUM,
                                       running_kl / GRAD_ACCUM)
                running, running_ce, running_kl = 0.0, 0.0, 0.0
                cur_lr = scheduler.get_last_lr()[0]
                wandb.log({"train/loss": avg, "train/loss_ce": avg_ce,
                           "train/loss_kl": avg_kl, "train/lr": cur_lr,
                           "train/grad_norm": float(gnorm), "step": global_step})
                if global_step % 20 == 0:
                    kl_str = f" kl={avg_kl:.4f}" if KL_LAMBDA > 0 else ""
                    print(f"  epoch {epoch} step {global_step} "
                          f"({i+1}/{len(order)}) loss={avg:.4f} "
                          f"ce={avg_ce:.4f}{kl_str} lr={cur_lr:.2e}", flush=True)

                if EVAL_EVERY_STEPS > 0 and global_step % EVAL_EVERY_STEPS == 0:
                    mid = eval_val_ce(base, val_windows, EVAL_MID_N)
                    print(f"    [mid-eval @ {global_step}] "
                          f"val_CE({EVAL_MID_N})={mid:.4f} PPL={np.exp(mid):.2f}",
                          flush=True)
                    wandb.log({"val/CE_mid": mid, "val/PPL_mid": float(np.exp(mid)),
                               "step": global_step})

                if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                    save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                                    epoch, global_step, history)

                if SAMPLE_EVERY_STEPS > 0 and global_step % SAMPLE_EVERY_STEPS == 0:
                    save_text_samples(base, tok, OUTPUT_DIR, global_step)

                if (MAX_TRAIN_MINUTES > 0
                        and (time.time() - t_start) / 60 >= MAX_TRAIN_MINUTES):
                    print(f"  reached MAX_TRAIN_MINUTES={MAX_TRAIN_MINUTES}; "
                          f"stopping at step {global_step}", flush=True)
                    stop = True
                    break

        val_ce = eval_val_ce(base, val_windows, VAL_WINDOWS)
        print(f"  [epoch {epoch}] val_CE={val_ce:.4f} PPL={np.exp(val_ce):.2f}",
              flush=True)
        wandb.log({"val/CE": val_ce, "val/PPL": float(np.exp(val_ce)),
                   "epoch": epoch + 1})
        history["epochs"].append({"epoch": epoch + 1, "val_CE": val_ce,
                                  "PPL": float(np.exp(val_ce))})

        per_epoch = OUTPUT_DIR / f"adapter_epoch{epoch+1}"
        peft_model.save_pretrained(str(per_epoch))
        save_checkpoint(latest_dir, peft_model, optimizer, scheduler,
                        epoch + 1, global_step, history)
        print(f"  saved adapter -> {per_epoch}", flush=True)
        if stop:
            break

    history["minutes"] = round((time.time() - t_start) / 60, 1)

    # best-val_CE 'adapter_best' (relative symlink; pointer-file fallback)
    best_epoch = None
    if history["epochs"]:
        best = min(history["epochs"], key=lambda e: e["val_CE"])
        best_epoch = best["epoch"]
        history["best_epoch"] = best_epoch
        history["best_val_CE"] = best["val_CE"]
        best_src = OUTPUT_DIR / f"adapter_epoch{best_epoch}"
        best_link = OUTPUT_DIR / "adapter_best"
        try:
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(best_src.name)
        except OSError:
            (OUTPUT_DIR / "adapter_best.txt").write_text(
                best_src.name, encoding="utf-8")

    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2),
                                             encoding="utf-8")
    save_text_samples(base, tok, OUTPUT_DIR, global_step)
    wandb.finish()

    print("\n" + SEP)
    print("  DONE")
    if history.get("baseline_val_CE") is not None:
        print(f"  baseline val_CE : {history['baseline_val_CE']:.4f}")
    if history["epochs"]:
        print(f"  final val_CE    : {history['epochs'][-1]['val_CE']:.4f}")
        print(f"  best epoch      : {best_epoch} "
              f"(val_CE={history['best_val_CE']:.4f}) -> adapter_best")
    print(f"  adapters + history in {OUTPUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
