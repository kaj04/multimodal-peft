# multimodal-peft — Research PEFT (AMORe)

Repo di codice della ricerca. GitHub: `kaj04/multimodal-peft`. Su Snellius:
`~/multimodal-peft` (rinominato da `4m-peft`). Questa è la VERA radice del progetto
di codice — lancia Claude Code da qui, non dalla cartella "Research AMORe".

## Cos'è il progetto (framing LOCKED con Ivan)
Il contributo NON è "LoRA style transfer su Janus" (un specialist SD+LoRA ci batterebbe,
più economico/nitido). E' **caratterizzare la MODULARITA' del PEFT su modelli multimodali
unificati**: specializzare UNA capacita' via PEFT senza degradare (a volte migliorando) le
altre — domanda che su un specialist non esiste nemmeno. Gli stili sono il *probe*; la
batteria di cross-modal forgetting e' la prova. Serve mostrare QUANDO TIENE / QUANDO ROMPE
/ PERCHE'. Target: preprint citabile (workshop ~NeurIPS 2026, deadline ~29 ago 2026).

## Modello: Janus-Pro (DeepSeek)
Scelto perche' e' un `nn.Module` HF standard -> `get_peft_model()` funziona (criterio nato
dai fallimenti: Chameleon gen disabilitata, Anole inference-only, 4M solo modalita' immagine).
Classe `JanusForConditionalGeneration`; understanding (SigLIP) + generation (VQ-VAE) disaccoppiati;
`generation_mode="text"|"image"`. Taglie 1B e 7B. Licenza MIT/DeepSeek.

### Gotcha inference (dal smoke test)
- Generazione immagini: serve `max_new_tokens=576` esplicito (576 token VQ/immagine);
  default 20 -> immagini nere.
- `decode_image_tokens` output HWC in [-1,1]: denormalizza `(x+1)/2`, clip, `*255`. Decodifica
  in `no_grad`.
- `processor.postprocess(return_tensors="PIL.Image.Image")` non supportato: conversione manuale np.

## Research questions
- RQ1: cross-modal forgetting (LoRA su understanding -> quanto degrada generation, e viceversa) -> transfer matrix.
- RQ2 (unica a Janus): il forgetting e' localizzato per pathway? LoRA su backbone condiviso vs encoder disaccoppiati.
- RQ3: rank sweep — guadagno task vs forgetting vs parametri.

## Task PEFT (scelti per distanza-dal-base, in parallelo)
- **image-gen = pixel art** (stile visivo distante; ricetta v5 r=48 KL=0.5).
- **text = poetry** (gutenberg-poetry) + **medicine** + **welsh** (gradiente stile->dominio->lingua).
  WELSH = esperimento TEXT headline (genuinamente non visto): PPL 40.71->14.64 (-64%),
  forgetting nullo, WikiText EN migliora 13.71->12.07, gen-CMMD 0.0017 (immagine intatta).

## Pipeline (script chiave)
- `02_train_generation.py` (image-gen), `03_train_text.py` (text, RESUME=1 default!).
- `07_forgetting_battery.py` (MMLU/POPE/MME/wikitext-PPL), `17` gen-CMMD, `18` ‖ΔW‖.
- eval-only PPL: `26_welsh_ppl.py` stile (base vs adapter su finestre identiche, senza retrain).
- runner SLURM con catene `afterok`: `run_text_*_night.sh`, `run_text_sweep.sh`.

## GOTCHA critici
- **RESUME/RUN_NAME**: `03_train_text.py` ha `RESUME=1` di default -> riusare lo stesso
  `RUN_NAME` riprende il checkpoint stale e CONTAMINA la baseline (history mostra
  baseline == ogni epoca). Usa SEMPRE un RUN_NAME fresco o `RESUME=0`, e verifica la
  baseline con una eval-only indipendente.
- **KL sweep direzione text**: KL vincola la distribuzione TEXT -> protegge l'immagine
  solo INDIRETTAMENTE. Il sweep DEVE includere KL=0 per esporre il forgetting cross-modale grezzo.

## Rigore scientifico (non negoziabile)
- SEMPRE metriche standard del paper (mai proxy inventate: token-acc NON sostituisce mIoU/CIDEr).
- SEMPRE dataset citati dai paper per confronto diretto.
- Se una metrica non si puo' calcolare pulita -> dillo esplicitamente e proponi alternativa reale.
- Baseline misurata sul pretrained PRIMA del FT, su tutti i probe (lezione "honest metrics" del 4M).

## Metriche (framework, ricerca 2026-06)
- IMMAGINE: CMMD primaria (small-N valida), KID, CLIPScore; FID/ArtFID solo a scala sui winner
  finali (FID inaffidabile a piccolo N — "Rethinking FID", CVPR 2024). ArtFID: il termine
  LPIPS-content NON si applica (generiamo da testo, no source image) -> usa CLIPScore-to-prompt.
- TESTO: MAUVE primaria ("FID per testo"), style-classifier, PPL sotto LM esterno (GPT-2, non
  circolare), distinct-n. Forgetting: wikitext-PPL/MMLU/POPE/MME (catturano MC ma MANCANO la
  degradazione generativa -> integra con MAUVE su prompt neutri).

## Ambienti (separati, vincolo HARD — transformers incompatibili)
- `janus` (transformers ufficiale, torch 2.6.0+cu118), `anole` (fork), `4m-peft` (stack 4M).
- Avvio giornaliero: `source ~/multimodal-peft/janus/env.sh`.

## Dati su Snellius
- COCO: `/scratch-shared/tesr125338/coco/` (train/val2017 + annotations).
- HF cache: `/scratch-shared/tesr125338/hf_cache` (set `HF_HOME`).
- I nodi GPU hanno internet. OnDemand file browser mostra home, NON scratch -> scrivi output
  ispezionabili sotto `~/multimodal-peft/janus/.../output/`.

## WandB
Project `4M-PEFT`, entity `francesco-colasurdo04-eindhoven-university-of-technology`.
`wandb login` una volta sul login node. Se offline: `WANDB_MODE=offline` + `wandb sync`.

## Nota: 4M postponed
Il filone 4M-21 (depth/NYU, perche' CE scende ma AbsRel no) e' POSTPONED, non abbandonato.
Repo storico `kaj04/4m-peft`. Vedi memoria `project_4m_peft`.
