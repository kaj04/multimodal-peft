#!/bin/bash
#SBATCH --job-name=anole_interleaved
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/fcolasurdo/4m-peft/anole/tests/output/slurm_interleaved_%j.out
#SBATCH --error=/home/fcolasurdo/4m-peft/anole/tests/output/slurm_interleaved_%j.err

mkdir -p /home/fcolasurdo/4m-peft/anole/tests/output

echo "================================================================"
echo "  Anole-7b-v0.1 — Test 4: interleaved generation only"
echo "  Started: $(date)"
echo "================================================================"

module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate anole

export CKPT_PATH=/scratch-shared/tesr125338/anole_ckpt
export ANOLE_REPO=~/anole
export ANOLE_OUTPUT_DIR=~/4m-peft/anole/tests/output

python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ['ANOLE_REPO'])

import torch
from pathlib import Path
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from constants import (MODEL_7B_PATH, TOKENIZER_TEXT_PATH,
                       TOKENIZER_IMAGE_CFG_PATH, TOKENIZER_IMAGE_PATH)

OUTPUT_DIR = Path(os.environ['ANOLE_OUTPUT_DIR'])
BOI, EOI = 8197, 8196

def split_tokens(tokens):
    segments, current, mode = [], [], 'text'
    for tok in tokens.tolist():
        if tok == BOI:
            if current: segments.append({'type': 'text',  'tokens': torch.tensor(current)})
            current, mode = [], 'image'
        elif tok == EOI:
            if current: segments.append({'type': 'image', 'tokens': torch.tensor(current)})
            current, mode = [], 'text'
        else:
            current.append(tok)
    if current: segments.append({'type': mode, 'tokens': torch.tensor(current)})
    return segments

print('Loading model...')
model = ChameleonInferenceModel(
    str(MODEL_7B_PATH), str(TOKENIZER_TEXT_PATH),
    str(TOKENIZER_IMAGE_CFG_PATH), str(TOKENIZER_IMAGE_PATH),
)
print('Model loaded.\n')

prompts = [
    ('paris',   'Please introduce the city of Paris with pictures.'),
    ('animals', 'Show me three different animals with a brief description of each.'),
    ('food',    'Describe two Italian dishes and show what they look like.'),
]

options = Options(max_gen_len=800)

for slug, prompt in prompts:
    print(f'Prompt: {prompt}')
    out_dir = OUTPUT_DIR / 'interleaved' / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = model.generate(prompt_text=prompt, options=options)
    segments = split_tokens(tokens[0])

    text_log, img_count = [f'PROMPT: {prompt}\n'], 0
    for seg in segments:
        if seg['type'] == 'text':
            texts = model.decode_text(seg['tokens'].unsqueeze(0))
            text = texts[0].strip() if texts else ''
            if text:
                print(f'  TEXT: {text[:120]}')
                text_log.append(f'[TEXT]\n{text}\n')
        else:
            img_count += 1
            images = model.decode_image(seg['tokens'].unsqueeze(0))
            if images:
                fname = f'image_{img_count:02d}.png'
                images[0].save(str(out_dir / fname))
                print(f'  IMAGE saved → {fname}')
                text_log.append(f'[IMAGE] → {fname}\n')

    (out_dir / 'output.txt').write_text('\n'.join(text_log), encoding='utf-8')
    print(f'  Done: {img_count} image(s) — saved to {out_dir}\n')

print('All interleaved tests done.')
PYEOF

echo ""
echo "================================================================"
echo "  Done: $(date)"
echo "  Output: ~/4m-peft/anole/tests/output/interleaved/"
echo "================================================================"
