#!/usr/bin/env python3
"""
generate_notebooks.py
Converts .py files with # %% cell markers to proper .ipynb Jupyter notebooks.

Run once on Snellius after git pull:
    python scripts/generate_notebooks.py

Output:
    notebooks/viz_depth_v2.ipynb
    notebooks/viz_caption_forgetting.ipynb
"""

import json
import re
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / 'notebooks'

TARGETS = [
    'viz_depth_v2.py',
    'viz_caption_forgetting.py',
]


def py_to_ipynb(py_path: Path) -> dict:
    text = py_path.read_text(encoding='utf-8')

    # Match cell markers: # %% or # %% [markdown]
    pattern = re.compile(r'^# %%(?:\s*\[([^\]]*)\])?\s*\n', re.MULTILINE)
    matches = list(pattern.finditer(text))

    cells = []
    for i, m in enumerate(matches):
        cell_type_hint = m.group(1) or ''
        content_start  = m.end()
        content_end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw            = text[content_start:content_end].rstrip('\n')

        if not raw.strip():
            continue

        if 'markdown' in cell_type_hint.lower():
            # Strip leading '# ' from markdown lines
            lines = []
            for line in raw.split('\n'):
                if line.startswith('# '):
                    lines.append(line[2:])
                elif line.rstrip() == '#':
                    lines.append('')
                else:
                    lines.append(line)
            raw = '\n'.join(lines)
            cell_type = 'markdown'
        else:
            cell_type = 'code'

        # Convert to source list (each line ends with \n, last line does not)
        lines  = raw.split('\n')
        source = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

        cell = {
            'cell_type': cell_type,
            'id': f'cell_{i:03d}',
            'metadata': {},
            'source': source,
        }
        if cell_type == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

        cells.append(cell)

    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': '4m-peft',
                'language': 'python',
                'name': '4m-peft',
            },
            'language_info': {
                'name': 'python',
                'version': '3.10.0',
            },
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


if __name__ == '__main__':
    for target in TARGETS:
        py_path    = NOTEBOOKS_DIR / target
        ipynb_path = NOTEBOOKS_DIR / target.replace('.py', '.ipynb')

        if not py_path.exists():
            print(f'  SKIP {py_path} — not found')
            continue

        nb = py_to_ipynb(py_path)
        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f'  {ipynb_path.name}  ({len(nb["cells"])} cells)')

    print('Done — open the .ipynb files in JupyterLab.')
