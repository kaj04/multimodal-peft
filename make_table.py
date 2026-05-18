#!/usr/bin/env python3
# make_table.py
#
# Aggregate multiple row.json files (one per LoRA run) into a markdown
# and LaTeX table for the paper.
#
# Usage:
#   python make_table.py \
#       --results \
#           /scratch-shared/tesr125338/output/eval/baseline/row.json \
#           /scratch-shared/tesr125338/output/eval/run23_caption/row.json \
#           /scratch-shared/tesr125338/output/eval/run25_semseg/row.json \
#           /scratch-shared/tesr125338/output/eval/run_det/row.json \
#           /scratch-shared/tesr125338/output/eval/run_depth/row.json \
#           /scratch-shared/tesr125338/output/eval/run_normal/row.json \
#           /scratch-shared/tesr125338/output/eval/run34_5task/row.json \
#       --output table.md

import argparse
import json
from pathlib import Path

# Columns shown in the table (metric → display name)
COLUMNS = [
    # Caption
    ('caption', 'CIDEr',           'CIDEr'),
    ('caption', 'BLEU4',           'BLEU-4'),
    # Semseg
    ('semseg',  'mIoU',            'mIoU'),
    # Detection
    ('det',     'AP50',            'AP50'),
    ('det',     'AP',              'AP'),
    # Depth
    ('depth',   'AbsRel',          'AbsRel↓'),
    ('depth',   'delta1',          'δ₁↑'),
    # Normal
    ('normal',  'mean_angle_err',  'Angle↓'),
    ('normal',  'within_11_25',    '<11.25°'),
]

# For "lower is better" metrics, mark with ↓
LOWER_IS_BETTER = {'AbsRel', 'mean_angle_err', 'Angle↓'}


def get_args():
    p = argparse.ArgumentParser(description='Aggregate row.json files into a table')
    p.add_argument('--results', nargs='+', required=True,
                   help='Paths to row.json files (one per run)')
    p.add_argument('--output',  default='table.md',
                   help='Output markdown file (also writes .tex alongside)')
    return p.parse_args()


def load_row(path):
    with open(path) as f:
        return json.load(f)


def fmt(value, lower_is_better=False):
    if value is None or value == -1:
        return '—'
    if isinstance(value, float):
        # Angle error: one decimal; others: 3 decimals or 1 decimal for %
        return f'{value:.3f}'
    return str(value)


def best_value(values, lower_is_better):
    """Return index of the best value (ignoring None/-1)."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None and v != -1]
    if not valid:
        return None
    if lower_is_better:
        return min(valid, key=lambda x: x[1])[0]
    return max(valid, key=lambda x: x[1])[0]


def build_table(rows):
    """rows: list of dicts from row.json"""
    col_headers = [c[2] for c in COLUMNS]
    run_names   = [r.get('run_name', '?') for r in rows]

    # Extract values
    data = []  # data[row_idx][col_idx] = float or None
    for row in rows:
        vals = []
        for task, metric, _ in COLUMNS:
            task_data = row.get(task, {})
            if isinstance(task_data, dict):
                v = task_data.get(metric, None)
                vals.append(float(v) if v is not None and v != -1 else None)
            else:
                vals.append(None)
        data.append(vals)

    # Find best per column (for bold)
    best_idx = []
    for col_i, (_, _, display) in enumerate(COLUMNS):
        col_vals = [data[row_i][col_i] for row_i in range(len(rows))]
        lb = display in LOWER_IS_BETTER
        best_idx.append(best_value(col_vals, lb))

    return run_names, col_headers, data, best_idx


def to_markdown(run_names, col_headers, data, best_idx):
    # Group columns by task
    task_groups = {}
    for i, (task, metric, display) in enumerate(COLUMNS):
        task_groups.setdefault(task, []).append(i)

    # Header
    lines = []
    header = '| Run | ' + ' | '.join(col_headers) + ' |'
    sep    = '|' + '---|' * (len(col_headers) + 1)
    lines.append(header)
    lines.append(sep)

    for row_i, run_name in enumerate(run_names):
        cells = []
        for col_i, v in enumerate(data[row_i]):
            s = fmt(v)
            if best_idx[col_i] == row_i and v is not None:
                s = f'**{s}**'
            cells.append(s)
        lines.append(f'| {run_name} | ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)


def to_latex(run_names, col_headers, data, best_idx):
    n_cols = len(col_headers)
    lines  = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l' + 'c' * n_cols + '}')
    lines.append(r'\toprule')

    # Task header row
    task_spans = {}
    for i, (task, _, _) in enumerate(COLUMNS):
        task_spans.setdefault(task, []).append(i)

    task_header = 'Model'
    for task, cols in task_spans.items():
        task_header += f' & \\multicolumn{{{len(cols)}}}{{c}}{{{task.capitalize()}}}'
    lines.append(task_header + r' \\')

    # Cmidrule per task group
    for task, cols in task_spans.items():
        start = cols[0] + 2   # +1 for Model col, +1 for 1-indexed
        end   = cols[-1] + 2
        lines.append(f'\\cmidrule(lr){{{start}-{end}}}')

    # Metric header
    metric_header = 'Model & ' + ' & '.join(col_headers)
    lines.append(metric_header + r' \\')
    lines.append(r'\midrule')

    for row_i, run_name in enumerate(run_names):
        cells = [run_name]
        for col_i, v in enumerate(data[row_i]):
            s = fmt(v)
            if best_idx[col_i] == row_i and v is not None:
                s = f'\\textbf{{{s}}}'
            cells.append(s)
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{Multi-task LoRA vs single-task baselines. '
                 r'Bold = best per column.}')
    lines.append(r'\label{tab:multitask_results}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def main():
    args = get_args()
    rows = [load_row(p) for p in args.results]

    run_names, col_headers, data, best_idx = build_table(rows)

    md  = to_markdown(run_names, col_headers, data, best_idx)
    tex = to_latex(run_names, col_headers, data, best_idx)

    out_md  = Path(args.output)
    out_tex = out_md.with_suffix('.tex')

    out_md.write_text(md)
    out_tex.write_text(tex)

    print(md)
    print(f'\nSaved → {out_md}  and  {out_tex}')


if __name__ == '__main__':
    main()
