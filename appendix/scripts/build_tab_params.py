#!/usr/bin/env python3
"""
Generate a LaTeX longtable of parameters from parameters.xlsx (constant sheet).

Columns:
- parameter
- definition  (from 'full_text')
- value_or_prior  (distribution + bounds if present; else value)
- unit

Usage:
    python3 scripts/build_tab_params.py \
        --xlsx parameters.xlsx \
        --out tab-params.tex \
        --caption "Model parameters" \
        --label tab-params
"""

import argparse
import pandas as pd
from pathlib import Path


latex_escape_map = {
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'
}

def latex_escape(s):
    if pd.isna(s) or s == '':
        return ''
    s = str(s)
    for k,v in latex_escape_map.items():
        s = s.replace(k, v)
    return s


def build_value_or_prior(row) -> str:
    dist = row.get('distribution', '')
    p1   = row.get('distri_param1', '')
    p2   = row.get('distri_param2', '')
    val  = row.get('value', '')
    if pd.notna(dist) and str(dist).strip() != '':
        # Render as: distribution (p1, p2) -- if p2 missing, still works
        inside = ', '.join([str(x) for x in (p1, p2) if pd.notna(x) and str(x)!=''])
        return f"{dist} ({inside})" if inside else f"{dist}"
    return str(val)


def df_to_longtable(df, caption, label):
    # Expected columns
    cols = ['parameter','definition','value_or_prior','unit']
    assert list(df.columns) == cols

    # 2) Column alignment with monospaced for code-like columns
    #    - parameter (tt), definition (wrap), value_or_prior (tt), unit (l)
    #    Requires \usepackage{array}, \usepackage{booktabs,longtable} in preamble
    align = r'>{\ttfamily}p{0.28\textwidth} p{0.42\textwidth} >{\ttfamily}p{0.22\textwidth} l'

    header = (
        r'\toprule' + '\n' +
        r'\textbf{Parameter} & \textbf{Definition} & \textbf{Value / Prior} & \textbf{Unit} \\' + '\n' +
        r'\midrule'
    )

    rows = []
    for _, r in df.iterrows():
        row_cells = [r[c] for c in cols]
        rows.append(' {} \\\\'.format(' & '.join(row_cells)))
    body = '\n'.join(rows)

    footer = r'\bottomrule'

    table = (
        r'\begin{longtable}{' + align + '}' + '\n' +
        r'\caption{' + caption + r'}\label{' + label + r'}\\' + '\n' +
        header + '\n' +
        r'\endfirsthead' + '\n' +
        r'\caption[]{' + caption + r' (continued)}\\' + '\n' +
        header + '\n' +
        r'\endhead' + '\n' +
        r'\hline \multicolumn{4}{r}{\textit{Continues on next page}} \\' + '\n' +
        r'\endfoot' + '\n' +
        footer + '\n' +
        r'\endlastfoot' + '\n' +
        body + '\n' +
        r'\end{longtable}'
    )
    return table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx',   default='parameters.xlsx', help='Path to parameters.xlsx')
    ap.add_argument('--sheet',  default='constant',        help='Sheet name for constant params')
    ap.add_argument('--out',    default='tab-params.tex',  help='Output .tex file')
    ap.add_argument('--caption',default='Model parameters',help='LaTeX table caption')
    ap.add_argument('--label',  default='tab-params',      help='LaTeX table label')
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    df = pd.read_excel(xlsx_path, sheet_name=args.sheet).fillna('')

    # Build value_or_prior
    df['value_or_prior'] = df.apply(build_value_or_prior, axis=1)

    # Map full_text -> definition
    if 'full_text' in df.columns:
        df = df.rename(columns={'full_text': 'definition'})
    elif 'definition' not in df.columns:
        df['definition'] = ''

    # Select & order columns
    keep = ['parameter', 'definition', 'value_or_prior', 'unit']
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {missing}")

    df = df[keep].copy()

    # Escape LaTeX in all cells
    for col in keep:
        df[col] = df[col].map(latex_escape)

    # Build longtable
    tex = (
        '% NOTE: Requires \\usepackage{booktabs,longtable} in the preamble\n\n' +
        df_to_longtable(df, args.caption, args.label) + '\n'
    )

    Path(args.out).write_text(tex, encoding='utf-8')
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()
