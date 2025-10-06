#!/usr/bin/env python3
"""
Collect alpha/Fe gradient results from AIP outputs into a single CSV.

Scans output/*/*_AIP_alpha_fe_results.npz and writes output/alpha_fe_gradient_summary.csv
with rows: galaxy, method, slope, slope_err, n_bins.
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path
import numpy as np


def load_gradient_row(npz_path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        data = np.load(npz_path, allow_pickle=True)
        galaxy = npz_path.stem.replace('_AIP_alpha_fe_results', '')
        grad = data.get('gradient_results', None)
        if grad is None:
            return rows
        # gradient_results may be a dict-like object
        if isinstance(grad, np.ndarray) and grad.shape == ():
            grad = grad.item()
        if not isinstance(grad, dict):
            return rows
        for method, vals in grad.items():
            # vals may be dict-like
            if isinstance(vals, np.ndarray) and vals.shape == ():
                vals = vals.item()
            if not isinstance(vals, dict):
                continue
            slope = vals.get('slope', np.nan)
            slope_err = vals.get('slope_err', vals.get('error', np.nan))
            n_bins = vals.get('N', vals.get('n_bins', np.nan))
            rows.append({
                'galaxy': galaxy,
                'method': method,
                'slope': slope,
                'slope_err': slope_err,
                'n_bins': n_bins,
            })
    except Exception:
        pass
    return rows


def main() -> int:
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    pattern = out_dir / '*' / '*_AIP_alpha_fe_results.npz'
    files = sorted(glob.glob(str(pattern)))
    rows: list[dict] = []
    for f in files:
        rows.extend(load_gradient_row(Path(f)))

    if not rows:
        print('No AIP results found to summarize.')
        return 0

    out_csv = out_dir / 'alpha_fe_gradient_summary.csv'
    with out_csv.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['galaxy', 'method', 'slope', 'slope_err', 'n_bins'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'Wrote {len(rows)} rows to {out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
