#!/usr/bin/env python3
"""
Convert output/alpha_fe_gradient_summary.csv into alpha_gradient_dual/combined_gradient_summary.csv
with columns: Galaxy,Mode,Slope,Slope_Error

Mapping:
- Galaxy: derived from left part of galaxy name (strip suffixes like _stack/_obstack)
- Mode: from 'method' (normalize keys like 'rdb_3bins' -> 'RDB', 'vnb' -> 'VNB')
- Slope: slope
- Slope_Error: slope_err (may be NaN if not available)
"""
from __future__ import annotations

import csv
from pathlib import Path


def normalize_galaxy(name: str) -> str:
    for suf in ('_stack', '_obstack', '_stacked'):
        if name.endswith(suf):
            return name.replace(suf, '')
    return name


def normalize_mode(method: str) -> str:
    m = method.strip().lower()
    if 'rdb' in m:
        return 'RDB'
    if 'vnb' in m:
        return 'VNB'
    return method.upper()


def main() -> int:
    src = Path('output/alpha_fe_gradient_summary.csv')
    dst_dir = Path('alpha_gradient_dual')
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / 'combined_gradient_summary.csv'
    if not src.exists():
        print(f'Missing source CSV: {src}')
        return 1
    rows_out = []
    with src.open('r', newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            gal_full = r.get('galaxy', '')
            method = r.get('method', '')
            slope = r.get('slope', '')
            slope_err = r.get('slope_err', '')
            if not gal_full or not method or slope == '':
                continue
            rows_out.append({
                'Galaxy': normalize_galaxy(gal_full),
                'Mode': normalize_mode(method),
                'Slope': slope,
                'Slope_Error': slope_err,
            })
    if not rows_out:
        print('No rows to write.')
        return 0
    with dst.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['Galaxy','Mode','Slope','Slope_Error'])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f'Wrote {len(rows_out)} rows to {dst}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
