#!/usr/bin/env python3
"""
Build a combined alpha/Fe gradient summary across galaxies using existing outputs.

For each galaxy under output/<name>_stack:
- Load AIP alpha/Fe 2D (from <gal>_AIP_alpha_fe_results.npz if available; else compute via run_aip_alpha_fe)
- Load RDB and VNB NPZs from Data/
- Compute RDB radial profile and optional VNB profile
- Fit gradients using alpha_gradient_analysis.fit_alpha_fe_gradient_multi_method
- Write rows for both RDB (3 bins) and VNB (all bins) to alpha_gradient_dual/combined_gradient_summary.csv

Outputs:
- alpha_gradient_dual/combined_gradient_summary.csv
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import numpy as np

import sys

# Ensure project root is on sys.path when running from a subfolder
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_gradient_analysis import (
    calculate_radial_alpha_fe_profile,
    calculate_vnb_alpha_fe_profile,
    fit_alpha_fe_gradient_multi_method,
)
from run_aip_alpha_fe import run_aip_for_galaxy


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DEST_DIR = ROOT / "alpha_gradient_dual"
DEST_DIR.mkdir(exist_ok=True)
DEST_CSV = DEST_DIR / "combined_gradient_summary.csv"


def safe_load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return dict(np.load(path, allow_pickle=True))
    except Exception:
        return None


def significance_label(slope: float, slope_err: float, p_value: float) -> str:
    if slope_err is None or slope_err == 0 or p_value is None or np.isnan(p_value):
        return "not_significant"
    ratio = abs(slope) / slope_err
    if p_value < 0.01 and ratio > 3:
        return "highly_significant"
    if p_value < 0.05 and ratio > 2:
        return "significant"
    if p_value < 0.1:
        return "marginal"
    return "not_significant"


def gather_galaxies() -> list[str]:
    gals = []
    if not OUTPUT.exists():
        return gals
    for d in sorted(OUTPUT.iterdir()):
        if d.is_dir() and d.name.endswith("_stack") and d.name.startswith("VCC"):
            gals.append(d.name.replace("_stack", ""))
    return gals


def ensure_aip_results(gal: str) -> dict | None:
    out_dir = OUTPUT / f"{gal}_stack"
    npz = out_dir / f"{gal}_AIP_alpha_fe_results.npz"
    data = safe_load_npz(npz)
    if data is None:
        # Compute on the fly; this will also save plots/npz
        run_aip_for_galaxy(gal, ROOT)
        data = safe_load_npz(npz)
    return data


def main() -> None:
    galaxies = gather_galaxies()
    if not galaxies:
        print("No galaxies found under output/*_stack")
        return

    rows = []
    for gal in galaxies:
        out_dir = OUTPUT / f"{gal}_stack"
        data_dir = out_dir / "Data"

        # Load AIP alpha/Fe 2D
        aip = ensure_aip_results(gal)
        if aip is None or 'alpha_fe_2d' not in aip:
            print(f"{gal}: missing AIP alpha/Fe; skipping")
            continue

        alpha_fe_2d = aip['alpha_fe_2d']
        alpha_fe_err = aip.get('alpha_fe_errors', np.full_like(alpha_fe_2d, np.nan))
        alpha_fe_data = {
            'galaxy_name': gal,
            'alpha_fe_2d': alpha_fe_2d,
            'alpha_fe_errors': alpha_fe_err,
            'mean_alpha_fe': float(np.nanmean(alpha_fe_2d)),
            'std_alpha_fe': float(np.nanstd(alpha_fe_2d)),
        }

        # Load RDB and VNB
        rdb = safe_load_npz(data_dir / f"{gal}_stack_RDB_results.npz")
        vnb = safe_load_npz(data_dir / f"{gal}_stack_VNB_results.npz")

        if rdb is None:
            print(f"{gal}: no RDB results; skipping")
            continue

        # Compute profiles
        radial_profile = calculate_radial_alpha_fe_profile(alpha_fe_data, rdb)
        vnb_profile = calculate_vnb_alpha_fe_profile(alpha_fe_data, vnb) if vnb is not None else None

        if radial_profile is None:
            print(f"{gal}: radial profile failed; skipping")
            continue

        # Fit gradients (include VNB if present)
        multi = fit_alpha_fe_gradient_multi_method(radial_profile, vnb_profile)
        if not multi:
            print(f"{gal}: gradient fit failed; skipping")
            continue

        # RDB 3-bin
        r = multi.get('rdb_3bins')
        if r is not None:
            rows.append({
                'Galaxy': gal,
                'Mode': 'RDB',
                'Slope': float(r['slope']),
                'Slope_Error': float(r['slope_error']) if r['slope_error'] is not None else np.nan,
                'P_value': float(r['p_value']) if r['p_value'] is not None else np.nan,
                'R_squared': float(r['r_squared']) if r['r_squared'] is not None else np.nan,
                'N_points': int(r['n_points']),
                'Significance': significance_label(r['slope'], r['slope_error'], r['p_value']),
                'Method': r['method'],
            })

        # VNB all bins
        v = multi.get('vnb')
        if v is not None:
            rows.append({
                'Galaxy': gal,
                'Mode': 'VNB',
                'Slope': float(v['slope']),
                'Slope_Error': float(v['slope_error']) if v['slope_error'] is not None else np.nan,
                'P_value': float(v['p_value']) if v['p_value'] is not None else np.nan,
                'R_squared': float(v['r_squared']) if v['r_squared'] is not None else np.nan,
                'N_points': int(v['n_points']),
                'Significance': significance_label(v['slope'], v['slope_error'], v['p_value']),
                'Method': v['method'],
            })

    # Write CSV
    if rows:
        cols = ['Galaxy', 'Mode', 'Slope', 'Slope_Error', 'P_value', 'R_squared', 'N_points', 'Significance', 'Method']
        with DEST_CSV.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"Wrote {DEST_CSV} with {len(rows)} rows")
    else:
        print("No gradient rows to write.")


if __name__ == "__main__":
    main()
