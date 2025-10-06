#!/usr/bin/env python3
"""
Generate a full science-focused report summarizing:
- AIP alpha/Fe radial fit results (slopes, intercepts, errors)
- Per-bin spectral index values (Hbeta, Fe5015, Mgb) from RDB

Outputs a Markdown file under FINAL_DELIVERABLES.

Usage:
  python generate_full_report.py --galaxies VCC1588 VCC1146
  python generate_full_report.py --galaxies VCC1588 --workspace .
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def safe_load_npz(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return dict(np.load(path, allow_pickle=True))
    except Exception:
        return None


def extract_aip_fit(galaxy: str, workspace: Path) -> Optional[Dict[str, Any]]:
    out_dir = workspace / 'output' / f'{galaxy}_stack'
    npz = safe_load_npz(out_dir / f'{galaxy}_AIP_alpha_fe_results.npz')
    if not npz:
        return None
    grad = npz.get('gradient_results', None)
    radial = npz.get('radial_profile', None)
    # Gradient results is likely an object array containing a dict
    if grad is not None and isinstance(grad, np.ndarray):
        try:
            grad = grad.item()
        except Exception:
            pass
    if radial is not None and isinstance(radial, np.ndarray):
        try:
            radial = radial.item()
        except Exception:
            pass
    if not isinstance(grad, dict):
        return None
    return {
        'gradient_results': grad,
        'radial_profile': radial,
        'map_path': str(out_dir / 'Plots' / f'{galaxy}_AIP_alpha_fe_map.png'),
        'profile_path': str(out_dir / 'Plots' / f'{galaxy}_AIP_alpha_fe_radial_profile.png'),
    }


def extract_rdb_indices(galaxy: str, workspace: Path) -> Optional[pd.DataFrame]:
    data_dir = workspace / 'output' / f'{galaxy}_stack' / 'Data'
    rdb_npz = safe_load_npz(data_dir / f'{galaxy}_stack_RDB_results.npz')
    if not rdb_npz:
        return None
    indices = None
    if 'bin_indices_multi' in rdb_npz:
        try:
            multi = rdb_npz['bin_indices_multi'].item()
            key = 'auto' if 'auto' in multi else list(multi.keys())[0]
            indices = multi[key]['bin_indices']
        except Exception:
            indices = None
    if indices is None and 'bin_indices' in rdb_npz:
        try:
            bi = rdb_npz['bin_indices'].item()
            indices = bi['bin_indices'] if 'bin_indices' in bi else bi
        except Exception:
            indices = None
    if indices is None:
        return None
    fe = np.array(indices.get('Fe5015', []), dtype=float)
    mg = np.array(indices.get('Mgb', []), dtype=float)
    hb = np.array(indices.get('Hbeta', []), dtype=float)
    n = max(len(fe), len(mg), len(hb))
    # Normalize lengths
    def pad(a, n):
        if len(a) < n:
            b = np.full(n, np.nan)
            b[:len(a)] = a
            return b
        return a
    fe, mg, hb = pad(fe, n), pad(mg, n), pad(hb, n)
    df = pd.DataFrame({
        'bin': np.arange(1, n+1, dtype=int),
        'Fe5015_A': fe,
        'Mgb_A': mg,
        'Hbeta_A': hb,
    })
    return df


def gradient_to_rows(grad: Dict[str, Any]) -> List[str]:
    rows = []
    for label, key in [('RDB (3 bins)', 'rdb_3bins'),
                       ('1.5 Re', 'radius_1p5_re'),
                       ('2.0 Re', 'radius_2p0_re'),
                       ('VNB', 'vnb')]:
        g = grad.get(key)
        if isinstance(g, np.ndarray):
            try:
                g = g.item()
            except Exception:
                pass
        if not g:
            rows.append(f"- {label}: N/A")
            continue
        slope = g.get('slope', np.nan)
        serr = g.get('slope_error', np.nan)
        inter = g.get('intercept', np.nan)
        r2 = g.get('r_squared', np.nan)
        n = g.get('n_points', g.get('n_bins', ''))
        rows.append(
            f"- {label}: slope = {slope:.4f} ± {serr:.4f} dex/Re; intercept = {inter:.4f}; R^2 = {r2:.3f}; N = {n}"
        )
    return rows


def build_report(galaxies: List[str], workspace: Path, out_file: Path) -> None:
    lines: List[str] = []
    lines.append("# Full Science Report")
    lines.append("")
    lines.append("This report summarizes AIP alpha/Fe radial fits and per-bin spectral indices for the selected galaxies.")
    lines.append("")
    for gal in galaxies:
        lines.append(f"## {gal}")
        # AIP
        aip = extract_aip_fit(gal, workspace)
        if aip:
            lines.append("### AIP alpha/Fe fit results")
            gr = aip['gradient_results']
            lines.extend(gradient_to_rows(gr))
            # figure paths (not embedded, just referenced)
            if Path(aip['profile_path']).exists():
                rel = Path(aip['profile_path']).as_posix()
                lines.append(f"- Radial profile figure: `{rel}`")
            if Path(aip['map_path']).exists():
                rel = Path(aip['map_path']).as_posix()
                lines.append(f"- AIP map figure: `{rel}`")
        else:
            lines.append("### AIP alpha/Fe fit results")
            lines.append("- Not available (run `run_aip_alpha_fe.py --galaxy {}` first)".format(gal))
        lines.append("")
        # RDB indices
        df = extract_rdb_indices(gal, workspace)
        lines.append("### RDB per-bin spectral indices")
        if df is None or df.empty:
            lines.append("- Not available in RDB results")
        else:
            # markdown table (limited rows if many)
            df_round = df.copy()
            for c in ['Fe5015_A', 'Mgb_A', 'Hbeta_A']:
                if c in df_round:
                    df_round[c] = df_round[c].astype(float).round(3)
            # to markdown
            lines.append(df_round.to_markdown(index=False))
        lines.append("")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxies', nargs='+', required=True, help='One or more galaxy IDs (e.g., VCC1588 VCC1146)')
    ap.add_argument('--workspace', default='.')
    ap.add_argument('--output', default='FINAL_DELIVERABLES/FULL_SCIENCE_REPORT.md')
    args = ap.parse_args()
    build_report(args.galaxies, Path(args.workspace), Path(args.output))
    print(f"✓ Wrote report: {args.output}")


if __name__ == '__main__':
    main()
