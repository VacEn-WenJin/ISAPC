#!/usr/bin/env python3
"""
Plot a pedagogical figure showing how spectral indices map to [alpha/Fe]
using the TMB03 model grid and a selected galaxy radial bin.

Usage:
  python plot_alpha_fe_from_indices.py --galaxy VCC1588 --bin 1

Outputs:
  FINAL_DELIVERABLES/alpha_fe_from_indices_<GAL>_bin<idx>.png

Notes:
  - Loads Fe5015, Mgb, Hbeta per-bin values from `<gal>_stack_RDB_results.npz`
  - Uses our enhanced 3D interpolation to compute [alpha/Fe], age, [Z/H]
  - Overlays the observed point on Fe5015–Mgb plane colored by model AoFe
  - Optionally draws 3 nearest model points used for interpolation
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_phy_visu_all_galaxies import load_tmb03_model
from Phy_Visu import calculate_enhanced_alpha_fe

ROOT = Path('.')


def _safe_load_npz(path: Path):
    try:
        if not path.exists():
            return None
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def load_bin_indices_with_errors(galaxy: str) -> Optional[dict]:
    """Return dict with arrays for Fe5015, Mgb, Hbeta and optional errors.

    Attempts multiple known schemas inside RDB results.
    """
    data_dir = ROOT / 'output' / f'{galaxy}_stack' / 'Data'
    rdb_path = data_dir / f'{galaxy}_stack_RDB_results.npz'
    d = _safe_load_npz(rdb_path)
    if d is None:
        return None

    fe = mg = hb = None
    fe_e = mg_e = hb_e = None

    try:
        if 'bin_indices_multi' in d:
            indices_data = d['bin_indices_multi'].item() if hasattr(d['bin_indices_multi'], 'item') else d['bin_indices_multi']
            method_key = 'auto' if isinstance(indices_data, dict) and 'auto' in indices_data else (list(indices_data.keys())[0] if isinstance(indices_data, dict) else None)
            if method_key is not None:
                bi = indices_data[method_key].get('bin_indices', indices_data[method_key])
                fe = np.asarray(bi.get('Fe5015', []), dtype=float)
                mg = np.asarray(bi.get('Mgb', []), dtype=float)
                hb = np.asarray(bi.get('Hbeta', []), dtype=float)
                # Optional errors if present in same structure
                fe_e = np.asarray(bi.get('Fe5015_err', []), dtype=float) if 'Fe5015_err' in bi else None
                mg_e = np.asarray(bi.get('Mgb_err', []), dtype=float) if 'Mgb_err' in bi else None
                hb_e = np.asarray(bi.get('Hbeta_err', []), dtype=float) if 'Hbeta_err' in bi else None
        elif 'bin_indices' in d:
            bi_struct = d['bin_indices'].item() if hasattr(d['bin_indices'], 'item') else d['bin_indices']
            bi = bi_struct.get('bin_indices', bi_struct)
            fe = np.asarray(bi.get('Fe5015', []), dtype=float)
            mg = np.asarray(bi.get('Mgb', []), dtype=float)
            hb = np.asarray(bi.get('Hbeta', []), dtype=float)
            fe_e = np.asarray(bi.get('Fe5015_err', []), dtype=float) if 'Fe5015_err' in bi else None
            mg_e = np.asarray(bi.get('Mgb_err', []), dtype=float) if 'Mgb_err' in bi else None
            hb_e = np.asarray(bi.get('Hbeta_err', []), dtype=float) if 'Hbeta_err' in bi else None
        else:
            # Some files store errors under a separate spectral_data block
            if 'spectral_data' in d:
                spec = d['spectral_data'].item() if hasattr(d['spectral_data'], 'item') else d['spectral_data']
                if all(k in spec for k in ['Fe5015', 'Mgb', 'Hbeta']):
                    fe = np.asarray(spec['Fe5015'].get('values', []), dtype=float)
                    mg = np.asarray(spec['Mgb'].get('values', []), dtype=float)
                    hb = np.asarray(spec['Hbeta'].get('values', []), dtype=float)
                    fe_e = np.asarray(spec['Fe5015'].get('errors', []), dtype=float) if 'errors' in spec['Fe5015'] else None
                    mg_e = np.asarray(spec['Mgb'].get('errors', []), dtype=float) if 'errors' in spec['Mgb'] else None
                    hb_e = np.asarray(spec['Hbeta'].get('errors', []), dtype=float) if 'errors' in spec['Hbeta'] else None
    except Exception:
        pass

    if fe is None or mg is None or hb is None:
        return None

    # Keep finite triplets; mask errors accordingly
    mask = np.isfinite(fe) & np.isfinite(mg) & np.isfinite(hb)
    out = {
        'Fe5015': fe[mask],
        'Mgb': mg[mask],
        'Hbeta': hb[mask],
    }
    if fe_e is not None and len(fe_e) == len(fe):
        out['Fe5015_err'] = np.asarray(fe_e)[mask]
    if mg_e is not None and len(mg_e) == len(mg):
        out['Mgb_err'] = np.asarray(mg_e)[mask]
    if hb_e is not None and len(hb_e) == len(hb):
        out['Hbeta_err'] = np.asarray(hb_e)[mask]
    return out


def derive_alpha_fe(fe5015: float, mgb: float, hbeta: float, model_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Compute [alpha/Fe], age, [Z/H], uncertainty using our enhanced method."""
    alpha, age, zh, unc, chi2 = calculate_enhanced_alpha_fe(fe5015, mgb, hbeta, model_df)
    return float(alpha), float(age), float(zh), float(unc)


def plot_alpha_fe_mapping(galaxy: str, bin_index: int = 0, output: Optional[Path] = None) -> Optional[Path]:
    model = load_tmb03_model()
    if model is None:
        print("❌ TMB03 model not found. Cannot plot index→alpha/Fe.")
        return None

    data = load_bin_indices_with_errors(galaxy)
    if data is None or len(data.get('Fe5015', [])) == 0:
        print(f"❌ No RDB bin indices found for {galaxy}.")
        return None

    n = len(data['Fe5015'])
    if bin_index < 0 or bin_index >= n:
        print(f"⚠️ Bin index {bin_index} out of range [0..{n-1}]; using 0.")
        bin_index = 0

    fe, mg, hb = data['Fe5015'][bin_index], data['Mgb'][bin_index], data['Hbeta'][bin_index]
    fe_e = data.get('Fe5015_err', np.array([np.nan]*n))[bin_index] if 'Fe5015_err' in data else np.nan
    mg_e = data.get('Mgb_err', np.array([np.nan]*n))[bin_index] if 'Mgb_err' in data else np.nan
    hb_e = data.get('Hbeta_err', np.array([np.nan]*n))[bin_index] if 'Hbeta_err' in data else np.nan

    # Compute derived parameters
    ao_fe, age, zh, unc = derive_alpha_fe(fe, mg, hb, model)

    # Find 3 nearest model points in 3D index space (for visual guide)
    try:
        # Column mapping heuristics
        fe_col = 'Fe5015' if 'Fe5015' in model.columns else 'Fe5015_SI'
        mg_col = 'Mgb' if 'Mgb' in model.columns else ('Mg_b' if 'Mg_b' in model.columns else 'Mgb_SI')
        hb_col = 'Hbeta' if 'Hbeta' in model.columns else ('Hb' if 'Hb' in model.columns else 'Hb_SI')
        ao_col = 'AoFe' if 'AoFe' in model.columns else ('alpha/Fe' if 'alpha/Fe' in model.columns else '[alpha/Fe]')
        feats = model[[fe_col, mg_col, hb_col]].values
        obs = np.array([fe, mg, hb])
        d3 = np.linalg.norm(feats - obs, axis=1)
        idx = np.argsort(d3)[:3]
        nearest = model.iloc[idx][[fe_col, mg_col, hb_col, ao_col]]
    except Exception:
        nearest = None

    # Build figure
    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.4], width_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0:2])
    ax_hb = fig.add_subplot(gs[0, 2])
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis('off')

    # Background model scatter colored by alpha/Fe
    sc = ax.scatter(model[fe_col], model[mg_col], c=model[ao_col], s=10, alpha=0.35, cmap='viridis')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('[alpha/Fe] (model)')

    # Observed bin point with optional errorbars
    has_err = np.isfinite(fe_e) and np.isfinite(mg_e)
    if has_err:
        ax.errorbar(fe, mg, xerr=fe_e, yerr=mg_e, fmt='o', color='crimson', ecolor='crimson', elinewidth=1.0, capsize=2, label=f'{galaxy} bin {bin_index+1}')
    else:
        ax.plot(fe, mg, 'o', color='crimson', label=f'{galaxy} bin {bin_index+1}')

    # Nearest model points (projected to Fe5015–Mgb plane)
    if nearest is not None:
        ax.scatter(nearest[fe_col], nearest[mg_col], marker='^', color='k', s=40, label='nearest models')
        # Dashed guidance from obs to nearest
        for _, row in nearest.iterrows():
            ax.plot([fe, row[fe_col]], [mg, row[mg_col]], ls='--', lw=0.7, color='k', alpha=0.5)

    ax.set_xlabel('Fe5015 [Å]')
    ax.set_ylabel('Mgb [Å]')
    ax.set_title(f'{galaxy}: indices → [alpha/Fe] (bin {bin_index+1})')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    # Hbeta context: show Hbeta distribution vs Fe5015 colored by AoFe, mark observed Hbeta
    sc2 = ax_hb.scatter(model[fe_col], model[hb_col], c=model[ao_col], s=8, alpha=0.3, cmap='viridis')
    ax_hb.axhline(hb, color='crimson', lw=1.0, ls='--', label=f'Hbeta obs = {hb:.2f} Å')
    if np.isfinite(hb_e):
        ax_hb.fill_between([model[fe_col].min(), model[fe_col].max()], hb - hb_e, hb + hb_e, color='crimson', alpha=0.1)
    ax_hb.set_xlabel('Fe5015 [Å]')
    ax_hb.set_ylabel('Hbeta [Å]')
    ax_hb.set_title('Hbeta context')
    ax_hb.grid(True, alpha=0.25)

    # Text panel with derived parameters
    lines = [
        f"Derived [alpha/Fe] = {ao_fe:.3f} ± {unc:.3f}",
        f"Derived age ≈ {age:.2f} Gyr",
        f"Derived [Z/H] ≈ {zh:.2f}",
    ]
    if np.isfinite(fe_e) or np.isfinite(mg_e) or np.isfinite(hb_e):
        lines.append(f"Observed indices (±1σ): Fe5015={fe:.2f}±{fe_e if np.isfinite(fe_e) else '…'}, Mgb={mg:.2f}±{mg_e if np.isfinite(mg_e) else '…'}, Hβ={hb:.2f}±{hb_e if np.isfinite(hb_e) else '…'}")
    else:
        lines.append(f"Observed indices: Fe5015={fe:.2f}, Mgb={mg:.2f}, Hβ={hb:.2f}")
    ax_txt.text(0.01, 0.6, "\n".join(lines), fontsize=11, va='top')
    ax_txt.text(0.01, 0.15, "Method: 3D interpolation on TMB03 (Fe5015, Mgb, Hβ) with enhanced physics corrections.", fontsize=9, color='dimgray')

    # Save
    out_dir = ROOT / 'FINAL_DELIVERABLES'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output if output is not None else (out_dir / f'alpha_fe_from_indices_{galaxy}_bin{bin_index+1}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Plot index→[alpha/Fe] mapping for a galaxy bin')
    ap.add_argument('--galaxy', required=True, help='Galaxy name e.g., VCC1588')
    ap.add_argument('--bin', type=int, default=1, help='Bin number (1-based). Default: 1')
    ap.add_argument('--out', type=str, default=None, help='Optional output PNG path')
    args = ap.parse_args()

    bin_index = max(0, args.bin - 1)
    out = Path(args.out) if args.out else None
    plot_alpha_fe_mapping(args.galaxy, bin_index=bin_index, output=out)


if __name__ == '__main__':
    main()
