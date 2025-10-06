#!/usr/bin/env python3
"""
Generate compact "visu" panels for VCC1588:
- Index grid (Mgb vs Fe5015) over TMB03 model points with bin annotations
- Compact RDB vs VNB gradient panel (3-bin RDB vs range-matched VNB)
- Copy/thumbnail existing normalized spectrum and AIP radial profile

Outputs are written to FINAL_DELIVERABLES/visu/ as small PNGs suitable for a short report.
"""
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse existing loader and gradient plot helper
from run_phy_visu_all_galaxies import load_tmb03_model
from enhanced_radial_plots_3bin_corrected import (
    calculate_gradient_3bin_rdb_vnb,
    create_enhanced_plot,
)

GAL = "VCC1588"

ROOT = Path('.')
OUT_BASE = ROOT / 'FINAL_DELIVERABLES' / 'visu'
OUT_BASE.mkdir(parents=True, exist_ok=True)

def load_bin_indices(galaxy: str):
    """Load Fe5015, Mgb, Hbeta arrays from RDB results if available."""
    data_dir = ROOT / 'output' / f'{galaxy}_stack' / 'Data'
    rdb_results = data_dir / f'{galaxy}_stack_RDB_results.npz'
    if not rdb_results.exists():
        return None
    d = np.load(rdb_results, allow_pickle=True)
    # Common schemas observed in repo
    if 'bin_indices_multi' in d:
        indices_data = d['bin_indices_multi'].item()
        key = 'auto' if 'auto' in indices_data else list(indices_data.keys())[0]
        bin_indices = indices_data[key]['bin_indices']
    elif 'bin_indices' in d:
        bi = d['bin_indices'].item()
        bin_indices = bi['bin_indices'] if 'bin_indices' in bi else bi
    else:
        return None
    fe = np.array(bin_indices.get('Fe5015', []), dtype=float)
    mg = np.array(bin_indices.get('Mgb', []), dtype=float)
    hb = np.array(bin_indices.get('Hbeta', []), dtype=float)
    # Keep finite triplets
    mask = np.isfinite(fe) & np.isfinite(mg) & np.isfinite(hb)
    return fe[mask], mg[mask], hb[mask]

def make_index_grid_panel(galaxy: str):
    tmb = load_tmb03_model()
    if tmb is None:
        print("No TMB03 model; skipping index grid panel.")
        return None
    indices = load_bin_indices(galaxy)
    if indices is None:
        print(f"No bin indices for {galaxy}; skipping index grid panel.")
        return None
    fe, mg, hb = indices
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))
    # Model background: color by AoFe if present, else by Age
    color_col = 'AoFe' if 'AoFe' in tmb.columns else ('Age' if 'Age' in tmb.columns else None)
    if color_col is not None:
        sc = ax.scatter(tmb['Fe5015'], tmb['Mgb'], c=tmb[color_col], s=10, alpha=0.4,
                        cmap='viridis')
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_col)
    else:
        ax.scatter(tmb['Fe5015'], tmb['Mgb'], s=6, alpha=0.3, c='gray')
    # Galaxy bins: trajectory with labels
    ax.plot(fe, mg, 'o-', color='crimson', linewidth=1.5, markersize=5,
            markerfacecolor='white', markeredgewidth=1.0)
    for i, (x, y) in enumerate(zip(fe, mg)):
        ax.text(x, y, str(i+1), ha='center', va='center', fontsize=7, color='black')
    ax.set_xlabel('Fe5015 [Å]')
    ax.set_ylabel('Mgb [Å]')
    ax.set_title(f'{galaxy}: Indices vs TMB03')
    ax.grid(True, alpha=0.3)
    out = OUT_BASE / f'{galaxy}_visu_index_grid.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out

def make_compact_rdb_vnb_panel(galaxy: str):
    res = calculate_gradient_3bin_rdb_vnb(galaxy)
    if res is None:
        print(f"No gradient data for {galaxy}; skipping RDB/VNB panel.")
        return None
    out = OUT_BASE / f'{galaxy}_visu_rdb_vnb.png'
    # Temporarily set smaller default figure inside helper by scaling DPI via savefig call
    create_enhanced_plot(res, str(out))
    return out

def copy_existing_small_panels(galaxy: str):
    copied = []
    # Normalized spectrum example (first panel)
    p2p_dir = ROOT / 'FINAL_DELIVERABLES' / f'{galaxy}_stack' / 'P2P'
    norm = None
    if p2p_dir.exists():
        cands = sorted(p2p_dir.glob(f'{galaxy}_stack_P2P_spectrum_norm_*.png'))
        norm = cands[0] if cands else None
    if norm and norm.exists():
        dst = OUT_BASE / f'{galaxy}_visu_norm.png'
        shutil.copyfile(norm, dst)
        copied.append(dst)
    # AIP radial profile (if present)
    gal_dir = ROOT / 'FINAL_DELIVERABLES' / f'{galaxy}_stack'
    prof = gal_dir / f'{galaxy}_AIP_alpha_fe_radial_profile.png'
    if prof.exists():
        dst = OUT_BASE / f'{galaxy}_visu_radial_profile.png'
        shutil.copyfile(prof, dst)
        copied.append(dst)
    return copied

def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    made = []
    p1 = make_index_grid_panel(GAL)
    if p1:
        print(f"✓ Index grid: {p1}")
        made.append(p1)
    p2 = make_compact_rdb_vnb_panel(GAL)
    if p2:
        print(f"✓ RDB/VNB panel: {p2}")
        made.append(p2)
    copies = copy_existing_small_panels(GAL)
    for c in copies:
        print(f"✓ Copied: {c}")
        made.append(c)
    if not made:
        print("No panels produced.")

if __name__ == '__main__':
    main()
