#!/usr/bin/env python3
"""
Run AIP alpha/Fe analysis and gradient derivation from existing ISAPC outputs.

This script:
- Loads P2P/VNB/RDB data for a given galaxy from ./output/<galaxy>_stack/Data
- Computes 2D alpha/Fe using Phy_Visu helpers (3D interpolation over Fe5015/Mgb/Hbeta)
- Builds RDB-based radial profile and fits gradients (linear, weighted)
- Saves plots and an NPZ summary into ./output/<galaxy>_stack/Plots

Usage:
  python run_aip_alpha_fe.py --galaxy VCC1588
"""
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

from run_phy_visu_all_galaxies import (
    load_tmb03_model,
    load_galaxy_data,
    calculate_alpha_fe_2d,
)
from alpha_gradient_analysis import (
    calculate_radial_alpha_fe_profile,
    fit_alpha_fe_gradient_multi_method,
)


def run_aip_for_galaxy(galaxy: str, workspace: Path = Path('.')) -> Optional[dict]:
    out_dir = workspace / 'output' / f'{galaxy}_stack'
    data_dir = out_dir / 'Data'
    plots_dir = out_dir / 'Plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    model = load_tmb03_model()
    if model is None:
        print('TMB03 model not found; cannot compute alpha/Fe.')
        return None

    gdata = load_galaxy_data(galaxy)
    if gdata is None:
        print(f'No data for {galaxy}.')
        return None

    # Compute alpha/Fe 2D
    alpha_fe_2d, alpha_fe_err, n_ok = calculate_alpha_fe_2d(gdata, model)
    if alpha_fe_2d is None or n_ok == 0:
        # Graceful placeholder so deliverables remain consistent
        print('Alpha/Fe calculation failed or no valid pixels. Writing placeholder plot.')
        fig, ax = plt.subplots(figsize=(6,3))
        ax.axis('off')
        ax.text(0.5, 0.6, f'{galaxy}', ha='center', va='center', fontsize=14)
        ax.text(0.5, 0.4, 'AIP: no valid indices for alpha/Fe', ha='center', va='center')
        fig.savefig(plots_dir / f'{galaxy}_AIP_alpha_fe_map.png', dpi=150)
        plt.close(fig)
        return None

    # Save quick 2D map
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(alpha_fe_2d, origin='lower', cmap='RdYlBu_r',
                   vmin=np.nanpercentile(alpha_fe_2d, 5),
                   vmax=np.nanpercentile(alpha_fe_2d, 95))
    ax.set_title(f'{galaxy}: alpha/Fe map (AIP)')
    plt.colorbar(im, ax=ax, label='[alpha/Fe]')
    fig.savefig(plots_dir / f'{galaxy}_AIP_alpha_fe_map.png', dpi=150)
    plt.close(fig)

    # Load RDB for radial bins
    rdb_path = data_dir / f'{galaxy}_stack_RDB_results.npz'
    if not rdb_path.exists():
        print('RDB results not found; radial gradient will be limited.')
        return None

    rdb_data = dict(np.load(rdb_path, allow_pickle=True))

    # Assemble alpha_fe_data dict compatible with gradient analysis
    alpha_fe_data = {
        'galaxy_name': galaxy,
        'alpha_fe_2d': alpha_fe_2d,
        'alpha_fe_errors': alpha_fe_err,
        'mean_alpha_fe': float(np.nanmean(alpha_fe_2d)),
        'std_alpha_fe': float(np.nanstd(alpha_fe_2d)),
    }

    radial_profile = calculate_radial_alpha_fe_profile(alpha_fe_data, rdb_data)
    if radial_profile is None:
        print('Radial profile failed.')
        return None

    # Fit gradients (RDB, 1.5Re, 2.0Re; VNB optional if available)
    vnb_path = data_dir / f'{galaxy}_stack_VNB_results.npz'
    vnb_data = dict(np.load(vnb_path, allow_pickle=True)) if vnb_path.exists() else None
    gradient_results = fit_alpha_fe_gradient_multi_method(radial_profile,
                                                          vnb_profile=None)

    # Save a compact summary plot
    try:
        radii = radial_profile['bin_radii'][~np.isnan(radial_profile['alpha_fe_mean'])]
        alpha = radial_profile['alpha_fe_mean'][~np.isnan(radial_profile['alpha_fe_mean'])]
        err = radial_profile['alpha_fe_error'][~np.isnan(radial_profile['alpha_fe_mean'])]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.errorbar(radii, alpha, yerr=err, fmt='o', capsize=3)
        ax.set_xlabel('R / Re')
        ax.set_ylabel('[alpha/Fe]')
        ax.set_title(f'{galaxy}: alpha/Fe radial profile (AIP)')
        fig.savefig(plots_dir / f'{galaxy}_AIP_alpha_fe_radial_profile.png', dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # Save NPZ summary for reuse
    np.savez(out_dir / f'{galaxy}_AIP_alpha_fe_results.npz',
             alpha_fe_2d=alpha_fe_2d,
             alpha_fe_errors=alpha_fe_err,
             radial_profile=radial_profile,
             gradient_results=gradient_results)

    return {
        'alpha_fe_2d': alpha_fe_2d,
        'radial_profile': radial_profile,
        'gradient_results': gradient_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy', required=True)
    ap.add_argument('--workspace', default='.')
    args = ap.parse_args()
    run_aip_for_galaxy(args.galaxy, Path(args.workspace))


if __name__ == '__main__':
    main()
