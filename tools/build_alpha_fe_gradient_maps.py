#!/usr/bin/env python3
"""
Compute |∇(alpha/Fe)| gradient magnitude maps from AIP results and save PNGs.

Inputs per galaxy (under output/<GALAXY>):
 - <GALAXY>_AIP_alpha_fe_results.npz containing 'alpha_fe_2d'

Outputs:
 - output/<GALAXY>/Plots/<GALAXY>_AIP_alpha_fe_gradient_mag.png

Usage:
  python tools/build_alpha_fe_gradient_maps.py
  python tools/build_alpha_fe_gradient_maps.py --galaxy G1,G2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def discover_galaxies(selected: list[str] | None = None) -> list[str]:
    out = Path('output')
    gals: list[str] = []
    for d in sorted(out.iterdir() if out.exists() else []):
        if not d.is_dir():
            continue
        name = d.name
        if selected and name not in selected:
            continue
        if (d / f'{name}_AIP_alpha_fe_results.npz').exists():
            gals.append(name)
    return gals


def compute_grad_mag(arr: np.ndarray) -> np.ndarray:
    # Finite mask
    a = np.asarray(arr, dtype=float)
    # Replace NaNs for gradient computation with nearest finite via simple inpainting
    if not np.isfinite(a).all():
        m = np.isfinite(a)
        if m.any():
            med = np.nanmedian(a[m])
            a = np.where(m, a, med)
        else:
            return np.full_like(a, np.nan)
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    return g


def save_grad_png(galaxy: str, alpha2d: np.ndarray):
    g = compute_grad_mag(alpha2d)
    plots = Path('output')/galaxy/'Plots'
    plots.mkdir(parents=True, exist_ok=True)
    vmin = np.nanpercentile(g, 5)
    vmax = np.nanpercentile(g, 95)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(g, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'{galaxy}: |∇(alpha/Fe)|')
    plt.colorbar(im, ax=ax, label='mag per pixel')
    fig.savefig(plots / f'{galaxy}_AIP_alpha_fe_gradient_mag.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Build gradient magnitude maps from AIP alpha/Fe 2D results')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated galaxy names (match output/<galaxy>)')
    args = ap.parse_args()

    selected = [g.strip() for g in args.galaxy.split(',') if g.strip()] if args.galaxy else None
    galaxies = discover_galaxies(selected)
    if not galaxies:
        print('No galaxies with AIP alpha/Fe NPZ found under output/.')
        return 0
    for g in galaxies:
        npz = Path('output')/g/f'{g}_AIP_alpha_fe_results.npz'
        try:
            d = np.load(npz, allow_pickle=True)
            alpha2d = d.get('alpha_fe_2d')
            if alpha2d is None:
                print(f'✗ {g}: NPZ lacks alpha_fe_2d')
                continue
            save_grad_png(g, alpha2d)
            print(f'✓ {g}: wrote gradient magnitude map')
        except Exception as e:
            print(f'✗ {g}: error {e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
