#!/usr/bin/env python3
"""
Generate normalized spectrum plots for the three innermost RDB bins and a
single overlay figure per galaxy, following the earlier convention:

Saves under: output/<GALAXY>/Plots/RDB/spectral_indices/detailed/
  - RDB0_norm.png, RDB1_norm.png, RDB2_norm.png (per-bin panels)
  - RDB_first3_norm_overlay.png (overlay, 3 bins in one figure)
  - RDB_first3_norm_shaded_overlay.png (overlay with index windows shaded)
  - RDB_first3_norm_combined.png (stacked image of the 3 per-bin panels)

Inputs: uses output/<GALAXY>/Data/<GALAXY>_RDB_binned.npz
If normalized panels already exist and --force is not passed, they are reused.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _smooth(y: np.ndarray, win: int = 5) -> np.ndarray:
    if y is None or y.size == 0:
        return y
    win = max(1, int(win))
    if win <= 1:
        return y
    k = np.ones(win) / float(win)
    return np.convolve(y, k, mode='same')


def normalize_spectrum(wave: np.ndarray, flux: np.ndarray, win: Tuple[float,float] | None = None) -> np.ndarray:
    x = np.asarray(flux, dtype=float)
    if win is None:
        med = np.nanmedian(x)
    else:
        m = (wave >= win[0]) & (wave <= win[1])
        med = np.nanmedian(x[m]) if np.any(m) else np.nanmedian(x)
    if not np.isfinite(med) or med == 0:
        med = 1.0
    return x / med


def build_overlays_for_galaxy(galaxy: str, force: bool = False, norm_win: Optional[Tuple[float,float]] = None) -> Optional[Path]:
    data_dir = Path('output') / galaxy / 'Data'
    plots_dir = Path('output') / galaxy / 'Plots' / 'RDB' / 'spectral_indices' / 'detailed'
    plots_dir.mkdir(parents=True, exist_ok=True)
    binned_npz = data_dir / f'{galaxy}_RDB_binned.npz'
    if not binned_npz.exists():
        print(f'✗ {galaxy}: missing {binned_npz.name}')
        return None
    d = np.load(binned_npz, allow_pickle=True)
    if 'wavelength' not in d or 'spectra' not in d:
        print(f'✗ {galaxy}: NPZ lacks wavelength/spectra')
        return None
    wave = d['wavelength']
    spec = d['spectra']  # shape (n_wave, n_bins)
    n_bins = spec.shape[1]
    bins = list(range(min(3, n_bins)))

    saved_norm_paths: List[Path] = []
    overlay_traces = []

    for i in bins:
        out_png = plots_dir / f'RDB{i}_norm.png'
        if out_png.exists() and not force:
            saved_norm_paths.append(out_png)
            # For overlay, read back normalized trace
            try:
                # Fallback: recompute normalization; we avoid disk read of image
                pass
            except Exception:
                pass
        # Build normalized spectrum from spectra array
        f = spec[:, i]
        norm = normalize_spectrum(wave, f, norm_win)
        norm_sm = _smooth(norm, 7)
        # Save per-bin panel if missing/force
        if (not out_png.exists()) or force:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.plot(wave, norm_sm, lw=1.2, color='tab:blue')
            ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
            ax.set_xlim(wave[0], wave[-1])
            # Full-range y-lims with small padding to ensure entire spectrum is visible
            try:
                y_min = float(np.nanmin(norm_sm))
                y_max = float(np.nanmax(norm_sm))
                if not (np.isfinite(y_min) and np.isfinite(y_max)):
                    raise ValueError('non-finite')
                if y_max <= y_min:
                    y_min, y_max = y_min - 0.1, y_max + 0.1
                pad = 0.05 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
            except Exception:
                ax.set_ylim(0.6, 1.4)
            ax.set_xlabel('Rest-frame Wavelength (Å)')
            ax.set_ylabel('Normalized Flux')
            ax.set_title(f'{galaxy} — RDB bin {i} (normalized)')
            fig.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close(fig)
        saved_norm_paths.append(out_png)
        overlay_traces.append((wave, norm_sm, f'bin {i}'))

    # Combined stacked image of the 3 per-bin panels
    if saved_norm_paths:
        import matplotlib.image as mpimg
        rows = len(saved_norm_paths)
        figC, axesC = plt.subplots(rows, 1, figsize=(12, 4 * rows))
        axes_list = [axesC] if rows == 1 else list(axesC)
        for axC, img_path in zip(axes_list, saved_norm_paths):
            img = mpimg.imread(img_path)
            axC.imshow(img, aspect='auto')
            axC.set_xticks([]); axC.set_yticks([])
        combined_path = plots_dir / 'RDB_first3_norm_combined.png'
        figC.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close(figC)

    # Overlay without shading
    if overlay_traces:
        figO, axO = plt.subplots(1, 1, figsize=(12, 4.5))
        colors = ['tab:blue','tab:orange','tab:green']
        for j, (w, y, lbl) in enumerate(overlay_traces):
            axO.plot(w, y, lw=1.5, color=colors[j % 3], label=lbl)
        axO.axhline(1.0, color='k', lw=0.8, alpha=0.5)
        axO.set_xlim(wave[0], wave[-1])
        # Full-range y-lims across all three traces
        try:
            allv = np.concatenate([y for (_, y, _) in overlay_traces])
            y_min = float(np.nanmin(allv))
            y_max = float(np.nanmax(allv))
            if not (np.isfinite(y_min) and np.isfinite(y_max)):
                raise ValueError('non-finite')
            if y_max <= y_min:
                y_min, y_max = y_min - 0.1, y_max + 0.1
            pad = 0.05 * (y_max - y_min)
            axO.set_ylim(y_min - pad, y_max + pad)
        except Exception:
            axO.set_ylim(0.6, 1.4)
        axO.set_xlabel('Rest-frame Wavelength (Å)')
        axO.set_ylabel('Normalized Flux')
        axO.set_title(f'{galaxy} RDB: First 3 bins (overlay)')
        axO.legend(loc='best', frameon=False, ncol=min(3, len(overlay_traces)))
        ov_path = plots_dir / 'RDB_first3_norm_overlay.png'
        figO.savefig(ov_path, dpi=150, bbox_inches='tight')
        plt.close(figO)

        # Overlay with index-window shading if spectral_indices windows available
        try:
            from visualization import _get_default_index_windows as _def_wins
            wins = _def_wins()
        except Exception:
            wins = {}
        figS, axS = plt.subplots(1, 1, figsize=(12, 5))
        for nm, wv in wins.items():
            blue = wv.get('blue'); band = wv.get('band', wv.get('line')); red = wv.get('red')
            if blue and band and red:
                axS.axvspan(blue[0], blue[1], color='lightgray', alpha=0.18)
                axS.axvspan(band[0], band[1], color='silver', alpha=0.18)
                axS.axvspan(red[0], red[1], color='lightgray', alpha=0.18)
                xlbl = (band[0] + band[1]) / 2.0
                axS.text(xlbl, 0.305, nm, ha='center', va='bottom', fontsize='x-small', transform=axS.get_xaxis_transform())
        for j, (w, y, lbl) in enumerate(overlay_traces):
            axS.plot(w, y, lw=1.2, color=colors[j % 3], label=lbl)
        axS.axhline(1.0, color='k', lw=0.7, alpha=0.5)
        axS.set_xlim(wave[0], wave[-1])
        # Full-range y-lims with small padding for shaded overlay as well
        try:
            allv = np.concatenate([y for (_, y, _) in overlay_traces])
            y_min = float(np.nanmin(allv))
            y_max = float(np.nanmax(allv))
            if not (np.isfinite(y_min) and np.isfinite(y_max)):
                raise ValueError('non-finite')
            if y_max <= y_min:
                y_min, y_max = y_min - 0.1, y_max + 0.1
            pad = 0.05 * (y_max - y_min)
            axS.set_ylim(y_min - pad, y_max + pad)
        except Exception:
            axS.set_ylim(0.7, 1.3)
        axS.set_xlabel('Rest-frame Wavelength (Å)')
        axS.set_ylabel('Normalized Flux')
        axS.set_title(f'{galaxy} RDB bins 0–2 (normalized, shaded indices)')
        axS.legend(loc='upper right', frameon=False, ncol=min(3, len(overlay_traces)))
        sh_path = plots_dir / 'RDB_first3_norm_shaded_overlay.png'
        figS.savefig(sh_path, dpi=150, bbox_inches='tight')
        plt.close(figS)

    return plots_dir


def main():
    ap = argparse.ArgumentParser(description='Build RDB inner-3 normalized overlay plots for galaxies')
    ap.add_argument('--galaxy', type=str, default=None, help='Comma-separated list or single galaxy name (e.g., VCC1146_obstack)')
    ap.add_argument('--force', action='store_true', help='Rebuild even if files exist')
    ap.add_argument('--norm-win', nargs=2, type=float, default=None, help='Normalization window [A] (default: median over full spectrum)')
    args = ap.parse_args()

    if args.galaxy:
        galaxies = [g.strip() for g in args.galaxy.split(',') if g.strip()]
    else:
        galaxies = []
        for d in sorted((Path('output').iterdir() if Path('output').exists() else [])):
            if d.is_dir() and (d / 'Data').exists() and any(str(f).endswith('_RDB_binned.npz') for f in (d / 'Data').glob('*_RDB_binned.npz')):
                galaxies.append(d.name)

    if not galaxies:
        print('No galaxies with RDB binned NPZ found under output/.')
        return 0

    norm_win = tuple(args.norm_win) if args.norm_win else None
    for g in galaxies:
        out_dir = build_overlays_for_galaxy(g, force=args.force, norm_win=norm_win)
        if out_dir:
            print(f'✓ {g}: saved overlays -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
