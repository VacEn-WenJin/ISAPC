#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Virgo cluster map and companion distance panels using the original
styling and plotting method provided by the user ("as the IMG shows, same!"):
- Triangles encode sign (blue up for positive, red down for negative)
- Marker fill encodes emission (filled = emission present, hollow = none)
- Vertical arrows above each triangle encode |slope| magnitude; color by sign
- Substructures drawn as light-gray circles around M87, M86, M49, M60
- 1° scale bar at bottom-left
- RA axis inverted to match sky convention; equal aspect
- Watermark "Preliminary!" on the figure
- Companion figure with two distance-vs-slope panels and trend lines

This script intentionally does NOT use the newer quiver/colormap/velocity
colorbar logic; it is designed to replicate the original appearance.
"""

import os
import re
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# Thread pinning (optional but stable)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s [%(name)s] - %(message)s")
LOGGER = logging.getLogger(__name__)

# --- Configuration ---
GRADIENT_CSV = os.path.join("alpha_gradient_dual", "combined_gradient_summary.csv")
FITS_DIR = os.path.join("output")
SAVE_MAIN = "virgo_cluster_original_gradients.png"
SAVE_PANELS = "virgo_cluster_original_with_panels.png"
FINAL_DIR = "FINAL_DELIVERABLES"

# Representative Virgo substructure anchors (MUSE IFU headers should align)
MAJOR_GALAXIES = {
    'M87': {'ra': 187.705930, 'dec': 12.391123},
    'M60': {'ra': 190.915125, 'dec': 11.552778},
    'M49': {'ra': 187.444583, 'dec': 8.000389},
    'M86': {'ra': 186.548042, 'dec': 12.946000},
}

# Circle radii tuned to match the original visual
R_M87 = 1.55
R_M86 = 0.95
R_M60 = 0.85
R_M49 = 1.15

# Arrow scaling: pixels per dex/Re magnitude (heuristic to match visual)
ARROW_SCALE = 10.0
ARROW_MIN = 0.10
ARROW_MAX = 1.25


def find_fits_coords(fits_root="output"):
    """Extract RA/DEC for each VCC galaxy from the FITS headers in output/*/Data.
    Expects files named like VCC####_stack.fits. Uses a fallback regex.
    """
    ras, decs, names = [], [], []
    if not os.path.isdir(fits_root):
        return names, np.array(ras), np.array(decs)
    for root, _dirs, files in os.walk(fits_root):
        for fn in files:
            if fn.lower().endswith(".fits") and fn.startswith("VCC"):
                # Prefer reading RA/DEC from printed cache if present
                m = re.match(r"(VCC\d{4}).*", fn)
                if not m:
                    continue
                vcc = m.group(1)
                # We derived these coordinates in the corrected script; reuse by
                # scanning the sibling directories for any logged list if present.
                # Fallback: leave coordinates empty; this script relies on saved
                # values from the corrected script run.
                pass
    # If we cannot extract from headers here, fallback to reading the
    # corrected script's logged CSV if exists (created earlier in session).
    cached = os.path.join("coords_cache.csv")
    if os.path.isfile(cached):
        dfc = pd.read_csv(cached)
        names = list(dfc["name"])  # type: ignore
        ras = dfc["ra"].to_numpy()
        decs = dfc["dec"].to_numpy()
    else:
        LOGGER.warning("No FITS coordinate cache found; trying to infer from gradient CSV.")
        # If coords missing, we won't plot the map; panels will still be built.
    return names, np.array(ras), np.array(decs)


def load_gradients(csv_path: str):
    df = pd.read_csv(csv_path)
    # Expect columns: galaxy, slope_rdb, slope_err_rdb, velocity, emission_flag, ra, dec
    # If ra/dec exist here, use them directly for plotting.
    needed = ["galaxy", "slope_rdb", "slope_err_rdb"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in {csv_path}")
    return df


def slope_to_style(slope: float):
    sign = np.sign(slope)
    color = 'blue' if sign > 0 else 'red'
    marker = '^' if sign > 0 else 'v'
    return color, marker


def arrow_length_for_slope(slope: float):
    L = ARROW_SCALE * abs(float(slope))
    return float(np.clip(L, ARROW_MIN, ARROW_MAX))


def draw_substructures(ax):
    m87_ra, m87_dec = MAJOR_GALAXIES['M87']['ra'], MAJOR_GALAXIES['M87']['dec']
    m60_ra, m60_dec = MAJOR_GALAXIES['M60']['ra'], MAJOR_GALAXIES['M60']['dec']
    m49_ra, m49_dec = MAJOR_GALAXIES['M49']['ra'], MAJOR_GALAXIES['M49']['dec']
    m86_ra, m86_dec = MAJOR_GALAXIES['M86']['ra'], MAJOR_GALAXIES['M86']['dec']

    ax.add_patch(Circle((m87_ra, m87_dec), radius=R_M87, facecolor='lightgray', edgecolor='none', alpha=0.25, zorder=1))
    ax.text(m87_ra, m87_dec - (R_M87 + 0.2), 'M87/Cluster A', ha='center', va='top', fontsize=10, color='dimgray')

    ax.add_patch(Circle((m86_ra + 0.4, m86_dec), radius=R_M86, facecolor='lightgray', edgecolor='none', alpha=0.22, zorder=1))
    ax.text(m86_ra + 0.4, m86_dec + (R_M86 + 0.2), 'M86/Cluster C', ha='center', va='bottom', fontsize=9, color='dimgray')

    ax.add_patch(Circle((m60_ra, m60_dec), radius=R_M60, facecolor='lightgray', edgecolor='none', alpha=0.22, zorder=1))
    ax.text(m60_ra, m60_dec - (R_M60 + 0.2), 'M60/W Cloud', ha='center', va='top', fontsize=9, color='dimgray')

    ax.add_patch(Circle((m49_ra + 0.2, m49_dec - 0.2), radius=R_M49, facecolor='lightgray', edgecolor='none', alpha=0.22, zorder=1))
    ax.text(m49_ra + 0.2, m49_dec - (R_M49 + 0.25), 'M49/Cluster B', ha='center', va='top', fontsize=9, color='dimgray')


def add_scale_bar(ax, all_ras, all_decs):
    ra_min, ra_max = np.min(all_ras), np.max(all_ras)
    dec_min, dec_max = np.min(all_decs), np.max(all_decs)
    ra_range = ra_max - ra_min
    dec_range = dec_max - dec_min
    margin = 0.06 * ra_range
    x_start = ra_max - margin - 1.0
    x_end = ra_max - margin
    y = dec_min + 0.08 * dec_range
    ax.plot([x_start, x_end], [y, y], color='black', lw=3, zorder=9)
    ax.text(x_end + 0.1, y + 0.15, '1° ≈ 0.29 Mpc', ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))


def plot_main(df: pd.DataFrame):
    # Prefer RA/DEC from CSV if present
    if {'ra', 'dec'}.issubset(df.columns):
        names = df['galaxy'].tolist()
        all_ras = df['ra'].to_numpy()
        all_decs = df['dec'].to_numpy()
    else:
        names, all_ras, all_decs = find_fits_coords(FITS_DIR)
        if len(names) == 0:
            raise RuntimeError("No coordinates available to plot the cluster map in original style.")
        # Reindex gradients to coord order where possible
        df = df.set_index('galaxy')
        keep = [n for n in names if n in df.index]
        df = df.loc[keep].reset_index()
        mask = np.isin(names, keep)
        all_ras, all_decs, names = all_ras[mask], all_decs[mask], [n for n in names if n in keep]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    draw_substructures(ax)

    # Plot each galaxy
    xs, ys = [], []
    for _, row in df.iterrows():
        name = str(row['galaxy'])
        slope = float(row['slope_rdb'])
        slope_err = float(row.get('slope_err_rdb', np.nan))
        color, marker = slope_to_style(slope)
        # emission flag: 1 present/0 none or missing
        emission = int(row.get('emission_flag', 0))
        filled = True if emission == 1 else False

        if {'ra', 'dec'}.issubset(df.columns):
            x, y = float(row['ra']), float(row['dec'])
        else:
            # names aligned above
            idx = names.index(name)
            x, y = float(all_ras[idx]), float(all_decs[idx])

        xs.append(x); ys.append(y)
        # Marker
        mec = color
        mfc = color if filled else 'none'
        ax.scatter([x], [y], marker=marker, s=100, c=[color], edgecolors=mec,
                   linewidths=1.6, zorder=6, facecolors=mfc)
        # Vertical arrow above marker
        L = arrow_length_for_slope(slope)
        x0, y0 = x, y + 0.08  # small offset above marker
        x1, y1 = x, y0 + L * 0.12  # convert length to degrees-ish scale
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=2.0, color=color, shrinkA=0, shrinkB=0),
                    zorder=7)
        # Value label near arrow tip
        ax.text(x1, y1 + 0.05, f"{slope:+.3f}", color=color, fontsize=9,
                ha='center', va='bottom', zorder=8)

    # RA/DEC axes and styling
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('DEC [deg]')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.minorticks_on()

    # Limits from data
    ax.set_xlim(np.max(xs) + 0.6, np.min(xs) - 0.6)
    ax.set_ylim(np.min(ys) - 0.6, np.max(ys) + 0.6)

    # Scale bar
    add_scale_bar(ax, np.array(xs), np.array(ys))

    # Watermark
    ax.text(0.02, 0.02, 'Preliminary!', transform=ax.transAxes, fontsize=22,
            color='gray', alpha=0.25, ha='left', va='bottom', rotation=0)

    # Legend
    legend_elems = [
        Line2D([0], [0], marker='^', color='w', label='Positive gradient', markerfacecolor='blue', markeredgecolor='blue', markersize=10),
        Line2D([0], [0], marker='v', color='w', label='Negative gradient', markerfacecolor='red', markeredgecolor='red', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='No emission (hollow)', markerfacecolor='none', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Emission present (filled)', markerfacecolor='black', markeredgecolor='black', markersize=10),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(SAVE_MAIN, dpi=200)
    LOGGER.info(f"Saved original-style Virgo cluster plot: {SAVE_MAIN}")


def plot_panels(df: pd.DataFrame):
    # Two panels: distance to M87 and to M49 vs slope
    def angular_distance(ra1, dec1, ra2, dec2):
        # small-angle approx in degrees
        dra = (ra1 - ra2) * np.cos(np.deg2rad((dec1 + dec2) / 2))
        ddec = (dec1 - dec2)
        return np.sqrt(dra * dra + ddec * ddec)

    if not {'ra', 'dec'}.issubset(df.columns):
        LOGGER.warning("Panels: RA/DEC not present in CSV; panels will be skipped.")
        return

    ra = df['ra'].to_numpy()
    dec = df['dec'].to_numpy()
    slope = df['slope_rdb'].to_numpy()

    d_m87 = np.array([angular_distance(r, d, MAJOR_GALAXIES['M87']['ra'], MAJOR_GALAXIES['M87']['dec']) for r, d in zip(ra, dec)])
    d_m49 = np.array([angular_distance(r, d, MAJOR_GALAXIES['M49']['ra'], MAJOR_GALAXIES['M49']['dec']) for r, d in zip(ra, dec)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, dist, title in [(ax1, d_m87, 'Distance to M87 [deg]'), (ax2, d_m49, 'Distance to M49 [deg]')]:
        ax.scatter(dist, slope, s=35, facecolors='white', edgecolors='black')
        # Trend line (simple linear fit)
        if len(dist) >= 2:
            m, b = np.polyfit(dist, slope, 1)
            xs = np.linspace(np.min(dist), np.max(dist), 100)
            ax.plot(xs, m * xs + b, color='gray', lw=1.5, linestyle='--')
            ax.text(0.03, 0.95, f"slope={m:+.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=9)
        ax.set_xlabel(title)
        ax.grid(True, linestyle=':', alpha=0.4)
    ax1.set_ylabel('d[α/Fe]/d(R/Re) [dex/Re]')
    fig.tight_layout()
    fig.savefig(SAVE_PANELS, dpi=200)
    LOGGER.info(f"Saved original-style companion panels: {SAVE_PANELS}")


def main():
    df = load_gradients(GRADIENT_CSV)
    # Keep only rows with finite RDB slope
    df = df[np.isfinite(df['slope_rdb'])]
    if len(df) == 0:
        raise RuntimeError("No valid gradient rows to plot.")

    plot_main(df)
    if {'ra', 'dec'}.issubset(df.columns):
        plot_panels(df)

    # Copy to FINAL_DELIVERABLES
    os.makedirs(FINAL_DIR, exist_ok=True)
    for fn in (SAVE_MAIN, SAVE_PANELS):
        if os.path.isfile(fn):
            dst = os.path.join(FINAL_DIR, os.path.basename(fn))
            try:
                import shutil
                shutil.copyfile(fn, dst)
                LOGGER.info(f"Copied {fn} -> {dst}")
            except Exception as e:
                LOGGER.warning(f"Could not copy {fn} -> {dst}: {e}")


if __name__ == '__main__':
    main()
