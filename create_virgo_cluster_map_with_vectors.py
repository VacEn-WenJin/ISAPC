#!/usr/bin/env python3
"""
Original-style Virgo Cluster map with vectors

This script recreates the cluster map exactly in the original style:
- Triangles (blue up / red down) encode gradient sign
- Filled vs hollow indicates emission presence
- Vertical arrows above triangles encode magnitude |slope| with sign color
- Light gray substructure circles (M87/A, M49/B, M60/W, M86/C)
- RA/DEC axes with equal scale; 1° scale bar at bottom-left
- Optional companion figure with distance panels and simple trend lines

Inputs:
- Coordinates from IFU FITS headers (fallback to hardcoded if missing)
- Gradients from alpha_gradient_dual/combined_gradient_summary.csv
- Emission flags from ISAPC_Galaxy.GALAXIES if available

Outputs:
- FINAL_DELIVERABLES/virgo_cluster_map_with_vectors.png
- FINAL_DELIVERABLES/virgo_cluster_map_with_vectors_panels.png
"""

import os
import glob
import logging
from typing import Dict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits

# Emission metadata (optional)
try:
    from ISAPC_Galaxy import GALAXIES as _GAL_LIST
except Exception:
    _GAL_LIST = []

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_ifu_coordinates() -> Dict[str, Dict[str, float]]:
    coords: Dict[str, Dict[str, float]] = {}
    muse_dir = "data/MUSE"
    if not os.path.exists(muse_dir):
        logger.warning("IFU data directory missing, using fallback coordinates")
        return {}
    for fp in glob.glob(os.path.join(muse_dir, "VCC*_stack.fits")):
        name = os.path.basename(fp).replace("_stack.fits", "")
        try:
            with fits.open(fp) as hdul:
                h = hdul[0].header
                ra = float(h.get('CRVAL1', h.get('RA')))
                dec = float(h.get('CRVAL2', h.get('DEC')))
                if ra is not None and dec is not None:
                    coords[name] = {"ra": ra, "dec": dec}
        except Exception as e:
            logger.warning(f"Coord read failed for {name}: {e}")
    return coords


def get_fallback_coordinates() -> Dict[str, Dict[str, float]]:
    return {
        'VCC0308': {'ra': 184.712195, 'dec': 14.687528},
        'VCC0667': {'ra': 185.559375, 'dec': 13.160167},
        'VCC0688': {'ra': 187.512625, 'dec': 11.378750},
        'VCC0990': {'ra': 186.933500, 'dec': 12.540889},
        'VCC1049': {'ra': 185.545958, 'dec': 13.491889},
        'VCC1146': {'ra': 186.341333, 'dec': 12.053500},
        'VCC1193': {'ra': 185.982292, 'dec': 14.226056},
        'VCC1368': {'ra': 187.702667, 'dec': 12.391139},
        'VCC1410': {'ra': 186.906208, 'dec': 12.813889},
        'VCC1431': {'ra': 187.774792, 'dec': 12.892750},
        'VCC1486': {'ra': 189.974208, 'dec': 13.734583},
        'VCC1499': {'ra': 188.256458, 'dec': 12.773500},
        'VCC1549': {'ra': 187.325708, 'dec': 13.621528},
        'VCC1588': {'ra': 189.325792, 'dec': 14.880750},
        'VCC1695': {'ra': 186.525708, 'dec': 13.176056},
        'VCC1811': {'ra': 187.836792, 'dec': 12.720333},
        'VCC1890': {'ra': 186.969792, 'dec': 13.942333},
        'VCC1902': {'ra': 187.230958, 'dec': 12.156972},
        'VCC1910': {'ra': 190.741792, 'dec': 12.400722},
        'VCC1949': {'ra': 186.844542, 'dec': 13.552083}
    }


def get_emission_flags() -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    def _norm(name: str) -> str:
        s = str(name).strip().upper().replace(' ', '')
        m = re.match(r'([A-Z]+)(\d+)$', s)
        if m:
            pre, dig = m.groups()
            try:
                return f"{pre}{int(dig)}"
            except Exception:
                return s
        return s
    try:
        for g in _GAL_LIST:
            name = g.get('name') or g.get('galaxy')
            if name:
                flags[_norm(name)] = bool(g.get('has_emission', False))
    except Exception:
        pass
    # Apply local overrides if present
    try:
        overrides_path = os.path.join('data', 'emission_overrides.csv')
        if os.path.exists(overrides_path):
            df = pd.read_csv(overrides_path)
            df = df.dropna(subset=['name'])
            default_rows = df[df['name'].astype(str).str.upper().eq('DEFAULT')]
            if not default_rows.empty and 'has_emission' in default_rows:
                default_val = bool(int(default_rows.iloc[0]['has_emission']))
                flags = {k: default_val for k in flags.keys()}
            for _, row in df.iterrows():
                raw = row['name']
                if str(raw).strip().upper() == 'DEFAULT':
                    continue
                key = _norm(raw)
                val = row.get('has_emission')
                if pd.notna(val):
                    flags[key] = bool(int(val))
    except Exception:
        pass
    return flags


def get_galaxy_velocities() -> Dict[str, int]:
    return {
        'VCC0308': 1572, 'VCC0667': 1431, 'VCC0688': 1061, 'VCC0990': 1345,
        'VCC1049': 1249, 'VCC1146': 1404, 'VCC1193': 1543, 'VCC1368': 1322,
        'VCC1410': 1411, 'VCC1431': 1379, 'VCC1486': 1588, 'VCC1499': 1386,
        'VCC1549': 1359, 'VCC1588': 1947, 'VCC1695': 1359, 'VCC1811': 1386,
        'VCC1890': 1438, 'VCC1902': 1452, 'VCC1910': 1995, 'VCC1949': 1283
    }

def get_major_velocities() -> Dict[str, int]:
    """Approximate systemic velocities (km/s) for major Virgo members."""
    return {
        'M87': 1307,
        'M86': -244,
        'M60': 1117,
        'M49': 997,
    }

def load_gradient_summary() -> Dict[str, Dict[str, float]]:
    path = "alpha_gradient_dual/combined_gradient_summary.csv"
    if not os.path.exists(path):
        logger.error(f"Missing gradient file: {path}")
        return {}
    df = pd.read_csv(path)
    grads: Dict[str, Dict[str, float]] = {}
    for gal in df['Galaxy'].unique():
        sub = df[df['Galaxy'] == gal]
        # Prefer RDB; fallback VNB
        pick = None
        if 'RDB' in sub['Mode'].values:
            pick = sub[sub['Mode'] == 'RDB'].iloc[0]
        elif 'VNB' in sub['Mode'].values:
            pick = sub[sub['Mode'] == 'VNB'].iloc[0]
        if pick is not None:
            grads[gal] = {
                'slope': float(pick['Slope']),
                'slope_error': float(pick['Slope_Error']) if 'Slope_Error' in pick else np.nan,
            }
    return grads


def ensure_outdir() -> str:
    outdir = os.path.join("FINAL_DELIVERABLES")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def plot_cluster_original():
    coords = extract_ifu_coordinates() or get_fallback_coordinates()
    grads = load_gradient_summary()
    emis = get_emission_flags()
    velocities = get_galaxy_velocities()

    valid = {k: v for k, v in coords.items() if k in grads}
    if not valid:
        logger.error("No valid galaxies with gradients and coordinates")
        return None

    # Prepare figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Major galaxies (positions)
    major = {
        'M87': (187.70591, 12.39112),
        'M86': (186.54958, 12.94694),
        'M60': (190.9162, 11.5522),
        'M49': (187.4441, 8.0035),
    }

    # Limits and aspect
    all_ras = [v['ra'] for v in valid.values()]
    all_decs = [v['dec'] for v in valid.values()]
    ra_min, ra_max = min(all_ras), max(all_ras)
    dec_min, dec_max = min(all_decs), max(all_decs)
    pad = max(ra_max - ra_min, dec_max - dec_min) * 0.15
    ax.set_xlim(ra_max + pad, ra_min - pad)
    ax.set_ylim(dec_min - pad, dec_max + pad)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Style constants & velocity-based coloring (relative) BEFORE substructure fills
    POS_C, NEG_C = '#1f77b4', '#d62728'
    vel_values = [velocities.get(g, np.nan) for g in valid.keys()]
    vel_arr = np.array([v for v in vel_values if np.isfinite(v)])
    v_mean = float(np.nanmean(vel_arr)) if len(vel_arr) else 0.0
    dv = {g: velocities.get(g, v_mean) - v_mean for g in valid.keys()}
    vmax = max(abs(v) for v in dv.values()) if dv else 1.0
    vmax = max(vmax, 1.0)
    cmap = plt.get_cmap('cool')

    # Uniform alpha for all icons/data points
    uniform_alpha = 0.95

    # Substructure circles (color by mean relative velocity of enclosed galaxies)
    r_m87, r_m49, r_m60, r_m86 = 1.55, 1.15, 0.85, 0.95
    sub_defs = [
        ("M87/Cluster A", major['M87'], r_m87, -1),
        ("M49/Cluster B", major['M49'], r_m49, -1),
        ("M60/W Cloud",   major['M60'], r_m60, -1),
        ("M86/Cluster C", major['M86'], r_m86, +1),
    ]
    for label, center, rad, label_sign in sub_defs:
        cx, cy = center
        enclosed = []
        for gal_name, pos in valid.items():
            dx = pos['ra'] - cx
            dy = pos['dec'] - cy
            if dx*dx + dy*dy <= rad*rad:
                enclosed.append(dv.get(gal_name, 0.0))
        mean_dv = np.nanmean(enclosed) if enclosed else 0.0
        norm_val = 0.5 + 0.5 * (mean_dv / vmax) if vmax > 0 else 0.5
        norm_val = min(max(norm_val, 0.0), 1.0)
        fill_color = cmap(norm_val)
        ax.add_patch(Circle((cx, cy), radius=rad, facecolor=fill_color, edgecolor='none', alpha=0.30, zorder=1))
        offset = (rad + 0.2) * (1 if label_sign > 0 else -1)
        va = 'bottom' if label_sign > 0 else 'top'
        ax.text(cx, cy + offset, label, ha='center', va=va,
                fontsize=9 if 'Cluster B' in label or 'Cluster C' in label else 10,
                color='dimgray')

    # Major galaxies (stars) colored by velocity relative to v_mean
    major_vel = get_major_velocities()
    for name, (ra, dec) in major.items():
        dvm = major_vel.get(name, v_mean) - v_mean
        norm_val = 0.5 + 0.5 * (dvm / vmax)
        norm_val = min(max(norm_val, 0.0), 1.0)
        star_color = cmap(norm_val)
        ax.scatter(ra, dec, s=700, marker='*', facecolors=star_color, edgecolors='black',
                   linewidth=2.0, zorder=8, alpha=uniform_alpha)
        ax.text(ra, dec - 0.18, name, ha='center', va='top', fontsize=11, fontweight='bold',
                alpha=uniform_alpha)
    # 0.10 dex/Re => 6% of DEC span
    dec_range = max(dec_max - dec_min, 1e-6)
    units_per_0p10_dec = 0.06 * dec_range

    # Non-overlapping simple name offsets (stack vertically by declination order)
    sorted_gals = sorted(valid.items(), key=lambda kv: kv[1]['dec'])
    name_offsets = {}
    for i, (g, xy) in enumerate(sorted_gals):
        name_offsets[g] = 0.15 + 0.04 * (i % 6)  # small stagger

    # Plot triangles and vertical arrows (annotate, not quiver, to match original)
    for gal, xy in valid.items():
        ra, dec = xy['ra'], xy['dec']
        slope = float(grads[gal]['slope'])
        has_em = bool(emis.get(gal, False))
        color = cmap(0.5 + 0.5 * (dv[gal] / vmax))
        marker = '^' if slope >= 0 else 'v'
        face = color if has_em else 'none'
        ax.scatter(ra, dec, s=260, marker=marker, facecolors=face,
                   edgecolors=("black" if has_em else color), linewidth=2,
                   zorder=6, alpha=uniform_alpha)

        # Arrow (start exactly at marker center)
        mag = abs(slope)
        L = (mag / 0.10) * units_per_0p10_dec
        L = float(np.clip(L, 0.02 * dec_range, 0.10 * dec_range))
        sign = 1.0 if slope >= 0 else -1.0
        y0 = dec
        y1 = y0 + sign * L
        ax.annotate('', xy=(ra, y1), xytext=(ra, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=2.2, color=color, shrinkA=0, shrinkB=0,
                                    alpha=uniform_alpha),
                    zorder=7)
        # Value label just beyond arrow tip
        ty = y1 + sign * (0.02 * dec_range)
        ax.text(ra, ty, f"{slope:+.2f}", ha='center', va='center', fontsize=9, fontweight='bold',
                color=color, zorder=8, alpha=uniform_alpha)

        # Name above
        ax.text(ra, dec + name_offsets.get(gal, 0.15), gal, ha='center', va='center',
                fontsize=9, fontweight='bold')

    # Axes labels and title
    ax.set_xlabel('Right Ascension (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Declination (degrees)', fontsize=14, fontweight='bold')
    ax.set_title('Virgo Cluster: α/Fe Radial Gradients d[α/Fe]/d(R/Re)', fontsize=16, fontweight='bold', pad=16)

    # Scale bar bottom-left (1°)
    x_margin = 0.06 * (ra_max - ra_min)
    x0 = ra_max - x_margin - 1.0
    x1 = ra_max - x_margin
    y = dec_min + 0.08 * (dec_max - dec_min)
    ax.plot([x0, x1], [y, y], color='black', lw=3, zorder=9)
    ax.text(x1 + 0.1, y + 0.15, '1° ≈ 0.29 Mpc', ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # (Watermark removed per request)

    # Legend elements: emission fill + velocity color coding
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    # Emission fill legend
    legend_emission = [
        Line2D([0],[0], marker='^', color='none', label='Emission present',
               markerfacecolor='black', markeredgecolor='black', markersize=9, lw=0),
        Line2D([0],[0], marker='^', color='none', label='No emission',
               markerfacecolor='none', markeredgecolor='black', markersize=9, lw=0)
    ]
    leg1 = ax.legend(handles=legend_emission, loc='upper right', frameon=True,
                     fontsize=9, title='Marker Fill (emission)')
    ax.add_artist(leg1)

    # Velocity legend (Δv)
    sample_vals = [-0.8 * vmax, 0.0, 0.8 * vmax]
    vel_handles = [
        Patch(facecolor=cmap(0.5 + 0.5 * (val / vmax) if vmax > 0 else 0.5), edgecolor='none',
              label=('Δv < 0' if i == 0 else ('Δv ≈ 0' if i == 1 else 'Δv > 0')))
        for i, val in enumerate(sample_vals)
    ]
    ax.legend(handles=vel_handles, loc='upper right', bbox_to_anchor=(1, 0.78),
              frameon=True, fontsize=9, title='Velocity (Δv)')

    # Velocity colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Velocity Δv (km/s)")
    try:
        cbar.set_ticks([-vmax, 0, vmax])
        cbar.set_ticklabels([f"{-vmax:.0f}", "0", f"{vmax:.0f}"])
    except Exception:
        pass

    # On-plot annotation for Δv=0 reference velocity
    try:
        # Place to the right of the colorbar
        cb_x = cbar.ax.get_position().x1
        cb_y0, cb_y1 = cbar.ax.get_position().y0, cbar.ax.get_position().y1
        ax.figure.text(cb_x + 0.01, (cb_y0 + cb_y1) / 2,
                       f"Δv = 0 at v = {v_mean:.0f} km/s",
                       va='center', ha='left', fontsize=9)
    except Exception:
        pass

    # Log the Δv=0 reference velocity used
    logger.info(f"Δv=0 reference (v_mean) = {v_mean:.1f} km/s")
    plt.tight_layout()
    outdir = ensure_outdir()
    out_png = os.path.join(outdir, 'virgo_cluster_map_with_vectors.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved original-style map: {out_png}")
    return valid, grads, major, dv, vmax, cmap


def plot_panels_original(valid, grads, major, dv_map=None, vmax=None, cmap=None):
    # Prepare data
    all_ras = [v['ra'] for v in valid.values()]
    all_decs = [v['dec'] for v in valid.values()]
    ra_min, ra_max = min(all_ras), max(all_ras)
    dec_min, dec_max = min(all_decs), max(all_decs)
    dec_range = max(dec_max - dec_min, 1e-6)
    units_per_0p10_dec = 0.06 * dec_range

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 1, 1, 1])
    ax_map = fig.add_subplot(gs[0, :])
    ax_m87 = fig.add_subplot(gs[1, 0])
    ax_m49 = fig.add_subplot(gs[1, 1])
    ax_m60 = fig.add_subplot(gs[2, 0])
    ax_m86 = fig.add_subplot(gs[2, 1])
    ax_ctr = fig.add_subplot(gs[3, :])

    # Map setup
    pad = max(ra_max - ra_min, dec_max - dec_min) * 0.15
    ax_map.set_xlim(ra_max + pad, ra_min - pad)
    ax_map.set_ylim(dec_min - pad, dec_max + pad)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3, linestyle='--')
    ax_map.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_map.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Substructure circles
    r_m87, r_m49, r_m60, r_m86 = 1.55, 1.15, 0.85, 0.95
    ax_map.add_patch(Circle(major['M87'], radius=r_m87, facecolor='lightgray', edgecolor='none', alpha=0.25, zorder=1))
    ax_map.text(major['M87'][0], major['M87'][1] - (r_m87 + 0.2), 'M87/Cluster A', ha='center', va='top', fontsize=10, color='dimgray')
    ax_map.add_patch(Circle(major['M49'], radius=r_m49, facecolor='lightgray', edgecolor='none', alpha=0.22, zorder=1))
    ax_map.text(major['M49'][0], major['M49'][1] - (r_m49 + 0.25), 'M49/Cluster B', ha='center', va='top', fontsize=9, color='dimgray')
    ax_map.add_patch(Circle(major['M60'], radius=r_m60, facecolor='lightgray', edgecolor='none', alpha=0.22, zorder=1))
    ax_map.text(major['M60'][0], major['M60'][1] - (r_m60 + 0.2), 'M60/W Cloud', ha='center', va='top', fontsize=9, color='dimgray')
    ax_map.add_patch(Circle(major['M86'], radius=r_m86, facecolor='lightgray', edgecolor='none', alpha=0.22, zorder=1))
    ax_map.text(major['M86'][0], major['M86'][1] + (r_m86 + 0.2), 'M86/Cluster C', ha='center', va='bottom', fontsize=9, color='dimgray')

    # Major stars
    for name, (ra, dec) in major.items():
        ax_map.scatter(ra, dec, s=600, marker='*', c='gold', edgecolors='black', linewidth=2.0, zorder=8)
        ax_map.text(ra, dec - 0.18, name, ha='center', va='top', fontsize=11, fontweight='bold')

    POS_C, NEG_C = '#1f77b4', '#d62728'

    # Plot galaxies on map and collect panel data
    centers = {
        'M87': major['M87'],
        'M49': major['M49'],
        'M60': major['M60'],
        'M86': major['M86'],
        'Center': (float(np.mean(all_ras)), float(np.mean(all_decs))),
    }
    panel_axes = {'M87': ax_m87, 'M49': ax_m49, 'M60': ax_m60, 'M86': ax_m86, 'Center': ax_ctr}
    panel_data = {k: {'x': [], 'y': [], 'c': []} for k in panel_axes}

    for gal, xy in valid.items():
        ra, dec = xy['ra'], xy['dec']
        slope = float(grads[gal]['slope'])
        if dv_map is not None and vmax:
            color = cmap(0.5 + 0.5 * (dv_map[gal] / vmax))
        else:
            color = POS_C if slope >= 0 else NEG_C
        marker = '^' if slope >= 0 else 'v'
        ax_map.scatter(ra, dec, s=240, marker=marker, facecolors='none', edgecolors=color, linewidth=2, zorder=6)
        L = (abs(slope) / 0.10) * units_per_0p10_dec
        L = float(np.clip(L, 0.02 * dec_range, 0.10 * dec_range))
        sign = 1.0 if slope >= 0 else -1.0
        y0 = dec
        y1 = y0 + sign * L
        ax_map.annotate('', xy=(ra, y1), xytext=(ra, y0),
                        arrowprops=dict(arrowstyle='-|>', lw=2.0, color=color, shrinkA=0, shrinkB=0),
                        zorder=7)
        ax_map.annotate('', xy=(ra, y1), xytext=(ra, y0),
                        arrowprops=dict(arrowstyle='-|>', lw=4.0, color='white', alpha=0.35, shrinkA=0, shrinkB=0),
                        zorder=6)
        ax_map.text(ra, y1 + sign * (0.02 * dec_range), f"{slope:+.2f}",
                    ha='center', va='center', fontsize=9, fontweight='bold', color=color,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white', alpha=0.6)])

        # Distances
        for k, (cr, cd) in centers.items():
            d = float(np.hypot(ra - cr, dec - cd))
            panel_data[k]['x'].append(d)
            panel_data[k]['y'].append(slope)
            panel_data[k]['c'].append(color)

    # Scale bar bottom-left on map
    x_margin = 0.06 * (ra_max - ra_min)
    x0 = ra_max - x_margin - 1.0
    x1 = ra_max - x_margin
    y = dec_min + 0.08 * (dec_max - dec_min)
    ax_map.plot([x0, x1], [y, y], color='black', lw=3, zorder=9)
    ax_map.text(x1 + 0.1, y + 0.15, '1° ≈ 0.29 Mpc', ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    ax_map.set_xlabel('RA (deg)')
    ax_map.set_ylabel('DEC (deg)')
    ax_map.set_title('Virgo Cluster: α/Fe Radial Gradients (Original Style)')

    # Build panels
    for key, axp in panel_axes.items():
        dd = panel_data[key]
        if dd['x']:
            axp.scatter(dd['x'], dd['y'], c=dd['c'], s=70, edgecolors='black', alpha=0.9)
            axp.axhline(0, color='black', linestyle='--', alpha=0.5)
            try:
                x = np.array(dd['x']); yv = np.array(dd['y'])
                if len(x) >= 3:
                    m, b = np.polyfit(x, yv, 1)
                    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                    axp.plot(xs, m * xs + b, 'k-', alpha=0.7)
                    axp.text(0.02, 0.05, f"slope={m:.3f}", transform=axp.transAxes,
                             bbox=dict(facecolor='white', alpha=0.7))
            except Exception:
                pass
        axp.set_xlabel('Angular Distance (deg)')
        axp.set_ylabel('α/Fe radial slope')
        axp.grid(True, alpha=0.3, linestyle='--')
        axp.xaxis.set_minor_locator(AutoMinorLocator(5))
        axp.yaxis.set_minor_locator(AutoMinorLocator(5))
        ymax = max(0.3, (max(map(abs, dd['y'])) if dd['y'] else 0.3) * 1.2)
        axp.set_ylim(-ymax, ymax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    outdir = ensure_outdir()
    out_png = os.path.join(outdir, 'virgo_cluster_map_with_vectors_panels.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved original-style panels: {out_png}")


if __name__ == "__main__":
    result = plot_cluster_original()
    if result:
        valid, grads, major, dv, vmax, cmap = result
        # Panels still use slope arrows; optionally color by dv
        plot_panels_original(valid, grads, major)
