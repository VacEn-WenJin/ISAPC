#!/usr/bin/env python3
"""
Final Virgo Cluster α/Fe Gradient Visualization - Corrected Version (MATCHED STYLE)

This script renders the Virgo cluster map matching the last corrected 2D style:
- Triangles encode gradient sign (up = positive, down = negative)
- Marker color encodes relative velocity Δv using the 'cool' colormap with symmetric norm
- Filled (emission) vs hollow (no emission) markers
- Vertical arrows above triangles encode |slope| magnitude and are colored by Δv
- Substructure circles (M87/A, M49/B, M60/W, M86/C) filled by the mean Δv of enclosed galaxies
- Major galaxy stars colored by Δv; Δv colorbar with ticks at [-vmax, 0, +vmax] and on-plot Δv=0 reference
- Equal RA/DEC scale, consistent alpha, no 'Preliminary' watermark
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Circle
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
import logging
import os
import glob
from typing import Dict
import re

# Local metadata for emission flags
try:
    from ISAPC_Galaxy import GALAXIES as _GAL_LIST
except Exception:
    _GAL_LIST = []

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_ifu_coordinates():
    """Extract galaxy coordinates from FITS headers"""
    coords = {}
    muse_dir = "data/MUSE"
    
    if not os.path.exists(muse_dir):
        logger.error(f"MUSE data directory not found: {muse_dir}")
        return {}
    
    fits_files = glob.glob(os.path.join(muse_dir, "VCC*_stack.fits"))
    logger.info(f"Found {len(fits_files)} FITS files")
    
    for fits_file in fits_files:
        try:
            # Extract galaxy name from filename
            basename = os.path.basename(fits_file)
            galaxy_name = basename.replace("_stack.fits", "")
            
            # Open FITS file and extract coordinates
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                
                # Try different possible coordinate keywords
                ra = None
                dec = None
                
                # Standard FITS keywords
                if 'CRVAL1' in header and 'CRVAL2' in header:
                    ra = float(header['CRVAL1'])
                    dec = float(header['CRVAL2'])
                elif 'RA' in header and 'DEC' in header:
                    ra = float(header['RA'])
                    dec = float(header['DEC'])
                elif 'CRVAL1' in header:
                    ra = float(header['CRVAL1'])
                    if 'CRVAL2' in header:
                        dec = float(header['CRVAL2'])
                
                if ra is not None and dec is not None:
                    coords[galaxy_name] = {'ra': ra, 'dec': dec}
                    logger.info(f"{galaxy_name}: RA={ra:.6f}, DEC={dec:.6f}")
                else:
                    logger.warning(f"Could not extract coordinates from {galaxy_name}")
                    
        except Exception as e:
            logger.error(f"Error processing {fits_file}: {e}")
    
    return coords

def get_fallback_coordinates():
    """Fallback coordinates if FITS extraction fails"""
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

def get_galaxy_velocities():
    """Get galaxy velocities for color coding"""
    return {
        'VCC0308': 1572, 'VCC0667': 1431, 'VCC0688': 1061, 'VCC0990': 1345,
        'VCC1049': 1249, 'VCC1146': 1404, 'VCC1193': 1543, 'VCC1368': 1322,
        'VCC1410': 1411, 'VCC1431': 1379, 'VCC1486': 1588, 'VCC1499': 1386,
        'VCC1549': 1359, 'VCC1588': 1947, 'VCC1695': 1359, 'VCC1811': 1386,
        'VCC1890': 1438, 'VCC1902': 1452, 'VCC1910': 1995, 'VCC1949': 1283
    }

def _norm_name(name: str) -> str:
    """Normalize galaxy names: strip spaces, uppercase, drop leading zeros in numeric part."""
    s = str(name).strip().upper().replace(' ', '')
    m = re.match(r'([A-Z]+)(\d+)$', s)
    if m:
        pre, dig = m.groups()
        try:
            return f"{pre}{int(dig)}"
        except Exception:
            return s
    return s


def get_emission_flags() -> Dict[str, bool]:
    """Build a map of galaxy -> has_emission from local GALAXIES list if available."""
    flags: Dict[str, bool] = {}
    try:
        if _GAL_LIST:
            for g in _GAL_LIST:
                name = g.get('name') or g.get('galaxy')
                if name:
                    flags[_norm_name(name)] = bool(g.get('has_emission', False))
    except Exception:
        pass
    # Apply local overrides if present
    try:
        overrides_path = os.path.join('data', 'emission_overrides.csv')
        if os.path.exists(overrides_path):
            df = pd.read_csv(overrides_path)
            df = df.dropna(subset=['name'])
            # Default row
            default_rows = df[df['name'].astype(str).str.upper().eq('DEFAULT')]
            if not default_rows.empty and 'has_emission' in default_rows:
                default_val = bool(int(default_rows.iloc[0]['has_emission']))
                # Initialize all known names to default
                flags = {k: default_val for k in flags.keys()}
            for _, row in df.iterrows():
                raw = row['name']
                if str(raw).strip().upper() == 'DEFAULT':
                    continue
                key = _norm_name(raw)
                val = row.get('has_emission')
                if pd.notna(val):
                    flags[key] = bool(int(val))
    except Exception:
        pass
    return flags

def get_major_velocities():
    """Approximate systemic velocities (km/s) for major Virgo members."""
    return {
        'M87': 1307,
        'M86': -244,
        'M60': 1117,
        'M49': 997,
    }

def load_final_gradient_data():
    """Load the definitive gradient data from enhanced_radial_plots analysis"""
    try:
        # Load the combined gradient summary with both RDB and VNB results
        combined_file = "alpha_gradient_dual/combined_gradient_summary.csv"
        fallback_file = "enhanced_radial_plots/enhanced_3bin_gradient_summary.csv"

        if not os.path.exists(combined_file):
            if os.path.exists(fallback_file):
                logger.warning(f"Using fallback gradient file: {fallback_file}")
                combined_file = fallback_file
            else:
                logger.error(f"Combined gradient file not found: {combined_file}")
                return {}

        # Load the CSV
        df = pd.read_csv(combined_file)
        logger.info(f"Loaded {len(df)} gradient measurements from {combined_file}")

        # Process data by galaxy
        galaxy_gradients = {}
        for galaxy_name in df['Galaxy'].unique():
            galaxy_data = df[df['Galaxy'] == galaxy_name]

            # Get RDB and VNB results for this galaxy
            rdb_data = galaxy_data[galaxy_data['Mode'] == 'RDB']
            vnb_data = galaxy_data[galaxy_data['Mode'] == 'VNB']

            galaxy_results = {}

            # Process RDB results
            if not rdb_data.empty:
                rdb_row = rdb_data.iloc[0]
                galaxy_results['RDB'] = {
                    'slope': rdb_row['Slope'],
                    'slope_error': rdb_row['Slope_Error'],
                    'significance': get_significance_level(rdb_row['Significance']) if 'Significance' in rdb_row else 0,
                    'p_value': rdb_row['P_value'] if 'P_value' in rdb_row else np.nan,
                    'r_squared': rdb_row['R_squared'] if 'R_squared' in rdb_row else np.nan,
                }

            # Process VNB results
            if not vnb_data.empty:
                vnb_row = vnb_data.iloc[0]
                galaxy_results['VNB'] = {
                    'slope': vnb_row['Slope'],
                    'slope_error': vnb_row['Slope_Error'],
                    'significance': get_significance_level(vnb_row['Significance']) if 'Significance' in vnb_row else 0,
                    'p_value': vnb_row['P_value'] if 'P_value' in vnb_row else np.nan,
                    'r_squared': vnb_row['R_squared'] if 'R_squared' in vnb_row else np.nan,
                }

            if galaxy_results:
                galaxy_gradients[galaxy_name] = galaxy_results

        return galaxy_gradients

    except Exception as e:
        logger.error(f"Error loading gradient data: {e}")
        return {}

def get_significance_level(sig_text):
    """Convert significance text to numeric level"""
    if sig_text == 'highly_significant':
        return 3
    elif sig_text == 'significant':
        return 2
    elif sig_text == 'marginal':
        return 1
    else:
        return 0

def get_best_gradient(galaxy_results):
    """Get the best gradient result for plotting (prefer RDB if significant, otherwise VNB)"""
    
    # Check if RDB exists and is significant
    if 'RDB' in galaxy_results:
        rdb = galaxy_results['RDB']
        if rdb['significance'] >= 2:  # Significant or highly significant
            return rdb, 'RDB'
    
    # Check if VNB exists and is significant
    if 'VNB' in galaxy_results:
        vnb = galaxy_results['VNB']
        if vnb['significance'] >= 2:
            return vnb, 'VNB'
    
    # If no significant results, prefer RDB for consistency, otherwise VNB
    if 'RDB' in galaxy_results:
        return galaxy_results['RDB'], 'RDB'
    elif 'VNB' in galaxy_results:
        return galaxy_results['VNB'], 'VNB'
    
    return None, None

def calculate_text_positions(galaxies_data, plot_limits):
    """Calculate non-overlapping text positions"""
    positions = {}
    ra_min, ra_max, dec_min, dec_max = plot_limits
    
    # Sort galaxies by declination for better spacing
    sorted_galaxies = sorted(galaxies_data.items(), key=lambda x: x[1]['dec'])
    
    for galaxy_name, data in sorted_galaxies:
        ra, dec = data['ra'], data['dec']
        
        # Base position
        name_offset = 0.15
        gradient_offset = -0.15
        
        # Check for overlaps and adjust
        overlap_found = True
        adjustment = 0
        
        while overlap_found and abs(adjustment) < 0.5:
            overlap_found = False
            test_name_pos = dec + name_offset + adjustment
            test_grad_pos = dec + gradient_offset + adjustment
            
            for other_name, other_pos in positions.items():
                if other_name != galaxy_name:
                    other_name_pos = other_pos['name_y']
                    other_grad_pos = other_pos['gradient_y']
                    
                    # Check if positions are too close
                    if (abs(test_name_pos - other_name_pos) < 0.12 or 
                        abs(test_grad_pos - other_grad_pos) < 0.12 or
                        abs(test_name_pos - other_grad_pos) < 0.12 or
                        abs(test_grad_pos - other_name_pos) < 0.12):
                        overlap_found = True
                        break
            
            if overlap_found:
                adjustment += 0.05
        
        positions[galaxy_name] = {
            'name_y': dec + name_offset + adjustment,
            'gradient_y': dec + gradient_offset + adjustment
        }
    
    return positions

def create_virgo_cluster_final():
    """Create the final, definitive Virgo cluster gradient plot"""
    
    # Get coordinates from FITS files or fallback
    logger.info("Extracting coordinates from FITS files...")
    ifu_coords = extract_ifu_coordinates()
    
    if not ifu_coords:
        logger.warning("Using fallback coordinates")
        ifu_coords = get_fallback_coordinates()
    
    # Get velocities and gradients
    galaxy_velocities = get_galaxy_velocities()
    major_velocities = get_major_velocities()
    emission_flags = get_emission_flags()
    galaxy_gradients = load_final_gradient_data()
    
    # Filter to galaxies with complete data
    valid_galaxies = {}
    for name in ifu_coords.keys():
        if name in galaxy_gradients and name in galaxy_velocities:
            valid_galaxies[name] = {
                'ra': ifu_coords[name]['ra'],
                'dec': ifu_coords[name]['dec'],
                'velocity': galaxy_velocities[name]
            }
    
    logger.info(f"Creating plot for {len(valid_galaxies)} galaxies with complete data")
    
    # Create figure with equal aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Major Virgo cluster galaxies for reference
    major_galaxies = {
        'M87': {'ra': 187.70591, 'dec': 12.39112, 'type': 'center'},
        'M86': {'ra': 186.54958, 'dec': 12.94694, 'type': 'major'},
        'M60': {'ra': 190.9162, 'dec': 11.5522, 'type': 'major'},
        'M49': {'ra': 187.4441, 'dec': 8.0035, 'type': 'major'}
    }
    
    # Velocity-based coloring setup
    POS_COLOR = '#1f77b4'
    NEG_COLOR = '#d62728'
    cmap = plt.get_cmap('cool')
    uniform_alpha = 0.95

    # Compute Δv relative to the mean of valid galaxy velocities
    vel_values = [galaxy_velocities.get(g, np.nan) for g in valid_galaxies.keys()]
    vel_arr = np.array([v for v in vel_values if np.isfinite(v)])
    v_mean = float(np.nanmean(vel_arr)) if len(vel_arr) else 0.0
    dv_gal = {g: galaxy_velocities.get(g, v_mean) - v_mean for g in valid_galaxies.keys()}
    vmax = max(abs(v) for v in dv_gal.values()) if dv_gal else 1.0
    vmax = max(vmax, 1.0)

    # Export a CSV with Δv for every plotted point (galaxies + major members)
    try:
        rows = []
        # Galaxies (triangles)
        for gname, ginfo in valid_galaxies.items():
            ra = float(ginfo['ra'])
            dec = float(ginfo['dec'])
            v = float(galaxy_velocities.get(gname, np.nan))
            dv = float(dv_gal.get(gname, np.nan))
            has_em = bool(emission_flags.get(_norm_name(gname), False))
            best, method = get_best_gradient(galaxy_gradients.get(gname, {}))
            slope = float(best['slope']) if best and 'slope' in best else np.nan
            slope_err = float(best['slope_error']) if best and 'slope_error' in best else np.nan
            rows.append({
                'name': gname,
                'category': 'galaxy',
                'ra_deg': ra,
                'dec_deg': dec,
                'v_kms': v,
                'v_mean_kms': v_mean,
                'delta_v_kms': dv,
                'has_emission': has_em,
                'slope_dex_per_Re': slope,
                'slope_error_dex_per_Re': slope_err,
                'gradient_method': method or ''
            })

        # Major Virgo members (stars)
        for mname, mdata in major_galaxies.items():
            ra = float(mdata['ra'])
            dec = float(mdata['dec'])
            v = float(major_velocities.get(mname, np.nan))
            dv = float(v - v_mean) if np.isfinite(v) else np.nan
            rows.append({
                'name': mname,
                'category': 'major',
                'ra_deg': ra,
                'dec_deg': dec,
                'v_kms': v,
                'v_mean_kms': v_mean,
                'delta_v_kms': dv,
                'has_emission': '',
                'slope_dex_per_Re': '',
                'slope_error_dex_per_Re': '',
                'gradient_method': ''
            })

        dv_df = pd.DataFrame(rows)
        os.makedirs('FINAL_DELIVERABLES', exist_ok=True)
        dv_path = os.path.join('FINAL_DELIVERABLES', 'virgo_cluster_relative_velocities.csv')
        dv_df.to_csv(dv_path, index=False)
        logger.info(f"Saved relative velocities CSV: {dv_path}")
    except Exception as e:
        logger.warning(f"Could not export relative velocities CSV: {e}")

    # Plot major galaxies colored by Δv (relative to v_mean)
    for name, data in major_galaxies.items():
        ra, dec = data['ra'], data['dec']
        dvm = major_velocities.get(name, v_mean) - v_mean
        norm_val = 0.5 + 0.5 * (dvm / vmax)
        norm_val = min(max(norm_val, 0.0), 1.0)
        star_color = cmap(norm_val)
        ax.scatter(ra, dec, s=700, marker='*', facecolors=star_color, edgecolors='black',
                   linewidth=2.0, zorder=8, alpha=uniform_alpha)
        ax.text(ra, dec - 0.18, name, ha='center', va='top', fontsize=11, fontweight='bold',
                alpha=uniform_alpha)

    # Add Virgo substructures as circles filled by mean Δv of enclosed galaxies
    m87_ra, m87_dec = major_galaxies['M87']['ra'], major_galaxies['M87']['dec']
    m60_ra, m60_dec = major_galaxies['M60']['ra'], major_galaxies['M60']['dec']
    m49_ra, m49_dec = major_galaxies['M49']['ra'], major_galaxies['M49']['dec']
    m86_ra, m86_dec = major_galaxies['M86']['ra'], major_galaxies['M86']['dec']
    # Radii approximate the prior ellipses' mean semi-axes
    r_m87 = 1.55  # ~mean of 2.0 and 1.1
    r_m86 = 0.95  # ~mean of 1.2 and 0.7, centered at slight RA offset
    r_m60 = 0.85  # ~mean of 1.1 and 0.6
    r_m49 = 1.15  # ~mean of 1.5 and 0.8, slight offset in original

    # Cluster A (M87)
    # Compute and render each substructure fill color from mean Δv of enclosed galaxies
    sub_defs = [
        ("M87/Cluster A", (m87_ra, m87_dec), r_m87, -1),
        ("M86/Cluster C", (m86_ra + 0.4, m86_dec), r_m86, +1),
        ("M60/W Cloud",   (m60_ra, m60_dec), r_m60, -1),
        ("M49/Cluster B", (m49_ra + 0.2, m49_dec - 0.2), r_m49, -1),
    ]
    for label, (cx, cy), rad, label_sign in sub_defs:
        enclosed = []
        for gal_name, pos in valid_galaxies.items():
            dx = pos['ra'] - cx
            dy = pos['dec'] - cy
            if dx*dx + dy*dy <= rad*rad:
                enclosed.append(dv_gal.get(gal_name, 0.0))
        mean_dv = np.nanmean(enclosed) if enclosed else 0.0
        norm_val = 0.5 + 0.5 * (mean_dv / vmax) if vmax > 0 else 0.5
        norm_val = min(max(norm_val, 0.0), 1.0)
        fill_color = cmap(norm_val)
        ax.add_patch(Circle((cx, cy), radius=rad, facecolor=fill_color, edgecolor='none', alpha=0.30, zorder=1))
        offset = (rad + 0.2) * (1 if label_sign > 0 else -1)
        va = 'bottom' if label_sign > 0 else 'top'
        ax.text(cx, cy + offset, label, ha='center', va=va, fontsize=9 if 'Cluster' in label and ('B' in label or 'C' in label) else 10, color='dimgray')
    
    # Calculate plot limits for text positioning
    all_ras = [data['ra'] for data in valid_galaxies.values()]
    all_decs = [data['dec'] for data in valid_galaxies.values()]
    plot_limits = (min(all_ras), max(all_ras), min(all_decs), max(all_decs))
    
    # Calculate non-overlapping text positions
    text_positions = calculate_text_positions(valid_galaxies, plot_limits)
    
    # Prepare base values for vertical vectors
    all_ras = [data['ra'] for data in valid_galaxies.values()]
    all_decs = [data['dec'] for data in valid_galaxies.values()]
    plot_limits = (min(all_ras), max(all_ras), min(all_decs), max(all_decs))
    text_positions = calculate_text_positions(valid_galaxies, plot_limits)

    dec_min, dec_max = min(all_decs), max(all_decs)
    dec_range = max(dec_max - dec_min, 1e-6)
    # Define a reference vector length: 0.06 of DEC range corresponds to 0.10 dex/Re
    units_per_0p10_dec = 0.06 * dec_range

    # Plot galaxies with gradient data
    for galaxy_name, coords in valid_galaxies.items():
        ra = coords['ra']
        dec = coords['dec']
        galaxy_results = galaxy_gradients[galaxy_name]
        best_gradient, method_used = get_best_gradient(galaxy_results)
        
        if best_gradient:
            slope = best_gradient['slope']
            slope_error = best_gradient['slope_error']
            # Marker based on gradient direction; color by Δv
            marker = '^' if slope >= 0 else 'v'
            marker_size = 260
            norm_val = 0.5 + 0.5 * (dv_gal.get(galaxy_name, 0.0) / vmax) if vmax > 0 else 0.5
            norm_val = min(max(norm_val, 0.0), 1.0)
            color = cmap(norm_val)
            # Emission-based fill vs hollow
            has_em = emission_flags.get(_norm_name(galaxy_name), False)
            if has_em:
                ax.scatter(ra, dec, s=marker_size, marker=marker,
                           facecolors=color, edgecolors='black', linewidth=2,
                           zorder=5, alpha=uniform_alpha)
            else:
                ax.scatter(ra, dec, s=marker_size, marker=marker,
                           facecolors='none', edgecolors=color, linewidth=2,
                           zorder=5, alpha=uniform_alpha)

            # Add vertical vector indicating |slope| magnitude starting at the triangle center
            try:
                mag = abs(float(slope))
                # Compute axis-scale-aware length
                L = (mag / 0.10) * units_per_0p10_dec
                # Clamp length to a reasonable range
                L = float(np.clip(L, 0.02 * dec_range, 0.10 * dec_range))
                sign = 1.0 if slope >= 0 else -1.0
                # Start arrow at the data point (triangle center)
                y0 = dec
                dx, dy = 0.0, sign * L
                # Draw a white underlay then colored arrow for contrast
                ax.quiver(ra, y0, dx, dy, angles='xy', scale_units='xy', scale=1,
                          width=0.004, headwidth=6, headlength=8, headaxislength=7,
                          color='white', zorder=6, alpha=uniform_alpha)
                ax.quiver(ra, y0, dx, dy, angles='xy', scale_units='xy', scale=1,
                          width=0.0026, headwidth=6, headlength=8, headaxislength=7,
                          color=color, zorder=7, alpha=uniform_alpha)
                # Value label near arrow tip
                ty = y0 + dy + sign * (0.02 * dec_range)
                ax.text(ra, ty, f"{slope:+.2f}", ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none'))
            except Exception:
                pass
            
            # Galaxy name above the arrow
            name_y = text_positions[galaxy_name]['name_y']
            ax.text(ra, name_y, galaxy_name, ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            
        else:
            # No valid data - use gray
            ax.scatter(ra, dec, s=200, marker='o', c='lightgray', 
                      edgecolors='gray', linewidth=1, alpha=0.5, zorder=3)
            
            name_y = text_positions[galaxy_name]['name_y']
            ax.text(ra, name_y, galaxy_name, ha='center', va='center', 
                   fontsize=9, color='gray',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Right Ascension (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Declination (degrees)', fontsize=14, fontweight='bold')
    ax.set_title('Virgo Cluster Galaxies: [α/Fe] vs. Radius Relationship (IFU Observations)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set equal aspect ratio and proper limits
    ra_min, ra_max = min(all_ras), max(all_ras)
    dec_min, dec_max = min(all_decs), max(all_decs)
    
    # Calculate padding to maintain equal scaling
    ra_range = ra_max - ra_min
    dec_range = dec_max - dec_min
    padding = max(ra_range, dec_range) * 0.15
    
    ax.set_xlim(ra_max + padding, ra_min - padding)  # Inverted for astronomical convention
    ax.set_ylim(dec_min - padding, dec_max + padding)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    # Legend elements: emission fill and Δv
    from matplotlib.patches import Patch
    legend_emission = [
        Line2D([0],[0], marker='^', color='none', label='Emission present',
               markerfacecolor='black', markeredgecolor='black', markersize=9, lw=0),
        Line2D([0],[0], marker='^', color='none', label='No emission',
               markerfacecolor='none', markeredgecolor='black', markersize=9, lw=0)
    ]
    leg1 = ax.legend(handles=legend_emission, loc='upper right', frameon=True,
                     fontsize=9, title='Marker Fill (emission)')
    ax.add_artist(leg1)

    # Velocity legend (Δv) and colorbar
    sample_vals = [-0.8 * vmax, 0.0, 0.8 * vmax]
    vel_handles = [
        Patch(facecolor=cmap(0.5 + 0.5 * (val / vmax) if vmax > 0 else 0.5), edgecolor='none',
              label=('Δv < 0' if i == 0 else ('Δv ≈ 0' if i == 1 else 'Δv > 0')))
        for i, val in enumerate(sample_vals)
    ]
    ax.legend(handles=vel_handles, loc='upper right', bbox_to_anchor=(1, 0.78),
              frameon=True, fontsize=9, title='Velocity (Δv)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Velocity Δv (km/s)")
    try:
        cbar.set_ticks([-vmax, 0, vmax])
        cbar.set_ticklabels([f"{-vmax:.0f}", "0", f"{vmax:.0f}"])
    except Exception:
        pass

    # Add Δv=0 reference near the scale bar
    try:
        # Recompute positions used for the scale bar
        ra_min, ra_max = min(all_ras), max(all_ras)
        dec_min, dec_max = min(all_decs), max(all_decs)
        ra_range = ra_max - ra_min
        dec_range = dec_max - dec_min
        margin = 0.06 * ra_range
        x_start = ra_max - margin - 1.0
        x_end = ra_max - margin
        y = dec_min + 0.08 * dec_range
        # Place Δv=0 text just below the scale bar area
        ax.text(x_start, y - 0.30, f"Δv = 0 at v = {v_mean:.0f} km/s",
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))
    except Exception:
        pass

    # Draw a 1° scale bar at bottom-left (match original code)
    try:
        ra_min, ra_max = min(all_ras), max(all_ras)
        dec_min, dec_max = min(all_decs), max(all_decs)
        ra_range = ra_max - ra_min
        dec_range = dec_max - dec_min
        # Place at ~6% inset from bottom-left (left side is high RA due to inverted axis)
        margin = 0.06 * ra_range
        x_start = ra_max - margin - 1.0
        x_end = ra_max - margin  # 1 degree bar to the right in data coords
        y = dec_min + 0.08 * dec_range
        ax.plot([x_start, x_end], [y, y], color='black', lw=3, solid_capstyle='butt', zorder=9)
        ax.text(x_end + 0.1, y + 0.15, '1° ≈ 0.29 Mpc', ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    except Exception:
        pass

    # No watermark per corrected styling
    
    plt.tight_layout()
    
    # Save the plot (FINAL_DELIVERABLES)
    outdir = 'FINAL_DELIVERABLES'
    os.makedirs(outdir, exist_ok=True)
    output_file = os.path.join(outdir, 'virgo_cluster_final_gradients.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Final Virgo cluster plot saved: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("VIRGO CLUSTER FINAL α/Fe GRADIENT ANALYSIS")
    print("="*70)
    print("Coordinates: Extracted from MUSE IFU FITS headers")
    # Note: No velocity colorbar in final styling
    print("Markers: Triangle up/down = gradient sign; filled = emission present; hollow = no emission")
    print("Arrows: Length ∝ |d[α/Fe]/d(R/Re)| magnitude")
    print("-"*70)
    
    for galaxy_name in sorted(valid_galaxies.keys()):
        if galaxy_name in galaxy_gradients:
            results = galaxy_gradients[galaxy_name]
            best_gradient, method_used = get_best_gradient(results)
            velocity = valid_galaxies[galaxy_name]['velocity']
            
            if best_gradient:
                slope = best_gradient['slope']
                error = best_gradient['slope_error']
                direction = "positive" if slope > 0 else "negative"
                
                print(f"{galaxy_name}: {slope:+.3f} ± {error:.3f} dex/Re ({method_used}) "
                      f"- {direction}, v={velocity} km/s")
            else:
                print(f"{galaxy_name}: No reliable measurement, v={velocity} km/s")
    
    print("="*70)

if __name__ == "__main__":
    create_virgo_cluster_final()

    # Also produce a companion figure with distance-vs-slope panels
    try:
        def _create_distance_panels(valid_galaxies, galaxy_gradients, major_galaxies):
            fig = plt.figure(figsize=(16, 20))
            gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 1, 1, 1])
            ax_map = fig.add_subplot(gs[0, :])
            ax_dist_m87 = fig.add_subplot(gs[1, 0])
            ax_dist_m49 = fig.add_subplot(gs[1, 1])
            ax_dist_m60 = fig.add_subplot(gs[2, 0])
            ax_dist_m86 = fig.add_subplot(gs[2, 1])
            ax_dist_center = fig.add_subplot(gs[3, :])

            # Prepare map limits and draw substructure ellipses (reuse logic)
            all_ras = [data['ra'] for data in valid_galaxies.values()]
            all_decs = [data['dec'] for data in valid_galaxies.values()]
            ra_min, ra_max = min(all_ras), max(all_ras)
            dec_min, dec_max = min(all_decs), max(all_decs)
            ra_range = ra_max - ra_min
            dec_range = dec_max - dec_min
            padding = max(ra_range, dec_range) * 0.15
            ax_map.set_xlim(ra_max + padding, ra_min - padding)
            ax_map.set_ylim(dec_min - padding, dec_max + padding)
            ax_map.set_aspect('equal')
            ax_map.grid(True, alpha=0.3, linestyle='--')
            ax_map.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax_map.yaxis.set_minor_locator(AutoMinorLocator(5))

            # Match main figure styling: velocity-based coloring, emission fill, arrows start at datapoints
            # Setup Δv colormap
            cmap = plt.get_cmap('cool')
            uniform_alpha = 0.95
            # Compute v_mean and Δv for each plotted galaxy using provided velocities
            vel_arr = np.array([g['velocity'] for g in valid_galaxies.values()], dtype=float)
            v_mean = float(np.nanmean(vel_arr)) if vel_arr.size else 0.0
            dv_gal = {name: (g['velocity'] - v_mean) for name, g in valid_galaxies.items()}
            vmax = max(abs(v) for v in dv_gal.values()) if dv_gal else 1.0
            vmax = max(vmax, 1.0)

            # Major positions
            m87_ra, m87_dec = major_galaxies['M87']['ra'], major_galaxies['M87']['dec']
            m60_ra, m60_dec = major_galaxies['M60']['ra'], major_galaxies['M60']['dec']
            m49_ra, m49_dec = major_galaxies['M49']['ra'], major_galaxies['M49']['dec']
            m86_ra, m86_dec = major_galaxies['M86']['ra'], major_galaxies['M86']['dec']
            # Substructure radii
            r_m87 = 1.55
            r_m86 = 0.95
            r_m60 = 0.85
            r_m49 = 1.15

            # Fill substructures by mean Δv of enclosed galaxies
            sub_defs = [
                ("M87/Cluster A", (m87_ra, m87_dec), r_m87, -1),
                ("M86/Cluster C", (m86_ra + 0.4, m86_dec), r_m86, +1),
                ("M60/W Cloud",   (m60_ra, m60_dec), r_m60, -1),
                ("M49/Cluster B", (m49_ra + 0.2, m49_dec - 0.2), r_m49, -1),
            ]
            for label, (cx, cy), rad, label_sign in sub_defs:
                enclosed = []
                for gal_name, pos in valid_galaxies.items():
                    dx = pos['ra'] - cx
                    dy = pos['dec'] - cy
                    if dx*dx + dy*dy <= rad*rad:
                        enclosed.append(dv_gal.get(gal_name, 0.0))
                mean_dv = np.nanmean(enclosed) if enclosed else 0.0
                norm_val = 0.5 + 0.5 * (mean_dv / vmax) if vmax > 0 else 0.5
                norm_val = min(max(norm_val, 0.0), 1.0)
                fill_color = cmap(norm_val)
                ax_map.add_patch(Circle((cx, cy), radius=rad, facecolor=fill_color, edgecolor='none', alpha=0.30, zorder=1))
                offset = (rad + 0.2) * (1 if label_sign > 0 else -1)
                va = 'bottom' if label_sign > 0 else 'top'
                ax_map.text(cx, cy + offset, label, ha='center', va=va, fontsize=9 if 'Cluster' in label and ('B' in label or 'C' in label) else 10, color='dimgray')

            # Units for arrows
            dec_range = max(dec_range, 1e-6)
            units_per_0p10_dec = 0.06 * dec_range

            # Dataset center (for distance panels)
            data_center_ra = float(np.mean(all_ras))
            data_center_dec = float(np.mean(all_decs))

            # Distance collectors
            panels = {
                'M87': ax_dist_m87,
                'M49': ax_dist_m49,
                'M60': ax_dist_m60,
                'M86': ax_dist_m86,
                'Center': ax_dist_center,
            }
            dist_data = {k: {'x': [], 'y': [], 'c': [], 's': []} for k in panels}

            # Emission flags
            emission_flags = get_emission_flags()

            # Plot markers and arrows on the map and fill distance data (match main style)
            for galaxy_name, coords in valid_galaxies.items():
                ra = coords['ra']
                dec = coords['dec']
                results = galaxy_gradients.get(galaxy_name)
                if not results:
                    continue
                best_gradient, _ = get_best_gradient(results)
                if not best_gradient:
                    continue
                slope = float(best_gradient['slope'])
                # Marker style based on gradient sign and emission; color by Δv
                marker = '^' if slope >= 0 else 'v'
                norm_val = 0.5 + 0.5 * (dv_gal.get(galaxy_name, 0.0) / vmax) if vmax > 0 else 0.5
                norm_val = min(max(norm_val, 0.0), 1.0)
                color = cmap(norm_val)
                has_em = emission_flags.get(galaxy_name, False)
                if has_em:
                    ax_map.scatter(ra, dec, s=220, marker=marker, facecolors=color,
                                   edgecolors='black', linewidth=2, zorder=5, alpha=uniform_alpha)
                else:
                    ax_map.scatter(ra, dec, s=220, marker=marker, facecolors='none',
                                   edgecolors=color, linewidth=2, zorder=5, alpha=uniform_alpha)

                # Arrow starting at datapoint center, colored by Δv with white underlay
                L = (abs(slope) / 0.10) * units_per_0p10_dec
                L = float(np.clip(L, 0.02 * dec_range, 0.10 * dec_range))
                sign = 1.0 if slope >= 0 else -1.0
                y0 = dec
                dx, dy = 0.0, sign * L
                ax_map.quiver(ra, y0, dx, dy, angles='xy', scale_units='xy', scale=1,
                              width=0.004, headwidth=6, headlength=8, headaxislength=7,
                              color='white', zorder=6, alpha=uniform_alpha)
                ax_map.quiver(ra, y0, dx, dy, angles='xy', scale_units='xy', scale=1,
                              width=0.0026, headwidth=6, headlength=8, headaxislength=7,
                              color=color, zorder=7, alpha=uniform_alpha)

                # Label near tip, colored by Δv
                ty = y0 + dy + sign * (0.02 * dec_range)
                ax_map.text(ra, ty, f"{slope:+.2f}", ha='center', va='center', fontsize=9,
                            fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none'))

                # Distances
                def ang_dist(r1, d1, r2, d2):
                    return float(np.sqrt((r1 - r2)**2 + (d1 - d2)**2))

                dists = {
                    'M87': ang_dist(ra, dec, m87_ra, m87_dec),
                    'M49': ang_dist(ra, dec, m49_ra, m49_dec),
                    'M60': ang_dist(ra, dec, m60_ra, m60_dec),
                    'M86': ang_dist(ra, dec, m86_ra, m86_dec),
                    'Center': ang_dist(ra, dec, data_center_ra, data_center_dec),
                }

                for k in panels:
                    dist_data[k]['x'].append(dists[k])
                    dist_data[k]['y'].append(slope)
                    dist_data[k]['c'].append(color)
                    dist_data[k]['s'].append(70)

            # Map extras: major galaxy stars colored by Δv like main plot
            major_velocities = get_major_velocities()
            for name, data in major_galaxies.items():
                dvm = major_velocities.get(name, v_mean) - v_mean
                norm_val = 0.5 + 0.5 * (dvm / vmax)
                norm_val = min(max(norm_val, 0.0), 1.0)
                star_color = cmap(norm_val)
                ax_map.scatter(data['ra'], data['dec'], s=400, marker='*', facecolors=star_color,
                               edgecolors='black', linewidth=1.5, zorder=8, alpha=uniform_alpha)
                ax_map.text(data['ra'], data['dec'] - 0.18, name, fontsize=11, ha='center', va='top',
                            fontweight='bold', alpha=uniform_alpha)

            # Scale bar and Δv=0 annotation (match placement)
            margin = 0.06 * ra_range
            x_start = ra_max - margin - 1.0
            x_end = ra_max - margin
            y = dec_min + 0.08 * dec_range
            ax_map.plot([x_start, x_end], [y, y], color='black', lw=3, solid_capstyle='butt', zorder=9)
            ax_map.text(x_end + 0.1, y + 0.15, '1° ≈ 0.29 Mpc', ha='left', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            # Δv=0 reference near the scale bar
            try:
                ax_map.text(x_start, y - 0.30, f"Δv = 0 at v = {v_mean:.0f} km/s",
                            ha='left', va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))
            except Exception:
                pass

            # Velocity legend and colorbar on inset map
            from matplotlib.patches import Patch
            sample_vals = [-0.8 * vmax, 0.0, 0.8 * vmax]
            vel_handles = [
                Patch(facecolor=cmap(0.5 + 0.5 * (val / vmax) if vmax > 0 else 0.5), edgecolor='none',
                      label=('Δv < 0' if i == 0 else ('Δv ≈ 0' if i == 1 else 'Δv > 0')))
                for i, val in enumerate(sample_vals)
            ]
            ax_map.legend(handles=vel_handles, loc='upper right', frameon=True, fontsize=8, title='Velocity (Δv)')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_map, fraction=0.046, pad=0.04)
            cbar.set_label("Relative Velocity Δv (km/s)")
            try:
                cbar.set_ticks([-vmax, 0, vmax])
                cbar.set_ticklabels([f"{-vmax:.0f}", "0", f"{vmax:.0f}"])
            except Exception:
                pass

            ax_map.set_xlabel('Right Ascension (degrees)')
            ax_map.set_ylabel('Declination (degrees)')
            ax_map.set_title('Virgo Cluster Galaxies: [α/Fe] vs. Radius Relationship (IFU Observations)')

            # Build distance panels
            for key, axp in panels.items():
                dd = dist_data[key]
                if dd['x']:
                    axp.scatter(dd['x'], dd['y'], s=dd['s'], c=dd['c'], edgecolors='black', alpha=0.9)
                    # horizontal line at zero
                    axp.axhline(0, color='black', linestyle='--', alpha=0.5)
                    # simple linear trend using numpy
                    try:
                        x = np.array(dd['x'])
                        yv = np.array(dd['y'])
                        if len(x) >= 3:
                            m, b = np.polyfit(x, yv, 1)
                            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                            ys = m * xs + b
                            axp.plot(xs, ys, 'k-', alpha=0.7)
                            axp.text(0.02, 0.05, f"slope={m:.3f}", transform=axp.transAxes,
                                     bbox=dict(facecolor='white', alpha=0.7))
                    except Exception:
                        pass

                axp.set_xlabel('Angular Distance (deg)')
                axp.set_ylabel('α/Fe radial slope')
                axp.grid(True, alpha=0.3, linestyle='--')
                axp.xaxis.set_minor_locator(AutoMinorLocator(5))
                axp.yaxis.set_minor_locator(AutoMinorLocator(5))
                # symmetric y-lims
                if dd['y']:
                    ymax = max(abs(float(np.min(dd['y']))), abs(float(np.max(dd['y'])))) * 1.2
                else:
                    ymax = 0.3
                axp.set_ylim(-ymax, ymax)
                axp.tick_params(axis='both', which='both', labelsize='small', right=True, top=True, direction='in')

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            outdir = 'FINAL_DELIVERABLES'
            os.makedirs(outdir, exist_ok=True)
            output_file = os.path.join(outdir, 'virgo_cluster_final_with_panels.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved companion figure with panels: {output_file}")

        # Reconstruct required inputs
        coords = extract_ifu_coordinates()
        if not coords:
            coords = get_fallback_coordinates()
        galaxy_gradients = load_final_gradient_data()
        galaxy_velocities = get_galaxy_velocities()
        valid_galaxies = {}
        for name in coords.keys():
            if name in galaxy_gradients and name in galaxy_velocities:
                valid_galaxies[name] = {
                    'ra': coords[name]['ra'],
                    'dec': coords[name]['dec'],
                    'velocity': galaxy_velocities[name]
                }

        major_galaxies = {
            'M87': {'ra': 187.70591, 'dec': 12.39112, 'type': 'center'},
            'M86': {'ra': 186.54958, 'dec': 12.94694, 'type': 'major'},
            'M60': {'ra': 190.9162, 'dec': 11.5522, 'type': 'major'},
            'M49': {'ra': 187.4441, 'dec': 8.0035, 'type': 'major'}
        }

        _create_distance_panels(valid_galaxies, galaxy_gradients, major_galaxies)
    except Exception as _e:
        logger.warning(f"Could not create companion distance panels: {_e}")
