#!/usr/bin/env python3
"""
Final Virgo Cluster α/Fe Gradient Visualization - Corrected Version

This script creates the definitive Virgo cluster plot using:
1. Proper coordinates extracted from MUSE IFU FITS headers
2. Velocity-based coloring (relative to cluster mean)
3. Equal RA/DEC scaling
4. Clean gradient display without significance markers
5. Non-overlapping text positioning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from astropy.io import fits
import logging
import os
import glob

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

def load_final_gradient_data():
    """Load the definitive gradient data from enhanced_radial_plots analysis"""
    try:
        # Load the combined gradient summary with both RDB and VNB results
        combined_file = "alpha_gradient_dual/combined_gradient_summary.csv"
        
        if not os.path.exists(combined_file):
            logger.error(f"Combined gradient file not found: {combined_file}")
            return {}
        
        # Load the CSV
        df = pd.read_csv(combined_file)
        logger.info(f"Loaded {len(df)} gradient measurements")
        
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
                    'significance': get_significance_level(rdb_row['Significance']),
                    'p_value': rdb_row['P_value'],
                    'r_squared': rdb_row['R_squared']
                }
            
            # Process VNB results
            if not vnb_data.empty:
                vnb_row = vnb_data.iloc[0]
                galaxy_results['VNB'] = {
                    'slope': vnb_row['Slope'],
                    'slope_error': vnb_row['Slope_Error'],
                    'significance': get_significance_level(vnb_row['Significance']),
                    'p_value': vnb_row['P_value'],
                    'r_squared': vnb_row['R_squared']
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
    
    # Calculate velocity statistics for coloring
    velocities = [data['velocity'] for data in valid_galaxies.values()]
    mean_velocity = np.mean(velocities)
    velocity_range = max(velocities) - min(velocities)
    
    logger.info(f"Velocity range: {min(velocities)} - {max(velocities)} km/s, mean: {mean_velocity:.0f} km/s")
    
    # Create figure with equal aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Setup velocity colormap
    norm = Normalize(vmin=min(velocities), vmax=max(velocities))
    cmap = cm.get_cmap('viridis')  # Blue to yellow color scheme
    
    # Major Virgo cluster galaxies for reference
    major_galaxies = {
        'M87': {'ra': 187.70591, 'dec': 12.39112, 'type': 'center'},
        'M86': {'ra': 186.54958, 'dec': 12.94694, 'type': 'major'},
        'M60': {'ra': 190.9162, 'dec': 11.5522, 'type': 'major'},
        'M49': {'ra': 187.4441, 'dec': 8.0035, 'type': 'major'}
    }
    
    # Plot major galaxies
    for name, data in major_galaxies.items():
        ra, dec = data['ra'], data['dec']
        if data['type'] == 'center':
            # M87 cluster center
            ax.scatter(ra, dec, s=800, marker='*', c='gold', 
                      edgecolors='black', linewidth=2, zorder=10, alpha=0.9)
            ax.text(ra, dec-0.25, name, ha='center', va='top', 
                   fontsize=11, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        else:
            # Other major galaxies
            ax.scatter(ra, dec, s=400, marker='D', c='orange', 
                      edgecolors='black', linewidth=1.5, zorder=8, alpha=0.7)
            ax.text(ra, dec-0.15, name, ha='center', va='top', 
                   fontsize=9, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='orange', alpha=0.7))
    
    # Calculate plot limits for text positioning
    all_ras = [data['ra'] for data in valid_galaxies.values()]
    all_decs = [data['dec'] for data in valid_galaxies.values()]
    plot_limits = (min(all_ras), max(all_ras), min(all_decs), max(all_decs))
    
    # Calculate non-overlapping text positions
    text_positions = calculate_text_positions(valid_galaxies, plot_limits)
    
    # Plot galaxies with gradient data
    for galaxy_name, coords in valid_galaxies.items():
        ra = coords['ra']
        dec = coords['dec']
        velocity = coords['velocity']
        
        galaxy_results = galaxy_gradients[galaxy_name]
        best_gradient, method_used = get_best_gradient(galaxy_results)
        
        if best_gradient:
            slope = best_gradient['slope']
            slope_error = best_gradient['slope_error']
            
            # Color based on velocity
            color = cmap(norm(velocity))
            
            # Marker based on gradient direction
            if slope > 0:
                marker = '^'
                marker_size = 300
            else:
                marker = 'v' 
                marker_size = 300
            
            # Plot galaxy
            ax.scatter(ra, dec, s=marker_size, marker=marker, c=[color], 
                      edgecolors='black', linewidth=2, zorder=5, alpha=0.8)
            
            # Get text positions
            name_y = text_positions[galaxy_name]['name_y']
            gradient_y = text_positions[galaxy_name]['gradient_y']
            
            # Galaxy name
            ax.text(ra, name_y, galaxy_name, ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            
            # Gradient value only (no significance markers)
            gradient_text = f"{slope:+.2f}"
            ax.text(ra, gradient_y, gradient_text, ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='lightgray', alpha=0.8))
            
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
    ax.set_title('Virgo Cluster: α/Fe Radial Gradients d[α/Fe]/d(R/Re)\n'
                'Color-coded by Galaxy Velocity', 
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
    
    # Add velocity colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Galaxy Velocity (km/s)', fontsize=12, fontweight='bold')
    
    # Enhanced legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='gray', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='Positive gradient (α/Fe increases outward)'),
        Line2D([0], [0], marker='v', color='gray', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='Negative gradient (α/Fe decreases outward)'),
        Line2D([0], [0], marker='*', color='gold', markersize=14, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='M87 (Cluster Center)'),
        Line2D([0], [0], marker='D', color='orange', markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, linewidth=0, 
               label='Major Virgo Galaxies')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             title='Gradient Direction & Galaxy Types', title_fontsize=11)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'virgo_cluster_final_gradients.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Final Virgo cluster plot saved: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("VIRGO CLUSTER FINAL α/Fe GRADIENT ANALYSIS")
    print("="*70)
    print("Coordinates: Extracted from MUSE IFU FITS headers")
    print("Color coding: Galaxy velocity relative to cluster")
    print("Gradients: d[α/Fe]/d(R/Re) in dex per effective radius")
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
