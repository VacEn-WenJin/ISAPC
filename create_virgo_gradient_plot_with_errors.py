"""
Enhanced Virgo Cluster Galaxy Gradient Visualization with Error Bars
Shows galaxy positions with gradient vectors colored by velocity, including error bars
Based on the original Phy_Visu.py approach but with enhanced error visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from alpha_gradient_analysis import analyze_single_galaxy
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VirgoGradientVisuWithErrors')

def get_real_ifu_positions():
    """Extract real IFU positions from MUSE FITS files with fallback coordinates"""
    import os
    from astropy.io import fits
    
    positions = {}
    data_dir = "data"
    
    # Known velocities for each galaxy
    velocities = {
        'VCC0308': 1572, 'VCC0667': 1431, 'VCC0688': 1061, 'VCC0990': 1740,
        'VCC1049': 639, 'VCC1146': 700, 'VCC1193': 757, 'VCC1368': 1055,
        'VCC1410': 1615, 'VCC1431': 1521, 'VCC1486': 111, 'VCC1499': 1823,
        'VCC1549': 1245, 'VCC1588': 1318, 'VCC1695': 1156, 'VCC1811': 1628,
        'VCC1890': 1672, 'VCC1902': 1519, 'VCC1910': 1745, 'VCC1949': 1198
    }
    
    # Emission line flags
    has_emission = {
        'VCC0308': False, 'VCC0667': False, 'VCC0688': False, 'VCC0990': False,
        'VCC1049': False, 'VCC1146': False, 'VCC1193': False, 'VCC1368': True,
        'VCC1410': False, 'VCC1431': True, 'VCC1486': False, 'VCC1499': False,
        'VCC1549': False, 'VCC1588': False, 'VCC1695': True, 'VCC1811': False,
        'VCC1890': False, 'VCC1902': False, 'VCC1910': False, 'VCC1949': True
    }
    
    # Fallback coordinates if FITS files not found (from enhanced version)
    fallback_coords = {
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
    
    for galaxy_name in velocities.keys():
        try:
            # Try to extract from FITS file first
            fits_file = os.path.join(data_dir, f"{galaxy_name}.fits")
            ra, dec = None, None
            
            if os.path.exists(fits_file):
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header
                    ra = header.get('CRVAL1', None)
                    dec = header.get('CRVAL2', None)
                    
                    if ra is not None and dec is not None:
                        logger.info(f"Real FITS coordinates for {galaxy_name}: RA={ra:.5f}, DEC={dec:.5f}")
            
            # Use fallback coordinates if FITS not found or incomplete
            if ra is None or dec is None:
                if galaxy_name in fallback_coords:
                    ra = fallback_coords[galaxy_name]['ra']
                    dec = fallback_coords[galaxy_name]['dec']
                    logger.info(f"Using fallback coordinates for {galaxy_name}: RA={ra:.5f}, DEC={dec:.5f}")
            
            if ra is not None and dec is not None:
                positions[galaxy_name] = {
                    'ra': ra,
                    'dec': dec,
                    'velocity': velocities[galaxy_name],
                    'has_emission': has_emission[galaxy_name]
                }
                        
        except Exception as e:
            logger.warning(f"Could not extract coordinates for {galaxy_name}: {e}")
    
    return positions

def create_virgo_cluster_gradient_plot_with_errors():
    """Create Virgo Cluster visualization with gradient vectors and error bars - enhanced Phy_Visu style"""
    
    # Get real IFU positions from MUSE FITS files
    logger.info("Extracting real IFU positions from MUSE FITS files...")
    galaxy_coords = get_real_ifu_positions()
    
    logger.info("Extracting RDB gradients with errors for all galaxies...")
    
    # Extract RDB gradients with uncertainties (3 bins only)
    galaxy_gradients = {}
    for galaxy_name in galaxy_coords.keys():
        try:
            logger.info(f"Processing {galaxy_name}...")
            result = analyze_single_galaxy(galaxy_name)
            
            if result and result.get('analysis_success', False):
                # Extract multi-method gradient data
                multi_method = result.get('multi_method_analysis', {})
                if 'RDB_3bins' in multi_method:
                    rdb_data = multi_method['RDB_3bins']
                    slope = rdb_data.get('slope', np.nan)
                    slope_error = rdb_data.get('slope_error', np.nan)
                    
                    if np.isfinite(slope) and np.isfinite(slope_error):
                        galaxy_gradients[galaxy_name] = {
                            'slope': slope,
                            'slope_error': slope_error,
                            'n_bins': rdb_data.get('n_bins', 3)
                        }
                        logger.info(f"  RDB gradient: {slope:.4f} ± {slope_error:.4f}")
                        
        except Exception as e:
            logger.warning(f"Error processing {galaxy_name}: {e}")
    
    # Create the plot - enhanced Phy_Visu style with errors
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Set up velocity color mapping
    velocities = [coords['velocity'] for coords in galaxy_coords.values()]
    vel_min, vel_max = min(velocities), max(velocities)
    norm = Normalize(vmin=vel_min, vmax=vel_max)
    cmap = plt.cm.plasma  # Use plasma for better velocity distinction
    
    # Plot Virgo Cluster structure
    logger.info("Creating enhanced Virgo Cluster visualization with error bars...")
    
    # Add M87 and other major Virgo galaxies (cluster landmarks)
    # M87 (Virgo A) - central galaxy
    m87_ra, m87_dec = 187.70591, 12.39112
    ax.scatter(m87_ra, m87_dec, s=600, marker='*', c='red', 
              edgecolors='black', linewidth=2, label='M87', zorder=10)
    ax.text(m87_ra, m87_dec-0.15, 'M87', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='red')
    
    # M86 (NGC 4406)
    m86_ra, m86_dec = 186.54829, 12.95668
    ax.scatter(m86_ra, m86_dec, s=400, marker='D', c='orange', 
              edgecolors='black', linewidth=2, label='M86', zorder=10)
    ax.text(m86_ra, m86_dec-0.12, 'M86', ha='center', va='top', fontsize=10, 
           fontweight='bold', color='orange')
    
    # M60 (NGC 4649)
    m60_ra, m60_dec = 190.91684, 11.55217
    ax.scatter(m60_ra, m60_dec, s=400, marker='s', c='blue', 
              edgecolors='black', linewidth=2, label='M60', zorder=10)
    ax.text(m60_ra, m60_dec-0.12, 'M60', ha='center', va='top', fontsize=10, 
           fontweight='bold', color='blue')
    
    # M49 (NGC 4472)
    m49_ra, m49_dec = 187.44419, 8.00003
    ax.scatter(m49_ra, m49_dec, s=400, marker='h', c='green', 
              edgecolors='black', linewidth=2, label='M49', zorder=10)
    
    # Plot each galaxy with gradient vectors and error bars
    for galaxy_name, coords in galaxy_coords.items():
        ra = coords['ra']
        dec = coords['dec']
        velocity = coords['velocity']
        has_emission = coords['has_emission']
        
        # Get gradient data
        gradient_data = galaxy_gradients.get(galaxy_name)
        
        # Color by velocity
        color = cmap(norm(velocity))
        
        # Plot galaxy position - triangles for better distinction
        if has_emission:
            marker_style = '^'  # Upward triangle for emission
            facecolors = color
            size = 200
        else:
            marker_style = 'o'  # Circle for non-emission
            facecolors = 'none'
            size = 180
        
        ax.scatter(ra, dec, c=facecolors, s=size, marker=marker_style,
                  edgecolors='black', linewidth=2, 
                  alpha=0.9, zorder=5)
        
        # Add galaxy label
        ax.annotate(galaxy_name.replace('VCC', ''), (ra, dec),
                   xytext=(10, 10), textcoords='offset points', 
                   fontsize=10, alpha=0.9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Add gradient vector with error bars and direction (enhanced style)
        if gradient_data:
            slope = gradient_data['slope']
            slope_error = gradient_data['slope_error']
            
            # Vector properties
            vector_scale = 10.0  # Scale factor for vector length
            vector_length = abs(slope) * vector_scale
            base_length = 0.05  # Minimum vector length
            total_length = base_length + vector_length
            
            # Determine vector direction (always vertical as in enhanced version)
            if slope > 0:
                # Positive gradient - vector points up (decreasing α/Fe outward)
                vector_end_y = dec + total_length
                arrow_direction = 'up'
            else:
                # Negative gradient - vector points down (increasing α/Fe outward)
                vector_end_y = dec - total_length
                arrow_direction = 'down'
            
            # Draw main gradient vector with velocity color and thickness proportional to significance
            significance = abs(slope) / slope_error if slope_error > 0 else 1
            vector_width = min(6, max(2, significance * 2))  # Width based on significance
            
            # Main vector line
            ax.plot([ra, ra], [dec, vector_end_y], color=color, linewidth=vector_width, 
                   alpha=0.9, zorder=6, solid_capstyle='round')
            
            # Add directional arrowhead
            arrow_size = 0.015
            if arrow_direction == 'up':
                # Upward arrow
                arrow_x = [ra-arrow_size, ra, ra+arrow_size, ra]
                arrow_y = [vector_end_y-arrow_size, vector_end_y, vector_end_y-arrow_size, vector_end_y]
            else:
                # Downward arrow
                arrow_x = [ra-arrow_size, ra, ra+arrow_size, ra]
                arrow_y = [vector_end_y+arrow_size, vector_end_y, vector_end_y+arrow_size, vector_end_y]
            
            # Draw filled arrowhead with velocity color
            ax.fill(arrow_x, arrow_y, color=color, alpha=0.9, zorder=7, edgecolor='black', linewidth=0.5)
            
            # Add error bars as T-shaped caps
            error_length = slope_error * vector_scale
            if arrow_direction == 'up':
                error_top = vector_end_y + error_length
                error_bottom = vector_end_y - error_length
            else:
                error_top = vector_end_y - error_length  
                error_bottom = vector_end_y + error_length
            
            # Draw error bar structure
            cap_width = 0.01
            # Main error bar line
            ax.plot([ra, ra], [error_bottom, error_top], 
                   color='black', linewidth=1.5, alpha=0.7, zorder=8)
            # Top cap
            ax.plot([ra-cap_width, ra+cap_width], [error_top, error_top], 
                   color='black', linewidth=2, alpha=0.8, zorder=8)
            # Bottom cap
            ax.plot([ra-cap_width, ra+cap_width], [error_bottom, error_bottom], 
                   color='black', linewidth=2, alpha=0.8, zorder=8)
            
            # Add gradient value label with direction indicator
            label_offset = 0.025
            label_y = vector_end_y + (label_offset if arrow_direction == 'up' else -label_offset)
            direction_symbol = "↗" if slope > 0 else "↘"
            
            ax.text(ra, label_y, f'{slope:.3f}±{slope_error:.3f} {direction_symbol}', 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, 
                            edgecolor=color, linewidth=1))
    
    # Add Virgo Cluster structure lines/regions
    # Draw approximate cluster substructure
    virgo_a_ra = [186.5, 187.5, 188.0, 187.2, 186.5]
    virgo_a_dec = [12.2, 12.8, 13.2, 13.8, 12.2]
    ax.plot(virgo_a_ra, virgo_a_dec, 'k--', alpha=0.4, linewidth=2, label='Virgo A Subcluster')
    
    # Add distance circles from M87
    m87_ra, m87_dec = 187.70591, 12.39112
    circle_radii = [2.0, 4.0]  # degrees - larger for real field
    for radius in circle_radii:
        circle = plt.Circle((m87_ra, m87_dec), radius, fill=False, 
                          linestyle=':', color='gray', alpha=0.4, linewidth=1.5)
        ax.add_patch(circle)
    
    # Formatting
    ax.set_xlabel('Right Ascension (degrees)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Declination (degrees)', fontsize=16, fontweight='bold')
    ax.set_title('Virgo Cluster Galaxies: α/Fe Radial Gradients with Uncertainties\n(RDB Method, 3 Inner Bins)', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Set equal aspect ratio and appropriate limits for real coordinates
    ax.set_aspect('equal')
    
    # Calculate field boundaries from galaxy positions
    if galaxy_coords:
        ras = [coords['ra'] for coords in galaxy_coords.values()]
        decs = [coords['dec'] for coords in galaxy_coords.values()]
        ra_range = max(ras) - min(ras)
        dec_range = max(decs) - min(decs)
        
        # Add padding
        padding = 0.3
        ax.set_xlim(min(ras) - padding, max(ras) + padding)
        ax.set_ylim(min(decs) - padding, max(decs) + padding)
    else:
        # Fallback limits
        ax.set_xlim(184.0, 191.5)
        ax.set_ylim(7.5, 15.5)
    
    # Add grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Add colorbar for velocity
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                       fraction=0.046, pad=0.04, aspect=25)
    cbar.set_label('Recession Velocity (km/s)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Create enhanced legend
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
               markersize=14, fillstyle='full', markeredgecolor='black', 
               markeredgewidth=2, label='Emission Line Galaxy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markersize=14, fillstyle='none', markeredgecolor='black', 
               markeredgewidth=2, label='Non-Emission Galaxy'),
        Line2D([0], [0], marker='|', color='green', markersize=16, 
               markeredgewidth=4, label='Positive α/Fe Gradient'),
        Line2D([0], [0], marker='|', color='red', markersize=16, 
               markeredgewidth=4, label='Negative α/Fe Gradient'),
        Line2D([0], [0], marker='*', color='red', markersize=18, 
               markeredgecolor='black', markeredgewidth=2, label='M87 (Cluster Center)'),
        Line2D([0], [0], color='black', linewidth=2, label='Gradient Uncertainty')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
             frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    
    # Add text box with gradient statistics
    if galaxy_gradients:
        slopes = [data['slope'] for data in galaxy_gradients.values()]
        errors = [data['slope_error'] for data in galaxy_gradients.values()]
        n_positive = sum(1 for s in slopes if s > 0)
        n_negative = sum(1 for s in slopes if s < 0)
        mean_slope = np.mean(slopes)
        mean_error = np.mean(errors)
        
        stats_text = f'Gradient Statistics:\n' \
                    f'Total galaxies: {len(slopes)}\n' \
                    f'Positive gradients: {n_positive}\n' \
                    f'Negative gradients: {n_negative}\n' \
                    f'Mean slope: {mean_slope:.3f} ± {mean_error:.3f}\n' \
                    f'Vector scale: 1 unit = 0.125 gradient'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='lightblue', alpha=0.9), fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('./enhanced_radial_plots')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "virgo_cluster_gradients_with_errors.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Enhanced Virgo Cluster gradient plot with errors saved: {output_file}")
    logger.info(f"Processed {len(galaxy_gradients)} galaxies with valid gradients and uncertainties")
    
    # Print summary
    if galaxy_gradients:
        print("\n" + "="*70)
        print("VIRGO CLUSTER α/Fe GRADIENT SUMMARY WITH UNCERTAINTIES")
        print("="*70)
        for galaxy_name, data in galaxy_gradients.items():
            slope = data['slope']
            error = data['slope_error']
            direction = "↗" if slope > 0 else "↘"
            significance = abs(slope) / error if error > 0 else 0
            sig_level = "***" if significance > 3 else ("**" if significance > 2 else ("*" if significance > 1 else ""))
            print(f"{galaxy_name}: {slope:+.4f} ± {error:.4f} {direction} {sig_level}")
        print("="*70)
        print("Significance levels: *** > 3σ, ** > 2σ, * > 1σ")
        print("="*70)

if __name__ == "__main__":
    create_virgo_cluster_gradient_plot_with_errors()
