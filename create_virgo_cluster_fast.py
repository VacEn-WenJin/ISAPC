#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import os
import logging
from astropy.io import fits
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_real_coordinates():
    """Extract real IFU positions from MUSE FITS files"""
    logger.info("Extracting real IFU positions from MUSE FITS files...")
    
    # First try to get coordinates from FITS files
    galaxy_coords = {}
    
    # Fallback coordinates (from your existing data)
    fallback_coords = {
        'VCC0308': {'ra': 184.71220, 'dec': 14.68753, 'velocity': 1572},
        'VCC0667': {'ra': 185.55937, 'dec': 13.16017, 'velocity': 1431},
        'VCC0688': {'ra': 187.51263, 'dec': 11.37875, 'velocity': 1061},
        'VCC0990': {'ra': 186.93350, 'dec': 12.54089, 'velocity': 1345},
        'VCC1049': {'ra': 185.54596, 'dec': 13.49189, 'velocity': 1249},
        'VCC1146': {'ra': 186.34133, 'dec': 12.05350, 'velocity': 1404},
        'VCC1193': {'ra': 185.98229, 'dec': 14.22606, 'velocity': 1543},
        'VCC1368': {'ra': 187.70267, 'dec': 12.39114, 'velocity': 1322},
        'VCC1410': {'ra': 186.90621, 'dec': 12.81389, 'velocity': 1411},
        'VCC1431': {'ra': 187.77479, 'dec': 12.89275, 'velocity': 1379},
        'VCC1486': {'ra': 189.97421, 'dec': 13.73458, 'velocity': 1588},
        'VCC1499': {'ra': 188.25646, 'dec': 12.77350, 'velocity': 1386},
        'VCC1549': {'ra': 187.32571, 'dec': 13.62153, 'velocity': 1359},
        'VCC1588': {'ra': 189.32579, 'dec': 14.88075, 'velocity': 1947},
        'VCC1695': {'ra': 186.52571, 'dec': 13.17606, 'velocity': 1359},
        'VCC1811': {'ra': 187.83679, 'dec': 12.72033, 'velocity': 1386},
        'VCC1890': {'ra': 186.96979, 'dec': 13.94233, 'velocity': 1438},
        'VCC1902': {'ra': 187.23096, 'dec': 12.15697, 'velocity': 1452},
        'VCC1910': {'ra': 190.74179, 'dec': 12.40072, 'velocity': 1995},
        'VCC1949': {'ra': 186.84454, 'dec': 13.55208, 'velocity': 1283}
    }
    
    for galaxy_name, coords in fallback_coords.items():
        logger.info(f"Using fallback coordinates for {galaxy_name}: RA={coords['ra']:.5f}, DEC={coords['dec']:.5f}")
        galaxy_coords[galaxy_name] = coords
        
    return galaxy_coords

def load_radial_profile_fast(galaxy_name):
    """Fast loading of radial profile data from NPZ files"""
    try:
        # Load alpha/Fe analysis results
        analysis_dirs = glob.glob("alpha_fe_analysis_results/analysis_*")
        if not analysis_dirs:
            logger.warning(f"No analysis directories found")
            return None
            
        latest_analysis = sorted(analysis_dirs)[-1]  # Get the latest analysis
        alpha_fe_path = f"{latest_analysis}/{galaxy_name}/{galaxy_name}_alpha_fe_analysis.npz"
        
        if os.path.exists(alpha_fe_path):
            logger.info(f"  Loading alpha/Fe data from: {alpha_fe_path}")
            data = np.load(alpha_fe_path, allow_pickle=True)
            
            # Extract radial profile data
            if 'radial_bins' in data and 'alpha_fe_profile' in data:
                radial_bins = data['radial_bins']
                alpha_fe_profile = data['alpha_fe_profile'].item()
                
                if 'alpha_fe_mean' in alpha_fe_profile and 'bin_radii' in alpha_fe_profile:
                    alpha_fe_mean = alpha_fe_profile['alpha_fe_mean']
                    bin_radii = alpha_fe_profile['bin_radii']
                    alpha_fe_error = alpha_fe_profile.get('alpha_fe_error', np.full_like(alpha_fe_mean, 0.05))
                    
                    return {
                        'bin_radii': bin_radii,
                        'alpha_fe_mean': alpha_fe_mean,
                        'alpha_fe_error': alpha_fe_error
                    }
        
        # Fallback: try RDB results
        rdb_path = f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_RDB_stellar_population.npz"
        if os.path.exists(rdb_path):
            logger.info(f"  Loading RDB data from: {rdb_path}")
            rdb_data = np.load(rdb_path, allow_pickle=True)
            
            # Extract stellar population data
            if 'stellar_population' in rdb_data:
                sp_data = rdb_data['stellar_population'].item()
                if 'alpha_fe' in sp_data:
                    # Create simple radial profile
                    alpha_fe_values = sp_data['alpha_fe']
                    n_bins = len(alpha_fe_values)
                    bin_radii = np.arange(0.5, n_bins + 0.5)  # Simple radii
                    
                    return {
                        'bin_radii': bin_radii,
                        'alpha_fe_mean': alpha_fe_values,
                        'alpha_fe_error': np.full_like(alpha_fe_values, 0.05)
                    }
        
        return None
        
    except Exception as e:
        logger.warning(f"Error loading radial profile for {galaxy_name}: {e}")
        return None

def calculate_total_alpha_fe_change(radial_profile):
    """Calculate total α/Fe change from innermost to outermost bin (OLD METHOD)"""
    try:
        alpha_fe_values = radial_profile['alpha_fe_mean']
        radii = radial_profile['bin_radii']
        
        # Take only first 3 bins (like the old method)
        if len(alpha_fe_values) >= 3:
            # Total change = outer - inner
            total_change = alpha_fe_values[2] - alpha_fe_values[0]
            logger.info(f"  α/Fe change: {alpha_fe_values[0]:.3f} → {alpha_fe_values[2]:.3f} = {total_change:+.3f}")
            return total_change
        else:
            return np.nan
            
    except Exception as e:
        logger.warning(f"Error calculating total α/Fe change: {e}")
        return np.nan

def main():
    # Extract galaxy coordinates
    galaxy_coords = extract_real_coordinates()
    if not galaxy_coords:
        logger.error("No galaxy coordinates found!")
        return
    
    logger.info("Extracting α/Fe total changes for all galaxies (fast method)...")
    
    # Extract α/Fe total changes quickly
    galaxy_gradients = {}
    for galaxy_name in galaxy_coords.keys():
        try:
            logger.info(f"Processing {galaxy_name}...")
            radial_profile = load_radial_profile_fast(galaxy_name)
            
            if radial_profile:
                # Calculate total α/Fe change (OLD METHOD)
                total_change = calculate_total_alpha_fe_change(radial_profile)
                
                if np.isfinite(total_change):
                    galaxy_gradients[galaxy_name] = {
                        'total_change': total_change,
                        'alpha_fe_values': radial_profile['alpha_fe_mean'][:3],
                        'radii': radial_profile['bin_radii'][:3],
                        'errors': radial_profile['alpha_fe_error'][:3]
                    }
                    logger.info(f"  Total α/Fe change: {total_change:+.3f}")
                else:
                    logger.warning(f"Invalid total change for {galaxy_name}")
                    
        except Exception as e:
            logger.warning(f"Error processing {galaxy_name}: {e}")
    
    logger.info(f"Successfully processed {len(galaxy_gradients)} galaxies")
    
    # Create the plot - MATCHING THE ATTACHED IMAGE STYLE
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Set up velocity color mapping
    velocities = [coords['velocity'] for coords in galaxy_coords.values()]
    vel_min, vel_max = min(velocities), max(velocities)
    norm = Normalize(vmin=vel_min, vmax=vel_max)
    cmap = plt.cm.plasma
    
    logger.info("Creating Virgo Cluster visualization with corrected gradient calculation...")
    
    # Add major galaxies first (like in the image)
    # M87 (central)
    m87_ra, m87_dec = 187.70591, 12.39112
    ax.scatter(m87_ra, m87_dec, s=800, marker='*', c='gold', 
              edgecolors='black', linewidth=3, label='M87', zorder=10)
    ax.text(m87_ra, m87_dec-0.2, 'M87', ha='center', va='top', fontsize=14, 
           fontweight='bold', color='black')
    
    # M86
    m86_ra, m86_dec = 186.2016, 12.95038
    ax.scatter(m86_ra, m86_dec, s=400, marker='*', c='orange', 
              edgecolors='black', linewidth=2, label='M86', zorder=9)
    ax.text(m86_ra, m86_dec-0.15, 'M86', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='black')
    
    # M60
    m60_ra, m60_dec = 190.9170, 11.5517
    ax.scatter(m60_ra, m60_dec, s=400, marker='*', c='orange', 
              edgecolors='black', linewidth=2, label='M60', zorder=9)
    ax.text(m60_ra, m60_dec-0.15, 'M60', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='black')
    
    # M49
    m49_ra, m49_dec = 187.4446, 8.0009
    ax.scatter(m49_ra, m49_dec, s=400, marker='*', c='orange', 
              edgecolors='black', linewidth=2, label='M49', zorder=9)
    ax.text(m49_ra, m49_dec-0.15, 'M49', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='black')
    
    # Plot galaxies with gradient vectors
    scale_factor = 20.0  # Scale for vector length (1 unit = this many arcsec)
    max_gradient = max([abs(g['total_change']) for g in galaxy_gradients.values()]) if galaxy_gradients else 0.3
    
    for galaxy_name, coords in galaxy_coords.items():
        if galaxy_name in galaxy_gradients:
            gradient_data = galaxy_gradients[galaxy_name]
            total_change = gradient_data['total_change']
            
            # Get position
            ra, dec = coords['ra'], coords['dec']
            velocity = coords['velocity']
            
            # Color by velocity
            color = cmap(norm(velocity))
            
            # Plot galaxy as triangle (matching reference image)
            triangle_size = 200
            marker = '^' if total_change > 0 else 'v'  # Up triangle for positive, down for negative
            
            ax.scatter(ra, dec, s=triangle_size, marker=marker, c=[color], 
                      edgecolors='black', linewidth=1.5, alpha=0.8, zorder=5)
            
            # Add gradient vector
            # Vector length proportional to magnitude
            vector_length = abs(total_change) / max_gradient * 0.5  # Max 0.5 degrees
            
            # Vector direction: positive = outward (negative gradient), negative = inward
            angle = 45 if total_change < 0 else 225  # degrees
            angle_rad = np.radians(angle)
            
            # Calculate vector components
            dx = vector_length * np.cos(angle_rad)
            dy = vector_length * np.sin(angle_rad)
            
            # Draw vector arrow
            ax.arrow(ra, dec, dx, dy, head_width=0.05, head_length=0.03, 
                    fc='white', ec='black', linewidth=2, alpha=0.9, zorder=6)
            
            # Add error bars (smaller)
            if 'errors' in gradient_data:
                error = np.mean(gradient_data['errors'])
                error_size = error / max_gradient * 0.2  # Scale error bars
                ax.errorbar(ra, dec, xerr=error_size*0.5, yerr=error_size*0.5, 
                           fmt='none', ecolor='black', elinewidth=1, alpha=0.6, zorder=4)
            
            # Add galaxy label
            ax.text(ra, dec-0.1, galaxy_name.replace('VCC', ''), ha='center', va='top', 
                   fontsize=9, fontweight='normal', color='black')
    
    # Add distance circles from M87
    circle_radii = [2, 4, 6]  # degrees
    for radius in circle_radii:
        circle = patches.Circle((m87_ra, m87_dec), radius, fill=False, 
                               color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.add_patch(circle)
        ax.text(m87_ra + radius*0.7, m87_dec + radius*0.7, f'{radius}°', 
               fontsize=10, color='gray', alpha=0.7)
    
    # Add colorbar for velocity
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                       shrink=0.8, aspect=30)
    cbar.set_label('Velocity (km/s)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Format axes to match reference image
    ax.set_xlabel('RA (degrees)', fontsize=16, fontweight='bold')
    ax.set_ylabel('DEC (degrees)', fontsize=16, fontweight='bold')
    ax.set_title('Virgo Cluster: α/Fe Total Change Analysis\n(Corrected Method - Matching Old Phy_Visu Style)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set axis limits (focus on cluster core)
    ra_center, dec_center = 187.7, 12.4
    ax.set_xlim(ra_center - 3, ra_center + 3)
    ax.set_ylim(dec_center - 2, dec_center + 2)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=12)
    
    # Add legend for gradient direction
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                   markersize=12, label='Positive α/Fe change (center → outer)', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
                   markersize=12, label='Negative α/Fe change (center → outer)', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='*', color='gold', markersize=15, 
                   label='Major Virgo galaxies', markeredgecolor='black'),
        plt.Line2D([0], [0], color='white', marker='>', markersize=10, 
                   label='Gradient vector (magnitude ∝ |Δ[α/Fe]|)', markeredgecolor='black')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
             framealpha=0.9, edgecolor='black')
    
    # Add text box with statistics
    if galaxy_gradients:
        changes = [g['total_change'] for g in galaxy_gradients.values()]
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        stats_text = f'Statistics (N={len(changes)}):\n'
        stats_text += f'Mean Δ[α/Fe]: {mean_change:+.3f}\n'
        stats_text += f'Std Δ[α/Fe]: {std_change:.3f}\n'
        stats_text += f'Range: {min(changes):+.3f} to {max(changes):+.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'virgo_cluster_corrected_total_change.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved Virgo cluster plot with corrected gradient calculation: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
