"""
Enhanced Galaxy Summary Plot with RA/DEC positions, gradient vectors, and velocity color coding
Similar to original Phy_Visu.py but focused on RDB (3 bins) and VNB gradients
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from alpha_gradient_analysis import analyze_single_galaxy
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GalaxyGradientSummary')

def load_galaxy_coordinates_and_properties():
    """Load galaxy coordinates and emission line properties"""
    # Hardcoded galaxy coordinates (approximate Virgo Cluster positions)
    # In practice, these should come from a catalog or database
    galaxy_coords = {
        'VCC0308': {'ra': 187.70, 'dec': 14.39, 'velocity': 1572, 'has_emission': False},
        'VCC0667': {'ra': 185.56, 'dec': 13.16, 'velocity': 1431, 'has_emission': False},
        'VCC0688': {'ra': 187.24, 'dec': 12.78, 'velocity': 1061, 'has_emission': False},
        'VCC0990': {'ra': 186.93, 'dec': 12.54, 'velocity': 1740, 'has_emission': False},
        'VCC1049': {'ra': 187.41, 'dec': 11.93, 'velocity': 639, 'has_emission': False},
        'VCC1146': {'ra': 186.74, 'dec': 12.71, 'velocity': 700, 'has_emission': False},
        'VCC1193': {'ra': 186.28, 'dec': 13.42, 'velocity': 757, 'has_emission': False},
        'VCC1368': {'ra': 187.85, 'dec': 12.36, 'velocity': 1055, 'has_emission': True},
        'VCC1410': {'ra': 186.91, 'dec': 12.81, 'velocity': 1615, 'has_emission': False},
        'VCC1431': {'ra': 187.78, 'dec': 12.89, 'velocity': 1521, 'has_emission': True},
        'VCC1486': {'ra': 187.95, 'dec': 12.44, 'velocity': 111, 'has_emission': False},
        'VCC1499': {'ra': 188.21, 'dec': 12.77, 'velocity': 1823, 'has_emission': False},
        'VCC1549': {'ra': 187.33, 'dec': 13.62, 'velocity': 1245, 'has_emission': False},
        'VCC1588': {'ra': 187.15, 'dec': 12.05, 'velocity': 1318, 'has_emission': False},
        'VCC1695': {'ra': 186.53, 'dec': 13.18, 'velocity': 1156, 'has_emission': True},
        'VCC1811': {'ra': 187.84, 'dec': 12.72, 'velocity': 1628, 'has_emission': False},
        'VCC1890': {'ra': 186.97, 'dec': 13.94, 'velocity': 1672, 'has_emission': False},
        'VCC1902': {'ra': 187.23, 'dec': 12.15, 'velocity': 1519, 'has_emission': False},
        'VCC1910': {'ra': 187.51, 'dec': 12.24, 'velocity': 1745, 'has_emission': False},
        'VCC1949': {'ra': 186.85, 'dec': 13.55, 'velocity': 1198, 'has_emission': True}
    }
    
    return galaxy_coords

def extract_gradient_data_limited_bins():
    """Extract gradient data for all galaxies with limited bins"""
    gradient_data = {}
    
    galaxies = [
        'VCC0308', 'VCC0667', 'VCC0688', 'VCC0990', 'VCC1049',
        'VCC1146', 'VCC1193', 'VCC1368', 'VCC1410', 'VCC1431',
        'VCC1486', 'VCC1499', 'VCC1549', 'VCC1588', 'VCC1695',
        'VCC1811', 'VCC1890', 'VCC1902', 'VCC1910', 'VCC1949'
    ]
    
    for galaxy_name in galaxies:
        logger.info(f"Processing gradient data for {galaxy_name}...")
        
        try:
            # Run analysis
            result = analyze_single_galaxy(galaxy_name)
            
            if result and result.get('analysis_success', False):
                radial_profile = result.get('radial_profile')
                vnb_profile = result.get('vnb_profile')
                multi_gradient_results = result.get('multi_gradient_results', {})
                
                galaxy_data = {'galaxy_name': galaxy_name}
                
                # Extract RDB data (limit to 3 bins)
                if radial_profile and 'bin_radii' in radial_profile:
                    radii = radial_profile['bin_radii']
                    alpha_fe = radial_profile['alpha_fe_mean']
                    
                    # Limit to first 3 bins
                    n_bins = min(3, len(radii))
                    if n_bins >= 2:  # Need at least 2 points for gradient
                        radii_rdb = radii[:n_bins].copy()
                        alpha_rdb = alpha_fe[:n_bins]
                        
                        # Set innermost bin to R=0
                        radii_rdb[0] = 0.0
                        
                        # Calculate gradient
                        if len(radii_rdb) >= 2:
                            valid_mask = np.isfinite(radii_rdb) & np.isfinite(alpha_rdb)
                            if np.sum(valid_mask) >= 2:
                                slope_rdb = np.polyfit(radii_rdb[valid_mask], alpha_rdb[valid_mask], 1)[0]
                                galaxy_data['rdb_slope'] = slope_rdb
                                galaxy_data['rdb_max_radius'] = np.max(radii_rdb[valid_mask])
                                logger.info(f"  RDB slope: {slope_rdb:.4f}")
                
                # Extract VNB data (limit to same radial range as RDB)
                if vnb_profile and 'bin_radii' in vnb_profile and 'rdb_max_radius' in galaxy_data:
                    radii_vnb = vnb_profile['bin_radii']
                    alpha_vnb = vnb_profile['alpha_fe_mean']
                    
                    # Filter VNB data to same radial range as RDB
                    max_radius = galaxy_data['rdb_max_radius']
                    range_mask = radii_vnb <= max_radius
                    
                    if np.sum(range_mask) >= 2:
                        radii_vnb_limited = radii_vnb[range_mask]
                        alpha_vnb_limited = alpha_vnb[range_mask]
                        
                        valid_mask = np.isfinite(radii_vnb_limited) & np.isfinite(alpha_vnb_limited)
                        if np.sum(valid_mask) >= 2:
                            slope_vnb = np.polyfit(radii_vnb_limited[valid_mask], alpha_vnb_limited[valid_mask], 1)[0]
                            galaxy_data['vnb_slope'] = slope_vnb
                            logger.info(f"  VNB slope: {slope_vnb:.4f}")
                
                gradient_data[galaxy_name] = galaxy_data
                
            else:
                logger.warning(f"Analysis failed for {galaxy_name}")
                
        except Exception as e:
            logger.error(f"Error processing {galaxy_name}: {e}")
    
    return gradient_data

def create_galaxy_summary_plot():
    """Create comprehensive galaxy summary plot like original Phy_Visu"""
    
    # Load data
    logger.info("Loading galaxy coordinates and properties...")
    galaxy_coords = load_galaxy_coordinates_and_properties()
    
    logger.info("Extracting gradient data...")
    gradient_data = extract_gradient_data_limited_bins()
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # RA/DEC positions with gradient vectors (left panel)
    ax1.set_aspect('equal')
    
    # Velocity color mapping
    velocities = [coords.get('velocity', 1000) for coords in galaxy_coords.values()]
    vel_min, vel_max = np.min(velocities), np.max(velocities)
    norm = Normalize(vmin=vel_min, vmax=vel_max)
    cmap = cm.get_cmap('viridis')
    
    # Plot each galaxy
    for galaxy_name in galaxy_coords.keys():
        coords = galaxy_coords[galaxy_name]
        grad_data = gradient_data.get(galaxy_name, {})
        
        ra = coords['ra']
        dec = coords['dec'] 
        velocity = coords['velocity']
        has_emission = coords['has_emission']
        
        # Color by velocity
        color = cmap(norm(velocity))
        
        # Symbol: solid for emission, hollow for non-emission
        if has_emission:
            marker_style = 'o'  # solid circle
            fillstyle = 'full'
            alpha = 0.8
        else:
            marker_style = 'o'  # hollow circle
            fillstyle = 'none'
            alpha = 0.9
        
        # Plot galaxy position
        ax1.scatter(ra, dec, c=[color], s=200, marker=marker_style, 
                   fillstyle=fillstyle, edgecolors='black', linewidth=2, alpha=alpha)
        
        # Add galaxy label
        ax1.annotate(galaxy_name.replace('VCC', ''), (ra, dec), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Add gradient vectors
        vector_length = 0.02  # Base length in RA/DEC units
        
        # RDB vector (red)
        if 'rdb_slope' in grad_data:
            slope_rdb = grad_data['rdb_slope']
            # Vector pointing up for positive gradient, down for negative
            dy_rdb = vector_length * np.sign(slope_rdb) * abs(slope_rdb) * 10  # Scale for visibility
            ax1.arrow(ra - 0.01, dec, 0, dy_rdb, head_width=0.005, head_length=0.002, 
                     fc='red', ec='red', alpha=0.8, linewidth=2)
        
        # VNB vector (blue)
        if 'vnb_slope' in grad_data:
            slope_vnb = grad_data['vnb_slope']
            dy_vnb = vector_length * np.sign(slope_vnb) * abs(slope_vnb) * 10
            ax1.arrow(ra + 0.01, dec, 0, dy_vnb, head_width=0.005, head_length=0.002, 
                     fc='blue', ec='blue', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('RA (degrees)', fontsize=14)
    ax1.set_ylabel('DEC (degrees)', fontsize=14)
    ax1.set_title('Galaxy Positions with α/Fe Gradient Vectors\\n(Red=RDB, Blue=VNB)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for velocity
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity (km/s)', fontsize=12)
    
    # Add legend for symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
               fillstyle='full', markeredgecolor='black', label='Emission Line Galaxy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=10, 
               fillstyle='none', markeredgecolor='black', markeredgewidth=2, label='Non-Emission Galaxy'),
        Line2D([0], [0], color='red', linewidth=3, label='RDB Gradient Vector'),
        Line2D([0], [0], color='blue', linewidth=3, label='VNB Gradient Vector')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Gradient comparison plot (right panel)
    rdb_slopes = []
    vnb_slopes = []
    galaxy_names = []
    colors_scatter = []
    symbols = []
    
    for galaxy_name, grad_data in gradient_data.items():
        if 'rdb_slope' in grad_data and 'vnb_slope' in grad_data:
            rdb_slopes.append(grad_data['rdb_slope'])
            vnb_slopes.append(grad_data['vnb_slope'])
            galaxy_names.append(galaxy_name)
            
            coords = galaxy_coords[galaxy_name]
            colors_scatter.append(cmap(norm(coords['velocity'])))
            symbols.append('o' if coords['has_emission'] else 's')
    
    # Plot RDB vs VNB comparison
    for i, (rdb, vnb, name, color, symbol) in enumerate(zip(rdb_slopes, vnb_slopes, galaxy_names, colors_scatter, symbols)):
        fillstyle = 'full' if galaxy_coords[name]['has_emission'] else 'none'
        ax2.scatter(rdb, vnb, c=[color], s=150, marker=symbol, 
                   fillstyle=fillstyle, edgecolors='black', linewidth=1.5, alpha=0.8)
        ax2.annotate(name.replace('VCC', ''), (rdb, vnb), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Add 1:1 line
    min_slope = min(min(rdb_slopes), min(vnb_slopes)) if rdb_slopes and vnb_slopes else -0.1
    max_slope = max(max(rdb_slopes), max(vnb_slopes)) if rdb_slopes and vnb_slopes else 0.1
    ax2.plot([min_slope, max_slope], [min_slope, max_slope], 'k--', alpha=0.5, label='1:1 Line')
    
    ax2.set_xlabel('RDB Gradient Slope (3 bins)', fontsize=14)
    ax2.set_ylabel('VNB Gradient Slope (same range)', fontsize=14)
    ax2.set_title('RDB vs VNB Gradient Comparison', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add correlation info
    if len(rdb_slopes) > 1 and len(vnb_slopes) > 1:
        correlation = np.corrcoef(rdb_slopes, vnb_slopes)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: r = {correlation:.3f}', 
                transform=ax2.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('./enhanced_radial_plots')
    output_file = output_dir / "galaxy_gradient_summary_with_vectors.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Galaxy summary plot saved: {output_file}")
    
    # Print summary statistics
    logger.info("\\n=== GRADIENT SUMMARY ===")
    logger.info(f"Total galaxies processed: {len(gradient_data)}")
    logger.info(f"Galaxies with both RDB and VNB gradients: {len(rdb_slopes)}")
    
    if rdb_slopes and vnb_slopes:
        logger.info(f"RDB slopes: mean = {np.mean(rdb_slopes):.4f}, std = {np.std(rdb_slopes):.4f}")
        logger.info(f"VNB slopes: mean = {np.mean(vnb_slopes):.4f}, std = {np.std(vnb_slopes):.4f}")
        logger.info(f"RDB-VNB correlation: r = {np.corrcoef(rdb_slopes, vnb_slopes)[0, 1]:.3f}")

if __name__ == "__main__":
    create_galaxy_summary_plot()
