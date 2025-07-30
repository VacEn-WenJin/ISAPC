"""
Quick Galaxy Summary Plot with RA/DEC positions, gradient vectors, and velocity color coding
Uses existing analysis results to avoid recomputation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('QuickGalaxySummary')

def get_galaxy_coordinates():
    """Get galaxy coordinates and emission line properties"""
    return {
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

def extract_gradients_from_existing_plots():
    """Extract gradient information from existing plot files"""
    # Hardcoded gradient slopes from recent analysis output
    # These would normally be extracted from analysis files
    gradient_data = {
        'VCC0308': {'rdb_slope': -0.0722, 'vnb_slope': 0.0040},
        'VCC0667': {'rdb_slope': -0.0663, 'vnb_slope': -0.0035},
        'VCC0688': {'rdb_slope': 0.0583, 'vnb_slope': None},  # No VNB data
        'VCC0990': {'rdb_slope': None, 'vnb_slope': None},  # Analysis failed
        'VCC1049': {'rdb_slope': 0.0593, 'vnb_slope': None},  # Limited VNB
        'VCC1146': {'rdb_slope': -0.0002, 'vnb_slope': None},  # Limited VNB
        'VCC1193': {'rdb_slope': 0.0490, 'vnb_slope': None},  # No VNB data
        'VCC1368': {'rdb_slope': 0.0220, 'vnb_slope': -0.0053},
        'VCC1410': {'rdb_slope': 0.0074, 'vnb_slope': None},  # No VNB data
        'VCC1431': {'rdb_slope': -0.1123, 'vnb_slope': -0.0014},
        'VCC1486': {'rdb_slope': -0.0098, 'vnb_slope': None},  # Limited VNB
        'VCC1499': {'rdb_slope': 0.0205, 'vnb_slope': None},  # No VNB data
        'VCC1549': {'rdb_slope': 0.0047, 'vnb_slope': None},  # No VNB data
        'VCC1588': {'rdb_slope': 0.0036, 'vnb_slope': None},  # No VNB data
        'VCC1695': {'rdb_slope': -0.0451, 'vnb_slope': -0.0034},
        'VCC1811': {'rdb_slope': 0.0173, 'vnb_slope': None},  # No VNB data
        'VCC1890': {'rdb_slope': -0.0235, 'vnb_slope': None},  # No VNB data
        'VCC1902': {'rdb_slope': 0.0021, 'vnb_slope': None},  # No VNB data
        'VCC1910': {'rdb_slope': 0.0093, 'vnb_slope': None},  # No VNB data
        'VCC1949': {'rdb_slope': -0.0188, 'vnb_slope': -0.0025}
    }
    
    return gradient_data

def create_quick_galaxy_summary_plot():
    """Create comprehensive galaxy summary plot quickly"""
    
    # Load data
    logger.info("Loading galaxy coordinates and gradient data...")
    galaxy_coords = get_galaxy_coordinates()
    gradient_data = extract_gradients_from_existing_plots()
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # RA/DEC positions with gradient vectors (left panel)
    ax1.set_aspect('equal')
    
    # Velocity color mapping
    velocities = [coords['velocity'] for coords in galaxy_coords.values()]
    vel_min, vel_max = np.min(velocities), np.max(velocities)
    norm = Normalize(vmin=vel_min, vmax=vel_max)
    cmap = plt.get_cmap('viridis')
    
    # Plot each galaxy
    rdb_slopes = []
    vnb_slopes = []
    galaxy_names = []
    colors_scatter = []
    symbols = []
    
    for galaxy_name in galaxy_coords.keys():
        coords = galaxy_coords[galaxy_name]
        grad_data = gradient_data.get(galaxy_name, {})
        
        ra = coords['ra']
        dec = coords['dec'] 
        velocity = coords['velocity']
        has_emission = coords['has_emission']
        
        # Color by velocity
        color = cmap(norm(velocity))
        
        # Plot galaxy position with different styles for emission/non-emission
        if has_emission:
            # Solid circle for emission galaxies
            ax1.scatter(ra, dec, c=[color], s=300, marker='o', 
                       edgecolors='black', linewidth=2, alpha=0.8)
        else:
            # Hollow circle for non-emission galaxies
            ax1.scatter(ra, dec, facecolors='none', edgecolors=color, s=300, marker='o', 
                       linewidth=3, alpha=0.9)
            # Add black edge
            ax1.scatter(ra, dec, facecolors='none', edgecolors='black', s=300, marker='o', 
                       linewidth=1, alpha=0.9)
        
        # Add galaxy label
        ax1.annotate(galaxy_name.replace('VCC', ''), (ra, dec), 
                    xytext=(8, 8), textcoords='offset points', fontsize=9, 
                    fontweight='bold', alpha=0.8)
        
        # Add gradient vectors
        vector_scale = 0.03  # Base length in RA/DEC units
        
        # RDB vector (red) - only if we have data
        if grad_data.get('rdb_slope') is not None:
            slope_rdb = grad_data['rdb_slope']
            # Vector pointing up for positive gradient, down for negative
            dy_rdb = vector_scale * np.sign(slope_rdb) * min(abs(slope_rdb) * 100, 1.0)
            ax1.arrow(ra - 0.02, dec, 0, dy_rdb, head_width=0.01, head_length=0.005, 
                     fc='red', ec='red', alpha=0.8, linewidth=3)
        
        # VNB vector (blue) - only if we have data
        if grad_data.get('vnb_slope') is not None:
            slope_vnb = grad_data['vnb_slope']
            dy_vnb = vector_scale * np.sign(slope_vnb) * min(abs(slope_vnb) * 100, 1.0)
            ax1.arrow(ra + 0.02, dec, 0, dy_vnb, head_width=0.01, head_length=0.005, 
                     fc='blue', ec='blue', alpha=0.8, linewidth=3)
        
        # Collect data for comparison plot
        if grad_data.get('rdb_slope') is not None and grad_data.get('vnb_slope') is not None:
            rdb_slopes.append(grad_data['rdb_slope'])
            vnb_slopes.append(grad_data['vnb_slope'])
            galaxy_names.append(galaxy_name)
            colors_scatter.append(color)
            symbols.append('o')
    
    ax1.set_xlabel('RA (degrees)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('DEC (degrees)', fontsize=14, fontweight='bold')
    ax1.set_title('Virgo Galaxies: α/Fe Gradient Vectors\\n(Red=RDB 3-bins, Blue=VNB same range)', 
                 fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for velocity
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity (km/s)', fontsize=14, fontweight='bold')
    
    # Add legend for symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='Emission Line Galaxy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=12, 
               markeredgecolor='black', markeredgewidth=3, label='Non-Emission Galaxy'),
        Line2D([0], [0], color='red', linewidth=4, label='RDB Gradient Vector (3 bins)'),
        Line2D([0], [0], color='blue', linewidth=4, label='VNB Gradient Vector (same range)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
    
    # Add note about vector direction
    ax1.text(0.02, 0.02, 'Vector Direction: ↑ Positive Gradient, ↓ Negative Gradient', 
            transform=ax1.transAxes, fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Gradient comparison plot (right panel)
    if rdb_slopes and vnb_slopes:
        # Plot RDB vs VNB comparison
        for i, (rdb, vnb, name, color) in enumerate(zip(rdb_slopes, vnb_slopes, galaxy_names, colors_scatter)):
            has_emission = galaxy_coords[name]['has_emission']
            if has_emission:
                # Solid circle for emission galaxies
                ax2.scatter(rdb, vnb, c=[color], s=200, marker='o', 
                           edgecolors='black', linewidth=2, alpha=0.8)
            else:
                # Hollow circle for non-emission galaxies  
                ax2.scatter(rdb, vnb, facecolors='none', edgecolors=color, s=200, marker='o', 
                           linewidth=3, alpha=0.9)
                ax2.scatter(rdb, vnb, facecolors='none', edgecolors='black', s=200, marker='o', 
                           linewidth=1, alpha=0.9)
            
            ax2.annotate(name.replace('VCC', ''), (rdb, vnb), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, 
                        fontweight='bold', alpha=0.8)
        
        # Add 1:1 line
        min_slope = min(min(rdb_slopes), min(vnb_slopes))
        max_slope = max(max(rdb_slopes), max(vnb_slopes))
        range_ext = (max_slope - min_slope) * 0.1
        ax2.plot([min_slope-range_ext, max_slope+range_ext], 
                [min_slope-range_ext, max_slope+range_ext], 'k--', alpha=0.7, linewidth=2, label='1:1 Line')
        
        # Add correlation info
        correlation = np.corrcoef(rdb_slopes, vnb_slopes)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: r = {correlation:.3f}\\nN = {len(rdb_slopes)} galaxies', 
                transform=ax2.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        logger.info(f"RDB vs VNB correlation: r = {correlation:.3f} (N={len(rdb_slopes)})")
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for comparison\\n(Need both RDB and VNB)', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=14, color='red')
    
    ax2.set_xlabel('RDB Gradient Slope (3 inner bins, R=0)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('VNB Gradient Slope (same radial range)', fontsize=14, fontweight='bold')
    ax2.set_title('RDB vs VNB Gradient Comparison', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if rdb_slopes and vnb_slopes:
        ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('./enhanced_radial_plots')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "galaxy_gradient_summary_with_vectors.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Galaxy summary plot with vectors saved: {output_file}")
    
    # Print summary statistics
    logger.info("\\n=== GRADIENT SUMMARY ===")
    logger.info(f"Total galaxies: {len(galaxy_coords)}")
    logger.info(f"Galaxies with RDB gradients: {sum(1 for g in gradient_data.values() if g.get('rdb_slope') is not None)}")
    logger.info(f"Galaxies with VNB gradients: {sum(1 for g in gradient_data.values() if g.get('vnb_slope') is not None)}")
    logger.info(f"Galaxies with both RDB and VNB: {len(rdb_slopes)}")
    
    if rdb_slopes:
        logger.info(f"RDB slopes: mean = {np.mean(rdb_slopes):.4f}, std = {np.std(rdb_slopes):.4f}")
    if vnb_slopes:
        logger.info(f"VNB slopes: mean = {np.mean(vnb_slopes):.4f}, std = {np.std(vnb_slopes):.4f}")

if __name__ == "__main__":
    create_quick_galaxy_summary_plot()
