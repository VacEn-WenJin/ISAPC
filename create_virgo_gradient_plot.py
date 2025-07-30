"""
Virgo Cluster Galaxy Gradient Visualization - Original Phy_Visu Style
Shows galaxy positions with gradient vectors colored by velocity
Based on the original Phy_Visu.py approach
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
logger = logging.getLogger('VirgoGradientVisu')

def create_virgo_cluster_gradient_plot():
    """Create Virgo Cluster visualization with gradient vectors - original Phy_Visu style"""
    
    # Galaxy coordinates in Virgo Cluster (RA, DEC in degrees)
    # Based on typical Virgo Cluster positions
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
    
    logger.info("Extracting RDB gradients for all galaxies...")
    
    # Extract RDB gradients (3 bins only)
    galaxy_gradients = {}
    for galaxy_name in galaxy_coords.keys():
        try:
            logger.info(f"Processing {galaxy_name}...")
            result = analyze_single_galaxy(galaxy_name)
            
            if result and result.get('analysis_success', False):
                # Extract RDB data (first 3 bins only)
                radial_profile = result.get('radial_profile')
                if radial_profile and 'bin_radii' in radial_profile:
                    radii = radial_profile['bin_radii'][:3]  # Only first 3 bins
                    alpha_fe = radial_profile['alpha_fe_mean'][:3]
                    
                    # Set innermost bin to R=0 as requested
                    if len(radii) > 0:
                        radii = radii.copy()
                        radii[0] = 0.0
                    
                    # Calculate gradient using linear fit
                    if len(radii) >= 2:
                        valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                        if np.sum(valid_mask) >= 2:
                            slope = np.polyfit(radii[valid_mask], alpha_fe[valid_mask], 1)[0]
                            galaxy_gradients[galaxy_name] = {
                                'slope': slope,
                                'radii': radii[valid_mask],
                                'alpha_fe': alpha_fe[valid_mask]
                            }
                            logger.info(f"  RDB gradient: {slope:.4f}")
                        
        except Exception as e:
            logger.warning(f"Error processing {galaxy_name}: {e}")
    
    # Create the plot - original Phy_Visu style
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Set up velocity color mapping
    velocities = [coords['velocity'] for coords in galaxy_coords.values()]
    vel_min, vel_max = min(velocities), max(velocities)
    norm = Normalize(vmin=vel_min, vmax=vel_max)
    cmap = plt.cm.viridis
    
    # Plot Virgo Cluster structure
    logger.info("Creating Virgo Cluster visualization...")
    
    # Add M87 position (approximate center of Virgo Cluster)
    m87_ra, m87_dec = 187.70, 12.39
    ax.scatter(m87_ra, m87_dec, s=500, marker='*', c='red', 
              edgecolors='black', linewidth=2, label='M87 (Cluster Center)', zorder=10)
    
    # Plot each galaxy with gradient vectors
    for galaxy_name, coords in galaxy_coords.items():
        ra = coords['ra']
        dec = coords['dec']
        velocity = coords['velocity']
        has_emission = coords['has_emission']
        
        # Get gradient data
        gradient_data = galaxy_gradients.get(galaxy_name)
        
        # Color by velocity
        color = cmap(norm(velocity))
        
        # Plot galaxy position - hollow for non-emission, solid for emission
        if has_emission:
            marker_style = 'o'
            facecolors = [color]
            alpha = 0.8
        else:
            marker_style = 'o'
            facecolors = 'none'
            alpha = 0.9
        
        ax.scatter(ra, dec, c=facecolors, s=150, marker=marker_style,
                  edgecolors='black', linewidth=1.5, 
                  alpha=alpha, zorder=5)
        
        # Add galaxy label
        ax.annotate(galaxy_name.replace('VCC', ''), (ra, dec),
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=9, alpha=0.8, fontweight='bold')
        
        # Add gradient vector as colored triangle (original Phy_Visu style)
        if gradient_data:
            slope = gradient_data['slope']
            
            # Vector length based on gradient magnitude
            base_length = 0.08  # Base length in RA/DEC units
            vector_length = base_length * (1 + abs(slope) * 5)  # Scale by gradient strength
            
            # Vector direction: up for positive, down for negative
            if slope > 0:
                # Positive gradient - triangle pointing up
                triangle_x = [ra, ra - 0.02, ra + 0.02, ra]
                triangle_y = [dec + vector_length, dec, dec, dec + vector_length]
            else:
                # Negative gradient - triangle pointing down  
                triangle_x = [ra, ra - 0.02, ra + 0.02, ra]
                triangle_y = [dec - vector_length, dec, dec, dec - vector_length]
            
            # Create triangle with velocity color
            triangle = plt.Polygon(list(zip(triangle_x, triangle_y)), 
                                 facecolor=color, edgecolor='black', 
                                 linewidth=1, alpha=0.8, zorder=6)
            ax.add_patch(triangle)
    
    # Add Virgo Cluster structure lines/regions
    # Draw approximate cluster substructure
    virgo_a_ra = [186.5, 187.5, 188.0, 187.2, 186.5]
    virgo_a_dec = [12.2, 12.8, 13.2, 13.8, 12.2]
    ax.plot(virgo_a_ra, virgo_a_dec, 'k--', alpha=0.3, linewidth=2, label='Virgo A Subcluster')
    
    # Add distance circles from M87
    circle_radii = [1.0, 2.0]  # degrees
    for radius in circle_radii:
        circle = plt.Circle((m87_ra, m87_dec), radius, fill=False, 
                          linestyle=':', color='gray', alpha=0.4, linewidth=1)
        ax.add_patch(circle)
    
    # Formatting
    ax.set_xlabel('Right Ascension (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Declination (degrees)', fontsize=14, fontweight='bold')
    ax.set_title('Virgo Cluster Galaxies: α/Fe Radial Gradients\\n(RDB Method, 3 Inner Bins)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(185.5, 188.5)
    ax.set_ylim(11.5, 14.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add colorbar for velocity
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                       fraction=0.046, pad=0.04, aspect=30)
    cbar.set_label('Recession Velocity (km/s)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Create legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=12, fillstyle='full', markeredgecolor='black', 
               label='Emission Line Galaxy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markersize=12, fillstyle='none', markeredgecolor='black', 
               markeredgewidth=2, label='Non-Emission Galaxy'),
        Line2D([0], [0], marker='^', color='blue', markersize=12, 
               label='Positive α/Fe Gradient'),
        Line2D([0], [0], marker='v', color='red', markersize=12, 
               label='Negative α/Fe Gradient'),
        Line2D([0], [0], marker='*', color='red', markersize=15, 
               markeredgecolor='black', label='M87 (Cluster Center)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
             frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    # Add text box with gradient statistics
    if galaxy_gradients:
        slopes = [data['slope'] for data in galaxy_gradients.values()]
        n_positive = sum(1 for s in slopes if s > 0)
        n_negative = sum(1 for s in slopes if s < 0)
        mean_slope = np.mean(slopes)
        
        stats_text = f'Gradient Statistics:\\n' \
                    f'Total galaxies: {len(slopes)}\\n' \
                    f'Positive gradients: {n_positive}\\n' \
                    f'Negative gradients: {n_negative}\\n' \
                    f'Mean slope: {mean_slope:.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('./enhanced_radial_plots')
    output_file = output_dir / "virgo_cluster_gradients_original_style.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Virgo Cluster gradient plot saved: {output_file}")
    logger.info(f"Processed {len(galaxy_gradients)} galaxies with valid gradients")
    
    # Print summary
    if galaxy_gradients:
        print("\\n" + "="*60)
        print("VIRGO CLUSTER α/Fe GRADIENT SUMMARY")
        print("="*60)
        for galaxy_name, data in galaxy_gradients.items():
            slope = data['slope']
            direction = "↗" if slope > 0 else "↘"
            print(f"{galaxy_name}: {slope:+.4f} {direction}")
        print("="*60)

if __name__ == "__main__":
    create_virgo_cluster_gradient_plot()
