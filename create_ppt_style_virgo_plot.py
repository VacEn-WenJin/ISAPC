#!/usr/bin/env python3
"""
Create PPT-style Virgo Cluster plot exactly as specified:
- Real IFU RA/DEC positions from FITS headers
- Vertical gradient vectors (up=positive, down=negative)
- Triangles at vector origins (solid=emission, hollow=no emission)
- Error bars as horizontal lines
- Color by velocity
- Match PPT presentation style
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import glob
import os

def get_ifu_positions():
    """Get actual IFU center positions from FITS headers"""
    positions = {}
    
    for fits_file in sorted(glob.glob('data/MUSE/VCC*_stack.fits')):
        try:
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                
                # Get reference coordinates from WCS
                crval1 = header.get('CRVAL1', None)  # RA
                crval2 = header.get('CRVAL2', None)  # DEC
                
                if crval1 is not None and crval2 is not None:
                    galaxy = fits_file.split('/')[-1].replace('_stack.fits', '')
                    positions[galaxy] = {'ra': crval1, 'dec': crval2}
                    
        except Exception as e:
            print(f"Error reading {fits_file}: {e}")
            
    return positions

def create_ppt_style_plot():
    """Create the PPT-style Virgo cluster plot"""
    
    # Major Virgo galaxies coordinates
    messier_coords = {
        'M87': {'ra': 187.70591, 'dec': 12.39112, 'name': 'M87 (NGC 4486)', 'color': 'red', 'marker': '*'},
        'M86': {'ra': 186.54829, 'dec': 12.95668, 'name': 'M86 (NGC 4406)', 'color': 'orange', 'marker': 'D'}, 
        'M60': {'ra': 190.91684, 'dec': 11.55217, 'name': 'M60 (NGC 4649)', 'color': 'blue', 'marker': 's'},
        'M49': {'ra': 187.44419, 'dec': 8.00003, 'name': 'M49 (NGC 4472)', 'color': 'green', 'marker': 'h'}
    }
    
    # Get IFU positions
    positions = get_ifu_positions()
    
    # Read gradient and velocity data
    gradient_df = pd.read_csv('enhanced_galaxy_summary/enhanced_alpha_fe_results.csv')
    velocity_df = pd.read_csv('alpha_gradient_plots/enhanced_gradient_velocity_summary.csv')
    
    # Merge data
    plot_data = []
    
    for _, grad_row in gradient_df.iterrows():
        galaxy = grad_row['Galaxy']
        
        if galaxy in positions:
            # Get velocity data
            vel_row = velocity_df[velocity_df['Galaxy'] == galaxy]
            if not vel_row.empty:
                vel_data = vel_row.iloc[0]
                
                # Extract data for plotting
                ra = positions[galaxy]['ra']
                dec = positions[galaxy]['dec']
                slope = grad_row['Slope'] if grad_row['Slope'] != '---' else 0.0
                slope_err = grad_row['Slope_Uncertainty'] if grad_row['Slope_Uncertainty'] != '---' else 0.0
                emission = grad_row['Emission_Lines'] == 'Yes'
                velocity = vel_data['Central_Velocity_Real_kms']
                
                # Convert to numeric if needed
                try:
                    slope = float(slope)
                    slope_err = float(slope_err)
                except:
                    slope = 0.0
                    slope_err = 0.0
                
                plot_data.append({
                    'galaxy': galaxy,
                    'ra': ra,
                    'dec': dec,
                    'slope': slope,
                    'slope_err': slope_err,
                    'emission': emission,
                    'velocity': velocity
                })
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(plot_data)
    
    # Calculate cluster extent and create equal-scale plot
    ra_min, ra_max = df['ra'].min(), df['ra'].max()
    dec_min, dec_max = df['dec'].min(), df['dec'].max()
    
    # Add padding around the data
    ra_padding = (ra_max - ra_min) * 0.1
    dec_padding = (dec_max - dec_min) * 0.1
    
    ra_range = (ra_max - ra_min) + 2 * ra_padding
    dec_range = (dec_max - dec_min) + 2 * dec_padding
    
    # Create larger plot with equal scales
    fig, ax = plt.subplots(figsize=(14, 14))  # Increased from (10, 10)
    
    # Set up equal-scale coordinate system
    ax.set_xlim(ra_min - ra_padding, ra_max + ra_padding)
    ax.set_ylim(dec_min - dec_padding, dec_max + dec_padding)
    ax.set_xlabel('Right Ascension (deg)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Declination (deg)', fontsize=16, fontweight='bold')
    ax.set_title('Virgo Cluster: α/Fe Abundance Gradients\n(Equal RA/DEC Scale, Real MUSE IFU Positions)', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Create velocity colormap
    velocities = df['velocity'].values
    vel_min, vel_max = velocities.min(), velocities.max()
    
    # Create enhanced velocity colormap with more contrast
    from matplotlib.cm import plasma, viridis
    import matplotlib.colors as mcolors
    
    # Normalize velocities for coloring with better contrast
    norm = mcolors.Normalize(vmin=vel_min, vmax=vel_max)
    cmap = plasma  # More vibrant colormap
    
    # Plot each galaxy with vectors scaled for 0.1 gradient = 1 unit
    vector_scale = 10.0  # 1 unit = 0.1 gradient (10.0 scale factor)
    triangle_size = 0.025  # Increased triangle size for better visibility
    
    for _, row in df.iterrows():
        ra = row['ra']
        dec = row['dec']
        slope = row['slope']
        slope_err = row['slope_err']
        emission = row['emission']
        velocity = row['velocity']
        galaxy = row['galaxy']
        
        # Calculate vector length and direction
        vector_length = abs(slope) * vector_scale
        direction = 1 if slope >= 0 else -1
        
        # Vector endpoints (vertical only)
        vector_end_dec = dec + direction * vector_length
        
        # Color based on velocity
        color = cmap(norm(velocity))
        
        # Draw the gradient vector as a vertical line with enhanced style
        if abs(slope) > 0.001:  # Only draw if significant
            ax.plot([ra, ra], [dec, vector_end_dec], 
                   color=color, linewidth=2.5, alpha=0.9, zorder=2,  # Reduced from 4.0
                   solid_capstyle='round')
            
            # Add enhanced arrowhead at the end
            arrow_size = 0.015  # Slightly larger arrow for bigger plot
            if slope > 0:
                # Upward arrow for positive gradient
                ax.arrow(ra, vector_end_dec-arrow_size*0.3, 0, arrow_size*0.3, 
                        head_width=arrow_size*1.2, head_length=arrow_size*0.6, 
                        fc=color, ec='black', linewidth=1.2, zorder=3)
            else:
                # Downward arrow for negative gradient
                ax.arrow(ra, vector_end_dec+arrow_size*0.3, 0, -arrow_size*0.3, 
                        head_width=arrow_size*1.2, head_length=arrow_size*0.6, 
                        fc=color, ec='black', linewidth=1.2, zorder=3)
            
            # Add gradient value label next to the vector
            label_y = vector_end_dec + (0.025 if slope > 0 else -0.025)  # Increased offset
            ax.text(ra + 0.015, label_y, f'{slope:+.2f}', 
                   fontsize=11, fontweight='bold', ha='left', va='center',  # Increased font size
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=1.2), 
                   zorder=7)
        
        # Draw enhanced error bar as perpendicular line with caps
        if slope_err > 0:
            err_length = slope_err * vector_scale
            
            # Position error bar at the tip of the gradient vector
            if abs(slope) > 0.001:
                error_y = vector_end_dec
            else:
                error_y = dec
            
            # Main error bar (horizontal line at vector tip)
            ax.plot([ra-0.01, ra+0.01], [error_y, error_y],
                   color='black', linewidth=3.0, alpha=0.9, zorder=4,
                   solid_capstyle='round')
            
            # Error bar end caps (small vertical lines)
            cap_size = 0.004
            ax.plot([ra-0.01, ra-0.01], [error_y-cap_size, error_y+cap_size],
                   color='black', linewidth=3.0, alpha=0.9, zorder=4)
            ax.plot([ra+0.01, ra+0.01], [error_y-cap_size, error_y+cap_size],
                   color='black', linewidth=3.0, alpha=0.9, zorder=4)
        
        # Draw enhanced triangle at origin
        if emission:
            # Solid triangle for emission lines with gradient edge
            triangle = plt.Polygon([(ra, dec + triangle_size), 
                                  (ra - triangle_size*0.866, dec - triangle_size*0.5),
                                  (ra + triangle_size*0.866, dec - triangle_size*0.5)],
                                 facecolor=color, edgecolor='black', 
                                 linewidth=2.0, zorder=5, alpha=0.9)  # Increased linewidth
        else:
            # Hollow triangle for no emission lines with thick colored edge
            triangle = plt.Polygon([(ra, dec + triangle_size), 
                                  (ra - triangle_size*0.866, dec - triangle_size*0.5),
                                  (ra + triangle_size*0.866, dec - triangle_size*0.5)],
                                 facecolor='white', edgecolor=color, 
                                 linewidth=3.0, zorder=5, alpha=0.9)  # Increased linewidth
        
        ax.add_patch(triangle)
        
        # Add enhanced galaxy label with better positioning
        label_offset = 0.035  # Increased offset for larger plot
        ax.text(ra + label_offset, dec, galaxy.replace('VCC', ''), 
               fontsize=12, fontweight='bold', ha='left', va='center',  # Increased font size
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', alpha=0.9, linewidth=1.0), 
               zorder=6)
    
    # Add major Virgo galaxies
    for name, coords in messier_coords.items():
        ra_m = coords['ra']
        dec_m = coords['dec']
        
        # Check if within plot bounds
        if (ra_min - ra_padding <= ra_m <= ra_max + ra_padding and 
            dec_min - dec_padding <= dec_m <= dec_max + dec_padding):
            
            ax.scatter(ra_m, dec_m, s=400, marker=coords['marker'],  # Increased from 250
                      c=coords['color'], edgecolors='black', linewidth=2.5,  # Increased linewidth
                      alpha=0.9, zorder=10, label=coords['name'])
            
            # Add labels for major galaxies
            ax.text(ra_m + 0.04, dec_m + 0.04, name,  # Increased offsets
                   fontsize=13, fontweight='bold', ha='left', va='bottom',  # Increased font size
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                            edgecolor=coords['color'], alpha=0.95, linewidth=2.0),  # Increased linewidth
                   zorder=11)
    
    # Add enhanced colorbar for velocity
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.9, pad=0.02, aspect=30)
    cbar.set_label('Central Velocity (km/s)', fontsize=14, fontweight='bold', 
                   labelpad=15)
    cbar.ax.tick_params(labelsize=12, width=1.5, length=6)
    
    # Add legend for triangle types with enhanced styling
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', 
               markeredgecolor='black', markersize=15,  # Increased from 12
               label='With Emission Lines', markeredgewidth=2.0),  # Increased linewidth
        Line2D([0], [0], marker='^', color='w', markerfacecolor='white', 
               markeredgecolor='purple', markersize=15,  # Increased from 12
               label='No Emission Lines', markeredgewidth=3.0),  # Increased linewidth
        Line2D([0], [0], color='purple', linewidth=2.5,  # Reduced from 4.0
               label='α/Fe Gradient Vector'),
        Line2D([0], [0], color='black', linewidth=3.0, 
               label='Gradient Uncertainty'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markeredgecolor='black', markersize=18,  # Increased from 15
               label='M87 (Central Giant)', markeredgewidth=2.5),  # Increased linewidth
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', 
               markeredgecolor='black', markersize=13,  # Increased from 10
               label='M86', markeredgewidth=2.5),  # Increased linewidth
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
               markeredgecolor='black', markersize=13,  # Increased from 10
               label='M60', markeredgewidth=2.5),  # Increased linewidth
        Line2D([0], [0], marker='h', color='w', markerfacecolor='green', 
               markeredgecolor='black', markersize=13,  # Increased from 10
               label='M49', markeredgewidth=2.5)  # Increased linewidth
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', 
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=11, title='Symbol Legend', title_fontsize=13,  # Increased font sizes
                      edgecolor='black')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(1.2)
    
    # Add enhanced gradient scale annotation
    ra_center = df['ra'].mean()
    dec_center = df['dec'].mean()
    ax.text(0.02, 0.02, f'Vector Scale: 1 unit = 0.1 α/Fe gradient\nField: RA {ra_min:.2f}°-{ra_max:.2f}°, DEC {dec_min:.2f}°-{dec_max:.2f}°', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                    edgecolor='orange', alpha=0.95, linewidth=1.2),
           verticalalignment='bottom')
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray')
    ax.set_aspect('equal', adjustable='box')
    
    # Add minor ticks for better precision
    ax.minorticks_on()
    ax.tick_params(which='major', labelsize=12, width=1.5, length=8)
    ax.tick_params(which='minor', width=1, length=4)
    
    # Enhanced border
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    # Make it look professional
    plt.tight_layout(pad=2.0)
    
    # Save the enhanced plot
    output_path = 'enhanced_radial_plots/virgo_cluster_with_messier_objects.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    print(f"Enhanced Virgo cluster plot with Messier objects saved to: {output_path}")
    print(f"Plotted {len(df)} galaxies with equal RA/DEC scales")
    print(f"RA range: {ra_min:.3f}° - {ra_max:.3f}°")
    print(f"DEC range: {dec_min:.3f}° - {dec_max:.3f}°")
    print(f"Velocity range: {vel_min:.0f} - {vel_max:.0f} km/s")
    print(f"Vector scale: 1 unit = 0.1 α/Fe gradient")
    print("Major galaxies included: M87, M86, M60, M49")
    
    plt.show()
    
    return output_path

if __name__ == "__main__":
    create_ppt_style_plot()
