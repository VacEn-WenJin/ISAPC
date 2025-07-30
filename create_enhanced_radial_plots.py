"""
Create enhanced radial plots using existing alpha gradient analysis
Focuses on proper physical radius and R=0 correction for RDB
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from alpha_gradient_analysis import analyze_single_galaxy
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CreateRadialPlots')

def create_radial_gradient_plots_for_all_galaxies():
    """Create enhanced radial gradient plots for all galaxies with proper corrections"""
    
    # Create output directory
    output_dir = Path('./enhanced_radial_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Clear existing plots
    logger.info(f"Clearing existing plots in {output_dir}")
    for file in output_dir.glob('*.png'):
        file.unlink()
    
    # Galaxy list
    galaxies = [
        'VCC0308', 'VCC0667', 'VCC0688', 'VCC0990', 'VCC1049',
        'VCC1146', 'VCC1193', 'VCC1368', 'VCC1410', 'VCC1431',
        'VCC1486', 'VCC1499', 'VCC1549', 'VCC1588', 'VCC1695',
        'VCC1811', 'VCC1890', 'VCC1902', 'VCC1910', 'VCC1949'
    ]
    
    successful_plots = 0
    all_galaxy_data = {}
    
    # Process each galaxy
    for galaxy_name in galaxies:
        logger.info(f"Processing {galaxy_name}...")
        
        try:
            # Run the analysis using existing function
            result = analyze_single_galaxy(galaxy_name)
            
            if result and result.get('analysis_success', False):
                all_galaxy_data[galaxy_name] = result
                
                # Create enhanced individual plot
                success = create_enhanced_individual_plot(galaxy_name, result, output_dir)
                if success:
                    successful_plots += 1
            else:
                logger.warning(f"Analysis failed for {galaxy_name}")
                
        except Exception as e:
            logger.error(f"Error processing {galaxy_name}: {e}")
    
    # Create summary plot
    if all_galaxy_data:
        create_summary_plot(all_galaxy_data, output_dir)
    
    logger.info(f"Enhanced radial plot creation completed: {successful_plots}/{len(galaxies)} successful")
    logger.info(f"Plots saved in: {output_dir}")

def create_enhanced_individual_plot(galaxy_name, analysis_result, output_dir):
    """
    Create enhanced individual radial plot with 3-panel format and corrections
    
    Parameters
    ----------
    galaxy_name : str
        Galaxy name
    analysis_result : dict
        Result from analyze_single_galaxy
    output_dir : Path
        Output directory
        
    Returns
    -------
    bool
        Success status
    """
    try:
        # Extract data
        radial_profile = analysis_result.get('radial_profile')
        vnb_profile = analysis_result.get('vnb_profile')
        multi_gradient_results = analysis_result.get('multi_gradient_results', {})
        
        if not radial_profile or not vnb_profile:
            logger.warning(f"Missing profile data for {galaxy_name}")
            return False
        
        # Set up figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{galaxy_name} - α/Fe Radial Gradients (Enhanced)', 
                    fontsize=16, fontweight='bold')
        
        # Method configurations
        methods = [
            {'key': 'rdb', 'name': 'RDB (Radial)', 'profile': radial_profile, 'color': '#ff7f0e', 'marker': 's', 'panel': 0},
            {'key': 'vnb', 'name': 'VNB (Voronoi)', 'profile': vnb_profile, 'color': '#1f77b4', 'marker': 'o', 'panel': 1},
            {'key': 'p2p', 'name': 'P2P (Pixel-to-Pixel)', 'profile': None, 'color': '#2ca02c', 'marker': '^', 'panel': 2}
        ]
        
        # Plot each method
        for method in methods:
            ax = axes[method['panel']]
            profile = method['profile']
            
            if profile and 'bin_radii' in profile and 'alpha_fe_mean' in profile:
                # Get data
                radii = profile['bin_radii'].copy()
                alpha_fe = profile['alpha_fe_mean']
                alpha_fe_error = profile.get('alpha_fe_error', np.full_like(alpha_fe, 0.05))
                
                # Apply R=0 correction for RDB method
                if method['key'] == 'rdb' and len(radii) > 0:
                    radii[0] = 0.0  # Set innermost bin to R=0 as requested
                
                # Filter valid data
                valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                if np.any(valid_mask):
                    radii_plot = radii[valid_mask]
                    alpha_plot = alpha_fe[valid_mask]
                    alpha_error_plot = alpha_fe_error[valid_mask]
                    
                    # Plot data with error bars
                    ax.errorbar(radii_plot, alpha_plot, yerr=alpha_error_plot,
                               fmt=method['marker'], color=method['color'], 
                               markersize=8, capsize=4, capthick=2, alpha=0.8,
                               label=f'{method["name"]} Data')
                    
                    # Fit and plot trend line
                    if len(radii_plot) > 2:
                        try:
                            # Simple linear fit
                            z = np.polyfit(radii_plot, alpha_plot, 1)
                            p = np.poly1d(z)
                            
                            # Create smooth line
                            r_fit = np.linspace(0, np.max(radii_plot), 100)
                            alpha_fit = p(r_fit)
                            
                            ax.plot(r_fit, alpha_fit, '--', color=method['color'], 
                                   alpha=0.7, linewidth=2,
                                   label=f'Fit: slope={z[0]:.3f}')
                            
                            # Add fit statistics
                            correlation = np.corrcoef(radii_plot, alpha_plot)[0, 1]
                            ax.text(0.05, 0.95, f'r = {correlation:.3f}\nslope = {z[0]:.3f}', 
                                   transform=ax.transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            
                        except Exception as e:
                            logger.debug(f"Could not fit trend line for {method['name']}: {e}")
                    
                    ax.set_title(f'{method["name"]} Method', fontsize=14, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No valid data', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, color='red')
                    ax.set_title(f'{method["name"]} Method (No Data)', fontsize=14, color='red')
            
            elif method['key'] == 'p2p':
                # Handle P2P method differently since it may not have a standard profile
                gradient_result = multi_gradient_results.get('p2p')
                if gradient_result and 'slope' in gradient_result:
                    # Create a simple representation of P2P results
                    ax.text(0.5, 0.5, f'P2P Gradient\nSlope: {gradient_result["slope"]:.3f}\n'
                                      f'R²: {gradient_result.get("r_squared", "N/A"):.3f}', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                    ax.set_title(f'{method["name"]} Method', fontsize=14, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No P2P data available', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, color='red')
                    ax.set_title(f'{method["name"]} Method (No Data)', fontsize=14, color='red')
            else:
                ax.text(0.5, 0.5, f'No {method["name"]} data available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='red')
                ax.set_title(f'{method["name"]} Method (No Data)', fontsize=14, color='red')
            
            # Formatting
            ax.set_xlabel('Radius (arcsec)', fontsize=12)
            ax.set_ylabel('[α/Fe]', fontsize=12)
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:  # Only add legend if there are items
                ax.legend(fontsize=10)
        
        # Add note about R=0 correction
        fig.text(0.5, 0.02, 'Note: RDB innermost bin placed at R=0 as requested', 
                ha='center', fontsize=10, style='italic', color='blue')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        
        # Save the plot
        output_file = output_dir / f"{galaxy_name}_enhanced_radial_gradient.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced radial plot saved: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating enhanced plot for {galaxy_name}: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return False

def create_summary_plot(all_galaxy_data, output_dir):
    """Create a summary plot showing all galaxies"""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = {'rdb': '#ff7f0e', 'vnb': '#1f77b4', 'p2p': '#2ca02c'}
        markers = {'rdb': 's', 'vnb': 'o', 'p2p': '^'}
        
        for galaxy_name, galaxy_data in all_galaxy_data.items():
            # Plot RDB data
            radial_profile = galaxy_data.get('radial_profile')
            if radial_profile and 'bin_radii' in radial_profile:
                radii = radial_profile['bin_radii'].copy()
                alpha_fe = radial_profile['alpha_fe_mean']
                
                # Apply R=0 correction
                if len(radii) > 0:
                    radii[0] = 0.0
                
                valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                if np.any(valid_mask):
                    ax.scatter(radii[valid_mask], alpha_fe[valid_mask], 
                              c=colors['rdb'], marker=markers['rdb'], alpha=0.6, s=30,
                              label='RDB' if galaxy_name == list(all_galaxy_data.keys())[0] else "")
            
            # Plot VNB data
            vnb_profile = galaxy_data.get('vnb_profile')
            if vnb_profile and 'bin_radii' in vnb_profile:
                radii = vnb_profile['bin_radii']
                alpha_fe = vnb_profile['alpha_fe_mean']
                
                valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                if np.any(valid_mask):
                    ax.scatter(radii[valid_mask], alpha_fe[valid_mask], 
                              c=colors['vnb'], marker=markers['vnb'], alpha=0.6, s=30,
                              label='VNB' if galaxy_name == list(all_galaxy_data.keys())[0] else "")
        
        ax.set_xlabel('Radius (arcsec)', fontsize=12)
        ax.set_ylabel('[α/Fe]', fontsize=12)
        ax.set_title('Combined α/Fe Radial Gradients - All Galaxies\\n(Enhanced with R=0 Correction)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        output_file = output_dir / "combined_enhanced_radial_gradients.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Combined summary plot saved: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")

if __name__ == "__main__":
    create_radial_gradient_plots_for_all_galaxies()
