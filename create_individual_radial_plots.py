#!/usr/bin/env python3
"""
Individual Radial Gradient Plot Creator
Creates radial plots using the alpha_gradient_analysis system that includes VNB/RDB/P2P methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import stats
from alpha_gradient_analysis import analyze_single_galaxy, load_galaxy_alpha_fe_data
from reader import load_galaxy_rdb
import glob
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RadialPlots')

class IndividualRadialPlotter:
    """Creates individual radial gradient plots using the full alpha gradient analysis."""
    
    def __init__(self):
        self.output_dir = Path('./individual_radial_plots')
        self.output_dir.mkdir(exist_ok=True)
        
        # Plotting parameters
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.figsize': [12, 8]
        })
    
    def get_galaxy_list(self):
        """Get list of available galaxies."""
        # Check for galaxy directories in data folder
        data_dir = Path('./data')
        galaxy_dirs = [d.name for d in data_dir.glob('VCC*') if d.is_dir()]
        return sorted(galaxy_dirs)
    
    def load_and_process_galaxy(self, galaxy_name):
        """Load galaxy data and calculate alpha/Fe profiles for all methods."""
        logger.info(f"Processing {galaxy_name}...")
        
        # Load basic galaxy data
        try:
            galaxy_data = load_galaxy_data(galaxy_name)
            if galaxy_data is None:
                logger.warning(f"Could not load basic data for {galaxy_name}")
                return None
        except Exception as e:
            logger.warning(f"Error loading {galaxy_name}: {e}")
            return None
        
        results = {
            'galaxy_name': galaxy_name,
            'data_available': {
                'p2p': galaxy_data is not None,
                'rdb': False,
                'vnb': False
            },
            'profiles': {}
        }
        
        # Try to load RDB data
        try:
            rdb_data = load_galaxy_rdb(galaxy_name)
            if rdb_data is not None:
                results['data_available']['rdb'] = True
                logger.info(f"  ✓ RDB data available for {galaxy_name}")
        except Exception as e:
            logger.info(f"  ✗ RDB data not available for {galaxy_name}: {e}")
            rdb_data = None
        
        # Try to calculate VNB profile
        try:
            vnb_profile = calculate_vnb_alpha_fe_profile(galaxy_name)
            if vnb_profile is not None and len(vnb_profile) > 0:
                results['data_available']['vnb'] = True
                results['profiles']['vnb'] = vnb_profile
                logger.info(f"  ✓ VNB profile calculated for {galaxy_name}: {len(vnb_profile)} bins")
            else:
                logger.info(f"  ✗ VNB profile calculation failed for {galaxy_name}")
        except Exception as e:
            logger.info(f"  ✗ VNB profile error for {galaxy_name}: {e}")
        
        # If RDB is available, try to extract radial bins
        if rdb_data is not None:
            try:
                # Look for binned data in RDB
                if 'bin_3' in rdb_data:
                    # Extract 3-bin data
                    bin_data = []
                    for bin_key in ['bin_1', 'bin_2', 'bin_3']:
                        if bin_key in rdb_data:
                            bin_info = rdb_data[bin_key]
                            if isinstance(bin_info, dict):
                                # Try to extract alpha/Fe and radius
                                try:
                                    # Look for radius information
                                    radius = None
                                    if 'distance' in bin_info:
                                        dist_info = bin_info['distance']
                                        if isinstance(dist_info, dict):
                                            if 'R/Re' in dist_info:
                                                radius = dist_info['R/Re']
                                            elif 'r_re' in dist_info:
                                                radius = dist_info['r_re']
                                            elif 'radius' in dist_info:
                                                radius = dist_info['radius']
                                    
                                    # Look for alpha/Fe information
                                    alpha_fe = None
                                    alpha_fe_error = None
                                    if 'spectral_indices' in bin_info:
                                        spec_indices = bin_info['spectral_indices']
                                        if isinstance(spec_indices, dict):
                                            # Look for pre-calculated alpha/Fe
                                            if 'alpha_fe' in spec_indices:
                                                alpha_fe = spec_indices['alpha_fe']
                                                if 'alpha_fe_error' in spec_indices:
                                                    alpha_fe_error = spec_indices['alpha_fe_error']
                                    
                                    if radius is not None and alpha_fe is not None:
                                        bin_data.append({
                                            'radius': radius,
                                            'alpha_fe': alpha_fe,
                                            'alpha_fe_error': alpha_fe_error if alpha_fe_error is not None else 0.05,
                                            'bin_name': bin_key
                                        })
                                        
                                except Exception as bin_error:
                                    logger.debug(f"Error processing {bin_key} for {galaxy_name}: {bin_error}")
                    
                    if bin_data:
                        results['profiles']['rdb_3bin'] = bin_data
                        logger.info(f"  ✓ RDB 3-bin profile extracted for {galaxy_name}: {len(bin_data)} bins")
                    else:
                        logger.info(f"  ✗ Could not extract usable RDB 3-bin data for {galaxy_name}")
                        
            except Exception as e:
                logger.info(f"  ✗ RDB processing error for {galaxy_name}: {e}")
        
        return results
    
    def create_individual_radial_plot(self, galaxy_results):
        """Create a radial plot for a single galaxy with all available methods."""
        galaxy_name = galaxy_results['galaxy_name']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plot_count = 0
        all_radii = []
        all_alpha_fe = []
        
        # Plot VNB data if available
        if 'vnb' in galaxy_results['profiles']:
            vnb_data = galaxy_results['profiles']['vnb']
            
            radii = [point['radius'] for point in vnb_data]
            alpha_fe_vals = [point['alpha_fe'] for point in vnb_data]
            alpha_fe_errors = [point.get('alpha_fe_error', 0.05) for point in vnb_data]
            
            ax.errorbar(radii, alpha_fe_vals, yerr=alpha_fe_errors,
                       fmt='o', alpha=0.7, markersize=6, capsize=4,
                       color='blue', ecolor='lightblue', 
                       label=f'VNB ({len(radii)} bins)')
            
            all_radii.extend(radii)
            all_alpha_fe.extend(alpha_fe_vals)
            plot_count += 1
            
            # Fit VNB gradient
            if len(radii) >= 3:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(radii, alpha_fe_vals)
                    r_fit = np.linspace(min(radii), max(radii), 100)
                    alpha_fe_fit = slope * r_fit + intercept
                    
                    fit_color = 'darkblue' if p_value < 0.05 else 'lightblue'
                    ax.plot(r_fit, alpha_fe_fit, '--', color=fit_color, linewidth=2, alpha=0.8,
                           label=f'VNB fit: {slope:.4f}±{std_err:.4f} dex/(R/Re)')
                except Exception:
                    pass
        
        # Plot RDB 3-bin data if available
        if 'rdb_3bin' in galaxy_results['profiles']:
            rdb_data = galaxy_results['profiles']['rdb_3bin']
            
            radii = [point['radius'] for point in rdb_data]
            alpha_fe_vals = [point['alpha_fe'] for point in rdb_data]
            alpha_fe_errors = [point['alpha_fe_error'] for point in rdb_data]
            
            ax.errorbar(radii, alpha_fe_vals, yerr=alpha_fe_errors,
                       fmt='s', alpha=0.7, markersize=8, capsize=4,
                       color='red', ecolor='lightcoral',
                       label=f'RDB 3-bins ({len(radii)} bins)')
            
            all_radii.extend(radii)
            all_alpha_fe.extend(alpha_fe_vals)
            plot_count += 1
            
            # Fit RDB gradient
            if len(radii) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(radii, alpha_fe_vals)
                    r_fit = np.linspace(min(radii), max(radii), 100)
                    alpha_fe_fit = slope * r_fit + intercept
                    
                    fit_color = 'darkred' if p_value < 0.05 else 'lightcoral'
                    ax.plot(r_fit, alpha_fe_fit, '-', color=fit_color, linewidth=2, alpha=0.8,
                           label=f'RDB fit: {slope:.4f}±{std_err:.4f} dex/(R/Re)')
                except Exception:
                    pass
        
        # If we have data, format the plot
        if plot_count > 0:
            # Overall gradient if we have mixed data
            if len(all_radii) > 3:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(all_radii, all_alpha_fe)
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    direction = "positive" if slope > 0 else "negative"
                    
                    stats_text = (f'Combined Gradient Analysis\n'
                                 f'Slope: {slope:.4f} ± {std_err:.4f} dex/(R/Re)\n'
                                 f'R² = {r_value**2:.3f}, p = {p_value:.4f}\n'
                                 f'{direction} gradient ({significance})')
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                           verticalalignment='top', fontsize=11)
                    
                except Exception as e:
                    logger.debug(f"Combined fit error for {galaxy_name}: {e}")
            
            # Formatting
            ax.set_xlabel('Radius (R/Re)', fontsize=14)
            ax.set_ylabel('[α/Fe] (dex)', fontsize=14)
            ax.set_title(f'{galaxy_name} - Alpha/Fe Radial Profile\n'
                        f'Available methods: {", ".join(galaxy_results["profiles"].keys())}',
                        fontsize=16, pad=20)
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set axis limits
            if all_radii:
                ax.set_xlim(0, max(all_radii) * 1.1)
                
                alpha_fe_range = max(all_alpha_fe) - min(all_alpha_fe)
                alpha_fe_center = (max(all_alpha_fe) + min(all_alpha_fe)) / 2
                alpha_fe_padding = max(alpha_fe_range * 0.2, 0.1)
                ax.set_ylim(alpha_fe_center - alpha_fe_range/2 - alpha_fe_padding,
                           alpha_fe_center + alpha_fe_range/2 + alpha_fe_padding)
        else:
            # No data available
            ax.text(0.5, 0.5, f'{galaxy_name}\nNo radial profile data available',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            ax.set_xlim(0, 3)
            ax.set_ylim(-0.5, 0.5)
        
        # Save the plot
        output_file = self.output_dir / f'{galaxy_name}_individual_radial_profile.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created individual radial plot for {galaxy_name}")
        return plot_count > 0
    
    def create_summary_grid(self, all_results):
        """Create a summary grid plot of all galaxies."""
        n_galaxies = len(all_results)
        cols = 5
        rows = (n_galaxies + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, galaxy_results in enumerate(all_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            galaxy_name = galaxy_results['galaxy_name']
            
            # Plot available data
            plot_count = 0
            
            # VNB data
            if 'vnb' in galaxy_results['profiles']:
                vnb_data = galaxy_results['profiles']['vnb']
                radii = [point['radius'] for point in vnb_data]
                alpha_fe_vals = [point['alpha_fe'] for point in vnb_data]
                ax.scatter(radii, alpha_fe_vals, alpha=0.7, s=20, color='blue', label='VNB')
                plot_count += 1
            
            # RDB data
            if 'rdb_3bin' in galaxy_results['profiles']:
                rdb_data = galaxy_results['profiles']['rdb_3bin']
                radii = [point['radius'] for point in rdb_data]
                alpha_fe_vals = [point['alpha_fe'] for point in rdb_data]
                ax.scatter(radii, alpha_fe_vals, alpha=0.7, s=30, color='red', marker='s', label='RDB')
                plot_count += 1
            
            ax.set_title(f'{galaxy_name}', fontsize=10)
            ax.set_xlabel('R/Re', fontsize=9)
            ax.set_ylabel('[α/Fe]', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            if plot_count == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            elif plot_count > 1:
                ax.legend(fontsize=7)
        
        # Hide unused subplots
        for i in range(len(all_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Individual Galaxy Alpha/Fe Radial Profiles - Summary Grid', fontsize=16)
        plt.tight_layout()
        
        summary_file = self.output_dir / 'all_galaxies_individual_radial_summary.png'
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created summary grid: {summary_file}")
    
    def run_all_plots(self):
        """Create individual radial plots for all available galaxies."""
        logger.info("🚀 Starting individual radial plot creation...")
        
        # Get galaxy list
        galaxy_list = self.get_galaxy_list()
        logger.info(f"Found {len(galaxy_list)} galaxies to process")
        
        all_results = []
        successful_plots = 0
        
        for galaxy_name in galaxy_list:
            # Process galaxy
            results = self.load_and_process_galaxy(galaxy_name)
            if results is None:
                continue
            
            all_results.append(results)
            
            # Create individual plot
            success = self.create_individual_radial_plot(results)
            if success:
                successful_plots += 1
        
        # Create summary grid
        if all_results:
            self.create_summary_grid(all_results)
        
        logger.info(f"✅ Individual radial plotting complete!")
        logger.info(f"Successfully created {successful_plots}/{len(galaxy_list)} individual plots")
        logger.info(f"Plots saved in: {self.output_dir}")
        
        return successful_plots


def main():
    """Main function to create all individual radial plots."""
    try:
        plotter = IndividualRadialPlotter()
        success_count = plotter.run_all_plots()
        
        print(f"\n{'='*80}")
        print("INDIVIDUAL RADIAL GRADIENT PLOTTING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully created {success_count} individual radial plots")
        print(f"✅ Summary grid created")
        print(f"📁 All plots saved in: {plotter.output_dir}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in individual radial plotting: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
