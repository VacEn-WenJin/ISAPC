#!/usr/bin/env python3
"""
Individual Radial Gradient Plot Creator
Creates dedicated radial gradient plots for each galaxy showing alpha/Fe vs radius.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import stats
import glob
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RadialPlots')

class RadialGradientPlotter:
    """Creates individual radial gradient plots for each galaxy."""
    
    def __init__(self):
        self.results_dir = Path('./alpha_fe_analysis_results')
        self.output_dir = Path('./alpha_fe_radial_plots')
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
            'figure.figsize': [10, 8]
        })
    
    def find_latest_analysis(self):
        """Find the most recent analysis directory."""
        analysis_dirs = [d for d in self.results_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('analysis_')]
        if not analysis_dirs:
            raise FileNotFoundError("No analysis directories found")
        
        latest_dir = max(analysis_dirs, key=lambda x: x.name)
        logger.info(f"Using analysis directory: {latest_dir}")
        return latest_dir
    
    def load_galaxy_data(self, galaxy_name, analysis_dir):
        """Load saved galaxy analysis data."""
        galaxy_file = analysis_dir / galaxy_name / f'{galaxy_name}_alpha_fe_analysis.npz'
        
        if not galaxy_file.exists():
            logger.warning(f"No data file found for {galaxy_name}")
            return None
        
        try:
            data = np.load(galaxy_file, allow_pickle=True)
            return {
                'galaxy_name': galaxy_name,
                'galaxy_type': str(data.get('galaxy_type', 'Unknown')),
                'redshift': float(data.get('redshift', 0.0)),
                'alpha_fe_map': data.get('alpha_fe_map'),
                'alpha_fe_error_map': data.get('alpha_fe_error_map'),
                'radius_map': data.get('radius_map'),
                'gradient_slope': float(data.get('gradient_slope', np.nan)),
                'gradient_error': float(data.get('gradient_error', np.nan)),
                'gradient_pvalue': float(data.get('gradient_pvalue', np.nan)),
                'valid_mask': data.get('valid_mask', None)
            }
        except Exception as e:
            logger.error(f"Error loading data for {galaxy_name}: {e}")
            return None
    
    def create_radial_gradient_plot(self, galaxy_data):
        """Create a dedicated radial gradient plot for a single galaxy."""
        galaxy_name = galaxy_data['galaxy_name']
        galaxy_type = galaxy_data['galaxy_type']
        
        # Extract valid data points
        alpha_fe_map = galaxy_data['alpha_fe_map']
        alpha_fe_error_map = galaxy_data['alpha_fe_error_map']
        radius_map = galaxy_data['radius_map']
        valid_mask = galaxy_data['valid_mask']
        
        if valid_mask is None:
            valid_mask = (~np.isnan(alpha_fe_map)) & (~np.isnan(radius_map))
        
        if not np.any(valid_mask):
            logger.warning(f"No valid data points for {galaxy_name}")
            return False
        
        # Extract valid points
        alpha_fe_valid = alpha_fe_map[valid_mask]
        alpha_fe_error_valid = alpha_fe_error_map[valid_mask] if alpha_fe_error_map is not None else np.full_like(alpha_fe_valid, 0.05)
        radius_valid = radius_map[valid_mask]
        
        # Remove any remaining NaN values
        finite_mask = np.isfinite(alpha_fe_valid) & np.isfinite(radius_valid) & np.isfinite(alpha_fe_error_valid)
        if not np.any(finite_mask):
            logger.warning(f"No finite data points for {galaxy_name}")
            return False
        
        alpha_fe_valid = alpha_fe_valid[finite_mask]
        alpha_fe_error_valid = alpha_fe_error_valid[finite_mask]
        radius_valid = radius_valid[finite_mask]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot with error bars
        scatter = ax.errorbar(radius_valid, alpha_fe_valid, yerr=alpha_fe_error_valid,
                             fmt='o', alpha=0.6, markersize=4, capsize=3,
                             color='steelblue', ecolor='lightblue',
                             label=f'Data points ({len(alpha_fe_valid)})')
        
        # Fit and plot gradient line if we have enough points
        if len(alpha_fe_valid) >= 3:
            try:
                # Linear fit
                slope, intercept, r_value, p_value, std_err = stats.linregress(radius_valid, alpha_fe_valid)
                
                # Create fit line
                r_fit = np.linspace(radius_valid.min(), radius_valid.max(), 100)
                alpha_fe_fit = slope * r_fit + intercept
                
                # Determine gradient significance and color
                if p_value < 0.05:
                    if slope > 0:
                        fit_color = 'red'
                        gradient_text = f'Positive gradient (significant)'
                    else:
                        fit_color = 'darkgreen'
                        gradient_text = f'Negative gradient (significant)'
                else:
                    fit_color = 'gray'
                    if slope > 0:
                        gradient_text = f'Positive gradient (not significant)'
                    else:
                        gradient_text = f'Negative gradient (not significant)'
                
                ax.plot(r_fit, alpha_fe_fit, '--', color=fit_color, linewidth=2,
                       label=f'Linear fit (slope = {slope:.4f} ± {std_err:.4f})')
                
                # Add statistics text box
                stats_text = (f'Slope: {slope:.4f} ± {std_err:.4f} dex/(R/Re)\n'
                             f'R² = {r_value**2:.3f}\n'
                             f'p-value: {p_value:.4f}\n'
                             f'{gradient_text}')
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=10)
                
            except Exception as e:
                logger.warning(f"Could not fit gradient for {galaxy_name}: {e}")
        
        # Formatting
        ax.set_xlabel('Radius (R/Re)', fontsize=14)
        ax.set_ylabel('[α/Fe] (dex)', fontsize=14)
        ax.set_title(f'{galaxy_name} - Radial Alpha/Fe Gradient\n'
                    f'Galaxy Type: {galaxy_type}', fontsize=16, pad=20)
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable axis limits
        ax.set_xlim(0, max(radius_valid.max() * 1.1, 2.0))
        
        alpha_fe_range = alpha_fe_valid.max() - alpha_fe_valid.min()
        alpha_fe_center = (alpha_fe_valid.max() + alpha_fe_valid.min()) / 2
        alpha_fe_padding = max(alpha_fe_range * 0.2, 0.1)
        ax.set_ylim(alpha_fe_center - alpha_fe_range/2 - alpha_fe_padding,
                   alpha_fe_center + alpha_fe_range/2 + alpha_fe_padding)
        
        # Save the plot
        output_file = self.output_dir / f'{galaxy_name}_radial_gradient.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created radial plot for {galaxy_name}")
        return True
    
    def create_summary_plot(self, all_galaxy_data):
        """Create a summary plot showing all galaxies' gradients."""
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, galaxy_data in enumerate(all_galaxy_data[:20]):  # Maximum 20 plots
            if i >= len(axes):
                break
            
            ax = axes[i]
            galaxy_name = galaxy_data['galaxy_name']
            galaxy_type = galaxy_data['galaxy_type']
            
            # Extract valid data points
            alpha_fe_map = galaxy_data['alpha_fe_map']
            radius_map = galaxy_data['radius_map']
            valid_mask = galaxy_data['valid_mask']
            
            if valid_mask is None:
                valid_mask = (~np.isnan(alpha_fe_map)) & (~np.isnan(radius_map))
            
            if not np.any(valid_mask):
                ax.text(0.5, 0.5, f'{galaxy_name}\nNo data', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{galaxy_name} ({galaxy_type})', fontsize=10)
                continue
            
            alpha_fe_valid = alpha_fe_map[valid_mask]
            radius_valid = radius_map[valid_mask]
            
            # Remove NaN values
            finite_mask = np.isfinite(alpha_fe_valid) & np.isfinite(radius_valid)
            if not np.any(finite_mask):
                ax.text(0.5, 0.5, f'{galaxy_name}\nNo finite data', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{galaxy_name} ({galaxy_type})', fontsize=10)
                continue
            
            alpha_fe_valid = alpha_fe_valid[finite_mask]
            radius_valid = radius_valid[finite_mask]
            
            # Plot data points
            ax.scatter(radius_valid, alpha_fe_valid, alpha=0.6, s=10, color='steelblue')
            
            # Fit gradient if possible
            if len(alpha_fe_valid) >= 3:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(radius_valid, alpha_fe_valid)
                    
                    r_fit = np.linspace(radius_valid.min(), radius_valid.max(), 50)
                    alpha_fe_fit = slope * r_fit + intercept
                    
                    if p_value < 0.05:
                        fit_color = 'red' if slope > 0 else 'darkgreen'
                    else:
                        fit_color = 'gray'
                    
                    ax.plot(r_fit, alpha_fe_fit, '--', color=fit_color, linewidth=1.5)
                    
                except Exception:
                    pass
            
            ax.set_title(f'{galaxy_name} ({galaxy_type})', fontsize=10)
            ax.set_xlabel('R/Re', fontsize=9)
            ax.set_ylabel('[α/Fe]', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for i in range(len(all_galaxy_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Alpha/Fe Radial Gradients - All Galaxies Summary', fontsize=16)
        plt.tight_layout()
        
        summary_file = self.output_dir / 'all_galaxies_radial_summary.png'
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created summary plot: {summary_file}")
    
    def run_all_plots(self):
        """Create radial gradient plots for all analyzed galaxies."""
        logger.info("🚀 Starting radial gradient plot creation...")
        
        # Find latest analysis
        analysis_dir = self.find_latest_analysis()
        
        # Get list of analyzed galaxies
        galaxy_dirs = [d for d in analysis_dir.iterdir() if d.is_dir()]
        galaxy_names = [d.name for d in galaxy_dirs]
        
        logger.info(f"Found {len(galaxy_names)} galaxies to plot")
        
        all_galaxy_data = []
        successful_plots = 0
        
        for galaxy_name in sorted(galaxy_names):
            logger.info(f"Processing {galaxy_name}...")
            
            # Load galaxy data
            galaxy_data = self.load_galaxy_data(galaxy_name, analysis_dir)
            if galaxy_data is None:
                continue
            
            all_galaxy_data.append(galaxy_data)
            
            # Create individual plot
            success = self.create_radial_gradient_plot(galaxy_data)
            if success:
                successful_plots += 1
        
        # Create summary plot
        if all_galaxy_data:
            self.create_summary_plot(all_galaxy_data)
        
        logger.info(f"✅ Completed radial plotting!")
        logger.info(f"Successfully created {successful_plots}/{len(galaxy_names)} individual plots")
        logger.info(f"Plots saved in: {self.output_dir}")
        
        return successful_plots


def main():
    """Main function to create all radial gradient plots."""
    try:
        plotter = RadialGradientPlotter()
        success_count = plotter.run_all_plots()
        
        print(f"\n{'='*80}")
        print("RADIAL GRADIENT PLOTTING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully created {success_count} individual radial plots")
        print(f"✅ Summary plot created")
        print(f"📁 All plots saved in: {plotter.output_dir}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in radial plotting: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
