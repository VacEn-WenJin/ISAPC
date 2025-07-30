#!/usr/bin/env python3
"""
Simple Individual Radial Gradient Plot Creator
Uses the analyze_single_galaxy function to get all gradient data and creates clean radial plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import stats
from alpha_gradient_analysis import analyze_single_galaxy
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SimpleRadialPlots')

class SimpleRadialPlotter:
    """Creates individual radial gradient plots using analyzed data."""
    
    def __init__(self):
        self.output_dir = Path('./simple_radial_plots')
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
        # Common VCC galaxies based on our successful analyses
        return [
            'VCC0308', 'VCC0667', 'VCC0688', 'VCC0990', 'VCC1049',
            'VCC1146', 'VCC1193', 'VCC1368', 'VCC1410', 'VCC1431',
            'VCC1486', 'VCC1499', 'VCC1549', 'VCC1588', 'VCC1695',
            'VCC1811', 'VCC1890', 'VCC1902', 'VCC1910', 'VCC1949'
        ]
    
    def analyze_galaxy_gradients(self, galaxy_name):
        """Use the existing analyze_single_galaxy function to get gradient data."""
        logger.info(f"Analyzing gradients for {galaxy_name}...")
        
        try:
            # Run the full gradient analysis
            results = analyze_single_galaxy(galaxy_name)
            
            if results is None:
                logger.warning(f"No results returned for {galaxy_name}")
                return None
            
            logger.info(f"✅ Analysis completed for {galaxy_name}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {galaxy_name}: {e}")
            return None
    
    def extract_radial_data_from_results(self, results):
        """Extract radial data points from analysis results."""
        radial_data = {}
        
        # Extract VNB profile if available
        if 'vnb_profile' in results and results['vnb_profile'] is not None:
            vnb_profile = results['vnb_profile']
            if 'bin_radii' in vnb_profile and 'alpha_fe_mean' in vnb_profile:
                # Convert arrays to lists and filter valid data
                radii = vnb_profile['bin_radii']
                alpha_fe = vnb_profile['alpha_fe_mean']
                alpha_fe_error = vnb_profile.get('alpha_fe_error', np.full_like(alpha_fe, 0.05))
                
                # Filter out NaN values
                valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                if np.any(valid_mask):
                    radial_data['vnb'] = {
                        'radii': radii[valid_mask].tolist(),
                        'alpha_fe': alpha_fe[valid_mask].tolist(),
                        'alpha_fe_error': alpha_fe_error[valid_mask].tolist(),
                        'method': 'VNB'
                    }
        
        # Extract RDB profile if available  
        if 'radial_profile' in results and results['radial_profile'] is not None:
            rdb_profile = results['radial_profile']
            if 'bin_radii' in rdb_profile and 'alpha_fe_mean' in rdb_profile:
                # Convert arrays to lists and filter valid data
                radii = rdb_profile['bin_radii']
                alpha_fe = rdb_profile['alpha_fe_mean']
                alpha_fe_error = rdb_profile.get('alpha_fe_error', np.full_like(alpha_fe, 0.05))
                
                # Filter out NaN values
                valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                if np.any(valid_mask):
                    radial_data['rdb'] = {
                        'radii': radii[valid_mask].tolist(),
                        'alpha_fe': alpha_fe[valid_mask].tolist(),
                        'alpha_fe_error': alpha_fe_error[valid_mask].tolist(),
                        'method': 'RDB'
                    }
        
        return radial_data
    
    def create_individual_radial_plot(self, galaxy_name, analysis_results):
        """Create a radial plot for a single galaxy."""
        if analysis_results is None:
            logger.warning(f"No analysis results for {galaxy_name}")
            return False
        
        # Extract radial data
        radial_data = self.extract_radial_data_from_results(analysis_results)
        
        if not radial_data:
            logger.warning(f"No radial data extracted for {galaxy_name}")
            return False
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {'vnb': 'blue', 'rdb': 'red'}
        markers = {'vnb': 'o', 'rdb': 's'}
        
        all_radii = []
        all_alpha_fe = []
        
        # Plot each method
        for method, data in radial_data.items():
            radii = data['radii']
            alpha_fe = data['alpha_fe']
            alpha_fe_error = data['alpha_fe_error']
            
            ax.errorbar(radii, alpha_fe, yerr=alpha_fe_error,
                       fmt=markers[method], alpha=0.7, markersize=8, capsize=4,
                       color=colors[method], ecolor='lightcoral' if method == 'rdb' else 'lightblue',
                       label=f'{data["method"]} ({len(radii)} points)')
            
            all_radii.extend(radii)
            all_alpha_fe.extend(alpha_fe)
            
            # Fit and plot trend line for this method
            if len(radii) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(radii, alpha_fe)
                    
                    r_fit = np.linspace(min(radii), max(radii), 100)
                    alpha_fe_fit = slope * r_fit + intercept
                    
                    fit_color = colors[method]
                    line_style = '--' if method == 'vnb' else '-'
                    
                    ax.plot(r_fit, alpha_fe_fit, line_style, color=fit_color, linewidth=2, alpha=0.7,
                           label=f'{data["method"]} fit: {slope:.4f}±{std_err:.4f}')
                    
                except Exception as e:
                    logger.debug(f"Fit error for {method} in {galaxy_name}: {e}")
        
        # Extract gradient results from analysis
        gradient_info = ""
        if 'gradient_results' in analysis_results:
            grad_res = analysis_results['gradient_results']
            
            if 'multi_method' in grad_res:
                multi_res = grad_res['multi_method']
                gradient_info = (f"Multi-method Analysis Summary:\n")
                
                for method, res in multi_res.items():
                    if isinstance(res, dict) and 'slope' in res:
                        slope = res['slope']
                        slope_err = res.get('slope_error', 0)
                        p_val = res.get('p_value', 1)
                        sig = "sig" if p_val < 0.05 else "n.s."
                        gradient_info += f"{method}: {slope:.4f}±{slope_err:.4f} ({sig})\n"
        
        # Add statistics text box
        if gradient_info:
            ax.text(0.05, 0.95, gradient_info.strip(), transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                   verticalalignment='top', fontsize=10)
        
        # Formatting
        ax.set_xlabel('Radius (R/Re)', fontsize=14)
        ax.set_ylabel('[α/Fe] (dex)', fontsize=14)
        ax.set_title(f'{galaxy_name} - Alpha/Fe Radial Profile\n'
                    f'Methods: {", ".join([d["method"] for d in radial_data.values()])}',
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
        
        # Save the plot
        output_file = self.output_dir / f'{galaxy_name}_radial_profile.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created radial plot for {galaxy_name}: {output_file}")
        return True
    
    def create_summary_grid(self, all_results):
        """Create a summary grid plot."""
        successful_results = [(name, res) for name, res in all_results if res is not None]
        n_galaxies = len(successful_results)
        
        if n_galaxies == 0:
            logger.warning("No successful results for summary grid")
            return
        
        cols = 5
        rows = (n_galaxies + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (galaxy_name, results) in enumerate(successful_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Extract and plot data
            radial_data = self.extract_radial_data_from_results(results)
            
            colors = {'vnb': 'blue', 'rdb': 'red'}
            markers = {'vnb': 'o', 'rdb': 's'}
            
            for method, data in radial_data.items():
                ax.scatter(data['radii'], data['alpha_fe'], 
                          alpha=0.7, s=20, color=colors[method], marker=markers[method])
            
            ax.set_title(f'{galaxy_name}', fontsize=10)
            ax.set_xlabel('R/Re', fontsize=8)
            ax.set_ylabel('[α/Fe]', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        # Hide unused subplots
        for i in range(len(successful_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Alpha/Fe Radial Profiles - All Galaxies Summary', fontsize=16)
        plt.tight_layout()
        
        summary_file = self.output_dir / 'all_galaxies_radial_summary.png'
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created summary grid: {summary_file}")
    
    def run_all_plots(self):
        """Create individual radial plots for all galaxies."""
        logger.info("🚀 Starting simple radial plot creation...")
        
        galaxy_list = self.get_galaxy_list()
        logger.info(f"Processing {len(galaxy_list)} galaxies")
        
        all_results = []
        successful_plots = 0
        
        for galaxy_name in galaxy_list:
            # Analyze galaxy
            results = self.analyze_galaxy_gradients(galaxy_name)
            all_results.append((galaxy_name, results))
            
            if results is not None:
                # Create individual plot
                success = self.create_individual_radial_plot(galaxy_name, results)
                if success:
                    successful_plots += 1
        
        # Create summary grid
        self.create_summary_grid(all_results)
        
        logger.info(f"✅ Simple radial plotting complete!")
        logger.info(f"Successfully created {successful_plots}/{len(galaxy_list)} individual plots")
        logger.info(f"Plots saved in: {self.output_dir}")
        
        return successful_plots


def main():
    """Main function."""
    try:
        plotter = SimpleRadialPlotter()
        success_count = plotter.run_all_plots()
        
        print(f"\n{'='*80}")
        print("SIMPLE RADIAL GRADIENT PLOTTING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully created {success_count} individual radial plots")
        print(f"✅ Summary grid created")
        print(f"📁 All plots saved in: {plotter.output_dir}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in simple radial plotting: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
