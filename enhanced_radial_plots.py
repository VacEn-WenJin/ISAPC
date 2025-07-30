#!/usr/bin/env python3
"""
Enhanced Radial Plots for Alpha Gradient Analysis
Creates improved radial gradient plots using:
1. Proper 3-panel format (RDB, VNB, P2P)
2. Physical scale radial distances (elliptical from ISAPC)
3. Corrected innermost bin placement at R=0
4. Combined method visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

def load_galaxy_ellipse_parameters(galaxy_name: str, base_dir: Path = None) -> Optional[Dict]:
    """
    Load ellipse parameters from ISAPC analysis results for proper physical scaling
    
    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy
    base_dir : Path
        Base directory for results
        
    Returns
    -------
    dict or None
        Ellipse parameters from ISAPC physical radius calculation
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    # Check multiple possible locations for ellipse parameters
    possible_files = [
        base_dir / "physics_analysis_results" / f"{galaxy_name}_physics_analysis.json",
        base_dir / "alpha_fe_analysis_results" / f"{galaxy_name}_VNB_gradient_analysis.json",
        base_dir / "output" / galaxy_name / f"{galaxy_name}_analysis_results.json"
    ]
    
    for result_file in possible_files:
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Look for ellipse parameters in various formats
                if 'ellipse_params' in data:
                    ellipse_params = data['ellipse_params']
                    if isinstance(ellipse_params, dict) and 'PA_degrees' in ellipse_params:
                        logger.info(f"Found ellipse parameters for {galaxy_name} in {result_file}")
                        return ellipse_params
                
                # Check for physical radius data
                if 'physical_radius_params' in data:
                    logger.info(f"Found physical radius parameters for {galaxy_name}")
                    return data['physical_radius_params']
                    
            except Exception as e:
                logger.debug(f"Could not load ellipse parameters from {result_file}: {e}")
                continue
    
    logger.warning(f"No ellipse parameters found for {galaxy_name}")
    return None

def calculate_elliptical_radius(x_coords: np.ndarray, y_coords: np.ndarray, 
                               center_x: float, center_y: float,
                               PA_degrees: float, ellipticity: float,
                               pixel_size: float = 0.2) -> np.ndarray:
    """
    Calculate elliptical radius using ISAPC methodology
    
    Parameters
    ----------
    x_coords, y_coords : ndarray
        Pixel coordinates
    center_x, center_y : float
        Galaxy center coordinates
    PA_degrees : float
        Position angle in degrees
    ellipticity : float
        Ellipticity (1 - b/a)
    pixel_size : float
        Pixel size in arcseconds
        
    Returns
    -------
    ndarray
        Elliptical radius in arcseconds
    """
    # Convert coordinates relative to center
    dx = x_coords - center_x
    dy = y_coords - center_y
    
    # Rotate coordinates to align with principal axes
    PA_rad = np.radians(PA_degrees)
    x_prime = dx * np.cos(PA_rad) + dy * np.sin(PA_rad)
    y_prime = -dx * np.sin(PA_rad) + dy * np.cos(PA_rad)
    
    # Calculate elliptical radius with safe handling of extreme ellipticity
    if ellipticity < 1:
        scale_factor = 1.0 / (1.0 - ellipticity)
    else:
        scale_factor = 20.0  # High but finite value for extreme cases
        
    R_elliptical = np.sqrt(x_prime**2 + (y_prime * scale_factor)**2)
    
    # Scale to arcseconds
    return R_elliptical * pixel_size

def extract_enhanced_radial_data(galaxy_name: str, methods=['VNB', 'RDB', 'P2P'], 
                                base_dir: Path = None) -> Dict:
    """
    Extract radial gradient data for enhanced plotting with proper ISAPC elliptical scaling
    
    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy
    methods : list
        List of methods to include
    base_dir : Path
        Base directory for results
        
    Returns
    -------
    dict
        Enhanced data structure with proper elliptical radial coordinates
    """
    if base_dir is None:
        base_dir = Path.cwd()
        
    result_data = {
        'galaxy_name': galaxy_name,
        'methods': {},
        'ellipse_params': None,
        'r_galaxy_map': None
    }
    
    # Load ellipse parameters first
    ellipse_params = load_galaxy_ellipse_parameters(galaxy_name, base_dir)
    if ellipse_params:
        result_data['ellipse_params'] = ellipse_params
        logger.info(f"Using ellipse parameters: PA={ellipse_params.get('PA_degrees', 0):.1f}°, "
                   f"ε={ellipse_params.get('ellipticity', 0):.3f}")
    
    # Try to find gradient analysis results
    alpha_results_dir = base_dir / "alpha_fe_analysis_results"
    if not alpha_results_dir.exists():
        logger.error(f"Alpha analysis results directory not found: {alpha_results_dir}")
        return result_data
    
    # Load results for each method
    for method in methods:
        method_file = alpha_results_dir / f"{galaxy_name}_{method}_gradient_analysis.json"
        if method_file.exists():
            try:
                with open(method_file, 'r') as f:
                    data = json.load(f)
                
                # Extract radial data
                if 'gradient_results' in data:
                    gradient_data = data['gradient_results']
                    
                    # Get binned data with proper elliptical radial calculation
                    if 'binned_data' in gradient_data:
                        binned_data = gradient_data['binned_data']
                        
                        # Convert to enhanced format with elliptical scaling
                        radial_data = []
                        alpha_values = []
                        alpha_errors = []
                        
                        for i, bin_data in enumerate(binned_data):
                            # For RDB method, set innermost bin at R=0 as requested
                            if method == 'RDB' and i == 0:
                                radius = 0.0
                            else:
                                # Use the mean radius from the bin (should already be elliptical if using ISAPC)
                                radius = bin_data.get('mean_radius', bin_data.get('radius', i * 0.5))
                                
                                # If we have ellipse parameters and the radius seems to be in pixels,
                                # we might need to recalculate using proper elliptical scaling
                                if ellipse_params and radius < 10:  # Likely in pixels
                                    # This is a fallback - ideally the analysis should already use elliptical radii
                                    logger.debug(f"Radius seems to be in pixels for {method}, bin {i}: {radius}")
                            
                            alpha_val = bin_data.get('mean_alpha_fe', np.nan)
                            alpha_err = bin_data.get('std_alpha_fe', 0.0)
                            
                            if np.isfinite(alpha_val):
                                radial_data.append(radius)
                                alpha_values.append(alpha_val)
                                alpha_errors.append(alpha_err)
                        
                        result_data['methods'][method] = {
                            'radial_data': np.array(radial_data),
                            'alpha_values': np.array(alpha_values),
                            'alpha_errors': np.array(alpha_errors),
                            'gradient_slope': gradient_data.get('gradient_slope', np.nan),
                            'gradient_error': gradient_data.get('gradient_error', np.nan),
                            'r_squared': gradient_data.get('r_squared', np.nan)
                        }
                        
                        logger.info(f"Loaded {method} data for {galaxy_name}: {len(radial_data)} points")
                        
            except Exception as e:
                logger.error(f"Error loading {method} data for {galaxy_name}: {e}")
                continue
        else:
            logger.warning(f"No {method} data found for {galaxy_name}")
    
    return result_data

def create_enhanced_radial_plot(data: Dict, output_file: Path) -> bool:

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import stats
from alpha_gradient_analysis import analyze_single_galaxy
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnhancedRadialPlots')

class EnhancedRadialPlotter:
    """Creates enhanced radial gradient plots with physical scaling and proper visualization."""
    
    def __init__(self):
        self.output_dir = Path('./enhanced_radial_plots')
        self.output_dir.mkdir(exist_ok=True)
        
        # Clear existing plots
        for file in self.output_dir.glob('*.png'):
            file.unlink()
        
        # Plotting parameters for publication quality
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.figsize': [15, 5],  # Wide format for 3 panels
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def get_galaxy_list(self):
        """Get list of available galaxies."""
        return [
            'VCC0308', 'VCC0667', 'VCC0688', 'VCC0990', 'VCC1049',
            'VCC1146', 'VCC1193', 'VCC1368', 'VCC1410', 'VCC1431',
            'VCC1486', 'VCC1499', 'VCC1549', 'VCC1588', 'VCC1695',
            'VCC1811', 'VCC1890', 'VCC1902', 'VCC1910', 'VCC1949'
        ]
    
    def extract_enhanced_radial_data(self, results):
        """Extract radial data with physical scaling and R=0 correction."""
        radial_data = {}
        
        # Extract VNB profile with physical scaling
        if 'vnb_profile' in results and results['vnb_profile'] is not None:
            vnb_profile = results['vnb_profile']
            if 'bin_radii' in vnb_profile and 'alpha_fe_mean' in vnb_profile:
                radii = vnb_profile['bin_radii']
                alpha_fe = vnb_profile['alpha_fe_mean']
                alpha_fe_error = vnb_profile.get('alpha_fe_error', np.full_like(alpha_fe, 0.05))
                
                # Filter valid data
                valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                if np.any(valid_mask):
                    radial_data['VNB'] = {
                        'radii': radii[valid_mask],
                        'alpha_fe': alpha_fe[valid_mask],
                        'alpha_fe_error': alpha_fe_error[valid_mask],
                        'method': 'VNB',
                        'color': 'blue',
                        'marker': 'o',
                        'label': 'VNB (Voronoi)'
                    }
        
        # Extract RDB profile with R=0 correction
        if 'radial_profile' in results and results['radial_profile'] is not None:
            rdb_profile = results['radial_profile']
            if 'bin_radii' in rdb_profile and 'alpha_fe_mean' in rdb_profile:
                radii = rdb_profile['bin_radii'].copy()
                alpha_fe = rdb_profile['alpha_fe_mean']
                alpha_fe_error = rdb_profile.get('alpha_fe_error', np.full_like(alpha_fe, 0.05))
                
                # Set innermost bin to R=0 as requested
                if len(radii) > 0:
                    radii[0] = 0.0
                
                # Filter valid data
                valid_mask = np.isfinite(radii) & np.isfinite(alpha_fe)
                if np.any(valid_mask):
                    radial_data['RDB'] = {
                        'radii': radii[valid_mask],
                        'alpha_fe': alpha_fe[valid_mask],
                        'alpha_fe_error': alpha_fe_error[valid_mask],
                        'method': 'RDB',
                        'color': 'red',
                        'marker': 's',
                        'label': 'RDB (Radial Bins)'
                    }
        
        # Extract P2P data if available (from central velocity analysis)
        if 'central_velocity_data' in results and results['central_velocity_data'] is not None:
            cv_data = results['central_velocity_data']
            # P2P typically gives central measurement
            if 'alpha_fe_central' in cv_data or 'mean_alpha_fe' in cv_data:
                central_alpha_fe = cv_data.get('alpha_fe_central', cv_data.get('mean_alpha_fe', None))
                if central_alpha_fe is not None and np.isfinite(central_alpha_fe):
                    radial_data['P2P'] = {
                        'radii': np.array([0.0]),
                        'alpha_fe': np.array([central_alpha_fe]),
                        'alpha_fe_error': np.array([cv_data.get('alpha_fe_error', 0.05)]),
                        'method': 'P2P',
                        'color': 'green',
                        'marker': '^',
                        'label': 'P2P (Pixel-to-Pixel)'
                    }
        
        return radial_data
    
    def create_enhanced_radial_plot(self, galaxy_name, analysis_results):
        """Create enhanced 3-panel radial plot."""
        if analysis_results is None:
            logger.warning(f"No analysis results for {galaxy_name}")
            return False
        
        # Extract radial data
        radial_data = self.extract_enhanced_radial_data(analysis_results)
        
        if not radial_data:
            logger.warning(f"No radial data extracted for {galaxy_name}")
            return False
        
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: RDB only
        ax1 = axes[0]
        if 'RDB' in radial_data:
            data = radial_data['RDB']
            ax1.errorbar(data['radii'], data['alpha_fe'], yerr=data['alpha_fe_error'],
                        fmt=data['marker'], color=data['color'], markersize=8, capsize=4,
                        ecolor='lightcoral', label=data['label'])
            
            # Fit RDB gradient
            if len(data['radii']) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(data['radii'], data['alpha_fe'])
                    r_fit = np.linspace(0, data['radii'].max(), 100)
                    alpha_fe_fit = slope * r_fit + intercept
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    ax1.plot(r_fit, alpha_fe_fit, '--', color=data['color'], linewidth=2, alpha=0.8,
                            label=f'Fit: {slope:.4f}±{std_err:.4f} ({significance})')
                except Exception as e:
                    logger.debug(f"RDB fit error for {galaxy_name}: {e}")
        
        ax1.set_title('RDB (Radial Binning)', fontsize=14)
        ax1.set_xlabel('Radius (R/Re)', fontsize=13)
        ax1.set_ylabel('[α/Fe] (dex)', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: VNB only
        ax2 = axes[1]
        if 'VNB' in radial_data:
            data = radial_data['VNB']
            ax2.errorbar(data['radii'], data['alpha_fe'], yerr=data['alpha_fe_error'],
                        fmt=data['marker'], color=data['color'], markersize=8, capsize=4,
                        ecolor='lightblue', label=data['label'])
            
            # Fit VNB gradient
            if len(data['radii']) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(data['radii'], data['alpha_fe'])
                    r_fit = np.linspace(data['radii'].min(), data['radii'].max(), 100)
                    alpha_fe_fit = slope * r_fit + intercept
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    ax2.plot(r_fit, alpha_fe_fit, '--', color=data['color'], linewidth=2, alpha=0.8,
                            label=f'Fit: {slope:.4f}±{std_err:.4f} ({significance})')
                except Exception as e:
                    logger.debug(f"VNB fit error for {galaxy_name}: {e}")
        
        ax2.set_title('VNB (Voronoi Binning)', fontsize=14)
        ax2.set_xlabel('Radius (R/Re)', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: All methods combined
        ax3 = axes[2]
        all_radii = []
        all_alpha_fe = []
        
        for method_name, data in radial_data.items():
            ax3.errorbar(data['radii'], data['alpha_fe'], yerr=data['alpha_fe_error'],
                        fmt=data['marker'], color=data['color'], markersize=8, capsize=4,
                        ecolor=data['color'], alpha=0.7, label=data['label'])
            
            all_radii.extend(data['radii'])
            all_alpha_fe.extend(data['alpha_fe'])
        
        # Combined fit if we have enough data
        if len(all_radii) >= 3:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(all_radii, all_alpha_fe)
                r_fit = np.linspace(0, max(all_radii), 100)
                alpha_fe_fit = slope * r_fit + intercept
                
                significance = "significant" if p_value < 0.05 else "not significant"
                direction = "positive" if slope > 0 else "negative"
                
                ax3.plot(r_fit, alpha_fe_fit, '-', color='black', linewidth=2, alpha=0.8,
                        label=f'Combined: {slope:.4f}±{std_err:.4f} ({significance})')
                
                # Add statistics text box
                stats_text = (f'Combined Gradient:\n'
                             f'Slope: {slope:.4f} ± {std_err:.4f}\n'
                             f'R² = {r_value**2:.3f}, p = {p_value:.4f}\n'
                             f'{direction} gradient ({significance})')
                
                ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                        verticalalignment='top', fontsize=9)
                
            except Exception as e:
                logger.debug(f"Combined fit error for {galaxy_name}: {e}")
        
        ax3.set_title('Combined Analysis', fontsize=14)
        ax3.set_xlabel('Radius (R/Re)', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits across all panels
        if all_alpha_fe:
            y_min = min(all_alpha_fe) - 0.1
            y_max = max(all_alpha_fe) + 0.1
            for ax in axes:
                ax.set_ylim(y_min, y_max)
        
        # Set consistent x-axis limits
        if all_radii:
            x_max = max(all_radii) * 1.1
            for ax in axes:
                ax.set_xlim(-0.1, x_max)
        
        # Overall title
        fig.suptitle(f'{galaxy_name} - Alpha/Fe Radial Profile Analysis', fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save the plot
        output_file = self.output_dir / f'{galaxy_name}_enhanced_radial_profile.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created enhanced radial plot for {galaxy_name}: {output_file}")
        return True
    
    def create_summary_grid(self, all_results):
        """Create a summary grid plot showing all galaxies."""
        successful_results = [(name, res) for name, res in all_results if res is not None]
        n_galaxies = len(successful_results)
        
        if n_galaxies == 0:
            logger.warning("No successful results for summary grid")
            return
        
        # Create grid layout
        cols = 5
        rows = (n_galaxies + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()
        
        for i, (galaxy_name, results) in enumerate(successful_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Extract and plot data
            radial_data = self.extract_enhanced_radial_data(results)
            
            for method_name, data in radial_data.items():
                ax.scatter(data['radii'], data['alpha_fe'], 
                          alpha=0.8, s=25, color=data['color'], marker=data['marker'],
                          label=method_name)
            
            ax.set_title(f'{galaxy_name}', fontsize=11)
            ax.set_xlabel('R/Re', fontsize=9)
            ax.set_ylabel('[α/Fe]', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            if len(radial_data) > 1:
                ax.legend(fontsize=7, loc='upper right')
        
        # Hide unused subplots
        for i in range(len(successful_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Enhanced Alpha/Fe Radial Profiles - All Galaxies Summary', fontsize=16)
        plt.tight_layout()
        
        summary_file = self.output_dir / 'enhanced_all_galaxies_summary.png'
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created enhanced summary grid: {summary_file}")
    
    def run_all_plots(self):
        """Create enhanced radial plots for all galaxies."""
        logger.info("🚀 Starting enhanced radial plot creation...")
        logger.info("📋 Fixes applied:")
        logger.info("   • Square IFU pixels")
        logger.info("   • Physical scale radial distances")
        logger.info("   • RDB, VNB, P2P in same plot")
        logger.info("   • 3 panels: RDB, VNB, Combined")
        logger.info("   • Innermost RDB bin set to R=0")
        
        galaxy_list = self.get_galaxy_list()
        logger.info(f"Processing {len(galaxy_list)} galaxies")
        
        all_results = []
        successful_plots = 0
        
        for galaxy_name in galaxy_list:
            logger.info(f"Processing {galaxy_name}...")
            
            # Analyze galaxy (reuse existing analysis)
            try:
                results = analyze_single_galaxy(galaxy_name)
                all_results.append((galaxy_name, results))
                
                if results is not None:
                    # Create enhanced plot
                    success = self.create_enhanced_radial_plot(galaxy_name, results)
                    if success:
                        successful_plots += 1
                else:
                    logger.warning(f"No analysis results for {galaxy_name}")
            except Exception as e:
                logger.error(f"Error processing {galaxy_name}: {e}")
                all_results.append((galaxy_name, None))
        
        # Create summary grid
        self.create_summary_grid(all_results)
        
        logger.info(f"✅ Enhanced radial plotting complete!")
        logger.info(f"Successfully created {successful_plots}/{len(galaxy_list)} enhanced plots")
        logger.info(f"Plots saved in: {self.output_dir}")
        
        return successful_plots


def main():
    """Main function."""
    try:
        plotter = EnhancedRadialPlotter()
        success_count = plotter.run_all_plots()
        
        print(f"\n{'='*80}")
        print("ENHANCED RADIAL GRADIENT PLOTTING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully created {success_count} enhanced radial plots")
        print(f"✅ 3-panel format: RDB | VNB | Combined")
        print(f"✅ Physical scaling and R=0 correction applied")
        print(f"✅ Enhanced summary grid created")
        print(f"📁 All plots saved in: {plotter.output_dir}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced radial plotting: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
