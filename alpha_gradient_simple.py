#!/usr/bin/env python3
"""
Simplified Alpha Abundance Gradient Analysis

This script creates radial profiles from 2D alpha/Fe data using simple geometric binning
and fits linear gradients following Liu Yiqing 2016 and Zhengzheng 2019 methodology.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy import stats
import os
import sys
import logging

def setup_logging():
    """Setup logging for gradient analysis"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def load_galaxy_alpha_fe_data(galaxy_name, analysis_dir="alpha_fe_analysis_results/analysis_20250720_091707"):
    """Load 2D alpha/Fe data for a galaxy"""
    try:
        npz_path = f"{analysis_dir}/{galaxy_name}/{galaxy_name}_alpha_fe_analysis.npz"
        
        if not os.path.exists(npz_path):
            logger.error(f"Alpha/Fe data not found for {galaxy_name}: {npz_path}")
            return None
            
        data = np.load(npz_path, allow_pickle=True)
        
        result = {
            'galaxy_name': str(data['galaxy_name']),
            'galaxy_type': str(data['galaxy_type']),
            'alpha_fe_2d': data['alpha_fe_2d'],
            'alpha_fe_errors': data['alpha_fe_errors'],
            'n_successful': int(data['n_successful_alpha_fe']),
            'mean_alpha_fe': float(data['mean_alpha_fe']),
            'std_alpha_fe': float(data['std_alpha_fe'])
        }
        
        logger.info(f"Loaded alpha/Fe data for {galaxy_name}: {result['alpha_fe_2d'].shape}, {result['n_successful']} valid pixels")
        return result
        
    except Exception as e:
        logger.error(f"Error loading alpha/Fe data for {galaxy_name}: {e}")
        return None

def get_effective_radius(galaxy_name):
    """Get effective radius from RDB data or use default"""
    try:
        rdb_path = f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_RDB_results.npz"
        
        if os.path.exists(rdb_path):
            rdb_data = np.load(rdb_path, allow_pickle=True)
            distance_info = rdb_data['distance'].item()
            return distance_info['effective_radius']
        else:
            # Default effective radius estimates for galaxies without RDB data
            defaults = {
                'VCC0308': 12.0, 'VCC0667': 8.0, 'VCC0688': 6.0, 'VCC0990': 5.0, 'VCC1049': 7.0,
                'VCC1146': 13.0, 'VCC1193': 8.0, 'VCC1368': 10.0, 'VCC1410': 6.0, 'VCC1431': 9.0,
                'VCC1486': 7.0, 'VCC1549': 14.0, 'VCC1588': 14.0, 'VCC1695': 9.0, 'VCC1811': 12.0,
                'VCC1890': 8.0, 'VCC1902': 12.0, 'VCC1910': 10.0, 'VCC1949': 9.0
            }
            return defaults.get(galaxy_name, 10.0)  # Default 10 kpc if not found
    except Exception as e:
        logger.warning(f"Could not get effective radius for {galaxy_name}: {e}")
        return 10.0

def create_radial_bins(alpha_fe_2d, center=None, n_bins=6, max_radius_factor=2.0):
    """
    Create simple concentric circular bins for radial analysis
    
    Parameters:
    -----------
    alpha_fe_2d : np.ndarray
        2D alpha/Fe data array
    center : tuple, optional
        (y, x) center coordinates. If None, uses image center
    n_bins : int
        Number of radial bins
    max_radius_factor : float
        Maximum radius as fraction of half the image size
    
    Returns:
    --------
    dict
        Bin information
    """
    try:
        ny, nx = alpha_fe_2d.shape
        
        if center is None:
            center_y, center_x = ny // 2, nx // 2
        else:
            center_y, center_x = center
        
        # Create coordinate grids
        y, x = np.ogrid[:ny, :nx]
        
        # Calculate distance from center for each pixel
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Define bin edges
        max_radius = min(center_x, center_y, nx - center_x, ny - center_y) * max_radius_factor
        bin_edges = np.linspace(0, max_radius, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create bin masks
        bin_masks = []
        for i in range(n_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            bin_masks.append(mask)
        
        return {
            'bin_centers': bin_centers,
            'bin_edges': bin_edges,
            'bin_masks': bin_masks,
            'center': (center_y, center_x),
            'distances': distances
        }
        
    except Exception as e:
        logger.error(f"Error creating radial bins: {e}")
        return None

def calculate_radial_profile_simple(alpha_fe_data, effective_radius, min_pixels_per_bin=10):
    """
    Calculate radial alpha/Fe profile using simple geometric binning
    
    Following Liu Yiqing 2016 and Zhengzheng 2019 methodology
    """
    try:
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        alpha_fe_errors = alpha_fe_data['alpha_fe_errors']
        
        # Create radial bins
        bin_info = create_radial_bins(alpha_fe_2d, n_bins=6)
        if bin_info is None:
            return None
        
        # Convert pixel radii to physical units (kpc) then to effective radius units
        # Assume 1 pixel = 0.2 arcsec, distance = 17 Mpc (typical for Virgo)
        pixel_scale_kpc = 0.2 * (17 * 1000) / 206265  # arcsec to kpc conversion
        
        radii_kpc = bin_info['bin_centers'] * pixel_scale_kpc
        radii_re = radii_kpc / effective_radius
        
        # Initialize profile arrays
        n_bins = len(bin_info['bin_centers'])
        profile = {
            'bin_radii_pixels': bin_info['bin_centers'],
            'bin_radii_kpc': radii_kpc,
            'bin_radii_re': radii_re,
            'alpha_fe_mean': np.full(n_bins, np.nan),
            'alpha_fe_median': np.full(n_bins, np.nan),
            'alpha_fe_std': np.full(n_bins, np.nan),
            'alpha_fe_error': np.full(n_bins, np.nan),
            'n_pixels': np.zeros(n_bins, dtype=int),
            'valid_bins': [],
            'effective_radius': effective_radius,
            'galaxy_name': alpha_fe_data['galaxy_name'],
            'bin_info': bin_info
        }
        
        # Calculate statistics for each bin
        for i, mask in enumerate(bin_info['bin_masks']):
            # Extract data for this bin
            alpha_fe_bin = alpha_fe_2d[mask]
            errors_bin = alpha_fe_errors[mask]
            
            # Filter valid data
            valid_mask = np.isfinite(alpha_fe_bin) & np.isfinite(errors_bin)
            alpha_fe_valid = alpha_fe_bin[valid_mask]
            errors_valid = errors_bin[valid_mask]
            
            n_valid = len(alpha_fe_valid)
            profile['n_pixels'][i] = n_valid
            
            if n_valid < min_pixels_per_bin:
                logger.debug(f"Bin {i}: insufficient pixels ({n_valid} < {min_pixels_per_bin})")
                continue
            
            # Calculate statistics
            profile['alpha_fe_mean'][i] = np.mean(alpha_fe_valid)
            profile['alpha_fe_median'][i] = np.median(alpha_fe_valid)
            profile['alpha_fe_std'][i] = np.std(alpha_fe_valid)
            
            # Error propagation
            measurement_error = np.sqrt(np.mean(errors_valid**2))
            scatter_error = np.std(alpha_fe_valid) / np.sqrt(n_valid)
            total_error = np.sqrt(measurement_error**2 + scatter_error**2)
            profile['alpha_fe_error'][i] = total_error
            
            profile['valid_bins'].append(i)
            
            logger.debug(f"Bin {i}: R={radii_re[i]:.2f} Re, [α/Fe]={profile['alpha_fe_mean'][i]:.3f}±{total_error:.3f}, N={n_valid}")
        
        logger.info(f"Calculated radial profile for {alpha_fe_data['galaxy_name']}: {len(profile['valid_bins'])}/{n_bins} valid bins")
        return profile
        
    except Exception as e:
        logger.error(f"Error calculating radial profile: {e}")
        import traceback
        traceback.print_exc()
        return None

def fit_alpha_gradient(profile, max_radius=2.0):
    """Fit linear gradient to alpha/Fe profile"""
    try:
        valid_bins = profile['valid_bins']
        
        if len(valid_bins) < 3:
            logger.warning(f"Insufficient bins for gradient fitting: {len(valid_bins)}")
            return None
        
        # Get data for valid bins
        radii = profile['bin_radii_re'][valid_bins]
        alpha_fe = profile['alpha_fe_mean'][valid_bins]
        errors = profile['alpha_fe_error'][valid_bins]
        
        # Apply radius cut
        radius_mask = radii <= max_radius
        if np.sum(radius_mask) < 3:
            logger.warning(f"Insufficient bins within R < {max_radius} Re")
            return None
        
        radii_fit = radii[radius_mask]
        alpha_fe_fit = alpha_fe[radius_mask]
        errors_fit = errors[radius_mask]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(radii_fit, alpha_fe_fit)
        
        # Calculate additional statistics
        predicted = slope * radii_fit + intercept
        residuals = alpha_fe_fit - predicted
        chi_squared = np.sum((residuals / errors_fit)**2) if np.all(errors_fit > 0) else np.sum(residuals**2)
        reduced_chi_squared = chi_squared / (len(radii_fit) - 2) if len(radii_fit) > 2 else chi_squared
        
        # Classification
        significance_ratio = abs(slope) / std_err if std_err > 0 else 0
        
        if p_value < 0.01 and significance_ratio > 3:
            significance = 'highly_significant'
        elif p_value < 0.05 and significance_ratio > 2:
            significance = 'significant'
        elif p_value < 0.1:
            significance = 'marginal'
        else:
            significance = 'not_significant'
        
        # Physical interpretation
        if significance in ['highly_significant', 'significant']:
            if slope < -0.05:
                gradient_type = 'negative_strong'
                interpretation = 'Strong central alpha enhancement'
            elif slope < -0.02:
                gradient_type = 'negative_moderate'
                interpretation = 'Moderate central alpha enhancement'
            elif slope > 0.05:
                gradient_type = 'positive_strong'
                interpretation = 'Strong central alpha depletion'
            elif slope > 0.02:
                gradient_type = 'positive_moderate'
                interpretation = 'Moderate central alpha depletion'
            else:
                gradient_type = 'flat'
                interpretation = 'Flat alpha profile'
        else:
            gradient_type = 'flat'
            interpretation = 'No significant gradient'
        
        results = {
            'slope': slope,
            'slope_error': std_err,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'n_points': len(radii_fit),
            'significance': significance,
            'gradient_type': gradient_type,
            'interpretation': interpretation,
            'fit_radii': radii_fit,
            'fit_alpha_fe': alpha_fe_fit,
            'fit_errors': errors_fit,
            'predicted': predicted,
            'residuals': residuals
        }
        
        logger.info(f"Gradient fit: slope = {slope:.4f} ± {std_err:.4f} [α/Fe]/Re, p = {p_value:.4f}, {significance}")
        return results
        
    except Exception as e:
        logger.error(f"Error fitting gradient: {e}")
        return None

def create_gradient_plot(galaxy_name, profile, gradient_results, alpha_fe_data, output_dir="alpha_gradient_plots"):
    """Create comprehensive gradient analysis plot"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 2D Alpha/Fe map with bins
        ax1 = fig.add_subplot(gs[0, :2])
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        
        im1 = ax1.imshow(alpha_fe_2d, origin='lower', cmap='RdYlBu_r',
                        vmin=np.nanpercentile(alpha_fe_2d, 5),
                        vmax=np.nanpercentile(alpha_fe_2d, 95))
        
        # Overlay bins
        bin_info = profile['bin_info']
        center_y, center_x = bin_info['center']
        
        for i, radius in enumerate(bin_info['bin_centers']):
            if i in profile['valid_bins']:
                circle = Circle((center_x, center_y), radius, 
                              fill=False, color='white', linewidth=1.5, alpha=0.8)
                ax1.add_patch(circle)
        
        ax1.set_title(f'{galaxy_name} - Alpha/Fe Map with Radial Bins')
        ax1.set_xlabel('X [pixels]')
        ax1.set_ylabel('Y [pixels]')
        
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('[α/Fe] [dex]')
        
        # 2. Radial profile
        ax2 = fig.add_subplot(gs[1, :2])
        
        valid_bins = profile['valid_bins']
        radii = profile['bin_radii_re'][valid_bins]
        alpha_fe_mean = profile['alpha_fe_mean'][valid_bins]
        alpha_fe_error = profile['alpha_fe_error'][valid_bins]
        
        ax2.errorbar(radii, alpha_fe_mean, yerr=alpha_fe_error,
                    fmt='ko', capsize=3, capthick=1, label='Radial bins')
        
        # Plot fit
        if gradient_results is not None:
            fit_radii = gradient_results['fit_radii']
            predicted = gradient_results['predicted']
            
            r_extended = np.linspace(0, max(fit_radii) * 1.1, 100)
            fit_extended = gradient_results['slope'] * r_extended + gradient_results['intercept']
            ax2.plot(r_extended, fit_extended, 'r-', linewidth=2,
                    label=f'Linear fit: slope = {gradient_results["slope"]:.3f} ± {gradient_results["slope_error"]:.3f}')
            
            ax2.plot(fit_radii, predicted, 'ro', markersize=6, alpha=0.7)
        
        ax2.set_xlabel('Radius [Re]')
        ax2.set_ylabel('[α/Fe] [dex]')
        ax2.set_title(f'{galaxy_name} - Alpha/Fe Radial Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Summary
        ax3 = fig.add_subplot(gs[:, 2])
        ax3.axis('off')
        
        summary_text = f"Galaxy: {galaxy_name}\n"
        summary_text += f"Type: {alpha_fe_data.get('galaxy_type', 'Unknown')}\n\n"
        
        summary_text += "Alpha/Fe Statistics:\n"
        summary_text += f"Mean: {alpha_fe_data['mean_alpha_fe']:.3f} ± {alpha_fe_data['std_alpha_fe']:.3f}\n"
        summary_text += f"Valid pixels: {alpha_fe_data['n_successful']}\n\n"
        
        summary_text += "Radial Profile:\n"
        summary_text += f"Valid bins: {len(profile['valid_bins'])}\n"
        summary_text += f"Re: {profile['effective_radius']:.3f} kpc\n\n"
        
        if gradient_results is not None:
            summary_text += "Gradient Analysis:\n"
            summary_text += f"Slope: {gradient_results['slope']:.4f} ± {gradient_results['slope_error']:.4f}\n"
            summary_text += f"Intercept: {gradient_results['intercept']:.3f}\n"
            summary_text += f"R²: {gradient_results['r_squared']:.3f}\n"
            summary_text += f"p-value: {gradient_results['p_value']:.4f}\n"
            summary_text += f"χ²/ν: {gradient_results['reduced_chi_squared']:.2f}\n"
            summary_text += f"Significance: {gradient_results['significance']}\n\n"
            summary_text += f"Interpretation:\n{gradient_results['interpretation']}"
        else:
            summary_text += "Gradient Analysis:\nInsufficient data for fitting"
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'{galaxy_name} - Alpha Abundance Gradient Analysis', fontsize=16, y=0.95)
        
        output_path = f"{output_dir}/{galaxy_name}_alpha_gradient.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating plot for {galaxy_name}: {e}")
        return None

def analyze_galaxy(galaxy_name):
    """Complete analysis for one galaxy"""
    logger.info(f"Analyzing {galaxy_name}")
    
    # Load data
    alpha_fe_data = load_galaxy_alpha_fe_data(galaxy_name)
    if alpha_fe_data is None:
        return None
    
    # Get effective radius
    effective_radius = get_effective_radius(galaxy_name)
    
    # Calculate profile
    profile = calculate_radial_profile_simple(alpha_fe_data, effective_radius)
    if profile is None:
        return None
    
    # Fit gradient
    gradient_results = fit_alpha_gradient(profile)
    
    # Create plot
    plot_path = create_gradient_plot(galaxy_name, profile, gradient_results, alpha_fe_data)
    
    return {
        'galaxy_name': galaxy_name,
        'profile': profile,
        'gradient_results': gradient_results,
        'plot_path': plot_path,
        'effective_radius': effective_radius,
        'success': gradient_results is not None
    }

def main():
    """Main analysis function"""
    logger.info("Starting Simplified Alpha Abundance Gradient Analysis")
    
    # Get list of galaxies
    analysis_dir = "alpha_fe_analysis_results/analysis_20250720_091707"
    galaxy_dirs = [d for d in os.listdir(analysis_dir) 
                   if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith('VCC')]
    
    logger.info(f"Found {len(galaxy_dirs)} galaxies: {sorted(galaxy_dirs)}")
    
    # Analyze each galaxy
    results = []
    successful = 0
    
    for galaxy_name in sorted(galaxy_dirs):
        try:
            result = analyze_galaxy(galaxy_name)
            if result is not None:
                results.append(result)
                if result['success']:
                    successful += 1
                    logger.info(f"✓ Successfully analyzed {galaxy_name}")
                else:
                    logger.warning(f"⚠ Partial analysis for {galaxy_name}")
            else:
                logger.error(f"✗ Failed to analyze {galaxy_name}")
        except Exception as e:
            logger.error(f"✗ Error analyzing {galaxy_name}: {e}")
    
    # Create summary
    logger.info(f"\nSUMMARY:")
    logger.info(f"Total galaxies: {len(galaxy_dirs)}")
    logger.info(f"Successful gradient fits: {successful}")
    logger.info(f"Plots saved in: alpha_gradient_plots/")
    
    # Save summary table
    create_summary_table(results)
    
    return results

def create_summary_table(results):
    """Create summary table of gradient results"""
    try:
        summary_data = []
        
        for result in results:
            if result['success']:
                grad = result['gradient_results']
                profile = result['profile']
                
                row = {
                    'Galaxy': result['galaxy_name'],
                    'Slope': grad['slope'],
                    'Slope_Error': grad['slope_error'],
                    'Intercept': grad['intercept'],
                    'R_squared': grad['r_squared'],
                    'P_value': grad['p_value'],
                    'Significance': grad['significance'],
                    'Gradient_Type': grad['gradient_type'],
                    'N_bins': grad['n_points'],
                    'Re_kpc': result['effective_radius'],
                    'Valid_Bins': len(profile['valid_bins'])
                }
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            output_path = "alpha_gradient_plots/alpha_gradient_summary.csv"
            df.to_csv(output_path, index=False, float_format='%.4f')
            logger.info(f"Saved summary table: {output_path}")
            
            print("\n" + "="*80)
            print("ALPHA ABUNDANCE GRADIENT SUMMARY")
            print("="*80)
            print(df.to_string(index=False, float_format='%.4f'))
            print("="*80)
    
    except Exception as e:
        logger.error(f"Error creating summary table: {e}")

if __name__ == "__main__":
    main()
