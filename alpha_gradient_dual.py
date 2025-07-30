#!/usr/bin/env python3
"""
Enhanced Alpha Abundance Gradient Analysis for VNB and RDB Modes

This script calculates alpha abundance gradients for both analysis modes:
- RDB: Using inner 3 bins only (following Liu Yiqing 2016)
- VNB: Using data within Re (following Zhengzheng 2019)

Includes comprehensive error handling and failure analysis.
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
import traceback
from pathlib import Path

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
            'std_alpha_fe': float(data['std_alpha_fe']),
            'has_rdb': bool(data['has_rdb']),
            'has_vnb': bool(data['has_vnb'])
        }
        
        logger.info(f"Loaded alpha/Fe data for {galaxy_name}: {result['alpha_fe_2d'].shape}, {result['n_successful']} valid pixels")
        return result
        
    except Exception as e:
        logger.error(f"Error loading alpha/Fe data for {galaxy_name}: {e}")
        return None

def find_analysis_file(galaxy_name, mode):
    """Find analysis file with flexible path search"""
    possible_paths = [
        f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_{mode}_results.npz",
        f"./output/{galaxy_name}/Data/{galaxy_name}_stack_{mode}_results.npz",
        f"./output/{galaxy_name}/{galaxy_name}_stack/Data/{galaxy_name}_stack_{mode}_results.npz"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_analysis_data(galaxy_name, mode):
    """
    Load analysis data for specified mode (RDB or VNB)
    """
    try:
        file_path = find_analysis_file(galaxy_name, mode)
        if file_path is None:
            logger.warning(f"{mode} data not found for {galaxy_name}")
            return None
            
        data = np.load(file_path, allow_pickle=True)
        logger.info(f"Loaded {mode} data for {galaxy_name} from: {file_path}")
        
        # Extract common information
        distance_info = data['distance'].item()
        effective_radius = distance_info['effective_radius']
        
        if mode == 'RDB':
            # RDB specific extraction
            binning_info = data['binning'].item()
            bin_indices_info = data['bin_indices'].item()
            
            result = {
                'mode': 'RDB',
                'bin_distances': distance_info['bin_distances'],
                'effective_radius': effective_radius,
                'center_x': binning_info['center_x'],
                'center_y': binning_info['center_y'],
                'bin_indices': bin_indices_info.get('bin_indices', {}),
                'pixel_indices': bin_indices_info.get('pixel_indices', {}),
                'file_path': file_path
            }
            
        elif mode == 'VNB':
            # VNB specific extraction - check if it has Voronoi binning info
            if 'binning' in data:
                binning_info = data['binning'].item()
                result = {
                    'mode': 'VNB',
                    'effective_radius': effective_radius,
                    'file_path': file_path,
                    'binning_info': binning_info
                }
                
                # Try to extract spatial information
                if 'bin_coordinates' in binning_info:
                    result['bin_coordinates'] = binning_info['bin_coordinates']
                if 'bin_pixels' in binning_info:
                    result['bin_pixels'] = binning_info['bin_pixels']
                    
            else:
                logger.warning(f"VNB data for {galaxy_name} missing binning information")
                return None
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading {mode} data for {galaxy_name}: {e}")
        traceback.print_exc()
        return None

def create_radial_bins_simple(alpha_fe_2d, effective_radius, center=None, n_bins=6, max_radius_factor=2.0):
    """
    Create simple concentric circular bins for radial analysis
    
    This follows the same approach as the working simple version
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
        distances_pixels = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Define bin edges in pixels
        max_radius_pixels = min(center_x, center_y, nx - center_x, ny - center_y) * max_radius_factor
        bin_edges_pixels = np.linspace(0, max_radius_pixels, n_bins + 1)
        bin_centers_pixels = (bin_edges_pixels[:-1] + bin_edges_pixels[1:]) / 2
        
        # Convert to physical units (kpc) then to effective radius units
        pixel_scale_kpc = 0.2 * (17 * 1000) / 206265  # arcsec to kpc conversion
        
        bin_centers_kpc = bin_centers_pixels * pixel_scale_kpc
        bin_centers_re = bin_centers_kpc / effective_radius
        
        # Create bin masks
        bin_masks = []
        for i in range(n_bins):
            mask = (distances_pixels >= bin_edges_pixels[i]) & (distances_pixels < bin_edges_pixels[i + 1])
            bin_masks.append(mask)
        
        return {
            'bin_centers_pixels': bin_centers_pixels,
            'bin_centers_kpc': bin_centers_kpc,
            'bin_centers_re': bin_centers_re,
            'bin_edges_pixels': bin_edges_pixels,
            'bin_masks': bin_masks,
            'center': (center_y, center_x),
            'distances_pixels': distances_pixels
        }
        
    except Exception as e:
        logger.error(f"Error creating radial bins: {e}")
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

def calculate_vnb_radial_profile(alpha_fe_data, vnb_data, min_pixels_per_bin=10):
    """
    Calculate radial profile for VNB mode using geometric binning within Re
    """
    try:
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        alpha_fe_errors = alpha_fe_data['alpha_fe_errors']
        effective_radius = vnb_data['effective_radius']
        
        # Use the same effective radius function as simple version
        effective_radius = get_effective_radius(alpha_fe_data['galaxy_name'])
        
        # Create radial bins (same as simple version)
        bin_info = create_radial_bins_simple(alpha_fe_2d, effective_radius, n_bins=6)
        if bin_info is None:
            return None
        
        # Filter bins to only include those within Re for VNB
        within_re_mask = bin_info['bin_centers_re'] <= 1.0
        valid_bin_indices = np.where(within_re_mask)[0]
        
        if len(valid_bin_indices) == 0:
            logger.warning(f"No bins within Re for VNB analysis of {alpha_fe_data['galaxy_name']}")
            return None
        
        # Initialize profile
        n_total_bins = len(bin_info['bin_centers_re'])
        profile = {
            'mode': 'VNB',
            'bin_radii_re': bin_info['bin_centers_re'],
            'alpha_fe_mean': np.full(n_total_bins, np.nan),
            'alpha_fe_median': np.full(n_total_bins, np.nan),
            'alpha_fe_std': np.full(n_total_bins, np.nan),
            'alpha_fe_error': np.full(n_total_bins, np.nan),
            'n_pixels': np.zeros(n_total_bins, dtype=int),
            'valid_bins': [],
            'effective_radius': effective_radius,
            'galaxy_name': alpha_fe_data['galaxy_name'],
            'bin_info': bin_info
        }
        
        # Calculate statistics for each bin within Re
        for i in valid_bin_indices:
            bin_mask = bin_info['bin_masks'][i]
            
            if not np.any(bin_mask):
                continue
                
            # Extract data for this bin
            alpha_fe_bin = alpha_fe_2d[bin_mask]
            errors_bin = alpha_fe_errors[bin_mask]
            
            # Filter valid data
            valid_mask = np.isfinite(alpha_fe_bin) & np.isfinite(errors_bin)
            alpha_fe_valid = alpha_fe_bin[valid_mask]
            errors_valid = errors_bin[valid_mask]
            
            n_valid = len(alpha_fe_valid)
            profile['n_pixels'][i] = n_valid
            
            if n_valid < min_pixels_per_bin:
                logger.debug(f"VNB Bin {i}: insufficient pixels ({n_valid} < {min_pixels_per_bin})")
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
            
            logger.debug(f"VNB Bin {i}: R={bin_info['bin_centers_re'][i]:.2f} Re, "
                        f"[α/Fe]={profile['alpha_fe_mean'][i]:.3f}±{total_error:.3f}, N={n_valid}")
        
        logger.info(f"VNB profile for {alpha_fe_data['galaxy_name']}: {len(profile['valid_bins'])}/{len(valid_bin_indices)} valid bins within Re")
        return profile
        
    except Exception as e:
        logger.error(f"Error calculating VNB radial profile: {e}")
        traceback.print_exc()
        return None

def calculate_rdb_radial_profile_inner3(alpha_fe_data, rdb_data, min_pixels_per_bin=5):
    """
    Calculate radial profile for RDB mode using only inner 3 bins
    """
    try:
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        alpha_fe_errors = alpha_fe_data['alpha_fe_errors']
        
        # Use the same effective radius function as simple version
        effective_radius = get_effective_radius(alpha_fe_data['galaxy_name'])
        
        # Create radial bins (same as simple version)
        bin_info = create_radial_bins_simple(alpha_fe_2d, effective_radius, n_bins=6)
        if bin_info is None:
            return None
        
        # Use only the inner 3 bins for RDB analysis
        n_rdb_bins = 3
        valid_bin_indices = list(range(n_rdb_bins))
        
        # Initialize profile
        n_total_bins = len(bin_info['bin_centers_re'])
        profile = {
            'mode': 'RDB',
            'bin_radii_re': bin_info['bin_centers_re'],
            'alpha_fe_mean': np.full(n_total_bins, np.nan),
            'alpha_fe_median': np.full(n_total_bins, np.nan),
            'alpha_fe_std': np.full(n_total_bins, np.nan),
            'alpha_fe_error': np.full(n_total_bins, np.nan),
            'n_pixels': np.zeros(n_total_bins, dtype=int),
            'valid_bins': [],
            'effective_radius': effective_radius,
            'galaxy_name': alpha_fe_data['galaxy_name'],
            'bin_info': bin_info
        }
        
        # Calculate statistics for inner 3 bins only
        for i in valid_bin_indices:
            bin_mask = bin_info['bin_masks'][i]
            
            if not np.any(bin_mask):
                continue
                
            # Extract data for this bin
            alpha_fe_bin = alpha_fe_2d[bin_mask]
            errors_bin = alpha_fe_errors[bin_mask]
            
            # Filter valid data
            valid_mask = np.isfinite(alpha_fe_bin) & np.isfinite(errors_bin)
            alpha_fe_valid = alpha_fe_bin[valid_mask]
            errors_valid = errors_bin[valid_mask]
            
            n_valid = len(alpha_fe_valid)
            profile['n_pixels'][i] = n_valid
            
            if n_valid < min_pixels_per_bin:
                logger.debug(f"RDB Bin {i}: insufficient pixels ({n_valid} < {min_pixels_per_bin})")
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
            
            logger.debug(f"RDB Bin {i}: R={bin_info['bin_centers_re'][i]:.2f} Re, "
                        f"[α/Fe]={profile['alpha_fe_mean'][i]:.3f}±{total_error:.3f}, N={n_valid}")
        
        logger.info(f"RDB profile for {alpha_fe_data['galaxy_name']}: {len(profile['valid_bins'])}/{n_rdb_bins} valid bins (inner only)")
        return profile
        
    except Exception as e:
        logger.error(f"Error calculating RDB radial profile: {e}")
        traceback.print_exc()
        return None

def fit_gradient(profile, mode, min_bins=3):
    """
    Fit linear gradient to alpha/Fe profile
    """
    try:
        valid_bins = profile['valid_bins']
        
        if len(valid_bins) < min_bins:
            logger.warning(f"{mode} gradient fitting: insufficient bins ({len(valid_bins)} < {min_bins})")
            return None
        
        # Get data for valid bins
        radii = profile['bin_radii_re'][valid_bins]
        alpha_fe = profile['alpha_fe_mean'][valid_bins]
        errors = profile['alpha_fe_error'][valid_bins]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(radii, alpha_fe)
        
        # Calculate additional statistics
        predicted = slope * radii + intercept
        residuals = alpha_fe - predicted
        chi_squared = np.sum((residuals / errors)**2) if np.all(errors > 0) else np.sum(residuals**2)
        reduced_chi_squared = chi_squared / (len(radii) - 2) if len(radii) > 2 else chi_squared
        
        # Significance classification
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
            'mode': mode,
            'slope': slope,
            'slope_error': std_err,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'n_points': len(radii),
            'significance': significance,
            'gradient_type': gradient_type,
            'interpretation': interpretation,
            'fit_radii': radii,
            'fit_alpha_fe': alpha_fe,
            'fit_errors': errors,
            'predicted': predicted,
            'residuals': residuals
        }
        
        logger.info(f"{mode} gradient: slope = {slope:.4f} ± {std_err:.4f} [α/Fe]/Re, p = {p_value:.4f}, {significance}")
        return results
        
    except Exception as e:
        logger.error(f"Error fitting {mode} gradient: {e}")
        return None

def create_dual_mode_plot(galaxy_name, vnb_profile, rdb_profile, vnb_gradient, rdb_gradient, 
                         alpha_fe_data, output_dir="alpha_gradient_dual"):
    """
    Create plot comparing VNB and RDB gradient analyses
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        
        # 1. 2D Alpha/Fe map
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(alpha_fe_2d, origin='lower', cmap='RdYlBu_r',
                        vmin=np.nanpercentile(alpha_fe_2d, 5),
                        vmax=np.nanpercentile(alpha_fe_2d, 95))
        
        ax1.set_title(f'{galaxy_name} - Alpha/Fe Map')
        ax1.set_xlabel('X [pixels]')
        ax1.set_ylabel('Y [pixels]')
        
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('[α/Fe] [dex]')
        
        # 2. VNB radial profile
        ax2 = fig.add_subplot(gs[1, 0])
        if vnb_profile is not None:
            valid_bins = vnb_profile['valid_bins']
            if len(valid_bins) > 0:
                radii = vnb_profile['bin_radii_re'][valid_bins]
                alpha_fe_mean = vnb_profile['alpha_fe_mean'][valid_bins]
                alpha_fe_error = vnb_profile['alpha_fe_error'][valid_bins]
                
                ax2.errorbar(radii, alpha_fe_mean, yerr=alpha_fe_error,
                            fmt='bo', capsize=3, capthick=1, label='VNB bins')
                
                if vnb_gradient is not None:
                    fit_radii = vnb_gradient['fit_radii']
                    predicted = vnb_gradient['predicted']
                    
                    r_extended = np.linspace(0, max(fit_radii) * 1.1, 100)
                    fit_extended = vnb_gradient['slope'] * r_extended + vnb_gradient['intercept']
                    ax2.plot(r_extended, fit_extended, 'b-', linewidth=2,
                            label=f'VNB fit: {vnb_gradient["slope"]:.3f} ± {vnb_gradient["slope_error"]:.3f}')
                    
                    ax2.plot(fit_radii, predicted, 'bo', markersize=6, alpha=0.7)
        
        ax2.set_xlabel('Radius [Re]')
        ax2.set_ylabel('[α/Fe] [dex]')
        ax2.set_title('VNB Analysis (within Re)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. RDB radial profile
        ax3 = fig.add_subplot(gs[1, 1])
        if rdb_profile is not None:
            valid_bins = rdb_profile['valid_bins']
            if len(valid_bins) > 0:
                radii = rdb_profile['bin_radii_re'][valid_bins]
                alpha_fe_mean = rdb_profile['alpha_fe_mean'][valid_bins]
                alpha_fe_error = rdb_profile['alpha_fe_error'][valid_bins]
                
                ax3.errorbar(radii, alpha_fe_mean, yerr=alpha_fe_error,
                            fmt='ro', capsize=3, capthick=1, label='RDB bins')
                
                if rdb_gradient is not None:
                    fit_radii = rdb_gradient['fit_radii']
                    predicted = rdb_gradient['predicted']
                    
                    r_extended = np.linspace(0, max(fit_radii) * 1.1, 100)
                    fit_extended = rdb_gradient['slope'] * r_extended + rdb_gradient['intercept']
                    ax3.plot(r_extended, fit_extended, 'r-', linewidth=2,
                            label=f'RDB fit: {rdb_gradient["slope"]:.3f} ± {rdb_gradient["slope_error"]:.3f}')
                    
                    ax3.plot(fit_radii, predicted, 'ro', markersize=6, alpha=0.7)
        
        ax3.set_xlabel('Radius [Re]')
        ax3.set_ylabel('[α/Fe] [dex]')
        ax3.set_title('RDB Analysis (inner 3 bins)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined comparison
        ax4 = fig.add_subplot(gs[1, 2:])
        
        colors = ['blue', 'red']
        labels = ['VNB', 'RDB']
        profiles = [vnb_profile, rdb_profile]
        gradients = [vnb_gradient, rdb_gradient]
        
        for i, (profile, gradient, color, label) in enumerate(zip(profiles, gradients, colors, labels)):
            if profile is not None:
                valid_bins = profile['valid_bins']
                if len(valid_bins) > 0:
                    radii = profile['bin_radii_re'][valid_bins]
                    alpha_fe_mean = profile['alpha_fe_mean'][valid_bins]
                    alpha_fe_error = profile['alpha_fe_error'][valid_bins]
                    
                    ax4.errorbar(radii, alpha_fe_mean, yerr=alpha_fe_error,
                                fmt=f'{color[0]}o', capsize=3, capthick=1, label=f'{label} data')
                    
                    if gradient is not None:
                        fit_radii = gradient['fit_radii']
                        r_extended = np.linspace(0, max(fit_radii) * 1.1, 100)
                        fit_extended = gradient['slope'] * r_extended + gradient['intercept']
                        ax4.plot(r_extended, fit_extended, f'{color[0]}-', linewidth=2,
                                label=f'{label}: {gradient["slope"]:.3f} ± {gradient["slope_error"]:.3f}')
        
        ax4.set_xlabel('Radius [Re]')
        ax4.set_ylabel('[α/Fe] [dex]')
        ax4.set_title('VNB vs RDB Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Summary
        ax5 = fig.add_subplot(gs[:, 3])
        ax5.axis('off')
        
        summary_text = f"Galaxy: {galaxy_name}\n"
        summary_text += f"Type: {alpha_fe_data.get('galaxy_type', 'Unknown')}\n\n"
        
        summary_text += "Alpha/Fe Statistics:\n"
        summary_text += f"Mean: {alpha_fe_data['mean_alpha_fe']:.3f} ± {alpha_fe_data['std_alpha_fe']:.3f}\n"
        summary_text += f"Valid pixels: {alpha_fe_data['n_successful']}\n\n"
        
        # VNB summary
        summary_text += "VNB Analysis (within Re):\n"
        if vnb_profile is not None and vnb_gradient is not None:
            summary_text += f"Bins: {len(vnb_profile['valid_bins'])}\n"
            summary_text += f"Slope: {vnb_gradient['slope']:.4f} ± {vnb_gradient['slope_error']:.4f}\n"
            summary_text += f"p-value: {vnb_gradient['p_value']:.4f}\n"
            summary_text += f"Significance: {vnb_gradient['significance']}\n"
        else:
            summary_text += "Analysis failed\n"
        summary_text += "\n"
        
        # RDB summary
        summary_text += "RDB Analysis (inner 3 bins):\n"
        if rdb_profile is not None and rdb_gradient is not None:
            summary_text += f"Bins: {len(rdb_profile['valid_bins'])}\n"
            summary_text += f"Slope: {rdb_gradient['slope']:.4f} ± {rdb_gradient['slope_error']:.4f}\n"
            summary_text += f"p-value: {rdb_gradient['p_value']:.4f}\n"
            summary_text += f"Significance: {rdb_gradient['significance']}\n"
        else:
            summary_text += "Analysis failed\n"
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'{galaxy_name} - Dual Mode Alpha Abundance Gradient Analysis', fontsize=16, y=0.98)
        
        output_path = f"{output_dir}/{galaxy_name}_dual_gradient.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved dual mode plot: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating dual mode plot for {galaxy_name}: {e}")
        return None

def analyze_galaxy_dual_mode(galaxy_name):
    """
    Complete dual mode analysis for one galaxy
    """
    logger.info(f"Starting dual mode analysis for {galaxy_name}")
    
    results = {
        'galaxy_name': galaxy_name,
        'vnb_success': False,
        'rdb_success': False,
        'vnb_profile': None,
        'rdb_profile': None,
        'vnb_gradient': None,
        'rdb_gradient': None,
        'vnb_error': None,
        'rdb_error': None,
        'plot_path': None
    }
    
    # Load alpha/Fe data
    alpha_fe_data = load_galaxy_alpha_fe_data(galaxy_name)
    if alpha_fe_data is None:
        results['vnb_error'] = "Alpha/Fe data not found"
        results['rdb_error'] = "Alpha/Fe data not found"
        return results
    
    # VNB Analysis
    try:
        vnb_data = load_analysis_data(galaxy_name, 'VNB')
        if vnb_data is not None:
            vnb_profile = calculate_vnb_radial_profile(alpha_fe_data, vnb_data)
            if vnb_profile is not None:
                vnb_gradient = fit_gradient(vnb_profile, 'VNB')
                if vnb_gradient is not None:
                    results['vnb_success'] = True
                    results['vnb_profile'] = vnb_profile
                    results['vnb_gradient'] = vnb_gradient
                else:
                    results['vnb_error'] = "Gradient fitting failed"
            else:
                results['vnb_error'] = "Profile calculation failed"
        else:
            results['vnb_error'] = "VNB data not found"
    except Exception as e:
        results['vnb_error'] = str(e)
        logger.error(f"VNB analysis error for {galaxy_name}: {e}")
    
    # RDB Analysis
    try:
        rdb_data = load_analysis_data(galaxy_name, 'RDB')
        if rdb_data is not None:
            rdb_profile = calculate_rdb_radial_profile_inner3(alpha_fe_data, rdb_data)
            if rdb_profile is not None:
                rdb_gradient = fit_gradient(rdb_profile, 'RDB')
                if rdb_gradient is not None:
                    results['rdb_success'] = True
                    results['rdb_profile'] = rdb_profile
                    results['rdb_gradient'] = rdb_gradient
                else:
                    results['rdb_error'] = "Gradient fitting failed"
            else:
                results['rdb_error'] = "Profile calculation failed"
        else:
            results['rdb_error'] = "RDB data not found"
    except Exception as e:
        results['rdb_error'] = str(e)
        logger.error(f"RDB analysis error for {galaxy_name}: {e}")
    
    # Create plot if at least one mode succeeded
    if results['vnb_success'] or results['rdb_success']:
        try:
            plot_path = create_dual_mode_plot(
                galaxy_name, 
                results['vnb_profile'], 
                results['rdb_profile'],
                results['vnb_gradient'], 
                results['rdb_gradient'],
                alpha_fe_data
            )
            results['plot_path'] = plot_path
        except Exception as e:
            logger.error(f"Plot creation error for {galaxy_name}: {e}")
    
    return results

def main():
    """
    Main function for dual mode analysis
    """
    logger.info("Starting Dual Mode Alpha Abundance Gradient Analysis")
    logger.info("VNB: within Re | RDB: inner 3 bins")
    
    # Get galaxy list
    analysis_dir = "alpha_fe_analysis_results/analysis_20250720_091707"
    galaxy_dirs = [d for d in os.listdir(analysis_dir) 
                   if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith('VCC')]
    
    logger.info(f"Found {len(galaxy_dirs)} galaxies to analyze")
    
    # Analyze all galaxies
    all_results = []
    vnb_successes = 0
    rdb_successes = 0
    vnb_failures = []
    rdb_failures = []
    
    for galaxy_name in sorted(galaxy_dirs):
        try:
            result = analyze_galaxy_dual_mode(galaxy_name)
            all_results.append(result)
            
            if result['vnb_success']:
                vnb_successes += 1
                logger.info(f"✓ VNB success: {galaxy_name}")
            else:
                vnb_failures.append((galaxy_name, result['vnb_error']))
                logger.warning(f"✗ VNB failed: {galaxy_name} - {result['vnb_error']}")
            
            if result['rdb_success']:
                rdb_successes += 1
                logger.info(f"✓ RDB success: {galaxy_name}")
            else:
                rdb_failures.append((galaxy_name, result['rdb_error']))
                logger.warning(f"✗ RDB failed: {galaxy_name} - {result['rdb_error']}")
                
        except Exception as e:
            logger.error(f"Complete failure for {galaxy_name}: {e}")
            vnb_failures.append((galaxy_name, f"Complete failure: {str(e)}"))
            rdb_failures.append((galaxy_name, f"Complete failure: {str(e)}"))
    
    # Create summary tables
    create_dual_mode_summary(all_results)
    
    # Print failure analysis
    print_failure_analysis(vnb_failures, rdb_failures)
    
    # Summary
    logger.info(f"\n" + "="*60)
    logger.info("DUAL MODE ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total galaxies: {len(galaxy_dirs)}")
    logger.info(f"VNB successes: {vnb_successes}")
    logger.info(f"RDB successes: {rdb_successes}")
    logger.info(f"VNB failures: {len(vnb_failures)}")
    logger.info(f"RDB failures: {len(rdb_failures)}")
    logger.info("="*60)
    
    return all_results

def create_dual_mode_summary(all_results):
    """Create summary tables for both modes"""
    try:
        os.makedirs("alpha_gradient_dual", exist_ok=True)
        
        # VNB summary
        vnb_data = []
        for result in all_results:
            if result['vnb_success'] and result['vnb_gradient'] is not None:
                grad = result['vnb_gradient']
                vnb_data.append({
                    'Galaxy': result['galaxy_name'],
                    'Mode': 'VNB',
                    'Slope': grad['slope'],
                    'Slope_Error': grad['slope_error'],
                    'Intercept': grad['intercept'],
                    'R_squared': grad['r_squared'],
                    'P_value': grad['p_value'],
                    'Significance': grad['significance'],
                    'N_bins': grad['n_points']
                })
        
        # RDB summary
        rdb_data = []
        for result in all_results:
            if result['rdb_success'] and result['rdb_gradient'] is not None:
                grad = result['rdb_gradient']
                rdb_data.append({
                    'Galaxy': result['galaxy_name'],
                    'Mode': 'RDB',
                    'Slope': grad['slope'],
                    'Slope_Error': grad['slope_error'],
                    'Intercept': grad['intercept'],
                    'R_squared': grad['r_squared'],
                    'P_value': grad['p_value'],
                    'Significance': grad['significance'],
                    'N_bins': grad['n_points']
                })
        
        # Save individual tables
        if vnb_data:
            vnb_df = pd.DataFrame(vnb_data)
            vnb_df.to_csv("alpha_gradient_dual/vnb_gradient_summary.csv", index=False, float_format='%.4f')
            logger.info("Saved VNB summary table")
        
        if rdb_data:
            rdb_df = pd.DataFrame(rdb_data)
            rdb_df.to_csv("alpha_gradient_dual/rdb_gradient_summary.csv", index=False, float_format='%.4f')
            logger.info("Saved RDB summary table")
        
        # Combined table
        combined_data = vnb_data + rdb_data
        if combined_data:
            combined_df = pd.DataFrame(combined_data)
            combined_df.to_csv("alpha_gradient_dual/combined_gradient_summary.csv", index=False, float_format='%.4f')
            logger.info("Saved combined summary table")
            
            print("\n" + "="*80)
            print("COMBINED GRADIENT SUMMARY")
            print("="*80)
            print(combined_df.to_string(index=False, float_format='%.4f'))
            print("="*80)
    
    except Exception as e:
        logger.error(f"Error creating summary tables: {e}")

def print_failure_analysis(vnb_failures, rdb_failures):
    """Print detailed failure analysis"""
    print("\n" + "="*60)
    print("FAILURE ANALYSIS")
    print("="*60)
    
    print(f"\nVNB Failures ({len(vnb_failures)}):")
    print("-" * 40)
    for galaxy, error in vnb_failures:
        print(f"{galaxy}: {error}")
    
    print(f"\nRDB Failures ({len(rdb_failures)}):")
    print("-" * 40)
    for galaxy, error in rdb_failures:
        print(f"{galaxy}: {error}")
    
    # Group errors by type
    vnb_error_types = {}
    for galaxy, error in vnb_failures:
        error_type = error.split(':')[0] if ':' in error else error
        if error_type not in vnb_error_types:
            vnb_error_types[error_type] = []
        vnb_error_types[error_type].append(galaxy)
    
    rdb_error_types = {}
    for galaxy, error in rdb_failures:
        error_type = error.split(':')[0] if ':' in error else error
        if error_type not in rdb_error_types:
            rdb_error_types[error_type] = []
        rdb_error_types[error_type].append(galaxy)
    
    print(f"\nVNB Error Types:")
    for error_type, galaxies in vnb_error_types.items():
        print(f"  {error_type}: {len(galaxies)} galaxies - {galaxies}")
    
    print(f"\nRDB Error Types:")
    for error_type, galaxies in rdb_error_types.items():
        print(f"  {error_type}: {len(galaxies)} galaxies - {galaxies}")

if __name__ == "__main__":
    main()
