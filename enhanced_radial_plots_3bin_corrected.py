#!/usr/bin/env python3
"""
Enhanced Radial Plots - Corrected for 3-bin RDB and Matching VNB Range

This script creates enhanced radial gradient plots using:
1. RDB method: Only first 3 bins (innermost region)
2. VNB method: Only bins within the same radial range as 3-bin RDB
3. Proper R/Re normalization
4. No P2P method (removed)
5. Strict filtering: Only plot galaxies with both RDB and VNB data in range
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import curve_fit
import os
import sys
from pathlib import Path
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_vnb_radii(vnb_binning, ellipse_params, metadata):
    """Calculate elliptical radii for VNB bins using correct ellipse parameters"""
    
    # Get bin center coordinates (VNB coordinates are relative to center)
    bin_x = vnb_binning['bin_x']
    bin_y = vnb_binning['bin_y']
    
    # Get ellipse parameters
    center_x = ellipse_params['center_x']
    center_y = ellipse_params['center_y']
    PA_degrees = ellipse_params['PA_degrees']
    ellipticity = ellipse_params['ellipticity']
    
    # VNB coordinates are already relative to center, so dx, dy are just bin_x, bin_y
    dx = bin_x
    dy = bin_y
    
    # Rotate to principal axes
    PA_rad = np.radians(PA_degrees)
    x_prime = dx * np.cos(PA_rad) + dy * np.sin(PA_rad)
    y_prime = -dx * np.sin(PA_rad) + dy * np.cos(PA_rad)
    
    # Calculate elliptical radius with correct scaling
    if ellipticity < 1:
        q = 1.0 - ellipticity  # Axis ratio (b/a)
        R_elliptical = np.sqrt(x_prime**2 + (y_prime / q)**2)
    else:
        R_elliptical = np.sqrt(x_prime**2 + y_prime**2)
        
    # Convert to arcseconds using standard MUSE pixel scale
    # The metadata pixel scales seem to be incorrect or refer to different coordinates
    pixel_scale = 0.2  # arcsec/pixel (standard MUSE scale)
    
    return R_elliptical * pixel_scale

def extract_alpha_fe_from_results(results_file):
    """Extract alpha/Fe values from results file"""
    try:
        data = np.load(results_file, allow_pickle=True)
        
        # Extract spectral indices - use 'auto' analysis if available
        if 'bin_indices_multi' in data:
            indices_data = data['bin_indices_multi'].item()
            if 'auto' in indices_data:
                indices = indices_data['auto']['bin_indices']
            else:
                indices = indices_data[list(indices_data.keys())[0]]['bin_indices']
        elif 'bin_indices' in data:
            indices = data['bin_indices'].item()
            if 'bin_indices' in indices:
                indices = indices['bin_indices']
        else:
            return None
        
        # Get Fe5015 and Mgb values
        fe_values = indices.get('Fe5015', np.array([]))
        mg_values = indices.get('Mgb', np.array([]))
        
        # Check if we have valid data
        if len(fe_values) == 0 or len(mg_values) == 0:
            return None
        if np.all(np.isnan(fe_values)) or np.all(np.isnan(mg_values)):
            return None
        
        # Calculate alpha/Fe ratios
        alpha_fe = []
        for i in range(len(fe_values)):
            if not np.isnan(fe_values[i]) and not np.isnan(mg_values[i]) and fe_values[i] != 0:
                alpha_fe.append(mg_values[i] / fe_values[i])
            else:
                alpha_fe.append(np.nan)
        
        return np.array(alpha_fe)
        
    except Exception as e:
        logger.warning(f"Could not extract alpha/Fe from {results_file}: {e}")
        return None

def calculate_gradient_3bin_rdb_vnb(galaxy_name):
    """Calculate gradients using 3-bin RDB and matching VNB range"""
    
    galaxy_dir = f"output/{galaxy_name}_stack/Data"
    if not os.path.exists(galaxy_dir):
        logger.error(f"Galaxy directory not found: {galaxy_dir}")
        return None
    
    results = {
        'galaxy_name': galaxy_name,
        'RDB': None,
        'VNB': None,
        'effective_radius': None
    }
    
    # Load RDB data (first 3 bins only)
    rdb_binned_file = f"{galaxy_dir}/{galaxy_name}_stack_RDB_binned.npz"
    rdb_results_file = f"{galaxy_dir}/{galaxy_name}_stack_RDB_results.npz"
    
    logger.debug(f"{galaxy_name}: RDB binned exists: {os.path.exists(rdb_binned_file)}")
    logger.debug(f"{galaxy_name}: RDB results exists: {os.path.exists(rdb_results_file)}")
    
    if os.path.exists(rdb_binned_file) and os.path.exists(rdb_results_file):
        try:
            rdb_binned = np.load(rdb_binned_file, allow_pickle=True)
            rdb_results = np.load(rdb_results_file, allow_pickle=True)
            
            # Get first 3 bins only
            bin_radii = rdb_binned['bin_radii'][:3]  # First 3 bins
            max_radius = np.max(bin_radii)
            
            # Get effective radius
            metadata = rdb_binned['metadata'].item()
            effective_radius = metadata.get('effective_radius', 12.0)
            results['effective_radius'] = effective_radius
            
            # Extract alpha/Fe for first 3 bins
            alpha_fe_data = extract_alpha_fe_from_results(rdb_results_file)
            logger.debug(f"{galaxy_name}: RDB alpha/Fe data: {alpha_fe_data}")
            
            if alpha_fe_data is not None and len(alpha_fe_data) >= 3:
                alpha_fe_3bins = alpha_fe_data[:3]
                logger.debug(f"{galaxy_name}: RDB first 3 bins alpha/Fe: {alpha_fe_3bins}")
                
                # Check if we have valid finite values
                valid_mask = np.isfinite(alpha_fe_3bins)
                logger.debug(f"{galaxy_name}: RDB valid mask: {valid_mask}")
                
                if np.sum(valid_mask) >= 2:  # Need at least 2 points
                    # Use only valid points for gradient calculation
                    radii_valid = bin_radii[valid_mask]
                    alpha_fe_valid = alpha_fe_3bins[valid_mask]
                    radii_norm = radii_valid / effective_radius
                    
                    # Calculate gradient
                    slope, intercept, r_value, p_value, std_err = stats.linregress(radii_norm, alpha_fe_valid)
                    
                    results['RDB'] = {
                        'radii': radii_valid,  # Only valid radii
                        'radii_norm': radii_norm,  # Normalized valid radii
                        'alpha_fe': alpha_fe_valid,  # Only valid alpha/Fe values
                        'max_radius': max_radius,
                        'slope': slope,
                        'slope_error': std_err,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'n_bins': np.sum(valid_mask)  # Number of valid bins used
                    }
                    
                    logger.info(f"{galaxy_name} RDB (3 bins): slope = {slope:.4f} ± {std_err:.4f} dex/Re, R² = {r_value**2:.3f}")
                else:
                    logger.warning(f"{galaxy_name}: RDB has insufficient valid data points ({np.sum(valid_mask)}/3)")
            else:
                logger.warning(f"{galaxy_name}: Could not extract RDB alpha/Fe data")
            
        except Exception as e:
            logger.error(f"Error processing RDB data for {galaxy_name}: {e}")
    
    # Load VNB data (filtered to same radial range)
    vnb_binned_file = f"{galaxy_dir}/{galaxy_name}_stack_VNB_binned.npz"
    vnb_binning_file = f"{galaxy_dir}/{galaxy_name}_stack_VNB_binning.npz"
    vnb_results_file = f"{galaxy_dir}/{galaxy_name}_stack_VNB_results.npz"
    
    logger.debug(f"{galaxy_name}: VNB binned exists: {os.path.exists(vnb_binned_file)}")
    logger.debug(f"{galaxy_name}: VNB binning exists: {os.path.exists(vnb_binning_file)}")
    logger.debug(f"{galaxy_name}: VNB results exists: {os.path.exists(vnb_results_file)}")
    
    if (os.path.exists(vnb_binned_file) and os.path.exists(vnb_binning_file) and 
        os.path.exists(vnb_results_file) and results['RDB'] is not None):
        
        try:
            vnb_binned = np.load(vnb_binned_file, allow_pickle=True)
            vnb_binning = np.load(vnb_binning_file, allow_pickle=True)
            vnb_results = np.load(vnb_results_file, allow_pickle=True)
            
            # Get ellipse parameters from RDB metadata
            ellipse_params = metadata['ellipse_params']
            
            # Calculate VNB bin radii
            vnb_radii = calculate_vnb_radii(vnb_binning, ellipse_params, metadata)
            logger.debug(f"{galaxy_name}: VNB radii range: {np.min(vnb_radii):.1f} - {np.max(vnb_radii):.1f} arcsec")
            
            # Filter VNB bins to same range as 3-bin RDB
            max_radius = results['RDB']['max_radius']
            valid_vnb_mask = vnb_radii <= max_radius
            logger.debug(f"{galaxy_name}: VNB bins within {max_radius:.1f}\": {np.sum(valid_vnb_mask)}/{len(vnb_radii)}")
            
            if np.sum(valid_vnb_mask) >= 2:  # Need at least 2 points for gradient
                vnb_radii_filtered = vnb_radii[valid_vnb_mask]
                
                # Extract alpha/Fe for VNB bins in range
                alpha_fe_data = extract_alpha_fe_from_results(vnb_results_file)
                logger.debug(f"{galaxy_name}: VNB alpha/Fe data length: {len(alpha_fe_data) if alpha_fe_data is not None else None}")
                
                if alpha_fe_data is not None:
                    alpha_fe_filtered = alpha_fe_data[valid_vnb_mask]
                    logger.debug(f"{galaxy_name}: VNB filtered alpha/Fe valid points: {np.sum(np.isfinite(alpha_fe_filtered))}")
                    
                    # Normalize radii by effective radius
                    radii_norm = vnb_radii_filtered / effective_radius
                    
                    # Calculate gradient (only use finite values)
                    finite_mask = np.isfinite(alpha_fe_filtered)
                    if np.sum(finite_mask) >= 2:
                        radii_norm_finite = radii_norm[finite_mask]
                        alpha_fe_finite = alpha_fe_filtered[finite_mask]
                        
                        slope, intercept, r_value, p_value, std_err = stats.linregress(radii_norm_finite, alpha_fe_finite)
                        
                        results['VNB'] = {
                            'radii': radii_norm_finite,  # Only finite radii (normalized)
                            'radii_norm': radii_norm_finite,  # Same as above for consistency
                            'alpha_fe': alpha_fe_finite,  # Only finite alpha/Fe values
                            'max_radius': max_radius,
                            'slope': slope,
                            'slope_error': std_err,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'n_bins': np.sum(finite_mask)  # Number of finite bins used
                        }
                        
                        logger.info(f"{galaxy_name} VNB ({np.sum(finite_mask)} finite bins ≤{max_radius:.1f}\"): slope = {slope:.4f} ± {std_err:.4f} dex/Re, R² = {r_value**2:.3f}")
                    else:
                        logger.warning(f"{galaxy_name}: VNB data not suitable for gradient calculation (only {np.sum(finite_mask)} finite values)")
                else:
                    logger.warning(f"{galaxy_name}: Could not extract VNB alpha/Fe data")
            else:
                logger.warning(f"{galaxy_name}: No VNB bins within RDB range ({max_radius:.1f}\")")
                
        except Exception as e:
            logger.error(f"Error processing VNB data for {galaxy_name}: {e}")
    elif results['RDB'] is None:
        logger.warning(f"{galaxy_name}: RDB data not available, skipping VNB processing")
    
    # Return results if we have any data (RDB or VNB or both)
    if results['RDB'] is not None or results['VNB'] is not None:
        if results['RDB'] is not None and results['VNB'] is not None:
            logger.info(f"✓ {galaxy_name}: Both RDB and VNB gradients calculated")
        elif results['RDB'] is not None:
            logger.info(f"◐ {galaxy_name}: Only RDB gradient available")
        elif results['VNB'] is not None:
            logger.info(f"◑ {galaxy_name}: Only VNB gradient available")
        return results
    else:
        logger.warning(f"✗ {galaxy_name}: Insufficient data")
        return None

def create_enhanced_plot(galaxy_data, output_file):
    """Create enhanced plot for a single galaxy"""
    
    galaxy_name = galaxy_data['galaxy_name']
    rdb_data = galaxy_data['RDB']
    vnb_data = galaxy_data['VNB']
    
    # Determine plot layout based on available data
    if rdb_data is not None and vnb_data is not None:
        # Both methods available
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        axes = [ax1, ax2]
        methods = ['RDB', 'VNB']
        data_sets = [rdb_data, vnb_data]
        colors = ['blue', 'red']
        markers = ['o', 's']
    elif rdb_data is not None:
        # Only RDB available
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
        axes = [ax1]
        methods = ['RDB']
        data_sets = [rdb_data]
        colors = ['blue']
        markers = ['o']
    elif vnb_data is not None:
        # Only VNB available
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
        axes = [ax1]
        methods = ['VNB']
        data_sets = [vnb_data]
        colors = ['red']
        markers = ['s']
    else:
        logger.error(f"No data available for {galaxy_name}")
        return
    
    # Plot each method
    for i, (ax, method, data, color, marker) in enumerate(zip(axes, methods, data_sets, colors, markers)):
        # Plot data points
        ax.errorbar(data['radii_norm'], data['alpha_fe'], 
                   yerr=0.05, fmt=marker, color=color, markersize=8, linewidth=2,
                   label=f'{method} ({data["n_bins"]} bins)', capsize=5)
        
        # Plot fit line
        x_fit = np.linspace(0, np.max(data['radii_norm']) * 1.1, 100)
        y_fit = data['slope'] * x_fit + data['intercept']
        ax.plot(x_fit, y_fit, '--', color=color, alpha=0.8, linewidth=2)
        
        # Formatting
        ax.set_xlabel('R/Re', fontsize=12, fontweight='bold')
        ax.set_ylabel('[α/Fe] (dex)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set title with gradient info
        ax.set_title(f'{galaxy_name} - {method} Method\n'
                    f'Gradient: {data["slope"]:+.3f} ± {data["slope_error"]:.3f} dex/Re\n'
                    f'R² = {data["r_squared"]:.3f}, p = {data["p_value"]:.3f}',
                    fontsize=11)
    
    # Main title
    if len(methods) == 2:
        plt.suptitle(f'{galaxy_name}: α/Fe Radial Gradients (3-bin RDB vs Matching VNB Range)',
                    fontsize=14, fontweight='bold')
    else:
        plt.suptitle(f'{galaxy_name}: α/Fe Radial Gradients ({methods[0]} Method Only)',
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def run_enhanced_radial_analysis():
    """Run the enhanced radial analysis for all galaxies"""
    
    # Create output directory
    output_dir = "enhanced_radial_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all available galaxies
    galaxy_dirs = glob.glob("output/VCC*_stack")
    galaxy_names = [os.path.basename(d).replace("_stack", "") for d in galaxy_dirs]
    galaxy_names.sort()
    
    logger.info(f"Found {len(galaxy_names)} galaxies to process")
    
    # Process each galaxy
    all_results = []
    gradient_summary = []
    
    for galaxy_name in galaxy_names:
        logger.info(f"Processing {galaxy_name}...")
        
        galaxy_data = calculate_gradient_3bin_rdb_vnb(galaxy_name)
        
        if galaxy_data is not None:
            # Create individual plot
            plot_file = f"{output_dir}/{galaxy_name}_enhanced_3bin_gradient.png"
            create_enhanced_plot(galaxy_data, plot_file)
            
            all_results.append(galaxy_data)
            
            # Add to summary - handle both or single method cases
            rdb_data = galaxy_data['RDB']
            vnb_data = galaxy_data['VNB']
            
            if rdb_data is not None:
                gradient_summary.append({
                    'Galaxy': galaxy_name,
                    'Mode': 'RDB',
                    'Slope': rdb_data['slope'],
                    'Slope_Error': rdb_data['slope_error'],
                    'Intercept': rdb_data['intercept'],
                    'R_squared': rdb_data['r_squared'],
                    'P_value': rdb_data['p_value'],
                    'N_bins': rdb_data['n_bins'],
                    'Max_Radius_arcsec': rdb_data['max_radius'],
                    'Data_Quality': 'Both' if vnb_data is not None else 'RDB_only'
                })
            
            if vnb_data is not None:
                gradient_summary.append({
                    'Galaxy': galaxy_name,
                    'Mode': 'VNB',
                    'Slope': vnb_data['slope'],
                    'Slope_Error': vnb_data['slope_error'],
                    'Intercept': vnb_data['intercept'],
                    'R_squared': vnb_data['r_squared'],
                    'P_value': vnb_data['p_value'],
                    'N_bins': vnb_data['n_bins'],
                    'Max_Radius_arcsec': vnb_data['max_radius'],
                    'Data_Quality': 'Both' if rdb_data is not None else 'VNB_only'
                })
            
            # Log summary
            if rdb_data is not None and vnb_data is not None:
                logger.info(f"  ✓ {galaxy_name}: RDB {rdb_data['slope']:+.3f}±{rdb_data['slope_error']:.3f}, "
                           f"VNB {vnb_data['slope']:+.3f}±{vnb_data['slope_error']:.3f} dex/Re")
            elif rdb_data is not None:
                logger.info(f"  ◐ {galaxy_name}: RDB {rdb_data['slope']:+.3f}±{rdb_data['slope_error']:.3f} dex/Re (RDB only)")
            elif vnb_data is not None:
                logger.info(f"  ◑ {galaxy_name}: VNB {vnb_data['slope']:+.3f}±{vnb_data['slope_error']:.3f} dex/Re (VNB only)")
        else:
            logger.warning(f"  ✗ {galaxy_name}: Insufficient data")
    
    # Save summary
    if gradient_summary:
        summary_df = pd.DataFrame(gradient_summary)
        summary_file = f"{output_dir}/enhanced_3bin_gradient_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved gradient summary: {summary_file}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("ENHANCED 3-BIN RADIAL GRADIENT ANALYSIS SUMMARY")
        print("="*80)
        print(f"Successfully processed: {len(all_results)} galaxies")
        print(f"RDB method: 3 innermost bins only")
        print(f"VNB method: Bins within same radial range as RDB")
        print("-"*80)
        
        rdb_results = summary_df[summary_df['Mode'] == 'RDB']
        vnb_results = summary_df[summary_df['Mode'] == 'VNB']
        
        # Count data quality categories
        both_data = summary_df[summary_df['Data_Quality'] == 'Both']
        rdb_only = summary_df[summary_df['Data_Quality'] == 'RDB_only']
        vnb_only = summary_df[summary_df['Data_Quality'] == 'VNB_only']
        
        print(f"Data availability:")
        print(f"  Both RDB & VNB: {len(both_data)//2} galaxies")  # Divide by 2 since each galaxy has 2 rows when both available
        print(f"  RDB only: {len(rdb_only)} galaxies")
        print(f"  VNB only: {len(vnb_only)} galaxies")
        print()
        
        if len(rdb_results) > 0:
            print(f"RDB gradients ({len(rdb_results)} measurements):")
            print(f"  Mean: {rdb_results['Slope'].mean():+.3f} ± {rdb_results['Slope'].std():.3f} dex/Re")
            print(f"  Range: {rdb_results['Slope'].min():+.3f} to {rdb_results['Slope'].max():+.3f} dex/Re")
        
        if len(vnb_results) > 0:
            print(f"VNB gradients ({len(vnb_results)} measurements):")
            print(f"  Mean: {vnb_results['Slope'].mean():+.3f} ± {vnb_results['Slope'].std():.3f} dex/Re")
            print(f"  Range: {vnb_results['Slope'].min():+.3f} to {vnb_results['Slope'].max():+.3f} dex/Re")
        
        print("\nDetailed Results:")
        print(f"{'Galaxy':<10} {'Method':<6} {'Gradient':<12} {'Error':<8} {'R²':<6} {'Bins':<5} {'Quality':<10}")
        print("-"*70)
        for _, row in summary_df.iterrows():
            print(f"{row['Galaxy']:<10} {row['Mode']:<6} {row['Slope']:+.4f} {row['Slope_Error']:<8.4f} "
                  f"{row['R_squared']:<6.3f} {row['N_bins']:<5} {row['Data_Quality']:<10}")
        
        print("="*80)
    
    return all_results

if __name__ == "__main__":
    run_enhanced_radial_analysis()
