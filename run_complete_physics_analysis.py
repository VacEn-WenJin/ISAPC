#!/usr/bin/env python3
"""
Complete Physics Analysis for All Virgo Cluster Galaxies
Run alpha abundance gradient and velocity analysis for all available galaxies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import logging
import datetime
import glob
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from Phy_Visu import calculate_enhanced_alpha_fe

def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"complete_physics_analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return str(log_file)

def find_all_galaxies():
    """Find all galaxies with completed analysis results"""
    output_dir = Path("./output")
    if not output_dir.exists():
        logging.error("Output directory not found!")
        return []
    
    # Find all galaxy directories ending with _stack
    galaxy_dirs = list(output_dir.glob("*_stack"))
    
    galaxies = []
    seen_names = set()  # Prevent duplicates
    
    for galaxy_dir in galaxy_dirs:
        # Extract galaxy name (remove _stack suffix)
        galaxy_name = galaxy_dir.name.replace("_stack", "")
        
        # Skip if already seen
        if galaxy_name in seen_names:
            continue
        seen_names.add(galaxy_name)
        
        # Check if required files exist
        data_dir = galaxy_dir / "Data"
        if data_dir.exists():
            p2p_file = data_dir / f"{galaxy_dir.name}_P2P_results.npz"
            rdb_file = data_dir / f"{galaxy_dir.name}_RDB_results.npz"
            
            if p2p_file.exists() and rdb_file.exists():
                galaxies.append({
                    'name': galaxy_name,
                    'dir': galaxy_dir,
                    'p2p_file': p2p_file,
                    'rdb_file': rdb_file
                })
                logging.info(f"Found galaxy: {galaxy_name}")
            else:
                logging.warning(f"Incomplete data for {galaxy_name}")
    
    logging.info(f"Total galaxies found: {len(galaxies)}")
    return galaxies

def load_galaxy_data(galaxy_info):
    """Load data for a single galaxy"""
    try:
        galaxy_name = galaxy_info['name']
        
        # Load P2P data (spectral indices, stellar population, kinematics)
        p2p_data = np.load(galaxy_info['p2p_file'], allow_pickle=True)
        
        # Load RDB data (radial binning information)
        rdb_data = np.load(galaxy_info['rdb_file'], allow_pickle=True)
        
        # Extract P2P data
        indices = p2p_data['indices'].item()
        stellar_pop = p2p_data['stellar_population'].item()
        stellar_kin = p2p_data['stellar_kinematics'].item()
        
        # Extract RDB radial information
        distance_info = rdb_data['distance'].item()
        
        galaxy_data = {
            'name': galaxy_name,
            'spectral_indices': {
                'Fe5015': indices['Fe5015'],
                'Mgb': indices['Mgb'],
                'Hbeta': indices['Hbeta']
            },
            'stellar_population': {
                'age': stellar_pop['age'],
                'metallicity': stellar_pop['metallicity']
            },
            'kinematics': {
                'velocity': stellar_kin['velocity_field'],    # P2P corrected velocity
                'dispersion': stellar_kin['dispersion_field'] # P2P dispersion
            },
            'radial_info': {
                'effective_radius': distance_info['effective_radius'],
                'bin_distances': distance_info.get('bin_distances', [])
            }
        }
        
        # Data quality check
        fe5015 = galaxy_data['spectral_indices']['Fe5015']
        n_total = fe5015.size
        n_valid_spectral = np.sum(~np.isnan(fe5015))
        n_valid_velocity = np.sum(~np.isnan(galaxy_data['kinematics']['velocity']))
        
        logging.info(f"{galaxy_name}: {fe5015.shape} pixels, "
                    f"{n_valid_spectral}/{n_total} valid spectral, "
                    f"{n_valid_velocity}/{n_total} valid velocity")
        
        return galaxy_data
        
    except Exception as e:
        logging.error(f"Error loading {galaxy_info['name']}: {e}")
        return None

def analyze_single_galaxy(galaxy_data, model_data):
    """Perform complete physics analysis for a single galaxy"""
    galaxy_name = galaxy_data['name']
    logging.info(f"Analyzing {galaxy_name}...")
    
    results = {
        'galaxy_name': galaxy_name,
        'success': False,
        'n_pixels_total': 0,
        'n_pixels_quality': 0,
        'n_alpha_fe_calculated': 0,
        'alpha_fe_stats': {},
        'velocity_stats': {},
        'gradient_results': {},
        'errors': []
    }
    
    try:
        # Get data arrays
        fe5015 = galaxy_data['spectral_indices']['Fe5015']
        mgb = galaxy_data['spectral_indices']['Mgb']
        hbeta = galaxy_data['spectral_indices']['Hbeta']
        age = galaxy_data['stellar_population']['age']
        metallicity = galaxy_data['stellar_population']['metallicity']
        velocity = galaxy_data['kinematics']['velocity']
        dispersion = galaxy_data['kinematics']['dispersion']
        effective_radius = galaxy_data['radial_info']['effective_radius']
        
        results['n_pixels_total'] = fe5015.size
        
        # Apply quality filters
        quality_mask = (
            (~np.isnan(fe5015)) & (~np.isnan(mgb)) & (~np.isnan(hbeta)) &
            (~np.isnan(age)) & (~np.isnan(metallicity)) &
            (fe5015 > -2) & (fe5015 < 15) &    # Extended range for robustness
            (mgb > 0) & (mgb < 15) &
            (hbeta > 0) & (hbeta < 15) &
            (age > 0.1) & (age < 20)           # Extended age range
        )
        
        results['n_pixels_quality'] = np.sum(quality_mask)
        
        if results['n_pixels_quality'] == 0:
            results['errors'].append("No pixels pass quality filters")
            return results
        
        # Calculate alpha/Fe for quality pixels
        alpha_fe_2d = np.full_like(fe5015, np.nan)
        alpha_fe_uncertainties = []
        
        # Get coordinates of quality pixels
        quality_coords = np.where(quality_mask)
        
        # Limit calculations for efficiency (sample if too many pixels)
        max_pixels = 500  # Adjust based on computational resources
        n_quality = len(quality_coords[0])
        
        if n_quality > max_pixels:
            # Randomly sample pixels for alpha/Fe calculation
            indices = np.random.choice(n_quality, max_pixels, replace=False)
            sample_coords = (quality_coords[0][indices], quality_coords[1][indices])
            logging.info(f"{galaxy_name}: Sampling {max_pixels} of {n_quality} quality pixels")
        else:
            sample_coords = quality_coords
            logging.info(f"{galaxy_name}: Calculating alpha/Fe for all {n_quality} quality pixels")
        
        successful_calculations = 0
        alpha_fe_values = []
        
        for idx in range(len(sample_coords[0])):
            i, j = sample_coords[0][idx], sample_coords[1][idx]
            
            try:
                result = calculate_enhanced_alpha_fe(
                    fe5015[i, j], mgb[i, j], hbeta[i, j],
                    model_data, age[i, j], metallicity[i, j],
                    method='3d_interpolation'
                )
                
                if isinstance(result, tuple) and len(result) >= 4:
                    alpha_fe_val, _, _, uncertainty, _ = result
                    if not np.isnan(alpha_fe_val) and abs(alpha_fe_val) < 1.0:  # Reasonable range
                        alpha_fe_2d[i, j] = alpha_fe_val
                        alpha_fe_values.append(alpha_fe_val)
                        if not np.isnan(uncertainty):
                            alpha_fe_uncertainties.append(uncertainty)
                        successful_calculations += 1
                        
            except Exception as e:
                # Don't log individual pixel errors to avoid spam
                continue
        
        results['n_alpha_fe_calculated'] = successful_calculations
        
        # Alpha/Fe statistics
        if successful_calculations > 0:
            results['alpha_fe_stats'] = {
                'mean': float(np.mean(alpha_fe_values)),
                'std': float(np.std(alpha_fe_values)),
                'min': float(np.min(alpha_fe_values)),
                'max': float(np.max(alpha_fe_values)),
                'median': float(np.median(alpha_fe_values)),
                'mean_uncertainty': float(np.mean(alpha_fe_uncertainties)) if alpha_fe_uncertainties else np.nan
            }
            
            logging.info(f"{galaxy_name}: [α/Fe] = {results['alpha_fe_stats']['mean']:.3f} ± "
                        f"{results['alpha_fe_stats']['std']:.3f} ({successful_calculations} pixels)")
        
        # Velocity analysis (using P2P corrected velocity)
        velocity_valid = velocity[~np.isnan(velocity)]
        dispersion_valid = dispersion[~np.isnan(dispersion)]
        
        if len(velocity_valid) > 0 and len(dispersion_valid) > 0:
            results['velocity_stats'] = {
                'v_mean': float(np.mean(velocity_valid)),
                'v_std': float(np.std(velocity_valid)),
                'v_range': [float(np.min(velocity_valid)), float(np.max(velocity_valid))],
                'sigma_mean': float(np.mean(dispersion_valid)),
                'sigma_std': float(np.std(dispersion_valid)),
                'v_over_sigma': float(np.mean(np.abs(velocity_valid)) / np.mean(dispersion_valid)),
                'n_velocity_pixels': len(velocity_valid)
            }
            
            logging.info(f"{galaxy_name}: V/σ = {results['velocity_stats']['v_over_sigma']:.2f}, "
                        f"σ = {results['velocity_stats']['sigma_mean']:.1f} km/s")
        
        # Radial gradient analysis (if we have enough alpha/Fe measurements)
        if successful_calculations >= 10:  # Minimum for meaningful gradient
            try:
                # Create radial coordinate
                center_x, center_y = fe5015.shape[1] // 2, fe5015.shape[0] // 2
                y_coords, x_coords = np.mgrid[0:fe5015.shape[0], 0:fe5015.shape[1]]
                radial_distance_pix = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                
                # Convert to relative radius (assuming effective radius ~ 10 pixels, rough estimate)
                pixel_scale = effective_radius / 10 if effective_radius > 0 else 1.0
                relative_radius = (radial_distance_pix * pixel_scale) / effective_radius if effective_radius > 0 else radial_distance_pix
                
                # Get alpha/Fe and radius values for valid pixels
                valid_alpha_fe_mask = ~np.isnan(alpha_fe_2d)
                alpha_fe_for_gradient = alpha_fe_2d[valid_alpha_fe_mask]
                radius_for_gradient = relative_radius[valid_alpha_fe_mask]
                
                if len(alpha_fe_for_gradient) >= 10:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        radius_for_gradient, alpha_fe_for_gradient
                    )
                    
                    results['gradient_results'] = {
                        'slope': float(slope),
                        'slope_error': float(std_err),
                        'intercept': float(intercept),
                        'r_value': float(r_value),
                        'p_value': float(p_value),
                        'n_points': len(alpha_fe_for_gradient),
                        'significance': 'significant' if abs(slope) > 2*std_err else 'not_significant',
                        'gradient_type': 'negative' if slope < -2*std_err else ('positive' if slope > 2*std_err else 'flat')
                    }
                    
                    logging.info(f"{galaxy_name}: Gradient = {slope:.4f} ± {std_err:.4f} dex/(R/Re), "
                                f"p = {p_value:.4f}")
                
            except Exception as e:
                logging.warning(f"{galaxy_name}: Gradient analysis failed: {e}")
                results['errors'].append(f"Gradient analysis failed: {e}")
        
        # Mark as successful if we have meaningful results
        results['success'] = (successful_calculations > 0 and len(velocity_valid) > 0)
        
        if results['success']:
            logging.info(f"{galaxy_name}: Analysis completed successfully")
        else:
            logging.warning(f"{galaxy_name}: Analysis completed with limited results")
            
    except Exception as e:
        logging.error(f"{galaxy_name}: Analysis failed: {e}")
        results['errors'].append(str(e))
    
    return results

def save_results(all_results, output_file):
    """Save all results to CSV and summary files"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path(f"./physics_analysis_results")
    results_dir.mkdir(exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    for result in all_results:
        row = {
            'galaxy_name': result['galaxy_name'],
            'success': result['success'],
            'n_pixels_total': result['n_pixels_total'],
            'n_pixels_quality': result['n_pixels_quality'],
            'n_alpha_fe_calculated': result['n_alpha_fe_calculated'],
            'quality_fraction': result['n_pixels_quality'] / result['n_pixels_total'] if result['n_pixels_total'] > 0 else 0,
            'alpha_fe_fraction': result['n_alpha_fe_calculated'] / result['n_pixels_quality'] if result['n_pixels_quality'] > 0 else 0
        }
        
        # Add alpha/Fe statistics
        if result['alpha_fe_stats']:
            for key, value in result['alpha_fe_stats'].items():
                row[f'alpha_fe_{key}'] = value
        
        # Add velocity statistics  
        if result['velocity_stats']:
            for key, value in result['velocity_stats'].items():
                row[f'velocity_{key}'] = value
        
        # Add gradient results
        if result['gradient_results']:
            for key, value in result['gradient_results'].items():
                row[f'gradient_{key}'] = value
        
        csv_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    csv_file = results_dir / f"complete_physics_analysis_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    logging.info(f"Results saved to: {csv_file}")
    
    # Create summary report
    summary_file = results_dir / f"analysis_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Complete Physics Analysis Summary\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write(f"="*60 + "\n\n")
        
        successful = [r for r in all_results if r['success']]
        f.write(f"Total galaxies analyzed: {len(all_results)}\n")
        f.write(f"Successful analyses: {len(successful)}\n")
        f.write(f"Success rate: {100*len(successful)/len(all_results):.1f}%\n\n")
        
        if successful:
            # Overall statistics
            total_pixels = sum(r['n_pixels_total'] for r in successful)
            total_quality = sum(r['n_pixels_quality'] for r in successful)
            total_alpha_fe = sum(r['n_alpha_fe_calculated'] for r in successful)
            
            f.write(f"Overall Statistics:\n")
            f.write(f"  Total pixels: {total_pixels:,}\n")
            f.write(f"  Quality pixels: {total_quality:,} ({100*total_quality/total_pixels:.1f}%)\n")
            f.write(f"  Alpha/Fe calculated: {total_alpha_fe:,} ({100*total_alpha_fe/total_quality:.1f}% of quality)\n\n")
            
            # Alpha/Fe statistics across all galaxies
            alpha_fe_means = [r['alpha_fe_stats']['mean'] for r in successful if r['alpha_fe_stats']]
            if alpha_fe_means:
                f.write(f"Alpha/Fe Abundance Statistics:\n")
                f.write(f"  Mean [α/Fe]: {np.mean(alpha_fe_means):.3f} ± {np.std(alpha_fe_means):.3f}\n")
                f.write(f"  Range: {np.min(alpha_fe_means):.3f} to {np.max(alpha_fe_means):.3f}\n\n")
            
            # Gradient statistics
            gradients = [r['gradient_results'] for r in successful if r['gradient_results']]
            if gradients:
                slopes = [g['slope'] for g in gradients]
                significant = [g for g in gradients if g['significance'] == 'significant']
                
                f.write(f"Gradient Analysis:\n")
                f.write(f"  Galaxies with gradients: {len(gradients)}\n")
                f.write(f"  Significant gradients: {len(significant)}\n")
                f.write(f"  Mean slope: {np.mean(slopes):.4f} ± {np.std(slopes):.4f} dex/(R/Re)\n\n")
        
        # Individual galaxy details
        f.write(f"Individual Galaxy Results:\n")
        f.write("-" * 60 + "\n")
        for result in all_results:
            f.write(f"{result['galaxy_name']:12s}: ")
            if result['success']:
                alpha_fe_mean = result['alpha_fe_stats'].get('mean', 'N/A')
                n_alpha = result['n_alpha_fe_calculated']
                v_sigma = result['velocity_stats'].get('v_over_sigma', 'N/A')
                f.write(f"[α/Fe]={alpha_fe_mean:.3f} ({n_alpha:3d} pix), V/σ={v_sigma:.2f}")
                if result['gradient_results']:
                    slope = result['gradient_results']['slope']
                    sig = result['gradient_results']['significance']
                    f.write(f", grad={slope:.4f} ({sig})")
            else:
                f.write("FAILED")
                if result['errors']:
                    f.write(f" - {result['errors'][0]}")
            f.write("\n")
    
    logging.info(f"Summary saved to: {summary_file}")
    return csv_file, summary_file

def main():
    """Main analysis function"""
    print("=" * 80)
    print("Complete Physics Analysis for All Virgo Cluster Galaxies")
    print("Alpha abundance gradients and velocity analysis")
    print("=" * 80)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting complete physics analysis")
    logging.info(f"Log file: {log_file}")
    
    # Load TMB03 model
    logging.info("Loading TMB03 stellar population model...")
    try:
        model_data = pd.read_csv('./TMB03/TMB03.csv')
        logging.info(f"TMB03 model loaded: {model_data.shape[0]} stellar models")
    except Exception as e:
        logging.error(f"Failed to load TMB03 model: {e}")
        return False
    
    # Find all galaxies
    logging.info("Searching for analyzed galaxies...")
    galaxies = find_all_galaxies()
    
    if not galaxies:
        logging.error("No galaxies found for analysis!")
        return False
    
    # Analyze each galaxy
    all_results = []
    
    for i, galaxy_info in enumerate(galaxies, 1):
        logging.info(f"\n--- Galaxy {i}/{len(galaxies)}: {galaxy_info['name']} ---")
        
        # Load galaxy data
        galaxy_data = load_galaxy_data(galaxy_info)
        if galaxy_data is None:
            logging.error(f"Failed to load data for {galaxy_info['name']}")
            continue
        
        # Analyze galaxy
        result = analyze_single_galaxy(galaxy_data, model_data)
        all_results.append(result)
    
    # Save results
    logging.info("\nSaving results...")
    csv_file, summary_file = save_results(all_results, "complete_physics_analysis")
    
    # Final summary
    successful = [r for r in all_results if r['success']]
    logging.info(f"\n{'='*60}")
    logging.info(f"COMPLETE PHYSICS ANALYSIS FINISHED")
    logging.info(f"{'='*60}")
    logging.info(f"Total galaxies: {len(all_results)}")
    logging.info(f"Successful: {len(successful)} ({100*len(successful)/len(all_results):.1f}%)")
    logging.info(f"Results saved to: {csv_file}")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info(f"Log file: {log_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
