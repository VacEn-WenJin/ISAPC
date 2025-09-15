#!/usr/bin/env python3
"""
Physics Visualization for All Galaxies
Alpha/Fe gradient analysis with enhanced error propagation and velocity correlation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import logging
import glob
from pathlib import Path
import time
from datetime import datetime
from galaxy_catalog import REDSHIFTS as GALAXY_REDSHIFTS, TYPES as GALAXY_TYPES, get_redshift

# Add current directory to path
sys.path.append('.')

# Import the Phy_Visu functions
from Phy_Visu import (
    calculate_enhanced_alpha_fe,
    get_enhanced_standardized_alpha_fe_data,
    calculate_enhanced_linear_gradients,
    coordinate_datasets_by_bins,
    estimate_alpha_fe_uncertainty
)

## Redshift/type mappings are imported from galaxy_catalog to avoid drift

def load_tmb03_model():
    """Load the TMB03 model data"""
    try:
        model_files = [
            './TMB03/TMB03.csv',
            './TMB03/TMB03_AOFe00.csv',
            './TMB03/TMB03_AOFe03.csv'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"Loading TMB03 model from: {model_file}")
                model_data = pd.read_csv(model_file)
                
                required_cols = ['Age', 'ZoH', 'Hb', 'Fe5015', 'Mgb', 'AoFe']
                if all(col in model_data.columns for col in required_cols):
                    print(f"✓ TMB03 model loaded: {model_data.shape[0]} entries")
                    return model_data
        
        print("❌ No suitable TMB03 model file found")
        return None
        
    except Exception as e:
        print(f"❌ Error loading TMB03 model: {e}")
        return None

def load_galaxy_data(galaxy_name):
    """Load galaxy data from ISAPC output files"""
    try:
        base_path = f'./output/{galaxy_name}_stack/Data'
        
        # Check which files exist
        p2p_file = f'{base_path}/{galaxy_name}_stack_P2P_results.npz'
        rdb_file = f'{base_path}/{galaxy_name}_stack_RDB_results.npz'
        vnb_file = f'{base_path}/{galaxy_name}_stack_VNB_results.npz'
        
        data = {'galaxy_name': galaxy_name}
        
        # Load P2P data (required)
        if os.path.exists(p2p_file):
            p2p_data = np.load(p2p_file, allow_pickle=True)
            
            # Extract spectral indices
            indices = p2p_data['indices'].item()
            data['Fe5015'] = indices['Fe5015']
            data['Mgb'] = indices['Mgb']
            data['Hbeta'] = indices['Hbeta']
            
            # Extract stellar population
            stellar_pop = p2p_data['stellar_population'].item()
            data['age'] = stellar_pop['age']
            data['metallicity'] = stellar_pop['metallicity']
            
            # Extract stellar kinematics
            stellar_kin = p2p_data['stellar_kinematics'].item()
            data['velocity'] = stellar_kin['velocity_field']
            data['dispersion'] = stellar_kin['dispersion_field']
            
            # Check for error data
            if 'stellar_kinematics_errors' in p2p_data:
                errors = p2p_data['stellar_kinematics_errors'].item()
                data['velocity_error'] = errors.get('velocity_error', None)
                data['dispersion_error'] = errors.get('dispersion_error', None)
            
            data['has_p2p'] = True
        else:
            print(f"⚠️  {galaxy_name}: No P2P data found")
            data['has_p2p'] = False
            return None
        
        # Load RDB data (optional)
        if os.path.exists(rdb_file):
            rdb_data = np.load(rdb_file, allow_pickle=True)
            
            binning = rdb_data['binning'].item()
            distance = rdb_data['distance'].item()
            
            data['bin_radii'] = distance['bin_distances']
            data['effective_radius'] = distance['effective_radius']
            data['has_rdb'] = True
        else:
            data['has_rdb'] = False
        
        # Load VNB data (optional)
        if os.path.exists(vnb_file):
            data['has_vnb'] = True
        else:
            data['has_vnb'] = False
        
        # Data quality assessment
        fe5015_valid = np.sum(~np.isnan(data['Fe5015']))
        mgb_valid = np.sum(~np.isnan(data['Mgb']))
        hbeta_valid = np.sum(~np.isnan(data['Hbeta']))
        
        data['n_valid_pixels'] = min(fe5015_valid, mgb_valid, hbeta_valid)
        data['data_shape'] = data['Fe5015'].shape
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading {galaxy_name}: {e}")
        return None

def calculate_alpha_fe_2d(galaxy_data, model_data, max_pixels=None):
    """Calculate alpha/Fe for 2D galaxy data"""
    try:
        fe5015 = galaxy_data['Fe5015']
        mgb = galaxy_data['Mgb']
        hbeta = galaxy_data['Hbeta']
        age = galaxy_data['age']
        metallicity = galaxy_data['metallicity']
        
        # Quality cuts
        reasonable_mask = (
            (fe5015 > -2) & (fe5015 < 10) &
            (mgb > 0) & (mgb < 10) &
            (hbeta > 0) & (hbeta < 10) &
            (age > 0.5) & (age < 15)
        )
        
        valid_mask = (~np.isnan(fe5015) & ~np.isnan(mgb) & ~np.isnan(hbeta) & 
                     ~np.isnan(age) & ~np.isnan(metallicity) & reasonable_mask)
        
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            return None, None, 0
        
        # Limit processing for very large datasets
        if max_pixels and n_valid > max_pixels:
            indices = np.where(valid_mask)
            selected = np.random.choice(len(indices[0]), max_pixels, replace=False)
            new_mask = np.zeros_like(valid_mask)
            new_mask[indices[0][selected], indices[1][selected]] = True
            valid_mask = new_mask
            n_valid = max_pixels
        
        # Initialize alpha/Fe array
        alpha_fe_2d = np.full_like(fe5015, np.nan)
        alpha_fe_errors = np.full_like(fe5015, np.nan)
        
        # Get valid pixel coordinates
        valid_coords = np.where(valid_mask)
        
        print(f"  Processing {n_valid} valid pixels...")
        
        successful = 0
        for i, (y, x) in enumerate(zip(valid_coords[0], valid_coords[1])):
            try:
                result = calculate_enhanced_alpha_fe(
                    fe5015[y, x], mgb[y, x], hbeta[y, x],
                    model_data, age[y, x], metallicity[y, x],
                    method='3d_interpolation'
                )
                
                if isinstance(result, tuple) and len(result) >= 4:
                    alpha_fe, _, _, uncertainty, _ = result
                    alpha_fe_2d[y, x] = alpha_fe
                    alpha_fe_errors[y, x] = uncertainty
                    successful += 1
                elif result is not None:
                    alpha_fe_2d[y, x] = result
                    successful += 1
                    
            except Exception as e:
                continue
        
        print(f"  Successful calculations: {successful}/{n_valid}")
        return alpha_fe_2d, alpha_fe_errors, successful
        
    except Exception as e:
        print(f"❌ Error in alpha/Fe calculation: {e}")
        return None, None, 0

def analyze_alpha_fe_gradients(galaxy_data, alpha_fe_2d, galaxy_name):
    """Analyze alpha/Fe gradients using radial binning"""
    try:
        if not galaxy_data.get('has_rdb', False):
            print(f"  No radial binning data for gradient analysis")
            return None
        
        # Simple center-based radial analysis as fallback
        center_y, center_x = np.array(alpha_fe_2d.shape) // 2
        y_coords, x_coords = np.mgrid[0:alpha_fe_2d.shape[0], 0:alpha_fe_2d.shape[1]]
        
        # Calculate distances from center
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Normalize by effective radius if available
        if 'effective_radius' in galaxy_data:
            # Convert pixel distance to physical distance (approximate)
            pixel_scale = 0.2  # arcsec/pixel (typical for MUSE)
            distances_arcsec = distances * pixel_scale
            distances_norm = distances_arcsec / galaxy_data['effective_radius']
        else:
            distances_norm = distances / np.max(distances)
        
        # Bin the data radially
        valid_mask = ~np.isnan(alpha_fe_2d)
        if np.sum(valid_mask) < 10:
            return None
        
        # Create radial bins
        max_radius = np.percentile(distances_norm[valid_mask], 90)
        bin_edges = np.linspace(0, max_radius, 6)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        alpha_fe_radial = []
        alpha_fe_errors = []
        
        for i in range(len(bin_edges) - 1):
            bin_mask = ((distances_norm >= bin_edges[i]) & 
                       (distances_norm < bin_edges[i+1]) & valid_mask)
            
            if np.sum(bin_mask) > 0:
                bin_values = alpha_fe_2d[bin_mask]
                alpha_fe_radial.append(np.nanmean(bin_values))
                alpha_fe_errors.append(np.nanstd(bin_values) / np.sqrt(np.sum(~np.isnan(bin_values))))
            else:
                alpha_fe_radial.append(np.nan)
                alpha_fe_errors.append(np.nan)
        
        alpha_fe_radial = np.array(alpha_fe_radial)
        alpha_fe_errors = np.array(alpha_fe_errors)
        
        # Linear fit
        valid_radial = ~np.isnan(alpha_fe_radial)
        if np.sum(valid_radial) >= 3:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                bin_centers[valid_radial], alpha_fe_radial[valid_radial]
            )
            
            gradient_result = {
                'bin_centers': bin_centers,
                'alpha_fe_radial': alpha_fe_radial,
                'alpha_fe_errors': alpha_fe_errors,
                'slope': slope,
                'slope_error': std_err,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'n_bins': np.sum(valid_radial)
            }
            
            return gradient_result
        
        return None
        
    except Exception as e:
        print(f"❌ Error in gradient analysis: {e}")
        return None

def process_galaxy(galaxy_name, model_data, output_dir):
    """Process a single galaxy for alpha/Fe analysis"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing {galaxy_name}")
        print(f"{'='*60}")
        
        # Load galaxy data
        galaxy_data = load_galaxy_data(galaxy_name)
        if galaxy_data is None:
            return None
        
        galaxy_type = GALAXY_TYPES.get(galaxy_name, 'Unknown')
        redshift = GALAXY_REDSHIFTS.get(galaxy_name, 0.004)
        
        print(f"Galaxy type: {galaxy_type}, Redshift: {redshift}")
        print(f"Data shape: {galaxy_data['data_shape']}")
        print(f"Valid pixels: {galaxy_data['n_valid_pixels']}")
        print(f"Available data: P2P={galaxy_data['has_p2p']}, "
              f"RDB={galaxy_data['has_rdb']}, VNB={galaxy_data['has_vnb']}")
        
        # Calculate alpha/Fe
        print("Calculating alpha/Fe abundance...")
        alpha_fe_2d, alpha_fe_errors, n_successful = calculate_alpha_fe_2d(
            galaxy_data, model_data, max_pixels=2000
        )
        
        if alpha_fe_2d is None or n_successful == 0:
            print(f"❌ No successful alpha/Fe calculations for {galaxy_name}")
            return None
        
        # Basic statistics
        alpha_fe_valid = alpha_fe_2d[~np.isnan(alpha_fe_2d)]
        mean_alpha_fe = np.mean(alpha_fe_valid)
        std_alpha_fe = np.std(alpha_fe_valid)
        
        print(f"Alpha/Fe statistics:")
        print(f"  Mean: {mean_alpha_fe:.3f} ± {std_alpha_fe:.3f}")
        print(f"  Range: {np.min(alpha_fe_valid):.3f} to {np.max(alpha_fe_valid):.3f}")
        print(f"  Valid pixels: {len(alpha_fe_valid)}")
        
        # Gradient analysis
        print("Analyzing radial gradients...")
        gradient_result = analyze_alpha_fe_gradients(galaxy_data, alpha_fe_2d, galaxy_name)
        
        if gradient_result:
            slope = gradient_result['slope']
            slope_err = gradient_result['slope_error']
            p_val = gradient_result['p_value']
            
            print(f"Radial gradient:")
            print(f"  Slope: {slope:.4f} ± {slope_err:.4f} dex/(R/Re)")
            print(f"  P-value: {p_val:.4f}")
            
            significance = "significant" if p_val < 0.05 else "not significant"
            direction = "negative" if slope < 0 else "positive"
            print(f"  {direction} gradient ({significance})")
        
        # Prepare results
        result = {
            'galaxy_name': galaxy_name,
            'galaxy_type': galaxy_type,
            'redshift': redshift,
            'data_shape': galaxy_data['data_shape'],
            'n_valid_pixels': galaxy_data['n_valid_pixels'],
            'n_successful_alpha_fe': n_successful,
            'has_rdb': galaxy_data['has_rdb'],
            'has_vnb': galaxy_data['has_vnb'],
            'mean_alpha_fe': mean_alpha_fe,
            'std_alpha_fe': std_alpha_fe,
            'alpha_fe_range': (np.min(alpha_fe_valid), np.max(alpha_fe_valid)),
            'alpha_fe_2d': alpha_fe_2d,
            'alpha_fe_errors': alpha_fe_errors,
            'gradient_result': gradient_result
        }
        
        # Save individual results
        galaxy_output_dir = os.path.join(output_dir, galaxy_name)
        os.makedirs(galaxy_output_dir, exist_ok=True)
        
        np.savez_compressed(
            os.path.join(galaxy_output_dir, f'{galaxy_name}_alpha_fe_analysis.npz'),
            **{k: v for k, v in result.items() if isinstance(v, (np.ndarray, float, int, str))}
        )
        
        print(f"✅ {galaxy_name} analysis completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error processing {galaxy_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_summary_report(results, output_dir):
    """Create a summary report of all galaxy analyses"""
    try:
        print(f"\n{'='*60}")
        print("CREATING SUMMARY REPORT")
        print(f"{'='*60}")
        
        # Filter successful results
        successful_results = [r for r in results if r is not None]
        n_successful = len(successful_results)
        n_total = len(results)
        
        print(f"Successfully processed: {n_successful}/{n_total} galaxies")
        
        if n_successful == 0:
            print("No successful analyses to summarize")
            return
        
        # Create summary DataFrame
        summary_data = []
        for result in successful_results:
            gradient = result.get('gradient_result')
            
            row = {
                'Galaxy': result['galaxy_name'],
                'Type': result['galaxy_type'],
                'Redshift': result['redshift'],
                'Shape': f"{result['data_shape'][0]}x{result['data_shape'][1]}",
                'Valid_Pixels': result['n_valid_pixels'],
                'Alpha_Fe_Success': result['n_successful_alpha_fe'],
                'Mean_Alpha_Fe': result['mean_alpha_fe'],
                'Std_Alpha_Fe': result['std_alpha_fe'],
                'Alpha_Fe_Min': result['alpha_fe_range'][0],
                'Alpha_Fe_Max': result['alpha_fe_range'][1],
                'Has_RDB': result['has_rdb'],
                'Has_VNB': result['has_vnb']
            }
            
            if gradient:
                row.update({
                    'Gradient_Slope': gradient['slope'],
                    'Gradient_Error': gradient['slope_error'],
                    'Gradient_P_Value': gradient['p_value'],
                    'Gradient_R_Value': gradient['r_value'],
                    'N_Gradient_Bins': gradient['n_bins']
                })
            else:
                row.update({
                    'Gradient_Slope': np.nan,
                    'Gradient_Error': np.nan,
                    'Gradient_P_Value': np.nan,
                    'Gradient_R_Value': np.nan,
                    'N_Gradient_Bins': 0
                })
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = os.path.join(output_dir, 'alpha_fe_analysis_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
        
        # Print key statistics
        print(f"\nKEY STATISTICS:")
        print(f"Mean alpha/Fe across all galaxies: {summary_df['Mean_Alpha_Fe'].mean():.3f}")
        print(f"Alpha/Fe range: {summary_df['Alpha_Fe_Min'].min():.3f} to {summary_df['Alpha_Fe_Max'].max():.3f}")
        
        # Gradient statistics
        significant_gradients = summary_df[summary_df['Gradient_P_Value'] < 0.05]
        print(f"Galaxies with significant gradients: {len(significant_gradients)}/{n_successful}")
        
        if len(significant_gradients) > 0:
            negative_gradients = significant_gradients[significant_gradients['Gradient_Slope'] < 0]
            positive_gradients = significant_gradients[significant_gradients['Gradient_Slope'] > 0]
            
            print(f"  Negative gradients: {len(negative_gradients)}")
            print(f"  Positive gradients: {len(positive_gradients)}")
        
        # By galaxy type
        print(f"\nBY GALAXY TYPE:")
        for gtype in summary_df['Type'].unique():
            if pd.notna(gtype):
                type_data = summary_df[summary_df['Type'] == gtype]
                mean_alpha = type_data['Mean_Alpha_Fe'].mean()
                n_type = len(type_data)
                print(f"  {gtype}: {n_type} galaxies, mean [α/Fe] = {mean_alpha:.3f}")
        
        return summary_df
        
    except Exception as e:
        print(f"❌ Error creating summary report: {e}")
        return None

def main():
    """Main function to process all galaxies"""
    print("="*80)
    print("PHYSICS VISUALIZATION: ALPHA/FE ANALYSIS FOR ALL GALAXIES")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'alpha_fe_analysis_results/analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load TMB03 model
    print("\nLoading TMB03 stellar population model...")
    model_data = load_tmb03_model()
    if model_data is None:
        print("❌ Failed to load TMB03 model. Cannot proceed.")
        return False
    
    # Find available galaxies
    galaxy_dirs = glob.glob('output/*_stack')
    available_galaxies = []
    
    for galaxy_dir in galaxy_dirs:
        galaxy_name = Path(galaxy_dir).name.replace('_stack', '')
        if galaxy_name in GALAXY_REDSHIFTS:
            available_galaxies.append(galaxy_name)
    
    available_galaxies.sort()
    print(f"\nFound {len(available_galaxies)} galaxies to process:")
    for i, galaxy in enumerate(available_galaxies, 1):
        gtype = GALAXY_TYPES.get(galaxy, 'Unknown')
        print(f"  {i:2d}. {galaxy} ({gtype})")
    
    # Process each galaxy
    results = []
    start_time = time.time()
    
    for i, galaxy_name in enumerate(available_galaxies, 1):
        print(f"\n{'='*20} GALAXY {i}/{len(available_galaxies)} {'='*20}")
        
        galaxy_start = time.time()
        result = process_galaxy(galaxy_name, model_data, output_dir)
        galaxy_time = time.time() - galaxy_start
        
        results.append(result)
        
        if result:
            print(f"✅ {galaxy_name} completed in {galaxy_time:.1f}s")
        else:
            print(f"❌ {galaxy_name} failed after {galaxy_time:.1f}s")
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(available_galaxies) - i)
        
        print(f"Progress: {i}/{len(available_galaxies)} | "
              f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
    
    # Create summary report
    summary_df = create_summary_report(results, output_dir)
    
    # Final summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r is not None])
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total galaxies processed: {len(available_galaxies)}")
    print(f"Successful analyses: {successful}")
    print(f"Failed analyses: {len(available_galaxies) - successful}")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Average time per galaxy: {total_time/len(available_galaxies):.1f} seconds")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved in: {output_dir}")
    
    if successful > 0:
        print(f"\n✅ SUCCESS: Alpha/Fe analysis completed for {successful} galaxies!")
        print("✅ Radial gradient analysis performed where possible")
        print("✅ Summary statistics generated")
        print("✅ Individual galaxy results saved")
    
    return successful > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
