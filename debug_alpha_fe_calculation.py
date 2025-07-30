#!/usr/bin/env python3
"""
Debug the α/Fe calculation issues and implement the proper methodology
following Liu Yi-Qing (2020) and similar literature.

Key improvements needed:
1. Check actual spectral index values from ISAPC
2. Verify TMB03 model range coverage
3. Implement proper stellar population synthesis approach
4. Add diagnostic output for debugging
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DebugAlphaFe')

def debug_isapc_spectral_indices(galaxy_name='VCC1910'):
    """Debug ISAPC spectral indices to understand the data"""
    try:
        indices_path = f"output/{galaxy_name}_stack/Data/{galaxy_name}_stack_P2P_indices.npz"
        
        if not os.path.exists(indices_path):
            logger.error(f"ISAPC indices not found: {indices_path}")
            return
            
        indices_data = np.load(indices_path, allow_pickle=True)
        
        print(f"\n🔍 DEBUGGING SPECTRAL INDICES FOR {galaxy_name}")
        print("="*60)
        
        print(f"Available indices in file: {list(indices_data.keys())}")
        
        for index_name in ['Fe5015', 'Mgb', 'Hbeta']:
            if index_name in indices_data:
                data = indices_data[index_name]
                valid_mask = np.isfinite(data)
                valid_data = data[valid_mask]
                
                print(f"\n{index_name}:")
                print(f"  Shape: {data.shape}")
                print(f"  Valid pixels: {np.sum(valid_mask)}/{data.size} ({100*np.sum(valid_mask)/data.size:.1f}%)")
                if len(valid_data) > 0:
                    print(f"  Range: {np.min(valid_data):.3f} to {np.max(valid_data):.3f} Å")
                    print(f"  Mean ± Std: {np.mean(valid_data):.3f} ± {np.std(valid_data):.3f} Å")
                    print(f"  Median: {np.median(valid_data):.3f} Å")
                else:
                    print(f"  No valid data!")
            else:
                print(f"\n{index_name}: NOT FOUND")
                
    except Exception as e:
        logger.error(f"Error debugging spectral indices: {e}")

def debug_tmb03_model():
    """Debug TMB03 model to understand parameter coverage"""
    try:
        tmb03_path = "TMB03/TMB03.csv"
        if not os.path.exists(tmb03_path):
            logger.error(f"TMB03 model not found: {tmb03_path}")
            return
            
        tmb03 = pd.read_csv(tmb03_path)
        
        print(f"\n🔍 DEBUGGING TMB03 MODEL")
        print("="*60)
        
        print(f"Model shape: {tmb03.shape}")
        print(f"Columns: {list(tmb03.columns)}")
        
        # Check key spectral indices
        for index_name in ['Fe5015', 'Mgb', 'Hb']:
            if index_name in tmb03.columns:
                values = tmb03[index_name].values
                print(f"\n{index_name} in TMB03:")
                print(f"  Range: {np.min(values):.3f} to {np.max(values):.3f} Å")
                print(f"  Mean ± Std: {np.mean(values):.3f} ± {np.std(values):.3f} Å")
            else:
                print(f"\n{index_name}: NOT FOUND in TMB03")
                
        # Check [α/Fe] range
        if 'AoFe' in tmb03.columns:
            alpha_values = tmb03['AoFe'].values
            print(f"\n[α/Fe] in TMB03:")
            print(f"  Range: {np.min(alpha_values):.3f} to {np.max(alpha_values):.3f}")
            print(f"  Unique values: {sorted(np.unique(alpha_values))}")
            
        # Check Age range
        if 'Age' in tmb03.columns:
            age_values = tmb03['Age'].values
            print(f"\nAge in TMB03:")
            print(f"  Range: {np.min(age_values):.1f} to {np.max(age_values):.1f} Gyr")
            print(f"  Unique values: {sorted(np.unique(age_values))}")
            
        # Check metallicity range
        if 'ZoH' in tmb03.columns:
            metal_values = tmb03['ZoH'].values
            print(f"\n[Z/H] in TMB03:")
            print(f"  Range: {np.min(metal_values):.3f} to {np.max(metal_values):.3f}")
            
    except Exception as e:
        logger.error(f"Error debugging TMB03 model: {e}")

def compare_isapc_tmb03_ranges(galaxy_name='VCC1910'):
    """Compare ISAPC observed indices with TMB03 model ranges"""
    try:
        print(f"\n🔍 COMPARING {galaxy_name} INDICES WITH TMB03 MODEL")
        print("="*70)
        
        # Load ISAPC indices
        indices_path = f"output/{galaxy_name}_stack/Data/{galaxy_name}_stack_P2P_indices.npz"
        if not os.path.exists(indices_path):
            print("ISAPC indices not found")
            return
            
        indices_data = np.load(indices_path, allow_pickle=True)
        
        # Load TMB03 model
        tmb03_path = "TMB03/TMB03.csv"
        if not os.path.exists(tmb03_path):
            print("TMB03 model not found")
            return
            
        tmb03 = pd.read_csv(tmb03_path)
        
        # Compare ranges
        index_mapping = {'Fe5015': 'Fe5015', 'Mgb': 'Mgb', 'Hbeta': 'Hb'}
        
        for isapc_name, tmb03_name in index_mapping.items():
            if isapc_name in indices_data and tmb03_name in tmb03.columns:
                # ISAPC data
                isapc_data = indices_data[isapc_name]
                isapc_valid = isapc_data[np.isfinite(isapc_data)]
                
                # TMB03 data
                tmb03_data = tmb03[tmb03_name].values
                
                print(f"\n{isapc_name} ({tmb03_name}):")
                print(f"  ISAPC range:  {np.min(isapc_valid):.3f} to {np.max(isapc_valid):.3f} Å")
                print(f"  TMB03 range:  {np.min(tmb03_data):.3f} to {np.max(tmb03_data):.3f} Å")
                
                # Check overlap
                isapc_min, isapc_max = np.min(isapc_valid), np.max(isapc_valid)
                tmb03_min, tmb03_max = np.min(tmb03_data), np.max(tmb03_data)
                
                overlap_min = max(isapc_min, tmb03_min)
                overlap_max = min(isapc_max, tmb03_max)
                
                if overlap_max > overlap_min:
                    overlap_fraction = (overlap_max - overlap_min) / (isapc_max - isapc_min)
                    print(f"  Overlap:      {overlap_min:.3f} to {overlap_max:.3f} Å ({100*overlap_fraction:.1f}%)")
                else:
                    print(f"  Overlap:      NO OVERLAP!")
                    
    except Exception as e:
        logger.error(f"Error comparing ranges: {e}")

def test_single_bin_alpha_fe_calculation(galaxy_name='VCC1910', bin_index=0):
    """Test α/Fe calculation for a single bin with detailed diagnostics"""
    try:
        print(f"\n🔍 TESTING α/Fe CALCULATION FOR {galaxy_name} BIN {bin_index}")
        print("="*70)
        
        # Load the corrected analyzer
        from corrected_alpha_fe_analyzer import CorrectedAlphaFeAnalyzer
        analyzer = CorrectedAlphaFeAnalyzer()
        
        # Load spectral indices
        spectral_indices = analyzer.load_isapc_spectral_indices(galaxy_name)
        if not spectral_indices:
            print("Failed to load spectral indices")
            return
            
        # Load binning info
        binning_info = analyzer.load_isapc_binning_info(galaxy_name, 'RDB')
        if not binning_info:
            print("Failed to load binning info")
            return
            
        # Calculate binned indices
        binned_indices = analyzer.calculate_binned_spectral_indices(spectral_indices, binning_info)
        if not binned_indices:
            print("Failed to calculate binned indices")
            return
            
        # Get values for specific bin
        if bin_index >= len(binned_indices['Fe5015']['values']):
            print(f"Bin {bin_index} not available")
            return
            
        fe5015 = binned_indices['Fe5015']['values'][bin_index]
        mgb = binned_indices['Mgb']['values'][bin_index]
        hbeta = binned_indices['Hbeta']['values'][bin_index]
        
        fe5015_err = binned_indices['Fe5015']['errors'][bin_index]
        mgb_err = binned_indices['Mgb']['errors'][bin_index]
        hbeta_err = binned_indices['Hbeta']['errors'][bin_index]
        
        print(f"Observed spectral indices for bin {bin_index}:")
        print(f"  Fe5015: {fe5015:.3f} ± {fe5015_err:.3f} Å")
        print(f"  Mgb:    {mgb:.3f} ± {mgb_err:.3f} Å")
        print(f"  Hβ:     {hbeta:.3f} ± {hbeta_err:.3f} Å")
        
        if not all(np.isfinite([fe5015, mgb, hbeta])):
            print("Invalid spectral indices - cannot calculate α/Fe")
            return
            
        # Test chi-squared method
        print(f"\nTesting chi-squared method:")
        alpha_fe, alpha_fe_err, fit_params = analyzer.calculate_alpha_fe_chi2_method(
            fe5015, mgb, hbeta, fe5015_err, mgb_err, hbeta_err
        )
        
        print(f"  Result: [α/Fe] = {alpha_fe:.3f} ± {alpha_fe_err:.3f}")
        print(f"  Best fit parameters: {fit_params}")
        
        # Test marginalization method
        print(f"\nTesting marginalization method:")
        alpha_fe2, alpha_fe_err2, fit_params2 = analyzer.calculate_alpha_fe_marginalization(
            fe5015, mgb, hbeta, fe5015_err, mgb_err, hbeta_err
        )
        
        print(f"  Result: [α/Fe] = {alpha_fe2:.3f} ± {alpha_fe_err2:.3f}")
        print(f"  Best fit parameters: {fit_params2}")
        
        # Manual check: Find closest TMB03 models
        if analyzer.tmb03_model is not None:
            print(f"\nManual closest model search:")
            model_indices = analyzer.tmb03_model[['Fe5015', 'Mgb', 'Hb']].values
            obs_indices = np.array([fe5015, mgb, hbeta])
            
            # Calculate simple Euclidean distances
            distances = np.sqrt(np.sum((model_indices - obs_indices)**2, axis=1))
            closest_idx = np.argmin(distances)
            closest_model = analyzer.tmb03_model.iloc[closest_idx]
            
            print(f"  Closest model distance: {distances[closest_idx]:.3f}")
            print(f"  Closest model [α/Fe]: {closest_model['AoFe']:.3f}")
            print(f"  Closest model indices: Fe5015={closest_model['Fe5015']:.3f}, "
                  f"Mgb={closest_model['Mgb']:.3f}, Hb={closest_model['Hb']:.3f}")
            print(f"  Closest model Age: {closest_model['Age']:.1f} Gyr")
            print(f"  Closest model [Z/H]: {closest_model['ZoH']:.2f}")
            
    except Exception as e:
        logger.error(f"Error in single bin test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debugging function"""
    print("🔍 DEBUGGING α/Fe CALCULATION ISSUES")
    print("="*80)
    
    # Test galaxy
    test_galaxy = 'VCC1910'
    
    # Debug steps
    debug_isapc_spectral_indices(test_galaxy)
    debug_tmb03_model()
    compare_isapc_tmb03_ranges(test_galaxy)
    test_single_bin_alpha_fe_calculation(test_galaxy, bin_index=0)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
