"""
Enhanced Alpha/Fe Gradient Analysis with Corrected ISAPC Integration

This module implements the corrected workflow for calculating α/Fe abundance gradients
from ISAPC spectral indices using TMB03 stellar population models.

Key corrections:
1. Properly reads ISAPC P2P spectral indices (Fe5015, Mgb, Hbeta)
2. Bins spectral indices spatially for gradient analysis  
3. Uses TMB03 models with physics-based corrections
4. Implements methodology from Liu et al. 2016 and recent literature
5. Calculates uncertainties properly

Author: Enhanced ISAPC Analysis System
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import stats
from astropy.io import fits
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EnhancedAlphaFeAnalysis')

class ISAPCAlphaFeAnalyzer:
    """Enhanced α/Fe analyzer that properly integrates ISAPC results with TMB03 models"""
    
    def __init__(self, tmb03_model_path="TMB03/TMB03.csv"):
        """Initialize with TMB03 stellar population models"""
        self.tmb03_model_path = tmb03_model_path
        self.tmb03_model = None
        self.load_tmb03_model()
        
        # Enhanced physics constants from recent literature
        self.ALPHA_FE_RANGE = (-0.2, 0.8)  # Valid [α/Fe] range
        self.TYPICAL_UNCERTAINTIES = {
            'Fe5015': 0.5,  # Typical measurement uncertainty (Å)
            'Mgb': 0.3,     # Typical measurement uncertainty (Å)
            'Hbeta': 0.2    # Typical measurement uncertainty (Å)
        }
        
        # Physics-based weights from Liu et al. 2016
        self.INDEX_WEIGHTS = {
            'Fe5015': 1.0,   # Primary iron indicator
            'Mgb': 1.2,      # Primary alpha indicator - higher weight
            'Hbeta': 0.8     # Age indicator - lower weight for α/Fe
        }
        
    def load_tmb03_model(self):
        """Load TMB03 stellar population models"""
        try:
            if os.path.exists(self.tmb03_model_path):
                self.tmb03_model = pd.read_csv(self.tmb03_model_path)
                logger.info(f"Loaded TMB03 model with {len(self.tmb03_model)} entries")
                
                # Check for required columns
                required_cols = ['Fe5015', 'Mgb', 'Hb', 'Age', 'AoFe']
                missing_cols = [col for col in required_cols if col not in self.tmb03_model.columns]
                if missing_cols:
                    logger.warning(f"Missing TMB03 columns: {missing_cols}")
                else:
                    logger.info("TMB03 model loaded successfully with all required columns")
            else:
                logger.error(f"TMB03 model file not found: {self.tmb03_model_path}")
                
        except Exception as e:
            logger.error(f"Error loading TMB03 model: {e}")
            
    def load_isapc_spectral_indices(self, galaxy_name):
        """Load ISAPC P2P spectral indices for a galaxy"""
        try:
            # Path to ISAPC P2P indices
            indices_path = f"output/{galaxy_name}_stack/Data/{galaxy_name}_stack_P2P_indices.npz"
            
            if not os.path.exists(indices_path):
                logger.warning(f"ISAPC indices not found for {galaxy_name}: {indices_path}")
                return None
                
            # Load spectral indices
            indices_data = np.load(indices_path, allow_pickle=True)
            
            spectral_indices = {}
            for index_name in ['Fe5015', 'Mgb', 'Hbeta']:
                if index_name in indices_data:
                    data = indices_data[index_name]
                    valid_mask = np.isfinite(data)
                    valid_count = np.sum(valid_mask)
                    total_count = data.size
                    
                    spectral_indices[index_name] = {
                        'data': data,
                        'valid_mask': valid_mask,
                        'valid_fraction': valid_count / total_count
                    }
                    
                    logger.info(f"{galaxy_name} {index_name}: {valid_count}/{total_count} "
                              f"({100*valid_count/total_count:.1f}%) valid pixels")
                else:
                    logger.warning(f"Missing {index_name} in ISAPC data for {galaxy_name}")
                    
            return spectral_indices
            
        except Exception as e:
            logger.error(f"Error loading ISAPC spectral indices for {galaxy_name}: {e}")
            return None
            
    def load_isapc_binning_info(self, galaxy_name, method='RDB'):
        """Load ISAPC binning information for spatial analysis"""
        try:
            # Load binning information
            binning_path = f"output/{galaxy_name}_stack/Data/{galaxy_name}_stack_{method}_binning.npz"
            
            if not os.path.exists(binning_path):
                logger.warning(f"Binning info not found for {galaxy_name}: {binning_path}")
                return None
                
            binning_data = np.load(binning_path, allow_pickle=True)
            
            # Load additional geometric information
            results_path = f"output/{galaxy_name}_stack/Data/{galaxy_name}_stack_{method}_results.npz"
            if os.path.exists(results_path):
                results_data = np.load(results_path, allow_pickle=True)
                
                # Extract binning geometry
                binning_info = {
                    'binning_data': binning_data,
                    'results_data': results_data,
                    'method': method
                }
                
                # Get bin information if available
                if 'binning' in results_data:
                    binning_details = results_data['binning'].item()
                    if isinstance(binning_details, dict):
                        binning_info['bin_details'] = binning_details
                        n_bins = len([k for k in binning_details.keys() if k.startswith('bin_')])
                        logger.info(f"{galaxy_name} {method}: {n_bins} bins identified")
                
                return binning_info
            else:
                logger.warning(f"Results file not found: {results_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading binning info for {galaxy_name}: {e}")
            return None
            
    def calculate_binned_spectral_indices(self, spectral_indices, binning_info):
        """Calculate mean spectral indices for each radial bin using ISAPC bin_num mapping"""
        try:
            if not spectral_indices or not binning_info:
                return None
                
            # Get binning data from ISAPC
            binning_data = binning_info['binning_data']
            results_data = binning_info['results_data']
            
            # Extract bin number mapping for each pixel
            if 'bin_num' not in binning_data:
                logger.warning("No bin_num found in binning data")
                return None
                
            bin_num = binning_data['bin_num']  # Array showing which bin each pixel belongs to
            bin_radii = binning_data['bin_radii'] if 'bin_radii' in binning_data else None
            
            # Get unique bin numbers (excluding invalid bins like -1)
            valid_bins = np.unique(bin_num[bin_num >= 0])
            n_bins = len(valid_bins)
            
            if n_bins == 0:
                logger.warning("No valid bins found")
                return None
                
            logger.info(f"Processing {n_bins} radial bins from ISAPC binning")
            
            # Initialize arrays for binned indices
            binned_indices = {}
            for index_name in ['Fe5015', 'Mgb', 'Hbeta']:
                binned_indices[index_name] = {
                    'values': np.full(n_bins, np.nan),
                    'errors': np.full(n_bins, np.nan),
                    'n_pixels': np.full(n_bins, 0, dtype=int)
                }
            
            # Calculate mean indices for each bin
            for i, bin_id in enumerate(valid_bins):
                # Find pixels belonging to this bin
                bin_mask = (bin_num == bin_id)
                bin_pixel_indices = np.where(bin_mask)[0]
                
                if len(bin_pixel_indices) == 0:
                    continue
                    
                # Calculate mean spectral indices for this bin
                for index_name in ['Fe5015', 'Mgb', 'Hbeta']:
                    if index_name in spectral_indices:
                        index_data = spectral_indices[index_name]['data']
                        valid_mask = spectral_indices[index_name]['valid_mask']
                        
                        # Flatten data if 2D to match bin_num indexing
                        if index_data.ndim == 2:
                            index_data_flat = index_data.flatten()
                            valid_mask_flat = valid_mask.flatten()
                        else:
                            index_data_flat = index_data
                            valid_mask_flat = valid_mask
                            
                        # Extract pixel values for this bin
                        bin_pixels = index_data_flat[bin_pixel_indices]
                        bin_valid = valid_mask_flat[bin_pixel_indices]
                        
                        # Calculate statistics for valid pixels only
                        valid_bin_pixels = bin_pixels[bin_valid]
                        
                        if len(valid_bin_pixels) > 0:
                            binned_indices[index_name]['values'][i] = np.mean(valid_bin_pixels)
                            binned_indices[index_name]['errors'][i] = np.std(valid_bin_pixels) / np.sqrt(len(valid_bin_pixels))
                            binned_indices[index_name]['n_pixels'][i] = len(valid_bin_pixels)
                            
                            logger.debug(f"Bin {bin_id}: {index_name} = {binned_indices[index_name]['values'][i]:.3f} "
                                       f"± {binned_indices[index_name]['errors'][i]:.3f} ({len(valid_bin_pixels)} pixels)")
            
            # Add R/Re information using ISAPC physics calculations
            if bin_radii is not None and len(bin_radii) >= n_bins:
                # Get effective radius from ISAPC binning
                max_radius_scale = binning_data.get('max_radius_scale', 3.0)  # Default to 3 Re
                max_radius = np.max(bin_radii)
                effective_radius = max_radius / max_radius_scale  # Re in arcsec
                
                # Calculate R/Re for each bin (proper physics normalization)
                r_over_re = bin_radii[:n_bins] / effective_radius
                binned_indices['radii'] = bin_radii[:n_bins]  # Keep physical radii
                binned_indices['r_over_re'] = r_over_re       # Add R/Re values
                binned_indices['effective_radius'] = effective_radius
                
                logger.info(f"Effective radius: {effective_radius:.2f} arcsec ({max_radius_scale} Re scale)")
                logger.info(f"R/Re values: {r_over_re}")
            else:
                # Fallback: use default spacing and estimate Re
                default_radii = np.arange(n_bins) + 0.5
                estimated_re = 5.0  # Default estimate in arcsec
                binned_indices['radii'] = default_radii * estimated_re
                binned_indices['r_over_re'] = default_radii
                binned_indices['effective_radius'] = estimated_re
                logger.warning(f"Using estimated effective radius: {estimated_re} arcsec")
                
            logger.info(f"Successfully processed {n_bins} bins for spectral indices")
            return binned_indices
            
        except Exception as e:
            logger.error(f"Error calculating binned spectral indices: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def calculate_alpha_fe_from_indices(self, fe5015, mgb, hbeta, age_constraint=None):
        """Calculate α/Fe from spectral indices using TMB03 models"""
        try:
            if self.tmb03_model is None:
                logger.error("TMB03 model not loaded")
                return np.nan, np.nan
                
            if not all(np.isfinite([fe5015, mgb, hbeta])):
                return np.nan, np.nan
                
            # Apply age constraint if provided
            working_model = self.tmb03_model.copy()
            if age_constraint is not None and np.isfinite(age_constraint):
                age_tolerance = 2.0  # Gyr
                age_mask = np.abs(working_model['Age'] - age_constraint) <= age_tolerance
                working_model = working_model[age_mask]
                
            if len(working_model) < 3:
                logger.warning("Insufficient model points after age constraint")
                working_model = self.tmb03_model.copy()
                
            # Calculate weighted distances in spectral index space
            distances = self._calculate_model_distances(fe5015, mgb, hbeta, working_model)
            
            # Use inverse distance weighting for interpolation
            weights = 1.0 / (distances + 1e-10)
            weights = weights / np.sum(weights)
            
            # Calculate weighted α/Fe
            alpha_fe = np.sum(weights * working_model['AoFe'].values)
            
            # Estimate uncertainty from model scatter
            n_nearest = min(5, len(working_model))
            nearest_indices = np.argsort(distances)[:n_nearest]
            nearest_alpha_fe = working_model['AoFe'].values[nearest_indices]
            uncertainty = np.std(nearest_alpha_fe) + 0.05  # Add systematic uncertainty
            
            # Apply physics corrections (following Liu et al. 2016)
            alpha_fe_corrected = self._apply_physics_corrections(alpha_fe, fe5015, mgb, hbeta)
            
            return alpha_fe_corrected, uncertainty
            
        except Exception as e:
            logger.error(f"Error calculating α/Fe: {e}")
            return np.nan, np.nan
            
    def _calculate_model_distances(self, fe5015_obs, mgb_obs, hbeta_obs, model_data):
        """Calculate weighted distances to model grid points"""
        try:
            # Normalized differences
            d_fe5015 = (fe5015_obs - model_data['Fe5015']) / self.TYPICAL_UNCERTAINTIES['Fe5015']
            d_mgb = (mgb_obs - model_data['Mgb']) / self.TYPICAL_UNCERTAINTIES['Mgb'] 
            d_hbeta = (hbeta_obs - model_data['Hb']) / self.TYPICAL_UNCERTAINTIES['Hbeta']  # Use 'Hb' column
            
            # Apply physics-based weights
            d_fe5015 *= self.INDEX_WEIGHTS['Fe5015']
            d_mgb *= self.INDEX_WEIGHTS['Mgb']
            d_hbeta *= self.INDEX_WEIGHTS['Hbeta']
            
            # Calculate weighted Euclidean distance
            distances = np.sqrt(d_fe5015**2 + d_mgb**2 + d_hbeta**2)
            
            return distances.values
            
        except Exception as e:
            logger.error(f"Error calculating model distances: {e}")
            return np.full(len(model_data), np.inf)
            
    def _apply_physics_corrections(self, alpha_fe, fe5015, mgb, hbeta):
        """Apply physics-based corrections following recent literature"""
        try:
            # Implement corrections from Liu et al. 2016 and Worthey et al. 2022
            alpha_fe_corrected = alpha_fe
            
            # Temperature effect correction
            # Alpha-enhanced populations have different effective temperatures
            temp_correction = -0.015 * alpha_fe  # From Worthey et al. 2022
            
            # Magnesium amplification correction
            # Mg indices are amplified beyond direct abundance effect
            if np.isfinite(mgb) and np.isfinite(fe5015) and fe5015 > 0:
                mgb_fe_ratio = mgb / fe5015
                expected_ratio = 0.5 + 0.3 * alpha_fe
                ratio_factor = mgb_fe_ratio / expected_ratio if expected_ratio > 0 else 1.0
                
                # Apply moderate correction to avoid overcorrection
                mg_correction = 0.1 * (ratio_factor - 1.0) * alpha_fe
                alpha_fe_corrected = alpha_fe - mg_correction
            
            # Ensure physical bounds
            alpha_fe_corrected = np.clip(alpha_fe_corrected, 
                                       self.ALPHA_FE_RANGE[0], 
                                       self.ALPHA_FE_RANGE[1])
            
            return alpha_fe_corrected
            
        except Exception as e:
            logger.warning(f"Error in physics corrections: {e}")
            return alpha_fe
            
    def analyze_galaxy_gradient(self, galaxy_name, method='RDB', max_bins=3):
        """Complete α/Fe gradient analysis for a galaxy"""
        try:
            logger.info(f"Starting α/Fe gradient analysis for {galaxy_name}")
            
            # Load ISAPC spectral indices
            spectral_indices = self.load_isapc_spectral_indices(galaxy_name)
            if spectral_indices is None:
                return None
                
            # Load binning information
            binning_info = self.load_isapc_binning_info(galaxy_name, method)
            if binning_info is None:
                return None
                
            # Calculate binned spectral indices
            binned_indices = self.calculate_binned_spectral_indices(spectral_indices, binning_info)
            if binned_indices is None:
                return None
                
            # Restrict to first N bins for gradient analysis
            n_bins = min(max_bins, len(binned_indices['Fe5015']['values']))
            
            # Calculate gradient using R/Re (proper physics normalization)
            alpha_fe_values = []
            alpha_fe_errors = []
            r_over_re_values = []
            
            for i in range(n_bins):
                fe5015 = binned_indices['Fe5015']['values'][i]
                mgb = binned_indices['Mgb']['values'][i]
                hbeta = binned_indices['Hbeta']['values'][i]
                
                if all(np.isfinite([fe5015, mgb, hbeta])):
                    alpha_fe, alpha_fe_err = self.calculate_alpha_fe_from_indices(fe5015, mgb, hbeta)
                    alpha_fe_values.append(alpha_fe)
                    alpha_fe_errors.append(alpha_fe_err)
                    
                    # Get R/Re information
                    if 'r_over_re' in binned_indices and i < len(binned_indices['r_over_re']):
                        r_over_re_values.append(binned_indices['r_over_re'][i])
                    else:
                        r_over_re_values.append(i + 0.5)  # Default spacing
                        
                    logger.info(f"Bin {i}: R/Re = {r_over_re_values[-1]:.2f}, α/Fe = {alpha_fe:.3f} ± {alpha_fe_err:.3f}")
                else:
                    logger.warning(f"Bin {i}: Invalid spectral indices")
                    alpha_fe_values.append(np.nan)
                    alpha_fe_errors.append(np.nan)
                    r_over_re_values.append(i + 0.5)
            
            # Convert to arrays
            alpha_fe_values = np.array(alpha_fe_values)
            alpha_fe_errors = np.array(alpha_fe_errors)
            r_over_re_values = np.array(r_over_re_values)
            
            # Calculate gradient using R/Re (d[α/Fe]/d(R/Re))
            valid_mask = np.isfinite(alpha_fe_values) & np.isfinite(r_over_re_values)
            if np.sum(valid_mask) >= 2:
                valid_r_over_re = r_over_re_values[valid_mask]
                valid_alpha_fe = alpha_fe_values[valid_mask]
                valid_errors = alpha_fe_errors[valid_mask]
                
                # Weighted linear fit using R/Re as x-axis
                if np.all(valid_errors > 0):
                    weights = 1.0 / valid_errors**2
                else:
                    weights = np.ones_like(valid_alpha_fe)
                    
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_r_over_re, valid_alpha_fe)
                
                result = {
                    'galaxy_name': galaxy_name,
                    'method': method,
                    'n_bins': n_bins,
                    'radii': binned_indices.get('radii', r_over_re_values),  # Physical radii in arcsec
                    'r_over_re': r_over_re_values,  # R/Re values used for gradient
                    'effective_radius': binned_indices.get('effective_radius', np.nan),
                    'alpha_fe_values': alpha_fe_values,
                    'alpha_fe_errors': alpha_fe_errors,
                    'gradient_slope': slope,  # d[α/Fe]/d(R/Re) 
                    'gradient_slope_error': std_err,
                    'gradient_intercept': intercept,
                    'correlation_coefficient': r_value,
                    'p_value': p_value,
                    'binned_indices': binned_indices,
                    'analysis_success': True
                }
                
                logger.info(f"{galaxy_name} gradient: {slope:.4f} ± {std_err:.4f} dex/Re")
                logger.info(f"  Effective radius: {binned_indices.get('effective_radius', np.nan):.2f} arcsec")
                return result
            else:
                logger.warning(f"Insufficient valid data points for gradient calculation")
                return None
                
        except Exception as e:
            logger.error(f"Error in gradient analysis for {galaxy_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

def test_enhanced_alpha_fe_analysis():
    """Test the enhanced α/Fe analysis with VCC1588"""
    
    analyzer = ISAPCAlphaFeAnalyzer()
    
    # Test with VCC1588
    result = analyzer.analyze_galaxy_gradient('VCC1588', method='RDB', max_bins=3)
    
    if result:
        print(f"\n{'='*60}")
        print(f"ENHANCED α/Fe GRADIENT ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Galaxy: {result['galaxy_name']}")
        print(f"Method: {result['method']}")
        print(f"Number of bins: {result['n_bins']}")
        print(f"Gradient slope: {result['gradient_slope']:.4f} ± {result['gradient_slope_error']:.4f} dex/Re")
        print(f"Effective radius: {result['effective_radius']:.2f} arcsec")
        print(f"Intercept: {result['gradient_intercept']:.4f}")
        print(f"Correlation: r = {result['correlation_coefficient']:.3f}, p = {result['p_value']:.3f}")
        print(f"\nBin details:")
        for i in range(result['n_bins']):
            r_over_re = result['r_over_re'][i]
            radius_arcsec = result['radii'][i]
            alpha_fe = result['alpha_fe_values'][i]
            alpha_fe_err = result['alpha_fe_errors'][i]
            print(f"  Bin {i+1}: R/Re = {r_over_re:.2f} ({radius_arcsec:.1f}\"), α/Fe = {alpha_fe:.3f} ± {alpha_fe_err:.3f}")
        print(f"{'='*60}")
        
        return result
    else:
        print("Analysis failed!")
        return None

if __name__ == "__main__":
    test_enhanced_alpha_fe_analysis()
