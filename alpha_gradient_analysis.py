#!/usr/bin/env python3
"""
Alpha Abundance Gradient Analysis for Virgo Cluster Galaxies

This script implements proper alpha abundance gradient calculations following:
- Liu Yiqing et al. 2016 methodology
- Zhengzheng Li et al. 2019 approach
- Standard radial binning and gradient fitting techniques

Key features:
- Radial binning of 2D alpha/Fe maps
- Linear gradient fitting: d[α/Fe]/d(R/Re)
- Error propagation and uncertainty estimation
- Comprehensive visualization plots
- Individual galaxy gradient profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy import stats
from scipy.optimize import curve_fit
import os
import sys
from astropy.io import fits
import logging
from galaxy_catalog import REDSHIFTS as GALAXY_REDSHIFTS, get_redshift

# Add current directory to path
sys.path.append('.')


def setup_logging():
    """Setup logging for gradient analysis"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def load_galaxy_alpha_fe_data(galaxy_name, analysis_dir="alpha_fe_analysis_results/analysis_20250720_091707"):
    """
    Load 2D alpha/Fe data for a galaxy
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy identifier (e.g., 'VCC1588')
    analysis_dir : str
        Path to analysis results directory
        
    Returns:
    --------
    dict
        Dictionary containing alpha/Fe data and metadata
    """
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
            'has_rdb': bool(data['has_rdb'])
        }
        
        logger.info(f"Loaded alpha/Fe data for {galaxy_name}: {result['alpha_fe_2d'].shape}, {result['n_successful']} valid pixels")
        return result
        
    except Exception as e:
        logger.error(f"Error loading alpha/Fe data for {galaxy_name}: {e}")
        return None

def load_galaxy_vnb_data(galaxy_name):
    """
    Load VNB (Voronoi binned) data for spatial information
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy identifier
        
    Returns:
    --------
    dict
        Dictionary containing VNB spatial and binning information
    """
    try:
        vnb_path = f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_VNB_results.npz"
        
        if not os.path.exists(vnb_path):
            logger.warning(f"VNB data not found for {galaxy_name}: {vnb_path}")
            return None
            
        vnb_data = np.load(vnb_path, allow_pickle=True)
        
        # Extract distance information
        distance_info = vnb_data['distance'].item()
        binning_info = vnb_data['binning'].item()
        
        result = {
            'bin_distances': distance_info['bin_distances'],
            'effective_radius': distance_info['effective_radius'],
            'pixelsize_x': distance_info['pixelsize_x'],
            'pixelsize_y': distance_info['pixelsize_y'],
            'target_snr': binning_info['target_snr'],
            'n_pixels_per_bin': binning_info['n_pixels'],
            'binning_info': binning_info  # Include full binning info with bin_num
        }
        
        logger.info(f"Loaded VNB data for {galaxy_name}: {len(result['bin_distances'])} bins, Re = {result['effective_radius']:.3f}, SNR = {result['target_snr']}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading VNB data for {galaxy_name}: {e}")
        return None

def load_galaxy_rdb_data(galaxy_name):
    """
    Load RDB (radial binned) data for spatial information
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy identifier
        
    Returns:
    --------
    dict
        Dictionary containing spatial and binning information
    """
    try:
        rdb_path = f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_RDB_results.npz"
        
        if not os.path.exists(rdb_path):
            logger.warning(f"RDB data not found for {galaxy_name}: {rdb_path}")
            return None
            
        rdb_data = np.load(rdb_path, allow_pickle=True)
        
        # Extract distance information
        distance_info = rdb_data['distance'].item()
        binning_info = rdb_data['binning'].item()
        
        result = {
            'bin_distances': distance_info['bin_distances'],
            'effective_radius': distance_info['effective_radius'],
            'pixelsize_x': distance_info['pixelsize_x'],
            'pixelsize_y': distance_info['pixelsize_y'],
            'bin_radii': binning_info['bin_radii'],
            'center_x': binning_info['center_x'],
            'center_y': binning_info['center_y'],
            'binning_info': binning_info  # Include full binning info with bin_num
        }
        
        logger.info(f"Loaded RDB data for {galaxy_name}: {len(result['bin_distances'])} bins, Re = {result['effective_radius']:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading RDB data for {galaxy_name}: {e}")
        return None
    """
    Load RDB (radial binned) data for spatial information
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy identifier
        
    Returns:
    --------
    dict
        Dictionary containing spatial and binning information
    """
    try:
        rdb_path = f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_RDB_results.npz"
        
        if not os.path.exists(rdb_path):
            logger.warning(f"RDB data not found for {galaxy_name}: {rdb_path}")
            return None
            
        rdb_data = np.load(rdb_path, allow_pickle=True)
        
        # Extract distance information
        distance_info = rdb_data['distance'].item()
        binning_info = rdb_data['binning'].item()
        
        result = {
            'bin_distances': distance_info['bin_distances'],
            'effective_radius': distance_info['effective_radius'],
            'pixelsize_x': distance_info['pixelsize_x'],
            'pixelsize_y': distance_info['pixelsize_y'],
            'bin_radii': binning_info['bin_radii'],
            'center_x': binning_info['center_x'],
            'center_y': binning_info['center_y'],
            'binning_info': binning_info  # Include full binning info with bin_num
        }
        
        logger.info(f"Loaded RDB data for {galaxy_name}: {len(result['bin_distances'])} bins, Re = {result['effective_radius']:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading RDB data for {galaxy_name}: {e}")
        return None

def load_galaxy_p2p_data(galaxy_name):
    """
    Load P2P (pixel-to-pixel) data for velocity information
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy identifier
        
    Returns:
    --------
    dict
        Dictionary containing P2P velocity field and metadata
    """
    try:
        p2p_path = f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_P2P_results.npz"
        
        if not os.path.exists(p2p_path):
            logger.warning(f"P2P data not found for {galaxy_name}: {p2p_path}")
            return None
            
        p2p_data = np.load(p2p_path, allow_pickle=True)
        
        # Extract stellar kinematics (this is the velocity before redshift correction)
        stellar_kin = p2p_data['stellar_kinematics'].item()
        
        result = {
            'velocity_field': stellar_kin['velocity_field'],
            'dispersion_field': stellar_kin['dispersion_field'],
            'galaxy_name': galaxy_name
        }
        
        # Add velocity errors if available
        if 'stellar_kinematics_errors' in p2p_data:
            errors = p2p_data['stellar_kinematics_errors'].item()
            result['velocity_error'] = errors.get('velocity_error', None)
            result['dispersion_error'] = errors.get('dispersion_error', None)
        
        logger.info(f"Loaded P2P data for {galaxy_name}: velocity field shape {result['velocity_field'].shape}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading P2P data for {galaxy_name}: {e}")
        return None

def calculate_central_velocity(p2p_data, galaxy_name, radius_factor=0.5):
    """
    Calculate mean velocity of single pixels within central region (0.5 Re)
    This is the velocity before redshift correction in the binning stage
    
    Parameters:
    -----------
    p2p_data : dict
        P2P velocity data from load_galaxy_p2p_data
    galaxy_name : str
        Galaxy identifier for redshift lookup
    radius_factor : float
        Fraction of effective radius to use for central region (default 0.5)
        
    Returns:
    --------
    dict
        Central velocity statistics with real velocity (adding back redshift)
    """
    try:
        velocity_field = p2p_data['velocity_field']
        galaxy_redshift = GALAXY_REDSHIFTS.get(galaxy_name, 0.004)  # Default Virgo redshift
        
        # Get velocity field dimensions
        ny, nx = velocity_field.shape
        center_y, center_x = ny // 2, nx // 2
        
        # Create coordinate grids
        y, x = np.indices((ny, nx))
        
        # Calculate distance from center in pixels
        # Assume typical pixel scale of 0.2 arcsec/pixel for MUSE
        pixel_scale = 0.2  # arcsec per pixel
        
        # Estimate effective radius in pixels (rough approximation)
        # This is just for the 0.5 Re mask - we'll get the real Re from RDB data
        estimated_re_pix = 25  # approximately 5 arcsec in 0.2"/pixel
        
        # Create mask for central region (within radius_factor * Re)
        radius_pix = estimated_re_pix * radius_factor
        central_mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius_pix
        
        # Extract valid velocities in central region
        central_velocities = velocity_field[central_mask]
        valid_velocities = central_velocities[np.isfinite(central_velocities)]
        
        if len(valid_velocities) == 0:
            logger.warning(f"No valid velocities found in central region for {galaxy_name}")
            return None
        
        # Calculate statistics of corrected velocities (before redshift correction)
        mean_velocity_corrected = np.mean(valid_velocities)
        std_velocity_corrected = np.std(valid_velocities)
        
        # Add back the redshift to get real heliocentric velocity
        # The P2P velocity field has been corrected by subtracting the redshift
        # So real_velocity = corrected_velocity + z * c
        c_kms = 299792.458  # Speed of light in km/s
        redshift_velocity = galaxy_redshift * c_kms
        
        mean_velocity_real = mean_velocity_corrected + redshift_velocity
        
        result = {
            'galaxy_name': galaxy_name,
            'galaxy_redshift': galaxy_redshift,
            'redshift_velocity_kms': redshift_velocity,
            'n_central_pixels': len(valid_velocities),
            'central_radius_factor': radius_factor,
            'mean_velocity_corrected': mean_velocity_corrected,  # Before adding back redshift
            'std_velocity_corrected': std_velocity_corrected,
            'mean_velocity_real': mean_velocity_real,  # Real heliocentric velocity
            'velocity_range_corrected': (np.min(valid_velocities), np.max(valid_velocities)),
            'central_mask': central_mask
        }
        
        logger.info(f"Central velocity for {galaxy_name}: "
                   f"z={galaxy_redshift:.4f}, "
                   f"v_real={mean_velocity_real:.1f}±{std_velocity_corrected:.1f} km/s, "
                   f"N={len(valid_velocities)} pixels")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating central velocity for {galaxy_name}: {e}")
        return None

def calculate_radial_alpha_fe_profile(alpha_fe_data, rdb_data, min_pixels_per_bin=5):
    """
    Calculate radial alpha/Fe profile using proper binning
    
    This implements the methodology from Liu Yiqing 2016 and Zhengzheng 2019:
    - Radial binning of 2D alpha/Fe maps
    - Statistical analysis within each bin
    - Error propagation
    
    Parameters:
    -----------
    alpha_fe_data : dict
        2D alpha/Fe data from load_galaxy_alpha_fe_data
    rdb_data : dict
        RDB spatial data from load_galaxy_rdb_data
    min_pixels_per_bin : int
        Minimum number of valid pixels required per bin
        
    Returns:
    --------
    dict
        Radial profile data
    """
    try:
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        alpha_fe_errors = alpha_fe_data['alpha_fe_errors']

        # Support both flat and nested RDB NPZ layouts
        bin_distances = None
        effective_radius = None
        if isinstance(rdb_data, dict):
            if 'bin_distances' in rdb_data and 'effective_radius' in rdb_data:
                bin_distances = rdb_data['bin_distances']
                effective_radius = rdb_data['effective_radius']
            else:
                dist = rdb_data.get('distance')
                if dist is not None:
                    # distance may be an object array with dict inside
                    try:
                        dist = dist.item() if hasattr(dist, 'item') else dist
                    except Exception:
                        pass
                    if isinstance(dist, dict):
                        bin_distances = dist.get('bin_distances')
                        effective_radius = dist.get('effective_radius')
        if bin_distances is None or effective_radius is None:
            logger.error("RDB data missing bin_distances/effective_radius")
            return None
        
        # Get binning information from RDB data
        # The bin_num array tells us which bin each pixel belongs to
        binning_info = rdb_data.get('binning_info') if isinstance(rdb_data, dict) else None
        if binning_info is None and isinstance(rdb_data, dict):
            binfo = rdb_data.get('binning')
            if binfo is not None:
                try:
                    binfo = binfo.item() if hasattr(binfo, 'item') else binfo
                except Exception:
                    pass
                if isinstance(binfo, dict):
                    binning_info = binfo
        if binning_info is None:
            logger.warning("No binning info found in RDB data, cannot calculate profile")
            return None
        
        bin_num = binning_info.get('bin_num') if isinstance(binning_info, dict) else None
        if bin_num is None:
            logger.warning("Binning info lacks bin_num; cannot calculate profile")
            return None
        
        # Get unique bin numbers (excluding invalid/unassigned pixels)
        unique_bins = np.unique(bin_num)
        valid_bins_list = [b for b in unique_bins if b >= 0]  # Remove any negative values
        
        # Initialize profile arrays
        n_bins = len(bin_distances)
        radial_profile = {
            'bin_radii': bin_distances / effective_radius,  # In units of Re
            'bin_radii_kpc': bin_distances,  # In kpc
            'alpha_fe_mean': np.full(n_bins, np.nan),
            'alpha_fe_median': np.full(n_bins, np.nan),
            'alpha_fe_std': np.full(n_bins, np.nan),
            'alpha_fe_error': np.full(n_bins, np.nan),
            'n_pixels': np.zeros(n_bins, dtype=int),
            'valid_bins': []
        }
        
        # The bin_num array is 1D, but alpha_fe_2d is 2D, so we need to reshape
        ny, nx = alpha_fe_2d.shape
        if len(bin_num) != ny * nx:
            logger.warning(f"Dimension mismatch: bin_num length {len(bin_num)} != alpha_fe size {ny*nx}")
            # Try to handle this by reshaping what we can
            min_size = min(len(bin_num), ny * nx)
            bin_num_2d = np.full((ny, nx), -1)  # Initialize with invalid bin
            bin_num_2d.flat[:min_size] = bin_num[:min_size]
        else:
            bin_num_2d = bin_num.reshape(ny, nx)
        
        # Process each radial bin
        for bin_idx in valid_bins_list:
            if bin_idx >= n_bins:
                continue  # Skip bins beyond our distance array
                
            # Find pixels belonging to this bin
            bin_mask = (bin_num_2d == bin_idx)
            
            if not np.any(bin_mask):
                continue
                
            # Extract alpha/Fe values for this bin
            alpha_fe_values = alpha_fe_2d[bin_mask]
            error_values = alpha_fe_errors[bin_mask]
            
            # Filter out invalid values
            valid_mask = np.isfinite(alpha_fe_values) & np.isfinite(error_values)
            alpha_fe_valid = alpha_fe_values[valid_mask]
            errors_valid = error_values[valid_mask]
            
            n_valid = len(alpha_fe_valid)
            radial_profile['n_pixels'][bin_idx] = n_valid
            
            if n_valid < min_pixels_per_bin:
                logger.debug(f"Bin {bin_idx}: insufficient pixels ({n_valid} < {min_pixels_per_bin})")
                continue
            
            # Calculate statistics
            radial_profile['alpha_fe_mean'][bin_idx] = np.mean(alpha_fe_valid)
            radial_profile['alpha_fe_median'][bin_idx] = np.median(alpha_fe_valid)
            radial_profile['alpha_fe_std'][bin_idx] = np.std(alpha_fe_valid)
            
            # Error propagation: combine measurement errors and scatter
            measurement_error = np.sqrt(np.mean(errors_valid**2))  # RMS of measurement errors
            scatter_error = np.std(alpha_fe_valid) / np.sqrt(n_valid)  # Standard error of mean
            total_error = np.sqrt(measurement_error**2 + scatter_error**2)
            radial_profile['alpha_fe_error'][bin_idx] = total_error
            
            radial_profile['valid_bins'].append(bin_idx)
            
            logger.debug(f"Bin {bin_idx}: R/Re={radial_profile['bin_radii'][bin_idx]:.2f}, "
                        f"[α/Fe]={radial_profile['alpha_fe_mean'][bin_idx]:.3f}±{total_error:.3f}, "
                        f"N={n_valid}")
        
        radial_profile['effective_radius'] = effective_radius
        radial_profile['galaxy_name'] = alpha_fe_data['galaxy_name']
        
        logger.info(f"Calculated radial profile for {alpha_fe_data['galaxy_name']}: "
                   f"{len(radial_profile['valid_bins'])}/{n_bins} valid bins")
        
        return radial_profile
        
    except Exception as e:
        logger.error(f"Error calculating radial profile: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_vnb_alpha_fe_profile(alpha_fe_data, vnb_data, min_pixels_per_bin=5):
    """
    Calculate VNB-based alpha/Fe profile using Voronoi binning
    
    Similar to RDB profile but uses VNB binning structure
    
    Parameters:
    -----------
    alpha_fe_data : dict
        2D alpha/Fe data from load_galaxy_alpha_fe_data
    vnb_data : dict
        VNB spatial data from load_galaxy_vnb_data
    min_pixels_per_bin : int
        Minimum number of valid pixels required per bin
        
    Returns:
    --------
    dict
        VNB radial profile data
    """
    try:
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        alpha_fe_errors = alpha_fe_data['alpha_fe_errors']
        bin_distances = vnb_data['bin_distances']
        effective_radius = vnb_data['effective_radius']
        
        # Get binning information from VNB data
        binning_info = vnb_data.get('binning_info')
        if binning_info is None:
            logger.warning("No binning info found in VNB data, cannot calculate profile")
            return None
        
        bin_num = binning_info['bin_num']
        
        # Get unique bin numbers (excluding invalid/unassigned pixels)
        unique_bins = np.unique(bin_num)
        valid_bins_list = [b for b in unique_bins if b >= 0]  # Remove any negative values
        
        # Initialize profile arrays
        n_bins = len(bin_distances)
        vnb_profile = {
            'bin_radii': bin_distances / effective_radius,  # In units of Re
            'bin_radii_kpc': bin_distances,  # In kpc
            'alpha_fe_mean': np.full(n_bins, np.nan),
            'alpha_fe_median': np.full(n_bins, np.nan),
            'alpha_fe_std': np.full(n_bins, np.nan),
            'alpha_fe_error': np.full(n_bins, np.nan),
            'n_pixels': np.zeros(n_bins, dtype=int),
            'valid_bins': [],
            'target_snr': vnb_data.get('target_snr', 'N/A'),
            'binning_method': 'VNB'
        }
        
        # The bin_num array is 1D, but alpha_fe_2d is 2D, so we need to reshape
        ny, nx = alpha_fe_2d.shape
        if len(bin_num) != ny * nx:
            logger.warning(f"VNB dimension mismatch: bin_num length {len(bin_num)} != alpha_fe size {ny*nx}")
            # Try to handle this by reshaping what we can
            min_size = min(len(bin_num), ny * nx)
            bin_num_2d = np.full((ny, nx), -1)  # Initialize with invalid bin
            bin_num_2d.flat[:min_size] = bin_num[:min_size]
        else:
            bin_num_2d = bin_num.reshape(ny, nx)
        
        # Process each VNB bin
        for bin_idx in valid_bins_list:
            if bin_idx >= n_bins:
                continue  # Skip bins beyond our distance array
                
            # Find pixels belonging to this bin
            bin_mask = (bin_num_2d == bin_idx)
            
            if not np.any(bin_mask):
                continue
                
            # Extract alpha/Fe values for this bin
            alpha_fe_values = alpha_fe_2d[bin_mask]
            error_values = alpha_fe_errors[bin_mask]
            
            # Filter out invalid values
            valid_mask = np.isfinite(alpha_fe_values) & np.isfinite(error_values)
            alpha_fe_valid = alpha_fe_values[valid_mask]
            errors_valid = error_values[valid_mask]
            
            n_valid = len(alpha_fe_valid)
            vnb_profile['n_pixels'][bin_idx] = n_valid
            
            if n_valid < min_pixels_per_bin:
                logger.debug(f"VNB Bin {bin_idx}: insufficient pixels ({n_valid} < {min_pixels_per_bin})")
                continue
            
            # Calculate statistics
            vnb_profile['alpha_fe_mean'][bin_idx] = np.mean(alpha_fe_valid)
            vnb_profile['alpha_fe_median'][bin_idx] = np.median(alpha_fe_valid)
            vnb_profile['alpha_fe_std'][bin_idx] = np.std(alpha_fe_valid)
            
            # Error propagation: combine measurement errors and scatter
            measurement_error = np.sqrt(np.mean(errors_valid**2))  # RMS of measurement errors
            scatter_error = np.std(alpha_fe_valid) / np.sqrt(n_valid)  # Standard error of mean
            total_error = np.sqrt(measurement_error**2 + scatter_error**2)
            vnb_profile['alpha_fe_error'][bin_idx] = total_error
            
            vnb_profile['valid_bins'].append(bin_idx)
            
            logger.debug(f"VNB Bin {bin_idx}: R/Re={vnb_profile['bin_radii'][bin_idx]:.2f}, "
                        f"[α/Fe]={vnb_profile['alpha_fe_mean'][bin_idx]:.3f}±{total_error:.3f}, "
                        f"N={n_valid}")
        
        vnb_profile['effective_radius'] = effective_radius
        vnb_profile['galaxy_name'] = alpha_fe_data['galaxy_name']
        
        logger.info(f"Calculated VNB profile for {alpha_fe_data['galaxy_name']}: "
                   f"{len(vnb_profile['valid_bins'])}/{n_bins} valid bins (SNR={vnb_profile['target_snr']})")
        
        return vnb_profile
        
    except Exception as e:
        logger.error(f"Error calculating VNB profile: {e}")
        import traceback
        traceback.print_exc()
        return None

def fit_alpha_fe_gradient_multi_method(radial_profile, vnb_profile=None, fit_method='linear'):
    """
    Fit linear gradient to alpha/Fe radial profile using multiple methods:
    1. RDB method: First 3 bins only
    2. 1.5 Re method: All bins within 1.5 Re  
    3. 2.0 Re method: All bins within 2.0 Re
    4. VNB method: Voronoi binned data (if available)
    
    Following Liu Yiqing 2016 and Zhengzheng 2019 methodology:
    - Linear model: [α/Fe](R) = [α/Fe]₀ + ∇[α/Fe] × (R/Re)
    - Weighted least squares fitting
    - Robust uncertainty estimation
    
    Parameters:
    -----------
    radial_profile : dict
        Radial profile from calculate_radial_alpha_fe_profile (RDB-based)
    vnb_profile : dict, optional
        VNB-based radial profile
    fit_method : str
        Fitting method ('linear', 'weighted_linear')
        
    Returns:
    --------
    dict
        Gradient fitting results for all available methods
    """
    def fit_single_method(radii, alpha_fe, errors, method_name, max_radius=None, max_bins=None):
        """Fit gradient for a single method"""
        try:
            # Apply constraints
            if max_radius is not None:
                mask = radii <= max_radius
            elif max_bins is not None:
                mask = np.arange(len(radii)) < max_bins
            else:
                mask = np.ones(len(radii), dtype=bool)
            
            if np.sum(mask) < 3:
                logger.warning(f"Insufficient bins for {method_name}: {np.sum(mask)}")
                return None
            
            radii_fit = radii[mask]
            alpha_fe_fit = alpha_fe[mask]
            errors_fit = errors[mask]
            
            # Perform linear fit
            if fit_method == 'weighted_linear' and np.all(errors_fit > 0):
                # Weighted least squares
                weights = 1.0 / errors_fit**2
                slope, intercept, r_value, p_value, std_err = stats.linregress(radii_fit, alpha_fe_fit)
                
                # Recalculate with weights for better uncertainty
                W = np.sum(weights)
                Wx = np.sum(weights * radii_fit)
                Wy = np.sum(weights * alpha_fe_fit)
                Wxx = np.sum(weights * radii_fit**2)
                Wxy = np.sum(weights * radii_fit * alpha_fe_fit)
                
                det = W * Wxx - Wx**2
                if det > 0:
                    slope_weighted = (W * Wxy - Wx * Wy) / det
                    intercept_weighted = (Wxx * Wy - Wx * Wxy) / det
                    slope_error = np.sqrt(W / det)
                    intercept_error = np.sqrt(Wxx / det)
                    
                    slope, intercept = slope_weighted, intercept_weighted
                    std_err = slope_error
            else:
                # Ordinary least squares
                slope, intercept, r_value, p_value, std_err = stats.linregress(radii_fit, alpha_fe_fit)
            
            # Calculate additional statistics
            predicted = slope * radii_fit + intercept
            residuals = alpha_fe_fit - predicted
            chi_squared = np.sum((residuals / errors_fit)**2) if np.all(errors_fit > 0) else np.sum(residuals**2)
            reduced_chi_squared = chi_squared / (len(radii_fit) - 2) if len(radii_fit) > 2 else chi_squared
            
            # Classify gradient significance
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
            
            return {
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
                'residuals': residuals,
                'method': method_name
            }
            
        except Exception as e:
            logger.error(f"Error fitting {method_name}: {e}")
            return None
    
    try:
        valid_bins = radial_profile['valid_bins']
        
        if len(valid_bins) < 3:
            logger.warning(f"Insufficient bins for gradient fitting: {len(valid_bins)}")
            return None
        
        # Get data for valid bins
        radii = radial_profile['bin_radii'][valid_bins]
        alpha_fe = radial_profile['alpha_fe_mean'][valid_bins]
        errors = radial_profile['alpha_fe_error'][valid_bins]
        
        # Method 1: RDB method - first 3 bins only
        rdb_results = fit_single_method(radii, alpha_fe, errors, "RDB_3bins", max_bins=3)
        
        # Method 2: 1.5 Re method - all bins within 1.5 Re
        re_1p5_results = fit_single_method(radii, alpha_fe, errors, "1.5_Re", max_radius=1.5)
        
        # Method 3: 2.0 Re method - all bins within 2.0 Re  
        re_2p0_results = fit_single_method(radii, alpha_fe, errors, "2.0_Re", max_radius=2.0)
        
        # Method 4: VNB method - Voronoi binned data
        vnb_results = None
        if vnb_profile is not None:
            vnb_valid_bins = vnb_profile['valid_bins']
            if len(vnb_valid_bins) >= 3:
                vnb_radii = vnb_profile['bin_radii'][vnb_valid_bins]
                vnb_alpha_fe = vnb_profile['alpha_fe_mean'][vnb_valid_bins]
                vnb_errors = vnb_profile['alpha_fe_error'][vnb_valid_bins]
                # Don't apply radius constraint for VNB - use all available bins
                vnb_results = fit_single_method(vnb_radii, vnb_alpha_fe, vnb_errors, "VNB")
        
        # Compile all results
        multi_results = {
            'rdb_3bins': rdb_results,
            'radius_1p5_re': re_1p5_results,
            'radius_2p0_re': re_2p0_results,
            'vnb': vnb_results,
            'all_radii': radii,
            'all_alpha_fe': alpha_fe,
            'all_errors': errors
        }
        
        # Log comparison
        methods_available = [name for name, result in [('RDB_3bins', rdb_results), 
                                                      ('1.5_Re', re_1p5_results), 
                                                      ('2.0_Re', re_2p0_results),
                                                      ('VNB', vnb_results)] if result is not None]
        
        logger.info(f"Multi-method gradient analysis:")
        for method_name in methods_available:
            if method_name == 'RDB_3bins' and rdb_results:
                logger.info(f"  {method_name}: slope = {rdb_results['slope']:.4f} ± {rdb_results['slope_error']:.4f}, N={rdb_results['n_points']}")
            elif method_name == '1.5_Re' and re_1p5_results:
                logger.info(f"  {method_name}: slope = {re_1p5_results['slope']:.4f} ± {re_1p5_results['slope_error']:.4f}, N={re_1p5_results['n_points']}")
            elif method_name == '2.0_Re' and re_2p0_results:
                logger.info(f"  {method_name}: slope = {re_2p0_results['slope']:.4f} ± {re_2p0_results['slope_error']:.4f}, N={re_2p0_results['n_points']}")
            elif method_name == 'VNB' and vnb_results:
                logger.info(f"  {method_name}: slope = {vnb_results['slope']:.4f} ± {vnb_results['slope_error']:.4f}, N={vnb_results['n_points']}")
        
        return multi_results
        
    except Exception as e:
        logger.error(f"Error fitting alpha/Fe gradient: {e}")
        return None

def create_alpha_gradient_plots(galaxy_name, radial_profile, multi_gradient_results, 
                               alpha_fe_data, rdb_data, p2p_data=None, central_velocity_data=None,
                               vnb_profile=None, output_dir="alpha_gradient_plots"):
    """
    Create comprehensive alpha abundance gradient plots including velocity visualization
    and comparison of all three gradient methods
    
    Following the visualization style from Liu Yiqing 2016 and Zhengzheng 2019:
    - 2D alpha/Fe map with radial bins
    - 2D velocity map colored by central velocity
    - Radial profile with all three gradient fits
    - Residuals plots for each method
    - Statistical summary comparing methods
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with subplots - expanded for three methods
        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 2D Alpha/Fe map with radial bins
        ax1 = fig.add_subplot(gs[0, :2])
        alpha_fe_2d = alpha_fe_data['alpha_fe_2d']
        
        # Show 2D map
        im1 = ax1.imshow(alpha_fe_2d, origin='lower', cmap='RdYlBu_r', 
                        vmin=np.nanpercentile(alpha_fe_2d, 5),
                        vmax=np.nanpercentile(alpha_fe_2d, 95))
        
        # Overlay radial bins and boundaries
        if rdb_data is not None:
            effective_radius = rdb_data['effective_radius']
            center_x = rdb_data['center_x']
            center_y = rdb_data['center_y']
            
            # Draw bin boundaries for all three methods
            for i, radius_kpc in enumerate(rdb_data['bin_distances']):
                radius_re = radius_kpc / effective_radius
                if i in radial_profile['valid_bins']:
                    # Convert kpc to pixels using effective radius scale
                    radius_pix = radius_kpc / effective_radius * 20  # rough conversion
                    
                    # Color code by method
                    if i < 3:  # RDB method
                        color, alpha, style = 'green', 0.8, '-'
                    elif radius_re <= 1.5:  # 1.5 Re method
                        color, alpha, style = 'orange', 0.7, '--'
                    elif radius_re <= 2.0:  # 2.0 Re method
                        color, alpha, style = 'red', 0.6, ':'
                    else:
                        continue
                        
                    circle = Circle((center_x, center_y), radius_pix, 
                                  fill=False, color=color, linewidth=2, alpha=alpha, linestyle=style)
                    ax1.add_patch(circle)
            
            # Add method boundaries and legend
            for radius_re, color, label in [(1.5, 'orange', '1.5 Re'), (2.0, 'red', '2.0 Re')]:
                radius_pix = radius_re * 20
                circle = Circle((center_x, center_y), radius_pix, 
                              fill=False, color=color, linewidth=3, alpha=0.9)
                ax1.add_patch(circle)
                ax1.text(center_x + radius_pix*0.7, center_y + radius_pix*0.7, 
                        label, color=color, fontsize=10, fontweight='bold')
        
        ax1.set_title(f'{galaxy_name} - Alpha/Fe Map with Multi-Method Bins')
        ax1.set_xlabel('X [pixels]')
        ax1.set_ylabel('Y [pixels]')
        
        # Colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('[α/Fe] [dex]')
        
        # 2. 2D Velocity map colored by central velocity
        ax2 = fig.add_subplot(gs[0, 2:])
        
        if p2p_data is not None and central_velocity_data is not None:
            velocity_field = p2p_data['velocity_field']
            central_mask = central_velocity_data['central_mask']
            mean_velocity_real = central_velocity_data['mean_velocity_real']
            
            # Create velocity map with real velocities (add back redshift)
            redshift_velocity = central_velocity_data['redshift_velocity_kms']
            velocity_real = velocity_field + redshift_velocity
            
            # Show velocity map
            vmin = np.nanpercentile(velocity_real, 5)
            vmax = np.nanpercentile(velocity_real, 95)
            
            im2 = ax2.imshow(velocity_real, origin='lower', cmap='RdBu_r', 
                            vmin=vmin, vmax=vmax)
            
            # Overlay central region (0.5 Re)
            center_y, center_x = np.array(velocity_field.shape) // 2
            central_contour = ax2.contour(central_mask.astype(float), levels=[0.5], 
                                        colors='yellow', linewidths=2, alpha=0.8)
            ax2.text(center_x, center_y + 15, 
                    f'Central velocity\n(0.5 Re): {mean_velocity_real:.1f} km/s', 
                    ha='center', va='bottom', color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            ax2.set_title(f'{galaxy_name} - Real Velocity Field\n(z={central_velocity_data["galaxy_redshift"]:.4f} added back)')
            ax2.set_xlabel('X [pixels]')
            ax2.set_ylabel('Y [pixels]')
            
            # Colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Velocity [km/s]')
        else:
            ax2.text(0.5, 0.5, 'Velocity data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f'{galaxy_name} - Velocity Field (N/A)')
        
        # 3. Radial profile with all three gradient fits
        ax3 = fig.add_subplot(gs[1, :3])
        
        valid_bins = radial_profile['valid_bins']
        radii = radial_profile['bin_radii'][valid_bins]
        alpha_fe_mean = radial_profile['alpha_fe_mean'][valid_bins]
        alpha_fe_error = radial_profile['alpha_fe_error'][valid_bins]
        
        # Plot all data points
        ax3.errorbar(radii, alpha_fe_mean, yerr=alpha_fe_error, 
                    fmt='ko', capsize=3, capthick=1, markersize=6, alpha=0.7,
                    label='All bins')
        
        # Plot gradient fits for each method
        colors = ['green', 'orange', 'red']
        method_names = ['RDB (3 bins)', '1.5 Re', '2.0 Re']
        method_keys = ['rdb_3bins', 'radius_1p5_re', 'radius_2p0_re']
        
        for i, (method_key, color, method_name) in enumerate(zip(method_keys, colors, method_names)):
            gradient_result = multi_gradient_results.get(method_key)
            if gradient_result is not None:
                fit_radii = gradient_result['fit_radii']
                predicted = gradient_result['predicted']
                
                # Plot fit line
                max_r = np.max(fit_radii)
                r_fit_extended = np.linspace(0, max_r, 100)
                fit_extended = gradient_result['slope'] * r_fit_extended + gradient_result['intercept']
                ax3.plot(r_fit_extended, fit_extended, color=color, linewidth=2, 
                        label=f'{method_name}: {gradient_result["slope"]:.3f}±{gradient_result["slope_error"]:.3f}')
                
                # Highlight fitted points
                ax3.plot(fit_radii, alpha_fe_mean[radii <= np.max(fit_radii)], 
                        'o', color=color, markersize=8, alpha=0.8)
        
        # Mark method boundaries
        for radius, color, style in [(1.5, 'orange', '--'), (2.0, 'red', ':')]:
            ax3.axvline(x=radius, color=color, linestyle=style, alpha=0.7)
        
        ax3.set_xlabel('Radius [Re]')
        ax3.set_ylabel('[α/Fe] [dex]')
        ax3.set_title(f'{galaxy_name} - Alpha/Fe Profile: Multi-Method Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 2.5)
        
        # 4-6. Residuals plots for each method
        for i, (method_key, color, method_name) in enumerate(zip(method_keys, colors, method_names)):
            ax_res = fig.add_subplot(gs[2, i])
            
            gradient_result = multi_gradient_results.get(method_key)
            if gradient_result is not None:
                residuals = gradient_result['residuals']
                fit_radii = gradient_result['fit_radii']
                fit_errors = gradient_result['fit_errors']
                
                ax_res.errorbar(fit_radii, residuals, yerr=fit_errors, 
                            fmt='o', color=color, capsize=3, capthick=1)
                ax_res.axhline(y=0, color='black', linestyle='--', alpha=0.7)
                
                ax_res.set_xlabel('Radius [Re]')
                ax_res.set_ylabel('Residuals [dex]')
                ax_res.set_title(f'{method_name} Residuals')
                ax_res.grid(True, alpha=0.3)
            else:
                ax_res.text(0.5, 0.5, 'No fit\navailable', 
                           ha='center', va='center', transform=ax_res.transAxes)
                ax_res.set_title(f'{method_name} (N/A)')
        
        # 7. Statistical summary
        ax_summary = fig.add_subplot(gs[3:, :])
        ax_summary.axis('off')
        
        # Prepare summary text with comparison
        summary_text = f"Galaxy: {galaxy_name}\n"
        summary_text += f"Type: {alpha_fe_data.get('galaxy_type', 'Unknown')}\n\n"
        
        summary_text += "Alpha/Fe Statistics:\n"
        summary_text += f"Mean: {alpha_fe_data['mean_alpha_fe']:.3f} ± {alpha_fe_data['std_alpha_fe']:.3f}\n"
        summary_text += f"Valid pixels: {alpha_fe_data['n_successful']}\n\n"
        
        summary_text += "Radial Profile:\n"
        summary_text += f"Total valid bins: {len(radial_profile['valid_bins'])}\n"
        summary_text += f"Re: {radial_profile['effective_radius']:.3f} kpc\n\n"
        
        summary_text += "GRADIENT COMPARISON:\n"
        summary_text += "="*50 + "\n"
        
        # Compare all three methods
        for method_key, method_name in zip(method_keys, method_names):
            gradient_result = multi_gradient_results.get(method_key)
            if gradient_result is not None:
                summary_text += f"\n{method_name}:\n"
                summary_text += f"  Slope: {gradient_result['slope']:.4f} ± {gradient_result['slope_error']:.4f} [α/Fe]/Re\n"
                summary_text += f"  N_bins: {gradient_result['n_points']}\n"
                summary_text += f"  R²: {gradient_result['r_squared']:.3f}\n"
                summary_text += f"  p-value: {gradient_result['p_value']:.4f}\n"
                summary_text += f"  Significance: {gradient_result['significance']}\n"
                if 'max_radius' in gradient_result:
                    summary_text += f"  Max radius: {gradient_result.get('max_radius', 'N/A')} Re\n"
                elif 'max_bins' in gradient_result:
                    summary_text += f"  Max bins: {gradient_result.get('max_bins', 'N/A')}\n"
            else:
                summary_text += f"\n{method_name}: No fit available\n"
        
        # Add velocity information
        if central_velocity_data is not None:
            summary_text += f"\nCentral Velocity (0.5 Re):\n"
            summary_text += f"Redshift: z = {central_velocity_data['galaxy_redshift']:.4f}\n"
            summary_text += f"Real velocity: {central_velocity_data['mean_velocity_real']:.1f} ± {central_velocity_data['std_velocity_corrected']:.1f} km/s\n"
            summary_text += f"Central pixels: {central_velocity_data['n_central_pixels']}\n"
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # Save plot
        plt.suptitle(f'{galaxy_name} - Multi-Method Alpha Gradient Analysis + Velocity', 
                    fontsize=16, y=0.98)
        
        output_path = f"{output_dir}/{galaxy_name}_alpha_gradient_multi_method_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved multi-method gradient+velocity plot: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating multi-method gradient plots for {galaxy_name}: {e}")
        return None

def analyze_single_galaxy(galaxy_name):
    """
    Complete alpha abundance gradient analysis for a single galaxy
    Including velocity analysis with proper redshift correction
    Now supports VNB, RDB, and P2P methods
    """
    logger.info(f"Starting enhanced alpha gradient + velocity analysis for {galaxy_name}")
    
    # Load alpha/Fe data
    alpha_fe_data = load_galaxy_alpha_fe_data(galaxy_name)
    if alpha_fe_data is None:
        return None
    
    # Load RDB data for spatial information
    rdb_data = load_galaxy_rdb_data(galaxy_name)
    if rdb_data is None:
        logger.warning(f"No RDB data available for {galaxy_name}, cannot calculate radial profile")
        return None
    
    # Load VNB data for Voronoi binning comparison
    vnb_data = load_galaxy_vnb_data(galaxy_name)
    vnb_profile = None
    if vnb_data is not None:
        vnb_profile = calculate_vnb_alpha_fe_profile(alpha_fe_data, vnb_data)
        if vnb_profile is None:
            logger.warning(f"Could not calculate VNB profile for {galaxy_name}")
    else:
        logger.warning(f"No VNB data available for {galaxy_name}, skipping VNB analysis")
    
    # Load P2P data for velocity analysis
    p2p_data = load_galaxy_p2p_data(galaxy_name)
    central_velocity_data = None
    if p2p_data is not None:
        central_velocity_data = calculate_central_velocity(p2p_data, galaxy_name)
        if central_velocity_data is None:
            logger.warning(f"Could not calculate central velocity for {galaxy_name}")
    else:
        logger.warning(f"No P2P data available for {galaxy_name}, skipping velocity analysis")
    
    # Calculate radial profile (RDB-based)
    radial_profile = calculate_radial_alpha_fe_profile(alpha_fe_data, rdb_data)
    if radial_profile is None:
        return None
    
    # Fit gradients with multiple methods including VNB
    multi_gradient_results = fit_alpha_fe_gradient_multi_method(radial_profile, vnb_profile)
    
    # Create enhanced plots with velocity and multi-method comparison
    plot_path = create_alpha_gradient_plots(galaxy_name, radial_profile, multi_gradient_results, 
                                          alpha_fe_data, rdb_data, p2p_data, central_velocity_data, vnb_profile)
    
    # Compile results
    analysis_results = {
        'galaxy_name': galaxy_name,
        'alpha_fe_data': alpha_fe_data,
        'radial_profile': radial_profile,
        'vnb_profile': vnb_profile,
        'multi_gradient_results': multi_gradient_results,
        'p2p_data': p2p_data,
        'central_velocity_data': central_velocity_data,
        'plot_path': plot_path,
        'analysis_success': multi_gradient_results is not None and any(
            multi_gradient_results.get(key) is not None 
            for key in ['rdb_3bins', 'radius_1p5_re', 'radius_2p0_re', 'vnb']
        ),
        'velocity_success': central_velocity_data is not None,
        'vnb_success': vnb_profile is not None
    }
    
    return analysis_results

def main():
    """
    Main function to run multi-method alpha gradient analysis for all galaxies
    Includes velocity analysis and compares three different gradient methods:
    1. RDB method: First 3 bins only
    2. 1.5 Re method: All bins within 1.5 Re  
    3. 2.0 Re method: All bins within 2.0 Re
    """
    logger.info("Starting Multi-Method Alpha Abundance Gradient Analysis")
    logger.info("Following Liu Yiqing 2016 and Zhengzheng 2019 methodology")
    logger.info("Comparing RDB (3 bins), 1.5 Re, and 2.0 Re methods")
    logger.info("Enhanced with velocity analysis and proper redshift correction")
    
    # List of galaxies to analyze
    analysis_dir = "alpha_fe_analysis_results/analysis_20250720_091707"
    galaxy_dirs = [d for d in os.listdir(analysis_dir) 
                   if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith('VCC')]
    
    logger.info(f"Found {len(galaxy_dirs)} galaxies to analyze: {galaxy_dirs}")
    
    # Analyze each galaxy
    all_results = []
    successful_analyses = 0
    successful_velocity = 0
    
    for galaxy_name in sorted(galaxy_dirs):
        try:
            result = analyze_single_galaxy(galaxy_name)
            if result is not None:
                all_results.append(result)
                if result['analysis_success']:
                    successful_analyses += 1
                    logger.info(f"Successfully analyzed {galaxy_name} (multi-method gradients)")
                else:
                    logger.warning(f"Partial analysis for {galaxy_name} (no gradient fits)")
                
                if result['velocity_success']:
                    successful_velocity += 1
                    logger.info(f"Successfully analyzed {galaxy_name} (velocity)")
                else:
                    logger.warning(f"No velocity analysis for {galaxy_name}")
            else:
                logger.error(f"Failed to analyze {galaxy_name}")
        except Exception as e:
            logger.error(f"Error analyzing {galaxy_name}: {e}")
    
    # Summary
    logger.info(f"\nMulti-Method Analysis Summary:")
    logger.info(f"Total galaxies: {len(galaxy_dirs)}")
    logger.info(f"Successful multi-method analyses: {successful_analyses}")
    logger.info(f"Successful velocity analyses: {successful_velocity}")
    logger.info(f"Multi-method plots saved in: alpha_gradient_plots/")
    
    # Create multi-method summary table
    create_gradient_summary_table(all_results)
    
    return all_results

def create_gradient_summary_table(all_results):
    """
    Create enhanced summary table of gradient results including all three methods and velocity data
    """
    try:
        summary_data = []
        
        for result in all_results:
            if result['analysis_success']:
                multi_gradients = result['multi_gradient_results']
                profile = result['radial_profile']
                velocity_data = result['central_velocity_data']
                
                # Create one row per method
                for method_key, method_name in [('rdb_3bins', 'RDB_3bins'), 
                                              ('radius_1p5_re', '1.5_Re'), 
                                              ('radius_2p0_re', '2.0_Re')]:
                    gradient = multi_gradients.get(method_key)
                    if gradient is not None:
                        row = {
                            'Galaxy': result['galaxy_name'],
                            'Method': method_name,
                            'Slope': gradient['slope'],
                            'Slope_Error': gradient['slope_error'],
                            'Intercept': gradient['intercept'],
                            'R_squared': gradient['r_squared'],
                            'P_value': gradient['p_value'],
                            'Significance': gradient['significance'],
                            'Gradient_Type': gradient['gradient_type'],
                            'N_bins': gradient['n_points'],
                            'Re_kpc': profile['effective_radius'],
                            'Mean_Alpha_Fe': result['alpha_fe_data']['mean_alpha_fe'],
                            'Has_Velocity': result['velocity_success']
                        }
                        
                        # Add method-specific information
                        if method_name == 'RDB_3bins':
                            row['Method_Description'] = 'First 3 RDB bins'
                        elif method_name == '1.5_Re':
                            row['Method_Description'] = 'All bins ≤ 1.5 Re'
                        elif method_name == '2.0_Re':
                            row['Method_Description'] = 'All bins ≤ 2.0 Re'
                        
                        # Add velocity information if available (same for all methods)
                        if velocity_data is not None:
                            row.update({
                                'Galaxy_Redshift': velocity_data['galaxy_redshift'],
                                'Central_Velocity_Real_kms': velocity_data['mean_velocity_real'],
                                'Central_Velocity_Std_kms': velocity_data['std_velocity_corrected'],
                                'Central_N_Pixels': velocity_data['n_central_pixels'],
                                'Redshift_Velocity_kms': velocity_data['redshift_velocity_kms']
                            })
                        else:
                            row.update({
                                'Galaxy_Redshift': GALAXY_REDSHIFTS.get(result['galaxy_name'], np.nan),
                                'Central_Velocity_Real_kms': np.nan,
                                'Central_Velocity_Std_kms': np.nan,
                                'Central_N_Pixels': 0,
                                'Redshift_Velocity_kms': np.nan
                            })
                        
                        summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = "alpha_gradient_plots/multi_method_gradient_velocity_summary.csv"
            df.to_csv(summary_path, index=False, float_format='%.4f')
            logger.info(f"Saved multi-method gradient+velocity summary: {summary_path}")
            
            # Print formatted comparison table
            print("\nMulti-Method Alpha Abundance Gradient Comparison:")
            print("="*120)
            
            # Group by galaxy and show all methods side by side
            galaxies = df['Galaxy'].unique()
            for galaxy in sorted(galaxies):
                galaxy_data = df[df['Galaxy'] == galaxy]
                print(f"\n{galaxy}:")
                print("-" * 80)
                
                for _, row in galaxy_data.iterrows():
                    print(f"  {row['Method']:10s}: slope = {row['Slope']:7.4f} ± {row['Slope_Error']:6.4f}, "
                          f"p = {row['P_value']:6.4f}, N = {row['N_bins']:1.0f}, {row['Significance']}")
                
                # Show velocity info once per galaxy
                if not galaxy_data.empty and not pd.isna(galaxy_data.iloc[0]['Central_Velocity_Real_kms']):
                    vel_row = galaxy_data.iloc[0]
                    print(f"  Velocity   : {vel_row['Central_Velocity_Real_kms']:.1f} ± {vel_row['Central_Velocity_Std_kms']:.1f} km/s "
                          f"(z = {vel_row['Galaxy_Redshift']:.4f})")
            
            # Summary statistics by method
            print(f"\n\nSummary Statistics by Method:")
            print("="*80)
            
            for method in ['RDB_3bins', '1.5_Re', '2.0_Re']:
                method_data = df[df['Method'] == method]
                if not method_data.empty:
                    print(f"\n{method}:")
                    print(f"  Galaxies with fits: {len(method_data)}")
                    print(f"  Significant gradients (p<0.05): {(method_data['P_value'] < 0.05).sum()}")
                    print(f"  Mean slope: {method_data['Slope'].mean():.4f} ± {method_data['Slope'].std():.4f}")
                    print(f"  Mean |slope|: {method_data['Slope'].abs().mean():.4f}")
                    print(f"  Slope range: [{method_data['Slope'].min():.4f}, {method_data['Slope'].max():.4f}]")
    
    except Exception as e:
        logger.error(f"Error creating multi-method summary table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
