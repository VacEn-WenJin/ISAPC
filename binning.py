"""
Enhanced binning module for ISAPC with full error propagation support
Implements Voronoi and Radial binning with spatial covariance handling
"""

import logging
import numpy as np
from scipy import spatial, stats
from pathlib import Path
import warnings
from typing import List, Tuple, Optional, Union

# Import error propagation utilities
from utils.error_propagation import (
    propagate_binning_errors,
    calculate_covariance_matrix,
    validate_errors_rms
)

logger = logging.getLogger(__name__)


class BinnedData:
    """Base class for binned data with error support"""
    
    def __init__(self, bin_num, bin_indices, spectra, wavelength, 
                 errors=None, metadata=None):
        """
        Initialize binned data
        
        Parameters
        ----------
        bin_num : array
            Bin number for each spectrum
        bin_indices : list
            List of indices for each bin
        spectra : array
            Binned spectra (n_wave, n_bins)
        wavelength : array
            Wavelength array
        errors : array, optional
            Error array (n_wave, n_bins)
        metadata : dict, optional
            Additional metadata
        """
        self.bin_num = bin_num
        self.bin_indices = bin_indices
        self.spectra = spectra
        self.wavelength = wavelength
        self.errors = errors
        self.metadata = metadata if metadata is not None else {}
        self.n_bins = len(bin_indices)
        
    def get_bin_spectrum(self, bin_id):
        """Get spectrum for a specific bin with errors"""
        if 0 <= bin_id < self.n_bins:
            if self.errors is not None:
                return self.wavelength, self.spectra[:, bin_id], self.errors[:, bin_id]
            else:
                return self.wavelength, self.spectra[:, bin_id], None
        else:
            raise ValueError(f"Invalid bin_id: {bin_id}")
    
    def get_bin_info(self, bin_id):
        """Get information about a specific bin"""
        if 0 <= bin_id < self.n_bins:
            info = {
                'indices': self.bin_indices[bin_id],
                'n_pixels': len(self.bin_indices[bin_id])
            }
            
            # Add metadata specific to this bin if available
            for key in ['bin_x', 'bin_y', 'bin_xbar', 'bin_ybar', 'sn', 'bin_radii']:
                if key in self.metadata:
                    info[key] = self.metadata[key][bin_id]
                    
            return info
        else:
            raise ValueError(f"Invalid bin_id: {bin_id}")


class RadialBinnedData(BinnedData):
    """Radial binned data container with error support"""
    
    def __init__(self, bin_num, bin_indices, spectra, wavelength, 
                 errors=None, metadata=None, bin_radii=None):
        """
        Initialize radial binned data
        
        Parameters
        ----------
        bin_num : array
            Bin number for each spectrum
        bin_indices : list
            List of indices for each bin
        spectra : array
            Binned spectra (n_wave, n_bins)
        wavelength : array
            Wavelength array
        errors : array, optional
            Error array (n_wave, n_bins)
        metadata : dict, optional
            Additional metadata
        bin_radii : array, optional
            Radial distance for each bin
        """
        super().__init__(bin_num, bin_indices, spectra, wavelength, errors, metadata)
        self.bin_radii = bin_radii
        
    def save(self, filename):
        """Save binned data to file with error support"""
        save_dict = {
            'bin_num': self.bin_num,
            'bin_indices': self.bin_indices,
            'spectra': self.spectra,
            'wavelength': self.wavelength,
            'metadata': self.metadata,
            'bin_radii': self.bin_radii
        }
        
        # Add errors if available
        if self.errors is not None:
            save_dict['errors'] = self.errors
            
        np.savez_compressed(filename, **save_dict)
        logger.info(f"Saved radial binned data to {filename}")
        
    @classmethod
    def load(cls, filename):
        """Load binned data from file with error support"""
        data = np.load(filename, allow_pickle=True)
        
        # Extract errors if available
        errors = data['errors'] if 'errors' in data else None
        
        return cls(
            bin_num=data['bin_num'],
            bin_indices=data['bin_indices'].tolist(),
            spectra=data['spectra'],
            wavelength=data['wavelength'],
            errors=errors,
            metadata=data['metadata'].item() if 'metadata' in data else None,
            bin_radii=data['bin_radii'] if 'bin_radii' in data else None
        )


class VoronoiBinnedData(BinnedData):
    """Voronoi binned data container with error support"""
    
    def __init__(self, bin_num, bin_indices, spectra, wavelength, 
                 errors=None, metadata=None):
        """
        Initialize Voronoi binned data
        
        Parameters
        ----------
        bin_num : array
            Bin number for each spectrum
        bin_indices : list
            List of indices for each bin
        spectra : array
            Binned spectra (n_wave, n_bins)
        wavelength : array
            Wavelength array
        errors : array, optional
            Error array (n_wave, n_bins)
        metadata : dict, optional
            Additional metadata including bin positions
        """
        super().__init__(bin_num, bin_indices, spectra, wavelength, errors, metadata)
        
        # Extract bin positions from metadata if available
        if metadata:
            self.bin_x = metadata.get('bin_x', None)
            self.bin_y = metadata.get('bin_y', None)
            self.bin_xbar = metadata.get('bin_xbar', None)
            self.bin_ybar = metadata.get('bin_ybar', None)
            self.sn = metadata.get('sn', None)
            self.n_pixels = metadata.get('n_pixels', None)
        
    def save(self, filename):
        """Save binned data to file with error support"""
        save_dict = {
            'bin_num': self.bin_num,
            'bin_indices': self.bin_indices,
            'spectra': self.spectra,
            'wavelength': self.wavelength,
            'metadata': self.metadata
        }
        
        # Add errors if available
        if self.errors is not None:
            save_dict['errors'] = self.errors
            
        np.savez_compressed(filename, **save_dict)
        logger.info(f"Saved Voronoi binned data to {filename}")
        
    @classmethod
    def load(cls, filename):
        """Load binned data from file with error support"""
        data = np.load(filename, allow_pickle=True)
        
        # Extract errors if available
        errors = data['errors'] if 'errors' in data else None
        
        return cls(
            bin_num=data['bin_num'],
            bin_indices=data['bin_indices'].tolist(),
            spectra=data['spectra'],
            wavelength=data['wavelength'],
            errors=errors,
            metadata=data['metadata'].item() if 'metadata' in data else None
        )


def calculate_wavelength_intersection(wavelength, velocity_field, nx):
    """
    Calculate wavelength range that accommodates all velocity shifts
    
    Parameters
    ----------
    wavelength : array
        Original wavelength array
    velocity_field : array
        2D velocity field
    nx : int
        Number of x pixels
        
    Returns
    -------
    wave_mask : array
        Boolean mask for valid wavelengths
    min_wave : float
        Minimum wavelength
    max_wave : float
        Maximum wavelength
    """
    # Get velocity range
    valid_vel = velocity_field[np.isfinite(velocity_field)]
    if len(valid_vel) == 0:
        return np.ones_like(wavelength, dtype=bool), wavelength[0], wavelength[-1]
    
    vel_min = np.percentile(valid_vel, 1)
    vel_max = np.percentile(valid_vel, 99)
    
    # Add buffer for safety
    vel_range = vel_max - vel_min
    vel_min -= 0.1 * vel_range
    vel_max += 0.1 * vel_range
    
    # Calculate wavelength limits
    c_kms = 299792.458
    wave_min = wavelength[0] * (1 + vel_max / c_kms)
    wave_max = wavelength[-1] * (1 + vel_min / c_kms)
    
    # Create mask
    wave_mask = (wavelength >= wave_min) & (wavelength <= wave_max)
    
    logger.info(f"Velocity range: [{vel_min:.1f}, {vel_max:.1f}] km/s")
    logger.info(f"Wavelength range: [{wave_min:.1f}, {wave_max:.1f}] Å")
    logger.info(f"Using {np.sum(wave_mask)} of {len(wavelength)} wavelength points")
    
    return wave_mask, wave_min, wave_max


def combine_spectra_efficiently(spectra, wavelength, bin_indices, velocity_field, nx, ny, 
                               edge_treatment="extend", use_separate_velocity=False,
                               errors=None, use_covariance=True, correlation_length=2.0):
    """
    Efficiently combine spectra into bins with proper velocity correction and error propagation
    
    Parameters
    ----------
    spectra : ndarray
        2D array of spectra (n_wave, n_spec)
    wavelength : ndarray
        Wavelength array
    bin_indices : list
        List of arrays containing indices for each bin
    velocity_field : ndarray, optional
        2D velocity field for correction
    nx, ny : int
        Spatial dimensions
    edge_treatment : str
        Edge treatment method: 'extend', 'truncate', or 'interpolate'
    use_separate_velocity : bool
        Whether to use separate stellar and gas velocities
    errors : ndarray, optional
        2D array of errors (n_wave, n_spec)
    use_covariance : bool
        Whether to use spatial covariance in error propagation
    correlation_length : float
        Spatial correlation length in pixels
        
    Returns
    -------
    binned_spectra : ndarray
        Combined spectra (n_wave, n_bins)
    binned_errors : ndarray, optional
        Propagated errors if errors were provided
    """
    n_bins = len(bin_indices)
    n_wave = len(wavelength)
    
    binned_spectra = np.zeros((n_wave, n_bins))
    binned_errors = np.zeros((n_wave, n_bins)) if errors is not None else None
    
    # Pre-calculate spatial coordinates for covariance
    if use_covariance and errors is not None:
        x_coords = np.arange(nx * ny) % nx
        y_coords = np.arange(nx * ny) // nx
    
    for i, indices in enumerate(bin_indices):
        if len(indices) == 0:
            continue
            
        # Get spectra for this bin
        bin_spectra = spectra[:, indices].copy()
        bin_errors_subset = errors[:, indices].copy() if errors is not None else None
        
        # Handle velocity correction
        if velocity_field is not None:
            # Get pixel coordinates
            rows = indices // nx
            cols = indices % nx
            
            # Ensure we're within bounds
            rows = np.clip(rows, 0, ny-1)
            cols = np.clip(cols, 0, nx-1)
            
            velocities = velocity_field[rows, cols]
            
            # Check for gas velocity field if using separate velocities
            gas_velocities = None
            if use_separate_velocity and hasattr(velocity_field, 'gas_velocity_field'):
                if velocity_field.gas_velocity_field is not None:
                    gas_velocities = velocity_field.gas_velocity_field[rows, cols]
            
            # Apply velocity corrections
            for j, vel in enumerate(velocities):
                if np.isfinite(vel) and np.abs(vel) < 1000:  # Sanity check on velocity
                    # Apply velocity shift
                    z = vel / 299792.458  # Convert to redshift
                    wave_shifted = wavelength * (1 + z)
                    
                    # Handle edge treatment
                    if edge_treatment == "extend":
                        # Extend edge values
                        flux_extended = np.concatenate([
                            [bin_spectra[0, j]] * 10,
                            bin_spectra[:, j],
                            [bin_spectra[-1, j]] * 10
                        ])
                        wave_extended = np.concatenate([
                            wavelength[0] - np.arange(10, 0, -1) * (wavelength[1] - wavelength[0]),
                            wavelength,
                            wavelength[-1] + np.arange(1, 11) * (wavelength[-1] - wavelength[-2])
                        ])
                        
                        # Interpolate back to original grid
                        bin_spectra[:, j] = np.interp(wavelength, wave_shifted, flux_extended)
                        
                        # Also interpolate errors if available
                        if bin_errors_subset is not None:
                            error_extended = np.concatenate([
                                [bin_errors_subset[0, j]] * 10,
                                bin_errors_subset[:, j],
                                [bin_errors_subset[-1, j]] * 10
                            ])
                            bin_errors_subset[:, j] = np.interp(wavelength, wave_shifted, error_extended)
                    
                    elif edge_treatment == "truncate":
                        # Simple interpolation with NaN for out-of-bounds
                        bin_spectra[:, j] = np.interp(wavelength, wave_shifted, 
                                                     bin_spectra[:, j], 
                                                     left=np.nan, right=np.nan)
                        if bin_errors_subset is not None:
                            bin_errors_subset[:, j] = np.interp(wavelength, wave_shifted,
                                                               bin_errors_subset[:, j],
                                                               left=np.nan, right=np.nan)
                    
                    elif edge_treatment == "interpolate":
                        # Use scipy for better interpolation
                        from scipy.interpolate import interp1d
                        f = interp1d(wave_shifted, bin_spectra[:, j], 
                                   kind='linear', bounds_error=False, 
                                   fill_value='extrapolate')
                        bin_spectra[:, j] = f(wavelength)
                        
                        if bin_errors_subset is not None:
                            f_err = interp1d(wave_shifted, bin_errors_subset[:, j],
                                           kind='linear', bounds_error=False,
                                           fill_value='extrapolate')
                            bin_errors_subset[:, j] = f_err(wavelength)
        
        # Combine spectra with optimal weighting
        if bin_errors_subset is not None:
            # Handle covariance if requested and we have multiple spectra
            if use_covariance and len(indices) > 1 and len(indices) < 100:
                # Calculate distances for this subset
                x_subset = x_coords[indices]
                y_subset = y_coords[indices]
                
                # Calculate pairwise distances
                dx = x_subset[:, np.newaxis] - x_subset[np.newaxis, :]
                dy = y_subset[:, np.newaxis] - y_subset[np.newaxis, :]
                dist_subset = np.sqrt(dx**2 + dy**2)
                
                # Calculate covariance matrix for this bin
                cov_matrix = calculate_covariance_matrix(
                    bin_spectra.T, dist_subset, correlation_length
                )
                
                # Use covariance for optimal combination
                try:
                    # Inverse covariance weighting
                    cov_inv = np.linalg.inv(cov_matrix)
                    weights = cov_inv.sum(axis=1) / cov_inv.sum()
                    
                    # Combine spectra
                    binned_spectra[:, i] = np.average(bin_spectra, weights=weights, axis=1)
                    
                    # Propagate errors
                    for k in range(n_wave):
                        # Scale covariance by wavelength-dependent errors
                        cov_scaled = cov_matrix * np.outer(bin_errors_subset[k, :], 
                                                          bin_errors_subset[k, :])
                        var = np.dot(weights, np.dot(cov_scaled, weights))
                        binned_errors[k, i] = np.sqrt(var)
                        
                except np.linalg.LinAlgError:
                    # Fall back to inverse variance weighting
                    logger.debug(f"Covariance matrix singular for bin {i}, using inverse variance weighting")
                    use_covariance = False
            
            if not use_covariance or len(indices) == 1:
                # Simple inverse variance weighting
                # Avoid division by zero
                safe_errors = np.maximum(bin_errors_subset, 1e-10)
                weights = 1.0 / safe_errors**2
                
                # Normalize weights
                weight_sum = np.sum(weights, axis=1, keepdims=True)
                weight_sum[weight_sum == 0] = 1.0  # Avoid division by zero
                weights /= weight_sum
                
                # Combine spectra
                binned_spectra[:, i] = np.sum(bin_spectra * weights, axis=1)
                
                # Propagate errors
                binned_errors[:, i] = 1.0 / np.sqrt(np.sum(1.0 / safe_errors**2, axis=1))
        else:
            # No errors provided - use simple mean
            valid_mask = np.isfinite(bin_spectra)
            n_valid = np.sum(valid_mask, axis=1)
            n_valid[n_valid == 0] = 1  # Avoid division by zero
            
            binned_spectra[:, i] = np.nansum(bin_spectra, axis=1) / n_valid
    
    if errors is not None:
        return binned_spectra, binned_errors
    else:
        return binned_spectra


def calculate_radial_bins(x, y, center_x=None, center_y=None, pa=0, ellipticity=0,
                         n_rings=10, log_spacing=False, r_galaxy=None):
    """
    Calculate radial bins with optional elliptical geometry
    
    Parameters
    ----------
    x, y : array
        Coordinates
    center_x, center_y : float, optional
        Center coordinates
    pa : float
        Position angle in degrees
    ellipticity : float
        Ellipticity (1 - b/a)
    n_rings : int
        Number of radial bins
    log_spacing : bool
        Use logarithmic spacing
    r_galaxy : array, optional
        Pre-calculated physical radii
        
    Returns
    -------
    bin_num : array
        Bin number for each position
    bin_edges : array
        Radial bin edges
    bin_radii : array
        Mean radius for each bin
    """
    # Default to center
    if center_x is None:
        center_x = np.mean(x)
    if center_y is None:
        center_y = np.mean(y)
    
    # Calculate elliptical radius if r_galaxy not provided
    if r_galaxy is None:
        # Convert PA to radians
        pa_rad = np.radians(pa)
        
        # Rotate coordinates
        dx = x - center_x
        dy = y - center_y
        
        x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
        y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
        
        # Calculate elliptical radius
        b_over_a = 1 - ellipticity
        r_ellipse = np.sqrt(x_rot**2 + (y_rot / b_over_a)**2)
    else:
        r_ellipse = r_galaxy
    
    # Determine bin edges
    r_min = 0
    r_max = np.nanmax(r_ellipse)
    
    if log_spacing:
        # Logarithmic spacing
        r_min_log = np.log10(max(r_min, 0.1))
        r_max_log = np.log10(r_max)
        bin_edges = np.logspace(r_min_log, r_max_log, n_rings + 1)
    else:
        # Linear spacing
        bin_edges = np.linspace(r_min, r_max, n_rings + 1)
    
    # Assign bins
    bin_num = np.digitize(r_ellipse, bin_edges) - 1
    
    # Handle edge cases
    bin_num[bin_num < 0] = 0
    bin_num[bin_num >= n_rings] = n_rings - 1
    
    # Calculate mean radius for each bin
    bin_radii = np.zeros(n_rings)
    for i in range(n_rings):
        mask = bin_num == i
        if np.any(mask):
            bin_radii[i] = np.mean(r_ellipse[mask])
        else:
            bin_radii[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
    
    return bin_num, bin_edges, bin_radii


def calculate_radial_bins_re_based(x, y, center_x=None, center_y=None, 
                                  pa=0, ellipticity=0, effective_radius=1.0,
                                  r_galaxy=None, max_radius_scale=3.0, n_bins=10):
    """
    Calculate radial bins based on effective radius (Re) units
    
    Parameters
    ----------
    x, y : array
        Coordinates
    center_x, center_y : float, optional
        Center coordinates
    pa : float
        Position angle in degrees
    ellipticity : float
        Ellipticity (1 - b/a)
    effective_radius : float
        Effective radius in arcsec
    r_galaxy : array, optional
        Pre-calculated physical radii
    max_radius_scale : float
        Maximum radius in units of Re
    n_bins : int
        Number of radial bins
        
    Returns
    -------
    bin_num : array
        Bin number for each position
    bin_edges : array
        Radial bin edges in Re units
    bin_radii : array
        Mean radius for each bin in arcsec
    """
    # Default to center
    if center_x is None:
        center_x = np.mean(x)
    if center_y is None:
        center_y = np.mean(y)
    
    # Calculate elliptical radius if r_galaxy not provided
    if r_galaxy is None:
        # Convert PA to radians
        pa_rad = np.radians(pa)
        
        # Rotate coordinates
        dx = x - center_x
        dy = y - center_y
        
        x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
        y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
        
        # Calculate elliptical radius in pixels
        b_over_a = 1 - ellipticity
        r_ellipse = np.sqrt(x_rot**2 + (y_rot / b_over_a)**2)
        
        # Assume pixel size if not in r_galaxy
        # This should be converted to arcsec externally
        logger.warning("Using pixel units for radius - should convert to arcsec")
    else:
        r_ellipse = r_galaxy
    
    # Create Re-based bins
    # Use finer sampling near center, coarser at large radii
    bin_edges_re = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Adjust to requested number of bins
    if n_bins != len(bin_edges_re) - 1:
        # Create custom spacing
        if n_bins < 5:
            # Simple linear spacing
            bin_edges_re = np.linspace(0, max_radius_scale, n_bins + 1)
        else:
            # Hybrid spacing - finer in center
            inner_bins = min(n_bins // 2, 5)
            outer_bins = n_bins - inner_bins
            
            inner_edges = np.linspace(0, 1.0, inner_bins + 1)
            outer_edges = np.linspace(1.0, max_radius_scale, outer_bins + 1)[1:]
            
            bin_edges_re = np.concatenate([inner_edges, outer_edges])
    
    # Convert to physical units
    bin_edges_arcsec = bin_edges_re * effective_radius
    
    # Assign bins
    bin_num = np.digitize(r_ellipse, bin_edges_arcsec) - 1
    
    # Handle edge cases
    bin_num[bin_num < 0] = 0
    bin_num[bin_num >= len(bin_edges_re) - 1] = len(bin_edges_re) - 2
    
    # Calculate mean radius for each bin
    n_bins_actual = len(bin_edges_re) - 1
    bin_radii = np.zeros(n_bins_actual)
    
    for i in range(n_bins_actual):
        mask = bin_num == i
        if np.any(mask):
            bin_radii[i] = np.mean(r_ellipse[mask])
        else:
            bin_radii[i] = (bin_edges_arcsec[i] + bin_edges_arcsec[i + 1]) / 2
    
    logger.info(f"Created {n_bins_actual} Re-based bins up to {max_radius_scale} Re")
    logger.info(f"Bin edges (Re): {bin_edges_re}")
    logger.info(f"Bin edges (arcsec): {bin_edges_arcsec}")
    
    return bin_num, bin_edges_arcsec, bin_radii


def run_voronoi_binning(x, y, signal, noise, target_snr=30, plot=0, quiet=True,
                       cvt=True, min_snr=1):
    """
    Run Voronoi binning algorithm
    
    Parameters
    ----------
    x, y : array
        Coordinates
    signal : array
        Signal values
    noise : array
        Noise values
    target_snr : float
        Target S/N ratio
    plot : int
        Plotting flag for vorbin
    quiet : bool
        Suppress output
    cvt : bool
        Apply CVT iteration
    min_snr : float
        Minimum SNR threshold
        
    Returns
    -------
    Results from vorbin algorithm
    """
    try:
        from vorbin.voronoi_2d_binning import voronoi_2d_binning
    except ImportError:
        raise ImportError("vorbin package required for Voronoi binning")
    
    # Filter valid pixels
    valid = (signal > 0) & (noise > 0) & np.isfinite(signal) & np.isfinite(noise)
    
    if np.sum(valid) < 10:
        raise ValueError("Too few valid pixels for Voronoi binning")
    
    x_valid = x[valid]
    y_valid = y[valid]
    signal_valid = signal[valid]
    noise_valid = noise[valid]
    
    # Run vorbin
    bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = voronoi_2d_binning(
        x_valid, y_valid, signal_valid, noise_valid, target_snr,
        plot=plot, quiet=quiet, cvt=cvt
    )
    
    # Map back to full array
    full_bin_num = np.full(len(x), -1, dtype=int)
    full_bin_num[valid] = bin_num
    
    return full_bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale


def optimize_voronoi_binning(x, y, signal, noise, target_bin_count=15,
                           min_bins=10, max_bins=20, cvt=True, quiet=True):
    """
    Optimize Voronoi binning to achieve target number of bins
    
    Parameters
    ----------
    x, y : array
        Coordinates
    signal : array
        Signal values
    noise : array
        Noise values
    target_bin_count : int
        Target number of bins
    min_bins : int
        Minimum acceptable bins
    max_bins : int
        Maximum acceptable bins
    cvt : bool
        Apply CVT iteration
    quiet : bool
        Suppress output
        
    Returns
    -------
    Best binning results
    """
    # Calculate pixel SNR
    snr = signal / noise
    median_snr = np.nanmedian(snr[snr > 0])
    
    # Initial SNR targets to try
    snr_targets = [
        median_snr * 0.5,
        median_snr * 0.7,
        median_snr,
        median_snr * 1.5,
        median_snr * 2.0,
        median_snr * 3.0
    ]
    
    best_result = None
    best_n_bins = 0
    best_diff = float('inf')
    
    for target_snr in snr_targets:
        try:
            result = run_voronoi_binning(
                x, y, signal, noise, target_snr=target_snr,
                plot=0, quiet=quiet, cvt=cvt
            )
            
            # Count bins
            bin_num = result[0]
            n_bins = len(np.unique(bin_num[bin_num >= 0]))
            
            # Check if within acceptable range
            if min_bins <= n_bins <= max_bins:
                diff = abs(n_bins - target_bin_count)
                if diff < best_diff:
                    best_diff = diff
                    best_n_bins = n_bins
                    best_result = result
                    
                    # Perfect match
                    if diff == 0:
                        break
                        
        except Exception as e:
            logger.debug(f"Failed with target SNR {target_snr}: {e}")
            continue
    
    if best_result is not None:
        logger.info(f"Optimized binning achieved {best_n_bins} bins "
                   f"(target was {target_bin_count})")
        return best_result
    else:
        # Fallback to median SNR
        logger.warning(f"Could not achieve target bin count, using median SNR = {median_snr}")
        return run_voronoi_binning(
            x, y, signal, noise, target_snr=median_snr,
            plot=0, quiet=quiet, cvt=cvt
        )


def create_bin_map(bin_num, nx, ny):
    """
    Create 2D bin map from 1D bin numbers
    
    Parameters
    ----------
    bin_num : array
        1D array of bin numbers
    nx, ny : int
        Image dimensions
        
    Returns
    -------
    bin_map : array
        2D bin map
    """
    if len(bin_num) == nx * ny:
        return bin_num.reshape(ny, nx)
    else:
        # Handle incomplete data
        bin_map = np.full((ny, nx), -1, dtype=int)
        n_copy = min(len(bin_num), nx * ny)
        bin_map.flat[:n_copy] = bin_num[:n_copy]
        return bin_map


def get_bin_statistics(binned_data):
    """
    Calculate statistics for binned data
    
    Parameters
    ----------
    binned_data : BinnedData
        Binned data object
        
    Returns
    -------
    dict
        Statistics dictionary
    """
    stats = {
        'n_bins': binned_data.n_bins,
        'n_pixels_per_bin': [len(indices) for indices in binned_data.bin_indices],
        'mean_spectra_per_bin': np.mean([len(indices) for indices in binned_data.bin_indices]),
        'min_spectra_per_bin': np.min([len(indices) for indices in binned_data.bin_indices]),
        'max_spectra_per_bin': np.max([len(indices) for indices in binned_data.bin_indices]),
    }
    
    # Add SNR if available
    if hasattr(binned_data, 'sn') and binned_data.sn is not None:
        stats['mean_snr'] = np.mean(binned_data.sn)
        stats['min_snr'] = np.min(binned_data.sn)
        stats['max_snr'] = np.max(binned_data.sn)
    
    # Add radial statistics if available
    if hasattr(binned_data, 'bin_radii') and binned_data.bin_radii is not None:
        stats['max_radius'] = np.max(binned_data.bin_radii)
        stats['radial_coverage'] = binned_data.bin_radii
    
    return stats