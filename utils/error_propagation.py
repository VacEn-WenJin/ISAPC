"""
Error propagation utilities for ISAPC
Implements MCMC, bootstrap, and analytical error propagation methods
"""

import logging
import numpy as np
from scipy import stats, interpolate
from typing import Tuple, Dict, Optional, Callable
import emcee
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class MCMCErrorEstimator:
    """
    MCMC-based error estimation for spectral analysis
    Accounts for positional uncertainties and correlated errors
    """
    
    def __init__(self, n_walkers: int = 32, n_steps: int = 1000, 
                 n_burn: int = 200, n_threads: int = 4):
        """
        Initialize MCMC error estimator
        
        Parameters
        ----------
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of MCMC steps
        n_burn : int
            Number of burn-in steps
        n_threads : int
            Number of parallel threads
        """
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burn = n_burn
        self.n_threads = n_threads
        
    def log_likelihood(self, theta: np.ndarray, data: np.ndarray, 
                      model_func: Callable, error: np.ndarray) -> float:
        """
        Log-likelihood function for MCMC
        
        Parameters
        ----------
        theta : array
            Model parameters
        data : array
            Observed data
        model_func : callable
            Model function
        error : array
            Measurement errors
            
        Returns
        -------
        float
            Log-likelihood value
        """
        model = model_func(theta)
        chi2 = np.sum(((data - model) / error) ** 2)
        return -0.5 * chi2
    
    def log_prior(self, theta: np.ndarray, bounds: list[Tuple]) -> float:
        """
        Log-prior function for MCMC
        
        Parameters
        ----------
        theta : array
            Model parameters
        bounds : list of tuples
            Parameter bounds [(min1, max1), (min2, max2), ...]
            
        Returns
        -------
        float
            Log-prior value
        """
        for i, (param, bound) in enumerate(zip(theta, bounds)):
            if not bound[0] <= param <= bound[1]:
                return -np.inf
        return 0.0
    
    def log_probability(self, theta: np.ndarray, data: np.ndarray,
                       model_func: Callable, error: np.ndarray,
                       bounds: list[Tuple]) -> float:
        """
        Log-probability function for MCMC
        
        Parameters
        ----------
        theta : array
            Model parameters
        data : array
            Observed data
        model_func : callable
            Model function
        error : array
            Measurement errors
        bounds : list of tuples
            Parameter bounds
            
        Returns
        -------
        float
            Log-probability value
        """
        lp = self.log_prior(theta, bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, data, model_func, error)
    
    def run_mcmc(self, initial_params: np.ndarray, data: np.ndarray,
                 model_func: Callable, error: np.ndarray,
                 bounds: list[Tuple], include_position_error: bool = True,
                 position_error: Optional[float] = None) -> Dict:
        """
        Run MCMC to estimate parameter uncertainties
        
        Parameters
        ----------
        initial_params : array
            Initial parameter values
        data : array
            Observed data
        model_func : callable
            Model function
        error : array
            Measurement errors
        bounds : list of tuples
            Parameter bounds
        include_position_error : bool
            Whether to include positional uncertainty
        position_error : float, optional
            Positional uncertainty in pixels
            
        Returns
        -------
        dict
            MCMC results including chains and statistics
        """
        n_dim = len(initial_params)
        
        # Initialize walkers with small perturbations
        pos = initial_params + 1e-4 * np.random.randn(self.n_walkers, n_dim)
        
        # Add position parameters if needed
        if include_position_error and position_error is not None:
            # Add x, y position parameters
            n_dim += 2
            pos = np.column_stack([
                pos,
                position_error * np.random.randn(self.n_walkers, 2)
            ])
            bounds = bounds + [(-3*position_error, 3*position_error)] * 2
        
        # Set up the sampler
        with Pool(self.n_threads) as pool:
            sampler = emcee.EnsembleSampler(
                self.n_walkers, n_dim, self.log_probability,
                args=(data, model_func, error, bounds),
                pool=pool
            )
            
            # Run MCMC
            logger.info(f"Running MCMC with {self.n_walkers} walkers for {self.n_steps} steps")
            sampler.run_mcmc(pos, self.n_steps, progress=True)
        
        # Get chains and remove burn-in
        chains = sampler.get_chain(discard=self.n_burn, flat=True)
        log_prob = sampler.get_log_prob(discard=self.n_burn, flat=True)
        
        # Calculate statistics
        results = {
            'chains': chains,
            'log_prob': log_prob,
            'mean': np.mean(chains, axis=0),
            'std': np.std(chains, axis=0),
            'median': np.median(chains, axis=0),
            'percentiles': np.percentile(chains, [16, 50, 84], axis=0),
            'acceptance_fraction': np.mean(sampler.acceptance_fraction),
            'autocorr_time': self._estimate_autocorr(sampler)
        }
        
        return results
    
    def _estimate_autocorr(self, sampler: emcee.EnsembleSampler) -> np.ndarray:
        """Estimate autocorrelation time with error handling"""
        try:
            return sampler.get_autocorr_time(quiet=True)
        except:
            return np.full(sampler.ndim, np.nan)


def propagate_errors_spectral_index(wave: np.ndarray, flux: np.ndarray,
                                   flux_err: np.ndarray, index_def: Dict,
                                   velocity: float = 0.0, 
                                   velocity_err: float = 0.0,
                                   n_monte_carlo: int = 1000) -> Tuple[float, float]:
    """
    Calculate spectral index with full error propagation including velocity uncertainty
    
    Parameters
    ----------
    wave : array
        Wavelength array
    flux : array
        Flux array
    flux_err : array
        Flux error array
    index_def : dict
        Index definition with 'blue', 'red', 'band' windows
    velocity : float
        Velocity correction in km/s
    velocity_err : float
        Velocity error in km/s
    n_monte_carlo : int
        Number of Monte Carlo iterations
        
    Returns
    -------
    index_value : float
        Spectral index value in Angstroms
    index_error : float
        Spectral index error in Angstroms
    """
    c_kms = 299792.458  # Speed of light in km/s
    
    # Apply velocity correction
    wave_rest = wave / (1 + velocity / c_kms)
    
    # Get index windows
    blue_band = index_def['blue']
    red_band = index_def['red']
    feature_band = index_def.get('band', index_def.get('line'))
    
    # Monte Carlo error propagation
    index_samples = np.zeros(n_monte_carlo)
    
    for i in range(n_monte_carlo):
        # Perturb flux with errors
        flux_mc = flux + flux_err * np.random.randn(len(flux))
        
        # Perturb velocity
        if velocity_err > 0:
            vel_mc = velocity + velocity_err * np.random.randn()
            wave_mc = wave / (1 + vel_mc / c_kms)
        else:
            wave_mc = wave_rest
        
        # Calculate pseudo-continuum
        blue_mask = (wave_mc >= blue_band[0]) & (wave_mc <= blue_band[1])
        red_mask = (wave_mc >= red_band[0]) & (wave_mc <= red_band[1])
        
        if np.any(blue_mask) and np.any(red_mask):
            blue_cont = np.median(flux_mc[blue_mask])
            red_cont = np.median(flux_mc[red_mask])
            
            # Linear interpolation for continuum
            cont_wave = np.array([np.mean(blue_band), np.mean(red_band)])
            cont_flux = np.array([blue_cont, red_cont])
            
            # Calculate index
            feature_mask = (wave_mc >= feature_band[0]) & (wave_mc <= feature_band[1])
            if np.any(feature_mask):
                feature_wave = wave_mc[feature_mask]
                feature_flux = flux_mc[feature_mask]
                
                # Interpolate continuum at feature wavelengths
                cont_at_feature = np.interp(feature_wave, cont_wave, cont_flux)
                
                # Calculate equivalent width
                if np.all(cont_at_feature > 0):
                    index_samples[i] = np.trapz(
                        1.0 - feature_flux / cont_at_feature, 
                        feature_wave
                    )
                else:
                    index_samples[i] = np.nan
            else:
                index_samples[i] = np.nan
        else:
            index_samples[i] = np.nan
    
    # Remove NaN values
    valid_samples = index_samples[~np.isnan(index_samples)]
    
    if len(valid_samples) > 10:
        # Robust statistics using median and MAD
        index_value = np.median(valid_samples)
        index_error = 1.4826 * np.median(np.abs(valid_samples - index_value))
        
        # Add systematic error component (0.5% of index value)
        systematic_error = 0.005 * np.abs(index_value)
        index_error = np.sqrt(index_error**2 + systematic_error**2)
    else:
        index_value = np.nan
        index_error = np.nan
    
    return index_value, index_error


def bootstrap_error_estimation(data_func: Callable, data: np.ndarray,
                              n_bootstrap: int = 1000, 
                              confidence: float = 0.68) -> Dict:
    """
    Bootstrap error estimation for any data analysis function
    
    Parameters
    ----------
    data_func : callable
        Function to apply to data
    data : array
        Input data
    n_bootstrap : int
        Number of bootstrap iterations
    confidence : float
        Confidence level for error estimation
        
    Returns
    -------
    dict
        Bootstrap results
    """
    n_data = len(data)
    results = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n_data, size=n_data)
        resampled_data = data[indices]
        
        # Apply function
        try:
            result = data_func(resampled_data)
            results.append(result)
        except:
            continue
    
    results = np.array(results)
    
    # Calculate statistics
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    return {
        'mean': np.mean(results, axis=0),
        'std': np.std(results, axis=0),
        'median': np.median(results, axis=0),
        'percentiles': np.percentile(results, [lower_percentile, 50, upper_percentile], axis=0),
        'n_valid': len(results)
    }


def calculate_covariance_matrix(spectra: np.ndarray, 
                               distances: np.ndarray,
                               correlation_length: float = 1.0) -> np.ndarray:
    """
    Calculate covariance matrix for spatially correlated spectra
    
    Parameters
    ----------
    spectra : array
        Spectra array (n_spec, n_wave)
    distances : array
        Distance matrix between spectra (n_spec, n_spec)
    correlation_length : float
        Spatial correlation length in pixels
        
    Returns
    -------
    array
        Covariance matrix
    """
    n_spec = spectra.shape[0]
    
    # Calculate variance for each spectrum
    variances = np.var(spectra, axis=1)
    
    # Calculate correlation matrix using exponential decay
    correlation = np.exp(-distances / correlation_length)
    
    # Construct covariance matrix
    cov_matrix = np.outer(np.sqrt(variances), np.sqrt(variances)) * correlation
    
    # Add small diagonal term for numerical stability
    cov_matrix += np.eye(n_spec) * 1e-10 * np.mean(variances)
    
    return cov_matrix


def propagate_binning_errors(spectra: np.ndarray, errors: np.ndarray,
                           bin_indices: list[np.ndarray],
                           covariance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate errors through optimal binning with covariance
    
    Parameters
    ----------
    spectra : array
        Spectra array (n_wave, n_spec)
    errors : array
        Error array (n_wave, n_spec)
    bin_indices : list
        List of arrays containing indices for each bin
    covariance : array, optional
        Spatial covariance matrix
        
    Returns
    -------
    binned_spectra : array
        Optimally combined spectra
    binned_errors : array
        Propagated errors
    """
    n_wave = spectra.shape[0]
    n_bins = len(bin_indices)
    
    binned_spectra = np.zeros((n_wave, n_bins))
    binned_errors = np.zeros((n_wave, n_bins))
    
    for i, indices in enumerate(bin_indices):
        if len(indices) == 0:
            continue
            
        # Extract spectra and errors for this bin
        bin_spectra = spectra[:, indices]
        bin_errors = errors[:, indices]
        
        if covariance is not None and len(indices) > 1:
            # Use covariance matrix for optimal weighting
            cov_subset = covariance[np.ix_(indices, indices)]
            
            # Inverse covariance weighting
            try:
                cov_inv = np.linalg.inv(cov_subset)
                weights = cov_inv.sum(axis=1) / cov_inv.sum()
                
                # Combine spectra
                binned_spectra[:, i] = np.average(bin_spectra, weights=weights, axis=1)
                
                # Propagate errors with covariance
                for j in range(n_wave):
                    var_j = np.dot(weights, np.dot(cov_subset * bin_errors[j, :]**2, weights))
                    binned_errors[j, i] = np.sqrt(var_j)
            except np.linalg.LinAlgError:
                # Fall back to inverse variance weighting
                weights = 1.0 / bin_errors**2
                weights /= np.sum(weights, axis=1, keepdims=True)
                
                binned_spectra[:, i] = np.sum(bin_spectra * weights, axis=1)
                binned_errors[:, i] = 1.0 / np.sqrt(np.sum(1.0 / bin_errors**2, axis=1))
        else:
            # Simple inverse variance weighting
            weights = 1.0 / bin_errors**2
            weights /= np.sum(weights, axis=1, keepdims=True)
            
            binned_spectra[:, i] = np.sum(bin_spectra * weights, axis=1)
            binned_errors[:, i] = 1.0 / np.sqrt(np.sum(1.0 / bin_errors**2, axis=1))
    
    return binned_spectra, binned_errors


def validate_errors_rms(observed: np.ndarray, model: np.ndarray,
                       errors: np.ndarray, scale_errors: bool = True) -> Tuple[float, np.ndarray]:
    """
    Validate and optionally scale errors based on RMS of residuals
    
    Parameters
    ----------
    observed : array
        Observed data
    model : array
        Model data
    errors : array
        Error estimates
    scale_errors : bool
        Whether to scale errors by RMS factor
        
    Returns
    -------
    rms_factor : float
        RMS scaling factor
    scaled_errors : array
        Scaled error array
    """
    # Calculate normalized residuals
    residuals = (observed - model) / errors
    
    # Remove outliers using sigma clipping
    from astropy.stats import sigma_clip
    clipped_residuals = sigma_clip(residuals, sigma=3, maxiters=5)
    
    # Calculate RMS
    rms_factor = np.sqrt(np.mean(clipped_residuals**2))
    
    # Scale errors if requested and RMS significantly different from 1
    if scale_errors and (rms_factor < 0.8 or rms_factor > 1.2):
        scaled_errors = errors * rms_factor
        logger.info(f"Scaling errors by RMS factor: {rms_factor:.3f}")
    else:
        scaled_errors = errors
    
    return rms_factor, scaled_errors


def error_weighted_mean(values: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Calculate error-weighted mean and its uncertainty
    
    Parameters
    ----------
    values : array
        Data values
    errors : array
        Error estimates for each value
        
    Returns
    -------
    mean : float
        Error-weighted mean
    mean_error : float
        Error in the weighted mean
    """
    # Remove NaN and infinite values
    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    
    if not np.any(valid):
        return np.nan, np.nan
    
    values_clean = values[valid]
    errors_clean = errors[valid]
    
    # Calculate weights (inverse variance)
    weights = 1.0 / (errors_clean**2)
    
    # Weighted mean
    weighted_mean = np.sum(weights * values_clean) / np.sum(weights)
    
    # Error in weighted mean
    mean_error = 1.0 / np.sqrt(np.sum(weights))
    
    return weighted_mean, mean_error