"""
Enhanced spectral indices calculation module with error propagation
Includes Lick indices, D4000, etc. with full uncertainty estimation
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy import interpolate
from typing import Dict, Tuple, Optional, Union

# Import spectres from utils
from utils.calc import spectres

# Import error propagation utilities
from utils.error_propagation import (
    propagate_errors_spectral_index, 
    MCMCErrorEstimator,
    validate_errors_rms,
    bootstrap_error_estimation
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Warning message control
SHOW_WARNINGS = False


def safe_tight_layout(fig=None):
    """
    Apply tight_layout with error handling to avoid warnings
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to adjust. If None, uses current figure.
    """
    import warnings
    import matplotlib.pyplot as plt
    
    if fig is None:
        fig = plt.gcf()
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            # Skip tight_layout if it fails
            pass


def set_warnings(show=True):
    """
    Control whether to show warning messages

    Parameters:
    -----------
    show : bool
        If True, show warnings; if False, suppress warnings
    """
    global SHOW_WARNINGS
    SHOW_WARNINGS = show


def warn(message, category=UserWarning):
    """
    Issue warning based on global settings

    Parameters:
    -----------
    message : str
        Warning message
    category : Warning, optional
        Warning category, defaults to UserWarning
    """
    if SHOW_WARNINGS:
        warnings.warn(message, category)


def convert_to_numeric_safely(data):
    """
    Convert data to numeric type safely, replacing non-numeric values with NaN
    """
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        # Already numeric
        return data
        
    try:
        # Try direct conversion
        return np.array(data, dtype=float)
    except (ValueError, TypeError):
        # Try element-wise conversion
        if hasattr(data, 'shape'):
            shape = data.shape
            flat_data = data.flatten()
        else:
            shape = None
            flat_data = data
            
        # Convert each element
        numeric_data = []
        for item in flat_data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                numeric_data.append(np.nan)
                
        # Reshape if needed
        if shape is not None:
            try:
                return np.array(numeric_data, dtype=float).reshape(shape)
            except:
                return np.array(numeric_data, dtype=float)
        else:
            return np.array(numeric_data, dtype=float)


class LineIndexCalculator:
    def __init__(
        self,
        wave,
        flux,
        fit_wave,
        fit_flux,
        em_wave=None,
        em_flux_list=None,
        velocity_correction=0,
        gas_velocity_correction=None,
        error=None,
        velocity_error=5.0,
        continuum_mode="auto",
        show_warnings=True,
    ):
        """
        Initialize the absorption line index calculator with enhanced error support

        Parameters:
        -----------
        wave : array-like
            Wavelength array of the original spectrum
        flux : array-like
            Flux array of the original spectrum
        fit_wave : array-like
            Wavelength array of the fitted spectrum, used for continuum calculation
        fit_flux : array-like
            Flux array of the fitted spectrum, used for continuum calculation
        em_wave : array-like, optional
            Wavelength array for emission lines
        em_flux_list : array-like, optional
            Combined emission line spectrum
        velocity_correction : float, optional
            Stellar velocity correction value in km/s, default is 0
        gas_velocity_correction : float, optional
            Gas velocity correction value in km/s. If None, uses velocity_correction
        error : array-like, optional
            Error array for flux
        velocity_error : float, optional
            Error in velocity correction in km/s, default is 5
        continuum_mode : str, optional
            Continuum calculation mode
            'auto': Use fitted spectrum only when original data is insufficient
            'fit': Always use fitted spectrum
            'original': Use original spectrum when possible (warn when insufficient)
        show_warnings : bool, optional
            Whether to show warnings from this calculator instance
        """

        self.Test_Mode = True  # test 1 pixel.
        self.show_warnings = show_warnings

        self.c = 299792.458  # Speed of light in km/s
        self.velocity = velocity_correction
        self.velocity_error = velocity_error
        # Use stellar velocity for gas if gas velocity not provided
        self.gas_velocity = (
            gas_velocity_correction
            if gas_velocity_correction is not None
            else velocity_correction
        )
        self.continuum_mode = continuum_mode

        # Create copies and ensure finite values
        self.wave = self._apply_velocity_correction(wave, self.velocity)
        self.flux = np.array(flux, copy=True)

        # Replace any NaN or Inf values with zeros
        if np.any(~np.isfinite(self.flux)):
            self.flux[~np.isfinite(self.flux)] = 0
            self._warn("Non-finite values in flux array replaced with zeros")

        self.fit_wave = fit_wave
        self.fit_flux = fit_flux

        # Handle NaNs in fitted flux
        if np.any(~np.isfinite(self.fit_flux)):
            self.fit_flux = np.array(self.fit_flux, copy=True)
            self.fit_flux[~np.isfinite(self.fit_flux)] = 0
            self._warn("Non-finite values in fitted flux array replaced with zeros")

        # Store residuals if we can calculate them (BEFORE error estimation)
        self.residuals = None
        try:
            if len(self.wave) == len(self.fit_wave):
                # Check if wavelength grids are similar (allowing small differences)
                if np.allclose(self.wave, self.fit_wave, rtol=1e-6):
                    self.residuals = self.flux - self.fit_flux
                else:
                    # Interpolate fit_flux to original wavelength grid
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(self.fit_wave, self.fit_flux, 
                                         kind='linear', bounds_error=False, 
                                         fill_value='extrapolate')
                    fit_flux_interp = interp_func(self.wave)
                    self.residuals = self.flux - fit_flux_interp
            elif hasattr(self, 'fit_flux') and self.fit_flux is not None:
                # Try to interpolate even if arrays have different lengths
                from scipy.interpolate import interp1d
                interp_func = interp1d(self.fit_wave, self.fit_flux, 
                                     kind='linear', bounds_error=False, 
                                     fill_value='extrapolate')
                fit_flux_interp = interp_func(self.wave)
                self.residuals = self.flux - fit_flux_interp
        except Exception as e:
            self._warn(f"Could not calculate residuals: {e}")
            self.residuals = None

        # Handle error array (AFTER residuals calculation)
        if error is not None:
            self.error = np.array(error, copy=True)
            # Validate errors
            if np.any(self.error <= 0) or np.any(~np.isfinite(self.error)):
                self._warn("Invalid error values detected, replacing with estimated errors")
                self.error = self._estimate_errors()
        else:
            # Estimate errors if not provided
            self.error = self._estimate_errors()

        # Process emission lines - now with separate gas velocity
        if em_wave is not None and em_flux_list is not None:
            # Apply gas velocity correction to emission lines
            self.em_wave = self._apply_velocity_correction(em_wave, self.gas_velocity)
            self.em_flux_list = em_flux_list

            # Handle NaNs in emission line flux
            if np.any(~np.isfinite(self.em_flux_list)):
                self.em_flux_list = np.array(self.em_flux_list, copy=True)
                self.em_flux_list[~np.isfinite(self.em_flux_list)] = 0
                self._warn(
                    "Non-finite values in emission flux array replaced with zeros"
                )

            self._subtract_emission_lines()

    def _warn(self, message, category=UserWarning):
        """Issue warning based on instance settings"""
        if self.show_warnings and SHOW_WARNINGS:
            warnings.warn(message, category)

    def _apply_velocity_correction(self, wave, velocity):
        """
        Apply velocity correction to the wavelength

        Parameters:
        -----------
        wave : array-like
            Original wavelength array
        velocity : float
            Velocity correction in km/s

        Returns:
        --------
        array-like : Corrected wavelength array
        """
        return wave / (1 + (velocity / self.c))

    def _estimate_errors(self):
        """
        Estimate errors from the spectrum
        
        Returns:
        --------
        array-like : Estimated error array
        """
        # Method 1: Use residuals if available
        if hasattr(self, 'residuals') and self.residuals is not None:
            residual_rms = np.sqrt(np.nanmean(self.residuals**2))
            if residual_rms > 0 and np.isfinite(residual_rms):
                return np.full_like(self.flux, residual_rms)
        
        # Method 2: Use local RMS in continuum regions
        try:
            # Find continuum regions (avoiding strong lines)
            continuum_mask = np.ones_like(self.flux, dtype=bool)
            
            # Mask out common line regions
            line_regions = [
                (4850, 4880),  # Hβ
                (4950, 5020),  # [OIII]
                (5160, 5195),  # Mg b
                (5260, 5290),  # Fe
            ]
            
            for start, end in line_regions:
                continuum_mask &= ~((self.wave >= start) & (self.wave <= end))
            
            if np.sum(continuum_mask) > 10:
                continuum_flux = self.flux[continuum_mask]
                # Use median absolute deviation
                mad = np.median(np.abs(continuum_flux - np.median(continuum_flux)))
                rms = 1.4826 * mad  # Convert MAD to RMS
                if rms > 0 and np.isfinite(rms):
                    return np.full_like(self.flux, rms)
        except:
            pass
        
        # Method 3: Use 1% of flux as error estimate
        median_flux = np.nanmedian(np.abs(self.flux))
        if median_flux > 0 and np.isfinite(median_flux):
            return np.full_like(self.flux, 0.01 * median_flux)
        
        # Fallback: constant error
        return np.ones_like(self.flux)

    def _subtract_emission_lines(self):
        """
        Subtract emission lines from the original spectrum
        The input em_flux_list is already a combined result
        """
        # Resample emission line spectrum to the original wavelength grid
        try:
            em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)

            # Verify the result is valid before subtraction
            if np.any(~np.isfinite(em_flux_resampled)):
                self._warn(
                    "Non-finite values in resampled emission flux, replacing with zeros"
                )
                em_flux_resampled[~np.isfinite(em_flux_resampled)] = 0

            # Subtract emission lines from the original spectrum
            self.flux -= em_flux_resampled
            
            # Also resample emission error if available
            if hasattr(self, 'em_error') and self.em_error is not None:
                em_error_resampled = spectres(self.wave, self.em_wave, self.em_error)
                # Add emission error in quadrature
                self.error = np.sqrt(self.error**2 + em_error_resampled**2)
                
        except Exception as e:
            self._warn(
                f"Error subtracting emission lines: {str(e)}. Continuing without emission line subtraction."
            )

    def _check_data_coverage(self, wave_range):
        """
        Check if the original data completely covers the given wavelength range

        Parameters:
        -----------
        wave_range : tuple
            (min_wave, max_wave)

        Returns:
        --------
        bool : Whether the range is fully covered
        """
        try:
            # Convert wavelength to numeric safely
            safe_wave = convert_to_numeric_safely(self.wave)
            
            # Check if range is covered
            if len(safe_wave) == 0:
                return False
                
            min_wave = np.nanmin(safe_wave)
            max_wave = np.nanmax(safe_wave)
            
            return (wave_range[0] >= min_wave) and (wave_range[1] <= max_wave)
        except Exception as e:
            self._warn(f"Error checking data coverage: {str(e)}")
            return False

    def define_line_windows(self, line_name):
        """
        Define absorption line and continuum windows

        Parameters:
        -----------
        line_name : str
            Absorption line name

        Returns:
        --------
        dict : Dictionary with blue, line, and red windows
        """
        # Try to get definition from config file first
        try:
            from config_manager import get_spectral_line_definition

            windows = get_spectral_line_definition(line_name)
            if windows is not None:
                if self.show_warnings:
                    logger.debug(f"Using line definition from config for {line_name}")
                return windows
        except ImportError:
            if self.show_warnings:
                logger.debug(
                    "Config manager not available, using built-in line definitions"
                )
        except Exception as e:
            if self.show_warnings:
                logger.debug(f"Error getting line definition from config: {e}")

        # Fall back to hardcoded definitions
        windows = {
            "Hbeta": {
                "blue": (4827.875, 4847.875),
                "line": (4847.875, 4876.625),
                "red": (4876.625, 4891.625),
            },
            "Mgb": {
                "blue": (5142.625, 5161.375),
                "line": (5160.125, 5192.625),
                "red": (5191.375, 5206.375),
            },
            "Fe5015": {
                "blue": (4946.500, 4977.750),
                "line": (4977.750, 5054.000),
                "red": (5054.000, 5065.250),
            },
            "Fe5270": {
                "blue": (5233.2, 5248.2),
                "line": (5245.7, 5285.7),
                "red": (5285.7, 5318.2),
            },
            "Fe5335": {
                "blue": (5304.6, 5315.9),
                "line": (5312.1, 5352.1),
                "red": (5353.4, 5363.4),
            },
            "D4000": {
                "blue": (3750.0, 3950.0),
                "red": (4050.0, 4250.0),
            },
        }

        return windows.get(line_name)

    def calculate_pseudo_continuum(self, wave_range, flux_range, region_type):
        """
        Calculate pseudo-continuum with error propagation

        Parameters:
        -----------
        wave_range : tuple or array-like
            Wavelength range
        flux_range : array-like or None
            Corresponding flux values (not needed if using fitted spectrum)
        region_type : str
            Region type ('blue' or 'red')

        Returns:
        --------
        tuple : (continuum_value, continuum_error)
        """
        # Safety check for wave_range
        if wave_range is None:
            self._warn(f"No wavelength range provided for {region_type} continuum")
            return 0, 0
        
        # Ensure wave_range is a tuple of two values
        if not isinstance(wave_range, tuple) and not isinstance(wave_range, list):
            self._warn(f"Invalid wave_range format for {region_type} continuum")
            return 0, 0
        
        if len(wave_range) != 2:
            self._warn(f"Wave range must have exactly 2 values for {region_type} continuum")
            return 0, 0

        if self.continuum_mode == "fit":
            # Use fitted spectrum
            try:
                # Convert to numeric safely
                safe_fit_wave = convert_to_numeric_safely(self.fit_wave)
                safe_fit_flux = convert_to_numeric_safely(self.fit_flux)
                
                mask = (safe_fit_wave >= wave_range[0]) & (safe_fit_wave <= wave_range[1])
                if np.any(mask):
                    cont_value = np.nanmedian(safe_fit_flux[mask])
                    # Estimate error from scatter
                    cont_error = np.nanstd(safe_fit_flux[mask]) / np.sqrt(np.sum(mask))
                    return cont_value, cont_error
                else:
                    self._warn(
                        f"No fitted data points in {region_type} continuum range"
                    )
                    return 0, 0
            except Exception as e:
                self._warn(
                    f"Error calculating continuum from fitted spectrum: {str(e)}"
                )
                return 0, 0

        elif self.continuum_mode == "auto":
            # Check original data coverage
            safe_wave = convert_to_numeric_safely(self.wave)
            safe_flux = convert_to_numeric_safely(self.flux)
            
            if self._check_data_coverage(wave_range):
                mask = (safe_wave >= wave_range[0]) & (safe_wave <= wave_range[1])
                if np.any(mask):
                    cont_value = np.nanmedian(safe_flux[mask])
                    # Use error array if available
                    if hasattr(self, 'error') and self.error is not None:
                        cont_error = np.sqrt(np.nanmean(self.error[mask]**2) / np.sum(mask))
                    else:
                        cont_error = np.nanstd(safe_flux[mask]) / np.sqrt(np.sum(mask))
                    return cont_value, cont_error
                else:
                    self._warn(f"No data points in {region_type} continuum range")
                    return 0, 0
            else:
                # Use fitted spectrum when data is insufficient
                safe_fit_wave = convert_to_numeric_safely(self.fit_wave)
                safe_fit_flux = convert_to_numeric_safely(self.fit_flux)
                
                mask = (safe_fit_wave >= wave_range[0]) & (safe_fit_wave <= wave_range[1])
                if np.any(mask):
                    cont_value = np.nanmedian(safe_fit_flux[mask])
                    cont_error = np.nanstd(safe_fit_flux[mask]) / np.sqrt(np.sum(mask))
                    return cont_value, cont_error
                else:
                    self._warn(
                        f"No fitted data points in {region_type} continuum range"
                    )
                    return 0, 0

        else:  # 'original'
            safe_wave = convert_to_numeric_safely(self.wave)
            safe_flux = convert_to_numeric_safely(self.flux)
            
            if not self._check_data_coverage(wave_range):
                self._warn(
                    f"Original data insufficient to cover {region_type} continuum region, returning 0"
                )
                return 0, 0
            mask = (safe_wave >= wave_range[0]) & (safe_wave <= wave_range[1])
            if np.any(mask):
                cont_value = np.nanmedian(safe_flux[mask])
                if hasattr(self, 'error') and self.error is not None:
                    cont_error = np.sqrt(np.nanmean(self.error[mask]**2) / np.sum(mask))
                else:
                    cont_error = np.nanstd(safe_flux[mask]) / np.sqrt(np.sum(mask))
                return cont_value, cont_error
            else:
                self._warn(f"No data points in {region_type} continuum range")
                return 0, 0

    def calculate_index(self, line_name, return_error=False):
        """
        Calculate absorption line index with error support
        
        Parameters:
        -----------
        line_name : str
            Absorption line name ('Hbeta', 'Mgb', 'Fe5015', etc.)
        return_error : bool
            Whether to return error

        Returns:
        --------
        float : Absorption line index value
        float : Error value (if return_error=True)
        """
        # Special handling for D4000
        if line_name == "D4000":
            return self.calculate_D4000(return_error=return_error)
        
        # Get window definition
        windows = self.define_line_windows(line_name)
        if windows is None:
            self._warn(f"Unknown absorption line: {line_name}")
            return np.nan if not return_error else (np.nan, np.nan)

        # Check for required keys in window definition
        if "blue" not in windows or "red" not in windows:
            self._warn(
                f"Incomplete window definition for {line_name}. Missing blue or red bands."
            )
            return np.nan if not return_error else (np.nan, np.nan)

        # Handle both 'line' and 'band' keys for compatibility
        if "band" in windows:
            line_range = windows["band"]
        elif "line" in windows:
            line_range = windows["line"]
        else:
            self._warn(f"Window definition for {line_name} has no line/band range.")
            return np.nan if not return_error else (np.nan, np.nan)

        # Get line region data
        try:
            # Convert flux and wave to numeric safely
            safe_wave = convert_to_numeric_safely(self.wave)
            safe_flux = convert_to_numeric_safely(self.flux)
            safe_error = convert_to_numeric_safely(self.error)
            
            # Continue with safe arrays
            line_mask = (safe_wave >= line_range[0]) & (safe_wave <= line_range[1])
            line_wave = safe_wave[line_mask]
            line_flux = safe_flux[line_mask]
            line_err = safe_error[line_mask]

            # Check data points
            if len(line_flux) < 3:
                self._warn(f"Insufficient data points for {line_name} line region")
                return np.nan if not return_error else (np.nan, np.nan)

            # Calculate continuum with errors
            blue_cont, blue_err = self.calculate_pseudo_continuum(windows["blue"], None, "blue")
            red_cont, red_err = self.calculate_pseudo_continuum(windows["red"], None, "red")

            # Check for valid continuum values
            if (
                blue_cont <= 0
                or red_cont <= 0
                or not np.isfinite(blue_cont)
                or not np.isfinite(red_cont)
            ):
                self._warn(
                    f"Invalid continuum values for {line_name}: blue={blue_cont}, red={red_cont}"
                )
                return np.nan if not return_error else (np.nan, np.nan)

            wave_cont = np.array([np.mean(windows["blue"]), np.mean(windows["red"])])
            flux_cont = np.array([blue_cont, red_cont])

            # Linear interpolation for continuum
            f_interp = interpolate.interp1d(wave_cont, flux_cont)
            cont_at_line = f_interp(line_wave)

            # Check for division by zero
            if np.any(cont_at_line <= 0):
                self._warn(
                    f"Zero or negative continuum values at line wavelengths for {line_name}"
                )
                return np.nan if not return_error else (np.nan, np.nan)

            # Calculate integral
            index = np.trapz((1.0 - line_flux / cont_at_line), line_wave)

            if return_error:
                # Error propagation for index calculation
                # Error in (1 - F/C) = (F/C) * sqrt((dF/F)^2 + (dC/C)^2)
                
                # Interpolate continuum errors
                f_err_interp = interpolate.interp1d(
                    wave_cont, 
                    [blue_err, red_err],
                    fill_value='extrapolate'
                )
                cont_err_at_line = f_err_interp(line_wave)
                
                # Relative errors
                flux_rel_err = line_err / np.abs(line_flux)
                cont_rel_err = cont_err_at_line / cont_at_line
                
                # Combined relative error
                combined_rel_err = np.sqrt(flux_rel_err**2 + cont_rel_err**2)
                
                # Error in (1 - F/C)
                err_integrand = (line_flux / cont_at_line) * combined_rel_err
                
                # Error in integral (simple approximation)
                index_error = np.sqrt(np.trapz(err_integrand**2, line_wave))
                
                # Add velocity error contribution
                if self.velocity_error > 0:
                    vel_contribution = self._calculate_velocity_error_contribution(line_name)
                    index_error = np.sqrt(index_error**2 + vel_contribution**2)
                
                return index, index_error

            return index
        except Exception as e:
            self._warn(f"Error calculating {line_name} index: {str(e)}")
            return np.nan if not return_error else (np.nan, np.nan)

    def calculate_D4000(self, return_error=False):
        """
        Calculate 4000 Å break strength with error propagation

        Parameters:
        -----------
        return_error : bool
            Whether to return error

        Returns:
        --------
        float : D4000 value
        float : Error value (if return_error=True)
        """
        # Get window definition
        windows = self.define_line_windows("D4000")
        if windows is None:
            # Use default if not defined
            windows = {
                "blue": (3750.0, 3950.0),
                "red": (4050.0, 4250.0)
            }

        # Find wavelength indices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            blue_idx = np.where((self.wave >= windows["blue"][0]) & 
                               (self.wave <= windows["blue"][1]))[0]
            red_idx = np.where((self.wave >= windows["red"][0]) & 
                              (self.wave <= windows["red"][1]))[0]

            if len(blue_idx) == 0 or len(red_idx) == 0:
                return np.nan if not return_error else (np.nan, np.nan)

            # Calculate mean flux in blue and red regions
            blue_flux = np.nanmean(self.flux[blue_idx])
            red_flux = np.nanmean(self.flux[red_idx])

            # Calculate D4000 as ratio of red to blue flux
            if blue_flux <= 0 or not np.isfinite(blue_flux) or not np.isfinite(red_flux):
                return np.nan if not return_error else (np.nan, np.nan)

            d4000 = red_flux / blue_flux

            if return_error:
                # Error propagation for ratio
                blue_error = np.sqrt(np.nanmean(self.error[blue_idx]**2) / len(blue_idx))
                red_error = np.sqrt(np.nanmean(self.error[red_idx]**2) / len(red_idx))
                
                # Relative errors
                blue_rel_err = blue_error / blue_flux
                red_rel_err = red_error / red_flux
                
                # D4000 error
                d4000_error = d4000 * np.sqrt(blue_rel_err**2 + red_rel_err**2)
                
                return d4000, d4000_error

            return d4000

    def calculate_index_with_error(self, line_name, n_monte_carlo=1000, 
                                  use_mcmc=False, mcmc_params=None):
        """
        Calculate absorption line index with full error propagation
        
        Parameters
        ----------
        line_name : str
            Absorption line name
        n_monte_carlo : int
            Number of Monte Carlo iterations
        use_mcmc : bool
            Whether to use MCMC for error estimation
        mcmc_params : dict, optional
            MCMC parameters
            
        Returns
        -------
        index_value : float
            Index value in Angstroms
        index_error : float
            Index error in Angstroms
        index_error_details : dict
            Detailed error breakdown
        """
        # Get window definition
        windows = self.define_line_windows(line_name)
        if windows is None:
            self._warn(f"Unknown absorption line: {line_name}")
            return np.nan, np.nan, {}
        
        # Check if we have proper error array
        if self.error is None or np.all(self.error == 1):
            # Use estimated errors
            self.error = self._estimate_errors()
        
        if use_mcmc and mcmc_params is not None:
            # MCMC-based error estimation
            mcmc_estimator = MCMCErrorEstimator(**mcmc_params)
            
            # Define model function for index calculation
            def index_model(params):
                vel, cont_factor = params
                # Apply velocity correction
                temp_calc = LineIndexCalculator(
                    self.wave * (1 + vel / self.c),  # Undo original correction, apply new
                    self.flux,
                    self.fit_wave,
                    self.fit_flux,
                    velocity_correction=0,  # Already applied above
                    error=self.error,
                    continuum_mode=self.continuum_mode,
                    show_warnings=False
                )
                # Scale continuum by factor
                temp_calc.fit_flux = self.fit_flux * cont_factor
                return temp_calc.calculate_index(line_name)
            
            # Run MCMC
            initial_params = [self.velocity, 1.0]
            bounds = [(self.velocity - 3*self.velocity_error, self.velocity + 3*self.velocity_error),
                     (0.9, 1.1)]  # Allow 10% continuum variation
            
            # For MCMC, we need to define a proper likelihood
            def log_likelihood(params, data, model_func, error):
                model_val = model_func(params)
                if not np.isfinite(model_val):
                    return -np.inf
                # Simple Gaussian likelihood
                return -0.5 * ((data - model_val) / error)**2
            
            # Get nominal index value
            nominal_index = self.calculate_index(line_name)
            
            mcmc_results = mcmc_estimator.run_mcmc(
                initial_params, nominal_index, index_model, 
                0.1,  # Assumed index error for likelihood
                bounds
            )
            
            index_value = mcmc_results['median'][0]
            index_error = mcmc_results['std'][0]
            
            error_details = {
                'method': 'MCMC',
                'acceptance_fraction': mcmc_results['acceptance_fraction'],
                'percentiles': mcmc_results['percentiles']
            }
        else:
            # Monte Carlo error propagation
            index_value, index_error = propagate_errors_spectral_index(
                self.wave, self.flux, self.error, windows,
                self.velocity, self.velocity_error, n_monte_carlo
            )
            
            error_details = {
                'method': 'Monte Carlo',
                'n_iterations': n_monte_carlo
            }
        
        # Add error components breakdown
        error_details.update({
            'velocity_error_contribution': self._calculate_velocity_error_contribution(line_name),
            'continuum_error_contribution': self._calculate_continuum_error_contribution(line_name),
            'measurement_error_contribution': index_error,
            'total_error': index_error
        })
        
        return index_value, index_error, error_details

    def _calculate_velocity_error_contribution(self, line_name):
        """Calculate the contribution of velocity error to index uncertainty"""
        # Calculate index at +/- velocity error
        vel_error = self.velocity_error
        
        # Store original velocity
        orig_vel = self.velocity
        
        # Calculate at +vel_error
        self.velocity = orig_vel + vel_error
        self.wave = self._apply_velocity_correction(self.wave * (1 + orig_vel/self.c), self.velocity)
        index_plus = self.calculate_index(line_name)
        
        # Calculate at -vel_error
        self.velocity = orig_vel - vel_error
        self.wave = self._apply_velocity_correction(self.wave * (1 + self.velocity/self.c), orig_vel - vel_error)
        index_minus = self.calculate_index(line_name)
        
        # Restore original
        self.velocity = orig_vel
        self.wave = self._apply_velocity_correction(self.wave * (1 + self.velocity/self.c), orig_vel)
        
        # Error contribution (assuming Gaussian)
        if np.isfinite(index_plus) and np.isfinite(index_minus):
            return (index_plus - index_minus) / (2 * np.sqrt(2))
        else:
            return 0.0

    def _calculate_continuum_error_contribution(self, line_name):
        """Calculate the contribution of continuum placement error to index uncertainty"""
        windows = self.define_line_windows(line_name)
        
        # Get continuum regions
        blue_mask = (self.wave >= windows["blue"][0]) & (self.wave <= windows["blue"][1])
        red_mask = (self.wave >= windows["red"][0]) & (self.wave <= windows["red"][1])
        
        # Calculate continuum uncertainty
        blue_std = np.std(self.flux[blue_mask]) if np.any(blue_mask) else 0
        red_std = np.std(self.flux[red_mask]) if np.any(red_mask) else 0
        
        # Get line window
        line_window = windows.get("line", windows.get("band"))
        if line_window is None:
            return 0.0
            
        # Approximate error contribution
        line_width = line_window[1] - line_window[0]
        continuum_error = np.sqrt(blue_std**2 + red_std**2) * line_width / 2
        
        return continuum_error

    def calculate_all_indices(self, return_errors=False):
        """
        Calculate all defined spectral indices with optional error estimation

        Parameters:
        -----------
        return_errors : bool
            Whether to return errors for each index

        Returns:
        --------
        dict : Dictionary of index values (and errors if requested)
        """
        result = {}
        for line_name in [
            "Hbeta",
            "Mgb",
            "Fe5015",
            "Fe5270", 
            "Fe5335",
            "D4000"
        ]:
            try:
                if return_errors:
                    index, error = self.calculate_index(line_name, return_error=True)
                    if not np.isnan(index):
                        result[line_name] = {'value': index, 'error': error}
                else:
                    index = self.calculate_index(line_name)
                    if not np.isnan(index):
                        result[line_name] = index
            except Exception as e:
                logger.debug(f"Error calculating {line_name}: {str(e)}")

        return result

    def calculate_all_indices_with_errors(self, n_monte_carlo=1000, parallel=True):
        """
        Calculate all defined spectral indices with errors
        
        Parameters
        ----------
        n_monte_carlo : int
            Number of Monte Carlo iterations
        parallel : bool
            Whether to use parallel processing
            
        Returns
        -------
        dict
            Dictionary with index values, errors, and error details
        """
        results = {}
        
        line_names = ["Hbeta", "Mgb", "Fe5015", "Fe5270", "Fe5335", "D4000"]
        
        if parallel:
            try:
                from multiprocessing import Pool
                with Pool() as pool:
                    index_results = pool.starmap(
                        self.calculate_index_with_error,
                        [(line_name, n_monte_carlo) for line_name in line_names]
                    )
                
                for line_name, (value, error, details) in zip(line_names, index_results):
                    if not np.isnan(value):
                        results[line_name] = {
                            'value': value,
                            'error': error,
                            'error_details': details
                        }
            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}. Falling back to serial processing.")
                parallel = False
        
        if not parallel:
            for line_name in line_names:
                try:
                    value, error, details = self.calculate_index_with_error(line_name, n_monte_carlo)
                    if not np.isnan(value):
                        results[line_name] = {
                            'value': value,
                            'error': error,
                            'error_details': details
                        }
                except Exception as e:
                    logger.debug(f"Error calculating {line_name} with errors: {str(e)}")
        
        return results

    def plot_all_lines(self, mode=None, number=None, save_path=None, show_index=False,
                      show_errors=False):
        """
        Plot all spectral lines in a complete figure with error visualization

        Parameters:
        -----------
        mode : str, optional
            Figure mode, must be one of 'P2P', 'VNB', 'RNB', or 'MUSE'
        number : int, optional
            Figure number, must be an integer
        save_path : str, optional
            Path to save the figure. If provided, the figure will be saved there
        show_index : bool, optional
            Whether to show index parameters, default is False
        show_errors : bool, optional
            Whether to show error bars and error estimates, default is False

        Returns:
        --------
        fig, axes : Figure and Axes objects for further customization
        """
        # Validate mode and number parameters
        if mode is not None and number is not None:
            mode_title = f"{mode}{number}" if mode is not None else None
        else:
            mode_title = None

        # Get all defined spectral lines
        all_windows = {
            name: self.define_line_windows(name)
            for name in ["Hbeta", "Mgb", "Fe5015", "Fe5270", "Fe5335"]
            if self.define_line_windows(name) is not None
        }

        # Set fixed X-axis range
        min_wave = 4800
        max_wave = 5250

        # Create figure and overall title
        fig = plt.figure(figsize=(15, 12))
        if mode_title:
            fig.suptitle(mode_title, fontsize=16, y=0.95)

        # Create subplots, adjust height ratios to accommodate overall title
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Set unified color scheme
        colors = {
            "blue": "tab:blue",
            "line": "tab:green",  # Used for 'band' or 'line' window
            "band": "tab:green",  # Add this key for compatibility
            "red": "tab:red",
            "orig_cont": "tab:orange",  # Original spectrum continuum color (orange)
            "fit_cont": "tab:green",  # Fitted spectrum continuum color (green)
            "inactive_cont": "gray",  # Inactive continuum color
        }

        # First panel: Original data comparison
        wave_mask = (self.wave >= min_wave) & (self.wave <= max_wave)
        fit_mask = (self.fit_wave >= min_wave) & (self.fit_wave <= max_wave)

        # Calculate y-axis range with improved error handling
        if hasattr(self, "em_flux_list"):
            try:
                em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
                flux_range = self.flux[wave_mask] + em_flux_resampled[wave_mask]
            except Exception as e:
                flux_range = self.flux[wave_mask]
                self._warn(f"Error resampling emission flux: {str(e)}")
        else:
            flux_range = self.flux[wave_mask]
        fit_range = self.fit_flux[fit_mask]

        # Ensure we have valid flux values
        valid_flux = flux_range[np.isfinite(flux_range)]
        valid_fit = fit_range[np.isfinite(fit_range)]

        if len(valid_flux) > 0 and len(valid_fit) > 0:
            y_min = min(np.nanmin(valid_flux), np.nanmin(valid_fit)) * 0.9
            y_max = max(np.nanmax(valid_flux), np.nanmax(valid_fit)) * 1.1
        else:
            # Default y-axis range if valid data cannot be determined
            y_min = -1
            y_max = 1
            self._warn(
                "Could not determine valid flux range for plotting. Using default range."
            )

        # Ensure y_min and y_max are valid (not NaN or Inf)
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
            y_min = -1
            y_max = 1
            self._warn("Invalid y-axis limits detected. Using default range.")

        # Plot spectra with error bars if requested
        if show_errors and hasattr(self, 'error') and self.error is not None:
            error_mask = wave_mask
            ax1.fill_between(
                self.wave[error_mask],
                self.flux[error_mask] - self.error[error_mask],
                self.flux[error_mask] + self.error[error_mask],
                alpha=0.2, color="tab:blue", label="Error band"
            )
            
        # Plot spectra
        if hasattr(self, "em_flux_list"):
            try:
                em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
                if not np.all(np.isfinite(em_flux_resampled)):
                    em_flux_resampled = np.zeros_like(self.wave)
                ax1.plot(
                    self.wave,
                    self.flux + em_flux_resampled,
                    color="tab:blue",
                    label="Original Spectrum",
                    alpha=0.8,
                )
                ax1.plot(
                    self.wave,
                    em_flux_resampled,
                    color="tab:orange",
                    label="Emission Lines",
                    alpha=0.8,
                )
            except Exception as e:
                ax1.plot(
                    self.wave,
                    self.flux,
                    color="tab:blue",
                    label="Original Spectrum",
                    alpha=0.8,
                )
                self._warn(f"Error plotting emission components: {str(e)}")
        else:
            ax1.plot(
                self.wave,
                self.flux,
                color="tab:blue",
                label="Original Spectrum",
                alpha=0.8,
            )
        ax1.plot(
            self.fit_wave,
            self.fit_flux,
            color="tab:red",
            label="Template Fit",
            alpha=0.8,
        )

        # Calculate y-axis range for second panel with error handling
        processed_flux = self.flux[wave_mask]
        fit_flux_range = self.fit_flux[fit_mask]

        # Ensure valid data for plot limits
        valid_proc = processed_flux[np.isfinite(processed_flux)]
        valid_fit = fit_flux_range[np.isfinite(fit_flux_range)]

        if len(valid_proc) > 0 and len(valid_fit) > 0:
            y_min_processed = min(np.nanmin(valid_proc), np.nanmin(valid_fit)) * 0.9
            y_max_processed = max(np.nanmax(valid_proc), np.nanmax(valid_fit)) * 1.1
        else:
            y_min_processed = -1
            y_max_processed = 1
            self._warn(
                "Could not determine valid flux range for processed panel. Using default range."
            )

        # Ensure valid limits
        if (
            not np.isfinite(y_min_processed)
            or not np.isfinite(y_max_processed)
            or y_min_processed >= y_max_processed
        ):
            y_min_processed = -1
            y_max_processed = 1
            self._warn(
                "Invalid y-axis limits for processed panel. Using default range."
            )

        # Second panel: Processed spectrum with errors
        if show_errors and hasattr(self, 'error') and self.error is not None:
            ax2.fill_between(
                self.wave[wave_mask],
                self.flux[wave_mask] - self.error[wave_mask],
                self.flux[wave_mask] + self.error[wave_mask],
                alpha=0.2, color="tab:blue"
            )
            
        ax2.plot(
            self.wave,
            self.flux,
            color="tab:blue",
            label="Processed Spectrum",
            alpha=0.8,
        )
        ax2.plot(
            self.fit_wave,
            self.fit_flux,
            "--",
            color="tab:red",
            label="Template Fit",
            alpha=0.8,
        )

        # Mark all spectral line regions in both panels
        for line_name, windows in all_windows.items():
            if windows is None:
                continue

            for panel in [ax1, ax2]:
                # Mark blue, line/band, and red regions
                alpha = 0.2  # Transparency

                # Check for all required keys in windows
                if (
                    "blue" not in windows
                    or ("line" not in windows and "band" not in windows)
                    or "red" not in windows
                ):
                    self._warn(
                        f"Incomplete window definition for {line_name}. Skipping."
                    )
                    continue

                # Get line/band range - handle both 'line' and 'band' key for compatibility
                band_range = windows.get("band", windows.get("line", None))
                if band_range is None:
                    self._warn(f"Missing band/line range for {line_name}. Skipping.")
                    continue

                # Only include in legend once for each region type
                if (
                    line_name == list(all_windows.keys())[0]
                ):  # First spectral line for legend
                    panel.axvspan(
                        windows["blue"][0],
                        windows["blue"][1],
                        alpha=alpha,
                        color=colors["blue"],
                        label="Blue window",
                    )
                    panel.axvspan(
                        band_range[0],
                        band_range[1],
                        alpha=alpha,
                        color=colors["line"],
                        label="Line region",
                    )
                    panel.axvspan(
                        windows["red"][0],
                        windows["red"][1],
                        alpha=alpha,
                        color=colors["red"],
                        label="Red window",
                    )
                else:
                    panel.axvspan(
                        windows["blue"][0],
                        windows["blue"][1],
                        alpha=alpha,
                        color=colors["blue"],
                    )
                    panel.axvspan(
                        band_range[0], band_range[1], alpha=alpha, color=colors["line"]
                    )
                    panel.axvspan(
                        windows["red"][0],
                        windows["red"][1],
                        alpha=alpha,
                        color=colors["red"],
                    )

                # Add text annotation at the bottom
                if panel == ax1:
                    y_text = y_min + 0.05 * (y_max - y_min)
                else:
                    y_text = y_min_processed + 0.05 * (
                        y_max_processed - y_min_processed
                    )

                # Basic label
                panel.text(
                    np.mean(band_range),
                    y_text,
                    line_name,
                    horizontalalignment="center",
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

                # Add continuum points in second panel
                if panel == ax2:
                    # Calculate continuum points with errors
                    blue_cont_orig, blue_err_orig = None, None
                    red_cont_orig, red_err_orig = None, None
                    blue_cont_fit, blue_err_fit = None, None
                    red_cont_fit, red_err_fit = None, None

                    # Check if original spectrum can be used
                    try:
                        if self._check_data_coverage(windows["blue"]):
                            mask = (self.wave >= windows["blue"][0]) & (
                                self.wave <= windows["blue"][1]
                            )
                            if np.any(mask):
                                blue_cont_orig = np.nanmedian(self.flux[mask])
                                if show_errors:
                                    blue_err_orig = np.nanstd(self.flux[mask]) / np.sqrt(np.sum(mask))
                        if self._check_data_coverage(windows["red"]):
                            mask = (self.wave >= windows["red"][0]) & (
                                self.wave <= windows["red"][1]
                            )
                            if np.any(mask):
                                red_cont_orig = np.nanmedian(self.flux[mask])
                                if show_errors:
                                    red_err_orig = np.nanstd(self.flux[mask]) / np.sqrt(np.sum(mask))
                    except Exception as e:
                        self._warn(f"Error calculating original continuum: {str(e)}")

                    # Calculate fitted spectrum continuum points
                    try:
                        mask_blue = (self.fit_wave >= windows["blue"][0]) & (
                            self.fit_wave <= windows["blue"][1]
                        )
                        mask_red = (self.fit_wave >= windows["red"][0]) & (
                            self.fit_wave <= windows["red"][1]
                        )
                        if np.any(mask_blue):
                            blue_cont_fit = np.nanmedian(self.fit_flux[mask_blue])
                            if show_errors:
                                blue_err_fit = np.nanstd(self.fit_flux[mask_blue]) / np.sqrt(np.sum(mask_blue))
                        if np.any(mask_red):
                            red_cont_fit = np.nanmedian(self.fit_flux[mask_red])
                            if show_errors:
                                red_err_fit = np.nanstd(self.fit_flux[mask_red]) / np.sqrt(np.sum(mask_red))
                    except Exception as e:
                        self._warn(f"Error calculating fitted continuum: {str(e)}")

                    # Skip if we don't have valid continuum points
                    if (
                        blue_cont_fit is None
                        or red_cont_fit is None
                        or not np.isfinite(blue_cont_fit)
                        or not np.isfinite(red_cont_fit)
                    ):
                        continue

                    wave_cont = np.array(
                        [np.mean(windows["blue"]), np.mean(windows["red"])]
                    )

                    # Determine which continuum is active based on calculation mode
                    is_orig_active = self.continuum_mode == "original" or (
                        self.continuum_mode == "auto"
                        and blue_cont_orig is not None
                        and red_cont_orig is not None
                        and np.isfinite(blue_cont_orig)
                        and np.isfinite(red_cont_orig)
                    )

                    # Plot original spectrum continuum points and line (if available)
                    if (
                        blue_cont_orig is not None
                        and red_cont_orig is not None
                        and np.isfinite(blue_cont_orig)
                        and np.isfinite(red_cont_orig)
                    ):
                        flux_cont_orig = np.array([blue_cont_orig, red_cont_orig])
                        marker_size = 10
                        
                        # Add error bars if available
                        if show_errors and blue_err_orig is not None and red_err_orig is not None:
                            err_cont_orig = np.array([blue_err_orig, red_err_orig])
                            panel.errorbar(
                                wave_cont, flux_cont_orig, yerr=err_cont_orig,
                                fmt='*', markersize=marker_size,
                                color=colors["orig_cont"] if is_orig_active else colors["inactive_cont"],
                                alpha=0.8 if is_orig_active else 0.5,
                                capsize=3
                            )
                        else:
                            panel.plot(
                                wave_cont,
                                flux_cont_orig,
                                "*",
                                color=colors["orig_cont"] if is_orig_active else colors["inactive_cont"],
                                markersize=marker_size,
                                alpha=0.8 if is_orig_active else 0.5,
                                label=f"Original spectrum continuum {'(active)' if is_orig_active else '(inactive)'}"
                                if line_name == list(all_windows.keys())[0]
                                else "",
                            )
                        
                        panel.plot(
                            wave_cont,
                            flux_cont_orig,
                            "--",
                            color=colors["orig_cont"] if is_orig_active else colors["inactive_cont"],
                            alpha=0.8 if is_orig_active else 0.5,
                        )

                    # Plot fitted spectrum continuum points and line
                    if (
                        blue_cont_fit is not None
                        and red_cont_fit is not None
                        and np.isfinite(blue_cont_fit)
                        and np.isfinite(red_cont_fit)
                    ):
                        flux_cont_fit = np.array([blue_cont_fit, red_cont_fit])
                        
                        # Add error bars if available
                        if show_errors and blue_err_fit is not None and red_err_fit is not None:
                            err_cont_fit = np.array([blue_err_fit, red_err_fit])
                            panel.errorbar(
                                wave_cont, flux_cont_fit, yerr=err_cont_fit,
                                fmt='*', markersize=10,
                                color=colors["fit_cont"] if not is_orig_active else colors["inactive_cont"],
                                alpha=0.8 if not is_orig_active else 0.5,
                                capsize=3
                            )
                        else:
                            panel.plot(
                                wave_cont,
                                flux_cont_fit,
                                "*",
                                color=colors["fit_cont"] if not is_orig_active else colors["inactive_cont"],
                                markersize=10,
                                alpha=0.8 if not is_orig_active else 0.5,
                                label=f"Template continuum {'(active)' if not is_orig_active else '(inactive)'}"
                                if line_name == list(all_windows.keys())[0]
                                else "",
                            )
                        
                        panel.plot(
                            wave_cont,
                            flux_cont_fit,
                            "--",
                            color=colors["fit_cont"] if not is_orig_active else colors["inactive_cont"],
                            alpha=0.8 if not is_orig_active else 0.5,
                        )

                    # Show index parameters if requested
                    if show_index:
                        try:
                            # Save current continuum_mode
                            original_mode = self.continuum_mode

                            # Calculate index using original spectrum
                            self.continuum_mode = "original"
                            try:
                                if show_errors:
                                    orig_index, orig_error = self.calculate_index(line_name, return_error=True)
                                else:
                                    orig_index = self.calculate_index(line_name)
                                    orig_error = None
                                if np.isnan(orig_index):
                                    orig_index = None
                            except Exception:
                                orig_index = None
                                orig_error = None

                            # Calculate index using fitted spectrum
                            self.continuum_mode = "fit"
                            try:
                                if show_errors:
                                    fit_index, fit_error = self.calculate_index(line_name, return_error=True)
                                else:
                                    fit_index = self.calculate_index(line_name)
                                    fit_error = None
                                if np.isnan(fit_index):
                                    fit_index = None
                            except Exception:
                                fit_index = None
                                fit_error = None

                            # Restore original continuum_mode
                            self.continuum_mode = original_mode

                            # Calculate text position
                            base_y_text = y_text + 0.05 * (
                                y_max_processed - y_min_processed
                            )

                            # Build display text
                            if orig_index is not None and fit_index is not None:
                                # Show both values separately
                                y_offset = 0.1 * (y_max_processed - y_min_processed)

                                # Show original spectrum value (top)
                                orig_text = f"{orig_index:.3f}"
                                if show_errors and orig_error is not None:
                                    orig_text += f"±{orig_error:.3f}"
                                    
                                panel.text(
                                    np.mean(band_range),
                                    base_y_text + y_offset,
                                    orig_text,
                                    color=colors["orig_cont"],
                                    horizontalalignment="center",
                                    verticalalignment="bottom",
                                    fontsize="x-small",
                                    bbox=dict(
                                        facecolor="white", alpha=0.7, edgecolor="none"
                                    ),
                                )

                                # Show fitted spectrum value (bottom)
                                fit_text = f"{fit_index:.3f}"
                                if show_errors and fit_error is not None:
                                    fit_text += f"±{fit_error:.3f}"
                                    
                                panel.text(
                                    np.mean(band_range),
                                    base_y_text + y_offset / 2,
                                    fit_text,
                                    color=colors["fit_cont"],
                                    horizontalalignment="center",
                                    verticalalignment="bottom",
                                    fontsize="x-small",
                                    bbox=dict(
                                        facecolor="white", alpha=0.7, edgecolor="none"
                                    ),
                                )

                            elif fit_index is not None:
                                # Only show fitted spectrum value
                                fit_text = f"{fit_index:.3f}"
                                if show_errors and fit_error is not None:
                                    fit_text += f"±{fit_error:.3f}"
                                    
                                panel.text(
                                    np.mean(band_range),
                                    base_y_text
                                    + 0.02 * (y_max_processed - y_min_processed),
                                    fit_text,
                                    color=colors["fit_cont"],
                                    horizontalalignment="center",
                                    verticalalignment="bottom",
                                    fontsize="x-small",
                                    bbox=dict(
                                        facecolor="white", alpha=0.7, edgecolor="none"
                                    ),
                                )

                        except Exception as e:
                            self._warn(f"Error calculating index for {line_name}: {e}")

        # Set panel properties
        ax1.set_xlim(min_wave, max_wave)
        ax1.set_ylim(y_min, y_max)
        ax2.set_xlim(min_wave, max_wave)
        ax2.set_ylim(y_min_processed, y_max_processed)

        # Set common properties for both panels
        for ax in [ax1, ax2]:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(
                axis="both",
                which="both",
                labelsize="x-small",
                right=True,
                top=True,
                direction="in",
            )
            ax.set_xlabel("Rest-frame Wavelength (Å)")
            ax.set_ylabel("Flux")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        # Add separate velocity information to title
        velocity_text = f"v_star={self.velocity:.1f}"
        if show_errors and self.velocity_error > 0:
            velocity_text += f"±{self.velocity_error:.1f}"
        velocity_text += " km/s"
        
        if self.gas_velocity != self.velocity:
            velocity_text += f", v_gas={self.gas_velocity:.1f} km/s"
            
        ax1.set_title(f"Original Data Comparison ({velocity_text})")
        ax2.set_title("Processed Spectrum with Continuum Fits")

        # Apply a safer approach to tight_layout
        try:
            safe_tight_layout(fig)
        except Warning:
            # Ignore warnings about tight_layout
            pass
        except Exception as e:
            logger.debug(f"Error applying tight_layout: {e}")
            # Apply a more basic spacing adjustment
            plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Save figure if path provided
        if save_path and mode_title:
            # Ensure save_path exists
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Build complete file path
            filepath = os.path.join(save_path, f"{mode_title}.pdf")

            try:
                # Save figure
                fig.savefig(filepath, format="pdf", bbox_inches="tight")
                print(f"Figure saved as: {filepath}")
                plt.close(fig)
            except Exception as e:
                self._warn(f"Error saving figure: {str(e)}")

        return fig, [ax1, ax2]


def calculate_lick_indices(
    wave, flux, index_definitions=None, indices_list=None, show_warnings=True,
    error=None, return_errors=False
):
    """
    Calculate Lick indices for a single spectrum with error support

    Parameters:
    -----------
    wave : ndarray
        Wavelength array
    flux : ndarray
        Flux array
    index_definitions : dict, optional
        Index definition dictionary
    indices_list : list of str, optional
        List of indices to calculate
    show_warnings : bool, optional
        Whether to show warnings
    error : ndarray, optional
        Error array
    return_errors : bool, optional
        Whether to return errors

    Returns:
    --------
    dict : Dictionary of index values (and errors if requested)
    """
    # Create LineIndexCalculator
    calculator = LineIndexCalculator(
        wave=wave,
        flux=flux,
        fit_wave=wave,
        fit_flux=flux,
        error=error,
        continuum_mode="original",
        show_warnings=show_warnings,
    )

    # Calculate all indices
    return calculator.calculate_all_indices(return_errors=return_errors)


def calculate_D4000(wave, flux, error=None, return_error=False):
    """
    Calculate 4000 Å break strength with error support

    Parameters:
    -----------
    wave : ndarray
        Wavelength array
    flux : ndarray
        Flux array
    error : ndarray, optional
        Error array
    return_error : bool, optional
        Whether to return error

    Returns:
    --------
    float : D4000 value
    float : D4000 error (if return_error=True)
    """
    # Create temporary calculator
    calculator = LineIndexCalculator(
        wave=wave,
        flux=flux,
        fit_wave=wave,
        fit_flux=flux,
        error=error,
        continuum_mode="original",
        show_warnings=False,
    )
    
    return calculator.calculate_D4000(return_error=return_error)