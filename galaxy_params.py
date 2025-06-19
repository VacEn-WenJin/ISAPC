"""
Enhanced Galaxy Parameters Calculation Tools with Error Propagation
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.stats import bootstrap

# Import error propagation utilities
try:
    from utils.error_propagation import (
        bootstrap_error_estimate,
        propagate_binning_errors,
        error_weighted_mean,
        MCMCErrorEstimator
    )
    HAS_ERROR_UTILS = True
except ImportError:
    HAS_ERROR_UTILS = False
    warnings.warn("Error propagation utilities not available. Error estimation will be limited.")


class GalaxyParameters:
    """Enhanced galaxy parameters calculation class with error propagation"""

    def __init__(
        self,
        velocity_field: np.ndarray,
        dispersion_field: np.ndarray,
        x: np.ndarray = None,
        y: np.ndarray = None,
        pixelsize: float = 1.0,
        distance: float = None,
        velocity_error: np.ndarray = None,
        dispersion_error: np.ndarray = None,
        distance_error: float = None,
        flux_field: np.ndarray = None,
    ):
        """
        Initialize galaxy parameters calculation with error support

        Parameters
        ----------
        velocity_field : ndarray
            Velocity field (2D array)
        dispersion_field : ndarray
            Dispersion field (2D array)
        x : ndarray, optional
            x coordinate array, default uses pixel coordinates
        y : ndarray, optional
            y coordinate array, default uses pixel coordinates
        pixelsize : float, default=1.0
            Pixel size (arcsec)
        distance : float, optional
            Galaxy distance (Mpc)
        velocity_error : ndarray, optional
            Velocity error field (2D array)
        dispersion_error : ndarray, optional
            Dispersion error field (2D array)
        distance_error : float, optional
            Distance error (Mpc)
        flux_field : ndarray, optional
            Flux field for weighting (2D array)
        """
        self.velocity_field = velocity_field
        self.dispersion_field = dispersion_field
        self.pixelsize = pixelsize
        self.distance = distance
        
        # Store error arrays
        self.velocity_error = velocity_error
        self.dispersion_error = dispersion_error
        self.distance_error = distance_error
        self.flux_field = flux_field

        n_y, n_x = velocity_field.shape
        if x is None or y is None:
            # Use pixel coordinates
            y_grid, x_grid = np.indices((n_y, n_x))
            self.x = x_grid.ravel()
            self.y = y_grid.ravel()
        else:
            self.x = x
            self.y = y

        # Initialize result storage with errors
        self.kinematic_pa = None
        self.kinematic_pa_error = None
        self.vsys = None
        self.vsys_error = None
        self.vmax = None
        self.vmax_error = None
        self.sigma_mean = None
        self.sigma_mean_error = None
        self.v_over_sigma = None
        self.v_over_sigma_error = None
        self.lambda_r = None
        self.lambda_r_error = None

    def fit_rotation_curve(
        self,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        pa_initial: float = 0.0,
        r_max: Optional[float] = None,
        use_mcmc: bool = False,
        n_bootstrap: int = 100,
    ) -> Dict:
        """
        Fit rotation curve and kinematic parameters with error estimation

        Parameters
        ----------
        center_x : float, optional
            Center x coordinate
        center_y : float, optional
            Center y coordinate
        pa_initial : float, default=0.0
            Initial position angle guess (degrees)
        r_max : float, optional
            Maximum fitting radius
        use_mcmc : bool, default=False
            Use MCMC for error estimation
        n_bootstrap : int, default=100
            Number of bootstrap iterations for error estimation

        Returns
        -------
        dict
            Dictionary of fitting parameters with errors
        """
        n_y, n_x = self.velocity_field.shape

        # Set default center
        if center_x is None:
            center_x = n_x / 2
        if center_y is None:
            center_y = n_y / 2

        try:
            # Extract valid velocity values
            vel_data = self.velocity_field.ravel()
            valid_mask = np.isfinite(vel_data)
            x = self.x[valid_mask]
            y = self.y[valid_mask]
            vel = vel_data[valid_mask]
            
            # Get errors if available
            if self.velocity_error is not None:
                vel_err = self.velocity_error.ravel()[valid_mask]
            else:
                # Estimate error from scatter
                vel_err = np.full_like(vel, np.std(vel) / np.sqrt(2))

            # Check if enough valid data points
            if len(vel) < 10:
                warnings.warn("Not enough valid velocity points for fitting")
                return {
                    "pa": pa_initial,
                    "pa_error": np.nan,
                    "vsys": 0.0,
                    "vsys_error": np.nan,
                    "vmax": 0.0,
                    "vmax_error": np.nan,
                    "center": (center_x, center_y),
                    "center_error": (np.nan, np.nan),
                    "rotation_curve": np.array([[0, 0]]),
                    "rotation_curve_error": np.array([[0, 0]]),
                }

            # Fit using the standard method first
            best_pa, best_amplitude, vsys = self._fit_pa_and_amplitude(
                x, y, vel, center_x, center_y, pa_initial
            )
            
            # Estimate errors using bootstrap
            if n_bootstrap > 0 and HAS_ERROR_UTILS:
                pa_samples = []
                amplitude_samples = []
                vsys_samples = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap resample
                    indices = np.random.choice(len(vel), size=len(vel), replace=True)
                    x_boot = x[indices]
                    y_boot = y[indices]
                    vel_boot = vel[indices]
                    
                    # Add noise based on errors
                    vel_boot += vel_err[indices] * np.random.randn(len(indices))
                    
                    # Fit
                    try:
                        pa_boot, amp_boot, vsys_boot = self._fit_pa_and_amplitude(
                            x_boot, y_boot, vel_boot, center_x, center_y, best_pa
                        )
                        pa_samples.append(pa_boot)
                        amplitude_samples.append(amp_boot)
                        vsys_samples.append(vsys_boot)
                    except:
                        continue
                
                # Calculate errors from bootstrap samples
                if len(pa_samples) > 10:
                    self.kinematic_pa_error = np.std(pa_samples)
                    self.vmax_error = np.std(amplitude_samples) / 2
                    self.vsys_error = np.std(vsys_samples)
                else:
                    self.kinematic_pa_error = 5.0  # Default 5 degree error
                    self.vmax_error = np.mean(vel_err)
                    self.vsys_error = np.mean(vel_err) / np.sqrt(len(vel))
            else:
                # Simple error estimates
                self.kinematic_pa_error = 5.0  # Default 5 degree error
                self.vmax_error = np.mean(vel_err)
                self.vsys_error = np.mean(vel_err) / np.sqrt(len(vel))

            # Store results
            self.kinematic_pa = best_pa
            self.vsys = vsys
            self.vmax = best_amplitude / 2

            # Build rotation curve with errors
            rotation_curve, rotation_curve_err = self._build_rotation_curve_with_errors(
                x, y, vel, vel_err, best_pa, center_x, center_y, r_max
            )

        except Exception as e:
            warnings.warn(f"Error in rotation curve fitting: {str(e)}")
            # Return default values with NaN errors
            return {
                "pa": pa_initial,
                "pa_error": np.nan,
                "vsys": 0.0,
                "vsys_error": np.nan,
                "vmax": 0.0,
                "vmax_error": np.nan,
                "center": (center_x, center_y),
                "center_error": (np.nan, np.nan),
                "rotation_curve": np.array([[0, 0]]),
                "rotation_curve_error": np.array([[0, 0]]),
            }

        # Return fitting result with errors
        result = {
            "pa": self.kinematic_pa,
            "pa_error": self.kinematic_pa_error,
            "vsys": self.vsys,
            "vsys_error": self.vsys_error,
            "vmax": self.vmax,
            "vmax_error": self.vmax_error,
            "center": (center_x, center_y),
            "center_error": (1.0, 1.0),  # Assume 1 pixel error in center
            "rotation_curve": rotation_curve,
            "rotation_curve_error": rotation_curve_err,
        }

        return result

    def _fit_pa_and_amplitude(self, x, y, vel, center_x, center_y, pa_initial):
        """
        Internal method to fit position angle and velocity amplitude
        """
        # Search for position angle
        best_pa = pa_initial
        best_amplitude = 0

        # Coarse search
        for test_pa in np.linspace(0, 180, 19):  # 10 degree step search
            pa_rad = np.radians(test_pa)

            # Project onto position angle
            proj_dist = (x - center_x) * np.cos(pa_rad) + (y - center_y) * np.sin(pa_rad)

            # Calculate velocity amplitude
            vel_pos = vel[proj_dist > 0]
            vel_neg = vel[proj_dist < 0]

            if len(vel_pos) > 0 and len(vel_neg) > 0:
                amplitude = np.median(vel_pos) - np.median(vel_neg)

                if abs(amplitude) > abs(best_amplitude):
                    best_amplitude = amplitude
                    best_pa = test_pa

        # Fine search
        search_range = 20  # Search within ±10 degrees of best PA
        for test_pa in np.linspace(
            best_pa - search_range / 2, best_pa + search_range / 2, 21
        ):
            pa_rad = np.radians(test_pa % 180)

            # Project onto position angle
            proj_dist = (x - center_x) * np.cos(pa_rad) + (y - center_y) * np.sin(pa_rad)

            # Calculate velocity amplitude
            vel_pos = vel[proj_dist > 0]
            vel_neg = vel[proj_dist < 0]

            if len(vel_pos) > 0 and len(vel_neg) > 0:
                amplitude = np.median(vel_pos) - np.median(vel_neg)

                if abs(amplitude) > abs(best_amplitude):
                    best_amplitude = amplitude
                    best_pa = test_pa % 180

        # Adjust position angle to conventional range [0, 180)
        if best_amplitude < 0:
            best_pa = (best_pa + 180) % 180
            best_amplitude = -best_amplitude

        # Calculate systemic velocity
        vsys = np.median(vel)

        return best_pa, best_amplitude, vsys

    def _build_rotation_curve_with_errors(self, x, y, vel, vel_err, pa, center_x, center_y, r_max):
        """
        Build rotation curve with error estimates
        """
        best_pa_rad = np.radians(pa)

        # Calculate distance to center and projected distance
        dx = x - center_x
        dy = y - center_y
        radius = np.sqrt(dx**2 + dy**2)
        proj_dist = dx * np.cos(best_pa_rad) + dy * np.sin(best_pa_rad)

        # Apply maximum radius limit
        if r_max is not None:
            radius_mask = radius <= r_max
            radius = radius[radius_mask]
            proj_dist = proj_dist[radius_mask]
            vel = vel[radius_mask]
            vel_err = vel_err[radius_mask]

        # Build rotation curve data
        radial_bins = 10
        if r_max is None:
            r_max = np.max(radius)

        r_bins = np.linspace(0, r_max, radial_bins + 1)
        rotation_curve = []
        rotation_curve_err = []

        for i in range(radial_bins):
            r_min = r_bins[i]
            r_max_bin = r_bins[i + 1]

            # Select data in this radial bin
            bin_mask = (radius >= r_min) & (radius < r_max_bin)

            if np.sum(bin_mask) > 5:  # Need at least 5 points
                # Calculate for positive and negative projection regions
                pos_mask = bin_mask & (proj_dist > 0)
                neg_mask = bin_mask & (proj_dist < 0)

                if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                    # Use error-weighted mean if errors available
                    if HAS_ERROR_UTILS:
                        v_pos, v_pos_err = error_weighted_mean(
                            vel[pos_mask], vel_err[pos_mask]
                        )
                        v_neg, v_neg_err = error_weighted_mean(
                            vel[neg_mask], vel_err[neg_mask]
                        )
                    else:
                        v_pos = np.median(vel[pos_mask])
                        v_neg = np.median(vel[neg_mask])
                        v_pos_err = np.std(vel[pos_mask]) / np.sqrt(np.sum(pos_mask))
                        v_neg_err = np.std(vel[neg_mask]) / np.sqrt(np.sum(neg_mask))
                    
                    v_rot = (v_pos - v_neg) / 2
                    v_rot_err = np.sqrt(v_pos_err**2 + v_neg_err**2) / 2
                    r_mean = np.mean(radius[bin_mask])

                    rotation_curve.append((r_mean, v_rot))
                    rotation_curve_err.append((r_mean, v_rot_err))

        # Convert to arrays
        if rotation_curve:
            rotation_curve = np.array(rotation_curve)
            rotation_curve_err = np.array(rotation_curve_err)
        else:
            rotation_curve = np.array([(0, 0)])
            rotation_curve_err = np.array([(0, 0)])

        return rotation_curve, rotation_curve_err

    def calculate_kinematics(self, n_bootstrap: int = 100) -> Dict:
        """
        Calculate kinematic statistics with error estimation

        Parameters
        ----------
        n_bootstrap : int, default=100
            Number of bootstrap iterations for error estimation

        Returns
        -------
        dict
            Dictionary of kinematic parameters with errors
        """
        try:
            # Get valid velocity and dispersion values
            vel = self.velocity_field.ravel()
            disp = self.dispersion_field.ravel()
            
            # Get errors if available
            if self.velocity_error is not None:
                vel_err = self.velocity_error.ravel()
            else:
                vel_err = np.full_like(vel, np.nanstd(vel) / np.sqrt(2))
                
            if self.dispersion_error is not None:
                disp_err = self.dispersion_error.ravel()
            else:
                disp_err = np.full_like(disp, np.nanstd(disp) / np.sqrt(2))

            # Extract valid data
            mask = np.isfinite(vel) & np.isfinite(disp) & (disp > 0)
            vel_valid = vel[mask]
            disp_valid = disp[mask]
            vel_err_valid = vel_err[mask]
            disp_err_valid = disp_err[mask]
            x_valid = self.x[mask]
            y_valid = self.y[mask]

            if len(vel_valid) == 0:
                warnings.warn("No valid velocity/dispersion values found")
                return {
                    "sigma_mean": np.nan,
                    "sigma_mean_error": np.nan,
                    "v_over_sigma": np.nan,
                    "v_over_sigma_error": np.nan,
                    "lambda_r": np.nan,
                    "lambda_r_error": np.nan,
                }

            # Calculate mean dispersion with error
            if HAS_ERROR_UTILS:
                self.sigma_mean, self.sigma_mean_error = error_weighted_mean(
                    disp_valid, disp_err_valid
                )
            else:
                self.sigma_mean = np.mean(disp_valid)
                self.sigma_mean_error = np.std(disp_valid) / np.sqrt(len(disp_valid))

            # Calculate V/σ with error propagation
            v_rms = np.sqrt(np.mean(vel_valid**2))
            v_rms_error = np.sqrt(np.mean(vel_err_valid**2) / len(vel_valid))
            
            if self.sigma_mean > 0:
                self.v_over_sigma = v_rms / self.sigma_mean
                # Error propagation for division
                self.v_over_sigma_error = self.v_over_sigma * np.sqrt(
                    (v_rms_error / v_rms)**2 + (self.sigma_mean_error / self.sigma_mean)**2
                )
            else:
                self.v_over_sigma = np.nan
                self.v_over_sigma_error = np.nan

            # Calculate λR parameter with error estimation
            if self.flux_field is not None:
                flux = self.flux_field.ravel()[mask]
            else:
                flux = np.ones_like(vel_valid)

            # Calculate center
            n_y, n_x = self.velocity_field.shape
            center_x = n_x / 2
            center_y = n_y / 2

            # Calculate radius for each pixel
            r = np.sqrt((x_valid - center_x) ** 2 + (y_valid - center_y) ** 2)

            # λR = Σ(Fi*Ri*|Vi|) / Σ(Fi*Ri*sqrt(Vi^2 + σi^2))
            numerator = np.sum(flux * r * np.abs(vel_valid))
            denominator = np.sum(flux * r * np.sqrt(vel_valid**2 + disp_valid**2))

            self.lambda_r = numerator / denominator if denominator > 0 else np.nan

            # Bootstrap error estimation for λR
            if n_bootstrap > 0 and not np.isnan(self.lambda_r):
                lambda_r_samples = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap resample
                    indices = np.random.choice(len(vel_valid), size=len(vel_valid), replace=True)
                    
                    # Add noise based on errors
                    vel_boot = vel_valid[indices] + vel_err_valid[indices] * np.random.randn(len(indices))
                    disp_boot = disp_valid[indices] + disp_err_valid[indices] * np.random.randn(len(indices))
                    flux_boot = flux[indices]
                    r_boot = r[indices]
                    
                    # Calculate λR for this sample
                    num_boot = np.sum(flux_boot * r_boot * np.abs(vel_boot))
                    den_boot = np.sum(flux_boot * r_boot * np.sqrt(vel_boot**2 + disp_boot**2))
                    
                    if den_boot > 0:
                        lambda_r_samples.append(num_boot / den_boot)
                
                if len(lambda_r_samples) > 10:
                    self.lambda_r_error = np.std(lambda_r_samples)
                else:
                    self.lambda_r_error = 0.1 * self.lambda_r  # 10% error as fallback
            else:
                self.lambda_r_error = 0.1 * self.lambda_r if not np.isnan(self.lambda_r) else np.nan

            # Return results with errors
            result = {
                "sigma_mean": self.sigma_mean,
                "sigma_mean_error": self.sigma_mean_error,
                "v_over_sigma": self.v_over_sigma,
                "v_over_sigma_error": self.v_over_sigma_error,
                "lambda_r": self.lambda_r,
                "lambda_r_error": self.lambda_r_error,
            }

            return result
        except Exception as e:
            warnings.warn(f"Error calculating kinematics: {str(e)}")
            return {
                "sigma_mean": np.nan, 
                "sigma_mean_error": np.nan,
                "v_over_sigma": np.nan, 
                "v_over_sigma_error": np.nan,
                "lambda_r": np.nan,
                "lambda_r_error": np.nan,
            }

    def calculate_physical_scales(self) -> Dict:
        """
        Calculate physical scale parameters with error propagation

        Returns
        -------
        dict
            Dictionary of physical scale parameters with errors
        """
        if self.distance is None:
            warnings.warn("Galaxy distance not provided, physical scales unavailable")
            return {
                "scale": np.nan, 
                "scale_error": np.nan,
                "r_eff_kpc": np.nan,
                "r_eff_kpc_error": np.nan,
            }

        try:
            # Calculate linear scale (kpc/arcsec)
            scale = self.distance * 1000 * np.pi / (180 * 3600)  # kpc/arcsec
            
            # Error propagation for scale
            if self.distance_error is not None:
                scale_error = scale * (self.distance_error / self.distance)
            else:
                # Assume 10% distance error as default
                scale_error = 0.1 * scale

            # Calculate pixel physical size
            pixel_kpc = scale * self.pixelsize
            pixel_kpc_error = scale_error * self.pixelsize

            # Calculate effective radius
            n_y, n_x = self.velocity_field.shape

            # Simple estimate: assume r_eff is 1/4 of image size
            r_eff_pix = min(n_y, n_x) / 4
            r_eff_kpc = r_eff_pix * pixel_kpc
            r_eff_kpc_error = r_eff_pix * pixel_kpc_error

            return {
                "scale": pixel_kpc, 
                "scale_error": pixel_kpc_error,
                "r_eff_kpc": r_eff_kpc,
                "r_eff_kpc_error": r_eff_kpc_error,
            }
        except Exception as e:
            warnings.warn(f"Error calculating physical scales: {str(e)}")
            return {
                "scale": np.nan, 
                "scale_error": np.nan,
                "r_eff_kpc": np.nan,
                "r_eff_kpc_error": np.nan,
            }

    def monte_carlo_full_analysis(self, n_iterations: int = 1000,
                                 center_x: Optional[float] = None,
                                 center_y: Optional[float] = None) -> Dict:
        """
        Perform full Monte Carlo error analysis for all parameters
        
        Parameters
        ----------
        n_iterations : int, default=1000
            Number of Monte Carlo iterations
        center_x : float, optional
            Center x coordinate
        center_y : float, optional  
            Center y coordinate
            
        Returns
        -------
        dict
            Dictionary with all parameters and their Monte Carlo errors
        """
        # Storage for Monte Carlo samples
        mc_results = {
            'pa': [], 'vsys': [], 'vmax': [],
            'sigma_mean': [], 'v_over_sigma': [], 'lambda_r': []
        }
        
        # Get original data
        vel_orig = self.velocity_field.copy()
        disp_orig = self.dispersion_field.copy()
        
        for i in range(n_iterations):
            # Perturb velocity and dispersion fields
            if self.velocity_error is not None:
                self.velocity_field = vel_orig + self.velocity_error * np.random.randn(*vel_orig.shape)
            else:
                noise_level = np.nanstd(vel_orig) * 0.1  # 10% noise
                self.velocity_field = vel_orig + noise_level * np.random.randn(*vel_orig.shape)
                
            if self.dispersion_error is not None:
                self.dispersion_field = disp_orig + self.dispersion_error * np.random.randn(*disp_orig.shape)
            else:
                noise_level = np.nanstd(disp_orig) * 0.1  # 10% noise
                self.dispersion_field = disp_orig + noise_level * np.random.randn(*disp_orig.shape)
            
            # Fit rotation curve
            try:
                rot_result = self.fit_rotation_curve(
                    center_x=center_x, center_y=center_y, 
                    n_bootstrap=0  # Don't do bootstrap within MC
                )
                mc_results['pa'].append(rot_result['pa'])
                mc_results['vsys'].append(rot_result['vsys'])
                mc_results['vmax'].append(rot_result['vmax'])
            except:
                continue
                
            # Calculate kinematics
            try:
                kin_result = self.calculate_kinematics(n_bootstrap=0)
                mc_results['sigma_mean'].append(kin_result['sigma_mean'])
                mc_results['v_over_sigma'].append(kin_result['v_over_sigma'])
                mc_results['lambda_r'].append(kin_result['lambda_r'])
            except:
                continue
        
        # Restore original data
        self.velocity_field = vel_orig
        self.dispersion_field = disp_orig
        
        # Calculate statistics from MC samples
        results = {}
        for param, samples in mc_results.items():
            clean_samples = [s for s in samples if np.isfinite(s)]
            if len(clean_samples) > 10:
                results[param] = np.median(clean_samples)
                results[f'{param}_error'] = np.std(clean_samples)
                results[f'{param}_percentiles'] = np.percentile(clean_samples, [16, 50, 84])
            else:
                results[param] = np.nan
                results[f'{param}_error'] = np.nan
                results[f'{param}_percentiles'] = [np.nan, np.nan, np.nan]
                
        return results