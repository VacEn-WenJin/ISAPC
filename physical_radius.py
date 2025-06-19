"""
Enhanced Physical Radius Calculation Module for ISAPC with Error Propagation
Calculates physically-motivated elliptical radii based on flux distribution
with full uncertainty quantification
"""

import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Tuple, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Import error propagation utilities if available
try:
    from utils.error_propagation import (
        bootstrap_error_estimate,
        monte_carlo_error_propagation,
        calculate_covariance_matrix
    )
    HAS_ERROR_UTILS = True
except ImportError:
    HAS_ERROR_UTILS = False
    logger.warning("Error propagation utilities not available. Error estimation will be limited.")


def detect_sources(flux_map, min_size=10, threshold=0.2):
    """
    Detect potential sources in a flux map using simple thresholding.
    
    Parameters
    ----------
    flux_map : ndarray
        2D flux map of the field
    min_size : int, optional
        Minimum size in pixels for a source
    threshold : float, optional
        Fraction of maximum flux to use as detection threshold
        
    Returns
    -------
    list
        List of (x,y) coordinates of detected source centers
    """
    import numpy as np
    from scipy import ndimage
    
    # Handle NaN values
    flux_copy = np.copy(flux_map)
    if np.any(np.isnan(flux_copy)):
        flux_copy = np.nan_to_num(flux_copy, nan=0.0)
    
    # Calculate threshold based on maximum flux
    max_flux = np.nanmax(flux_copy)
    threshold_value = max_flux * threshold
    
    # Create binary mask of pixels above threshold
    binary_mask = flux_copy > threshold_value
    
    # Label connected regions
    labeled_array, num_features = ndimage.label(binary_mask)
    
    if num_features == 0:
        return []
    
    # Get properties of each region
    centers = []
    for i in range(1, num_features + 1):
        # Create mask for this region
        region_mask = labeled_array == i
        region_size = np.sum(region_mask)
        
        # Skip if region is too small
        if region_size < min_size:
            continue
        
        # Find center of mass
        y_indices, x_indices = np.where(region_mask)
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        
        # Add to centers list
        centers.append((center_x, center_y))
    
    # Sort by region brightness (brightest first)
    if centers:
        brightnesses = []
        for center_x, center_y in centers:
            # Get mean brightness in a small region around center
            region_size = 3  # 3x3 box
            x_min = max(0, center_x - region_size)
            x_max = min(flux_copy.shape[1] - 1, center_x + region_size)
            y_min = max(0, center_y - region_size)
            y_max = min(flux_copy.shape[0] - 1, center_y + region_size)
            
            region = flux_copy[y_min:y_max+1, x_min:x_max+1]
            brightness = np.nanmean(region)
            brightnesses.append(brightness)
        
        # Sort centers by brightness (descending)
        centers = [center for _, center in sorted(zip(brightnesses, centers), reverse=True)]
    
    return centers


def calculate_galaxy_radius(flux_2d, pixel_size_x=0.2, pixel_size_y=None, focus_central=True,
                          flux_error=None, n_monte_carlo=0):
    """
    Calculate elliptical galaxy radius (R_galaxy) based on flux distribution
    with optional error propagation
    
    Parameters
    ----------
    flux_2d : numpy.ndarray
        2D array of flux values
    pixel_size_x : float, default=0.2
        Pixel size in x-direction (arcsec)
    pixel_size_y : float, optional
        Pixel size in y-direction (arcsec), defaults to pixel_size_x
    focus_central : bool, default=True
        Whether to focus on the central galaxy
    flux_error : numpy.ndarray, optional
        2D array of flux errors (for error propagation)
    n_monte_carlo : int, default=0
        Number of Monte Carlo iterations for error estimation
    
    Returns
    -------
    tuple
        If flux_error is None or n_monte_carlo == 0:
            (R_galaxy, ellipse_params)
        Otherwise:
            (R_galaxy, ellipse_params) with error information in ellipse_params
    """
    if pixel_size_y is None:
        pixel_size_y = pixel_size_x
        
    try:
        # Step 1: Identify and mask artifacts and prepare flux data
        # Get dimensions
        ny, nx = flux_2d.shape
        
        # Create initial mask for valid pixels (non-zero, finite flux)
        valid_mask = np.isfinite(flux_2d) & (flux_2d > 0)
        
        if np.sum(valid_mask) < 0.1 * flux_2d.size:
            logger.warning("Very few valid flux values, using all positive values")
            valid_mask = flux_2d > 0
            
        if np.sum(valid_mask) == 0:
            logger.warning("No valid flux values found, creating synthetic data")
            # Create a fallback result with simple circular geometry
            y_indices, x_indices = np.indices(flux_2d.shape)
            center_y, center_x = flux_2d.shape[0]/2, flux_2d.shape[1]/2
            R_galaxy = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2) * pixel_size_x
            ellipse_params = {
                'center_x': center_x,
                'center_y': center_y,
                'PA_degrees': 0,
                'ellipticity': 0,
                'a': 1,
                'b': 1
            }
            return R_galaxy, ellipse_params
        
        # Get statistics of valid flux
        flux_values = flux_2d[valid_mask]
        median_flux = np.median(flux_values)
        mad = np.median(np.abs(flux_values - median_flux))  # Median Absolute Deviation - robust to outliers
        
        # Identify extreme outliers (artifacts) - values more than N MADs from median
        artifact_threshold_high = median_flux + 5 * mad
        artifact_threshold_low = max(0, median_flux - 5 * mad)  # Don't go below zero
        
        # Create clean mask that excludes artifacts
        clean_mask = valid_mask & (flux_2d < artifact_threshold_high) & (flux_2d > artifact_threshold_low)
        
        # Check if we still have enough valid pixels
        if np.sum(clean_mask) < 0.05 * flux_2d.size:
            logger.warning("Too few pixels after artifact removal, using percentile-based approach")
            # Use percentile-based approach instead
            low_percentile = np.percentile(flux_values, 10)
            high_percentile = np.percentile(flux_values, 90)
            clean_mask = valid_mask & (flux_2d >= low_percentile) & (flux_2d <= high_percentile)
            
        # Step 2: Detect multiple sources (if any)
        # Apply Gaussian filter to smooth the data
        from scipy.ndimage import gaussian_filter
        smoothed_flux = gaussian_filter(np.nan_to_num(flux_2d, nan=0), sigma=3.0)
        
        # Try to detect multiple local maxima
        from scipy import ndimage
        
        # Create mask for maxima detection
        maxima_mask = smoothed_flux > np.mean(smoothed_flux[smoothed_flux > 0])
        
        # Use distance transform to find local maxima
        local_max = ndimage.maximum_filter(smoothed_flux, size=10) == smoothed_flux
        local_max = local_max & maxima_mask  # Only consider high-signal regions
        
        # Get coordinates of local maxima
        coords = np.array(np.where(local_max)).T
        
        # Skip if no local maxima found
        if len(coords) == 0:
            logger.warning("No local maxima found, using brightest pixel")
            # Find brightest pixel
            brightest_idx = np.unravel_index(np.argmax(smoothed_flux), smoothed_flux.shape)
            coords = np.array([brightest_idx])
        
        # If multiple sources detected, choose the appropriate one
        if len(coords) > 1:
            logger.info(f"Detected {len(coords)} potential sources")
            
            if focus_central:
                # Calculate distance from center of field
                field_center_y, field_center_x = ny // 2, nx // 2
                dist_from_center = np.sqrt((coords[:, 0] - field_center_y)**2 + 
                                          (coords[:, 1] - field_center_x)**2)
                
                # Choose the closest source to the center
                central_idx = np.argmin(dist_from_center)
                center_y, center_x = coords[central_idx]
                
                logger.info(f"Focusing on central source at ({center_x}, {center_y})")
                
                # Modify mask to focus on this source
                # Create a distance map from this source
                y_indices, x_indices = np.indices(flux_2d.shape)
                dist_map = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                
                # Keep only pixels in clean_mask that are closer to chosen source than to other sources
                for i, (y, x) in enumerate(coords):
                    if i != central_idx:
                        dist_to_other = np.sqrt((y_indices - y)**2 + (x_indices - x)**2)
                        closer_to_other = dist_to_other < dist_map
                        clean_mask[closer_to_other] = False
            else:
                # Use the brightest source
                source_brightness = np.array([smoothed_flux[y, x] for y, x in coords])
                brightest_idx = np.argmax(source_brightness)
                center_y, center_x = coords[brightest_idx]
                
                logger.info(f"Using brightest source at ({center_x}, {center_y})")
                
                # Modify mask to focus on this source
                y_indices, x_indices = np.indices(flux_2d.shape)
                dist_map = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                
                # Keep only pixels in clean_mask that are closer to chosen source than to other sources
                for i, (y, x) in enumerate(coords):
                    if i != brightest_idx:
                        dist_to_other = np.sqrt((y_indices - y)**2 + (x_indices - x)**2)
                        closer_to_other = dist_to_other < dist_map
                        clean_mask[closer_to_other] = False
        else:
            # Single source - use its coordinates as center
            center_y, center_x = coords[0]
            logger.info(f"Single source detected at ({center_x}, {center_y})")
        
        # Step 3: Create a central region mask to focus on main galaxy
        # Focus on central region around the chosen center
        y_indices, x_indices = np.indices(flux_2d.shape)
        dist_from_center = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        
        # Use a percentile-based approach to determine central region size
        valid_dists = dist_from_center[clean_mask]
        if len(valid_dists) > 0:
            central_dist_threshold = np.percentile(valid_dists, 75)  # Include up to 75th percentile distance
            central_mask = dist_from_center <= central_dist_threshold
        else:
            # Fallback to a fixed radius
            central_mask = dist_from_center <= min(ny, nx) / 4
            
        # Combine masks - we want pixels that are both clean and in the central region
        final_mask = clean_mask & central_mask
        
        # If the final mask is too restrictive, fall back to just the clean mask
        if np.sum(final_mask) < 0.02 * flux_2d.size:
            logger.warning("Central region mask too restrictive, using clean mask only")
            final_mask = clean_mask
            
        # Step 4: Calculate second moments for shape determination
        # Create coordinate arrays relative to the chosen center
        dx = x_indices - center_x
        dy = y_indices - center_y
        
        # Calculate flux-weighted moments
        if np.sum(final_mask) > 0:
            # Extract valid data
            valid_dx = dx[final_mask]
            valid_dy = dy[final_mask]
            valid_flux = flux_2d[final_mask]
            
            # Normalize flux for weighting (avoid division by zero)
            if np.sum(valid_flux) > 0:
                flux_weights = valid_flux / np.sum(valid_flux)
            else:
                flux_weights = np.ones_like(valid_flux) / len(valid_flux)
            
            # Calculate weighted moments
            I_xx = np.sum(valid_dx**2 * flux_weights)
            I_yy = np.sum(valid_dy**2 * flux_weights)
            I_xy = np.sum(valid_dx * valid_dy * flux_weights)
        else:
            # Default to circular shape
            I_xx = 1.0
            I_yy = 1.0
            I_xy = 0.0
            
        # Step 5: Determine ellipse parameters
        try:
            # Calculate position angle
            PA = 0.5 * np.arctan2(2 * I_xy, I_xx - I_yy)
            PA_degrees = (PA * 180 / np.pi) % 180
            
            # Calculate semi-major and semi-minor axis lengths (eigenvalues)
            term1 = (I_xx + I_yy) / 2
            term2 = np.sqrt(((I_xx - I_yy) / 2)**2 + I_xy**2)
            
            a_sq = term1 + term2  # Larger eigenvalue
            b_sq = term1 - term2  # Smaller eigenvalue
            
            # Ensure positive values with error checking
            if a_sq <= 0 or not np.isfinite(a_sq):
                logger.warning("Invalid semi-major axis calculation, using default value")
                a_sq = max(I_xx, I_yy)  # Fallback
                
            if b_sq <= 0 or not np.isfinite(b_sq):
                logger.warning("Invalid semi-minor axis calculation, using default value")
                b_sq = min(max(0.1, min(I_xx, I_yy)), a_sq * 0.9)  # Fallback with constraints
            
            # Calculate axis lengths
            a = np.sqrt(a_sq)
            b = np.sqrt(b_sq)
            
            # Calculate ellipticity with constraints
            ellipticity = 1 - (b / a)
            
            # Constrain ellipticity to reasonable range [0, 0.95]
            ellipticity = max(0, min(0.95, ellipticity))
            
            # Handle degenerate cases where ellipticity is very low
            if ellipticity < 0.05:
                # For nearly circular objects, the PA is not well-defined
                PA_degrees = 0
                ellipticity = 0
        except Exception as e:
            logger.warning(f"Error calculating ellipse parameters: {str(e)}")
            # Default to circular shape
            a = np.sqrt(max(I_xx, I_yy, 0.1))
            b = a
            PA_degrees = 0
            ellipticity = 0
                
        # Step 6: Calculate R_galaxy for each pixel
        # Prepare coordinate arrays
        dx = x_indices - center_x
        dy = y_indices - center_y
        
        # Rotate coordinates to align with principal axes
        PA_rad = np.radians(PA_degrees)
        x_prime = dx * np.cos(PA_rad) + dy * np.sin(PA_rad)
        y_prime = -dx * np.sin(PA_rad) + dy * np.cos(PA_rad)
        
        # Calculate elliptical radius with safe handling of extreme ellipticity
        if ellipticity < 1:
            scale_factor = 1.0 / (1.0 - ellipticity)
        else:
            scale_factor = 20.0  # High but finite value for extreme cases
            
        R_galaxy = np.sqrt(x_prime**2 + (y_prime * scale_factor)**2)
        
        # Scale to arcseconds
        R_galaxy_arcsec = R_galaxy * pixel_size_x
        
        # Store ellipse parameters
        ellipse_params = {
            'center_x': center_x,
            'center_y': center_y,
            'PA_degrees': PA_degrees,
            'ellipticity': ellipticity,
            'a': a,  # Semi-major axis scale
            'b': b,  # Semi-minor axis scale
            'clean_mask': clean_mask,  # Store the mask for potential visualization
            'final_mask': final_mask   # Store the final mask used
        }
        
        # Add error estimation if requested
        if flux_error is not None and n_monte_carlo > 0 and HAS_ERROR_UTILS:
            logger.info(f"Running Monte Carlo error propagation with {n_monte_carlo} iterations")
            
            # Storage for Monte Carlo samples
            mc_results = {
                'center_x': [],
                'center_y': [],
                'PA': [],
                'ellipticity': [],
                'a': [],
                'b': []
            }
            
            for i in range(n_monte_carlo):
                # Perturb flux with errors
                flux_perturbed = flux_2d + flux_error * np.random.randn(*flux_2d.shape)
                
                try:
                    # Recalculate with perturbed flux
                    _, ellipse_params_mc = calculate_galaxy_radius(
                        flux_perturbed, pixel_size_x, pixel_size_y, focus_central, 
                        flux_error=None, n_monte_carlo=0  # Don't recurse
                    )
                    
                    # Store ellipse parameters
                    mc_results['center_x'].append(ellipse_params_mc['center_x'])
                    mc_results['center_y'].append(ellipse_params_mc['center_y'])
                    mc_results['PA'].append(ellipse_params_mc['PA_degrees'])
                    mc_results['ellipticity'].append(ellipse_params_mc['ellipticity'])
                    mc_results['a'].append(ellipse_params_mc['a'])
                    mc_results['b'].append(ellipse_params_mc['b'])
                    
                except Exception as e:
                    logger.debug(f"Monte Carlo iteration {i} failed: {e}")
                    continue
            
            # Calculate statistics from Monte Carlo samples
            n_valid = len(mc_results['center_x'])
            
            if n_valid > 10:
                # Calculate ellipse parameter errors
                ellipse_params['center_x_error'] = np.std(mc_results['center_x'])
                ellipse_params['center_y_error'] = np.std(mc_results['center_y'])
                ellipse_params['PA_error'] = np.std(mc_results['PA'])
                ellipse_params['ellipticity_error'] = np.std(mc_results['ellipticity'])
                ellipse_params['a_error'] = np.std(mc_results['a'])
                ellipse_params['b_error'] = np.std(mc_results['b'])
                
                logger.info(f"Monte Carlo error propagation completed with {n_valid} valid samples")
            else:
                logger.warning("Too few valid Monte Carlo samples, using default errors")
                # Fallback error estimates
                ellipse_params['center_x_error'] = 2.0
                ellipse_params['center_y_error'] = 2.0
                ellipse_params['PA_error'] = 10.0
                ellipse_params['ellipticity_error'] = 0.1
                ellipse_params['a_error'] = 0.1 * a
                ellipse_params['b_error'] = 0.1 * b
        
        return R_galaxy_arcsec, ellipse_params
        
    except Exception as e:
        logger.error(f"Error calculating galaxy radius: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Return a simple radius measure as fallback
        y_indices, x_indices = np.indices(flux_2d.shape)
        center_y, center_x = flux_2d.shape[0]/2, flux_2d.shape[1]/2
        R_galaxy = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2) * pixel_size_x
        ellipse_params = {
            'center_x': center_x,
            'center_y': center_y,
            'PA_degrees': 0,
            'ellipticity': 0,
            'a': 1,
            'b': 1
        }
        return R_galaxy, ellipse_params


def calculate_effective_radius(flux_2d, R_galaxy, ellipse_params, pixel_size_x=0.2, 
                             pixel_size_y=None, flux_error=None, n_bootstrap=0):
    """
    Calculate the effective radius (Re) containing 50% of the galaxy light
    with optional error estimation
    
    Parameters
    ----------
    flux_2d : numpy.ndarray
        2D array of flux values
    R_galaxy : numpy.ndarray
        2D array of elliptical radius values from calculate_galaxy_radius
    ellipse_params : dict
        Dictionary with ellipse parameters from calculate_galaxy_radius
    pixel_size_x : float, default=0.2
        Pixel size in x-direction (arcsec)
    pixel_size_y : float, optional
        Pixel size in y-direction (arcsec), defaults to pixel_size_x
    flux_error : numpy.ndarray, optional
        2D array of flux errors for error estimation
    n_bootstrap : int, default=0
        Number of bootstrap iterations for error estimation
        
    Returns
    -------
    float or tuple
        If flux_error is None or n_bootstrap == 0: Re
        Otherwise: (Re, Re_error)
    """
    if pixel_size_y is None:
        pixel_size_y = pixel_size_x
        
    try:
        # Create a mask for valid flux values
        valid_mask = np.isfinite(flux_2d) & (flux_2d > 0)
        
        if np.sum(valid_mask) == 0:
            logger.warning("No valid flux values found, using default effective radius")
            if flux_error is None or n_bootstrap == 0:
                return 5.0  # Default value in arcseconds
            else:
                return 5.0, 1.0  # Default with error
        
        # Get total flux and sort pixels by radius
        total_flux = np.sum(flux_2d[valid_mask])
        
        # Create sorted arrays of radius and corresponding flux
        R_valid = R_galaxy[valid_mask]
        flux_valid = flux_2d[valid_mask]
        
        # Sort by radius
        sort_idx = np.argsort(R_valid)
        R_sorted = R_valid[sort_idx]
        flux_sorted = flux_valid[sort_idx]
        
        # Calculate cumulative flux
        cumulative_flux = np.cumsum(flux_sorted)
        cumulative_flux_fraction = cumulative_flux / total_flux
        
        # Find effective radius (where cumulative flux fraction reaches 0.5)
        half_light_idx = np.searchsorted(cumulative_flux_fraction, 0.5)
        
        # If half-light index is valid, get Re
        if half_light_idx < len(R_sorted):
            Re = R_sorted[half_light_idx]
        else:
            # Fallback to maximum radius
            logger.warning("Half-light index out of range, using maximum radius")
            Re = np.max(R_valid)
        
        # Error estimation if requested
        if flux_error is not None and n_bootstrap > 0 and HAS_ERROR_UTILS:
            Re_samples = []
            
            for i in range(n_bootstrap):
                # Perturb flux values
                flux_perturbed = flux_2d + flux_error * np.random.randn(*flux_2d.shape)
                
                # Ensure positive values
                flux_perturbed = np.maximum(flux_perturbed, 0)
                
                try:
                    # Calculate Re for this realization
                    Re_boot = calculate_effective_radius(
                        flux_perturbed, R_galaxy, ellipse_params, 
                        pixel_size_x, pixel_size_y, 
                        flux_error=None, n_bootstrap=0  # Don't recurse
                    )
                    
                    if np.isfinite(Re_boot) and Re_boot > 0:
                        Re_samples.append(Re_boot)
                except:
                    continue
            
            # Calculate error from bootstrap samples
            if len(Re_samples) > 100:
                Re_error = np.std(Re_samples)
            else:
                logger.warning("Too few bootstrap samples for Re error estimation")
                Re_error = Re * 0.15  # 15% error as fallback
            
            return Re, Re_error
        else:
            return Re
        
    except Exception as e:
        logger.error(f"Error calculating effective radius: {str(e)}")
        # Return a reasonable default value
        if flux_error is None or n_bootstrap == 0:
            return 5.0  # Default value in arcseconds
        else:
            return 5.0, 1.0


def visualize_galaxy_radius(flux_2d, R_galaxy, ellipse_params, output_path=None):
    """
    Visualize the elliptical radius calculation
    
    Parameters
    ----------
    flux_2d : numpy.ndarray
        2D array of flux values
    R_galaxy : numpy.ndarray
        2D array of elliptical radius values
    ellipse_params : dict
        Dictionary with ellipse parameters
    output_path : str, optional
        Path to save the visualization
        
    Returns
    -------
    tuple
        (fig1, fig2) - Figure objects for the flux+ellipse plot and the radius map
    """
    try:
        # Create figure for flux map with ellipses
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Plot the flux map
        im = ax1.imshow(flux_2d, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax1, label='Flux')
        
        # Extract parameters
        x_center = ellipse_params['center_x']
        y_center = ellipse_params['center_y']
        PA_degrees = ellipse_params['PA_degrees']
        ellipticity = ellipse_params['ellipticity']
        
        # Mark the center
        ax1.plot(x_center, y_center, 'r+', markersize=10)
        
        # Draw ellipses at different radii
        for r in np.linspace(5, min(flux_2d.shape) / 3, 6):
            ellipse = Ellipse(
                (x_center, y_center),
                2 * r,                    # Major axis length
                2 * r * (1 - ellipticity),  # Minor axis length
                angle=PA_degrees,
                fill=False,
                edgecolor='white',
                linestyle='--'
            )
            ax1.add_patch(ellipse)
        
        ax1.set_title(f'Galaxy Flux with Elliptical Isophotes\nPA={PA_degrees:.1f}°, ε={ellipticity:.2f}')
        
        # Create figure for R_galaxy map
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        im = ax2.imshow(R_galaxy, origin='lower', cmap='plasma')
        plt.colorbar(im, ax=ax2, label='R_galaxy (arcsec)')
        
        ax2.set_title('Elliptical Radius (R_galaxy)')
        
        # Save if output path provided
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True, parents=True)
            fig1.savefig(output_dir / 'flux_ellipse.png', dpi=150, bbox_inches='tight')
            fig2.savefig(output_dir / 'galaxy_radius.png', dpi=150, bbox_inches='tight')
            
        return fig1, fig2
    
    except Exception as e:
        logger.error(f"Error visualizing galaxy radius: {str(e)}")
        return None, None