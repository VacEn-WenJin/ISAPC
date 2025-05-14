"""
Physical Radius Calculation Module for ISAPC
Calculates physically-motivated elliptical radii based on flux distribution
"""

import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

logger = logging.getLogger(__name__)

def calculate_galaxy_radius(flux_2d, pixel_size_x=0.2, pixel_size_y=None):
    """
    Calculate elliptical galaxy radius (R_galaxy) based on flux distribution
    with artifact masking and uniform weighting
    
    Parameters
    ----------
    flux_2d : numpy.ndarray
        2D array of flux values
    pixel_size_x : float, default=0.2
        Pixel size in x-direction (arcsec)
    pixel_size_y : float, optional
        Pixel size in y-direction (arcsec), defaults to pixel_size_x
    
    Returns
    -------
    tuple
        (R_galaxy, ellipse_params)
        - R_galaxy: 2D array of elliptical radius values
        - ellipse_params: dict with ellipse parameters (center_x, center_y, PA_degrees, ellipticity)
    """
    if pixel_size_y is None:
        pixel_size_y = pixel_size_x
        
    try:
        # Step 1: Identify and mask artifacts
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
        
        # Create mask that excludes artifacts
        clean_mask = valid_mask & (flux_2d < artifact_threshold_high) & (flux_2d > artifact_threshold_low)
        
        # Check if we still have enough valid pixels
        if np.sum(clean_mask) < 0.05 * flux_2d.size:
            logger.warning("Too few pixels after artifact removal, using percentile-based approach")
            # Use percentile-based approach instead
            low_percentile = np.percentile(flux_values, 10)
            high_percentile = np.percentile(flux_values, 90)
            clean_mask = valid_mask & (flux_2d >= low_percentile) & (flux_2d <= high_percentile)
        
        # Step 2: Find the brightest central region (to avoid edge artifacts)
        # We'll use a smoothed version of the flux map to identify the central region
        from scipy.ndimage import gaussian_filter
        
        # Apply Gaussian smoothing
        smoothed_flux = gaussian_filter(np.nan_to_num(flux_2d, nan=0), sigma=3.0)
        
        # Create a mask for the central region (brightest 50% after smoothing)
        smoothed_threshold = np.percentile(smoothed_flux[smoothed_flux > 0], 50)
        central_mask = smoothed_flux > smoothed_threshold
        
        # Combine masks - we want pixels that are both clean and in the central region
        final_mask = clean_mask & central_mask
        
        # If the final mask is too restrictive, fall back to just the clean mask
        if np.sum(final_mask) < 0.02 * flux_2d.size:
            logger.warning("Central region mask too restrictive, using clean mask only")
            final_mask = clean_mask
        
        # Step 3: Calculate center using centroid of selected pixels
        y_indices, x_indices = np.indices(flux_2d.shape)
        
        if np.sum(final_mask) > 0:
            # Use uniform weighting (all valid pixels have equal weight)
            x_center = np.mean(x_indices[final_mask])
            y_center = np.mean(y_indices[final_mask])
        else:
            # If no valid pixels, use geometric center
            logger.warning("No valid pixels for center calculation, using geometric center")
            x_center = nx / 2
            y_center = ny / 2
        
        # Step 4: Calculate second moments using uniform weighting
        dx = x_indices - x_center
        dy = y_indices - y_center
        
        if np.sum(final_mask) > 0:
            # Calculate unweighted moments
            I_xx = np.mean(dx[final_mask]**2)
            I_yy = np.mean(dy[final_mask]**2)
            I_xy = np.mean(dx[final_mask] * dy[final_mask])
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
        dx = x_indices - x_center
        dy = y_indices - y_center
        
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
            'center_x': x_center,
            'center_y': y_center,
            'PA_degrees': PA_degrees,
            'ellipticity': ellipticity,
            'a': a,  # Semi-major axis scale
            'b': b,  # Semi-minor axis scale
            'clean_mask': clean_mask,  # Store the mask for potential visualization
            'final_mask': final_mask   # Store the final mask used
        }
        
        return R_galaxy_arcsec, ellipse_params
        
    except Exception as e:
        logger.error(f"Error calculating galaxy radius: {str(e)}")
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


def calculate_effective_radius(flux_2d, R_galaxy, ellipse_params, pixel_size_x=0.2, pixel_size_y=None):
    """
    Calculate the effective radius (Re) containing 50% of the galaxy light
    
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
        
    Returns
    -------
    float
        Effective radius in arcseconds
    """
    if pixel_size_y is None:
        pixel_size_y = pixel_size_x
        
    try:
        # Create a mask for valid flux values
        valid_mask = np.isfinite(flux_2d) & (flux_2d > 0)
        
        if np.sum(valid_mask) == 0:
            logger.warning("No valid flux values found, using default effective radius")
            return 5.0  # Default value in arcseconds
        
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
        
        return Re
        
    except Exception as e:
        logger.error(f"Error calculating effective radius: {str(e)}")
        # Return a reasonable default value
        return 5.0  # Default value in arcseconds


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