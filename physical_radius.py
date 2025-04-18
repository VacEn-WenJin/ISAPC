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
    with logarithmic weighting to reduce impact of artifacts
    
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
        # Step 1: Prepare flux data and apply log weighting
        # Replace non-positive and NaN values with a small positive number
        valid_flux = flux_2d[np.isfinite(flux_2d) & (flux_2d > 0)]
        
        if len(valid_flux) == 0:
            logger.warning("No positive flux values found in data")
            # Create a fallback result
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
        
        # Find minimum positive value and use for replacement
        min_positive = np.min(valid_flux)
        # Create a copy to avoid modifying the original
        flux_work = flux_2d.copy()
        # Replace invalid values
        flux_work[~np.isfinite(flux_work) | (flux_work <= 0)] = min_positive / 10
        
        # Apply log scaling to compress the dynamic range
        log_flux = np.log10(flux_work)
        
        # Calculate statistics on log scale
        mean_log_flux = np.nanmean(log_flux)
        std_log_flux = np.nanstd(log_flux)
        
        # Create mask to exclude artifacts and noise
        # Lower bound removes background and noise
        # Upper bound removes potential artifacts
        flux_mask = (log_flux > mean_log_flux - 0.5*std_log_flux) & (log_flux < mean_log_flux + 2*std_log_flux)
        
        # If mask is too restrictive, relax the criteria
        if np.sum(flux_mask) < 0.1 * flux_2d.size:
            logger.warning("Flux mask too restrictive, relaxing criteria")
            flux_mask = (log_flux > mean_log_flux - std_log_flux) & (log_flux < mean_log_flux + 3*std_log_flux)
            
            # If still too few pixels, use a percentile-based approach
            if np.sum(flux_mask) < 0.05 * flux_2d.size:
                logger.warning("Using percentile-based flux masking")
                sorted_flux = np.sort(log_flux[np.isfinite(log_flux)])
                if len(sorted_flux) > 0:
                    low_thresh = sorted_flux[int(0.2 * len(sorted_flux))]  # 20th percentile
                    high_thresh = sorted_flux[int(0.8 * len(sorted_flux))]  # 80th percentile
                    flux_mask = (log_flux >= low_thresh) & (log_flux <= high_thresh)
        
        # Step 2: Calculate flux-weighted center using the masked data
        y_indices, x_indices = np.indices(flux_2d.shape)
        
        # Use log-weighted flux for center calculation
        weights = log_flux.copy()
        # Zero out weights for pixels outside our mask
        weights[~flux_mask] = 0
        
        # Apply mask and calculate weighted mean for center
        total_weight = np.sum(weights[flux_mask])
        
        if total_weight > 0:
            # Calculate flux-weighted center
            x_center = np.sum(x_indices[flux_mask] * weights[flux_mask]) / total_weight
            y_center = np.sum(y_indices[flux_mask] * weights[flux_mask]) / total_weight
        else:
            # Fallback to geometric center
            x_center = flux_2d.shape[1] / 2
            y_center = flux_2d.shape[0] / 2
            logger.warning("Weighted center calculation failed. Using geometric center.")
        
        # Step 3: Calculate flux-weighted second moments for ellipse
        dx = x_indices - x_center
        dy = y_indices - y_center
        
        # Create weights for moments calculation - use squared flux to emphasize bright regions
        # but limit extreme values
        weights_moments = np.power(10, log_flux.clip(mean_log_flux - std_log_flux, 
                                                     mean_log_flux + std_log_flux))
        
        # Zero weights for masked regions
        weights_moments[~flux_mask] = 0
        
        # Calculate weighted moments
        total_weight = np.sum(weights_moments[flux_mask])
        
        if total_weight > 0:
            I_xx = np.sum(weights_moments[flux_mask] * dx[flux_mask]**2) / total_weight
            I_yy = np.sum(weights_moments[flux_mask] * dy[flux_mask]**2) / total_weight
            I_xy = np.sum(weights_moments[flux_mask] * dx[flux_mask] * dy[flux_mask]) / total_weight
        else:
            # Fallback to unweighted calculation
            valid_mask = flux_mask & np.isfinite(dx) & np.isfinite(dy)
            if np.any(valid_mask):
                I_xx = np.mean(dx[valid_mask]**2)
                I_yy = np.mean(dy[valid_mask]**2)
                I_xy = np.mean(dx[valid_mask] * dy[valid_mask])
            else:
                # Last resort: use identity matrix (circular shape)
                I_xx = 1.0
                I_yy = 1.0
                I_xy = 0.0
            
        # Step 4: Determine ellipse parameters with robust error handling
        try:
            # Calculate position angle
            PA = 0.5 * np.arctan2(2 * I_xy, I_xx - I_yy)
            PA_degrees = (PA * 180 / np.pi) % 180
            
            # Calculate semi-major and semi-minor axis lengths
            # These formulas compute eigenvalues of the inertia tensor
            term1 = (I_xx + I_yy) / 2
            term2 = np.sqrt(((I_xx - I_yy) / 2)**2 + I_xy**2)
            
            # Semi-major and semi-minor axes (eigenvalues)
            a_sq = term1 + term2  # Larger eigenvalue
            b_sq = term1 - term2  # Smaller eigenvalue
            
            # Ensure positive values with error checking
            if a_sq <= 0 or not np.isfinite(a_sq):
                logger.warning("Invalid semi-major axis calculation, using default value")
                a_sq = max(I_xx, I_yy)  # Fallback
                
            if b_sq <= 0 or not np.isfinite(b_sq):
                logger.warning("Invalid semi-minor axis calculation, using default value")
                b_sq = min(max(0.1, min(I_xx, I_yy)), a_sq * 0.9)  # Fallback with constraints
            
            # Calculate ellipticity with constraints
            if a_sq > 0 and b_sq > 0:
                # Eigenvalues give us the square of the semi-axis lengths
                a = np.sqrt(a_sq)
                b = np.sqrt(b_sq)
                
                # Ellipticity defined as 1 - (minor/major)
                ellipticity = 1 - (b / a)
                
                # Constrain ellipticity to reasonable range [0, 0.95]
                ellipticity = max(0, min(0.95, ellipticity))
            else:
                a = np.sqrt(max(I_xx, I_yy, 0.1))
                b = np.sqrt(max(min(I_xx, I_yy), 0.1))
                ellipticity = 0
                logger.warning("Using circular model (e=0) due to calculation issues")
                
            # Fix potential issues with angles
            if not np.isfinite(PA_degrees):
                PA_degrees = 0
                
            # Handle degenerate cases where ellipticity is very low
            if ellipticity < 0.05:
                # For nearly circular objects, the PA is not well-defined
                # Set to 0 to avoid numerical instability
                PA_degrees = 0
                ellipticity = 0
        except Exception as e:
            logger.warning(f"Error calculating ellipse parameters: {str(e)}")
            # Default to circular shape
            a = np.sqrt(max(I_xx, I_yy, 0.1))
            b = a
            PA_degrees = 0
            ellipticity = 0
                
        # Step 5: Calculate R_galaxy for each pixel
        # Prepare coordinate arrays
        dx = x_indices - x_center
        dy = y_indices - y_center
        
        # Rotate coordinates to align with principal axes
        PA_rad = np.radians(PA_degrees)
        x_prime = dx * np.cos(PA_rad) + dy * np.sin(PA_rad)
        y_prime = -dx * np.sin(PA_rad) + dy * np.cos(PA_rad)
        
        # Calculate elliptical radius
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
            'b': b   # Semi-minor axis scale
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