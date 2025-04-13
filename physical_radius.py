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
        # Step 1: Calculate flux statistics
        mean_flux = np.nanmean(flux_2d[flux_2d > 0])
        std_flux = np.nanstd(flux_2d[flux_2d > 0])
        
        # Threshold: 1 sigma above mean
        flux_threshold = mean_flux + std_flux
        
        # Create mask for high-flux pixels
        high_flux_mask = flux_2d > flux_threshold
        
        # If no pixels above threshold, use all pixels with positive flux
        if not np.any(high_flux_mask):
            logger.warning("No pixels above flux threshold, using all positive flux pixels")
            high_flux_mask = flux_2d > 0
            
        # Step 2: Calculate flux-weighted center
        y_indices, x_indices = np.indices(flux_2d.shape)
        
        # Apply mask and calculate weighted mean
        total_weight = np.sum(flux_2d[high_flux_mask])
        # if total_weight > 0:
        #     x_center = np.sum(x_indices[high_flux_mask] * flux_2d[high_flux_mask]) / total_weight
        #     y_center = np.sum(y_indices[high_flux_mask] * flux_2d[high_flux_mask]) / total_weight
        # else:
        #     # Fallback to geometric center if weighting fails
        #     x_center = flux_2d.shape[1] / 2
        #     y_center = flux_2d.shape[0] / 2
        #     logger.warning("Flux weighting failed. Using geometric center instead.")
        
        # Uniform center
        x_center = flux_2d.shape[1] / 2
        y_center = flux_2d.shape[0] / 2

        # Step 3: Calculate flux-weighted second moments
        dx = x_indices - x_center
        dy = y_indices - y_center
        
        # Create weights from flux
        weights = flux_2d.copy()
        weights[~high_flux_mask] = 0  # Zero weight for low-flux pixels
        
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            I_xx = np.sum(weights * dx**2) / total_weight
            I_yy = np.sum(weights * dy**2) / total_weight
            I_xy = np.sum(weights * dx * dy) / total_weight
        else:
            # Fallback to unity weights if weighting fails
            valid_mask = high_flux_mask & np.isfinite(dx) & np.isfinite(dy)
            I_xx = np.mean(dx[valid_mask]**2)
            I_yy = np.mean(dy[valid_mask]**2)
            I_xy = np.mean(dx[valid_mask] * dy[valid_mask])
            
        # Step 4: Determine ellipse parameters
        # Calculate position angle
        PA = 0.5 * np.arctan2(2 * I_xy, I_xx - I_yy)
        PA_degrees = (PA * 180 / np.pi) % 180
        
        # Calculate variances along principal axes
        a = I_xx*np.cos(PA)**2 + 2*I_xy*np.sin(PA)*np.cos(PA) + I_yy*np.sin(PA)**2
        b = I_xx*np.sin(PA)**2 - 2*I_xy*np.sin(PA)*np.cos(PA) + I_yy*np.cos(PA)**2
        
        # Calculate ellipticity (handle edge cases)
        if a > 0 and b > 0:
            ellipticity = 1 - np.sqrt(b / a)
            
            # Constrain ellipticity to reasonable range
            ellipticity = max(0, min(0.95, ellipticity))
        else:
            ellipticity = 0
            logger.warning("Invalid axis lengths, using circular model (e=0)")
        
        # Step 5: Calculate R_galaxy for each pixel
        # Prepare coordinate arrays
        dx = x_indices - x_center
        dy = y_indices - y_center
        
        # Rotate coordinates to align with principal axes
        x_prime = dx * np.cos(PA) + dy * np.sin(PA)
        y_prime = -dx * np.sin(PA) + dy * np.cos(PA)
        
        # Calculate elliptical radius
        scale_factor = 1.0 / max(1.0 - ellipticity, 0.001)  # Avoid division by zero
        R_galaxy = np.sqrt(x_prime**2 + (y_prime * scale_factor)**2)
        
        # Scale to arcseconds
        R_galaxy_arcsec = R_galaxy * pixel_size_x
        
        # Store ellipse parameters
        ellipse_params = {
            'center_x': x_center,
            'center_y': y_center,
            'PA_degrees': PA_degrees,
            'ellipticity': ellipticity,
            'a': np.sqrt(a),  # Semi-major axis scale
            'b': np.sqrt(b)   # Semi-minor axis scale
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