"""
Enhanced visualization utilities for ISAPC with error visualization capabilities
Handles consistent plotting for all analysis types with uncertainty display
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from typing import Optional, Tuple, Dict, Any, Union

from typing import List
import warnings
from scipy import stats
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)


def safe_tight_layout(fig=None):
    """
    Apply tight_layout with error handling to avoid warnings
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to adjust. If None, uses current figure.
    """
    import warnings
    
    if fig is None:
        fig = plt.gcf()
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            # Skip tight_layout if it fails
            pass


def safe_plot_array(values, bin_map, ax=None, title=None, cmap='viridis', label=None, 
                   vmin=None, vmax=None, errors=None, show_snr=False, snr_threshold=None):
    """
    Safely plot values mapped onto bins, handling non-numeric data types with error support
    
    Parameters
    ----------
    values : array-like
        Values for each bin
    bin_map : numpy.ndarray
        2D array of bin numbers
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap name
    label : str, optional
        Colorbar label
    vmin, vmax : float, optional
        Value range limits
    errors : array-like, optional
        Error values for each bin
    show_snr : bool, default=False
        Show S/N ratio instead of values
    snr_threshold : float, optional
        Mark regions below this S/N threshold
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    # Convert bin_map to integer type for mapping
    bin_map_int = np.asarray(bin_map, dtype=np.int32)
    
    # Convert values to numeric safely
    try:
        # If values is already numeric numpy array, this won't change it
        if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number):
            numeric_values = values
        else:
            # For non-numeric arrays, try to convert element by element
            numeric_values = np.zeros(len(values), dtype=float)
            for i, val in enumerate(values):
                try:
                    numeric_values[i] = float(val)
                except (ValueError, TypeError):
                    numeric_values[i] = np.nan
    except Exception as e:
        logger.warning(f"Error converting values to numeric: {e}")
        # Create NaN array as fallback
        numeric_values = np.full(np.max(bin_map_int) + 1, np.nan)
    
    # Convert errors to numeric if provided
    numeric_errors = None
    if errors is not None:
        try:
            if isinstance(errors, np.ndarray) and np.issubdtype(errors.dtype, np.number):
                numeric_errors = errors
            else:
                numeric_errors = np.zeros(len(errors), dtype=float)
                for i, err in enumerate(errors):
                    try:
                        numeric_errors[i] = float(err)
                    except (ValueError, TypeError):
                        numeric_errors[i] = np.nan
        except Exception as e:
            logger.warning(f"Error converting errors to numeric: {e}")
            numeric_errors = None
    
    # Create value map using bin numbers
    value_map = np.full_like(bin_map, np.nan, dtype=float)
    
    # Valid bins are non-negative and within range of values
    max_bin = min(np.max(bin_map_int), len(numeric_values) - 1)
    
    # Calculate S/N if requested and errors available
    if show_snr and numeric_errors is not None:
        # Calculate S/N ratio
        snr_values = np.zeros_like(numeric_values)
        for i in range(len(numeric_values)):
            if np.isfinite(numeric_values[i]) and np.isfinite(numeric_errors[i]) and numeric_errors[i] > 0:
                snr_values[i] = np.abs(numeric_values[i]) / numeric_errors[i]
            else:
                snr_values[i] = np.nan
        
        # Use S/N values for plotting
        plot_values = snr_values
        if label and not label.endswith('S/N'):
            label = f"{label} S/N"
    else:
        plot_values = numeric_values
    
    # Populate value map
    for bin_idx in range(max_bin + 1):
        # Safety check to avoid index errors
        if bin_idx < len(plot_values):
            value = plot_values[bin_idx]
            # Check if value is valid
            if np.isfinite(value):
                value_map[bin_map_int == bin_idx] = value
    
    # Mark low S/N regions if threshold provided
    if show_snr and snr_threshold is not None:
        # Create mask for low S/N
        low_snr_mask = value_map < snr_threshold
        
        # Create masked array for better visualization
        masked_data = np.ma.array(value_map, mask=~np.isfinite(value_map))
        
        # Plot with special handling for low S/N
        if vmin is None or vmax is None:
            valid_data = masked_data.compressed()
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = max(0, np.nanpercentile(valid_data, 5))
                if vmax is None:
                    vmax = np.nanpercentile(valid_data, 95)
        
        im = ax.imshow(masked_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Overlay low S/N regions
        low_snr_overlay = np.ma.array(np.ones_like(value_map), mask=~low_snr_mask)
        ax.imshow(low_snr_overlay, origin='lower', cmap='gray', alpha=0.5, vmin=0, vmax=1)
        
        # Add text annotation
        ax.text(0.02, 0.98, f'Gray: S/N < {snr_threshold}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Create masked array for better visualization
        masked_data = np.ma.array(value_map, mask=~np.isfinite(value_map))
        
        # Determine color limits
        if vmin is None or vmax is None:
            valid_data = masked_data.compressed()
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = np.nanpercentile(valid_data, 5)
                if vmax is None:
                    vmax = np.nanpercentile(valid_data, 95)
        
        # Plot the data
        im = ax.imshow(masked_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if label:
        cbar.set_label(label)
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Set aspect ratio for better visualization
    ax.set_aspect('equal')
    
    return ax


def standardize_figure_saving(fig, filename, dpi=150, transparent=False, ext=None):
    """
    Save figure with standardized settings and proper directory creation
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str or Path
        Output filename
    dpi : int, default=150
        Resolution for saving
    transparent : bool, default=False
        Whether to use transparent background
    ext : str, optional
        File extension override
    """
    try:
        # Convert to Path object
        path = Path(filename)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use specified extension or get from filename
        if ext:
            path = path.with_suffix(f".{ext}")
        
        # Apply tight layout with error handling
        safe_tight_layout(fig)
        
        # Save the figure
        fig.savefig(
            path, 
            dpi=dpi, 
            bbox_inches='tight', 
            transparent=transparent
        )
        
        logger.debug(f"Saved figure to {path}")
        
    except Exception as e:
        logger.warning(f"Error saving figure: {e}")
        try:
            # Fallback to simpler saving without tight_layout
            fig.savefig(filename, dpi=dpi)
        except Exception as e2:
            logger.error(f"Failed to save figure: {e2}")


def prepare_flux_map(cube):
    """
    Create a representative flux map from a data cube with robust handling of mismatched dimensions
    
    Parameters
    ----------
    cube : MUSECube
        MUSE cube with spectral data
        
    Returns
    -------
    numpy.ndarray
        2D flux map matching the dimensions of the cube
    """
    try:
        # Get expected dimensions first
        if hasattr(cube, "_n_y") and hasattr(cube, "_n_x"):
            expected_shape = (cube._n_y, cube._n_x)
        else:
            # Use a default if no dimensions specified
            expected_shape = None
        
        # Method 1: Use a specific wavelength range (V-band: 5000-5500Å)
        if hasattr(cube, "_lambda_gal") and hasattr(cube, "_cube_data"):
            try:
                wave = cube._lambda_gal
                cube_data = cube._cube_data
                
                # Verify dimensions match
                if cube_data.ndim == 3:
                    # Make sure wavelengths match
                    if len(wave) == cube_data.shape[0]:
                        # Create V-band flux map
                        wave_mask = (wave >= 5000) & (wave <= 5500)
                        if np.sum(wave_mask) > 0:
                            # Sum flux across V-band
                            flux_map = np.nanmean(cube_data[wave_mask], axis=0)
                            
                            # Verify flux_map has correct dimensions
                            if expected_shape and flux_map.shape != expected_shape:
                                logger.warning(f"Flux map shape {flux_map.shape} doesn't match expected {expected_shape}")
                                # Try to reshape if number of elements matches
                                if flux_map.size == expected_shape[0] * expected_shape[1]:
                                    flux_map = flux_map.reshape(expected_shape)
                                else:
                                    logger.warning("Cannot reshape flux map to expected dimensions")
                            
                            if np.any(np.isfinite(flux_map)):
                                logger.debug("Created flux map using V-band (5000-5500Å)")
                                return flux_map
            except Exception as e:
                logger.debug(f"Error in method 1: {e}")
        
        # Method 2: Use median of full spectrum
        if hasattr(cube, "_cube_data"):
            try:
                cube_data = cube._cube_data
                if cube_data.ndim == 3:
                    flux_map = np.nanmedian(cube_data, axis=0)
                    
                    # Check dimensions
                    if expected_shape and flux_map.shape != expected_shape:
                        if flux_map.size == expected_shape[0] * expected_shape[1]:
                            flux_map = flux_map.reshape(expected_shape)
                    
                    if np.any(np.isfinite(flux_map)):
                        logger.debug("Created flux map using full spectrum median")
                        return flux_map
            except Exception as e:
                logger.debug(f"Error in method 2: {e}")
        
        # Method 3: Try to use spectra in 2D format
        if hasattr(cube, "_spectra") and hasattr(cube, "_n_x") and hasattr(cube, "_n_y"):
            try:
                spectra = cube._spectra
                nx, ny = cube._n_x, cube._n_y
                
                # Calculate median flux for each spectrum
                median_flux = np.nanmedian(spectra, axis=0)
                
                # Create flux map with correct dimensions
                flux_map = np.full((ny, nx), np.nan)
                
                # Fill flux map if sizes match
                if len(median_flux) == nx * ny:
                    flux_map = median_flux.reshape(ny, nx)
                elif len(median_flux) < nx * ny:
                    # Fill partial data
                    flux_map.flat[:len(median_flux)] = median_flux
                else:
                    # Truncate data
                    flux_map = median_flux[:nx*ny].reshape(ny, nx)
                
                if np.any(np.isfinite(flux_map)):
                    logger.debug("Created flux map using reshaped spectra median")
                    return flux_map
            except Exception as e:
                logger.debug(f"Error in method 3: {e}")
                
        # Method 4: Try to extract physical radius map if available
        if hasattr(cube, "_physical_radius") and cube._physical_radius is not None:
            try:
                # Use a function of the physical radius as a flux map
                r_galaxy = cube._physical_radius
                
                # Create a simulated flux using the inverse of radius (highest at center)
                flux_map = np.exp(-r_galaxy / np.nanmedian(r_galaxy[np.isfinite(r_galaxy)]))
                
                # Check for expected dimensions
                if expected_shape and flux_map.shape != expected_shape:
                    if flux_map.size == expected_shape[0] * expected_shape[1]:
                        flux_map = flux_map.reshape(expected_shape)
                    else:
                        # Create interpolated map or a different size
                        logger.warning("Physical radius dimensions don't match expected cube dimensions")
                
                if np.any(np.isfinite(flux_map)):
                    logger.debug("Created flux map using physical radius")
                    return flux_map
            except Exception as e:
                logger.debug(f"Error in method 4: {e}")
        
        # Method 5: Create a synthetic flux map with expected dimensions
        if expected_shape:
            logger.warning(f"Creating synthetic flux map with dimensions {expected_shape}")
            ny, nx = expected_shape
            
            # Create simple synthetic galaxy
            y, x = np.indices((ny, nx))
            center_y, center_x = ny // 2, nx // 2
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            flux_map = np.exp(-r / (max(nx, ny) / 4))
            
            return flux_map
            
        # Final fallback with fixed dimensions
        logger.warning("Using fallback flux map with fixed dimensions (32x32)")
        return np.ones((32, 32))
        
    except Exception as e:
        logger.error(f"Error creating flux map: {e}")
        # Final fallback - if we have expected shape, use that
        if expected_shape:
            return np.ones(expected_shape)
        else:
            return np.ones((32, 32))


def process_wcs(wcs_obj):
    """
    Process WCS object to get a 2D slice suitable for plotting
    
    Parameters
    ----------
    wcs_obj : astropy.wcs.WCS
        WCS object
        
    Returns
    -------
    astropy.wcs.WCS or None
        Processed 2D WCS object or None if processing fails
    """
    if wcs_obj is None:
        return None
        
    try:
        # Handle WCS with more than 2 dimensions by slicing
        if wcs_obj.naxis > 2:
            try:
                from astropy.wcs import WCS
                
                # Try different approaches based on the astropy version
                if hasattr(wcs_obj, 'celestial'):
                    # Newer astropy versions
                    return wcs_obj.celestial
                elif hasattr(wcs_obj, 'sub'):
                    # Older astropy versions
                    return wcs_obj.sub([1, 2])
                else:
                    # Really old versions - create a new WCS manually
                    header = wcs_obj.to_header()
                    new_header = {}
                    for key in header:
                        if '1' in key or '2' in key:  # Keep only 1st and 2nd axes
                            new_header[key] = header[key]
                    return WCS(new_header)
            except Exception as e:
                logger.warning(f"Error creating 2D WCS: {e}")
                return None
        return wcs_obj
    except Exception as e:
        logger.warning(f"Error processing WCS: {e}")
        return None


def get_physical_extent(cube):
    """
    Get physical extent of a cube in arcseconds
    
    Parameters
    ----------
    cube : MUSECube
        Data cube
        
    Returns
    -------
    tuple or None
        (xmin, xmax, ymin, ymax) in arcseconds, or None if not available
    """
    try:
        if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
            nx, ny = cube._n_x, cube._n_y
            pixel_size_x = cube._pxl_size_x
            pixel_size_y = cube._pxl_size_y
            
            # Calculate extent centered at (0,0)
            extent = [
                -nx/2 * pixel_size_x, 
                nx/2 * pixel_size_x, 
                -ny/2 * pixel_size_y, 
                ny/2 * pixel_size_y
            ]
            return extent
        return None
    except Exception as e:
        logger.warning(f"Error calculating physical extent: {e}")
        return None


def get_figure_size_for_cube(cube, base_size=8):
    """
    Calculate figure size based on cube dimensions and pixel scale
    
    Parameters
    ----------
    cube : MUSECube
        Data cube
    base_size : float, default=8
        Base figure size
        
    Returns
    -------
    tuple
        (width, height) figure size
    """
    # Default square figure
    figsize = (base_size, base_size)
    
    # Adjust for physical scale if available
    if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y") and \
       hasattr(cube, "_n_x") and hasattr(cube, "_n_y"):
        # Get pixel aspect ratio
        pixel_ratio = cube._pxl_size_x / cube._pxl_size_y
        
        # Get image aspect ratio (width/height)
        image_ratio = (cube._n_x * cube._pxl_size_x) / (cube._n_y * cube._pxl_size_y)
        
        # Calculate figsize to maintain physical scale
        if image_ratio >= 1:  # Wider than tall
            figsize = (base_size, base_size / image_ratio)
        else:  # Taller than wide
            figsize = (base_size * image_ratio, base_size)
    
    return figsize


def plot_bin_indices(indices_dict, bin_map, title=None, save_path=None):
    """
    Plot spectral indices for binned data with safe handling of non-numeric data
    
    Parameters
    ----------
    indices_dict : dict
        Dictionary of spectral indices
    bin_map : numpy.ndarray
        2D array of bin numbers
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plots
    """
    # Get number of indices to plot
    n_indices = len(indices_dict)
    if n_indices == 0:
        logger.warning("No spectral indices to plot")
        return None
    
    # Determine grid layout
    if n_indices <= 3:
        n_rows, n_cols = 1, n_indices
    else:
        n_rows = (n_indices + 2) // 3  # Ceiling division
        n_cols = min(3, n_indices)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    # Ensure axes is array-like
    axes = np.atleast_1d(axes)
    
    # Plot each index with safe plotting
    for i, (index_name, values) in enumerate(indices_dict.items()):
        if i < axes.size:
            # Get current axis
            if axes.ndim == 1:
                ax = axes[i]
            else:
                ax = axes.flat[i]
                
            # Plot with safe handling of non-numeric data
            safe_plot_array(
                values=values,
                bin_map=bin_map,
                ax=ax,
                title=index_name,
                cmap='plasma',
                label='Index Value'
            )
    
    # Hide unused axes
    for i in range(len(indices_dict), axes.size):
        if axes.ndim == 1 and i < len(axes):
            axes[i].axis('off')
        elif axes.ndim > 1:
            axes.flat[i].axis('off')
    
    # Add title to figure
    if title:
        fig.suptitle(title, fontsize=16)
        # Adjust for title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
    
    # Save if path provided
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig


def plot_bin_map(bin_num, values=None, ax=None, cmap='viridis', title=None, 
                colorbar_label=None, physical_scale=True, pixel_size=None, 
                wcs=None, vmin=None, vmax=None, log_scale=False, cube=None):
    """
    Plot a map of binned data with consistent physical scaling
    
    Parameters
    ----------
    bin_num : numpy.ndarray
        2D array of bin numbers
    values : numpy.ndarray, optional
        Values for each bin
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    cmap : str, default='viridis'
        Colormap name
    title : str, optional
        Plot title
    colorbar_label : str, optional
        Label for colorbar
    physical_scale : bool, default=True
        Use physical coordinates
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation
    vmin, vmax : float, optional
        Value range for colormap
    log_scale : bool, default=False
        Use logarithmic scale for values
    cube : MUSECube, optional
        MUSE cube object, used to determine figure size
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    # Create axis if needed
    if ax is None:
        # Get appropriate figure size if cube is provided
        if cube is not None:
            figsize = get_figure_size_for_cube(cube)
        else:
            figsize = (8, 8)
            
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get dimensions
    ny, nx = bin_num.shape
    
    # Get pixel size from cube if provided and not explicitly specified
    if pixel_size is None and cube is not None:
        if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
            pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
            # Enable physical scale if we have pixel size
            physical_scale = True
    
    # Process pixel size if provided as a single value
    if isinstance(pixel_size, (int, float)):
        pixel_size = (pixel_size, pixel_size)
    
    # Safety check - make sure bin_num is proper array
    bin_num = np.asarray(bin_num)
    
    # Create a masked array for the bin data
    if values is not None:
        # Safety check - ensure values is proper array
        values = np.asarray(values)
        
        # Create a map of values by bin number
        value_map = np.zeros_like(bin_num, dtype=float)
        value_map.fill(np.nan)  # Fill with NaN by default
        
        # Validate bin numbers and ensure they're within range
        valid_bins = np.unique(bin_num[bin_num >= 0])
        
        # Check if values has enough elements
        if len(values) < len(valid_bins):
            logger.warning(f"Values array has {len(values)} elements but there are {len(valid_bins)} bins. " +
                          "Using available values and filling the rest with NaN.")
            # Fill only the bins we have values for
            max_bin_to_fill = min(len(values), np.max(valid_bins) + 1)
            for i in range(max_bin_to_fill):
                value_map[bin_num == i] = values[i] if i < len(values) else np.nan
        else:
            # Normal case - fill all bins
            for i in valid_bins:
                if i < len(values):
                    value_map[bin_num == i] = values[i]
        
        # Create masked array
        masked_data = np.ma.array(value_map, mask=~np.isfinite(value_map))
        
        # Determine color normalization
        valid_data = masked_data.compressed()
        if len(valid_data) > 0:
            if vmin is None:
                vmin = np.nanmin(valid_data)
            if vmax is None:
                vmax = np.nanmax(valid_data)
                
            if log_scale and np.all(valid_data > 0):
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        # Just plot bin numbers
        masked_data = np.ma.array(bin_num, mask=bin_num < 0)
        norm = None
    
    # Try WCS plotting first if physical scale is requested
    wcs_obj = process_wcs(wcs) if wcs is not None else None
    
    if wcs_obj is not None and physical_scale:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Get current figure
            fig = ax.figure
            
            # Create new axis with WCS projection if needed
            if not isinstance(ax, WCSAxes):
                position = ax.get_position()
                ax.remove()
                ax = fig.add_subplot(111, projection=wcs_obj)
                ax.set_position(position)
            
            # Plot the data with WCS coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm)
            
            # Add coordinate grid
            ax.grid(color='white', ls='solid', alpha=0.3)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            
            # Ensure aspect ratio is correct for WCS
            ax.set_aspect('equal')
        except Exception as e:
            logger.warning(f"Error plotting with WCS: {e}")
            wcs_obj = None
    
    # Use physical coordinates if WCS not available or failed
    if wcs_obj is None:
        if physical_scale and pixel_size is not None:
            # Calculate physical extent
            pixel_size_x, pixel_size_y = pixel_size
            extent = [
                -nx/2 * pixel_size_x, 
                nx/2 * pixel_size_x, 
                -ny/2 * pixel_size_y, 
                ny/2 * pixel_size_y
            ]
            
            # Plot with physical coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm, 
                         extent=extent, aspect=1.0)  # aspect=1.0 for equal physical scaling
            
            ax.set_xlabel('Δ RA (arcsec)')
            ax.set_ylabel('Δ Dec (arcsec)')
        else:
            # Plot with pixel coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm)
            ax.set_aspect('equal')  # Equal aspect ratio for pixels
            
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
    
    # Add colorbar if we have values
    if values is not None:
        cbar = plt.colorbar(im, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    # Set title
    if title:
        ax.set_title(title)
    
    return ax


def plot_flux_with_radial_bins(flux_map, bin_radii, center_x, center_y, pa=0, ellipticity=0,
                             wcs=None, pixel_size=None, ax=None, title=None, cmap='inferno',
                             log_scale=True, cube=None):
    """
    Plot flux map with radial bins overlay using consistent physical scaling
    
    Parameters
    ----------
    flux_map : numpy.ndarray
        2D flux map
    bin_radii : array-like
        Radii of bins in arcseconds
    center_x, center_y : float
        Center coordinates in pixels
    pa : float, default=0
        Position angle in degrees
    ellipticity : float, default=0
        Ellipticity (0-1)
    wcs : astropy.wcs.WCS, optional
        WCS object
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='inferno'
        Colormap
    log_scale : bool, default=True
        Use logarithmic scale for flux
    cube : MUSECube, optional
        MUSE cube object for determining figure size
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    # Create figure and axis if needed
    if ax is None:
        # Get appropriate figure size if cube is provided
        if cube is not None:
            figsize = get_figure_size_for_cube(cube)
        else:
            figsize = (8, 8)
            
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get dimensions
    ny, nx = flux_map.shape
    
    # Get pixel size from cube if provided and not explicitly specified
    if pixel_size is None and cube is not None:
        if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
            pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
    
    # Default pixel size if still not set
    if pixel_size is None:
        pixel_size = (0.2, 0.2)  # Default MUSE pixel size
        logger.warning("No pixel size provided, using default (0.2 arcsec)")
    
    # Process pixel size if provided as a single value
    if isinstance(pixel_size, (int, float)):
        pixel_size = (pixel_size, pixel_size)
        
    pixel_size_x, pixel_size_y = pixel_size
    
    # Handle NaN values and mask
    masked_flux = np.ma.array(flux_map, mask=~np.isfinite(flux_map))
    
    # Determine color normalization
    valid_flux = masked_flux.compressed()
    if len(valid_flux) > 0 and np.any(valid_flux > 0):
        # Logarithmic scale with safety for flux
        if log_scale:
            min_positive = np.nanmax([np.min(valid_flux[valid_flux > 0]), 1e-10])
            norm = LogNorm(vmin=min_positive, vmax=np.nanmax(valid_flux))
        else:
            # Linear scale
            vmin = np.nanmin(valid_flux)
            vmax = np.nanmax(valid_flux)
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        # Default normalization
        norm = Normalize(vmin=0, vmax=1)
    
    # Process WCS
    wcs_obj = process_wcs(wcs)
    
    # Use physical coordinates if available
    if wcs_obj is not None:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create new axis with WCS projection if needed
            if not isinstance(ax, WCSAxes):
                position = ax.get_position()
                ax.remove()
                ax = fig.add_subplot(111, projection=wcs_obj)
                ax.set_position(position)
            
            # Plot flux map
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap, norm=norm)
            
            # Add coordinate grid
            ax.grid(color='white', ls='solid', alpha=0.3)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            
            # Draw radial bins if we have physical size
            if True:  # Physical size is now guaranteed
                # Convert to physical coordinates
                # For WCS, we need to work with pixel coordinates
                # Calculate radii in pixels for ellipse drawing
                
                # Add ellipses for each radius
                for radius in bin_radii:
                    # For WCS, draw in pixel coordinates but with appropriate physical size
                    # Convert radius from arcsec to pixels
                    radius_pix_x = radius / pixel_size_x
                    radius_pix_y = radius / pixel_size_y
                    
                    # Account for ellipticity
                    pix_height = 2 * radius_pix_y * (1 - ellipticity)
                    
                    # Create ellipse
                    ell = Ellipse(
                        (center_x, center_y),  # Center in pixel coordinates
                        2 * radius_pix_x,      # Width (diameter) in pixels
                        pix_height,            # Height with ellipticity in pixels
                        angle=pa,              # Position angle in degrees
                        fill=False,
                        edgecolor='white',
                        linestyle='-',
                        linewidth=1.5,
                        alpha=0.8
                    )
                    ax.add_patch(ell)
            
        except Exception as e:
            logger.warning(f"Error plotting with WCS: {e}")
            wcs_obj = None
    
    # Fall back to physical coordinates without WCS
    if wcs_obj is None:
        # Calculate extent
        extent = [
            -nx/2 * pixel_size_x, 
            nx/2 * pixel_size_x, 
            -ny/2 * pixel_size_y, 
            ny/2 * pixel_size_y
        ]
        
        # Plot flux map with physical coordinates
        im = ax.imshow(masked_flux, origin='lower', cmap=cmap, norm=norm, 
                     extent=extent, aspect=1.0)  # aspect=1.0 for equal physical scaling
        
        # Draw radial bins
        # Convert center from pixels to physical coordinates
        center_x_phys = (center_x - nx/2) * pixel_size_x
        center_y_phys = (center_y - ny/2) * pixel_size_y
        
        for radius in bin_radii:
            if ellipticity == 0 or not np.isfinite(ellipticity):
                # Draw circle
                ell = Ellipse(
                    (center_x_phys, center_y_phys),
                    2 * radius,  # Diameter
                    2 * radius,
                    angle=0,
                    fill=False,
                    edgecolor='white',
                    linestyle='-',
                    linewidth=1.5,
                    alpha=0.8
                )
            else:
                # Draw ellipse
                ell = Ellipse(
                    (center_x_phys, center_y_phys),
                    2 * radius,  # Major axis
                    2 * radius * (1 - ellipticity),  # Minor axis
                    angle=pa,
                    fill=False,
                    edgecolor='white',
                    linestyle='-',
                    linewidth=1.5,
                    alpha=0.8
                )
            ax.add_patch(ell)
        
        ax.set_xlabel('Δ RA (arcsec)')
        ax.set_ylabel('Δ Dec (arcsec)')
    
    # Set title
    if title:
        ax.set_title(title)
        
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Flux')
    
    return fig, ax


def plot_ellipse_wcs(ax, ra, dec, major, minor, pa, **kwargs):
    """
    Plot ellipse in WCS coordinates
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        WCSAxes object
    ra, dec : float
        Center coordinates in degrees
    major, minor : float
        Major and minor axes in arcseconds
    pa : float
        Position angle in degrees
    **kwargs : dict
        Additional arguments for Ellipse
        
    Returns
    -------
    matplotlib.patches.Ellipse
        Ellipse patch
    """
    from matplotlib.patches import Ellipse
    import numpy as np
    
    # Default ellipse properties
    default_kwargs = {
        'fill': False,
        'edgecolor': 'white',
        'linestyle': '-',
        'linewidth': 1.5,
        'alpha': 0.8,
        'zorder': 10
    }
    
    # Combine with user-provided kwargs
    ellipse_kwargs = {**default_kwargs, **kwargs}
    
    # Convert major/minor from arcseconds to degrees
    major_deg = major / 3600
    minor_deg = minor / 3600
    
    # Create ellipse in sky coordinates
    # Note: angle is east of north, PA is measured from north to east
    angle = pa  # Adjust as needed for WCS convention
    
    # Create the ellipse patch
    ell = Ellipse((ra, dec), major_deg, minor_deg, angle=angle, **ellipse_kwargs)
    
    return ell


def plot_bin_boundaries_on_flux(bin_num, flux_map, cube, galaxy_name=None, binning_type="Voronoi", 
                               bin_centers=None, bin_radii=None, center_x=None, center_y=None,
                               pa=0, ellipticity=0, save_path=None):
    """
    Plot bin boundaries overlaid on flux map using consistent physical scaling
    
    Parameters
    ----------
    bin_num : numpy.ndarray
        2D array of bin numbers
    flux_map : numpy.ndarray
        2D flux map
    cube : MUSECube
        MUSE cube object
    galaxy_name : str, optional
        Galaxy name for title
    binning_type : str, default="Voronoi"
        Type of binning (Voronoi, Radial, etc.)
    bin_centers : tuple, optional
        (x, y) arrays of bin centers, for Voronoi
    bin_radii : array-like, optional
        Radii of bins in arcseconds, for radial
    center_x, center_y : float, optional
        Center coordinates in pixels, for radial
    pa : float, default=0
        Position angle in degrees, for radial
    ellipticity : float, default=0
        Ellipticity (0-1), for radial
    save_path : str or Path, optional
        Path to save the figure
        
    Returns
    -------
    tuple
        (fig, ax) - Figure and axis objects
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Always use physical coordinates if available
        physical_scale = True
        pixel_size = None
        
        if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
            pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
        else:
            physical_scale = False
            
        # Get WCS if available
        wcs = None
        if hasattr(cube, '_wcs'):
            wcs = cube._wcs
        
        # Ensure flux_map has the same shape as bin_num
        if flux_map.shape != bin_num.shape:
            logger.warning(f"Shape mismatch: flux_map {flux_map.shape} vs bin_num {bin_num.shape}")
            # Resize flux_map if needed
            if flux_map.size == bin_num.size:
                flux_map = flux_map.reshape(bin_num.shape)
            else:
                logger.error("Cannot reshape flux_map to match bin_num")
                return None, None
        
        # Plot flux map using physical coordinates
        im = plot_bin_map(
            flux_map,
            ax=ax,
            cmap='inferno',
            title=f"{galaxy_name} - {binning_type} Binning on Flux Map" if galaxy_name else f"{binning_type} Binning",
            physical_scale=physical_scale,
            pixel_size=pixel_size,
            wcs=wcs,
            log_scale=True,
            colorbar_label='Flux'
        )
        
        # Get dimensions
        ny, nx = bin_num.shape
        
        # Overlay bin boundaries based on binning type
        if binning_type.upper() == "VORONOI" and bin_centers is not None:
            # For Voronoi, plot bin centers and boundaries
            x_centers, y_centers = bin_centers
            
            # Plot bin centers
            if physical_scale and pixel_size is not None:
                # Convert to physical coordinates
                pixel_size_x, pixel_size_y = pixel_size
                
                # Convert from pixel to physical (arcsec)
                phys_x = (x_centers - nx/2) * pixel_size_x
                phys_y = (y_centers - ny/2) * pixel_size_y
                
                # Plot centers in physical space
                ax.plot(phys_x, phys_y, 'w+', markersize=6, alpha=0.8)
                
            else:
                # Plot centers in pixel space
                ax.plot(x_centers, y_centers, 'w+', markersize=6, alpha=0.8)
                
            # Draw bin boundaries - need to detect boundaries
            # This is complex for Voronoi - consider just showing the bin centers
            
        elif binning_type.upper() in ["RADIAL", "RDB"] and bin_radii is not None:
            # For radial, draw ellipses
            if center_x is None:
                center_x = nx / 2
            if center_y is None:
                center_y = ny / 2
                
            # Convert PA to proper angle definition if needed
            # In matplotlib, angle is clockwise from +x axis
                
            if physical_scale and pixel_size is not None:
                # Draw in physical coordinates
                pixel_size_x, pixel_size_y = pixel_size
                
                # Convert center to physical coords
                center_x_phys = (center_x - nx/2) * pixel_size_x
                center_y_phys = (center_y - ny/2) * pixel_size_y
                
                # Draw ellipses for each radius
                for radius in bin_radii:
                    ell = Ellipse(
                        (center_x_phys, center_y_phys),
                        2 * radius,  # Diameter
                        2 * radius * (1 - ellipticity),  # Account for ellipticity
                        angle=pa,
                        fill=False,
                        edgecolor='white',
                        linestyle='-',
                        linewidth=1.5,
                        alpha=0.8
                    )
                    ax.add_patch(ell)
            else:
                # Draw in pixel coordinates
                for radius in bin_radii:
                    # Convert radius from arcsec to pixels
                    if pixel_size is not None:
                        radius_pix = radius / pixel_size[0]
                    else:
                        radius_pix = radius  # Assume already in pixels
                        
                    ell = Ellipse(
                        (center_x, center_y),
                        2 * radius_pix,  # Diameter
                        2 * radius_pix * (1 - ellipticity),  # Account for ellipticity
                        angle=pa,
                        fill=False,
                        edgecolor='white',
                        linestyle='-',
                        linewidth=1.5,
                        alpha=0.8
                    )
                    ax.add_patch(ell)
        
        # Handle bin edges - use segmentation edge detection
        # Convert bin_num to integer type for contour finding
        bin_num_int = bin_num.astype(np.int32)
        
        # Find bin edges by looking for changes in bin numbers
        from scipy import ndimage
        
        # First create a binary edge image
        edges = np.zeros_like(bin_num_int)
        
        # Detect edges by comparing each pixel with its neighbors
        for shift_y in [-1, 0, 1]:
            for shift_x in [-1, 0, 1]:
                if shift_x == 0 and shift_y == 0:
                    continue  # Skip center
                    
                # Shift image
                shifted = np.zeros_like(bin_num_int)
                if shift_y < 0:
                    if shift_x < 0:
                        shifted[:-shift_y, :-shift_x] = bin_num_int[shift_y:, shift_x:]
                    elif shift_x == 0:
                        shifted[:-shift_y, :] = bin_num_int[shift_y:, :]
                    else:  # shift_x > 0
                        shifted[:-shift_y, shift_x:] = bin_num_int[shift_y:, :-shift_x]
                elif shift_y == 0:
                    if shift_x < 0:
                        shifted[:, :-shift_x] = bin_num_int[:, shift_x:]
                    else:  # shift_x > 0
                        shifted[:, shift_x:] = bin_num_int[:, :-shift_x]
                else:  # shift_y > 0
                    if shift_x < 0:
                        shifted[shift_y:, :-shift_x] = bin_num_int[:-shift_y, shift_x:]
                    elif shift_x == 0:
                        shifted[shift_y:, :] = bin_num_int[:-shift_y, :]
                    else:  # shift_x > 0
                        shifted[shift_y:, shift_x:] = bin_num_int[:-shift_y, :-shift_x]
                
                # Detect edges (where bin numbers change)
                edges = np.logical_or(edges, bin_num_int != shifted)
        
        # Plot bin edges using contour
        if physical_scale and pixel_size is not None:
            # Set up physical coordinate grid
            pixel_size_x, pixel_size_y = pixel_size
            y_grid, x_grid = np.mgrid[:ny, :nx]
            
            # Convert to physical coordinates
            x_phys = (x_grid - nx/2) * pixel_size_x
            y_phys = (y_grid - ny/2) * pixel_size_y
            
            # Plot contours
            ax.contour(x_phys, y_phys, edges, levels=[0.5], colors='white', alpha=0.6, linewidths=0.8)
        else:
            # Plot in pixel coordinates
            ax.contour(edges, levels=[0.5], colors='white', alpha=0.6, linewidths=0.8)
        
        # Save figure if requested
        if save_path is not None:
            # Ensure directory exists
            if isinstance(save_path, str):
                save_path = Path(save_path)
                
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Save with tight layout
            standardize_figure_saving(fig, save_path)
            
        return fig, ax
        
    except Exception as e:
        logger.error(f"Error in plot_bin_boundaries_on_flux: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def plot_kinematics_summary(velocity_field, dispersion_field, rotation_curve=None, params=None,
                          equal_aspect=True, physical_scale=True, pixel_size=None, wcs=None,
                          rotation_axis=True, cube=None):
    """
    Plot kinematics summary with physical scaling
    
    Parameters
    ----------
    velocity_field : numpy.ndarray
        2D velocity field
    dispersion_field : numpy.ndarray
        2D dispersion field
    rotation_curve : dict, optional
        Rotation curve data including 'radius' and 'velocity'
    params : dict, optional
        Kinematic parameters including 'pa', 'center', etc.
    equal_aspect : bool, default=True
        Use equal aspect ratio
    physical_scale : bool, default=True
        Use physical coordinates
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    wcs : astropy.wcs.WCS, optional
        WCS object
    rotation_axis : bool, default=True
        Show rotation axis
    cube : MUSECube, optional
        MUSE cube for determining figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plots
    """
    # Create figure with appropriate size
    if cube is not None:
        map_figsize = get_figure_size_for_cube(cube)
        fig_width = max(15, map_figsize[0] * 2)  # Ensure enough width for two panels
        fig_height = map_figsize[1] + 4  # Add height for rotation curve
    else:
        fig_width, fig_height = 15, 10
        
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create separate upper and lower rows
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 1.5], hspace=0.3, wspace=0.3)
    
    # Process WCS
    wcs_obj = process_wcs(wcs) if wcs is not None else None
    
    # Get dimensions
    ny, nx = velocity_field.shape
    
    # If pixel_size not provided but cube is, use cube's pixel size
    if pixel_size is None and cube is not None:
        if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
            pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
    
    # Calculate physical extent if needed
    extent = None
    if physical_scale and pixel_size is not None:
        pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
        extent = [
            -nx/2 * pixel_size_x, 
            nx/2 * pixel_size_x, 
            -ny/2 * pixel_size_y, 
            ny/2 * pixel_size_y
        ]
    
    # Determine color limits for velocity
    valid_vel = velocity_field[np.isfinite(velocity_field)]
    if len(valid_vel) > 0:
        vmax = np.percentile(np.abs(valid_vel), 95)
        vmin = -vmax
    else:
        vmin, vmax = -100, 100
        
    # Determine color limits for dispersion
    valid_disp = dispersion_field[np.isfinite(dispersion_field)]
    if len(valid_disp) > 0:
        vmin_disp = np.percentile(valid_disp, 5)
        vmax_disp = np.percentile(valid_disp, 95)
    else:
        vmin_disp, vmax_disp = 0, 100
    
    # Upper left: Velocity field
    ax_vel = fig.add_subplot(gs[0, 0])
    
    # Try WCS plotting for velocity field
    if wcs_obj is not None and physical_scale:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create new axis with WCS projection
            ax_vel.remove()
            ax_vel = fig.add_subplot(gs[0, 0], projection=wcs_obj)
            
            im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm', 
                                 vmin=vmin, vmax=vmax)
            
            ax_vel.grid(color='white', ls='solid', alpha=0.3)
            ax_vel.set_xlabel('RA')
            ax_vel.set_ylabel('Dec')
        except Exception as e:
            logger.warning(f"Error plotting velocity field with WCS: {e}")
            # Fall back to standard plotting with physical coordinates
            ax_vel = fig.add_subplot(gs[0, 0])
            
            if extent is not None:
                im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm',
                                     vmin=vmin, vmax=vmax, extent=extent, aspect=1.0)
                
                ax_vel.set_xlabel('Δ RA (arcsec)')
                ax_vel.set_ylabel('Δ Dec (arcsec)')
            else:
                im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm',
                                     vmin=vmin, vmax=vmax)
                ax_vel.set_aspect('equal')  # Equal aspect ratio for pixels
                
                ax_vel.set_xlabel('Pixels')
                ax_vel.set_ylabel('Pixels')
    else:
        # Standard plotting for velocity field with physical scaling if available
        if extent is not None:
            im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax, extent=extent, aspect=1.0)
            
            ax_vel.set_xlabel('Δ RA (arcsec)')
            ax_vel.set_ylabel('Δ Dec (arcsec)')
        else:
            im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax)
            ax_vel.set_aspect('equal')  # Equal aspect ratio for pixels
            
            ax_vel.set_xlabel('Pixels')
            ax_vel.set_ylabel('Pixels')
    
    plt.colorbar(im_vel, ax=ax_vel, label='Velocity (km/s)')
    ax_vel.set_title('Velocity Field')
    
    # Add rotation axis if requested and parameters available
    if rotation_axis and params is not None:
        if 'pa' in params and 'center' in params:
            pa = params['pa']
            center = params['center']
            
            # Calculate rotation axis endpoints
            if physical_scale and extent is not None:
                # Physical coordinates
                r = max(nx/2 * pixel_size_x, ny/2 * pixel_size_y) * 0.8
                cx = (center[0] - nx/2) * pixel_size_x if len(center) >= 2 else 0
                cy = (center[1] - ny/2) * pixel_size_y if len(center) >= 2 else 0
            else:
                # Pixel coordinates
                r = max(nx, ny) * 0.4
                cx = center[0] if len(center) >= 1 else nx/2
                cy = center[1] if len(center) >= 2 else ny/2
            
            # Calculate endpoint coordinates
            pa_rad = np.radians(pa)
            x1 = cx + r * np.cos(pa_rad)
            y1 = cy + r * np.sin(pa_rad)
            x2 = cx - r * np.cos(pa_rad)
            y2 = cy - r * np.sin(pa_rad)
            
            # Plot rotation axis
            ax_vel.plot([x1, x2], [y1, y2], 'k-', lw=2, alpha=0.7)
    
    # Upper right: Velocity dispersion
    ax_disp = fig.add_subplot(gs[0, 1])
    
    # Try WCS plotting for dispersion
    if wcs_obj is not None and physical_scale:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create new axis with WCS projection
            ax_disp.remove()
            ax_disp = fig.add_subplot(gs[0, 1], projection=wcs_obj)
            
            im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp)
            
            ax_disp.grid(color='white', ls='solid', alpha=0.3)
            ax_disp.set_xlabel('RA')
            ax_disp.set_ylabel('Dec')
        except Exception as e:
            logger.warning(f"Error plotting dispersion field with WCS: {e}")
            # Fall back to standard plotting with physical coordinates
            ax_disp = fig.add_subplot(gs[0, 1])
            
            if extent is not None:
                im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                       vmin=vmin_disp, vmax=vmax_disp, extent=extent, aspect=1.0)
                
                ax_disp.set_xlabel('Δ RA (arcsec)')
                ax_disp.set_ylabel('Δ Dec (arcsec)')
            else:
                im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                       vmin=vmin_disp, vmax=vmax_disp)
                ax_disp.set_aspect('equal')  # Equal aspect ratio for pixels
                
                ax_disp.set_xlabel('Pixels')
                ax_disp.set_ylabel('Pixels')
    else:
        # Standard plotting for dispersion with physical scaling if available
        if extent is not None:
            im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp, extent=extent, aspect=1.0)
            
            ax_disp.set_xlabel('Δ RA (arcsec)')
            ax_disp.set_ylabel('Δ Dec (arcsec)')
        else:
            im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp)
            ax_disp.set_aspect('equal')  # Equal aspect ratio for pixels
            
            ax_disp.set_xlabel('Pixels')
            ax_disp.set_ylabel('Pixels')
    
    plt.colorbar(im_disp, ax=ax_disp, label='Dispersion (km/s)')
    ax_disp.set_title('Velocity Dispersion')
    
    # Lower: Rotation curve if available
    if rotation_curve is not None and 'radius' in rotation_curve and 'velocity' in rotation_curve:
        ax_curve = fig.add_subplot(gs[1, :])
        
        radius = rotation_curve['radius']
        velocity = rotation_curve['velocity']
        
        # Plot data points
        ax_curve.plot(radius, velocity, 'o', color='darkblue', alpha=0.7, label='Data')
        
        # Plot fit if available
        if 'fit' in rotation_curve:
            fit_radius = rotation_curve.get('fit_radius', radius)
            fit_velocity = rotation_curve['fit']
            ax_curve.plot(fit_radius, fit_velocity, '-', color='red', lw=2, alpha=0.8, label='Fit')
        
        # Add parameter annotations
        if params is not None:
            param_text = []
            
            if 'pa' in params:
                param_text.append(f"PA = {params['pa']:.1f}°")
            
            if 'vsys' in params:
                param_text.append(f"Vsys = {params['vsys']:.1f} km/s")
            
            if 'vmax' in params:
                param_text.append(f"Vmax = {params['vmax']:.1f} km/s")
            
            if 'r_eff' in params:
                param_text.append(f"Reff = {params['r_eff']:.1f} arcsec")
            
            if 'i' in params:
                param_text.append(f"i = {params['i']:.1f}°")
            
            if param_text:
                param_str = '\n'.join(param_text)
                ax_curve.text(
                    0.95, 0.95, param_str,
                    transform=ax_curve.transAxes,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
                )
        
        ax_curve.set_xlabel('Radius (arcsec)')
        ax_curve.set_ylabel('Velocity (km/s)')
        ax_curve.set_title('Rotation Curve')
        ax_curve.grid(True, alpha=0.3)
        ax_curve.legend()
    else:
        # Create empty plot for parameters if no rotation curve
        ax_params = fig.add_subplot(gs[1, :])
        
        if params is not None:
            param_text = []
            
            if 'pa' in params:
                param_text.append(f"PA = {params['pa']:.1f}°")
            
            if 'vsys' in params:
                param_text.append(f"Vsys = {params['vsys']:.1f} km/s")
            
            if 'ellipticity' in params:
                param_text.append(f"Ellipticity = {params['ellipticity']:.2f}")
                
            if 'i' in params:
                param_text.append(f"i = {params['i']:.1f}°")
                
            if 'center' in params:
                center = params['center']
                if len(center) >= 2:
                    param_text.append(f"Center = ({center[0]:.1f}, {center[1]:.1f})")
            
            if param_text:
                param_str = '\n'.join(param_text)
                ax_params.text(
                    0.5, 0.5, param_str,
                    ha='center', va='center',
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
                )
        
        ax_params.set_title('Kinematic Parameters')
        ax_params.set_xticks([])
        ax_params.set_yticks([])
    
    # Apply tight layout safely
    safe_tight_layout(fig)
    
    return fig


def plot_gas_kinematics(velocity_field, dispersion_field, equal_aspect=True, 
                      physical_scale=True, pixel_size=None, wcs=None, rot_angle=0.0,
                      cube=None):
    """
    Plot gas kinematics maps
    
    Parameters
    ----------
    velocity_field : numpy.ndarray
        2D velocity field
    dispersion_field : numpy.ndarray
        2D dispersion field
    equal_aspect : bool, default=True
        Use equal aspect ratio
    physical_scale : bool, default=True
        Use physical coordinates
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    wcs : astropy.wcs.WCS, optional
        WCS object
    rot_angle : float, default=0.0
        Rotation angle in degrees for axes
    cube : MUSECube, optional
        MUSE cube for determining figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plots
    """
    # Create figure with appropriate size
    if cube is not None:
        map_figsize = get_figure_size_for_cube(cube)
        fig_width = max(12, map_figsize[0] * 2)  # Ensure enough width for two panels
        fig_height = map_figsize[1]
    else:
        fig_width, fig_height = 12, 5
        
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Process WCS
    wcs_obj = process_wcs(wcs) if wcs is not None else None
    
    # Get dimensions
    ny, nx = velocity_field.shape
    
    # If pixel_size not provided but cube is, use cube's pixel size
    if pixel_size is None and cube is not None:
        if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
            pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
    
    # Calculate physical extent if needed
    extent = None
    if physical_scale and pixel_size is not None:
        pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
        extent = [
            -nx/2 * pixel_size_x, 
            nx/2 * pixel_size_x, 
            -ny/2 * pixel_size_y, 
            ny/2 * pixel_size_y
        ]
    
    # Determine color limits for velocity
    valid_vel = velocity_field[np.isfinite(velocity_field)]
    if len(valid_vel) > 0:
        # Create symmetric range
        vmax = np.percentile(np.abs(valid_vel), 95)
        vmin = -vmax
    else:
        vmin, vmax = -100, 100
    
    # Determine color limits for dispersion
    valid_disp = dispersion_field[np.isfinite(dispersion_field)]
    if len(valid_disp) > 0:
        vmin_disp = np.percentile(valid_disp, 5)
        vmax_disp = np.percentile(valid_disp, 95)
    else:
        vmin_disp, vmax_disp = 0, 100
    
    # Plot with WCS if available
    if wcs_obj is not None and physical_scale:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create new axes with WCS projection
            for i, ax in enumerate(axes):
                # Remember position
                position = ax.get_position()
                ax.remove()
                axes[i] = fig.add_subplot(1, 2, i+1, projection=wcs_obj)
                axes[i].set_position(position)
            
            # Velocity field
            im_vel = axes[0].imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax)
            axes[0].grid(color='white', ls='solid', alpha=0.3)
            axes[0].set_xlabel('RA')
            axes[0].set_ylabel('Dec')
            
            # Velocity dispersion
            im_disp = axes[1].imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp)
            axes[1].grid(color='white', ls='solid', alpha=0.3)
            axes[1].set_xlabel('RA')
            axes[1].set_ylabel('Dec')
        except Exception as e:
            logger.warning(f"Error plotting with WCS: {e}")
            wcs_obj = None
    
    # Fall back to standard plotting if WCS failed or isn't available
    if wcs_obj is None:
        # Recreate axes if needed
        if 'WCSAxes' in str(type(axes[0])):
            axes[0].remove()
            axes[1].remove()
            axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
        
        if extent is not None:
            # Physical coordinates without WCS
            im_vel = axes[0].imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax, extent=extent, aspect=1.0)
                                 
            im_disp = axes[1].imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp, extent=extent, aspect=1.0)
            
            # Set labels with rotation angle if specified
            if rot_angle != 0:
                axes[0].set_xlabel(f'Δ RA cos({rot_angle:.0f}°) + Δ DEC sin({rot_angle:.0f}°) (arcsec)')
                axes[0].set_ylabel(f'-Δ RA sin({rot_angle:.0f}°) + Δ DEC cos({rot_angle:.0f}°) (arcsec)')
                axes[1].set_xlabel(f'Δ RA cos({rot_angle:.0f}°) + Δ DEC sin({rot_angle:.0f}°) (arcsec)')
                axes[1].set_ylabel(f'-Δ RA sin({rot_angle:.0f}°) + Δ DEC cos({rot_angle:.0f}°) (arcsec)')
            else:
                axes[0].set_xlabel('Δ RA (arcsec)')
                axes[0].set_ylabel('Δ Dec (arcsec)')
                axes[1].set_xlabel('Δ RA (arcsec)')
                axes[1].set_ylabel('Δ Dec (arcsec)')
        else:
            # Pixel coordinates
            im_vel = axes[0].imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax)
            im_disp = axes[1].imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp)
            
            # Use equal aspect ratio for pixel coordinates
            if equal_aspect:
                axes[0].set_aspect('equal')
                axes[1].set_aspect('equal')
            
            axes[0].set_xlabel('Pixels')
            axes[0].set_ylabel('Pixels')
            axes[1].set_xlabel('Pixels')
            axes[1].set_ylabel('Pixels')
    
    # Add colorbars
    plt.colorbar(im_vel, ax=axes[0], label='Velocity (km/s)')
    plt.colorbar(im_disp, ax=axes[1], label='Dispersion (km/s)')
    
    # Set titles
    axes[0].set_title('Gas Velocity Field')
    axes[1].set_title('Gas Velocity Dispersion')
    
    # Add grid for reference
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    # Set figure title
    fig.suptitle('Gas Kinematics', fontsize=14)
    
    # Apply tight layout safely
    safe_tight_layout(fig)
    
    return fig


def plot_spectrum_fit(wavelength, observed_flux, model_flux, stellar_flux=None, gas_flux=None, 
                    error=None, title=None):
    """
    Plot spectrum fit with components (backward compatibility wrapper)
    
    Parameters
    ----------
    wavelength : numpy.ndarray
        Wavelength array
    observed_flux : numpy.ndarray
        Observed flux array
    model_flux : numpy.ndarray
        Model flux array
    stellar_flux : numpy.ndarray, optional
        Stellar component flux
    gas_flux : numpy.ndarray, optional
        Gas emission component flux
    error : numpy.ndarray, optional
        Error array
    title : str, optional
        Plot title
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    # Call the enhanced version with default settings for backward compatibility
    fig, axes = plot_spectrum_fit_with_errors(
        wavelength, observed_flux, model_flux, stellar_flux, gas_flux,
        error, title, figsize=(10, 6), show_residuals=False, show_snr=False
    )
    
    # Return just the main axis for compatibility
    return fig, axes[0] if isinstance(axes, list) else axes


def plot_spectrum_fit_with_errors(wavelength, observed_flux, model_flux, stellar_flux=None, 
                                 gas_flux=None, error=None, title=None, figsize=(12, 8),
                                 show_residuals=True, show_snr=False):
    """
    Enhanced spectrum fit plot with error bars and residuals
    
    Parameters
    ----------
    wavelength : numpy.ndarray
        Wavelength array
    observed_flux : numpy.ndarray
        Observed flux array
    model_flux : numpy.ndarray
        Model flux array
    stellar_flux : numpy.ndarray, optional
        Stellar component flux
    gas_flux : numpy.ndarray, optional
        Gas emission component flux
    error : numpy.ndarray, optional
        Error array
    title : str, optional
        Plot title
    figsize : tuple, default=(12, 8)
        Figure size
    show_residuals : bool, default=True
        Show residuals panel
    show_snr : bool, default=False
        Show S/N ratio panel
        
    Returns
    -------
    fig, axes : tuple
        Figure and axes objects
    """
    # Create figure with subplots
    if show_residuals and show_snr and error is not None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, 
                               gridspec_kw={'height_ratios': [3, 1, 1]},
                               sharex=True)
        ax_main, ax_res, ax_snr = axes
    elif show_residuals:
        fig, axes = plt.subplots(2, 1, figsize=figsize,
                               gridspec_kw={'height_ratios': [3, 1]},
                               sharex=True)
        ax_main, ax_res = axes
        ax_snr = None
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
        axes = [ax_main]
        ax_res = None
        ax_snr = None
    
    # Plot observed spectrum with error bars
    if error is not None:
        # Plot with error band
        ax_main.fill_between(wavelength, observed_flux - error, observed_flux + error,
                           color='gray', alpha=0.2, label='Error')
        ax_main.plot(wavelength, observed_flux, 'k-', alpha=0.7, label='Observed', lw=1)
    else:
        ax_main.plot(wavelength, observed_flux, 'k-', alpha=0.7, label='Observed', lw=1.5)
    
    # Plot model
    ax_main.plot(wavelength, model_flux, 'r-', lw=1.5, alpha=0.8, label='Best fit')
    
    # Plot components if available
    if stellar_flux is not None:
        ax_main.plot(wavelength, stellar_flux, 'b-', lw=1.5, alpha=0.6, label='Stellar')
    
    if gas_flux is not None:
        ax_main.plot(wavelength, gas_flux, 'g-', lw=1.5, alpha=0.6, label='Gas')
    
    # Calculate reasonable y-axis range with error handling
    valid_flux = observed_flux[np.isfinite(observed_flux)]
    valid_model = model_flux[np.isfinite(model_flux)]
    
    if len(valid_flux) > 0 and len(valid_model) > 0:
        min_y = np.nanmin([np.nanmin(valid_flux), np.nanmin(valid_model)])
        max_y = np.nanmax([np.nanmax(valid_flux), np.nanmax(valid_model)])
        
        # Add a margin
        range_y = max_y - min_y
        y_margin = 0.1 * range_y
        ax_main.set_ylim(min_y - y_margin, max_y + y_margin)
    
    # Plot residuals
    if show_residuals and ax_res is not None:
        residuals = observed_flux - model_flux
        
        if error is not None:
            # Normalized residuals
            normalized_residuals = residuals / error
            ax_res.fill_between(wavelength, -1, 1, color='gray', alpha=0.2)
            ax_res.plot(wavelength, normalized_residuals, 'ko', markersize=2, alpha=0.6)
            ax_res.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax_res.set_ylabel('Normalized\nResiduals')
            ax_res.set_ylim(-5, 5)
            
            # Add 3-sigma lines
            ax_res.axhline(y=3, color='gray', linestyle=':', alpha=0.5)
            ax_res.axhline(y=-3, color='gray', linestyle=':', alpha=0.5)
        else:
            # Regular residuals
            ax_res.plot(wavelength, residuals, 'ko', markersize=2, alpha=0.6)
            ax_res.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax_res.set_ylabel('Residuals')
    
    # Plot S/N ratio
    if show_snr and ax_snr is not None and error is not None:
        snr = np.abs(observed_flux) / error
        ax_snr.plot(wavelength, snr, 'g-', alpha=0.7)
        ax_snr.set_ylabel('S/N Ratio')
        ax_snr.set_ylim(0, np.percentile(snr[np.isfinite(snr)], 95) * 1.1)
        ax_snr.axhline(y=3, color='r', linestyle='--', alpha=0.5, label='S/N = 3')
        ax_snr.grid(True, alpha=0.3)
    
    # Add grid, legend and labels
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right')
    ax_main.set_ylabel('Flux')
    
    # Set x-label on bottom plot
    if ax_snr is not None:
        ax_snr.set_xlabel('Wavelength (Å)')
    elif ax_res is not None:
        ax_res.set_xlabel('Wavelength (Å)')
    else:
        ax_main.set_xlabel('Wavelength (Å)')
    
    # Set title
    if title:
        ax_main.set_title(title)
    else:
        ax_main.set_title('Spectrum Fit')
    
    # Apply tight layout safely
    safe_tight_layout(fig)
    
    return fig, axes


def plot_binned_spectra(cube, spectra, wavelength, indices=None, title=None, save_path=None):
    """
    Plot binned spectra for selected bins
    
    Parameters
    ----------
    cube : MUSECube
        MUSE cube object
    spectra : numpy.ndarray
        2D array of spectra [n_wave, n_bins]
    wavelength : numpy.ndarray
        1D wavelength array
    indices : list, optional
        List of bin indices to plot. If None, a representative selection is made.
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save the figure
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    n_bins = spectra.shape[1]
    
    # Select indices to plot if not provided
    if indices is None:
        if n_bins <= 10:
            indices = list(range(n_bins))
        else:
            # Select a representative set of bins
            indices = [0]  # First bin
            if n_bins >= 3:
                indices.append(n_bins // 2)  # Middle bin
            if n_bins >= 2:
                indices.append(n_bins - 1)  # Last bin
            
            # Add some evenly spaced bins
            if n_bins > 10:
                step = n_bins // 5
                for i in range(step, n_bins - 1, step):
                    if i not in indices:
                        indices.append(i)
                indices.sort()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot spectra with offset for clarity
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    max_flux = 0
    
    for i, bin_idx in enumerate(indices):
        if bin_idx < n_bins:
            # Get spectrum and normalize for better visibility
            spectrum = spectra[:, bin_idx]
            if np.any(np.isfinite(spectrum)):
                valid_flux = spectrum[np.isfinite(spectrum)]
                norm_factor = np.max(np.abs(valid_flux)) if len(valid_flux) > 0 else 1.0
                if norm_factor > 0:
                    normalized = spectrum / norm_factor
                else:
                    normalized = spectrum
                
                # Apply offset
                offset = i * 1.5
                plot_data = normalized + offset
                max_flux = max(max_flux, np.nanmax(plot_data))
                
                # Plot with label
                ax.plot(wavelength, plot_data, color=colors[i], label=f'Bin {bin_idx}')
    
    # Customize plot
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux + Offset')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Binned Spectra')
    
    # Add wavelength range markers if available
    if hasattr(cube, '_goodwavelengthrange') and len(cube._goodwavelengthrange) == 2:
        for wl in cube._goodwavelengthrange:
            ax.axvline(x=wl, color='gray', linestyle='--', alpha=0.7)
            
    # Add legend with smaller font and multiple columns if many bins
    if len(indices) > 10:
        ax.legend(fontsize='small', ncol=2)
    else:
        ax.legend()
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits to show all spectra
    ax.set_ylim(-0.5, max_flux + 0.5)
    
    # Save if path provided
    if save_path:
        standardize_figure_saving(fig, save_path, dpi=150)
    
    return fig, ax


def plot_ppxf_fit(wavelength, observed, bestfit, residuals=None, emission=None, 
                save_path=None, title=None, redshift=0.0):
    """
    Create a comprehensive pPXF fitting results plot
    
    Parameters
    ----------
    wavelength : numpy.ndarray
        Wavelength array
    observed : numpy.ndarray
        Observed spectrum
    bestfit : numpy.ndarray
        Best-fit model
    residuals : numpy.ndarray, optional
        Residuals between observed and bestfit
    emission : numpy.ndarray, optional
        Emission line component
    save_path : str or Path, optional
        Path to save the figure
    title : str, optional
        Plot title
    redshift : float, default=0.0
        Redshift value for rest-frame conversion
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot observed data
    ax.plot(wavelength, observed, 'k-', label='Observed', alpha=0.7)
    
    # Plot best-fit model
    ax.plot(wavelength, bestfit, 'r-', label='Best fit', lw=1.5, alpha=0.8)
    
    # Plot residuals with offset if provided
    if residuals is not None:
        # Calculate offset to show residuals below the spectrum
        offset = np.nanmin(observed) - np.nanmax(residuals) - 0.1 * np.nanmax(observed)
        ax.plot(wavelength, residuals + offset, 'b-', label=f'Residuals (offset:{offset:.1f})', alpha=0.7)
    
    # Plot emission component if provided
    if emission is not None and np.any(emission != 0):
        # Calculate scaling factor to make emission lines visible
        scale_factor = 0.5 * np.nanmax(observed) / np.nanmax(emission) if np.nanmax(emission) > 0 else 1.0
        scaled_emission = emission * scale_factor
        
        ax.plot(wavelength, scaled_emission, 'g-', 
               label=f'Emission lines (×{scale_factor:.1f})', alpha=0.6, lw=1.5)
    
    # Add gray bands for masked/masked regions if available
    # This could be expanded based on your masking approach
    
    # Add labels for important spectral features if in range
    features = {
        4861: 'Hβ',
        4959: '[OIII]',
        5007: '[OIII]',
        5177: 'Mgb',
        5892: 'NaD'
    }
    
    for wave, name in features.items():
        if min(wavelength) < wave < max(wavelength):
            ax.axvline(x=wave, color='gray', linestyle=':', alpha=0.7)
            # Add text slightly above the x-axis
            ypos = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(wave, ypos, name, rotation=90, 
                   va='bottom', ha='center', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Set axis labels with rest-frame note if redshift provided
    if redshift != 0:
        ax.set_xlabel(f'Wavelength (Å, rest-frame, z={redshift:.4f})')
    else:
        ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Spectral Fitting Results')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Save if path provided
    if save_path:
        standardize_figure_saving(fig, save_path, dpi=150)
    
    return fig, ax


def plot_parameter_profile(radius, values, errors=None, param_name='Parameter', 
                          galaxy_name=None, save_path=None, log_scale=False):
    """
    Create a plot of parameter values vs. radius
    
    Parameters
    ----------
    radius : numpy.ndarray
        Radius values in arcsec
    values : numpy.ndarray
        Parameter values
    errors : numpy.ndarray, optional
        Error bars for parameter values
    param_name : str, default='Parameter'
        Name of the parameter being plotted
    galaxy_name : str, optional
        Galaxy name for title
    save_path : str or Path, optional
        Path to save the figure
    log_scale : bool, default=False
        Use logarithmic scale for y-axis
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by radius for cleaner profile plots
    idx = np.argsort(radius)
    r_sorted = radius[idx]
    values_sorted = values[idx]
    
    # Plot profile line with points
    if errors is not None:
        errors_sorted = errors[idx]
        ax.errorbar(r_sorted, values_sorted, yerr=errors_sorted, 
                   fmt='o-', markersize=6, linewidth=1.5, capsize=3)
    else:
        ax.plot(r_sorted, values_sorted, 'o-', markersize=6, linewidth=1.5)
    
    # Set axis labels
    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel(param_name)
    
    # Add logarithmic scale if requested and values are positive
    if log_scale and np.all(values[np.isfinite(values)] > 0):
        ax.set_yscale('log')
    
    # Set title
    if galaxy_name:
        ax.set_title(f'{galaxy_name} - {param_name} Radial Profile')
    else:
        ax.set_title(f'{param_name} Radial Profile')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        standardize_figure_saving(fig, save_path, dpi=150)
    
    return fig, ax


def create_diagnostic_plot(cube, indices=None, title=None, output_dir=None):
    """
    Create a comprehensive diagnostic plot for a MUSE cube
    
    Parameters
    ----------
    cube : MUSECube
        MUSE cube object
    indices : list, optional
        List of spaxel indices to sample
    title : str, optional
        Plot title
    output_dir : str or Path, optional
        Directory to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plots
    """
    # Get appropriate figure size based on cube dimensions
    map_figsize = get_figure_size_for_cube(cube)
    fig_width = max(16, map_figsize[0] * 3)  # Ensure enough width for three panels
    fig_height = max(12, map_figsize[1] * 2)  # Ensure enough height for two rows
    
    # Create a multi-panel diagnostic figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Set up the GridSpec - 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
    
    # 1. Wavelength coverage and throughput (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if hasattr(cube, '_lambda_gal') and hasattr(cube, '_spectra'):
        wave = cube._lambda_gal
        # Calculate median spectrum across spaxels
        med_spec = np.nanmedian(cube._spectra, axis=1)
        ax1.plot(wave, med_spec, 'k-', alpha=0.7)
        ax1.set_xlabel('Wavelength (Å)')
        ax1.set_ylabel('Median Flux')
        ax1.set_title('Spectral Coverage')
        
        # Highlight useful spectral regions
        regions = {
            (4850, 4870): 'Hβ',
            (4950, 5020): '[OIII]',
            (5160, 5190): 'Mgb'
        }
        
        for (w1, w2), name in regions.items():
            if w1 > min(wave) and w2 < max(wave):
                ax1.axvspan(w1, w2, alpha=0.2, color='blue')
                # Add label at top of the shaded region
                y_pos = ax1.get_ylim()[1] * 0.9
                ax1.text((w1 + w2)/2, y_pos, name, 
                        ha='center', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Mark wavelength range used for analysis if available
        if hasattr(cube, '_goodwavelengthrange') and len(cube._goodwavelengthrange) == 2:
            for wl in cube._goodwavelengthrange:
                ax1.axvline(x=wl, color='red', linestyle='--', alpha=0.7)
            ax1.axvspan(
                cube._goodwavelengthrange[0], 
                cube._goodwavelengthrange[1], 
                alpha=0.1, color='green',
                label='Analysis Range'
            )
            ax1.legend(loc='upper right', fontsize='small')
    else:
        ax1.text(0.5, 0.5, 'Wavelength data not available', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Flux map - use the prepare_flux_map function (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        flux_map = prepare_flux_map(cube)
        
        # Use proper non-square pixels with physical scale
        if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
            pixel_size_x = cube._pxl_size_x
            pixel_size_y = cube._pxl_size_y
            
            # Calculate extent
            extent = [
                -cube._n_x/2 * pixel_size_x, 
                cube._n_x/2 * pixel_size_x, 
                -cube._n_y/2 * pixel_size_y, 
                cube._n_y/2 * pixel_size_y
            ]
            
            im2 = ax2.imshow(
                flux_map, 
                origin='lower', 
                cmap='inferno',
                norm=LogNorm(vmin=np.nanpercentile(flux_map[flux_map > 0], 1),
                            vmax=np.nanpercentile(flux_map, 99)),
                extent=extent,
                aspect=1.0  # aspect=1.0 for equal physical scaling
            )
            
            ax2.set_xlabel('Δ RA (arcsec)')
            ax2.set_ylabel('Δ Dec (arcsec)')
        else:
            im2 = ax2.imshow(
                flux_map, 
                origin='lower', 
                cmap='inferno',
                norm=LogNorm(vmin=np.nanpercentile(flux_map[flux_map > 0], 1),
                            vmax=np.nanpercentile(flux_map, 99))
            )
            ax2.set_aspect('equal')  # Equal aspect ratio for pixel coordinates
            
            ax2.set_xlabel('Pixels')
            ax2.set_ylabel('Pixels')
            
        plt.colorbar(im2, ax=ax2, label='Flux')
        ax2.set_title('Flux Map')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error creating flux map:\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Sample spectral fits (if available) (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if hasattr(cube, '_bin_bestfit') and hasattr(cube, '_binned_data'):
        try:
            # Get data for bin 0 (or another representative bin)
            bin_idx = 0
            wave = cube._binned_data.wavelength
            spec = cube._binned_data.spectra[:, bin_idx]
            bestfit = cube._bin_bestfit[:, bin_idx]
            
            # Plot
            ax3.plot(wave, spec, 'k-', alpha=0.7, label='Data')
            ax3.plot(wave, bestfit, 'r-', alpha=0.8, label='Fit')
            
            # Add residuals at bottom
            residuals = spec - bestfit
            offset = np.nanmin(spec) - 1.2 * np.nanmax(np.abs(residuals))
            ax3.plot(wave, residuals + offset, 'b-', alpha=0.7, label='Residuals')
            
            ax3.set_xlabel('Wavelength (Å)')
            ax3.set_ylabel('Flux')
            ax3.set_title('Sample Spectral Fit')
            ax3.legend(loc='upper right', fontsize='small')
            
            # Limit x range to useful region
            if hasattr(cube, '_goodwavelengthrange') and len(cube._goodwavelengthrange) == 2:
                ax3.set_xlim(cube._goodwavelengthrange[0], cube._goodwavelengthrange[1])
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error plotting spectral fit:\n{str(e)}', 
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'Spectral fit data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Velocity field (if available) (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    if hasattr(cube, '_bin_velocity') and hasattr(cube, '_bin_num'):
        try:
            # Create 2D velocity field
            bin_num = cube._bin_num
            velocity = cube._bin_velocity
            
            if bin_num.ndim == 1:
                # Reshape 1D to 2D
                bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
            else:
                bin_num_2d = bin_num
            
            vel_map = np.full_like(bin_num_2d, np.nan, dtype=float)
            for i in range(len(velocity)):
                if i < len(velocity):
                    vel_map[bin_num_2d == i] = velocity[i]
            
            # Calculate limits for symmetric colormap
            valid_vel = vel_map[np.isfinite(vel_map)]
            if len(valid_vel) > 0:
                vmax = np.nanpercentile(np.abs(valid_vel), 95)
                vmin = -vmax
            else:
                vmin, vmax = -100, 100
            
            # Use proper non-square pixels with physical scale
            if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
                pixel_size_x = cube._pxl_size_x
                pixel_size_y = cube._pxl_size_y
                
                # Calculate extent
                extent = [
                    -cube._n_x/2 * pixel_size_x, 
                    cube._n_x/2 * pixel_size_x, 
                    -cube._n_y/2 * pixel_size_y, 
                    cube._n_y/2 * pixel_size_y
                ]
                
                im4 = ax4.imshow(
                    vel_map, 
                    origin='lower', 
                    cmap='coolwarm',
                    vmin=vmin, vmax=vmax,
                    extent=extent,
                    aspect=1.0  # aspect=1.0 for equal physical scaling
                )
                
                ax4.set_xlabel('Δ RA (arcsec)')
                ax4.set_ylabel('Δ Dec (arcsec)')
            else:
                im4 = ax4.imshow(
                    vel_map, 
                    origin='lower', 
                    cmap='coolwarm',
                    vmin=vmin, vmax=vmax
                )
                ax4.set_aspect('equal')  # Equal aspect ratio for pixel coordinates
                
                ax4.set_xlabel('Pixels')
                ax4.set_ylabel('Pixels')
                
            plt.colorbar(im4, ax=ax4, label='Velocity (km/s)')
            ax4.set_title('Velocity Field')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error creating velocity field:\n{str(e)}', 
                    ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'Velocity field not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Dispersion field (if available) (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])
    if hasattr(cube, '_bin_dispersion') and hasattr(cube, '_bin_num'):
        try:
            # Create 2D dispersion field
            bin_num = cube._bin_num
            dispersion = cube._bin_dispersion
            
            if bin_num.ndim == 1:
                # Reshape 1D to 2D
                bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
            else:
                bin_num_2d = bin_num
            
            disp_map = np.full_like(bin_num_2d, np.nan, dtype=float)
            for i in range(len(dispersion)):
                if i < len(dispersion):
                    disp_map[bin_num_2d == i] = dispersion[i]
            
            # Use proper non-square pixels with physical scale
            if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
                pixel_size_x = cube._pxl_size_x
                pixel_size_y = cube._pxl_size_y
                
                # Calculate extent
                extent = [
                    -cube._n_x/2 * pixel_size_x, 
                    cube._n_x/2 * pixel_size_x, 
                    -cube._n_y/2 * pixel_size_y, 
                    cube._n_y/2 * pixel_size_y
                ]
                
                # Plot
                valid_disp = disp_map[np.isfinite(disp_map)]
                if len(valid_disp) > 0:
                    vmin = np.nanpercentile(valid_disp, 5)
                    vmax = np.nanpercentile(valid_disp, 95)
                else:
                    vmin, vmax = 0, 100
                    
                im5 = ax5.imshow(
                    disp_map, 
                    origin='lower', 
                    cmap='viridis',
                    vmin=vmin, vmax=vmax,
                    extent=extent,
                    aspect=1.0  # aspect=1.0 for equal physical scaling
                )
                
                ax5.set_xlabel('Δ RA (arcsec)')
                ax5.set_ylabel('Δ Dec (arcsec)')
            else:
                # Plot with pixel coordinates
                valid_disp = disp_map[np.isfinite(disp_map)]
                if len(valid_disp) > 0:
                    vmin = np.nanpercentile(valid_disp, 5)
                    vmax = np.nanpercentile(valid_disp, 95)
                else:
                    vmin, vmax = 0, 100
                    
                im5 = ax5.imshow(
                    disp_map, 
                    origin='lower', 
                    cmap='viridis',
                    vmin=vmin, vmax=vmax
                )
                ax5.set_aspect('equal')  # Equal aspect ratio for pixel coordinates
                
                ax5.set_xlabel('Pixels')
                ax5.set_ylabel('Pixels')
                
            plt.colorbar(im5, ax=ax5, label='Dispersion (km/s)')
            ax5.set_title('Velocity Dispersion')
        except Exception as e:
            ax5.text(0.5, 0.5, f'Error creating dispersion field:\n{str(e)}', 
                    ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'Dispersion field not available', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Information and statistics panel (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')  # No axes needed for text
    
    # Gather information about the cube and analysis
    info_text = []
    
    # Basic cube properties
    if hasattr(cube, '_filename'):
        info_text.append(f"Filename: {os.path.basename(cube._filename)}")
    
    if hasattr(cube, '_n_x') and hasattr(cube, '_n_y'):
        info_text.append(f"Dimensions: {cube._n_x} × {cube._n_y} pixels")
    
    if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
        info_text.append(f"Pixel scale: {cube._pxl_size_x:.3f} × {cube._pxl_size_y:.3f} arcsec")
    
    if hasattr(cube, '_redshift'):
        info_text.append(f"Redshift: {cube._redshift:.4f}")
    
    # Wavelength range
    if hasattr(cube, '_lambda_gal'):
        info_text.append(f"Wavelength range: {np.min(cube._lambda_gal):.1f} - {np.max(cube._lambda_gal):.1f} Å")
    
    # Binning information
    if hasattr(cube, '_bin_num'):
        n_bins = np.max(cube._bin_num) + 1 if np.any(cube._bin_num >= 0) else 0
        info_text.append(f"Number of bins: {n_bins}")
    
    # Quality metrics
    if hasattr(cube, '_bin_velocity') and hasattr(cube, '_bin_dispersion'):
        # Calculate V/σ
        try:
            velocity = np.asarray(cube._bin_velocity)
            dispersion = np.asarray(cube._bin_dispersion)
            valid = np.isfinite(velocity) & np.isfinite(dispersion) & (dispersion > 0)
            if np.any(valid):
                v_sigma = np.abs(velocity[valid]) / dispersion[valid]
                info_text.append(f"Median V/σ: {np.median(v_sigma):.2f}")
                info_text.append(f"Max V/σ: {np.max(v_sigma):.2f}")
        except:
            pass
    
    # Add gathered information as text
    if info_text:
        ax6.text(0.05, 0.95, '\n'.join(info_text), 
                va='top', ha='left', transform=ax6.transAxes,
                fontsize=10)
    else:
        ax6.text(0.5, 0.5, 'No cube information available', 
                ha='center', va='center', transform=ax6.transAxes)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    # Apply tight layout safely
    safe_tight_layout(fig)
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if title:
            filename = f"{title.replace(' ', '_')}_diagnostic.png"
        else:
            filename = "cube_diagnostic.png"
        
        standardize_figure_saving(fig, output_dir / filename, dpi=150)
    
    return fig


def plot_spatial_map(data, ax=None, title=None, cmap='viridis', physical_scale=True, 
                   pixel_size=None, wcs=None, colorbar_label=None, vmin=None, vmax=None, 
                   log_scale=False, cube=None):
    """
    General purpose function to plot spatial data with proper physical scaling
    
    Parameters
    ----------
    data : numpy.ndarray
        2D data array to plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, creates a new one if None
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap name
    physical_scale : bool, default=True
        Use physical coordinates
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation
    colorbar_label : str, optional
        Label for colorbar
    vmin, vmax : float, optional
        Value range for colormap
    log_scale : bool, default=False
        Use logarithmic scale for values
    cube : MUSECube, optional
        MUSE cube object for determining figure size
        
    Returns
    -------
    fig, ax, im : tuple
        Figure, axis and image objects
    """
    # Create axis if needed
    if ax is None:
        # Get appropriate figure size if cube is provided
        if cube is not None:
            figsize = get_figure_size_for_cube(cube)
        else:
            figsize = (8, 8)
            
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Get dimensions
    ny, nx = data.shape
    
    # If pixel_size not provided but cube is, use cube's pixel size
    if pixel_size is None and cube is not None:
        if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
            pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
    
    # Process pixel size if provided as a single value
    if isinstance(pixel_size, (int, float)):
        pixel_size = (pixel_size, pixel_size)
    
    # Handle NaN values in data
    masked_data = np.ma.array(data, mask=~np.isfinite(data))
    
    # Determine color normalization
    valid_data = masked_data.compressed()
    if len(valid_data) > 0:
        if vmin is None:
            vmin = np.nanmin(valid_data)
        if vmax is None:
            vmax = np.nanmax(valid_data)
            
        if log_scale and np.all(valid_data > 0):
            norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None
        
    # Try WCS plotting first
    wcs_obj = process_wcs(wcs) if wcs is not None else None
    
    if wcs_obj is not None and physical_scale:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Get current figure
            fig = ax.figure
            
            # Create new axis with WCS projection if needed
            if not isinstance(ax, WCSAxes):
                position = ax.get_position()
                ax.remove()
                ax = fig.add_subplot(111, projection=wcs_obj)
                ax.set_position(position)
            
            # Plot the data with WCS coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm)
            
            # Add coordinate grid
            ax.grid(color='white', ls='solid', alpha=0.3)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            
            # Ensure aspect ratio is correct for WCS
            ax.set_aspect('equal')
        except Exception as e:
            logger.warning(f"Error plotting with WCS: {e}")
            wcs_obj = None
    
    # Use physical coordinates if WCS not available or failed
    if wcs_obj is None:
        if physical_scale and pixel_size is not None:
            # Calculate physical extent
            pixel_size_x, pixel_size_y = pixel_size
            extent = [
                -nx/2 * pixel_size_x, 
                nx/2 * pixel_size_x, 
                -ny/2 * pixel_size_y, 
                ny/2 * pixel_size_y
            ]
            
            # Plot with physical coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm, 
                         extent=extent, aspect=1.0)  # aspect=1.0 for equal physical scaling
            
            # Set axis labels
            ax.set_xlabel('Δ RA (arcsec)')
            ax.set_ylabel('Δ Dec (arcsec)')
        else:
            # Plot with pixel coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm)
            ax.set_aspect('equal')  # Equal aspect ratio for pixels
            
            # Set axis labels
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig, ax, im


def plot_kinematic_maps_with_errors(velocity_field, dispersion_field, 
                                   velocity_error=None, dispersion_error=None,
                                   equal_aspect=True, physical_scale=True, 
                                   pixel_size=None, wcs=None, cube=None,
                                   figsize=None, title=None):
    """
    Plot kinematic maps with error visualization
    
    Parameters
    ----------
    velocity_field : numpy.ndarray
        2D velocity field
    dispersion_field : numpy.ndarray
        2D dispersion field
    velocity_error : numpy.ndarray, optional
        2D velocity error field
    dispersion_error : numpy.ndarray, optional
        2D dispersion error field
    equal_aspect : bool, default=True
        Use equal aspect ratio
    physical_scale : bool, default=True
        Use physical coordinates
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    wcs : astropy.wcs.WCS, optional
        WCS object
    cube : MUSECube, optional
        MUSE cube for determining figure size
    figsize : tuple, optional
        Figure size, auto-determined if None
    title : str, optional
        Figure title
        
    Returns
    -------
    fig, axes : tuple
        Figure and axes array
    """
    # Determine figure layout based on available error data
    has_errors = velocity_error is not None and dispersion_error is not None
    
    if has_errors:
        n_cols = 3
        n_rows = 2
        if figsize is None:
            if cube is not None:
                map_size = get_figure_size_for_cube(cube)
                figsize = (map_size[0] * 3, map_size[1] * 2)
            else:
                figsize = (15, 10)
    else:
        n_cols = 2
        n_rows = 1
        if figsize is None:
            if cube is not None:
                map_size = get_figure_size_for_cube(cube)
                figsize = (map_size[0] * 2, map_size[1])
            else:
                figsize = (12, 5)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)
    
    # Get dimensions
    ny, nx = velocity_field.shape
    
    # Get pixel size from cube if not provided
    if pixel_size is None and cube is not None:
        if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
            pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
    
    # Calculate physical extent if needed
    extent = None
    if physical_scale and pixel_size is not None:
        pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
        extent = [
            -nx/2 * pixel_size_x, 
            nx/2 * pixel_size_x, 
            -ny/2 * pixel_size_y, 
            ny/2 * pixel_size_y
        ]
    
    # Velocity field
    vel_valid = np.isfinite(velocity_field)
    if np.any(vel_valid):
        vmax = np.percentile(np.abs(velocity_field[vel_valid]), 95)
        vmin = -vmax
        
        im1 = axes[0, 0].imshow(velocity_field, origin='lower', cmap='coolwarm',
                               vmin=vmin, vmax=vmax, extent=extent if extent else None,
                               aspect=1.0 if extent else 'equal')
        plt.colorbar(im1, ax=axes[0, 0], label='Velocity (km/s)')
        axes[0, 0].set_title('Velocity Field')
        
        if has_errors:
            # Velocity error
            vel_err_valid = np.isfinite(velocity_error)
            if np.any(vel_err_valid):
                err_range = np.percentile(velocity_error[vel_err_valid], [5, 95])
                im2 = axes[0, 1].imshow(velocity_error, origin='lower', cmap='plasma',
                                       vmin=err_range[0], vmax=err_range[1], 
                                       extent=extent if extent else None,
                                       aspect=1.0 if extent else 'equal')
                plt.colorbar(im2, ax=axes[0, 1], label='Error (km/s)')
                axes[0, 1].set_title('Velocity Error')
                
                # Velocity S/N
                vel_snr = np.abs(velocity_field) / velocity_error
                snr_valid = np.isfinite(vel_snr) & (velocity_error > 0)
                if np.any(snr_valid):
                    snr_range = np.percentile(vel_snr[snr_valid], [5, 95])
                    im3 = axes[0, 2].imshow(vel_snr, origin='lower', cmap='viridis',
                                           vmin=max(0.1, snr_range[0]), vmax=snr_range[1],
                                           extent=extent if extent else None,
                                           aspect=1.0 if extent else 'equal')
                    plt.colorbar(im3, ax=axes[0, 2], label='S/N')
                    axes[0, 2].set_title('Velocity S/N')
    
    # Dispersion field
    disp_valid = np.isfinite(dispersion_field) & (dispersion_field > 0)
    if np.any(disp_valid):
        disp_range = np.percentile(dispersion_field[disp_valid], [5, 95])
        
        row_idx = 1 if has_errors else 0
        col_idx = 0 if has_errors else 1
        
        im4 = axes[row_idx, col_idx].imshow(dispersion_field, origin='lower', cmap='inferno',
                                           vmin=disp_range[0], vmax=disp_range[1],
                                           extent=extent if extent else None,
                                           aspect=1.0 if extent else 'equal')
        plt.colorbar(im4, ax=axes[row_idx, col_idx], label='Dispersion (km/s)')
        axes[row_idx, col_idx].set_title('Velocity Dispersion')
        
        if has_errors:
            # Dispersion error
            disp_err_valid = np.isfinite(dispersion_error)
            if np.any(disp_err_valid):
                err_range = np.percentile(dispersion_error[disp_err_valid], [5, 95])
                im5 = axes[1, 1].imshow(dispersion_error, origin='lower', cmap='plasma',
                                       vmin=err_range[0], vmax=err_range[1],
                                       extent=extent if extent else None,
                                       aspect=1.0 if extent else 'equal')
                plt.colorbar(im5, ax=axes[1, 1], label='Error (km/s)')
                axes[1, 1].set_title('Dispersion Error')
                
                # Dispersion S/N
                disp_snr = dispersion_field / dispersion_error
                snr_valid = np.isfinite(disp_snr) & (dispersion_error > 0)
                if np.any(snr_valid):
                    snr_range = np.percentile(disp_snr[snr_valid], [5, 95])
                    im6 = axes[1, 2].imshow(disp_snr, origin='lower', cmap='viridis',
                                           vmin=max(0.1, snr_range[0]), vmax=snr_range[1],
                                           extent=extent if extent else None,
                                           aspect=1.0 if extent else 'equal')
                    plt.colorbar(im6, ax=axes[1, 2], label='S/N')
                    axes[1, 2].set_title('Dispersion S/N')
    
    # Set axis labels
    for ax in axes.flat:
        if extent:
            ax.set_xlabel('Δ RA (arcsec)')
            ax.set_ylabel('Δ Dec (arcsec)')
        else:
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
        ax.grid(True, alpha=0.3)
    
    safe_tight_layout(fig)
    
    return fig, axes


def plot_spectral_indices_with_errors(indices_dict, errors_dict=None, bin_map=None,
                                     indices_2d=None, errors_2d=None,
                                     figsize=(16, 12), title=None, save_path=None):
    """
    Plot spectral indices with error visualization
    
    Parameters
    ----------
    indices_dict : dict
        Dictionary of index_name -> values (1D array for bins)
    errors_dict : dict, optional
        Dictionary of index_name -> errors (1D array for bins)
    bin_map : numpy.ndarray, optional
        2D bin map for converting 1D to 2D
    indices_2d : dict, optional
        Dictionary of index_name -> 2D arrays (alternative to indices_dict + bin_map)
    errors_2d : dict, optional
        Dictionary of index_name -> 2D error arrays
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    # Determine data source
    if indices_2d is not None:
        # Use 2D data directly
        index_names = list(indices_2d.keys())
        is_2d = True
    elif indices_dict is not None and bin_map is not None:
        # Convert 1D to 2D using bin map
        index_names = list(indices_dict.keys())
        is_2d = False
    else:
        logger.error("Must provide either indices_2d or (indices_dict + bin_map)")
        return None
    
    n_indices = len(index_names)
    if n_indices == 0:
        logger.warning("No spectral indices to plot")
        return None
    
    # Determine layout
    has_errors = (errors_dict is not None) or (errors_2d is not None)
    n_cols = 3 if has_errors else 1
    n_rows = n_indices
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)
    
    # Column titles
    if has_errors:
        axes[0, 0].set_title('Index Value', fontsize=14, pad=10)
        axes[0, 1].set_title('Index Error', fontsize=14, pad=10)
        axes[0, 2].set_title('Index S/N', fontsize=14, pad=10)
    
    for i, index_name in enumerate(index_names):
        # Get values
        if is_2d:
            values_2d = indices_2d[index_name]
            errors_2d_idx = errors_2d.get(index_name) if errors_2d else None
        else:
            # Convert 1D to 2D
            values_2d = np.full_like(bin_map, np.nan, dtype=float)
            values_1d = indices_dict[index_name]
            
            for bin_idx in range(len(values_1d)):
                if np.isfinite(values_1d[bin_idx]):
                    values_2d[bin_map == bin_idx] = values_1d[bin_idx]
            
            if has_errors and errors_dict and index_name in errors_dict:
                errors_2d_idx = np.full_like(bin_map, np.nan, dtype=float)
                errors_1d = errors_dict[index_name]
                
                for bin_idx in range(len(errors_1d)):
                    if np.isfinite(errors_1d[bin_idx]):
                        errors_2d_idx[bin_map == bin_idx] = errors_1d[bin_idx]
            else:
                errors_2d_idx = None
        
        # Plot value map
        valid = np.isfinite(values_2d)
        if np.any(valid):
            val_range = np.percentile(values_2d[valid], [2, 98])
            im1 = axes[i, 0].imshow(values_2d, origin='lower', cmap='plasma',
                                   vmin=val_range[0], vmax=val_range[1])
            plt.colorbar(im1, ax=axes[i, 0], label='Å')
            
            if has_errors and errors_2d_idx is not None:
                # Plot error map
                err_valid = np.isfinite(errors_2d_idx) & (errors_2d_idx > 0)
                if np.any(err_valid):
                    err_range = np.percentile(errors_2d_idx[err_valid], [5, 95])
                    im2 = axes[i, 1].imshow(errors_2d_idx, origin='lower', cmap='viridis',
                                           vmin=err_range[0], vmax=err_range[1])
                    plt.colorbar(im2, ax=axes[i, 1], label='Å')
                    
                    # Plot S/N map
                    snr = np.abs(values_2d) / errors_2d_idx
                    snr_valid = np.isfinite(snr) & err_valid
                    if np.any(snr_valid):
                        snr_range = np.percentile(snr[snr_valid], [5, 95])
                        im3 = axes[i, 2].imshow(snr, origin='lower', cmap='inferno',
                                               vmin=max(0.1, snr_range[0]), vmax=snr_range[1])
                        plt.colorbar(im3, ax=axes[i, 2], label='S/N')
        
        # Set row label
        axes[i, 0].set_ylabel(index_name, fontsize=12)
    
    # Remove ticks for cleaner appearance
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    safe_tight_layout(fig)
    
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig


def plot_radial_profiles_with_errors(radii, profiles_dict, errors_dict=None,
                                    figsize=(12, 8), title=None, save_path=None,
                                    xlabel='Radius (arcsec)', log_x=False, log_y=False):
    """
    Plot radial profiles with error bars
    
    Parameters
    ----------
    radii : numpy.ndarray
        Radial coordinates
    profiles_dict : dict
        Dictionary of profile_name -> values
    errors_dict : dict, optional
        Dictionary of profile_name -> errors
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save figure
    xlabel : str
        X-axis label
    log_x : bool
        Use log scale for x-axis
    log_y : bool or dict
        Use log scale for y-axis (can be dict with profile names as keys)
        
    Returns
    -------
    fig, axes : tuple
        Figure and axes array
    """
    profile_names = list(profiles_dict.keys())
    n_profiles = len(profile_names)
    
    if n_profiles == 0:
        logger.warning("No profiles to plot")
        return None, None
    
    # Determine layout
    if n_profiles <= 2:
        nrows, ncols = 1, n_profiles
    elif n_profiles <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = int(np.ceil(n_profiles / 3))
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)
    
    for i, profile_name in enumerate(profile_names):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        values = profiles_dict[profile_name]
        valid = np.isfinite(values) & np.isfinite(radii)
        
        if np.any(valid):
            # Sort by radius for cleaner lines
            idx_sort = np.argsort(radii[valid])
            r_sorted = radii[valid][idx_sort]
            v_sorted = values[valid][idx_sort]
            
            if errors_dict is not None and profile_name in errors_dict:
                # Plot with error bars
                errors = errors_dict[profile_name]
                e_sorted = errors[valid][idx_sort]
                
                # Filter out zero or negative errors
                valid_err = np.isfinite(e_sorted) & (e_sorted > 0)
                
                if np.any(valid_err):
                    ax.errorbar(r_sorted[valid_err], v_sorted[valid_err], 
                               yerr=e_sorted[valid_err],
                               fmt='o-', capsize=3, capthick=1, markersize=4,
                               label=profile_name, alpha=0.8)
                    
                    # Add shaded error region
                    ax.fill_between(r_sorted[valid_err], 
                                   v_sorted[valid_err] - e_sorted[valid_err],
                                   v_sorted[valid_err] + e_sorted[valid_err],
                                   alpha=0.2)
                else:
                    ax.plot(r_sorted, v_sorted, 'o-', markersize=4, label=profile_name)
            else:
                # Plot without error bars
                ax.plot(r_sorted, v_sorted, 'o-', markersize=4, label=profile_name)
        
        # Set scales
        if log_x:
            ax.set_xscale('log')
            
        if isinstance(log_y, dict):
            if profile_name in log_y and log_y[profile_name]:
                ax.set_yscale('log')
        elif log_y:
            ax.set_yscale('log')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(profile_name)
        ax.grid(True, alpha=0.3)
        
        # Add zero line if appropriate
        if ax.get_ylim()[0] < 0 < ax.get_ylim()[1]:
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_profiles, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    safe_tight_layout(fig)
    
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig, axes


def plot_error_diagnostic(velocity_error, dispersion_error, velocity=None, dispersion=None,
                         figsize=(12, 8), title="Kinematic Error Analysis"):
    """
    Create diagnostic plots for error analysis
    
    Parameters
    ----------
    velocity_error : numpy.ndarray
        2D velocity error array
    dispersion_error : numpy.ndarray
        2D dispersion error array
    velocity : numpy.ndarray, optional
        2D velocity array for comparison
    dispersion : numpy.ndarray, optional
        2D dispersion array for comparison
    figsize : tuple
        Figure size
    title : str
        Figure title
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Error distributions
    ax1 = fig.add_subplot(gs[0, 0])
    vel_err_valid = velocity_error[np.isfinite(velocity_error)]
    if len(vel_err_valid) > 0:
        ax1.hist(vel_err_valid, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.median(vel_err_valid), color='red', linestyle='--', 
                   label=f'Median: {np.median(vel_err_valid):.1f}')
        ax1.set_xlabel('Velocity Error (km/s)')
        ax1.set_ylabel('Count')
        ax1.set_title('Velocity Error Distribution')
        ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    disp_err_valid = dispersion_error[np.isfinite(dispersion_error)]
    if len(disp_err_valid) > 0:
        ax2.hist(disp_err_valid, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(np.median(disp_err_valid), color='red', linestyle='--',
                   label=f'Median: {np.median(disp_err_valid):.1f}')
        ax2.set_xlabel('Dispersion Error (km/s)')
        ax2.set_ylabel('Count')
        ax2.set_title('Dispersion Error Distribution')
        ax2.legend()
    
    # Error vs value scatter plots
    if velocity is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        valid = np.isfinite(velocity) & np.isfinite(velocity_error)
        if np.any(valid):
            ax3.scatter(np.abs(velocity[valid]), velocity_error[valid], 
                       alpha=0.5, s=1, c='blue')
            ax3.set_xlabel('|Velocity| (km/s)')
            ax3.set_ylabel('Velocity Error (km/s)')
            ax3.set_title('Velocity Error vs |Velocity|')
            ax3.grid(True, alpha=0.3)
    
    if dispersion is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        valid = np.isfinite(dispersion) & np.isfinite(dispersion_error)
        if np.any(valid):
            ax4.scatter(dispersion[valid], dispersion_error[valid],
                       alpha=0.5, s=1, c='orange')
            ax4.set_xlabel('Dispersion (km/s)')
            ax4.set_ylabel('Dispersion Error (km/s)')
            ax4.set_title('Dispersion Error vs Dispersion')
            ax4.grid(True, alpha=0.3)
    
    # S/N distributions
    if velocity is not None and dispersion is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        vel_snr = np.abs(velocity) / velocity_error
        disp_snr = dispersion / dispersion_error
        
        vel_snr_valid = vel_snr[np.isfinite(vel_snr)]
        disp_snr_valid = disp_snr[np.isfinite(disp_snr)]
        
        if len(vel_snr_valid) > 0:
            ax5.hist(vel_snr_valid, bins=50, alpha=0.5, color='blue', 
                    label='Velocity S/N', density=True)
        if len(disp_snr_valid) > 0:
            ax5.hist(disp_snr_valid, bins=50, alpha=0.5, color='orange',
                    label='Dispersion S/N', density=True)
        
        ax5.set_xlabel('S/N Ratio')
        ax5.set_ylabel('Normalized Count')
        ax5.set_title('S/N Distributions')
        ax5.legend()
        ax5.set_xlim(0, np.percentile(np.concatenate([vel_snr_valid, disp_snr_valid]), 95))
    
    # Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = []
    stats_text.append("Error Statistics:")
    stats_text.append("")
    stats_text.append(f"Velocity Error:")
    stats_text.append(f"  Median: {np.nanmedian(velocity_error):.1f} km/s")
    stats_text.append(f"  Mean: {np.nanmean(velocity_error):.1f} km/s")
    stats_text.append(f"  90th percentile: {np.nanpercentile(velocity_error, 90):.1f} km/s")
    stats_text.append("")
    stats_text.append(f"Dispersion Error:")
    stats_text.append(f"  Median: {np.nanmedian(dispersion_error):.1f} km/s")
    stats_text.append(f"  Mean: {np.nanmean(dispersion_error):.1f} km/s")
    stats_text.append(f"  90th percentile: {np.nanpercentile(dispersion_error, 90):.1f} km/s")
    
    if velocity is not None and dispersion is not None:
        vel_snr_median = np.nanmedian(np.abs(velocity) / velocity_error)
        disp_snr_median = np.nanmedian(dispersion / dispersion_error)
        stats_text.append("")
        stats_text.append(f"Median S/N:")
        stats_text.append(f"  Velocity: {vel_snr_median:.1f}")
        stats_text.append(f"  Dispersion: {disp_snr_median:.1f}")
    
    ax6.text(0.1, 0.9, '\n'.join(stats_text), transform=ax6.transAxes,
            verticalalignment='top', fontsize=10, family='monospace')
    
    fig.suptitle(title, fontsize=14)
    safe_tight_layout(fig)
    
    return fig


def plot_monte_carlo_corner(samples, parameter_names, truths=None, 
                           figsize=(12, 12), title=None, save_path=None):
    """
    Create corner plot for Monte Carlo samples
    
    Parameters
    ----------
    samples : numpy.ndarray
        Array of samples (n_samples, n_parameters)
    parameter_names : list
        List of parameter names
    truths : array-like, optional
        True values to mark on plots
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : Figure
        Corner plot figure
    """
    try:
        import corner
    except ImportError:
        logger.error("corner package required for corner plots. Install with: pip install corner")
        return None
    
    fig = corner.corner(samples, labels=parameter_names, truths=truths,
                       quantiles=[0.16, 0.5, 0.84], show_titles=True,
                       title_kwargs={"fontsize": 12}, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig

def plot_error_ellipse(ax, center, cov_matrix, n_std=1.0, **kwargs):
    """
    Plot error ellipse for 2D Gaussian distribution
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    center : array-like
        (x, y) center of ellipse
    cov_matrix : numpy.ndarray
        2x2 covariance matrix
    n_std : float, default=1.0
        Number of standard deviations (1, 2, or 3 for 68%, 95%, 99.7%)
    **kwargs : dict
        Additional arguments for Ellipse patch
        
    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        Ellipse patch object
    """
    from matplotlib.patches import Ellipse
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Calculate angle
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Width and height are 2 * sqrt(eigenvalue) * n_std
    width = 2 * np.sqrt(eigenvalues[0]) * n_std
    height = 2 * np.sqrt(eigenvalues[1]) * n_std
    
    # Default ellipse properties
    default_kwargs = {
        'fill': False,
        'edgecolor': 'red',
        'linewidth': 2,
        'linestyle': '-',
        'alpha': 0.8
    }
    default_kwargs.update(kwargs)
    
    # Create and add ellipse
    ellipse = Ellipse(center, width, height, angle, **default_kwargs)
    ax.add_patch(ellipse)
    
    return ellipse


def plot_confidence_bands(ax, x, y, y_lower, y_upper, label=None, color='blue', alpha=0.3):
    """
    Plot line with confidence bands
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    x : numpy.ndarray
        X coordinates
    y : numpy.ndarray
        Y values (central estimate)
    y_lower : numpy.ndarray
        Lower confidence bound
    y_upper : numpy.ndarray
        Upper confidence bound
    label : str, optional
        Label for legend
    color : str, default='blue'
        Color for line and band
    alpha : float, default=0.3
        Transparency for confidence band
        
    Returns
    -------
    line : matplotlib.lines.Line2D
        Line object
    band : matplotlib.collections.PolyCollection
        Confidence band object
    """
    # Plot central line
    line = ax.plot(x, y, color=color, label=label, linewidth=2)[0]
    
    # Plot confidence band
    band = ax.fill_between(x, y_lower, y_upper, color=color, alpha=alpha)
    
    return line, band


def plot_correlation_matrix(correlation_matrix, labels=None, ax=None, cmap='RdBu_r',
                          vmin=-1, vmax=1, annotate=True, fmt='.2f'):
    """
    Plot correlation matrix as heatmap
    
    Parameters
    ----------
    correlation_matrix : numpy.ndarray
        Correlation matrix to plot
    labels : list, optional
        Labels for axes
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    cmap : str, default='RdBu_r'
        Colormap
    vmin, vmax : float
        Color scale limits
    annotate : bool, default=True
        Add correlation values as text
    fmt : str, default='.2f'
        Format string for annotations
        
    Returns
    -------
    im : matplotlib.image.AxesImage
        Image object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot correlation matrix
    im = ax.imshow(correlation_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Set ticks and labels
    n = correlation_matrix.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    
    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    # Add correlation values
    if annotate:
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:{fmt}}',
                             ha='center', va='center',
                             color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    
    ax.set_title('Correlation Matrix')
    
    return im


def plot_parameter_distributions(params_dict, errors_dict=None, figsize=(12, 8),
                               title=None, save_path=None):
    """
    Plot parameter distributions with error estimates
    
    Parameters
    ----------
    params_dict : dict
        Dictionary of parameter_name -> values
    errors_dict : dict, optional
        Dictionary of parameter_name -> errors
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig, axes : tuple
        Figure and axes array
    """
    param_names = list(params_dict.keys())
    n_params = len(param_names)
    
    if n_params == 0:
        logger.warning("No parameters to plot")
        return None, None
    
    # Determine layout
    ncols = min(3, n_params)
    nrows = int(np.ceil(n_params / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    for i, param_name in enumerate(param_names):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        values = params_dict[param_name]
        valid = np.isfinite(values)
        
        if np.any(valid):
            # Plot histogram
            n, bins, patches = ax.hist(values[valid], bins=30, alpha=0.7, 
                                     edgecolor='black', density=True)
            
            # Add statistics
            mean_val = np.mean(values[valid])
            median_val = np.median(values[valid])
            std_val = np.std(values[valid])
            
            # Plot mean and median
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--',
                      label=f'Median: {median_val:.2f}')
            
            # Add error band if available
            if errors_dict and param_name in errors_dict:
                errors = errors_dict[param_name]
                if np.any(np.isfinite(errors)):
                    mean_error = np.mean(errors[np.isfinite(errors)])
                    ax.axvspan(mean_val - mean_error, mean_val + mean_error,
                             alpha=0.2, color='red', label=f'±Error: {mean_error:.2f}')
            
            # Add normal distribution overlay
            x = np.linspace(bins[0], bins[-1], 100)
            ax.plot(x, stats.norm.pdf(x, mean_val, std_val), 'k-', 
                   linewidth=2, label=f'σ: {std_val:.2f}')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Density')
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    safe_tight_layout(fig)
    
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig, axes


def plot_bootstrap_results(bootstrap_samples, parameter_names, true_values=None,
                         figsize=(12, 10), title=None, save_path=None):
    """
    Plot bootstrap analysis results
    
    Parameters
    ----------
    bootstrap_samples : numpy.ndarray
        Array of bootstrap samples (n_samples, n_parameters)
    parameter_names : list
        List of parameter names
    true_values : array-like, optional
        True parameter values to mark
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    n_params = len(parameter_names)
    
    fig, axes = plt.subplots(n_params, 2, figsize=figsize)
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    for i, param_name in enumerate(parameter_names):
        # Distribution plot
        ax_dist = axes[i, 0]
        data = bootstrap_samples[:, i]
        
        # Histogram
        n, bins, _ = ax_dist.hist(data, bins=50, alpha=0.7, density=True, 
                                edgecolor='black')
        
        # Calculate statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        percentiles = np.percentile(data, [16, 84])  # 68% CI
        
        # Plot statistics
        ax_dist.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.3f}')
        ax_dist.axvline(median_val, color='green', linestyle='--',
                       label=f'Median: {median_val:.3f}')
        ax_dist.axvspan(percentiles[0], percentiles[1], alpha=0.2, 
                       color='gray', label='68% CI')
        
        # True value if provided
        if true_values is not None:
            ax_dist.axvline(true_values[i], color='black', linestyle='-',
                          linewidth=2, label=f'True: {true_values[i]:.3f}')
        
        ax_dist.set_xlabel(param_name)
        ax_dist.set_ylabel('Density')
        ax_dist.legend(fontsize='small')
        ax_dist.grid(True, alpha=0.3)
        
        # Convergence plot
        ax_conv = axes[i, 1]
        n_samples = len(data)
        sample_sizes = np.logspace(1, np.log10(n_samples), 50).astype(int)
        
        means = []
        stds = []
        for n in sample_sizes:
            subsample = data[:n]
            means.append(np.mean(subsample))
            stds.append(np.std(subsample))
        
        # Plot convergence
        ax_conv.plot(sample_sizes, means, 'b-', label='Mean', linewidth=2)
        ax_conv.fill_between(sample_sizes, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3, color='blue')
        
        if true_values is not None:
            ax_conv.axhline(true_values[i], color='black', linestyle='--',
                          label='True value')
        
        ax_conv.set_xscale('log')
        ax_conv.set_xlabel('Number of samples')
        ax_conv.set_ylabel(param_name)
        ax_conv.set_title('Bootstrap Convergence')
        ax_conv.legend()
        ax_conv.grid(True, alpha=0.3)
    
    safe_tight_layout(fig)
    
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig


def plot_mcmc_chains(chains, parameter_names, burn_in=None, figsize=(12, 10),
                    title=None, save_path=None):
    """
    Plot MCMC chains for diagnostics
    
    Parameters
    ----------
    chains : numpy.ndarray
        MCMC chains (n_steps, n_walkers, n_parameters)
    parameter_names : list
        List of parameter names
    burn_in : int, optional
        Number of burn-in steps to mark
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    n_steps, n_walkers, n_params = chains.shape
    
    fig, axes = plt.subplots(n_params, 2, figsize=figsize, 
                           gridspec_kw={'width_ratios': [3, 1]})
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    for i, param_name in enumerate(parameter_names):
        # Chain plot
        ax_chain = axes[i, 0]
        
        # Plot each walker
        for j in range(n_walkers):
            ax_chain.plot(chains[:, j, i], alpha=0.5, linewidth=0.5)
        
        # Mark burn-in
        if burn_in is not None:
            ax_chain.axvline(burn_in, color='red', linestyle='--', 
                           label='Burn-in')
        
        ax_chain.set_ylabel(param_name)
        if i == n_params - 1:
            ax_chain.set_xlabel('Step')
        ax_chain.grid(True, alpha=0.3)
        
        if i == 0 and burn_in is not None:
            ax_chain.legend()
        
        # Posterior distribution
        ax_post = axes[i, 1]
        
        # Use samples after burn-in
        start_idx = burn_in if burn_in is not None else 0
        samples = chains[start_idx:, :, i].flatten()
        
        ax_post.hist(samples, bins=50, orientation='horizontal', 
                    alpha=0.7, density=True)
        
        # Add statistics
        mean_val = np.mean(samples)
        percentiles = np.percentile(samples, [16, 50, 84])
        
        ax_post.axhline(mean_val, color='red', linestyle='--')
        ax_post.axhline(percentiles[1], color='green', linestyle='--')
        ax_post.axhspan(percentiles[0], percentiles[2], alpha=0.2, color='gray')
        
        ax_post.set_ylim(ax_chain.get_ylim())
        ax_post.set_xlabel('Density')
        ax_post.grid(True, alpha=0.3)
    
    safe_tight_layout(fig)
    
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig


# Enhance existing safe_plot_array function to include S/N option
# This replaces the existing safe_plot_array function

def safe_plot_array(values, bin_map, ax=None, title=None, cmap='viridis', label=None, 
                   vmin=None, vmax=None, errors=None, show_snr=False, snr_threshold=None):
    """
    Safely plot values mapped onto bins, handling non-numeric data types with error support
    
    Parameters
    ----------
    values : array-like
        Values for each bin
    bin_map : numpy.ndarray
        2D array of bin numbers
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap name
    label : str, optional
        Colorbar label
    vmin, vmax : float, optional
        Value range limits
    errors : array-like, optional
        Error values for each bin
    show_snr : bool, default=False
        Show S/N ratio instead of values
    snr_threshold : float, optional
        Mark regions below this S/N threshold
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    # Convert bin_map to integer type for mapping
    bin_map_int = np.asarray(bin_map, dtype=np.int32)
    
    # Convert values to numeric safely
    try:
        # If values is already numeric numpy array, this won't change it
        if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number):
            numeric_values = values
        else:
            # For non-numeric arrays, try to convert element by element
            numeric_values = np.zeros(len(values), dtype=float)
            for i, val in enumerate(values):
                try:
                    numeric_values[i] = float(val)
                except (ValueError, TypeError):
                    numeric_values[i] = np.nan
    except Exception as e:
        logger.warning(f"Error converting values to numeric: {e}")
        # Create NaN array as fallback
        numeric_values = np.full(np.max(bin_map_int) + 1, np.nan)
    
    # Convert errors to numeric if provided
    numeric_errors = None
    if errors is not None:
        try:
            if isinstance(errors, np.ndarray) and np.issubdtype(errors.dtype, np.number):
                numeric_errors = errors
            else:
                numeric_errors = np.zeros(len(errors), dtype=float)
                for i, err in enumerate(errors):
                    try:
                        numeric_errors[i] = float(err)
                    except (ValueError, TypeError):
                        numeric_errors[i] = np.nan
        except Exception as e:
            logger.warning(f"Error converting errors to numeric: {e}")
            numeric_errors = None
    
    # Create value map using bin numbers
    value_map = np.full_like(bin_map, np.nan, dtype=float)
    
    # Valid bins are non-negative and within range of values
    max_bin = min(np.max(bin_map_int), len(numeric_values) - 1)
    
    # Calculate S/N if requested and errors available
    if show_snr and numeric_errors is not None:
        # Calculate S/N ratio
        snr_values = np.zeros_like(numeric_values)
        for i in range(len(numeric_values)):
            if np.isfinite(numeric_values[i]) and np.isfinite(numeric_errors[i]) and numeric_errors[i] > 0:
                snr_values[i] = np.abs(numeric_values[i]) / numeric_errors[i]
            else:
                snr_values[i] = np.nan
        
        # Use S/N values for plotting
        plot_values = snr_values
        if label and not label.endswith('S/N'):
            label = f"{label} S/N"
    else:
        plot_values = numeric_values
    
    # Populate value map
    for bin_idx in range(max_bin + 1):
        # Safety check to avoid index errors
        if bin_idx < len(plot_values):
            value = plot_values[bin_idx]
            # Check if value is valid
            if np.isfinite(value):
                value_map[bin_map_int == bin_idx] = value
    
    # Mark low S/N regions if threshold provided
    if show_snr and snr_threshold is not None:
        # Create mask for low S/N
        low_snr_mask = value_map < snr_threshold
        
        # Create masked array for better visualization
        masked_data = np.ma.array(value_map, mask=~np.isfinite(value_map))
        
        # Plot with special handling for low S/N
        if vmin is None or vmax is None:
            valid_data = masked_data.compressed()
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = max(0, np.nanpercentile(valid_data, 5))
                if vmax is None:
                    vmax = np.nanpercentile(valid_data, 95)
        
        im = ax.imshow(masked_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Overlay low S/N regions
        low_snr_overlay = np.ma.array(np.ones_like(value_map), mask=~low_snr_mask)
        ax.imshow(low_snr_overlay, origin='lower', cmap='gray', alpha=0.5, vmin=0, vmax=1)
        
        # Add text annotation
        ax.text(0.02, 0.98, f'Gray: S/N < {snr_threshold}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Create masked array for better visualization
        masked_data = np.ma.array(value_map, mask=~np.isfinite(value_map))
        
        # Determine color limits
        if vmin is None or vmax is None:
            valid_data = masked_data.compressed()
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = np.nanpercentile(valid_data, 5)
                if vmax is None:
                    vmax = np.nanpercentile(valid_data, 95)
        
        # Plot the data
        im = ax.imshow(masked_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if label:
        cbar.set_label(label)
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Set aspect ratio for better visualization
    ax.set_aspect('equal')
    
    return ax


def create_error_summary_plot(results_dict, figsize=(16, 12), title=None, save_path=None):
    """
    Create comprehensive error summary plot for analysis results
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing analysis results with errors
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    # Extract available data
    has_kinematics = 'stellar_kinematics' in results_dict
    has_errors = False
    
    if has_kinematics:
        vel = results_dict['stellar_kinematics'].get('velocity')
        disp = results_dict['stellar_kinematics'].get('dispersion')
        vel_err = results_dict['stellar_kinematics'].get('velocity_error')
        disp_err = results_dict['stellar_kinematics'].get('dispersion_error')
        has_errors = vel_err is not None and disp_err is not None
    
    # 1. Error distributions
    if has_errors:
        ax1 = fig.add_subplot(gs[0, 0])
        if isinstance(vel_err, np.ndarray):
            vel_err_flat = vel_err[np.isfinite(vel_err)]
            if len(vel_err_flat) > 0:
                ax1.hist(vel_err_flat, bins=50, alpha=0.7, color='blue', 
                        label=f'Velocity (median: {np.median(vel_err_flat):.1f})')
        
        if isinstance(disp_err, np.ndarray):
            disp_err_flat = disp_err[np.isfinite(disp_err)]
            if len(disp_err_flat) > 0:
                ax1.hist(disp_err_flat, bins=50, alpha=0.7, color='orange',
                        label=f'Dispersion (median: {np.median(disp_err_flat):.1f})')
        
        ax1.set_xlabel('Error (km/s)')
        ax1.set_ylabel('Count')
        ax1.set_title('Kinematic Error Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. S/N distributions
    if has_errors and has_kinematics:
        ax2 = fig.add_subplot(gs[0, 1])
        
        vel_snr = np.abs(vel) / vel_err
        disp_snr = disp / disp_err
        
        vel_snr_valid = vel_snr[np.isfinite(vel_snr)]
        disp_snr_valid = disp_snr[np.isfinite(disp_snr)]
        
        if len(vel_snr_valid) > 0:
            ax2.hist(vel_snr_valid, bins=50, alpha=0.5, color='blue',
                    label=f'Velocity (median: {np.median(vel_snr_valid):.1f})',
                    density=True)
        
        if len(disp_snr_valid) > 0:
            ax2.hist(disp_snr_valid, bins=50, alpha=0.5, color='orange',
                    label=f'Dispersion (median: {np.median(disp_snr_valid):.1f})',
                    density=True)
        
        ax2.set_xlabel('S/N Ratio')
        ax2.set_ylabel('Normalized Count')
        ax2.set_title('S/N Distributions')
        ax2.legend()
        ax2.set_xlim(0, np.percentile(np.concatenate([vel_snr_valid, disp_snr_valid]), 95))
        ax2.grid(True, alpha=0.3)
    
    # 3. Spatial error pattern
    if has_errors and 'binning' in results_dict:
        ax3 = fig.add_subplot(gs[0, 2])
        bin_map = results_dict['binning'].get('bin_num')
        
        if bin_map is not None:
            # Calculate mean error per bin
            if isinstance(bin_map, np.ndarray) and bin_map.ndim == 2:
                # Already 2D
                safe_plot_array(vel_err, bin_map, ax=ax3, 
                              title='Velocity Error Map',
                              cmap='plasma', label='Error (km/s)')
    
    # 4-6. Parameter errors if available
    if 'stellar_population' in results_dict:
        pop_data = results_dict['stellar_population']
        
        # Age errors
        if 'age_error' in pop_data:
            ax4 = fig.add_subplot(gs[1, 0])
            age_err = pop_data['age_error']
            if isinstance(age_err, np.ndarray):
                age_err_valid = age_err[np.isfinite(age_err)]
                if len(age_err_valid) > 0:
                    ax4.hist(age_err_valid / 1e9, bins=30, alpha=0.7)
                    ax4.set_xlabel('Age Error (Gyr)')
                    ax4.set_ylabel('Count')
                    ax4.set_title('Age Error Distribution')
                    ax4.grid(True, alpha=0.3)
        
        # Metallicity errors
        if 'metallicity_error' in pop_data:
            ax5 = fig.add_subplot(gs[1, 1])
            met_err = pop_data['metallicity_error']
            if isinstance(met_err, np.ndarray):
                met_err_valid = met_err[np.isfinite(met_err)]
                if len(met_err_valid) > 0:
                    ax5.hist(met_err_valid, bins=30, alpha=0.7)
                    ax5.set_xlabel('Metallicity Error')
                    ax5.set_ylabel('Count')
                    ax5.set_title('Metallicity Error Distribution')
                    ax5.grid(True, alpha=0.3)
    
    # 7. Spectral indices errors
    if 'indices' in results_dict or 'bin_indices' in results_dict:
        ax6 = fig.add_subplot(gs[1, 2])
        
        indices_data = results_dict.get('indices', results_dict.get('bin_indices', {}))
        
        # Look for error data
        error_fractions = []
        index_names = []
        
        for idx_name in indices_data:
            if f'{idx_name}_error' in indices_data:
                values = indices_data[idx_name]
                errors = indices_data[f'{idx_name}_error']
                
                if isinstance(values, np.ndarray) and isinstance(errors, np.ndarray):
                    valid = np.isfinite(values) & np.isfinite(errors) & (values != 0)
                    if np.any(valid):
                        rel_err = errors[valid] / np.abs(values[valid])
                        error_fractions.append(np.median(rel_err))
                        index_names.append(idx_name)
        
        if error_fractions:
            y_pos = np.arange(len(index_names))
            ax6.barh(y_pos, error_fractions)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(index_names)
            ax6.set_xlabel('Median Relative Error')
            ax6.set_title('Spectral Index Errors')
            ax6.grid(True, alpha=0.3)
    
    # 8. Error correlation matrix
    if has_errors:
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Create correlation matrix for errors
        error_data = []
        error_labels = []
        
        if isinstance(vel_err, np.ndarray):
            error_data.append(vel_err.flatten())
            error_labels.append('Vel Error')
        
        if isinstance(disp_err, np.ndarray):
            error_data.append(disp_err.flatten())
            error_labels.append('Disp Error')
        
        if len(error_data) >= 2:
            # Calculate correlation
            valid_mask = np.all([np.isfinite(d) for d in error_data], axis=0)
            if np.sum(valid_mask) > 10:
                error_array = np.array([d[valid_mask] for d in error_data])
                corr_matrix = np.corrcoef(error_array)
                
                plot_correlation_matrix(corr_matrix, error_labels, ax=ax7)
    
    # 9. Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    stats_text = ['Error Analysis Summary:\n']
    
    if has_errors:
        if isinstance(vel_err, np.ndarray):
            stats_text.append(f'Velocity Error:')
            stats_text.append(f'  Median: {np.nanmedian(vel_err):.1f} km/s')
            stats_text.append(f'  90th %ile: {np.nanpercentile(vel_err, 90):.1f} km/s')
        
        if isinstance(disp_err, np.ndarray):
            stats_text.append(f'\nDispersion Error:')
            stats_text.append(f'  Median: {np.nanmedian(disp_err):.1f} km/s')
            stats_text.append(f'  90th %ile: {np.nanpercentile(disp_err, 90):.1f} km/s')
        
        # Quality metrics
        stats_text.append(f'\nQuality Metrics:')
        if len(vel_snr_valid) > 0:
            stats_text.append(f'  Median Vel S/N: {np.median(vel_snr_valid):.1f}')
            stats_text.append(f'  Fraction S/N > 3: {np.sum(vel_snr_valid > 3) / len(vel_snr_valid):.2f}')
    
    ax8.text(0.1, 0.9, '\n'.join(stats_text), transform=ax8.transAxes,
            verticalalignment='top', fontsize=10, family='monospace')
    
    safe_tight_layout(fig)
    
    if save_path:
        standardize_figure_saving(fig, save_path)
    
    return fig