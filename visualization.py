"""
Visualization utilities for ISAPC
Handles consistent plotting for all analysis types
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
    Create a representative flux map from a data cube
    
    Parameters
    ----------
    cube : MUSECube
        Data cube
        
    Returns
    -------
    numpy.ndarray
        2D flux map
    """
    try:
        # Make sure we use the data that matches the cube dimensions
        if hasattr(cube, '_cube_data') and cube._cube_data is not None:
            cube_data = cube._cube_data
            
            # Verify shape
            if len(cube_data.shape) == 3:
                # Use a more representative calculation - sum over a reasonable wavelength range
                if hasattr(cube, '_lambda_gal'):
                    wave = cube._lambda_gal
                    
                    # Make sure wave length matches cube_data first dimension
                    if len(wave) == cube_data.shape[0]:
                        # Use V-band range (5000-5500Å) for a standard representation
                        wave_mask = (wave >= 5000) & (wave <= 5500)
                        
                        if np.any(wave_mask):
                            # Sum over this wavelength range for a more representative flux map
                            flux_map = np.nansum(cube_data[wave_mask], axis=0)
                            logger.info("Created flux map using V-band wavelength range (5000-5500Å)")
                            return flux_map
                
                # Fall back to median if specific ranges don't work
                flux_map = np.nanmedian(cube_data, axis=0)
                logger.info("Created flux map using median of all wavelengths")
                return flux_map
        
        # If we can't use _cube_data, try with _spectra
        if hasattr(cube, '_spectra') and hasattr(cube, '_n_x') and hasattr(cube, '_n_y'):
            spectra = cube._spectra
            
            # Reshape if needed
            if len(spectra.shape) == 2:
                # _spectra is [n_wave, n_pixels]
                n_wave = spectra.shape[0]
                
                if hasattr(cube, '_lambda_gal') and len(cube._lambda_gal) == n_wave:
                    wave = cube._lambda_gal
                    wave_mask = (wave >= 5000) & (wave <= 5500)
                    
                    if np.any(wave_mask):
                        # Calculate median in this range
                        flux_1d = np.nanmedian(spectra[wave_mask], axis=0)
                    else:
                        # Use full range
                        flux_1d = np.nanmedian(spectra, axis=0)
                    
                    # Reshape to 2D
                    flux_map = np.zeros((cube._n_y, cube._n_x))
                    valid_pixels = min(len(flux_1d), cube._n_x * cube._n_y)
                    flux_map.flat[:valid_pixels] = flux_1d[:valid_pixels]
                    
                    logger.info("Created flux map from _spectra")
                    return flux_map
        
        # Last resort - create a dummy map
        flux_map = np.ones((cube._n_y, cube._n_x))
        logger.warning("Created dummy flux map - no suitable data found")
        return flux_map
        
    except Exception as e:
        logger.error(f"Error creating flux map: {e}")
        # Return a dummy flux map
        return np.ones((cube._n_y, cube._n_x))


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


def plot_bin_map(bin_num, values=None, ax=None, cmap='viridis', title=None, 
                colorbar_label=None, physical_scale=True, pixel_size=None, 
                wcs=None, vmin=None, vmax=None, log_scale=False):
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
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 9))
    
    # Get dimensions
    ny, nx = bin_num.shape
    
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
            
            # Plot with proper non-square pixels
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm, 
                         extent=extent, aspect='auto')
            
            # Explicitly set the aspect ratio to match physical scale
            ax.set_aspect(pixel_size_y / pixel_size_x)
            
            # Set axis labels
            ax.set_xlabel('Δ RA (arcsec)')
            ax.set_ylabel('Δ Dec (arcsec)')
        else:
            # Plot with pixel coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm)
            
            # Set axis labels
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
            
            # Use equal aspect ratio for pixel coordinates
            ax.set_aspect('equal')
    
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
                             log_scale=True):
    """
    Plot flux map with radial bins overlay
    
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
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    # Create figure and axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Get dimensions
    ny, nx = flux_map.shape
    
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
            if pixel_size is not None:
                pixel_size_x, pixel_size_y = pixel_size
                
                # Convert to physical coordinates
                center_x_phys = (center_x - nx/2) * pixel_size_x
                center_y_phys = (center_y - ny/2) * pixel_size_y
                
                # Add ellipses for each radius
                for radius in bin_radii:
                    # Convert to WCS coordinates
                    world_coords = wcs_obj.wcs_pix2world([[center_x, center_y]], 0)[0]
                    ell = plot_ellipse_wcs(
                        ax, world_coords[0], world_coords[1], 
                        radius, radius * (1 - ellipticity), pa
                    )
                    ax.add_patch(ell)
            
        except Exception as e:
            logger.warning(f"Error plotting with WCS: {e}")
            wcs_obj = None
    
    # Fall back to physical coordinates without WCS
    if wcs_obj is None:
        if pixel_size is not None:
            pixel_size_x, pixel_size_y = pixel_size
            
            # Calculate extent
            extent = [
                -nx/2 * pixel_size_x, 
                nx/2 * pixel_size_x, 
                -ny/2 * pixel_size_y, 
                ny/2 * pixel_size_y
            ]
            
            # Plot flux map with proper non-square pixels
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap, norm=norm, 
                         extent=extent, aspect='auto')
            
            # Explicitly set the aspect ratio to match physical scale
            ax.set_aspect(pixel_size_y / pixel_size_x)
            
            # Draw radial bins
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
        else:
            # Plot in pixel coordinates
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap, norm=norm)
            
            # Use equal aspect ratio for pixel coordinates
            ax.set_aspect('equal')
            
            for radius in bin_radii:
                if ellipticity == 0 or not np.isfinite(ellipticity):
                    # Draw circle in pixel units
                    ell = Ellipse(
                        (center_x, center_y),
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
                    # Draw ellipse in pixel units
                    ell = Ellipse(
                        (center_x, center_y),
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
            
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flux' + (' (log scale)' if log_scale else ''))
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Flux Map with Radial Bins')
    
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
    Plot binning boundaries overlaid on flux map with correct physical scaling
    
    Parameters
    ----------
    bin_num : numpy.ndarray
        2D array of bin numbers
    flux_map : numpy.ndarray
        2D flux map
    cube : MUSECube
        MUSE data cube
    galaxy_name : str, optional
        Galaxy name for title
    binning_type : str, default="Voronoi"
        Type of binning: "Voronoi" or "Radial"
    bin_centers : tuple, optional
        (x_centers, y_centers) for Voronoi bins
    bin_radii : array-like, optional
        Radii for radial bins
    center_x, center_y : float, optional
        Center coordinates for radial bins
    pa : float, default=0
        Position angle for radial bins
    ellipticity : float, default=0
        Ellipticity for radial bins
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    # Create figure with rectangular aspect ratio
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Get dimensions
    ny, nx = flux_map.shape
    
    # Convert to proper dimensionless numpy arrays
    bin_num_2d = np.asarray(bin_num)
    
    # Handle NaN values in flux map
    flux_map_clean = np.copy(flux_map)
    flux_map_clean[~np.isfinite(flux_map_clean)] = np.nanmin(flux_map_clean[np.isfinite(flux_map_clean)])
    
    # Determine if we have physical scale information
    has_physical_scale = hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y")
    
    # Get WCS if available
    wcs_obj = None
    if hasattr(cube, "_wcs") and cube._wcs is not None:
        try:
            from astropy.wcs import WCS
            wcs_obj = WCS(cube._wcs.to_header())
        except:
            wcs_obj = None
    
    # Calculate physical extent for proper scaling - always use physical coordinates
    if has_physical_scale:
        pixel_size_x = cube._pxl_size_x
        pixel_size_y = cube._pxl_size_y
        
        # Calculate extent - always centered at (0,0)
        extent = [
            -nx/2 * pixel_size_x, 
            nx/2 * pixel_size_x, 
            -ny/2 * pixel_size_y, 
            ny/2 * pixel_size_y
        ]
    else:
        # Fallback to pixel coordinates if no physical scale
        extent = None
    
    # Plot flux map with logarithmic scaling for better contrast
    if np.any(flux_map_clean > 0):
        vmin = np.nanpercentile(flux_map_clean[flux_map_clean > 0], 5)
        vmax = np.nanpercentile(flux_map_clean, 99)
    else:
        vmin, vmax = 1, 10
    
    # Always use physical coordinates if available
    if extent is not None:
        # Plot with proper non-square pixels
        im = ax.imshow(flux_map_clean, origin='lower', cmap='inferno', 
                     norm=LogNorm(vmin=vmin, vmax=vmax), 
                     extent=extent, aspect='auto')
        
        # Explicitly set the aspect ratio to match physical scale
        ax.set_aspect(pixel_size_y / pixel_size_x)
        
        ax.set_xlabel('Δ RA (arcsec)')
        ax.set_ylabel('Δ Dec (arcsec)')
    else:
        # Fallback to pixel coordinates
        im = ax.imshow(flux_map_clean, origin='lower', cmap='inferno',
                     norm=LogNorm(vmin=vmin, vmax=vmax))
                     
        # Use equal aspect ratio for pixel coordinates
        ax.set_aspect('equal')
        
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Pixels')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Flux')
    
    # Draw bin boundaries based on binning type
    if binning_type.lower() == "voronoi":
        # Create a LineCollection for edges between bins
        from matplotlib.collections import LineCollection
        
        edges = []
        # Check horizontal edges
        for j in range(ny):
            for i in range(nx-1):
                if (bin_num_2d[j, i] != bin_num_2d[j, i+1] and 
                    bin_num_2d[j, i] >= 0 and bin_num_2d[j, i+1] >= 0):
                    if extent is not None:
                        # Physical coordinates
                        x1 = (i - nx/2) * pixel_size_x
                        x2 = (i+1 - nx/2) * pixel_size_x
                        y1 = (j - ny/2) * pixel_size_y
                        y2 = y1  # Same y coordinate
                        edges.append([(x1, y1), (x2, y2)])
                    else:
                        # Pixel coordinates
                        edges.append([(i, j), (i+1, j)])
        
        # Check vertical edges
        for i in range(nx):
            for j in range(ny-1):
                if (bin_num_2d[j, i] != bin_num_2d[j+1, i] and 
                    bin_num_2d[j, i] >= 0 and bin_num_2d[j+1, i] >= 0):
                    if extent is not None:
                        # Physical coordinates
                        x1 = (i - nx/2) * pixel_size_x
                        y1 = (j - ny/2) * pixel_size_y
                        x2 = x1  # Same x coordinate
                        y2 = (j+1 - ny/2) * pixel_size_y
                        edges.append([(x1, y1), (x2, y2)])
                    else:
                        # Pixel coordinates
                        edges.append([(i, j), (i, j+1)])
        
        # Add LineCollection to plot
        if edges:
            lc = LineCollection(edges, colors='white', linewidths=0.8, alpha=0.7, zorder=10)
            ax.add_collection(lc)
        
        # Add bin centers if provided and needed (set to False to disable)
        show_bin_numbers = False
        if show_bin_numbers and bin_centers is not None:
            x_centers, y_centers = bin_centers
            
            # Limit the number of labeled centers to avoid clutter
            max_centers = min(20, len(x_centers))
            step = max(1, len(x_centers) // max_centers)
            
            for i in range(0, len(x_centers), step):
                if i < len(x_centers):
                    if extent is not None:
                        # Physical coordinates
                        x = (x_centers[i] - nx/2) * pixel_size_x
                        y = (y_centers[i] - ny/2) * pixel_size_y
                        ax.scatter(x, y, marker='o', s=5, color='cyan', alpha=0.7, zorder=11)
                    else:
                        # Pixel coordinates
                        ax.scatter(x_centers[i], y_centers[i], marker='o', s=5, color='cyan', alpha=0.7, zorder=11)
    
    elif binning_type.lower() == "radial":
        # For radial binning, draw ellipses
        from matplotlib.patches import Ellipse
        
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
            
        # Convert center to physical coordinates if needed
        if extent is not None:
            center_x_phys = (center_x - nx/2) * pixel_size_x
            center_y_phys = (center_y - ny/2) * pixel_size_y
            
            # Draw center
            ax.scatter(center_x_phys, center_y_phys, marker='+', color='cyan', s=50, linewidth=1.5, zorder=11)
        else:
            # Pixel coordinates
            ax.scatter(center_x, center_y, marker='+', color='cyan', s=50, linewidth=1.5, zorder=11)
        
        # Draw ellipses for each bin radius
        if bin_radii is not None:
            # Determine if we should show bin numbers (set to False to disable)
            show_bin_numbers = False
            
            for i, radius in enumerate(bin_radii):
                if extent is not None:
                    # Physical coordinates - account for aspect ratio
                    if ellipticity < 1:
                        # Calculate the minor axis accounting for the aspect ratio of the plot
                        minor_axis = radius * (1 - ellipticity)
                    else:
                        minor_axis = radius * 0.1  # Fallback for extreme ellipticity
                        
                    # Create ellipse with correct aspect ratio
                    ell = Ellipse(
                        (center_x_phys, center_y_phys),
                        2 * radius,
                        2 * minor_axis,
                        angle=pa,
                        fill=False,
                        edgecolor='white',
                        linewidth=1.0,
                        alpha=0.8,
                        zorder=10
                    )
                    ax.add_patch(ell)
                    
                    # Add bin number if needed
                    if show_bin_numbers:
                        angle_rad = np.pi/4
                        text_x = center_x_phys + radius * np.cos(angle_rad)
                        text_y = center_y_phys + minor_axis * np.sin(angle_rad)
                        
                        ax.text(text_x, text_y, str(i), fontsize=9, ha='center', va='center',
                              color='white', bbox=dict(facecolor='black', alpha=0.7, pad=1),
                              zorder=12)
                else:
                    # Pixel coordinates - convert radius if it seems to be in arcsec
                    if radius > 100 and hasattr(cube, "_pxl_size_x"):
                        radius_pix = radius / cube._pxl_size_x
                    else:
                        radius_pix = radius
                    
                    if ellipticity < 1:
                        minor_axis = radius_pix * (1 - ellipticity)
                    else:
                        minor_axis = radius_pix * 0.1
                        
                    ell = Ellipse(
                        (center_x, center_y),
                        2 * radius_pix,
                        2 * minor_axis,
                        angle=pa,
                        fill=False,
                        edgecolor='white',
                        linewidth=1.0,
                        alpha=0.8,
                        zorder=10
                    )
                    ax.add_patch(ell)
                    
                    # Add bin number if needed
                    if show_bin_numbers:
                        angle_rad = np.pi/4
                        text_x = center_x + radius_pix * np.cos(angle_rad)
                        text_y = center_y + minor_axis * np.sin(angle_rad)
                        
                        ax.text(text_x, text_y, str(i), fontsize=9, ha='center', va='center',
                              color='white', bbox=dict(facecolor='black', alpha=0.7, pad=1),
                              zorder=12)
    
    # Set title
    if galaxy_name:
        ax.set_title(f"{galaxy_name} - {binning_type} Binning on Flux")
    else:
        ax.set_title(f"{binning_type} Binning on Flux")
    
    # Make sure axis limits match the data range
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
    # Add grid for reference
    ax.grid(color='gray', linestyle=':', alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_kinematics_summary(velocity_field, dispersion_field, rotation_curve=None, params=None,
                          equal_aspect=True, physical_scale=True, pixel_size=None, wcs=None,
                          rotation_axis=True):
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
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plots
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create separate upper and lower rows
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 1.5], hspace=0.3, wspace=0.3)
    
    # Process WCS
    wcs_obj = process_wcs(wcs) if wcs is not None else None
    
    # Get dimensions
    ny, nx = velocity_field.shape
    
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
                                     vmin=vmin, vmax=vmax, extent=extent, aspect='auto')
                
                # Explicitly set the aspect ratio to match physical scale
                ax_vel.set_aspect(pixel_size_y / pixel_size_x)
                
                ax_vel.set_xlabel('Δ RA (arcsec)')
                ax_vel.set_ylabel('Δ Dec (arcsec)')
            else:
                im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm',
                                     vmin=vmin, vmax=vmax)
                                     
                # Use equal aspect ratio for pixel coordinates
                ax_vel.set_aspect('equal')
                
                ax_vel.set_xlabel('Pixels')
                ax_vel.set_ylabel('Pixels')
    else:
        # Standard plotting for velocity field with physical scaling if available
        if extent is not None:
            im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax, extent=extent, aspect='auto')
            
            # Explicitly set the aspect ratio to match physical scale
            ax_vel.set_aspect(pixel_size_y / pixel_size_x)
            
            ax_vel.set_xlabel('Δ RA (arcsec)')
            ax_vel.set_ylabel('Δ Dec (arcsec)')
        else:
            im_vel = ax_vel.imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax)
                                 
            # Use equal aspect ratio for pixel coordinates
            ax_vel.set_aspect('equal')
            
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
                                       vmin=vmin_disp, vmax=vmax_disp, extent=extent, aspect='auto')
                                       
                # Explicitly set the aspect ratio to match physical scale
                ax_disp.set_aspect(pixel_size_y / pixel_size_x)
                
                ax_disp.set_xlabel('Δ RA (arcsec)')
                ax_disp.set_ylabel('Δ Dec (arcsec)')
            else:
                im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                       vmin=vmin_disp, vmax=vmax_disp)
                                       
                # Use equal aspect ratio for pixel coordinates
                ax_disp.set_aspect('equal')
                
                ax_disp.set_xlabel('Pixels')
                ax_disp.set_ylabel('Pixels')
    else:
        # Standard plotting for dispersion with physical scaling if available
        if extent is not None:
            im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp, extent=extent, aspect='auto')
                                   
            # Explicitly set the aspect ratio to match physical scale
            ax_disp.set_aspect(pixel_size_y / pixel_size_x)
            
            ax_disp.set_xlabel('Δ RA (arcsec)')
            ax_disp.set_ylabel('Δ Dec (arcsec)')
        else:
            im_disp = ax_disp.imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp)
                                   
            # Use equal aspect ratio for pixel coordinates
            ax_disp.set_aspect('equal')
            
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
                      physical_scale=True, pixel_size=None, wcs=None, rot_angle=0.0):
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
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plots
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Process WCS
    wcs_obj = process_wcs(wcs) if wcs is not None else None
    
    # Get dimensions
    ny, nx = velocity_field.shape
    
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
            # Physical coordinates without WCS - use proper non-square pixels
            im_vel = axes[0].imshow(velocity_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax, extent=extent, aspect='auto')
                                 
            im_disp = axes[1].imshow(dispersion_field, origin='lower', cmap='viridis',
                                   vmin=vmin_disp, vmax=vmax_disp, extent=extent, aspect='auto')
            
            # Set the correct aspect ratio for both plots
            for ax in axes:
                ax.set_aspect(pixel_size_y / pixel_size_x)
            
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
            for ax in axes:
                ax.set_aspect('equal')
            
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
    Plot spectrum fit with components
    
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot observed spectrum
    ax.plot(wavelength, observed_flux, 'k-', alpha=0.7, label='Observed')
    
    # Plot model
    ax.plot(wavelength, model_flux, 'r-', lw=1.5, alpha=0.8, label='Best fit')
    
    # Plot components if available
    if stellar_flux is not None:
        ax.plot(wavelength, stellar_flux, 'b-', lw=1.5, alpha=0.6, label='Stellar')
    
    if gas_flux is not None:
        ax.plot(wavelength, gas_flux, 'g-', lw=1.5, alpha=0.6, label='Gas')
    
    # Plot error if available
    if error is not None:
        ax.fill_between(wavelength, observed_flux - error, observed_flux + error,
                      color='gray', alpha=0.2, label='Error')
    
    # Calculate reasonable y-axis range with error handling
    valid_flux = observed_flux[np.isfinite(observed_flux)]
    valid_model = model_flux[np.isfinite(model_flux)]
    
    if len(valid_flux) > 0 and len(valid_model) > 0:
        min_y = np.nanmin([np.nanmin(valid_flux), np.nanmin(valid_model)])
        max_y = np.nanmax([np.nanmax(valid_flux), np.nanmax(valid_model)])
        
        # Add a margin
        range_y = max_y - min_y
        y_margin = 0.1 * range_y
        ax.set_ylim(min_y - y_margin, max_y + y_margin)
    
    # Add grid, legend and labels
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Spectrum Fit')
    
    # Apply tight layout safely
    safe_tight_layout(fig)
    
    return fig, ax


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
    # Create a multi-panel diagnostic figure
    fig = plt.figure(figsize=(16, 12))
    
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
    
    # 2. Flux map (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        flux_map = prepare_flux_map(cube)
        
        # Use proper non-square pixels if physical scaling is available
        if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
            pixel_size_x = cube._pxl_size_x
            pixel_size_y = cube._pxl_size_y
            
            # Calculate the aspect ratio
            aspect_ratio = pixel_size_y / pixel_size_x
            
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
                aspect='auto'  # Use 'auto' to allow non-square pixels
            )
            
            # Explicitly set the aspect ratio
            ax2.set_aspect(aspect_ratio)
            
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
            
            # Use equal aspect ratio for pixel coordinates
            ax2.set_aspect('equal')
            
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
            
            # Use proper non-square pixels if physical scaling is available
            if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
                pixel_size_x = cube._pxl_size_x
                pixel_size_y = cube._pxl_size_y
                
                # Calculate the aspect ratio
                aspect_ratio = pixel_size_y / pixel_size_x
                
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
                    aspect='auto'  # Use 'auto' to allow non-square pixels
                )
                
                # Explicitly set the aspect ratio
                ax4.set_aspect(aspect_ratio)
                
                ax4.set_xlabel('Δ RA (arcsec)')
                ax4.set_ylabel('Δ Dec (arcsec)')
            else:
                im4 = ax4.imshow(
                    vel_map, 
                    origin='lower', 
                    cmap='coolwarm',
                    vmin=vmin, vmax=vmax
                )
                
                # Use equal aspect ratio for pixel coordinates
                ax4.set_aspect('equal')
                
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
            
            # Plot with proper non-square pixels if physical scaling is available
            if hasattr(cube, '_pxl_size_x') and hasattr(cube, '_pxl_size_y'):
                pixel_size_x = cube._pxl_size_x
                pixel_size_y = cube._pxl_size_y
                
                # Calculate the aspect ratio
                aspect_ratio = pixel_size_y / pixel_size_x
                
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
                    aspect='auto'  # Use 'auto' to allow non-square pixels
                )
                
                # Explicitly set the aspect ratio
                ax5.set_aspect(aspect_ratio)
                
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
                
                # Use equal aspect ratio for pixel coordinates
                ax5.set_aspect('equal')
                
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
                   log_scale=False):
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
        
    Returns
    -------
    fig, ax, im : tuple
        Figure, axis and image objects
    """
    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = ax.figure
        
    # Get dimensions
    ny, nx = data.shape
    
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
            
            # Display the data with non-square pixels
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm, 
                         extent=extent, aspect='auto')
                         
            # Explicitly set the aspect ratio to match physical scale
            ax.set_aspect(pixel_size_y / pixel_size_x)
            
            # Set axis labels
            ax.set_xlabel('Δ RA (arcsec)')
            ax.set_ylabel('Δ Dec (arcsec)')
        else:
            # Plot with pixel coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, norm=norm)
            
            # Use equal aspect ratio for pixel coordinates
            ax.set_aspect('equal')
            
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