"""
Visualization module for IFU data analysis
Contains functions for plotting spectra, kinematic maps, binning, and more
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
import traceback

logger = logging.getLogger(__name__)

# Add this to visualization.py at the top of the file after imports
import contextlib


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    """Context manager for matplotlib figures to ensure they are closed properly.

    Usage:
        with figure_context(figsize=(10, 5)) as fig:
            # Do plotting operations on fig
            # Figure will be automatically closed when exiting the with block
    """
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)

def standardize_figure_saving(fig, file_path, dpi=150, close_after=True):
    """Standard approach to save figures with consistent settings"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save with consistent settings
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {file_path}")

        # Close figure if requested
        if close_after:
            plt.close(fig)

        return True
    except Exception as e:
        logger.warning(f"Error saving figure to {file_path}: {e}")
        # Still try to close the figure on error
        if close_after:
            try:
                plt.close(fig)
            except:
                pass
        return False

def plot_spectrum(
    wavelength,
    flux,
    ax=None,
    title="Spectrum",
    xlabel="Wavelength (Å)",
    ylabel="Flux",
    color="k",
    linewidth=1,
    alpha=1,
    label=None,
):
    """
    Plot a single spectrum.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array
    flux : ndarray
        Flux array
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Spectrum'
        Title for the plot
    xlabel : str, default='Wavelength (Å)'
        X-axis label
    ylabel : str, default='Flux'
        Y-axis label
    color : str, default='k'
        Line color
    linewidth : float, default=1
        Line width
    alpha : float, default=1
        Line transparency
    label : str, optional
        Label for the legend

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Handle NaN values
    valid_mask = np.isfinite(wavelength) & np.isfinite(flux)
    if not np.any(valid_mask):
        ax.text(
            0.5,
            0.5,
            "No valid data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    wave = wavelength[valid_mask]
    fl = flux[valid_mask]

    ax.plot(wave, fl, color=color, linewidth=linewidth, alpha=alpha, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if label is not None:
        ax.legend()

    return ax


def plot_spectrum_fit(
    wavelength,
    observed_flux,
    model_flux,
    stellar_flux=None,
    gas_flux=None,
    residual=None,
    mask=None,
    ranges=None,
    title="Spectrum Fit",
    figsize=(12, 8),
):
    """
    Plot observed spectrum with fitted model components.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array
    observed_flux : ndarray
        Observed flux array
    model_flux : ndarray
        Model flux array
    stellar_flux : ndarray, optional
        Stellar component flux
    gas_flux : ndarray, optional
        Gas component flux
    residual : ndarray, optional
        Residual flux (observed - model)
    mask : ndarray, optional
        Boolean mask for regions to highlight
    ranges : list of tuples, optional
        List of wavelength ranges to highlight
    title : str, default='Spectrum Fit'
        Title for the plot
    figsize : tuple, default=(12, 8)
        Figure size

    Returns
    -------
    tuple
        (figure, axes) tuple
    """
    # Create residual if not provided
    if residual is None:
        residual = observed_flux - model_flux

    # Handle NaN values
    valid_mask = (
        np.isfinite(wavelength) & np.isfinite(observed_flux) & np.isfinite(model_flux)
    )
    if not np.any(valid_mask):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No valid data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, (ax,)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot observed and fitted spectrum
    ax1.plot(wavelength, observed_flux, "k-", lw=1.5, label="Observed")
    ax1.plot(wavelength, model_flux, "r-", lw=1.5, label="Model")

    # Plot components if provided and valid
    if stellar_flux is not None and np.any(np.isfinite(stellar_flux)):
        ax1.plot(wavelength, stellar_flux, "b-", lw=1.5, label="Stellar")

    if gas_flux is not None and np.any(np.isfinite(gas_flux)):
        ax1.plot(wavelength, gas_flux, "g-", lw=1.5, label="Gas")

    # Highlight masked regions if provided
    if mask is not None:
        masked_regions = np.ma.masked_where(~mask, observed_flux)
        ax1.plot(wavelength, masked_regions, "y-", lw=1.5, label="Masked")

    # Highlight specific wavelength ranges if provided
    if ranges is not None:
        for i, (wmin, wmax) in enumerate(ranges):
            ax1.axvspan(wmin, wmax, color=f"C{i}", alpha=0.2)

    # Plot residuals
    ax2.plot(wavelength, residual, "k-", lw=1.5)
    ax2.axhline(y=0, color="r", linestyle="-", lw=1.0)

    # Set y-axis limits with safeguards
    try:
        # Calculate y-axis limits for main plot
        valid_flux = np.concatenate([observed_flux[valid_mask], model_flux[valid_mask]])
        if stellar_flux is not None and np.any(np.isfinite(stellar_flux)):
            valid_flux = np.concatenate([valid_flux, stellar_flux[valid_mask]])
        if gas_flux is not None and np.any(np.isfinite(gas_flux)):
            valid_flux = np.concatenate([valid_flux, gas_flux[valid_mask]])

        ymin = np.nanpercentile(valid_flux, 1)
        ymax = np.nanpercentile(valid_flux, 99)
        yrange = ymax - ymin

        # Add 10% padding
        ax1.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)

        # Calculate y-axis limits for residual plot
        valid_residual = residual[valid_mask]
        res_ymin = np.nanpercentile(valid_residual, 1)
        res_ymax = np.nanpercentile(valid_residual, 99)
        res_yrange = max(res_ymax - res_ymin, 1e-10)  # Avoid empty range

        # Add 10% padding
        ax2.set_ylim(res_ymin - 0.1 * res_yrange, res_ymax + 0.1 * res_yrange)
    except Exception as e:
        # Fall back to automatic scaling if percentile calculation fails
        warnings.warn(f"Error calculating plot limits: {str(e)}")

    # Add labels and legends
    ax1.set_ylabel("Flux")
    ax1.legend(loc="upper right")
    ax1.set_title(title)

    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("Residual")

    # Remove horizontal space between subplots
    plt.subplots_adjust(hspace=0)

    # Hide x-labels for top subplot
    for label in ax1.get_xticklabels():
        label.set_visible(False)

    return fig, (ax1, ax2)


def plot_velocity_field(
    velocity_field, mask=None, ax=None, title="Velocity Field", equal_aspect=False
):
    """
    Plot the velocity field.

    Parameters
    ----------
    velocity_field : ndarray
        2D array of velocity values
    mask : ndarray, optional
        Boolean mask for values to exclude
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Velocity Field'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Create masked array if mask provided
    if mask is not None:
        vel_plot = np.ma.array(velocity_field, mask=mask)
    else:
        vel_plot = np.ma.array(velocity_field, mask=~np.isfinite(velocity_field))

    # Check if there are any valid values
    if np.all(vel_plot.mask):
        ax.text(
            0.5,
            0.5,
            "No valid velocity data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return ax

    # Get symmetric color range
    valid_values = vel_plot.compressed()
    if len(valid_values) > 0:
        vabs = np.nanpercentile(np.abs(valid_values), 95)
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = -100, 100  # Default range if no valid data

    # Plot velocity field
    im = ax.imshow(
        vel_plot,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="equal" if equal_aspect else "auto",
    )

    plt.colorbar(im, ax=ax, label="Velocity (km/s)")

    ax.set_title(title)

    return ax


def plot_dispersion_field(
    dispersion_field,
    mask=None,
    ax=None,
    title="Velocity Dispersion",
    equal_aspect=False,
):
    """
    Plot the velocity dispersion field.

    Parameters
    ----------
    dispersion_field : ndarray
        2D array of velocity dispersion values
    mask : ndarray, optional
        Boolean mask for values to exclude
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Velocity Dispersion'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Create masked array if mask provided or for NaN values
    if mask is not None:
        disp_plot = np.ma.array(dispersion_field, mask=mask)
    else:
        disp_plot = np.ma.array(dispersion_field, mask=~np.isfinite(dispersion_field))

    # Check if there are any valid values
    if np.all(disp_plot.mask):
        ax.text(
            0.5,
            0.5,
            "No valid dispersion data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return ax

    # Get colormap limits
    valid_values = disp_plot.compressed()
    if len(valid_values) > 0:
        vmin = max(0, np.nanpercentile(valid_values, 5))
        vmax = np.nanpercentile(valid_values, 95)
    else:
        vmin, vmax = 0, 100  # Default range if no valid data

    # Ensure valid range
    if vmin >= vmax:
        vmin = 0
        vmax = max(100, np.nanmax(disp_plot))

    # Plot dispersion field
    im = ax.imshow(
        disp_plot,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal" if equal_aspect else "auto",
    )

    plt.colorbar(im, ax=ax, label="Velocity Dispersion (km/s)")

    ax.set_title(title)

    return ax


def plot_binning_map(
    bin_map,
    snr_map=None,
    ax=None,
    title="Binning Map",
    equal_aspect=False,
    cmap="tab20",
):
    """
    Plot binning map.

    Parameters
    ----------
    bin_map : ndarray
        2D array of bin indices
    snr_map : ndarray, optional
        2D array of SNR values
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Binning Map'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
    cmap : str or matplotlib.colors.Colormap, default='tab20'
        Colormap to use

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    # Create masked array for unbinned pixels
    masked_bin_map = np.ma.array(bin_map, mask=(bin_map < 0))

    # Check if any valid bins exist
    if np.all(masked_bin_map.mask):
        ax.text(
            0.5,
            0.5,
            "No valid binning data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return fig, ax

    # Get number of bins
    try:
        n_bins = int(np.max(bin_map)) + 1
    except:
        n_bins = 1  # Default if calculation fails

    # Plot bin map
    im = ax.imshow(
        masked_bin_map,
        origin="lower",
        cmap=cmap,
        aspect="equal" if equal_aspect else "auto",
        vmin=-0.5,
        vmax=min(n_bins, 20) - 0.5,
    )

    # If SNR map provided, add it as contours
    if snr_map is not None and np.any(np.isfinite(snr_map)):
        try:
            # Smooth SNR map for better visualization
            smoothed_snr = gaussian_filter(np.nan_to_num(snr_map), sigma=1)

            # Create contour levels
            valid_snr = smoothed_snr[np.isfinite(smoothed_snr)]
            if len(valid_snr) > 0:
                snr_min = np.nanmin(valid_snr)
                snr_max = np.nanmax(valid_snr)
                if snr_min < snr_max:
                    snr_levels = np.linspace(snr_min, snr_max, 5)

                    # Plot contours
                    ct = ax.contour(
                        smoothed_snr, levels=snr_levels, colors="white", alpha=0.5
                    )

                    # Add contour labels
                    ax.clabel(ct, inline=True, fontsize=8, fmt="%.1f")
        except Exception as e:
            warnings.warn(f"Error plotting SNR contours: {str(e)}")

    ax.set_title(title)

    return fig, ax


def plot_rotation_curve(
    rotation_curve, plot_model=True, vmax=None, pa=None, title="Rotation Curve", ax=None
):
    """
    Plot rotation curve.

    Parameters
    ----------
    rotation_curve : ndarray
        Array with [radius, velocity] pairs
    plot_model : bool, default=True
        Whether to plot the model curve
    vmax : float, optional
        Maximum rotation velocity
    pa : float, optional
        Position angle in degrees
    title : str, default='Rotation Curve'
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Extract radius and velocity
    try:
        radius = rotation_curve[:, 0]
        velocity = rotation_curve[:, 1]

        # Filter out NaN values
        valid_mask = np.isfinite(radius) & np.isfinite(velocity)
        if not np.any(valid_mask):
            ax.text(
                0.5,
                0.5,
                "No valid rotation curve data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return fig, ax

        radius = radius[valid_mask]
        velocity = velocity[valid_mask]

        # Plot data points
        ax.plot(radius, velocity, "ko", label="Data")

        # Plot model curve if requested
        if plot_model and vmax is not None and np.isfinite(vmax):
            # Create a dense radius array for smooth curve
            if len(radius) > 0:
                r_model = np.linspace(0, np.max(radius) * 1.1, 100)

                # Arctan rotation curve model
                v_model = 2 * vmax / np.pi * np.arctan(r_model / 5)

                # Plot model
                ax.plot(r_model, v_model, "r-", label="Model")

        # Add annotations
        if vmax is not None and np.isfinite(vmax):
            ax.axhline(
                y=vmax,
                color="b",
                linestyle="--",
                label=f"$V_{{max}}$ = {vmax:.1f} km/s",
            )
    except Exception as e:
        warnings.warn(f"Error plotting rotation curve: {str(e)}")
        ax.text(
            0.5,
            0.5,
            "Error plotting rotation curve",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Add legend and labels
    ax.legend(loc="best")
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Rotation Velocity (km/s)")

    # Add PA information if provided
    if pa is not None and np.isfinite(pa):
        ax.text(
            0.05,
            0.95,
            f"PA = {pa:.1f}°",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
        )

    ax.set_title(title)

    return fig, ax


def plot_rotation_model(
    velocity_field,
    mask=None,
    center_x=None,
    center_y=None,
    pa=None,
    model_field=None,
    ax=None,
    title="Rotation Model",
    equal_aspect=False,
):
    """
    Plot rotation model with velocity field.

    Parameters
    ----------
    velocity_field : ndarray
        2D array of velocity values
    mask : ndarray, optional
        Boolean mask for values to exclude
    center_x : float, optional
        X-coordinate of rotation center
    center_y : float, optional
        Y-coordinate of rotation center
    pa : float, optional
        Position angle in degrees
    model_field : ndarray, optional
        2D array of model velocity values
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Rotation Model'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Create masked array for NaN values
    if mask is not None:
        vel_plot = np.ma.array(velocity_field, mask=mask)
    else:
        vel_plot = np.ma.array(velocity_field, mask=~np.isfinite(velocity_field))

    # Check if there are any valid values
    if np.all(vel_plot.mask):
        ax.text(
            0.5,
            0.5,
            "No valid velocity data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return ax

    # Get symmetric color range
    valid_values = vel_plot.compressed()
    if len(valid_values) > 0:
        vabs = np.nanpercentile(np.abs(valid_values), 95)
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = -100, 100  # Default range if no valid data

    # Plot velocity field
    im = ax.imshow(
        vel_plot,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="equal" if equal_aspect else "auto",
    )

    # Add model contours if provided
    if model_field is not None and np.any(np.isfinite(model_field)):
        try:
            # Create contour levels
            levels = np.linspace(vmin, vmax, 11)

            # Plot contours
            ct = ax.contour(model_field, levels=levels, colors="white", alpha=0.7)

            # Add contour labels
            ax.clabel(ct, inline=True, fontsize=8, fmt="%.1f")
        except Exception as e:
            warnings.warn(f"Error plotting model contours: {str(e)}")

    # Add rotation center if provided
    if (
        center_x is not None
        and center_y is not None
        and np.isfinite(center_x)
        and np.isfinite(center_y)
    ):
        ax.plot(center_x, center_y, "wo", markersize=10, markeredgecolor="k")

    # Add rotation axis if PA provided
    if pa is not None and center_x is not None and center_y is not None:
        if np.isfinite(pa) and np.isfinite(center_x) and np.isfinite(center_y):
            try:
                # Convert PA to radians
                pa_rad = np.radians(pa)

                # Get image dimensions
                ny, nx = velocity_field.shape
                radius = min(nx, ny) // 2

                # Calculate endpoints of rotation axis line
                x1 = center_x + radius * np.cos(pa_rad)
                y1 = center_y + radius * np.sin(pa_rad)
                x2 = center_x - radius * np.cos(pa_rad)
                y2 = center_y - radius * np.sin(pa_rad)

                # Plot rotation axis
                ax.plot([x1, x2], [y1, y2], "w--", lw=2)

                # Add PA annotation
                ax.text(
                    0.05,
                    0.95,
                    f"PA = {pa:.1f}°",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    color="w",
                    bbox=dict(facecolor="k", alpha=0.5),
                )
            except Exception as e:
                warnings.warn(f"Error plotting rotation axis: {str(e)}")

    plt.colorbar(im, ax=ax, label="Velocity (km/s)")

    ax.set_title(title)

    return ax


def plot_parameter_map(
    data,
    bin_map,
    ax=None,
    title=None,
    cmap="viridis",
    label=None,
    vmin=None,
    vmax=None,
    equal_aspect=True,
):
    """
    Plot parameter map with robust handling of different array dimensions

    Parameters
    ----------
    data : ndarray
        Parameter values (can be 1D or 2D)
    bin_map : ndarray
        2D bin map
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap to use
    label : str, optional
        Colorbar label
    vmin, vmax : float, optional
        Minimum and maximum values for colorbar
    equal_aspect : bool, default=True
        Whether to use equal aspect ratio

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))  # Square figure size

    try:
        # Create parameter map
        param_map = np.full_like(bin_map, np.nan, dtype=float)

        # Check if data is 1D or 2D
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                # For each bin, fill map with corresponding value
                for i, value in enumerate(data):
                    if i < len(data) and np.isfinite(value):
                        param_map[bin_map == i] = value
            elif len(data.shape) == 2:
                # Check if shapes match
                if data.shape == bin_map.shape:
                    # Direct 2D array, just copy where bin_map is valid
                    valid_mask = bin_map >= 0
                    param_map[valid_mask] = data[valid_mask]
                else:
                    # Reshape if possible, otherwise just fill where we can
                    try:
                        data_reshaped = data.reshape(bin_map.shape)
                        valid_mask = bin_map >= 0
                        param_map[valid_mask] = data_reshaped[valid_mask]
                    except:
                        # Go back to bin-based filling
                        for i in range(np.max(bin_map) + 1):
                            mask = bin_map == i
                            if np.any(mask) and i < data.size:
                                param_map[mask] = data.flat[i]
            else:
                # Higher dimensional - try to flatten and use what we can
                flat_data = data.ravel()
                for i in range(min(np.max(bin_map) + 1, len(flat_data))):
                    mask = bin_map == i
                    if np.any(mask) and i < len(flat_data):
                        param_map[mask] = flat_data[i]

        # Plot parameter map
        valid_param = param_map[np.isfinite(param_map)]
        if len(valid_param) > 0:
            # Determine color scale
            if vmin is None:
                vmin = np.nanpercentile(valid_param, 5)
            if vmax is None:
                vmax = np.nanpercentile(valid_param, 95)

            # Ensure valid range
            if vmin >= vmax:
                vmin = np.nanmin(valid_param)
                vmax = np.nanmax(valid_param)
                # If still invalid, use default range
                if vmin >= vmax:
                    vmin, vmax = 0, 1

            # Create masked array for better visualization
            masked_param = np.ma.array(param_map, mask=~np.isfinite(param_map))

            # Plot the map
            im = ax.imshow(
                masked_param,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal" if equal_aspect else "auto",
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if label:
                cbar.set_label(label)
        else:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        if title:
            ax.set_title(title)

        # Set equal aspect ratio if requested
        if equal_aspect:
            ax.set_aspect("equal")

        # Add grid for reference
        ax.grid(True, alpha=0.3)

        return ax

    except Exception as e:
        logger.warning(f"Error plotting parameter map: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error plotting parameter: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if title:
            ax.set_title(title)
        return ax


def safe_plot_array(
    data,
    bin_map,
    ax=None,
    title=None,
    cmap='viridis',
    label=None,
    physical_scale=False,
    pixel_size=None
):
    """
    Plot array data safely handling different dimensions and shapes
    
    Parameters
    ----------
    data : ndarray or list
        Data to plot (1D, 2D, or other)
    bin_map : ndarray
        Bin map with bin numbers
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap name
    label : str, optional
        Colorbar label
    physical_scale : bool, default=False
        Whether to use physical (arcsec) scale for axes
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec, used if physical_scale=True
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    try:
        # Convert to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check if bin_map is 1D and needs to be reshaped
        bin_map_np = np.asarray(bin_map)
        if bin_map_np.ndim == 1:
            # If we have a 1D bin_map, we need to create a 2D visualization
            # Try to find the ny, nx dimensions in attributes of the calling object
            if hasattr(ax, "_ny") and hasattr(ax, "_nx"):
                ny, nx = ax._ny, ax._nx
            else:
                # Try to guess a reasonable shape that's close to square
                total_pixels = len(bin_map_np)
                nx = int(np.sqrt(total_pixels))
                ny = total_pixels // nx
                if nx * ny < total_pixels:
                    ny += 1

            # Create a 2D map with the reconstructed shape
            shape = (ny, nx)
            try:
                # Try to reshape into 2D array (will only work if it's a perfect rectangle)
                bin_map_2d = bin_map_np.reshape(shape)
            except ValueError:
                # If reshape fails, create a new array and fill values
                bin_map_2d = np.full(shape, -1, dtype=int)
                for i, bin_idx in enumerate(bin_map_np):
                    if i < ny * nx:  # Ensure we don't exceed dimensions
                        row, col = i // nx, i % nx
                        bin_map_2d[row, col] = bin_idx

            bin_map_np = bin_map_2d

        # Create parameter map
        param_map = np.full_like(bin_map_np, np.nan, dtype=float)

        # Different approaches based on data dimensions
        if data.ndim == 0:  # Scalar
            # Use scalar value for all valid bins
            param_map[bin_map_np >= 0] = float(data)

        elif data.ndim == 1:  # 1D array
            # Map each value to corresponding bin
            for i, val in enumerate(data):
                if i < len(data) and np.isfinite(val):
                    param_map[bin_map_np == i] = val

        elif data.ndim == 2 and data.shape == bin_map_np.shape:  # Matching 2D array
            # Direct copy to valid bins
            valid = bin_map_np >= 0
            param_map[valid] = data[valid]

        else:  # Other dimensions or shapes
            # Try to flatten and use what we can
            flat_data = data.flatten()
            for i in range(min(np.max(bin_map_np) + 1, len(flat_data))):
                mask = bin_map_np == i
                if np.any(mask):
                    param_map[mask] = flat_data[i]

        # Check if we have valid data
        valid_data = param_map[np.isfinite(param_map)]
        if len(valid_data) > 0:
            # If physical scaling is requested
            if physical_scale and pixel_size is not None:
                # Create physical coordinates for extent
                ny, nx = bin_map_np.shape
                pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
                
                # Create physical coordinate grid (centered on image center)
                x_min = -nx/2 * pixel_size_x
                x_max = nx/2 * pixel_size_x
                y_min = -ny/2 * pixel_size_y
                y_max = ny/2 * pixel_size_y
                
                # Plot with physical coordinates
                extent = [x_min, x_max, y_min, y_max]
                
                # Calculate color scale
                vmin = np.nanpercentile(valid_data, 5)
                vmax = np.nanpercentile(valid_data, 95)
                
                # Handle equal values
                if vmin >= vmax:
                    vmin = np.nanmin(valid_data) - 0.1
                    vmax = np.nanmax(valid_data) + 0.1
                
                # Create masked array for better display
                masked_param = np.ma.array(param_map, mask=~np.isfinite(param_map))
                
                # Plot with imshow and physical coordinates
                im = ax.imshow(
                    masked_param, 
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    aspect='equal'
                )
                
                # Set axis labels
                ax.set_xlabel('Δ RA (arcsec)')
                ax.set_ylabel('Δ Dec (arcsec)')
                
            else:
                # Calculate color scale
                vmin = np.nanpercentile(valid_data, 5)
                vmax = np.nanpercentile(valid_data, 95)

                # Handle equal values
                if vmin >= vmax:
                    vmin = np.nanmin(valid_data) - 0.1
                    vmax = np.nanmax(valid_data) + 0.1

                # Plot with imshow
                im = ax.imshow(
                    param_map,
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect='equal',
                )
                
                # Set axis labels
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Pixels')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if label:
                cbar.set_label(label)
                
        else:
            ax.text(
                0.5,
                0.5,
                "No valid data to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # Add title
        if title:
            ax.set_title(title)

        # Add grid
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.warning(f"Error in safe_plot_array: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error plotting: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if title:
            ax.set_title(title)

    return ax


def plot_bin_map(
    bin_num,
    values=None,
    ax=None,
    cmap="tab20",
    title=None,
    vmin=None,
    vmax=None,
    log_scale=False,
    colorbar_label=None,
    physical_scale=False,
    pixel_size=None,
    wcs=None,
):
    """
    Plot values for binned data, handling both 1D and 2D bin formats
    
    Parameters
    ----------
    bin_num : ndarray
        Bin map or bin indices
    values : ndarray
        Values for each bin
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    cmap : str, default='viridis'
        Colormap name
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Minimum and maximum color values
    log_scale : bool, default=False
        Whether to use log scale for values
    colorbar_label : str, optional
        Label for colorbar
    physical_scale : bool, default=False
        Whether to use physical (arcsec) scale for axes
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec, used if physical_scale=True
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Store the figure for later closing
        should_close_fig = True
    else:
        should_close_fig = False

    try:
        # Convert inputs to numpy arrays if not already
        bin_num_np = np.asarray(bin_num)
        values_np = np.asarray(values) if values is not None else None

        # Check for 1D vs 2D bin_num
        if bin_num_np.ndim == 1:
            # For 1D arrays, create a bar plot
            return plot_binned_values(
                bin_num_np,
                values_np,
                ax=ax,
                title=title,
                cmap=cmap,
                colorbar_label=colorbar_label,
                vmin=vmin,
                vmax=vmax,
                log_scale=log_scale,
            )
        else:
            # For 2D arrays with WCS or physical scaling
            if physical_scale or wcs is not None:
                # Create value map from bin numbers if needed
                if values_np is None:
                    # Just showing bin numbers
                    value_map = bin_num_np
                    masked_data = np.ma.array(value_map, mask=(value_map < 0))
                    if colorbar_label is None:
                        colorbar_label = "Bin Number"
                else:
                    # Create a value map from bin numbers
                    value_map = np.full_like(bin_num_np, np.nan, dtype=float)
                    
                    # Map values to bins - handle cases where values are for bins
                    if len(values_np) <= np.max(bin_num_np) + 1:
                        # Values are per bin
                        for i, val in enumerate(values_np):
                            if i < len(values_np) and np.isfinite(val):
                                value_map[bin_num_np == i] = val
                    else:
                        # Values might be directly mappable
                        if values_np.shape == bin_num_np.shape:
                            value_map = values_np.copy()
                    
                    # Create a masked array for NaN values
                    masked_data = np.ma.array(value_map, mask=~np.isfinite(value_map))
                    
                    # Handle log scale if requested
                    if log_scale and np.any(masked_data > 0):
                        # Create a separate mask for non-positive values
                        positive_mask = masked_data > 0
                        if np.any(positive_mask):
                            # Apply log10 to positive values
                            log_data = np.log10(masked_data.data)
                            log_data[~positive_mask.data] = np.nan
                            masked_data = np.ma.array(log_data, mask=~positive_mask)
                            
                            # Update colorbar label
                            if colorbar_label and not colorbar_label.startswith("Log"):
                                colorbar_label = f"Log10({colorbar_label})"
                
                # Use plot_with_wcs_grid for WCS plotting or physical scaling
                return plot_with_wcs_grid(
                    masked_data,
                    wcs=wcs,
                    ax=ax,
                    title=title,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar_label=colorbar_label,
                    pixel_size=pixel_size
                )
            else:
                # Use standard plotting without physical scaling or WCS
                return safe_plot_array(
                    values_np,
                    bin_num_np,
                    ax=ax,
                    title=title,
                    cmap=cmap,
                    label=colorbar_label,
                )

    except Exception as e:
        logger.warning(f"Error in plot_bin_map: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error plotting: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if title:
            ax.set_title(title)
        return ax


def plot_binned_values(
    bin_indices,
    values,
    ax=None,
    title=None,
    cmap="viridis",
    colorbar_label=None,
    vmin=None,
    vmax=None,
    log_scale=False,
):
    """
    Plot values for binned data using a bar plot approach

    Parameters
    ----------
    bin_indices : array-like
        Bin indices (can be 1D array or list of bin indices)
    values : array-like
        Values for each bin
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    cmap : str, default='viridis'
        Colormap name or colormap object
    colorbar_label : str, optional
        Label for colorbar
    vmin, vmax : float, optional
        Minimum and maximum values for colorbar
    log_scale : bool, default=False
        Whether to use log scale

    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    try:
        # Convert inputs to numpy arrays if not already
        values_array = np.asarray(values).flatten()  # Ensure 1D

        # Filter out invalid values
        valid_indices = np.isfinite(values_array)
        if np.sum(valid_indices) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            if title:
                ax.set_title(title)
            return ax

        # Create plotting indices (just sequential numbers)
        x = np.arange(len(values_array))

        # Filter to valid data points
        x_valid = x[valid_indices]
        values_valid = values_array[valid_indices]

        # Handle log scale if requested
        if log_scale and np.any(values_valid > 0):
            # Find positive values
            positive_mask = values_valid > 0
            if np.any(positive_mask):
                # Get minimum positive value
                min_positive = np.min(values_valid[positive_mask])

                # Replace non-positive values with a small value
                values_valid[~positive_mask] = min_positive * 0.1

                # Apply log10
                values_valid = np.log10(values_valid)

                # Adjust colorbar label
                if colorbar_label and not colorbar_label.startswith("Log"):
                    colorbar_label = f"Log10({colorbar_label})"

        # Get color limits
        if len(values_valid) > 0:
            if vmin is None:
                vmin = np.min(values_valid)
            if vmax is None:
                vmax = np.max(values_valid)

            # Ensure valid range
            if abs(vmax - vmin) < 1e-10:  # Nearly equal
                vmin = vmin * 0.9 if vmin != 0 else -1
                vmax = vmax * 1.1 if vmax != 0 else 1

            # Create colormap and normalize
            if isinstance(cmap, str):
                cmap_obj = plt.cm.get_cmap(cmap)
            else:
                cmap_obj = cmap

            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap_obj)

            # Get colors for valid values
            colors = sm.to_rgba(values_valid)

            # Create bar plot
            ax.bar(x_valid, values_valid, color=colors, alpha=0.8)

            # Set labels
            ax.set_xlabel("Bin Index")
            ax.set_ylabel(colorbar_label if colorbar_label else "Value")

            # Set reasonable number of x ticks
            max_ticks = min(20, len(x_valid))
            if len(x_valid) > max_ticks:
                step = max(1, len(x_valid) // max_ticks)
                idx = np.arange(0, len(x_valid), step)
                ax.set_xticks(x_valid[idx])
                ax.set_xticklabels([str(int(i)) for i in x_valid[idx]])

            # Add colorbar
            plt.colorbar(sm, ax=ax, label=colorbar_label)
        else:
            ax.text(
                0.5,
                0.5,
                "No valid data to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # Add title
        if title:
            ax.set_title(title)

        # Add grid
        ax.grid(True, alpha=0.3)

        return ax

    except Exception as e:
        logger.warning(f"Error plotting binned values: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if title:
            ax.set_title(title)
        return ax


def add_rotation_markers(ax, center_x, center_y, pa, radius=None, color="w"):
    """
    Add markers for rotation center and axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add markers to
    center_x : float
        X-coordinate of rotation center
    center_y : float
        Y-coordinate of rotation center
    pa : float
        Position angle in degrees
    radius : float, optional
        Length of rotation axis
    color : str, default='w'
        Color of markers

    Returns
    -------
    matplotlib.axes.Axes
        Axis with markers
    """
    try:
        # Check parameters
        if (
            not np.isfinite(center_x)
            or not np.isfinite(center_y)
            or not np.isfinite(pa)
        ):
            return ax

        # Add rotation center
        ax.plot(
            center_x, center_y, "o", color=color, markersize=10, markeredgecolor="k"
        )

        # Add rotation axis if radius provided
        if radius is not None:
            # Convert PA to radians
            pa_rad = np.radians(pa)

            # Calculate endpoints of rotation axis line
            x1 = center_x + radius * np.cos(pa_rad)
            y1 = center_y + radius * np.sin(pa_rad)
            x2 = center_x - radius * np.cos(pa_rad)
            y2 = center_y - radius * np.sin(pa_rad)

            # Plot rotation axis
            ax.plot([x1, x2], [y1, y2], "--", color=color, lw=2)

            # Add PA annotation
            ax.text(
                0.05,
                0.95,
                f"PA = {pa:.1f}°",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                color=color,
                bbox=dict(facecolor="k", alpha=0.5),
            )
    except Exception as e:
        warnings.warn(f"Error adding rotation markers: {str(e)}")

    return ax


def plot_kinematics_summary(
    velocity_field,
    dispersion_field,
    bin_map=None,
    rotation_curve=None,
    params=None,
    equal_aspect=False,
    physical_scale=False,
    pixel_size=None,
    wcs=None,
    rot_angle=0.0,
):
    """
    Create a summary plot of kinematic analysis with robust error handling.

    Parameters
    ----------
    velocity_field : ndarray
        2D array of velocity values
    dispersion_field : ndarray
        2D array of velocity dispersion values
    bin_map : ndarray, optional
        2D array of bin indices
    rotation_curve : ndarray, optional
        Array with [radius, velocity] pairs
    params : dict, optional
        Dictionary of kinematic parameters
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
    physical_scale : bool, default=False
        Whether to use physical (arcsec) scale for axes
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec, used if physical_scale=True
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation
    rot_angle : float, default=0.0
        Rotation angle of the IFU in degrees, used if WCS not available

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    """
    # Create figure
    if rotation_curve is not None and np.any(np.isfinite(rotation_curve)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax3 = None

    try:
        # Create mask for bad values
        mask = ~np.isfinite(velocity_field) | ~np.isfinite(dispersion_field)

        # If bin_map provided, add to mask
        if bin_map is not None:
            mask = mask | (bin_map < 0)

        # Plot velocity field with error handling
        try:
            # Check shape of velocity field
            if len(velocity_field.shape) == 2:
                # Regular 2D velocity field
                valid_vel = velocity_field[~mask]
                if len(valid_vel) > 0:
                    vmin = np.nanpercentile(valid_vel, 5)
                    vmax = np.nanpercentile(valid_vel, 95)

                    # For velocity, use symmetric color scale
                    vabs = max(abs(vmin), abs(vmax))
                    if vmin < 0 and vmax > 0:
                        vmin, vmax = -vabs, vabs
                    
                    # Create masked array for velocity
                    masked_vel = np.ma.array(velocity_field, mask=mask)
                    
                    # Use WCS if available for proper coordinate handling
                    if wcs is not None and physical_scale:
                        try:
                            # Try to use WCS coordinates
                            from astropy.visualization.wcsaxes import WCSAxes
                            
                            # Create new axis with WCS projection if needed
                            if not isinstance(ax1, WCSAxes):
                                # Store old position
                                pos = ax1.get_position()
                                # Remove original axis
                                ax1.remove()
                                # Create new axis with WCS projection
                                ax1 = fig.add_subplot(1, 2 if ax3 is None else 3, 1, projection=wcs)
                                # Restore position
                                ax1.set_position(pos)
                            
                            # Plot with WCS coordinates
                            im1 = ax1.imshow(
                                masked_vel,
                                origin="lower",
                                cmap="coolwarm",
                                vmin=vmin,
                                vmax=vmax,
                            )
                            
                            plt.colorbar(im1, ax=ax1, label="Velocity (km/s)")
                            
                            # Add coordinate grid
                            ax1.grid(color='white', ls='solid', alpha=0.3)
                            ax1.set_xlabel('RA')
                            ax1.set_ylabel('Dec')
                        except Exception as e:
                            logger.warning(f"Error using WCS for velocity plot: {e}")
                            # Fall back to basic physical scaling
                            physical_plot_velocity = True
                            wcs = None
                    
                    # Use physical scaling with pixels if WCS not available
                    if (wcs is None and physical_scale and pixel_size is not None) or not physical_scale:
                        ny, nx = velocity_field.shape
                        
                        if physical_scale and pixel_size is not None:
                            # Get pixel sizes
                            pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
                            
                            # Create physical coordinate grid (centered on image center)
                            x_min = -nx/2 * pixel_size_x
                            x_max = nx/2 * pixel_size_x
                            y_min = -ny/2 * pixel_size_y
                            y_max = ny/2 * pixel_size_y
                            
                            # Plot with physical coordinates
                            extent = [x_min, x_max, y_min, y_max]
                            
                            im1 = ax1.imshow(
                                masked_vel,
                                origin="lower",
                                cmap="coolwarm",
                                vmin=vmin,
                                vmax=vmax,
                                extent=extent,
                                aspect="equal" if equal_aspect else "auto",
                            )
                            
                            # Apply rotation to axes labels if needed
                            if rot_angle != 0:
                                # Create rotated axis labels
                                x_label = f'Δ RA cos({rot_angle:.0f}°) + Δ DEC sin({rot_angle:.0f}°) (arcsec)'
                                y_label = f'-Δ RA sin({rot_angle:.0f}°) + Δ DEC cos({rot_angle:.0f}°) (arcsec)'
                            else:
                                x_label = 'Δ RA (arcsec)'
                                y_label = 'Δ DEC (arcsec)'
                                
                            ax1.set_xlabel(x_label)
                            ax1.set_ylabel(y_label)
                        else:
                            # Standard pixel-based plotting
                            im1 = ax1.imshow(
                                masked_vel,
                                origin="lower",
                                cmap="coolwarm",
                                vmin=vmin,
                                vmax=vmax,
                                aspect="equal" if equal_aspect else "auto",
                            )
                            
                            # Set default axis labels
                            ax1.set_xlabel('Pixels')
                            ax1.set_ylabel('Pixels')
                        
                        plt.colorbar(im1, ax=ax1, label="Velocity (km/s)")
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "No valid velocity data",
                        ha="center",
                        va="center",
                        transform=ax1.transAxes,
                    )
            elif velocity_field.shape[0] == 1:
                # Special case for 1D adapter - create a simple colored plot
                ax1.text(
                    0.5,
                    0.5,
                    "Binned velocity data (non-spatial)",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )

                # Create a simple bar plot of velocity values
                bin_indices = np.arange(velocity_field.shape[1])
                ax1.bar(bin_indices, velocity_field[0], color="blue", alpha=0.7)
                ax1.set_xlabel("Bin Index")
                ax1.set_ylabel("Velocity (km/s)")
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "Incompatible velocity field shape",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
        except Exception as e:
            logger.error(f"Error plotting velocity field: {e}")
            ax1.text(
                0.5,
                0.5,
                "Error plotting velocity field",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        ax1.set_title("Velocity Field")

        # Plot dispersion field with error handling
        try:
            # Check shape of dispersion field
            if len(dispersion_field.shape) == 2:
                # Regular 2D dispersion field
                valid_disp = dispersion_field[~mask]
                if len(valid_disp) > 0:
                    vmin = np.nanpercentile(valid_disp, 5)
                    vmax = np.nanpercentile(valid_disp, 95)
                    
                    # Create masked array for dispersion
                    masked_disp = np.ma.array(dispersion_field, mask=mask)
                    
                    # Use WCS if available for proper coordinate handling
                    if wcs is not None and physical_scale:
                        try:
                            # Try to use WCS coordinates
                            from astropy.visualization.wcsaxes import WCSAxes
                            
                            # Create new axis with WCS projection if needed
                            if not isinstance(ax2, WCSAxes):
                                # Store old position
                                pos = ax2.get_position()
                                # Remove original axis
                                ax2.remove()
                                # Create new axis with WCS projection
                                ax2 = fig.add_subplot(1, 2 if ax3 is None else 3, 2, projection=wcs)
                                # Restore position
                                ax2.set_position(pos)
                            
                            # Plot with WCS coordinates
                            im2 = ax2.imshow(
                                masked_disp,
                                origin="lower",
                                cmap="viridis",
                                vmin=vmin,
                                vmax=vmax,
                            )
                            
                            plt.colorbar(im2, ax=ax2, label="Dispersion (km/s)")
                            
                            # Add coordinate grid
                            ax2.grid(color='white', ls='solid', alpha=0.3)
                            ax2.set_xlabel('RA')
                            ax2.set_ylabel('Dec')
                        except Exception as e:
                            logger.warning(f"Error using WCS for dispersion plot: {e}")
                            # Fall back to basic physical scaling
                            wcs = None
                    
                    # Use physical scaling with pixels if WCS not available
                    if (wcs is None and physical_scale and pixel_size is not None) or not physical_scale:
                        ny, nx = dispersion_field.shape
                        
                        if physical_scale and pixel_size is not None:
                            # Get pixel sizes
                            pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
                            
                            # Create physical coordinate grid (centered on image center)
                            x_min = -nx/2 * pixel_size_x
                            x_max = nx/2 * pixel_size_x
                            y_min = -ny/2 * pixel_size_y
                            y_max = ny/2 * pixel_size_y
                            
                            # Plot with physical coordinates
                            extent = [x_min, x_max, y_min, y_max]
                            
                            im2 = ax2.imshow(
                                masked_disp,
                                origin="lower",
                                cmap="viridis",
                                vmin=vmin,
                                vmax=vmax,
                                extent=extent,
                                aspect="equal" if equal_aspect else "auto",
                            )
                            
                            # Apply rotation to axes labels if needed
                            if rot_angle != 0:
                                # Create rotated axis labels
                                x_label = f'Δ RA cos({rot_angle:.0f}°) + Δ DEC sin({rot_angle:.0f}°) (arcsec)'
                                y_label = f'-Δ RA sin({rot_angle:.0f}°) + Δ DEC cos({rot_angle:.0f}°) (arcsec)'
                            else:
                                x_label = 'Δ RA (arcsec)'
                                y_label = 'Δ DEC (arcsec)'
                                
                            ax2.set_xlabel(x_label)
                            ax2.set_ylabel(y_label)
                        else:
                            # Standard pixel-based plotting
                            im2 = ax2.imshow(
                                masked_disp,
                                origin="lower",
                                cmap="viridis",
                                vmin=vmin,
                                vmax=vmax,
                                aspect="equal" if equal_aspect else "auto",
                            )
                            
                            # Set default axis labels
                            ax2.set_xlabel('Pixels')
                            ax2.set_ylabel('Pixels')
                        
                        plt.colorbar(im2, ax=ax2, label="Dispersion (km/s)")
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "No valid dispersion data",
                        ha="center",
                        va="center",
                        transform=ax2.transAxes,
                    )
            elif dispersion_field.shape[0] == 1:
                # Special case for 1D adapter - create a simple colored plot
                ax2.text(
                    0.5,
                    0.5,
                    "Binned dispersion data (non-spatial)",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )

                # Create a simple bar plot of dispersion values
                bin_indices = np.arange(dispersion_field.shape[1])
                ax2.bar(bin_indices, dispersion_field[0], color="green", alpha=0.7)
                ax2.set_xlabel("Bin Index")
                ax2.set_ylabel("Dispersion (km/s)")
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Incompatible dispersion field shape",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
        except Exception as e:
            logger.error(f"Error plotting dispersion field: {e}")
            ax2.text(
                0.5,
                0.5,
                "Error plotting dispersion field",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        ax2.set_title("Velocity Dispersion")

        # Add model contours and rotation parameters if available
        if params is not None:
            try:
                # Extract parameters
                center_x = params.get("center_x", None)
                center_y = params.get("center_y", None)
                pa = params.get("pa", None)

                # Check parameter validity
                if (
                    center_x is not None
                    and center_y is not None
                    and pa is not None
                    and np.isfinite(center_x)
                    and np.isfinite(center_y)
                    and np.isfinite(pa)
                ):
                    # Add rotation center and axis to velocity plot
                    add_rotation_markers(
                        ax1,
                        center_x,
                        center_y,
                        pa,
                        radius=min(velocity_field.shape) // 3,
                    )
            except Exception as e:
                logger.warning(f"Error adding rotation markers: {e}")

        # Plot rotation curve if provided in the third panel
        if rotation_curve is not None and ax3 is not None:
            try:
                plot_rotation_curve(
                    rotation_curve,
                    plot_model=params is not None,
                    vmax=params.get("vmax", None) if params else None,
                    pa=params.get("pa", None) if params else None,
                    title="Rotation Curve",
                    ax=ax3,
                )

                # Add parameters if provided
                if params is not None:
                    # Format parameter values as text
                    param_text = []
                    if "vmax" in params and np.isfinite(params["vmax"]):
                        param_text.append(f"$V_{{max}}$ = {params['vmax']:.1f} km/s")
                    if "pa" in params and np.isfinite(params["pa"]):
                        param_text.append(f"PA = {params['pa']:.1f}°")
                    if "vsys" in params and np.isfinite(params["vsys"]):
                        param_text.append(f"$V_{{sys}}$ = {params['vsys']:.1f} km/s")
                    if "center_x" in params and "center_y" in params:
                        if np.isfinite(params["center_x"]) and np.isfinite(
                            params["center_y"]
                        ):
                            param_text.append(
                                f"Center = ({params['center_x']:.1f}, {params['center_y']:.1f})"
                            )
                    if "sigma_mean" in params and np.isfinite(params["sigma_mean"]):
                        param_text.append(
                            f"$\\sigma_{{mean}}$ = {params['sigma_mean']:.1f} km/s"
                        )

                    # Add text box
                    if param_text:
                        ax3.text(
                            0.05,
                            0.05,
                            "\n".join(param_text),
                            transform=ax3.transAxes,
                            fontsize=10,
                            verticalalignment="bottom",
                            horizontalalignment="left",
                            bbox=dict(facecolor="white", alpha=0.7),
                        )
            except Exception as e:
                logger.warning(f"Error plotting rotation curve: {e}")
                if ax3:
                    ax3.text(
                        0.5,
                        0.5,
                        "Error plotting rotation curve",
                        ha="center",
                        va="center",
                        transform=ax3.transAxes,
                    )

    except Exception as e:
        logger.error(f"Error in plot_kinematics_summary: {e}")
        # Ensure we still return a figure even after error
        for ax in [ax1, ax2, ax3]:
            if ax:
                ax.text(
                    0.5,
                    0.5,
                    "Error generating plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_bin_kinematics(
    bin_num,
    velocity,
    dispersion,
    ax=None,
    figsize=(12, 5),
    title=None,
    cbar_labels=None,
):
    """
    Plot binned kinematics (velocity and dispersion) side by side

    Parameters
    ----------
    bin_num : ndarray
        Bin map (2D array)
    velocity : ndarray
        Velocity values for each bin
    dispersion : ndarray
        Dispersion values for each bin
    ax : tuple of matplotlib.axes.Axes, optional
        Tuple of two axes for plotting
    figsize : tuple, default=(12, 5)
        Figure size
    title : str, optional
        Title for the plot
    cbar_labels : tuple, optional
        Labels for the colorbars (velocity, dispersion)

    Returns
    -------
    tuple
        (fig, axes) - Figure and axes objects
    """
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    try:
        # Default colorbar labels
        if cbar_labels is None:
            cbar_labels = ("Velocity (km/s)", "Dispersion (km/s)")

        # Plot velocity
        vel_ax = safe_plot_array(
            velocity,
            bin_num,
            ax=ax1,
            cmap="coolwarm",
            title="Velocity Field" if title is None else f"{title} - Velocity",
            label=cbar_labels[0],
        )

        # Plot dispersion
        disp_ax = safe_plot_array(
            dispersion,
            bin_num,
            ax=ax2,
            cmap="viridis",
            title="Velocity Dispersion" if title is None else f"{title} - Dispersion",
            label=cbar_labels[1],
        )

        plt.tight_layout()
        return fig, (vel_ax, disp_ax)

    except Exception as e:
        logger.warning(f"Error plotting bin kinematics: {e}")
        for ax in (ax1, ax2):
            ax.text(
                0.5,
                0.5,
                f"Error plotting: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return fig, (ax1, ax2)


def plot_bin_lic(
    bin_num,
    velocity,
    dispersion=None,
    ax=None,
    figsize=(8, 8),
    title=None,
    cmap="coolwarm",
):
    """
    Plot Line Integral Convolution visualization for binned velocity field

    Parameters
    ----------
    bin_num : ndarray
        Bin map (2D array)
    velocity : ndarray
        Velocity values for each bin
    dispersion : ndarray, optional
        Dispersion values for each bin (for overlay)
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    figsize : tuple, default=(8, 8)
        Figure size
    title : str, optional
        Title for the plot
    cmap : str, default='coolwarm'
        Colormap for velocity

    Returns
    -------
    tuple
        (fig, ax) - Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    try:
        # Check if bin_num is 1D and needs reshaping
        bin_num_np = np.asarray(bin_num)
        if bin_num_np.ndim == 1:
            # Try to get dimensions
            if (
                hasattr(ax, "figure")
                and hasattr(ax.figure, "ny")
                and hasattr(ax.figure, "nx")
            ):
                ny, nx = ax.figure.ny, ax.figure.nx
            else:
                # Try to guess a reasonable shape
                total_pixels = len(bin_num_np)
                nx = int(np.sqrt(total_pixels))
                ny = total_pixels // nx
                if nx * ny < total_pixels:
                    ny += 1

            # Create a 2D array for bin map
            shape = (ny, nx)
            bin_num_2d = np.full(shape, -1)

            for i, bin_id in enumerate(bin_num_np):
                if i < ny * nx:
                    row, col = i // nx, i % nx
                    bin_num_2d[row, col] = bin_id

            bin_num_np = bin_num_2d

        # Create 2D velocity field from binned data
        vel_map = np.full_like(bin_num_np, np.nan, dtype=float)

        # Map velocity values to bins
        for i, vel in enumerate(velocity):
            if i < len(velocity) and np.isfinite(vel):
                vel_map[bin_num_np == i] = vel

        # Create masked array
        masked_vel = np.ma.array(vel_map, mask=~np.isfinite(vel_map))

        # Check if we have enough valid data
        if np.sum(~masked_vel.mask) < 10:
            ax.text(
                0.5,
                0.5,
                "Not enough valid velocity data for visualization",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            if title:
                ax.set_title(title)
            return fig, ax

        # Fill NaN values for gradient calculation
        filled_vel = masked_vel.filled(0)

        # Calculate velocity gradients
        vy, vx = np.gradient(filled_vel)

        # Normalize the gradient vectors
        norm = np.sqrt(vx**2 + vy**2)
        mask = norm > 0
        if np.any(mask):
            vx[mask] /= norm[mask]
            vy[mask] /= norm[mask]

        # Plot velocity field
        vmin = np.nanpercentile(vel_map[~masked_vel.mask], 5)
        vmax = np.nanpercentile(vel_map[~masked_vel.mask], 95)

        # For velocity, use symmetric color scale
        vabs = max(abs(vmin), abs(vmax))
        if vmin < 0 and vmax > 0:
            vmin, vmax = -vabs, vabs

        im = ax.imshow(
            masked_vel, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax
        )
        plt.colorbar(im, ax=ax, label="Velocity (km/s)")

        # Try to add streamlines (with error handling for compatibility)
        try:
            # Set up streamplot
            y, x = np.mgrid[: vel_map.shape[0], : vel_map.shape[1]]

            # Use a more compatible call to streamplot
            # First try with just the required parameters
            ax.streamplot(x, y, vx, vy)
        except Exception as e:
            logger.warning(f"Could not create streamplot: {e}")

        # Add title
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Velocity Field Visualization")

        return fig, ax

    except Exception as e:
        logger.warning(f"Error creating velocity field plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if title:
            ax.set_title(title)
        return fig, ax


def plot_bin_emission_lines(
    bin_num,
    emission_flux,
    line_names=None,
    max_lines=6,
    figsize=None,
    log_scale=True,
    title=None,
):
    """
    Plot emission line maps for binned data

    Parameters
    ----------
    bin_num : ndarray
        Bin map (2D array)
    emission_flux : dict
        Dictionary of emission line fluxes
    line_names : list, optional
        List of emission lines to plot. If None, use all available.
    max_lines : int, default=6
        Maximum number of emission lines to plot
    figsize : tuple, optional
        Figure size. If None, calculated based on number of lines.
    log_scale : bool, default=True
        Whether to use log scale for flux
    title : str, optional
        Title for the plot

    Returns
    -------
    tuple
        (fig, axes) - Figure and axes objects
    """
    # Get emission lines to plot
    if line_names is None:
        line_names = list(emission_flux.keys())

    # Limit to max_lines
    line_names = line_names[:max_lines]
    n_lines = len(line_names)

    if n_lines == 0:
        logger.warning("No emission lines to plot")
        return None, None

    # Calculate figure layout
    if n_lines <= 3:
        nrows, ncols = 1, n_lines
    else:
        nrows = (n_lines + 2) // 3  # Ceiling division
        ncols = min(3, n_lines)

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easy iteration
    if n_lines == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()

    try:
        # Plot each emission line
        for i, line_name in enumerate(line_names):
            if i < len(axes):
                # Get flux for this line
                if line_name in emission_flux:
                    flux = emission_flux[line_name]

                    # Plot flux
                    safe_plot_array(
                        flux,
                        bin_num,
                        ax=axes[i],
                        cmap="inferno",
                        title=line_name,
                        label="Log Flux" if log_scale else "Flux",
                    )
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"No data for {line_name}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )
                    axes[i].set_title(line_name)

        # Hide any unused subplots
        for i in range(n_lines, len(axes)):
            axes[i].axis("off")

        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)

        plt.tight_layout()
        return fig, axes

    except Exception as e:
        logger.warning(f"Error plotting emission lines: {e}")
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return fig, axes


def plot_bin_indices(
    bin_num, indices_values, index_names=None, max_indices=6, figsize=None, title=None
):
    """
    Plot spectral index maps for binned data

    Parameters
    ----------
    bin_num : ndarray
        Bin map (2D array)
    indices_values : dict
        Dictionary of spectral index values
    index_names : list, optional
        List of indices to plot. If None, use all available.
    max_indices : int, default=6
        Maximum number of indices to plot
    figsize : tuple, optional
        Figure size. If None, calculated based on number of indices.
    title : str, optional
        Title for the plot

    Returns
    -------
    tuple
        (fig, axes) - Figure and axes objects
    """
    # Get indices to plot
    if index_names is None:
        index_names = list(indices_values.keys())

    # Limit to max_indices
    index_names = index_names[:max_indices]
    n_indices = len(index_names)

    if n_indices == 0:
        logger.warning("No spectral indices to plot")
        return None, None

    # Calculate figure layout
    if n_indices <= 3:
        nrows, ncols = 1, n_indices
    else:
        nrows = (n_indices + 2) // 3  # Ceiling division
        ncols = min(3, n_indices)

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easy iteration
    if n_indices == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()

    try:
        # Plot each index
        for i, index_name in enumerate(index_names):
            if i < len(axes):
                # Get values for this index
                if index_name in indices_values:
                    values = indices_values[index_name]

                    # Plot values
                    safe_plot_array(
                        values,
                        bin_num,
                        ax=axes[i],
                        cmap="plasma",
                        title=index_name,
                        label="Index Value",
                    )
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"No data for {index_name}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )
                    axes[i].set_title(index_name)

        # Hide any unused subplots
        for i in range(n_indices, len(axes)):
            axes[i].axis("off")

        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)

        plt.tight_layout()
        return fig, axes

    except Exception as e:
        logger.warning(f"Error plotting spectral indices: {e}")
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return fig, axes


def plot_bin_spectrum_fit(
    wavelength,
    bin_idx,
    bin_spectrum,
    bin_bestfit,
    bin_gas_bestfit=None,
    bin_stellar_bestfit=None,
    ax=None,
    title=None,
    plot_range=None,
    figsize=(12, 6),
):
    """
    Plot spectrum fitting results for a specific bin

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array
    bin_idx : int
        Bin index to plot
    bin_spectrum : ndarray
        2D array of bin spectra [n_wave, n_bins]
    bin_bestfit : ndarray
        2D array of best-fit spectra [n_wave, n_bins]
    bin_gas_bestfit : ndarray, optional
        2D array of gas component best-fit [n_wave, n_bins]
    bin_stellar_bestfit : ndarray, optional
        2D array of stellar component best-fit [n_wave, n_bins]
    ax : matplotlib.axes, optional
        Axis to plot on (will create new figure if None)
    title : str, optional
        Title for the plot (will use default if None)
    plot_range : tuple, optional
        Wavelength range to plot (min, max)
    figsize : tuple, default=(12, 6)
        Figure size for new figure

    Returns
    -------
    tuple
        (fig, axes) - Figure and axes objects
    """
    # Create figure if needed
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        # If ax is provided, create a residual subplot below it
        fig = ax.figure
        gs = ax.get_gridspec()
        ax1 = ax

        # Create residual subplot
        for i, axis in enumerate(fig.axes):
            if axis == ax:
                # Create new subplot below this one
                ax2 = fig.add_subplot(gs[i + 1])
                break

    try:
        # Extract spectra for this bin
        if bin_spectrum.ndim == 2 and bin_spectrum.shape[1] > bin_idx:
            observed = bin_spectrum[:, bin_idx]
        else:
            observed = bin_spectrum.flatten()

        if bin_bestfit.ndim == 2 and bin_bestfit.shape[1] > bin_idx:
            model = bin_bestfit[:, bin_idx]
        else:
            model = bin_bestfit.flatten()

        # Get gas and stellar components if provided
        if bin_gas_bestfit is not None:
            if bin_gas_bestfit.ndim == 2 and bin_gas_bestfit.shape[1] > bin_idx:
                gas = bin_gas_bestfit[:, bin_idx]
            else:
                gas = bin_gas_bestfit.flatten()
        else:
            gas = None

        if bin_stellar_bestfit is not None:
            if bin_stellar_bestfit.ndim == 2 and bin_stellar_bestfit.shape[1] > bin_idx:
                stellar = bin_stellar_bestfit[:, bin_idx]
            else:
                stellar = bin_stellar_bestfit.flatten()
        elif gas is not None:
            # Calculate stellar = total - gas
            stellar = model - gas
        else:
            stellar = None

        # Calculate residual
        residual = observed - model

        # Filter wavelength range if provided
        if plot_range is not None:
            wmin, wmax = plot_range
            wave_mask = (wavelength >= wmin) & (wavelength <= wmax)

            # Apply mask if any valid points
            if np.any(wave_mask):
                plot_wave = wavelength[wave_mask]
                plot_observed = observed[wave_mask]
                plot_model = model[wave_mask]
                plot_residual = residual[wave_mask]

                if gas is not None:
                    plot_gas = gas[wave_mask]
                else:
                    plot_gas = None

                if stellar is not None:
                    plot_stellar = stellar[wave_mask]
                else:
                    plot_stellar = None
            else:
                # Use full range if no valid points in specified range
                plot_wave = wavelength
                plot_observed = observed
                plot_model = model
                plot_residual = residual
                plot_gas = gas
                plot_stellar = stellar
        else:
            # Use full range
            plot_wave = wavelength
            plot_observed = observed
            plot_model = model
            plot_residual = residual
            plot_gas = gas
            plot_stellar = stellar

        # Check if we have any valid data
        if not np.any(np.isfinite(plot_observed)) or not np.any(
            np.isfinite(plot_model)
        ):
            ax1.text(
                0.5,
                0.5,
                "No valid data for this bin",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax2.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

            # Set title
            if title:
                ax1.set_title(title)
            else:
                ax1.set_title(f"Bin {bin_idx} - No valid data")

            return fig, (ax1, ax2)

        # Plot observed data and model
        ax1.plot(plot_wave, plot_observed, "k-", label="Observed", alpha=0.7, lw=1)
        ax1.plot(plot_wave, plot_model, "r-", label="Model", lw=1.5)

        # Plot components if available
        if plot_stellar is not None and np.any(np.isfinite(plot_stellar)):
            ax1.plot(plot_wave, plot_stellar, "b-", label="Stellar", lw=1, alpha=0.8)

        if plot_gas is not None and np.any(np.isfinite(plot_gas)):
            ax1.plot(plot_wave, plot_gas, "g-", label="Gas", lw=1, alpha=0.8)

        # Plot residual
        ax2.plot(plot_wave, plot_residual, "k-", lw=1)
        ax2.axhline(y=0, color="r", linestyle="-", lw=0.5)

        # Calculate axis limits robustly
        try:
            # Get all valid spectra for y-axis limits
            all_data = [
                d
                for d in [plot_observed, plot_model, plot_stellar, plot_gas]
                if d is not None and np.any(np.isfinite(d))
            ]

            if all_data:
                # Concatenate all data
                all_values = np.concatenate([d[np.isfinite(d)] for d in all_data])

                if len(all_values) > 0:
                    # Set y limits with 5% padding
                    ymin = np.percentile(all_values, 1)
                    ymax = np.percentile(all_values, 99)
                    yrange = ymax - ymin
                    ax1.set_ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)

                    # Set residual limits
                    valid_residual = plot_residual[np.isfinite(plot_residual)]
                    if len(valid_residual) > 0:
                        res_ymin = np.percentile(valid_residual, 1)
                        res_ymax = np.percentile(valid_residual, 99)
                        res_yrange = max(res_ymax - res_ymin, 1e-10)
                        ax2.set_ylim(
                            res_ymin - 0.05 * res_yrange, res_ymax + 0.05 * res_yrange
                        )
        except Exception as e:
            # Fall back to automatic scaling
            logger.warning(f"Error calculating plot limits: {e}")

        # Add labels and title
        ax1.set_ylabel("Flux")
        ax1.legend(loc="upper right")

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title(f"Bin {bin_idx} - Spectrum Fit")

        ax2.set_xlabel("Wavelength (Å)")
        ax2.set_ylabel("Residual")

        # Hide x-ticks for top panel
        ax1.set_xticklabels([])

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.1)

        return fig, (ax1, ax2)

    except Exception as e:
        # Handle errors
        logger.warning(f"Error plotting bin spectrum: {e}")

        # Display error message
        if ax1 is not None:
            ax1.text(
                0.5,
                0.5,
                f"Error plotting spectrum: {str(e)}",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
        if ax2 is not None:
            ax2.axis("off")  # Hide residual axis

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title(f"Bin {bin_idx} - Error")

        return fig, (ax1, ax2)


def plot_kinematics_summary(
    velocity_field,
    dispersion_field,
    bin_map=None,
    rotation_curve=None,
    params=None,
    equal_aspect=False,
    physical_scale=False,
    pixel_size=None,
    wcs=None,
):
    """
    Create a summary plot of kinematic analysis with robust error handling.

    Parameters
    ----------
    velocity_field : ndarray
        2D array of velocity values
    dispersion_field : ndarray
        2D array of velocity dispersion values
    bin_map : ndarray, optional
        2D array of bin indices
    rotation_curve : ndarray, optional
        Array with [radius, velocity] pairs
    params : dict, optional
        Dictionary of kinematic parameters
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
    physical_scale : bool, default=False
        Whether to use physical (arcsec) scale for axes
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec, used if physical_scale=True
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    """
    # Create figure
    if rotation_curve is not None and np.any(np.isfinite(rotation_curve)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax3 = None

    try:
        # Create mask for bad values
        mask = ~np.isfinite(velocity_field) | ~np.isfinite(dispersion_field)

        # If bin_map provided, add to mask
        if bin_map is not None:
            mask = mask | (bin_map < 0)

        # Plot velocity field with proper WCS
        try:
            # Check shape of velocity field
            if len(velocity_field.shape) == 2:
                # Regular 2D velocity field
                valid_vel = velocity_field[~mask]
                if len(valid_vel) > 0:
                    vmin = np.nanpercentile(valid_vel, 5)
                    vmax = np.nanpercentile(valid_vel, 95)

                    # For velocity, use symmetric color scale
                    vabs = max(abs(vmin), abs(vmax))
                    if vmin < 0 and vmax > 0:
                        vmin, vmax = -vabs, vabs
                    
                    # Create masked array
                    masked_vel = np.ma.array(velocity_field, mask=mask)
                    
                    # Use WCS plotting
                    plot_with_wcs_grid(
                        masked_vel,
                        wcs=wcs if (physical_scale or wcs is not None) else None,
                        ax=ax1,
                        title="Velocity Field",
                        cmap="coolwarm",
                        vmin=vmin,
                        vmax=vmax,
                        colorbar_label="Velocity (km/s)",
                        pixel_size=pixel_size if physical_scale else None
                    )
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "No valid velocity data",
                        ha="center",
                        va="center",
                        transform=ax1.transAxes,
                    )
                    ax1.set_title("Velocity Field")
            else:
                # Handling for non-2D data - use the existing code
                ax1.text(
                    0.5,
                    0.5,
                    "Binned velocity data (non-spatial)",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                
                # Create a simple bar plot of velocity values
                bin_indices = np.arange(velocity_field.shape[1])
                ax1.bar(bin_indices, velocity_field[0], color="blue", alpha=0.7)
                ax1.set_xlabel("Bin Index")
                ax1.set_ylabel("Velocity (km/s)")
                ax1.set_title("Velocity Field")
                
        except Exception as e:
            logger.error(f"Error plotting velocity field: {e}")
            ax1.text(
                0.5,
                0.5,
                "Error plotting velocity field",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title("Velocity Field")

        # Plot dispersion field with proper WCS
        try:
            # Check shape of dispersion field
            if len(dispersion_field.shape) == 2:
                # Regular 2D dispersion field
                valid_disp = dispersion_field[~mask]
                if len(valid_disp) > 0:
                    vmin = np.nanpercentile(valid_disp, 5)
                    vmax = np.nanpercentile(valid_disp, 95)
                    
                    # Create masked array
                    masked_disp = np.ma.array(dispersion_field, mask=mask)
                    
                    # Use WCS plotting
                    plot_with_wcs_grid(
                        masked_disp,
                        wcs=wcs if (physical_scale or wcs is not None) else None,
                        ax=ax2,
                        title="Velocity Dispersion",
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        colorbar_label="Dispersion (km/s)",
                        pixel_size=pixel_size if physical_scale else None
                    )
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "No valid dispersion data",
                        ha="center",
                        va="center",
                        transform=ax2.transAxes,
                    )
                    ax2.set_title("Velocity Dispersion")
            else:
                # Handling for non-2D data - use the existing code
                ax2.text(
                    0.5,
                    0.5,
                    "Binned dispersion data (non-spatial)",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                
                # Create a simple bar plot of dispersion values
                bin_indices = np.arange(dispersion_field.shape[1])
                ax2.bar(bin_indices, dispersion_field[0], color="green", alpha=0.7)
                ax2.set_xlabel("Bin Index")
                ax2.set_ylabel("Dispersion (km/s)")
                ax2.set_title("Velocity Dispersion")
                
        except Exception as e:
            logger.error(f"Error plotting dispersion field: {e}")
            ax2.text(
                0.5,
                0.5,
                "Error plotting dispersion field",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Velocity Dispersion")

        # [Rest of the function for rotation curve plotting remains the same]

    except Exception as e:
        logger.error(f"Error in plot_kinematics_summary: {e}")
        # Ensure we still return a figure even after error
        for ax in [ax1, ax2, ax3]:
            if ax:
                ax.text(
                    0.5,
                    0.5,
                    "Error generating plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_field_physical(
    data,
    ax=None,
    title=None,
    cmap='viridis',
    vmin=None,
    vmax=None,
    colorbar_label=None,
    wcs=None,
    pixel_size=None,
    rot_angle=0.0,
):
    """
    Plot a 2D field with proper physical coordinates
    
    Parameters
    ----------
    data : ndarray
        2D array of values to plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap
    vmin, vmax : float, optional
        Color scale limits
    colorbar_label : str, optional
        Label for colorbar
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec, used if WCS not available
    rot_angle : float, default=0.0
        Rotation angle of the IFU in degrees, used if WCS not available
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
        
    # Create masked array for NaN values
    masked_data = np.ma.array(data, mask=~np.isfinite(data))
    
    # Set color limits
    if vmin is None:
        vmin = np.ma.min(masked_data)
    if vmax is None:
        vmax = np.ma.max(masked_data)
    
    # Plot based on available coordinate information
    if wcs is not None:
        try:
            # Try to use WCS coordinates
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create new axis with WCS projection if needed
            if not isinstance(ax, WCSAxes):
                fig = ax.figure
                ax.remove()
                ax = fig.add_subplot(111, projection=wcs)
            
            # Plot with WCS coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add coordinate grid
            ax.grid(color='white', ls='solid', alpha=0.3)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
        except Exception as e:
            logger.warning(f"Error using WCS for plotting: {e}")
            # Fall back to basic plotting with rotation
            wcs = None
    
    if wcs is None:
        # Use simple physical scaling with rotation if needed
        ny, nx = data.shape
        
        # Default pixel sizes if not provided
        if pixel_size is None:
            pixel_size_x, pixel_size_y = 0.2, 0.2
        else:
            pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
        
        # Plot with extent to get physical scale
        x_min = -nx/2 * pixel_size_x
        x_max = nx/2 * pixel_size_x
        y_min = -ny/2 * pixel_size_y
        y_max = ny/2 * pixel_size_y
        
        extent = [x_min, x_max, y_min, y_max]
        im = ax.imshow(masked_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=extent, aspect='equal')
        
        # Apply rotation to axes labels if needed
        if rot_angle != 0:
            # Convert to radians
            theta = np.radians(rot_angle)
            
            # Create rotated axis labels
            x_label = f'Δ RA cos({rot_angle:.0f}°) + Δ DEC sin({rot_angle:.0f}°) (arcsec)'
            y_label = f'-Δ RA sin({rot_angle:.0f}°) + Δ DEC cos({rot_angle:.0f}°) (arcsec)'
        else:
            x_label = 'Δ RA (arcsec)'
            y_label = 'Δ DEC (arcsec)'
            
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    # Add title
    if title:
        ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_real_pixels(data, wcs, ax=None, title=None, cmap='viridis', vmin=None, vmax=None, 
                    colorbar_label=None, show_grid=False):
    """
    Plot 2D data with each pixel at its actual RA/DEC position
    
    Parameters
    ----------
    data : ndarray
        2D array of data to plot
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap
    vmin, vmax : float, optional
        Color scale limits
    colorbar_label : str, optional
        Label for colorbar
    show_grid : bool, default=False
        Whether to show grid lines
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create masked array for NaN values
    masked_data = np.ma.array(data, mask=~np.isfinite(data))
    
    # Set color limits
    if vmin is None:
        vmin = np.ma.min(masked_data)
    if vmax is None:
        vmax = np.ma.max(masked_data)
    
    # Get data dimensions
    ny, nx = data.shape
    
    # Create arrays for pixel corners
    # For each pixel, we need the four corners in pixel coordinates
    patches = []
    colors = []
    
    for j in range(ny):
        for i in range(nx):
            if np.isfinite(data[j, i]):
                # Pixel corners in pixel coordinates
                corners = [
                    [i - 0.5, j - 0.5],  # bottom left
                    [i + 0.5, j - 0.5],  # bottom right
                    [i + 0.5, j + 0.5],  # top right
                    [i - 0.5, j + 0.5]   # top left
                ]
                
                # Convert pixel corners to RA, DEC
                ra_dec_corners = []
                for x, y in corners:
                    ra, dec = wcs.wcs_pix2world(x, y, 0)
                    ra_dec_corners.append([ra, dec])
                
                # Create polygon patch
                patch = Polygon(ra_dec_corners, closed=True, edgecolor='gray' if show_grid else 'none', 
                               linewidth=0.2 if show_grid else 0)
                patches.append(patch)
                colors.append(data[j, i])
    
    # Create patch collection
    if patches:
        collection = PatchCollection(patches, cmap=cmap, alpha=1, edgecolor='gray' if show_grid else 'none',
                                    linewidth=0.2 if show_grid else 0)
        collection.set_array(np.array(colors))
        collection.set_clim(vmin, vmax)
        
        # Add collection to axis
        ax.add_collection(collection)
        
        # Add colorbar
        cbar = plt.colorbar(collection, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    else:
        ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
               ha='center', va='center')
    
    # Set axis limits to show all patches
    ax.autoscale_view()
    
    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Set axis labels
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('DEC (deg)')
    
    # Add title
    if title:
        ax.set_title(title)
    
    return ax


def plot_with_wcs_grid(data, wcs, ax=None, title=None, cmap='viridis', vmin=None, vmax=None, 
                      colorbar_label=None, show_grid=True, pixel_size=None):
    """
    Plot 2D data with proper WCS-based pixel grid to show RA/DEC coordinates
    
    Parameters
    ----------
    data : ndarray
        2D array of data to plot
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap name
    vmin, vmax : float, optional
        Minimum and maximum values for colorbar
    colorbar_label : str, optional
        Label for colorbar
    show_grid : bool, default=True
        Whether to show the coordinate grid
    pixel_size : tuple, optional
        Fallback pixel size (x, y) in arcsec if WCS isn't available
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create masked array for NaN values
    masked_data = np.ma.array(data, mask=~np.isfinite(data))
    
    # Set color limits
    if vmin is None and np.any(~masked_data.mask):
        vmin = np.ma.min(masked_data)
    if vmax is None and np.any(~masked_data.mask):
        vmax = np.ma.max(masked_data)
    
    try:
        # Use WCS for plotting
        if wcs is not None:
            # Import astropy's WCSAxes for coordinate transformation
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create new axis with WCS projection if needed
            if not isinstance(ax, WCSAxes):
                fig = ax.figure
                ax.remove()
                ax = fig.add_subplot(111, projection=wcs)
            
            # Plot data in WCS coordinates
            im = ax.imshow(masked_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add coordinate grid
            if show_grid:
                ax.grid(color='white', ls='solid', alpha=0.3)
                
            # Set axis labels to RA/DEC
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            
            # Let WCSAxes handle coordinate display automatically
        else:
            # Fallback to physical scaling without WCS
            if pixel_size is not None:
                # Use physical coordinates based on pixel size
                ny, nx = data.shape
                pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
                
                # Create physical coordinate grid centered on IFU center
                extent = [
                    -nx/2 * pixel_size_x, 
                    nx/2 * pixel_size_x, 
                    -ny/2 * pixel_size_y, 
                    ny/2 * pixel_size_y
                ]
                
                im = ax.imshow(
                    masked_data,
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    aspect='equal'
                )
                
                # Set axis labels
                ax.set_xlabel('Δ RA (arcsec)')
                ax.set_ylabel('Δ Dec (arcsec)')
                
                # Add grid if requested
                if show_grid:
                    ax.grid(alpha=0.3)
            else:
                # Simple pixel coordinates if no WCS or pixel size
                im = ax.imshow(
                    masked_data,
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Pixels')
    
    except Exception as e:
        logger.warning(f"Error in WCS plotting: {e}")
        # Fallback to simple imshow
        im = ax.imshow(
            masked_data,
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Pixels')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    # Add title
    if title:
        ax.set_title(title)
    
    return ax

def plot_flux_with_radial_bins(
    flux_map,
    bin_radii=None,
    center_x=None,
    center_y=None,
    pa=0,
    ellipticity=0,
    wcs=None,
    pixel_size=None,
    ax=None,
    title=None,
    cmap='inferno',
    log_scale=True,
):
    """
    Plot flux map with overlaid radial bins to visualize the physical radius calculation
    
    Parameters
    ----------
    flux_map : ndarray
        2D array of flux values
    bin_radii : array-like, optional
        Radii of bins in arcsec
    center_x, center_y : float, optional
        Center coordinates in pixels
    pa : float, default=0
        Position angle in degrees
    ellipticity : float, default=0
        Ellipticity (0-1)
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='inferno'
        Colormap
    log_scale : bool, default=True
        Whether to use log scale for flux
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create masked array for NaN values
    masked_flux = np.ma.array(flux_map, mask=~np.isfinite(flux_map))
    
    # Apply log scale if requested and data is positive
    if log_scale and np.any(masked_flux > 0):
        # Create mask for non-positive values
        pos_mask = masked_flux > 0
        if np.any(pos_mask):
            # Apply log10 to positive values
            log_data = np.log10(masked_flux.data)
            log_data[~pos_mask.data] = np.nan
            masked_flux = np.ma.array(log_data, mask=~pos_mask)
            colorbar_label = "Log10(Flux)"
        else:
            colorbar_label = "Flux"
    else:
        colorbar_label = "Flux"
    
    # Use WCS if available
    if wcs is not None:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create WCS axis if needed
            if not isinstance(ax, WCSAxes):
                fig = ax.figure
                ax.remove()
                ax = fig.add_subplot(111, projection=wcs)
            
            # Plot flux with WCS coordinates
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap)
            
            # Add radial bin ellipses if provided
            if bin_radii is not None and center_x is not None and center_y is not None:
                # Convert center pixel coordinates to world coordinates
                center_ra, center_dec = wcs.wcs_pix2world(center_x, center_y, 0)
                
                # Try to estimate pixel scale for proper scaling of ellipses
                try:
                    # Get pixel scale near center (may vary across field)
                    pixel_scale_x = np.abs(wcs.wcs.cd[0, 0]) * 3600  # deg to arcsec
                    pixel_scale_y = np.abs(wcs.wcs.cd[1, 1]) * 3600
                    
                    # Average pixel scale in arcsec for radius conversion
                    pixel_scale = (pixel_scale_x + pixel_scale_y) / 2
                    
                    # Get WCS rotation angle
                    wcs_pa = np.degrees(np.arctan2(wcs.wcs.cd[0, 1], wcs.wcs.cd[0, 0]))
                    
                    # Add total PA to WCS PA
                    total_pa = pa + wcs_pa
                    
                    for radius in bin_radii:
                        # Convert radius to degrees (WCS units)
                        radius_deg = radius / 3600  # arcsec to deg
                        
                        # Create ellipse
                        from matplotlib.patches import Ellipse
                        ell = Ellipse(
                            (center_ra, center_dec),
                            2 * radius_deg,  # major axis (diameter)
                            2 * radius_deg * (1 - ellipticity),  # minor axis
                            angle=total_pa,
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=1,
                            alpha=0.7,
                            transform=ax.get_transform('world')
                        )
                        ax.add_patch(ell)
                
                except Exception as e:
                    logger.warning(f"Could not draw ellipses in WCS space: {e}")
            
            # Set axis labels
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.grid(True, color='white', ls='solid', alpha=0.2)
        
        except Exception as e:
            logger.warning(f"Error using WCS for plotting: {e}")
            # Fall back to regular plotting
            wcs = None
    
    # Standard plotting if WCS is not available
    if wcs is None:
        # Use physical pixel size if provided
        if pixel_size is not None:
            ny, nx = flux_map.shape
            pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
            
            # Create physical coordinate grid
            extent = [
                -nx/2 * pixel_size_x, 
                nx/2 * pixel_size_x, 
                -ny/2 * pixel_size_y, 
                ny/2 * pixel_size_y
            ]
            
            # Plot with physical coordinates
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap, extent=extent)
            
            # Add radial bin ellipses if provided
            if bin_radii is not None and center_x is not None and center_y is not None:
                # Convert center pixel coordinates to physical coordinates
                center_phys_x = (center_x - nx/2) * pixel_size_x
                center_phys_y = (center_y - ny/2) * pixel_size_y
                
                for radius in bin_radii:
                    # Create ellipse
                    from matplotlib.patches import Ellipse
                    ell = Ellipse(
                        (center_phys_x, center_phys_y),
                        2 * radius,  # major axis (diameter)
                        2 * radius * (1 - ellipticity),  # minor axis
                        angle=pa,
                        fill=False,
                        edgecolor='white',
                        linestyle='-',
                        linewidth=1,
                        alpha=0.7
                    )
                    ax.add_patch(ell)
            
            # Set axis labels
            ax.set_xlabel('Δ RA (arcsec)')
            ax.set_ylabel('Δ Dec (arcsec)')
        
        else:
            # Simple pixel coordinates
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap)
            
            # Add radial bin ellipses if provided
            if bin_radii is not None and center_x is not None and center_y is not None and pixel_size is not None:
                pixel_size_x, pixel_size_y = pixel_size
                
                for radius in bin_radii:
                    # Convert radius from arcsec to pixels
                    radius_px_x = radius / pixel_size_x
                    radius_px_y = radius / pixel_size_y
                    
                    # Average radius in pixels
                    radius_px = (radius_px_x + radius_px_y) / 2
                    
                    # Create ellipse
                    from matplotlib.patches import Ellipse
                    ell = Ellipse(
                        (center_x, center_y),
                        2 * radius_px,  # major axis (diameter)
                        2 * radius_px * (1 - ellipticity),  # minor axis
                        angle=pa,
                        fill=False,
                        edgecolor='white',
                        linestyle='-',
                        linewidth=1,
                        alpha=0.7
                    )
                    ax.add_patch(ell)
            
            # Set axis labels
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Galaxy Flux with Radial Bins')
    # standardize_figure_saving()
    plt.close(fig)
    return ax


def plot_binning_on_flux(
    flux_map, 
    bin_num, 
    bin_indices=None, 
    bin_centers=None, 
    bin_radii=None,
    center_x=None, 
    center_y=None, 
    pa=0, 
    ellipticity=0, 
    title=None, 
    cmap='inferno', 
    save_path=None, 
    binning_type='Voronoi',
    wcs=None,
    pixel_size=None,
    show_numbers=True,
    log_scale=True
):
    """
    Plot binning boundaries overlaid on a flux map
    
    Parameters
    ----------
    flux_map : ndarray
        2D array of flux values
    bin_num : ndarray
        2D array of bin numbers
    bin_indices : list, optional
        List of arrays with indices for each bin
    bin_centers : tuple, optional
        (x_centers, y_centers) coordinates of bin centers
    bin_radii : ndarray, optional
        Radii for radial bins (for RDB)
    center_x, center_y : float, optional
        Center coordinates for radial bins
    pa : float, default=0
        Position angle in degrees (for radial bins)
    ellipticity : float, default=0
        Ellipticity (0-1) (for radial bins)
    title : str, optional
        Plot title
    cmap : str, default='inferno'
        Colormap for flux map
    save_path : str or Path, optional
        Path to save figure
    binning_type : str, default='Voronoi'
        Type of binning: 'Voronoi' or 'Radial'
    wcs : astropy.wcs.WCS, optional
        WCS object for coordinate transformation
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcsec
    show_numbers : bool, default=True
        Show bin numbers on plot
    log_scale : bool, default=True
        Use logarithmic scale for flux values
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Polygon, Ellipse
    import numpy as np
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Process flux map for plotting
    if flux_map is None:
        # Create dummy flux map if not provided
        flux_map = np.ones_like(bin_num, dtype=float)
    
    # Handle NaN values and mask
    masked_flux = np.ma.array(flux_map, mask=~np.isfinite(flux_map))
    
    # Determine color normalization
    valid_flux = masked_flux.compressed()
    if len(valid_flux) > 0:
        if log_scale and np.any(valid_flux > 0):
            # Logarithmic scale
            min_positive = np.min(valid_flux[valid_flux > 0])
            norm = LogNorm(vmin=min_positive, vmax=np.max(valid_flux))
        else:
            # Linear scale
            vmin = np.percentile(valid_flux, 2)
            vmax = np.percentile(valid_flux, 98)
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        # Fallback
        norm = Normalize(vmin=0, vmax=1)
    



    if wcs is not None:
        try:
            # *** FIX: Handle WCS with more than 2 dimensions by slicing ***
            from astropy.wcs import WCS
            
            if wcs.naxis > 2:
                # Create a new 2D WCS from the spatial dimensions
                # Get the celestial axes indices (usually 0,1 for 2D images)
                if hasattr(wcs, 'wcs'):  # For newer astropy versions
                    celestial = wcs.wcs.get_axis_types()
                    spatial_axes = [i for i, ax in enumerate(celestial) 
                                   if ax['coordinate_type'] == 'celestial']
                    if len(spatial_axes) >= 2:
                        # Create a new 2D WCS with just the spatial dimensions
                        wcs_2d = wcs.celestial
                    else:
                        # Fallback: slice the first two dimensions
                        wcs_2d = wcs.slice([0, 1])
                else:
                    # Simple slice of the first two dimensions
                    wcs_2d = wcs.slice([0, 1])
            else:
                wcs_2d = wcs
                
            # Use astropy WCS for plotting
            from astropy.visualization.wcsaxes import WCSAxes
            
            # Create new axis with WCS projection if needed
            if not isinstance(ax, WCSAxes):
                old_pos = ax.get_position()
                ax.remove()
                ax = fig.add_subplot(111, projection=wcs_2d)
                ax.set_position(old_pos)  # Maintain position
            
            # Plot flux map with WCS coordinates
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap, norm=norm)
            
            # Add coordinate grid
            ax.grid(color='white', ls='solid', alpha=0.3)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
        except Exception as e:
            logger.warning(f"Error using WCS for plotting: {e}")
            # Fall back to regular plotting
            wcs = None
            
    if wcs is None:
        # Standard plotting with physical coordinates if available
        if pixel_size is not None:
            ny, nx = flux_map.shape
            pixel_size_x, pixel_size_y = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
            
            # Create coordinate grid centered on image
            extent = [
                -nx/2 * pixel_size_x, 
                nx/2 * pixel_size_x, 
                -ny/2 * pixel_size_y, 
                ny/2 * pixel_size_y
            ]
            
            im = ax.imshow(
                masked_flux, 
                origin='lower', 
                cmap=cmap, 
                norm=norm,
                extent=extent, 
                aspect='equal'
            )
            
            # Set axis labels
            ax.set_xlabel('Δ RA (arcsec)')
            ax.set_ylabel('Δ Dec (arcsec)')
        else:
            # Simple pixel coordinates
            im = ax.imshow(masked_flux, origin='lower', cmap=cmap, norm=norm)
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flux' + (' (log scale)' if log_scale else ''))
    
    # Overlay bin boundaries based on binning type
    if binning_type.lower() == 'voronoi':
        # For Voronoi binning, we need to find bin boundaries
        ny, nx = bin_num.shape
        
        # Create edge map - detect edges between different bins
        edge_map = np.zeros((ny, nx), dtype=bool)
        
        # Check horizontal edges
        for j in range(ny):
            for i in range(nx-1):
                if bin_num[j, i] != bin_num[j, i+1] and bin_num[j, i] >= 0 and bin_num[j, i+1] >= 0:
                    edge_map[j, i] = True
                    edge_map[j, i+1] = True
        
        # Check vertical edges
        for j in range(ny-1):
            for i in range(nx):
                if bin_num[j, i] != bin_num[j+1, i] and bin_num[j, i] >= 0 and bin_num[j+1, i] >= 0:
                    edge_map[j, i] = True
                    edge_map[j+1, i] = True
        
        # Plot edges
        y_edge, x_edge = np.where(edge_map)
        
        # Convert to physical coordinates if needed
        if pixel_size is not None:
            x_edge_phys = (x_edge - nx/2) * pixel_size_x
            y_edge_phys = (y_edge - ny/2) * pixel_size_y
            ax.scatter(x_edge_phys, y_edge_phys, s=1, color='white', alpha=0.7)
        else:
            ax.scatter(x_edge, y_edge, s=1, color='white', alpha=0.7)
        
        # Plot bin centers and numbers if requested
        if bin_centers is not None and show_numbers:
            x_centers, y_centers = bin_centers
            
            # Convert to physical coordinates if needed
            if pixel_size is not None:
                x_centers_phys = (x_centers - nx/2) * pixel_size_x
                y_centers_phys = (y_centers - ny/2) * pixel_size_y
                
                for i, (x, y) in enumerate(zip(x_centers_phys, y_centers_phys)):
                    ax.text(x, y, str(i), color='white', fontsize=8, 
                           ha='center', va='center', 
                           bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            else:
                for i, (x, y) in enumerate(zip(x_centers, y_centers)):
                    ax.text(x, y, str(i), color='white', fontsize=8, 
                           ha='center', va='center', 
                           bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    
    elif binning_type.lower() == 'radial':
        # For radial binning, overlay ellipses
        if bin_radii is not None and center_x is not None and center_y is not None:
            ny, nx = bin_num.shape
            
            # Convert center to physical coordinates if needed
            if pixel_size is not None:
                center_x_phys = (center_x - nx/2) * pixel_size_x
                center_y_phys = (center_y - ny/2) * pixel_size_y
            else:
                center_x_phys, center_y_phys = center_x, center_y
            
            # Draw ellipses for each bin radius
            for i, radius in enumerate(bin_radii):
                if pixel_size is not None:
                    # Scale radius to physical units for plotting
                    # For radial binning, radius should already be in arcsec
                    
                    # Create ellipse based on PA and ellipticity
                    if ellipticity == 0 or not np.isfinite(ellipticity):
                        # Draw circle
                        ellipse = Ellipse(
                            (center_x_phys, center_y_phys),
                            2 * radius,  # Diameter
                            2 * radius,
                            angle=0,
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=1,
                            alpha=0.7
                        )
                    else:
                        # Draw ellipse
                        ellipse = Ellipse(
                            (center_x_phys, center_y_phys),
                            2 * radius,  # Major axis (diameter)
                            2 * radius * (1 - ellipticity),  # Minor axis
                            angle=pa,
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=1,
                            alpha=0.7
                        )
                    
                    ax.add_patch(ellipse)
                    
                    # Add bin number at specific angle
                    if show_numbers:
                        label_angle = np.radians(45)  # Place labels at 45 degrees
                        label_x = center_x_phys + radius * np.cos(label_angle)
                        label_y = center_y_phys + radius * np.sin(label_angle)
                        
                        ax.text(
                            label_x, 
                            label_y, 
                            str(i), 
                            color='white', 
                            fontsize=8,
                            ha='center', 
                            va='center',
                            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                        )
                else:
                    # Draw in pixel coordinates
                    if ellipticity == 0 or not np.isfinite(ellipticity):
                        # Draw circle in pixel units
                        ellipse = Ellipse(
                            (center_x, center_y),
                            2 * radius / pixel_size_x if pixel_size is not None else 2 * radius,
                            2 * radius / pixel_size_y if pixel_size is not None else 2 * radius,
                            angle=0,
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=1,
                            alpha=0.7
                        )
                    else:
                        # Draw ellipse in pixel units
                        ellipse = Ellipse(
                            (center_x, center_y),
                            2 * radius / pixel_size_x if pixel_size is not None else 2 * radius,
                            2 * radius / pixel_size_y * (1 - ellipticity) if pixel_size is not None else 2 * radius * (1 - ellipticity),
                            angle=pa,
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=1,
                            alpha=0.7
                        )
                    
                    ax.add_patch(ellipse)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{binning_type} Binning on Flux Map")
    
    # Save figure if requested
    if save_path:
        # Make sure the directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


