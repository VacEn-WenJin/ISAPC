"""
Spectral Binning Tools - Support for Voronoi binning and radial binning
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import utilities
from utils.calc import spectres

logger = logging.getLogger(__name__)

# Speed of light in km/s
C_KMS = 299792.458


@dataclass
class BinnedSpectra:
    """Base class for binned spectra data"""

    bin_num: np.ndarray  # Bin number for each pixel
    bin_indices: List[np.ndarray]  # List of arrays containing pixel indices for each bin
    spectra: np.ndarray  # 2D array of binned spectra [n_wavelength, n_bins]
    wavelength: np.ndarray  # 1D array of wavelength
    metadata: Dict  # Metadata dictionary

    def save(self, filename: Union[str, Path]) -> None:
        """Save binned data to file"""
        # Convert bin_indices to a list and wrap in object array to preserve heterogeneous sizes
        bin_indices_obj = np.empty(len(self.bin_indices), dtype=object)
        for i, indices in enumerate(self.bin_indices):
            bin_indices_obj[i] = indices

        np.savez_compressed(
            filename,
            bin_num=self.bin_num,
            bin_indices=bin_indices_obj,  # Store as object array
            spectra=self.spectra,
            wavelength=self.wavelength,
            metadata=self.metadata,
        )
        logger.info(f"Saved binned data to {filename}")

    @classmethod
    def load(cls, filename: Union[str, Path]):
        """Load binned data from file"""
        data = np.load(filename, allow_pickle=True)

        # Extract bin_indices as a list of arrays
        bin_indices = []
        for obj in data["bin_indices"]:
            bin_indices.append(obj)

        return cls(
            bin_num=data["bin_num"],
            bin_indices=bin_indices,  # Use the extracted list
            spectra=data["spectra"],
            wavelength=data["wavelength"],
            metadata=data["metadata"].item(),
        )

    def create_visualization_plots(self, output_dir, galaxy_name):
        """Create basic visualization plots for binned data"""
        from pathlib import Path
        import visualization
        
        plots_dir = Path(output_dir)
        plots_dir.mkdir(exist_ok=True, parents=True)

        # 1. Plot bin map
        try:
            # Get dimensions from metadata
            if "ny" in self.metadata and "nx" in self.metadata:
                ny, nx = self.metadata["ny"], self.metadata["nx"]
            else:
                # Estimate from bin_num shape
                ny, nx = (
                    self.bin_num.shape
                    if hasattr(self.bin_num, "shape")
                    else (1, len(self.bin_num))
                )

            # Create 2D bin map
            bin_map_2d = self.bin_num.reshape(ny, nx)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Get pixel size if available
            pixel_size = None
            if "pixelsize_x" in self.metadata and "pixelsize_y" in self.metadata:
                pixel_size = (self.metadata["pixelsize_x"], self.metadata["pixelsize_y"])
            
            visualization.plot_bin_map(
                bin_map_2d,
                ax=ax,
                cmap="tab20",
                title=f"{galaxy_name} - Binning Map",
                physical_scale=pixel_size is not None,
                pixel_size=pixel_size
            )

            # Save
            visualization.standardize_figure_saving(
                fig, plots_dir / f"{galaxy_name}_bin_map.png"
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create bin map visualization: {e}")

        # 2. Plot sample spectra
        try:
            # Choose a few bins to plot
            n_bins = self.spectra.shape[1]
            bins_to_plot = [0]  # Always plot first bin

            # Add some bins throughout the range
            if n_bins > 1:
                bins_to_plot.append(n_bins // 2)
            if n_bins > 2:
                bins_to_plot.append(n_bins - 1)

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))

            for bin_id in bins_to_plot:
                ax.plot(self.wavelength, self.spectra[:, bin_id], label=f"Bin {bin_id}")

            ax.set_xlabel("Wavelength (Å)")
            ax.set_ylabel("Flux")
            ax.set_title(f"{galaxy_name} - Sample Bin Spectra")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save
            visualization.standardize_figure_saving(
                fig, plots_dir / f"{galaxy_name}_sample_spectra.png"
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create sample spectra visualization: {e}")


@dataclass
class VoronoiBinnedData(BinnedSpectra):
    """Class for Voronoi binned spectra"""

    def create_visualization_plots(self, output_dir, galaxy_name):
        """Create Voronoi specific visualization plots"""
        import visualization
        from pathlib import Path
        
        # Call parent method first
        super().create_visualization_plots(output_dir, galaxy_name)

        plots_dir = Path(output_dir)
        plots_dir.mkdir(exist_ok=True, parents=True)

        # Add Voronoi specific plots
        try:
            # Plot SNR distribution
            if "sn" in self.metadata:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(self.metadata["sn"], bins=20, alpha=0.7)
                ax.axvline(
                    x=self.metadata.get("target_snr", 0),
                    color="r",
                    linestyle="--",
                    label=f"Target SNR: {self.metadata.get('target_snr', 0)}",
                )
                ax.set_xlabel("Signal-to-Noise Ratio")
                ax.set_ylabel("Number of Bins")
                ax.set_title(f"{galaxy_name} - Voronoi Bin SNR Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Save
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_voronoi_snr.png"
                )
                plt.close(fig)

            # Plot bin size distribution
            if "n_pixels" in self.metadata:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(self.metadata["n_pixels"], bins=20, alpha=0.7)
                ax.set_xlabel("Number of Pixels per Bin")
                ax.set_ylabel("Number of Bins")
                ax.set_title(f"{galaxy_name} - Voronoi Bin Size Distribution")
                ax.grid(True, alpha=0.3)

                # Save
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_voronoi_bin_size.png"
                )
                plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to create Voronoi visualization: {e}")
            plt.close("all")


@dataclass
class RadialBinnedData(BinnedSpectra):
    """Class for radial binned spectra"""

    bin_radii: np.ndarray  # Radius of each bin

    def create_visualization_plots(self, output_dir, galaxy_name):
        """Create radial specific visualization plots with physical coordinates"""
        import visualization
        from pathlib import Path
        
        # Call parent method first
        super().create_visualization_plots(output_dir, galaxy_name)

        plots_dir = Path(output_dir)
        plots_dir.mkdir(exist_ok=True, parents=True)

        # Add radial specific plots
        try:
            # Get pixel size for coordinate conversion
            pixel_size_x = self.metadata.get("pixelsize_x", 1.0)
            pixel_size_y = self.metadata.get("pixelsize_y", 1.0)
            pixel_size = (pixel_size_x, pixel_size_y)

            # Plot radius vs. bin number
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(np.arange(len(self.bin_radii)), self.bin_radii, "o-")
            ax.set_xlabel("Bin Number")
            ax.set_ylabel("Radius (arcsec)")
            ax.set_title(f"{galaxy_name} - Radial Bin Distribution")
            ax.grid(True, alpha=0.3)

            # Save
            visualization.standardize_figure_saving(
                fig, plots_dir / f"{galaxy_name}_radial_bins.png"
            )
            plt.close(fig)

            # Plot bin map with circles representing bins using physical coordinates
            try:
                # Get dimensions from bin_num
                if "ny" in self.metadata and "nx" in self.metadata:
                    ny, nx = self.metadata["ny"], self.metadata["nx"]
                else:
                    # Estimate from bin_num shape
                    ny, nx = (
                        self.bin_num.shape
                        if hasattr(self.bin_num, "shape")
                        else (1, len(self.bin_num))
                    )

                bin_map_2d = self.bin_num.reshape(ny, nx)
                center_x = self.metadata.get("center_x", nx / 2)
                center_y = self.metadata.get("center_y", ny / 2)
                pa = self.metadata.get("pa", 0)
                ellipticity = self.metadata.get("ellipticity", 0)

                # Get WCS if available
                wcs = self.metadata.get("wcs", None)

                # Create the visualization plot
                fig, ax = visualization.plot_flux_with_radial_bins(
                    bin_map_2d,  # Use bin map as a base
                    self.bin_radii,
                    center_x,
                    center_y,
                    pa=pa,
                    ellipticity=ellipticity,
                    wcs=wcs,
                    pixel_size=pixel_size,
                    title=f"{galaxy_name} - Radial Bin Map",
                )

                # Save
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_radial_bin_map.png"
                )
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to create radial bin map: {e}")
                plt.close("all")
        except Exception as e:
            logger.warning(f"Failed to create radial visualization: {e}")
            plt.close("all")


def calculate_wavelength_intersection(wavelength, velocity_field, n_x):
    """
    Calculate common wavelength range accounting for velocity shifts.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Original wavelength array
    velocity_field : numpy.ndarray
        Velocity field (2D array)
    n_x : int
        Number of pixels in x direction

    Returns
    -------
    tuple
        (mask, min_wave, max_wave)
    """
    c = 299792.458  # Speed of light in km/s

    # Find minimum and maximum velocities
    valid_velocities = velocity_field[np.isfinite(velocity_field)]
    if len(valid_velocities) == 0:
        # No valid velocities, return full range
        return (
            np.ones_like(wavelength, dtype=bool),
            np.min(wavelength),
            np.max(wavelength),
        )

    min_vel = np.min(valid_velocities)
    max_vel = np.max(valid_velocities)

    # Calculate wavelength limits - accounting for redshift/blueshift
    min_factor = 1 + min_vel / c  # For blueshift (v < 0), factor < 1
    max_factor = 1 + max_vel / c  # For redshift (v > 0), factor > 1

    # Determine range that works for all velocities
    if min_vel < 0 and max_vel > 0:
        # We have both blue and redshifts - more complex case
        # The range must be within both limits
        rest_min = np.min(wavelength) / max_factor  # Bluer limit for redshifts
        rest_max = np.max(wavelength) / min_factor  # Redder limit for blueshifts
    else:
        # Simpler case - all velocities in same direction
        rest_min = np.min(wavelength) / max(min_factor, max_factor)
        rest_max = np.max(wavelength) / min(min_factor, max_factor)

    # Add safety margin (1%)
    margin = 0.01 * (rest_max - rest_min)
    min_wave = rest_min + margin
    max_wave = rest_max - margin

    # Create mask for wavelength range
    mask = (wavelength >= min_wave) & (wavelength <= max_wave)

    # Ensure we have some valid wavelengths
    if np.sum(mask) < 10:
        # If too few wavelength points, use most of the original range
        logger.warning(
            "Velocity range too wide for wavelength intersection, using 80% of original range"
        )
        wlen = len(wavelength)
        start_idx = int(wlen * 0.1)
        end_idx = int(wlen * 0.9)
        mask = np.zeros_like(wavelength, dtype=bool)
        mask[start_idx:end_idx] = True
        min_wave = wavelength[start_idx]
        max_wave = wavelength[end_idx - 1]

    return mask, min_wave, max_wave


def combine_spectra_efficiently(
    spectra,
    wavelength,
    bin_indices,
    velocity_field=None,
    n_x=None,
    n_y=None,
    edge_treatment="extend",
    use_separate_velocity=True
):
    """
    Efficiently combine spectra into bins with improved velocity correction
    and support for separate stellar and gas velocities.

    Parameters
    ----------
    spectra : numpy.ndarray
        Array of spectra [n_wave, n_spectra]
    wavelength : numpy.ndarray
        Wavelength array
    bin_indices : list
        List of arrays with indices for each bin
    velocity_field : numpy.ndarray, optional
        Velocity field for correction
    n_x : int, optional
        Number of pixels in x direction
    n_y : int, optional
        Number of pixels in y direction
    edge_treatment : str, default='extend'
        How to handle spectrum edges:
        - 'extend': Extend edge values rather than filling with zeros
        - 'mask': Use NaN for edge values
        - 'zero': Fill with zeros (original behavior)
    use_separate_velocity : bool, default=True
        Whether to use separate stellar and gas velocities when available
        
    Returns
    -------
    numpy.ndarray
        Combined bin spectra array [n_wave, n_bins]
    """
    n_wave = len(wavelength)
    n_bins = len(bin_indices)
    c = 299792.458  # Speed of light in km/s

    # Initialize output array
    bin_spectra = np.zeros((n_wave, n_bins))

    # Check if velocity correction is requested and available
    do_correction = velocity_field is not None and n_x is not None

    # Process each bin
    for i, indices in enumerate(bin_indices):
        # Skip empty bins
        if len(indices) == 0:
            bin_spectra[:, i] = np.nan
            continue

        # Skip bins with invalid indices
        if np.max(indices) >= spectra.shape[1]:
            logger.warning(f"Bin {i} has indices beyond spectra shape, skipping")
            bin_spectra[:, i] = np.nan
            continue

        try:
            # Extract spectra for this bin
            bin_spectra_list = []

            # Try alternate velocity correction methods if one fails
            velocity_correction_success = False

            # First attempt: Calculate median velocity for the bin
            if do_correction:
                try:
                    # Get velocity field for this bin
                    bin_velocities = []
                    for idx in indices:
                        row = idx // n_x
                        col = idx % n_x
                        if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                            vel = velocity_field[row, col]
                            if np.isfinite(vel):
                                bin_velocities.append(vel)

                    # Check for gas velocity field if available and use_separate_velocity is enabled
                    gas_velocity_field = None
                    bin_gas_velocities = []
                    
                    if use_separate_velocity and hasattr(velocity_field, 'gas_velocity_field'):
                        gas_velocity_field = velocity_field.gas_velocity_field
                        
                        # Get gas velocities for this bin
                        for idx in indices:
                            row = idx // n_x
                            col = idx % n_x
                            if (row < gas_velocity_field.shape[0] and 
                                col < gas_velocity_field.shape[1]):
                                gas_vel = gas_velocity_field[row, col]
                                if np.isfinite(gas_vel):
                                    bin_gas_velocities.append(gas_vel)
                    
                    # Use median velocity for the bin if we have valid velocities
                    if bin_velocities:
                        median_velocity = np.median(bin_velocities)
                        
                        # Get median gas velocity if available
                        median_gas_velocity = (
                            np.median(bin_gas_velocities) if bin_gas_velocities 
                            else median_velocity
                        )

                        # Only apply correction for non-zero velocities
                        if abs(median_velocity) > 1.0:  # Minimum 1 km/s to apply correction
                            # Combine all spectra first, then apply correction once
                            combined_spectrum = np.zeros(n_wave)
                            weight_sum = 0

                            for idx in indices:
                                spec = spectra[:, idx]
                                if not np.all(~np.isfinite(spec)):
                                    combined_spectrum += spec
                                    weight_sum += 1

                            if weight_sum > 0:
                                combined_spectrum /= weight_sum

                                # Apply velocity shift using Doppler formula
                                lam_shifted = wavelength * (1 + median_velocity / c)

                                # Use spectres for resampling with edge handling
                                try:
                                    if edge_treatment == "extend":
                                        # Use nearest valid values for edges
                                        corrected_spectrum = spectres(
                                            wavelength,
                                            lam_shifted,
                                            combined_spectrum,
                                            fill=None,
                                            preserve_edges=True,
                                        )
                                    elif edge_treatment == "mask":
                                        # Use NaN for edges
                                        corrected_spectrum = spectres(
                                            wavelength,
                                            lam_shifted,
                                            combined_spectrum,
                                            fill=np.nan,
                                            preserve_edges=False,
                                        )
                                    else:
                                        # Original behavior - fill with zeros
                                        corrected_spectrum = spectres(
                                            wavelength,
                                            lam_shifted,
                                            combined_spectrum,
                                            fill=0.0,
                                            preserve_edges=False,
                                        )

                                    bin_spectra[:, i] = corrected_spectrum
                                    velocity_correction_success = True
                                    continue  # Skip to next bin
                                except Exception as e:
                                    logger.debug(f"Bulk velocity correction failed: {e}")
                except Exception as e:
                    logger.debug(f"Bin velocity correction failed: {e}")

            # Second attempt: Process each spectrum individually if first method failed
            if not velocity_correction_success:
                # Process each spectrum individually
                individual_velocity_data = []
                
                for idx in indices:
                    # Get spectrum
                    spectrum = spectra[:, idx]

                    # Skip if all NaN
                    if np.all(~np.isfinite(spectrum)):
                        continue

                    # Apply velocity correction if available
                    if do_correction:
                        try:
                            row = idx // n_x
                            col = idx % n_x

                            # Check bounds
                            if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                                vel = velocity_field[row, col]

                                # Only correct if velocity is valid and non-negligible
                                if np.isfinite(vel) and abs(vel) > 1.0:
                                    # Apply velocity shift using Doppler formula
                                    lam_shifted = wavelength * (1 + vel / c)

                                    try:
                                        # Use spectres for resampling with improved edge handling
                                        if edge_treatment == "extend":
                                            corrected_spectrum = spectres(
                                                wavelength,
                                                lam_shifted,
                                                spectrum,
                                                fill=None,
                                                preserve_edges=True,
                                            )
                                        elif edge_treatment == "mask":
                                            # Use NaN values at edges
                                            corrected_spectrum = spectres(
                                                wavelength,
                                                lam_shifted,
                                                spectrum,
                                                fill=np.nan,
                                                preserve_edges=False,
                                            )
                                        else:
                                            # Original behavior
                                            corrected_spectrum = spectres(
                                                wavelength,
                                                lam_shifted,
                                                spectrum,
                                                fill=0.0,
                                                preserve_edges=False,
                                            )
                                                                                
                                        # Store velocity-corrected spectrum
                                        individual_velocity_data.append({
                                            'spectrum': corrected_spectrum, 
                                            'velocity': vel
                                        })
                                        continue  # Skip the default append
                                    except Exception as e:
                                        # Fall through to add original spectrum
                                        logger.debug(f"Individual velocity correction failed: {e}")
                        except Exception as e:
                            # Fall through to add original spectrum
                            pass

                    # Add the original spectrum if no velocity correction
                    bin_spectra_list.append(spectrum)

                # First try to use velocity-corrected spectra if available
                if individual_velocity_data:
                    try:
                        # Extract spectra and velocities
                        vel_corrected_spectra = [data['spectrum'] for data in individual_velocity_data]
                        velocities = [data['velocity'] for data in individual_velocity_data]
                        
                        # Stack into array
                        spectra_array = np.array(vel_corrected_spectra)
                        
                        # Calculate median velocity-corrected spectrum
                        bin_spectra[:, i] = np.nanmedian(spectra_array, axis=0)
                        
                        # Handle NaN wavelengths
                        all_nan = np.all(~np.isfinite(spectra_array), axis=0)
                        bin_spectra[all_nan, i] = np.nan
                        
                        # Log velocity statistics
                        if len(velocities) > 1:
                            vel_median = np.median(velocities)
                            vel_std = np.std(velocities)
                            if vel_std > 20:  # If velocity dispersion is high
                                logger.debug(f"Bin {i} has high velocity dispersion: {vel_std:.1f} km/s")
                    except Exception as e:
                        logger.debug(f"Failed to combine velocity-corrected spectra: {e}")
                        # Fall back to regular spectra
                        if bin_spectra_list:
                            spectra_array = np.array(bin_spectra_list)
                            bin_spectra[:, i] = np.nanmedian(spectra_array, axis=0)
                            all_nan = np.all(~np.isfinite(spectra_array), axis=0)
                            bin_spectra[all_nan, i] = np.nan
                        else:
                            bin_spectra[:, i] = np.nan
                # Use regular spectra if no velocity data
                elif bin_spectra_list:
                    # Convert to array for easier operations
                    spectra_array = np.array(bin_spectra_list)

                    # Compute median spectrum - more robust than mean
                    bin_spectra[:, i] = np.nanmedian(spectra_array, axis=0)

                    # Set all-NaN wavelengths to NaN in result
                    all_nan = np.all(~np.isfinite(spectra_array), axis=0)
                    bin_spectra[all_nan, i] = np.nan
                else:
                    # No valid spectra
                    bin_spectra[:, i] = np.nan

        except Exception as e:
            logger.error(f"Error combining spectra for bin {i}: {e}")
            bin_spectra[:, i] = np.nan

    return bin_spectra


def optimize_voronoi_binning(
    x, y, signal, noise, target_bin_count=15, min_bins=10, max_bins=20, 
    snr_range=(10, 100), cvt=True, quiet=False
):
    """
    Optimize Voronoi binning to achieve a target number of bins
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    signal : ndarray
        Signal data
    noise : ndarray
        Noise data
    target_bin_count : int, default=15
        Target number of bins (middle of range)
    min_bins : int, default=10
        Minimum acceptable number of bins
    max_bins : int, default=20
        Maximum acceptable number of bins
    snr_range : tuple, default=(10, 100)
        Initial SNR range to search (min, max)
    cvt : bool, default=True
        Whether to use CVT in Voronoi binning
    quiet : bool, default=False
        Whether to suppress output
    
    Returns
    -------
    tuple
        (bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale)
        Same return values as run_voronoi_binning
    """
    from vorbin.voronoi_2d_binning import voronoi_2d_binning
    import warnings

    logger.info(f"Optimizing Voronoi binning for {target_bin_count} bins (range: {min_bins}-{max_bins})")
    
    # Find valid pixels
    valid_mask = (
        np.isfinite(x) & np.isfinite(y) & 
        np.isfinite(signal) & np.isfinite(noise) & 
        (noise > 0)
    )
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    signal_valid = signal[valid_mask]
    noise_valid = noise[valid_mask]
    
    if not quiet:
        logger.info(f"Using {np.sum(valid_mask)} valid pixels for optimization")
    
    if np.sum(valid_mask) < 3:
        logger.warning(f"Too few valid pixels ({np.sum(valid_mask)}) for binning optimization")
        # Fall back to standard binning with target_snr=30
        return run_voronoi_binning(x, y, signal, noise, 30, 0, True, 1.0)
    
    # Calculate pixel SNR statistics
    pixel_snr = signal_valid / noise_valid
    median_snr = np.median(pixel_snr)
    max_snr = np.max(pixel_snr)
    
    if not quiet:
        logger.info(f"Pixel SNR - Median: {median_snr:.1f}, Max: {max_snr:.1f}")
    
    # Initialize search parameters
    min_snr, max_snr = snr_range
    
    # Adjust initial SNR range based on pixel SNR statistics
    if max_snr < median_snr * 2:
        min_snr = median_snr * 1.1
        max_snr = median_snr * 10
    
    # Try binary search to find optimal target_snr
    num_iterations = 10  # Maximum iterations for binary search
    best_result = None
    best_bin_count = 0
    
    for iteration in range(num_iterations):
        # Calculate midpoint SNR
        target_snr = (min_snr + max_snr) / 2
        
        if not quiet and iteration > 0:
            logger.info(f"Iteration {iteration}: Trying SNR={target_snr:.1f} (range: {min_snr:.1f}-{max_snr:.1f})")
        
        # Run Voronoi binning with current target_snr
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                
                try:
                    # Try with provided CVT setting
                    binning_result = voronoi_2d_binning(
                        x_valid, y_valid, signal_valid, noise_valid,
                        target_snr, cvt=cvt, plot=0, quiet=True, wvt=True
                    )
                except Exception:
                    # Fall back to attempt without CVT
                    binning_result = voronoi_2d_binning(
                        x_valid, y_valid, signal_valid, noise_valid,
                        target_snr, cvt=False, plot=0, quiet=True, wvt=True
                    )
                
                # Extract bin_num from result to count unique bins
                if isinstance(binning_result, tuple) and len(binning_result) >= 1:
                    bin_num = binning_result[0]
                    unique_bins = np.unique(bin_num[bin_num >= 0])
                    bin_count = len(unique_bins)
                    
                    if not quiet:
                        logger.info(f"SNR={target_snr:.1f} produced {bin_count} bins")
                    
                    # Check if this is the best result so far
                    if min_bins <= bin_count <= max_bins:
                        # Score based on distance from target
                        current_score = abs(bin_count - target_bin_count)
                        best_score = abs(best_bin_count - target_bin_count) if best_bin_count > 0 else float('inf')
                        
                        if best_result is None or current_score < best_score:
                            best_result = binning_result
                            best_bin_count = bin_count
                            
                            if not quiet:
                                logger.info(f"New best result: {bin_count} bins with SNR={target_snr:.1f}")
                    
                    # Update search range
                    if bin_count > target_bin_count:
                        # Too many bins, increase SNR
                        min_snr = target_snr
                    else:
                        # Too few bins, decrease SNR
                        max_snr = target_snr
                    
                    # Check for convergence
                    if bin_count == target_bin_count or abs(min_snr - max_snr) < 0.5:
                        if not quiet:
                            logger.info(f"Converged on {bin_count} bins with SNR={target_snr:.1f}")
                        break
                else:
                    # Invalid result, adjust SNR and continue
                    max_snr = target_snr  # Try a lower SNR next time
                    if not quiet:
                        logger.warning("Invalid binning result, trying lower SNR")
        except Exception as e:
            # Error in binning, adjust SNR and continue
            if not quiet:
                logger.warning(f"Error in binning with SNR={target_snr:.1f}: {e}")
            max_snr = target_snr  # Try a lower SNR next time
    
    # If we found a good result, return it
    if best_result is not None:
        if not quiet:
            logger.info(f"Final optimization result: {best_bin_count} bins")
        
        # Map bin numbers back to original shape
        bin_num_orig = np.full_like(x, -1, dtype=int)
        if isinstance(best_result, tuple) and len(best_result) >= 1:
            bin_num_orig[valid_mask] = best_result[0]
            
            # Prepare remaining outputs
            if len(best_result) >= 8:
                bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = best_result
                return bin_num_orig, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale
            elif len(best_result) >= 6:
                bin_num, x_gen, y_gen, sn, n_pixels, scale = best_result
                # Create dummy x_bar, y_bar
                x_bar, y_bar = x_gen.copy(), y_gen.copy()
                return bin_num_orig, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale
    
    # If optimization failed, try once more with a fixed target_snr
    if not quiet:
        logger.warning("Optimization failed, falling back to standard binning")
    
    # Use fallback target_snr based on median pixel SNR
    fallback_snr = max(15, min(50, median_snr * 3))
    if not quiet:
        logger.info(f"Using fallback SNR={fallback_snr:.1f}")
    
    return run_voronoi_binning(x, y, signal, noise, fallback_snr, 0, False, 1.0)


def run_voronoi_binning(
    x, y, signal, noise, target_snr, plot=0, quiet=False, cvt=True, min_snr=0.0
):
    """
    Run Voronoi binning on input data with comprehensive error handling and adaptive SNR.

    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    signal : ndarray
        Signal data
    noise : ndarray
        Noise data
    target_snr : float
        Target signal-to-noise ratio. Recommended to be between:
        - Minimum: 1.5 × median pixel SNR
        - Maximum: 50 × maximum pixel SNR
    plot : int, default=0
        Plotting flag (0=no plot, 1=plot)
    quiet : bool, default=False
        Quiet mode
    cvt : bool, default=True
        Use CVT optimization
    min_snr : float, default=0.0
        Minimum SNR for valid bins

    Returns
    -------
    tuple
        Binning results (bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale)
        - bin_num: array of bin numbers for each pixel
        - x_gen, y_gen: coordinates of bin generators
        - x_bar, y_bar: luminosity-weighted centroids of bins
        - sn: signal-to-noise ratio of each bin
        - n_pixels: number of pixels in each bin
        - scale: scale length of the bins
    """
    try:
        # Try to import vorbin
        import warnings

        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        # Silence numpy warnings during binning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Ensure data is valid
            valid_mask = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(signal)
                & np.isfinite(noise)
                & (noise > 0)
            )
            if not quiet:
                logger.info(
                    f"Using {np.sum(valid_mask)} valid pixels out of {len(x)} for Voronoi binning"
                )

            if np.sum(valid_mask) < 3:
                raise ValueError(
                    f"Too few valid pixels ({np.sum(valid_mask)}) for Voronoi binning"
                )

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            signal_valid = signal[valid_mask]
            noise_valid = noise[valid_mask]

            # Calculate current median and max SNR
            pixel_snr = signal_valid / noise_valid
            median_snr = np.median(pixel_snr)
            max_pixel_snr = np.max(pixel_snr)

            if not quiet:
                logger.info(
                    f"Median SNR in data: {median_snr:.1f}, Maximum SNR: {max_pixel_snr:.1f}"
                )

            # Check if the target SNR is reasonable based on data quality
            # Recommended range: 1.5× to 50× maximum pixel SNR
            min_recommended = median_snr * 1.5
            max_recommended = max_pixel_snr * 50.0

            if target_snr < min_recommended:
                if not quiet:
                    logger.warning(
                        f"Target SNR ({target_snr:.1f}) is lower than recommended minimum ({min_recommended:.1f})"
                    )
                    logger.info(
                        f"Recommended range is {min_recommended:.1f} to {max_recommended:.1f}"
                    )
            elif target_snr > max_recommended:
                if not quiet:
                    logger.warning(
                        f"Target SNR ({target_snr:.1f}) is higher than recommended maximum ({max_recommended:.1f})"
                    )
                    logger.info(
                        f"Recommended range is {min_recommended:.1f} to {max_recommended:.1f}"
                    )

            # First try with the user's specified target_snr
            success = False
            error_message = ""
            binning_result = None  # Will hold the successful result

            # Try the main target first
            try:
                if not quiet:
                    logger.info(
                        f"Trying Voronoi binning with target SNR = {target_snr:.1f}"
                    )

                for use_cvt in [cvt, False]:
                    try:
                        # Set exceptionally high max_iterations for better convergence
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            # UPDATED: Capture all 8 return values
                            binning_result = voronoi_2d_binning(
                                x_valid,
                                y_valid,
                                signal_valid,
                                noise_valid,
                                target_snr,
                                cvt=use_cvt,
                                plot=0,
                                quiet=True,
                                wvt=True,
                            )
                            
                            # Check number of outputs and unpack appropriately
                            if isinstance(binning_result, tuple):
                                if len(binning_result) >= 8:
                                    # Full result with 8 parameters
                                    bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = binning_result
                                elif len(binning_result) >= 6:
                                    # Partial result with 6 parameters - create dummy x_bar, y_bar
                                    bin_num, x_gen, y_gen, sn, n_pixels, scale = binning_result
                                    x_bar, y_bar = x_gen.copy(), y_gen.copy()  # Use generators as centroids
                                else:
                                    raise ValueError(f"Unexpected number of return values: {len(binning_result)}")
                            else:
                                raise ValueError("Unexpected return format from voronoi_2d_binning")

                        # Check if we have any valid bins
                        if len(x_gen) > 0:
                            success = True
                            if not quiet and use_cvt != cvt:
                                logger.warning(
                                    f"Using CVT={use_cvt} for better results"
                                )
                            break
                    except Exception as e:
                        error_message = str(e)
                        if (
                            "index 0 is out of bounds for axis 0 with size 0"
                            in error_message
                        ):
                            # Specific error that requires different approach
                            continue
                        elif "zero-size array to reduction operation" in error_message:
                            # Another empty array error
                            continue
                        elif "too many values to unpack" in error_message:
                            # Error with voronoi_2d_binning unpacking
                            continue
            except Exception as e:
                error_message = str(e)

            # If the specified target failed, try a systematic approach with multiple SNR values
            if not success:
                if not quiet:
                    logger.warning(
                        f"Failed with target SNR = {target_snr:.1f}: {error_message}"
                    )
                
                # Create a logarithmically spaced array of SNR values to try
                # Start from max_recommended down to min_recommended with 100 steps
                num_steps = 300
                
                # Ensure we have valid values for the range
                max_search = max_recommended * 0.99  # Slightly below max to avoid boundary issues
                min_search = max(min_recommended * 1.01, max_pixel_snr * 1.1)  # Ensure we're above min
                
                # Create log-spaced values from high to low
                if max_search > min_search:
                    snr_values_to_try = np.logspace(
                        np.log10(max_search), 
                        np.log10(min_search), 
                        num_steps
                    )
                    
                    if not quiet:
                        logger.info(f"Trying {num_steps} SNR values from {max_search:.1f} to {min_search:.1f}")
                    
                    # Try each SNR value until one works
                    for alternative_target in snr_values_to_try:
                        if not quiet and (num_steps > 10):
                            # Only log occasionally to avoid flooding the logs
                            if np.random.random() < 0.03:  # Log approximately 3% of attempts
                                logger.info(f"Trying SNR = {alternative_target:.1f}")
                        
                        # Try with both CVT options
                        for use_cvt in [cvt, False]:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    
                                    # UPDATED: Capture all 8 return values
                                    binning_result = voronoi_2d_binning(
                                        x_valid,
                                        y_valid,
                                        signal_valid,
                                        noise_valid,
                                        alternative_target,
                                        cvt=use_cvt,
                                        plot=0,
                                        quiet=True,
                                        wvt=True,
                                    )
                                    
                                    # Check number of outputs and unpack appropriately
                                    if isinstance(binning_result, tuple):
                                        if len(binning_result) >= 8:
                                            # Full result with 8 parameters
                                            bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = binning_result
                                        elif len(binning_result) >= 6:
                                            # Partial result with 6 parameters - create dummy x_bar, y_bar
                                            bin_num, x_gen, y_gen, sn, n_pixels, scale = binning_result
                                            x_bar, y_bar = x_gen.copy(), y_gen.copy()  # Use generators as centroids
                                        else:
                                            raise ValueError(f"Unexpected number of return values: {len(binning_result)}")
                                    else:
                                        raise ValueError("Unexpected return format from voronoi_2d_binning")
                                    
                                    # Check if we got valid results
                                    if len(x_gen) > 0:
                                        success = True
                                        if not quiet:
                                            logger.warning(
                                                f"Success with SNR = {alternative_target:.1f}, CVT={use_cvt}"
                                            )
                                        break  # Break out of CVT loop
                            except Exception as e:
                                # Just continue to the next attempt
                                continue
                            
                        if success:
                            break  # Break out of SNR loop if we found a working value
                else:
                    logger.warning(f"Invalid search range: max={max_search:.1f}, min={min_search:.1f}")

            # If still no success, use create_grid_binning as a more reliable fallback
            if not success:
                if not quiet:
                    logger.warning(
                        f"All Voronoi binning attempts failed with error: {error_message}"
                    )
                    logger.warning("Using fallback grid binning scheme")

                # Get additional information for grid binning
                xmin, xmax = np.min(x_valid), np.max(x_valid)
                ymin, ymax = np.min(y_valid), np.max(y_valid)

                # Create a grid-based binning - more reliable than quadrants for arbitrary shapes
                bin_result = create_grid_binning(
                    x_valid,
                    y_valid,
                    signal_valid,
                    noise_valid,
                    nx=min(4, int(np.sqrt(len(x_valid) / 5))),  # Auto-adjust grid size
                    ny=min(4, int(np.sqrt(len(x_valid) / 5))),
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )

                if bin_result is not None:
                    # Grid binning returns 6 parameters, need to create dummy x_bar, y_bar
                    bin_num, x_gen, y_gen, sn, n_pixels, scale = bin_result
                    x_bar, y_bar = x_gen.copy(), y_gen.copy()  # Use generators as centroids
                    success = True
                    if not quiet:
                        logger.info(f"Created {len(x_gen)} grid bins as fallback")
                else:
                    # Fall back to _create_fallback_binning
                    fallback_result = _create_fallback_binning(x, y)
                    if len(fallback_result) == 6:
                        bin_num, x_gen, y_gen, sn, n_pixels, scale = fallback_result
                        x_bar, y_bar = x_gen.copy(), y_gen.copy()  # Use generators as centroids
                    else:
                        # Handle unexpected return value format
                        raise ValueError("Unexpected format from fallback binning")

            # Map bin numbers back to the original arrays
            full_bin_num = np.full_like(x, -1, dtype=int)
            full_bin_num[valid_mask] = bin_num

            # Filter out low SNR bins if needed, but don't filter ALL bins
            if min_snr > 0:
                # Find bins with SNR below minimum
                bad_bins = np.where(sn < min_snr)[0]

                # Don't filter all bins
                if len(bad_bins) < len(sn):  # Only if some bins remain
                    # Mark these bins as unbinned (-1)
                    for bad_bin in bad_bins:
                        full_bin_num[full_bin_num == bad_bin] = -1

                    if not quiet and len(bad_bins) > 0:
                        logger.info(
                            f"Filtered out {len(bad_bins)} bins with SNR < {min_snr}"
                        )
                else:
                    if not quiet:
                        logger.warning(
                            f"All bins have SNR < {min_snr}, but keeping them to avoid empty result"
                        )

            # UPDATED: Return all 8 parameters
            return full_bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale

    except ImportError:
        logger.error(
            "vorbin package not found. Please install with 'pip install vorbin'"
        )
        raise
    except ValueError as ve:
        # For specific value errors, fall back to simple binning
        logger.error(f"Voronoi binning value error: {ve}")
        logger.info("Using simple radial binning as fallback")
        fallback_result = _create_fallback_binning(x, y)
        if len(fallback_result) == 6:
            bin_num, x_gen, y_gen, sn, n_pixels, scale = fallback_result
            x_bar, y_bar = x_gen.copy(), y_gen.copy()  # Use generators as centroids
            return bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale
        else:
            raise ValueError("Unexpected format from fallback binning")
    except Exception as e:
        logger.error(f"Error in Voronoi binning: {e}")
        logger.info("Using simple radial binning as fallback")
        fallback_result = _create_fallback_binning(x, y)
        if len(fallback_result) == 6:
            bin_num, x_gen, y_gen, sn, n_pixels, scale = fallback_result
            x_bar, y_bar = x_gen.copy(), y_gen.copy()  # Use generators as centroids
            return bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale
        else:
            raise ValueError("Unexpected format from fallback binning")


def _create_fallback_binning(x, y, n_bins=5):
    """
    Create a simple fallback binning when Voronoi binning fails

    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    n_bins : int, default=5
        Number of bins to create

    Returns
    -------
    tuple
        Simple binning results compatible with Voronoi binning return
        (bin_num, x_gen, y_gen, sn, n_pixels, scale)
    """
    # Calculate distance from center for each point
    x_center = np.median(x[np.isfinite(x)])
    y_center = np.median(y[np.isfinite(y)])

    r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    # Create simple radial bins
    valid_mask = np.isfinite(r)
    if np.sum(valid_mask) < 1:
        # If no valid points, return a simple error state
        bin_num = np.full_like(x, -1, dtype=int)
        return (
            bin_num,
            np.array([0]),
            np.array([0]),
            np.array([1.0]),
            np.array([1]),
            1.0,
        )

    r_valid = r[valid_mask]
    max_r = np.max(r_valid)

    # Create bin edges
    bin_edges = np.linspace(0, max_r, n_bins + 1)

    # Assign bins
    bin_num = np.full_like(x, -1, dtype=int)
    x_gen = []
    y_gen = []
    sn = []
    n_pixels = []

    for i in range(n_bins):
        bin_mask = valid_mask & (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(bin_mask):
            bin_num[bin_mask] = i
            x_gen.append(np.mean(x[bin_mask]))
            y_gen.append(np.mean(y[bin_mask]))
            n_pixels.append(np.sum(bin_mask))
            sn.append(5.0)  # Use reasonable SNR value instead of placeholder

    # If no bins were created (unlikely but possible), create at least one bin
    if not x_gen:
        bin_num[valid_mask] = 0
        x_gen = [np.mean(x[valid_mask])]
        y_gen = [np.mean(y[valid_mask])]
        n_pixels = [np.sum(valid_mask)]
        sn = [5.0]

    logger.warning(f"Created simple fallback binning with {len(x_gen)} bins")

    return (
        bin_num,
        np.array(x_gen),
        np.array(y_gen),
        np.array(sn),
        np.array(n_pixels),
        1.0,
    )


def calculate_radial_bins(
    x,
    y,
    center_x=None,
    center_y=None,
    pa=0,
    ellipticity=0,
    n_rings=10,
    log_spacing=False,
    r_galaxy=None  # Pre-calculated physical radius
):
    """
    Calculate radial bins with improved error handling and proper centering
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    center_x : float, optional
        x coordinate of center. If None, uses center of the IFU field.
    center_y : float, optional
        y coordinate of center. If None, uses center of the IFU field.
    pa : float, default=0
        Position angle (degrees)
    ellipticity : float, default=0
        Ellipticity (0-1)
    n_rings : int, default=10
        Number of radial rings
    log_spacing : bool, default=False
        Use logarithmic spacing
    r_galaxy : ndarray, optional
        Pre-calculated elliptical galaxy radius array (flattened)
        
    Returns
    -------
    tuple
        (bin_num, bin_edges, bin_radii)
    """
    try:
        # If r_galaxy is provided, use it directly for binning
        if r_galaxy is not None and len(r_galaxy) == len(x):
            radius = r_galaxy
            logger.info("Using provided R_galaxy values for radial binning")
        else:
            # Original calculation method
            # Determine the center if not provided
            if center_x is None or center_y is None:
                # Use the more robust median of valid positions
                valid_x = x[np.isfinite(x)]
                valid_y = y[np.isfinite(y)]
                
                if len(valid_x) > 0 and len(valid_y) > 0:
                    if center_x is None:
                        center_x = np.median(valid_x)
                    if center_y is None:
                        center_y = np.median(valid_y)
                else:
                    # Fallback to simple estimation
                    x_min, x_max = np.nanmin(x), np.nanmax(x)
                    y_min, y_max = np.nanmin(y), np.nanmax(y)
                    
                    if center_x is None:
                        center_x = (x_min + x_max) / 2.0
                    if center_y is None:
                        center_y = (y_min + y_max) / 2.0
                
                logger.info(f"Using center coordinates: ({center_x:.2f}, {center_y:.2f})")

            # Convert position angle to radians
            pa_rad = np.radians(pa)

            # Calculate semi-major axis
            dx = x - center_x
            dy = y - center_y

            # Rotate coordinates
            x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
            y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)

            # Apply ellipticity with error handling
            if ellipticity < 1:
                y_rot_scaled = y_rot / (1 - ellipticity)
            else:
                y_rot_scaled = y_rot * 20  # Use a high but finite value for extreme cases
                logger.warning("Ellipticity too high, using fallback scaling")

            # Calculate radius
            radius = np.sqrt(x_rot**2 + y_rot_scaled**2)

        # Check for valid data points
        valid_mask = np.isfinite(radius)
        if np.sum(valid_mask) < 1:
            logger.warning("No valid data points for binning")
            # Return a simple single bin with all pixels
            bin_num = np.zeros_like(radius, dtype=int)
            bin_edges = np.array([np.nanmax(radius)])
            bin_radii = np.array([0.0])
            return bin_num, bin_edges, bin_radii

        # Calculate bin edges more robustly using percentiles
        # This helps avoid outliers affecting the binning
        valid_radii = radius[valid_mask]
        
        # Use robust range: 1st percentile to 95th percentile
        # This can avoid extreme outliers while keeping most of the galaxy
        min_radius = np.percentile(valid_radii, 1)
        max_radius = np.percentile(valid_radii, 95)
        
        # Add margin to max_radius to ensure we cover enough area
        max_radius *= 1.1  # Add 10% margin
        
        # Ensure min_radius is not too close to 0 for log spacing
        if log_spacing and min_radius < max_radius / 1000:
            min_radius = max_radius / 1000
            
        # Ensure we have a reasonable range
        if (
            max_radius <= min_radius
            or not np.isfinite(max_radius)
            or not np.isfinite(min_radius)
        ):
            logger.warning("Invalid radius range, using simple binning")
            bin_num = np.zeros_like(radius, dtype=int)
            bin_edges = np.array([max_radius])
            bin_radii = np.array([0.0])
            return bin_num, bin_edges, bin_radii

        # Adjust n_rings if we have very few points
        n_points = np.sum(valid_mask)
        if n_points < 2 * n_rings:
            new_n_rings = max(1, n_points // 2)
            logger.warning(
                f"Too few points ({n_points}) for {n_rings} rings. Adjusting to {new_n_rings} rings."
            )
            n_rings = new_n_rings

        # Create full bin edges array including 0
        if log_spacing and min_radius > 0:
            # Logarithmic spacing 
            min_log = np.log10(min_radius)
            max_log = np.log10(max_radius)
            bin_edges_full = np.hstack([
                [0],  # Include 0 as the first edge
                np.logspace(min_log, max_log, n_rings)
            ])
        else:
            # Linear spacing
            bin_edges_full = np.hstack([
                [0],  # Include 0 as the first edge
                np.linspace(min_radius, max_radius, n_rings)
            ])

        # Bin edges are everything except the first edge (0)
        bin_edges = bin_edges_full[1:]

        # Assign bin numbers
        bin_num = np.zeros_like(radius, dtype=int)
        
        # Assign all invalid points to bin 0
        bin_num[~valid_mask] = 0
        
        # For valid points, determine bin by radius
        for i in range(n_rings):
            if i == 0:
                # First bin: all points with radius <= first edge
                mask = valid_mask & (radius <= bin_edges[i])
            else:
                # Other bins: points with radius between previous and current edge
                mask = valid_mask & (radius > bin_edges[i - 1]) & (radius <= bin_edges[i])

            bin_num[mask] = i

        # Check for empty bins and handle them
        # Count pixels in each bin
        bin_counts = np.bincount(bin_num[valid_mask], minlength=n_rings)
        empty_bins = np.where(bin_counts == 0)[0]

        if len(empty_bins) > 0:
            logger.warning(f"Found {len(empty_bins)} empty bins. Adjusting binning.")

            # Determine which bins have data
            non_empty_bins = np.where(bin_counts > 0)[0]

            if len(non_empty_bins) == 0:
                # All bins are empty (shouldn't happen, but just in case)
                logger.error("All bins are empty. Using single bin.")
                bin_num = np.zeros_like(radius, dtype=int)
                bin_edges = np.array([max_radius])
                bin_radii = np.array([np.nanmean(radius[valid_mask])])
                return bin_num, bin_edges, bin_radii

            # Create a mapping from old bin numbers to new contiguous ones
            new_bin_map = np.full(n_rings, -1, dtype=int)
            for i, bin_id in enumerate(non_empty_bins):
                new_bin_map[bin_id] = i

            # Remap the bin numbers
            new_bin_num = np.full_like(bin_num, -1)
            for old_bin, new_bin in enumerate(new_bin_map):
                if new_bin >= 0:  # Only map non-empty bins
                    new_bin_num[bin_num == old_bin] = new_bin

            # Create new bin_edges by only using edges that define non-empty bins
            new_edges_indices = np.sort(np.concatenate([[0], non_empty_bins, non_empty_bins + 1]))
            new_edges_indices = np.unique(new_edges_indices)
            new_edges_indices = new_edges_indices[new_edges_indices < len(bin_edges_full)]
            new_bin_edges = bin_edges_full[new_edges_indices[1:]]  # Skip first edge (0)

            # Update values
            bin_num = new_bin_num
            bin_edges = new_bin_edges
            n_rings = len(non_empty_bins)

        # Calculate bin radii (at middle of each bin)
        bin_radii = np.zeros(n_rings)
        bin_edges_full = np.concatenate(([0], bin_edges))
        
        for i in range(n_rings):
            mask = bin_num == i
            if np.any(mask & valid_mask):
                # Use median radius of points in bin (more robust than mean)
                bin_radii[i] = np.nanmedian(radius[mask & valid_mask])
            else:
                # Use middle of bin edges if no points (shouldn't happen after our fixes)
                bin_radii[i] = 0.5 * (bin_edges_full[i] + bin_edges_full[i + 1])

        return bin_num, bin_edges, bin_radii

    except Exception as e:
        logger.error(f"Error in radial binning: {e}")
        # Return a simple single bin as fallback
        bin_num = np.zeros_like(x, dtype=int)
        bin_edges = np.array([np.nanmax(np.sqrt(x**2 + y**2))])
        bin_radii = np.array([0.0])
        return bin_num, bin_edges, bin_radii


def calculate_radial_bins_re_based(x, y, center_x=None, center_y=None, pa=0, ellipticity=0,
                               effective_radius=None, r_galaxy=None, max_radius_scale=3.0,
                               n_bins=4):  # Default changed to 4 bins
    """
    Calculate radial bins using effective radius (Re) as the scaling factor.
    Specifically creates 4 bins: 0-0.5 Re, 0.5-1 Re, 1-1.5 Re, >1.5 Re
    
    Parameters:
    -----------
    x, y : array-like
        Pixel coordinates
    center_x, center_y : float, optional
        Center coordinates. If None, uses image center.
    pa : float, optional
        Position angle in degrees
    ellipticity : float, optional
        Ellipticity (1-b/a)
    effective_radius : float, optional
        Effective radius (Re) in arcseconds
    r_galaxy : array-like, optional
        Pre-calculated galaxy radii in arcsec
    max_radius_scale : float, optional
        Maximum radius as a multiple of Re (default: 3.0)
    n_bins : int, optional
        Number of radial bins (default: 4, fixed bin ranges used regardless)
        
    Returns:
    --------
    bin_num : ndarray
        Array of bin numbers for each pixel
    bin_edges : ndarray
        Array of bin edges in arcseconds
    bin_radii : ndarray
        Array of bin radii (midpoints) in arcseconds
    """
    if center_x is None:
        center_x = np.max(x) / 2
    if center_y is None:
        center_y = np.max(y) / 2
        
    # Ensure we have coordinates as numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Use pre-calculated r_galaxy if provided
    if r_galaxy is not None:
        r = r_galaxy
    else:
        # Calculate radii in pixels
        dx = x - center_x
        dy = y - center_y
        
        # Convert to elliptical radius
        pa_rad = np.radians(pa)
        x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
        y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
        
        # Apply ellipticity
        a = 1.0
        b = 1.0 - ellipticity
        
        # Calculate elliptical radius
        r = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    
    # If effective radius is not provided, estimate it
    if effective_radius is None or effective_radius <= 0:
        # Estimate Re as half the 90th percentile radius
        effective_radius = np.nanpercentile(r, 90) / 2
        logger.warning(f"Effective radius not provided, estimated as {effective_radius:.2f}")
    
    # Create exactly 4 bins with specific Re fractions:
    # 0-0.5 Re, 0.5-1 Re, 1-1.5 Re, >1.5 Re (up to max_radius_scale)
    bin_edges = np.array([0.0, 0.5, 1.0, 1.5, max_radius_scale]) * effective_radius
    
    # Force 4 bins regardless of n_bins parameter
    n_bins = 4 
    
    # Calculate bin radii (midpoints)
    bin_radii = np.zeros(n_bins)
    for i in range(n_bins):
        if i < n_bins - 1:  # Normal bins
            bin_radii[i] = (bin_edges[i] + bin_edges[i+1]) / 2
        else:  # Last bin (open-ended)
            # For last bin, use midpoint between lower edge and 90th percentile
            r_in_bin = r[(r >= bin_edges[i]) & (r < bin_edges[i+1])]
            if len(r_in_bin) > 0:
                # Use percentile if we have enough points
                radius_90 = np.nanpercentile(r_in_bin, 90)
                bin_radii[i] = (bin_edges[i] + radius_90) / 2
            else:
                # Fallback to simple midpoint
                bin_radii[i] = (bin_edges[i] + bin_edges[i+1]) / 2
    
    # Assign each pixel to a bin
    bin_num = np.full_like(r, -1)  # Initialize with -1 (no bin)
    
    # For each bin, assign pixels within its radius range
    for i in range(n_bins):
        bin_mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        bin_num[bin_mask] = i
    
    # Log the specific bin edges used
    logger.info(f"Created 4 radial bins with Re={effective_radius:.2f}:")
    logger.info(f"  Bin 0: 0-0.5 Re ({bin_edges[0]:.2f}-{bin_edges[1]:.2f} arcsec)")
    logger.info(f"  Bin 1: 0.5-1 Re ({bin_edges[1]:.2f}-{bin_edges[2]:.2f} arcsec)")
    logger.info(f"  Bin 2: 1-1.5 Re ({bin_edges[2]:.2f}-{bin_edges[3]:.2f} arcsec)")
    logger.info(f"  Bin 3: >1.5 Re ({bin_edges[3]:.2f}-{bin_edges[4]:.2f} arcsec)")
    
    return bin_num, bin_edges, bin_radii


def create_grid_binning(
    x, y, signal, noise, nx=3, ny=3, xmin=None, xmax=None, ymin=None, ymax=None
):
    """
    Create a grid-based binning as a reliable fallback
    
    Parameters
    ----------
    x, y : ndarray
        Coordinates
    signal, noise : ndarray
        Signal and noise data
    nx, ny : int
        Number of bins in x and y directions
    xmin, xmax, ymin, ymax : float, optional
        Coordinate bounds
        
    Returns
    -------
    tuple or None
        Binning results (bin_num, x_gen, y_gen, sn, n_pixels, scale)
    """
    try:
        # Ensure at least one bin
        nx = max(1, nx)
        ny = max(1, ny)

        # Get bounds if not provided
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x)
        if ymin is None:
            ymin = np.min(y)
        if ymax is None:
            ymax = np.max(y)

        # Add small margin
        eps = 1e-6
        xmin -= eps
        xmax += eps
        ymin -= eps
        ymax += eps

        # Create bin edges
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)

        # Initialize arrays
        bin_num = np.full(len(x), -1, dtype=int)
        x_gen = []
        y_gen = []
        sn_values = []
        n_pixels = []

        # Assign bins
        bin_id = 0
        for i in range(nx):
            for j in range(ny):
                # Get bin boundaries
                x_min, x_max = x_edges[i], x_edges[i + 1]
                y_min, y_max = y_edges[j], y_edges[j + 1]

                # Select points in this bin
                mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)

                # Create bin if it has data points
                if np.any(mask):
                    bin_num[mask] = bin_id
                    x_gen.append(np.mean(x[mask]))
                    y_gen.append(np.mean(y[mask]))

                    # Calculate SNR
                    s = np.mean(signal[mask])
                    n = np.mean(noise[mask])
                    sn = s / n if n > 0 else 1.0

                    sn_values.append(sn)
                    n_pixels.append(np.sum(mask))
                    bin_id += 1

        # Check if any bins were created
        if bin_id == 0:
            return None

        return (
            bin_num,
            np.array(x_gen),
            np.array(y_gen),
            np.array(sn_values),
            np.array(n_pixels),
            1.0,
        )

    except Exception as e:
        logger.error(f"Error in grid binning: {e}")
        return None