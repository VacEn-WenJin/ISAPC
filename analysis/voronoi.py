"""
Voronoi binning analysis module for ISAPC
Version 5.0.0 - Enhanced with improved SNR target selection
"""

import logging
import time
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vorbin.voronoi_2d_binning import voronoi_2d_binning

import spectral_indices
from binning import (
    VoronoiBinnedData,
    calculate_wavelength_intersection,
    combine_spectra_efficiently,
)
from utils.io import save_standardized_results

logger = logging.getLogger(__name__)

# Speed of light in km/s
C_KMS = 299792.458


def run_vnb_analysis(args, cube, p2p_results=None):
    """
    Run Voronoi binning analysis on MUSE data cube with improved coordinate handling
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict, optional
        P2P analysis results to use for binning
        
    Returns
    -------
    dict
        Analysis results
    """
    logger.info("Starting Voronoi binning analysis...")
    start_time = time.time()

    # Disable warnings for spectral indices
    spectral_indices.set_warnings(False)

    # Get galaxy name and create directories
    galaxy_name = Path(args.filename).stem
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / "Data"
    plots_dir = galaxy_dir / "Plots" / "VNB"

    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Try to load P2P results if not provided but auto-reuse is enabled
    if p2p_results is None and hasattr(args, "auto_reuse") and args.auto_reuse:
        try:
            # Try loading from standard paths
            p2p_results_path = data_dir / f"{galaxy_name}_P2P_results.npz"
            std_results_path = data_dir / f"{galaxy_name}_P2P_standardized.npz"

            if p2p_results_path.exists():
                p2p_results = np.load(p2p_results_path, allow_pickle=True)
                logger.info("Successfully loaded P2P results for VNB analysis")
            elif std_results_path.exists():
                p2p_results = np.load(std_results_path, allow_pickle=True)
                logger.info(
                    "Successfully loaded standardized P2P results for VNB analysis"
                )
        except Exception as e:
            logger.warning(f"Error loading P2P results: {e}")
            p2p_results = None

    # Set up binning parameters
    target_snr = args.target_snr if hasattr(args, "target_snr") else 30
    min_snr = args.min_snr if hasattr(args, "min_snr") else 1
    use_cvt = args.cvt if hasattr(args, "cvt") else True
    high_snr_mode = args.high_snr_mode if hasattr(args, "high_snr_mode") else False
    use_physical_radius = args.physical_radius if hasattr(args, "physical_radius") else False

    # Extract coordinates, signal, and noise for binning
    # IMPORTANT: Use original coordinates from the cube
    x = cube.x  # These are the original x coordinates from the cube
    y = cube.y  # These are the original y coordinates from the cube

    # Calculate SNR using a specific wavelength range for continuum
    wave_mask = (cube._lambda_gal >= 5075) & (cube._lambda_gal <= 5125)
    if np.sum(wave_mask) > 0:
        signal = np.nanmedian(cube._spectra[wave_mask], axis=0)
        noise = np.nanstd(cube._spectra[wave_mask], axis=0)
        logger.info("Using wavelength range 5075-5125 Å for SNR calculation")
    else:
        signal = np.nanmedian(cube._spectra, axis=0)
        noise = np.nanmedian(np.sqrt(cube._log_variance), axis=0)
        logger.info("Using full spectrum for SNR calculation")

    # Preprocess signal and noise
    min_threshold = 1.0
    low_signal_mask = (signal < min_threshold) | ~np.isfinite(signal)
    low_noise_mask = (noise < min_threshold) | ~np.isfinite(noise)

    if np.any(low_signal_mask) or np.any(low_noise_mask):
        logger.warning(
            f"Found {np.sum(low_signal_mask)} pixels with signal < {min_threshold}"
        )
        logger.warning(
            f"Found {np.sum(low_noise_mask)} pixels with noise < {min_threshold}"
        )

        signal[low_signal_mask] = min_threshold
        noise[low_noise_mask] = min_threshold

        low_snr_mask = (signal / noise) < 1.0
        if np.any(low_snr_mask):
            logger.warning(
                f"Setting both signal and noise to {min_threshold} for {np.sum(low_snr_mask)} pixels with SNR < 1"
            )
            signal[low_snr_mask] = min_threshold
            noise[low_snr_mask] = min_threshold

    # Ensure no zeros or NaNs
    signal = np.nan_to_num(signal, nan=min_threshold)
    noise = np.nan_to_num(noise, nan=min_threshold)

    # Calculate per-pixel SNR values
    pixel_snr = signal / noise
    max_pixel_snr = np.nanmax(pixel_snr)
    median_snr = np.nanmedian(pixel_snr)

    logger.info(
        f"Maximum pixel SNR: {max_pixel_snr:.1f}, Median pixel SNR: {median_snr:.1f}"
    )

    # Determine recommended SNR range
    min_recommended_snr = min(2, median_snr * 0.5)
    max_recommended_snr = max(max_pixel_snr * 1.2, median_snr * 10)

    # Ensure reasonable range
    min_recommended_snr = max(2, min_recommended_snr)
    max_recommended_snr = max(50, max_recommended_snr)

    # Adjust target_snr if needed
    if target_snr < min_recommended_snr or target_snr > max_recommended_snr:
        logger.warning(
            f"Specified target SNR {target_snr} is outside recommended range "
            f"({min_recommended_snr:.1f} - {max_recommended_snr:.1f})"
        )

        safe_target_snr = max(min_recommended_snr, min(target_snr, max_recommended_snr))
        logger.info(f"Adjusting target SNR to {safe_target_snr:.1f}")
    else:
        safe_target_snr = target_snr

    # Calculate physical radius if requested
    ellipse_params = None
    if use_physical_radius:
        try:
            from physical_radius import calculate_galaxy_radius
            
            # Create flux map for radius calculation
            flux_2d = np.nanmedian(cube._cube_data, axis=0)
            
            # Calculate physical radius
            R_galaxy, ellipse_params = calculate_galaxy_radius(
                flux_2d,
                pixel_size_x=cube._pxl_size_x,
                pixel_size_y=cube._pxl_size_y
            )
            
            logger.info(f"Calculated physical radius with PA={ellipse_params['PA_degrees']:.1f}°, "
                        f"ε={ellipse_params['ellipticity']:.2f}")
                        
            # Store for later use
            cube._physical_radius = R_galaxy
            cube._ellipse_params = ellipse_params
            
        except Exception as e:
            logger.warning(f"Error calculating physical radius: {e}")
            use_physical_radius = False

    # Run Voronoi binning
    logger.info(f"Running Voronoi binning with target SNR = {safe_target_snr:.1f}")
    
    # Import run_voronoi_binning from binning module - this is where the fix is applied
    from binning import run_voronoi_binning
    
    # Call the function with original coordinates - no transformation
    # In run_vnb_analysis function in voronoi.py:

    # Import the updated function
    from binning import run_voronoi_binning

    # Change the function call to unpack 8 parameters
    bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = run_voronoi_binning(
        x, y, signal, noise, 
        target_snr=safe_target_snr,
        plot=0, 
        quiet=False, 
        cvt=use_cvt,
        min_snr=min_snr
    )

    # The rest of the function stays the same, but using bin_num consistently

    # Create bin indices
    bin_indices = []
    all_indices = np.arange(len(bin_num))
    max_bin = int(np.nanmax(bin_num))

    for i in range(max_bin + 1):
        bin_indices.append(all_indices[bin_num == i])

    # Get velocity field from P2P results if available, for velocity correction
    velocity_field = None
    if p2p_results is not None:
        try:
            # Try standard format first
            if (
                "stellar_kinematics" in p2p_results
                and "velocity_field" in p2p_results["stellar_kinematics"]
            ):
                velocity_field = p2p_results["stellar_kinematics"]["velocity_field"]
            # Try direct format
            elif "velocity_field" in p2p_results:
                velocity_field = p2p_results["velocity_field"]

            if velocity_field is not None:
                logger.info(
                    "Using velocity field from P2P results for velocity correction"
                )
        except Exception as e:
            logger.warning(f"Error extracting velocity field from P2P results: {e}")

    # Calculate wavelength intersection accounting for velocity shifts
    from binning import calculate_wavelength_intersection
    
    if velocity_field is not None:
        wave_mask, min_wave, max_wave = calculate_wavelength_intersection(
            cube._lambda_gal, velocity_field, cube._n_x
        )
        logger.info(
            f"Velocity correction wavelength range: {min_wave:.1f} - {max_wave:.1f} Å"
        )
    else:
        wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)

    # Apply wavelength mask
    wavelength = cube._lambda_gal[wave_mask]
    spectra = cube._spectra[wave_mask]

    # Combine spectra into bins with velocity correction
    from binning import combine_spectra_efficiently
    
    binned_spectra = combine_spectra_efficiently(
        spectra, wavelength, bin_indices, velocity_field, cube._n_x
    )

    # Create metadata
    metadata = {
        "nx": cube._n_x,
        "ny": cube._n_y,
        "target_snr": target_snr,
        "sn": sn,
        "n_pixels": n_pixels,
        "scale": scale,
        "time": time.time(),
        "galaxy_name": galaxy_name,
        "analysis_type": "VNB",
        "pixelsize_x": cube._pxl_size_x,
        "pixelsize_y": cube._pxl_size_y,
        "redshift": cube._redshift if hasattr(cube, "_redshift") else 0.0,
    }
    
    # Add physical radius information if used
    if use_physical_radius and ellipse_params is not None:
        metadata["physical_radius"] = True
        metadata["ellipse_params"] = ellipse_params
        if hasattr(cube, "_physical_radius"):
            metadata["R_galaxy_map"] = cube._physical_radius
        metadata["flux_map"] = np.nanmedian(cube._cube_data, axis=0)

    # Create VoronoiBinnedData object
    from binning import VoronoiBinnedData
    
    binned_data = VoronoiBinnedData(
        bin_num=bin_num,
        bin_indices=bin_indices,
        spectra=binned_spectra,
        wavelength=wavelength,
        metadata=metadata,
    )

    # Set up binning in the cube - this connects our binned data to the cube
    cube.setup_binning("VNB", binned_data)

    # Run analysis using the binned data
    velocity_field, dispersion_field, bestfit_field, optimal_tmpls, poly_coeffs = (
        cube.fit_spectra(
            template_filename=args.template,
            ppxf_vel_init=args.vel_init,
            ppxf_vel_disp_init=args.sigma_init,
            ppxf_deg=args.poly_degree if hasattr(args, "poly_degree") else 3,
            n_jobs=args.n_jobs,
        )
    )

    # Fit emission lines if requested
    emission_result = None
    if not args.no_emission:
        emission_result = cube.fit_emission_lines(
            template_filename=args.template,
            ppxf_vel_init=velocity_field,  # Use stellar velocity field as initial guess
            ppxf_sig_init=args.sigma_init,
            ppxf_deg=2,
            n_jobs=args.n_jobs,
        )

    # Before returning the results, add:
    # Ensure emission lines are consistently processed
    if emission_result is not None and hasattr(cube, "_post_process_emission_results"):
        cube._post_process_emission_results()

    # Calculate spectral indices if requested
    indices_result = None
    if not args.no_indices:
        indices_result = cube.calculate_spectral_indices(n_jobs=args.n_jobs)

    # In run_vnb_analysis/run_rdb_analysis:

    # Calculate bin distances
    bin_distances = np.zeros(len(bin_indices))
    for i in range(len(bin_indices)):
        # Get pixel indices for this bin
        pixel_indices = bin_indices[i]

        # Skip empty bins
        if len(pixel_indices) == 0:
            bin_distances[i] = np.nan
            continue

        # Calculate average distance for this bin
        distances = []
        for idx in pixel_indices:
            row = idx // cube._n_x
            col = idx % cube._n_x

            # Calculate distance from center
            center_y, center_x = cube._n_y // 2, cube._n_x // 2
            distance = np.sqrt(
                ((row - center_y) * cube._pxl_size_y) ** 2
                + ((col - center_x) * cube._pxl_size_x) ** 2
            )
            distances.append(distance)

        # Set average distance
        bin_distances[i] = np.mean(distances)

    # Prepare standardized output dictionary
    vnb_results = {
        "analysis_type": "VNB",
        "stellar_kinematics": {
            "velocity": cube._bin_velocity,
            "dispersion": cube._bin_dispersion,
            "velocity_field": velocity_field,
            "dispersion_field": dispersion_field,
        },
        "distance": {
            "bin_distances": bin_distances,
            "pixelsize_x": cube._pxl_size_x,
            "pixelsize_y": cube._pxl_size_y,
        },
        "binning": {
            "bin_num": bin_num,
            "bin_indices": bin_indices,
            "bin_x": x_gen,
            "bin_y": y_gen,
            "n_pixels": n_pixels,
            "snr": sn,
            "target_snr": target_snr,
        },
    }
    
    # Add physical radius information if used
    if use_physical_radius and ellipse_params is not None:
        vnb_results["physical_radius"] = {
            "center_x": ellipse_params["center_x"],
            "center_y": ellipse_params["center_y"],
            "pa": ellipse_params["PA_degrees"],
            "ellipticity": ellipse_params["ellipticity"],
        }

    # Add emission line results if available
    if emission_result is not None:
        vnb_results["emission"] = {}

        # Copy emission line fields from cube if available
        if hasattr(cube, "_bin_emission_flux"):
            vnb_results["emission"]["flux"] = cube._bin_emission_flux
        if hasattr(cube, "_bin_emission_vel"):
            vnb_results["emission"]["velocity"] = cube._bin_emission_vel
        if hasattr(cube, "_bin_emission_sig"):
            vnb_results["emission"]["dispersion"] = cube._bin_emission_sig

        # Copy emission fields from emission_result
        for key in ["emission_flux", "emission_vel", "emission_sig"]:
            if key in emission_result:
                field_name = key.split("_")[1]  # extract 'flux', 'vel', 'sig'
                vnb_results["emission"][field_name] = emission_result[key]

        # Add emission line wavelengths if available
        if "emission_wavelength" in emission_result:
            vnb_results["emission"]["wavelengths"] = emission_result[
                "emission_wavelength"
            ]

        # After calculating spectral indices
    if indices_result is not None:
        vnb_results["indices"] = indices_result  # This will be pixel-based

        # Add bin-level indices if available
        if hasattr(cube, "_bin_indices_result") and cube._bin_indices_result:
            vnb_results["bin_indices"] = cube._bin_indices_result

    # Add this after spectral indices calculation in both voronoi.py and radial.py:


    try:
        import visualization
        
        # First create the standard plots as before
        # [existing plotting code...]
        
        # Create a new bin overlay plot
        try:
            # Get flux map from cube - use median flux along wavelength
            flux_map = np.nanmedian(cube._cube_data, axis=0)
            
            # Get binning information
            bin_num = vnb_results["binning"]["bin_num"]
            
            # If bin_num is 1D, reshape to match the cube dimensions
            if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
            else:
                bin_num_2d = bin_num
            
            # Get bin centers if available
            bin_centers = None
            if "bin_x" in vnb_results["binning"] and "bin_y" in vnb_results["binning"]:
                bin_centers = (vnb_results["binning"]["bin_x"], vnb_results["binning"]["bin_y"])
            
            # Create and save binning overlay plot
            fig, ax = visualization.plot_binning_on_flux(
                flux_map=flux_map,
                bin_num=bin_num_2d,
                bin_centers=bin_centers,
                title=f"{galaxy_name} - Voronoi Binning",
                cmap="inferno",
                save_path=plots_dir / f"{galaxy_name}_VNB_binning_overlay.png",
                binning_type="Voronoi",
                wcs=cube._wcs if hasattr(cube, "_wcs") else None,
                pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                log_scale=True
            )
            plt.close(fig)
            
            logger.info(f"Created VNB binning overlay plot")
            
        except Exception as e:
            logger.warning(f"Error creating VNB binning overlay plot: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            plt.close("all")
                
    except Exception as e:
        logger.error(f"Error in create_vnb_plots: {str(e)}")
        logger.error(traceback.format_exc())
        plt.close("all")

    # Create spectral index visualizations if requested
    if indices_result is not None and not args.no_plots:
        try:
            # Create directory for spectral index visualization
            indices_plots_dir = plots_dir / "spectral_indices"
            indices_plots_dir.mkdir(exist_ok=True, parents=True)

            # Plot indices for a few sample bins
            n_bins_to_plot = min(5, cube._n_bins)
            bin_indices_to_plot = []

            # Try to select bins with valid velocities
            if hasattr(cube, "_bin_velocity"):
                valid_bins = [
                    i
                    for i in range(len(cube._bin_velocity))
                    if np.isfinite(cube._bin_velocity[i])
                ]

                if len(valid_bins) > 0:
                    # Get evenly spaced bins
                    step = max(1, len(valid_bins) // n_bins_to_plot)
                    bin_indices_to_plot = valid_bins[::step][:n_bins_to_plot]

            # If no valid bins found, just use the first few
            if not bin_indices_to_plot:
                bin_indices_to_plot = range(n_bins_to_plot)

            # Plot each bin
            for bin_idx in bin_indices_to_plot:
                try:
                    # Use the dedicated method to ensure consistency with calculation
                    fig, axes = cube.plot_bin_index_calculation(
                        bin_idx, save_dir=indices_plots_dir
                    )
                    if fig is not None:
                        plt.close(fig)  # Close to avoid memory issues

                except Exception as e:
                    logger.warning(
                        f"Error plotting spectral indices for bin {bin_idx}: {e}"
                    )

        except Exception as e:
            logger.warning(f"Error creating spectral index visualizations: {e}")

    # Save binned data object
    binned_data_path = data_dir / f"{galaxy_name}_VNB_binned_data.npz"
    binned_data.save(binned_data_path)
    logger.info(f"Saved binned data to {binned_data_path}")

    # Save results
    save_standardized_results(galaxy_name, "VNB", vnb_results, output_dir)

    # Create visualization plots if requested
    if not hasattr(args, "no_plots") or not args.no_plots:
        binned_data.create_visualization_plots(plots_dir, galaxy_name)
        create_vnb_plots(cube, vnb_results, galaxy_name, plots_dir, args)

    # Add these lines near the end of the run_vnb_analysis function in voronoi.py
    # Just before returning the results

    # Create bin spectrum plots if requested
    if not args.no_plots:
        # Create directory for bin plots
        bin_plots_dir = plots_dir / "bin_spectra"
        bin_plots_dir.mkdir(exist_ok=True, parents=True)

        # Plot a sample of bin spectra
        try:
            cube.plot_bin_fits(n_bins=min(5, cube._n_bins), save_dir=bin_plots_dir)
        except Exception as e:
            logger.warning(f"Error creating bin fitting plots: {e}")

    # Create comprehensive visualization plots
    if not args.no_plots:
        try:
            # Create visualization plots directory
            viz_plots_dir = plots_dir / "visualizations"
            viz_plots_dir.mkdir(exist_ok=True, parents=True)

            # Generate all visualization plots
            cube.plot_bin_analysis_results(output_dir=viz_plots_dir)

            logger.info(f"Created visualization plots in {viz_plots_dir}")
        except Exception as e:
            logger.warning(f"Error creating visualization plots: {e}")

    logger.info(f"VNB analysis completed in {time.time() - start_time:.1f} seconds")

    return vnb_results


def create_vnb_plots(cube, vnb_results, galaxy_name, plots_dir, args):
    """
    Create visualization plots for VNB analysis
    
    Parameters
    ----------
    cube : MUSECube
        MUSE cube with binned data
    vnb_results : dict
        VNB analysis results
    galaxy_name : str
        Galaxy name
    plots_dir : Path
        Directory to save plots
    args : argparse.Namespace
        Command line arguments
    """
    try:
        import visualization
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Create bin map plot with physical scaling
        if "binning" in vnb_results and "bin_num" in vnb_results["binning"]:
            try:
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                else:
                    bin_num_2d = bin_num
                    
                # Create bin map with physical scaling
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d,
                    None,
                    ax=ax,
                    cmap="tab20",
                    title=f"{galaxy_name} - Voronoi Bins",
                    physical_scale=True,
                    pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_VNB_bin_map.png"
                )
                plt.close(fig)
                
                # Create SNR map
                if "binning" in vnb_results and "snr" in vnb_results["binning"]:
                    snr = vnb_results["binning"]["snr"]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    visualization.plot_bin_map(
                        bin_num_2d,
                        snr,
                        ax=ax,
                        cmap="viridis",
                        title=f"{galaxy_name} - Bin SNR",
                        colorbar_label="Signal-to-Noise Ratio",
                        physical_scale=True,
                        pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                    )
                    visualization.standardize_figure_saving(
                        fig, plots_dir / f"{galaxy_name}_VNB_snr_map.png"
                    )
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating bin map: {e}")
                plt.close("all")
                
        # Create kinematics plots
        if "stellar_kinematics" in vnb_results:
            try:
                velocity = vnb_results["stellar_kinematics"]["velocity"]
                dispersion = vnb_results["stellar_kinematics"]["dispersion"]
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                else:
                    bin_num_2d = bin_num
                    
                # Velocity map with physical scaling
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d,
                    velocity,
                    ax=ax,
                    cmap="coolwarm",
                    title=f"{galaxy_name} - VNB Velocity",
                    vmin=np.percentile(velocity[np.isfinite(velocity)], 5),
                    vmax=np.percentile(velocity[np.isfinite(velocity)], 95),
                    colorbar_label="Velocity (km/s)",
                    physical_scale=True,
                    pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_VNB_velocity_map.png"
                )
                plt.close(fig)
                
                # Dispersion map with physical scaling
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d,
                    dispersion,
                    ax=ax,
                    cmap="viridis",
                    title=f"{galaxy_name} - VNB Dispersion",
                    vmin=np.percentile(dispersion[np.isfinite(dispersion)], 5),
                    vmax=np.percentile(dispersion[np.isfinite(dispersion)], 95),
                    colorbar_label="Dispersion (km/s)",
                    physical_scale=True,
                    pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_VNB_dispersion_map.png"
                )
                plt.close(fig)
                
            except Exception as e:
                logger.warning(f"Error creating kinematics plots: {e}")
                plt.close("all")
                
        # Create emission line plots if available
        if "emission" in vnb_results:
            try:
                emission = vnb_results["emission"]
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                else:
                    bin_num_2d = bin_num
                    
                # Process each emission line field
                for field_name, field_data in emission.items():
                    if field_name in ["flux", "velocity", "dispersion"] and isinstance(field_data, dict):
                        for line_name, line_values in field_data.items():
                            try:
                                if np.any(np.isfinite(line_values)):
                                    # Create map with physical scaling
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    # Determine colormap based on field type
                                    if field_name == "flux":
                                        cmap = "inferno"
                                        title = f"{galaxy_name} - {line_name} Flux"
                                        label = "Flux"
                                        log_scale = True
                                    elif field_name == "velocity":
                                        cmap = "coolwarm"
                                        title = f"{galaxy_name} - {line_name} Velocity"
                                        label = "Velocity (km/s)"
                                        log_scale = False
                                    else:  # dispersion
                                        cmap = "viridis"
                                        title = f"{galaxy_name} - {line_name} Dispersion"
                                        label = "Dispersion (km/s)"
                                        log_scale = False
                                        
                                    visualization.plot_bin_map(
                                        bin_num_2d,
                                        line_values,
                                        ax=ax,
                                        cmap=cmap,
                                        title=title,
                                        colorbar_label=label,
                                        log_scale=log_scale,
                                        physical_scale=True,
                                        pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                                    )
                                    visualization.standardize_figure_saving(
                                        fig, plots_dir / f"{galaxy_name}_VNB_{line_name}_{field_name}.png"
                                    )
                                    plt.close(fig)
                            except Exception as e:
                                logger.warning(f"Error creating plot for {line_name} {field_name}: {e}")
                                plt.close("all")
            except Exception as e:
                logger.warning(f"Error creating emission line plots: {e}")
                plt.close("all")
                
        # Create spectral indices plots
        if "bin_indices" in vnb_results:
            try:
                bin_indices = vnb_results["bin_indices"]
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                else:
                    bin_num_2d = bin_num
                    
                # Process each spectral index
                for idx_name, idx_values in bin_indices.items():
                    if np.any(np.isfinite(idx_values)):
                        # Create map with physical scaling
                        fig, ax = plt.subplots(figsize=(10, 8))
                        visualization.plot_bin_map(
                            bin_num_2d,
                            idx_values,
                            ax=ax,
                            cmap="plasma",
                            title=f"{galaxy_name} - {idx_name}",
                            colorbar_label="Index Value",
                            physical_scale=True,
                            pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                        )
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_VNB_{idx_name}.png"
                        )
                        plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating spectral indices plots: {e}")
                plt.close("all")
        
        # Create a new bin overlay plot showing binning on flux map
        try:
            # Get flux map from cube - use median flux along wavelength
            flux_map = np.nanmedian(cube._cube_data, axis=0)
            
            # Get binning information
            bin_num = vnb_results["binning"]["bin_num"]
            
            # If bin_num is 1D, reshape to match the cube dimensions
            if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
            else:
                bin_num_2d = bin_num
            
            # Get bin centers if available
            bin_centers = None
            if "bin_x" in vnb_results["binning"] and "bin_y" in vnb_results["binning"]:
                bin_centers = (vnb_results["binning"]["bin_x"], vnb_results["binning"]["bin_y"])
            
            # Process WCS for proper handling
            wcs_obj = None
            if hasattr(cube, "_wcs") and cube._wcs is not None:
                wcs_obj = cube._wcs
                # Handle WCS with more than 2 dimensions by slicing
                if wcs_obj.naxis > 2:
                    try:
                        from astropy.wcs import WCS
                        
                        # Try different approaches based on the astropy version
                        try:
                            # Newer astropy versions
                            if hasattr(wcs_obj, 'celestial'):
                                wcs_obj = wcs_obj.celestial
                            # Older astropy versions
                            elif hasattr(wcs_obj, 'sub'):
                                wcs_obj = wcs_obj.sub([1, 2])
                            # Really old versions
                            else:
                                # Just grab the spatial part manually
                                header = wcs_obj.to_header()
                                new_header = {}
                                for key in header:
                                    if '1' in key or '2' in key:  # Keep only 1st and 2nd axes
                                        new_header[key] = header[key]
                                wcs_obj = WCS(new_header)
                        except Exception as e1:
                            logger.warning(f"Error creating 2D WCS: {e1}")
                            wcs_obj = None
                            
                    except Exception as e:
                        logger.warning(f"Error processing WCS: {e}")
                        wcs_obj = None
            
            # Create a dedicated new figure for the overlay plot
            plt.figure(figsize=(10, 9))
            
            # Define save path
            overlay_path = plots_dir / f"{galaxy_name}_VNB_binning_overlay.png"
            
            # Utility function to create the overlay plot
            def create_overlay_plot():
                """Create a high-quality bin overlay plot"""
                from matplotlib.colors import LogNorm, Normalize
                import matplotlib.colors as mcolors
                from matplotlib.collections import LineCollection
                
                # Get current figure and axis
                fig = plt.gcf()
                ax = plt.gca()
                
                # Clear any previous content
                ax.clear()
                
                # Get dimensions
                ny, nx = flux_map.shape
                
                # Handle NaN values and mask
                masked_flux = np.ma.array(flux_map, mask=~np.isfinite(flux_map))
                
                # Determine color normalization
                valid_flux = masked_flux.compressed()
                if len(valid_flux) > 0 and np.any(valid_flux > 0):
                    # Logarithmic scale with safety
                    min_positive = np.nanmax([np.min(valid_flux[valid_flux > 0]), 1e-10])
                    norm = LogNorm(vmin=min_positive, vmax=np.nanmax(valid_flux))
                else:
                    # Linear scale fallback
                    norm = Normalize(vmin=0, vmax=1)
                
                # Plot flux map
                im = ax.imshow(masked_flux, origin='lower', cmap='inferno', norm=norm)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Flux (log scale)')
                
                # Better method to detect bin edges - work with the bin number array directly
                bin_edges = []
                
                # Check for horizontal edges (better algorithm)
                for j in range(ny):
                    for i in range(nx-1):
                        if (bin_num_2d[j, i] != bin_num_2d[j, i+1] and 
                            bin_num_2d[j, i] >= 0 and bin_num_2d[j, i+1] >= 0):
                            # Found an edge - store segment endpoints
                            if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
                                # Use physical coordinates
                                pixel_size_x = cube._pxl_size_x
                                pixel_size_y = cube._pxl_size_y
                                
                                # Convert to physical units with center at (0,0)
                                x1 = (i - nx/2) * pixel_size_x
                                x2 = (i+1 - nx/2) * pixel_size_x
                                y1 = (j - ny/2) * pixel_size_y
                                y2 = y1  # Same y-coordinate
                                
                                bin_edges.append([(x1, y1), (x2, y2)])
                            else:
                                # Use pixel coordinates
                                bin_edges.append([(i, j), (i+1, j)])
                
                # Check for vertical edges
                for i in range(nx):
                    for j in range(ny-1):
                        if (bin_num_2d[j, i] != bin_num_2d[j+1, i] and 
                            bin_num_2d[j, i] >= 0 and bin_num_2d[j+1, i] >= 0):
                            # Found an edge - store segment endpoints
                            if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
                                # Use physical coordinates
                                pixel_size_x = cube._pxl_size_x
                                pixel_size_y = cube._pxl_size_y
                                
                                # Convert to physical units with center at (0,0)
                                x1 = (i - nx/2) * pixel_size_x
                                x2 = x1  # Same x-coordinate
                                y1 = (j - ny/2) * pixel_size_y
                                y2 = (j+1 - ny/2) * pixel_size_y
                                
                                bin_edges.append([(x1, y1), (x2, y2)])
                            else:
                                # Use pixel coordinates
                                bin_edges.append([(i, j), (i, j+1)])
                
                # Add edges as a LineCollection for better rendering
                if bin_edges:
                    line_segments = LineCollection(
                        bin_edges, 
                        colors='white',
                        linewidths=0.5,
                        alpha=0.7,
                        zorder=10  # Ensure lines are drawn on top of the image
                    )
                    ax.add_collection(line_segments)
                
                # Plot bin centers if available
                if bin_centers is not None:
                    x_centers, y_centers = bin_centers
                    
                    # Show only a subset of bin labels to avoid clutter
                    max_labels = min(30, len(x_centers))
                    step = max(1, len(x_centers) // max_labels)
                    
                    # Convert to physical coordinates if needed
                    if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
                        pixel_size_x = cube._pxl_size_x
                        pixel_size_y = cube._pxl_size_y
                        
                        for i in range(0, len(x_centers), step):
                            x = (x_centers[i] - nx/2) * pixel_size_x
                            y = (y_centers[i] - ny/2) * pixel_size_y
                            ax.text(
                                x, y, str(i), 
                                color='white', 
                                fontsize=8, 
                                ha='center', 
                                va='center', 
                                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'),
                                zorder=11  # On top of everything
                            )
                    else:
                        for i in range(0, len(x_centers), step):
                            ax.text(
                                x_centers[i], y_centers[i], 
                                str(i), 
                                color='white', 
                                fontsize=8, 
                                ha='center', 
                                va='center', 
                                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'),
                                zorder=11
                            )
                
                # Set axis labels
                if hasattr(cube, "_pxl_size_x") and hasattr(cube, "_pxl_size_y"):
                    ax.set_xlabel('Δ RA (arcsec)')
                    ax.set_ylabel('Δ Dec (arcsec)')
                    ax.set_aspect('equal')  # Ensure physical units are shown with correct aspect ratio
                else:
                    ax.set_xlabel('Pixels')
                    ax.set_ylabel('Pixels')
                
                # Set title
                ax.set_title(f"{galaxy_name} - Voronoi Binning")
                
                # Make sure the figure is properly sized
                plt.tight_layout()
                
                # Save with high resolution
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
                
                return fig
            
            # Create the overlay plot
            overlay_fig = create_overlay_plot()
            
            # Make sure to close the figure to avoid memory leaks
            if overlay_fig is not None:
                plt.close(overlay_fig)
            else:
                plt.close()  # Close the current figure anyway
            
            logger.info(f"Created VNB binning overlay plot")
            
        except Exception as e:
            logger.warning(f"Error creating VNB binning overlay plot: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            plt.close("all")  # Close all figures in case of error
                
    except Exception as e:
        logger.error(f"Error in create_vnb_plots: {str(e)}")
        logger.error(traceback.format_exc())
        plt.close("all")  # Close all figures in case of error
