"""
Voronoi binning analysis module for ISAPC
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm  # Required for color normalization

import galaxy_params
import visualization
from binning import (
    VoronoiBinnedData,
    run_voronoi_binning,
    calculate_wavelength_intersection,
    combine_spectra_efficiently,
    optimize_voronoi_binning
)
from utils.io import save_standardized_results

logger = logging.getLogger(__name__)


def run_vnb_analysis(args, cube, p2p_results=None):
    """
    Run Voronoi binning analysis on MUSE data cube with improved binning
    
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

    # Set up warning handling for spectral indices
    try:
        import spectral_indices
        spectral_indices.set_warnings(False)
    except ImportError:
        logger.warning("Could not import spectral_indices, indices may be limited")

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

    # Set up binning parameters - use lower SNR as requested
    # Use 15 as the default target SNR instead of 30
    target_snr = args.target_snr if hasattr(args, "target_snr") else 15
    min_snr = args.min_snr if hasattr(args, "min_snr") else 1
    use_cvt = args.cvt if hasattr(args, "cvt") else True
    high_snr_mode = args.high_snr_mode if hasattr(args, "high_snr_mode") else False
    use_physical_radius = True  # Always use physical radius in improved version
    verbose = args.verbose if hasattr(args, "verbose") else False

    # Get cube coordinates, ensuring we use the original x,y coordinates from the cube
    x = cube.x if hasattr(cube, "x") else np.arange(cube._n_x * cube._n_y) % cube._n_x
    y = cube.y if hasattr(cube, "y") else np.arange(cube._n_x * cube._n_y) // cube._n_x

    # Calculate SNR using a specific wavelength range for continuum
    wave_mask = (cube._lambda_gal >= 5075) & (cube._lambda_gal <= 5125)
    if np.sum(wave_mask) > 0:
        signal = np.nanmedian(cube._spectra[wave_mask], axis=0)
        noise = np.nanstd(cube._spectra[wave_mask], axis=0)
        logger.info("Using wavelength range 5075-5125 Å for SNR calculation")
    else:
        signal = np.nanmedian(cube._spectra, axis=0)
        noise = np.nanmedian(np.sqrt(cube._variance), axis=0) if hasattr(cube, "_variance") else np.ones_like(signal) * 0.01
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

    # Calculate physical radius if requested
    r_galaxy = None
    ellipse_params = None
    Re_arcsec = None
    try:
        # Always try to calculate physical radius
        from physical_radius import calculate_galaxy_radius, calculate_effective_radius, detect_sources
        
        # Create flux map for radius calculation
        flux_2d = np.nanmedian(cube._cube_data, axis=0)
        
        # Detect potential sources
        sources = detect_sources(flux_2d)
        if sources and len(sources) > 1:
            logger.info(f"Detected {len(sources)} potential sources")
            # Focus on the central source (usually the largest)
            central_source = sources[0]  # First source is typically the main one
            logger.info(f"Focusing on central source at ({central_source[0]}, {central_source[1]})")
        
        # Calculate physical radius and ellipse parameters with improved method
        r_galaxy, ellipse_params = calculate_galaxy_radius(
            flux_2d,
            pixel_size_x=cube._pxl_size_x,
            pixel_size_y=cube._pxl_size_y,
            focus_central=True  # Ensure we focus on the central galaxy
        )
        
        logger.info(f"Calculated physical radius with PA={ellipse_params['PA_degrees']:.1f}°, "
                    f"ε={ellipse_params['ellipticity']:.2f}")
                    
        # Store for later use
        cube._physical_radius = r_galaxy
        cube._ellipse_params = ellipse_params
        
        # Calculate effective radius (Re)
        Re_arcsec = calculate_effective_radius(
            flux_2d,
            r_galaxy,
            ellipse_params,
            pixel_size_x=cube._pxl_size_x,
            pixel_size_y=cube._pxl_size_y
        )
        
        logger.info(f"Calculated effective radius Re = {Re_arcsec:.2f} arcsec")
                    
        # Store for later use
        cube._effective_radius = Re_arcsec
        
    except Exception as e:
        logger.warning(f"Error calculating physical radius: {e}")
        use_physical_radius = False

    # Determine recommended SNR range
    min_recommended = min(2, median_snr * 0.5)
    max_recommended = max(max_pixel_snr * 1.2, median_snr * 10)

    # Ensure reasonable range
    min_recommended = max(2, min_recommended)
    max_recommended = max(50, max_recommended)

    # Adjust target_snr if needed
    if target_snr < min_recommended or target_snr > max_recommended:
        logger.warning(
            f"Specified target SNR {target_snr} is outside recommended range "
            f"({min_recommended:.1f} - {max_recommended:.1f})"
        )

        safe_target_snr = max(min_recommended, min(target_snr, max_recommended))
        logger.info(f"Adjusting target SNR to {safe_target_snr:.1f}")
    else:
        safe_target_snr = target_snr

    # Log the number of valid pixels being used
    valid_mask = np.isfinite(signal) & np.isfinite(noise) & (signal > 0) & (noise > 0)
    logger.info(f"Using {np.sum(valid_mask)} valid pixels out of {len(signal)} for Voronoi binning")
    logger.info(f"Median SNR in data: {median_snr:.1f}, Maximum SNR: {max_pixel_snr:.1f}")
    logger.info(f"Trying Voronoi binning with target SNR = {safe_target_snr:.1f}")

    # Run the binning algorithm using optimization for 10-20 bins
    if hasattr(args, "optimize_bins") and args.optimize_bins:
        logger.info("Using optimized binning to target 10-20 bins")
        bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = optimize_voronoi_binning(
            x, y, signal, noise, 
            target_bin_count=15,  # Target around 15 bins
            min_bins=10,          # Accept minimum 10 bins 
            max_bins=20,          # Accept maximum 20 bins
            cvt=use_cvt,
            quiet=not verbose
        )
    else:
        # Run with standard parameters if optimization not requested
        bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = run_voronoi_binning(
            x, y, signal, noise, 
            target_snr=safe_target_snr,
            plot=0, 
            quiet=False, 
            cvt=use_cvt,
            min_snr=min_snr
        )

    # Create bin indices
    bin_indices = []
    all_indices = np.arange(len(bin_num))
    max_bin = int(np.nanmax(bin_num))

    for i in range(max_bin + 1):
        bin_indices.append(all_indices[bin_num == i])

    # Get velocity field from P2P results if available, for velocity correction
    velocity_field = None
    gas_velocity_field = None  # Additional gas velocity field for separate correction
    
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
                
                # Try to get gas velocity field as well
                if "emission" in p2p_results and "velocity" in p2p_results["emission"]:
                    # Try to extract a representative gas velocity field
                    for line_name, vel_map in p2p_results["emission"]["velocity"].items():
                        # Use the first gas velocity map with valid values
                        if np.any(np.isfinite(vel_map)):
                            gas_velocity_field = vel_map
                            logger.info(f"Using {line_name} velocity field for gas")
                            
                            # Attach gas velocity field to the stellar field for easy access
                            velocity_field.gas_velocity_field = gas_velocity_field
                            break
        except Exception as e:
            logger.warning(f"Error extracting velocity field from P2P results: {e}")

    # Calculate wavelength intersection accounting for velocity shifts
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
    binned_spectra = combine_spectra_efficiently(
        spectra, wavelength, bin_indices, velocity_field, cube._n_x, cube._n_y,
        edge_treatment="extend",
        use_separate_velocity=True  # Enable separate stellar and gas velocity handling
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
        "bin_x": x_gen,
        "bin_y": y_gen,
        "bin_xbar": x_bar,
        "bin_ybar": y_bar,
    }
    
    # Add physical radius information if used
    if use_physical_radius and ellipse_params is not None:
        metadata["physical_radius"] = True
        metadata["ellipse_params"] = ellipse_params
        metadata["effective_radius"] = Re_arcsec  # Add effective radius
        if hasattr(cube, "_physical_radius"):
            metadata["R_galaxy_map"] = cube._physical_radius

    # Add WCS if available
    if hasattr(cube, "_wcs") and cube._wcs is not None:
        metadata["has_wcs"] = True

    # Create VoronoiBinnedData object
    binned_data = VoronoiBinnedData(
        bin_num=bin_num,
        bin_indices=bin_indices,
        spectra=binned_spectra,
        wavelength=wavelength,
        metadata=metadata,
    )

    # Set up binning in the cube - this connects our binned data to the cube
    if hasattr(cube, "setup_binning"):
        cube.setup_binning("VNB", binned_data)
    
    # Log the number of bins created
    logger.info(f"Cube set up for VNB analysis with {len(bin_indices)} bins")
    
    # Run stellar component analysis
    logger.info("Fitting stellar kinematics for binned spectra...")
    
    try:
        # Run ppxf or equivalent to get kinematics
        stellar_result = cube.fit_spectra(
            template_filename=args.template,
            ppxf_vel_init=args.vel_init,
            ppxf_vel_disp_init=args.sigma_init,
            ppxf_deg=args.poly_degree if hasattr(args, "poly_degree") else 3,
            n_jobs=args.n_jobs,
        )

        # Unpack results
        stellar_velocity_field, stellar_dispersion_field, bestfit_field, optimal_tmpls, poly_coeffs = stellar_result
    except Exception as e:
        logger.error(f"Error fitting stellar kinematics: {e}")
        # Create dummy values
        stellar_velocity_field = np.zeros(len(bin_indices))
        stellar_dispersion_field = np.zeros(len(bin_indices))
        bestfit_field = np.zeros_like(binned_spectra)
        optimal_tmpls = None
        poly_coeffs = None

    logger.info(f"Stellar kinematics completed in {time.time() - start_time:.1f} seconds")

    # Extract stellar population parameters
    stellar_pop_params = None
    if hasattr(cube, "_bin_weights") and cube._bin_weights is not None:
        try:
            logger.info("Extracting stellar population parameters for bins...")
            start_sp_time = time.time()

            # Initialize weight parser
            from stellar_population import WeightParser
            weight_parser = WeightParser(args.template)

            # Prepare arrays for physical parameters
            n_bins = len(bin_indices)
            stellar_pop_params = {
                "log_age": np.full(n_bins, np.nan),
                "age": np.full(n_bins, np.nan),
                "metallicity": np.full(n_bins, np.nan),
            }

            # Process weights for each bin
            weights = cube._bin_weights
            
            for bin_idx in range(n_bins):
                try:
                    bin_weights = weights[:, bin_idx]
                    
                    # Skip bins with invalid weights
                    if np.sum(bin_weights) > 0 and np.all(np.isfinite(bin_weights)):
                        params = weight_parser.get_physical_params(bin_weights)
                        for param_name, value in params.items():
                            stellar_pop_params[param_name][bin_idx] = value
                except Exception as e:
                    logger.debug(f"Error calculating stellar params for bin {bin_idx}: {e}")

            logger.info(f"Stellar population parameters extracted in {time.time() - start_sp_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Failed to extract stellar population parameters: {e}")
            stellar_pop_params = None

    # Calculate bin distances (in arcsec)
    bin_distances = None
    if "bin_x" in metadata and "bin_y" in metadata:
        try:
            # Get IFU center
            center_x = cube._n_x // 2
            center_y = cube._n_y // 2

            # Calculate distances in pixels
            dx = metadata["bin_x"] - center_x
            dy = metadata["bin_y"] - center_y

            # Convert to arcseconds
            bin_distances = np.sqrt(
                (dx * cube._pxl_size_x) ** 2 + (dy * cube._pxl_size_y) ** 2
            )

            logger.info(f"Calculated bin distances from center")
        except Exception as e:
            logger.warning(f"Error calculating bin distances: {e}")

    # IMPORTANT: Create the vnb_results dictionary BEFORE accessing it
    vnb_results = {
        "analysis_type": "VNB",
        "binning": {
            "bin_num": bin_num,
            "bin_x": x_gen,
            "bin_y": y_gen,
            "bin_xbar": x_bar,
            "bin_ybar": y_bar,
            "snr": sn,
            "n_pixels": n_pixels,
            "scale": scale,
            "target_snr": target_snr,
        },
        "stellar_kinematics": {
            "velocity": stellar_velocity_field,
            "dispersion": stellar_dispersion_field,
        },
        "distance": {
            "bin_distances": bin_distances,
            "effective_radius": Re_arcsec,  # Add effective radius
            "pixelsize_x": cube._pxl_size_x,
            "pixelsize_y": cube._pxl_size_y,
        },
        "meta_data": {
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
            "bin_x": x_gen,
            "bin_y": y_gen,
            "bin_xbar": x_bar,
            "bin_ybar": y_bar,
        },
    }

    # Add stellar population parameters if available
    if stellar_pop_params is not None:
        vnb_results["stellar_population"] = stellar_pop_params

    # Fit emission lines if requested
    emission_result = None
    if not hasattr(args, "no_emission") or not args.no_emission:
        try:
            logger.info("Fitting emission lines for binned spectra...")
            start_em_time = time.time()
            
            # Get configured emission lines
            emission_lines = None
            if hasattr(args, "configured_emission_lines"):
                emission_lines = args.configured_emission_lines

            # Run emission line fitting
            emission_result = cube.fit_emission_lines(
                template_filename=args.template,
                line_names=emission_lines,
                ppxf_vel_init=stellar_velocity_field,  # Use stellar velocity field as initial guess
                ppxf_sig_init=args.sigma_init,
                ppxf_deg=2,  # Simpler polynomial for emission lines
                n_jobs=args.n_jobs,
            )
            
            logger.info(f"Emission line fitting completed in {time.time() - start_em_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Error fitting emission lines: {e}")
            emission_result = None

    # Calculate spectral indices if requested
    indices_result = None
    if not hasattr(args, "no_indices") or not args.no_indices:
        try:
            logger.info("Calculating spectral indices for binned spectra...")
            start_idx_time = time.time()
            
            # Get configured spectral indices
            indices_list = None
            if hasattr(args, "configured_indices"):
                indices_list = args.configured_indices
                    
            # Get methods list (continuum modes)
            methods = ['auto', 'original', 'fit']
            if hasattr(args, "indices_methods"):
                methods = args.indices_methods
            
            # Calculate indices with multiple methods
            indices_result = cube.calculate_spectral_indices_multi_method(
                indices_list=indices_list,
                n_jobs=args.n_jobs,
                verbose=verbose,
                save_mode='VNB',
                save_path=plots_dir
            )
            
            logger.info(f"Spectral indices calculation completed in {time.time() - start_idx_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Error calculating spectral indices: {e}")
            import traceback
            logger.error(traceback.format_exc())
            indices_result = None

    # Add spectral indices results if available - AFTER creating vnb_results
    if indices_result is not None:
        if isinstance(indices_result, dict) and "auto" in indices_result:
            # Results with multiple methods format
            vnb_results["bin_indices_multi"] = indices_result
            # For backward compatibility, use auto method as default
            vnb_results["bin_indices"] = indices_result["auto"]
        else:
            # Standard single-method format
            vnb_results["bin_indices"] = indices_result

    # Save binned data for later reuse
    try:
        binned_data.save(data_dir / f"{galaxy_name}_VNB_binned.npz")
    except Exception as e:
        logger.warning(f"Error saving binned data: {e}")

    # Add emission line results if available
    if emission_result is not None:
        vnb_results["emission"] = {}
        
        # Extract emission line information - flux, velocity, dispersion
        for key in ["flux", "velocity", "dispersion"]:
            if key in emission_result and emission_result[key]:
                vnb_results["emission"][key] = emission_result[key]
                
        # Add signal/noise information
        for key in ["signal", "noise", "snr"]:
            if key in emission_result and emission_result[key] is not None:
                vnb_results[key] = emission_result[key]

    # Save standardized results
    should_save = not hasattr(args, "no_save") or not args.no_save
    if should_save:
        save_standardized_results(galaxy_name, "VNB", vnb_results, output_dir)

    # Create visualization plots
    should_plot = not hasattr(args, "no_plots") or not args.no_plots
    if should_plot:
        try:
            # Check for potential dimension mismatches before plotting
            if 'stellar_kinematics' in vnb_results and hasattr(vnb_results['stellar_kinematics']['velocity'], 'shape'):
                logger.info(f"Velocity field shape: {vnb_results['stellar_kinematics']['velocity'].shape}")
            create_vnb_plots(cube, vnb_results, galaxy_name, plots_dir, args)
        except Exception as e:
            logger.warning(f"Error creating plots: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    logger.info("Voronoi binning analysis completed")
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
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from matplotlib.patches import Ellipse
        from matplotlib.colors import LogNorm
        
        # Create bin map plot with physical scaling
        if "binning" in vnb_results and "bin_num" in vnb_results["binning"]:
            try:
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    # Make sure the dimensions match the cube
                    if len(bin_num) == cube._n_x * cube._n_y:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({cube._n_y}x{cube._n_x})")
                        # Create a dummy bin_num_2d with correct dimensions
                        bin_num_2d = np.zeros((cube._n_y, cube._n_x), dtype=int)
                        # Fill with what we can
                        valid_len = min(len(bin_num), cube._n_y * cube._n_x)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
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
                    wcs=cube._wcs if hasattr(cube, "_wcs") else None
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_VNB_bin_map.png"
                )
                plt.close(fig)
                
                # Create SNR map
                if "snr" in vnb_results["binning"]:
                    try:
                        snr = vnb_results["binning"]["snr"]
                        
                        # Ensure snr is a numpy array of proper type
                        snr = np.asarray(snr, dtype=float)
                        
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
                            wcs=cube._wcs if hasattr(cube, "_wcs") else None
                        )
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_VNB_snr_map.png"
                        )
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating SNR map: {e}")
                        plt.close('all')
                
                # Create pixels-per-bin map
                if "n_pixels" in vnb_results["binning"]:
                    try:
                        n_pixels = vnb_results["binning"]["n_pixels"]
                        
                        # Ensure n_pixels is a numpy array of proper type
                        n_pixels = np.asarray(n_pixels, dtype=float)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        visualization.plot_bin_map(
                            bin_num_2d,
                            n_pixels,
                            ax=ax,
                            cmap="magma",
                            title=f"{galaxy_name} - Pixels per Bin",
                            colorbar_label="Number of Pixels",
                            physical_scale=True,
                            pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                            wcs=cube._wcs if hasattr(cube, "_wcs") else None,
                            log_scale=True
                        )
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_VNB_pixels_per_bin.png"
                        )
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating pixels-per-bin map: {e}")
                        plt.close('all')
            except Exception as e:
                logger.warning(f"Error creating bin map: {e}")
                plt.close("all")
        
        # Create effective radius visualization
        if "distance" in vnb_results and "effective_radius" in vnb_results["distance"]:
            try:
                Re = vnb_results["distance"]["effective_radius"]
                
                if Re is not None and Re > 0:
                    # Get ellipse parameters if available
                    ellipse_params = None
                    if "meta_data" in vnb_results and "ellipse_params" in vnb_results["meta_data"]:
                        ellipse_params = vnb_results["meta_data"]["ellipse_params"]
                    elif hasattr(cube, "_ellipse_params"):
                        ellipse_params = cube._ellipse_params
                    
                    # Create flux map
                    flux_map = visualization.prepare_flux_map(cube)
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot flux with log scale if possible
                    valid_flux = flux_map[np.isfinite(flux_map) & (flux_map > 0)]
                    if len(valid_flux) > 0:
                        vmin = np.percentile(valid_flux, 1)
                        vmax = np.percentile(valid_flux, 99)
                        norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
                    else:
                        norm = None
                        
                    im = ax.imshow(flux_map, origin='lower', cmap='inferno', norm=norm)
                    plt.colorbar(im, ax=ax, label='Flux')
                    
                    # Draw effective radius ellipse
                    if ellipse_params is not None:
                        # Get center and shape parameters
                        center_x = ellipse_params["center_x"]
                        center_y = ellipse_params["center_y"]
                        pa = ellipse_params["PA_degrees"]
                        ellipticity = ellipse_params["ellipticity"]
                        
                        # Convert Re to pixels
                        Re_pix = Re / cube._pxl_size_x
                        
                        # Draw the ellipse representing Re
                        ell = Ellipse(
                            (center_x, center_y),
                            2 * Re_pix,  # Diameter
                            2 * Re_pix * (1 - ellipticity),  # Account for ellipticity
                            angle=pa,
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=2.0
                        )
                        ax.add_patch(ell)
                        
                        # Add center marker
                        ax.plot(center_x, center_y, '+', color='white', markersize=10)
                    else:
                        # Use a circular Re if no ellipse parameters
                        center_x = cube._n_x // 2
                        center_y = cube._n_y // 2
                        
                        # Convert Re to pixels
                        Re_pix = Re / cube._pxl_size_x
                        
                        # Draw the circle representing Re
                        from matplotlib.patches import Circle
                        circ = Circle(
                            (center_x, center_y),
                            Re_pix,  # Radius
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=2.0
                        )
                        ax.add_patch(circ)
                        
                        # Add center marker
                        ax.plot(center_x, center_y, '+', color='white', markersize=10)
                    
                    ax.set_title(f'{galaxy_name} - Effective Radius (Re = {Re:.2f} arcsec)')
                    
                    visualization.standardize_figure_saving(
                        fig, plots_dir / f"{galaxy_name}_VNB_effective_radius.png"
                    )
                    plt.close(fig)
                
            except Exception as e:
                logger.warning(f"Error creating effective radius visualization: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                plt.close("all")
                
        # Create kinematics plots
        if "stellar_kinematics" in vnb_results:
            try:
                velocity = vnb_results["stellar_kinematics"]["velocity"]
                dispersion = vnb_results["stellar_kinematics"]["dispersion"]
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Convert to proper numpy arrays
                velocity = np.asarray(velocity, dtype=float)
                dispersion = np.asarray(dispersion, dtype=float)
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    # Make sure the dimensions match the cube
                    if len(bin_num) == cube._n_x * cube._n_y:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({cube._n_y}x{cube._n_x})")
                        # Create a dummy bin_num_2d with correct dimensions
                        bin_num_2d = np.zeros((cube._n_y, cube._n_x), dtype=int)
                        # Fill with what we can
                        valid_len = min(len(bin_num), cube._n_y * cube._n_x)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
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
                    wcs=cube._wcs if hasattr(cube, "_wcs") else None
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
                    wcs=cube._wcs if hasattr(cube, "_wcs") else None
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_VNB_dispersion_map.png"
                )
                plt.close(fig)
                
                # Create rotation curve and V/σ plot
                if "distance" in vnb_results and "bin_distances" in vnb_results["distance"]:
                    try:
                        bin_distances = vnb_results["distance"]["bin_distances"]
                        
                        # Ensure proper numpy array type
                        bin_distances = np.asarray(bin_distances, dtype=float)
                        
                        # Sort bins by distance from center
                        sorted_indices = np.argsort(bin_distances)
                        sorted_distances = bin_distances[sorted_indices]
                        sorted_velocity = velocity[sorted_indices]
                        sorted_dispersion = dispersion[sorted_indices]
                        
                        # Calculate V/σ
                        v_sigma = np.abs(sorted_velocity) / sorted_dispersion
                        
                        # Create combined plot
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        # Plot rotation curve
                        ax1.plot(sorted_distances, sorted_velocity, 'o-')
                        ax1.set_xlabel('Radius (arcsec)')
                        ax1.set_ylabel('Velocity (km/s)')
                        ax1.set_title(f'{galaxy_name} - Rotation Curve')
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot V/σ
                        ax2.plot(sorted_distances, v_sigma, 'o-')
                        ax2.set_xlabel('Radius (arcsec)')
                        ax2.set_ylabel('|V|/σ')
                        ax2.set_title(f'{galaxy_name} - V/σ Profile')
                        ax2.grid(True, alpha=0.3)
                        
                        # Horizontal line at V/σ = 1
                        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7)
                        
                        plt.tight_layout()
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_VNB_rotation_vsigma.png"
                        )
                        plt.close(fig)
                        
                        # Create profiles in Re units if effective radius is available
                        if "effective_radius" in vnb_results["distance"]:
                            try:
                                Re = vnb_results["distance"]["effective_radius"]
                                
                                if Re is not None and Re > 0:
                                    # Create a version of distances in Re units
                                    distances_in_Re = sorted_distances / Re
                                    
                                    # Velocity profile in Re units
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(distances_in_Re, sorted_velocity, 'o-', markersize=6, linewidth=1.5)
                                    ax.set_xlabel('R/Re')
                                    ax.set_ylabel('Velocity (km/s)')
                                    ax.set_title(f'{galaxy_name} - Rotation Curve (Re = {Re:.2f} arcsec)')
                                    ax.grid(True, alpha=0.3)
                                    
                                    # Add vertical line at Re
                                    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                    ax.legend()
                                    
                                    visualization.standardize_figure_saving(
                                        fig, plots_dir / f"{galaxy_name}_VNB_velocity_vs_Re.png"
                                    )
                                    plt.close(fig)
                                    
                                    # Dispersion profile in Re units
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(distances_in_Re, sorted_dispersion, 'o-', markersize=6, linewidth=1.5)
                                    ax.set_xlabel('R/Re')
                                    ax.set_ylabel('Dispersion (km/s)')
                                    ax.set_title(f'{galaxy_name} - Dispersion Profile (Re = {Re:.2f} arcsec)')
                                    ax.grid(True, alpha=0.3)
                                    
                                    # Add vertical line at Re
                                    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                    ax.legend()
                                    
                                    visualization.standardize_figure_saving(
                                        fig, plots_dir / f"{galaxy_name}_VNB_dispersion_vs_Re.png"
                                    )
                                    plt.close(fig)
                                    
                                    # V/σ profile in Re units
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(distances_in_Re, v_sigma, 'o-', markersize=6, linewidth=1.5)
                                    ax.set_xlabel('R/Re')
                                    ax.set_ylabel('|V|/σ')
                                    ax.set_title(f'{galaxy_name} - V/σ Profile (Re = {Re:.2f} arcsec)')
                                    ax.grid(True, alpha=0.3)
                                    
                                    # Add horizontal line at V/σ = 1
                                    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
                                    
                                    # Add vertical line at Re
                                    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                    ax.legend()
                                    
                                    visualization.standardize_figure_saving(
                                        fig, plots_dir / f"{galaxy_name}_VNB_vsigma_vs_Re.png"
                                    )
                                    plt.close(fig)
                                    
                            except Exception as e:
                                logger.warning(f"Error creating profiles in Re units: {e}")
                                plt.close('all')
                        
                    except Exception as e:
                        logger.warning(f"Error creating rotation curve plot: {e}")
                        plt.close('all')
                
                # Create kinematics summary plot
                try:
                    # Create 2D velocity and dispersion fields for plotting
                    vel_2d = np.full(bin_num_2d.shape, np.nan, dtype=float)
                    disp_2d = np.full(bin_num_2d.shape, np.nan, dtype=float)
                    
                    # Fill in values for each bin
                    for i, (vel, disp) in enumerate(zip(velocity, dispersion)):
                        if i < len(velocity):
                            vel_2d[bin_num_2d == i] = vel
                        if i < len(dispersion):
                            disp_2d[bin_num_2d == i] = disp
                    
                    # Create summary plot
                    fig = visualization.plot_kinematics_summary(
                        velocity_field=vel_2d,
                        dispersion_field=disp_2d,
                        equal_aspect=True,
                        physical_scale=True,
                        pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                        wcs=cube._wcs if hasattr(cube, "_wcs") else None
                    )
                    
                    if fig is not None:
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_VNB_kinematics_summary.png"
                        )
                        plt.close(fig)
                    else:
                        logger.warning("Failed to create kinematics summary plot")
                except Exception as e:
                    logger.warning(f"Error creating kinematics summary: {e}")
                    plt.close("all")
                
            except Exception as e:
                logger.warning(f"Error creating kinematics plots: {e}")
                plt.close("all")
                
        # Create stellar population plots if available
        if "stellar_population" in vnb_results:
            try:
                # Create directory for stellar population plots
                stellar_dir = plots_dir / "stellar_population"
                stellar_dir.mkdir(exist_ok=True, parents=True)
                
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    if len(bin_num) == cube._n_x * cube._n_y:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({cube._n_y}x{cube._n_x})")
                        bin_num_2d = np.zeros((cube._n_y, cube._n_x), dtype=int)
                        valid_len = min(len(bin_num), cube._n_y * cube._n_x)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
                else:
                    bin_num_2d = bin_num
                
                # Process each parameter
                for param_name, param_values in vnb_results["stellar_population"].items():
                    # Convert to numpy array if needed
                    param_values = np.asarray(param_values, dtype=float)
                    
                    # Define parameter info
                    param_info = {
                        "log_age": {"title": "Log Age [yr]", "cmap": "plasma"},
                        "age": {"title": "Age [Gyr]", "cmap": "plasma", "scale_factor": 1e-9},
                        "metallicity": {"title": "Metallicity [Z/H]", "cmap": "viridis"}
                    }
                    
                    # Get parameter settings
                    info = param_info.get(param_name, {"title": param_name, "cmap": "viridis"})
                    
                    # Apply scale factor if needed
                    values_to_plot = param_values
                    if "scale_factor" in info:
                        values_to_plot = param_values * info["scale_factor"]
                    
                    # Create bin map with this parameter
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot the map
                    visualization.plot_bin_map(
                        bin_num_2d,
                        values_to_plot,
                        ax=ax,
                        cmap=info["cmap"],
                        title=f"{galaxy_name} - Stellar {info['title']}",
                        colorbar_label=info["title"],
                        physical_scale=True,
                        pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                        wcs=cube._wcs if hasattr(cube, "_wcs") else None
                    )
                    
                    visualization.standardize_figure_saving(
                        fig, stellar_dir / f"{galaxy_name}_VNB_{param_name}.png"
                    )
                    plt.close(fig)
                    
                    # Create radial profile if distances available
                    if "distance" in vnb_results and "bin_distances" in vnb_results["distance"]:
                        try:
                            bin_distances = vnb_results["distance"]["bin_distances"]
                            
                            # Sort bins by distance from center
                            sorted_indices = np.argsort(bin_distances)
                            sorted_distances = bin_distances[sorted_indices]
                            sorted_values = values_to_plot[sorted_indices]
                            
                            # Create profile plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(sorted_distances, sorted_values, 'o-')
                            ax.set_xlabel('Radius (arcsec)')
                            ax.set_ylabel(info["title"])
                            ax.set_title(f'{galaxy_name} - Stellar {info["title"]} Profile')
                            ax.grid(True, alpha=0.3)
                            
                            visualization.standardize_figure_saving(
                                fig, stellar_dir / f"{galaxy_name}_VNB_{param_name}_profile.png"
                            )
                            plt.close(fig)
                            
                            # Create profile in Re units if effective radius is available
                            if "effective_radius" in vnb_results["distance"]:
                                try:
                                    Re = vnb_results["distance"]["effective_radius"]
                                    
                                    if Re is not None and Re > 0:
                                        # Create a version of distances in Re units
                                        distances_in_Re = sorted_distances / Re
                                        
                                        # Create profile plot in Re units
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.plot(distances_in_Re, sorted_values, 'o-', markersize=6, linewidth=1.5)
                                        ax.set_xlabel('R/Re')
                                        ax.set_ylabel(info["title"])
                                        ax.set_title(f'{galaxy_name} - Stellar {info["title"]} Profile (Re = {Re:.2f} arcsec)')
                                        ax.grid(True, alpha=0.3)
                                        
                                        # Add vertical line at Re
                                        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                        ax.legend()
                                        
                                        visualization.standardize_figure_saving(
                                            fig, stellar_dir / f"{galaxy_name}_VNB_{param_name}_vs_Re.png"
                                        )
                                        plt.close(fig)
                                        
                                except Exception as e:
                                    logger.warning(f"Error creating profile in Re units for parameter {param_name}: {e}")
                                    plt.close('all')
                                    
                        except Exception as e:
                            logger.warning(f"Error creating profile for parameter {param_name}: {e}")
                            plt.close('all')
                    
            except Exception as e:
                logger.warning(f"Error creating stellar population plots: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                plt.close('all')
                
        # Create emission line plots if available
        if "emission" in vnb_results:
            try:
                emission = vnb_results["emission"]
                bin_num = vnb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    # Make sure the dimensions match the cube
                    if len(bin_num) == cube._n_x * cube._n_y:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({cube._n_y}x{cube._n_x})")
                        # Create a dummy bin_num_2d with correct dimensions
                        bin_num_2d = np.zeros((cube._n_y, cube._n_x), dtype=int)
                        # Fill with what we can
                        valid_len = min(len(bin_num), cube._n_y * cube._n_x)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
                else:
                    bin_num_2d = bin_num
                    
                # Create emission line plots directory
                emission_dir = plots_dir / "emission_lines"
                emission_dir.mkdir(exist_ok=True, parents=True)
                
                # Process each emission line field
                for field_name, field_data in emission.items():
                    if field_name in ["flux", "velocity", "dispersion"] and isinstance(field_data, dict):
                        for line_name, line_values in field_data.items():
                            try:
                                # Convert to numpy array
                                line_values = np.asarray(line_values, dtype=float)
                                
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
                                        wcs=cube._wcs if hasattr(cube, "_wcs") else None
                                    )
                                    visualization.standardize_figure_saving(
                                        fig, emission_dir / f"{galaxy_name}_VNB_{line_name}_{field_name}.png"
                                    )
                                    plt.close(fig)
                                    
                                    # Create radial profile if distances available
                                    if "distance" in vnb_results and "bin_distances" in vnb_results["distance"]:
                                        try:
                                            bin_distances = vnb_results["distance"]["bin_distances"]
                                            
                                            # Sort bins by distance from center
                                            sorted_indices = np.argsort(bin_distances)
                                            sorted_distances = bin_distances[sorted_indices]
                                            sorted_values = line_values[sorted_indices]
                                            
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            ax.plot(sorted_distances, sorted_values, 'o-')
                                            ax.set_xlabel('Radius (arcsec)')
                                            
                                            if field_name == "flux":
                                                ax.set_ylabel('Flux')
                                                if np.all(sorted_values[np.isfinite(sorted_values)] > 0):
                                                    ax.set_yscale('log')
                                            elif field_name == "velocity":
                                                ax.set_ylabel('Velocity (km/s)')
                                            else:
                                                ax.set_ylabel('Dispersion (km/s)')
                                                
                                            ax.set_title(f'{galaxy_name} - {line_name} {field_name.capitalize()} Profile')
                                            ax.grid(True, alpha=0.3)
                                            
                                            visualization.standardize_figure_saving(
                                                fig, emission_dir / f"{galaxy_name}_VNB_{line_name}_{field_name}_profile.png"
                                            )
                                            plt.close(fig)
                                            
                                            # Create profile in Re units if effective radius is available
                                            if "effective_radius" in vnb_results["distance"]:
                                                try:
                                                    Re = vnb_results["distance"]["effective_radius"]
                                                    
                                                    if Re is not None and Re > 0:
                                                        # Create a version of distances in Re units
                                                        distances_in_Re = sorted_distances / Re
                                                        
                                                        # Create profile plot in Re units
                                                        fig, ax = plt.subplots(figsize=(10, 6))
                                                        ax.plot(distances_in_Re, sorted_values, 'o-', markersize=6, linewidth=1.5)
                                                        ax.set_xlabel('R/Re')
                                                        
                                                        if field_name == "flux":
                                                            ax.set_ylabel('Flux')
                                                            if np.all(sorted_values[np.isfinite(sorted_values)] > 0):
                                                                ax.set_yscale('log')
                                                        elif field_name == "velocity":
                                                            ax.set_ylabel('Velocity (km/s)')
                                                        else:
                                                            ax.set_ylabel('Dispersion (km/s)')
                                                            
                                                        ax.set_title(f'{galaxy_name} - {line_name} {field_name.capitalize()} Profile (Re = {Re:.2f} arcsec)')
                                                        ax.grid(True, alpha=0.3)
                                                        
                                                        # Add vertical line at Re
                                                        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                                        ax.legend()
                                                        
                                                        visualization.standardize_figure_saving(
                                                            fig, emission_dir / f"{galaxy_name}_VNB_{line_name}_{field_name}_vs_Re.png"
                                                        )
                                                        plt.close(fig)
                                                        
                                                except Exception as e:
                                                    logger.warning(f"Error creating profile in Re units for {line_name} {field_name}: {e}")
                                                    plt.close('all')
                                                    
                                        except Exception as e:
                                            logger.warning(f"Error creating radial profile for {line_name} {field_name}: {e}")
                                            plt.close('all')
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
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    if len(bin_num) == cube._n_x * cube._n_y:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({cube._n_y}x{cube._n_x})")
                        # Create a dummy bin_num_2d with correct dimensions
                        bin_num_2d = np.zeros((cube._n_y, cube._n_x), dtype=int)
                        # Fill with what we can
                        valid_len = min(len(bin_num), cube._n_y * cube._n_x)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
                else:
                    bin_num_2d = bin_num
                
                # Create spectral indices directory
                indices_dir = plots_dir / "spectral_indices"
                indices_dir.mkdir(exist_ok=True, parents=True)
                
                # Convert bin_indices to a regular dictionary if it isn't already
                if not isinstance(bin_indices, dict):
                    try:
                        # Try to convert np.lib.npyio.NpzFile or other array-like
                        bin_indices_dict = {}
                        for key in bin_indices:
                            # Convert to native Python types
                            if isinstance(bin_indices[key], np.ndarray):
                                bin_indices_dict[str(key)] = bin_indices[key].tolist()
                            else:
                                bin_indices_dict[str(key)] = bin_indices[key]
                        bin_indices = bin_indices_dict
                    except:
                        logger.warning("Could not convert bin_indices to dictionary")
                        bin_indices = {}
                
                # Process each spectral index
                for idx_name, idx_values in bin_indices.items():
                    try:
                        # Make sure we have a proper numpy array
                        idx_array = np.asarray(idx_values, dtype=float)
                        
                        if np.any(np.isfinite(idx_array)):
                            # Create map with physical scaling
                            fig, ax = plt.subplots(figsize=(10, 8))
                            visualization.plot_bin_map(
                                bin_num_2d,
                                idx_array,
                                ax=ax,
                                cmap="plasma",
                                title=f"{galaxy_name} - {idx_name}",
                                colorbar_label="Index Value",
                                physical_scale=True,
                                pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                                wcs=cube._wcs if hasattr(cube, "_wcs") else None
                            )
                            visualization.standardize_figure_saving(
                                fig, indices_dir / f"{galaxy_name}_VNB_{idx_name}.png"
                            )
                            plt.close(fig)
                            
                            # Create radial profile if distances available
                            if "distance" in vnb_results and "bin_distances" in vnb_results["distance"]:
                                try:
                                    bin_distances = vnb_results["distance"]["bin_distances"]
                                    
                                    # Ensure bin_distances is same length as idx_array
                                    if len(bin_distances) == len(idx_array):
                                        # Sort bins by distance from center
                                        sorted_indices = np.argsort(bin_distances)
                                        sorted_distances = bin_distances[sorted_indices]
                                        sorted_values = idx_array[sorted_indices]
                                        
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.plot(sorted_distances, sorted_values, 'o-')
                                        ax.set_xlabel('Radius (arcsec)')
                                        ax.set_ylabel(f'{idx_name} Value')
                                        ax.set_title(f'{galaxy_name} - {idx_name} Radial Profile')
                                        ax.grid(True, alpha=0.3)
                                        
                                        visualization.standardize_figure_saving(
                                            fig, indices_dir / f"{galaxy_name}_VNB_{idx_name}_profile.png"
                                        )
                                        plt.close(fig)
                                        
                                        # Create profile in Re units if effective radius is available
                                        if "effective_radius" in vnb_results["distance"]:
                                            try:
                                                Re = vnb_results["distance"]["effective_radius"]
                                                
                                                if Re is not None and Re > 0:
                                                    # Create a version of distances in Re units
                                                    distances_in_Re = sorted_distances / Re
                                                    
                                                    # Create profile plot in Re units
                                                    fig, ax = plt.subplots(figsize=(10, 6))
                                                    ax.plot(distances_in_Re, sorted_values, 'o-', markersize=6, linewidth=1.5)
                                                    ax.set_xlabel('R/Re')
                                                    ax.set_ylabel(f'{idx_name} Value')
                                                    ax.set_title(f'{galaxy_name} - {idx_name} Profile (Re = {Re:.2f} arcsec)')
                                                    ax.grid(True, alpha=0.3)
                                                    
                                                    # Add vertical line at Re
                                                    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                                    ax.legend()
                                                    
                                                    visualization.standardize_figure_saving(
                                                        fig, indices_dir / f"{galaxy_name}_VNB_{idx_name}_vs_Re.png"
                                                    )
                                                    plt.close(fig)
                                                    
                                            except Exception as e:
                                                logger.warning(f"Error creating profile in Re units for index {idx_name}: {e}")
                                                plt.close('all')
                                except Exception as e:
                                    logger.warning(f"Error creating radial profile for index {idx_name}: {e}")
                                    plt.close('all')
                    except Exception as e:
                        logger.warning(f"Error creating plot for index {idx_name}: {e}")
                        plt.close("all")
                
                # Create detailed spectral index plots for selected bins
                if hasattr(cube, "_spectra") and hasattr(cube, "_lambda_gal") and hasattr(cube, "_template_weights"):
                    try:
                        # Import spectral_indices
                        import spectral_indices
                        from spectral_indices import LineIndexCalculator
                        
                        # Create spectral indices subdirectory
                        idx_detail_dir = indices_dir / "detailed"
                        idx_detail_dir.mkdir(exist_ok=True, parents=True)
                        
                        # Select a subset of bins to plot
                        n_bins = len(vnb_results["binning"]["bin_x"])
                        bins_to_plot = []
                        
                        if n_bins <= 20:
                            bins_to_plot = list(range(n_bins))
                        else:
                            # Plot first, middle and last, plus some evenly spaced ones
                            bins_to_plot = [0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1]
                            # Add a few more evenly spaced ones
                            if n_bins > 50:
                                step = n_bins // 5
                                for i in range(0, n_bins, step):
                                    if i not in bins_to_plot:
                                        bins_to_plot.append(i)
                                bins_to_plot.sort()
                        
                        # Get binned spectra
                        if hasattr(cube, "_binned_data") and cube._binned_data is not None:
                            binned_spectra = cube._binned_data.spectra
                            wavelength = cube._binned_data.wavelength
                            
                            # Plot selected bins
                            for bin_idx in bins_to_plot:
                                if bin_idx < binned_spectra.shape[1]:
                                    try:
                                        # Get stellar template fit
                                        if hasattr(cube, "_bin_bestfit") and cube._bin_bestfit is not None:
                                            bestfit = cube._bin_bestfit[:, bin_idx]
                                        else:
                                            # Skip if no bestfit available
                                            continue
                                            
                                        # Get template and optimal weights
                                        if hasattr(cube, "_sps") and hasattr(cube, "_optimal_weights"):
                                            template = cube._sps.lam_temp
                                            weights = cube._optimal_weights[:, bin_idx]
                                        else:
                                            template = wavelength
                                            weights = np.ones_like(wavelength)
                                        
                                        # Get gas emission if available
                                        em_wave = None
                                        em_flux = None
                                        if hasattr(cube, "_gas_bestfit") and cube._gas_bestfit is not None:
                                            em_wave = wavelength
                                            em_flux = cube._gas_bestfit[:, bin_idx]
                                        
                                        # Get velocity
                                        vel = vnb_results["stellar_kinematics"]["velocity"][bin_idx]
                                        
                                        # Create calculator
                                        calc = LineIndexCalculator(
                                            wave=wavelength,
                                            flux=binned_spectra[:, bin_idx],
                                            fit_wave=template,
                                            fit_flux=weights,
                                            em_wave=em_wave,
                                            em_flux_list=em_flux,
                                            velocity_correction=vel,
                                            continuum_mode="fit",
                                            show_warnings=False
                                        )
                                        
                                        # Plot all lines
                                        fig, axes = calc.plot_all_lines(
                                            mode="VNB",
                                            number=bin_idx,
                                            save_path=str(idx_detail_dir),
                                            show_index=True
                                        )
                                        plt.close(fig)
                                    except Exception as e:
                                        logger.warning(f"Error creating spectral index plot for bin {bin_idx}: {e}")
                                        import traceback
                                        logger.debug(traceback.format_exc())
                                        plt.close("all")
                    except Exception as e:
                        logger.warning(f"Error creating detailed spectral index plots: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        plt.close("all")
            except Exception as e:
                logger.warning(f"Error creating spectral indices plots: {e}")
                plt.close("all")
        
        # Create overview and diagnostic plots
        try:
            # Create a comprehensive diagnostic plot
            diagnostic_fig = visualization.create_diagnostic_plot(
                cube, 
                title=f"{galaxy_name} - VNB Analysis Overview",
                output_dir=plots_dir
            )
            plt.close(diagnostic_fig)
        except Exception as e:
            logger.warning(f"Error creating diagnostic plot: {e}")
            plt.close('all')
        
        # Create binned spectra plots
        if hasattr(cube, "_binned_data") and cube._binned_data is not None:
            try:
                # Create binned spectra visualization directory
                spectra_dir = plots_dir / "spectra"
                spectra_dir.mkdir(exist_ok=True, parents=True)
                
                # Get binned data
                binned_spectra = cube._binned_data.spectra
                wavelength = cube._binned_data.wavelength
                
                # Create a plot of representative binned spectra
                fig, ax = visualization.plot_binned_spectra(
                    cube,
                    binned_spectra,
                    wavelength,
                    title=f"{galaxy_name} - VNB Binned Spectra",
                    save_path=spectra_dir / f"{galaxy_name}_VNB_binned_spectra.png"
                )
                plt.close(fig)
                
                # Create a more detailed plot showing the spectrum range used for analysis
                if hasattr(cube, "_goodwavelengthrange"):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot median spectrum of all bins
                    median_spec = np.nanmedian(binned_spectra, axis=1)
                    ax.plot(wavelength, median_spec, 'k-', label='Median Spectrum')
                    
                    # Highlight the good wavelength range
                    good_range = cube._goodwavelengthrange
                    ax.axvspan(good_range[0], good_range[1], alpha=0.2, color='green', 
                              label='Analysis Range')
                    
                    # Add labels for key spectral features
                    features = {
                        4861: 'Hβ',
                        4959: '[OIII]',
                        5007: '[OIII]',
                        5177: 'Mgb'
                    }
                    
                    for wave, name in features.items():
                        if min(wavelength) < wave < max(wavelength):
                            ax.axvline(x=wave, color='red', linestyle=':', alpha=0.7)
                            # Get y position for text
                            ypos = ax.get_ylim()[0] + 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                            ax.text(wave, ypos, name, rotation=90, va='top', ha='center')
                    
                    ax.set_xlabel('Wavelength (Å)')
                    ax.set_ylabel('Flux')
                    ax.set_title(f"{galaxy_name} - VNB Spectral Range")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    visualization.standardize_figure_saving(
                        fig, spectra_dir / f"{galaxy_name}_VNB_spectral_range.png"
                    )
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating binned spectra plots: {e}")
                plt.close('all')
        
        # Create detailed spectral fit plots
        if hasattr(cube, "_binned_data") and hasattr(cube, "_bin_bestfit"):
            try:
                # Create spectral fit visualization directory
                fit_dir = plots_dir / "spectral_fits"
                fit_dir.mkdir(exist_ok=True, parents=True)
                
                # Get binned data
                binned_spectra = cube._binned_data.spectra
                wavelength = cube._binned_data.wavelength
                bestfit = cube._bin_bestfit
                
                # Get emission component if available
                emission = None
                if hasattr(cube, "_gas_bestfit") and cube._gas_bestfit is not None:
                    emission = cube._gas_bestfit
                
                # Number of bins to plot
                n_bins = binned_spectra.shape[1]
                bins_to_plot = []
                
                if n_bins <= 12:
                    # Plot all bins if few
                    bins_to_plot = list(range(n_bins))
                else:
                    # Plot selection of bins (first, some middle, last)
                    bins_to_plot = [0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1]
                
                # Get bin distances if available
                bin_distances = None
                if "distance" in vnb_results and "bin_distances" in vnb_results["distance"]:
                    bin_distances = vnb_results["distance"]["bin_distances"]
                
                for bin_id in bins_to_plot:
                    if bin_id < binned_spectra.shape[1] and bin_id < bestfit.shape[1]:
                        try:
                            # Create comprehensive spectral fit plot
                            observed = binned_spectra[:, bin_id]
                            model = bestfit[:, bin_id]
                            residuals = observed - model
                            
                            # Get emission component if available
                            em_component = None
                            if emission is not None and bin_id < emission.shape[1]:
                                em_component = emission[:, bin_id]
                            
                            # Create title with bin info
                            title = f"{galaxy_name} - VNB Bin {bin_id} Spectral Fit"
                            
                            # Add distance information if available
                            if bin_distances is not None and bin_id < len(bin_distances):
                                title += f" (r = {bin_distances[bin_id]:.1f} arcsec)"
                            
                            # Create plot
                            fig, ax = visualization.plot_ppxf_fit(
                                wavelength, observed, model, residuals, em_component,
                                title=title,
                                redshift=cube._redshift if hasattr(cube, "_redshift") else 0.0,
                                save_path=fit_dir / f"{galaxy_name}_VNB_fit_bin{bin_id}.png"
                            )
                            plt.close(fig)
                        except Exception as e:
                            logger.warning(f"Error creating spectral fit plot for bin {bin_id}: {e}")
                            plt.close('all')
                            
                # Create a combined spectral fit figure showing a selection of bins
                try:
                    # For combined plot, limit to 4 bins maximum
                    combined_bins = bins_to_plot[:min(4, len(bins_to_plot))]
                    
                    if combined_bins:
                        fig, axes = plt.subplots(len(combined_bins), 1, figsize=(12, 3*len(combined_bins)), sharex=True)
                        
                        # Handle single subplot case
                        if len(combined_bins) == 1:
                            axes = [axes]
                        
                        for i, bin_id in enumerate(combined_bins):
                            # Get data
                            observed = binned_spectra[:, bin_id]
                            model = bestfit[:, bin_id]
                            
                            # Plot data and model
                            axes[i].plot(wavelength, observed, 'k-', alpha=0.7, label='Observed')
                            axes[i].plot(wavelength, model, 'r-', alpha=0.9, label='Model')
                            
                            # Add emission if available
                            if emission is not None and bin_id < emission.shape[1]:
                                em_component = emission[:, bin_id]
                                if np.any(em_component != 0):
                                    # Scale emission to be visible
                                    scale = 0.5 * np.nanmax(observed) / np.nanmax(em_component) if np.nanmax(em_component) > 0 else 1
                                    axes[i].plot(wavelength, em_component * scale, 'g-', alpha=0.7, 
                                               label=f'Emission (×{scale:.1f})')
                            
                            # Add bin information
                            bin_label = f"Bin {bin_id}"
                            if bin_distances is not None and bin_id < len(bin_distances):
                                bin_label += f" (r = {bin_distances[bin_id]:.1f} arcsec)"
                            
                            axes[i].set_title(bin_label)
                            axes[i].grid(True, alpha=0.3)
                            axes[i].legend(loc='upper right', fontsize='small')
                            
                            # Add y-axis label only for the middle subplot
                            if i == len(combined_bins) // 2:
                                axes[i].set_ylabel('Flux')
                        
                        # Set common x-axis label
                        axes[-1].set_xlabel('Wavelength (Å)')
                        
                        # Adjust layout and save
                        plt.suptitle(f"{galaxy_name} - VNB Spectral Fits", y=0.98)
                        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
                        
                        visualization.standardize_figure_saving(
                            fig, fit_dir / f"{galaxy_name}_VNB_combined_fits.png"
                        )
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error creating combined spectral fits plot: {e}")
                    plt.close('all')
                    
            except Exception as e:
                logger.warning(f"Error creating spectral fit plots: {e}")
                plt.close('all')
        
        # Create a bin overlay plot showing binning on flux map
        try:
            # Get flux map from cube using improved function
            flux_map = visualization.prepare_flux_map(cube)
            
            # Get binning information
            bin_num = vnb_results["binning"]["bin_num"]
            
            # If bin_num is 1D, reshape to match the cube dimensions
            if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                if len(bin_num) == cube._n_x * cube._n_y:
                    bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                else:
                    # Handle case where bin_num length doesn't match cube dimensions
                    logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({cube._n_y}x{cube._n_x})")
                    bin_num_2d = np.zeros((cube._n_y, cube._n_x), dtype=int)
                    valid_len = min(len(bin_num), cube._n_y * cube._n_x)
                    bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
            else:
                bin_num_2d = bin_num
            
            # Get bin centers if available
            bin_centers = None
            if "bin_x" in vnb_results["binning"] and "bin_y" in vnb_results["binning"]:
                bin_centers = (vnb_results["binning"]["bin_x"], vnb_results["binning"]["bin_y"])
            
            # Create binning overlay plot using the improved function
            fig, ax = visualization.plot_bin_boundaries_on_flux(
                bin_num_2d,
                flux_map,
                cube,
                galaxy_name=galaxy_name,
                binning_type="Voronoi",
                bin_centers=bin_centers,
                save_path=plots_dir / f"{galaxy_name}_VNB_binning_overlay.png"
            )
            
            logger.info(f"Created VNB binning overlay plot")
            
        except Exception as e:
            logger.warning(f"Error creating VNB binning overlay plot: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            plt.close("all")  # Close all figures in case of error
                
    except Exception as e:
        logger.error(f"Error in create_vnb_plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close("all")  # Close all figures in case of error