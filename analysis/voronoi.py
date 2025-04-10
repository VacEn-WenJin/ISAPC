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
    Run Voronoi binning analysis on MUSE data cube with improved SNR targeting

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

    # Set up target SNR and other binning parameters
    # Base target SNR from arguments, but will be adjusted if binning fails
    target_snr = args.target_snr if hasattr(args, "target_snr") else 30
    min_snr = args.min_snr if hasattr(args, "min_snr") else 1
    use_cvt = args.cvt if hasattr(args, "cvt") else True
    
    # New parameter for high-SNR mode
    high_snr_mode = args.high_snr_mode if hasattr(args, "high_snr_mode") else False
    
    # Check if we should use physical radius calculation
    use_physical_radius = args.physical_radius if hasattr(args, "physical_radius") else False

    # Extract coordinates, signal, and noise for VNB
    # Use wavelength-integrated signal-to-noise
    x = cube.x
    y = cube.y

    # Following the notebook's approach, use a specific wavelength range for SNR calculation
    # This range (5075-5125 Å) is often used for continuum SNR assessment
    wave_mask = (cube._lambda_gal >= 5075) & (cube._lambda_gal <= 5125)
    if np.sum(wave_mask) > 0:
        # Calculate SNR using this specific range
        signal = np.nanmedian(cube._spectra[wave_mask], axis=0)
        noise = np.nanstd(cube._spectra[wave_mask], axis=0)
        logger.info("Using wavelength range 5075-5125 Å for SNR calculation")
    else:
        # Fallback to full spectrum if this range is not available
        signal = np.nanmedian(cube._spectra, axis=0)
        noise = np.nanmedian(np.sqrt(cube._log_variance), axis=0)
        logger.info(
            "Using full spectrum for SNR calculation (preferred range not available)"
        )

    # Preprocess signal and noise to avoid problems with very low values
    # Handle problematic pixels by setting minimum values
    min_threshold = 1.0

    # Find pixels with problematic values (very low or NaN)
    low_signal_mask = (signal < min_threshold) | ~np.isfinite(signal)
    low_noise_mask = (noise < min_threshold) | ~np.isfinite(noise)

    # Set minimum values for signal and noise consistently
    if np.any(low_signal_mask) or np.any(low_noise_mask):
        logger.warning(
            f"Found {np.sum(low_signal_mask)} pixels with signal < {min_threshold}"
        )
        logger.warning(
            f"Found {np.sum(low_noise_mask)} pixels with noise < {min_threshold}"
        )

        # Apply minimum value to signal
        signal[low_signal_mask] = min_threshold

        # Apply minimum value to noise
        noise[low_noise_mask] = min_threshold

        # For pixels where SNR < 1, set both signal and noise to the same value (=1)
        # This effectively sets SNR = 1 for these pixels
        low_snr_mask = (signal / noise) < 1.0
        if np.any(low_snr_mask):
            logger.warning(
                f"Setting both signal and noise to {min_threshold} for {np.sum(low_snr_mask)} pixels with SNR < 1"
            )
            signal[low_snr_mask] = min_threshold
            noise[low_snr_mask] = min_threshold

    # Additional safety check - ensure no zeros or NaNs
    signal = np.nan_to_num(signal, nan=min_threshold)
    noise = np.nan_to_num(noise, nan=min_threshold)

    # Calculate per-pixel SNR values
    pixel_snr = signal / noise  # Element-wise division for each pixel
    valid_snr = pixel_snr  # All pixels should be valid now due to our preprocessing
    max_pixel_snr = np.nanmax(valid_snr)
    median_snr = np.nanmedian(valid_snr)

    logger.info(
        f"Maximum pixel SNR: {max_pixel_snr:.1f}, Median pixel SNR: {median_snr:.1f}"
    )

    # Determine recommended SNR range with wider bounds as suggested
    min_recommended_snr = min(2, median_snr * .5)
    max_recommended_snr = max(max_pixel_snr * 1.2, median_snr * 10)

    # # Ensure min and max are in a reasonable range
    min_recommended_snr = max(2, min_recommended_snr)  # At least 2
    max_recommended_snr = max(50, max_recommended_snr)  # At least 15

    # If user-specified target_snr is outside the recommended range, adjust
    if target_snr < min_recommended_snr or target_snr > max_recommended_snr:
        logger.warning(
            f"Specified target SNR {target_snr} is outside recommended range "
            f"({min_recommended_snr:.1f} - {max_recommended_snr:.1f})"
        )

        # Adjust target_snr to be within range
        safe_target_snr = max(min_recommended_snr, min(target_snr, max_recommended_snr))
        logger.info(f"Adjusting target SNR to {safe_target_snr:.1f}")
    else:
        safe_target_snr = target_snr

    logger.info(f"Running Voronoi binning with target SNR = {safe_target_snr:.1f}")

    # If requested to use physical radius, calculate it now for later use in binning
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

    # Modified high-SNR mode for better binning
    if high_snr_mode:
        logger.info("Using high-SNR optimization mode for Voronoi binning")
        
        # Sort pixels by SNR in descending order
        sorted_indices = np.argsort(-pixel_snr)  # Negative for descending order
        
        # Take the top X% of pixels with highest SNR as seeds
        top_percent = 10  # Use top 10% as seeds
        n_seed_pixels = max(1, int(len(sorted_indices) * top_percent / 100))
        seed_indices = sorted_indices[:n_seed_pixels]
        
        logger.info(f"Using {n_seed_pixels} high-SNR seed pixels (top {top_percent}%)")
        
        # Extract seed pixels
        seed_x = x[seed_indices]
        seed_y = y[seed_indices]
        seed_signal = signal[seed_indices]
        seed_noise = noise[seed_indices]
        
        # Run a preliminary binning on just the seed pixels
        # This ensures we have good initial bins with high SNR
        try:
            from vorbin.voronoi_2d_binning import voronoi_2d_binning
            seed_bin_num, seed_x_gen, seed_y_gen, seed_sn, seed_n_pixels, seed_scale = voronoi_2d_binning(
                seed_x, seed_y, seed_signal, seed_noise,
                safe_target_snr, plot=0, quiet=True, cvt=use_cvt
            )
            
            logger.info(f"Created {len(seed_x_gen)} high-SNR seed bins")
            
            # Now run the main binning using these seed bins as anchors
            # We would need to implement a custom version of voronoi_2d_binning to use seed bins
            # For now, we'll use a simpler approach of two-stage binning
            
            # Create a mask for remaining pixels
            remaining_mask = np.ones(len(x), dtype=bool)
            remaining_mask[seed_indices] = False
            
            # Initialize a bin map for all pixels, starting with seed bins
            full_bin_num = np.full(len(x), -1, dtype=int)
            for i, seed_idx in enumerate(seed_indices):
                full_bin_num[seed_idx] = seed_bin_num[i]
            
            # Assign each remaining pixel to the nearest seed bin
            from scipy.spatial import cKDTree
            seed_points = np.column_stack([seed_x_gen, seed_y_gen])
            tree = cKDTree(seed_points)
            
            # Get remaining points
            remaining_x = x[remaining_mask]
            remaining_y = y[remaining_mask]
            remaining_indices = np.where(remaining_mask)[0]
            
            # Find nearest seed bin for each remaining point
            query_points = np.column_stack([remaining_x, remaining_y])
            distances, nearest_bins = tree.query(query_points)
            
            # Assign each remaining pixel to nearest bin
            for i, bin_idx in enumerate(nearest_bins):
                full_bin_num[remaining_indices[i]] = bin_idx
                
            # Generate bin metadata for compatibility
            from collections import defaultdict
            bin_indices_dict = defaultdict(list)
            for i, bin_id in enumerate(full_bin_num):
                if bin_id >= 0:
                    bin_indices_dict[bin_id].append(i)
            
            bin_indices_list = [bin_indices_dict.get(i, []) for i in range(max(bin_indices_dict.keys()) + 1)]
            n_pixels = np.array([len(indices) for indices in bin_indices_list])
            
            # Calculate average position for each bin
            x_gen = []
            y_gen = []
            sn = []
            
            for i, indices in enumerate(bin_indices_list):
                if indices:
                    x_gen.append(np.mean(x[indices]))
                    y_gen.append(np.mean(y[indices]))
                    
                    # Calculate SNR for the bin
                    bin_signal = np.mean(signal[indices])
                    bin_noise = np.mean(noise[indices]) / np.sqrt(len(indices))
                    sn.append(bin_signal / bin_noise)
                else:
                    x_gen.append(0)
                    y_gen.append(0)
                    sn.append(0)
            
            x_gen = np.array(x_gen)
            y_gen = np.array(y_gen)
            sn = np.array(sn)
            
            success = True
            best_result = (full_bin_num, x_gen, y_gen, sn, n_pixels, 1.0)
            logger.info(f"High-SNR binning created {len(x_gen)} bins")
            
        except Exception as e:
            logger.warning(f"High-SNR optimization failed: {str(e)}")
            logger.info("Falling back to standard Voronoi binning")
            success = False
            high_snr_mode = False
    else:
        success = False  # Force standard binning if high-SNR mode is not requested

    # Standard Voronoi binning if high-SNR mode is not used or failed
    if not high_snr_mode or not success:
        # First attempt with the selected target SNR
        try:
            success = True

            # Use all valid pixels
            x_valid = x
            y_valid = y
            signal_valid = signal
            noise_valid = noise

            logger.info(f"Running Voronoi binning with {len(x_valid)} valid pixels")

            # If using physical radius, apply elliptical transformation to coordinates
            if use_physical_radius and ellipse_params is not None:
                # Extract parameters
                center_x = ellipse_params['center_x']
                center_y = ellipse_params['center_y']
                PA_rad = np.radians(ellipse_params['PA_degrees'])
                ellipticity = ellipse_params['ellipticity']
                
                # Apply transformation: centered, rotated, and stretched coordinates
                dx = x_valid - center_x
                dy = y_valid - center_y
                
                # Rotate coordinates
                x_rot = dx * np.cos(PA_rad) + dy * np.sin(PA_rad)
                y_rot = -dx * np.sin(PA_rad) + dy * np.cos(PA_rad)
                
                # Apply ellipticity stretch
                y_rot_scaled = y_rot / (1 - ellipticity) if ellipticity < 1 else y_rot
                
                # Use transformed coordinates for binning
                logger.info(f"Using elliptical coordinates for Voronoi binning (PA={ellipse_params['PA_degrees']:.1f}°, ε={ellipticity:.2f})")
                x_for_binning = x_rot
                y_for_binning = y_rot_scaled
            else:
                # Use original coordinates
                x_for_binning = x_valid
                y_for_binning = y_valid

            # Handle return values more robustly to accommodate different version of vorbin
            result = voronoi_2d_binning(
                x_for_binning,
                y_for_binning,
                signal_valid,
                noise_valid,
                safe_target_snr,
                plot=0,
                quiet=True,
                cvt=use_cvt,
            )

            # Robust handling of return values
            if isinstance(result, tuple):
                # Check length of returned tuple
                if len(result) >= 6:
                    # Extract the first 6 values we need
                    bin_num = result[0]
                    x_gen = result[1]
                    y_gen = result[2]
                    sn = result[3]
                    n_pixels = result[4]
                    scale = result[5]
                    best_result = (bin_num, x_gen, y_gen, sn, n_pixels, scale)
                    logger.info(
                        f"Voronoi binning succeeded with target SNR = {safe_target_snr:.1f}"
                    )
                else:
                    # Not enough values
                    raise ValueError(
                        f"Voronoi binning returned {len(result)} values, expected at least 6"
                    )
            else:
                # Single return value (unlikely but handle anyway)
                raise ValueError("Unexpected return format from Voronoi binning")

        except Exception as e:
            success = False
            best_result = None
            logger.warning(f"Initial Voronoi binning failed: {str(e)}")
            logger.info("Trying alternative binning approach")

            # Try a more comprehensive search through recommended SNR values
            search_range = np.linspace(max_recommended_snr, min_recommended_snr, 20)

            for snr_value in search_range:
                # Skip the original value which already failed
                if abs(snr_value - safe_target_snr) < 0.1:
                    continue

                # Try with CVT first
                try:
                    logger.info(
                        f"Trying Voronoi binning with target SNR = {snr_value:.1f}, CVT=True"
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = voronoi_2d_binning(
                            x_for_binning,
                            y_for_binning,
                            signal_valid,
                            noise_valid,
                            snr_value,
                            plot=0,
                            quiet=True,
                            cvt=True,
                        )

                        # Extract the first 6 values
                        if isinstance(result, tuple) and len(result) >= 6:
                            bin_num = result[0]
                            x_gen = result[1]
                            y_gen = result[2]
                            sn = result[3]
                            n_pixels = result[4]
                            scale = result[5]
                            best_result = (bin_num, x_gen, y_gen, sn, n_pixels, scale)
                            success = True
                            logger.info(
                                f"Voronoi binning succeeded with target SNR = {snr_value:.1f}"
                            )
                            break
                        else:
                            raise ValueError(
                                "Unexpected return format from Voronoi binning"
                            )
                except Exception as e:
                    # Now try without CVT
                    try:
                        logger.info(
                            f"Trying Voronoi binning with target SNR = {snr_value:.1f}, CVT=False"
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result = voronoi_2d_binning(
                                x_for_binning,
                                y_for_binning,
                                signal_valid,
                                noise_valid,
                                snr_value,
                                plot=0,
                                quiet=True,
                                cvt=False,
                            )

                            # Extract the first 6 values
                            if isinstance(result, tuple) and len(result) >= 6:
                                bin_num = result[0]
                                x_gen = result[1]
                                y_gen = result[2]
                                sn = result[3]
                                n_pixels = result[4]
                                scale = result[5]
                                best_result = (bin_num, x_gen, y_gen, sn, n_pixels, scale)
                                success = True
                                logger.info(
                                    f"Voronoi binning succeeded with target SNR = {snr_value:.1f}"
                                )
                                break
                            else:
                                raise ValueError(
                                    "Unexpected return format from Voronoi binning"
                                )
                    except Exception as e:
                        continue

    # Fallback to grid binning if all else fails
    if not success or best_result is None:
        logger.warning(
            "All Voronoi binning attempts failed. Using fallback grid binning."
        )

        # Create a simple grid binning as fallback
        from binning import create_grid_binning

        grid_result = create_grid_binning(
            x,
            y,
            signal,
            noise,
            nx=min(5, int(np.sqrt(len(x) / 20))),  # Adjust grid size based on data
            ny=min(5, int(np.sqrt(len(x) / 20))),
        )

        if grid_result is not None:
            best_result = grid_result
            success = True
            logger.info(f"Created grid binning with {len(best_result[1])} bins")
        else:
            logger.error("Failed to create bins. Cannot proceed with analysis.")
            return {"status": "error", "message": "Failed to create bins"}

    # Extract binning result
    bin_num, x_gen, y_gen, sn, n_pixels, scale = best_result

    # Map bin numbers back to the original arrays
    full_bin_num = bin_num

    # Filter out low SNR bins if requested, but don't filter ALL bins
    if min_snr > 0:
        # Find bins with SNR below minimum
        bad_bins = np.where(sn < min_snr)[0]

        # Don't filter all bins
        if len(bad_bins) < len(sn):  # Only if some bins remain
            # Mark these bins as unbinned (-1)
            for bad_bin in bad_bins:
                full_bin_num[full_bin_num == bad_bin] = -1

            logger.info(f"Filtered out {len(bad_bins)} bins with SNR < {min_snr}")
        else:
            logger.warning(
                f"All bins have SNR < {min_snr}, but keeping them to avoid empty result"
            )

    # Create bin indices
    bin_indices = []
    all_indices = np.arange(len(full_bin_num))

    # Maximum bin number
    max_bin = int(np.nanmax(full_bin_num))

    for i in range(max_bin + 1):
        bin_indices.append(all_indices[full_bin_num == i])

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

    # Calculate intersection of wavelength ranges accounting for velocity shifts
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
    binned_data = VoronoiBinnedData(
        bin_num=full_bin_num,
        bin_indices=bin_indices,
        spectra=binned_spectra,
        wavelength=wavelength,
        metadata=metadata,
    )

    # Set up binning in the cube
    cube.setup_binning("VNB", binned_data)

    # Run analysis using the enhanced MUSECube methods
    # These will automatically use the binned data
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
            "bin_num": full_bin_num,
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
                
    except Exception as e:
        logger.error(f"Error in create_vnb_plots: {str(e)}")
        logger.error(traceback.format_exc())
        plt.close("all")