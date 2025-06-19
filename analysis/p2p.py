"""
Pixel-to-pixel analysis module for ISAPC
Version 5.1.0 - Enhanced with full error propagation
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import galaxy_params
import spectral_indices
import visualization
from stellar_population import WeightParser
from utils.io import save_standardized_results
from utils.error_propagation import (
    bootstrap_error_estimation,
    monte_carlo_error_propagation,
    propagate_errors_multiplication,
    propagate_errors_division
)

logger = logging.getLogger(__name__)

# Speed of light in km/s
C_KMS = 299792.458


def calculate_distance_to_center(x, y, pxl_size_x, pxl_size_y=None):
    """
    Calculate physical distance to center

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates in pixels
    y : numpy.ndarray
        y coordinates in pixels
    pxl_size_x : float
        x pixel size in arcsec
    pxl_size_y : float, optional
        y pixel size in arcsec, default same as x

    Returns
    -------
    numpy.ndarray
        Distance to center in arcsec
    """
    if pxl_size_y is None:
        pxl_size_y = pxl_size_x

    # Convert to physical units (arcsec)
    phys_x = x * pxl_size_x
    phys_y = y * pxl_size_y

    # Calculate Euclidean distance
    distance = np.sqrt(phys_x**2 + phys_y**2)

    return distance


def create_radial_profile_plots_wrapper(results, galaxy_name, plots_dir, physical_scale=True, analysis_type="P2P"):
    """
    Wrapper for create_radial_profile_plots with signature matching how it's called in run_p2p_analysis
    """
    # We don't have access to the cube here, so we can't use physical scaling
    logger.info(f"Creating radial profile plots for {galaxy_name} (no physical scaling)")
    return create_radial_profile_plots(
        results=results,
        cube=None,  # We don't have cube here
        galaxy_name=galaxy_name,
        plots_dir=plots_dir,
        physical_scale=False,  # Can't use physical scaling without cube
        analysis_type=analysis_type
    )


def safe_extract(array, mask):
    """
    Safely extract values from an array using a mask
    
    Parameters
    ----------
    array : ndarray
        Array to extract from
    mask : ndarray
        Boolean mask
    
    Returns
    -------
    ndarray
        Extracted values
    """
    try:
        # First convert to numeric if needed
        if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number):
            numeric_array = np.zeros(array.shape, dtype=float)
            for i, val in enumerate(np.ravel(array)):
                try:
                    numeric_array.flat[i] = float(val)
                except (ValueError, TypeError):
                    numeric_array.flat[i] = np.nan
            # Now safely extract using the mask
            return numeric_array[mask]
        else:
            # Standard extraction for numeric arrays
            return array[mask]
    except Exception as e:
        logger.debug(f"Error safely extracting values: {e}")
        return np.array([])


def run_p2p_analysis(args, cube, Pmode=False):
    """
    Run pixel-to-pixel analysis with full error propagation

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    Pmode : bool, default=False
        Whether this is a standalone P2P analysis (not part of VNB/RDB)

    Returns
    -------
    dict
        Analysis results with key physical parameters and errors
    """
    logger.info("Starting pixel-to-pixel analysis with error propagation...")
    start_time = time.time()

    # Disable warnings for spectral indices
    spectral_indices.set_warnings(False)

    # Extract galaxy name from filename
    galaxy_name = Path(args.filename).stem

    # Create standardized output directories
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / "Data"
    plots_dir = galaxy_dir / "Plots" / "P2P"

    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Get error array from cube
    error_array = None
    if hasattr(cube, '_error') and cube._error is not None:
        error_array = cube._error
        logger.info("Using error array from cube")
    elif hasattr(cube, '_variance') and cube._variance is not None:
        error_array = np.sqrt(cube._variance)
        logger.info("Using variance array from cube")
    else:
        # Estimate errors from data
        logger.info("No error array found, estimating from spectral noise")
        from scipy.stats import median_abs_deviation
        error_array = np.zeros_like(cube._spectra)
        for i in range(cube._spectra.shape[1]):
            if np.any(np.isfinite(cube._spectra[:, i])):
                noise = median_abs_deviation(cube._spectra[:, i], nan_policy='omit')
                error_array[:, i] = noise * 1.4826  # Convert MAD to std

    # Fit stellar continuum with error propagation
    result = cube.fit_spectra(
        template_filename=args.template,
        ppxf_vel_init=args.vel_init,
        ppxf_vel_disp_init=args.sigma_init,
        ppxf_deg=args.poly_degree if hasattr(args, "poly_degree") else 3,
        n_jobs=args.n_jobs,
    )

    (
        stellar_velocity_field,
        stellar_dispersion_field,
        bestfit_field,
        optimal_tmpls,
        poly_coeffs,
    ) = result

    # Extract errors from ppxf results if available
    stellar_velocity_error = None
    stellar_dispersion_error = None
    
    if hasattr(cube, '_ppxf_results') and cube._ppxf_results:
        logger.info("Extracting stellar kinematic errors from ppxf results")
        
        n_y, n_x = stellar_velocity_field.shape
        stellar_velocity_error = np.full((n_y, n_x), np.nan)
        stellar_dispersion_error = np.full((n_y, n_x), np.nan)
        
        for row, col, pp_result in cube._ppxf_results:
            if 'error' in pp_result and pp_result['error'] is not None:
                # ppxf errors are typically [vel_err, sigma_err, ...]
                if len(pp_result['error']) >= 2:
                    stellar_velocity_error[row, col] = pp_result['error'][0]
                    stellar_dispersion_error[row, col] = pp_result['error'][1]
                    
    logger.info(
        f"Stellar component fitting completed in {time.time() - start_time:.1f} seconds"
    )

    # Get configured emission lines
    emission_lines = None
    if hasattr(args, "configured_emission_lines"):
        emission_lines = args.configured_emission_lines

    # Fit emission lines with error propagation
    emission_result = None
    if not args.no_emission:
        start_time = time.time()
        emission_result = cube.fit_emission_lines(
            template_filename=args.template,
            line_names=emission_lines,
            ppxf_vel_init=stellar_velocity_field,
            ppxf_sig_init=args.sigma_init,
            ppxf_deg=2,
            n_jobs=args.n_jobs,
        )
        logger.info(
            f"Emission line fitting completed in {time.time() - start_time:.1f} seconds"
        )

    # Extract emission line errors if available
    gas_velocity_error = None
    gas_dispersion_error = None
    
    if hasattr(cube, '_ppxf_gas_results') and cube._ppxf_gas_results:
        logger.info("Extracting gas kinematic errors from ppxf results")
        
        n_y, n_x = stellar_velocity_field.shape
        gas_velocity_error = np.full((n_y, n_x), np.nan)
        gas_dispersion_error = np.full((n_y, n_x), np.nan)
        
        for row, col, pp_result in cube._ppxf_gas_results:
            if 'error' in pp_result and pp_result['error'] is not None:
                if len(pp_result['error']) >= 2:
                    gas_velocity_error[row, col] = pp_result['error'][0]
                    gas_dispersion_error[row, col] = pp_result['error'][1]

    # Get configured spectral indices
    indices_list = None
    if hasattr(args, "configured_indices"):
        indices_list = args.configured_indices

    # Calculate spectral indices with error propagation
    indices_result = None
    indices_errors = None
    if not args.no_indices:
        start_time = time.time()
        
        # Check if cube has calculate_spectral_indices_with_errors method
        if hasattr(cube, 'calculate_spectral_indices_with_errors'):
            indices_result, indices_errors = cube.calculate_spectral_indices_with_errors(
                indices_list=indices_list,
                n_jobs=args.n_jobs,
                error_array=error_array
            )
        else:
            # Fallback to standard calculation
            indices_result = cube.calculate_spectral_indices(
                indices_list=indices_list,
                n_jobs=args.n_jobs,
            )
            
        logger.info(
            f"Spectral indices calculation completed in {time.time() - start_time:.1f} seconds"
        )

    # Prepare emission line data (if available)
    gas_velocity_field = None
    gas_dispersion_field = None

    if emission_result is not None:
        try:
            # 1. Try to directly get from emission_result dictionary
            if (
                "velocity_field" in emission_result
                and emission_result["velocity_field"] is not None
            ):
                gas_velocity_field = emission_result["velocity_field"]
                logger.info("Extracted gas velocity field from emission_result")

            if (
                "dispersion_field" in emission_result
                and emission_result["dispersion_field"] is not None
            ):
                gas_dispersion_field = emission_result["dispersion_field"]
                logger.info("Extracted gas dispersion field from emission_result")

            # 2. Try to extract from emission_vel and emission_sig in emission_result
            if (
                gas_velocity_field is None
                and "emission_vel" in emission_result
                and emission_result["emission_vel"]
            ):
                # Get first available emission line
                for line_name, vel_map in emission_result["emission_vel"].items():
                    if not np.all(np.isnan(vel_map)):
                        gas_velocity_field = vel_map
                        logger.info(
                            f"Using velocity field from emission line: {line_name}"
                        )
                        break

                # Get corresponding dispersion field
                if (
                    gas_velocity_field is not None
                    and "emission_sig" in emission_result
                    and emission_result["emission_sig"]
                ):
                    for line_name, disp_map in emission_result["emission_sig"].items():
                        if not np.all(np.isnan(disp_map)):
                            gas_dispersion_field = disp_map
                            logger.info(
                                f"Using dispersion field from emission line: {line_name}"
                            )
                            break

            # 3. Try to get from cube object
            if (
                gas_velocity_field is None
                and hasattr(cube, "_emission_vel")
                and cube._emission_vel
            ):
                # Get first available emission line
                for line_name, vel_map in cube._emission_vel.items():
                    if not np.all(np.isnan(vel_map)):
                        gas_velocity_field = vel_map
                        logger.info(
                            f"Using velocity field from cube's emission line: {line_name}"
                        )
                        break

                # Get corresponding dispersion field
                if (
                    gas_velocity_field is not None
                    and hasattr(cube, "_emission_sig")
                    and cube._emission_sig
                ):
                    for line_name, disp_map in cube._emission_sig.items():
                        if not np.all(np.isnan(disp_map)):
                            gas_dispersion_field = disp_map
                            logger.info(
                                f"Using dispersion field from cube's emission line: {line_name}"
                            )
                            break

            # 4. Try to extract from cube._ppxf_gas_results if available
            if (
                gas_velocity_field is None
                and hasattr(cube, "_ppxf_gas_results")
                and cube._ppxf_gas_results
            ):
                # Initialize arrays as NaN
                gas_velocity_field = np.full((cube._n_y, cube._n_x), np.nan)
                gas_dispersion_field = np.full((cube._n_y, cube._n_x), np.nan)

                # Fill with values from each pixel's fit
                for row, col, result in cube._ppxf_gas_results:
                    if "gas_sol" in result and result["gas_sol"] is not None:
                        gas_sol = result["gas_sol"]
                        try:
                            # Try to access gas_sol as array
                            if (
                                isinstance(gas_sol, (list, np.ndarray))
                                and len(gas_sol) >= 2
                            ):
                                gas_velocity_field[row, col] = gas_sol[0]
                                gas_dispersion_field[row, col] = gas_sol[1]
                            elif gas_sol is not None:
                                # If gas_sol is scalar, assume it's the velocity
                                gas_velocity_field[row, col] = float(gas_sol)
                                # Use default value for dispersion
                                gas_dispersion_field[row, col] = args.sigma_init
                        except (TypeError, IndexError) as e:
                            logger.debug(
                                f"Error extracting gas kinematics at ({row},{col}): {e}"
                            )

                logger.info(
                    "Extracted gas velocity and dispersion fields from ppxf_gas_results"
                )

            # Validate extracted gas kinematic fields
            if gas_velocity_field is not None:
                valid_pixels = np.count_nonzero(~np.isnan(gas_velocity_field))
                if (
                    valid_pixels < 10
                ):  # Assume fewer than 10 valid pixels is not useful for analysis
                    logger.warning(
                        f"Too few valid pixels in gas velocity field ({valid_pixels}), using stellar field instead"
                    )
                    gas_velocity_field = None
                    gas_dispersion_field = None
                else:
                    logger.info(
                        f"Found {valid_pixels} valid pixels in gas velocity field"
                    )

        except Exception as e:
            logger.error(f"Failed to extract gas velocity and dispersion fields: {e}")
            gas_velocity_field = None
            gas_dispersion_field = None

            # Decide which velocity and dispersion field to use with their errors
    velocity_error_field = None
    dispersion_error_field = None
    
    if gas_velocity_field is not None and gas_dispersion_field is not None:
        # Check gas velocity field quality/coverage
        valid_gas_pixels = np.sum(~np.isnan(gas_velocity_field))
        valid_stellar_pixels = np.sum(~np.isnan(stellar_velocity_field))
        total_pixels = gas_velocity_field.size

        gas_coverage = valid_gas_pixels / total_pixels
        stellar_coverage = valid_stellar_pixels / total_pixels

        # If gas coverage is reasonable
        if gas_coverage > 0.3 or gas_coverage > 0.8 * stellar_coverage:
            logger.info(
                f"Using emission line velocity field for kinematics (coverage: {gas_coverage:.2f})"
            )
            velocity_field = gas_velocity_field
            dispersion_field = gas_dispersion_field
            velocity_error_field = gas_velocity_error
            dispersion_error_field = gas_dispersion_error
            using_emission = True
        else:
            logger.info(
                f"Insufficient emission line coverage ({gas_coverage:.2f}), using stellar velocity field"
            )
            velocity_field = stellar_velocity_field
            dispersion_field = stellar_dispersion_field
            velocity_error_field = stellar_velocity_error
            dispersion_error_field = stellar_dispersion_error
            using_emission = False
    else:
        logger.info("No emission line data available, using stellar velocity field")
        velocity_field = stellar_velocity_field
        dispersion_field = stellar_dispersion_field
        velocity_error_field = stellar_velocity_error
        dispersion_error_field = stellar_dispersion_error
        using_emission = False

    # Calculate galaxy parameters with error propagation
    start_time = time.time()
    
    # Enhanced GalaxyParameters with error support
    gp = galaxy_params.GalaxyParameters(
        velocity_field=velocity_field,
        dispersion_field=dispersion_field,
        velocity_error=velocity_error_field,
        dispersion_error=dispersion_error_field,
        pixelsize=cube._pxl_size_x,
    )

    rotation_result = gp.fit_rotation_curve()
    kinematics_result = gp.calculate_kinematics()

    logger.info(
        f"Galaxy parameters calculation completed in {time.time() - start_time:.1f} seconds"
    )

    # Calculate distance to center for each pixel
    n_y, n_x = stellar_velocity_field.shape
    y_indices, x_indices = np.indices((n_y, n_x))

    # Use image center as default center
    center_y, center_x = n_y // 2, n_x // 2

    # Calculate coordinates relative to center
    rel_x = x_indices - center_x
    rel_y = y_indices - center_y

    # Calculate distance (arcsec)
    distance_field = calculate_distance_to_center(
        rel_x, rel_y, cube._pxl_size_x, cube._pxl_size_y
    )

    # Extract stellar population parameters with errors using enhanced WeightParser
    stellar_pop_params = None
    stellar_pop_errors = None
    
    if hasattr(cube, "_template_weights") and cube._template_weights is not None:
        try:
            logger.info("Extracting stellar population parameters with errors...")
            start_time = time.time()

            # Initialize weight parser
            weight_parser = WeightParser(args.template)

            # Get weight covariance if available
            weight_covariance = None
            if hasattr(cube, '_weight_covariance'):
                weight_covariance = cube._weight_covariance

            # Prepare arrays for physical parameters
            n_y, n_x = stellar_velocity_field.shape
            stellar_pop_params = {
                "log_age": np.full((n_y, n_x), np.nan),
                "age": np.full((n_y, n_x), np.nan),
                "metallicity": np.full((n_y, n_x), np.nan),
            }
            
            stellar_pop_errors = {
                "log_age_error": np.full((n_y, n_x), np.nan),
                "age_error": np.full((n_y, n_x), np.nan),
                "metallicity_error": np.full((n_y, n_x), np.nan),
            }

            # Process weights based on shape
            weights = cube._template_weights

            if len(weights.shape) == 3:  # [n_templates, n_y, n_x]
                # Calculate for each valid pixel
                valid_mask = ~np.isnan(stellar_velocity_field)
                valid_indices = np.where(valid_mask)

                for i in range(len(valid_indices[0])):
                    y, x = valid_indices[0][i], valid_indices[1][i]

                    try:
                        pixel_weights = weights[:, y, x]
                        
                        # Get pixel-specific weight errors if available
                        weight_errors = None
                        if hasattr(cube, '_template_weight_errors'):
                            weight_errors = cube._template_weight_errors[:, y, x]
                        
                        # Get pixel-specific covariance if available
                        pixel_covariance = None
                        if weight_covariance is not None and len(weight_covariance.shape) == 4:
                            pixel_covariance = weight_covariance[:, :, y, x]
                        
                        # Get parameters with errors
                        params = weight_parser.get_physical_params(
                            pixel_weights,
                            weight_errors=weight_errors,
                            weight_covariance=pixel_covariance,
                            n_monte_carlo=100 if weight_errors is not None else 0
                        )
                        
                        # Store parameters
                        stellar_pop_params["log_age"][y, x] = params.get("log_age", np.nan)
                        stellar_pop_params["age"][y, x] = params.get("age", np.nan)
                        stellar_pop_params["metallicity"][y, x] = params.get("metallicity", np.nan)
                        
                        # Store errors if available
                        stellar_pop_errors["log_age_error"][y, x] = params.get("log_age_error", np.nan)
                        stellar_pop_errors["age_error"][y, x] = params.get("age_error", np.nan)
                        stellar_pop_errors["metallicity_error"][y, x] = params.get("metallicity_error", np.nan)
                        
                    except Exception as e:
                        logger.debug(
                            f"Error calculating stellar params for pixel ({x}, {y}): {e}"
                        )

            logger.info(
                f"Stellar population parameters extracted in {time.time() - start_time:.1f} seconds"
            )
        except Exception as e:
            logger.error(f"Failed to extract stellar population parameters: {e}")
    elif (
        emission_result is not None
        and "weights" in emission_result
        and emission_result["weights"] is not None
    ):
        try:
            logger.info(
                "Extracting stellar population parameters from emission_result..."
            )
            start_time = time.time()

            # Initialize weight parser
            weight_parser = WeightParser(args.template)

            # Prepare arrays for physical parameters
            n_y, n_x = stellar_velocity_field.shape
            stellar_pop_params = {
                "log_age": np.full((n_y, n_x), np.nan),
                "age": np.full((n_y, n_x), np.nan),
                "metallicity": np.full((n_y, n_x), np.nan),
            }

            # Use weights from emission_result
            weights = emission_result["weights"]

            if len(weights.shape) == 2:  # [n_spectra, n_templates]
                # Calculate for each valid pixel
                valid_mask = ~np.isnan(stellar_velocity_field)
                valid_indices = np.where(valid_mask)

                for i in range(len(valid_indices[0])):
                    y, x = valid_indices[0][i], valid_indices[1][i]
                    idx = y * n_x + x

                    try:
                        if idx < len(weights):
                            pixel_weights = weights[idx]
                            if np.sum(pixel_weights) > 0:
                                params = weight_parser.get_physical_params(
                                    pixel_weights
                                )
                                for param_name, value in params.items():
                                    stellar_pop_params[param_name][y, x] = value
                    except Exception as e:
                        logger.debug(
                            f"Error calculating stellar params for pixel ({x}, {y}): {e}"
                        )

            logger.info(
                f"Stellar population parameters extracted in {time.time() - start_time:.1f} seconds"
            )
        except Exception as e:
            logger.error(
                f"Failed to extract stellar population parameters from emission_result: {e}"
            )
    else:
        logger.warning("No weights found for stellar population analysis")

    # Create standardized results dictionary with errors
    p2p_results = {
        "analysis_type": "P2P",
        "stellar_kinematics": {
            "velocity_field": stellar_velocity_field,
            "dispersion_field": stellar_dispersion_field,
            "velocity_error": stellar_velocity_error,
            "dispersion_error": stellar_dispersion_error,
        },
        "global_kinematics": {
            **rotation_result,
            **kinematics_result,
            "based_on_emission": using_emission,
        },
        "distance": {
            "field": distance_field,
            "pixelsize_x": cube._pxl_size_x,
            "pixelsize_y": cube._pxl_size_y,
        },
    }

    # Add emission results if available
    if emission_result is not None:
        p2p_results["emission"] = emission_result
        if gas_velocity_field is not None:
            p2p_results["emission"]["velocity_field"] = gas_velocity_field
            p2p_results["emission"]["dispersion_field"] = gas_dispersion_field
            p2p_results["emission"]["velocity_error"] = gas_velocity_error
            p2p_results["emission"]["dispersion_error"] = gas_dispersion_error

    # Add stellar population parameters with errors if available
    if stellar_pop_params is not None:
        p2p_results["stellar_population"] = stellar_pop_params
        if stellar_pop_errors is not None:
            p2p_results["stellar_population_errors"] = stellar_pop_errors

    # Add spectral indices with errors
    if indices_result is not None:
        p2p_results["indices"] = indices_result
        if indices_errors is not None:
            p2p_results["indices_errors"] = indices_errors

    # Save results - Only save if this is a genuine P2P analysis (not binned)
    should_save = Pmode or (
        not hasattr(args, "_is_binned_analysis") and not hasattr(args, "no_save")
    )

    if should_save:
        save_standardized_results(galaxy_name, "P2P", p2p_results, output_dir)

    # Create visualizations with error support
    if should_save and not args.no_plots:
        create_p2p_plots_with_errors(
            args,
            cube,
            p2p_results,
            galaxy_name,
            bestfit_field,
            optimal_tmpls,
            emission_result,
            using_emission,
        )
        # Create radial profile plots with errors
        create_radial_profile_plots_with_errors(
            p2p_results,
            cube,
            galaxy_name,
            plots_dir,
            physical_scale=True,
            analysis_type="P2P"
        )

    logger.info("Pixel-to-pixel analysis with error propagation completed")
    return p2p_results


def create_p2p_plots_with_errors(
    args,
    cube,
    p2p_results,
    galaxy_name,
    bestfit_field,
    optimal_tmpls,
    emission_result,
    using_emission,
):
    """
    Create plots for pixel-to-pixel analysis with error visualization

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict
        Analysis results with errors
    galaxy_name : str
        Galaxy name for file naming
    bestfit_field : ndarray
        Best-fit spectra
    optimal_tmpls : ndarray
        Optimal templates
    emission_result : dict
        Full emission line results
    using_emission : bool
        Whether emission lines were used for kinematics
    """
    # Set up plots directory
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    plots_dir = galaxy_dir / "Plots" / "P2P"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # First create standard plots
    create_p2p_plots(
        args, cube, p2p_results, galaxy_name, 
        bestfit_field, optimal_tmpls, emission_result, using_emission
    )

    # Now create additional error plots
    
    # Extract results and errors
    rotation_result = p2p_results["global_kinematics"]

    # Determine which velocity field to use
    if using_emission and "emission" in p2p_results:
        velocity_field = p2p_results["emission"].get("velocity_field")
        dispersion_field = p2p_results["emission"].get("dispersion_field")
        velocity_error = p2p_results["emission"].get("velocity_error")
        dispersion_error = p2p_results["emission"].get("dispersion_error")
        kinematics_type = "gas"
    else:
        velocity_field = p2p_results["stellar_kinematics"]["velocity_field"]
        dispersion_field = p2p_results["stellar_kinematics"]["dispersion_field"]
        velocity_error = p2p_results["stellar_kinematics"].get("velocity_error")
        dispersion_error = p2p_results["stellar_kinematics"].get("dispersion_error")
        kinematics_type = "stellar"

    # Create kinematic maps with errors if available
    if velocity_error is not None and dispersion_error is not None:
        try:
            fig, axes = visualization.plot_kinematic_maps_with_errors(
                velocity_field=velocity_field,
                dispersion_field=dispersion_field,
                velocity_error=velocity_error,
                dispersion_error=dispersion_error,
                equal_aspect=args.equal_aspect,
                physical_scale=True,
                pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                wcs=cube._wcs if hasattr(cube, "_wcs") else None,
                cube=cube,
                title=f"{galaxy_name} - {kinematics_type.capitalize()} Kinematics with Errors"
            )
            
            visualization.standardize_figure_saving(
                fig, plots_dir / f"{galaxy_name}_P2P_{kinematics_type}_kinematics_errors.png"
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating kinematic error plots: {e}")
            plt.close("all")

    # Create stellar population parameter plots with errors
    if "stellar_population_errors" in p2p_results:
        create_stellar_pop_plots_with_errors(
            p2p_results["stellar_population"],
            p2p_results["stellar_population_errors"],
            plots_dir,
            galaxy_name
        )

    # Create spectral index plots with errors
    if "indices_errors" in p2p_results:
        create_indices_plots_with_errors(
            cube,
            p2p_results["indices"],
            p2p_results["indices_errors"],
            plots_dir,
            galaxy_name
        )

    # Create error summary plot
    try:
        fig = visualization.create_error_summary_plot(
            p2p_results,
            figsize=(16, 12),
            title=f"{galaxy_name} - P2P Error Analysis",
            save_path=plots_dir / f"{galaxy_name}_P2P_error_summary.png"
        )
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating error summary plot: {e}")
        plt.close("all")

def create_p2p_plots(
    args,
    cube,
    p2p_results,
    galaxy_name,
    bestfit_field,
    optimal_tmpls,
    emission_result,
    using_emission,
):
    """
    Create plots for pixel-to-pixel analysis

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict
        Analysis results
    galaxy_name : str
        Galaxy name for file naming
    bestfit_field : ndarray
        Best-fit spectra (used for plotting only)
    optimal_tmpls : ndarray
        Optimal templates (used for plotting only)
    emission_result : dict
        Full emission line results (used for plotting only)
    using_emission : bool
        Whether emission lines were used for kinematics
    """
    # Set up plots directory
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    plots_dir = galaxy_dir / "Plots" / "P2P"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Extract results
    rotation_result = p2p_results["global_kinematics"]

    # Determine which velocity field to use for primary display based on global kinematics
    if (
        using_emission
        and "emission" in p2p_results
        and "velocity_field" in p2p_results["emission"]
    ):
        velocity_field = p2p_results["emission"]["velocity_field"]
        dispersion_field = (
            p2p_results["emission"]["dispersion_field"]
            if "dispersion_field" in p2p_results["emission"]
            else None
        )
        kinematics_type = "gas"  # For filename label
    else:
        velocity_field = p2p_results["stellar_kinematics"]["velocity_field"]
        dispersion_field = p2p_results["stellar_kinematics"]["dispersion_field"]
        kinematics_type = "stellar"  # For filename label
        using_emission = False  # Ensure flag is correct

    # If dispersion field not found, create NaN array with velocity field shape
    if dispersion_field is None:
        dispersion_field = np.full_like(velocity_field, np.nan)
        logger.warning("Dispersion field not found, using NaN array")

    # Create kinematics plot with physical scaling and WCS
    # In create_p2p_plots:
    try:
        fig = visualization.plot_kinematics_summary(
            velocity_field=velocity_field,
            dispersion_field=dispersion_field,
            rotation_curve=rotation_result["rotation_curve"],
            params=rotation_result,
            equal_aspect=args.equal_aspect,
            physical_scale=True,
            pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
            wcs=cube._wcs if hasattr(cube, "_wcs") else None
        )

        # Use filename to distinguish gas/stellar kinematics
        fig.savefig(
            plots_dir / f"{galaxy_name}_P2P_{kinematics_type}_kinematics.png", dpi=150
        )
        plt.close(fig)  # Ensure figure is closed
    except Exception as e:
        logger.error(f"Error creating kinematics plot: {str(e)}")
        plt.close("all")  # Close all figures in case of error

    # Create stellar population parameter plots
    if "stellar_population" in p2p_results:
        create_stellar_pop_plots(
            p2p_results["stellar_population"], plots_dir, galaxy_name
        )
        plt.close(fig)

    # If using stellar kinematics but emission line data is available, create gas kinematics plot
    if (
        not using_emission
        and "emission" in p2p_results
        and "velocity_field" in p2p_results["emission"]
    ):
        gas_vel = p2p_results["emission"]["velocity_field"]
        gas_disp = (
            p2p_results["emission"]["dispersion_field"]
            if "dispersion_field" in p2p_results["emission"]
            else np.full_like(gas_vel, np.nan)
        )

        # Create gas kinematics plot (without rotation curve fit)
        try:
            # Check if plot_gas_kinematics function exists
            if hasattr(visualization, "plot_gas_kinematics"):
                fig = visualization.plot_gas_kinematics(
                    velocity_field=gas_vel,
                    dispersion_field=gas_disp,
                    equal_aspect=args.equal_aspect,
                    physical_scale=True,
                    pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                    wcs=cube._wcs if hasattr(cube, "_wcs") else None,
                    rot_angle=cube._ifu_rot_angle if hasattr(cube, "_ifu_rot_angle") else 0.0,
                )
            else:
                # If function doesn't exist, create basic plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Calculate valid data range
                valid_vel = gas_vel[~np.isnan(gas_vel)]
                valid_disp = gas_disp[~np.isnan(gas_disp)]

                if len(valid_vel) > 0:
                    vmin_vel = np.percentile(valid_vel, 5)
                    vmax_vel = np.percentile(valid_vel, 95)
                    
                    # Use physical scaling
                    if hasattr(cube, "_wcs") and cube._wcs is not None:
                        # Plot with WCS coordinates
                        from astropy.visualization.wcsaxes import WCSAxes
                        
                        # Create new axis with WCS projection
                        fig = plt.figure(figsize=(12, 5))
                        ax0 = fig.add_subplot(121, projection=cube._wcs)
                        ax1 = fig.add_subplot(122, projection=cube._wcs)
                        
                        im0 = ax0.imshow(
                            gas_vel,
                            origin="lower",
                            cmap="RdBu_r",
                            vmin=vmin_vel,
                            vmax=vmax_vel,
                        )
                        plt.colorbar(im0, ax=ax0, label="Velocity [km/s]")
                        ax0.set_title("Gas Velocity Field")
                        ax0.grid(color='white', ls='solid', alpha=0.3)
                        ax0.set_xlabel('RA')
                        ax0.set_ylabel('Dec')
                        
                        # Dispersion
                        if len(valid_disp) > 0:
                            vmin_disp = np.percentile(valid_disp, 5)
                            vmax_disp = np.percentile(valid_disp, 95)
                            im1 = ax1.imshow(
                                gas_disp,
                                origin="lower",
                                cmap="viridis",
                                vmin=vmin_disp,
                                vmax=vmax_disp,
                            )
                            plt.colorbar(im1, ax=ax1, label="Dispersion [km/s]")
                            ax1.set_title("Gas Velocity Dispersion")
                            ax1.grid(color='white', ls='solid', alpha=0.3)
                            ax1.set_xlabel('RA')
                            ax1.set_ylabel('Dec')
                    else:
                        # Physical scaling with pixel size
                        ny, nx = gas_vel.shape
                        pixel_size_x = cube._pxl_size_x
                        pixel_size_y = cube._pxl_size_y
                        rot_angle = cube._ifu_rot_angle if hasattr(cube, "_ifu_rot_angle") else 0.0
                        
                        # Create physical coordinate grid
                        x_min = -nx/2 * pixel_size_x
                        x_max = nx/2 * pixel_size_x
                        y_min = -ny/2 * pixel_size_y
                        y_max = ny/2 * pixel_size_y
                        
                        # Plot with physical coordinates
                        extent = [x_min, x_max, y_min, y_max]
                        
                        im0 = axes[0].imshow(
                            gas_vel,
                            origin="lower",
                            cmap="RdBu_r",
                            vmin=vmin_vel,
                            vmax=vmax_vel,
                            extent=extent,
                            aspect="auto" if not args.equal_aspect else 1,
                        )
                        plt.colorbar(im0, ax=axes[0], label="Velocity [km/s]")
                        axes[0].set_title("Gas Velocity Field")
                        
                        # Apply rotation to axes labels if needed
                        if rot_angle != 0:
                            # Create rotated axis labels
                            x_label = f'Δ RA cos({rot_angle:.0f}°) + Δ DEC sin({rot_angle:.0f}°) (arcsec)'
                            y_label = f'-Δ RA sin({rot_angle:.0f}°) + Δ DEC cos({rot_angle:.0f}°) (arcsec)'
                        else:
                            x_label = 'Δ RA (arcsec)'
                            y_label = 'Δ DEC (arcsec)'
                            
                        axes[0].set_xlabel(x_label)
                        axes[0].set_ylabel(y_label)

                if len(valid_disp) > 0:
                    vmin_disp = np.percentile(valid_disp, 5)
                    vmax_disp = np.percentile(valid_disp, 95)
                    
                    if hasattr(cube, "_wcs") and cube._wcs is not None:
                        # Already handled in WCS case above
                        pass
                    else:
                        # Physical scaling with pixel size
                        ny, nx = gas_disp.shape
                        pixel_size_x = cube._pxl_size_x
                        pixel_size_y = cube._pxl_size_y
                        rot_angle = cube._ifu_rot_angle if hasattr(cube, "_ifu_rot_angle") else 0.0
                        
                        # Create physical coordinate grid
                        x_min = -nx/2 * pixel_size_x
                        x_max = nx/2 * pixel_size_x
                        y_min = -ny/2 * pixel_size_y
                        y_max = ny/2 * pixel_size_y
                        
                        # Plot with physical coordinates
                        extent = [x_min, x_max, y_min, y_max]
                        
                        im1 = axes[1].imshow(
                            gas_disp,
                            origin="lower",
                            cmap="viridis",
                            vmin=vmin_disp,
                            vmax=vmax_disp,
                            extent=extent,
                            aspect="auto" if not args.equal_aspect else 1,
                        )
                        plt.colorbar(im1, ax=axes[1], label="Dispersion [km/s]")
                        axes[1].set_title("Gas Velocity Dispersion")
                        
                        # Apply rotation to axes labels if needed
                        if rot_angle != 0:
                            # Create rotated axis labels
                            x_label = f'Δ RA cos({rot_angle:.0f}°) + Δ DEC sin({rot_angle:.0f}°) (arcsec)'
                            y_label = f'-Δ RA sin({rot_angle:.0f}°) + Δ DEC cos({rot_angle:.0f}°) (arcsec)'
                        else:
                            x_label = 'Δ RA (arcsec)'
                            y_label = 'Δ DEC (arcsec)'
                            
                        axes[1].set_xlabel(x_label)
                        axes[1].set_ylabel(y_label)

                fig.suptitle("Gas Kinematics")
                plt.tight_layout()

            fig.savefig(plots_dir / f"{galaxy_name}_P2P_gas_kinematics.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating gas kinematics plot: {str(e)}")
            plt.close("all")

    # Create sample pixel spectrum fits
    create_sample_fits(
        cube, velocity_field, bestfit_field, emission_result, plots_dir, galaxy_name
    )

    # Create emission line maps if available
    if "emission" in p2p_results:
        create_emission_maps(p2p_results["emission"], plots_dir, galaxy_name)

    # Create spectral index plots
    if "indices" in p2p_results:
        create_indices_plots(cube, p2p_results["indices"], plots_dir, galaxy_name)

        # Use LineIndexCalculator to create detailed index plots for central pixel
        n_y, n_x = velocity_field.shape
        central_y, central_x = n_y // 2, n_x // 2

        # Check if central pixel has valid data
        if np.isnan(velocity_field[central_y, central_x]) or np.isnan(
            dispersion_field[central_y, central_x]
        ):
            # Find a valid pixel
            valid_mask = ~np.isnan(velocity_field) & ~np.isnan(dispersion_field)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)
                # Use the first valid pixel
                central_y, central_x = valid_indices[0][0], valid_indices[1][0]
                logger.info(
                    f"Central pixel invalid, using alternative pixel at ({central_x}, {central_y})"
                )
            else:
                logger.warning(
                    "No valid pixels found for spectral index plotting. Skipping."
                )
                return

        # Get data for central pixel
        central_idx = central_y * n_x + central_x

        try:
            # Get spectral data
            observed_spectrum = cube._spectra[:, central_idx]
            model_spectrum = bestfit_field[:, central_y, central_x]

            # Get gas model if available
            gas_model = None
            gas_model = cube._gas_bestfit_field[:, central_y, central_x]

            # Create LIC with error handling
            calculator = spectral_indices.LineIndexCalculator(
                wave=cube._lambda_gal,
                flux=observed_spectrum,
                fit_wave=cube._sps.lam_temp,
                fit_flux=optimal_tmpls[:, central_y, central_x],
                em_wave=cube._lambda_gal if gas_model is not None else None,
                em_flux_list=gas_model,
                velocity_correction=velocity_field[central_y, central_x],
                continuum_mode="auto",
                show_warnings=False,
            )

            # Plot spectral lines with indices
            fig, axes = calculator.plot_all_lines(
                mode="P2P", number=0, save_path=str(plots_dir), show_index=True
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating spectral line plots: {str(e)}")
            plt.close("all")


def create_stellar_pop_plots(stellar_pop_params, plots_dir, galaxy_name):
    """
    Create plots for stellar population parameters

    Parameters
    ----------
    stellar_pop_params : dict
        Dictionary containing stellar population parameters
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    """
    # Create plots for each parameter
    param_info = {
        "log_age": {
            "title": "Log Age [yr]",
            "cmap": "plasma",
            "vmin_percentile": 5,
            "vmax_percentile": 95,
        },
        "age": {
            "title": "Age [Gyr]",
            "cmap": "plasma",
            "vmin_percentile": 5,
            "vmax_percentile": 95,
            "scale_factor": 1e-9,  # Convert to Gyr
        },
        "metallicity": {
            "title": "Metallicity [Z/H]",
            "cmap": "viridis",
            "vmin_percentile": 5,
            "vmax_percentile": 95,
        },
    }

    for param_name, info in param_info.items():
        if param_name in stellar_pop_params:
            try:
                param_map = stellar_pop_params[param_name]

                # Apply scale factor if defined
                if "scale_factor" in info:
                    param_map = param_map * info["scale_factor"]

                # Check data validity
                valid_values = param_map[~np.isnan(param_map)]
                if len(valid_values) > 0:
                    fig, ax = plt.subplots(figsize=(8, 7))

                    # Calculate display range
                    vmin = np.percentile(valid_values, info["vmin_percentile"])
                    vmax = np.percentile(valid_values, info["vmax_percentile"])

                    im = ax.imshow(
                        param_map,
                        origin="lower",
                        cmap=info["cmap"],
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                    )
                    plt.colorbar(im, ax=ax, label=info["title"])
                    ax.set_title(f"Stellar {info['title']}")

                    fig.savefig(
                        plots_dir / f"{galaxy_name}_P2P_stellar_{param_name}.png",
                        dpi=150,
                    )
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating {param_name} map: {str(e)}")
                plt.close("all")


def create_stellar_pop_plots_with_errors(stellar_pop_params, stellar_pop_errors, 
                                        plots_dir, galaxy_name):
    """
    Create plots for stellar population parameters with error visualization
    """
    # Create directory for stellar population plots
    stellar_dir = plots_dir / "stellar_population"
    stellar_dir.mkdir(exist_ok=True, parents=True)

    # Parameter information
    param_info = {
        "age": {
            "title": "Age [Gyr]",
            "error_key": "age_error",
            "scale_factor": 1e-9,
            "cmap": "plasma"
        },
        "metallicity": {
            "title": "Metallicity [Z/H]",
            "error_key": "metallicity_error",
            "scale_factor": 1.0,
            "cmap": "viridis"
        }
    }

    for param_name, info in param_info.items():
        if param_name in stellar_pop_params and info["error_key"] in stellar_pop_errors:
            try:
                param_map = stellar_pop_params[param_name] * info["scale_factor"]
                error_map = stellar_pop_errors[info["error_key"]] * info["scale_factor"]
                
                # Create figure with value, error, and S/N
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Value map
                valid_values = param_map[~np.isnan(param_map)]
                if len(valid_values) > 0:
                    vmin = np.percentile(valid_values, 5)
                    vmax = np.percentile(valid_values, 95)
                    
                    im1 = axes[0].imshow(
                        param_map, origin="lower", cmap=info["cmap"],
                        vmin=vmin, vmax=vmax, aspect="auto"
                    )
                    plt.colorbar(im1, ax=axes[0], label=info["title"])
                    axes[0].set_title(f"Stellar {info['title']}")
                
                # Error map
                valid_errors = error_map[~np.isnan(error_map)]
                if len(valid_errors) > 0:
                    im2 = axes[1].imshow(
                        error_map, origin="lower", cmap="plasma",
                        vmin=0, vmax=np.percentile(valid_errors, 95),
                        aspect="auto"
                    )
                    plt.colorbar(im2, ax=axes[1], label=f"Error ({info['title']})")
                    axes[1].set_title(f"{info['title']} Error")
                
                # S/N map
                snr_map = np.abs(param_map) / error_map
                valid_snr = snr_map[np.isfinite(snr_map)]
                if len(valid_snr) > 0:
                    im3 = axes[2].imshow(
                        snr_map, origin="lower", cmap="viridis",
                        vmin=0, vmax=np.percentile(valid_snr, 95),
                        aspect="auto"
                    )
                    plt.colorbar(im3, ax=axes[2], label="S/N")
                    axes[2].set_title(f"{info['title']} S/N")
                
                fig.tight_layout()
                fig.savefig(
                    stellar_dir / f"{galaxy_name}_P2P_stellar_{param_name}_with_errors.png",
                    dpi=150
                )
                plt.close(fig)
                
            except Exception as e:
                logger.error(f"Error creating {param_name} plots with errors: {e}")
                plt.close("all")


def create_indices_plots_with_errors(cube, indices_result, indices_errors, 
                                   plots_dir, galaxy_name, pixel_size=None):
    """
    Create spectral indices plots with error visualization
    """
    # Create directory for index plots
    indices_dir = plots_dir / "spectral_indices"
    indices_dir.mkdir(exist_ok=True, parents=True)

    # If pixel_size not provided, get from cube
    if pixel_size is None and hasattr(cube, "_pxl_size_x"):
        pixel_size = (cube._pxl_size_x, cube._pxl_size_y)

    # Plot maps for each index with errors
    for name in indices_result.keys():
        if name in indices_errors:
            try:
                index_map = indices_result[name]
                error_map = indices_errors[name]
                
                # Skip if not valid arrays
                if not isinstance(index_map, np.ndarray) or not isinstance(error_map, np.ndarray):
                    continue
                
                # Create figure with value, error, and S/N
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Index value map
                valid_values = index_map[~np.isnan(index_map)]
                if len(valid_values) > 0:
                    vmin = np.percentile(valid_values, 5)
                    vmax = np.percentile(valid_values, 95)
                    
                    if pixel_size is not None:
                        # Physical coordinates
                        pixel_size_x, pixel_size_y = pixel_size
                        ny, nx = index_map.shape
                        extent = [
                            -nx/2 * pixel_size_x, nx/2 * pixel_size_x,
                            -ny/2 * pixel_size_y, ny/2 * pixel_size_y
                        ]
                        
                        im1 = axes[0].imshow(
                            index_map, origin="lower", cmap="viridis",
                            vmin=vmin, vmax=vmax, aspect="equal", extent=extent
                        )
                        axes[0].set_xlabel('Δ RA (arcsec)')
                        axes[0].set_ylabel('Δ Dec (arcsec)')
                    else:
                        im1 = axes[0].imshow(
                            index_map, origin="lower", cmap="viridis",
                            vmin=vmin, vmax=vmax, aspect="auto"
                        )
                        axes[0].set_xlabel('Pixels')
                        axes[0].set_ylabel('Pixels')
                        
                    plt.colorbar(im1, ax=axes[0], label="Index Value")
                    axes[0].set_title(f"{name} Index")
                
                # Error map
                valid_errors = error_map[~np.isnan(error_map)]
                if len(valid_errors) > 0:
                    if pixel_size is not None:
                        im2 = axes[1].imshow(
                            error_map, origin="lower", cmap="plasma",
                            vmin=0, vmax=np.percentile(valid_errors, 95),
                            aspect="equal", extent=extent
                        )
                        axes[1].set_xlabel('Δ RA (arcsec)')
                        axes[1].set_ylabel('Δ Dec (arcsec)')
                    else:
                        im2 = axes[1].imshow(
                            error_map, origin="lower", cmap="plasma",
                            vmin=0, vmax=np.percentile(valid_errors, 95),
                            aspect="auto"
                        )
                        axes[1].set_xlabel('Pixels')
                        axes[1].set_ylabel('Pixels')
                        
                    plt.colorbar(im2, ax=axes[1], label="Error")
                    axes[1].set_title(f"{name} Error")
                
                # S/N map
                snr_map = np.abs(index_map) / error_map
                valid_snr = snr_map[np.isfinite(snr_map)]
                if len(valid_snr) > 0:
                    if pixel_size is not None:
                        im3 = axes[2].imshow(
                            snr_map, origin="lower", cmap="inferno",
                            vmin=0, vmax=np.percentile(valid_snr, 95),
                            aspect="equal", extent=extent
                        )
                        axes[2].set_xlabel('Δ RA (arcsec)')
                        axes[2].set_ylabel('Δ Dec (arcsec)')
                    else:
                        im3 = axes[2].imshow(
                            snr_map, origin="lower", cmap="inferno",
                            vmin=0, vmax=np.percentile(valid_snr, 95),
                            aspect="auto"
                        )
                        axes[2].set_xlabel('Pixels')
                        axes[2].set_ylabel('Pixels')
                        
                    plt.colorbar(im3, ax=axes[2], label="S/N")
                    axes[2].set_title(f"{name} S/N")
                
                for ax in axes:
                    ax.grid(True, alpha=0.3)
                
                fig.tight_layout()
                fig.savefig(
                    indices_dir / f"{galaxy_name}_P2P_{name}_with_errors.png",
                    dpi=150
                )
                plt.close(fig)
                
            except Exception as e:
                logger.error(f"Error creating index plots for {name}: {e}")
                plt.close("all")


def create_radial_profile_plots_with_errors(
    results, cube, galaxy_name, plots_dir, physical_scale=True, analysis_type="P2P"
):
    """
    Create radial profile plots with error bars
    """
    # Create directory for radial profiles
    radial_dir = Path(plots_dir) / "radial"
    radial_dir.mkdir(exist_ok=True, parents=True)
    
    # Call existing function and add error support
    create_radial_profile_plots(
        results, cube, galaxy_name, plots_dir, physical_scale, analysis_type
    )
    
    # Add additional plots with error bars if errors are available
    if "stellar_kinematics" in results:
        kin = results["stellar_kinematics"]
        
        if "velocity_error" in kin and kin["velocity_error"] is not None:
            try:
                # Get velocity and error fields
                vfield = kin["velocity_field"]
                vfield_err = kin["velocity_error"]
                
                # Get radius
                ny, nx = vfield.shape
                y, x = np.indices((ny, nx))
                center_y, center_x = ny // 2, nx // 2
                r_pix = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                if physical_scale and hasattr(cube, "_pxl_size_x"):
                    radius = r_pix * cube._pxl_size_x
                else:
                    radius = r_pix
                
                # Create mask for valid values
                mask = np.isfinite(vfield) & np.isfinite(vfield_err) & np.isfinite(radius)
                
                if np.any(mask):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Get valid values
                    valid_r = radius[mask]
                    valid_v = vfield[mask]
                    valid_v_err = vfield_err[mask]
                    
                    # Bin data with error propagation
                    from scipy.stats import binned_statistic
                    
                    n_bins = 15
                    bin_edges = np.linspace(np.min(valid_r), np.max(valid_r), n_bins + 1)
                    
                    # Calculate mean and propagate errors in bins
                    v_mean, _, bin_number = binned_statistic(
                        valid_r, valid_v, statistic='mean', bins=bin_edges
                    )
                    
                    # Propagate errors through binning
                    v_err_prop = np.zeros(n_bins)
                    for i in range(n_bins):
                        mask_bin = (bin_number == i + 1)
                        if np.any(mask_bin):
                            # Error propagation for mean: σ_mean = sqrt(sum(σ_i^2)) / n
                            v_err_prop[i] = np.sqrt(np.sum(valid_v_err[mask_bin]**2)) / np.sum(mask_bin)
                    
                    # Calculate bin centers
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    
                    # Plot with error bars
                    ax.errorbar(
                        bin_centers, v_mean, yerr=v_err_prop,
                        fmt='o-', capsize=4, markersize=6, lw=1.5,
                        label="Velocity with errors"
                    )
                    
                    ax.set_xlabel(f"Radius {'(arcsec)' if physical_scale else '(pixels)'}")
                    ax.set_ylabel("Velocity (km/s)")
                    ax.set_title(f"{galaxy_name} - Radial Velocity Profile with Errors")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    visualization.standardize_figure_saving(
                        fig, radial_dir / f"{galaxy_name}_velocity_profile_errors.png"
                    )
                    plt.close(fig)
                    
            except Exception as e:
                logger.error(f"Error creating velocity profile with errors: {e}")
                plt.close("all")


def create_emission_maps(emission_params, plots_dir, galaxy_name, pixel_size=None):
    """
    Create emission line flux and ratio maps

    Parameters
    ----------
    emission_params : dict
        Emission line parameters
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcseconds for physical scaling
    """
    # Find all emission line flux maps
    flux_maps = {}

    # Collect all flux maps
    for key, value in emission_params.items():
        if key.startswith("flux_") and isinstance(value, np.ndarray):
            line_name = key[5:]  # Remove 'flux_' prefix
            flux_maps[line_name] = value

    # If no 'flux_' prefixed keys found, check for 'flux'
    if not flux_maps and "flux" in emission_params:
        # If flux is a dictionary, it might contain line fluxes
        if isinstance(emission_params["flux"], dict):
            for line_name, flux in emission_params["flux"].items():
                flux_maps[line_name] = flux
        # If flux is an array, assume it's a single emission line
        elif isinstance(emission_params["flux"], np.ndarray):
            flux_maps["Combined"] = emission_params["flux"]

    # Create maps for each emission line
    for line_name, flux_map in flux_maps.items():
        try:
            # Check data validity
            valid_values = flux_map[~np.isnan(flux_map) & (flux_map > 0)]
            if len(valid_values) > 0:
                fig, ax = plt.subplots(figsize=(8, 7))

                # Use log scale for display
                norm = mcolors.LogNorm(
                    vmin=np.percentile(valid_values, 1),
                    vmax=np.percentile(valid_values, 99),
                )
                
                # Apply physical scaling if pixel size provided
                if pixel_size is not None:
                    pixel_size_x, pixel_size_y = pixel_size
                    ny, nx = flux_map.shape
                    x_min = -nx/2 * pixel_size_x
                    x_max = nx/2 * pixel_size_x
                    y_min = -ny/2 * pixel_size_y
                    y_max = ny/2 * pixel_size_y
                    extent = [x_min, x_max, y_min, y_max]
                    
                    im = ax.imshow(
                        flux_map, 
                        origin="lower", 
                        cmap="inferno", 
                        norm=norm, 
                        aspect="equal",
                        extent=extent
                    )
                    ax.set_xlabel('Δ RA (arcsec)')
                    ax.set_ylabel('Δ Dec (arcsec)')
                else:
                    im = ax.imshow(
                        flux_map, origin="lower", cmap="inferno", norm=norm, aspect="auto"
                    )
                    ax.set_xlabel('Pixels')
                    ax.set_ylabel('Pixels')
                    
                plt.colorbar(im, ax=ax, label="Flux")
                ax.set_title(f"{line_name} Flux")
                ax.grid(True, alpha=0.3)

                fig.tight_layout()
                fig.savefig(
                    plots_dir / f"{galaxy_name}_P2P_{line_name}_flux.png", dpi=150
                )
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating flux map for {line_name}: {str(e)}")
            plt.close("all")

    # Create BPT-style diagnostic plots if line ratios available
    if "line_ratios" in emission_params and "OIII_Hb" in emission_params["line_ratios"]:
        try:
            # Get line ratio
            oiii_hb = emission_params["line_ratios"]["OIII_Hb"]

            # Check data validity
            valid_values = oiii_hb[~np.isnan(oiii_hb) & (oiii_hb > 0)]
            if len(valid_values) > 0:
                fig, ax = plt.subplots(figsize=(8, 7))

                # Use log scale for display
                norm = mcolors.LogNorm(
                    vmin=np.percentile(valid_values, 1),
                    vmax=np.percentile(valid_values, 99),
                )
                
                # Apply physical scaling if pixel size provided
                if pixel_size is not None:
                    pixel_size_x, pixel_size_y = pixel_size
                    ny, nx = oiii_hb.shape
                    x_min = -nx/2 * pixel_size_x
                    x_max = nx/2 * pixel_size_x
                    y_min = -ny/2 * pixel_size_y
                    y_max = ny/2 * pixel_size_y
                    extent = [x_min, x_max, y_min, y_max]
                    
                    im = ax.imshow(
                        oiii_hb, 
                        origin="lower", 
                        cmap="viridis", 
                        norm=norm, 
                        aspect="equal",
                        extent=extent
                    )
                    ax.set_xlabel('Δ RA (arcsec)')
                    ax.set_ylabel('Δ Dec (arcsec)')
                else:
                    im = ax.imshow(
                        oiii_hb, origin="lower", cmap="viridis", norm=norm, aspect="auto"
                    )
                    ax.set_xlabel('Pixels')
                    ax.set_ylabel('Pixels')

                plt.colorbar(im, ax=ax, label="Ratio")
                ax.set_title("OIII/Hβ Ratio")
                ax.grid(True, alpha=0.3)

                fig.tight_layout()
                fig.savefig(plots_dir / f"{galaxy_name}_P2P_OIII_Hb_ratio.png", dpi=150)
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating OIII/Hb ratio map: {str(e)}")
            plt.close("all")


def create_sample_fits(
    cube, velocity_field, bestfit_field, emission_result, plots_dir, galaxy_name
):
    """
    Create spectrum fits plots for sample pixels

    Parameters
    ----------
    cube : MUSECube
        MUSE data cube object
    velocity_field : ndarray
        Velocity field
    bestfit_field : ndarray
        Best-fit spectra
    emission_result : dict
        Emission line results
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    """
    n_y, n_x = velocity_field.shape

    # Select sample positions
    center_y, center_x = n_y // 2, n_x // 2
    sample_positions = [
        (center_y, center_x),  # Center
        (center_y, min(center_x + n_x // 4, n_x - 1)),  # Right
        (min(center_y + n_y // 4, n_y - 1), center_x),  # Top
        (max(center_y - n_y // 4, 0), max(center_x - n_x // 4, 0)),  # Bottom-left
    ]

    # Filter to ensure positions are valid
    valid_positions = []
    for y, x in sample_positions:
        if 0 <= y < n_y and 0 <= x < n_x and np.isfinite(velocity_field[y, x]):
            valid_positions.append((y, x))

    # If no valid positions found, try to find at least one
    if not valid_positions:
        valid_mask = np.isfinite(velocity_field)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)
            valid_positions = [(valid_indices[0][0], valid_indices[1][0])]
        else:
            logger.warning("No valid pixels found for spectrum plots. Skipping.")
            return

    for i, (y, x) in enumerate(valid_positions):
        try:
            # Get spaxel index
            idx = y * n_x + x

            # Get observed spectrum
            observed = cube._spectra[:, idx]

            # Get model spectrum
            model = bestfit_field[:, y, x]

            # Get gas component if available
            gas_comp = None
            if emission_result is not None:
                # Try to get gas component from emission_result
                if "gas_bestfit" in emission_result:
                    gas_bestfit = emission_result["gas_bestfit"]
                    if gas_bestfit is not None:
                        # Check shape and extract appropriately
                        if len(gas_bestfit.shape) == 3:  # [n_wave, n_y, n_x]
                            gas_comp = gas_bestfit[:, y, x]
                        elif len(gas_bestfit.shape) == 2:  # [n_wave, n_spectra]
                            gas_comp = gas_bestfit[:, idx]
                elif "gas_bestfit_field" in emission_result:
                    gas_bestfit = emission_result["gas_bestfit_field"]
                    if gas_bestfit is not None:
                        if len(gas_bestfit.shape) == 3:
                            gas_comp = gas_bestfit[:, y, x]
                        elif len(gas_bestfit.shape) == 2:
                            gas_comp = gas_bestfit[:, idx]

                # If not found in emission_result, try cube
                if gas_comp is None and hasattr(cube, "_gas_bestfit_field"):
                    gas_comp = cube._gas_bestfit_field[:, y, x]

                # Verify it's a valid array
                if gas_comp is not None and not np.any(np.isfinite(gas_comp)):
                    gas_comp = None

            # Create stellar component by subtracting gas
            stellar_comp = model.copy()
            if gas_comp is not None:
                stellar_comp -= gas_comp

            # Create plot with error handling
            try:
                fig, axes = visualization.plot_spectrum_fit(
                    wavelength=cube._lambda_gal,
                    observed_flux=observed,
                    model_flux=model,
                    stellar_flux=stellar_comp,
                    gas_flux=gas_comp,
                    title=f"Pixel ({x}, {y})",
                )

                fig.savefig(plots_dir / f"{galaxy_name}_P2P_spectrum_{i}.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                logger.error(
                    f"Error in plot_spectrum_fit for pixel ({x}, {y}): {str(e)}"
                )
                plt.close("all")

        except Exception as e:
            logger.error(f"Error creating spectrum plot for pixel ({x}, {y}): {str(e)}")
            plt.close("all")


def create_indices_plots(cube, indices_result, plots_dir, galaxy_name, pixel_size=None):
    """
    Create spectral indices plots

    Parameters
    ----------
    cube : MUSECube
        MUSE data cube object
    indices_result : dict
        Spectral indices results
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    pixel_size : tuple, optional
        (pixel_size_x, pixel_size_y) in arcseconds for physical scaling
    """
    # Plot maps for each index
    for name, index_map in indices_result.items():
        try:
            fig, ax = plt.subplots(figsize=(8, 7))

            # Calculate valid range
            valid_values = index_map[~np.isnan(index_map)]
            if len(valid_values) > 0:
                vmin = np.percentile(valid_values, 5)
                vmax = np.percentile(valid_values, 95)

                # Check for valid range
                if vmin < vmax and np.isfinite(vmin) and np.isfinite(vmax):
                    # Apply physical scaling if pixel size provided
                    if pixel_size is not None:
                        pixel_size_x, pixel_size_y = pixel_size
                        ny, nx = index_map.shape
                        x_min = -nx/2 * pixel_size_x
                        x_max = nx/2 * pixel_size_x
                        y_min = -ny/2 * pixel_size_y
                        y_max = ny/2 * pixel_size_y
                        extent = [x_min, x_max, y_min, y_max]
                        
                        # Plot index map
                        im = ax.imshow(
                            index_map,
                            origin="lower",
                            cmap="viridis",
                            vmin=vmin,
                            vmax=vmax,
                            aspect="equal",
                            extent=extent
                        )
                        ax.set_xlabel('Δ RA (arcsec)')
                        ax.set_ylabel('Δ Dec (arcsec)')
                    else:
                        # Plot index map
                        im = ax.imshow(
                            index_map,
                            origin="lower",
                            cmap="viridis",
                            vmin=vmin,
                            vmax=vmax,
                            aspect="auto",
                        )
                        ax.set_xlabel('Pixels')
                        ax.set_ylabel('Pixels')
                        
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"{name} Index")
                    ax.grid(True, alpha=0.3)

                    fig.tight_layout()
                    fig.savefig(
                        plots_dir / f"{galaxy_name}_P2P_{name}_index.png", dpi=150
                    )
                else:
                    logger.warning(
                        f"Invalid value range for {name} index map: vmin={vmin}, vmax={vmax}"
                    )
            else:
                logger.warning(f"No valid values in {name} index map")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating index map for {name}: {str(e)}")
            plt.close("all")  # Ensure all figures are closed


def create_radial_profile_plots(
    results, cube, galaxy_name, plots_dir, physical_scale=True, analysis_type="P2P"
):
    """
    Create radial profile plots for analysis results
    
    Parameters
    ----------
    results : dict
        Analysis results
    cube : MUSECube
        MUSE data cube
    galaxy_name : str
        Galaxy name
    plots_dir : str or Path
        Directory to save plots
    physical_scale : bool, default=True
        Whether to use physical scale for radius
    analysis_type : str, default="P2P"
        Type of analysis (P2P, VNB, RDB)
    """
    # Check if cube was provided for physical scaling
    if physical_scale and cube is None:
        logger.warning("No cube provided for physical scaling, using pixel units")
        physical_scale = False
    
    # Create directory for radial profiles
    radial_dir = Path(plots_dir) / "radial"
    radial_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract coordinates and velocity/dispersion fields
    try:
        # Get fields from results
        vfield = results.get("velocity_field", None)
        dfield = results.get("dispersion_field", None)
        
        # Check if we have velocity and dispersion fields
        if vfield is None or dfield is None:
            if "stellar_kinematics" in results:
                vfield = results["stellar_kinematics"].get("velocity_field", None)
                dfield = results["stellar_kinematics"].get("dispersion_field", None)
        
        # Skip if we don't have velocity and dispersion fields
        if vfield is None or dfield is None:
            logger.warning("No velocity or dispersion fields found for radial profiles")
            return
            
        # Get dimensions
        ny, nx = vfield.shape
        
        # Calculate coordinates
        y, x = np.indices((ny, nx))
        center_y, center_x = ny // 2, nx // 2
        
        # Calculate radius in pixels
        r_pix = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Convert to physical units if requested
        if physical_scale and hasattr(cube, "_pxl_size_x"):
            radius = r_pix * cube._pxl_size_x
        else:
            radius = r_pix
            
        # Check for physical radius and use if available
        r_galaxy = None
        if hasattr(cube, "_physical_radius") and cube._physical_radius is not None:
            r_galaxy = cube._physical_radius
            radius = r_galaxy
            
        # Create mask for valid radius and valid values
        # Use a safer approach for checking finite values
        r_mask = np.ones_like(radius, dtype=bool)
        try:
            r_mask = np.isfinite(radius) & (radius > 0)
        except TypeError:
            # If isfinite fails, try to convert to float
            try:
                r_mask = np.isfinite(radius.astype(float)) & (radius.astype(float) > 0)
            except:
                logger.warning("Could not create radius mask, using all pixels")
        
        # Create safe masks for velocity and dispersion
        v_mask = np.ones_like(vfield, dtype=bool) & r_mask
        try:
            v_mask = r_mask & np.isfinite(vfield)
        except TypeError:
            # Try to handle non-numeric arrays
            try:
                v_mask = r_mask & np.isfinite(vfield.astype(float))
            except:
                logger.warning("Could not create velocity mask, using radius mask only")
                v_mask = r_mask
        
        d_mask = np.ones_like(dfield, dtype=bool) & r_mask
        try:
            d_mask = r_mask & np.isfinite(dfield)
        except TypeError:
            # Try to handle non-numeric arrays
            try:
                d_mask = r_mask & np.isfinite(dfield.astype(float))
            except:
                logger.warning("Could not create dispersion mask, using radius mask only")
                d_mask = r_mask
        
        # Create velocity profile
        if np.any(v_mask):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get valid velocity values safely
            valid_r = safe_extract(radius, v_mask)
            valid_v = safe_extract(vfield, v_mask)
            
            if len(valid_r) > 0 and len(valid_v) > 0:
                # Calculate binned statistics
                try:
                    from scipy.stats import binned_statistic
                    
                    # Bin data radially
                    bin_centers, v_mean, v_std = radial_binning_statistics(
                        valid_r, valid_v, n_bins=20
                    )
                    
                    # Plot binned profile
                    ax.errorbar(
                        bin_centers, v_mean, yerr=v_std, 
                        fmt='o-', capsize=4, markersize=6, lw=1.5,
                        label="Binned profile"
                    )
                    
                    # Plot scatter if not too many points
                    if len(valid_r) <= 1000:
                        ax.scatter(
                            valid_r, valid_v, 
                            alpha=0.3, s=5, color='gray',
                            label="Individual pixels"
                        )
                except Exception as e:
                    # Fallback to simple scatter plot
                    logger.debug(f"Error in velocity binned statistics: {e}")
                    ax.scatter(valid_r, valid_v, alpha=0.5, s=10)
                
                # Set labels and title
                ax.set_xlabel(f"Radius {'(arcsec)' if physical_scale else '(pixels)'}")
                ax.set_ylabel("Velocity (km/s)")
                ax.set_title(f"{galaxy_name} - Radial Velocity Profile")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Save figure
                visualization.standardize_figure_saving(
                    fig, radial_dir / f"{galaxy_name}_velocity_profile.png",
                    dpi=150
                )
            plt.close(fig)
            
        # Create dispersion profile
        if np.any(d_mask):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get valid dispersion values safely
            valid_r = safe_extract(radius, d_mask)
            valid_d = safe_extract(dfield, d_mask)
            
            if len(valid_r) > 0 and len(valid_d) > 0:
                # Calculate binned statistics
                try:
                    from scipy.stats import binned_statistic
                    
                    # Bin data radially
                    bin_centers, d_mean, d_std = radial_binning_statistics(
                        valid_r, valid_d, n_bins=20
                    )
                    
                    # Plot binned profile
                    ax.errorbar(
                        bin_centers, d_mean, yerr=d_std, 
                        fmt='o-', capsize=4, markersize=6, lw=1.5,
                        label="Binned profile"
                    )
                    
                    # Plot scatter if not too many points
                    if len(valid_r) <= 1000:
                        ax.scatter(
                            valid_r, valid_d, 
                            alpha=0.3, s=5, color='gray',
                            label="Individual pixels"
                        )
                except Exception as e:
                    # Fallback to simple scatter plot
                    logger.debug(f"Error in dispersion binned statistics: {e}")
                    ax.scatter(valid_r, valid_d, alpha=0.5, s=10)
                
                # Set labels and title
                ax.set_xlabel(f"Radius {'(arcsec)' if physical_scale else '(pixels)'}")
                ax.set_ylabel("Dispersion (km/s)")
                ax.set_title(f"{galaxy_name} - Radial Dispersion Profile")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Save figure
                visualization.standardize_figure_saving(
                    fig, radial_dir / f"{galaxy_name}_dispersion_profile.png",
                    dpi=150
                )
            plt.close(fig)
            
        # Create spectral indices profiles
        if "spectral_indices" in results:
            indices_dir = radial_dir / "indices"
            indices_dir.mkdir(exist_ok=True, parents=True)
            
            # Process each index and method
            for method_key in ["auto", "original", "fit"]:
                # Check if this method exists in the results
                if method_key not in results["spectral_indices"]:
                    # Skip if this method isn't in the results
                    continue
                    
                # Get indices for this method
                method_indices = results["spectral_indices"][method_key]
                
                # Handle both dict and non-dict formats
                if isinstance(method_indices, dict):
                    indices_dict = method_indices
                else:
                    # Skip if not a dictionary
                    logger.warning(f"Spectral indices for method {method_key} not in expected format")
                    continue
                
                # Process each index
                for idx_name, index_map in indices_dict.items():
                    try:
                        # Create safe mask for this index
                        idx_mask = np.ones_like(r_mask, dtype=bool) & r_mask
                        
                        # Convert index_map to a numeric array if possible
                        numeric_index_map = safe_convert_to_numeric(index_map)
                        
                        # Skip if conversion failed
                        if numeric_index_map is None:
                            logger.error(f"Could not convert {idx_name} to numeric values for {method_key}")
                            continue
                            
                        # Apply isnan safely now
                        try:
                            idx_mask = r_mask & ~np.isnan(numeric_index_map)
                        except:
                            logger.warning(f"Couldn't create mask for {idx_name}, using radius mask")
                            
                        # Extract valid values safely
                        valid_r = safe_extract(radius, idx_mask)
                        valid_idx = safe_extract(numeric_index_map, idx_mask)
                        
                        if len(valid_r) > 0 and len(valid_idx) > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Calculate binned statistics
                            try:
                                bin_centers, idx_mean, idx_std = radial_binning_statistics(
                                    valid_r, valid_idx, n_bins=15
                                )
                                
                                # Plot binned profile
                                ax.errorbar(
                                    bin_centers, idx_mean, yerr=idx_std, 
                                    fmt='o-', capsize=4, markersize=6, lw=1.5,
                                    label="Binned profile"
                                )
                                
                                # Plot scatter if not too many points
                                if len(valid_r) <= 1000:
                                    ax.scatter(
                                        valid_r, valid_idx, 
                                        alpha=0.3, s=5, color='gray',
                                        label="Individual pixels"
                                    )
                            except Exception as e:
                                # Fallback to simple scatter plot
                                logger.debug(f"Error in index binned statistics for {idx_name}: {e}")
                                ax.scatter(valid_r, valid_idx, alpha=0.5, s=10)
                            
                            # Set labels and title
                            ax.set_xlabel(f"Radius {'(arcsec)' if physical_scale else '(pixels)'}")
                            ax.set_ylabel(f"{idx_name} Index")
                            ax.set_title(f"{galaxy_name} - {idx_name} Radial Profile ({method_key})")
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            # Save figure
                            visualization.standardize_figure_saving(
                                fig, indices_dir / f"{galaxy_name}_{idx_name}_{method_key}_profile.png",
                                dpi=150
                            )
                            plt.close(fig)
                        else:
                            logger.warning(f"No valid data for {idx_name} with method {method_key}")
                    except Exception as e:
                        logger.error(f"Error creating index map for {method_key}: {e}")
                        plt.close("all")
        
    except Exception as e:
        logger.error(f"Error creating radial profile plots: {e}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close("all")


def radial_binning_statistics(radius, values, n_bins=20):
    """
    Calculate binned statistics for radial profiles
    
    Parameters
    ----------
    radius : ndarray
        Radius values
    values : ndarray
        Data values
    n_bins : int, default=20
        Number of bins
    
    Returns
    -------
    tuple
        (bin_centers, mean_values, std_values)
    """
    from scipy.stats import binned_statistic
    
    # Calculate reasonable bin edges
    rmin, rmax = np.min(radius), np.max(radius)
    bin_edges = np.linspace(rmin, rmax, n_bins + 1)
    
    # Calculate statistics
    mean_values, bin_edges, bin_number = binned_statistic(
        radius, values, statistic='mean', bins=bin_edges
    )
    
    # Calculate standard deviation in each bin
    std_values, _, _ = binned_statistic(
        radius, values, statistic='std', bins=bin_edges
    )
    
    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    return bin_centers, mean_values, std_values


def safe_convert_to_numeric(array):
    """
    Safely convert an array to numeric values
    
    Parameters
    ----------
    array : any
        Array to convert
    
    Returns
    -------
    ndarray or None
        Numeric array or None if conversion failed
    """
    # Check if already a numeric array
    if isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.number):
        return array
    
    # Try to convert
    try:
        # First, convert to a flat list if it's an array-like object
        if hasattr(array, 'flatten'):
            try:
                flat_list = array.flatten()
            except:
                flat_list = array
        elif hasattr(array, 'tolist'):
            try:
                flat_list = array.tolist()
            except:
                flat_list = array
        else:
            flat_list = array
        
        # Try to convert each element
        numeric_list = []
        for item in flat_list:
            try:
                numeric_list.append(float(item))
            except (ValueError, TypeError):
                numeric_list.append(np.nan)
        
        # Convert back to an array with original shape if possible
        if hasattr(array, 'shape'):
            try:
                return np.array(numeric_list, dtype=float).reshape(array.shape)
            except:
                return np.array(numeric_list, dtype=float)
        else:
            return np.array(numeric_list, dtype=float)
    except Exception as e:
        logger.debug(f"Error converting to numeric: {e}")
        return None