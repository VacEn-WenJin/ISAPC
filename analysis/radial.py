"""
Radial binning analysis module for ISAPC
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm  # Required for color normalization
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots

import galaxy_params
import visualization
from binning import (
    RadialBinnedData,
    calculate_radial_bins,
    calculate_wavelength_intersection,
    combine_spectra_efficiently,
)
from utils.io import save_standardized_results

logger = logging.getLogger(__name__)


def run_rdb_analysis(args, cube, p2p_results=None):
    """
    Run radial binning analysis on MUSE data cube
    
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
    logger.info("Starting radial binning analysis...")
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
    plots_dir = galaxy_dir / "Plots" / "RDB"

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
                logger.info("Successfully loaded P2P results for RDB analysis")
            elif std_results_path.exists():
                p2p_results = np.load(std_results_path, allow_pickle=True)
                logger.info(
                    "Successfully loaded standardized P2P results for RDB analysis"
                )
        except Exception as e:
            logger.warning(f"Error loading P2P results: {e}")
            p2p_results = None

    # Set up binning parameters
    n_rings = args.n_rings if hasattr(args, "n_rings") else 10
    log_spacing = args.log_spacing if hasattr(args, "log_spacing") else False
    pa = args.pa if hasattr(args, "pa") else 0.0
    ellipticity = args.ellipticity if hasattr(args, "ellipticity") else 0.0
    center_x = args.center_x if hasattr(args, "center_x") else None
    center_y = args.center_y if hasattr(args, "center_y") else None
    use_physical_radius = args.physical_radius if hasattr(args, "physical_radius") else False

    # Get cube coordinates, ensuring we use the original x,y coordinates from the cube
    x = cube.x if hasattr(cube, "x") else np.arange(cube._n_x * cube._n_y) % cube._n_x
    y = cube.y if hasattr(cube, "y") else np.arange(cube._n_x * cube._n_y) // cube._n_x

    # Calculate physical radius if requested
    r_galaxy = None
    ellipse_params = None
    if use_physical_radius:
        try:
            from physical_radius import calculate_galaxy_radius
            
            # Create flux map for radius calculation
            flux_2d = np.nanmedian(cube._cube_data, axis=0)
            
            # Calculate physical radius and ellipse parameters
            r_galaxy, ellipse_params = calculate_galaxy_radius(
                flux_2d,
                pixel_size_x=cube._pxl_size_x,
                pixel_size_y=cube._pxl_size_y
            )
            
            # Update parameters with calculated values
            pa = ellipse_params["PA_degrees"]
            ellipticity = ellipse_params["ellipticity"]
            center_x = ellipse_params["center_x"]
            center_y = ellipse_params["center_y"]
            
            logger.info(f"Using physical radius with PA={pa:.1f}°, ε={ellipticity:.2f}, "
                        f"center=({center_x:.1f}, {center_y:.1f})")
                        
            # Store for later use
            cube._physical_radius = r_galaxy
            cube._ellipse_params = ellipse_params
            
        except Exception as e:
            logger.warning(f"Error calculating physical radius: {e}")
            use_physical_radius = False

    # Run radial binning
    logger.info(f"Running radial binning with {n_rings} rings, "
                f"PA={pa:.1f}°, ellipticity={ellipticity:.2f}")
    
    # Calculate radial bins
    bin_num, bin_edges, bin_radii = calculate_radial_bins(
        x, y,
        center_x=center_x,
        center_y=center_y,
        pa=pa,
        ellipticity=ellipticity,
        n_rings=n_rings,
        log_spacing=log_spacing,
        r_galaxy=r_galaxy.ravel() if r_galaxy is not None else None
    )

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
        spectra, wavelength, bin_indices, velocity_field, cube._n_x, cube._n_y
    )

    # Create metadata
    metadata = {
        "nx": cube._n_x,
        "ny": cube._n_y,
        "n_rings": n_rings,
        "pa": pa,
        "ellipticity": ellipticity,
        "center_x": center_x if center_x is not None else cube._n_x // 2,
        "center_y": center_y if center_y is not None else cube._n_y // 2,
        "log_spacing": log_spacing,
        "bin_edges": bin_edges,
        "bin_radii": bin_radii,
        "time": time.time(),
        "galaxy_name": galaxy_name,
        "analysis_type": "RDB",
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

    # Add WCS if available
    if hasattr(cube, "_wcs") and cube._wcs is not None:
        metadata["has_wcs"] = True

    # Create RadialBinnedData object
    binned_data = RadialBinnedData(
        bin_num=bin_num,
        bin_indices=bin_indices,
        spectra=binned_spectra,
        wavelength=wavelength,
        metadata=metadata,
        bin_radii=bin_radii,
    )

    # Set up binning in the cube - this connects our binned data to the cube
    if hasattr(cube, "setup_binning"):
        cube.setup_binning("RDB", binned_data)
    
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
                
            # Calculate indices
            indices_result = cube.calculate_spectral_indices(
                indices_list=indices_list,
                n_jobs=args.n_jobs,
            )
            
            logger.info(f"Spectral indices calculation completed in {time.time() - start_idx_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Error calculating spectral indices: {e}")
            indices_result = None

    # Save binned data for later reuse
    try:
        binned_data.save(data_dir / f"{galaxy_name}_RDB_binned.npz")
    except Exception as e:
        logger.warning(f"Error saving binned data: {e}")

    # Calculate rotation or kinematic model if available
    global_kinematics = None
    if hasattr(args, "rotation_model") and args.rotation_model:
        try:
            logger.info("Calculating rotation model for radial velocity profile...")
            
            # Create GalaxyParameters object
            gp = galaxy_params.GalaxyParameters(
                velocity_field=stellar_velocity_field,
                dispersion_field=stellar_dispersion_field,
                pixelsize=cube._pxl_size_x,
                radius=bin_radii,
            )
            
            # Fit rotation curve
            rotation_result = gp.fit_rotation_curve()
            
            # Calculate kinematics
            kinematics_result = gp.calculate_kinematics()
            
            # Combine results
            global_kinematics = {**rotation_result, **kinematics_result}
            
            logger.info("Rotation model completed")
        except Exception as e:
            logger.warning(f"Error calculating rotation model: {e}")
    
    # Create standardized results dictionary
    rdb_results = {
        "analysis_type": "RDB",
        "binning": {
            "bin_num": bin_num,
            "n_rings": n_rings,
            "pa": pa,
            "ellipticity": ellipticity,
            "center_x": center_x if center_x is not None else cube._n_x // 2,
            "center_y": center_y if center_y is not None else cube._n_y // 2,
            "log_spacing": log_spacing,
            "bin_edges": bin_edges,
            "bin_radii": bin_radii,
        },
        "stellar_kinematics": {
            "velocity": stellar_velocity_field,
            "dispersion": stellar_dispersion_field,
        },
        "distance": {
            "bin_distances": bin_radii,  # Use bin radii as distances
            "pixelsize_x": cube._pxl_size_x,
            "pixelsize_y": cube._pxl_size_y,
        },
    }
    
    # Add global kinematics if available
    if global_kinematics is not None:
        rdb_results["global_kinematics"] = global_kinematics

    # Add emission line results if available
    if emission_result is not None:
        rdb_results["emission"] = {}
        
        # Extract emission line information - flux, velocity, dispersion
        for key in ["flux", "velocity", "dispersion"]:
            if key in emission_result and emission_result[key]:
                rdb_results["emission"][key] = emission_result[key]
                
        # Add signal/noise information
        for key in ["signal", "noise", "snr"]:
            if key in emission_result and emission_result[key] is not None:
                rdb_results[key] = emission_result[key]

    # Add spectral indices results if available
    if indices_result is not None:
        rdb_results["bin_indices"] = indices_result

    # Save standardized results
    should_save = not hasattr(args, "no_save") or not args.no_save
    if should_save:
        save_standardized_results(galaxy_name, "RDB", rdb_results, output_dir)

    # Create visualization plots
    should_plot = not hasattr(args, "no_plots") or not args.no_plots
    if should_plot:
        create_rdb_plots(cube, rdb_results, galaxy_name, plots_dir, args)

    logger.info("Radial binning analysis completed")
    return rdb_results


def create_rdb_plots(cube, rdb_results, galaxy_name, plots_dir, args):
    """
    Create visualization plots for RDB analysis
    
    Parameters
    ----------
    cube : MUSECube
        MUSE cube with binned data
    rdb_results : dict
        RDB analysis results
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
        
        # Create radial bin map
        if "binning" in rdb_results and "bin_num" in rdb_results["binning"]:
            try:
                bin_num = rdb_results["binning"]["bin_num"]
                bin_radii = rdb_results["binning"]["bin_radii"]
                center_x = rdb_results["binning"]["center_x"]
                center_y = rdb_results["binning"]["center_y"]
                pa = rdb_results["binning"]["pa"]
                ellipticity = rdb_results["binning"]["ellipticity"]
                
                # Reshape to 2D for plotting
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
                
                # Create radial bin map with physical scaling
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d,
                    None,
                    ax=ax,
                    cmap="tab20",
                    title=f"{galaxy_name} - Radial Bins",
                    physical_scale=True,
                    pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                    wcs=cube._wcs if hasattr(cube, "_wcs") else None
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_bin_map.png"
                )
                plt.close(fig)
                
                # Create radial profile plots
                if "distance" in rdb_results and "bin_distances" in rdb_results["distance"]:
                    try:
                        bin_distances = rdb_results["distance"]["bin_distances"]
                        
                        # Create radial distance vs bin number plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(np.arange(len(bin_distances)), bin_distances, "o-")
                        ax.set_xlabel("Bin Number")
                        ax.set_ylabel("Radius (arcsec)")
                        ax.set_title(f"{galaxy_name} - Radial Bin Distribution")
                        ax.grid(True, alpha=0.3)
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_RDB_bin_distances.png"
                        )
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating bin distance plot: {e}")
                        plt.close("all")
            except Exception as e:
                logger.warning(f"Error creating radial bin map: {e}")
                plt.close("all")
        
        # Create kinematics plots
        if "stellar_kinematics" in rdb_results and "distance" in rdb_results and "bin_distances" in rdb_results["distance"]:
            try:
                velocity = rdb_results["stellar_kinematics"]["velocity"]
                dispersion = rdb_results["stellar_kinematics"]["dispersion"]
                bin_distances = rdb_results["distance"]["bin_distances"]
                bin_num = rdb_results["binning"]["bin_num"]
                
                # Convert to numpy arrays
                velocity = np.asarray(velocity, dtype=float)
                dispersion = np.asarray(dispersion, dtype=float)
                bin_distances = np.asarray(bin_distances, dtype=float)
                
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
                    
                # Create radial velocity profile
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(bin_distances, velocity, "o-", markersize=6, linewidth=1.5)
                ax.set_xlabel("Radius (arcsec)")
                ax.set_ylabel("Velocity (km/s)")
                ax.set_title(f"{galaxy_name} - Radial Velocity Profile")
                ax.grid(True, alpha=0.3)
                
                # Add rotation curve fit if available
                if (
                    "global_kinematics" in rdb_results 
                    and "rotation_curve" in rdb_results["global_kinematics"]
                    and "fit" in rdb_results["global_kinematics"]["rotation_curve"]
                ):
                    try:
                        fit = rdb_results["global_kinematics"]["rotation_curve"]["fit"]
                        fit_radius = rdb_results["global_kinematics"]["rotation_curve"].get("fit_radius", bin_distances)
                        ax.plot(fit_radius, fit, "r-", lw=2, alpha=0.7, label="Rotation Curve Fit")
                        ax.legend()
                    except Exception as e:
                        logger.warning(f"Error plotting rotation curve fit: {e}")
                
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_velocity_profile.png"
                )
                plt.close(fig)
                
                # Create radial dispersion profile
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(bin_distances, dispersion, "o-", markersize=6, linewidth=1.5)
                ax.set_xlabel("Radius (arcsec)")
                ax.set_ylabel("Dispersion (km/s)")
                ax.set_title(f"{galaxy_name} - Radial Dispersion Profile")
                ax.grid(True, alpha=0.3)
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_dispersion_profile.png"
                )
                plt.close(fig)
                
                # Create velocity map with physical scaling
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d,
                    velocity,
                    ax=ax,
                    cmap="coolwarm",
                    title=f"{galaxy_name} - RDB Velocity",
                    colorbar_label="Velocity (km/s)",
                    physical_scale=True,
                    pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                    wcs=cube._wcs if hasattr(cube, "_wcs") else None
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_velocity_map.png"
                )
                plt.close(fig)
                
                # Create dispersion map with physical scaling
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d,
                    dispersion,
                    ax=ax,
                    cmap="viridis",
                    title=f"{galaxy_name} - RDB Dispersion",
                    colorbar_label="Dispersion (km/s)",
                    physical_scale=True,
                    pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                    wcs=cube._wcs if hasattr(cube, "_wcs") else None
                )
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_dispersion_map.png"
                )
                plt.close(fig)
                
                # Create combined kinematics profile
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot velocity profile
                axes[0].plot(bin_distances, velocity, "o-", markersize=6, linewidth=1.5)
                axes[0].set_xlabel("Radius (arcsec)")
                axes[0].set_ylabel("Velocity (km/s)")
                axes[0].set_title("Stellar Velocity Profile")
                axes[0].grid(True, alpha=0.3)
                
                # Plot dispersion profile
                axes[1].plot(bin_distances, dispersion, "o-", markersize=6, linewidth=1.5)
                axes[1].set_xlabel("Radius (arcsec)")
                axes[1].set_ylabel("Dispersion (km/s)")
                axes[1].set_title("Stellar Dispersion Profile")
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_kinematics_profile.png"
                )
                plt.close(fig)
                
                # Create V/σ profile
                fig, ax = plt.subplots(figsize=(10, 6))
                v_sigma = np.abs(velocity) / dispersion
                ax.plot(bin_distances, v_sigma, "o-", markersize=6, linewidth=1.5)
                ax.set_xlabel("Radius (arcsec)")
                ax.set_ylabel("|V|/σ")
                ax.set_title(f"{galaxy_name} - V/σ Radial Profile")
                ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
                ax.grid(True, alpha=0.3)
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_vsigma_profile.png"
                )
                plt.close(fig)
                
                # Create kinematics summary plot
                try:
                    # Parameters for the summary plot
                    rotation_curve = None
                    params = None
                    
                    if "global_kinematics" in rdb_results:
                        if "rotation_curve" in rdb_results["global_kinematics"]:
                            rotation_curve = rdb_results["global_kinematics"]["rotation_curve"]
                        params = {k: v for k, v in rdb_results["global_kinematics"].items() 
                                 if k not in ["rotation_curve"]}
                    
                    # Convert bin map to 2D velocity and dispersion fields
                    vel_2d = np.full_like(bin_num_2d, np.nan, dtype=float)
                    disp_2d = np.full_like(bin_num_2d, np.nan, dtype=float)
                    
                    for i in range(len(velocity)):
                        if i < len(velocity) and i < np.max(bin_num_2d) + 1:
                            vel_2d[bin_num_2d == i] = velocity[i]
                        if i < len(dispersion) and i < np.max(bin_num_2d) + 1:
                            disp_2d[bin_num_2d == i] = dispersion[i]
                    
                    # Create summary plot
                    fig = visualization.plot_kinematics_summary(
                        velocity_field=vel_2d,
                        dispersion_field=disp_2d,
                        rotation_curve=rotation_curve,
                        params=params,
                        equal_aspect=True,
                        physical_scale=True,
                        pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                        wcs=cube._wcs if hasattr(cube, "_wcs") else None
                    )
                    
                    if fig is not None:
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_RDB_kinematics_summary.png"
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
                
        # Create emission line plots if available
        if "emission" in rdb_results and "distance" in rdb_results and "bin_distances" in rdb_results["distance"]:
            try:
                emission = rdb_results["emission"]
                bin_distances = rdb_results["distance"]["bin_distances"]
                bin_num = rdb_results["binning"]["bin_num"]
                
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
                                        fig, emission_dir / f"{galaxy_name}_RDB_{line_name}_{field_name}_map.png"
                                    )
                                    plt.close(fig)
                                    
                                    # Create radial profile
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Check dimensions match
                                    if len(bin_distances) == len(line_values):
                                        ax.plot(bin_distances, line_values, "o-", markersize=6, linewidth=1.5)
                                        ax.set_xlabel("Radius (arcsec)")
                                        
                                        # Set y-axis based on field type
                                        if field_name == "flux":
                                            ax.set_ylabel("Flux")
                                            if np.all(line_values[np.isfinite(line_values)] > 0):
                                                ax.set_yscale("log")
                                        elif field_name == "velocity":
                                            ax.set_ylabel("Velocity (km/s)")
                                        else:  # dispersion
                                            ax.set_ylabel("Dispersion (km/s)")
                                            
                                        ax.set_title(f"{galaxy_name} - {line_name} {field_name.capitalize()} Profile")
                                        ax.grid(True, alpha=0.3)
                                        
                                        visualization.standardize_figure_saving(
                                            fig, emission_dir / f"{galaxy_name}_RDB_{line_name}_{field_name}_profile.png"
                                        )
                                    else:
                                        logger.warning(f"Dimension mismatch for {line_name} {field_name} profile: distances={len(bin_distances)}, values={len(line_values)}")
                                    
                                    plt.close(fig)
                            except Exception as e:
                                logger.warning(f"Error creating plot for {line_name} {field_name}: {e}")
                                plt.close("all")
            except Exception as e:
                logger.warning(f"Error creating emission line plots: {e}")
                plt.close("all")
                
        # Create spectral indices plots
        if "bin_indices" in rdb_results and "distance" in rdb_results and "bin_distances" in rdb_results["distance"]:
            try:
                bin_indices = rdb_results["bin_indices"]
                bin_distances = rdb_results["distance"]["bin_distances"]
                bin_num = rdb_results["binning"]["bin_num"]
                
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
                                fig, indices_dir / f"{galaxy_name}_RDB_{idx_name}_map.png"
                            )
                            plt.close(fig)
                            
                            # Create radial profile
                            if len(bin_distances) == len(idx_array):
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(bin_distances, idx_array, "o-", markersize=6, linewidth=1.5)
                                ax.set_xlabel("Radius (arcsec)")
                                ax.set_ylabel(f"{idx_name} Value")
                                ax.set_title(f"{galaxy_name} - {idx_name} Radial Profile")
                                ax.grid(True, alpha=0.3)
                                visualization.standardize_figure_saving(
                                    fig, indices_dir / f"{galaxy_name}_RDB_{idx_name}_profile.png"
                                )
                                plt.close(fig)
                            else:
                                logger.warning(f"Dimension mismatch for {idx_name} profile: distances={len(bin_distances)}, values={len(idx_array)}")
                    except Exception as e:
                        logger.warning(f"Error creating plot for index {idx_name}: {e}")
                        plt.close("all")
                
                # Create combined index map
                try:
                    # Get necessary data
                    bin_indices = rdb_results["bin_indices"]
                    bin_num = rdb_results["binning"]["bin_num"]
                    
                    # Only proceed if we have dictionary data with indices
                    if isinstance(bin_indices, dict) and len(bin_indices) > 0:
                        # Create a multi-panel figure for index maps
                        # Limit to 6 indices maximum
                        indices_to_plot = list(bin_indices.keys())[:6]
                        
                        n_indices = len(indices_to_plot)
                        if n_indices > 0:
                            # Calculate grid dimensions
                            if n_indices <= 3:
                                n_cols = n_indices
                                n_rows = 1
                            else:
                                n_cols = 3
                                n_rows = (n_indices + 2) // 3  # Ceiling division
                            
                            # Create figure
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
                            
                            # Handle single axis case
                            if n_indices == 1:
                                axes = np.array([axes])
                            
                            # Flatten axes for easy iteration
                            if n_indices > 1:
                                axes = axes.flatten()
                            
                            # Plot each spectral index
                            for i, idx_name in enumerate(indices_to_plot):
                                if i < len(axes):
                                    # Get index values
                                    idx_values = bin_indices[idx_name]
                                    
                                    try:
                                        # Convert to numpy array
                                        idx_array = np.asarray(idx_values, dtype=float)
                                        
                                        if np.any(np.isfinite(idx_array)):
                                            # Plot bin map with index values
                                            visualization.plot_bin_map(
                                                bin_num_2d,
                                                idx_array,
                                                ax=axes[i],
                                                cmap="plasma",
                                                title=idx_name,
                                                colorbar_label="Index Value",
                                                physical_scale=True,
                                                pixel_size=(cube._pxl_size_x, cube._pxl_size_y),
                                                wcs=cube._wcs if hasattr(cube, "_wcs") else None
                                            )
                                    except Exception as e:
                                        logger.warning(f"Error plotting index {idx_name}: {e}")
                                        axes[i].set_title(f"{idx_name} - Error")
                                        axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                                                 ha='center', va='center', transform=axes[i].transAxes)
                            
                            # Hide any unused subplots
                            for i in range(n_indices, len(axes)):
                                axes[i].axis('off')
                            
                            # Add overall title
                            plt.suptitle(f"{galaxy_name} - Spectral Indices Maps", fontsize=16, y=0.98)
                            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
                            
                            # Save figure
                            visualization.standardize_figure_saving(
                                fig, indices_dir / f"{galaxy_name}_RDB_indices_map.png"
                            )
                            plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error creating combined indices map: {e}")
                    plt.close('all')
                
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
                        n_bins = len(bin_distances)
                        bins_to_plot = []
                        
                        if n_bins <= 10:
                            bins_to_plot = list(range(n_bins))
                        else:
                            # Plot first, middle and last, plus some evenly spaced ones
                            bins_to_plot = [0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1]
                        
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
                                        vel = rdb_results["stellar_kinematics"]["velocity"][bin_idx]
                                        
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
                                            mode="RDB",
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
                title=f"{galaxy_name} - RDB Analysis Overview",
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
                    title=f"{galaxy_name} - RDB Binned Spectra",
                    save_path=spectra_dir / f"{galaxy_name}_RDB_binned_spectra.png"
                )
                plt.close(fig)
                
                # Plot spectra vs radius
                if "binning" in rdb_results and "bin_radii" in rdb_results["binning"]:
                    try:
                        bin_radii = rdb_results["binning"]["bin_radii"]
                        
                        # Create plot showing spectra color-coded by radius
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Use a color map to represent radius
                        cmap = plt.cm.plasma
                        norm = Normalize(vmin=np.min(bin_radii), vmax=np.max(bin_radii))
                        
                        # Plot each spectrum with color based on radius
                        for i in range(min(binned_spectra.shape[1], len(bin_radii))):
                            spectrum = binned_spectra[:, i]
                            radius = bin_radii[i]
                            
                            # Normalize for better visibility
                            norm_factor = np.nanmax(np.abs(spectrum))
                            if norm_factor > 0:
                                normalized = spectrum / norm_factor
                            else:
                                normalized = spectrum
                            
                            # Apply offset based on radius
                            offset = i * 1.2
                            plot_data = normalized + offset
                            
                            # Get color from colormap
                            color = cmap(norm(radius))
                            
                            # Plot with radius in label
                            ax.plot(wavelength, plot_data, color=color, 
                                   label=f'r = {radius:.1f} arcsec')
                        
                        # Add colorbar
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax, label='Radius (arcsec)')
                        
                        # Customize plot
                        ax.set_xlabel('Wavelength (Å)')
                        ax.set_ylabel('Normalized Flux + Offset')
                        ax.set_title(f"{galaxy_name} - RDB Spectra vs. Radius")
                        
                        # Add legend with smaller font only for a few spectra
                        if binned_spectra.shape[1] <= 10:
                            ax.legend(fontsize='small', loc='upper right')
                        
                        ax.grid(True, alpha=0.3)
                        
                        # Save figure
                        visualization.standardize_figure_saving(
                            fig, spectra_dir / f"{galaxy_name}_RDB_spectra_vs_radius.png"
                        )
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating spectra vs radius plot: {e}")
                        plt.close('all')
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
                
                if n_bins <= 10:
                    # Plot all bins if few
                    bins_to_plot = list(range(n_bins))
                else:
                    # Plot selection of bins (first, some middle, last)
                    bins_to_plot = [0, n_bins // 3, 2 * n_bins // 3, n_bins - 1]
                
                # Get bin radii
                bin_radii = None
                if "binning" in rdb_results and "bin_radii" in rdb_results["binning"]:
                    bin_radii = rdb_results["binning"]["bin_radii"]
                
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
                            title = f"{galaxy_name} - RDB Bin {bin_id} Spectral Fit"
                            
                            # Add radius information if available
                            if bin_radii is not None and bin_id < len(bin_radii):
                                title += f" (r = {bin_radii[bin_id]:.1f} arcsec)"
                            
                            # Create plot
                            fig, ax = visualization.plot_ppxf_fit(
                                wavelength, observed, model, residuals, em_component,
                                title=title,
                                redshift=cube._redshift if hasattr(cube, "_redshift") else 0.0,
                                save_path=fit_dir / f"{galaxy_name}_RDB_fit_bin{bin_id}.png"
                            )
                            plt.close(fig)
                        except Exception as e:
                            logger.warning(f"Error creating spectral fit plot for bin {bin_id}: {e}")
                            plt.close('all')
                
                # Create spectral fitting parameter vs. radius plots
                # Include template weights and/or kinematics
                if hasattr(cube, "_optimal_weights") and "binning" in rdb_results:
                    try:
                        optimal_weights = cube._optimal_weights
                        bin_radii = rdb_results["binning"]["bin_radii"]
                        
                        if optimal_weights.shape[1] == len(bin_radii):
                            # Create template weights directory
                            templates_dir = plots_dir / "templates"
                            templates_dir.mkdir(exist_ok=True, parents=True)
                            
                            # Create plot of template weights vs. radius
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Limit to showing a manageable number of templates
                            n_templates = optimal_weights.shape[0]
                            if n_templates > 10:
                                # Select templates with highest weights
                                template_importance = np.sum(optimal_weights, axis=1)
                                top_templates = np.argsort(template_importance)[-10:]
                                weights_to_plot = optimal_weights[top_templates, :]
                                template_labels = [f"Template {i}" for i in top_templates]
                            else:
                                weights_to_plot = optimal_weights
                                template_labels = [f"Template {i}" for i in range(n_templates)]
                            
                            # Plot template weights vs. radius
                            for i in range(weights_to_plot.shape[0]):
                                ax.plot(bin_radii, weights_to_plot[i, :], 'o-', 
                                       label=template_labels[i], markersize=4)
                            
                            ax.set_xlabel('Radius (arcsec)')
                            ax.set_ylabel('Template Weight')
                            ax.set_title(f"{galaxy_name} - Template Weights vs. Radius")
                            ax.grid(True, alpha=0.3)
                            
                            # Use smaller font for legend if many templates
                            if weights_to_plot.shape[0] > 5:
                                ax.legend(fontsize='small', ncol=2)
                            else:
                                ax.legend()
                            
                            # Save figure
                            visualization.standardize_figure_saving(
                                fig, templates_dir / f"{galaxy_name}_RDB_template_weights.png"
                            )
                            plt.close(fig)
                            
                            # Create a 2D "template weight map" (templates vs. radius)
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Create a 2D heatmap of template weights
                            im = ax.imshow(weights_to_plot, aspect='auto', 
                                         extent=[min(bin_radii), max(bin_radii), 
                                                weights_to_plot.shape[0]-0.5, -0.5],
                                         cmap='viridis')
                            
                            # Add colorbar
                            cbar = plt.colorbar(im, ax=ax)
                            cbar.set_label('Template Weight')
                            
                            # Set labels
                            ax.set_xlabel('Radius (arcsec)')
                            ax.set_ylabel('Template Index')
                            ax.set_title(f"{galaxy_name} - Template Weights Map")
                            
                            # Save figure
                            visualization.standardize_figure_saving(
                                fig, templates_dir / f"{galaxy_name}_RDB_template_weights_map.png"
                            )
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating template weight plots: {e}")
                        plt.close('all')
            except Exception as e:
                logger.warning(f"Error creating spectral fit plots: {e}")
                plt.close('all')
        
        # Create a bin overlay plot showing binning on flux map
        try:
            # Get flux map from cube using improved function
            flux_map = visualization.prepare_flux_map(cube)
            
            # Get binning information
            bin_num = rdb_results["binning"]["bin_num"]
            bin_radii = rdb_results["binning"]["bin_radii"]
            center_x = rdb_results["binning"]["center_x"]
            center_y = rdb_results["binning"]["center_y"]
            pa = rdb_results["binning"]["pa"]
            ellipticity = rdb_results["binning"]["ellipticity"]
            
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
            
            # Create binning overlay plot using the improved function
            fig, ax = visualization.plot_bin_boundaries_on_flux(
                bin_num_2d,
                flux_map,
                cube,
                galaxy_name=galaxy_name,
                binning_type="Radial",
                bin_radii=bin_radii,
                center_x=center_x,
                center_y=center_y,
                pa=pa,
                ellipticity=ellipticity,
                save_path=plots_dir / f"{galaxy_name}_RDB_binning_overlay.png"
            )
            
            logger.info(f"Created RDB binning overlay plot")
            
        except Exception as e:
            logger.warning(f"Error creating RDB binning overlay plot: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            plt.close("all")  # Close all figures in case of error
                
    except Exception as e:
        logger.error(f"Error in create_rdb_plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close("all")  # Close all figures in case of error