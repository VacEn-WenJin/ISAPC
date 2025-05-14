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
    Re_arcsec = None
    if use_physical_radius:
        try:
            from physical_radius import calculate_galaxy_radius, calculate_effective_radius
            
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
        metadata["effective_radius"] = Re_arcsec  # Add effective radius
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

    # Extract stellar population parameters
    stellar_pop_params = None
    if hasattr(cube, "_bin_weights") and cube._bin_weights is not None:
        try:
            logger.info("Extracting stellar population parameters for bins...")
            start_time = time.time()

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

            logger.info(f"Stellar population parameters extracted in {time.time() - start_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Failed to extract stellar population parameters: {e}")
            stellar_pop_params = None

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
            
            # print(indices_list)
            # Calculate indices
            indices_result = cube.calculate_spectral_indices(
                indices_list=indices_list,
                n_jobs=args.n_jobs,
                save_mode='RDB',
                save_path=plots_dir,
                verbose=True
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
            "effective_radius": Re_arcsec,  # Add effective radius
            "pixelsize_x": cube._pxl_size_x,
            "pixelsize_y": cube._pxl_size_y,
        },
        "meta_data":{
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
        },
    }
    
    # Add stellar population parameters if available
    if stellar_pop_params is not None:
        rdb_results["stellar_population"] = stellar_pop_params
    
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
    Create visualization plots for RDB analysis with robust dimension handling
    
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
        from matplotlib.colors import Normalize, LogNorm
        from matplotlib.patches import Ellipse
        
        # First, ensure expected cube dimensions are available
        ny, nx = cube._n_y, cube._n_x
        expected_shape = (ny, nx)
        logger.info(f"Expected data shape: {expected_shape}")
        
        # Create flux map with correct dimensions
        flux_map = visualization.prepare_flux_map(cube)
        
        # Ensure flux map has correct dimensions
        if flux_map.shape != expected_shape:
            logger.warning(f"Flux map shape {flux_map.shape} doesn't match expected {expected_shape}")
            # Try to resize flux_map to match expected dimensions
            if flux_map.size > 0:
                from skimage.transform import resize
                try:
                    # Resize with cubic interpolation
                    flux_map = resize(flux_map, expected_shape, order=3, mode='reflect', anti_aliasing=True)
                    logger.info(f"Resized flux map to {flux_map.shape}")
                except Exception as resize_error:
                    logger.error(f"Error resizing flux map: {resize_error}")
                    # Create synthetic map
                    y, x = np.indices(expected_shape)
                    cy, cx = ny // 2, nx // 2
                    r = np.sqrt((x - cx)**2 + (y - cy)**2)
                    flux_map = np.exp(-r / (max(nx, ny) / 4))
            else:
                # Create synthetic map
                y, x = np.indices(expected_shape)
                cy, cx = ny // 2, nx // 2
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                flux_map = np.exp(-r / (max(nx, ny) / 4))
        
        # Create radial bin map
        if "binning" in rdb_results and "bin_num" in rdb_results["binning"]:
            try:
                bin_num = rdb_results["binning"]["bin_num"]
                bin_radii = rdb_results["binning"]["bin_radii"]
                center_x = rdb_results["binning"]["center_x"]
                center_y = rdb_results["binning"]["center_y"]
                pa = rdb_results["binning"]["pa"]
                ellipticity = rdb_results["binning"]["ellipticity"]
                
                # Ensure bin_num has correct dimensions
                bin_num_2d = None
                if isinstance(bin_num, np.ndarray):
                    if bin_num.ndim == 2:
                        # Already 2D - check dimensions
                        if bin_num.shape == expected_shape:
                            bin_num_2d = bin_num
                        else:
                            logger.warning(f"bin_num shape {bin_num.shape} doesn't match expected {expected_shape}")
                            # Try to resize
                            try:
                                from skimage.transform import resize
                                bin_num_2d = resize(bin_num, expected_shape, order=0, preserve_range=True).astype(int)
                            except Exception as e:
                                logger.error(f"Error resizing bin_num: {e}")
                    elif bin_num.ndim == 1:
                        # Check if length matches flattened dimensions
                        if len(bin_num) == ny * nx:
                            # Perfect match - reshape
                            bin_num_2d = bin_num.reshape(expected_shape)
                        elif len(bin_num) < ny * nx:
                            # Partial data - create full array and fill
                            bin_num_2d = np.zeros(expected_shape, dtype=int)
                            bin_num_2d.flat[:len(bin_num)] = bin_num
                        else:
                            # Too much data - truncate
                            bin_num_2d = bin_num[:ny*nx].reshape(expected_shape)
                
                # If bin_num_2d is still None, create default
                if bin_num_2d is None:
                    logger.warning("Creating default bin_num_2d")
                    bin_num_2d = np.zeros(expected_shape, dtype=int)
                
                # Sort bin_radii for contour levels
                sorted_bin_radii = np.sort(bin_radii)
                
                # 1. Create simplified bin map
                # ---------------------------
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Use physical coordinates consistently
                physical_scale = True
                pixel_size = (cube._pxl_size_x, cube._pxl_size_y)
                
                # Use a simpler visualization approach
                im = ax.imshow(bin_num_2d, origin='lower', cmap='tab20')
                plt.colorbar(im, ax=ax, label='Bin Number')
                
                ax.set_title(f"{galaxy_name} - Radial Bins")
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Pixels')
                
                # Draw radial bins as circles/ellipses
                try:
                    # Attempt to add ellipses
                    for radius in sorted_bin_radii:
                        # Convert radius to pixels for proper scaling
                        radius_pix = radius / pixel_size[0]
                        
                        # Create ellipse
                        ell = Ellipse(
                            (center_x, center_y),
                            2 * radius_pix,  # Major axis diameter
                            2 * radius_pix * (1 - ellipticity),  # Minor axis diameter
                            angle=pa,
                            fill=False,
                            edgecolor='white',
                            linestyle='-',
                            linewidth=1.0
                        )
                        ax.add_patch(ell)
                except Exception as e:
                    logger.warning(f"Error drawing ellipses: {e}")
                
                ax.grid(False)
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_bin_map.png"
                )
                plt.close(fig)
                
                # 2. Create physical radius map with contours
                # -----------------------------------------
                try:
                    # Generate physical radius map
                    if hasattr(cube, "_physical_radius") and cube._physical_radius is not None:
                        r_galaxy = cube._physical_radius
                        ellipse_params = cube._ellipse_params if hasattr(cube, "_ellipse_params") else None
                    else:
                        # Calculate physical radius
                        from physical_radius import calculate_galaxy_radius
                        r_galaxy, ellipse_params = calculate_galaxy_radius(
                            flux_map,
                            pixel_size_x=cube._pxl_size_x, 
                            pixel_size_y=cube._pxl_size_y
                        )
                    
                    # Check dimensions
                    if r_galaxy.shape != expected_shape:
                        logger.warning(f"Physical radius shape {r_galaxy.shape} doesn't match expected {expected_shape}")
                        # Try to resize
                        if r_galaxy.size > 0:
                            from skimage.transform import resize
                            try:
                                r_galaxy = resize(r_galaxy, expected_shape, order=1, mode='reflect', anti_aliasing=True)
                                logger.info(f"Resized physical radius to {r_galaxy.shape}")
                            except Exception as e:
                                logger.error(f"Error resizing physical radius: {e}")
                                # Create synthetic radius
                                y, x = np.indices(expected_shape)
                                cy, cx = ny // 2, nx // 2
                                r_galaxy = np.sqrt((x - cx)**2 + (y - cy)**2) * pixel_size[0]
                                
                    # Create figure for physical radius contours
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot using physical coordinates
                    extent = [
                        -nx/2 * pixel_size[0],
                        nx/2 * pixel_size[0],
                        -ny/2 * pixel_size[1],
                        ny/2 * pixel_size[1]
                    ]
                    
                    # Plot flux map as background with log scale
                    valid_flux = flux_map[np.isfinite(flux_map) & (flux_map > 0)]
                    if len(valid_flux) > 0:
                        vmin = np.percentile(valid_flux, 1)
                        vmax = np.percentile(valid_flux, 99)
                        norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
                    else:
                        norm = None
                    
                    im = ax.imshow(
                        flux_map,
                        origin='lower',
                        cmap='inferno',
                        norm=norm,
                        extent=extent,
                        aspect='equal'
                    )
                    plt.colorbar(im, ax=ax, label='Flux')
                    
                    # Make sure radii are sorted and unique for contours
                    contour_levels = np.unique(sorted_bin_radii)
                    
                    # Generate contours with consistent colormap
                    try:
                        contour = ax.contour(
                            np.linspace(extent[0], extent[1], nx),
                            np.linspace(extent[2], extent[3], ny),
                            r_galaxy,
                            levels=contour_levels,
                            colors='white',
                            linewidths=1.5,
                            alpha=0.7
                        )
                        
                        # Add contour labels
                        plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
                    except Exception as e:
                        logger.warning(f"Error creating contours: {e}")
                    
                    # Add ellipses for bin edges
                    for radius in sorted_bin_radii:
                        # Convert center to physical coordinates
                        center_x_phys = (center_x - nx/2) * pixel_size[0]
                        center_y_phys = (center_y - ny/2) * pixel_size[1]
                        
                        # Create ellipse
                        ell = Ellipse(
                            (center_x_phys, center_y_phys),
                            2 * radius,  # Diameter
                            2 * radius * (1 - ellipticity),  # Account for ellipticity
                            angle=pa,
                            fill=False,
                            edgecolor='yellow',
                            linestyle='--',
                            linewidth=1.0,
                            alpha=0.6
                        )
                        ax.add_patch(ell)
                    
                    # Set labels and title
                    ax.set_xlabel('Δ RA (arcsec)')
                    ax.set_ylabel('Δ Dec (arcsec)')
                    ax.set_title(f'{galaxy_name} - Physical Radius Contours')
                    
                    # Add center marker
                    center_x_phys = (center_x - nx/2) * pixel_size[0]
                    center_y_phys = (center_y - ny/2) * pixel_size[1]
                    ax.plot(center_x_phys, center_y_phys, '+', color='yellow', markersize=10)
                    
                    # Save figure
                    visualization.standardize_figure_saving(
                        fig, plots_dir / f"{galaxy_name}_RDB_physical_coords.png"
                    )
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error creating physical radius contour plot: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    plt.close("all")
                
                # 3. Create radial profile plot
                # ----------------------------
                if len(sorted_bin_radii) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(np.arange(len(sorted_bin_radii)), sorted_bin_radii, "o-", markersize=6, linewidth=1.5)
                    ax.set_xlabel("Bin Number")
                    ax.set_ylabel("Radius (arcsec)")
                    ax.set_title(f"{galaxy_name} - Radial Bin Distribution")
                    ax.grid(True, alpha=0.3)
                    visualization.standardize_figure_saving(
                        fig, plots_dir / f"{galaxy_name}_RDB_bin_distances.png"
                    )
                    plt.close(fig)
                
            except Exception as e:
                logger.warning(f"Error creating radial bin map: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                plt.close("all")
                
        # Create effective radius visualization
        if "distance" in rdb_results and "effective_radius" in rdb_results["distance"]:
            try:
                Re = rdb_results["distance"]["effective_radius"]
                
                if Re is not None and Re > 0:
                    # Get ellipse parameters if available
                    ellipse_params = None
                    if "meta_data" in rdb_results and "ellipse_params" in rdb_results["meta_data"]:
                        ellipse_params = rdb_results["meta_data"]["ellipse_params"]
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
                        fig, plots_dir / f"{galaxy_name}_RDB_effective_radius.png"
                    )
                    plt.close(fig)
                
            except Exception as e:
                logger.warning(f"Error creating effective radius visualization: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                plt.close("all")
        
        # Create kinematics plots
        if "stellar_kinematics" in rdb_results and "distance" in rdb_results:
            try:
                # Get bin-based data
                velocity = np.asarray(rdb_results["stellar_kinematics"]["velocity"])
                dispersion = np.asarray(rdb_results["stellar_kinematics"]["dispersion"])
                
                if "bin_distances" in rdb_results["distance"]:
                    bin_distances = np.asarray(rdb_results["distance"]["bin_distances"])
                else:
                    # Create sequential distances if needed
                    bin_distances = np.arange(len(velocity))
                
                # Get binning info
                bin_num = rdb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    # Make sure the dimensions match the cube
                    if len(bin_num) == nx * ny:
                        bin_num_2d = bin_num.reshape(ny, nx)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({ny}x{nx})")
                        # Create a dummy bin_num_2d with correct dimensions
                        bin_num_2d = np.zeros((ny, nx), dtype=int)
                        # Fill with what we can
                        valid_len = min(len(bin_num), ny * nx)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
                else:
                    bin_num_2d = bin_num
                
                # Create radial velocity profile
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Ensure dimensions match for plotting
                valid_distances = bin_distances
                valid_velocity = velocity
                
                if len(valid_distances) != len(valid_velocity):
                    logger.warning(f"Dimension mismatch: distances={len(valid_distances)}, velocity={len(valid_velocity)}")
                    # Find common length
                    min_len = min(len(valid_distances), len(valid_velocity))
                    valid_distances = valid_distances[:min_len]
                    valid_velocity = valid_velocity[:min_len]
                
                ax.plot(valid_distances, valid_velocity, "o-", markersize=6, linewidth=1.5)
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
                        
                        # Handle fit_radius
                        if "fit_radius" in rdb_results["global_kinematics"]["rotation_curve"]:
                            fit_radius = rdb_results["global_kinematics"]["rotation_curve"]["fit_radius"]
                        else:
                            # Use distances if possible, or create linear space
                            if len(fit) == len(valid_distances):
                                fit_radius = valid_distances
                            else:
                                fit_radius = np.linspace(np.min(valid_distances), np.max(valid_distances), len(fit))
                        
                        # Check dimensions match
                        if len(fit) == len(fit_radius):
                            ax.plot(fit_radius, fit, "r-", lw=2, alpha=0.7, label="Rotation Curve Fit")
                            ax.legend()
                        else:
                            logger.warning(f"Rotation curve dimensions don't match: fit={len(fit)}, radius={len(fit_radius)}")
                    except Exception as e:
                        logger.warning(f"Error plotting rotation curve fit: {e}")
                
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_velocity_profile.png"
                )
                plt.close(fig)
                
                # Create velocity profile in Re units if effective radius is available
                if "effective_radius" in rdb_results["distance"]:
                    try:
                        Re = rdb_results["distance"]["effective_radius"]
                        
                        if Re is not None and Re > 0:
                            # Create a version of distances in Re units
                            distances_in_Re = valid_distances / Re
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(distances_in_Re, valid_velocity, "o-", markersize=6, linewidth=1.5)
                            ax.set_xlabel("R/Re")
                            ax.set_ylabel("Velocity (km/s)")
                            ax.set_title(f"{galaxy_name} - Radial Velocity Profile (Re = {Re:.2f} arcsec)")
                            ax.grid(True, alpha=0.3)
                            
                            # Add vertical line at Re
                            ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                            ax.legend()
                            
                            # Add rotation curve fit if available
                            if (
                                "global_kinematics" in rdb_results 
                                and "rotation_curve" in rdb_results["global_kinematics"]
                                and "fit" in rdb_results["global_kinematics"]["rotation_curve"]
                            ):
                                try:
                                    fit = rdb_results["global_kinematics"]["rotation_curve"]["fit"]
                                    
                                    # Handle fit_radius and convert to Re units
                                    if "fit_radius" in rdb_results["global_kinematics"]["rotation_curve"]:
                                        fit_radius = rdb_results["global_kinematics"]["rotation_curve"]["fit_radius"]
                                        fit_radius_Re = fit_radius / Re
                                    else:
                                        # Use distances if possible, or create linear space
                                        if len(fit) == len(distances_in_Re):
                                            fit_radius_Re = distances_in_Re
                                        else:
                                            fit_radius_Re = np.linspace(np.min(distances_in_Re), np.max(distances_in_Re), len(fit))
                                    
                                    # Check dimensions match
                                    if len(fit) == len(fit_radius_Re):
                                        ax.plot(fit_radius_Re, fit, "r-", lw=2, alpha=0.7, label="Rotation Curve Fit")
                                        ax.legend()
                                except Exception as e:
                                    logger.warning(f"Error plotting rotation curve fit in Re units: {e}")
                            
                            visualization.standardize_figure_saving(
                                fig, plots_dir / f"{galaxy_name}_RDB_velocity_vs_Re.png"
                            )
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating velocity profile in Re units: {e}")
                        plt.close('all')
                
                # Create radial dispersion profile
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Ensure dimensions match for plotting
                valid_dispersion = dispersion
                if len(valid_distances) != len(valid_dispersion):
                    logger.warning(f"Dimension mismatch: distances={len(valid_distances)}, dispersion={len(valid_dispersion)}")
                    # Find common length
                    min_len = min(len(valid_distances), len(valid_dispersion))
                    valid_distances = valid_distances[:min_len]
                    valid_dispersion = valid_dispersion[:min_len]
                
                ax.plot(valid_distances, valid_dispersion, "o-", markersize=6, linewidth=1.5)
                ax.set_xlabel("Radius (arcsec)")
                ax.set_ylabel("Dispersion (km/s)")
                ax.set_title(f"{galaxy_name} - Radial Dispersion Profile")
                ax.grid(True, alpha=0.3)
                visualization.standardize_figure_saving(
                    fig, plots_dir / f"{galaxy_name}_RDB_dispersion_profile.png"
                )
                plt.close(fig)
                
                # Create dispersion profile in Re units if effective radius is available
                if "effective_radius" in rdb_results["distance"]:
                    try:
                        Re = rdb_results["distance"]["effective_radius"]
                        
                        if Re is not None and Re > 0:
                            # Create a version of distances in Re units
                            distances_in_Re = valid_distances / Re
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(distances_in_Re, valid_dispersion, "o-", markersize=6, linewidth=1.5)
                            ax.set_xlabel("R/Re")
                            ax.set_ylabel("Dispersion (km/s)")
                            ax.set_title(f"{galaxy_name} - Radial Dispersion Profile (Re = {Re:.2f} arcsec)")
                            ax.grid(True, alpha=0.3)
                            
                            # Add vertical line at Re
                            ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                            ax.legend()
                            
                            visualization.standardize_figure_saving(
                                fig, plots_dir / f"{galaxy_name}_RDB_dispersion_vs_Re.png"
                            )
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating dispersion profile in Re units: {e}")
                        plt.close('all')
                
                # Create combined kinematics profile
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot velocity profile
                axes[0].plot(valid_distances, valid_velocity, "o-", markersize=6, linewidth=1.5)
                axes[0].set_xlabel("Radius (arcsec)")
                axes[0].set_ylabel("Velocity (km/s)")
                axes[0].set_title("Stellar Velocity Profile")
                axes[0].grid(True, alpha=0.3)
                
                # Plot dispersion profile
                axes[1].plot(valid_distances, valid_dispersion, "o-", markersize=6, linewidth=1.5)
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
                if len(valid_velocity) == len(valid_dispersion) and np.all(valid_dispersion > 0):
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        v_sigma = np.abs(valid_velocity) / valid_dispersion
                        ax.plot(valid_distances, v_sigma, "o-", markersize=6, linewidth=1.5)
                        ax.set_xlabel("Radius (arcsec)")
                        ax.set_ylabel("|V|/σ")
                        ax.set_title(f"{galaxy_name} - V/σ Radial Profile")
                        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
                        ax.grid(True, alpha=0.3)
                        visualization.standardize_figure_saving(
                            fig, plots_dir / f"{galaxy_name}_RDB_vsigma_profile.png"
                        )
                        plt.close(fig)
                        
                        # Create V/σ profile in Re units if effective radius is available
                        if "effective_radius" in rdb_results["distance"]:
                            try:
                                Re = rdb_results["distance"]["effective_radius"]
                                
                                if Re is not None and Re > 0:
                                    # Create a version of distances in Re units
                                    distances_in_Re = valid_distances / Re
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(distances_in_Re, v_sigma, "o-", markersize=6, linewidth=1.5)
                                    ax.set_xlabel("R/Re")
                                    ax.set_ylabel("|V|/σ")
                                    ax.set_title(f"{galaxy_name} - V/σ Radial Profile (Re = {Re:.2f} arcsec)")
                                    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
                                    ax.grid(True, alpha=0.3)
                                    
                                    # Add vertical line at Re
                                    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                    ax.legend()
                                    
                                    visualization.standardize_figure_saving(
                                        fig, plots_dir / f"{galaxy_name}_RDB_vsigma_vs_Re.png"
                                    )
                                    plt.close(fig)
                            except Exception as e:
                                logger.warning(f"Error creating V/σ profile in Re units: {e}")
                                plt.close('all')
                    except Exception as e:
                        logger.warning(f"Error creating V/σ profile: {e}")
                        plt.close('all')
                
                # Create velocity map
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    visualization.plot_bin_map(
                        bin_num_2d, 
                        velocity, 
                        ax=ax,
                        cmap="coolwarm",
                        title=f"{galaxy_name} - RDB Velocity",
                        colorbar_label="Velocity (km/s)",
                        physical_scale=True,
                        pixel_size=pixel_size,
                        wcs=cube._wcs if hasattr(cube, "_wcs") else None
                    )
                    visualization.standardize_figure_saving(
                        fig, plots_dir / f"{galaxy_name}_RDB_velocity_map.png"
                    )
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error creating velocity map: {e}")
                    plt.close('all')
                
                # Create dispersion map
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    visualization.plot_bin_map(
                        bin_num_2d, 
                        dispersion, 
                        ax=ax,
                        cmap="viridis",
                        title=f"{galaxy_name} - RDB Dispersion",
                        colorbar_label="Dispersion (km/s)",
                        physical_scale=True,
                        pixel_size=pixel_size,
                        wcs=cube._wcs if hasattr(cube, "_wcs") else None
                    )
                    visualization.standardize_figure_saving(
                        fig, plots_dir / f"{galaxy_name}_RDB_dispersion_map.png"
                    )
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error creating dispersion map: {e}")
                    plt.close('all')
                    
                # Create kinematics summary
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
                        rotation_curve=rdb_results["global_kinematics"]["rotation_curve"] if "global_kinematics" in rdb_results else None,
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
                import traceback
                logger.debug(traceback.format_exc())
                plt.close("all")
                
        # Create stellar population plots if available
        if "stellar_population" in rdb_results:
            try:
                # Create directory for stellar population plots
                stellar_dir = plots_dir / "stellar_population"
                stellar_dir.mkdir(exist_ok=True, parents=True)
                
                bin_num = rdb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    if len(bin_num) == nx * ny:
                        bin_num_2d = bin_num.reshape(ny, nx)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({ny}x{nx})")
                        bin_num_2d = np.zeros((ny, nx), dtype=int)
                        valid_len = min(len(bin_num), ny * nx)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
                else:
                    bin_num_2d = bin_num
                
                # Process each parameter
                for param_name, param_values in rdb_results["stellar_population"].items():
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
                        fig, stellar_dir / f"{galaxy_name}_RDB_{param_name}.png"
                    )
                    plt.close(fig)
                    
                    # Create radial profile if distances available
                    if "distance" in rdb_results and "bin_distances" in rdb_results["distance"]:
                        try:
                            bin_distances = rdb_results["distance"]["bin_distances"]
                            
                            # Ensure dimensions match for plotting
                            if len(bin_distances) != len(values_to_plot):
                                logger.warning(f"Dimension mismatch: distances={len(bin_distances)}, {param_name}={len(values_to_plot)}")
                                # Find common length
                                min_len = min(len(bin_distances), len(values_to_plot))
                                valid_distances = bin_distances[:min_len]
                                valid_values = values_to_plot[:min_len]
                            else:
                                valid_distances = bin_distances
                                valid_values = values_to_plot
                            
                            # Create profile plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(valid_distances, valid_values, 'o-', markersize=6, linewidth=1.5)
                            ax.set_xlabel('Radius (arcsec)')
                            ax.set_ylabel(info["title"])
                            ax.set_title(f'{galaxy_name} - Stellar {info["title"]} Profile')
                            ax.grid(True, alpha=0.3)
                            
                            visualization.standardize_figure_saving(
                                fig, stellar_dir / f"{galaxy_name}_RDB_{param_name}_profile.png"
                            )
                            plt.close(fig)
                            
                            # Create profile in Re units if effective radius is available
                            if "effective_radius" in rdb_results["distance"]:
                                try:
                                    Re = rdb_results["distance"]["effective_radius"]
                                    
                                    if Re is not None and Re > 0:
                                        # Create a version of distances in Re units
                                        distances_in_Re = valid_distances / Re
                                        
                                        # Create profile plot in Re units
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.plot(distances_in_Re, valid_values, 'o-', markersize=6, linewidth=1.5)
                                        ax.set_xlabel('R/Re')
                                        ax.set_ylabel(info["title"])
                                        ax.set_title(f'{galaxy_name} - Stellar {info["title"]} Profile (Re = {Re:.2f} arcsec)')
                                        ax.grid(True, alpha=0.3)
                                        
                                        # Add vertical line at Re
                                        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                        ax.legend()
                                        
                                        visualization.standardize_figure_saving(
                                            fig, stellar_dir / f"{galaxy_name}_RDB_{param_name}_vs_Re.png"
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
        if "emission" in rdb_results and "distance" in rdb_results:
            try:
                emission = rdb_results["emission"]
                
                # Get bin distances
                if "bin_distances" in rdb_results["distance"]:
                    bin_distances = np.asarray(rdb_results["distance"]["bin_distances"])
                else:
                    # Create placeholder distances
                    bin_distances = np.arange(10)  # Default value
                
                # Get bin_num for spatial maps
                bin_num = rdb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    if len(bin_num) == nx * ny:
                        bin_num_2d = bin_num.reshape(ny, nx)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({ny}x{nx})")
                        bin_num_2d = np.zeros((ny, nx), dtype=int)
                        valid_len = min(len(bin_num), ny * nx)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
                else:
                    bin_num_2d = bin_num
                
                # Create emission line plots directory
                emission_dir = plots_dir / "emission_lines"
                emission_dir.mkdir(exist_ok=True, parents=True)
                    
                # Process each emission line field
                for field_name, field_data in emission.items():
                    # Skip non-dictionary data
                    if not isinstance(field_data, dict):
                        continue
                        
                    for line_name, line_values in field_data.items():
                        try:
                            # Make sure we have numeric values
                            try:
                                # Convert to numpy array of floats
                                line_array = np.asarray(line_values, dtype=float)
                            except (ValueError, TypeError):
                                # Skip if conversion fails
                                logger.warning(f"Could not convert {line_name} {field_name} to numeric values")
                                continue
                                
                            if np.any(np.isfinite(line_array)):
                                # Create spatial map
                                try:
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
                                        line_array,
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
                                except Exception as e:
                                    logger.warning(f"Error creating map for {line_name} {field_name}: {e}")
                                    plt.close('all')
                                
                                # Create radial profile only
                                # Ensure dimensions match
                                if len(bin_distances) != len(line_array):
                                    # Truncate to common length
                                    min_len = min(len(bin_distances), len(line_array))
                                    valid_distances = bin_distances[:min_len]
                                    valid_values = line_array[:min_len]
                                else:
                                    valid_distances = bin_distances
                                    valid_values = line_array
                                    
                                # Create radial profile
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(valid_distances, valid_values, "o-", markersize=6, linewidth=1.5)
                                ax.set_xlabel("Radius (arcsec)")
                                
                                # Set y-axis based on field type
                                if field_name == "flux":
                                    ax.set_ylabel("Flux")
                                    if np.all(valid_values[np.isfinite(valid_values)] > 0):
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
                                plt.close(fig)
                                
                                # Create profile in Re units if effective radius is available
                                if "effective_radius" in rdb_results["distance"]:
                                    try:
                                        Re = rdb_results["distance"]["effective_radius"]
                                        
                                        if Re is not None and Re > 0:
                                            # Create a version of distances in Re units
                                            distances_in_Re = valid_distances / Re
                                            
                                            # Create profile plot in Re units
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            ax.plot(distances_in_Re, valid_values, "o-", markersize=6, linewidth=1.5)
                                            ax.set_xlabel("R/Re")
                                            
                                            # Set y-axis based on field type
                                            if field_name == "flux":
                                                ax.set_ylabel("Flux")
                                                if np.all(valid_values[np.isfinite(valid_values)] > 0):
                                                    ax.set_yscale("log")
                                            elif field_name == "velocity":
                                                ax.set_ylabel("Velocity (km/s)")
                                            else:  # dispersion
                                                ax.set_ylabel("Dispersion (km/s)")
                                                
                                            ax.set_title(f"{galaxy_name} - {line_name} {field_name.capitalize()} Profile (Re = {Re:.2f} arcsec)")
                                            ax.grid(True, alpha=0.3)
                                            
                                            # Add vertical line at Re
                                            ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                            ax.legend()
                                            
                                            visualization.standardize_figure_saving(
                                                fig, emission_dir / f"{galaxy_name}_RDB_{line_name}_{field_name}_vs_Re.png"
                                            )
                                            plt.close(fig)
                                    except Exception as e:
                                        logger.warning(f"Error creating profile in Re units for {line_name} {field_name}: {e}")
                                        plt.close('all')
                        except Exception as e:
                            logger.warning(f"Error creating plot for {line_name} {field_name}: {e}")
                            plt.close('all')
            except Exception as e:
                logger.warning(f"Error creating emission line plots: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                plt.close("all")
                
        # Handle spectral indices
        if "bin_indices" in rdb_results:
            try:
                bin_indices = rdb_results["bin_indices"]
                
                # Get bin distances if available
                if "distance" in rdb_results and "bin_distances" in rdb_results["distance"]:
                    bin_distances = np.asarray(rdb_results["distance"]["bin_distances"])
                else:
                    # Create placeholder distances
                    bin_distances = np.arange(10)  # Default value
                
                # Get bin_num for spatial maps
                bin_num = rdb_results["binning"]["bin_num"]
                
                # Reshape to 2D for plotting if needed
                if isinstance(bin_num, np.ndarray) and bin_num.ndim == 1:
                    if len(bin_num) == nx * ny:
                        bin_num_2d = bin_num.reshape(ny, nx)
                    else:
                        # Handle case where bin_num length doesn't match cube dimensions
                        logger.warning(f"bin_num length ({len(bin_num)}) doesn't match cube dimensions ({ny}x{nx})")
                        bin_num_2d = np.zeros((ny, nx), dtype=int)
                        valid_len = min(len(bin_num), ny * nx)
                        bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
                else:
                    bin_num_2d = bin_num
                
                # Create spectral indices directory
                indices_dir = plots_dir / "spectral_indices"
                indices_dir.mkdir(exist_ok=True, parents=True)
                
                # Different strategies to handle bin_indices based on type
                try:
                    if isinstance(bin_indices, dict):
                        # Process each key-value pair in the dictionary
                        for idx_name, idx_values in bin_indices.items():
                            try:
                                # Skip if not a numeric array
                                if isinstance(idx_values, (dict, np.lib.npyio.NpzFile)):
                                    logger.warning(f"Skipping nested dictionary for index {idx_name}")
                                    continue
                                    
                                # Convert to numpy array
                                try:
                                    idx_array = np.asarray(idx_values, dtype=float)
                                except (ValueError, TypeError):
                                    logger.warning(f"Could not convert {idx_name} to numeric array")
                                    continue
                                
                                # Create radial profile if array is valid
                                if np.any(np.isfinite(idx_array)):
                                    # Create spatial map
                                    try:
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
                                    except Exception as e:
                                        logger.warning(f"Error creating map for index {idx_name}: {e}")
                                        plt.close('all')
                                    
                                    # Ensure lengths match
                                    if len(bin_distances) != len(idx_array):
                                        # Truncate to common length
                                        min_len = min(len(bin_distances), len(idx_array))
                                        valid_distances = bin_distances[:min_len]
                                        valid_values = idx_array[:min_len]
                                    else:
                                        valid_distances = bin_distances
                                        valid_values = idx_array
                                    
                                    # Create radial profile
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(valid_distances, valid_values, "o-", markersize=6, linewidth=1.5)
                                    ax.set_xlabel("Radius (arcsec)")
                                    ax.set_ylabel(f"{idx_name} Value")
                                    ax.set_title(f"{galaxy_name} - {idx_name} Radial Profile")
                                    ax.grid(True, alpha=0.3)
                                    
                                    visualization.standardize_figure_saving(
                                        fig, indices_dir / f"{galaxy_name}_RDB_{idx_name}_profile.png"
                                    )
                                    plt.close(fig)
                                    
                                    # Create profile in Re units if effective radius is available
                                    if "effective_radius" in rdb_results["distance"]:
                                        try:
                                            Re = rdb_results["distance"]["effective_radius"]
                                            
                                            if Re is not None and Re > 0:
                                                # Create a version of distances in Re units
                                                distances_in_Re = valid_distances / Re
                                                
                                                # Create profile plot in Re units
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                ax.plot(distances_in_Re, valid_values, "o-", markersize=6, linewidth=1.5)
                                                ax.set_xlabel("R/Re")
                                                ax.set_ylabel(f"{idx_name} Value")
                                                ax.set_title(f"{galaxy_name} - {idx_name} Profile (Re = {Re:.2f} arcsec)")
                                                ax.grid(True, alpha=0.3)
                                                
                                                # Add vertical line at Re
                                                ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                                ax.legend()
                                                
                                                visualization.standardize_figure_saving(
                                                    fig, indices_dir / f"{galaxy_name}_RDB_{idx_name}_vs_Re.png"
                                                )
                                                plt.close(fig)
                                        except Exception as e:
                                            logger.warning(f"Error creating profile in Re units for index {idx_name}: {e}")
                                            plt.close('all')
                            except Exception as e:
                                logger.warning(f"Error processing index {idx_name}: {e}")
                                plt.close('all')
                    elif isinstance(bin_indices, np.ndarray) and bin_indices.dtype == np.dtype('O'):
                        # Handle object array
                        for i, val in enumerate(bin_indices):
                            if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
                                # Create synthetic name
                                idx_name = f"index_{i}"
                                
                                # Create spatial map
                                try:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    visualization.plot_bin_map(
                                        bin_num_2d,
                                        val,
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
                                except Exception as e:
                                    logger.warning(f"Error creating map for index {idx_name}: {e}")
                                    plt.close('all')
                                
                                # Ensure lengths match
                                if len(bin_distances) != len(val):
                                    # Truncate to common length
                                    min_len = min(len(bin_distances), len(val))
                                    valid_distances = bin_distances[:min_len]
                                    valid_values = val[:min_len]
                                else:
                                    valid_distances = bin_distances
                                    valid_values = val
                                
                                # Create radial profile
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(valid_distances, valid_values, "o-", markersize=6, linewidth=1.5)
                                ax.set_xlabel("Radius (arcsec)")
                                ax.set_ylabel(f"{idx_name} Value")
                                ax.set_title(f"{galaxy_name} - {idx_name} Radial Profile")
                                ax.grid(True, alpha=0.3)
                                
                                visualization.standardize_figure_saving(
                                    fig, indices_dir / f"{galaxy_name}_RDB_{idx_name}_profile.png"
                                )
                                plt.close(fig)
                                
                                # Create profile in Re units if effective radius is available
                                if "effective_radius" in rdb_results["distance"]:
                                    try:
                                        Re = rdb_results["distance"]["effective_radius"]
                                        
                                        if Re is not None and Re > 0:
                                            # Create a version of distances in Re units
                                            distances_in_Re = valid_distances / Re
                                            
                                            # Create profile plot in Re units
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            ax.plot(distances_in_Re, valid_values, "o-", markersize=6, linewidth=1.5)
                                            ax.set_xlabel("R/Re")
                                            ax.set_ylabel(f"{idx_name} Value")
                                            ax.set_title(f"{galaxy_name} - {idx_name} Profile (Re = {Re:.2f} arcsec)")
                                            ax.grid(True, alpha=0.3)
                                            
                                            # Add vertical line at Re
                                            ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Re')
                                            ax.legend()
                                            
                                            visualization.standardize_figure_saving(
                                                fig, indices_dir / f"{galaxy_name}_RDB_{idx_name}_vs_Re.png"
                                            )
                                            plt.close(fig)
                                    except Exception as e:
                                        logger.warning(f"Error creating profile in Re units for index {idx_name}: {e}")
                                        plt.close('all')
                except Exception as e:
                    logger.warning(f"Error processing bin_indices: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
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
                        
                        # Get binned data
                        if hasattr(cube, "_binned_data") and cube._binned_data is not None:
                            binned_spectra = cube._binned_data.spectra
                            wavelength = cube._binned_data.wavelength
                            
                            # Select a subset of bins to plot
                            n_bins = binned_spectra.shape[1]
                            bins_to_plot = []
                            
                            if n_bins <= 10:
                                bins_to_plot = list(range(n_bins))
                            else:
                                # Plot first, middle and last
                                bins_to_plot = [0, n_bins // 2, n_bins - 1]
                            
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
        
        # Create sample spectra plots if available
        if hasattr(cube, "_binned_data") and cube._binned_data is not None:
            try:
                # Create binned spectra visualization directory
                spectra_dir = plots_dir / "spectra"
                spectra_dir.mkdir(exist_ok=True, parents=True)
                
                # Get binned data
                binned_spectra = cube._binned_data.spectra
                wavelength = cube._binned_data.wavelength
                
                # Create a simple plot of sample binned spectra
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot 5 spectra at most
                n_bins = binned_spectra.shape[1]
                samples = n_bins if n_bins <= 5 else 5
                
                # Choose evenly spaced bin indices
                indices = np.linspace(0, n_bins-1, samples).astype(int)
                
                for i, bin_idx in enumerate(indices):
                    if bin_idx < n_bins:
                        spectrum = binned_spectra[:, bin_idx]
                        
                        # Normalize spectrum for better visualization
                        norm_factor = np.nanmax(np.abs(spectrum))
                        if norm_factor > 0:
                            normalized = spectrum / norm_factor
                        else:
                            normalized = spectrum
                            
                        # Apply offset for clarity
                        offset = i * 1.2
                        ax.plot(wavelength, normalized + offset, label=f'Bin {bin_idx}')
                
                ax.set_xlabel('Wavelength (Å)')
                ax.set_ylabel('Normalized Flux + Offset')
                ax.set_title(f"{galaxy_name} - Sample RDB Binned Spectra")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                visualization.standardize_figure_saving(
                    fig, spectra_dir / f"{galaxy_name}_RDB_sample_spectra.png"
                )
                plt.close(fig)
                
            except Exception as e:
                logger.warning(f"Error creating sample spectra plots: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                plt.close('all')
    
    except Exception as e:
        logger.error(f"Error in create_rdb_plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close("all")  # Close all figures in case of error