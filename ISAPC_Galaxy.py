#!/usr/bin/env python3
"""
ISAPC Galaxy Analysis Module
Automates galaxy analysis with three modes: P2P, VNB, and RDB
"""

import argparse
import logging
import os
import sys
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import ISAPC modules
from analysis.p2p import run_p2p_analysis
from analysis.radial import run_rdb_analysis
from analysis.voronoi import run_voronoi_analysis
from utils.io import save_standardized_results, load_standardized_results, load_all_galaxy_results, create_summary_metadata
from visualization import plot_physical_radius, plot_normalized_radius, standardize_figure_saving
from muse import MUSECube

# Predefined list of VCC galaxies with metadata
GALAXIES = [
    {"name": "VCC0308", "redshift": 0.0055, "type": "dE", "has_emission": True},
    {"name": "VCC0667", "redshift": 0.0048, "type": "Sd", "has_emission": True},
    {"name": "VCC0688", "redshift": 0.0038, "type": "Sc", "has_emission": True},
    {"name": "VCC0990", "redshift": 0.0058, "type": "dS0(N)", "has_emission": True},
    {"name": "VCC1049", "redshift": 0.0021, "type": "dE(N)", "has_emission": True},
    {"name": "VCC1146", "redshift": 0.0023, "type": "E", "has_emission": True},
    {"name": "VCC1193", "redshift": 0.0025, "type": "Sd", "has_emission": True},
    {"name": "VCC1368", "redshift": 0.0035, "type": "SBa", "has_emission": True},
    {"name": "VCC1410", "redshift": 0.0054, "type": "Sd", "has_emission": True},
    {"name": "VCC1431", "redshift": 0.0050, "type": "dE", "has_emission": True},
    {"name": "VCC1486", "redshift": 0.0004, "type": "Sc", "has_emission": True},
    {"name": "VCC1499", "redshift": 0.0055, "type": "dE", "has_emission": True},
    {"name": "VCC1549", "redshift": 0.0046, "type": "dE(N)", "has_emission": False},
    {"name": "VCC1588", "redshift": 0.0042, "type": "Sd", "has_emission": True},
    {"name": "VCC1695", "redshift": 0.0058, "type": "dE", "has_emission": True},
    {"name": "VCC1811", "redshift": 0.0023, "type": "Sc", "has_emission": True},
    {"name": "VCC1890", "redshift": 0.0040, "type": "dE", "has_emission": True},
    {"name": "VCC1902", "redshift": 0.0038, "type": "SBa", "has_emission": True},
    {"name": "VCC1910", "redshift": 0.0007, "type": "dE(N)", "has_emission": True},
    {"name": "VCC1949", "redshift": 0.0058, "type": "dS0(N)", "has_emission": True},
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ISAPC")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ISAPC Galaxy Analysis v2.0 - Analyze galaxies using P2P, VNB, and RDB"
    )

    # Input/output arguments
    parser.add_argument(
        "filename", nargs="?", default=None,
        help="Path to MUSE cube file (.fits). Optional if using predefined list."
    )
    parser.add_argument(
        "-o", "--output-dir", default="./output", help="Output directory"
    )
    parser.add_argument(
        "-t", "--template", default="", help="Template file path"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving results"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip creating plots"
    )
    
    # Analysis mode selection
    parser.add_argument(
        "--analysis-mode", 
        choices=["all", "p2p", "vnb", "rdb"], 
        default="all",
        help="Analysis mode: p2p, vnb, rdb, or all (default)"
    )
    
    # Galaxy fitting parameters
    parser.add_argument(
        "--vel-init", type=float, default=0.0, help="Initial velocity (km/s)"
    )
    parser.add_argument(
        "--sigma-init", type=float, default=50.0, help="Initial velocity dispersion (km/s)"
    )
    parser.add_argument(
        "--poly-degree", type=int, default=3, help="Polynomial degree for continuum fitting"
    )
    parser.add_argument(
        "--redshift", type=float, default=None, help="Galaxy redshift (overrides file value)"
    )
    
    # Voronoi binning parameters
    parser.add_argument(
        "--target-snr", type=float, default=20.0, help="Target SNR for Voronoi binning"
    )
    parser.add_argument(
        "--min-snr", type=float, default=0.1, help="Minimum SNR for inclusion in binning"
    )
    
    # Radial binning parameters
    parser.add_argument(
        "--n-rings", type=int, default=10, help="Number of rings for radial binning"
    )
    parser.add_argument(
        "--log-spacing", action="store_true", help="Use logarithmic spacing for radial bins"
    )
    parser.add_argument(
        "--pa", type=float, help="Position angle (degrees)"
    )
    parser.add_argument(
        "--ellipticity", type=float, help="Ellipticity (0-1)"
    )
    parser.add_argument(
        "--center-x", type=float, help="X coordinate of center (pixels)"
    )
    parser.add_argument(
        "--center-y", type=float, help="Y coordinate of center (pixels)"
    )
    parser.add_argument(
        "--bin-method", 
        choices=["equal_width", "logarithmic", "equal_flux", "equal_snr"], 
        default="equal_flux",
        help="Method for radial bin spacing"
    )
    parser.add_argument(
        "--min-pixels-per-bin", type=int, default=5, 
        help="Minimum number of pixels per bin"
    )
    
    # Galaxy list mode
    parser.add_argument(
        "--galaxy-list", help="Text file with list of galaxy filenames to process"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", 
        help="Skip galaxies with existing analysis results"
    )
    
    # Predefined galaxy list options
    parser.add_argument(
        "--use-predefined", action="store_true",
        help="Use the predefined galaxy list (VCC galaxies)"
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Base directory for galaxy files when using predefined list"
    )
    
    # Physical radius parameters
    parser.add_argument(
        "--physical-radius", action="store_true",
        help="Calculate and use physical radius for all modes"
    )
    
    # Re-use analysis results between modes
    parser.add_argument(
        "--auto-reuse", action="store_true",
        help="Automatically reuse results from previous analyses (e.g., P2P results for VNB and RDB)"
    )
    
    # Optional settings for emission line and indices analysis
    parser.add_argument(
        "--no-emission", action="store_true", help="Skip emission line fitting"
    )
    parser.add_argument(
        "--no-indices", action="store_true", help="Skip spectral indices calculation"
    )
    parser.add_argument(
        "--equal-aspect", action="store_true", help="Use equal aspect ratio in plots"
    )

    return parser.parse_args()


def run_galaxy_analysis(args):
    """
    Run full galaxy analysis with all modes
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    dict
        Dictionary containing all analysis results
    """
    # Check if filename is provided
    if args.filename is None:
        logger.error("No input filename provided")
        return None
        
    # Load cube
    try:
        logger.info(f"Loading data cube: {args.filename}")
        start_time = time.time()
        
        cube = MUSECube(args.filename)
        
        # Set redshift if provided in arguments
        if hasattr(args, "redshift") and args.redshift is not None:
            cube._redshift = args.redshift
            logger.info(f"Setting redshift to {args.redshift} (from command line)")
        
        logger.info(f"Loaded cube with shape {cube._cube_data.shape}")
        logger.info(f"Redshift: {cube._redshift}")
        logger.info(f"Cube loaded in {time.time() - start_time:.1f} seconds")
    except Exception as e:
        logger.error(f"Error loading cube: {e}")
        return None
    
    # Get galaxy name from filename
    galaxy_name = Path(args.filename).stem
    logger.info(f"Starting analysis for galaxy: {galaxy_name}")
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    
    # Create directories
    galaxy_dir.mkdir(exist_ok=True, parents=True)
    (galaxy_dir / "Data").mkdir(exist_ok=True)
    (galaxy_dir / "Plots").mkdir(exist_ok=True)
    
    # Create metadata file with galaxy information
    metadata_path = galaxy_dir / "galaxy_info.json"
    if not metadata_path.exists():
        metadata = {
            "galaxy_name": galaxy_name,
            "filename": str(Path(args.filename).absolute()),
            "timestamp": time.time(),
            "cube_shape": cube._cube_data.shape,
            "redshift": cube._redshift if hasattr(cube, "_redshift") else 0.0,
            "pixel_size_x": cube._pxl_size_x,
            "pixel_size_y": cube._pxl_size_y,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Prepare results dictionary
    results = {}
    
    # Run P2P analysis if requested
    p2p_results = None
    if args.analysis_mode in ["all", "p2p"]:
        logger.info("Running P2P (Pixel-to-Pixel) analysis...")
        
        # Check if existing results should be loaded
        if args.skip_existing:
            std_results_path = galaxy_dir / "Data" / f"{galaxy_name}_P2P_standardized.npz"
            results_path = galaxy_dir / "Data" / f"{galaxy_name}_P2P_results.npz"
            
            if std_results_path.exists() or results_path.exists():
                logger.info("Found existing P2P results, skipping analysis")
                # Load existing results
                try:
                    p2p_results = load_standardized_results(galaxy_name, "P2P", galaxy_dir / "Data")
                    if p2p_results is not None:
                        logger.info("Successfully loaded P2P results")
                        results["P2P"] = p2p_results
                    else:
                        logger.warning("Failed to load existing P2P results")
                except Exception as e:
                    logger.warning(f"Error loading existing P2P results: {e}")
            else:
                # Run P2P analysis
                p2p_results = run_p2p_analysis(args, cube, Pmode=True)
                if p2p_results is not None:
                    logger.info("P2P analysis completed successfully")
                    results["P2P"] = p2p_results
                else:
                    logger.error("P2P analysis failed")
        else:
            # Run P2P analysis
            p2p_results = run_p2p_analysis(args, cube, Pmode=True)
            if p2p_results is not None:
                logger.info("P2P analysis completed successfully")
                results["P2P"] = p2p_results
            else:
                logger.error("P2P analysis failed")
    
    # Run VNB analysis if requested
    vnb_results = None
    if args.analysis_mode in ["all", "vnb"]:
        logger.info("Running VNB (Voronoi Binning) analysis...")
        
        # Check if existing results should be loaded
        if args.skip_existing:
            std_results_path = galaxy_dir / "Data" / f"{galaxy_name}_VNB_standardized.npz"
            results_path = galaxy_dir / "Data" / f"{galaxy_name}_VNB_results.npz"
            
            if std_results_path.exists() or results_path.exists():
                logger.info("Found existing VNB results, skipping analysis")
                # Load existing results
                try:
                    vnb_results = load_standardized_results(galaxy_name, "VNB", galaxy_dir / "Data")
                    if vnb_results is not None:
                        logger.info("Successfully loaded VNB results")
                        results["VNB"] = vnb_results
                    else:
                        logger.warning("Failed to load existing VNB results")
                except Exception as e:
                    logger.warning(f"Error loading existing VNB results: {e}")
            else:
                # Run VNB analysis using P2P results if available
                vnb_results = run_voronoi_analysis(args, cube, p2p_results)
                if vnb_results is not None:
                    logger.info("VNB analysis completed successfully")
                    results["VNB"] = vnb_results
                else:
                    logger.error("VNB analysis failed")
        else:
            # Run VNB analysis using P2P results if available
            vnb_results = run_voronoi_analysis(args, cube, p2p_results)
            if vnb_results is not None:
                logger.info("VNB analysis completed successfully")
                results["VNB"] = vnb_results
            else:
                logger.error("VNB analysis failed")
    
    # Run RDB analysis if requested
    rdb_results = None
    if args.analysis_mode in ["all", "rdb"]:
        logger.info("Running RDB (Radial Binning) analysis...")
        
        # Check if existing results should be loaded
        if args.skip_existing:
            std_results_path = galaxy_dir / "Data" / f"{galaxy_name}_RDB_standardized.npz"
            results_path = galaxy_dir / "Data" / f"{galaxy_name}_RDB_results.npz"
            
            if std_results_path.exists() or results_path.exists():
                logger.info("Found existing RDB results, skipping analysis")
                # Load existing results
                try:
                    rdb_results = load_standardized_results(galaxy_name, "RDB", galaxy_dir / "Data")
                    if rdb_results is not None:
                        logger.info("Successfully loaded RDB results")
                        results["RDB"] = rdb_results
                    else:
                        logger.warning("Failed to load existing RDB results")
                except Exception as e:
                    logger.warning(f"Error loading existing RDB results: {e}")
            else:
                # Run RDB analysis using P2P results if available
                rdb_results = run_rdb_analysis(args, cube, p2p_results)
                if rdb_results is not None:
                    logger.info("RDB analysis completed successfully")
                    results["RDB"] = rdb_results
                else:
                    logger.error("RDB analysis failed")
        else:
            # Run RDB analysis using P2P results if available
            rdb_results = run_rdb_analysis(args, cube, p2p_results)
            if rdb_results is not None:
                logger.info("RDB analysis completed successfully")
                results["RDB"] = rdb_results
            else:
                logger.error("RDB analysis failed")
    
    # Create summary plots if multiple analyses were performed
    if len(results) > 1 and not args.no_plots:
        logger.info("Creating summary comparison plots...")
        create_summary_plots(galaxy_name, results, args.output_dir)
    
    logger.info(f"All analyses for galaxy {galaxy_name} completed")
    return results


def create_summary_plots(galaxy_name, results, output_dir):
    """
    Create summary plots comparing results from different analyses
    
    Parameters
    ----------
    galaxy_name : str
        Galaxy name
    results : dict
        Dictionary with analysis results
    output_dir : str or Path
        Output directory
    """
    # Convert to Path
    output_dir = Path(output_dir)
    
    # Create comparison directory
    galaxy_dir = output_dir / galaxy_name
    comparison_dir = galaxy_dir / "Plots" / "Comparison"
    comparison_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract available analysis modes
    modes = list(results.keys())
    
    # Don't create comparison if only one mode is available
    if len(modes) <= 1:
        return
    
    try:
        # 1. Compare physical radius distributions
        radius_fields = {}
        for mode in modes:
            # Check for physical radius
            if "radius" in results[mode] and "physical_radius" in results[mode]["radius"]:
                radius_fields[mode] = results[mode]["radius"]["physical_radius"]
        
        # Create physical radius comparison
        if len(radius_fields) > 1:
            fig, axes = plt.subplots(1, len(radius_fields), figsize=(6*len(radius_fields), 5))
            
            # Ensure axes is iterable
            if len(radius_fields) == 1:
                axes = [axes]
            
            # Get common color range
            vmin, vmax = float('inf'), float('-inf')
            for radius_field in radius_fields.values():
                valid_values = radius_field[np.isfinite(radius_field)]
                if len(valid_values) > 0:
                    vmin = min(vmin, np.percentile(valid_values, 5))
                    vmax = max(vmax, np.percentile(valid_values, 95))
            
            # Apply reasonable constraints
            vmin = max(0, vmin)
            vmax = min(100, max(5, vmax))  # Reasonable upper limit
            
            # Plot each radius field
            for i, (mode, radius_field) in enumerate(radius_fields.items()):
                im = axes[i].imshow(radius_field, origin='lower', cmap='plasma',
                                 vmin=vmin, vmax=vmax)
                axes[i].set_title(f"{mode} Physical Radius")
                plt.colorbar(im, ax=axes[i], label='Radius (arcsec)')
            
            plt.suptitle(f"{galaxy_name} - Physical Radius Comparison", fontsize=14)
            standardize_figure_saving(fig, comparison_dir / f"{galaxy_name}_radius_comparison.png")
            plt.close(fig)
        
        # 2. Compare kinematics
        velocity_fields = {}
        for mode in modes:
            # Check for velocity field
            if mode == "P2P" and "stellar_kinematics" in results[mode] and "velocity_field" in results[mode]["stellar_kinematics"]:
                velocity_fields[mode] = results[mode]["stellar_kinematics"]["velocity_field"]
            elif mode == "VNB" and "stellar_kinematics" in results[mode] and "velocity_field" in results[mode]["stellar_kinematics"]:
                velocity_fields[mode] = results[mode]["stellar_kinematics"]["velocity_field"]
        
        # Create velocity field comparison
        if len(velocity_fields) > 1:
            fig, axes = plt.subplots(1, len(velocity_fields), figsize=(6*len(velocity_fields), 5))
            
            # Ensure axes is iterable
            if len(velocity_fields) == 1:
                axes = [axes]
            
            # Get common color range
            vmin, vmax = float('inf'), float('-inf')
            for vel_field in velocity_fields.values():
                valid_values = vel_field[np.isfinite(vel_field)]
                if len(valid_values) > 0:
                    # Create symmetric range for velocity
                    max_abs = np.percentile(np.abs(valid_values), 95)
                    vmin = min(vmin, -max_abs)
                    vmax = max(vmax, max_abs)
            
            # Plot each velocity field
            for i, (mode, vel_field) in enumerate(velocity_fields.items()):
                im = axes[i].imshow(vel_field, origin='lower', cmap='coolwarm',
                                 vmin=vmin, vmax=vmax)
                axes[i].set_title(f"{mode} Velocity Field")
                plt.colorbar(im, ax=axes[i], label='Velocity (km/s)')
            
            plt.suptitle(f"{galaxy_name} - Velocity Field Comparison", fontsize=14)
            standardize_figure_saving(fig, comparison_dir / f"{galaxy_name}_velocity_comparison.png")
            plt.close(fig)
        
        # 3. Compare Hbeta index if available
        hbeta_maps = {}
        
        # For P2P, extract directly from indices
        if "P2P" in results and "indices" in results["P2P"] and "Hbeta" in results["P2P"]["indices"]:
            hbeta_maps["P2P"] = results["P2P"]["indices"]["Hbeta"]
        
        # For VNB, extract from pixel_indices
        if "VNB" in results and "bin_indices" in results["VNB"] and "pixel_indices" in results["VNB"]["bin_indices"]:
            pixel_indices = results["VNB"]["bin_indices"]["pixel_indices"]
            if "Hbeta" in pixel_indices:
                hbeta_maps["VNB"] = pixel_indices["Hbeta"]
        
        # Create Hbeta comparison
        if len(hbeta_maps) > 1:
            fig, axes = plt.subplots(1, len(hbeta_maps), figsize=(6*len(hbeta_maps), 5))
            
            # Ensure axes is iterable
            if len(hbeta_maps) == 1:
                axes = [axes]
            
            # Get common color range
            vmin, vmax = float('inf'), float('-inf')
            for hbeta_map in hbeta_maps.values():
                valid_values = hbeta_map[np.isfinite(hbeta_map)]
                if len(valid_values) > 0:
                    vmin = min(vmin, np.percentile(valid_values, 5))
                    vmax = max(vmax, np.percentile(valid_values, 95))
            
            # Plot each Hbeta map
            for i, (mode, hbeta_map) in enumerate(hbeta_maps.items()):
                im = axes[i].imshow(hbeta_map, origin='lower', cmap='viridis',
                                 vmin=vmin, vmax=vmax)
                axes[i].set_title(f"{mode} Hbeta Index")
                plt.colorbar(im, ax=axes[i], label='Hbeta Index')
            
            plt.suptitle(f"{galaxy_name} - Hbeta Index Comparison", fontsize=14)
            standardize_figure_saving(fig, comparison_dir / f"{galaxy_name}_hbeta_comparison.png")
            plt.close(fig)
        
        # 4. Compare age maps
        age_maps = {}
        
        # For P2P, extract directly
        if "P2P" in results and "stellar_population" in results["P2P"] and "age" in results["P2P"]["stellar_population"]:
            age_maps["P2P"] = results["P2P"]["stellar_population"]["age"] * 1e-9  # Convert to Gyr
        
        # For VNB, extract bin values and create 2D map
        if "VNB" in results and "stellar_population" in results["VNB"] and "age" in results["VNB"]["stellar_population"]:
            if "binning" in results["VNB"] and "bin_num" in results["VNB"]["binning"]:
                # Get age values and bin map
                age_values = results["VNB"]["stellar_population"]["age"] * 1e-9  # Convert to Gyr
                bin_map = results["VNB"]["binning"]["bin_num"]
                
                # Create 2D age map
                age_map = np.full_like(bin_map, np.nan, dtype=float)
                
                # Fill age values by bin
                for i, age in enumerate(age_values):
                    if np.isfinite(age):
                        age_map[bin_map == i] = age
                
                age_maps["VNB"] = age_map
        
        # Create age comparison
        if len(age_maps) > 1:
            fig, axes = plt.subplots(1, len(age_maps), figsize=(6*len(age_maps), 5))
            
            # Ensure axes is iterable
            if len(age_maps) == 1:
                axes = [axes]
            
            # Get common color range
            vmin, vmax = float('inf'), float('-inf')
            for age_map in age_maps.values():
                valid_values = age_map[np.isfinite(age_map)]
                if len(valid_values) > 0:
                    vmin = min(vmin, np.percentile(valid_values, 5))
                    vmax = max(vmax, np.percentile(valid_values, 95))
            
            # Apply reasonable constraints for ages in Gyr
            vmin = max(0.1, vmin)
            vmax = min(15, max(1, vmax))
            
            # Plot each age map
            for i, (mode, age_map) in enumerate(age_maps.items()):
                im = axes[i].imshow(age_map, origin='lower', cmap='plasma',
                                 vmin=vmin, vmax=vmax)
                axes[i].set_title(f"{mode} Age Map")
                plt.colorbar(im, ax=axes[i], label='Age (Gyr)')
            
            plt.suptitle(f"{galaxy_name} - Age Comparison", fontsize=14)
            standardize_figure_saving(fig, comparison_dir / f"{galaxy_name}_age_comparison.png")
            plt.close(fig)
    
    except Exception as e:
        logger.error(f"Error creating comparison plots: {e}")
        plt.close("all")


def process_galaxy_list(args):
    """
    Process multiple galaxies from a list file
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    try:
        # Read galaxy list file
        galaxy_list_path = Path(args.galaxy_list)
        if not galaxy_list_path.exists():
            logger.error(f"Galaxy list file not found: {galaxy_list_path}")
            return None
            
        with open(galaxy_list_path, 'r') as f:
            galaxy_files = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    galaxy_files.append(line)
        
        logger.info(f"Found {len(galaxy_files)} galaxies in list file")
        
        # Create summary directory
        output_dir = Path(args.output_dir)
        summary_dir = output_dir / "Summary"
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a tracking file for processed galaxies
        tracking_file = summary_dir / "processed_galaxies.csv"
        
        # Initialize or load tracking data
        if tracking_file.exists():
            import csv
            with open(tracking_file, 'r') as f:
                reader = csv.DictReader(f)
                tracking_data = {row['filename']: row for row in reader}
        else:
            tracking_data = {}
            
        # Create tracking file if it doesn't exist
        if not tracking_file.exists():
            import csv
            with open(tracking_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'galaxy_name', 'status', 'timestamp', 'p2p', 'vnb', 'rdb'])
        
        # Process each galaxy
        results = {}
        for i, galaxy_file in enumerate(galaxy_files):
            # Get absolute path to the galaxy file
            galaxy_path = Path(galaxy_file)
            if not galaxy_path.is_absolute():
                # Try relative to the current directory
                if Path(galaxy_file).exists():
                    galaxy_path = Path(galaxy_file).absolute()
                # Try relative to the list file directory
                elif (galaxy_list_path.parent / galaxy_file).exists():
                    galaxy_path = (galaxy_list_path.parent / galaxy_file).absolute()
                else:
                    logger.error(f"Galaxy file not found: {galaxy_file}")
                    # Update tracking data
                    tracking_data[galaxy_file] = {
                        'filename': galaxy_file,
                        'galaxy_name': Path(galaxy_file).stem,
                        'status': 'not_found',
                        'timestamp': time.time(),
                        'p2p': 'N/A',
                        'vnb': 'N/A',
                        'rdb': 'N/A'
                    }
                    continue
            
            galaxy_name = galaxy_path.stem
            logger.info(f"Processing galaxy {i+1}/{len(galaxy_files)}: {galaxy_name}")
            
            # Check if already processed and skip_existing is enabled
            if args.skip_existing and galaxy_file in tracking_data:
                status = tracking_data[galaxy_file].get('status', '')
                if status == 'completed':
                    logger.info(f"Skipping {galaxy_name} - already processed")
                    continue
            
            # Update args with current filename
            args.filename = str(galaxy_path)
            
            # Run analysis
            try:
                galaxy_results = run_galaxy_analysis(args)
                
                # Store results by galaxy name
                if galaxy_results is not None:
                    results[galaxy_name] = galaxy_results
                    logger.info(f"Analysis for {galaxy_name} completed")
                    
                    # Update tracking data
                    tracking_data[galaxy_file] = {
                        'filename': galaxy_file,
                        'galaxy_name': galaxy_name,
                        'status': 'completed',
                        'timestamp': time.time(),
                        'p2p': 'yes' if 'P2P' in galaxy_results else 'no',
                        'vnb': 'yes' if 'VNB' in galaxy_results else 'no',
                        'rdb': 'yes' if 'RDB' in galaxy_results else 'no'
                    }
                else:
                    logger.error(f"Analysis for {galaxy_file} failed")
                    # Update tracking data
                    tracking_data[galaxy_file] = {
                        'filename': galaxy_file,
                        'galaxy_name': galaxy_name,
                        'status': 'failed',
                        'timestamp': time.time(),
                        'p2p': 'N/A',
                        'vnb': 'N/A',
                        'rdb': 'N/A'
                    }
            except Exception as e:
                logger.error(f"Error processing {galaxy_name}: {e}")
                # Update tracking data
                tracking_data[galaxy_file] = {
                    'filename': galaxy_file,
                    'galaxy_name': galaxy_name,
                    'status': 'error',
                    'timestamp': time.time(),
                    'p2p': 'N/A',
                    'vnb': 'N/A',
                    'rdb': 'N/A'
                }
            
            # Update tracking file after each galaxy
            try:
                import csv
                with open(tracking_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'galaxy_name', 'status', 'timestamp', 'p2p', 'vnb', 'rdb'])
                    writer.writeheader()
                    for data in tracking_data.values():
                        writer.writerow(data)
            except Exception as e:
                logger.warning(f"Error updating tracking file: {e}")
        
        # Create summary of all galaxies
        if results:
            create_all_galaxy_summary(results, args.output_dir)
        
        logger.info(f"All {len(galaxy_files)} galaxies processed")
        return results
    
    except Exception as e:
        logger.error(f"Error processing galaxy list: {e}")
        return None


def process_predefined_galaxies(args, galaxies=None, base_dir=None):
    """
    Process galaxies from the predefined list
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    galaxies : list, optional
        List of galaxy dictionaries, defaults to GALAXIES
    base_dir : str or Path, optional
        Base directory for galaxy files, defaults to current directory
        
    Returns
    -------
    dict
        Dictionary of results by galaxy name
    """
    if galaxies is None:
        galaxies = GALAXIES
        
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)
    
    logger.info(f"Processing {len(galaxies)} galaxies from predefined list")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create galaxy data file with complete info
    galaxy_info_path = output_dir / "galaxy_info.json"
    
    # Save galaxy information
    with open(galaxy_info_path, 'w') as f:
        json.dump(galaxies, f, indent=2)
    
    # Create summary directory
    summary_dir = output_dir / "Summary"
    summary_dir.mkdir(exist_ok=True, parents=True)
    
    # Create tracking file for processed galaxies
    tracking_file = summary_dir / "processed_galaxies.csv"
    
    # Initialize or load tracking data
    tracking_data = {}
    if tracking_file.exists():
        import csv
        try:
            with open(tracking_file, 'r') as f:
                reader = csv.DictReader(f)
                tracking_data = {row['name']: row for row in reader}
        except Exception as e:
            logger.warning(f"Error loading tracking file: {e}")
    
    # Create tracking file if it doesn't exist
    if not tracking_file.exists():
        import csv
        with open(tracking_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'type', 'redshift', 'has_emission', 'status', 'timestamp', 'p2p', 'vnb', 'rdb'])
    
    # Process each galaxy
    results = {}
    for i, galaxy in enumerate(galaxies):
        galaxy_name = galaxy["name"]
        logger.info(f"Processing galaxy {i+1}/{len(galaxies)}: {galaxy_name} ({galaxy['type']}, z={galaxy['redshift']})")
        
        # Check if already processed and skip_existing is enabled
        if args.skip_existing and galaxy_name in tracking_data:
            status = tracking_data[galaxy_name].get('status', '')
            if status == 'completed':
                logger.info(f"Skipping {galaxy_name} - already processed")
                continue
        
        # Look for galaxy file in standard locations
        galaxy_paths = [
            base_dir / f"{galaxy_name}.fits",
            base_dir / f"{galaxy_name}.FITS",
            base_dir / f"{galaxy_name}.fit",
            base_dir / f"{galaxy_name}_MUSE.fits",
            base_dir / "data" / f"{galaxy_name}.fits",
            base_dir / "data" / f"{galaxy_name}_MUSE.fits",
            base_dir / galaxy_name / f"{galaxy_name}.fits",
            base_dir / galaxy_name / f"{galaxy_name}_MUSE.fits",
        ]
        
        # Find first valid file
        galaxy_path = None
        for path in galaxy_paths:
            if path.exists():
                galaxy_path = path
                break
        
        if galaxy_path is None:
            logger.error(f"Could not find data file for galaxy {galaxy_name}")
            # Update tracking data
            tracking_data[galaxy_name] = {
                'name': galaxy_name,
                'type': galaxy['type'],
                'redshift': galaxy['redshift'],
                'has_emission': str(galaxy['has_emission']),
                'status': 'not_found',
                'timestamp': time.time(),
                'p2p': 'N/A',
                'vnb': 'N/A',
                'rdb': 'N/A'
            }
            continue
        
        # Update args with current filename and parameters
        args.filename = str(galaxy_path)
        
        # Set no-emission parameter based on galaxy info if not explicitly set
        if not hasattr(args, "no_emission") or args.no_emission is None:
            args.no_emission = not galaxy['has_emission']
            
        # Set redshift for cube if available
        if hasattr(args, "redshift") and args.redshift is None:
            args.redshift = galaxy['redshift']
        
        # Run analysis
        try:
            galaxy_results = run_galaxy_analysis(args)
            
            # Store results by galaxy name
            if galaxy_results is not None:
                results[galaxy_name] = galaxy_results
                logger.info(f"Analysis for {galaxy_name} completed")
                
                # Update tracking data
                tracking_data[galaxy_name] = {
                    'name': galaxy_name,
                    'type': galaxy['type'],
                    'redshift': galaxy['redshift'],
                    'has_emission': str(galaxy['has_emission']),
                    'status': 'completed',
                    'timestamp': time.time(),
                    'p2p': 'yes' if 'P2P' in galaxy_results else 'no',
                    'vnb': 'yes' if 'VNB' in galaxy_results else 'no',
                    'rdb': 'yes' if 'RDB' in galaxy_results else 'no'
                }
            else:
                logger.error(f"Analysis for {galaxy_name} failed")
                # Update tracking data
                tracking_data[galaxy_name] = {
                    'name': galaxy_name,
                    'type': galaxy['type'],
                    'redshift': galaxy['redshift'],
                    'has_emission': str(galaxy['has_emission']),
                    'status': 'failed',
                    'timestamp': time.time(),
                    'p2p': 'N/A',
                    'vnb': 'N/A',
                    'rdb': 'N/A'
                }
        except Exception as e:
            logger.error(f"Error processing {galaxy_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Update tracking data
            tracking_data[galaxy_name] = {
                'name': galaxy_name,
                'type': galaxy['type'],
                'redshift': galaxy['redshift'],
                'has_emission': str(galaxy['has_emission']),
                'status': 'error',
                'timestamp': time.time(),
                'p2p': 'N/A',
                'vnb': 'N/A',
                'rdb': 'N/A'
            }
        
        # Update tracking file after each galaxy
        try:
            import csv
            with open(tracking_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['name', 'type', 'redshift', 'has_emission', 'status', 'timestamp', 'p2p', 'vnb', 'rdb'])
                writer.writeheader()
                for data in tracking_data.values():
                    writer.writerow(data)
        except Exception as e:
            logger.warning(f"Error updating tracking file: {e}")
    
    # Create summary of all galaxies
    if results:
        create_all_galaxy_summary(results, args.output_dir)
        
        # Create morphology-based comparison plots
        try:
            create_morphology_comparison(results, galaxies, summary_dir)
        except Exception as e:
            logger.error(f"Error creating morphology comparison: {e}")
    
    logger.info(f"All {len(galaxies)} galaxies processed")
    return results


def create_all_galaxy_summary(results, output_dir):
    """
    Create summary plots and tables for all processed galaxies
    
    Parameters
    ----------
    results : dict
        Dictionary of results indexed by galaxy name
    output_dir : str or Path
        Output directory
    """
    # Convert to Path
    output_dir = Path(output_dir)
    
    # Create summary directory
    summary_dir = output_dir / "Summary"
    summary_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Create list of available galaxies and modes
        galaxies = list(results.keys())
        
        if not galaxies:
            logger.warning("No galaxy results to summarize")
            return
        
        logger.info(f"Creating summary for {len(galaxies)} galaxies")
        
        # Create JSON summary file
        summary_data = {}
        
        for galaxy_name in galaxies:
            # Create summary for this galaxy
            galaxy_summary = {}
            
            # Add available analysis modes
            available_modes = list(results[galaxy_name].keys())
            galaxy_summary["analysis_modes"] = available_modes
            
            # Add physical parameters if available
            for mode in available_modes:
                # Check for R_eff
                if "radius" in results[galaxy_name][mode] and "R_eff" in results[galaxy_name][mode]["radius"]:
                    galaxy_summary["R_eff"] = float(results[galaxy_name][mode]["radius"]["R_eff"])
                    galaxy_summary["R_eff_source"] = mode
                    break  # Use first available R_eff
            
            # Check for global kinematics
            for mode in ["P2P", "RDB"]:  # Prefer P2P for global kinematics
                if mode in available_modes and "global_kinematics" in results[galaxy_name][mode]:
                    global_kin = results[galaxy_name][mode]["global_kinematics"]
                    
                    if "pa" in global_kin:
                        galaxy_summary["PA"] = float(global_kin["pa"])
                    
                    if "ellipticity" in global_kin:
                        galaxy_summary["ellipticity"] = float(global_kin["ellipticity"])
                    
                    if "vsys" in global_kin:
                        galaxy_summary["vsys"] = float(global_kin["vsys"])
                    
                    galaxy_summary["kinematics_source"] = mode
                    break
            
            # Add timestamp
            for mode in available_modes:
                if "time" in results[galaxy_name][mode]:
                    galaxy_summary["timestamp"] = float(results[galaxy_name][mode]["time"])
                    break
            
            # Add pixel size
            for mode in available_modes:
                if "radius" in results[galaxy_name][mode]:
                    if "pixelsize_x" in results[galaxy_name][mode]["radius"]:
                        galaxy_summary["pixelsize_x"] = float(results[galaxy_name][mode]["radius"]["pixelsize_x"])
                    if "pixelsize_y" in results[galaxy_name][mode]["radius"]:
                        galaxy_summary["pixelsize_y"] = float(results[galaxy_name][mode]["radius"]["pixelsize_y"])
                    break
            
            # Store in summary
            summary_data[galaxy_name] = galaxy_summary
        
        # Save summary to JSON file
        with open(summary_dir / "galaxy_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Create summary table of key parameters
        try:
            import pandas as pd
            
            # Convert to DataFrame
            summary_df = pd.DataFrame()
            
            for galaxy, data in summary_data.items():
                row = {"Galaxy": galaxy}
                
                # Add common columns
                if "R_eff" in data:
                    row["R_eff (arcsec)"] = data["R_eff"]
                
                if "PA" in data:
                    row["PA (deg)"] = data["PA"]
                
                if "ellipticity" in data:
                    row["Ellipticity"] = data["ellipticity"]
                
                if "vsys" in data:
                    row["Vsys (km/s)"] = data["vsys"]
                
                if "analysis_modes" in data:
                    row["Analysis Modes"] = ", ".join(data["analysis_modes"])
                
                # Add row to DataFrame
                summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
            
            # Save to CSV
            if not summary_df.empty:
                summary_df.to_csv(summary_dir / "galaxy_parameters.csv", index=False)
                
                # Create LaTeX table if possible
                try:
                    latex_table = summary_df.to_latex(index=False, float_format="%.2f")
                    with open(summary_dir / "galaxy_parameters.tex", 'w') as f:
                        f.write(latex_table)
                except Exception as e:
                    logger.warning(f"Error creating LaTeX table: {e}")
        
        except ImportError:
            logger.warning("pandas not found, skipping summary table creation")
        except Exception as e:
            logger.warning(f"Error creating summary table: {e}")
        
        # Create comparison plots between galaxies
        # Only compare galaxies with R_eff information
        galaxies_with_reff = [g for g in galaxies if "R_eff" in summary_data[g]]
        
        if len(galaxies_with_reff) > 1:
            logger.info(f"Creating comparison plots for {len(galaxies_with_reff)} galaxies with R_eff")
            
            # Compare normalized radius vs. spectral indices
            try:
                create_multi_galaxy_radius_plots(
                    galaxies_with_reff, 
                    results, 
                    summary_data,
                    summary_dir
                )
            except Exception as e:
                logger.error(f"Error creating multi-galaxy radius plots: {e}")
        
        logger.info("Galaxy summary creation completed")
    
    except Exception as e:
        logger.error(f"Error creating all galaxy summary: {e}")


def create_multi_galaxy_radius_plots(galaxies, results, summary_data, summary_dir):
    """
    Create plots comparing parameters vs. normalized radius across galaxies
    
    Parameters
    ----------
    galaxies : list
        List of galaxy names
    results : dict
        Dictionary of results indexed by galaxy name
    summary_data : dict
        Dictionary of summary data
    summary_dir : Path
        Output directory for summary plots
    """
    # Create radius profiles directory
    radius_dir = summary_dir / "Radius_Profiles"
    radius_dir.mkdir(exist_ok=True, parents=True)
    
    # For each parameter, create a plot comparing galaxies
    parameters = [
        {"name": "Hbeta", "label": "Hbeta Index", "source": "indices"},
        {"name": "Mgb", "label": "Mgb Index", "source": "indices"},
        {"name": "Fe5015", "label": "Fe5015 Index", "source": "indices"},
        {"name": "age", "label": "Age (Gyr)", "source": "stellar_population", "scale": 1e-9},
        {"name": "metallicity", "label": "Metallicity [Z/H]", "source": "stellar_population"},
    ]
    
    # For each parameter, create plots comparing RDB results
    for param in parameters:
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Track if any data was plotted
            plotted = False
            
            # Plot each galaxy
            for galaxy in galaxies:
                # Get data from RDB results
                if "RDB" in results[galaxy] and param["source"] in results[galaxy]["RDB"]:
                    if param["name"] in results[galaxy]["RDB"][param["source"]]:
                        # Get parameter values
                        values = results[galaxy]["RDB"][param["source"]][param["name"]]
                        
                        # Apply scaling if needed
                        if "scale" in param:
                            values = values * param["scale"]
                        
                        # Get radii
                        if "radius" in results[galaxy]["RDB"] and "bin_distances" in results[galaxy]["RDB"]["radius"]:
                            radii = results[galaxy]["RDB"]["radius"]["bin_distances"]
                            
                            # Get R_eff for normalization
                            if "R_eff" in summary_data[galaxy]:
                                r_eff = summary_data[galaxy]["R_eff"]
                                
                                # Calculate normalized radii
                                norm_radii = radii / r_eff
                                
                                # Plot for this galaxy
                                ax.plot(norm_radii, values, 'o-', label=galaxy, alpha=0.7)
                                plotted = True
            
            # Only save if data was plotted
            if plotted:
                ax.set_xlabel('R/R_eff')
                ax.set_ylabel(param["label"])
                ax.set_title(f'All Galaxies - {param["label"]} vs. Normalized Radius')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add vertical line at R_eff
                ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label="R_eff")
                
                # Save figure
                standardize_figure_saving(fig, radius_dir / f"all_galaxies_{param['name']}_vs_radius.png")
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating multi-galaxy plot for {param['name']}: {e}")
            plt.close("all")


def create_morphology_comparison(results, galaxies, summary_dir):
    """
    Create comparison plots grouped by galaxy morphology
    
    Parameters
    ----------
    results : dict
        Dictionary of results by galaxy name
    galaxies : list
        List of galaxy dictionaries
    summary_dir : Path
        Output directory for summary plots
    """
    # Create morphology directory
    morphology_dir = summary_dir / "Morphology"
    morphology_dir.mkdir(exist_ok=True, parents=True)
    
    # Group galaxies by morphological type
    galaxy_types = {}
    for galaxy in galaxies:
        if galaxy['name'] in results:
            # Extract first letter of type for broad classification
            broad_type = galaxy['type'][0]  # E, S, d
            if broad_type not in galaxy_types:
                galaxy_types[broad_type] = []
            galaxy_types[broad_type].append(galaxy['name'])
    
    # For each parameter, create a plot comparing galaxies by morphology
    parameters = [
        {"name": "Hbeta", "label": "Hbeta Index", "source": "indices"},
        {"name": "Mgb", "label": "Mgb Index", "source": "indices"},
        {"name": "Fe5015", "label": "Fe5015 Index", "source": "indices"},
        {"name": "age", "label": "Age (Gyr)", "source": "stellar_population", "scale": 1e-9},
        {"name": "metallicity", "label": "Metallicity [Z/H]", "source": "stellar_population"},
    ]
    
    # For each morphological type and parameter, create plots comparing RDB results
    for broad_type, galaxies_of_type in galaxy_types.items():
        if len(galaxies_of_type) < 2:
            continue  # Skip if only one galaxy of this type
            
        for param in parameters:
            try:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Track if any data was plotted
                plotted = False
                
                # Plot each galaxy of this type
                for galaxy in galaxies_of_type:
                    # Get data from RDB results
                    if "RDB" in results[galaxy] and param["source"] in results[galaxy]["RDB"]:
                        if param["name"] in results[galaxy]["RDB"][param["source"]]:
                            # Get parameter values
                            values = results[galaxy]["RDB"][param["source"]][param["name"]]
                            
                            # Apply scaling if needed
                            if "scale" in param:
                                values = values * param["scale"]
                            
                            # Get radii
                            if "radius" in results[galaxy]["RDB"] and "bin_distances" in results[galaxy]["RDB"]["radius"]:
                                radii = results[galaxy]["RDB"]["radius"]["bin_distances"]
                                
                                # Get R_eff for normalization
                                if "radius" in results[galaxy]["RDB"] and "R_eff" in results[galaxy]["RDB"]["radius"]:
                                    r_eff = results[galaxy]["RDB"]["radius"]["R_eff"]
                                    
                                    # Calculate normalized radii
                                    norm_radii = radii / r_eff
                                    
                                    # Plot for this galaxy
                                    ax.plot(norm_radii, values, 'o-', label=galaxy, alpha=0.7)
                                    plotted = True
                
                # Only save if data was plotted
                if plotted:
                    ax.set_xlabel('R/R_eff')
                    ax.set_ylabel(param["label"])
                    ax.set_title(f'Type {broad_type} Galaxies - {param["label"]} vs. Normalized Radius')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Add vertical line at R_eff
                    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label="R_eff")
                    
                    # Save figure
                    standardize_figure_saving(fig, morphology_dir / f"type_{broad_type}_{param['name']}_vs_radius.png")
                
                plt.close(fig)
                
            except Exception as e:
                logger.error(f"Error creating morphology plot for {param['name']}: {e}")
                plt.close("all")


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Set template file
    if not args.template:
        # Try to find template file in the current directory
        default_templates = [
            "pp_ssp.fits",
            "templates/pp_ssp.fits",
            "ISAPC/templates/pp_ssp.fits",
            "../templates/pp_ssp.fits",
        ]
        
        for template in default_templates:
            if os.path.exists(template):
                args.template = template
                logger.info(f"Using template file: {template}")
                break
        
        if not args.template:
            logger.error("No template file specified and no default found")
            sys.exit(1)
    
    # Check which mode to use
    if args.use_predefined:
        # Use predefined galaxy list
        process_predefined_galaxies(args, base_dir=args.base_dir)
    elif args.galaxy_list:
        # Use custom galaxy list file
        logger.info(f"Processing galaxy list from {args.galaxy_list}")
        process_galaxy_list(args)
    else:
        # Run single galaxy analysis
        run_galaxy_analysis(args)


if __name__ == "__main__":
    main()