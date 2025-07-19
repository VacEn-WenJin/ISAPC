"""
ISAPC - IFU Spectrum Analysis Pipeline Cluster
Main program and command-line interface
Version 5.1.0 - Enhanced with error propagation support
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from analysis.p2p import run_p2p_analysis
from analysis.radial import run_rdb_analysis
from analysis.voronoi import run_vnb_analysis
from utils.io import (
    load_results_from_npz,
    save_results_to_npz,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ISAPC")


def setup_parser():
    """Set up the command-line argument parser with error propagation options"""
    parser = argparse.ArgumentParser(
        description="ISAPC - IFU Spectrum Analysis Pipeline Cluster"
    )

    # Required arguments
    parser.add_argument("filename", help="MUSE data cube filename")
    parser.add_argument(
        "-z", "--redshift", type=float, required=True, help="Galaxy redshift"
    )
    parser.add_argument(
        "-t", "--template", required=True, help="Stellar template filename"
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")

    # Analysis mode
    parser.add_argument(
        "-m",
        "--mode",
        choices=["P2P", "VNB", "RDB", "ALL"],
        default="ALL",
        help="Analysis mode",
    )

    # Fitting parameters
    parser.add_argument(
        "--vel-init", type=float, default=0, help="Initial velocity for fitting (km/s)"
    )
    parser.add_argument(
        "--sigma-init",
        type=float,
        default=150,
        help="Initial velocity dispersion for fitting (km/s)",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=3,
        help="Degree of polynomial for continuum fitting",
    )

    # Binning parameters
    parser.add_argument(
        "--target-snr", type=float, default=30, help="Target SNR for Voronoi binning"
    )
    parser.add_argument(
        "--min-snr", type=float, default=1, help="Minimum SNR for valid data"
    )
    parser.add_argument(
        "--cvt", action="store_true", help="Use CVT algorithm for Voronoi binning"
    )
    parser.add_argument(
        "--n-rings", type=int, default=10, help="Number of rings for radial binning"
    )
    parser.add_argument(
        "--log-spacing",
        action="store_true",
        help="Use logarithmic spacing for radial bins",
    )
    parser.add_argument(
        "--pa",
        type=float,
        default=None,
        help="Position angle for radial bins (degrees)",
    )
    parser.add_argument(
        "--ellipticity",
        type=float,
        default=None,
        help="Ellipticity for radial bins (0-1)",
    )
    parser.add_argument(
        "--center-x", type=float, default=None, help="X coordinate of center (pixels)"
    )
    parser.add_argument(
        "--center-y", type=float, default=None, help="Y coordinate of center (pixels)"
    )

    # Error propagation options
    error_group = parser.add_argument_group("Error Propagation Options")
    error_group.add_argument(
        "--error-mode",
        choices=["analytical", "monte_carlo", "bootstrap", "hybrid"],
        default="analytical",
        help="Error propagation method (default: analytical)",
    )
    error_group.add_argument(
        "--n-monte-carlo",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations for error estimation",
    )
    error_group.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap samples for error estimation",
    )
    error_group.add_argument(
        "--use-covariance",
        action="store_true",
        default=True,
        help="Use spatial covariance in error propagation (default: True)",
    )
    error_group.add_argument(
        "--no-covariance",
        action="store_false",
        dest="use_covariance",
        help="Disable spatial covariance in error propagation",
    )
    error_group.add_argument(
        "--correlation-length",
        type=float,
        default=2.0,
        help="Spatial correlation length in pixels (default: 2.0)",
    )
    error_group.add_argument(
        "--error-validation",
        action="store_true",
        help="Validate error propagation using multiple methods",
    )
    error_group.add_argument(
        "--save-error-maps",
        action="store_true",
        help="Save detailed error maps for all parameters",
    )
    error_group.add_argument(
        "--plot-errors",
        action="store_true",
        help="Create additional plots showing error distributions",
    )
    error_group.add_argument(
        "--error-percentiles",
        nargs=3,
        type=float,
        default=[16, 50, 84],
        help="Percentiles for error reporting (default: 16 50 84)",
    )

    # Stellar population error options
    sp_error_group = parser.add_argument_group("Stellar Population Error Options")
    sp_error_group.add_argument(
        "--sp-monte-carlo",
        type=int,
        default=100,
        help="Number of Monte Carlo realizations for stellar population errors",
    )
    sp_error_group.add_argument(
        "--sp-error-method",
        choices=["weights", "spectra", "both"],
        default="weights",
        help="Method for stellar population error estimation",
    )
    sp_error_group.add_argument(
        "--save-mc-realizations",
        action="store_true",
        help="Save individual Monte Carlo realizations",
    )

    # Spectral index error options
    idx_error_group = parser.add_argument_group("Spectral Index Error Options")
    idx_error_group.add_argument(
        "--index-error-method",
        choices=["analytical", "bootstrap", "monte_carlo"],
        default="analytical",
        help="Error estimation method for spectral indices",
    )
    idx_error_group.add_argument(
        "--index-bootstrap-samples",
        type=int,
        default=200,
        help="Number of bootstrap samples for spectral index errors",
    )
    idx_error_group.add_argument(
        "--index-error-validation",
        action="store_true",
        help="Cross-validate spectral index errors using multiple methods",
    )

    # Plotting options
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot generation"
    )
    parser.add_argument(
        "--equal-aspect", action="store_true", help="Use equal aspect ratio for plots"
    )

    # Performance options
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)",
    )

    # Analysis options
    parser.add_argument(
        "--no-emission", action="store_true", help="Skip emission line fitting"
    )
    parser.add_argument(
        "--no-indices", action="store_true", help="Skip spectral indices calculation"
    )

    # Data reuse options (modified)
    parser.add_argument(
        "--auto-reuse",
        action="store_true",
        default=True,
        help="Automatically load and reuse previous results if available",
    )
    parser.add_argument(
        "--no-auto-reuse",
        action="store_false",
        dest="auto_reuse",
        help="Disable automatic reuse of previous results",
    )

    # Configuration options
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    parser.add_argument(
        "--indices",
        type=str,
        nargs="+",
        help="List of spectral indices to calculate (overrides config defaults)",
    )
    parser.add_argument(
        "--emission-lines",
        type=str,
        nargs="+",
        help="List of emission lines to fit (overrides config defaults)",
    )
    parser.add_argument(
        "--physical-radius",
        action="store_true",
        help="Use flux-based elliptical physical radius for radial binning"
    )
    parser.add_argument(
        "--high-snr-mode",
        action="store_true",
        help="Use high-SNR optimization for Voronoi binning"
    )
    parser.add_argument(
        "--optimize-bins",
        action="store_true",
        help="Optimize binning to achieve target number of bins (10-20)"
    )

    # Advanced error options
    advanced_error_group = parser.add_argument_group("Advanced Error Options")
    advanced_error_group.add_argument(
        "--error-floor",
        type=float,
        default=0.01,
        help="Minimum relative error floor (default: 0.01 = 1%%)",
    )
    advanced_error_group.add_argument(
        "--systematic-error",
        type=float,
        default=0.0,
        help="Additional systematic error to add in quadrature (fraction)",
    )
    advanced_error_group.add_argument(
        "--error-correlation-model",
        choices=["exponential", "gaussian", "powerlaw"],
        default="exponential",
        help="Spatial correlation model for errors",
    )
    advanced_error_group.add_argument(
        "--save-covariance-matrices",
        action="store_true",
        help="Save full covariance matrices (warning: large files)",
    )
    advanced_error_group.add_argument(
        "--error-debug",
        action="store_true",
        help="Enable detailed error propagation debugging",
    )

    # Verbose/quiet options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-critical output"
    )

    return parser


def configure_error_propagation(args):
    """
    Configure error propagation settings based on command-line arguments
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    # Import error configuration module
    # from utils.error_propagation import configure_error_settings
    
    # Create error configuration dictionary
    error_config = {
        "mode": args.error_mode,
        "n_monte_carlo": args.n_monte_carlo,
        "n_bootstrap": args.n_bootstrap,
        "use_covariance": args.use_covariance,
        "correlation_length": args.correlation_length,
        "error_floor": args.error_floor,
        "systematic_error": args.systematic_error,
        "correlation_model": args.error_correlation_model,
        "validation": args.error_validation,
        "save_maps": args.save_error_maps,
        "plot_errors": args.plot_errors,
        "percentiles": args.error_percentiles,
        "debug": args.error_debug,
        "save_covariance": args.save_covariance_matrices,
    }
    
    # Stellar population specific settings
    sp_error_config = {
        "n_monte_carlo": args.sp_monte_carlo,
        "method": args.sp_error_method,
        "save_realizations": args.save_mc_realizations,
    }
    
    # Spectral index specific settings
    idx_error_config = {
        "method": args.index_error_method,
        "n_bootstrap": args.index_bootstrap_samples,
        "validation": args.index_error_validation,
    }
    
    # Configure global error settings
    # configure_error_settings(error_config, sp_error_config, idx_error_config)
    
    # Store configurations in args for passing to analysis functions
    args.error_config = error_config
    args.sp_error_config = sp_error_config
    args.idx_error_config = idx_error_config
    
    logger.info(f"Error propagation configured: mode={args.error_mode}, "
                f"covariance={args.use_covariance}, MC samples={args.n_monte_carlo}")


def main():
    """Main entry point for ISAPC with error propagation support"""
    # Parse command-line arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Configure logging level based on verbose/quiet flags
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure error propagation settings
    configure_error_propagation(args)

    # Load custom config file if specified
    if args.config:
        from config_manager import load_config

        load_config(args.config)
        logger.info(f"Loaded custom configuration from {args.config}")
    else:
        # Load default configuration
        from config_manager import get_config

        get_config()  # Initialize with default config
        logger.info("Using default configuration")

    # Extract galaxy name from filename
    galaxy_name = Path(args.filename).stem

    # Create output directories
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / "Data"
    error_dir = galaxy_dir / "Errors"  # New directory for error-related outputs

    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    if args.save_error_maps or args.save_covariance_matrices:
        error_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Starting ISAPC analysis for {galaxy_name} in mode {args.mode}")
    logger.info(f"Error propagation mode: {args.error_mode}")

    # Import the MUSECube class here to avoid circular imports
    try:
        from muse import MUSECube
    except ImportError:
        logger.error(
            "Failed to import MUSECube class. Please ensure muse.py is in the Python path."
        )
        return 1

    # Load data cube with error support
    try:
        logger.info(f"Loading data cube from {args.filename}")
        cube = MUSECube(
            filename=args.filename, 
            redshift=args.redshift, 
            use_good_wavelength=True
        )
        logger.info("Data cube loaded successfully")
        
        # Check if errors were loaded
        if hasattr(cube, '_error') or hasattr(cube, '_variance'):
            logger.info("Error/variance data successfully loaded")
        else:
            logger.warning("No error data found in cube - errors will be estimated")
            
    except Exception as e:
        logger.error(f"Error loading data cube: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

    # Get spectral indices and emission lines from config or command line
    from config_manager import get_emission_lines, get_spectral_indices

    # Use command line args if provided, otherwise use config
    indices_list = args.indices if args.indices else get_spectral_indices()
    emission_lines = (
        args.emission_lines if args.emission_lines else get_emission_lines()
    )

    # Store configured parameters in args for use by analysis functions
    args.configured_indices = indices_list
    args.configured_emission_lines = emission_lines

    # Initialize shared results
    p2p_results = None

    # Execute analysis
    try:
        # Check for existing P2P results if we're running VNB or RDB
        if args.mode in ["VNB", "RDB"] and args.auto_reuse:
            p2p_results_path = data_dir / f"{galaxy_name}_P2P_results.npz"
            std_results_path = data_dir / f"{galaxy_name}_P2P_standardized.npz"

            # Try both potential file paths
            if p2p_results_path.exists():
                logger.info(f"Found P2P results at {p2p_results_path}")
                try:
                    p2p_results = load_results_from_npz(p2p_results_path)
                    logger.info("Successfully loaded P2P results")
                except Exception as e:
                    logger.warning(f"Failed to load P2P results: {e}")
                    p2p_results = None
            elif std_results_path.exists():
                logger.info(f"Found standardized P2P results at {std_results_path}")
                try:
                    p2p_results = load_results_from_npz(std_results_path)
                    logger.info("Successfully loaded standardized P2P results")
                except Exception as e:
                    logger.warning(f"Failed to load standardized P2P results: {e}")
                    p2p_results = None

            if p2p_results is None:
                logger.warning(
                    "No P2P results found. VNB/RDB analyses may have limited functionality."
                )

        # P2P analysis
        if args.mode in ["P2P", "ALL"]:
            logger.info("Running P2P analysis with error propagation...")
            p2p_results = run_p2p_analysis(args, cube, Pmode=True)

            # Save results with errors
            save_results_to_npz(
                output_file=data_dir / f"{galaxy_name}_P2P_results.npz",
                data_dict=p2p_results,
                include_errors=True  # Ensure errors are saved
            )
            logger.info(
                f"Saved P2P results to {data_dir / f'{galaxy_name}_P2P_results.npz'}"
            )
            
            # Save error maps if requested
            if args.save_error_maps and "stellar_kinematics" in p2p_results:
                save_error_maps(p2p_results, error_dir, galaxy_name, "P2P")

        # VNB analysis
        if args.mode in ["VNB", "ALL"]:
            logger.info("Running VNB analysis with error propagation...")
            vnb_results = run_vnb_analysis(args, cube, p2p_results)

            # Save results with errors
            save_results_to_npz(
                output_file=data_dir / f"{galaxy_name}_VNB_results.npz",
                data_dict=vnb_results,
                include_errors=True
            )
            logger.info(
                f"Saved VNB results to {data_dir / f'{galaxy_name}_VNB_results.npz'}"
            )
            
            # Save error maps if requested
            if args.save_error_maps and "stellar_kinematics" in vnb_results:
                save_error_maps(vnb_results, error_dir, galaxy_name, "VNB")

        # RDB analysis
        if args.mode in ["RDB", "ALL"]:
            logger.info("Running RDB analysis with error propagation...")
            rdb_results = run_rdb_analysis(args, cube, p2p_results)

            # Save results with errors
            save_results_to_npz(
                output_file=data_dir / f"{galaxy_name}_RDB_results.npz",
                data_dict=rdb_results,
                include_errors=True
            )
            logger.info(
                f"Saved RDB results to {data_dir / f'{galaxy_name}_RDB_results.npz'}"
            )
            
            # Save error maps if requested
            if args.save_error_maps and "stellar_kinematics" in rdb_results:
                save_error_maps(rdb_results, error_dir, galaxy_name, "RDB")

        logger.info("ISAPC analysis with error propagation completed successfully")
        
        # Print error summary if verbose
        if args.verbose:
            print_error_summary(args)
            
        return 0

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


def save_error_maps(results, error_dir, galaxy_name, analysis_type):
    """Save error maps as separate files"""
    import numpy as np
    
    error_data = {}
    
    # Extract all error fields
    if "stellar_kinematics" in results:
        kin = results["stellar_kinematics"]
        if "velocity_error" in kin and kin["velocity_error"] is not None:
            error_data["velocity_error"] = kin["velocity_error"]
        if "dispersion_error" in kin and kin["dispersion_error"] is not None:
            error_data["dispersion_error"] = kin["dispersion_error"]
    
    if "stellar_population_errors" in results:
        for key, value in results["stellar_population_errors"].items():
            if value is not None:
                error_data[key] = value
    
    if "indices_errors" in results:
        for key, value in results["indices_errors"].items():
            if value is not None:
                error_data[f"index_{key}_error"] = value
    
    # Save error maps
    if error_data:
        error_file = error_dir / f"{galaxy_name}_{analysis_type}_error_maps.npz"
        np.savez_compressed(error_file, **error_data)
        logger.info(f"Saved error maps to {error_file}")


def print_error_summary(args):
    """Print summary of error propagation settings"""
    print("\n" + "="*60)
    print("ERROR PROPAGATION SUMMARY")
    print("="*60)
    print(f"Mode: {args.error_mode}")
    print(f"Monte Carlo samples: {args.n_monte_carlo}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Spatial covariance: {args.use_covariance}")
    print(f"Correlation length: {args.correlation_length} pixels")
    print(f"Error floor: {args.error_floor*100:.1f}%")
    print(f"Systematic error: {args.systematic_error*100:.1f}%")
    print(f"Correlation model: {args.error_correlation_model}")
    print("="*60 + "\n")


if __name__ == "__main__":
    sys.exit(main())