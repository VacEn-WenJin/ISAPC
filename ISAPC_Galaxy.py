#!/usr/bin/env python3
"""
ISAPC Galaxy Analysis Module
Automates galaxy analysis with three modes: P2P, VNB, and RDB using main.py
Enhanced to support error propagation arguments
"""

import argparse
import logging
import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ISAPC")

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


def parse_args():
    """Parse command line arguments including pass-through for error options"""
    parser = argparse.ArgumentParser(
        description="ISAPC Galaxy Analysis v2.0 - Analyze galaxies using P2P, VNB, and RDB",
        epilog="Any additional arguments will be passed through to main.py (e.g., error propagation options)"
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
    
    # Analysis mode selection
    parser.add_argument(
        "--mode", 
        choices=["ALL", "P2P", "VNB", "RDB"], 
        default="ALL",
        help="Analysis mode: P2P, VNB, RDB, or ALL (default)"
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
    parser.add_argument(
        "--high-snr-mode", action="store_true", help="Use high-SNR optimization"
    )
    
    # Performance options
    parser.add_argument(
        "-j", "--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)"
    )
    
    # Parse known args and collect unknown ones for pass-through
    args, unknown = parser.parse_known_args()
    
    # Store unknown arguments to pass to main.py
    args.pass_through_args = unknown
    
    # Log if we have pass-through arguments
    if unknown:
        logger.info(f"Additional arguments to pass to main.py: {' '.join(unknown)}")
    
    return args


def run_main_py(args, galaxy_file, redshift, output_dir=None, has_emission=True):
    """
    Run main.py with the given arguments including pass-through error options
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    galaxy_file : str
        Galaxy file path
    redshift : float
        Redshift value
    output_dir : str, optional
        Custom output directory
    has_emission : bool, default=True
        Whether galaxy has emission lines
        
    Returns
    -------
    int
        Return code from main.py (0 for success)
    """
    # Use specified output_dir if provided, otherwise use args.output_dir
    if output_dir is None:
        output_dir = args.output_dir
    
    # Build command
    cmd = [
        sys.executable,
        "main.py",
        galaxy_file,
        "--redshift", str(redshift),
        "--template", args.template,
        "--output-dir", output_dir,
        "--mode", args.mode,
        "--n-jobs", str(args.n_jobs)
    ]
    
    # Add optional parameters
    if args.vel_init != 0.0:
        cmd.extend(["--vel-init", str(args.vel_init)])
    
    if args.sigma_init != 50.0:
        cmd.extend(["--sigma-init", str(args.sigma_init)])
    
    if args.poly_degree != 3:
        cmd.extend(["--poly-degree", str(args.poly_degree)])
    
    if args.target_snr != 20.0:
        cmd.extend(["--target-snr", str(args.target_snr)])
    
    if args.min_snr != 0.1:
        cmd.extend(["--min-snr", str(args.min_snr)])
    
    if args.n_rings != 10:
        cmd.extend(["--n-rings", str(args.n_rings)])
    
    if args.log_spacing:
        cmd.append("--log-spacing")
    
    if args.pa is not None:
        cmd.extend(["--pa", str(args.pa)])
    
    if args.ellipticity is not None:
        cmd.extend(["--ellipticity", str(args.ellipticity)])
    
    if args.center_x is not None:
        cmd.extend(["--center-x", str(args.center_x)])
    
    if args.center_y is not None:
        cmd.extend(["--center-y", str(args.center_y)])
    
    if args.physical_radius:
        cmd.append("--physical-radius")
    
    if args.high_snr_mode:
        cmd.append("--high-snr-mode")
    
    if args.no_emission or not has_emission:
        cmd.append("--no-emission")
    
    if args.no_indices:
        cmd.append("--no-indices")
    
    if args.equal_aspect:
        cmd.append("--equal-aspect")
    
    # Add pass-through arguments (error options, etc.)
    if hasattr(args, 'pass_through_args') and args.pass_through_args:
        cmd.extend(args.pass_through_args)
        logger.info(f"Passing through additional arguments: {' '.join(args.pass_through_args)}")
    
    # Run the command
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Log output
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(f"main.py: {line}")
    
    if result.stderr:
        for line in result.stderr.splitlines():
            if "ERROR" in line or "Error" in line or "error" in line:
                logger.error(f"main.py: {line}")
            else:
                logger.warning(f"main.py: {line}")
    
    return result.returncode


def run_galaxy_analysis(args):
    """
    Run galaxy analysis using main.py
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    bool
        True if analysis was successful
    """
    # Check if filename is provided
    if args.filename is None:
        logger.error("No input filename provided")
        return False
    
    # Get galaxy name from filename
    galaxy_name = Path(args.filename).stem
    # Remove " stack" suffix if present
    if galaxy_name.endswith(" stack"):
        galaxy_name = galaxy_name[:-6]
    logger.info(f"Processing galaxy: {galaxy_name}")
    
    # Determine redshift - either from args or try to extract from filename
    redshift = args.redshift
    has_emission = True
    if redshift is None:
        # Try to match against predefined galaxies
        for galaxy in GALAXIES:
            if galaxy_name == galaxy["name"]:
                redshift = galaxy["redshift"]
                has_emission = galaxy["has_emission"]
                logger.info(f"Using redshift {redshift} from predefined galaxies")
                break
        
        if redshift is None:
            logger.error("No redshift provided or found for this galaxy")
            return False
    
    # Check if we should skip if results already exist
    if args.skip_existing:
        output_dir = Path(args.output_dir) / galaxy_name / "Data"
        results_files = [
            output_dir / f"{galaxy_name}_P2P_results.npz",
            output_dir / f"{galaxy_name}_VNB_results.npz",
            output_dir / f"{galaxy_name}_RDB_results.npz",
        ]
        
        # Check if results exist for the requested modes
        if args.mode == "ALL" and all(f.exists() for f in results_files):
            logger.info(f"Skipping {galaxy_name}: ALL results already exist")
            return True
        elif args.mode == "P2P" and results_files[0].exists():
            logger.info(f"Skipping {galaxy_name}: P2P results already exist")
            return True
        elif args.mode == "VNB" and results_files[1].exists():
            logger.info(f"Skipping {galaxy_name}: VNB results already exist")
            return True
        elif args.mode == "RDB" and results_files[2].exists():
            logger.info(f"Skipping {galaxy_name}: RDB results already exist")
            return True
    
    # Run main.py
    result = run_main_py(args, args.filename, redshift, has_emission=has_emission)
    
    if result == 0:
        logger.info(f"Successfully analyzed galaxy {galaxy_name}")
        return True
    else:
        logger.error(f"Failed to analyze galaxy {galaxy_name}")
        return False


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
        tracking_data = {}
        if tracking_file.exists():
            import csv
            try:
                with open(tracking_file, 'r') as f:
                    reader = csv.DictReader(f)
                    tracking_data = {row['filename']: row for row in reader}
            except Exception as e:
                logger.warning(f"Error loading tracking file: {e}")
        
        # Process each galaxy
        success_count = 0
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
                    continue
            
            galaxy_name = galaxy_path.stem
            # Remove " stack" suffix if present
            if galaxy_name.endswith(" stack"):
                galaxy_name = galaxy_name[:-6]
            logger.info(f"Processing galaxy {i+1}/{len(galaxy_files)}: {galaxy_name}")
            
            # Check if already processed and skip_existing is enabled
            if args.skip_existing and galaxy_file in tracking_data:
                status = tracking_data[galaxy_file].get('status', '')
                if status == 'completed':
                    logger.info(f"Skipping {galaxy_name} - already processed")
                    success_count += 1
                    continue
            
            # Update args with current filename
            args.filename = str(galaxy_path)
            
            # Try to find redshift from filename (VCC naming convention)
            args.redshift = None
            has_emission = True
            for galaxy in GALAXIES:
                if galaxy_name == galaxy["name"]:
                    args.redshift = galaxy["redshift"]
                    has_emission = galaxy["has_emission"]
                    break
            
            # Run analysis
            custom_output_dir = f"{args.output_dir}/{galaxy_name}"
            result = run_main_py(
                args,
                str(galaxy_path),
                args.redshift,
                output_dir=custom_output_dir,
                has_emission=has_emission
            )
            
            if result == 0:
                success_count += 1
                # Update tracking data
                tracking_data[galaxy_file] = {
                    'filename': galaxy_file,
                    'galaxy_name': galaxy_name,
                    'status': 'completed',
                    'timestamp': time.time(),
                }
            else:
                # Update tracking data
                tracking_data[galaxy_file] = {
                    'filename': galaxy_file,
                    'galaxy_name': galaxy_name,
                    'status': 'failed',
                    'timestamp': time.time(),
                }
            
            # Update tracking file after each galaxy
            try:
                import csv
                with open(tracking_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'galaxy_name', 'status', 'timestamp'])
                    writer.writeheader()
                    for data in tracking_data.values():
                        writer.writerow(data)
            except Exception as e:
                logger.warning(f"Error updating tracking file: {e}")
        
        logger.info(f"Processed {len(galaxy_files)} galaxies: {success_count} successful, {len(galaxy_files) - success_count} failed")
        return success_count == len(galaxy_files)
    
    except Exception as e:
        logger.error(f"Error processing galaxy list: {e}")
        return False


def process_predefined_galaxies(args):
    """
    Process galaxies from the predefined list using main.py
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    galaxies = GALAXIES
    
    if args.base_dir is None:
        base_dir = Path.cwd() / "data" / "MUSE"  # Default to data/MUSE subdirectory
    else:
        base_dir = Path(args.base_dir)
    
    logger.info(f"Processing {len(galaxies)} galaxies from predefined list")
    if args.pass_through_args:
        logger.info(f"With error propagation options: {' '.join(args.pass_through_args)}")
    
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
    
    # Process each galaxy
    success_count = 0
    for i, galaxy in enumerate(galaxies):
        galaxy_name = galaxy["name"]
        logger.info(f"Processing galaxy {i+1}/{len(galaxies)}: {galaxy_name} ({galaxy['type']}, z={galaxy['redshift']})")
        
        # Check if already processed and skip_existing is enabled
        if args.skip_existing and galaxy_name in tracking_data:
            status = tracking_data[galaxy_name].get('status', '')
            if status == 'completed':
                logger.info(f"Skipping {galaxy_name} - already processed")
                success_count += 1
                continue
        
        # Look for galaxy file with different naming patterns
        galaxy_paths = [
            # Standard FITS file
            base_dir / f"{galaxy_name}.fits",
            # Stack file format as shown in your command
            base_dir / f"{galaxy_name} stack.fits",
            base_dir / f"{galaxy_name}_stack.fits",
            # With different extensions
            base_dir / f"{galaxy_name}.FITS",
            base_dir / f"{galaxy_name}.fit",
            # Alternative naming conventions
            base_dir / f"{galaxy_name}_MUSE.fits",
            # In subdirectories by galaxy name
            base_dir / galaxy_name / f"{galaxy_name}.fits",
            base_dir / galaxy_name / f"{galaxy_name}_stack.fits",
            base_dir / galaxy_name / f"{galaxy_name} stack.fits",
        ]
        
        # Find first valid file
        galaxy_path = None
        for path in galaxy_paths:
            if path.exists():
                galaxy_path = path
                logger.info(f"Found galaxy file at: {path}")
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
            }
            continue
        
        # Run main.py directly with galaxy information
        # Set emission line handling based on galaxy data
        logger.info(f"Running main.py for {galaxy_name} with file: {galaxy_path}")
        
        # Define output directory for this galaxy
        galaxy_output_dir = str(output_dir / galaxy_name)
        
        # Run main.py using our helper function
        result = run_main_py(
            args,
            str(galaxy_path),
            galaxy['redshift'],
            output_dir=galaxy_output_dir,
            has_emission=galaxy['has_emission']
        )
        
        # Check result
        if result == 0:
            logger.info(f"Successfully processed galaxy {galaxy_name}")
            success_count += 1
            
            # Update tracking data
            tracking_data[galaxy_name] = {
                'name': galaxy_name,
                'type': galaxy['type'],
                'redshift': galaxy['redshift'],
                'has_emission': str(galaxy['has_emission']),
                'status': 'completed',
                'timestamp': time.time(),
            }
        else:
            logger.error(f"Failed to process galaxy {galaxy_name}")
            
            # Update tracking data
            tracking_data[galaxy_name] = {
                'name': galaxy_name,
                'type': galaxy['type'],
                'redshift': galaxy['redshift'],
                'has_emission': str(galaxy['has_emission']),
                'status': 'failed',
                'timestamp': time.time(),
            }
        
        # Update tracking file after each galaxy
        try:
            import csv
            with open(tracking_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['name', 'type', 'redshift', 'has_emission', 'status', 'timestamp'])
                writer.writeheader()
                for data in tracking_data.values():
                    writer.writerow(data)
        except Exception as e:
            logger.warning(f"Error updating tracking file: {e}")
    
    logger.info(f"Processed {len(galaxies)} galaxies: {success_count} successful, {len(galaxies) - success_count} failed")
    return success_count == len(galaxies)


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Track whether no_emission was explicitly set
    args.no_emission_set = "--no-emission" in sys.argv
    
    # Set template file
    if not args.template:
        # Try to find template file in the current directory
        default_templates = [
            "pp_ssp.fits",
            "templates/pp_ssp.fits",
            "./templates/spectra_emiles_9.0.npz",
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
    
    # Log if we're using error propagation
    if args.pass_through_args:
        error_args = [arg for arg in args.pass_through_args if 'error' in arg]
        if error_args:
            logger.info("Error propagation enabled with options:")
            for i in range(0, len(args.pass_through_args), 2):
                if i+1 < len(args.pass_through_args):
                    logger.info(f"  {args.pass_through_args[i]} {args.pass_through_args[i+1]}")
                else:
                    logger.info(f"  {args.pass_through_args[i]}")
    
    # Check which mode to use
    if args.use_predefined:
        # Use predefined galaxy list
        process_predefined_galaxies(args)
    elif args.galaxy_list:
        # Use custom galaxy list file
        logger.info(f"Processing galaxy list from {args.galaxy_list}")
        process_galaxy_list(args)
    else:
        # Run single galaxy analysis
        success = run_galaxy_analysis(args)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()