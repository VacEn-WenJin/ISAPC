#!/usr/bin/env python3
"""
Complete ISAPC Pipeline for All Virgo Galaxies
Step 1: Run full ISAPC analysis (P2P, VNB, RDB) for all available galaxies
Step 2: Run physics visualization analysis on all completed results
"""

import os
import sys
import glob
import time
import datetime
import logging
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from run_complete_pipeline import main as run_isapc_pipeline

def setup_logging():
    """Setup logging for the complete workflow"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"complete_workflow_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return str(log_file)

def find_all_galaxies_for_analysis():
    """Find all galaxies available for ISAPC analysis"""
    # Look for MUSE data files to identify available galaxies
    data_dir = Path("./data/MUSE")
    
    if not data_dir.exists():
        logging.error("MUSE data directory not found!")
        return []
    
    # Find all galaxy directories in MUSE data
    galaxy_patterns = [
        "VCC*.fits",
        "NGC*.fits", 
        "*_*.fits"
    ]
    
    galaxies = set()
    
    for pattern in galaxy_patterns:
        files = list(data_dir.glob(pattern))
        for file_path in files:
            # Extract galaxy name from filename (remove .fits extension)
            galaxy_name = file_path.stem
            # Remove common suffixes like _cube, _stack, etc.
            for suffix in ['_cube', '_stack', '_combined']:
                if galaxy_name.endswith(suffix):
                    galaxy_name = galaxy_name.replace(suffix, '')
            galaxies.add(galaxy_name)
    
    # Also check for any existing directories that might have analysis
    existing_dirs = list(Path("./output").glob("*_stack")) if Path("./output").exists() else []
    for dir_path in existing_dirs:
        galaxy_name = dir_path.name.replace("_stack", "")
        galaxies.add(galaxy_name)
    
    galaxy_list = sorted(list(galaxies))
    logging.info(f"Found {len(galaxy_list)} galaxies for analysis: {galaxy_list}")
    
    return galaxy_list

def check_galaxy_completion(galaxy_name):
    """Check if a galaxy has completed ISAPC analysis"""
    output_dir = Path(f"./output/{galaxy_name}_stack")
    
    if not output_dir.exists():
        return False, "No output directory"
    
    data_dir = output_dir / "Data"
    if not data_dir.exists():
        return False, "No data directory"
    
    # Check for required files
    required_files = [
        f"{galaxy_name}_stack_P2P_results.npz",
        f"{galaxy_name}_stack_VNB_results.npz", 
        f"{galaxy_name}_stack_RDB_results.npz"
    ]
    
    missing_files = []
    for filename in required_files:
        if not (data_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        return False, f"Missing files: {missing_files}"
    
    return True, "Complete"

def run_isapc_for_galaxy(galaxy_name):
    """Run ISAPC analysis for a single galaxy"""
    logging.info(f"Starting ISAPC analysis for {galaxy_name}")
    
    try:
        # Here you would call your existing ISAPC pipeline
        # For now, I'll create a placeholder that calls the main pipeline
        
        # Check if galaxy data exists
        data_file = Path(f"./data/MUSE/{galaxy_name}.fits")
        if not data_file.exists():
            # Try alternative naming patterns
            alt_patterns = [
                f"{galaxy_name}_cube.fits",
                f"{galaxy_name}_stack.fits",
                f"{galaxy_name}_combined.fits"
            ]
            
            data_file = None
            for pattern in alt_patterns:
                test_file = Path(f"./data/MUSE/{pattern}")
                if test_file.exists():
                    data_file = test_file
                    break
            
            if data_file is None:
                logging.error(f"No MUSE data file found for {galaxy_name}")
                return False
        
        # Run ISAPC pipeline (this would be your existing main analysis)
        # You need to modify this to call your actual ISAPC main function
        # with the specific galaxy name
        
        # For now, simulate the call:
        logging.info(f"Running ISAPC pipeline for {galaxy_name}...")
        logging.info(f"Input data: {data_file}")
        
        # TODO: Replace this with actual ISAPC pipeline call
        # success = run_isapc_pipeline(galaxy_name=galaxy_name, input_file=str(data_file))
        
        # Placeholder for testing - you'll need to implement the actual call
        logging.warning(f"ISAPC pipeline call for {galaxy_name} - PLACEHOLDER")
        logging.info("You need to implement the actual ISAPC pipeline call here")
        
        return False  # Set to True when actual implementation is done
        
    except Exception as e:
        logging.error(f"Error running ISAPC for {galaxy_name}: {e}")
        return False

def run_physics_analysis_for_all():
    """Run physics visualization analysis for all completed galaxies"""
    logging.info("Starting physics visualization analysis for all completed galaxies")
    
    try:
        # Import and run the physics analysis
        from run_complete_physics_analysis import main as run_physics_main
        success = run_physics_main()
        return success
    except Exception as e:
        logging.error(f"Error in physics analysis: {e}")
        return False

def main():
    """Main workflow function"""
    print("=" * 80)
    print("Complete ISAPC + Physics Analysis Workflow")
    print("Step 1: Run ISAPC analysis for all galaxies")
    print("Step 2: Run physics visualization analysis")
    print("=" * 80)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting complete ISAPC + Physics workflow")
    logging.info(f"Log file: {log_file}")
    
    start_time = time.time()
    
    # Step 1: Find all galaxies
    logging.info("=" * 60)
    logging.info("STEP 1: Finding galaxies for ISAPC analysis")
    logging.info("=" * 60)
    
    galaxies = find_all_galaxies_for_analysis()
    if not galaxies:
        logging.error("No galaxies found for analysis!")
        return False
    
    # Step 2: Check existing completions and run ISAPC where needed
    logging.info("=" * 60)
    logging.info("STEP 2: Running ISAPC analysis for incomplete galaxies")
    logging.info("=" * 60)
    
    completed_galaxies = []
    failed_galaxies = []
    
    for i, galaxy_name in enumerate(galaxies, 1):
        logging.info(f"\n--- Galaxy {i}/{len(galaxies)}: {galaxy_name} ---")
        
        # Check if already completed
        is_complete, status = check_galaxy_completion(galaxy_name)
        
        if is_complete:
            logging.info(f"{galaxy_name}: Already completed - {status}")
            completed_galaxies.append(galaxy_name)
        else:
            logging.info(f"{galaxy_name}: Needs analysis - {status}")
            
            # Run ISAPC analysis
            success = run_isapc_for_galaxy(galaxy_name)
            
            if success:
                # Verify completion
                is_complete, status = check_galaxy_completion(galaxy_name)
                if is_complete:
                    logging.info(f"{galaxy_name}: ISAPC analysis completed successfully")
                    completed_galaxies.append(galaxy_name)
                else:
                    logging.error(f"{galaxy_name}: ISAPC analysis failed verification - {status}")
                    failed_galaxies.append(galaxy_name)
            else:
                logging.error(f"{galaxy_name}: ISAPC analysis failed")
                failed_galaxies.append(galaxy_name)
    
    # Step 3: Run physics analysis on all completed galaxies
    logging.info("=" * 60)
    logging.info("STEP 3: Running physics visualization analysis")
    logging.info("=" * 60)
    
    logging.info(f"Completed galaxies: {len(completed_galaxies)}")
    logging.info(f"Failed galaxies: {len(failed_galaxies)}")
    
    if completed_galaxies:
        logging.info(f"Running physics analysis on {len(completed_galaxies)} completed galaxies")
        physics_success = run_physics_analysis_for_all()
    else:
        logging.error("No completed galaxies available for physics analysis!")
        physics_success = False
    
    # Final summary
    total_time = time.time() - start_time
    
    logging.info("=" * 60)
    logging.info("WORKFLOW SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total galaxies found: {len(galaxies)}")
    logging.info(f"Successfully completed ISAPC: {len(completed_galaxies)}")
    logging.info(f"Failed ISAPC: {len(failed_galaxies)}")
    logging.info(f"Physics analysis: {'SUCCESS' if physics_success else 'FAILED'}")
    logging.info(f"Total runtime: {total_time/3600:.2f} hours")
    
    if completed_galaxies:
        logging.info(f"Completed galaxies: {', '.join(completed_galaxies)}")
    if failed_galaxies:
        logging.info(f"Failed galaxies: {', '.join(failed_galaxies)}")
    
    logging.info(f"Log file: {log_file}")
    
    # Overall success if we have any completed galaxies and physics worked
    overall_success = len(completed_galaxies) > 0 and physics_success
    
    logging.info(f"OVERALL WORKFLOW: {'SUCCESS' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
