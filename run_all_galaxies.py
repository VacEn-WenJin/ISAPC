#!/usr/bin/env python3
"""
Complete ISAPC + Physics Workflow for All Virgo Galaxies
Runs ISAPC analysis for all available galaxies, then physics visualization
"""

import os
import sys
import subprocess
import time
import datetime
import logging
from pathlib import Path

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

def find_all_muse_galaxies():
    """Find all available MUSE galaxy data files"""
    muse_dir = Path("./data/MUSE")
    
    if not muse_dir.exists():
        logging.error("MUSE data directory not found!")
        return []
    
    # Find all FITS files
    fits_files = list(muse_dir.glob("*.fits"))
    
    galaxies = []
    for fits_file in fits_files:
        # Extract galaxy name (remove _stack.fits or .fits)
        galaxy_name = fits_file.stem
        if galaxy_name.endswith("_stack"):
            galaxy_name = galaxy_name.replace("_stack", "")
        
        galaxies.append({
            'name': galaxy_name,
            'file': fits_file,
            'size_gb': fits_file.stat().st_size / (1024**3)
        })
    
    # Sort by name
    galaxies.sort(key=lambda x: x['name'])
    
    logging.info(f"Found {len(galaxies)} MUSE galaxies:")
    for g in galaxies:
        logging.info(f"  {g['name']}: {g['file'].name} ({g['size_gb']:.1f} GB)")
    
    return galaxies

def get_galaxy_redshift(galaxy_name):
    """Get redshift for a galaxy"""
    # Common Virgo cluster redshifts - Update these with actual values!
    virgo_redshifts = {
        'VCC0308': 0.0036,
        'VCC0667': 0.0034,
        'VCC0688': 0.0031,
        'VCC0990': 0.0029,
        'VCC1049': 0.0033,
        'VCC1146': 0.0035,
        'VCC1193': 0.0032,
        'VCC1368': 0.0037,
        'VCC1410': 0.0034,
        'VCC1431': 0.0031,
        'VCC1588': 0.0028,
        'VCC1627': 0.0032,
        'VCC1826': 0.0035,
        'VCC1545': 0.0031,
        'VCC1833': 0.0034,
        'VCC2095': 0.0030,
    }
    
    # Default Virgo cluster redshift if not found
    default_redshift = 0.0033
    
    redshift = virgo_redshifts.get(galaxy_name, default_redshift)
    logging.info(f"Using redshift {redshift} for {galaxy_name}")
    
    return redshift

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
    
    existing_files = []
    missing_files = []
    
    for filename in required_files:
        file_path = data_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            existing_files.append(f"{filename} ({size_mb:.1f} MB)")
        else:
            missing_files.append(filename)
    
    if missing_files:
        status = f"Missing: {missing_files}"
        if existing_files:
            status += f", Existing: {existing_files}"
        return False, status
    
    return True, f"Complete: {existing_files}"

def run_isapc_for_galaxy(galaxy_info):
    """Run ISAPC analysis for a single galaxy using subprocess"""
    galaxy_name = galaxy_info['name']
    fits_file = galaxy_info['file']
    
    logging.info(f"Starting ISAPC analysis for {galaxy_name}")
    logging.info(f"Input file: {fits_file} ({galaxy_info['size_gb']:.1f} GB)")
    
    # Get galaxy parameters
    redshift = get_galaxy_redshift(galaxy_name)
    template_file = "./data/templates/spectra_emiles_9.0.npz"
    output_dir = f"./output/{galaxy_name}_stack"
    
    # Prepare ISAPC command
    cmd = [
        sys.executable, "main.py",
        str(fits_file),
        "-z", str(redshift),
        "-t", template_file,
        "-o", output_dir,
        "-m", "ALL",  # Run all analysis modes
        "--error-propagation",  # Enable error propagation
        "--correlation-length", "3.0",
        "--error-floor", "0.01",
        "--systematic-error", "0.02"
    ]
    
    logging.info(f"ISAPC command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        
        # Run ISAPC
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per galaxy
        )
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            logging.info(f"{galaxy_name}: ISAPC completed successfully in {runtime/60:.1f} minutes")
            # Only log first 1000 chars of stdout to avoid spam
            stdout_preview = result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout
            logging.debug(f"{galaxy_name}: STDOUT preview:\n{stdout_preview}")
            return True
        else:
            logging.error(f"{galaxy_name}: ISAPC failed with return code {result.returncode}")
            logging.error(f"{galaxy_name}: STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"{galaxy_name}: ISAPC timed out after 2 hours")
        return False
    except Exception as e:
        logging.error(f"{galaxy_name}: ISAPC failed with exception: {e}")
        return False

def run_physics_analysis():
    """Run physics visualization analysis for all completed galaxies"""
    logging.info("Starting physics visualization analysis")
    
    try:
        cmd = [sys.executable, "run_complete_physics_analysis.py"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for physics analysis
        )
        
        if result.returncode == 0:
            logging.info("Physics analysis completed successfully")
            return True
        else:
            logging.error(f"Physics analysis failed with return code {result.returncode}")
            logging.error(f"Physics analysis stderr:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("Physics analysis timed out after 1 hour")
        return False
    except Exception as e:
        logging.error(f"Physics analysis failed: {e}")
        return False

def main():
    """Main workflow function"""
    print("=" * 80)
    print("Complete ISAPC + Physics Analysis Workflow")
    print("Processing all Virgo cluster galaxies")
    print("=" * 80)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting complete ISAPC + Physics workflow")
    logging.info(f"Log file: {log_file}")
    
    start_time = time.time()
    
    # Check prerequisites
    if not Path("main.py").exists():
        logging.error("main.py not found! Run from ISAPC directory")
        return False
    
    if not Path("data/templates/spectra_emiles_9.0.npz").exists():
        logging.error("Template file not found!")
        return False
    
    # Find all galaxies
    logging.info("=" * 60)
    logging.info("STEP 1: Finding MUSE galaxies")
    logging.info("=" * 60)
    
    galaxies = find_all_muse_galaxies()
    if not galaxies:
        logging.error("No MUSE galaxies found!")
        return False
    
    # Check existing completions
    logging.info("=" * 60)
    logging.info("STEP 2: Checking existing completions")
    logging.info("=" * 60)
    
    completed_galaxies = []
    incomplete_galaxies = []
    
    for galaxy_info in galaxies:
        galaxy_name = galaxy_info['name']
        is_complete, status = check_galaxy_completion(galaxy_name)
        
        if is_complete:
            logging.info(f"{galaxy_name}: ✓ {status}")
            completed_galaxies.append(galaxy_info)
        else:
            logging.info(f"{galaxy_name}: ✗ {status}")
            incomplete_galaxies.append(galaxy_info)
    
    logging.info(f"Already completed: {len(completed_galaxies)}")
    logging.info(f"Need processing: {len(incomplete_galaxies)}")
    
    # Run ISAPC for incomplete galaxies
    if incomplete_galaxies:
        logging.info("=" * 60)
        logging.info("STEP 3: Running ISAPC analysis for incomplete galaxies")
        logging.info("=" * 60)
        
        newly_completed = []
        failed_galaxies = []
        
        for i, galaxy_info in enumerate(incomplete_galaxies, 1):
            galaxy_name = galaxy_info['name']
            
            logging.info(f"\n--- Galaxy {i}/{len(incomplete_galaxies)}: {galaxy_name} ---")
            
            success = run_isapc_for_galaxy(galaxy_info)
            
            if success:
                # Verify completion
                is_complete, status = check_galaxy_completion(galaxy_name)
                if is_complete:
                    logging.info(f"{galaxy_name}: ✓ Analysis completed and verified")
                    newly_completed.append(galaxy_info)
                    completed_galaxies.append(galaxy_info)
                else:
                    logging.error(f"{galaxy_name}: ✗ Analysis completed but verification failed: {status}")
                    failed_galaxies.append(galaxy_info)
            else:
                logging.error(f"{galaxy_name}: ✗ Analysis failed")
                failed_galaxies.append(galaxy_info)
        
        logging.info(f"Newly completed: {len(newly_completed)}")
        logging.info(f"Failed: {len(failed_galaxies)}")
    
    # Run physics analysis
    logging.info("=" * 60)
    logging.info("STEP 4: Running physics visualization analysis")
    logging.info("=" * 60)
    
    if completed_galaxies:
        logging.info(f"Running physics analysis on {len(completed_galaxies)} completed galaxies:")
        for g in completed_galaxies:
            logging.info(f"  - {g['name']}")
        
        physics_success = run_physics_analysis()
    else:
        logging.error("No completed galaxies available for physics analysis!")
        physics_success = False
    
    # Final summary
    total_time = time.time() - start_time
    
    logging.info("=" * 60)
    logging.info("FINAL WORKFLOW SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total galaxies found: {len(galaxies)}")
    logging.info(f"Successfully completed ISAPC: {len(completed_galaxies)}")
    logging.info(f"Physics analysis: {'SUCCESS' if physics_success else 'FAILED'}")
    logging.info(f"Total runtime: {total_time/3600:.2f} hours")
    
    if completed_galaxies:
        logging.info(f"Completed galaxies:")
        for g in completed_galaxies:
            logging.info(f"  - {g['name']}")
    
    logging.info(f"Log file: {log_file}")
    
    # Overall success
    overall_success = len(completed_galaxies) > 0 and physics_success
    logging.info(f"OVERALL WORKFLOW: {'SUCCESS' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
