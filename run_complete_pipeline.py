#!/usr/bin/env python3
"""
Complete ISAPC Pipeline Runner
Runs the complete IFU spectroscopy analysis pipeline for all Virgo Cluster galaxies
"""

import os
import sys
import logging
import subprocess
import time
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Galaxy list with proper redshifts - TESTING WITH SINGLE GALAXY
GALAXIES = [
    {"name": "VCC0308", "redshift": 0.0055, "type": "dE"},
    # {"name": "VCC0667", "redshift": 0.0048, "type": "Sd"},
    # {"name": "VCC0688", "redshift": 0.0038, "type": "Sc"},
    # {"name": "VCC0990", "redshift": 0.0058, "type": "dS0(N)"},
    # {"name": "VCC1049", "redshift": 0.0021, "type": "dE(N)"},
    # {"name": "VCC1146", "redshift": 0.0023, "type": "E"},
    # {"name": "VCC1193", "redshift": 0.0025, "type": "Sd"},
    # {"name": "VCC1368", "redshift": 0.0035, "type": "SBa"},
    # {"name": "VCC1410", "redshift": 0.0054, "type": "Sd"},
    # {"name": "VCC1431", "redshift": 0.0050, "type": "dE"},
    # {"name": "VCC1486", "redshift": 0.0004, "type": "Sc"},
    # {"name": "VCC1499", "redshift": 0.0055, "type": "dE"},
    # {"name": "VCC1549", "redshift": 0.0046, "type": "dE(N)"},
    # {"name": "VCC1588", "redshift": 0.0042, "type": "Sd"},
    # {"name": "VCC1695", "redshift": 0.0058, "type": "dE"},
    # {"name": "VCC1811", "redshift": 0.0023, "type": "Sc"},
    # {"name": "VCC1890", "redshift": 0.0040, "type": "dE"},
    # {"name": "VCC1902", "redshift": 0.0038, "type": "SBa"},
    # {"name": "VCC1910", "redshift": 0.0007, "type": "dE(N)"},
    # {"name": "VCC1949", "redshift": 0.0058, "type": "dS0(N)"},
]

# Pipeline configuration
PIPELINE_CONFIG = {
    "template_file": "data/templates/spectra_emiles_9.0.npz",
    "output_dir": "output",
    "data_dir": "data/MUSE",
    "n_workers": 4,  # Parallel processing
    "modes": ["P2P", "VNB", "RDB"],  # Analysis modes
    "target_snr": 20.0,
    "min_snr": 1.0,
    "n_rings": 6,  # Number of radial rings
    "n_jobs": 4,  # Number of parallel jobs for individual analysis
    "vel_init": 0.0,
    "sigma_init": 50.0,
    "poly_degree": 3,
    "save_error_maps": True,
    "auto_reuse": True,
    "cvt": True,
    "physical_radius": True,
}

# Thread-safe counter for progress tracking
progress_lock = threading.Lock()
progress_counter = {"completed": 0, "total": len(GALAXIES)}

def run_single_galaxy(galaxy_info):
    """Run complete analysis for a single galaxy"""
    galaxy_name = galaxy_info["name"]
    redshift = galaxy_info["redshift"]
    galaxy_type = galaxy_info["type"]
    
    start_time = time.time()
    logger.info(f"Starting analysis for {galaxy_name} (z={redshift:.4f}, type={galaxy_type})")
    
    # Build file paths
    data_file = Path(PIPELINE_CONFIG["data_dir"]) / f"{galaxy_name}_stack.fits"
    template_file = Path(PIPELINE_CONFIG["template_file"])
    output_dir = Path(PIPELINE_CONFIG["output_dir"])
    
    # Check if data file exists
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return {"galaxy": galaxy_name, "status": "failed", "error": "Data file not found"}
    
    # Check if template file exists
    if not template_file.exists():
        logger.error(f"Template file not found: {template_file}")
        return {"galaxy": galaxy_name, "status": "failed", "error": "Template file not found"}
    
    try:
        # Build command for main.py using conda environment
        cmd = [
            "conda", "run", "-n", "Siqi_AstPy_312", "python", "main.py",
            str(data_file),
            "-z", str(redshift),
            "-t", str(template_file),
            "-o", str(output_dir),
            "-m", "ALL",  # Run all modes
            "--target-snr", str(PIPELINE_CONFIG["target_snr"]),
            "--min-snr", str(PIPELINE_CONFIG["min_snr"]),
            "--n-rings", str(PIPELINE_CONFIG["n_rings"]),
            "--n-jobs", str(PIPELINE_CONFIG["n_jobs"]),
            "--vel-init", str(PIPELINE_CONFIG["vel_init"]),
            "--sigma-init", str(PIPELINE_CONFIG["sigma_init"]),
            "--poly-degree", str(PIPELINE_CONFIG["poly_degree"]),
        ]
        
        # Add valid optional flags
        if PIPELINE_CONFIG["save_error_maps"]:
            cmd.append("--save-error-maps")
        if PIPELINE_CONFIG["auto_reuse"]:
            cmd.append("--auto-reuse")
        if PIPELINE_CONFIG["cvt"]:
            cmd.append("--cvt")
        if PIPELINE_CONFIG["physical_radius"]:
            cmd.append("--physical-radius")
        
        # Run the analysis
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per galaxy
        )
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully completed {galaxy_name} in {elapsed_time:.1f} seconds")
            
            # Update progress
            with progress_lock:
                progress_counter["completed"] += 1
                progress = progress_counter["completed"] / progress_counter["total"] * 100
                logger.info(f"Progress: {progress_counter['completed']}/{progress_counter['total']} ({progress:.1f}%)")
            
            return {
                "galaxy": galaxy_name,
                "status": "success",
                "elapsed_time": elapsed_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            logger.error(f"Analysis failed for {galaxy_name}")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return {
                "galaxy": galaxy_name,
                "status": "failed",
                "error": f"Return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    
    except subprocess.TimeoutExpired:
        logger.error(f"Analysis timed out for {galaxy_name}")
        return {"galaxy": galaxy_name, "status": "timeout", "error": "Analysis timed out"}
    except Exception as e:
        logger.error(f"Unexpected error for {galaxy_name}: {e}")
        return {"galaxy": galaxy_name, "status": "error", "error": str(e)}

def run_physics_visualization():
    """Run the physics visualization and alpha/Fe analysis"""
    logger.info("Starting physics visualization and alpha/Fe analysis...")
    
    try:
        # Run Phy_Visu.py using conda environment
        cmd = ["conda", "run", "-n", "Siqi_AstPy_312", "python", "Phy_Visu.py"]
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info("Physics visualization completed successfully")
            return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
        else:
            logger.error(f"Physics visualization failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return {"status": "failed", "error": f"Return code {result.returncode}"}
    
    except subprocess.TimeoutExpired:
        logger.error("Physics visualization timed out")
        return {"status": "timeout", "error": "Physics visualization timed out"}
    except Exception as e:
        logger.error(f"Unexpected error in physics visualization: {e}")
        return {"status": "error", "error": str(e)}

def create_summary_report(results):
    """Create a summary report of the pipeline run"""
    report_file = Path("pipeline_summary.txt")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    with open(report_file, "w") as f:
        f.write("ISAPC Pipeline Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total galaxies processed: {len(results)}\n")
        f.write(f"Successful analyses: {len(successful)}\n")
        f.write(f"Failed analyses: {len(failed)}\n")
        f.write(f"Success rate: {len(successful)/len(results)*100:.1f}%\n\n")
        
        if successful:
            f.write("Successful Galaxies:\n")
            f.write("-" * 20 + "\n")
            for result in successful:
                elapsed = result.get("elapsed_time", 0)
                f.write(f"  {result['galaxy']}: {elapsed:.1f} seconds\n")
            f.write("\n")
        
        if failed:
            f.write("Failed Galaxies:\n")
            f.write("-" * 15 + "\n")
            for result in failed:
                f.write(f"  {result['galaxy']}: {result['status']} - {result.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        f.write("Configuration Used:\n")
        f.write("-" * 18 + "\n")
        for key, value in PIPELINE_CONFIG.items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Summary report written to {report_file}")

def main():
    """Main pipeline runner"""
    logger.info("Starting ISAPC Complete Pipeline")
    logger.info(f"Processing {len(GALAXIES)} galaxies")
    logger.info(f"Using {PIPELINE_CONFIG['n_workers']} parallel workers")
    
    # Check that required files exist
    template_file = Path(PIPELINE_CONFIG["template_file"])
    if not template_file.exists():
        logger.error(f"Template file not found: {template_file}")
        return 1
    
    data_dir = Path(PIPELINE_CONFIG["data_dir"])
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(PIPELINE_CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    results = []
    
    # Process galaxies in parallel
    with ThreadPoolExecutor(max_workers=PIPELINE_CONFIG["n_workers"]) as executor:
        # Submit all jobs
        future_to_galaxy = {executor.submit(run_single_galaxy, galaxy): galaxy for galaxy in GALAXIES}
        
        # Collect results as they complete
        for future in as_completed(future_to_galaxy):
            galaxy = future_to_galaxy[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Exception for {galaxy['name']}: {e}")
                results.append({
                    "galaxy": galaxy["name"],
                    "status": "exception",
                    "error": str(e)
                })
    
    # Process physics visualization
    physics_result = run_physics_visualization()
    
    # Create summary report
    create_summary_report(results)
    
    # Final summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r["status"] == "success"])
    
    logger.info("Pipeline Complete!")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Successful analyses: {successful}/{len(GALAXIES)} ({successful/len(GALAXIES)*100:.1f}%)")
    logger.info(f"Physics visualization: {physics_result['status']}")
    
    return 0 if successful > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
