#!/usr/bin/env python3
"""
ISAPC Data Status Checker and Issue Fixer
Checks the current state of your data and fixes common issues
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from astropy.io import fits
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_data_files():
    """Check the status of MUSE data files"""
    logger.info("Checking MUSE data files...")
    
    data_dir = Path("data/MUSE")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    galaxies = [
        "VCC0308", "VCC0667", "VCC0688", "VCC0990", "VCC1049", "VCC1146",
        "VCC1193", "VCC1368", "VCC1410", "VCC1431", "VCC1486", "VCC1499",
        "VCC1549", "VCC1588", "VCC1695", "VCC1811", "VCC1890",
        "VCC1902", "VCC1910", "VCC1949"
    ]
    
    missing_files = []
    valid_files = []
    
    for galaxy in galaxies:
        filename = data_dir / f"{galaxy}_stack.fits"
        if filename.exists():
            try:
                # Try to open and check the file
                with fits.open(filename) as hdul:
                    # Check if it has data
                    if len(hdul) > 0 and hdul[0].data is not None:
                        shape = hdul[0].data.shape
                        logger.info(f"✓ {galaxy}: {shape}")
                        valid_files.append(galaxy)
                    else:
                        logger.warning(f"⚠ {galaxy}: No data in file")
                        missing_files.append(galaxy)
            except Exception as e:
                logger.error(f"✗ {galaxy}: Error reading file - {e}")
                missing_files.append(galaxy)
        else:
            logger.error(f"✗ {galaxy}: File not found")
            missing_files.append(galaxy)
    
    logger.info(f"Summary: {len(valid_files)} valid files, {len(missing_files)} missing/invalid files")
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
    
    return len(valid_files) > 0

def check_template_file():
    """Check if the template file exists"""
    logger.info("Checking template file...")
    
    template_file = Path("data/templates/spectra_emiles_9.0.npz")
    if template_file.exists():
        try:
            data = np.load(template_file)
            logger.info(f"✓ Template file found with keys: {list(data.keys())}")
            return True
        except Exception as e:
            logger.error(f"✗ Error loading template file: {e}")
            return False
    else:
        logger.error(f"✗ Template file not found: {template_file}")
        return False

def check_tmb03_model():
    """Check if TMB03 model files exist"""
    logger.info("Checking TMB03 model files...")
    
    tmb03_dir = Path("TMB03")
    if not tmb03_dir.exists():
        logger.error(f"TMB03 directory not found: {tmb03_dir}")
        return False
    
    # Check main model file
    main_model = tmb03_dir / "TMB03.csv"
    if main_model.exists():
        try:
            df = pd.read_csv(main_model)
            logger.info(f"✓ TMB03.csv found with {len(df)} rows and columns: {list(df.columns)}")
            
            # Check for required columns
            required_cols = ['Fe5015', 'Mgb', 'Hb', 'AoFe']  # Note: Hb instead of Hbeta
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in TMB03.csv: {missing_cols}")
            else:
                logger.info("✓ All required columns found in TMB03.csv")
                
            return True
        except Exception as e:
            logger.error(f"✗ Error reading TMB03.csv: {e}")
            return False
    else:
        logger.error(f"✗ TMB03.csv not found: {main_model}")
        return False

def check_output_structure():
    """Check the output directory structure"""
    logger.info("Checking output directory structure...")
    
    output_dir = Path("output")
    if not output_dir.exists():
        logger.info("Creating output directory...")
        output_dir.mkdir(exist_ok=True)
    
    # Check for existing results
    result_files = list(output_dir.glob("**/Data/*_results.npz"))
    if result_files:
        logger.info(f"Found {len(result_files)} existing result files")
        for result_file in result_files[:5]:  # Show first 5
            logger.info(f"  - {result_file}")
        if len(result_files) > 5:
            logger.info(f"  ... and {len(result_files) - 5} more")
    else:
        logger.info("No existing result files found")
    
    return True

def check_dependencies():
    """Check if all required Python packages are installed"""
    logger.info("Checking Python dependencies...")
    
    required_packages = [
        "numpy", "pandas", "matplotlib", "scipy", "astropy", 
        "ppxf", "vorbin", "joblib", "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def fix_common_issues():
    """Fix common issues in the codebase"""
    logger.info("Fixing common issues...")
    
    # Fix 1: Ensure proper import paths
    init_file = Path("utils/__init__.py")
    if not init_file.exists():
        init_file.parent.mkdir(exist_ok=True)
        init_file.write_text("# Utils package\n")
        logger.info("✓ Created utils/__init__.py")
    
    # Fix 2: Create logs directory
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir(exist_ok=True)
        logger.info("✓ Created logs directory")
    
    # Fix 3: Create results directories
    results_dirs = [
        "alpha_fe_analysis_results",
        "physics_analysis_results",
        "diagnostics"
    ]
    
    for dir_name in results_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✓ Created {dir_name} directory")
    
    return True

def create_test_run_script():
    """Create a test script to run on a single galaxy"""
    logger.info("Creating test run script...")
    
    test_script = """#!/usr/bin/env python3
'''
Test script to run ISAPC on a single galaxy
'''
import sys
import subprocess
from pathlib import Path

def test_single_galaxy():
    galaxy = "VCC1588"  # Test galaxy
    redshift = 0.0042
    
    # Check if data file exists
    data_file = Path(f"data/MUSE/{galaxy}_stack.fits")
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return 1
    
    # Check if template exists
    template_file = Path("data/templates/spectra_emiles_9.0.npz")
    if not template_file.exists():
        print(f"Error: Template file not found: {template_file}")
        return 1
    
    # Run the analysis
    cmd = [
        sys.executable, "main.py",
        str(data_file),
        "-z", str(redshift),
        "-t", str(template_file),
        "-o", "output",
        "-m", "ALL",
        "--target-snr", "20",
        "--n-rings", "6",
        "--physical-radius",
        "--use-re-bins",
        "--max-radius-scale", "3.0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✓ Successfully analyzed {galaxy}")
        return 0
    else:
        print(f"✗ Failed to analyze {galaxy}")
        return 1

if __name__ == "__main__":
    sys.exit(test_single_galaxy())
"""
    
    test_file = Path("test_single_galaxy.py")
    test_file.write_text(test_script)
    test_file.chmod(0o755)
    logger.info(f"✓ Created test script: {test_file}")
    
    return True

def main():
    """Main function to run all checks"""
    logger.info("=== ISAPC Data Status Checker ===")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Template File", check_template_file),
        ("TMB03 Model", check_tmb03_model),
        ("Output Structure", check_output_structure),
        ("Common Issues", fix_common_issues),
        ("Test Script", create_test_run_script)
    ]
    
    results = {}
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"Error in {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    all_good = True
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{check_name}: {status}")
        if not result:
            all_good = False
    
    if all_good:
        logger.info("\n🎉 All checks passed! You can now run:")
        logger.info("  python test_single_galaxy.py  # Test on one galaxy")
        logger.info("  python run_complete_pipeline.py  # Run on all galaxies")
        logger.info("  python complete_physics_analysis.py  # Final analysis")
    else:
        logger.warning("\n⚠ Some checks failed. Please fix the issues before running the pipeline.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
