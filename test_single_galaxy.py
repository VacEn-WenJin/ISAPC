#!/usr/bin/env python3
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
        "--cvt"
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
