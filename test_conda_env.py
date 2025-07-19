#!/usr/bin/env python3
"""
Simple test script using conda environment
"""
import subprocess
import sys
from pathlib import Path

def test_single_galaxy():
    """Test a single galaxy using the conda environment"""
    
    # Build command
    cmd = [
        "conda", "run", "-n", "Siqi_AstPy_312", "python", "main.py",
        "data/MUSE/VCC1588_stack.fits",
        "-z", "0.0042",
        "-t", "data/templates/spectra_emiles_9.0.npz",
        "-o", "output",
        "-m", "ALL",
        "--target-snr", "20",
        "--n-rings", "6",
        "--physical-radius",
        "--cvt"
    ]
    
    print("Running command:", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        print("Return code:", result.returncode)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

if __name__ == "__main__":
    success = test_single_galaxy()
    sys.exit(0 if success else 1)
