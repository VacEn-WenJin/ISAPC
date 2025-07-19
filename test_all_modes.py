#!/usr/bin/env python3
"""
Test script for all three analysis modes on a single galaxy
"""

import subprocess
import sys
import os

def main():
    """Run all three modes test"""
    
    # Change to ISAPC directory
    os.chdir('/home/siqi/WkpSpace/ISAPC_Jul/ISAPC')
    
    # Command arguments
    args = [
        "conda", "run", "-n", "Siqi_AstPy_312", 
        "python", "main.py", 
        "data/MUSE/VCC0308_stack.fits",
        "-z", "0.0055",
        "-t", "data/templates/spectra_emiles_9.0.npz",
        "-o", "output",
        "-m", "ALL",
        "--target-snr", "20.0",
        "--min-snr", "1.0",
        "--n-rings", "6",
        "--vel-init", "0.0",
        "--sigma-init", "50.0",
        "--poly-degree", "3",
        "--n-jobs", "4",
        "--save-error-maps",
        "--auto-reuse",
        "--cvt",
        "--physical-radius"
    ]
    
    print("Running complete ISAPC analysis with all modes (P2P, VNB, RDB)...")
    print("Command:", " ".join(args))
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(args, capture_output=False, text=True)
        return result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
