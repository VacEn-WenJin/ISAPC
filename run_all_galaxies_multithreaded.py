#!/usr/bin/env python3
"""
ISAPC Batch Processing Script - Multithreaded
Run all MUSE galaxies with proven working parameters
Using 3 jobs for improved performance
"""

import subprocess
import glob
import os
import time
from pathlib import Path

def get_galaxy_redshift(galaxy_name):
    """Get redshift for each galaxy"""
    redshifts = {
        'VCC0308': 0.0042,
        'VCC0667': 0.0042, 
        'VCC0688': 0.0042,
        'VCC0990': 0.0042,
        'VCC1049': 0.0042,
        'VCC1146': 0.0042,
        'VCC1193': 0.0042,
        'VCC1368': 0.0042,
        'VCC1410': 0.0042,
        'VCC1431': 0.0042,
        'VCC1588': 0.0042,
        'VCC1826': 0.0042,
        'VCC1833': 0.0042,
        'VCC1857': 0.0042,
        'VCC1899': 0.0042,
        'VCC1903': 0.0042,
        'VCC1952': 0.0042,
        'VCC2050': 0.0042,
        'VCC2058': 0.0042,
        'VCC2092': 0.0042,
    }
    return redshifts.get(galaxy_name, 0.0042)

def run_isapc_for_galaxy(fits_file):
    """Run ISAPC for a single galaxy with proven parameters"""
    
    # Extract galaxy name
    galaxy_name = Path(fits_file).stem.replace('_stack', '')
    redshift = get_galaxy_redshift(galaxy_name)
    
    print(f"\n{'='*60}")
    print(f"Processing {galaxy_name} (z={redshift})")
    print(f"{'='*60}")
    
    # Build command with exact working parameters from VCC1588 + multithreading
    cmd = [
        'python', 'main.py',
        fits_file,
        '-z', str(redshift),
        '-t', 'data/templates/spectra_emiles_9.0.npz',
        '-o', 'output',
        '-m', 'ALL',
        '--target-snr', '20.0',
        '--min-snr', '1.0',
        '--n-rings', '6',
        '--vel-init', '0.0',
        '--sigma-init', '50.0',
        '--poly-degree', '3',
        '--n-jobs', '3',  # Use 3 jobs for multithreading
        '--save-error-maps',
        '--auto-reuse',
        '--cvt',
        '--physical-radius'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {galaxy_name} completed successfully in {duration:.1f} seconds")
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True, duration, None
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ {galaxy_name} failed after {duration:.1f} seconds")
        print(f"Error code: {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        
        return False, duration, e.stderr

def main():
    """Main execution function"""
    
    print("ISAPC Multithreaded Batch Processing")
    print("=====================================")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all FITS files
    fits_files = sorted(glob.glob('data/MUSE/*_stack.fits'))
    
    if not fits_files:
        print("❌ No FITS files found in data/MUSE/")
        return
    
    print(f"Found {len(fits_files)} galaxies to process")
    
    # Track results
    results = {
        'successful': [],
        'failed': [],
        'total_time': 0
    }
    
    overall_start = time.time()
    
    # Process each galaxy
    for i, fits_file in enumerate(fits_files, 1):
        print(f"\n🔄 Processing galaxy {i}/{len(fits_files)}")
        
        success, duration, error = run_isapc_for_galaxy(fits_file)
        
        galaxy_name = Path(fits_file).stem.replace('_stack', '')
        
        if success:
            results['successful'].append((galaxy_name, duration))
        else:
            results['failed'].append((galaxy_name, duration, error))
        
        results['total_time'] += duration
        
        # Progress update
        print(f"\nProgress: {i}/{len(fits_files)} galaxies processed")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Time elapsed: {time.time() - overall_start:.1f} seconds")
    
    # Final summary
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total galaxies processed: {len(fits_files)}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Overall time: {overall_time:.1f} seconds")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['successful']:
        print(f"\n✅ Successful galaxies:")
        for galaxy, duration in results['successful']:
            print(f"  - {galaxy}: {duration:.1f}s")
    
    if results['failed']:
        print(f"\n❌ Failed galaxies:")
        for galaxy, duration, error in results['failed']:
            print(f"  - {galaxy}: failed after {duration:.1f}s")
    
    print(f"\n📊 Average processing time: {results['total_time']/len(fits_files):.1f} seconds per galaxy")

if __name__ == "__main__":
    main()
