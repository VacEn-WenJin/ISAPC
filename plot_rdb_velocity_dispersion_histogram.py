#!/usr/bin/env python3
"""
Script to create a histogram of velocity dispersion from ISAPC RDB (Radial Binning) results
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_rdb_velocity_dispersions(output_dir):
    """
    Load velocity dispersion data from ISAPC RDB analysis results
    
    Args:
        output_dir (str): Path to ISAPC output directory
        
    Returns:
        dict: Dictionary containing velocity dispersion data and metadata
    """
    velocity_data = {
        'dispersions': [],
        'galaxies': [],
        'bin_radii': [],
        'metadata': []
    }
    
    # Find all VCC galaxy directories
    galaxy_dirs = [d for d in os.listdir(output_dir) 
                   if d.startswith('VCC') and os.path.isdir(os.path.join(output_dir, d))]
    
    logging.info(f"Loading ISAPC RDB data from: {output_dir}")
    
    for galaxy_dir in sorted(galaxy_dirs):
        galaxy_path = os.path.join(output_dir, galaxy_dir, 'Data')
        if not os.path.exists(galaxy_path):
            continue
            
        galaxy_name = galaxy_dir.replace('_stack', '')
        
        # Look for RDB results file
        results_file = f"{galaxy_dir}_RDB_results.npz"
        file_path = os.path.join(galaxy_path, results_file)
        
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # Extract velocity dispersion from stellar_kinematics
                if 'stellar_kinematics' in data.files:
                    stellar_kin = data['stellar_kinematics'].item()
                    if isinstance(stellar_kin, dict) and 'dispersion' in stellar_kin:
                        dispersion_data = stellar_kin['dispersion']
                        
                        # Get binning information for radial distances
                        bin_radii = None
                        if 'distance' in data.files:
                            distance_data = data['distance'].item()
                            if isinstance(distance_data, dict) and 'bin_distances' in distance_data:
                                bin_radii = distance_data['bin_distances']
                        
                        velocity_data['dispersions'].append(dispersion_data)
                        velocity_data['galaxies'].append(galaxy_name)
                        velocity_data['bin_radii'].append(bin_radii)
                        velocity_data['metadata'].append({
                            'galaxy': galaxy_name,
                            'file_path': file_path,
                            'dispersion_shape': dispersion_data.shape if hasattr(dispersion_data, 'shape') else None,
                            'bin_radii': bin_radii
                        })
                        
                        logging.info(f"✓ Loaded velocity dispersion for {galaxy_name}: shape={dispersion_data.shape}")
                        
                    else:
                        logging.warning(f"No dispersion data in stellar_kinematics for {galaxy_name}")
                else:
                    logging.warning(f"No stellar_kinematics data for {galaxy_name}")
                        
            except Exception as e:
                logging.error(f"Error loading velocity dispersion from {galaxy_name}: {e}")
    
    return velocity_data

def apply_rdb_velocity_dispersion_corrections(velocity_dispersions):
    """
    Apply velocity dispersion corrections specifically for RDB analysis
    RDB analysis typically works with binned data, so corrections may be different
    """
    
    # RDB-specific velocity dispersion correction coefficients
    corrections = {
        'Fe5015': -0.0008,  # Å/(km/s)⁻¹
        'Mgb': -0.0006,     # Å/(km/s)⁻¹
        'Hbeta': -0.0003    # Å/(km/s)⁻¹
    }
    
    # Reference velocity dispersion for RDB analysis
    # RDB often uses different reference than P2P due to binning effects
    sigma_ref = 200.0  # km/s (TMB03 standard, but may need adjustment for binned data)
    
    logger.info(f"\n⚙️  RDB VELOCITY DISPERSION CORRECTIONS")
    logger.info(f"Reference velocity dispersion: {sigma_ref} km/s")
    logger.info(f"Note: RDB analysis uses radially binned data")
    logger.info(f"Correction coefficients:")
    for index, coeff in corrections.items():
        logger.info(f"  {index}: {coeff} Å/(km/s)⁻¹")
    
    # Flatten all velocity dispersion arrays and remove NaN values
    all_dispersions = []
    for dispersion_data in velocity_dispersions:
        if hasattr(dispersion_data, 'flatten'):
            flat_data = dispersion_data.flatten()
            # Remove NaN values
            valid_data = flat_data[~np.isnan(flat_data)]
            all_dispersions.extend(valid_data)
    
    all_dispersions = np.array(all_dispersions)
    
    # Calculate corrections for each spectral index
    correction_values = {}
    for index, coeff in corrections.items():
        # Correction = coefficient * (sigma - sigma_ref)
        correction_values[index] = coeff * (all_dispersions - sigma_ref)
    
    return correction_values, corrections

def create_rdb_velocity_dispersion_histogram(velocity_dispersions, galaxy_names, rdb_data_info, output_dir="./"):
    """Create histogram of RDB velocity dispersion values with corrections"""
    
    if len(velocity_dispersions) == 0:
        logger.error("No RDB velocity dispersion data found!")
        return
    
    # Flatten all velocity dispersion arrays and remove NaN values
    all_dispersions = []
    for i, dispersion_data in enumerate(velocity_dispersions):
        if hasattr(dispersion_data, 'flatten'):
            flat_data = dispersion_data.flatten()
            # Remove NaN values
            valid_data = flat_data[~np.isnan(flat_data)]
            all_dispersions.extend(valid_data)
            logger.info(f"Galaxy {galaxy_names[i]}: {len(valid_data)} valid dispersion measurements")
        else:
            logger.warning(f"Galaxy {galaxy_names[i]}: Unexpected data format")
    
    all_dispersions = np.array(all_dispersions)
    logger.info(f"Total valid velocity dispersion measurements: {len(all_dispersions)}")
    
    if len(all_dispersions) == 0:
        logger.error("No valid velocity dispersion measurements found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ISAPC RDB Velocity Dispersion Analysis\n(Radial Binning Results)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Main histogram
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(all_dispersions, bins=40, alpha=0.7, color='darkblue', 
                               edgecolor='black', density=False)
    ax1.set_xlabel('Velocity Dispersion (km/s)')
    ax1.set_ylabel('Number of Measurements')
    ax1.set_title(f'RDB Velocity Dispersion Distribution\n({len(galaxy_names)} galaxies, {len(all_dispersions)} measurements)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_sigma = np.mean(all_dispersions)
    std_sigma = np.std(all_dispersions)
    median_sigma = np.median(all_dispersions)
    
    ax1.axvline(mean_sigma, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_sigma:.1f} km/s')
    ax1.axvline(median_sigma, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_sigma:.1f} km/s')
    ax1.axvline(200, color='purple', linestyle=':', linewidth=2, 
               label='TMB03 Reference: 200 km/s')
    ax1.legend()
    
    # Plot 2: Log-scale histogram for better visualization of tails
    ax2 = axes[0, 1]
    ax2.hist(all_dispersions, bins=40, alpha=0.7, color='darkgreen', 
             edgecolor='black', density=True)
    ax2.set_xlabel('Velocity Dispersion (km/s)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('RDB Dispersion Distribution (Normalized)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(200, color='purple', linestyle=':', linewidth=2, 
               label='TMB03 Reference: 200 km/s')
    ax2.legend()
    
    # Plot 3: Velocity dispersion corrections
    ax3 = axes[1, 0]
    
    # Calculate corrections using the original 2D data
    correction_values, corrections = apply_rdb_velocity_dispersion_corrections(velocity_dispersions)
    
    # Plot correction distributions
    colors = ['red', 'blue', 'green']
    for i, (index, corr_vals) in enumerate(correction_values.items()):
        ax3.hist(corr_vals, bins=30, alpha=0.6, color=colors[i], 
                label=f'{index} (coeff: {corrections[index]})', density=True)
    
    ax3.set_xlabel('Correction Value (Å)')
    ax3.set_ylabel('Density')
    ax3.set_title('RDB Velocity Dispersion Corrections')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics and data summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create statistics table
    stats_text = f"""
    RDB VELOCITY DISPERSION STATISTICS
    
    Total Measurements: {len(all_dispersions):,}
    Galaxies with RDB Data: {len(galaxy_names)}
    
    DISTRIBUTION STATISTICS:
    Mean σ: {mean_sigma:.1f} ± {std_sigma:.1f} km/s
    Median σ: {median_sigma:.1f} km/s
    Range: {np.min(all_dispersions):.1f} - {np.max(all_dispersions):.1f} km/s
    
    PERCENTILES:
    5th: {np.percentile(all_dispersions, 5):.1f} km/s
    25th: {np.percentile(all_dispersions, 25):.1f} km/s
    75th: {np.percentile(all_dispersions, 75):.1f} km/s
    95th: {np.percentile(all_dispersions, 95):.1f} km/s
    
    RDB-SPECIFIC CHARACTERISTICS:
    • Radially binned measurements
    • Typically fewer data points per galaxy
    • May show different dispersion patterns
    
    CORRECTION RANGES (Å):
    Fe5015: {np.min(correction_values['Fe5015']):.4f} to {np.max(correction_values['Fe5015']):.4f}
    Mgb: {np.min(correction_values['Mgb']):.4f} to {np.max(correction_values['Mgb']):.4f}
    Hβ: {np.min(correction_values['Hbeta']):.4f} to {np.max(correction_values['Hbeta']):.4f}
    
    ⚠️  Note: RDB analysis uses radial binning which may
    affect velocity dispersion measurements compared to P2P
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "isapc_rdb_velocity_dispersion_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"RDB dispersion histogram saved to: {output_path}")
    
    # Print detailed summary
    logger.info(f"\n📊 RDB VELOCITY DISPERSION ANALYSIS SUMMARY")
    logger.info(f"="*60)
    logger.info(f"Total RDB velocity dispersion measurements: {len(all_dispersions):,}")
    logger.info(f"Galaxies with RDB data: {len(galaxy_names)}")
    logger.info(f"Mean velocity dispersion: {mean_sigma:.1f} ± {std_sigma:.1f} km/s")
    logger.info(f"Range: {np.min(all_dispersions):.1f} - {np.max(all_dispersions):.1f} km/s")
    
    # Print per-galaxy breakdown
    logger.info(f"\nPER-GALAXY RDB DATA BREAKDOWN:")
    for i, metadata in enumerate(rdb_data_info):
        if metadata:  # Only show galaxies with data
            galaxy = metadata['galaxy']
            shape = metadata['dispersion_shape']
            if shape:
                total_pixels = shape[0] * shape[1]
                logger.info(f"  {galaxy}: {shape} spatial map ({total_pixels} total pixels)")
            else:
                logger.info(f"  {galaxy}: Data available (shape unknown)")
    
    plt.show()
    
    return all_dispersions, correction_values

def main():
    """Main function to run the RDB velocity dispersion analysis"""
    
    logger.info("🔍 ISAPC RDB Velocity Dispersion Analysis")
    logger.info("="*60)
    
    # Load RDB data
    data_dir = "/home/siqi/WkpSpace/ISAPC_Jul/ISAPC/output"
    
    logger.info(f"Loading ISAPC RDB data from: {data_dir}")
    velocity_data = load_rdb_velocity_dispersions(data_dir)
    
    if len(velocity_data['dispersions']) == 0:
        logger.error("No RDB velocity dispersion data found in ISAPC results!")
        logger.info("Checking available RDB files...")
        
        # List available RDB files for debugging
        data_path = Path(data_dir)
        rdb_files_found = []
        for galaxy_dir in data_path.glob("VCC*_stack"):
            rdb_files = list(galaxy_dir.glob("Data/*RDB*.npz"))
            if rdb_files:
                rdb_files_found.extend(rdb_files)
        
        if rdb_files_found:
            logger.info("Found RDB files:")
            for f in rdb_files_found[:10]:  # Show first 10
                logger.info(f"  {f}")
            if len(rdb_files_found) > 10:
                logger.info(f"  ... and {len(rdb_files_found) - 10} more")
        else:
            logger.error("No RDB files found!")
        return
    
    # Create histogram and apply corrections
    velocity_dispersions, correction_values = create_rdb_velocity_dispersion_histogram(
        velocity_data['dispersions'], velocity_data['galaxies'], velocity_data['metadata']
    )
    
    logger.info("\n✅ RDB velocity dispersion analysis complete!")

if __name__ == "__main__":
    main()
