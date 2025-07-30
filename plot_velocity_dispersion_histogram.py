#!/usr/bin/env python3
"""
Script to create a histogram of velocity dispersion from ISAPC results
and apply the necessary velocity dispersion corrections for spectral indices.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_isapc_data(data_dir):
    """Load ISAPC data and extract velocity dispersion information"""
    
    velocity_dispersions = []
    galaxy_names = []
    
    data_path = Path(data_dir)
    
    # Look for all galaxy output directories
    for galaxy_dir in data_path.glob("VCC*_stack"):
        galaxy_name = galaxy_dir.name.replace("_stack", "")
        galaxy_names.append(galaxy_name)
        
        # Check for P2P stellar kinematics data (most likely to contain velocity dispersion)
        p2p_kinematics_file = galaxy_dir / "Data" / f"{galaxy_dir.name}_P2P_stellar_kinematics.npz"
        
        if p2p_kinematics_file.exists():
            try:
                # Load P2P stellar kinematics data
                data = np.load(p2p_kinematics_file)
                
                logger.info(f"\n{galaxy_name} - P2P Stellar Kinematics file contains:")
                for key in data.keys():
                    logger.info(f"  - {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
                
                # Look for velocity dispersion field
                if 'dispersion_field' in data:
                    dispersion_field = data['dispersion_field']
                    # Extract valid (finite) velocity dispersions
                    valid_dispersions = dispersion_field[np.isfinite(dispersion_field)]
                    velocity_dispersions.extend(valid_dispersions)
                    logger.info(f"  Found {len(valid_dispersions)} valid dispersion measurements")
                
            except Exception as e:
                logger.warning(f"Error loading {p2p_kinematics_file}: {e}")
        
        # Also check P2P results file
        p2p_results_file = galaxy_dir / "Data" / f"{galaxy_dir.name}_P2P_results.npz"
        
        if p2p_results_file.exists():
            try:
                # Load P2P results data
                data = np.load(p2p_results_file, allow_pickle=True)
                
                logger.info(f"\n{galaxy_name} - P2P Results file contains:")
                for key in data.keys():
                    if hasattr(data[key], 'shape'):
                        logger.info(f"  - {key}: {data[key].shape}")
                    else:
                        logger.info(f"  - {key}: {type(data[key])}")
                
                # Check if there's stellar kinematics data in results
                if 'stellar_kinematics' in data:
                    stellar_kinematics = data['stellar_kinematics'].item()  # Convert from numpy object
                    if isinstance(stellar_kinematics, dict) and 'dispersion_field' in stellar_kinematics:
                        dispersion_field = stellar_kinematics['dispersion_field']
                        valid_dispersions = dispersion_field[np.isfinite(dispersion_field)]
                        velocity_dispersions.extend(valid_dispersions)
                        logger.info(f"  Found {len(valid_dispersions)} valid dispersion measurements in stellar_kinematics")
                
            except Exception as e:
                logger.warning(f"Error loading {p2p_results_file}: {e}")
    
    return np.array(velocity_dispersions), galaxy_names

def apply_velocity_dispersion_corrections(sigma_values):
    """
    Apply velocity dispersion corrections to spectral indices
    
    Based on the coefficients found in the ISAPC codebase:
    - Fe5015: -0.0008 Å/(km/s)⁻¹
    - Mgb: -0.0006 Å/(km/s)⁻¹ 
    - Hβ: -0.0003 Å/(km/s)⁻¹
    
    Note: These coefficients are currently without proper citation (see ISAPC_Complete_References.tex)
    """
    
    # Velocity dispersion correction coefficients (CRITICAL: SOURCE UNKNOWN)
    corrections = {
        'Fe5015': -0.0008,  # Å/(km/s)⁻¹
        'Mgb': -0.0006,     # Å/(km/s)⁻¹
        'Hbeta': -0.0003    # Å/(km/s)⁻¹
    }
    
    # Reference velocity dispersion (typically 100 or 200 km/s)
    sigma_ref = 200.0  # km/s (TMB03 standard)
    
    logger.info(f"\n⚠️  VELOCITY DISPERSION CORRECTIONS")
    logger.info(f"Reference velocity dispersion: {sigma_ref} km/s")
    logger.info(f"Correction coefficients (⚠️  SOURCE UNKNOWN):")
    for index, coeff in corrections.items():
        logger.info(f"  {index}: {coeff} Å/(km/s)⁻¹")
    
    # Calculate corrections for each spectral index
    correction_values = {}
    for index, coeff in corrections.items():
        # Correction = coefficient * (sigma - sigma_ref)
        correction_values[index] = coeff * (sigma_values - sigma_ref)
    
    return correction_values, corrections

def create_velocity_dispersion_histogram(velocity_dispersions, galaxy_names, output_dir="./"):
    """Create histogram of velocity dispersion values with corrections"""
    
    if len(velocity_dispersions) == 0:
        logger.error("No velocity dispersion data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ISAPC Velocity Dispersion Analysis and Corrections', fontsize=16, fontweight='bold')
    
    # Plot 1: Main histogram
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(velocity_dispersions, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Velocity Dispersion (km/s)')
    ax1.set_ylabel('Number of Pixels')
    ax1.set_title('Distribution of Velocity Dispersions')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_sigma = np.mean(velocity_dispersions)
    std_sigma = np.std(velocity_dispersions)
    median_sigma = np.median(velocity_dispersions)
    
    ax1.axvline(mean_sigma, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sigma:.1f} km/s')
    ax1.axvline(median_sigma, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_sigma:.1f} km/s')
    ax1.axvline(200, color='purple', linestyle=':', linewidth=2, label='TMB03 Reference: 200 km/s')
    ax1.legend()
    
    # Plot 2: Cumulative distribution
    ax2 = axes[0, 1]
    sorted_sigma = np.sort(velocity_dispersions)
    cumulative = np.arange(1, len(sorted_sigma) + 1) / len(sorted_sigma)
    ax2.plot(sorted_sigma, cumulative, linewidth=2, color='darkgreen')
    ax2.set_xlabel('Velocity Dispersion (km/s)')
    ax2.set_ylabel('Cumulative Fraction')
    ax2.set_title('Cumulative Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(200, color='purple', linestyle=':', linewidth=2, label='TMB03 Reference: 200 km/s')
    ax2.legend()
    
    # Plot 3: Velocity dispersion corrections
    ax3 = axes[1, 0]
    
    # Calculate corrections
    correction_values, corrections = apply_velocity_dispersion_corrections(velocity_dispersions)
    
    # Plot correction distributions
    colors = ['red', 'blue', 'green']
    for i, (index, corr_vals) in enumerate(correction_values.items()):
        ax3.hist(corr_vals, bins=30, alpha=0.6, color=colors[i], 
                label=f'{index} (coeff: {corrections[index]})', density=True)
    
    ax3.set_xlabel('Correction Value (Å)')
    ax3.set_ylabel('Density')
    ax3.set_title('Velocity Dispersion Corrections for Spectral Indices')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create statistics table
    stats_text = f"""
    VELOCITY DISPERSION STATISTICS
    
    Total Measurements: {len(velocity_dispersions):,}
    Galaxies Processed: {len(galaxy_names)}
    
    Mean σ: {mean_sigma:.1f} ± {std_sigma:.1f} km/s
    Median σ: {median_sigma:.1f} km/s
    Range: {np.min(velocity_dispersions):.1f} - {np.max(velocity_dispersions):.1f} km/s
    
    Percentiles:
    5th: {np.percentile(velocity_dispersions, 5):.1f} km/s
    25th: {np.percentile(velocity_dispersions, 25):.1f} km/s
    75th: {np.percentile(velocity_dispersions, 75):.1f} km/s
    95th: {np.percentile(velocity_dispersions, 95):.1f} km/s
    
    CORRECTION RANGES (Å):
    Fe5015: {np.min(correction_values['Fe5015']):.4f} to {np.max(correction_values['Fe5015']):.4f}
    Mgb: {np.min(correction_values['Mgb']):.4f} to {np.max(correction_values['Mgb']):.4f}
    Hβ: {np.min(correction_values['Hbeta']):.4f} to {np.max(correction_values['Hbeta']):.4f}
    
    ⚠️  WARNING: Correction coefficients lack proper citation
    See ISAPC_Complete_References.tex for details
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "isapc_velocity_dispersion_histogram_with_corrections.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved to: {output_path}")
    
    # Print summary
    logger.info(f"\n📊 VELOCITY DISPERSION ANALYSIS SUMMARY")
    logger.info(f"Total velocity dispersion measurements: {len(velocity_dispersions):,}")
    logger.info(f"Galaxies with data: {len(galaxy_names)}")
    logger.info(f"Mean velocity dispersion: {mean_sigma:.1f} ± {std_sigma:.1f} km/s")
    logger.info(f"Range: {np.min(velocity_dispersions):.1f} - {np.max(velocity_dispersions):.1f} km/s")
    
    plt.show()
    
    return velocity_dispersions, correction_values

def main():
    """Main function to run the velocity dispersion analysis"""
    
    logger.info("🔍 ISAPC Velocity Dispersion Analysis")
    logger.info("="*50)
    
    # Load ISAPC data
    data_dir = "/home/siqi/WkpSpace/ISAPC_Jul/ISAPC/output"
    
    logger.info(f"Loading ISAPC data from: {data_dir}")
    velocity_dispersions, galaxy_names = load_isapc_data(data_dir)
    
    if len(velocity_dispersions) == 0:
        logger.error("No velocity dispersion data found in ISAPC results!")
        return
    
    # Create histogram and apply corrections
    velocity_dispersions, correction_values = create_velocity_dispersion_histogram(
        velocity_dispersions, galaxy_names
    )
    
    logger.info("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
