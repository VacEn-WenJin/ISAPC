#!/usr/bin/env python3
"""
Complete Physics Analysis and Visualization for Virgo Cluster Galaxies
Addresses key issues in the alpha/Fe abundance analysis pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Ellipse
from matplotlib.colors import Normalize
import logging
import os
import sys
from pathlib import Path
import datetime
from scipy import stats
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from reader import Read_otp
    from Phy_Visu import (
        Read_Galaxy, get_coordinated_alpha_fe_age_data, 
        load_enhanced_model_data, calculate_alpha_fe_gradient_linear,
        create_velocity_gradient_vector_plot
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Galaxy list
GALAXIES = [
    "VCC0308", "VCC0667", "VCC0688", "VCC0990", "VCC1049", "VCC1146",
    "VCC1193", "VCC1368", "VCC1410", "VCC1431", "VCC1486", "VCC1499",
    "VCC1549", "VCC1588", "VCC1695", "VCC1811", "VCC1890",
    "VCC1902", "VCC1910", "VCC1949"
]

# Configuration
CONFIG = {
    "model_file": "TMB03/TMB03.csv",
    "output_dir": "physics_analysis_results",
    "continuum_mode": "fit",
    "bins_limit": 6,
    "enable_error_propagation": True,
    "create_summary_plots": True,
    "save_individual_plots": True,
    "dpi": 150,
    "figsize": (12, 8)
}

def setup_analysis_directories():
    """Create directory structure for analysis results"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path(CONFIG["output_dir"])
    dirs = {
        "base": base_dir,
        "individual": base_dir / "individual_galaxies",
        "summary": base_dir / "summary_plots",
        "data": base_dir / "data_tables",
        "diagnostics": base_dir / "diagnostics"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created analysis directories in {base_dir}")
    return dirs

def load_galaxy_data(galaxy_name):
    """Load P2P, VNB, and RDB data for a galaxy"""
    try:
        # Try using the reader function
        p2p_data = Read_otp(galaxy_name, "P2P")
        vnb_data = Read_otp(galaxy_name, "VNB")
        rdb_data = Read_otp(galaxy_name, "RDB")
        
        # Fallback to Phy_Visu reader if needed
        if not rdb_data:
            p2p_data, vnb_data, rdb_data = Read_Galaxy(galaxy_name)
        
        return p2p_data, vnb_data, rdb_data
    
    except Exception as e:
        logger.error(f"Error loading data for {galaxy_name}: {e}")
        return None, None, None

def calculate_mean_velocity_in_re(p2p_data, effective_radius=None):
    """Calculate mean velocity within effective radius"""
    try:
        if not p2p_data or "stellar_kinematics" not in p2p_data:
            return None, None
        
        velocity_field = p2p_data["stellar_kinematics"].get("velocity_field")
        if velocity_field is None:
            return None, None
        
        # Get dimensions
        ny, nx = velocity_field.shape
        y, x = np.indices((ny, nx))
        center_y, center_x = ny // 2, nx // 2
        
        # Calculate radius
        if effective_radius is not None:
            # Use effective radius for masking
            radius_pix = effective_radius / 0.2  # Assume 0.2" pixel scale
            mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius_pix
        else:
            # Use central region
            mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) <= min(nx, ny) // 4
        
        # Calculate mean velocity and dispersion
        valid_velocities = velocity_field[mask & np.isfinite(velocity_field)]
        
        if len(valid_velocities) > 0:
            mean_vel = np.mean(valid_velocities)
            vel_std = np.std(valid_velocities)
            return mean_vel, vel_std
        else:
            return None, None
    
    except Exception as e:
        logger.error(f"Error calculating mean velocity: {e}")
        return None, None

def analyze_single_galaxy(galaxy_name, model_data, dirs):
    """Analyze a single galaxy for alpha/Fe gradients"""
    logger.info(f"Analyzing {galaxy_name}")
    
    # Load data
    p2p_data, vnb_data, rdb_data = load_galaxy_data(galaxy_name)
    
    if not rdb_data:
        logger.warning(f"No RDB data for {galaxy_name}")
        return None
    
    # Calculate alpha/Fe data
    try:
        alpha_fe_result = get_coordinated_alpha_fe_age_data(
            galaxy_name, rdb_data, model_data,
            bins_limit=CONFIG["bins_limit"],
            continuum_mode=CONFIG["continuum_mode"]
        )
        
        if not alpha_fe_result or len(alpha_fe_result.get("alpha_fe_values", [])) == 0:
            logger.warning(f"No alpha/Fe data calculated for {galaxy_name}")
            return None
        
        # Calculate gradient
        gradient_result = calculate_alpha_fe_gradient_linear(alpha_fe_result)
        
        # Calculate mean velocity
        mean_vel, vel_std = calculate_mean_velocity_in_re(
            p2p_data, 
            effective_radius=alpha_fe_result.get("effective_radius")
        )
        
        # Compile results
        result = {
            "galaxy": galaxy_name,
            "alpha_fe_values": alpha_fe_result.get("alpha_fe_values", []),
            "alpha_fe_uncertainties": alpha_fe_result.get("alpha_fe_uncertainties", []),
            "radius_values": alpha_fe_result.get("radius_values", []),
            "effective_radius": alpha_fe_result.get("effective_radius"),
            "gradient_slope": gradient_result.get("slope"),
            "gradient_slope_error": gradient_result.get("slope_error"),
            "gradient_intercept": gradient_result.get("intercept"),
            "gradient_r_squared": gradient_result.get("r_squared"),
            "gradient_p_value": gradient_result.get("p_value"),
            "mean_velocity": mean_vel,
            "velocity_dispersion": vel_std,
            "n_bins": len(alpha_fe_result.get("alpha_fe_values", []))
        }
        
        # Create individual galaxy plot
        if CONFIG["save_individual_plots"]:
            create_individual_galaxy_plot(result, dirs["individual"])
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing {galaxy_name}: {e}")
        return None

def create_individual_galaxy_plot(result, output_dir):
    """Create diagnostic plot for individual galaxy"""
    galaxy_name = result["galaxy"]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Alpha/Fe vs radius
    if result["alpha_fe_values"] and result["radius_values"]:
        alpha_fe = np.array(result["alpha_fe_values"])
        radius = np.array(result["radius_values"])
        uncertainties = np.array(result.get("alpha_fe_uncertainties", []))
        
        if len(uncertainties) == len(alpha_fe):
            ax1.errorbar(radius, alpha_fe, yerr=uncertainties, 
                        fmt='o', markersize=8, capsize=5, capthick=2)
        else:
            ax1.plot(radius, alpha_fe, 'o', markersize=8)
        
        # Plot gradient fit if available
        if result["gradient_slope"] is not None:
            x_fit = np.linspace(min(radius), max(radius), 100)
            y_fit = result["gradient_slope"] * x_fit + result["gradient_intercept"]
            ax1.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.7)
            
            # Add fit statistics
            slope = result["gradient_slope"]
            r_sq = result["gradient_r_squared"]
            p_val = result["gradient_p_value"]
            ax1.text(0.05, 0.95, f"Slope: {slope:.3f}\nR²: {r_sq:.3f}\np: {p_val:.3f}",
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.set_xlabel("Radius (R/Re)")
    ax1.set_ylabel("[α/Fe]")
    ax1.set_title(f"{galaxy_name} - Alpha/Fe Gradient")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Alpha/Fe histogram
    if result["alpha_fe_values"]:
        ax2.hist(result["alpha_fe_values"], bins=min(10, len(result["alpha_fe_values"])), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(result["alpha_fe_values"]), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(result["alpha_fe_values"]):.3f}')
        ax2.set_xlabel("[α/Fe]")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Alpha/Fe Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient statistics
    stats_text = f"""
    Galaxy: {galaxy_name}
    Effective Radius: {result['effective_radius']:.2f}" if result['effective_radius'] else "N/A"
    Number of Bins: {result['n_bins']}
    Mean [α/Fe]: {np.mean(result['alpha_fe_values']):.3f} if result['alpha_fe_values'] else "N/A"
    Gradient Slope: {result['gradient_slope']:.3f} if result['gradient_slope'] else "N/A"
    Mean Velocity: {result['mean_velocity']:.1f} km/s if result['mean_velocity'] else "N/A"
    Velocity Dispersion: {result['velocity_dispersion']:.1f} km/s if result['velocity_dispersion'] else "N/A"
    """
    
    ax3.text(0.05, 0.95, stats_text.strip(), transform=ax3.transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title("Analysis Summary")
    
    # Plot 4: Quality indicators
    quality_metrics = []
    if result["gradient_r_squared"] is not None:
        quality_metrics.append(f"R² = {result['gradient_r_squared']:.3f}")
    if result["gradient_p_value"] is not None:
        quality_metrics.append(f"p-value = {result['gradient_p_value']:.3f}")
    if result["n_bins"] > 0:
        quality_metrics.append(f"N bins = {result['n_bins']}")
    
    if quality_metrics:
        ax4.text(0.05, 0.95, "\n".join(quality_metrics), transform=ax4.transAxes,
                verticalalignment='top', fontsize=12)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title("Quality Metrics")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{galaxy_name}_analysis.png", dpi=CONFIG["dpi"], bbox_inches='tight')
    plt.close()

def create_summary_plots(results, dirs):
    """Create summary plots for all galaxies"""
    logger.info("Creating summary plots")
    
    # Filter successful results
    valid_results = [r for r in results if r and r.get("gradient_slope") is not None]
    
    if not valid_results:
        logger.warning("No valid results for summary plots")
        return
    
    # Extract data
    galaxy_names = [r["galaxy"] for r in valid_results]
    gradients = [r["gradient_slope"] for r in valid_results]
    gradient_errors = [r["gradient_slope_error"] for r in valid_results]
    mean_velocities = [r["mean_velocity"] for r in valid_results if r["mean_velocity"] is not None]
    vel_dispersions = [r["velocity_dispersion"] for r in valid_results if r["velocity_dispersion"] is not None]
    
    # Create vector plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create random positions for galaxies (you can replace with actual RA/Dec)
    np.random.seed(42)
    n_galaxies = len(valid_results)
    x_pos = np.random.uniform(0, 10, n_galaxies)
    y_pos = np.random.uniform(0, 10, n_galaxies)
    
    # Create vectors
    for i, result in enumerate(valid_results):
        gradient = result["gradient_slope"]
        velocity = result["mean_velocity"] if result["mean_velocity"] else 0
        
        # Vector length represents gradient magnitude
        vector_length = abs(gradient) * 20  # Scale factor
        
        # Vector angle represents something (you can customize this)
        vector_angle = np.random.uniform(0, 2*np.pi)  # Random for now
        
        # Vector components
        dx = vector_length * np.cos(vector_angle)
        dy = vector_length * np.sin(vector_angle)
        
        # Color by velocity
        color = velocity if velocity else 0
        
        # Plot arrow
        arrow = Arrow(x_pos[i], y_pos[i], dx, dy, 
                     width=0.3, color=plt.cm.RdYlBu(color/200 + 0.5))
        ax.add_patch(arrow)
        
        # Add galaxy label
        ax.text(x_pos[i], y_pos[i], result["galaxy"], 
               fontsize=8, ha='center', va='center')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_xlabel("RA (arbitrary units)")
    ax.set_ylabel("Dec (arbitrary units)")
    ax.set_title("Alpha/Fe Gradient Vectors (length = |gradient|, color = velocity)")
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=Normalize(vmin=-100, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Mean Velocity (km/s)")
    
    plt.tight_layout()
    plt.savefig(dirs["summary"] / "gradient_vectors.png", dpi=CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    
    # Create histogram of gradients
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gradients, bins=min(15, len(gradients)), alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(gradients), color='red', linestyle='--', 
               label=f'Mean: {np.mean(gradients):.3f}')
    ax.axvline(np.median(gradients), color='green', linestyle='--', 
               label=f'Median: {np.median(gradients):.3f}')
    ax.set_xlabel("Alpha/Fe Gradient (d[α/Fe]/d(R/Re))")
    ax.set_ylabel("Number of Galaxies")
    ax.set_title("Distribution of Alpha/Fe Gradients")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["summary"] / "gradient_distribution.png", dpi=CONFIG["dpi"], bbox_inches='tight')
    plt.close()

def save_results_table(results, dirs):
    """Save results to CSV table"""
    logger.info("Saving results table")
    
    # Create DataFrame
    data = []
    for result in results:
        if result:
            data.append({
                "Galaxy": result["galaxy"],
                "Effective_Radius_arcsec": result.get("effective_radius"),
                "N_Bins": result.get("n_bins", 0),
                "Mean_Alpha_Fe": np.mean(result["alpha_fe_values"]) if result["alpha_fe_values"] else np.nan,
                "Gradient_Slope": result.get("gradient_slope"),
                "Gradient_Slope_Error": result.get("gradient_slope_error"),
                "Gradient_R_Squared": result.get("gradient_r_squared"),
                "Gradient_P_Value": result.get("gradient_p_value"),
                "Mean_Velocity_km_s": result.get("mean_velocity"),
                "Velocity_Dispersion_km_s": result.get("velocity_dispersion")
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = dirs["data"] / "alpha_fe_analysis_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Results table saved to {output_file}")
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Total galaxies analyzed: {len(df)}")
    logger.info(f"Successful gradient measurements: {df['Gradient_Slope'].notna().sum()}")
    
    if df['Gradient_Slope'].notna().sum() > 0:
        logger.info(f"Mean gradient: {df['Gradient_Slope'].mean():.3f}")
        logger.info(f"Median gradient: {df['Gradient_Slope'].median():.3f}")
        logger.info(f"Standard deviation: {df['Gradient_Slope'].std():.3f}")

def main():
    """Main analysis function"""
    logger.info("Starting complete physics analysis")
    
    # Setup directories
    dirs = setup_analysis_directories()
    
    # Load model data
    logger.info("Loading TMB03 model data")
    model_data = load_enhanced_model_data(CONFIG["model_file"])
    
    if model_data is None or len(model_data) == 0:
        logger.error("Failed to load model data")
        return 1
    
    # Analyze each galaxy
    results = []
    for galaxy_name in GALAXIES:
        result = analyze_single_galaxy(galaxy_name, model_data, dirs)
        results.append(result)
    
    # Create summary plots
    if CONFIG["create_summary_plots"]:
        create_summary_plots(results, dirs)
    
    # Save results table
    save_results_table(results, dirs)
    
    # Print final summary
    successful = len([r for r in results if r and r.get("gradient_slope") is not None])
    logger.info(f"\nAnalysis complete!")
    logger.info(f"Successfully analyzed: {successful}/{len(GALAXIES)} galaxies")
    logger.info(f"Results saved to: {dirs['base']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
