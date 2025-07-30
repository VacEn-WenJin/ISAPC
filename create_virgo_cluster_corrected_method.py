"""
Virgo Cluster Gradient Visualization - Corrected to Match Old Method
Shows total α/Fe change across galaxy (not slope) like the original analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from alpha_gradient_analysis import analyze_single_galaxy
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VirgoClusterCorrected')

def get_real_ifu_positions():
    """Extract real IFU positions from MUSE FITS files with fallback coordinates"""
    import os
    from astropy.io import fits
    
    positions = {}
    data_dir = "data"
    
    # Known velocities for each galaxy
    velocities = {
        'VCC0308': 1572, 'VCC0667': 1431, 'VCC0688': 1061, 'VCC0990': 1740,
        'VCC1049': 639, 'VCC1146': 700, 'VCC1193': 757, 'VCC1368': 1055,
        'VCC1410': 1615, 'VCC1431': 1521, 'VCC1486': 111, 'VCC1499': 1823,
        'VCC1549': 1245, 'VCC1588': 1318, 'VCC1695': 1156, 'VCC1811': 1628,
        'VCC1890': 1672, 'VCC1902': 1519, 'VCC1910': 1745, 'VCC1949': 1198
    }
    
    # Emission line flags
    has_emission = {
        'VCC0308': False, 'VCC0667': False, 'VCC0688': False, 'VCC0990': False,
        'VCC1049': False, 'VCC1146': False, 'VCC1193': False, 'VCC1368': True,
        'VCC1410': False, 'VCC1431': True, 'VCC1486': False, 'VCC1499': False,
        'VCC1549': False, 'VCC1588': False, 'VCC1695': True, 'VCC1811': False,
        'VCC1890': False, 'VCC1902': False, 'VCC1910': False, 'VCC1949': True
    }
    
    # Real coordinates from enhanced version (extracted from MUSE FITS)
    real_coords = {
        'VCC0308': {'ra': 184.712195, 'dec': 14.687528},
        'VCC0667': {'ra': 185.559375, 'dec': 13.160167},
        'VCC0688': {'ra': 187.512625, 'dec': 11.378750},
        'VCC0990': {'ra': 186.933500, 'dec': 12.540889},
        'VCC1049': {'ra': 185.545958, 'dec': 13.491889},
        'VCC1146': {'ra': 186.341333, 'dec': 12.053500},
        'VCC1193': {'ra': 185.982292, 'dec': 14.226056},
        'VCC1368': {'ra': 187.702667, 'dec': 12.391139},
        'VCC1410': {'ra': 186.906208, 'dec': 12.813889},
        'VCC1431': {'ra': 187.774792, 'dec': 12.892750},
        'VCC1486': {'ra': 189.974208, 'dec': 13.734583},
        'VCC1499': {'ra': 188.256458, 'dec': 12.773500},
        'VCC1549': {'ra': 187.325708, 'dec': 13.621528},
        'VCC1588': {'ra': 189.325792, 'dec': 14.880750},
        'VCC1695': {'ra': 186.525708, 'dec': 13.176056},
        'VCC1811': {'ra': 187.836792, 'dec': 12.720333},
        'VCC1890': {'ra': 186.969792, 'dec': 13.942333},
        'VCC1902': {'ra': 187.230958, 'dec': 12.156972},
        'VCC1910': {'ra': 190.741792, 'dec': 12.400722},
        'VCC1949': {'ra': 186.844542, 'dec': 13.552083}
    }
    
    for galaxy_name in velocities.keys():
        try:
            # Try to extract from FITS file first
            fits_file = os.path.join(data_dir, f"{galaxy_name}.fits")
            ra, dec = None, None
            
            if os.path.exists(fits_file):
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header
                    ra = header.get('CRVAL1', None)
                    dec = header.get('CRVAL2', None)
                    
                    if ra is not None and dec is not None:
                        logger.info(f"Real FITS coordinates for {galaxy_name}: RA={ra:.5f}, DEC={dec:.5f}")
            
            # Use fallback coordinates if FITS not found or incomplete
            if ra is None or dec is None:
                if galaxy_name in real_coords:
                    ra = real_coords[galaxy_name]['ra']
                    dec = real_coords[galaxy_name]['dec']
                    logger.info(f"Using fallback coordinates for {galaxy_name}: RA={ra:.5f}, DEC={dec:.5f}")
            
            if ra is not None and dec is not None:
                positions[galaxy_name] = {
                    'ra': ra,
                    'dec': dec,
                    'velocity': velocities[galaxy_name],
                    'has_emission': has_emission[galaxy_name]
                }
                        
        except Exception as e:
            logger.warning(f"Could not extract coordinates for {galaxy_name}: {e}")
    
    return positions

def calculate_alpha_fe_gradient_with_re(radial_profile, effective_radius):
    """
    Calculate α/Fe gradient using proper R/Re normalization with error estimation
    Following Liu Yiqing 2016 and Zhengzheng Li 2019 methodology:
    Linear model: [α/Fe](R) = [α/Fe]₀ + ∇[α/Fe] × (R/Re)
    
    Parameters:
    -----------
    radial_profile : dict
        Radial profile containing bin_radii, alpha_fe_mean, alpha_fe_error
    effective_radius : float
        Effective radius in arcsec (Re)
        
    Returns:
    --------
    dict
        Gradient results with slope, error, statistics
    """
    try:
        if not radial_profile or 'alpha_fe_mean' not in radial_profile:
            logger.warning("No valid radial profile data")
            return None
        
        alpha_fe = radial_profile['alpha_fe_mean']
        radii = radial_profile['bin_radii']
        errors = radial_profile.get('alpha_fe_error', np.full_like(alpha_fe, 0.05))
        
        # Use first 3 bins (RDB method) like original analysis
        if len(alpha_fe) < 3 or len(radii) < 3:
            logger.warning(f"Insufficient bins: {len(alpha_fe)}")
            return None
        
        alpha_fe_3bins = alpha_fe[:3]
        radii_3bins = radii[:3]  # These are already in arcsec
        errors_3bins = errors[:3]
        
        # Remove NaN values
        valid_mask = np.isfinite(alpha_fe_3bins) & np.isfinite(radii_3bins) & np.isfinite(errors_3bins)
        if np.sum(valid_mask) < 2:
            logger.warning(f"Insufficient valid data points: {np.sum(valid_mask)}")
            return None
        
        alpha_valid = alpha_fe_3bins[valid_mask]
        radii_valid = radii_3bins[valid_mask]
        errors_valid = errors_3bins[valid_mask]
        
        # Convert radii to R/Re units (CRITICAL for proper gradient calculation)
        radii_normalized = radii_valid / effective_radius
        
        logger.info(f"  Radii (arcsec): {radii_valid}")
        logger.info(f"  Radii (R/Re): {radii_normalized}")
        logger.info(f"  α/Fe values: {alpha_valid}")
        logger.info(f"  Effective radius Re = {effective_radius:.3f} arcsec")
        
        # Perform weighted linear regression: α/Fe vs R/Re
        if len(errors_valid) > 0 and np.all(errors_valid > 0):
            # Weighted least squares
            weights = 1.0 / errors_valid**2
            
            # Calculate weighted regression coefficients
            W = np.sum(weights)
            Wx = np.sum(weights * radii_normalized)
            Wy = np.sum(weights * alpha_valid)
            Wxx = np.sum(weights * radii_normalized**2)
            Wxy = np.sum(weights * radii_normalized * alpha_valid)
            
            det = W * Wxx - Wx**2
            if det > 0:
                slope = (W * Wxy - Wx * Wy) / det
                intercept = (Wxx * Wy - Wx * Wxy) / det
                slope_error = np.sqrt(W / det)
                intercept_error = np.sqrt(Wxx / det)
            else:
                logger.warning("Singular matrix in weighted regression")
                return None
        else:
            # Ordinary least squares fallback
            from scipy import stats
            slope, intercept, r_value, p_value, slope_error = stats.linregress(radii_normalized, alpha_valid)
            intercept_error = slope_error * np.sqrt(np.sum(radii_normalized**2) / len(radii_normalized))
        
        # Calculate statistics
        predicted = slope * radii_normalized + intercept
        residuals = alpha_valid - predicted
        chi_squared = np.sum((residuals / errors_valid)**2) if np.all(errors_valid > 0) else np.sum(residuals**2)
        reduced_chi_squared = chi_squared / (len(alpha_valid) - 2) if len(alpha_valid) > 2 else chi_squared
        
        # R-squared calculation
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((alpha_valid - np.mean(alpha_valid))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Statistical significance
        significance_ratio = abs(slope) / slope_error if slope_error > 0 else 0
        
        result = {
            'slope': slope,  # d[α/Fe]/d(R/Re) in dex per effective radius
            'slope_error': slope_error,
            'intercept': intercept,  # Central α/Fe value
            'intercept_error': intercept_error,
            'r_squared': r_squared,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'n_points': len(alpha_valid),
            'significance_ratio': significance_ratio,
            'radii_re': radii_normalized,
            'alpha_fe_fit': alpha_valid,
            'errors_fit': errors_valid,
            'predicted': predicted,
            'residuals': residuals,
            'effective_radius': effective_radius
        }
        
        logger.info(f"  Gradient: {slope:.4f} ± {slope_error:.4f} dex/Re")
        logger.info(f"  Intercept: {intercept:.4f} ± {intercept_error:.4f} dex")
        logger.info(f"  R²: {r_squared:.3f}, χ²ᵣ: {reduced_chi_squared:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating α/Fe gradient: {e}")
        return None

def create_virgo_cluster_plot_corrected_method():
    """Create Virgo Cluster visualization with corrected gradient calculation (total change)"""
    
    # Get real IFU positions
    logger.info("Extracting real IFU positions from MUSE FITS files...")
    galaxy_coords = get_real_ifu_positions()
    
    if not galaxy_coords:
        logger.error("No galaxy coordinates found!")
        return
    
    logger.info("Extracting α/Fe gradients for all galaxies (corrected R/Re method)...")
    
    # Extract α/Fe gradients using proper R/Re normalization
    galaxy_gradients = {}
    for galaxy_name in galaxy_coords.keys():
        try:
            logger.info(f"Processing {galaxy_name}...")
            result = analyze_single_galaxy(galaxy_name)
            
            if result and result.get('analysis_success', False):
                # Extract radial profile and effective radius
                radial_profile = result.get('radial_profile')
                effective_radius = result.get('effective_radius')
                
                if radial_profile and effective_radius and effective_radius > 0:
                    # Calculate gradient using proper R/Re normalization
                    gradient_result = calculate_alpha_fe_gradient_with_re(radial_profile, effective_radius)
                    
                    if gradient_result and np.isfinite(gradient_result['slope']):
                        galaxy_gradients[galaxy_name] = {
                            'gradient': gradient_result['slope'],
                            'gradient_error': gradient_result['slope_error'],
                            'intercept': gradient_result['intercept'],
                            'r_squared': gradient_result['r_squared'],
                            'effective_radius': effective_radius,
                            'alpha_fe_values': radial_profile['alpha_fe_mean'][:3],
                            'radii_arcsec': radial_profile['bin_radii'][:3],
                            'radii_re': gradient_result['radii_re'],
                            'errors': radial_profile.get('alpha_fe_error', np.full(3, 0.05))[:3]
                        }
                        logger.info(f"  Gradient: {gradient_result['slope']:+.4f} ± {gradient_result['slope_error']:.4f} dex/Re")
                else:
                    logger.warning(f"Missing radial profile or effective radius for {galaxy_name}")
                        
        except Exception as e:
            logger.warning(f"Error processing {galaxy_name}: {e}")
    
    # Create the plot - MATCHING THE ATTACHED IMAGE STYLE
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Set up velocity color mapping
    velocities = [coords['velocity'] for coords in galaxy_coords.values()]
    vel_min, vel_max = min(velocities), max(velocities)
    norm = Normalize(vmin=vel_min, vmax=vel_max)
    cmap = plt.cm.plasma
    
    logger.info("Creating Virgo Cluster visualization with corrected gradient calculation...")
    
    # Add major galaxies first (like in the image)
    # M87 (central)
    m87_ra, m87_dec = 187.70591, 12.39112
    ax.scatter(m87_ra, m87_dec, s=800, marker='*', c='gold', 
              edgecolors='black', linewidth=3, label='M87', zorder=10)
    ax.text(m87_ra, m87_dec-0.2, 'M87', ha='center', va='top', fontsize=14, 
           fontweight='bold', color='black')
    
    # M86
    m86_ra, m86_dec = 186.54829, 12.95668
    ax.scatter(m86_ra, m86_dec, s=600, marker='*', c='gold', 
              edgecolors='black', linewidth=2, zorder=10)
    ax.text(m86_ra, m86_dec-0.15, 'M86', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='black')
    
    # M60
    m60_ra, m60_dec = 190.91684, 11.55217
    ax.scatter(m60_ra, m60_dec, s=600, marker='*', c='gold', 
              edgecolors='black', linewidth=2, zorder=10)
    ax.text(m60_ra, m60_dec-0.15, 'M60', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='black')
    
    # M49
    m49_ra, m49_dec = 187.44419, 8.00003
    ax.scatter(m49_ra, m49_dec, s=600, marker='*', c='gold', 
              edgecolors='black', linewidth=2, zorder=10)
    ax.text(m49_ra, m49_dec-0.15, 'M49', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='black')
    
    # Plot each galaxy with gradient arrows - PROPER GRADIENT VALUES
    for galaxy_name, coords in galaxy_coords.items():
        ra = coords['ra']
        dec = coords['dec']
        velocity = coords['velocity']
        has_emission = coords['has_emission']
        
        # Get gradient data
        gradient_data = galaxy_gradients.get(galaxy_name)
        
        # Galaxy markers - triangles like in the image
        if gradient_data:
            gradient = gradient_data['gradient']
            gradient_error = gradient_data['gradient_error']
            
            # Color and marker based on gradient direction and magnitude
            # Using proper significance testing
            significance = abs(gradient) / gradient_error if gradient_error > 0 else 0
            
            if significance > 2.0:  # Significant gradient (2σ)
                if gradient > 0:  # Positive gradient: α/Fe increases outward
                    marker = '^'
                    color = 'blue'
                    marker_size = 300
                    gradient_type = 'positive'
                else:  # Negative gradient: α/Fe decreases outward  
                    marker = 'v'  
                    color = 'red'
                    marker_size = 300
                    gradient_type = 'negative'
            else:  # Not significant - flat profile
                marker = 'o'
                color = 'gray'
                marker_size = 200
                gradient_type = 'flat'
            
            # Filled vs hollow based on emission lines
            if has_emission:
                facecolor = color
                edgecolor = 'black'
                alpha = 0.8
            else:
                facecolor = 'none'
                edgecolor = color
                alpha = 0.9
            
            ax.scatter(ra, dec, c=facecolor, s=marker_size, marker=marker,
                      edgecolors=edgecolor, linewidth=2, alpha=alpha, zorder=5)
            
            # Add gradient arrow - length proportional to significance
            if significance > 1.0:
                arrow_length = min(significance / 5.0, 0.3)  # Max 0.3 degrees
                
                # Arrow direction based on gradient sign
                if gradient > 0:
                    angle = 45  # Northeast for positive gradients
                else:
                    angle = 225  # Southwest for negative gradients
                    
                angle_rad = np.radians(angle)
                dx = arrow_length * np.cos(angle_rad)
                dy = arrow_length * np.sin(angle_rad)
                
                ax.arrow(ra, dec, dx, dy, head_width=0.05, head_length=0.03, 
                        fc='white', ec='black', linewidth=2, alpha=0.9, zorder=6)
            
            # Add error bars
            error_size = gradient_error * 0.1  # Scale error bars
            ax.errorbar(ra, dec, xerr=error_size, yerr=error_size, 
                       fmt='none', ecolor='black', elinewidth=1, alpha=0.6, zorder=4)
            
            # Add gradient value label (proper scientific notation)
            if abs(gradient) >= 0.01:
                label_text = f"{gradient:+.3f}±{gradient_error:.3f}"
            else:
                label_text = f"{gradient:+.4f}±{gradient_error:.4f}"
                
            ax.text(ra, dec-0.1, label_text, ha='center', va='top', 
                   fontsize=9, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Add galaxy name
            ax.text(ra+0.1, dec+0.1, galaxy_name.replace('VCC', ''), 
                   ha='left', va='bottom', fontsize=9, color='black',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    
    # Add cluster structure (gray shaded regions like in image)
    # Virgo A (M87 region)
    virgo_a_circle = plt.Circle((m87_ra, m87_dec), 1.5, 
                               facecolor='lightgray', alpha=0.3, zorder=1)
    ax.add_patch(virgo_a_circle)
    ax.text(m87_ra, m87_dec-1.8, 'M87/Cluster A', ha='center', va='center',
           fontsize=12, fontweight='bold', color='gray')
    
    # M49 region (Virgo B)
    virgo_b_circle = plt.Circle((m49_ra, m49_dec), 1.0, 
                               facecolor='lightgray', alpha=0.3, zorder=1)
    ax.add_patch(virgo_b_circle)
    ax.text(m49_ra, m49_dec-1.2, 'M49/Cluster B', ha='center', va='center',
           fontsize=12, fontweight='bold', color='gray')
    
    # M60 region (W Cloud)
    virgo_w_circle = plt.Circle((m60_ra, m60_dec), 0.8, 
                               facecolor='lightgray', alpha=0.3, zorder=1)
    ax.add_patch(virgo_w_circle)
    ax.text(m60_ra-0.5, m60_dec+1.0, 'M60/W Cloud', ha='center', va='center',
           fontsize=10, fontweight='bold', color='gray')
    
    # Formatting to match the image
    ax.set_xlabel('RA (deg)', fontsize=16, fontweight='bold')
    ax.set_ylabel('DEC (deg)', fontsize=16, fontweight='bold')
    ax.set_title('Virgo Cluster Galaxies: [α/Fe] Radial Gradients d[α/Fe]/d(R/Re)\n(Corrected R/Re Normalization Method)', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Set equal aspect ratio and limits based on real coordinates
    ax.set_aspect('equal')
    if galaxy_coords:
        ras = [coords['ra'] for coords in galaxy_coords.values()]
        decs = [coords['dec'] for coords in galaxy_coords.values()]
        ra_range = max(ras) - min(ras)
        dec_range = max(decs) - min(decs)
        
        padding = 0.5
        ax.set_xlim(min(ras) - padding, max(ras) + padding)
        ax.set_ylim(min(decs) - padding, max(decs) + padding)
    
    # Invert RA axis (astronomical convention)
    ax.invert_xaxis()
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create legend matching the image
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='^', color='blue', markersize=14, 
               fillstyle='full', markeredgecolor='black', linewidth=0,
               label='Positive α/Fe gradient (d[α/Fe]/d(R/Re) > 0)'),
        Line2D([0], [0], marker='v', color='red', markersize=14, 
               fillstyle='full', markeredgecolor='black', linewidth=0,
               label='Negative α/Fe gradient (d[α/Fe]/d(R/Re) < 0)'),
        Line2D([0], [0], marker='o', color='gray', markersize=12, 
               fillstyle='full', markeredgecolor='black', linewidth=0,
               label='Flat α/Fe profile (|gradient| < 2σ)'),
        Line2D([0], [0], marker='^', color='blue', markersize=14, 
               fillstyle='full', markeredgecolor='black', linewidth=0,
               label='With emission lines (filled)'),
        Line2D([0], [0], marker='^', color='w', markersize=14, 
               fillstyle='none', markeredgecolor='blue', linewidth=2,
               label='Without emission lines (hollow)'),
        Line2D([0], [0], marker='*', color='gold', markersize=16, 
               markeredgecolor='black', linewidth=2,
               label='Cluster/Subcluster Center')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
             frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    
    # Add explanation text
    explanation = ("Blue symbols: α/Fe increases with radius (d[α/Fe]/d(R/Re) > 0) | Red symbols: α/Fe decreases with radius (d[α/Fe]/d(R/Re) < 0)\n"
                  "Triangles: Significant gradients (>2σ) | Circles: Flat profiles (|gradient| < 2σ)\n"
                  "Filled symbols: Galaxies with emission lines | Hollow symbols: Galaxies without emission lines\n"
                  "Values show gradient in dex per effective radius with ±1σ uncertainties")
    
    ax.text(0.5, 0.02, explanation, transform=ax.transAxes, 
           ha='center', va='bottom', fontsize=11, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Add scale reference
    scale_text = '1° ≈ 0.29 Mpc'
    ax.text(0.98, 0.02, scale_text, transform=ax.transAxes, 
           ha='right', va='bottom', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Add gradient statistics
    if galaxy_gradients:
        gradients = [data['gradient'] for data in galaxy_gradients.values()]
        errors = [data['gradient_error'] for data in galaxy_gradients.values()]
        
        # Count significant gradients
        significance_ratios = [abs(g)/e if e > 0 else 0 for g, e in zip(gradients, errors)]
        n_positive = sum(1 for g, s in zip(gradients, significance_ratios) if g > 0 and s > 2)
        n_negative = sum(1 for g, s in zip(gradients, significance_ratios) if g < 0 and s > 2)
        n_flat = sum(1 for s in significance_ratios if s <= 2)
        
        mean_gradient = np.mean(gradients)
        std_gradient = np.std(gradients)
        
        stats_text = f'Gradient Statistics (N={len(gradients)}):\n' \
                    f'Mean d[α/Fe]/d(R/Re): {mean_gradient:+.4f} ± {std_gradient:.4f}\n' \
                    f'Significant positive: {n_positive}\n' \
                    f'Significant negative: {n_negative}\n' \
                    f'Flat profiles: {n_flat}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='lightblue', alpha=0.9), fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('./enhanced_radial_plots')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "virgo_cluster_proper_gradient_method.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Proper gradient Virgo Cluster plot saved: {output_file}")
    logger.info(f"Processed {len(galaxy_gradients)} galaxies with valid α/Fe gradients")
    
    # Print summary with proper gradient values
    if galaxy_gradients:
        print("\n" + "="*80)
        print("VIRGO CLUSTER α/Fe GRADIENT SUMMARY - PROPER R/Re METHOD")
        print("="*80)
        print("Values show d[α/Fe]/d(R/Re) gradients in dex per effective radius")
        print("="*80)
        for galaxy_name, data in galaxy_gradients.items():
            gradient = data['gradient']
            error = data['gradient_error']
            significance = abs(gradient) / error if error > 0 else 0
            
            if significance > 3:
                sig_level = "***"  # Highly significant
            elif significance > 2:
                sig_level = "**"   # Significant  
            elif significance > 1:
                sig_level = "*"    # Marginal
            else:
                sig_level = ""     # Not significant
                
            direction = "↗" if gradient > 0 else "↘"
            gradient_type = "positive" if gradient > 0 else "negative"
            
            print(f"{galaxy_name}: {gradient:+.4f} ± {error:.4f} dex/Re {direction} {sig_level} ({gradient_type})")
            print(f"          Re = {data['effective_radius']:.2f}\", R²={data['r_squared']:.3f}")
        print("="*80)
        print("Significance: *** >3σ, ** >2σ, * >1σ")
        print("NOTE: Proper gradient calculation using R/Re normalization")
        print("="*80)

if __name__ == "__main__":
    create_virgo_cluster_plot_corrected_method()
