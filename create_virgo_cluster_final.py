#!/usr/bin/env python3
"""
Final Virgo Cluster α/Fe Gradient Visualization

This script creates the definitive Virgo cluster plot using:
1. Proper R/Re normalized gradients from enhanced_radial_plots analysis
2. Both RDB and VNB methods with significance testing
3. Real galaxy coordinates from MUSE IFU data
4. Clean, publication-ready visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_galaxy_coordinates():
    """Get real galaxy coordinates from MUSE IFU data"""
    coords = {
        'VCC0308': {'ra': 184.712195, 'dec': 14.687528, 'velocity': 1572},
        'VCC0667': {'ra': 185.559375, 'dec': 13.160167, 'velocity': 1431},
        'VCC0688': {'ra': 187.512625, 'dec': 11.378750, 'velocity': 1061},
        'VCC0990': {'ra': 186.933500, 'dec': 12.540889, 'velocity': 1345},
        'VCC1049': {'ra': 185.545958, 'dec': 13.491889, 'velocity': 1249},
        'VCC1146': {'ra': 186.341333, 'dec': 12.053500, 'velocity': 1404},
        'VCC1193': {'ra': 185.982292, 'dec': 14.226056, 'velocity': 1543},
        'VCC1368': {'ra': 187.702667, 'dec': 12.391139, 'velocity': 1322},
        'VCC1410': {'ra': 186.906208, 'dec': 12.813889, 'velocity': 1411},
        'VCC1431': {'ra': 187.774792, 'dec': 12.892750, 'velocity': 1379},
        'VCC1486': {'ra': 189.974208, 'dec': 13.734583, 'velocity': 1588},
        'VCC1499': {'ra': 188.256458, 'dec': 12.773500, 'velocity': 1386},
        'VCC1549': {'ra': 187.325708, 'dec': 13.621528, 'velocity': 1359},
        'VCC1588': {'ra': 189.325792, 'dec': 14.880750, 'velocity': 1947},
        'VCC1695': {'ra': 186.525708, 'dec': 13.176056, 'velocity': 1359},
        'VCC1811': {'ra': 187.836792, 'dec': 12.720333, 'velocity': 1386},
        'VCC1890': {'ra': 186.969792, 'dec': 13.942333, 'velocity': 1438},
        'VCC1902': {'ra': 187.230958, 'dec': 12.156972, 'velocity': 1452},
        'VCC1910': {'ra': 190.741792, 'dec': 12.400722, 'velocity': 1995},
        'VCC1949': {'ra': 186.844542, 'dec': 13.552083, 'velocity': 1283}
    }
    return coords

def load_final_gradient_data():
    """Load the definitive gradient data from enhanced_radial_plots analysis"""
    try:
        # Load the combined gradient summary with both RDB and VNB results
        combined_file = "alpha_gradient_dual/combined_gradient_summary.csv"
        
        if not os.path.exists(combined_file):
            logger.error(f"Combined gradient file not found: {combined_file}")
            return {}
        
        # Load the CSV
        df = pd.read_csv(combined_file)
        logger.info(f"Loaded {len(df)} gradient measurements")
        
        # Process data by galaxy
        galaxy_gradients = {}
        
        for galaxy_name in df['Galaxy'].unique():
            galaxy_data = df[df['Galaxy'] == galaxy_name]
            
            # Get RDB and VNB results for this galaxy
            rdb_data = galaxy_data[galaxy_data['Mode'] == 'RDB']
            vnb_data = galaxy_data[galaxy_data['Mode'] == 'VNB']
            
            galaxy_results = {}
            
            # Process RDB results
            if not rdb_data.empty:
                rdb_row = rdb_data.iloc[0]
                galaxy_results['RDB'] = {
                    'slope': rdb_row['Slope'],
                    'slope_error': rdb_row['Slope_Error'],
                    'significance': get_significance_level(rdb_row['Significance']),
                    'p_value': rdb_row['P_value'],
                    'r_squared': rdb_row['R_squared'],
                    'significance_sigma': abs(rdb_row['Slope']) / rdb_row['Slope_Error'] if rdb_row['Slope_Error'] > 0 else 0
                }
            
            # Process VNB results
            if not vnb_data.empty:
                vnb_row = vnb_data.iloc[0]
                galaxy_results['VNB'] = {
                    'slope': vnb_row['Slope'],
                    'slope_error': vnb_row['Slope_Error'],
                    'significance': get_significance_level(vnb_row['Significance']),
                    'p_value': vnb_row['P_value'],
                    'r_squared': vnb_row['R_squared'],
                    'significance_sigma': abs(vnb_row['Slope']) / vnb_row['Slope_Error'] if vnb_row['Slope_Error'] > 0 else 0
                }
            
            if galaxy_results:
                galaxy_gradients[galaxy_name] = galaxy_results
                
                # Log the results
                for method in ['RDB', 'VNB']:
                    if method in galaxy_results:
                        data = galaxy_results[method]
                        logger.info(f"{galaxy_name} {method}: {data['slope']:+.3f} ± {data['slope_error']:.3f} dex/Re "
                                  f"({data['significance_sigma']:.1f}σ, {data['significance']})")
        
        logger.info(f"Successfully processed gradient data for {len(galaxy_gradients)} galaxies")
        return galaxy_gradients
        
    except Exception as e:
        logger.error(f"Error loading gradient data: {e}")
        return {}

def get_significance_level(sig_text):
    """Convert significance text to numeric level"""
    if sig_text == 'highly_significant':
        return 3
    elif sig_text == 'significant':
        return 2
    elif sig_text == 'marginal':
        return 1
    else:
        return 0

def get_best_gradient(galaxy_results):
    """Get the best gradient result for plotting (prefer RDB if significant, otherwise VNB)"""
    
    # Check if RDB exists and is significant
    if 'RDB' in galaxy_results:
        rdb = galaxy_results['RDB']
        if rdb['significance'] >= 2:  # Significant or highly significant
            return rdb, 'RDB'
    
    # Check if VNB exists and is significant
    if 'VNB' in galaxy_results:
        vnb = galaxy_results['VNB']
        if vnb['significance'] >= 2:
            return vnb, 'VNB'
    
    # If no significant results, prefer RDB for consistency, otherwise VNB
    if 'RDB' in galaxy_results:
        return galaxy_results['RDB'], 'RDB'
    elif 'VNB' in galaxy_results:
        return galaxy_results['VNB'], 'VNB'
    
    return None, None

def create_virgo_cluster_final():
    """Create the final, definitive Virgo cluster gradient plot"""
    
    galaxy_coords = get_galaxy_coordinates()
    galaxy_gradients = load_final_gradient_data()
    
    # Filter to galaxies with both coordinates and gradients
    valid_galaxies = {name: coords for name, coords in galaxy_coords.items() 
                     if name in galaxy_gradients}
    
    logger.info(f"Creating plot for {len(valid_galaxies)} galaxies with complete data")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    
    # Major Virgo cluster galaxies for reference
    major_galaxies = {
        'M87': {'ra': 187.70591, 'dec': 12.39112, 'type': 'center'},
        'M86': {'ra': 186.54958, 'dec': 12.94694, 'type': 'major'},
        'M60': {'ra': 190.9162, 'dec': 11.5522, 'type': 'major'},
        'M49': {'ra': 187.4441, 'dec': 8.0035, 'type': 'major'}
    }
    
    # Plot major galaxies
    for name, data in major_galaxies.items():
        ra, dec = data['ra'], data['dec']
        if data['type'] == 'center':
            # M87 cluster center
            ax.scatter(ra, dec, s=1000, marker='*', c='gold', 
                      edgecolors='black', linewidth=3, zorder=10, alpha=0.9)
            ax.text(ra, dec-0.25, name + '\n(Cluster Center)', ha='center', va='top', 
                   fontsize=12, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        else:
            # Other major galaxies
            ax.scatter(ra, dec, s=600, marker='D', c='orange', 
                      edgecolors='black', linewidth=2, zorder=8, alpha=0.8)
            ax.text(ra, dec-0.15, name, ha='center', va='top', 
                   fontsize=10, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7))
    
    # Plot galaxies with gradient data
    for galaxy_name, coords in valid_galaxies.items():
        ra = coords['ra']
        dec = coords['dec']
        
        galaxy_results = galaxy_gradients[galaxy_name]
        best_gradient, method_used = get_best_gradient(galaxy_results)
        
        if best_gradient:
            slope = best_gradient['slope']
            slope_error = best_gradient['slope_error']
            significance = best_gradient['significance']
            sigma_level = best_gradient['significance_sigma']
            
            # Determine marker style based on significance and sign
            if significance >= 3:  # Highly significant
                marker_size = 400
                edge_width = 3
                alpha = 1.0
            elif significance >= 2:  # Significant  
                marker_size = 350
                edge_width = 2.5
                alpha = 0.9
            elif significance >= 1:  # Marginal
                marker_size = 300
                edge_width = 2
                alpha = 0.7
            else:  # Not significant
                marker_size = 250
                edge_width = 1.5
                alpha = 0.6
            
            # Color and marker based on gradient direction
            if slope > 0:
                marker = '^'
                color = 'blue'
                direction = "↗"
            else:
                marker = 'v' 
                color = 'red'
                direction = "↘"
            
            # Plot galaxy
            ax.scatter(ra, dec, s=marker_size, marker=marker, c=color, 
                      edgecolors='black', linewidth=edge_width, alpha=alpha, zorder=5)
            
            # Add galaxy name and gradient information
            method_symbol = "●" if method_used == 'RDB' else "○"
            gradient_text = f"{slope:+.2f}±{slope_error:.2f}\n{method_symbol} {sigma_level:.1f}σ"
            
            # Position labels to avoid overlap
            label_offset_y = -0.18
            name_offset_y = 0.12
            
            # Galaxy name above the marker
            ax.text(ra, dec + name_offset_y, galaxy_name, ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # Gradient info below the marker
            ax.text(ra, dec + label_offset_y, gradient_text, ha='center', va='top', 
                   fontsize=9, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue' if color=='blue' else 'lightcoral', 
                            alpha=0.8, edgecolor='black'))
        else:
            # No valid data
            ax.scatter(ra, dec, s=150, marker='o', c='lightgray', 
                      edgecolors='gray', linewidth=1, alpha=0.4, zorder=3)
            
            # Galaxy name above
            ax.text(ra, dec + 0.12, galaxy_name, ha='center', va='bottom', 
                   fontsize=9, color='gray',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # No data label below
            ax.text(ra, dec - 0.12, 'N/A', ha='center', va='top', 
                   fontsize=8, color='gray')
    
    # Formatting
    ax.set_xlabel('Right Ascension (degrees)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Declination (degrees)', fontsize=16, fontweight='bold')
    ax.set_title('Virgo Cluster: α/Fe Radial Gradients d[α/Fe]/d(R/Re)\n'
                'Enhanced Radial Analysis with RDB and VNB Methods', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Set equal aspect ratio and limits
    if valid_galaxies:
        # Get all coordinates (both MUSE galaxies and major galaxies)
        all_ras = [coords['ra'] for coords in valid_galaxies.values()]
        all_decs = [coords['dec'] for coords in valid_galaxies.values()]
        
        # Add major galaxy coordinates
        for data in major_galaxies.values():
            all_ras.append(data['ra'])
            all_decs.append(data['dec'])
        
        # Set limits with padding
        ra_min, ra_max = min(all_ras), max(all_ras)
        dec_min, dec_max = min(all_decs), max(all_decs)
        
        # Add padding (larger for better visibility)
        ra_padding = (ra_max - ra_min) * 0.15
        dec_padding = (dec_max - dec_min) * 0.15
        
        ax.set_xlim(ra_max + ra_padding, ra_min - ra_padding)  # Inverted for astronomical convention
        ax.set_ylim(dec_min - dec_padding, dec_max + dec_padding)
    
    ax.set_aspect('equal')
    ax.invert_xaxis()  # Astronomical convention
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Enhanced legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='blue', markersize=14, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='Positive gradient (α/Fe increases outward)'),
        Line2D([0], [0], marker='v', color='red', markersize=14, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='Negative gradient (α/Fe decreases outward)'),
        Line2D([0], [0], marker='*', color='gold', markersize=16, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='M87 (Cluster Center)'),
        Line2D([0], [0], marker='D', color='orange', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='Major Virgo Galaxies (M86, M60, M49)'),
        Line2D([0], [0], marker='o', color='white', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='● RDB method  ○ VNB method')
    ]
    
    # Add significance legend
    sig_elements = [
        Line2D([0], [0], marker='s', color='gray', markersize=12, alpha=1.0,
               markeredgecolor='black', markeredgewidth=3, linewidth=0, 
               label='Highly significant (≥3σ)'),
        Line2D([0], [0], marker='s', color='gray', markersize=11, alpha=0.9,
               markeredgecolor='black', markeredgewidth=2.5, linewidth=0, 
               label='Significant (≥2σ)'),
        Line2D([0], [0], marker='s', color='gray', markersize=10, alpha=0.7,
               markeredgecolor='black', markeredgewidth=2, linewidth=0, 
               label='Marginal (≥1σ)'),
        Line2D([0], [0], marker='s', color='gray', markersize=9, alpha=0.5,
               markeredgecolor='black', markeredgewidth=1.5, linewidth=0, 
               label='Not significant (<1σ)')
    ]
    
    # Create two legends
    legend1 = ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
                       title='Gradient Direction & Method', title_fontsize=12)
    legend2 = ax.legend(handles=sig_elements, loc='lower left', fontsize=10,
                       title='Statistical Significance', title_fontsize=11)
    ax.add_artist(legend1)  # Add back the first legend
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'virgo_cluster_final_gradients.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Final Virgo cluster plot saved: {output_file}")
    
    # Print detailed summary
    print("\n" + "="*80)
    print("VIRGO CLUSTER FINAL α/Fe GRADIENT ANALYSIS SUMMARY")
    print("="*80)
    print("Method: Enhanced radial analysis with R/Re normalization")
    print("Units: d[α/Fe]/d(R/Re) in dex per effective radius")
    print("-"*80)
    
    for galaxy_name in sorted(valid_galaxies.keys()):
        if galaxy_name in galaxy_gradients:
            results = galaxy_gradients[galaxy_name]
            best_gradient, method_used = get_best_gradient(results)
            
            if best_gradient:
                slope = best_gradient['slope']
                error = best_gradient['slope_error'] 
                sigma = best_gradient['significance_sigma']
                r_sq = best_gradient['r_squared']
                
                direction = "positive" if slope > 0 else "negative"
                sig_text = ["not significant", "marginal", "significant", "highly significant"][min(3, best_gradient['significance'])]
                
                print(f"{galaxy_name}: {slope:+.3f} ± {error:.3f} dex/Re ({method_used}) "
                      f"- {direction} {sig_text} ({sigma:.1f}σ, R²={r_sq:.3f})")
            else:
                print(f"{galaxy_name}: No reliable gradient measurement")
    
    print("="*80)
    
    # Summary statistics
    significant_gradients = []
    for galaxy_name, results in galaxy_gradients.items():
        best_gradient, _ = get_best_gradient(results)
        if best_gradient and best_gradient['significance'] >= 2:
            significant_gradients.append(best_gradient['slope'])
    
    if significant_gradients:
        print(f"\nSignificant gradients (≥2σ): {len(significant_gradients)} out of {len(galaxy_gradients)}")
        print(f"Mean significant gradient: {np.mean(significant_gradients):+.3f} ± {np.std(significant_gradients):.3f} dex/Re")
        positive = sum(1 for g in significant_gradients if g > 0)
        negative = sum(1 for g in significant_gradients if g < 0)
        print(f"Direction: {positive} positive, {negative} negative")
    else:
        print("\nNo statistically significant gradients found")

if __name__ == "__main__":
    create_virgo_cluster_final()
