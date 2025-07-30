#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_galaxy_coordinates():
    """Get real galaxy coordinates"""
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

def get_galaxy_coordinates():

def load_gradient_data():
    """Load gradient data from the summary CSV file"""
    try:
        # Use the latest analysis directory
        analysis_dir = "alpha_fe_analysis_results/analysis_20250720_135902"
        summary_file = os.path.join(analysis_dir, "alpha_fe_analysis_summary.csv")
        
        if not os.path.exists(summary_file):
            logger.error(f"Summary file not found: {summary_file}")
            return {}
        
        # Load the CSV
        df = pd.read_csv(summary_file)
        
        galaxy_gradients = {}
        for _, row in df.iterrows():
            galaxy_name = row['Galaxy']
            slope = row['Gradient_Slope']
            slope_error = row['Gradient_Error']
            
            # Check if data is valid
            if pd.notna(slope) and pd.notna(slope_error) and slope_error > 0:
                significance = abs(slope) / slope_error
                
                galaxy_gradients[galaxy_name] = {
                    'gradient': slope,
                    'gradient_error': slope_error,
                    'significance': significance,
                    'effective_radius': 12.0,  # Default value
                    'r_squared': row.get('Gradient_R_Value', 0.0) ** 2 if pd.notna(row.get('Gradient_R_Value')) else 0.0
                }
                
                logger.info(f"{galaxy_name}: {slope:+.4f} ± {slope_error:.4f} dex/Re (significance: {significance:.1f}σ)")
        
        logger.info(f"Loaded gradient data for {len(galaxy_gradients)} galaxies")
        return galaxy_gradients
        
    except Exception as e:
        logger.error(f"Error loading gradient data: {e}")
        return {}

def create_virgo_plot():
    """Create clean Virgo cluster plot"""
    
    galaxy_coords = get_galaxy_coordinates()
    logger.info(f"Processing {len(galaxy_coords)} galaxies...")
    
    # Load gradient data from CSV
    galaxy_gradients = load_gradient_data()
    
    # Filter to only include galaxies we have coordinates for
    galaxy_gradients = {name: data for name, data in galaxy_gradients.items() if name in galaxy_coords}
    
    logger.info(f"Successfully loaded {len(galaxy_gradients)} galaxies with both coordinates and gradients")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # M87 center
    m87_ra, m87_dec = 187.70591, 12.39112
    ax.scatter(m87_ra, m87_dec, s=600, marker='*', c='gold', 
              edgecolors='black', linewidth=2, label='M87', zorder=10)
    ax.text(m87_ra, m87_dec-0.15, 'M87', ha='center', va='top', fontsize=12, 
           fontweight='bold', color='black')
    
    # Plot galaxies
    for galaxy_name, coords in galaxy_coords.items():
        ra = coords['ra']
        dec = coords['dec']
        
        gradient_data = galaxy_gradients.get(galaxy_name)
        
        if gradient_data:
            gradient = gradient_data['gradient']
            gradient_error = gradient_data['gradient_error']
            significance = gradient_data['significance']
            
            # Determine marker based on significance and sign
            if significance > 2.0:  # Significant
                if gradient > 0:
                    marker = '^'
                    color = 'blue'
                    size = 250
                else:
                    marker = 'v'
                    color = 'red'
                    size = 250
            else:  # Not significant
                marker = 'o'
                color = 'gray'
                size = 150
            
            # Plot galaxy
            ax.scatter(ra, dec, s=size, marker=marker, c=color, 
                      edgecolors='black', linewidth=1.5, alpha=0.8, zorder=5)
            
            # Add gradient value
            label_text = f"{gradient:+.3f}"
            ax.text(ra, dec-0.08, label_text, ha='center', va='top', 
                   fontsize=8, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
        else:
            # No data - gray circle
            ax.scatter(ra, dec, s=100, marker='o', c='lightgray', 
                      edgecolors='gray', linewidth=1, alpha=0.5, zorder=3)
    
    # Formatting
    ax.set_xlabel('RA (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('DEC (degrees)', fontsize=14, fontweight='bold')
    ax.set_title('Virgo Cluster: α/Fe Radial Gradients d[α/Fe]/d(R/Re)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set limits
    if galaxy_coords:
        ras = [coords['ra'] for coords in galaxy_coords.values()]
        decs = [coords['dec'] for coords in galaxy_coords.values()]
        ax.set_xlim(min(ras) - 0.5, max(ras) + 0.5)
        ax.set_ylim(min(decs) - 0.5, max(decs) + 0.5)
    
    ax.set_aspect('equal')
    ax.invert_xaxis()  # Astronomical convention
    ax.grid(True, alpha=0.3)
    
    # Simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='blue', markersize=12, 
               markeredgecolor='black', linewidth=0, label='Positive gradient (significant)'),
        Line2D([0], [0], marker='v', color='red', markersize=12, 
               markeredgecolor='black', linewidth=0, label='Negative gradient (significant)'),
        Line2D([0], [0], marker='o', color='gray', markersize=10, 
               markeredgecolor='black', linewidth=0, label='Flat/insignificant'),
        Line2D([0], [0], marker='*', color='gold', markersize=14, 
               markeredgecolor='black', linewidth=0, label='M87 cluster center')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_file = 'virgo_cluster_gradients_clean.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Clean plot saved: {output_file}")
    
    # Print summary
    if galaxy_gradients:
        print("\n" + "="*60)
        print("VIRGO CLUSTER α/Fe GRADIENT SUMMARY")
        print("="*60)
        for galaxy_name, data in galaxy_gradients.items():
            gradient = data['gradient']
            error = data['gradient_error']
            significance = data['significance']
            
            sig_marker = "***" if significance > 3 else ("**" if significance > 2 else "*" if significance > 1 else "")
            direction = "↗" if gradient > 0 else "↘"
            
            print(f"{galaxy_name}: {gradient:+.4f} ± {error:.4f} dex/Re {direction} {sig_marker}")
        print("="*60)

if __name__ == "__main__":
    create_virgo_plot()
