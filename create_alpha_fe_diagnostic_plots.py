#!/usr/bin/env python3
"""
Comprehensive α/Fe Diagnostic Plots for ISAPC Analysis Validation

Creates three types of diagnostic plots:
1. Multi-dimensional spectral index to α/Fe relationships with TMB03 models
2. Individual galaxy α/Fe vs R/Re profiles with gradient fits
3. Virgo cluster overview with velocity color-coding

Author: Enhanced ISAPC Analysis System
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import os
from enhanced_alpha_fe_analyzer import ISAPCAlphaFeAnalyzer
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class AlphaFeDiagnosticPlotter:
    """Creates comprehensive diagnostic plots for α/Fe analysis validation"""
    
    def __init__(self):
        """Initialize the diagnostic plotter"""
        self.analyzer = ISAPCAlphaFeAnalyzer()
        self.galaxies = [
            'VCC0308', 'VCC0667', 'VCC0688', 'VCC0990', 'VCC1049',
            'VCC1146', 'VCC1193', 'VCC1368', 'VCC1410', 'VCC1431',
            'VCC1486', 'VCC1499', 'VCC1549', 'VCC1588', 'VCC1695',
            'VCC1811', 'VCC1890', 'VCC1902', 'VCC1910', 'VCC1949'
        ]
        
        # Galaxy velocities (km/s) from NED
        self.galaxy_velocities = {
            'VCC0308': 1124, 'VCC0667': 1405, 'VCC0688': 1149, 'VCC0990': 1842,
            'VCC1049': 1261, 'VCC1146': 1333, 'VCC1193': 1095, 'VCC1368': 1191,
            'VCC1410': 1682, 'VCC1431': 1115, 'VCC1486': 1885, 'VCC1499': 1331,
            'VCC1549': 1144, 'VCC1588': 1553, 'VCC1695': 1293, 'VCC1811': 1594,
            'VCC1890': 1156, 'VCC1902': 1325, 'VCC1910': 1240, 'VCC1949': 1106
        }
        
        # Load results if available
        self.load_existing_results()
        
    def load_existing_results(self):
        """Load existing analysis results"""
        results_file = "enhanced_alpha_fe_results/virgo_cluster_alpha_fe_gradients_enhanced.csv"
        if os.path.exists(results_file):
            self.results_df = pd.read_csv(results_file)
            print(f"Loaded results for {len(self.results_df)} galaxies")
        else:
            print("No existing results found - will create sample data")
            self.results_df = None
            
    def plot_tmb03_index_relationships(self):
        """Plot 1: Multi-dimensional spectral index to α/Fe relationships"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TMB03 Model: Spectral Index to α/Fe Relationships\n(Diagnostic for Methodology Validation)', 
                     fontsize=16, fontweight='bold')
        
        # Load TMB03 model
        tmb03 = self.analyzer.tmb03_model
        if tmb03 is None:
            print("TMB03 model not available")
            return
            
        # Create colormap for [α/Fe]
        alpha_fe_values = tmb03['AoFe'].values
        
        # Plot 1: Fe5015 vs Mgb (main diagnostic)
        ax = axes[0, 0]
        scatter = ax.scatter(tmb03['Fe5015'], tmb03['Mgb'], c=alpha_fe_values, 
                           cmap='RdYlBu_r', s=30, alpha=0.7)
        ax.set_xlabel('Fe5015 Index (Å)')
        ax.set_ylabel('Mgb Index (Å)')
        ax.set_title('Fe5015 vs Mgb - colored by [α/Fe]')
        ax.grid(True, alpha=0.3)
        
        # Add α/Fe contours
        alpha_fe_levels = [0.0, 0.2, 0.4]
        for level in alpha_fe_levels:
            mask = np.abs(alpha_fe_values - level) < 0.05
            if np.sum(mask) > 0:
                ax.plot(tmb03['Fe5015'][mask], tmb03['Mgb'][mask], 
                       'k-', alpha=0.5, linewidth=1)
        
        plt.colorbar(scatter, ax=ax, label='[α/Fe]')
        
        # Plot 2: Fe5015 vs Hβ
        ax = axes[0, 1]
        scatter = ax.scatter(tmb03['Fe5015'], tmb03['Hb'], c=alpha_fe_values, 
                           cmap='RdYlBu_r', s=30, alpha=0.7)
        ax.set_xlabel('Fe5015 Index (Å)')
        ax.set_ylabel('Hβ Index (Å)')
        ax.set_title('Fe5015 vs Hβ - colored by [α/Fe]')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='[α/Fe]')
        
        # Plot 3: Mgb vs Hβ
        ax = axes[0, 2]
        scatter = ax.scatter(tmb03['Mgb'], tmb03['Hb'], c=alpha_fe_values, 
                           cmap='RdYlBu_r', s=30, alpha=0.7)
        ax.set_xlabel('Mgb Index (Å)')
        ax.set_ylabel('Hβ Index (Å)')
        ax.set_title('Mgb vs Hβ - colored by [α/Fe]')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='[α/Fe]')
        
        # Plot 4: Age effects
        ax = axes[1, 0]
        scatter = ax.scatter(tmb03['Age'], tmb03['AoFe'], c=tmb03['ZoH'], 
                           cmap='viridis', s=30, alpha=0.7)
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel('[α/Fe]')
        ax.set_title('Age vs [α/Fe] - colored by [Z/H]')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='[Z/H]')
        
        # Plot 5: Metallicity effects
        ax = axes[1, 1]
        scatter = ax.scatter(tmb03['ZoH'], tmb03['AoFe'], c=tmb03['Age'], 
                           cmap='plasma', s=30, alpha=0.7)
        ax.set_xlabel('[Z/H]')
        ax.set_ylabel('[α/Fe]')
        ax.set_title('[Z/H] vs [α/Fe] - colored by Age')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Age (Gyr)')
        
        # Plot 6: 3D projection (Fe5015, Mgb, α/Fe)
        ax = axes[1, 2]
        # Create a synthetic "depth" using combination of indices
        depth = tmb03['Fe5015'] + tmb03['Mgb']
        scatter = ax.scatter(depth, tmb03['AoFe'], c=tmb03['Age'], 
                           cmap='coolwarm', s=30, alpha=0.7)
        ax.set_xlabel('Fe5015 + Mgb (Combined Index)')
        ax.set_ylabel('[α/Fe]')
        ax.set_title('Combined Index vs [α/Fe] - colored by Age')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Age (Gyr)')
        
        plt.tight_layout()
        plt.savefig('diagnostic_plots/tmb03_index_relationships.png', dpi=300, bbox_inches='tight')
        plt.savefig('diagnostic_plots/tmb03_index_relationships.pdf', bbox_inches='tight')
        plt.show()
        
    def plot_individual_galaxy_profiles(self, max_galaxies=6):
        """Plot 2: Individual galaxy α/Fe vs R/Re profiles with gradient fits"""
        
        # Select galaxies with good data quality
        selected_galaxies = ['VCC1910', 'VCC1949', 'VCC1049', 'VCC1146', 'VCC1368', 'VCC1588']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Individual Galaxy α/Fe Radial Profiles\n(Enhanced ISAPC Analysis with R/Re Normalization)', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, galaxy_name in enumerate(selected_galaxies[:max_galaxies]):
            ax = axes[i]
            
            # Analyze this galaxy
            try:
                result = self.analyzer.analyze_galaxy_gradient(galaxy_name)
                if result is None:
                    ax.text(0.5, 0.5, f'{galaxy_name}\nNo data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{galaxy_name} - No Data')
                    continue
                    
                r_over_re = result['r_over_re']
                alpha_fe_values = result['alpha_fe_values']
                alpha_fe_errors = result['alpha_fe_errors']
                gradient = result['gradient']
                gradient_error = result['gradient_error']
                effective_radius = result['effective_radius']
                
                # Plot data points with error bars
                ax.errorbar(r_over_re, alpha_fe_values, yerr=alpha_fe_errors,
                           fmt='o', markersize=8, capsize=5, capthick=2,
                           color='blue', ecolor='lightblue', alpha=0.8)
                
                # Plot gradient fit
                r_fit = np.linspace(0, max(r_over_re) * 1.1, 100)
                alpha_fe_fit = alpha_fe_values[0] + gradient * r_fit  # Linear fit from center
                ax.plot(r_fit, alpha_fe_fit, 'r--', linewidth=2, alpha=0.7,
                       label=f'Gradient: {gradient:.4f}±{gradient_error:.4f} dex/Re')
                
                # Styling
                ax.set_xlabel('R/Re')
                ax.set_ylabel('[α/Fe]')
                ax.set_title(f'{galaxy_name}\nRe = {effective_radius:.1f}", v = {self.galaxy_velocities.get(galaxy_name, "?")} km/s')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)
                
                # Add significance indicator
                significance = abs(gradient / gradient_error) if gradient_error > 0 else 0
                if significance > 3:
                    ax.text(0.05, 0.95, '***', transform=ax.transAxes, fontsize=16, 
                           color='red', fontweight='bold')
                elif significance > 2:
                    ax.text(0.05, 0.95, '**', transform=ax.transAxes, fontsize=16, 
                           color='orange', fontweight='bold')
                elif significance > 1:
                    ax.text(0.05, 0.95, '*', transform=ax.transAxes, fontsize=16, 
                           color='green', fontweight='bold')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'{galaxy_name}\nError: {str(e)[:30]}...', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{galaxy_name} - Error')
        
        plt.tight_layout()
        plt.savefig('diagnostic_plots/individual_galaxy_profiles.png', dpi=300, bbox_inches='tight')
        plt.savefig('diagnostic_plots/individual_galaxy_profiles.pdf', bbox_inches='tight')
        plt.show()
        
    def plot_virgo_cluster_overview(self):
        """Plot 3: Virgo cluster overview with velocity color-coding"""
        
        # Galaxy coordinates (RA, DEC) - approximate from your previous plots
        galaxy_coords = {
            'VCC0308': (184.6, 8.1), 'VCC0667': (186.4, 7.5), 'VCC0688': (186.7, 8.0),
            'VCC0990': (185.9, 6.9), 'VCC1049': (188.2, 8.2), 'VCC1146': (187.4, 13.4),
            'VCC1193': (188.6, 8.4), 'VCC1368': (187.7, 11.9), 'VCC1410': (189.3, 16.7),
            'VCC1431': (188.1, 11.0), 'VCC1486': (186.9, 7.9), 'VCC1499': (188.4, 13.1),
            'VCC1549': (188.4, 11.3), 'VCC1588': (189.4, 15.7), 'VCC1695': (189.9, 12.8),
            'VCC1811': (189.0, 15.5), 'VCC1890': (191.1, 11.5), 'VCC1902': (191.0, 13.2),
            'VCC1910': (191.1, 12.4), 'VCC1949': (191.0, 12.1)
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Spatial distribution with velocity colors
        ax = ax1
        
        ras, decs, velocities, gradients = [], [], [], []
        galaxy_names = []
        
        for galaxy in self.galaxies:
            if galaxy in galaxy_coords and galaxy in self.galaxy_velocities:
                ra, dec = galaxy_coords[galaxy]
                vel = self.galaxy_velocities[galaxy]
                
                # Get gradient if available from results
                if self.results_df is not None:
                    galaxy_result = self.results_df[self.results_df['galaxy'] == galaxy]
                    if len(galaxy_result) > 0:
                        gradient = galaxy_result['gradient_slope'].iloc[0]
                    else:
                        gradient = 0.0
                else:
                    gradient = 0.0
                
                ras.append(ra)
                decs.append(dec)
                velocities.append(vel)
                gradients.append(gradient)
                galaxy_names.append(galaxy)
        
        # Create scatter plot with velocity colors
        scatter = ax.scatter(ras, decs, c=velocities, s=200, cmap='viridis',
                           alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add galaxy labels
        for i, name in enumerate(galaxy_names):
            ax.annotate(name.replace('VCC', ''), (ras[i], decs[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        # Add gradient direction indicators
        for i, (ra, dec, grad) in enumerate(zip(ras, decs, gradients)):
            if abs(grad) > 0.01:  # Only show significant gradients
                arrow_length = min(abs(grad) * 50, 0.5)  # Scale arrow length
                if grad > 0:  # Positive gradient - arrow pointing up
                    ax.arrow(ra, dec, 0, arrow_length, head_width=0.1, 
                            head_length=0.05, fc='red', ec='red', alpha=0.7)
                else:  # Negative gradient - arrow pointing down
                    ax.arrow(ra, dec, 0, -arrow_length, head_width=0.1, 
                            head_length=0.05, fc='blue', ec='blue', alpha=0.7)
        
        ax.set_xlabel('Right Ascension (degrees)')
        ax.set_ylabel('Declination (degrees)')
        ax.set_title('Virgo Cluster Galaxies - Velocity Color-Coded\n(Arrows indicate α/Fe gradient direction)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for velocity
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Galaxy Velocity (km/s)')
        
        # Plot 2: Gradient vs velocity relationship
        ax = ax2
        
        # Filter out zero gradients (failed analyses)
        valid_mask = (np.array(gradients) != 0.0) & np.isfinite(gradients)
        valid_velocities = np.array(velocities)[valid_mask]
        valid_gradients = np.array(gradients)[valid_mask]
        valid_names = np.array(galaxy_names)[valid_mask]
        
        if len(valid_gradients) > 0:
            # Color by gradient sign
            colors = ['red' if g > 0 else 'blue' for g in valid_gradients]
            
            scatter2 = ax.scatter(valid_velocities, valid_gradients, c=colors, s=150,
                                alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add galaxy labels
            for i, name in enumerate(valid_names):
                ax.annotate(name.replace('VCC', ''), 
                           (valid_velocities[i], valid_gradients[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Calculate correlation - only if we have finite values
            if len(valid_gradients) > 3 and np.all(np.isfinite(valid_velocities)) and np.all(np.isfinite(valid_gradients)):
                from scipy.stats import pearsonr
                try:
                    corr, p_value = pearsonr(valid_velocities, valid_gradients)
                    ax.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_value:.3f}',
                           transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
                except:
                    ax.text(0.05, 0.95, 'Correlation: N/A',
                           transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        ax.set_xlabel('Galaxy Velocity (km/s)')
        ax.set_ylabel('α/Fe Gradient (dex/Re)')
        ax.set_title('α/Fe Gradient vs Galaxy Velocity\n(Red: positive, Blue: negative)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diagnostic_plots/virgo_cluster_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig('diagnostic_plots/virgo_cluster_overview.pdf', bbox_inches='tight')
        plt.show()
        
    def check_velocity_dispersion_consistency(self):
        """Check if TMB03 model velocity dispersions match our ISAPC data"""
        
        print("\n" + "="*70)
        print("VELOCITY DISPERSION CONSISTENCY CHECK")
        print("="*70)
        
        # Typical velocity dispersions for early-type galaxies
        typical_sigma = {
            'dwarf': (50, 100),      # km/s
            'normal': (100, 300),    # km/s
            'massive': (300, 400)    # km/s
        }
        
        print(f"Expected velocity dispersions for galaxy types:")
        for gal_type, (sigma_min, sigma_max) in typical_sigma.items():
            print(f"  {gal_type.capitalize()}: {sigma_min}-{sigma_max} km/s")
        
        print(f"\nOur galaxy sample velocities:")
        velocities = list(self.galaxy_velocities.values())
        print(f"  Range: {min(velocities)}-{max(velocities)} km/s")
        print(f"  Mean: {np.mean(velocities):.0f} ± {np.std(velocities):.0f} km/s")
        
        # Check if we need velocity dispersion corrections
        print(f"\nTMB03 model assumptions:")
        print(f"  - Models are typically calibrated for σ ~ 200-250 km/s")
        print(f"  - Our galaxies span 1095-1885 km/s (recession velocities)")
        print(f"  - Need to convert to velocity dispersions using virial theorem")
        
        # Estimate velocity dispersions from recession velocities (very rough)
        # σ ≈ v_rec / sqrt(3) for rough estimate (virial relation)
        estimated_sigma = np.array(velocities) / np.sqrt(3) * 0.1  # Very rough conversion
        
        print(f"\nRough estimated velocity dispersions:")
        print(f"  Range: {min(estimated_sigma):.0f}-{max(estimated_sigma):.0f} km/s")
        print(f"  Mean: {np.mean(estimated_sigma):.0f} ± {np.std(estimated_sigma):.0f} km/s")
        
        print(f"\nRECOMMENDATION:")
        if np.mean(estimated_sigma) > 250:
            print(f"  ⚠️  Our galaxies may have higher σ than TMB03 calibration")
            print(f"  ➤  Consider velocity dispersion corrections in ISAPC workflow")
        else:
            print(f"  ✅ Velocity dispersions appear consistent with TMB03 models")
            
        return estimated_sigma
        
    def create_all_diagnostic_plots(self):
        """Create all diagnostic plots"""
        
        # Create output directory
        os.makedirs('diagnostic_plots', exist_ok=True)
        
        print("Creating comprehensive α/Fe diagnostic plots...")
        print("="*50)
        
        # Check velocity dispersion consistency first
        self.check_velocity_dispersion_consistency()
        
        print("\n1. Creating TMB03 index relationship plots...")
        self.plot_tmb03_index_relationships()
        
        print("\n2. Creating individual galaxy profile plots...")
        self.plot_individual_galaxy_profiles()
        
        print("\n3. Creating Virgo cluster overview plot...")
        self.plot_virgo_cluster_overview()
        
        print(f"\n✅ All diagnostic plots saved to diagnostic_plots/ directory")
        print("="*50)

def main():
    """Main function to create all diagnostic plots"""
    plotter = AlphaFeDiagnosticPlotter()
    plotter.create_all_diagnostic_plots()

if __name__ == "__main__":
    main()
