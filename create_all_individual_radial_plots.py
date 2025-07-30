#!/usr/bin/env python3
"""
Create Individual Radial Plots for All Galaxies
Creates detailed radial gradient plots for each galaxy using the comprehensive analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
from enhanced_alpha_fe_analyzer import ISAPCAlphaFeAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AllIndividualRadialPlotter:
    """Creates individual radial gradient plots for all galaxies."""
    
    def __init__(self):
        self.output_dir = Path('./individual_radial_plots')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize the analyzer
        self.analyzer = ISAPCAlphaFeAnalyzer()
        
        # Load existing comprehensive results
        self.results_file = Path('complete_virgo_alpha_fe_results/complete_virgo_alpha_fe_analysis.csv')
        
    def load_comprehensive_results(self):
        """Load the comprehensive analysis results."""
        if not self.results_file.exists():
            logger.error(f"Results file not found: {self.results_file}")
            return None
            
        df = pd.read_csv(self.results_file)
        logger.info(f"Loaded results for {len(df)} galaxies")
        return df
        
    def create_individual_plot(self, galaxy_name, gradient_data):
        """Create individual radial plot for a single galaxy."""
        try:
            logger.info(f"Creating individual plot for {galaxy_name}")
            
            # Analyze this galaxy with the enhanced analyzer
            results = self.analyzer.analyze_galaxy_gradient(galaxy_name)
            
            if not results or not results.get('success', False):
                logger.warning(f"Failed to analyze {galaxy_name}")
                return False
                
            # Create the plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{galaxy_name} - α/Fe Radial Analysis', fontsize=16, fontweight='bold')
            
            # Extract data
            radii = results['profile']['radius_re']
            alpha_fe = results['profile']['alpha_fe']
            alpha_fe_err = results['profile']['alpha_fe_error']
            gradient = results['gradient']['slope']
            gradient_err = results['gradient']['error']
            significance = results['gradient']['significance']
            
            # Panel 1: Radial Profile
            ax1 = axes[0, 0]
            ax1.errorbar(radii, alpha_fe, yerr=alpha_fe_err, 
                        fmt='o', color='navy', markersize=8, capsize=5, capthick=2,
                        label=f'{galaxy_name} data')
            
            # Plot gradient line
            r_line = np.linspace(0, max(radii), 100)
            alpha_line = results['profile']['alpha_fe'][0] + gradient * r_line
            ax1.plot(r_line, alpha_line, '--', color='red', linewidth=2,
                    label=f'Gradient: {gradient:.3f}±{gradient_err:.3f} dex/Re')
            
            ax1.set_xlabel('Radius (R/Re)', fontsize=12)
            ax1.set_ylabel('[α/Fe] (dex)', fontsize=12)
            ax1.set_title('Radial α/Fe Profile', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Panel 2: Spectral Indices
            ax2 = axes[0, 1]
            if 'indices' in results:
                indices = results['indices']
                for idx_name in ['Fe5015', 'Mgb', 'Hbeta']:
                    if idx_name.lower() in indices:
                        idx_data = indices[idx_name.lower()]
                        ax2.errorbar(radii, idx_data['values'], yerr=idx_data.get('errors', None),
                                   fmt='o-', label=idx_name, markersize=6, capsize=3)
            
            ax2.set_xlabel('Radius (R/Re)', fontsize=12)
            ax2.set_ylabel('Index Strength (Å)', fontsize=12)
            ax2.set_title('Spectral Index Profiles', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Panel 3: Statistics Summary  
            ax3 = axes[1, 0]
            ax3.axis('off')
            
            # Create statistics text
            stats_text = f"""
GALAXY: {galaxy_name}
{'='*30}

GRADIENT ANALYSIS:
• Slope: {gradient:.4f} ± {gradient_err:.4f} dex/Re
• Significance: {significance:.2f}σ
• Correlation: r = {results['gradient'].get('correlation', 'N/A')}
• P-value: {results['gradient'].get('p_value', 'N/A'):.3f}

PHYSICAL PROPERTIES:
• Effective Radius: {results.get('effective_radius', 'N/A'):.1f}"
• Number of Bins: {len(radii)}
• Radial Range: {min(radii):.2f} - {max(radii):.2f} Re

α/Fe ABUNDANCE:
• Center: {alpha_fe[0]:.3f} ± {alpha_fe_err[0]:.3f} dex
• Outer: {alpha_fe[-1]:.3f} ± {alpha_fe_err[-1]:.3f} dex
• Change: {alpha_fe[-1] - alpha_fe[0]:+.3f} dex
            """
            
            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            # Panel 4: Gradient Significance
            ax4 = axes[1, 1]
            
            # Create significance bar chart
            sig_levels = ['1σ', '2σ', '3σ']
            sig_values = [1, 2, 3]
            colors = ['yellow', 'orange', 'red']
            
            bars = ax4.bar(sig_levels, sig_values, color=colors, alpha=0.6, edgecolor='black')
            
            # Mark current significance
            if significance >= 3:
                marker_color = 'red'
                marker_alpha = 1.0
            elif significance >= 2:
                marker_color = 'orange' 
                marker_alpha = 1.0
            elif significance >= 1:
                marker_color = 'yellow'
                marker_alpha = 1.0
            else:
                marker_color = 'gray'
                marker_alpha = 0.5
                
            ax4.axhline(y=significance, color=marker_color, linewidth=4, alpha=marker_alpha,
                       label=f'This galaxy: {significance:.1f}σ')
            
            ax4.set_ylabel('Significance (σ)', fontsize=12)
            ax4.set_title('Gradient Significance', fontsize=14, fontweight='bold')
            ax4.set_ylim(0, 4)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.legend()
            
            # Add gradient direction indicator
            if gradient > 0:
                direction = "↗ Positive (increasing outward)"
                color = 'blue'
            else:
                direction = "↘ Negative (decreasing outward)"  
                color = 'red'
                
            ax4.text(0.5, 0.85, direction, transform=ax4.transAxes, fontsize=12,
                    ha='center', color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            output_file = self.output_dir / f"{galaxy_name}_individual_radial_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Individual plot saved: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating plot for {galaxy_name}: {e}")
            return False
    
    def create_all_individual_plots(self):
        """Create individual plots for all galaxies."""
        logger.info("🚀 Starting individual radial plot creation for all galaxies...")
        
        # Load comprehensive results
        df = self.load_comprehensive_results()
        if df is None:
            return 0
            
        # Filter successful galaxies
        successful_galaxies = df[df['analysis_success'] == True]['galaxy'].tolist()
        logger.info(f"Found {len(successful_galaxies)} successful galaxies to plot")
        
        successful_plots = 0
        
        for galaxy_name in successful_galaxies:
            logger.info(f"Processing {galaxy_name}...")
            
            # Get gradient data for this galaxy
            galaxy_data = df[df['galaxy'] == galaxy_name].iloc[0]
            
            # Create individual plot
            success = self.create_individual_plot(galaxy_name, galaxy_data)
            if success:
                successful_plots += 1
            
        logger.info(f"✅ Individual plotting complete!")
        logger.info(f"Successfully created {successful_plots}/{len(successful_galaxies)} individual plots")
        
        return successful_plots

def main():
    """Main function to create all individual radial plots."""
    print("\n🌌 INDIVIDUAL RADIAL GRADIENT PLOTTING")
    print("="*60)
    
    try:
        plotter = AllIndividualRadialPlotter()
        success_count = plotter.create_all_individual_plots()
        
        print("\n" + "="*60)
        print("INDIVIDUAL RADIAL GRADIENT PLOTTING COMPLETE")
        print("="*60)
        print(f"✅ Successfully created {success_count} individual radial plots")
        print(f"📁 Plots saved in: individual_radial_plots/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Error in individual radial plotting: {e}")
        print(f"❌ Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
