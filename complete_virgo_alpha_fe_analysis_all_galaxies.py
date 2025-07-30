#!/usr/bin/env python3
"""
Complete Virgo Cluster Alpha/Fe Analysis for ALL Available Galaxies

This script processes all available Virgo cluster galaxies with ISAPC results,
calculating alpha/Fe abundance gradients, creating comprehensive plots, and
providing detailed analysis of the entire sample.

Features:
1. Processes all available galaxies automatically
2. Robust error handling for failed analyses
3. Comprehensive plotting and visualization
4. Statistical analysis of the complete sample
5. Individual galaxy profile plots
6. Summary statistics and correlations

Author: Enhanced ISAPC Analysis System
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import os
import glob
from scipy import stats
from matplotlib.gridspec import GridSpec
import seaborn as sns
from enhanced_alpha_fe_analyzer import ISAPCAlphaFeAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CompleteVirgoAnalysis')

class CompleteVirgoAlphaFeAnalysis:
    """Complete analysis of all Virgo cluster galaxies"""
    
    def __init__(self, output_dir="/home/siqi/WkpSpace/ISAPC_Jul/ISAPC/output"):
        self.output_dir = output_dir
        self.analyzer = ISAPCAlphaFeAnalyzer()
        self.results_dir = Path("complete_virgo_alpha_fe_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Get all available galaxies
        self.available_galaxies = self._find_available_galaxies()
        logger.info(f"Found {len(self.available_galaxies)} galaxies with ISAPC results")
        
    def _find_available_galaxies(self):
        """Find all galaxies with ISAPC results"""
        galaxy_dirs = glob.glob(os.path.join(self.output_dir, "VCC*_stack"))
        galaxies = []
        
        for galaxy_dir in sorted(galaxy_dirs):
            galaxy_name = os.path.basename(galaxy_dir).replace("_stack", "")
            
            # Check if essential files exist
            data_dir = os.path.join(galaxy_dir, "Data")
            if os.path.exists(data_dir):
                # Look for P2P indices
                indices_file = os.path.join(data_dir, f"{galaxy_name}_stack_P2P_indices.npz")
                if os.path.exists(indices_file):
                    galaxies.append(galaxy_name)
                    logger.debug(f"✓ {galaxy_name}: ISAPC P2P data available")
                else:
                    logger.warning(f"✗ {galaxy_name}: Missing P2P indices file")
            else:
                logger.warning(f"✗ {galaxy_name}: Missing Data directory")
                
        return galaxies
    
    def process_all_galaxies(self):
        """Process all available galaxies for alpha/Fe analysis"""
        
        results = {}
        summary_data = []
        failed_galaxies = []
        
        print(f"\n{'='*100}")
        print(f"COMPLETE VIRGO CLUSTER α/Fe GRADIENT ANALYSIS")
        print(f"{'='*100}")
        print(f"Processing {len(self.available_galaxies)} galaxies with ISAPC results")
        print(f"Using enhanced methodology with TMB03 stellar population models")
        print(f"{'='*100}")
        
        for i, galaxy_name in enumerate(self.available_galaxies, 1):
            print(f"\n[{i:2d}/{len(self.available_galaxies)}] Processing {galaxy_name}...")
            
            try:
                # Analyze galaxy with RDB method
                result = self.analyzer.analyze_galaxy_gradient(galaxy_name, method='RDB', max_bins=3)
                
                if result and result.get('analysis_success', False):
                    results[galaxy_name] = result
                    
                    # Extract key metrics
                    slope = result['gradient_slope']
                    slope_error = result['gradient_slope_error']
                    n_bins = result['n_bins']
                    correlation = result['correlation_coefficient']
                    p_value = result['p_value']
                    
                    # Calculate significance
                    significance = abs(slope) / slope_error if slope_error > 0 else 0
                    sig_level = "***" if significance > 3 else ("**" if significance > 2 else ("*" if significance > 1 else ""))
                    # Store summary data
                    summary_data.append({
                        'galaxy': galaxy_name,
                        'gradient_slope': slope,
                        'gradient_error': slope_error,
                        'significance': significance,
                        'n_bins': n_bins,
                        'correlation': correlation,
                        'p_value': p_value,
                        'effective_radius': result.get('effective_radius', np.nan),
                        'alpha_fe_center': result['alpha_fe_values'][0] if len(result['alpha_fe_values']) > 0 else np.nan,
                        'alpha_fe_outer': result['alpha_fe_values'][-1] if len(result['alpha_fe_values']) > 0 else np.nan,
                        'analysis_success': True
                    })
                    
                    # Print detailed result
                    direction = "↗" if slope > 0 else "↘"
                    re_arcsec = result.get('effective_radius', np.nan)
                    print(f"    ✓ Gradient: {slope:+.4f} ± {slope_error:.4f} dex/Re {direction} {sig_level}")
                    print(f"      Bins: {n_bins}, Re = {re_arcsec:.1f}\", r={correlation:.3f}, p={p_value:.3f}")
                    
                    # Print α/Fe profile
                    if len(result['alpha_fe_values']) > 0:
                        r_over_re_vals = result.get('r_over_re', [])
                        if len(r_over_re_vals) > 0:
                            alpha_fe_summary = f"α/Fe: {result['alpha_fe_values'][0]:.3f}→{result['alpha_fe_values'][-1]:.3f}"
                            print(f"      {alpha_fe_summary} (center→outer)")
                    
                else:
                    # Analysis failed
                    failed_galaxies.append(galaxy_name)
                    summary_data.append({
                        'galaxy': galaxy_name,
                        'gradient_slope': np.nan,
                        'gradient_error': np.nan,
                        'significance': 0,
                        'n_bins': 0,
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'effective_radius': np.nan,
                        'alpha_fe_center': np.nan,
                        'alpha_fe_outer': np.nan,
                        'analysis_success': False
                    })
                    print(f"    ✗ FAILED: Analysis unsuccessful")
                    
            except Exception as e:
                failed_galaxies.append(galaxy_name)
                print(f"    ✗ ERROR: {e}")
                logger.error(f"Error processing {galaxy_name}: {e}")
                summary_data.append({
                    'galaxy': galaxy_name,
                    'gradient_slope': np.nan,
                    'gradient_error': np.nan,
                    'significance': 0,
                    'n_bins': 0,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'effective_radius': np.nan,
                    'alpha_fe_center': np.nan,
                    'alpha_fe_outer': np.nan,
                    'analysis_success': False
                })
        
        print(f"\n{'='*100}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*100}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save results
        self._save_results(results, summary_df, failed_galaxies)
        
        # Print summary statistics
        self._print_summary_statistics(summary_df, failed_galaxies)
        
        return results, summary_df
    
    def _save_results(self, results, summary_df, failed_galaxies):
        """Save all analysis results"""
        
        # Save summary CSV
        summary_file = self.results_dir / "complete_virgo_alpha_fe_analysis.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to: {summary_file}")
        
        # Save detailed results (pickle)
        import pickle
        detailed_file = self.results_dir / "complete_virgo_detailed_results.pkl"
        with open(detailed_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Detailed results saved to: {detailed_file}")
        
        # Save failed galaxies list
        if failed_galaxies:
            failed_file = self.results_dir / "failed_galaxies.txt"
            with open(failed_file, 'w') as f:
                f.write("Galaxies that failed α/Fe analysis:\\n")
                for galaxy in failed_galaxies:
                    f.write(f"{galaxy}\\n")
            logger.info(f"Failed galaxies list saved to: {failed_file}")
    
    def _print_summary_statistics(self, summary_df, failed_galaxies):
        """Print comprehensive summary statistics"""
        
        # Calculate statistics
        valid_results = summary_df[summary_df['analysis_success'] == True]
        n_total = len(summary_df)
        n_successful = len(valid_results)
        n_failed = len(failed_galaxies)
        
        if n_successful > 0:
            valid_gradients = valid_results[np.isfinite(valid_results['gradient_slope'])]
            n_positive = np.sum(valid_gradients['gradient_slope'] > 0)
            n_negative = np.sum(valid_gradients['gradient_slope'] < 0)
            n_significant = np.sum(valid_gradients['significance'] > 2)  # >2σ
            n_highly_sig = np.sum(valid_gradients['significance'] > 3)  # >3σ
            
            mean_gradient = np.mean(valid_gradients['gradient_slope'])
            std_gradient = np.std(valid_gradients['gradient_slope'])
            median_gradient = np.median(valid_gradients['gradient_slope'])
            
            mean_alpha_center = np.nanmean(valid_results['alpha_fe_center'])
            mean_alpha_outer = np.nanmean(valid_results['alpha_fe_outer'])
        else:
            n_positive = n_negative = n_significant = n_highly_sig = 0
            mean_gradient = std_gradient = median_gradient = np.nan
            mean_alpha_center = mean_alpha_outer = np.nan
        
        print(f"\\n{'='*100}")
        print(f"COMPLETE VIRGO CLUSTER α/Fe ANALYSIS SUMMARY")
        print(f"{'='*100}")
        print(f"Total galaxies processed: {n_total}")
        print(f"Successful analyses: {n_successful} ({100*n_successful/n_total:.1f}%)")
        print(f"Failed analyses: {n_failed} ({100*n_failed/n_total:.1f}%)")
        
        if n_successful > 0:
            print(f"\\nGRADIENT STATISTICS:")
            print(f"Positive gradients: {n_positive} ({100*n_positive/n_successful:.1f}%)")
            print(f"Negative gradients: {n_negative} ({100*n_negative/n_successful:.1f}%)")
            print(f"Significant results (>2σ): {n_significant} ({100*n_significant/n_successful:.1f}%)")
            print(f"Highly significant (>3σ): {n_highly_sig} ({100*n_highly_sig/n_successful:.1f}%)")
            print(f"\\nMean gradient: {mean_gradient:.4f} ± {std_gradient:.4f} dex/Re")
            print(f"Median gradient: {median_gradient:.4f} dex/Re")
            print(f"\\nα/Fe ABUNDANCE LEVELS:")
            print(f"Mean α/Fe at center: {mean_alpha_center:.3f}")
            print(f"Mean α/Fe at outer: {mean_alpha_outer:.3f}")
            print(f"Mean change (outer-center): {mean_alpha_outer-mean_alpha_center:+.3f}")
        
        if failed_galaxies:
            print(f"\\nFAILED GALAXIES: {', '.join(failed_galaxies)}")
        
        print(f"{'='*100}")
    
    def create_comprehensive_plots(self, results, summary_df):
        """Create comprehensive visualization plots"""
        
        print(f"\\n🎨 Creating comprehensive visualization plots...")
        
        # Filter valid results
        valid_results = summary_df[summary_df['analysis_success'] == True]
        
        if len(valid_results) == 0:
            logger.error("No valid results to plot!")
            return
        
        # Create master figure with multiple panels
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Gradient distribution histogram
        ax1 = fig.add_subplot(gs[0, 0])
        gradients = valid_results['gradient_slope'].values
        gradients_finite = gradients[np.isfinite(gradients)]
        
        ax1.hist(gradients_finite, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8, label='Zero gradient')
        ax1.axvline(np.mean(gradients_finite), color='orange', linestyle='-', alpha=0.8, 
                   label=f'Mean: {np.mean(gradients_finite):.3f}')
        ax1.set_xlabel('α/Fe Gradient (dex/Re)')
        ax1.set_ylabel('Number of Galaxies')
        ax1.set_title('α/Fe Gradient Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Significance vs Gradient
        ax2 = fig.add_subplot(gs[0, 1])
        significances = valid_results['significance'].values
        colors = ['red' if s > 3 else 'orange' if s > 2 else 'blue' if s > 1 else 'gray' 
                 for s in significances]
        
        ax2.scatter(gradients_finite, significances[np.isfinite(gradients)], 
                   c=colors, alpha=0.7, s=50)
        ax2.axhline(1, color='gray', linestyle=':', alpha=0.5, label='1σ')
        ax2.axhline(2, color='orange', linestyle=':', alpha=0.5, label='2σ')
        ax2.axhline(3, color='red', linestyle=':', alpha=0.5, label='3σ')
        ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('α/Fe Gradient (dex/Re)')
        ax2.set_ylabel('Significance (σ)')
        ax2.set_title('Gradient Significance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: α/Fe center vs outer
        ax3 = fig.add_subplot(gs[0, 2])
        center_vals = valid_results['alpha_fe_center'].values
        outer_vals = valid_results['alpha_fe_outer'].values
        
        # Remove NaN values for plotting
        mask = np.isfinite(center_vals) & np.isfinite(outer_vals)
        center_clean = center_vals[mask]
        outer_clean = outer_vals[mask]
        
        ax3.scatter(center_clean, outer_clean, alpha=0.7, s=50, c='green')
        # Add 1:1 line
        min_val = min(np.min(center_clean), np.min(outer_clean))
        max_val = max(np.max(center_clean), np.max(outer_clean))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 line')
        ax3.set_xlabel('α/Fe Center')
        ax3.set_ylabel('α/Fe Outer')
        ax3.set_title('Center vs Outer α/Fe')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Effective radius vs Gradient
        ax4 = fig.add_subplot(gs[0, 3])
        re_vals = valid_results['effective_radius'].values
        re_finite = re_vals[np.isfinite(re_vals) & np.isfinite(gradients)]
        grad_finite_re = gradients[np.isfinite(re_vals) & np.isfinite(gradients)]
        
        ax4.scatter(re_finite, grad_finite_re, alpha=0.7, s=50, c='purple')
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Effective Radius (arcsec)')
        ax4.set_ylabel('α/Fe Gradient (dex/Re)')
        ax4.set_title('Gradient vs Galaxy Size')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5-8: Individual galaxy profiles (sample of 4 best cases)
        best_galaxies = valid_results.nlargest(4, 'significance')
        
        for i, (_, galaxy_data) in enumerate(best_galaxies.iterrows()):
            ax = fig.add_subplot(gs[1, i])
            galaxy_name = galaxy_data['galaxy']
            
            if galaxy_name in results:
                result = results[galaxy_name]
                r_over_re = result.get('r_over_re', [])
                alpha_fe_vals = result.get('alpha_fe_values', [])
                alpha_fe_errs = result.get('alpha_fe_errors', [])
                
                if len(r_over_re) > 0 and len(alpha_fe_vals) > 0:
                    ax.errorbar(r_over_re, alpha_fe_vals, yerr=alpha_fe_errs,
                               fmt='o-', capsize=5, capthick=2, linewidth=2)
                    
                    # Add linear fit
                    slope = galaxy_data['gradient_slope']
                    intercept = alpha_fe_vals[0] - slope * r_over_re[0]  # Approximate
                    r_fit = np.linspace(min(r_over_re), max(r_over_re), 100)
                    alpha_fit = intercept + slope * r_fit
                    ax.plot(r_fit, alpha_fit, 'r--', alpha=0.7)
                    
            ax.set_xlabel('R/Re')
            ax.set_ylabel('α/Fe')
            ax.set_title(f'{galaxy_name}\\n({galaxy_data["significance"]:.1f}σ)')
            ax.grid(True, alpha=0.3)
        
        # Panel 9-12: Statistics and correlations
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Create correlation matrix for key parameters
        corr_data = valid_results[['gradient_slope', 'significance', 'correlation', 
                                 'effective_radius', 'alpha_fe_center', 'alpha_fe_outer']].copy()
        corr_data = corr_data.select_dtypes(include=[np.number])  # Only numeric columns
        
        # Remove rows with any NaN values for correlation
        corr_data_clean = corr_data.dropna()
        if len(corr_data_clean) > 3:  # Need at least 4 points for meaningful correlation
            corr_matrix = corr_data_clean.corr()
            im = ax5.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax5.set_xticks(range(len(corr_matrix.columns)))
            ax5.set_yticks(range(len(corr_matrix.columns)))
            ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax5.set_yticklabels(corr_matrix.columns)
            ax5.set_title('Parameter Correlations')
            
            # Add correlation values as text
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    ax5.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}', 
                            ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax5, shrink=0.8)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\\nfor correlation analysis', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Parameter Correlations - Insufficient Data')
        
        # Panel 13: Summary statistics text
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        n_successful = len(valid_results)
        n_total = len(summary_df)
        
        if n_successful > 0:
            stats_text = f"""
COMPLETE VIRGO α/Fe ANALYSIS SUMMARY

SAMPLE STATISTICS:
• Total galaxies: {n_total}
• Successful analyses: {n_successful} ({100*n_successful/n_total:.1f}%)
• Failed analyses: {n_total-n_successful}

GRADIENT STATISTICS:
• Mean gradient: {np.nanmean(gradients_finite):.4f} ± {np.nanstd(gradients_finite):.4f} dex/Re
• Median gradient: {np.nanmedian(gradients_finite):.4f} dex/Re
• Range: {np.nanmin(gradients_finite):.4f} to {np.nanmax(gradients_finite):.4f} dex/Re

SIGNIFICANCE:
• >1σ significant: {np.sum(significances > 1)} ({100*np.sum(significances > 1)/n_successful:.1f}%)
• >2σ significant: {np.sum(significances > 2)} ({100*np.sum(significances > 2)/n_successful:.1f}%)
• >3σ significant: {np.sum(significances > 3)} ({100*np.sum(significances > 3)/n_successful:.1f}%)

α/Fe ABUNDANCE LEVELS:
• Mean α/Fe (center): {np.nanmean(valid_results['alpha_fe_center']):.3f}
• Mean α/Fe (outer): {np.nanmean(valid_results['alpha_fe_outer']):.3f}
• Typical change: {np.nanmean(valid_results['alpha_fe_outer'] - valid_results['alpha_fe_center']):+.3f}

METHODOLOGY:
• TMB03 stellar population models
• 3-bin radial analysis (RDB method)
• Physics-based α/Fe corrections
• Effective radius normalization
            """
        else:
            stats_text = "No successful analyses to summarize"
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Panel 14-16: Individual best examples (remaining space)
        remaining_best = valid_results.nlargest(8, 'significance').iloc[4:8]  # Next 4 best
        
        for i, (_, galaxy_data) in enumerate(remaining_best.iterrows()):
            if i < 3:  # Only 3 more panels available
                ax = fig.add_subplot(gs[3, i])
                galaxy_name = galaxy_data['galaxy']
                
                if galaxy_name in results:
                    result = results[galaxy_name]
                    r_over_re = result.get('r_over_re', [])
                    alpha_fe_vals = result.get('alpha_fe_values', [])
                    alpha_fe_errs = result.get('alpha_fe_errors', [])
                    
                    if len(r_over_re) > 0 and len(alpha_fe_vals) > 0:
                        ax.errorbar(r_over_re, alpha_fe_vals, yerr=alpha_fe_errs,
                                   fmt='o-', capsize=5, capthick=2, linewidth=2)
                        
                        # Add linear fit
                        slope = galaxy_data['gradient_slope']
                        intercept = alpha_fe_vals[0] - slope * r_over_re[0]
                        r_fit = np.linspace(min(r_over_re), max(r_over_re), 100)
                        alpha_fit = intercept + slope * r_fit
                        ax.plot(r_fit, alpha_fit, 'r--', alpha=0.7)
                        
                ax.set_xlabel('R/Re')
                ax.set_ylabel('α/Fe')
                ax.set_title(f'{galaxy_name}\\n({galaxy_data["significance"]:.1f}σ)')
                ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Complete Virgo Cluster α/Fe Gradient Analysis - All Available Galaxies', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save comprehensive plot
        plot_file = self.results_dir / "complete_virgo_alpha_fe_comprehensive_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive analysis plot saved to: {plot_file}")
        
        plt.show()
        
        return plot_file

def main():
    """Main function to run complete Virgo analysis"""
    
    print(f"🌌 COMPLETE VIRGO CLUSTER α/Fe ANALYSIS")
    print(f"="*80)
    
    # Initialize analysis
    analysis = CompleteVirgoAlphaFeAnalysis()
    
    # Process all galaxies
    results, summary_df = analysis.process_all_galaxies()
    
    # Create comprehensive plots
    analysis.create_comprehensive_plots(results, summary_df)
    
    print(f"\\n✅ COMPLETE VIRGO α/Fe ANALYSIS FINISHED")
    print(f"All results saved in: {analysis.results_dir}")
    print(f"="*80)

if __name__ == "__main__":
    main()
