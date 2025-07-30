"""
Complete Virgo Cluster α/Fe Gradient Analysis with Enhanced ISAPC Integration

This script processes all Virgo cluster galaxies using the corrected ISAPC workflow
to calculate α/Fe abundance gradients using TMB03 stellar population models.

Key improvements:
1. Proper ISAPC P2P spectral index extraction
2. Correct spatial binning using bin_num mapping
3. Enhanced α/Fe calculation with physics corrections
4. Literature-based methodology validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from enhanced_alpha_fe_analyzer import ISAPCAlphaFeAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VirgoCompleteAnalysis')

def process_all_virgo_galaxies():
    """Process all Virgo cluster galaxies with enhanced α/Fe analysis"""
    
    # Initialize analyzer
    analyzer = ISAPCAlphaFeAnalyzer()
    
    # Galaxy list from the project
    galaxy_list = [
        'VCC0308', 'VCC0667', 'VCC0688', 'VCC0990', 'VCC1049', 'VCC1146', 
        'VCC1193', 'VCC1368', 'VCC1410', 'VCC1431', 'VCC1486', 'VCC1499',
        'VCC1549', 'VCC1588', 'VCC1695', 'VCC1811', 'VCC1890', 'VCC1902', 
        'VCC1910', 'VCC1949'
    ]
    
    # Results storage
    results = {}
    summary_data = []
    
    print(f"\n{'='*80}")
    print(f"VIRGO CLUSTER α/Fe GRADIENT ANALYSIS - ENHANCED ISAPC INTEGRATION")
    print(f"{'='*80}")
    print(f"Processing {len(galaxy_list)} galaxies with corrected methodology")
    print(f"{'='*80}")
    
    # Process each galaxy
    for i, galaxy_name in enumerate(galaxy_list, 1):
        print(f"\n[{i:2d}/{len(galaxy_list)}] Processing {galaxy_name}...")
        
        try:
            # Analyze galaxy with RDB method (3 bins)
            result = analyzer.analyze_galaxy_gradient(galaxy_name, method='RDB', max_bins=3)
            
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
                    'alpha_fe_outer': result['alpha_fe_values'][-1] if len(result['alpha_fe_values']) > 0 else np.nan
                })
                
                # Print result with R/Re information
                direction = "↗" if slope > 0 else "↘"
                re_arcsec = result.get('effective_radius', np.nan)
                print(f"    Result: {slope:+.4f} ± {slope_error:.4f} dex/Re {direction} {sig_level}")
                print(f"    Bins: {n_bins}, Re = {re_arcsec:.1f}\", Correlation: r={correlation:.3f}, p={p_value:.3f}")
                
                # Print α/Fe values with R/Re
                if len(result['alpha_fe_values']) > 0:
                    r_over_re_vals = result.get('r_over_re', [])
                    alpha_fe_str = ", ".join([f"R/Re={r_over_re_vals[i]:.2f}: {val:.3f}±{err:.3f}" 
                                            for i, (val, err) in enumerate(zip(result['alpha_fe_values'], result['alpha_fe_errors']))
                                            if np.isfinite(val) and i < len(r_over_re_vals)])
                    print(f"    α/Fe profile: {alpha_fe_str}")
                
            else:
                print(f"    FAILED: Analysis unsuccessful")
                summary_data.append({
                    'galaxy': galaxy_name,
                    'gradient_slope': np.nan,
                    'gradient_error': np.nan,
                    'significance': 0,
                    'n_bins': 0,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'effective_radius': np.nan,
                    'effective_radius': np.nan,
                    'alpha_fe_center': np.nan,
                    'alpha_fe_outer': np.nan
                })
                
        except Exception as e:
            print(f"    ERROR: {e}")
            logger.error(f"Error processing {galaxy_name}: {e}")
            summary_data.append({
                'galaxy': galaxy_name,
                'gradient_slope': np.nan,
                'gradient_error': np.nan,
                'significance': 0,
                'n_bins': 0,
                'correlation': np.nan,
                'p_value': np.nan,
                'alpha_fe_center': np.nan,
                'alpha_fe_outer': np.nan
            })
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate statistics
    valid_results = summary_df[np.isfinite(summary_df['gradient_slope'])]
    n_successful = len(valid_results)
    n_positive = np.sum(valid_results['gradient_slope'] > 0)
    n_negative = np.sum(valid_results['gradient_slope'] < 0)
    n_significant = np.sum(valid_results['significance'] > 2)  # >2σ
    
    mean_gradient = np.mean(valid_results['gradient_slope'])
    std_gradient = np.std(valid_results['gradient_slope'])
    
    print(f"Successfully analyzed: {n_successful}/{len(galaxy_list)} galaxies ({100*n_successful/len(galaxy_list):.1f}%)")
    print(f"Positive gradients: {n_positive}")
    print(f"Negative gradients: {n_negative}")
    print(f"Significant results (>2σ): {n_significant}")
    print(f"Mean gradient: {mean_gradient:.4f} ± {std_gradient:.4f} dex/Re")
    
    # Save results
    output_dir = Path("enhanced_alpha_fe_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save summary CSV
    summary_file = output_dir / "virgo_cluster_alpha_fe_gradients_enhanced.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # Save detailed results
    import pickle
    detailed_file = output_dir / "virgo_cluster_detailed_results.pkl"
    with open(detailed_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Print detailed summary table
    print(f"\n{'='*100}")
    print(f"DETAILED RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"{'Galaxy':<8} {'Gradient':<12} {'Error':<8} {'Sig':<4} {'Bins':<4} {'Re(\")  ':<7} {'α/Fe_cen':<8} {'α/Fe_out':<8} {'Notes':<20}")
    print(f"{'-'*105}")
    
    for _, row in summary_df.iterrows():
        galaxy = row['galaxy']
        slope = row['gradient_slope']
        error = row['gradient_error']
        sig = row['significance']
        n_bins = row['n_bins']
        re_arcsec = row['effective_radius']
        alpha_cen = row['alpha_fe_center']
        alpha_out = row['alpha_fe_outer']
        
        if np.isfinite(slope):
            slope_str = f"{slope:+.4f}"
            error_str = f"±{error:.4f}"
            sig_str = f"{sig:.1f}σ"
            re_str = f"{re_arcsec:.1f}" if np.isfinite(re_arcsec) else "---"
            alpha_cen_str = f"{alpha_cen:.3f}" if np.isfinite(alpha_cen) else "---"
            alpha_out_str = f"{alpha_out:.3f}" if np.isfinite(alpha_out) else "---"
            
            # Notes
            notes = []
            if sig > 3:
                notes.append("Highly Significant")
            elif sig > 2:
                notes.append("Significant")
            elif sig > 1:
                notes.append("Marginal")
            else:
                notes.append("Not Significant")
                
            if slope > 0.01:
                notes.append("Strong Positive")
            elif slope < -0.01:
                notes.append("Strong Negative")
                
            notes_str = ", ".join(notes)[:19]
            
        else:
            slope_str = "FAILED"
            error_str = "---"
            sig_str = "---"
            re_str = "---"
            alpha_cen_str = "---"
            alpha_out_str = "---"
            notes_str = "Analysis Failed"
        
        print(f"{galaxy:<8} {slope_str:<12} {error_str:<8} {sig_str:<4} {n_bins:<4} {re_str:<7} {alpha_cen_str:<8} {alpha_out_str:<8} {notes_str:<20}")
    
    print(f"{'-'*105}")
    print(f"Total: {len(summary_df)} galaxies, {n_successful} successful analyses")
    print(f"All gradients calculated as d[α/Fe]/d(R/Re) using ISAPC effective radii")
    print(f"{'='*105}")
    
    return results, summary_df

def create_comparison_with_original():
    """Compare enhanced results with original analysis"""
    
    # Load enhanced results
    enhanced_file = "enhanced_alpha_fe_results/virgo_cluster_alpha_fe_gradients_enhanced.csv"
    if Path(enhanced_file).exists():
        enhanced_df = pd.read_csv(enhanced_file)
        print(f"\nLoaded enhanced results: {len(enhanced_df)} galaxies")
        
        # Look for original results to compare
        original_files = [
            "alpha_gradient_dual/combined_gradient_summary.csv",
            "vcc1431_gradient_summary.csv"
        ]
        
        for orig_file in original_files:
            if Path(orig_file).exists():
                try:
                    orig_df = pd.read_csv(orig_file)
                    print(f"Found original results: {orig_file} with {len(orig_df)} entries")
                    
                    # Try to compare VCC1431 specifically
                    if 'VCC1431' in enhanced_df['galaxy'].values:
                        enhanced_vcc1431 = enhanced_df[enhanced_df['galaxy'] == 'VCC1431'].iloc[0]
                        
                        print(f"\nVCC1431 COMPARISON:")
                        print(f"Enhanced method: {enhanced_vcc1431['gradient_slope']:+.4f} ± {enhanced_vcc1431['gradient_error']:.4f}")
                        
                        if 'gradient' in orig_df.columns:
                            orig_vcc1431 = orig_df[orig_df['galaxy'].str.contains('1431', na=False)]
                            if len(orig_vcc1431) > 0:
                                orig_grad = orig_vcc1431.iloc[0]['gradient']
                                print(f"Original method:  {orig_grad:+.4f}")
                                print(f"Difference: {enhanced_vcc1431['gradient_slope'] - orig_grad:+.4f}")
                    
                except Exception as e:
                    print(f"Error comparing with {orig_file}: {e}")
    
if __name__ == "__main__":
    # Run complete analysis
    results, summary_df = process_all_virgo_galaxies()
    
    # Compare with original if available
    create_comparison_with_original()
    
    print(f"\n{'='*80}")
    print(f"ENHANCED α/Fe GRADIENT ANALYSIS COMPLETE")
    print(f"All results saved in enhanced_alpha_fe_results/ directory")
    print(f"{'='*80}")
