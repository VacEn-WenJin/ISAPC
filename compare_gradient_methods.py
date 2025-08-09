#!/usr/bin/env python3
"""
Compare old vs new gradient calculations with innermost bin set to R=0
"""

import pandas as pd
from pathlib import Path
import logging
from enhanced_alpha_fe_analyzer import ISAPCAlphaFeAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_gradient_calculations():
    """Compare gradient calculations before and after setting innermost bin to R=0"""
    
    print("\n🔄 COMPARING GRADIENT CALCULATIONS")
    print("="*70)
    print("Old method: Use original ISAPC radii")
    print("New method: Set innermost bin radius to 0")
    print("="*70)
    
    # Load old results for comparison
    old_results_file = Path('complete_virgo_alpha_fe_results/complete_virgo_alpha_fe_analysis.csv')
    if old_results_file.exists():
        old_df = pd.read_csv(old_results_file)
        print(f"Loaded {len(old_df)} old results for comparison")
    else:
        print("❌ Old results file not found")
        return
    
    # Initialize analyzer with new method
    analyzer = ISAPCAlphaFeAnalyzer()
    
    # Test with a few key galaxies
    test_galaxies = ['VCC1910', 'VCC1049', 'VCC1368', 'VCC1146', 'VCC1588']
    
    print("\n📊 GRADIENT COMPARISON:")
    print("-" * 70)
    print(f"{'Galaxy':<8} {'Old Gradient':<15} {'New Gradient':<15} {'Change':<10} {'Significance'}")
    print("-" * 70)
    
    significant_changes = 0
    total_tested = 0
    
    for galaxy_name in test_galaxies:
        try:
            # Get old results
            old_row = old_df[old_df['galaxy'] == galaxy_name]
            if len(old_row) == 0:
                print(f"{galaxy_name:<8} {'NOT FOUND':<15}")
                continue
                
            old_gradient = old_row['gradient_slope'].iloc[0]
            old_error = old_row['gradient_error'].iloc[0]
            
            # Get new results with R=0 for innermost bin
            new_results = analyzer.analyze_galaxy_gradient(galaxy_name, method='RDB', max_bins=3)
            
            if new_results:
                new_gradient = new_results['gradient_slope']
                new_error = new_results['gradient_slope_error']
                
                # Calculate change
                change = new_gradient - old_gradient
                change_percent = (change / abs(old_gradient)) * 100 if abs(old_gradient) > 1e-6 else 0
                
                # Determine significance of change
                if abs(change) > 0.01:
                    significance = "MAJOR"
                    significant_changes += 1
                elif abs(change) > 0.005:
                    significance = "MODERATE"
                    significant_changes += 1
                elif abs(change) > 0.001:
                    significance = "MINOR"
                else:
                    significance = "NEGLIGIBLE"
                
                print(f"{galaxy_name:<8} {old_gradient:+.4f}±{old_error:.4f} {new_gradient:+.4f}±{new_error:.4f} {change:+.4f} {significance}")
                total_tested += 1
                
            else:
                print(f"{galaxy_name:<8} {'ANALYSIS FAILED':<45}")
                
        except Exception as e:
            print(f"{galaxy_name:<8} ERROR: {str(e)[:40]}")
    
    print("-" * 70)
    print(f"\n📈 SUMMARY:")
    print(f"   Total galaxies tested: {total_tested}")
    print(f"   Significant changes: {significant_changes}")
    print(f"   Change rate: {significant_changes/total_tested*100:.1f}%")
    
    print("\n🔍 KEY INSIGHTS:")
    print("   • Setting innermost bin to R=0 changes the gradient reference point")
    print("   • This affects the linear fit calculation and gradient slopes")
    print("   • Gradients may become steeper/shallower depending on the original inner radius")
    print("   • Physical interpretation: gradients now measured from galaxy center (R=0)")
    
    print("\n" + "="*70)
    print("GRADIENT COMPARISON COMPLETE")
    print("="*70)

if __name__ == "__main__":
    compare_gradient_calculations()
