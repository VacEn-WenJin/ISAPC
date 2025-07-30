#!/usr/bin/env python3
"""
Visualize Alpha/Fe Analysis Results
Create summary plots and analysis of the physics visualization results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_results():
    """Load the analysis results"""
    try:
        result_dir = None
        for dirname in os.listdir('alpha_fe_analysis_results'):
            if dirname.startswith('analysis_') and os.path.isdir(f'alpha_fe_analysis_results/{dirname}'):
                result_dir = f'alpha_fe_analysis_results/{dirname}'
                break
        
        if result_dir is None:
            print("No analysis results found")
            return None
        
        summary_file = os.path.join(result_dir, 'alpha_fe_analysis_summary.csv')
        df = pd.read_csv(summary_file)
        
        print(f"Loaded results from: {result_dir}")
        print(f"Number of galaxies: {len(df)}")
        
        return df, result_dir
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None

def create_summary_plots(df, output_dir):
    """Create summary plots of the alpha/Fe analysis"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Alpha/Fe Abundance Analysis Results - MUSE Galaxy Sample', fontsize=16, fontweight='bold')
    
    # 1. Alpha/Fe distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(df['Mean_Alpha_Fe'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['Mean_Alpha_Fe'].mean(), color='red', linestyle='--', 
                label=f'Mean = {df["Mean_Alpha_Fe"].mean():.3f}')
    ax1.set_xlabel('[α/Fe] (dex)')
    ax1.set_ylabel('Number of Galaxies')
    ax1.set_title('Distribution of Mean [α/Fe]')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Alpha/Fe by galaxy type
    ax2 = axes[0, 1]
    galaxy_types = df['Type'].value_counts()
    type_means = df.groupby('Type')['Mean_Alpha_Fe'].agg(['mean', 'std']).reset_index()
    
    bars = ax2.bar(type_means['Type'], type_means['mean'], 
                   yerr=type_means['std'], capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Galaxy Type')
    ax2.set_ylabel('Mean [α/Fe] (dex)')
    ax2.set_title('Mean [α/Fe] by Galaxy Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(alpha=0.3)
    
    # Add numbers on bars
    for i, (bar, count) in enumerate(zip(bars, galaxy_types[type_means['Type']])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # 3. Alpha/Fe vs redshift
    ax3 = axes[0, 2]
    scatter = ax3.scatter(df['Redshift'], df['Mean_Alpha_Fe'], 
                         c=df['Alpha_Fe_Success'], cmap='viridis', 
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel('Mean [α/Fe] (dex)')
    ax3.set_title('Mean [α/Fe] vs Redshift')
    ax3.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Successful α/Fe pixels')
    
    # 4. Number of successful pixels by galaxy type
    ax4 = axes[1, 0]
    success_by_type = df.groupby('Type')['Alpha_Fe_Success'].agg(['mean', 'std']).reset_index()
    bars4 = ax4.bar(success_by_type['Type'], success_by_type['mean'], 
                    yerr=success_by_type['std'], capsize=5, alpha=0.7, color='lightgreen')
    ax4.set_xlabel('Galaxy Type')
    ax4.set_ylabel('Successful α/Fe pixels')
    ax4.set_title('Data Quality by Galaxy Type')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(alpha=0.3)
    
    # 5. Gradient analysis
    ax5 = axes[1, 1]
    gradient_data = df[df['N_Gradient_Bins'] > 0].copy()
    
    if len(gradient_data) > 0:
        # Color by significance
        colors = ['red' if p < 0.05 else 'blue' for p in gradient_data['Gradient_P_Value']]
        scatter5 = ax5.scatter(gradient_data['Gradient_Slope'], gradient_data['Gradient_P_Value'],
                              c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add zero line and significance line
        ax5.axvline(0, color='gray', linestyle='-', alpha=0.5)
        ax5.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
        
        ax5.set_xlabel('Gradient Slope (dex/Re)')
        ax5.set_ylabel('P-value')
        ax5.set_title('Radial Gradient Significance')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # Add galaxy labels for significant gradients
        sig_gradients = gradient_data[gradient_data['Gradient_P_Value'] < 0.05]
        for _, row in sig_gradients.iterrows():
            ax5.annotate(row['Galaxy'], 
                        (row['Gradient_Slope'], row['Gradient_P_Value']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    else:
        ax5.text(0.5, 0.5, 'No gradient data available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Radial Gradient Analysis')
    
    # 6. Alpha/Fe range (error bars)
    ax6 = axes[1, 2]
    galaxy_names = df['Galaxy']
    mean_values = df['Mean_Alpha_Fe']
    std_values = df['Std_Alpha_Fe']
    
    # Sort by mean alpha/Fe
    sort_idx = np.argsort(mean_values)
    
    x_pos = np.arange(len(galaxy_names))
    ax6.errorbar(x_pos, mean_values[sort_idx], yerr=std_values[sort_idx], 
                fmt='o', capsize=3, alpha=0.7, markersize=4)
    
    ax6.set_xlabel('Galaxy (sorted by [α/Fe])')
    ax6.set_ylabel('[α/Fe] (dex)')
    ax6.set_title('Individual Galaxy [α/Fe] Values')
    ax6.set_xticks(x_pos[::2])  # Show every other label
    ax6.set_xticklabels(galaxy_names[sort_idx][::2], rotation=45, fontsize=8)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'alpha_fe_analysis_summary.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.savefig(plot_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Summary plots saved: {plot_file}")
    
    plt.show()

def print_detailed_summary(df):
    """Print detailed summary statistics"""
    
    print("\n" + "="*80)
    print("DETAILED ALPHA/FE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total galaxies analyzed: {len(df)}")
    print(f"Mean [α/Fe] across sample: {df['Mean_Alpha_Fe'].mean():.3f} ± {df['Mean_Alpha_Fe'].std():.3f}")
    print(f"Range: {df['Mean_Alpha_Fe'].min():.3f} to {df['Mean_Alpha_Fe'].max():.3f}")
    print(f"Total successful α/Fe measurements: {df['Alpha_Fe_Success'].sum()}")
    
    print(f"\nBY GALAXY TYPE:")
    type_stats = df.groupby('Type').agg({
        'Mean_Alpha_Fe': ['count', 'mean', 'std'],
        'Alpha_Fe_Success': 'sum'
    }).round(3)
    print(type_stats)
    
    print(f"\nRADIAL GRADIENT ANALYSIS:")
    gradient_galaxies = df[df['N_Gradient_Bins'] > 0]
    print(f"Galaxies with gradient analysis: {len(gradient_galaxies)}")
    
    if len(gradient_galaxies) > 0:
        significant = gradient_galaxies[gradient_galaxies['Gradient_P_Value'] < 0.05]
        print(f"Significant gradients (p < 0.05): {len(significant)}")
        
        if len(significant) > 0:
            print(f"\nSignificant gradient details:")
            for _, row in significant.iterrows():
                direction = "negative" if row['Gradient_Slope'] < 0 else "positive"
                print(f"  {row['Galaxy']} ({row['Type']}): {row['Gradient_Slope']:.4f} ± {row['Gradient_Error']:.4f} "
                      f"dex/Re ({direction}, p={row['Gradient_P_Value']:.4f})")
        
        # Gradient trends by type
        print(f"\nGradient trends by galaxy type:")
        for gtype in gradient_galaxies['Type'].unique():
            type_grad = gradient_galaxies[gradient_galaxies['Type'] == gtype]
            mean_slope = type_grad['Gradient_Slope'].mean()
            n_type = len(type_grad)
            print(f"  {gtype}: {n_type} galaxies, mean slope = {mean_slope:.4f} dex/Re")
    
    print(f"\nDATA QUALITY:")
    print(f"Average successful pixels per galaxy: {df['Alpha_Fe_Success'].mean():.1f}")
    print(f"Best data quality: {df.loc[df['Alpha_Fe_Success'].idxmax(), 'Galaxy']} "
          f"({df['Alpha_Fe_Success'].max()} pixels)")
    print(f"Galaxies with RDB analysis: {df['Has_RDB'].sum()}")
    print(f"Galaxies with VNB analysis: {df['Has_VNB'].sum()}")
    
    print(f"\nPHYSICAL INTERPRETATION:")
    enhanced_galaxies = df[df['Mean_Alpha_Fe'] > 0.25]
    solar_galaxies = df[(df['Mean_Alpha_Fe'] >= -0.05) & (df['Mean_Alpha_Fe'] <= 0.05)]
    depleted_galaxies = df[df['Mean_Alpha_Fe'] < -0.05]
    
    print(f"Alpha-enhanced galaxies ([α/Fe] > 0.25): {len(enhanced_galaxies)}")
    if len(enhanced_galaxies) > 0:
        print(f"  Types: {', '.join(enhanced_galaxies['Type'].unique())}")
    
    print(f"Solar-like galaxies (-0.05 ≤ [α/Fe] ≤ 0.05): {len(solar_galaxies)}")
    if len(solar_galaxies) > 0:
        print(f"  Types: {', '.join(solar_galaxies['Type'].unique())}")
    
    print(f"Alpha-depleted galaxies ([α/Fe] < -0.05): {len(depleted_galaxies)}")
    if len(depleted_galaxies) > 0:
        print(f"  Types: {', '.join(depleted_galaxies['Type'].unique())}")

def main():
    """Main visualization function"""
    
    print("ALPHA/FE ANALYSIS VISUALIZATION")
    print("="*50)
    
    # Load results
    df, result_dir = load_results()
    if df is None:
        return
    
    # Create visualizations
    print("\nCreating summary plots...")
    create_summary_plots(df, result_dir)
    
    # Print detailed summary
    print_detailed_summary(df)
    
    print(f"\n✅ Analysis complete! Results available in: {result_dir}")

if __name__ == "__main__":
    main()
