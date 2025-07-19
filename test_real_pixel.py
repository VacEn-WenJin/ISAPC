#!/usr/bin/env python3
"""
Test script for single pixel from real MUSE data
Tests the LineIndexCalculator with real data
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_indices import LineIndexCalculator
import logging
from muse import MUSE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_single_pixel():
    """Test spectral index calculation for a single pixel from real MUSE data"""
    print("=" * 60)
    print("Testing Single Pixel from Real MUSE Data")
    print("=" * 60)
    
    try:
        # Step 1: Load MUSE data
        print("\n1. Loading MUSE data...")
        muse_file = "data/MUSE/VCC0308_stack/VCC0308_linear.fits"
        
        # Create MUSE object and load data
        muse = MUSE(muse_file)
        print(f"   ✓ Loaded MUSE data: {muse._n_x}x{muse._n_y} pixels, {muse._n_wave} wavelengths")
        print(f"   ✓ Wavelength range: {muse._lambda.min():.1f} - {muse._lambda.max():.1f} Å")
        
        # Select a pixel near the center
        center_x = muse._n_x // 2
        center_y = muse._n_y // 2
        pixel_idx = center_y * muse._n_x + center_x
        
        print(f"   ✓ Selected pixel: ({center_x}, {center_y}), index {pixel_idx}")
        
        # Step 2: Get single pixel spectrum
        print("\n2. Extracting single pixel spectrum...")
        spectrum = muse._cube[:, center_y, center_x]
        wave = muse._lambda.copy()
        
        # Check for valid data
        valid_mask = np.isfinite(spectrum) & (spectrum > 0)
        if np.sum(valid_mask) < len(spectrum) * 0.5:
            print(f"   ⚠ Warning: Only {np.sum(valid_mask)}/{len(spectrum)} valid pixels")
        
        print(f"   ✓ Spectrum range: {spectrum[valid_mask].min():.3e} - {spectrum[valid_mask].max():.3e}")
        
        # Step 3: Create a simple template (smoothed spectrum)
        print("\n3. Creating template spectrum...")
        from scipy.ndimage import uniform_filter1d
        template_flux = uniform_filter1d(spectrum, size=5)
        
        # Estimate errors
        residuals = spectrum - template_flux
        error_estimate = np.full_like(spectrum, np.std(residuals[valid_mask]))
        
        print(f"   ✓ Template created with error estimate: {error_estimate[0]:.3e}")
        
        # Step 4: Test LineIndexCalculator
        print("\n4. Testing LineIndexCalculator with real data...")
        try:
            calculator = LineIndexCalculator(
                wave=wave,
                flux=spectrum,
                fit_wave=wave,
                fit_flux=template_flux,
                velocity_correction=0.0,  # No velocity correction for test
                error=error_estimate,
                velocity_error=5.0,
                continuum_mode='auto',
                show_warnings=True
            )
            print("   ✓ LineIndexCalculator created successfully")
            
        except Exception as e:
            print(f"   ✗ Error creating LineIndexCalculator: {e}")
            return False
        
        # Step 5: Test spectral indices
        print("\n5. Testing spectral indices...")
        indices_to_test = ['Hbeta', 'Mgb', 'Fe5015', 'Fe5270', 'Fe5335']
        
        results = {}
        for index_name in indices_to_test:
            print(f"\n   Testing {index_name}:")
            try:
                # Test basic calculation
                value = calculator.calculate_index(index_name)
                print(f"      Value: {value:.4f} Å")
                
                # Test with error calculation
                value_with_error, error_estimate = calculator.calculate_index(index_name, return_error=True)
                print(f"      Value with error: {value_with_error:.4f} ± {error_estimate:.4f} Å")
                
                # Check if values are finite
                if np.isfinite(value) and np.isfinite(value_with_error) and np.isfinite(error_estimate):
                    print(f"      ✓ {index_name} calculation successful")
                    results[index_name] = {
                        'value': value_with_error,
                        'error': error_estimate,
                        'status': 'success'
                    }
                else:
                    print(f"      ✗ {index_name} returned non-finite values")
                    results[index_name] = {
                        'value': np.nan,
                        'error': np.nan,
                        'status': 'nan_values'
                    }
                    
            except Exception as e:
                print(f"      ✗ Error calculating {index_name}: {e}")
                results[index_name] = {
                    'value': np.nan,
                    'error': np.nan,
                    'status': f'error: {e}'
                }
        
        # Step 6: Create visualization
        print("\n6. Creating visualization...")
        try:
            fig, axes = create_real_data_plot(calculator, results, wave, spectrum, template_flux)
            plt.savefig('/tmp/test_real_pixel.png', dpi=150, bbox_inches='tight')
            print("   ✓ Plot saved to /tmp/test_real_pixel.png")
            plt.close()
        except Exception as e:
            print(f"   ✗ Error creating plot: {e}")
        
        # Step 7: Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        successful_indices = [name for name, data in results.items() if data['status'] == 'success']
        failed_indices = [name for name, data in results.items() if data['status'] != 'success']
        
        print(f"Successful indices ({len(successful_indices)}): {', '.join(successful_indices)}")
        if failed_indices:
            print(f"Failed indices ({len(failed_indices)}): {', '.join(failed_indices)}")
        
        if len(successful_indices) > 0:
            print("\n✓ Real data spectral index calculation is working!")
            return True
        else:
            print("\n✗ All real data spectral index calculations failed!")
            return False
            
    except Exception as e:
        print(f"✗ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_real_data_plot(calculator, results, wave, spectrum, template_flux):
    """Create a diagnostic plot for real data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Spectrum with line regions
    ax1.plot(wave, spectrum, 'b-', label='Observed', alpha=0.7, linewidth=1)
    ax1.plot(wave, template_flux, 'r--', label='Template', alpha=0.7, linewidth=1)
    
    # Add error band
    if hasattr(calculator, 'error'):
        ax1.fill_between(wave, 
                         spectrum - calculator.error,
                         spectrum + calculator.error,
                         alpha=0.2, color='blue', label='Error band')
    
    # Mark line regions
    line_colors = {'Hbeta': 'green', 'Mgb': 'orange', 'Fe5015': 'purple', 'Fe5270': 'brown', 'Fe5335': 'pink'}
    
    for line_name in ['Hbeta', 'Mgb', 'Fe5015', 'Fe5270', 'Fe5335']:
        windows = calculator.define_line_windows(line_name)
        if windows:
            color = line_colors.get(line_name, 'gray')
            
            # Blue continuum
            if 'blue' in windows:
                ax1.axvspan(windows['blue'][0], windows['blue'][1], alpha=0.15, color=color)
            
            # Line region
            line_region = windows.get('line', windows.get('band'))
            if line_region:
                ax1.axvspan(line_region[0], line_region[1], alpha=0.3, color=color)
                # Add text label
                ax1.text(np.mean(line_region), ax1.get_ylim()[1] * 0.95, line_name,
                        ha='center', va='center', fontweight='bold', color=color, fontsize=8)
                
            # Red continuum
            if 'red' in windows:
                ax1.axvspan(windows['red'][0], windows['red'][1], alpha=0.15, color=color)
    
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Flux')
    ax1.set_title('Real MUSE Spectrum with Line Regions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Focus on spectral line region
    ax1.set_xlim(4800, 5350)
    
    # Plot 2: Results summary
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    
    if successful_results:
        indices = list(successful_results.keys())
        values = [successful_results[idx]['value'] for idx in indices]
        errors = [successful_results[idx]['error'] for idx in indices]
        
        x_pos = np.arange(len(indices))
        bars = ax2.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, color='green')
        
        ax2.set_xlabel('Spectral Index')
        ax2.set_ylabel('Index Value (Å)')
        ax2.set_title('Spectral Index Results (Real Data)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(indices, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (value, error) in enumerate(zip(values, errors)):
            if np.isfinite(value):
                ax2.text(i, value + error + max(values) * 0.02, f'{value:.3f}±{error:.3f}', 
                        ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No successful calculations', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14, color='red')
        ax2.set_title('No Results')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

if __name__ == "__main__":
    success = test_real_single_pixel()
    if success:
        print("\n🎉 Real pixel test completed successfully!")
    else:
        print("\n❌ Real pixel test failed!")
