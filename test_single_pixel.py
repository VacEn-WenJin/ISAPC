#!/usr/bin/env python3
"""
Test script for single pixel spectral index calculation
Tests the LineIndexCalculator with mathematical error propagation
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_indices import LineIndexCalculator
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_spectrum():
    """Create a simple test spectrum with known features"""
    # Wavelength range covering the spectral lines of interest
    wave = np.linspace(4800, 5300, 1000)
    
    # Create a simple continuum with some slope
    continuum = 1.0 + 0.0001 * (wave - 5000)
    
    # Add some absorption lines at known positions
    # Hbeta at ~4861
    hbeta_center = 4861.3
    hbeta_depth = 0.3
    hbeta_width = 5.0
    hbeta_profile = hbeta_depth * np.exp(-0.5 * ((wave - hbeta_center) / hbeta_width)**2)
    
    # Mgb at ~5175
    mgb_center = 5175.0
    mgb_depth = 0.25
    mgb_width = 8.0
    mgb_profile = mgb_depth * np.exp(-0.5 * ((wave - mgb_center) / mgb_width)**2)
    
    # Fe5015 at ~5015
    fe5015_center = 5015.0
    fe5015_depth = 0.15
    fe5015_width = 10.0
    fe5015_profile = fe5015_depth * np.exp(-0.5 * ((wave - fe5015_center) / fe5015_width)**2)
    
    # Combine to create spectrum
    flux = continuum - hbeta_profile - mgb_profile - fe5015_profile
    
    # Add some noise
    noise_level = 0.02
    noise = np.random.normal(0, noise_level, len(flux))
    flux += noise
    
    # Create error array
    error = np.full_like(flux, noise_level)
    
    return wave, flux, error

def create_template_spectrum(wave, flux):
    """Create a simple template spectrum (smoothed version of input)"""
    # Simple smoothing using a running average
    from scipy.ndimage import uniform_filter1d
    template_flux = uniform_filter1d(flux, size=5)
    
    return wave.copy(), template_flux

def test_single_pixel():
    """Test spectral index calculation for a single pixel"""
    print("=" * 60)
    print("Testing Single Pixel Spectral Index Calculation")
    print("=" * 60)
    
    # Step 1: Create test data
    print("\n1. Creating test spectrum...")
    wave, flux, error = create_test_spectrum()
    template_wave, template_flux = create_template_spectrum(wave, flux)
    
    print(f"   Wavelength range: {wave.min():.1f} - {wave.max():.1f} Å")
    print(f"   Flux range: {flux.min():.3f} - {flux.max():.3f}")
    print(f"   Mean error: {np.mean(error):.3f}")
    
    # Step 2: Create LineIndexCalculator
    print("\n2. Creating LineIndexCalculator...")
    try:
        calculator = LineIndexCalculator(
            wave=wave,
            flux=flux,
            fit_wave=template_wave,
            fit_flux=template_flux,
            velocity_correction=50.0,  # 50 km/s test velocity
            error=error,
            velocity_error=10.0,  # 10 km/s velocity error
            continuum_mode='auto',
            show_warnings=True
        )
        print("   ✓ LineIndexCalculator created successfully")
        print(f"   ✓ Residuals calculated: {calculator.residuals is not None}")
        print(f"   ✓ Error array shape: {calculator.error.shape}")
        print(f"   ✓ Error range: {calculator.error.min():.3f} - {calculator.error.max():.3f}")
        
    except Exception as e:
        print(f"   ✗ Error creating LineIndexCalculator: {e}")
        return False
    
    # Step 3: Test individual spectral indices
    print("\n3. Testing individual spectral indices...")
    indices_to_test = ['Hbeta', 'Mgb', 'Fe5015', 'D4000']
    
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
    
    # Step 4: Test all indices at once
    print("\n4. Testing all indices calculation...")
    try:
        all_indices = calculator.calculate_all_indices(return_errors=True)
        print(f"   ✓ All indices calculated: {len(all_indices)} indices")
        for name, data in all_indices.items():
            if isinstance(data, dict):
                print(f"      {name}: {data['value']:.4f} ± {data['error']:.4f}")
            else:
                print(f"      {name}: {data:.4f}")
    except Exception as e:
        print(f"   ✗ Error calculating all indices: {e}")
    
    # Step 5: Create visualization
    print("\n5. Creating visualization...")
    try:
        fig, axes = create_test_plot(calculator, results)
        plt.savefig('/tmp/test_single_pixel.png', dpi=150, bbox_inches='tight')
        print("   ✓ Plot saved to /tmp/test_single_pixel.png")
        plt.close()
    except Exception as e:
        print(f"   ✗ Error creating plot: {e}")
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful_indices = [name for name, data in results.items() if data['status'] == 'success']
    failed_indices = [name for name, data in results.items() if data['status'] != 'success']
    
    print(f"Successful indices ({len(successful_indices)}): {', '.join(successful_indices)}")
    if failed_indices:
        print(f"Failed indices ({len(failed_indices)}): {', '.join(failed_indices)}")
    
    if len(successful_indices) > 0:
        print("\n✓ Basic spectral index calculation is working!")
        return True
    else:
        print("\n✗ All spectral index calculations failed!")
        return False

def create_test_plot(calculator, results):
    """Create a diagnostic plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Spectrum with line regions
    ax1.plot(calculator.wave, calculator.flux, 'b-', label='Observed', alpha=0.7)
    ax1.plot(calculator.fit_wave, calculator.fit_flux, 'r--', label='Template', alpha=0.7)
    
    # Add error band
    ax1.fill_between(calculator.wave, 
                     calculator.flux - calculator.error,
                     calculator.flux + calculator.error,
                     alpha=0.2, color='blue', label='Error band')
    
    # Mark line regions
    line_colors = {'Hbeta': 'green', 'Mgb': 'orange', 'Fe5015': 'purple', 'D4000': 'red'}
    
    for line_name in ['Hbeta', 'Mgb', 'Fe5015']:
        windows = calculator.define_line_windows(line_name)
        if windows:
            color = line_colors.get(line_name, 'gray')
            if 'blue' in windows:
                ax1.axvspan(windows['blue'][0], windows['blue'][1], alpha=0.2, color=color)
            
            # Handle both 'line' and 'band' keys
            line_region = windows.get('line', windows.get('band'))
            if line_region:
                ax1.axvspan(line_region[0], line_region[1], alpha=0.3, color=color)
                
            if 'red' in windows:
                ax1.axvspan(windows['red'][0], windows['red'][1], alpha=0.2, color=color)
            
            # Add text label
            if line_region:
                ax1.text(np.mean(line_region), ax1.get_ylim()[1] * 0.9, line_name,
                        ha='center', va='center', fontweight='bold', color=color)
    
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Flux')
    ax1.set_title('Test Spectrum with Line Regions')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Results summary
    indices = list(results.keys())
    values = [results[idx]['value'] for idx in indices]
    errors = [results[idx]['error'] for idx in indices]
    
    x_pos = np.arange(len(indices))
    bars = ax2.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
    
    # Color bars based on success/failure
    for i, (idx, data) in enumerate(results.items()):
        if data['status'] == 'success':
            bars[i].set_color('green')
        else:
            bars[i].set_color('red')
    
    ax2.set_xlabel('Spectral Index')
    ax2.set_ylabel('Index Value (Å)')
    ax2.set_title('Spectral Index Results')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(indices, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (value, error) in enumerate(zip(values, errors)):
        if np.isfinite(value):
            ax2.text(i, value + error + 0.01, f'{value:.3f}±{error:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

if __name__ == "__main__":
    success = test_single_pixel()
    if success:
        print("\n🎉 Single pixel test completed successfully!")
    else:
        print("\n❌ Single pixel test failed!")
