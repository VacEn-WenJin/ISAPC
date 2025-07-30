#!/usr/bin/env python3

import sys
sys.path.append('.')
from enhanced_radial_plots_3bin_corrected import calculate_gradient_3bin_rdb_vnb

# Test VCC1431 only
print("Testing VCC1431...")
result = calculate_gradient_3bin_rdb_vnb('VCC1431')
if result:
    print("SUCCESS!")
    rdb = result['RDB']
    vnb = result['VNB']
    
    print(f"RDB gradient: {rdb['slope']:.4f} ± {rdb['slope_error']:.4f} dex/Re")
    print(f"RDB n_bins: {rdb['n_bins']}")
    print(f"RDB radii_norm shape: {len(rdb['radii_norm'])}")
    print(f"RDB alpha_fe shape: {len(rdb['alpha_fe'])}")
    
    print(f"VNB gradient: {vnb['slope']:.4f} ± {vnb['slope_error']:.4f} dex/Re")
    print(f"VNB n_bins: {vnb['n_bins']}")
    print(f"VNB radii_norm shape: {len(vnb['radii_norm'])}")
    print(f"VNB alpha_fe shape: {len(vnb['alpha_fe'])}")
    
    # Create a summary CSV
    import pandas as pd
    summary = [{
        'Galaxy': 'VCC1431',
        'RDB_slope': rdb['slope'],
        'RDB_slope_error': rdb['slope_error'],
        'RDB_R_squared': rdb['r_squared'],
        'RDB_n_bins': rdb['n_bins'],
        'VNB_slope': vnb['slope'],
        'VNB_slope_error': vnb['slope_error'],
        'VNB_R_squared': vnb['r_squared'],
        'VNB_n_bins': vnb['n_bins']
    }]
    
    df = pd.DataFrame(summary)
    df.to_csv('vcc1431_gradient_summary.csv', index=False)
    print("Saved summary to vcc1431_gradient_summary.csv")
    
else:
    print("FAILED!")
