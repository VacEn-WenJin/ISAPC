#!/usr/bin/env python3
"""
Generate a concise CSV summary of the ISAPC + AIP run for all galaxies.

Columns:
  galaxy, isapc_status, aip_map, aip_profile, norm_spectra_count, notes
"""
import csv
from pathlib import Path

def main():
    root = Path('output')
    out_csv = Path('FINAL_DELIVERABLES') / 'run_summary.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for gdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith('_stack')):
        galaxy = gdir.name.replace('_stack','')
        data = gdir / 'Data'
        plots = gdir / 'Plots'
        status = 'ok' if (data / f'{galaxy}_stack_P2P_results.npz').exists() else 'missing'
        aip_map = 1 if (plots / f'{galaxy}_AIP_alpha_fe_map.png').exists() else 0
        aip_profile = 1 if (plots / f'{galaxy}_AIP_alpha_fe_radial_profile.png').exists() else 0
        # Count normalized spectra saved either as '<galaxy>_P2P_spectrum_norm_*.png'
        # or '<galaxy>_stack_P2P_spectrum_norm_*.png' in any subfolder under Plots
        norm_patterns = [
            f'{galaxy}_P2P_spectrum_norm_*.png',
            f'{galaxy}_stack_P2P_spectrum_norm_*.png'
        ]
        norm_count = 0
        if plots.exists():
            for pat in norm_patterns:
                norm_count += len(list(plots.rglob(pat)))
        notes = ''
        if status != 'ok':
            notes = 'missing P2P results'
        elif aip_map == 0:
            notes = 'no AIP map'
        rows.append([galaxy, status, aip_map, aip_profile, norm_count, notes])

    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['galaxy','isapc_status','aip_map','aip_profile','norm_spectra_count','notes'])
        w.writerows(rows)
    print(f'Wrote {out_csv} with {len(rows)} rows')

if __name__ == '__main__':
    main()
