#!/usr/bin/env python3
"""
Force rest-frame wavelength in saved RDB binned NPZ files.

- Scans output/*_stack/Data/*_stack_RDB_binned.npz
- Estimates z_obs from absorption line minima (Hbeta, Mgb, Fe5270, Fe5335) on bins 0-2
- If wave_frame != 'rest' or rest_frame_applied False or |z_obs| > tol:
    * Divide wavelength by (1+z_sys) where z_sys from metadata or galaxy_catalog
    * Update metadata: wave_frame='rest', rest_frame_applied=True, systemic_redshift_used=z_sys
    * Save file in place

Usage:
  PYTHONPATH=. python tools/force_rest_frame_on_binned.py --tol 0.0015 --dry-run
  PYTHONPATH=. python tools/force_rest_frame_on_binned.py
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from typing import Optional, Tuple

try:
    from galaxy_catalog import get_redshift
except Exception:
    def get_redshift(_g: str) -> float:
        return 0.0

C = 299792.458
LINES = {
    'Hbeta': 4861.33,
    'Mgb': 5175.0,
    'Fe5270': 5270.0,
    'Fe5335': 5335.0,
}

def estimate_z_obs(wave: np.ndarray, flux: np.ndarray, z_guess: float) -> Optional[float]:
    vals = []
    for rest in LINES.values():
        guess = rest * (1 + z_guess)
        mask = (wave > guess - 25) & (wave < guess + 25)
        if mask.sum() < 10:
            continue
        w_sub = wave[mask]
        f_sub = flux[mask]
        if f_sub.size < 5:
            continue
        sm = np.convolve(f_sub, np.ones(5)/5.0, mode='same')
        center = w_sub[np.argmin(sm)]
        z_obs = center / rest - 1.0
        vals.append(z_obs)
    if not vals:
        return None
    return float(np.nanmedian(vals))


def process_file(path: Path, tol: float, dry: bool = False) -> Tuple[bool, str]:
    data = np.load(path, allow_pickle=True)
    wave = data['wavelength']
    spec = data['spectra']
    meta = data['metadata'].item() if 'metadata' in data else {}
    gal = path.name.split('_stack_RDB_binned.npz')[0]

    wave_frame = meta.get('wave_frame', 'unknown')
    rest_applied = bool(meta.get('rest_frame_applied', False))
    z_sys = meta.get('systemic_redshift_used')
    if z_sys in (None, ''):
        z_sys = get_redshift(gal.replace('_stack', ''))
    z_sys = float(z_sys or 0.0)

    # Estimate z_obs from first up to 3 bins
    bin_count = min(3, spec.shape[1])
    zvals = []
    for b in range(bin_count):
        z = estimate_z_obs(wave, spec[:, b], z_sys)
        if z is not None:
            zvals.append(z)
    z_med = float(np.nanmedian(zvals)) if zvals else 0.0

    needs_fix = (wave_frame != 'rest') or (not rest_applied) or (abs(z_med) > tol)
    if not needs_fix:
        return False, f"OK {gal}: wave_frame={wave_frame} rest={rest_applied} z_med={z_med:.5f}"

    if dry:
        return True, f"FIX {gal} (dry-run): wave_frame={wave_frame} rest={rest_applied} z_med={z_med:.5f} -> divide by (1+{z_sys:.5f})"

    # Apply rest-frame correction
    wave_rest = wave / (1.0 + z_sys) if (1.0 + z_sys) != 0 else wave.copy()

    # Update metadata flags
    meta['wave_frame'] = 'rest'
    meta['rest_frame_applied'] = True
    meta['systemic_redshift_used'] = float(z_sys)
    meta['rest_frame_classification'] = 'rest-forced'

    # Save in place, preserving other arrays
    out = {k: data[k] for k in data.files if k not in ('wavelength', 'metadata')}
    np.savez(path, wavelength=wave_rest, metadata=meta, **out)
    return True, f"APPLIED {gal}: z_med={z_med:.5f} -> wave /= (1+{z_sys:.5f})"


def main():
    ap = argparse.ArgumentParser(description='Force rest-frame wavelength in RDB binned NPZs')
    ap.add_argument('--tol', type=float, default=0.0015, help='z tolerance for assuming rest (|z_med|<=tol)')
    ap.add_argument('--dry-run', action='store_true', help='Only report, do not modify files')
    args = ap.parse_args()

    pattern = Path('output').glob('*_stack/Data/*_stack_RDB_binned.npz')
    fixed, total = 0, 0
    for f in sorted(pattern):
        total += 1
        try:
            changed, msg = process_file(f, tol=args.tol, dry=args.dry_run)
            print(msg)
            if changed:
                fixed += 1
        except Exception as e:
            print(f"ERR {f}: {e}")
    print(f"Summary: total={total} fixed={fixed} (tol={args.tol}) dry={args.dry_run}")

if __name__ == '__main__':
    main()
