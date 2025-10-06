#!/usr/bin/env python3
"""Check velocity / rest-frame alignment for binned RDB spectra.

Logic:
 1. Load each *_stack_RDB_binned.npz under output/*_stack/Data.
 2. Read metadata: wave_frame, systemic_redshift_used, rest_frame_classification.
 3. Determine reference redshift:
      if wave_frame == 'rest' or rest_frame_applied -> z_ref = 0
      else -> z_ref = systemic_redshift_used (fallback to catalog if missing)
 4. For first 3 bins, estimate line centers (Hbeta, Mgb, Fe5270, Fe5335) and compute
    Δv = (z_obs - z_ref) * c. Report median per galaxy and flag if |median| > threshold.

Usage:
  python tools/check_velocity_alignment.py [--threshold 80] [--details]

Exit codes:
  0 if all galaxies pass threshold
  1 if any galaxy exceeds threshold
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
try:
    from galaxy_catalog import get_redshift  # type: ignore
except Exception:  # pragma: no cover - fallback when catalog module is unavailable
    def get_redshift(_gal: str) -> float:  # type: ignore
        return 0.0

C = 299792.458
LINES = {
    'Hbeta': 4861.33,
    'Mgb': 5175.0,
    'Fe5270': 5270.0,
    'Fe5335': 5335.0,
}

def measure_offsets(wave, spec, z_ref):
    dv_list = []
    for rest in LINES.values():
        guess = rest * (1 + z_ref)
        mask = (wave > guess - 25) & (wave < guess + 25)
        if mask.sum() < 10:
            continue
        w_sub = wave[mask]
        f_sub = spec[mask]
        if f_sub.size < 5:
            continue
        sm = np.convolve(f_sub, np.ones(5)/5, mode='same')
        center = w_sub[np.argmin(sm)]
        z_obs = center / rest - 1.0
        dv = (z_obs - z_ref) * C
        dv_list.append(dv)
    if not dv_list:
        return None
    return float(np.nanmedian(dv_list)), dv_list


def main():
    ap = argparse.ArgumentParser(description='Check rest-frame velocity alignment for RDB binned data')
    ap.add_argument('--threshold', type=float, default=80.0, help='|median Δv| threshold (km/s) for flagging')
    ap.add_argument('--details', action='store_true', help='Print per-bin detailed values')
    args = ap.parse_args()

    pattern = Path('output').glob('*_stack/Data/*_stack_RDB_binned.npz')
    rows = []
    failed = []
    for f in sorted(pattern):
        gal = f.name.split('_stack_RDB_binned.npz')[0]
        data = np.load(f, allow_pickle=True)
        wave = data['wavelength']
        spec = data['spectra']
        meta = data['metadata'].item() if 'metadata' in data else {}
        wave_frame = meta.get('wave_frame', 'unknown')
        rest_applied = meta.get('rest_frame_applied', False)
        z_sys = meta.get('systemic_redshift_used', None)
        if z_sys is None:
            try:
                z_sys = get_redshift(gal.replace('_stack',''))
            except Exception:
                z_sys = 0.0
        # Decide reference frame
        if wave_frame == 'rest' or rest_applied:
            z_ref = 0.0
        else:
            z_ref = float(z_sys)
        # Optional: subtract per-bin stellar velocities if available to avoid false flags when already in rest frame
        v_bins = None
        # Try to find velocities from standardized results
        try:
            std_path = f.with_name(f.name.replace('_RDB_binned.npz', '_RDB_results.npz'))
            if std_path.exists():
                std = np.load(std_path, allow_pickle=True)
                if 'stellar_kinematics' in std and 'velocity_binned' in std['stellar_kinematics'].item():
                    v_bins = std['stellar_kinematics'].item()['velocity_binned']
        except Exception:
            v_bins = None
        # Fallback: try cube-level bin velocities in metadata if present
        if v_bins is None and isinstance(meta, dict):
            v_bins = meta.get('velocity_binned', None)

        # Measure first 3 bins
        bin_count = min(3, spec.shape[1])
        all_dv = []
        for b in range(bin_count):
            # If in rest frame and per-bin stellar velocities are available, subtract their effect on the fly
            w = wave
            s = spec[:, b]
            if (wave_frame == 'rest' or rest_applied) and v_bins is not None:
                try:
                    v = float(v_bins[b])
                    if np.isfinite(v) and abs(v) < 1000:
                        w = wave / (1 + v / C)
                except Exception:
                    pass
            res = measure_offsets(w, s, z_ref)
            if res:
                med, dv_vals = res
                all_dv.append(med)
                if args.details:
                    print(f"{gal} bin {b} median Δv={med:.1f} km/s values={['%.1f'%v for v in dv_vals]}")
        if all_dv:
            gal_med = float(np.nanmedian(all_dv))
            gal_std = float(np.nanstd(all_dv))
        else:
            gal_med = float('nan'); gal_std = float('nan')
        status = 'OK'
        if not np.isnan(gal_med) and abs(gal_med) > args.threshold:
            status = 'FLAG'
            failed.append(gal)
        rows.append((gal, wave_frame, rest_applied, z_sys, gal_med, gal_std, status))
    # Print summary
    print(f"{'Galaxy':10s} {'Frame':7s} {'Rest?':5s} {'z_sys':8s} {'med_dv':8s} {'std':8s} Status")
    for gal, frame, rest_applied, z_sys, med, std, status in rows:
        print(f"{gal:10s} {frame:7s} {str(rest_applied):5s} {z_sys:8.5f} {med:8.1f} {std:8.1f} {status}")
    if failed:
        print(f"\nGalaxies exceeding |Δv|>{args.threshold} km/s: {', '.join(failed)}")
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
