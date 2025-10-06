#!/usr/bin/env python3
"""Batch builder for all RDB (radial bin) normalized spectral plots.

Purpose
-------
Generate (or verify) the suite of first three radial-bin normalized spectra
plots for every available MUSE galaxy cube (<name>_stack.fits) under
``data/MUSE``. It focuses on the artifacts created inside the RDB mode of
``main.py`` in ``analysis/radial.py``:

  Required plot set (per galaxy):
    - RDB0_norm.png
    - RDB1_norm.png
    - RDB2_norm.png
    - RDB_first3_norm_overlay.png
    - RDB_first3_norm_shaded_overlay.png
    - RDB_first3_norm_combined.png

If any required plot is missing (or ``--force`` supplied), the script invokes
``main.py`` in RDB mode to (re)generate them using the trusted parameter set
mirroring prior validated runs (target SNR=20, 6 rings, poly-degree=3, etc.).

Usage
-----
  python -u tools/build_rdb_all_plots.py [--force] [--galaxy VCC1588,VCC1146] [--check-only]

Options
-------
  --force       Re-run RDB mode even if all required plots already exist.
  --galaxy      Comma-separated subset of galaxy names to limit processing.
  --check-only  Only report status; do not run any processing.

Exit codes
----------
  0  All requested galaxies possess the full required plot set (after any runs).
  1  Some galaxies failed to build or are still missing required plots.
  2  No galaxies found / configuration error.

Notes
-----
  - Relies on ``galaxy_catalog.get_redshift``; falls back to z=0 if unavailable.
  - Uses ``--auto-reuse`` and ``--cvt`` to minimize recomputation.
  - Plots are created under:
      output/<GAL>_stack/Plots/RDB/spectral_indices/detailed/
  - Safe to re-run. Use ``--force`` if you need to refresh all plots.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# Attempt local import; provide fallback if missing
try:
    from galaxy_catalog import get_redshift  # type: ignore
except Exception:  # pragma: no cover - fallback
    def get_redshift(_gal: str) -> float:  # type: ignore
        return 0.0

ROOT = Path(__file__).resolve().parents[1]
MUSE_DIR = ROOT / 'data' / 'MUSE'
TEMPLATE = ROOT / 'data' / 'templates' / 'spectra_emiles_9.0.npz'

# Required plot filenames (basename only) residing in detailed directory
REQUIRED_PLOTS = [
    'RDB0_norm.png',
    'RDB1_norm.png',
    'RDB2_norm.png',
    'RDB_first3_norm_overlay.png',
    'RDB_first3_norm_shaded_overlay.png',
    'RDB_first3_norm_combined.png',
]


def find_galaxies(selected: List[str] | None = None) -> List[Dict]:
    """Locate *_stack.fits MUSE cubes.

    Parameters
    ----------
    selected : list[str] | None
        Optional whitelist of galaxy names.
    """
    gals: List[Dict] = []
    for fits in sorted(MUSE_DIR.glob('*_stack.fits')):
        name = fits.stem.replace('_stack', '')
        if selected and name not in selected:
            continue
        gals.append({'name': name, 'file': fits})
    return gals


def detailed_dir(gal: str) -> Path:
    return ROOT / 'output' / f'{gal}_stack' / 'Plots' / 'RDB' / 'spectral_indices' / 'detailed'


def missing_plots(gal: str) -> List[str]:
    ddir = detailed_dir(gal)
    missing: List[str] = []
    for fname in REQUIRED_PLOTS:
        if not (ddir / fname).exists():
            missing.append(fname)
    return missing


def run_rdb(gal: str, fits_path: Path) -> bool:
    """Invoke main.py in RDB mode to create missing plots."""
    if not TEMPLATE.exists():
        print(f"✗ {gal}: template missing: {TEMPLATE}")
        return False
    z = get_redshift(gal)
    cmd = [
        sys.executable, str(ROOT / 'main.py'),
        str(fits_path),
        '-z', str(z),
        '-t', str(TEMPLATE),
        '-o', str(ROOT / 'output'),
        '-m', 'RDB',
        '--target-snr', '20.0', '--min-snr', '1.0',
        '--n-rings', '6', '--n-jobs', '4',
        '--vel-init', '0.0', '--sigma-init', '50.0',
        '--poly-degree', '3', '--auto-reuse', '--cvt'
    ]
    print(f"→ {gal}: running RDB (command truncated) ...")
    try:
        res = subprocess.run(cmd, text=True, capture_output=True, timeout=5400)
        if res.returncode != 0:
            tail = res.stderr[-800:] if res.stderr else ''
            print(f"✗ {gal}: RDB run failed (code {res.returncode})\n{tail}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {gal}: RDB run timed out")
        return False
    # Verify creation
    missing = missing_plots(gal)
    if missing:
        print(f"✗ {gal}: still missing after run: {missing}")
        return False
    print(f"✓ {gal}: all required plots present")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build or verify RDB normalized plots for all galaxies")
    ap.add_argument('--force', action='store_true', help='Force re-run even if all plots exist')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated list of galaxies to process')
    ap.add_argument('--check-only', action='store_true', help='Only report status, do not run main.py')
    args = ap.parse_args()

    selected = [g.strip() for g in args.galaxy.split(',') if g.strip()] if args.galaxy else None

    if not MUSE_DIR.exists():
        print(f"MUSE data directory not found: {MUSE_DIR}")
        return 2
    gals = find_galaxies(selected)
    if not gals:
        print("No galaxies found (selection may be too restrictive).")
        return 2

    total = len(gals)
    ok = 0
    skipped = 0
    failed: List[str] = []

    print(f"Discovered {total} galaxy(ies). Starting verification...")

    for idx, g in enumerate(gals, 1):
        gal = g['name']
        fits_path = g['file']
        miss = missing_plots(gal)
        if not miss and not args.force:
            print(f"[{idx}/{total}] ✓ {gal}: all plots exist")
            ok += 1
            continue
        if miss and args.check_only:
            print(f"[{idx}/{total}] ✗ {gal}: missing {miss}")
            failed.append(gal)
            continue
        if not miss and args.force:
            print(f"[{idx}/{total}] ↻ {gal}: forcing rebuild (plots present)")
        else:
            print(f"[{idx}/{total}] • {gal}: missing {miss if miss else 'None (force rebuild)'} -> running RDB")
        success = run_rdb(gal, fits_path)
        if success:
            ok += 1
        else:
            failed.append(gal)

    print("\nSummary")
    print("-------")
    print(f"Total galaxies processed: {total}")
    print(f"Successful (all plots present): {ok}")
    if failed:
        print(f"Failed / incomplete: {len(failed)}")
        for gal in failed:
            print(f"  - {gal}")
    if args.check_only:
        print("Mode: check-only (no processing runs executed)")

    return 0 if (ok == total) else 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
