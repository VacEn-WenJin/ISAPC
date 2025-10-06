#!/usr/bin/env python3
"""
Build RDB 3-bin normalized overlay plots for all galaxies.
- Scans data/MUSE for *_stack.fits files
- For each galaxy, checks for overlay at
  output/<GAL>_stack/Plots/RDB/spectral_indices/detailed/RDB_first3_norm_overlay.png
- If missing or --force, runs main.py in RDB mode with --auto-reuse to generate

Usage:
  python -u tools/build_rdb_overlays_all.py [--force] [--galaxy VCC1588[,VCC1146,...]]

Notes:
- Requires template at ./data/templates/spectra_emiles_9.0.npz
- Uses galaxy_catalog.get_redshift(gal) for z
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys

# Local imports
try:
    from galaxy_catalog import get_redshift
except Exception:
    def get_redshift(_gal: str) -> float:
        return 0.0

ROOT = Path(__file__).resolve().parents[1]
MUSE_DIR = ROOT / 'data' / 'MUSE'
TEMPLATE = ROOT / 'data' / 'templates' / 'spectra_emiles_9.0.npz'


def find_galaxies(selected: list[str] | None = None) -> list[dict]:
    gals = []
    for fits in sorted(MUSE_DIR.glob('*.fits')):
        name = fits.stem
        if name.endswith('_stack'):
            name = name[:-6]
        if selected and name not in selected:
            continue
        gals.append({'name': name, 'file': fits})
    return gals


def overlay_path(gal: str) -> Path:
    return ROOT / 'output' / f'{gal}_stack' / 'Plots' / 'RDB' / 'spectral_indices' / 'detailed' / 'RDB_first3_norm_overlay.png'


def build_overlay(gal: str, fits_path: Path, force: bool = False) -> bool:
    out_png = overlay_path(gal)
    if out_png.exists() and not force:
        print(f"✓ {gal}: overlay exists -> {out_png}")
        return True
    if not TEMPLATE.exists():
        print(f"✗ Template missing: {TEMPLATE}")
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
    print(f"→ {gal}: running RDB to generate overlay...")
    try:
        res = subprocess.run(cmd, text=True, capture_output=True, timeout=3600)
        if res.returncode != 0:
            print(f"✗ {gal}: RDB failed (code {res.returncode})\n{res.stderr[-800:]} ")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {gal}: RDB timed out")
        return False
    ok = out_png.exists()
    print(f"{'✓' if ok else '✗'} {gal}: overlay {'created' if ok else 'missing after run'} -> {out_png}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='Rebuild overlays even if they exist')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated galaxy names to limit run')
    args = ap.parse_args()

    selected = [g.strip() for g in args.galaxy.split(',') if g.strip()] if args.galaxy else None

    if not MUSE_DIR.exists():
        print(f"MUSE data directory not found: {MUSE_DIR}")
        return 2

    gals = find_galaxies(selected)
    if not gals:
        print("No galaxies found under data/MUSE.")
        return 0

    ok = 0
    for g in gals:
        if build_overlay(g['name'], g['file'], force=args.force):
            ok += 1
    print(f"Done. {ok}/{len(gals)} overlays present.")
    return 0 if ok == len(gals) else 1


if __name__ == '__main__':
    raise SystemExit(main())
