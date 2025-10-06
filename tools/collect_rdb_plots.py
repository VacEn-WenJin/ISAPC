#!/usr/bin/env python3
"""Collect all first 3 bin RDB normalized plots into a single combined folder.

Features
--------
For every galaxy (data/MUSE/*_stack.fits), gather the standardized set of
RDB normalized/overlay plots produced by RDB mode and place them under a
common destination directory for easy browsing or packaging.

Default destination layout (without --flat):
  combined_rdb_plots/
    VCC1588/
      RDB0_norm.png
      RDB1_norm.png
      RDB2_norm.png
      RDB_first3_norm_overlay.png
      RDB_first3_norm_shaded_overlay.png
      RDB_first3_norm_combined.png
    VCC1146/
      ...

If --flat is supplied the files are instead placed directly in the destination
directory with galaxy name prefixed:
  combined_rdb_plots/
    VCC1588_RDB0_norm.png
    VCC1588_RDB_first3_norm_overlay.png
    ...

By default files are copied. You can instead:
  --move       Move files (remove originals)
  --symlink    Create symlinks instead of copies

Usage
-----
  python tools/collect_rdb_plots.py
  python tools/collect_rdb_plots.py --dest combined_rdb_plots_flat --flat
  python tools/collect_rdb_plots.py --galaxy VCC1588,VCC1146 --symlink

Exit Codes
----------
  0 success (some galaxies may have missing files but process completes)
  2 configuration error (no galaxies)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
MUSE_DIR = ROOT / 'data' / 'MUSE'

REQUIRED_PLOTS = [
    'RDB0_norm.png',
    'RDB1_norm.png',
    'RDB2_norm.png',
    'RDB_first3_norm_overlay.png',
    'RDB_first3_norm_shaded_overlay.png',
    'RDB_first3_norm_combined.png',
]


def find_galaxies(selected: List[str] | None = None) -> List[Dict]:
    gals: List[Dict] = []
    # Prefer discovering from output/* directories to include *_obstack as well
    out = ROOT / 'output'
    if out.exists():
        for d in sorted(out.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            # Expect a Data dir with RDB_binned
            if (d / 'Data').exists() and any((d / 'Data').glob(f'{name}_RDB_binned.npz')):
                base = name
                if selected and base not in selected:
                    continue
                gals.append({'name': base, 'file': None})
    # Fallback to MUSE stacks if output discovery fails
    if not gals:
        for fits in sorted(MUSE_DIR.glob('*_stack.fits')):
            name = fits.stem.replace('_stack', '')
            if selected and name not in selected:
                continue
            gals.append({'name': name, 'file': fits})
    return gals


def source_dir(gal: str) -> Path:
    # Support both *_stack and *_obstack (or raw name already includes suffix)
    # Try exact name first, then name_stack fallback
    d1 = ROOT / 'output' / gal / 'Plots' / 'RDB' / 'spectral_indices' / 'detailed'
    if d1.exists():
        return d1
    return ROOT / 'output' / f'{gal}_stack' / 'Plots' / 'RDB' / 'spectral_indices' / 'detailed'


def collect(gal: str, dest_root: Path, mode: str, flat: bool) -> dict:
    sdir = source_dir(gal)
    if flat:
        out_dir = dest_root
    else:
        out_dir = dest_root / gal
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {'galaxy': gal, 'copied': [], 'missing': []}
    for fname in REQUIRED_PLOTS:
        src = sdir / fname
        if not src.exists():
            report['missing'].append(fname)
            continue
        if flat:
            target_name = f"{gal}_{fname}"
        else:
            target_name = fname
        dst = out_dir / target_name
        try:
            if mode == 'copy':
                shutil.copy2(src, dst)
            elif mode == 'move':
                shutil.move(src, dst)
            elif mode == 'symlink':
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            report['copied'].append(target_name)
        except Exception as e:
            report['missing'].append(f"{fname} (error: {e})")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description='Collect RDB plot set into a combined folder')
    ap.add_argument('--dest', type=str, default='combined_rdb_plots', help='Destination directory')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated subset of galaxies (use names as in output/*)')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--move', action='store_true', help='Move instead of copy')
    group.add_argument('--symlink', action='store_true', help='Symlink instead of copy')
    ap.add_argument('--flat', action='store_true', help='Do not create per-galaxy subdirectories; prefix filenames with galaxy name')
    args = ap.parse_args()

    selected = [g.strip() for g in args.galaxy.split(',') if g.strip()] if args.galaxy else None
    gals = find_galaxies(selected)
    if not gals:
        print('No galaxies found.')
        return 2

    dest_root = Path(args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    mode = 'copy'
    if args.move:
        mode = 'move'
    elif args.symlink:
        mode = 'symlink'
    print(f"Destination: {dest_root} (mode={mode}, flat={args.flat})")

    summaries = []
    total_missing = 0
    for i, g in enumerate(gals, 1):
        gal = g['name']
        rep = collect(gal, dest_root, mode, args.flat)
        summaries.append(rep)
        if rep['missing']:
            total_missing += len(rep['missing'])
            print(f"[{i}/{len(gals)}] {gal}: missing {len(rep['missing'])} -> {rep['missing'][:3]}{'...' if len(rep['missing'])>3 else ''}")
        else:
            print(f"[{i}/{len(gals)}] {gal}: collected {len(rep['copied'])} files")

    print('\nSummary')
    print('-------')
    print(f"Galaxies processed: {len(gals)}")
    print(f"Total missing entries: {total_missing}")
    if total_missing:
        incomplete = [r['galaxy'] for r in summaries if r['missing']]
        print(f"Galaxies with missing files: {len(incomplete)} -> {', '.join(incomplete)}")
    else:
        print('All required files present for all galaxies.')
    print(f"Output collected at: {dest_root}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
