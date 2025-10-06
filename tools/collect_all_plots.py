#!/usr/bin/env python3
"""
Collect all plots into a unified PLOT_OUTPUT folder per galaxy.

Copies PNG/PDF files from:
  - output/<galaxy>/Plots/**
  - output/<galaxy>/** (top-level PDFs like RDB_auto*.pdf)
into:
  PLOT_OUTPUT/<galaxy>/** (preserving relative subfolder structure under Plots)

Usage:
  python tools/collect_all_plots.py
  python tools/collect_all_plots.py --galaxy VCC1949_best2_obstack,VCC1695_selected_obstack
  python tools/collect_all_plots.py --dest PLOT_OUTPUT
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def iter_plot_files(galaxy_dir: Path):
    # Include Plots subtree and any PDFs directly under galaxy dir (e.g., RDB_auto*.pdf)
    plots_dir = galaxy_dir / 'Plots'
    if plots_dir.exists():
        for p in plots_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.png', '.pdf'):
                rel = p.relative_to(plots_dir)
                yield p, Path('Plots') / rel
    # Top-level PDFs under output/<galaxy>
    for p in galaxy_dir.glob('*.pdf'):
        yield p, p.name


def collect(dest_root: Path, galaxy_name: str, src_root: Path = Path('output')) -> int:
    src_gal = src_root / galaxy_name
    if not src_gal.exists():
        return 0
    count = 0
    for src, rel in iter_plot_files(src_gal):
        dst = dest_root / galaxy_name / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            count += 1
        except Exception:
            pass
    return count


def main():
    ap = argparse.ArgumentParser(description='Collect plot PNG/PDFs into PLOT_OUTPUT per galaxy')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated galaxy names (defaults to all under output/)')
    ap.add_argument('--dest', type=str, default='PLOT_OUTPUT', help='Destination root folder')
    args = ap.parse_args()

    src_root = Path('output')
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    if args.galaxy:
        galaxies = [g.strip() for g in args.galaxy.split(',') if g.strip()]
    else:
        galaxies = [d.name for d in src_root.iterdir() if d.is_dir()]

    total = 0
    for g in galaxies:
        n = collect(dest_root, g, src_root)
        print(f'{g}: copied {n} files')
        total += n
    print(f'Total copied: {total} -> {dest_root.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
