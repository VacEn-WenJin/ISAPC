#!/usr/bin/env python3
"""
Collect only the inner-3 normalized overlay plot (RDB_first3_norm_overlay.png)
for all galaxies under output/* and build a single multi-page PDF gallery.

Outputs:
- combined_inner3_overlays/  (flat copies, galaxy-prefixed)
- combined_inner3_overlays_gallery.pdf (one page per galaxy)

Usage:
  python tools/collect_inner3_overlay_gallery.py
  python tools/collect_inner3_overlay_gallery.py --dest combined_inner3_overlays_alt --pdf out.pdf
  python tools/collect_inner3_overlay_gallery.py --select VCC1146_obstack,VCC1049_obstack
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def discover_galaxies(selected: List[str] | None = None) -> List[str]:
    out = Path('output')
    gals: List[str] = []
    if not out.exists():
        return gals
    for d in sorted(out.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if selected and name not in selected:
            continue
        # Require the overlay to exist
        ov = d / 'Plots' / 'RDB' / 'spectral_indices' / 'detailed' / 'RDB_first3_norm_overlay.png'
        if ov.exists():
            gals.append(name)
    return gals


def collect_and_pdf(galaxies: List[str], dest_dir: Path, pdf_path: Path) -> Tuple[int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    with PdfPages(pdf_path) as pdf:
        for g in galaxies:
            src = Path('output') / g / 'Plots' / 'RDB' / 'spectral_indices' / 'detailed' / 'RDB_first3_norm_overlay.png'
            if not src.exists():
                missing += 1
                continue
            # copy (overwrite)
            dst = dest_dir / f'{g}_RDB_first3_norm_overlay.png'
            try:
                data = plt.imread(src)
                plt.imsave(dst, data)
                copied += 1
            except Exception:
                # Fallback to raw copy
                import shutil
                shutil.copy2(src, dst)
                copied += 1
            # add to pdf
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.axis('off')
            try:
                img = plt.imread(dst)
                ax.imshow(img)
                ax.set_title(g, fontsize=12)
            except Exception:
                ax.text(0.5, 0.5, f'{g}\n(missing image)', ha='center', va='center')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return copied, missing


def main():
    ap = argparse.ArgumentParser(description='Collect inner-3 overlay plots into one folder and PDF gallery')
    ap.add_argument('--select', type=str, default='', help='Comma-separated galaxy names (as in output/*)')
    ap.add_argument('--dest', type=str, default='combined_inner3_overlays', help='Destination flat folder for copies')
    ap.add_argument('--pdf', type=str, default='combined_inner3_overlays_gallery.pdf', help='Output PDF path')
    args = ap.parse_args()

    selected = [s.strip() for s in args.select.split(',') if s.strip()] if args.select else None
    gals = discover_galaxies(selected)
    if not gals:
        print('No galaxies with inner-3 overlay found.')
        return 0
    print(f'Found {len(gals)} galaxies with overlays')
    dest_dir = Path(args.dest)
    pdf_path = Path(args.pdf)
    copied, missing = collect_and_pdf(gals, dest_dir, pdf_path)
    print(f'Copied: {copied}, Missing: {missing}')
    print(f'Folder: {dest_dir.resolve()}')
    print(f'PDF: {pdf_path.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
