#!/usr/bin/env python3
"""
Collect per-galaxy AIP alpha/Fe 2D map and radial profile into a multi-page PDF.

By default, auto-discovers galaxies under output/* with both PNGs present:
 - output/<GALAXY>/Plots/<GALAXY>_AIP_alpha_fe_map.png
 - output/<GALAXY>/Plots/<GALAXY>_AIP_alpha_fe_radial_profile.png

Usage:
  python tools/build_alpha_fe_gallery.py                     # all
  python tools/build_alpha_fe_gallery.py --galaxy G1,G2      # subset
  python tools/build_alpha_fe_gallery.py --pdf AIP_2D.pdf    # custom PDF name
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def find_galaxies(selected: list[str] | None = None) -> list[str]:
    out = Path('output')
    gals: list[str] = []
    for d in sorted(out.iterdir() if out.exists() else []):
        if not d.is_dir():
            continue
        name = d.name
        if selected and name not in selected:
            continue
        plots = d / 'Plots'
        if (plots / f'{name}_AIP_alpha_fe_map.png').exists() and (plots / f'{name}_AIP_alpha_fe_radial_profile.png').exists():
            gals.append(name)
    return gals


def add_page(pdf: PdfPages, map_path: Path, radial_path: Path, title: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    try:
        img1 = plt.imread(map_path)
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title(f'{title}: alpha/Fe map')
    except Exception:
        ax1.axis('off')
        ax1.set_title(f'{title}: map missing')
    try:
        img2 = plt.imread(radial_path)
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title('Radial profile')
    except Exception:
        ax2.axis('off')
        ax2.set_title('Radial profile missing')
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Build a gallery PDF of alpha/Fe 2D maps + radial profiles')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated galaxy names (match output/<galaxy>)')
    ap.add_argument('--pdf', type=str, default='AIP_alphaFe_2D_gallery.pdf', help='Output PDF path')
    args = ap.parse_args()

    selected = [g.strip() for g in args.galaxy.split(',') if g.strip()] if args.galaxy else None
    galaxies = find_galaxies(selected)
    if not galaxies:
        print('No galaxies with both alpha/Fe map and radial profile found under output/.')
        return 0

    with PdfPages(args.pdf) as pdf:
        for g in galaxies:
            root = Path('output') / g / 'Plots'
            add_page(pdf, root / f'{g}_AIP_alpha_fe_map.png', root / f'{g}_AIP_alpha_fe_radial_profile.png', g)
            print(f'Added: {g}')
    print(f'Gallery PDF: {Path(args.pdf).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
