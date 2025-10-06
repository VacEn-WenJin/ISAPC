#!/usr/bin/env python3
"""
For each galaxy's exposures, show a grid of the central-10% spectra panels and
mark which exposures are used in the stack with a red box.

Inputs and behavior:
- Discovers exposures using tools/ob_data_qc.py list_exposures (default ob_data/)
- Ensures single-panel PNGs exist (calls ob_data_qc inspect if needed)
- Selection ("used in stack") is determined by:
  1) --select/--select-file if provided
  2) Else from out_dir/<galaxy>/<galaxy>_central10_qc.csv keep==True
  3) Else none marked

Outputs:
- output/_obqa/<GALAXY>/<GALAXY>_central10_grid_marked.png (mosaic with red boxes)
- combined_stack_selection_gallery.pdf (one page per galaxy) in repo root by default

Usage examples:
  python tools/build_stack_selection_gallery.py --galaxy VCC1049_obstack --auto
  python tools/build_stack_selection_gallery.py --galaxy VCC1146_obstack,VCC1695_obstack --select-file selects.txt
  python tools/build_stack_selection_gallery.py  # process all galaxies with single_panels available
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _import_ob_qc():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools import ob_data_qc as qc
    return qc


def discover_galaxies(ob_root: Path, out_dir: Path) -> List[str]:
    gals: Set[str] = set()
    # Prefer those that already have single_panels
    base = out_dir
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and (d / 'single_panels').exists():
                gals.add(d.name)
    # Also include any directories under ob_root
    if ob_root.exists():
        for d in ob_root.iterdir():
            if d.is_dir():
                gals.add(d.name)
        # flat layout: collect stems like <gal>_*.fits
        for f in ob_root.glob('*.fits'):
            name = f.stem.split('_')[0]
            gals.add(name)
    return sorted(gals)


def ensure_single_panels(qc, galaxy: str, ob_root: Path, out_dir: Path, wl: Tuple[float,float] | None = None, norm_win: Tuple[float,float] | None = None) -> Path:
    """Run inspect to generate single_panels if missing; return the panel dir."""
    gdir = out_dir / galaxy
    sp_dir = gdir / 'single_panels'
    if sp_dir.exists() and any(sp_dir.glob('*_central10.png')):
        return sp_dir
    # build args namespace for inspect
    class _A: pass
    args = _A()
    args.galaxy = galaxy
    args.ob_root = str(ob_root)
    args.out_dir = str(out_dir)
    args.wl = wl
    args.norm_win = norm_win
    args.style = 'both'
    args.max_out_frac = 0.010
    args.max_rms = 0.10
    args.single_panels = True
    args.select = ''
    args.select_file = ''
    qc.inspect_cmd(args)
    return sp_dir


def read_selection(out_dir: Path, galaxy: str, select: List[str] | None, select_file: Path | None, auto: bool) -> Set[str]:
    sel: Set[str] = set()
    if select:
        sel.update([s.strip() for s in select if s.strip()])
    if select_file and select_file.exists():
        with open(select_file, 'r') as f:
            for line in f:
                t = line.strip()
                if t and not t.startswith('#'):
                    sel.add(t)
    if sel:
        return sel
    if auto:
        # Read QC CSV keep flags
        csv_path = out_dir / galaxy / f'{galaxy}_central10_qc.csv'
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        keep_flag = str(row.get('keep', '')).strip().lower()
                        if keep_flag in ('true','1','yes','y','t'):
                            fn = row.get('file', '')
                            if fn:
                                sel.add(fn)
            except Exception:
                pass
    return sel


def build_marked_grid(galaxy: str, sp_dir: Path, selection: Set[str], out_png: Path) -> None:
    # Find all single panel images and map stems back to files
    panels = sorted(sp_dir.glob('*_central10.png'))
    if not panels:
        return
    # Use a grid similar to qc.plot_central_grid
    n = len(panels)
    ncols = 2 if n <= 5 else int((n) ** 0.5 + 0.999)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.9 * nrows))
    axes = np.atleast_2d(axes)
    flat_axes = axes.ravel().tolist()
    import matplotlib.image as mpimg
    for i, ax in enumerate(flat_axes):
        if i >= n:
            ax.axis('off')
            continue
        img_path = panels[i]
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        stem = img_path.name.replace('_central10.png', '')
        # Determine original filename: prefer matching full .fits name in selection; fall back to stem
        used = False
        # A selection entry may be stem or full filename; check both
        if stem in selection:
            used = True
        else:
            # find any .fits in selection whose stem equals stem
            used = any(Path(s).stem == stem for s in selection)
        # Draw red box if used
        if used:
            h, w = img.shape[0], img.shape[1]
            rect = Rectangle((0, 0), w, h, linewidth=4, edgecolor='red', facecolor='none', transform=ax.transData)
            ax.add_patch(rect)
        # Add small title label
        ax.set_title(stem if len(stem) <= 40 else stem[:37] + '…', fontsize=9)
    fig.suptitle(f'{galaxy}: central 10% spectra — used in stack highlighted', fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Build per-galaxy selection grids with used exposures highlighted')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated galaxy names (as in ob_data and output/_obqa)')
    ap.add_argument('--ob-root', type=str, default='ob_data', help='Root folder for raw exposures')
    ap.add_argument('--out-dir', type=str, default='output/_obqa', help='QC output directory (where single_panels are saved)')
    ap.add_argument('--select', type=str, default='', help='Comma-separated exposure filenames or stems to mark as used')
    ap.add_argument('--select-file', type=str, default='', help='Text file listing exposures used (one per line)')
    ap.add_argument('--auto', action='store_true', help='Use keep==True from QC CSV as selection when no explicit selection given')
    ap.add_argument('--wl', nargs=2, type=float, default=None, help='Optional wavelength range for generating panels')
    ap.add_argument('--norm-win', nargs=2, type=float, default=None, help='Normalization window for generating panels')
    ap.add_argument('--pdf', type=str, default='combined_stack_selection_gallery.pdf', help='Output combined PDF path')
    args = ap.parse_args()

    qc = _import_ob_qc()
    ob_root = Path(args.ob_root)
    out_dir = Path(args.out_dir)
    selected_gals = [g.strip() for g in args.galaxy.split(',') if g.strip()] if args.galaxy else None
    galaxies = selected_gals or discover_galaxies(ob_root, out_dir)
    if not galaxies:
        print('No galaxies found (no exposures or single_panels).')
        return 0

    # Build per-galaxy grids and a combined PDF
    with PdfPages(args.pdf) as pdf:
        for g in galaxies:
            sp_dir = ensure_single_panels(qc, g, ob_root, out_dir, wl=tuple(args.wl) if args.wl else None, norm_win=tuple(args.norm_win) if args.norm_win else None)
            sel = read_selection(out_dir, g,
                                 select=[s.strip() for s in args.select.split(',') if s.strip()] if args.select else None,
                                 select_file=Path(args.select_file) if args.select_file else None,
                                 auto=args.auto)
            out_png = out_dir / g / f'{g}_central10_grid_marked.png'
            build_marked_grid(g, sp_dir, sel, out_png)
            # Append to PDF
            try:
                img = plt.imread(out_png)
                h, w = img.shape[0], img.shape[1]
                # Preserve aspect ratio on the PDF page; scale width to 12 inches
                width_in = 12.0
                height_in = max(4.0, width_in * (h / float(w) if w else 1.0))
                fig2, ax2 = plt.subplots(figsize=(width_in, height_in))
                ax2.imshow(img)
                ax2.axis('off')
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)
            except Exception:
                pass
            print(f'✓ {g}: wrote {out_png}')
    print(f'Combined PDF: {Path(args.pdf).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
