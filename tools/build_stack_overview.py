#!/usr/bin/env python3
"""
Build a single-page stack overview per galaxy:
- Left: stacked white-light image
- Right: grid of original exposure white-light images; highlight selected exposures
- Footer: list of filenames chosen for the stack

Selection sources (priority order):
  1) --select/--select-file
  2) QC CSV keep==True from output/_obqa/<gal>/<gal>_central10_qc.csv if --auto
  3) All exposures under ob_data/<gal>/ if none provided

Also writes a combined multi-page PDF if --pdf is supplied.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]


def _resolve_base_galaxy(ob_root: Path, galaxy: str) -> str:
    """Infer base galaxy folder name from a possibly suffixed stack name.

    Strategy:
    - If a subdir in ob_root is a substring of galaxy, choose the longest match
    - Else if galaxy contains a token like VCC####, return that
    - Else return the first token before an underscore
    """
    # Longest subdir match
    candidates = [d.name for d in ob_root.iterdir() if d.is_dir()]
    matches = [nm for nm in candidates if nm in galaxy]
    if matches:
        return max(matches, key=len)
    # Regex-like simple find for VCC####
    import re
    m = re.search(r"(VCC\d{3,5})", galaxy, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback to first token
    return galaxy.split('_')[0]


def list_exposures(ob_root: Path, galaxy: str) -> List[Path]:
    # Reuse pattern logic similar to ob_data_qc.list_exposures
    pats = ["*icubes*.fits", "*icube*.fits", "*cube*.fits", "*.fits"]
    base = _resolve_base_galaxy(ob_root, galaxy)
    gdir = ob_root / base
    files: List[Path] = []
    if gdir.exists():
        for p in pats:
            files.extend(sorted(gdir.glob(p)))
    else:
        for p in pats:
            files.extend(sorted(ob_root.glob(p)))
    # unique by name
    seen = set()
    out: List[Path] = []
    for f in files:
        if f.is_file() and f.name not in seen:
            out.append(f)
            seen.add(f.name)
    return out


def _stack_png_from_any(stack_dir: Path, galaxy: str, base: str, stack_suffix: str) -> Path | None:
    cands = [
        stack_dir / f"{galaxy}{stack_suffix}",
        stack_dir / f"{base}{stack_suffix}",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def _stack_wl_from_fits(cands: List[Path]) -> np.ndarray | None:
    for fp in cands:
        if fp.exists():
            try:
                with fits.open(fp, memmap=True) as hdul:
                    data = hdul[0].data
                    if data is None:
                        # search first 3D ext
                        for h in hdul[1:]:
                            if hasattr(h, 'data') and isinstance(h.data, np.ndarray) and h.data.ndim == 3:
                                data = h.data
                                break
                    if data is None:
                        continue
                    arr = np.asarray(data, dtype=float)
                    wl = np.nansum(arr, axis=0)
                    return np.nan_to_num(wl, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                continue
    return None


def _compute_quick_stack_wl(files: List[Path], max_n: int = 16) -> np.ndarray | None:
    if not files:
        return None
    imgs = []
    for f in files[:max_n]:
        try:
            imgs.append(load_white_light(f))
        except Exception:
            pass
    if not imgs:
        return None
    # Median combine, robust to outliers
    return np.nanmedian(np.stack(imgs, axis=0), axis=0)


def load_white_light(fp: Path, trim: int = 300) -> np.ndarray:
    with fits.open(fp, memmap=True) as hdul:
        data = None
        for h in hdul:
            if hasattr(h, 'data') and isinstance(h.data, np.ndarray) and getattr(h.data, 'ndim', 0) == 3:
                data = h.data.astype(float)
                break
        if data is None:
            raise ValueError(f"No 3D cube in {fp}")
    a = int(max(0, trim))
    b = int(max(0, trim))
    a = min(a, data.shape[0]-1)
    b = min(b, data.shape[0]-1-a)
    wl = np.nansum(data[a:data.shape[0]-b, :, :], axis=0)
    wl = np.nan_to_num(wl, nan=0.0, posinf=0.0, neginf=0.0)
    return wl


def read_selection(out_dir: Path, galaxy: str, select: List[str] | None, select_file: Path | None, auto: bool) -> Set[str]:
    sel: Set[str] = set()
    if select:
        sel.update([s.strip() for s in select if s.strip()])
    if select_file and select_file.exists():
        for line in select_file.read_text().splitlines():
            t = line.strip()
            if t and not t.startswith('#'):
                sel.add(t)
    if sel:
        return sel
    if auto:
        csv_path = out_dir / galaxy / f"{galaxy}_central10_qc.csv"
        if csv_path.exists():
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keep_flag = str(row.get('keep','')).strip().lower()
                    if keep_flag in ('true','1','yes','y','t'):
                        fn = row.get('file','')
                        if fn:
                            sel.add(fn)
    return sel


def render_overview(galaxy: str, ob_root: Path, out_dir: Path, select: Set[str], stack_png: Path | None, out_png: Path,
                    stack_wl: np.ndarray | None = None) -> None:
    files = list_exposures(ob_root, galaxy)
    if not files:
        raise RuntimeError(f"No exposures found for {galaxy} under {ob_root}")
    # Load white-light for each exposure
    imgs = []
    names = []
    for f in files:
        try:
            wl = load_white_light(f)
            imgs.append(wl)
            names.append(f.name)
        except Exception:
            continue
    if not imgs:
        raise RuntimeError(f"No usable exposure images for {galaxy}")

    # Figure layout: left column for stack, right for grid of exposures
    n = len(imgs)
    ncols = 2 if n <= 5 else int((n) ** 0.5 + 0.999)
    nrows = (n + ncols - 1) // ncols
    # Stack panel height approximates grid height
    grid_h = 2.8 * nrows
    fig = plt.figure(figsize=(14, max(5.0, grid_h + 1.6)))

    # Left: stacked white-light (if provided)
    ax_left = fig.add_axes([0.04, 0.15, 0.36, 0.80])
    if stack_wl is not None:
        ax_left.imshow(stack_wl, origin='lower', cmap='gray')
        ax_left.set_title(f"{galaxy} stacked white-light")
    elif stack_png and stack_png.exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(stack_png)
        ax_left.imshow(img, origin='lower', cmap='gray')
        ax_left.set_title(f"{galaxy} stacked white-light (png)")
    else:
        ax_left.text(0.5, 0.5, 'Stack image not found', ha='center', va='center')
    ax_left.axis('off')

    # Right: exposure grid
    # Create a grid of axes within the right area
    right_x, right_y, right_w, right_h = 0.44, 0.15, 0.54, 0.80
    ax_grid = []
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx >= n:
                break
            w = right_w / ncols
            h = right_h / nrows
            ax = fig.add_axes([right_x + j * w, right_y + (nrows - 1 - i) * h, w, h])
            ax_grid.append(ax)
    for idx, ax in enumerate(ax_grid):
        if idx >= len(imgs):
            ax.axis('off')
            continue
        ax.imshow(imgs[idx], origin='lower', cmap='gray')
        ax.axis('off')
        stem = Path(names[idx]).stem
        used = (names[idx] in select) or any(Path(s).stem == stem for s in select)
        if used:
            ny, nx = imgs[idx].shape
            ax.add_patch(Rectangle((0, 0), nx, ny, linewidth=3.0, edgecolor='red', facecolor='none'))
        ax.set_title(stem if len(stem) <= 28 else stem[:25] + '…', fontsize=8)

    # Footer: selected filenames
    sel_list = sorted(select) if select else ["<none>"]
    footer = f"Selected for stack ({len(sel_list)}):\n" + "\n".join(sel_list)
    fig.text(0.02, 0.03, footer, fontsize=8, family='monospace', va='bottom')
    fig.suptitle(f"{galaxy} — stack overview", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Build per-galaxy stack overview pages')
    ap.add_argument('--galaxy', type=str, default='', help='Comma-separated galaxy names (e.g., VCC1949_best2_obstack)')
    ap.add_argument('--ob-root', type=str, default='ob_data', help='Directory with exposure cubes')
    ap.add_argument('--out-dir', type=str, default='output/_obqa', help='QC output directory (for selections)')
    ap.add_argument('--stack-dir', type=str, default='data/PCWI', help='Directory with stacked outputs (stack.png)')
    ap.add_argument('--stack-suffix', type=str, default='_stack.png', help='Suffix for stacked white-light image')
    ap.add_argument('--select', type=str, default='', help='Comma-separated filenames or stems to mark as selected')
    ap.add_argument('--select-file', type=str, default='', help='Text file with list of selected exposures')
    ap.add_argument('--auto', action='store_true', help='Use QC CSV keep==True if no selections provided')
    ap.add_argument('--pdf', type=str, default='', help='Optional combined PDF path')
    args = ap.parse_args()

    ob_root = Path(args.ob_root)
    out_dir = Path(args.out_dir)
    stack_dir = Path(args.stack_dir)
    selected_gals = [g.strip() for g in args.galaxy.split(',') if g.strip()] if args.galaxy else []
    if not selected_gals:
        # Infer from ob_root directories
        selected_gals = [d.name for d in ob_root.iterdir() if d.is_dir()]
    if not selected_gals:
        print('No galaxies specified or discovered.')
        return 0

    pdf = PdfPages(args.pdf) if args.pdf else None
    try:
        for gal in selected_gals:
            sel = read_selection(out_dir, gal,
                                 select=[s.strip() for s in args.select.split(',') if s.strip()] if args.select else None,
                                 select_file=Path(args.select_file) if args.select_file else None,
                                 auto=args.auto)
            base = _resolve_base_galaxy(Path(args.ob_root), gal)
            stack_png = _stack_png_from_any(stack_dir, gal, base, args.stack_suffix)
            # Try FITS-based stack previews from common locations
            stack_wl = _stack_wl_from_fits([
                Path('data/IFU') / f'{gal}.fits',
                Path('data/IFU') / f'{base}_stack.fits',
                stack_dir / f'{gal}_stack.fits',
                stack_dir / f'{base}_stack.fits',
            ])
            # As a last resort, build a quick median white-light stack from exposures
            if stack_wl is None:
                files_for_quick = list_exposures(Path(args.ob_root), gal)
                stack_wl = _compute_quick_stack_wl(files_for_quick)
            out_png = out_dir / gal / f"{gal}_stack_overview.png"
            try:
                render_overview(gal, ob_root, out_dir, sel, stack_png, out_png, stack_wl=stack_wl)
                print(f"✓ {gal}: wrote {out_png}")
                if pdf is not None:
                    img = plt.imread(out_png)
                    h, w = img.shape[0], img.shape[1]
                    width_in = 12.0
                    height_in = max(5.0, width_in * (h / float(w) if w else 1.0))
                    fig2, ax2 = plt.subplots(figsize=(width_in, height_in))
                    ax2.imshow(img)
                    ax2.axis('off')
                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)
            except Exception as e:
                print(f"! {gal}: failed to build overview: {e}")
        if pdf is not None:
            print(f"Combined PDF: {Path(args.pdf).resolve()}")
    finally:
        if pdf is not None:
            pdf.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
