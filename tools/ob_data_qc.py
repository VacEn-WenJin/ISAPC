#!/usr/bin/env python3
"""
Ob-data QC and stacking utilities.

Features:
- Inspect exposures for a galaxy under ob_data/<galaxy>/ and plot the central 10% aperture spectrum per exposure.
- Compute simple QC metrics and write a CSV (rms, outlier rate, keep flag).
- Optionally stack selected exposures into a new IFU cube FITS (median or mean).

Usage examples:
  # Inspect and create a grid figure and CSV
  python tools/ob_data_qc.py inspect --galaxy VCC1146 --out-dir output/_obqa --wl 4800 5250

  # Stack a subset explicitly
  python tools/ob_data_qc.py stack --galaxy VCC1146 --select image24258_icubes.fits,image24262_icubes.fits \
       --method median --out data/IFU/VCC1146_stack.fits

  # Auto-select based on QC and stack
  python tools/ob_data_qc.py stack --galaxy VCC1146 --auto-select --out data/IFU/VCC1146_stack.fits
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def list_exposures(ob_root: Path, galaxy: str) -> List[Path]:
    """List exposure FITS files for a galaxy.

    Supports two layouts:
    1) Nested: ob_root/<galaxy>/*icubes*.fits
    2) Flat:   ob_root/*icubes*.fits with FITS header OBJECT/TARGET containing galaxy string

    Accepts common cube names: *icube*.fits, *icubes*.fits, *cube*.fits.
    """
    pats = ["*icubes*.fits", "*icube*.fits", "*cube*.fits", "*.fits"]

    # Case 1: per-galaxy subdirectory
    gdir = ob_root / galaxy
    if gdir.exists() and gdir.is_dir():
        files: List[Path] = []
        for p in pats:
            files.extend(sorted(gdir.glob(p)))
        # De-duplicate, prefer specific patterns first
        uniq: List[Path] = []
        seen = set()
        for f in files:
            if f.name not in seen and f.is_file():
                uniq.append(f)
                seen.add(f.name)
        return uniq

    # Case 2: flat directory — inspect headers to match galaxy
    cand: List[Path] = []
    for p in pats:
        cand.extend(sorted((ob_root).glob(p)))

    # Helper to check header for galaxy string (case-insensitive)
    def header_matches(fp: Path, name: str) -> bool:
        try:
            with fits.open(fp, memmap=True) as hdul:
                hdr = None
                for h in hdul:
                    if hasattr(h, 'data') and isinstance(h.data, np.ndarray) and h.data.ndim in (2, 3):
                        hdr = h.header
                        break
                if hdr is None:
                    hdr = hdul[0].header if len(hdul) > 0 else None
                if hdr is None:
                    return False
                keys = (
                    'OBJECT', 'TARGET', 'TARGNAME', 'OBJNAME', 'NAME',
                    'EXTNAME', 'OBSERVER', 'PROGNAME'
                )
                nm = name.lower()
                for k in keys:
                    val = hdr.get(k)
                    if isinstance(val, str) and nm in val.lower():
                        return True
        except Exception:
            return False
        return False

    matched = [fp for fp in cand if fp.is_file() and header_matches(fp, galaxy)]
    if matched:
        # Deduplicate by filename in case multiple patterns matched the same file
        uniq: List[Path] = []
        seen = set()
        for f in matched:
            if f.name not in seen:
                uniq.append(f)
                seen.add(f.name)
        return uniq
    # No header matches found in flat layout -> return empty to avoid mis-assignment
    return []


def load_cube(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a FITS cube, returning (lambda, cube) where cube is (n_wave, ny, nx).
    Tries common axis orders and keywords (CRVAL3/CDELT3, etc.).
    """
    with fits.open(path, memmap=True) as hdul:
        # Find data HDU: prefer first with 3D data
        data = None
        header = None
        for h in hdul:
            if hasattr(h, 'data') and isinstance(h.data, np.ndarray) and h.data.ndim == 3:
                data = h.data
                header = h.header
                break
        if data is None:
            raise ValueError(f"No 3D cube found in {path}")
        # Normalize to (n_wave, ny, nx)
        if data.shape[0] in (data.shape[1], data.shape[2]) and data.shape[-1] != data.shape[0]:
            # Heuristic: data might be (ny, nx, n_wave)
            if data.shape[-1] > 16 and data.shape[0] < 128 and data.shape[1] < 128:
                data = np.moveaxis(data, -1, 0)
        elif data.shape[0] < 32 and data.shape[-1] > data.shape[0]:
            # Still likely (ny, nx, n_wave)
            data = np.moveaxis(data, -1, 0)
        cube = np.asarray(data, dtype=float)
        # Build wavelength axis from header if available
        # Try standard WCS keywords; support PCWI's CD3_3 as step
        crval3 = header.get('CRVAL3') if header is not None else None
        cdelt3 = header.get('CDELT3') if header is not None else None
        if cdelt3 is None and header is not None:
            cdelt3 = header.get('CD3_3')
        crpix3 = header.get('CRPIX3') if header is not None else 1.0
        if crval3 is not None and cdelt3 is not None:
            # Many MUSE cubes store wavelength in meters; convert if needed
            wl0 = float(crval3)
            dw = float(cdelt3)
            n = cube.shape[0]
            pix = np.arange(n, dtype=float) + 1.0
            wave = wl0 + (pix - crpix3) * dw
            # Convert to Angstrom if looks like meters
            if wave.mean() < 100:  # meters
                wave = wave * 1e10
        else:
            # Fallback: guess evenly spaced between 4800-9300
            n = cube.shape[0]
            wave = np.linspace(4800.0, 9300.0, n, dtype=float)
        return wave, cube


def central_mask(ny: int, nx: int, area_frac: float = 0.10) -> np.ndarray:
    """Create a circular central mask covering approximately area_frac of pixels.
    """
    yy, xx = np.indices((ny, nx))
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    # Radius chosen so that circle area ~ area_frac * (ny*nx)
    # pi*r^2 = area_frac * ny*nx => r = sqrt(area_frac * ny*nx / pi)
    r = np.sqrt(area_frac * ny * nx / np.pi)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def extract_central_spectrum(wave: np.ndarray, cube: np.ndarray, area_frac: float = 0.10) -> np.ndarray:
    """Median spectrum over central mask, robust to NaNs/outliers."""
    n_wave, ny, nx = cube.shape
    m = central_mask(ny, nx, area_frac)
    # Reshape to (n_wave, n_pix)
    spx = cube[:, m]
    with np.errstate(invalid='ignore'):
        spec = np.nanmedian(spx, axis=1)
    return spec


def normalize_spectrum(wave: np.ndarray, spec: np.ndarray, win: Tuple[float, float] | None = (5050.0, 5100.0)) -> np.ndarray:
    if win is not None:
        mask = (wave >= win[0]) & (wave <= win[1])
        if mask.sum() >= 10:
            ref = np.nanmedian(spec[mask])
        else:
            ref = np.nanmedian(spec[np.isfinite(spec)])
    else:
        ref = np.nanmedian(spec[np.isfinite(spec)])
    ref = ref if np.isfinite(ref) and ref > 0 else 1.0
    return spec / ref


def qc_metrics(wave: np.ndarray, spec: np.ndarray) -> dict:
    """Compute simple QC metrics to flag obvious sky-line problems.
    Approach: smooth then count >3.5σ positive excursions; compute RMS in continuum window.
    """
    s = np.asarray(spec, dtype=float)
    # Smooth with moving average
    k = 9
    ker = np.ones(k) / k
    sm = np.convolve(np.nan_to_num(s, nan=np.nanmedian(s)), ker, mode='same')
    resid = s - sm
    finite = np.isfinite(resid)
    if not np.any(finite):
        return {"rms": np.nan, "n_out": 0, "out_frac": 0.0}
    sigma = np.nanstd(resid[finite])
    thr = 3.5 * sigma if np.isfinite(sigma) and sigma > 0 else np.inf
    n_out = int(np.sum((resid > thr) & finite)) if np.isfinite(thr) else 0
    out_frac = float(n_out) / float(np.sum(finite)) if np.sum(finite) > 0 else 0.0
    # Continuum RMS (e.g., 5050-5100 if available)
    cmask = (wave >= 5050.0) & (wave <= 5100.0)
    crms = float(np.nanstd(resid[cmask])) if cmask.sum() > 5 else float(np.nanstd(resid[finite]))
    return {"rms": crms, "n_out": n_out, "out_frac": out_frac}


def plot_central_grid(items: List[Tuple[str, np.ndarray, np.ndarray]], out_png: Path,
                      xlim: Tuple[float, float] | None = None, title: str = "Central 10% aperture spectra (per exposure)") -> None:
    n = len(items)
    if n == 0:
        return
    # grid heuristic: up to 2 columns for 5 items as in sample; otherwise ~sqrt
    if n <= 5:
        ncols = 2
    else:
        ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if i < n:
                name, w, nf = items[i]
                ax.plot(w, nf, lw=0.8)
                ax.set_title(name, fontsize=10)
                if r == nrows - 1:
                    ax.set_xlabel("Wavelength (Angstrom)")
                if c == 0:
                    ax.set_ylabel("Normalized Flux")
                if xlim:
                    ax.set_xlim(*xlim)
            else:
                ax.axis('off')
            i += 1
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_central_overlay(items: List[Tuple[str, np.ndarray, np.ndarray]], out_png: Path,
                         xlim: Tuple[float, float] | None = None,
                         title: str = "Central 10% spectra overlay (per exposure)") -> None:
    if not items:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    # Choose a consistent x range
    if xlim is None:
        xmin = min(float(w.min()) for _, w, _ in items)
        xmax = max(float(w.max()) for _, w, _ in items)
        xlim = (xmin, xmax)
    for name, w, nf in items:
        ax.plot(w, nf, lw=0.8, label=name)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(title)
    if len(items) <= 15:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_single_panels(items: List[Tuple[str, np.ndarray, np.ndarray]], filenames: List[str], out_dir: Path,
                       xlim: Tuple[float, float] | None = None) -> None:
    """Save one PNG per exposure with a clear single-panel plot.

    items: list of (label, wave, norm_flux) where label already includes metrics
    filenames: original exposure filenames, same order as items
    out_dir: directory where to create a 'single_panels' subfolder
    """
    if not items:
        return
    sp_dir = out_dir / 'single_panels'
    sp_dir.mkdir(parents=True, exist_ok=True)
    for (label, w, nf), fname in zip(items, filenames):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(w, nf, lw=0.9)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(label)
        fig.tight_layout()
        stem = Path(fname).stem
        out_png = sp_dir / f"{stem}_central10.png"
        fig.savefig(out_png, dpi=170)
        plt.close(fig)


def inspect_cmd(args: argparse.Namespace) -> int:
    ob_root = Path(args.ob_root)
    galaxy = args.galaxy
    files = list_exposures(ob_root, galaxy)
    # Optional filtering: only preview selected exposures
    if args.select or getattr(args, 'select_file', ''):
        sel_names = set()
        if args.select:
            sel_names.update([s.strip() for s in args.select.split(',') if s.strip()])
        if getattr(args, 'select_file', ''):
            sel_path = Path(args.select_file)
            if sel_path.exists():
                with open(sel_path, 'r') as f:
                    for line in f:
                        t = line.strip()
                        if t and not t.startswith('#'):
                            sel_names.add(t)
        files = [p for p in files if p.name in sel_names or p.stem in sel_names]
    if not files:
        print(f"No exposures found for {galaxy} under {ob_root} (checked subfolder and flat layout)")
        return 2
    items = []
    rows = []
    item_files: List[str] = []
    for p in files:
        try:
            wave, cube = load_cube(p)
            spec = extract_central_spectrum(wave, cube, area_frac=0.10)
            # Normalize per exposure; use entire spectrum if no window is specified
            norm = normalize_spectrum(wave, spec, win=(args.norm_win[0], args.norm_win[1]) if args.norm_win else None)
            # Do not restrict wavelength unless explicitly requested
            if args.wl:
                m = (wave >= args.wl[0]) & (wave <= args.wl[1])
                wv = wave[m]
                nv = norm[m]
            else:
                wv, nv = wave, norm
            metrics = qc_metrics(wave, norm)
            # Compose a legend label including simple metrics for easier manual review
            label = f"{p.stem} (rms={metrics['rms']:.3f}, out={metrics['out_frac']*100:.1f}%)"
            items.append((label, wv, nv))
            item_files.append(p.name)
            rows.append({"file": p.name, **metrics})
        except Exception as e:
            rows.append({"file": p.name, "rms": np.nan, "n_out": -1, "out_frac": 0.0, "error": str(e)})
    # Decide keep flag with simple rule: out_frac < 0.01 and rms < 0.1
    for r in rows:
        if "error" in r:
            r["keep"] = False
        else:
            r["keep"] = (r["out_frac"] < args.max_out_frac) and (r["rms"] < args.max_rms)
    out_dir = Path(args.out_dir) / galaxy
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save figures: default is overlay; grid optional
    xlim = (args.wl[0], args.wl[1]) if args.wl else None
    if args.style in ("overlay", "both"):
        out_png_overlay = out_dir / f"{galaxy}_central10_overlay.png"
        plot_central_overlay(items, out_png_overlay, xlim=xlim)
        print(f"Saved overlay figure: {out_png_overlay}")
    if args.style in ("grid", "both"):
        out_png_grid = out_dir / f"{galaxy}_central10_grid.png"
        plot_central_grid(items, out_png_grid, xlim=xlim)
        print(f"Saved grid figure: {out_png_grid}")
    if getattr(args, 'single_panels', False):
        plot_single_panels(items, item_files, out_dir, xlim=xlim)
        print(f"Saved single-panel images under: {out_dir / 'single_panels'}")
    # Write CSV
    out_csv = out_dir / f"{galaxy}_central10_qc.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved QC CSV: {out_csv}")

    # Also write helper lists to simplify manual selection
    try:
        keep_list = out_dir / f"{galaxy}_keep_suggested.txt"
        all_list = out_dir / f"{galaxy}_all_exposures.txt"
        with open(keep_list, 'w') as f_keep, open(all_list, 'w') as f_all:
            for r in rows:
                fname = r.get('file', '')
                if fname:
                    f_all.write(f"{fname}\n")
                if r.get('keep', False):
                    f_keep.write(f"{fname}\n")
        print(f"Saved helper files: {keep_list.name}, {all_list.name}")
    except Exception:
        pass
    return 0


def stack_cmd(args: argparse.Namespace) -> int:
    ob_root = Path(args.ob_root)
    galaxy = args.galaxy
    files = list_exposures(ob_root, galaxy)
    if not files:
        print(f"No exposures found under {ob_root}/{galaxy}")
        return 2
    # Determine selection
    if args.select or args.select_file:
        keep_names = set()
        if args.select:
            keep_names.update([s.strip() for s in args.select.split(',') if s.strip()])
        if args.select_file:
            sel_path = Path(args.select_file)
            if sel_path.exists():
                with open(sel_path, 'r') as f:
                    for line in f:
                        t = line.strip()
                        if t and not t.startswith('#'):
                            keep_names.add(t)
            else:
                print(f"Select-file not found: {sel_path}")
        keep = [p for p in files if p.name in keep_names or p.stem in keep_names]
    elif args.auto_select:
        # Read QC CSV produced by inspect (no pandas dependency)
        qc_csv = Path(args.out_dir) / galaxy / f"{galaxy}_central10_qc.csv"
        keep = []
        if qc_csv.exists():
            try:
                with open(qc_csv, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        keep_flag = row.get('keep', 'False')
                        # Interpret various truthy values
                        if isinstance(keep_flag, str):
                            k = keep_flag.strip().lower()
                            is_keep = k in ('true', '1', 'yes', 'y', 't')
                        else:
                            is_keep = bool(keep_flag)
                        if is_keep:
                            fn = row.get('file', '')
                            pick = next((p for p in files if p.name == fn or p.stem == fn), None)
                            if pick is not None:
                                keep.append(pick)
            except Exception:
                # On any CSV parse issue, fallback to keeping all
                keep = files
        else:
            # Fallback: keep all
            keep = files
    else:
        keep = files
    if not keep:
        # If auto-select yielded nothing, fallback to all files (safer default)
        if args.auto_select:
            print("Auto-select chose no exposures; falling back to stack all available exposures.")
            keep = files
        else:
            print("No exposures selected to stack.")
            return 3
    # Load all selected
    waves = []
    cubes = []
    for p in keep:
        w, c = load_cube(p)
        waves.append(w)
        cubes.append(c)
    # Ensure common wavelength intersection
    wmin = max(w.min() for w in waves)
    wmax = min(w.max() for w in waves)
    if wmax <= wmin:
        print("No overlapping wavelength region across exposures.")
        return 4
    # Interpolate each cube to the common grid (use the first as template)
    ref = waves[0]
    mask = (ref >= wmin) & (ref <= wmax)
    wave_ref = ref[mask]
    cubes_resamp = []
    for w, c in zip(waves, cubes):
        # If wavelength grids are identical in shape and values, fast-path slice
        if w.shape == wave_ref.shape and np.allclose(w, wave_ref):
            cubes_resamp.append(c[mask])
            continue
        # Linear interpolation per spaxel
        n_wave, ny, nx = c.shape
        out = np.empty((wave_ref.size, ny, nx), dtype=float)
        for j in range(ny):
            arr = c[:, j, :]
            for i in range(nx):
                out[:, j, i] = np.interp(wave_ref, w, arr[:, i], left=np.nan, right=np.nan)
        cubes_resamp.append(out)
    # Harmonize spatial dimensions by center-cropping to the minimal (ny, nx)
    nys = [c.shape[1] for c in cubes_resamp]
    nxs = [c.shape[2] for c in cubes_resamp]
    ny_min, nx_min = min(nys), min(nxs)
    cropped = []
    for c in cubes_resamp:
        n_wave, ny, nx = c.shape
        # compute center crop indices
        sy = max(0, (ny - ny_min) // 2)
        sx = max(0, (nx - nx_min) // 2)
        ey = sy + ny_min
        ex = sx + nx_min
        cropped.append(c[:, sy:ey, sx:ex])
    cubes_resamp = cropped
    # Stack
    stack_arr = np.nanmedian(np.stack(cubes_resamp, axis=0), axis=0) if args.method == 'median' else np.nanmean(np.stack(cubes_resamp, axis=0), axis=0)
    # Write FITS using first header
    first = keep[0]
    with fits.open(first) as hdul:
        # Find data HDU
        for h in hdul:
            if hasattr(h, 'data') and isinstance(h.data, np.ndarray) and h.data.ndim == 3:
                hdr = h.header.copy()
                break
        else:
            hdr = fits.Header()
    # Update wavelength WCS if we interpolated
    hdr['CRVAL3'] = float(wave_ref[0])
    if wave_ref.size > 1:
        hdr['CDELT3'] = float(wave_ref[1] - wave_ref[0])
    hdr['CRPIX3'] = 1.0
    hdr['BUNIT'] = 'arbitrary'
    hdu = fits.PrimaryHDU(data=stack_arr, header=hdr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(out_path, overwrite=True)
    print(f"Wrote stacked IFU: {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description='QC and stack ob_data exposures')
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_ins = sub.add_parser('inspect', help='Inspect exposures and plot central 10% spectra per exposure')
    ap_ins.add_argument('--galaxy', required=True)
    ap_ins.add_argument('--ob-root', default='ob_data')
    ap_ins.add_argument('--out-dir', default='output/_obqa')
    ap_ins.add_argument('--wl', nargs=2, type=float, default=None, help='Optional wavelength range to plot [A] (default: full)')
    ap_ins.add_argument('--norm-win', nargs=2, type=float, default=None, help='Normalization window [A] (default: median over full spectrum)')
    ap_ins.add_argument('--style', choices=['overlay','grid','both'], default='overlay', help='Figure style to save (default: overlay)')
    ap_ins.add_argument('--max-out-frac', type=float, default=0.010, help='Max outlier fraction threshold to keep')
    ap_ins.add_argument('--max-rms', type=float, default=0.10, help='Max RMS threshold to keep')
    ap_ins.add_argument('--single-panels', action='store_true', help='Also save one PNG per exposure (single panel)')
    ap_ins.add_argument('--select', default='', help='Comma-separated exposure filenames or stems to preview')
    ap_ins.add_argument('--select-file', default='', help='Path to a text file with one exposure per line to preview')
    ap_ins.set_defaults(func=inspect_cmd)

    ap_stk = sub.add_parser('stack', help='Stack selected exposures into an IFU cube')
    ap_stk.add_argument('--galaxy', required=True)
    ap_stk.add_argument('--ob-root', default='ob_data')
    ap_stk.add_argument('--out', required=True, help='Output stacked FITS path (e.g., data/IFU/<gal>_stack.fits)')
    ap_stk.add_argument('--select', default='', help='Comma-separated exposure filenames or stems to keep')
    ap_stk.add_argument('--select-file', default='', help='Path to a text file with one exposure filename/stem per line to keep')
    ap_stk.add_argument('--auto-select', action='store_true', help='Keep exposures flagged by inspect QC CSV')
    ap_stk.add_argument('--out-dir', default='output/_obqa')
    ap_stk.add_argument('--method', choices=['median', 'mean'], default='median')
    ap_stk.set_defaults(func=stack_cmd)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
