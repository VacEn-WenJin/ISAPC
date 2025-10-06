"""
Stack selected PCWI exposure cubes into a single aligned cube per galaxy.

Designed to regenerate stacks while excluding known-bad exposures (e.g., poor sky subtraction).

Usage examples:
  - Manual whitelist of exposures for VCC1049 and write to data/PCWI:
      python tools/pcwi_stack.py \
        --name VCC1049 \
        --files ob_data/image24262_icubes.fits,ob_data/image24264_icubes.fits,ob_data/image24265_icubes.fits,ob_data/image24267_icubes.fits \
        --output-dir data/PCWI

  - Same but exclude one bad exposure from a glob:
      python tools/pcwi_stack.py \
        --name VCC1049 \
        --files "ob_data/image2426*_icubes.fits" \
        --exclude 24258 \
        --output-dir data/PCWI

Notes
-----
 - Exposures are aligned using a 2D Gaussian fit to a white-light image
   and then shifted via cubic interpolation to a common reference center.
 - The spectral axis is trimmed by `--cutwav` Angstroms on both ends to
   avoid edge artifacts (default=2 Å, similar to legacy scripts).
 - Output is a primary HDU with updated CRVAL3/NAXIS3 and CRPIX1/2 set to
   the chosen reference center. Units are preserved from inputs.
 - Per-exposure quicklook PNGs and the list of included files are saved
   for provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from astropy.io import fits
from scipy import interpolate, optimize
import matplotlib.pyplot as plt


# ----------------------
# Gaussian center helpers
# ----------------------

def _moments(img: np.ndarray) -> Tuple[float, float, float, float, float]:
    total = np.nansum(img)
    if not np.isfinite(total) or total == 0:
        # Fallback to geometric center
        x = (img.shape[0] - 1) / 2
        y = (img.shape[1] - 1) / 2
        return np.nanmax(img), x, y, max(1.0, img.shape[0] / 10), max(1.0, img.shape[1] / 10)
    X, Y = np.indices(img.shape)
    x = float(np.nansum(X * img) / total)
    y = float(np.nansum(Y * img) / total)
    col = img[:, int(round(y))]
    row = img[int(round(x)), :]
    width_x = math.sqrt(abs(np.nansum(((np.arange(col.size) - y) ** 2) * col) / np.nansum(col))) if np.nansum(col) else 1.0
    width_y = math.sqrt(abs(np.nansum(((np.arange(row.size) - x) ** 2) * row) / np.nansum(row))) if np.nansum(row) else 1.0
    height = float(np.nanmax(img))
    return height, x, y, width_x, width_y


def _gaussian2d(height: float, cx: float, cy: float, wx: float, wy: float):
    wx = float(max(1e-6, wx))
    wy = float(max(1e-6, wy))
    return lambda x, y: height * np.exp(-(((cx - x) / wx) ** 2 + ((cy - y) / wy) ** 2) / 2.0)


def _fit_center(img: np.ndarray, x_rng: Tuple[int, int], y_rng: Tuple[int, int]) -> Tuple[float, float]:
    x0, x1 = max(0, x_rng[0]), min(img.shape[0], x_rng[1])
    y0, y1 = max(0, y_rng[0]), min(img.shape[1], y_rng[1])
    sub = img[x0:x1, y0:y1]
    p0 = _moments(sub)
    err = lambda p: np.ravel(_gaussian2d(*p)(*np.indices(sub.shape)) - sub)
    try:
        p, _ = optimize.leastsq(err, p0, maxfev=5000)
        cx = float(p[1] + x0)
        cy = float(p[2] + y0)
    except Exception:
        # Fallback to moments center
        _, cx, cy, _, _ = p0
        cx += x0
        cy += y0
    return cx, cy


def _shift_cube_to_center(cube: np.ndarray, cx: float, cy: float, tx: float, ty: float) -> np.ndarray:
    dx, dy = (tx - cx), (ty - cy)
    x1 = np.arange(cube.shape[1])
    y1 = np.arange(cube.shape[2])
    x2 = x1 - dx
    y2 = y1 - dy
    out = np.zeros_like(cube)
    # Interpolate each wavelength plane
    for i in range(cube.shape[0]):
        f = interpolate.interp2d(x1, y1, cube[i, :, :].T, kind="cubic")
        out[i, :, :] = f(x2, y2).T
    return out


def _extract_window(cube: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> np.ndarray:
    """Extract a (2*rx+1, 2*ry+1) window around integer center (cx,cy), padding with NaNs if needed."""
    nx, ny = cube.shape[1], cube.shape[2]
    x0, x1 = cx - rx, cx + rx + 1
    y0, y1 = cy - ry, cy + ry + 1
    out = np.full((cube.shape[0], 2 * rx + 1, 2 * ry + 1), np.nan, dtype=cube.dtype)
    xs0, xs1 = max(0, x0), min(nx, x1)
    ys0, ys1 = max(0, y0), min(ny, y1)
    tx0, ty0 = xs0 - x0, ys0 - y0
    tx1, ty1 = tx0 + (xs1 - xs0), ty0 + (ys1 - ys0)
    if xs1 > xs0 and ys1 > ys0:
        out[:, tx0:tx1, ty0:ty1] = cube[:, xs0:xs1, ys0:ys1]
    return out


@dataclass
class StackConfig:
    name: str
    files: List[Path]
    exclude_ids: List[str]
    output_dir: Path
    cutwav: float = 2.0
    ref_x: int = 11
    ref_y: int = 43
    fit_box_x: int = 5
    fit_box_y: int = 15
    window_x: int = 11  # half-size in x => full size 2*11+1 = 23
    window_y: int = 43  # half-size in y => full size 2*43+1 = 87


def _read_primary_cube(file: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(file) as hdu:
        # Primary preferred; if empty, try first extension with 3D data
        if hdu[0].data is not None and hdu[0].data.ndim == 3:
            data = hdu[0].data.astype(np.float64)
            hdr = hdu[0].header.copy()
        else:
            found = None
            for ext in hdu[1:]:
                if ext.data is not None and getattr(ext.data, "ndim", 0) == 3:
                    found = ext
                    break
            if found is None:
                raise ValueError(f"No 3D data cube found in {file}")
            data = found.data.astype(np.float64)
            hdr = hdu[0].header.copy()
            for k in found.header:
                if k not in ("XTENSION", "BITPIX", "NAXIS", "PCOUNT", "GCOUNT"):
                    hdr[k] = found.header[k]
    # Normalize NaNs
    if np.any(~np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, hdr


def build_stack(cfg: StackConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Filter excludes by matching id substring against basename (e.g., "24258" in "image24258_icubes.fits")
    include_files: List[Path] = []
    for f in cfg.files:
        bn = f.name
        if any(eid in bn for eid in cfg.exclude_ids):
            continue
        include_files.append(f)
    if not include_files:
        raise SystemExit("No exposures left after applying excludes.")

    shifted_dir = cfg.output_dir / f"{cfg.name}_shifted"
    shifted_dir.mkdir(exist_ok=True)

    # Read first file to set spectral trimming and header template
    first_data, first_hdr = _read_primary_cube(include_files[0])
    w0 = float(first_hdr["CRVAL3"]) if "CRVAL3" in first_hdr else 0.0
    dw = float(first_hdr.get("CD3_3", first_hdr.get("CDELT3", 1.0)))
    nw = int(first_hdr["NAXIS3"]) if "NAXIS3" in first_hdr else first_data.shape[0]
    wave = w0 + np.arange(nw) * dw
    # spectral trimming indices
    wmin = float(np.min(wave)) + cfg.cutwav
    wmax = float(np.max(wave)) - cfg.cutwav
    indwav = np.where((wave > wmin) & (wave < wmax))[0]
    if indwav.size < 10:
        raise SystemExit("Too few wavelength points after trimming; adjust --cutwav")

    stack_planes: List[np.ndarray] = []
    used_files: List[str] = []

    # Process each exposure: determine center, shift, crop/pad, trim spectrum
    for file in include_files:
        data, hdr = _read_primary_cube(file)
        # quick-look white-light (avoid strong skyline edges)
        wimg = data[max(300, 0): data.shape[0] - max(300, 0), :, :].sum(axis=0)
        # Save preview
        plt.imsave(shifted_dir / f"{cfg.name}_{file.stem}_img.png", wimg)

        # Fit center near expected reference (same search box as legacy: +/-5 in x, +/-15 in y)
        cx, cy = _fit_center(
            wimg,
            (cfg.ref_x - cfg.fit_box_x, cfg.ref_x + cfg.fit_box_x),
            (cfg.ref_y - cfg.fit_box_y, cfg.ref_y + cfg.fit_box_y),
        )

        shifted = _shift_cube_to_center(data, cx, cy, cfg.ref_x, cfg.ref_y)

        # Save shifted exposure (optional but useful)
        out_hdr = hdr.copy()
        out_hdr["CRPIX1"] = cfg.ref_x
        out_hdr["CRPIX2"] = cfg.ref_y
        out_hdr["CRVAL3"] = float(np.min(wave[indwav]))
        out_hdr["NAXIS3"] = int(indwav.size)
        fits.PrimaryHDU(shifted[indwav, :, :], header=out_hdr).writeto(
            shifted_dir / f"{cfg.name}_{file.stem}_shifted.fits", overwrite=True
        )

        # Extract standard window around reference center
        window = _extract_window(shifted[indwav, :, :], cfg.ref_x, cfg.ref_y, cfg.window_x, cfg.window_y)
        stack_planes.append(window)
        used_files.append(str(file))

    # Median combine across exposures, ignoring NaNs from padding
    stack_cube = np.nanmedian(np.stack(stack_planes, axis=0), axis=0)

    # Write stack
    stack_hdr = first_hdr.copy()
    stack_hdr["CRPIX1"] = cfg.ref_x
    stack_hdr["CRPIX2"] = cfg.ref_y
    stack_hdr["CRVAL3"] = float(np.min(wave[indwav]))
    stack_hdr["NAXIS3"] = int(indwav.size)
    out_path = cfg.output_dir / f"{cfg.name}_stack.fits"
    fits.PrimaryHDU(stack_cube, header=stack_hdr).writeto(out_path, overwrite=True)

    # Provenance: list used files
    with open(cfg.output_dir / f"{cfg.name}_stack_inputs.json", "w") as f:
        json.dump({"name": cfg.name, "files": used_files, "exclude_ids": cfg.exclude_ids}, f, indent=2)

    # Quicklook of the stack white-light image
    wimg_stack = np.nansum(stack_cube, axis=0)
    plt.imsave(cfg.output_dir / f"{cfg.name}_stack.png", wimg_stack)

    return out_path


def _parse_file_list(files_arg: str) -> List[Path]:
    paths: List[Path] = []
    parts = [p for p in files_arg.split(",") if p.strip()]
    for p in parts:
        p = p.strip()
        if any(ch in p for ch in "*?[]"):  # glob pattern
            for g in sorted(Path().glob(p)):
                if g.is_file():
                    paths.append(g)
        else:
            gp = Path(p)
            if gp.exists() and gp.is_file():
                paths.append(gp)
    # De-duplicate preserving order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stack selected PCWI exposure cubes into an aligned stack for one galaxy.")
    ap.add_argument("--name", required=True, help="Galaxy name (e.g., VCC1049)")
    ap.add_argument("--files", required=True, help="Comma-separated list or glob(s) of exposure FITS files to include")
    ap.add_argument("--exclude", default="", help="Comma-separated exposure id substrings to exclude (e.g., 24258,24648)")
    ap.add_argument("--output-dir", default="data/PCWI", help="Directory to write outputs (stack FITS, previews, inputs JSON)")
    ap.add_argument("--cutwav", type=float, default=2.0, help="Trim this many Å from each end of the spectrum")
    ap.add_argument("--ref-x", type=int, default=11, help="Reference X pixel for alignment (legacy)"
                   )
    ap.add_argument("--ref-y", type=int, default=43, help="Reference Y pixel for alignment (legacy)"
                   )
    ap.add_argument("--fit-box-x", type=int, default=5, help="Half-size in X for center fit search window")
    ap.add_argument("--fit-box-y", type=int, default=15, help="Half-size in Y for center fit search window")
    ap.add_argument("--win-x", type=int, default=11, help="Half-size of output window in X (final cube will be 2*win_x+1)")
    ap.add_argument("--win-y", type=int, default=43, help="Half-size of output window in Y (final cube will be 2*win_y+1)")

    args = ap.parse_args(argv)
    files = _parse_file_list(args.files)
    if not files:
        raise SystemExit("No input exposure files found for --files")

    cfg = StackConfig(
        name=args.name,
        files=files,
        exclude_ids=[e.strip() for e in args.exclude.split(",") if e.strip()],
        output_dir=Path(args.output_dir),
        cutwav=args.cutwav,
        ref_x=args.ref_x,
        ref_y=args.ref_y,
        fit_box_x=args.fit_box_x,
        fit_box_y=args.fit_box_y,
        window_x=args.win_x,
        window_y=args.win_y,
    )

    out = build_stack(cfg)
    print(f"Wrote stacked cube: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
