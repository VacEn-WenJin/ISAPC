#!/usr/bin/env python3
"""
SNR RA/DEC overlay plots with normalized spectra thumbnails for demo galaxies.

For each target galaxy (default: VCC1588, VCC1146), this script:
- Loads RDB results NPZ to get SNR map and IFU pixel scales.
- Reads the MUSE FITS header for RA/DEC center.
- Builds an RA/DEC image of the SNR map using original pixel scale.
- Overlays first few radial bins and places 1–2 normalized spectrum thumbnails
  on the map at representative radii.

Outputs PNG figures into FINAL_DELIVERABLES/.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from astropy.io import fits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SNR_RA_DEC")


def _load_rdb_npz(gal: str):
    path = f"./output/{gal}_stack/Data/{gal}_stack_RDB_results.npz"
    if not os.path.exists(path):
        logger.error(f"RDB NPZ not found: {path}")
        return None
    return np.load(path, allow_pickle=True)


def _get_center_radec(gal: str) -> Optional[Tuple[float, float]]:
    # Prefer stacked MUSE FITS
    for candidate in [f"data/MUSE/{gal}_stack.fits", f"data/{gal}.fits", f"data/MUSE/{gal}.fits"]:
        if os.path.exists(candidate):
            try:
                with fits.open(candidate) as hdul:
                    h = hdul[0].header
                    ra = float(h.get('CRVAL1')) if h.get('CRVAL1') is not None else None
                    dec = float(h.get('CRVAL2')) if h.get('CRVAL2') is not None else None
                    if ra is not None and dec is not None:
                        logger.info(f"{gal}: FITS center RA,DEC=({ra:.6f},{dec:.6f}) from {os.path.basename(candidate)}")
                        return ra, dec
            except Exception as e:
                logger.warning(f"FITS read failed for {candidate}: {e}")
    logger.warning(f"No FITS center found for {gal}")
    return None


def _find_norm_spectrum_images(gal: str) -> List[str]:
    patterns = [
        f"output/{gal}_stack/Plots/**/{gal}_stack_P2P_spectrum_norm_*.png",
        f"FINAL_DELIVERABLES/**/{gal}_stack_P2P_spectrum_norm_*.png",
        f"**/{gal}_P2P_spectrum_norm_*.png",
    ]
    hits: List[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))
    # Prefer stack images
    hits = sorted(set(hits))
    return hits[:2]  # take first two


def _place_thumbnail(ax, img_path: str, xy: Tuple[float, float], zoom: float = 0.15):
    try:
        arr = plt.imread(img_path)
        imagebox = OffsetImage(arr, zoom=zoom)
        ab = AnnotationBbox(imagebox, xy, frameon=True, bboxprops=dict(edgecolor='black', linewidth=1, alpha=0.8))
        ax.add_artist(ab)
    except Exception as e:
        logger.warning(f"Failed to place thumbnail {img_path}: {e}")


def plot_snr_ra_dec_with_spectra(gal: str, outdir: str = "FINAL_DELIVERABLES") -> Optional[str]:
    data = _load_rdb_npz(gal)
    if data is None:
        return None

    # Extract arrays
    snr = data.get('snr')
    if snr is None:
        logger.error(f"No 'snr' array in NPZ for {gal}")
        return None
    snr = np.asarray(snr, dtype=float)
    # Meta and binning
    meta = data.get('meta_data')
    meta = meta.item() if meta is not None and hasattr(meta, 'item') else (meta or {})
    nx = int(meta.get('nx', snr.shape[1]))
    ny = int(meta.get('ny', snr.shape[0]))
    px = float(meta.get('pixelsize_x', 0.2))  # arcsec/pix
    py = float(meta.get('pixelsize_y', 0.2))

    binning = data.get('binning')
    binning = binning.item() if binning is not None and hasattr(binning, 'item') else (binning or {})
    cx = float(binning.get('center_x', nx/2))
    cy = float(binning.get('center_y', ny/2))

    distance = data.get('distance')
    distance = distance.item() if distance is not None and hasattr(distance, 'item') else (distance or {})
    bin_radii_arcsec = np.asarray(distance.get('bin_radii', []), dtype=float)
    re_arcsec = float(distance.get('effective_radius', np.nan))

    # FITS RA/DEC center
    radec = _get_center_radec(gal)
    if radec is None:
        logger.error(f"Cannot determine RA/DEC center for {gal}")
        return None
    ra0, dec0 = radec

    # Build extent in degrees
    # Small-angle: dRA = -(x - cx)*px/(3600*cos(dec)), dDEC = (y - cy)*py/3600
    px_deg = px / 3600.0
    py_deg = py / 3600.0
    cosd = np.cos(np.deg2rad(dec0)) if np.isfinite(dec0) else 1.0
    width_deg = nx * px_deg / max(cosd, 1e-6)
    height_deg = ny * py_deg
    ra_min = ra0 - (cx) * px_deg / max(cosd, 1e-6)
    ra_max = ra0 + (nx - cx) * px_deg / max(cosd, 1e-6)
    dec_min = dec0 - (cy) * py_deg
    dec_max = dec0 + (ny - cy) * py_deg

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    vmin = np.nanpercentile(snr[np.isfinite(snr)], 5) if np.any(np.isfinite(snr)) else np.nanmin(snr)
    vmax = np.nanpercentile(snr[np.isfinite(snr)], 95) if np.any(np.isfinite(snr)) else np.nanmax(snr)
    im = ax.imshow(snr, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
                   extent=[ra_min, ra_max, dec_min, dec_max], aspect='equal')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('SNR')

    # Overlay a few radial bins as ellipses at RA/DEC center
    if np.isfinite(re_arcsec):
        for r_arcsec, color, lw in zip(bin_radii_arcsec[:3], ['white', 'orange', 'red'], [1.8, 1.5, 1.5]):
            try:
                w = 2 * (r_arcsec * px_deg / max(cosd, 1e-6))
                h = 2 * (r_arcsec * py_deg)
                e = Ellipse((ra0, dec0), width=w, height=h, angle=0, fill=False, edgecolor=color, linewidth=lw, alpha=0.9)
                ax.add_patch(e)
            except Exception as e:
                logger.warning(f"Ellipse overlay failed: {e}")

    # Place up to two spectrum thumbnails at ~0.5Re east and ~1.0Re north
    thumbs = _find_norm_spectrum_images(gal)
    if thumbs:
        # Positions
        pos1 = (ra0 + (0.5 * re_arcsec) / 3600.0 / max(cosd, 1e-6), dec0)
        pos2 = (ra0, dec0 + (1.0 * re_arcsec) / 3600.0)
        _place_thumbnail(ax, thumbs[0], pos1, zoom=0.18)
        if len(thumbs) > 1:
            _place_thumbnail(ax, thumbs[1], pos2, zoom=0.18)
    else:
        logger.warning(f"No normalized spectrum images found for {gal}")

    # Labels and cosmetics
    ax.set_xlabel('Right Ascension (deg)')
    ax.set_ylabel('Declination (deg)')
    ax.set_title(f'{gal}: SNR map in RA/DEC with normalized spectra overlays')
    ax.invert_xaxis()  # RA decreases to the right
    ax.grid(True, alpha=0.3, linestyle='--')

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{gal}_snr_ra_dec_with_spectra.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {outpath}")
    return outpath


def main():
    targets = ["VCC1588", "VCC1146"]
    for gal in targets:
        plot_snr_ra_dec_with_spectra(gal)


if __name__ == "__main__":
    main()
