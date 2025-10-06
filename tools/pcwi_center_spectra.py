"""
Extract the spectrum of the center pixel from PCWI exposure cubes.

For each input FITS cube, fit a center (like the stacker), round to the nearest
integer pixel, extract the 1D spectrum at [ix, iy], and save per-exposure CSV/PNG
plus a combined overlay plot. Optional rest-frame correction via --z.

Additionally, compute an aperture spectrum by stacking the central 10% of
spaxels (based on anisotropic PCWI pixel scales) for each exposure, and produce
both per-exposure outputs and an across-exposure stacked spectrum.

Examples
--------
    # Per-exposure spectra and overlay for VCC1049 candidate exposures
  python tools/pcwi_center_spectra.py \
    --files "ob_data/image2425*_icubes.fits,ob_data/image2426*_icubes.fits" \
    --out data/PCWI/VCC1049_centers \
    --ref-x 11 --ref-y 43

  # From pcwi.link for a given galaxy (skipping sky frames)
  python tools/pcwi_center_spectra.py \
    --from-link ob_data/pcwi.link --name VCC1049 --root ob_data \
    --out data/PCWI/VCC1049_centers
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

try:
    from .pcwi_stack import _read_primary_cube, _fit_center  # type: ignore
except Exception:
    from scipy import optimize  # type: ignore

    def _read_primary_cube(file: Path):
        with fits.open(file) as hdu:
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
        if np.any(~np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data, hdr

    def _moments(img: np.ndarray):
        total = np.nansum(img)
        if not np.isfinite(total) or total == 0:
            x = (img.shape[0] - 1) / 2
            y = (img.shape[1] - 1) / 2
            return np.nanmax(img), x, y, max(1.0, img.shape[0] / 10), max(1.0, img.shape[1] / 10)
        X, Y = np.indices(img.shape)
        x = float(np.nansum(X * img) / total)
        y = float(np.nansum(Y * img) / total)
        col = img[:, int(round(y))]
        row = img[int(round(x)), :]
        width_x = np.sqrt(abs(np.nansum(((np.arange(col.size) - y) ** 2) * col) / np.nansum(col))) if np.nansum(col) else 1.0
        width_y = np.sqrt(abs(np.nansum(((np.arange(row.size) - x) ** 2) * row) / np.nansum(row))) if np.nansum(row) else 1.0
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
            _, cx, cy, _, _ = p0
            cx += x0
            cy += y0
        return cx, cy


def _parse_file_list(files_arg: str) -> List[Path]:
    paths: List[Path] = []
    parts = [p for p in files_arg.split(",") if p.strip()]
    for p in parts:
        p = p.strip()
        if any(ch in p for ch in "*?[]"):
            for g in sorted(Path().glob(p)):
                if g.is_file():
                    paths.append(g)
        else:
            gp = Path(p)
            if gp.exists() and gp.is_file():
                paths.append(gp)
    # unique preserve order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _files_from_link(link: Path, root: Path, name: str, include_sky: bool = False) -> List[Path]:
    files: List[Path] = []
    with open(link) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            img = parts[0]
            label = parts[-1]
            if not include_sky and label.endswith("sky"):
                continue
            if label == name:
                p = root / f"image{img}_icubes.fits"
                if p.exists():
                    files.append(p)
    return files


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract and plot center-pixel spectra for PCWI exposure cubes.")
    ap.add_argument("--files", default="", help="Comma-separated list or glob(s) of FITS files")
    ap.add_argument("--from-link", dest="from_link", default="", help="pcwi.link path to resolve exposures by name")
    ap.add_argument("--name", default=None, help="Galaxy name for --from-link")
    ap.add_argument("--root", default="ob_data", help="Directory holding image*_icubes.fits for --from-link")
    ap.add_argument("--include-sky", action="store_true", help="Include *_sky exposures when using --from-link")
    ap.add_argument("--out", default="data/PCWI/centers", help="Output directory for spectra")
    ap.add_argument("--ref-x", type=int, default=11, help="Reference X for center-fit search box")
    ap.add_argument("--ref-y", type=int, default=43, help="Reference Y for center-fit search box")
    ap.add_argument("--fit-box-x", type=int, default=5, help="Half-size in X for center fit search window")
    ap.add_argument("--fit-box-y", type=int, default=15, help="Half-size in Y for center fit search window")
    ap.add_argument("--z", type=float, default=None, help="Optional redshift to convert to rest frame (λ_rest=λ/(1+z))")
    ap.add_argument("--make-panels", action="store_true", help="Also produce multi-panel figure (one panel per exposure)")
    ap.add_argument("--aperture10", action="store_true", help="Also compute 10% central-aperture spectra and overlays")

    args = ap.parse_args(argv)

    files: List[Path] = []
    if args.files:
        files = _parse_file_list(args.files)
    elif args.from_link and args.name:
        files = _files_from_link(Path(args.from_link), Path(args.root), args.name, include_sky=args.include_sky)
    else:
        raise SystemExit("Provide --files or --from-link with --name")
    if not files:
        raise SystemExit("No input files found")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    overlay_waves: list[np.ndarray] = []
    overlay_fluxes: list[np.ndarray] = []
    labels: list[str] = []
    # For 10% central-aperture outputs
    ap_overlay_waves: list[np.ndarray] = []
    ap_overlay_fluxes: list[np.ndarray] = []
    # panel collections
    panel_series: list[tuple[np.ndarray, np.ndarray, str]] = []
    panel_series_ap: list[tuple[np.ndarray, np.ndarray, str]] = []

    for fp in files:
        data, hdr = _read_primary_cube(fp)
        # white-light for center fit
        wimg = data[max(300, 0): data.shape[0] - max(300, 0), :, :].sum(axis=0)
        cx, cy = _fit_center(
            wimg,
            (args.ref_x - args.fit_box_x, args.ref_x + args.fit_box_x),
            (args.ref_y - args.fit_box_y, args.ref_y + args.fit_box_y),
        )
        ix, iy = int(round(cx)), int(round(cy))
        # guard bounds
        ix = max(0, min(ix, data.shape[1] - 1))
        iy = max(0, min(iy, data.shape[2] - 1))

        # wavelength axis
        w0 = float(hdr.get("CRVAL3", 0.0))
        dw = float(hdr.get("CD3_3", hdr.get("CDELT3", 1.0)))
        nw = int(hdr.get("NAXIS3", data.shape[0]))
        wave = w0 + np.arange(nw) * dw
        if args.z is not None:
            wave_plot = wave / (1.0 + float(args.z))
        else:
            wave_plot = wave
        flux = data[:, ix, iy]

        # Save CSV for this exposure
        csv_path = outdir / f"{fp.stem}_center_spec.csv"
        with open(csv_path, "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["wave_angstrom", "flux"])
            w.writerows(zip(wave_plot.tolist(), flux.tolist()))

        # Plot per-exposure
        png_path = outdir / f"{fp.stem}_center_spec.png"
        plt.figure(figsize=(8, 4))
        plt.plot(wave_plot, flux, lw=0.8)
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Flux (FLAM)")
        plt.title(f"{fp.name} @ center [{ix},{iy}] (z={(args.z if args.z is not None else 'obs')})")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()

        overlay_waves.append(wave_plot)
        overlay_fluxes.append(flux)
        labels.append(fp.stem)
        panel_series.append((wave_plot, flux, fp.stem))

        # Optional: central 10% aperture spectrum (anisotropic metric)
        if args.aperture10:
            # Pixel scales adopted from legacy PCWI scripts
            xscale, yscale = 2.65, 0.58  # arcsec/px
            nx, ny = data.shape[1], data.shape[2]
            xi = np.arange(nx)[:, None]
            yi = np.arange(ny)[None, :]
            dist = np.sqrt(((xi - cx) * xscale) ** 2 + ((yi - cy) * yscale) ** 2)
            # Determine radius enclosing 10% of valid spaxels
            flat = dist.reshape(-1)
            # Exclude NaN slices by checking any non-finite across wavelength; approximate with white-light image mask
            wmask = np.isfinite(wimg)
            flat_mask = wmask.reshape(-1)
            good_d = flat[flat_mask]
            if good_d.size == 0:
                r10 = np.nan
                mask = np.zeros_like(dist, dtype=bool)
            else:
                r10 = np.quantile(good_d, 0.10)
                mask = dist <= r10
            # Stack spectrum across selected spaxels (median for robustness)
            if mask.any():
                ap_spec = np.nanmedian(data[:, mask], axis=1)
            else:
                ap_spec = flux.copy()

            # Save CSV/PNG for aperture spectrum
            csv_path_ap = outdir / f"{fp.stem}_center10pct_spec.csv"
            with open(csv_path_ap, "w", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(["wave_angstrom", "flux"])
                w.writerows(zip(wave_plot.tolist(), ap_spec.tolist()))

            png_path_ap = outdir / f"{fp.stem}_center10pct_spec.png"
            plt.figure(figsize=(8, 4))
            plt.plot(wave_plot, ap_spec, lw=0.8)
            plt.xlabel("Wavelength (Angstrom)")
            plt.ylabel("Flux (FLAM)")
            plt.title(f"{fp.name} @ central 10% aperture (r10={r10:.2f} arcsec eq.)")
            plt.tight_layout()
            plt.savefig(png_path_ap, dpi=120)
            plt.close()

            ap_overlay_waves.append(wave_plot)
            ap_overlay_fluxes.append(ap_spec)
            panel_series_ap.append((wave_plot, ap_spec, fp.stem))

    # Combined overlay (only if at least 2 inputs)
    if len(overlay_waves) >= 1:
        plt.figure(figsize=(10, 5))
        for w, f, lab in zip(overlay_waves, overlay_fluxes, labels):
            # normalize by median of central region to compare shapes
            if len(f) > 400:
                sl = slice(200, -200)
            else:
                sl = slice(len(f)//4, 3*len(f)//4)
            denom = np.nanmedian(f[sl]) if np.isfinite(np.nanmedian(f[sl])) else 1.0
            plt.plot(w, f/denom, lw=0.8, label=lab)
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Normalized Flux")
        plt.legend(ncol=2, fontsize=8)
        plt.title("Center pixel spectra overlay")
        plt.tight_layout()
        ov_path = outdir / ("center_spectra_overlay_rest.png" if args.z is not None else "center_spectra_overlay_obs.png")
        plt.savefig(ov_path, dpi=140)
        plt.close()
        print(f"Wrote overlay: {ov_path}")

    # Multi-panel (one per exposure) for center-pixel spectra
    if args.make_panels and panel_series:
        n = len(panel_series)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 2.6*nrows), squeeze=False, sharex=True, sharey=True)
        for i, (w, f, lab) in enumerate(panel_series):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            if len(f) > 400:
                sl = slice(200, -200)
            else:
                sl = slice(len(f)//4, 3*len(f)//4)
            denom = np.nanmedian(f[sl]) if np.isfinite(np.nanmedian(f[sl])) else 1.0
            ax.plot(w, f/denom, lw=0.8)
            ax.set_title(lab, fontsize=9)
        for j in range(n, nrows*ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis('off')
        fig.text(0.5, 0.04, 'Wavelength (Angstrom)', ha='center')
        fig.text(0.04, 0.5, 'Normalized Flux', va='center', rotation='vertical')
        fig.suptitle('Center-pixel spectra (per exposure)', y=0.995)
        fig.tight_layout(rect=[0.05, 0.05, 1.0, 0.97])
        panels_path = outdir / ("center_spectra_panels_rest.png" if args.z is not None else "center_spectra_panels_obs.png")
        fig.savefig(panels_path, dpi=150)
        plt.close(fig)
        print(f"Wrote panels: {panels_path}")

    # Aperture-10% overlays and panels
    if args.aperture10 and ap_overlay_waves:
        # Overlay
        plt.figure(figsize=(10, 5))
        for w, f, lab in zip(ap_overlay_waves, ap_overlay_fluxes, labels):
            if len(f) > 400:
                sl = slice(200, -200)
            else:
                sl = slice(len(f)//4, 3*len(f)//4)
            denom = np.nanmedian(f[sl]) if np.isfinite(np.nanmedian(f[sl])) else 1.0
            plt.plot(w, f/denom, lw=0.8, label=lab)
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Normalized Flux")
        plt.legend(ncol=2, fontsize=8)
        plt.title("Central 10% aperture spectra overlay")
        plt.tight_layout()
        ov_path_ap = outdir / ("center10pct_overlay_rest.png" if args.z is not None else "center10pct_overlay_obs.png")
        plt.savefig(ov_path_ap, dpi=140)
        plt.close()
        print(f"Wrote overlay: {ov_path_ap}")

        # Panels
        if args.make_panels and panel_series_ap:
            n = len(panel_series_ap)
            ncols = 2 if n > 1 else 1
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 2.6*nrows), squeeze=False, sharex=True, sharey=True)
            for i, (w, f, lab) in enumerate(panel_series_ap):
                r, c = divmod(i, ncols)
                ax = axes[r][c]
                if len(f) > 400:
                    sl = slice(200, -200)
                else:
                    sl = slice(len(f)//4, 3*len(f)//4)
                denom = np.nanmedian(f[sl]) if np.isfinite(np.nanmedian(f[sl])) else 1.0
                ax.plot(w, f/denom, lw=0.8)
                ax.set_title(lab, fontsize=9)
            for j in range(n, nrows*ncols):
                r, c = divmod(j, ncols)
                axes[r][c].axis('off')
            fig.text(0.5, 0.04, 'Wavelength (Angstrom)', ha='center')
            fig.text(0.04, 0.5, 'Normalized Flux', va='center', rotation='vertical')
            fig.suptitle('Central 10% aperture spectra (per exposure)', y=0.995)
            fig.tight_layout(rect=[0.05, 0.05, 1.0, 0.97])
            panels_path_ap = outdir / ("center10pct_panels_rest.png" if args.z is not None else "center10pct_panels_obs.png")
            fig.savefig(panels_path_ap, dpi=150)
            plt.close(fig)
            print(f"Wrote panels: {panels_path_ap}")

        # Across-exposure stacked spectrum (median across per-exposure aperture spectra)
        try:
            # Ensure equal length (assumes same wavelength grid); otherwise skip
            if len({len(w) for w in ap_overlay_waves}) == 1:
                wref = ap_overlay_waves[0]
                stack_flux = np.nanmedian(np.vstack(ap_overlay_fluxes), axis=0)
                # Save
                csv_path = outdir / ("stack_center10pct_spec_rest.csv" if args.z is not None else "stack_center10pct_spec_obs.csv")
                with open(csv_path, "w", newline="") as cf:
                    w = csv.writer(cf)
                    w.writerow(["wave_angstrom", "flux"])
                    w.writerows(zip(wref.tolist(), stack_flux.tolist()))
                # Plot
                png_path = outdir / ("stack_center10pct_spec_rest.png" if args.z is not None else "stack_center10pct_spec_obs.png")
                plt.figure(figsize=(8, 4))
                plt.plot(wref, stack_flux, lw=0.9)
                plt.xlabel("Wavelength (Angstrom)")
                plt.ylabel("Flux (FLAM)")
                plt.title("Stacked spectrum: central 10% (median across exposures)")
                plt.tight_layout()
                plt.savefig(png_path, dpi=130)
                plt.close()
                print(f"Wrote stack: {png_path}")
        except Exception as e:
            print(f"Could not create across-exposure stack (10% aperture): {e}")

    print(f"Wrote per-exposure spectra to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
