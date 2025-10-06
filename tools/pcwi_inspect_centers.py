"""
Inspect PCWI exposure cubes and report the fitted center per file.

Supports selecting files via glob(s) or by parsing a pcwi.link file for a
given galaxy name. Produces a CSV/JSON summary and optional quicklook PNGs
with the detected center marked to assist manual inclusion/exclusion decisions.

Examples
--------
  # Inspect specific exposures by glob and save previews
  python tools/pcwi_inspect_centers.py \
    --files "ob_data/image2426*_icubes.fits" \
    --save-previews \
    --out data/PCWI/VCC1049_centers

  # Inspect exposures for VCC1049 from pcwi.link, skipping *_sky entries
  python tools/pcwi_inspect_centers.py \
    --from-link ob_data/pcwi.link \
    --name VCC1049 \
    --root ob_data \
    --save-previews \
    --out data/PCWI/VCC1049_centers
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Reuse helpers from pcwi_stack if available
try:
    from .pcwi_stack import _read_primary_cube, _fit_center  # type: ignore
except Exception:
    # Fallback local copies to keep this script runnable on its own
    from scipy import interpolate, optimize  # type: ignore

    def _read_primary_cube(file: Path):
        from astropy.io import fits as _fits
        with _fits.open(file) as hdu:
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


@dataclass
class InspectConfig:
    files: List[Path]
    outdir: Path
    name: str | None = None
    ref_x: int = 11
    ref_y: int = 43
    fit_box_x: int = 5
    fit_box_y: int = 15
    save_previews: bool = False


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
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img = parts[0]
            label = parts[-1]
            if not include_sky and label.endswith("sky"):
                continue
            if label == name:
                files.append(root / f"image{img}_icubes.fits")
    return [p for p in files if p.exists()]


def inspect(cfg: InspectConfig) -> Path:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in cfg.files:
        data, hdr = _read_primary_cube(f)
        # collapse white-light while avoiding extreme skyline edges like legacy
        wimg = data[max(300, 0): data.shape[0] - max(300, 0), :, :].sum(axis=0)
        cx, cy = _fit_center(
            wimg,
            (cfg.ref_x - cfg.fit_box_x, cfg.ref_x + cfg.fit_box_x),
            (cfg.ref_y - cfg.fit_box_y, cfg.ref_y + cfg.fit_box_y),
        )
        obj = str(hdr.get("OBJECT", ""))
        inst = str(hdr.get("INSTRUME", ""))
        ra = float(hdr.get("RA", np.nan))
        dec = float(hdr.get("DEC", np.nan))
        rows.append({
            "file": str(f),
            "basename": f.name,
            "object": obj,
            "instrument": inst,
            "center_x": round(cx, 3),
            "center_y": round(cy, 3),
            "ra_deg": ra,
            "dec_deg": dec,
        })
        if cfg.save_previews:
            plt.figure(figsize=(6, 4))
            plt.imshow(wimg, origin="lower", cmap="gray")
            plt.scatter([cy], [cx], c="red", s=30, marker="+")  # note imshow axes: [y,x]
            plt.axhline(cfg.ref_x, color="cyan", lw=0.6)
            plt.axvline(cfg.ref_y, color="cyan", lw=0.6)
            plt.title(f"{f.name}: cx={cx:.2f}, cy={cy:.2f}")
            plt.tight_layout()
            plt.savefig(cfg.outdir / f"{f.stem}_center.png", dpi=150)
            plt.close()

    # Write CSV and JSON
    csv_path = cfg.outdir / (f"{cfg.name}_centers.csv" if cfg.name else "centers.csv")
    json_path = cfg.outdir / (f"{cfg.name}_centers.json" if cfg.name else "centers.json")
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    with open(json_path, "w") as fp:
        json.dump(rows, fp, indent=2)
    print(f"Wrote: {csv_path}\nWrote: {json_path}")
    return csv_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report Gaussian-fitted centers for PCWI exposure cubes.")
    ap.add_argument("--files", default="", help="Comma-separated list or glob(s) for exposure FITS files")
    ap.add_argument("--from-link", dest="from_link", default="", help="Path to pcwi.link to resolve exposures by name")
    ap.add_argument("--name", default=None, help="Galaxy name when using --from-link (e.g., VCC1049)")
    ap.add_argument("--root", default="ob_data", help="Directory containing image*_icubes.fits when using --from-link")
    ap.add_argument("--include-sky", action="store_true", help="Include *_sky entries when reading from link file")
    ap.add_argument("--out", default="data/PCWI/centers", help="Output directory for reports and previews")
    ap.add_argument("--save-previews", action="store_true", help="Save white-light preview PNGs with centers marked")
    ap.add_argument("--ref-x", type=int, default=11)
    ap.add_argument("--ref-y", type=int, default=43)
    ap.add_argument("--fit-box-x", type=int, default=5)
    ap.add_argument("--fit-box-y", type=int, default=15)

    args = ap.parse_args(argv)

    files: List[Path] = []
    if args.files:
        files = _parse_file_list(args.files)
    elif args.from_link and args.name:
        files = _files_from_link(Path(args.from_link), Path(args.root), args.name, include_sky=args.include_sky)
    else:
        raise SystemExit("Provide --files or --from-link with --name")

    if not files:
        raise SystemExit("No exposure files found to inspect")

    cfg = InspectConfig(
        files=files,
        outdir=Path(args.out),
        name=args.name,
        ref_x=args.ref_x,
        ref_y=args.ref_y,
        fit_box_x=args.fit_box_x,
        fit_box_y=args.fit_box_y,
        save_previews=args.save_previews,
    )
    inspect(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
