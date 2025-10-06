#!/usr/bin/env python3
"""
Virgo Cluster Phase-Space Diagram

Creates a phase-space diagram with:
- x-axis: projected distance to Virgo center (default center=M87) in Mpc
- y-axis: normalized velocity (v - v_sys) / sigma

Features:
- Optional gray background points from a catalog (e.g., NGVS SBF catalog),
  parsed flexibly from CSV/TSV/whitespace tables with RA/Dec/velocity.
- Overlays our analyzed galaxies using the same 2D style as the final cluster map:
  triangle markers (up/down by α/Fe gradient sign), filled if emission, color by Δv,
  and small vertical arrows whose length encodes |gradient| and direction encodes sign.

Usage examples:
  python virgo_phase_diagram.py \
    --catalog /path/to/apjad3453t1_mrt.txt \
    --output FINAL_DELIVERABLES/virgo_phase_space_diagram.png

  python virgo_phase_diagram.py --v-sys 1307 --sigma 700 --distance-mpc 16.5

Defaults:
- Virgo center (M87): RA=187.70591 deg, Dec=12.39112 deg
- Distance to Virgo: 16.5 Mpc (for projected separation conversion)
- v_sys: computed from overlay sample if not supplied (fallback to 1454.389 km/s)
- sigma: computed from background catalog if provided else overlay sample; fallback 700 km/s

Inputs expected in overlay CSV (auto-loaded from FINAL_DELIVERABLES/virgo_cluster_relative_velocities.csv):
  name, category, ra_deg, dec_deg, v_kms, has_emission, slope_dex_per_Re, slope_error_dex_per_Re, gradient_method, delta_v_kms, v_mean_kms

Output:
    FINAL_DELIVERABLES/virgo_phase_space_diagram.png (by default)
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Tuple, List
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except Exception:
    SkyCoord = None  # type: ignore
    u = None  # type: ignore


DEFAULT_CENTER_RA = 187.70591  # M87
DEFAULT_CENTER_DEC = 12.39112
DEFAULT_DISTANCE_MPC = 16.5
DEFAULT_OUTPUT = os.path.join("FINAL_DELIVERABLES", "virgo_phase_space_diagram.png")
OVERLAY_CSV = os.path.join("FINAL_DELIVERABLES", "virgo_cluster_relative_velocities.csv")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s [%(name)s] - %(message)s",
    )


def angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angular separation in degrees.

    Uses astropy if available; else uses a spherical law of cosines fallback.
    """
    if SkyCoord is not None:
        c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
        c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")
        return c1.separation(c2).deg
    # Fallback (numerically stable enough for small separations)
    ra1r, dec1r, ra2r, dec2r = np.radians([ra1, dec1, ra2, dec2])
    cos_d = (
        np.sin(dec1r) * np.sin(dec2r)
        + np.cos(dec1r) * np.cos(dec2r) * np.cos(ra1r - ra2r)
    )
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_d)))


def projected_distance_mpc(ra: np.ndarray, dec: np.ndarray, center_ra: float, center_dec: float, distance_mpc: float) -> np.ndarray:
    """Compute projected physical distance (Mpc) from center using small-angle approx: R = D * theta_rad."""
    seps_deg = np.array([angular_separation_deg(r, d, center_ra, center_dec) for r, d in zip(ra, dec)])
    theta_rad = np.radians(seps_deg)
    return distance_mpc * theta_rad


def try_read_catalog(path: str) -> pd.DataFrame:
    """Read a background catalog robustly from CSV/TSV/whitespace or CDS-style ASCII (mrt) formats.

    Heuristics:
    - If the file extension contains 'mrt' or header contains 'Byte-by-byte Description', prefer astropy.ascii (CDS).
    - Else try CSV, TSV, whitespace in that order.
    """
    # Detect CDS/ASCII style
    is_cds = False
    try:
        low = os.path.basename(path).lower()
        if low.endswith(".mrt") or low.endswith("_mrt.txt"):
            is_cds = True
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
            if "Byte-by-byte Description" in head or "Format Units   Label" in head:
                is_cds = True
    except Exception:
        pass

    # Prefer CDS parser when applicable
    if is_cds:
        try:
            from astropy.io import ascii  # type: ignore

            table = ascii.read(path)
            df = table.to_pandas()
            logging.info(f"Loaded background catalog (CDS/ascii): {path} with {len(df)} rows")
            return df
        except Exception as e:
            logging.warning(f"CDS/ascii parse failed for {path}: {e}; falling back to CSV/TSV/whitespace.")

    # Try pandas with default CSV
    for reader_kwargs, label in [
        ({}, "CSV"),
        ({"sep": "\t"}, "TSV"),
        ({"delim_whitespace": True, "comment": "#", "engine": "python"}, "whitespace"),
    ]:
        try:
            df = pd.read_csv(path, **reader_kwargs)
            if df.shape[1] >= 3:
                logging.info(f"Loaded background catalog ({label}): {path} with {len(df)} rows")
                return df
        except Exception:
            continue

    # Last resort: astropy ascii generic
    try:
        from astropy.io import ascii  # type: ignore

        table = ascii.read(path)
        df = table.to_pandas()
        logging.info(f"Loaded background catalog (astropy-ascii): {path} with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Failed to load catalog from {path}: {e}")
        raise


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols:
            return df.columns[cols.index(cand.lower())]
    return None


def _parse_ra_series_to_deg(series: pd.Series) -> pd.Series:
    """Convert RA series to degrees; handles numeric or sexagesimal strings if astropy is available."""
    if series.dtype.kind in ("i", "u", "f"):
        return pd.to_numeric(series, errors="coerce")
    # Likely strings; try astropy
    if SkyCoord is not None:
        def _to_deg(val):
            try:
                return SkyCoord(str(val), unit=(u.hourangle, u.deg)).ra.deg
            except Exception:
                try:
                    # Try interpreting as degrees string
                    return SkyCoord(float(val) * u.deg, 0 * u.deg).ra.deg
                except Exception:
                    return np.nan
        return series.astype(str).map(_to_deg)
    # Fallback: try to_numeric
    return pd.to_numeric(series, errors="coerce")


def _parse_dec_series_to_deg(series: pd.Series) -> pd.Series:
    """Convert Dec series to degrees; handles numeric or sexagesimal strings if astropy is available."""
    if series.dtype.kind in ("i", "u", "f"):
        return pd.to_numeric(series, errors="coerce")
    if SkyCoord is not None:
        def _to_deg(val):
            try:
                return SkyCoord("00:00:00", str(val), unit=(u.hourangle, u.deg)).dec.deg
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return np.nan
        return series.astype(str).map(_to_deg)
    return pd.to_numeric(series, errors="coerce")


def harmonize_background(df: pd.DataFrame, ra_col: Optional[str], dec_col: Optional[str], vel_col: Optional[str]) -> pd.DataFrame:
    """Rename RA/Dec/velocity columns to ra_deg/dec_deg/v_kms when possible; accept sexagesimal and/or redshift inputs."""
    lower_map = {c.lower(): c for c in df.columns}
    if ra_col is None:
        ra_col = pick_column(df, [
            "ra_deg", "ra", "ra(deg)", "radeg", "raj2000", "ra_j2000", "alpha", "alpha_j2000", "ra(hms)", "ra_hms", "radeg",
        ])
    if dec_col is None:
        dec_col = pick_column(df, [
            "dec_deg", "dec", "dec(deg)", "decdeg", "dej2000", "dec_j2000", "delta", "delta_j2000", "dec(dms)", "dec_dms", "dedeg",
        ])
    if vel_col is None:
        vel_col = pick_column(df, [
            "v_kms", "velocity", "vel", "vhelio", "v_helio", "cz", "czhel", "vrad", "v", "cz_helio", "czhelio", "helio_cz", "vhelio",
        ])
    # If still missing velocity, see if redshift is available and convert z->km/s
    z_col = None
    if vel_col is None:
        z_col = pick_column(df, ["z", "redshift", "z_helio", "zhelio"])

    missing = [name for name, col in [("RA", ra_col), ("Dec", dec_col), ("Velocity", vel_col)] if col is None]
    if missing:
        logging.warning(
            f"Background catalog missing columns: {', '.join(missing)}. Available columns: {list(df.columns)}"
        )
    out = pd.DataFrame()
    if ra_col in df:
        out["ra_deg"] = _parse_ra_series_to_deg(df[ra_col])
    if dec_col in df:
        out["dec_deg"] = _parse_dec_series_to_deg(df[dec_col])
    if vel_col in df:
        out["v_kms"] = pd.to_numeric(df[vel_col], errors="coerce")
    elif z_col in df:
        z = pd.to_numeric(df[z_col], errors="coerce")
        out["v_kms"] = z * 299792.458  # km/s

    # Drop rows with missing key fields
    out = out.dropna(subset=[c for c in ["ra_deg", "dec_deg", "v_kms"] if c in out.columns])
    return out


def load_overlay_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only galaxies (exclude major markers from overlay layer)
    df = df[df["category"] == "galaxy"].copy()
    # Ensure required columns exist
    required = ["name", "ra_deg", "dec_deg", "v_kms", "has_emission", "slope_dex_per_Re", "delta_v_kms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Overlay CSV missing required columns: {missing}")
    # Helper to normalize galaxy names (strip spaces, uppercase, remove leading zeros in numeric part)
    def _norm_name(name: str) -> str:
        s = str(name).strip().upper().replace(' ', '')
        m = re.match(r'([A-Z]+)(\d+)$', s)
        if m:
            prefix, digits = m.groups()
            try:
                num = int(digits)  # drops leading zeros
                return f"{prefix}{num}"
            except Exception:
                return s
        return s

    # Apply emission overrides if present (supports DEFAULT row to set others)
    try:
        overrides_path = os.path.join('data', 'emission_overrides.csv')
        if os.path.exists(overrides_path):
            ov = pd.read_csv(overrides_path)
            ov = ov.dropna(subset=['name'])
            ov['key'] = ov['name'].map(_norm_name)
            df['key'] = df['name'].map(_norm_name)
            # Default if DEFAULT row exists
            default_rows = ov[ov['key'].str.upper().eq('DEFAULT')]
            if not default_rows.empty and 'has_emission' in default_rows:
                default_val = bool(int(default_rows.iloc[0]['has_emission']))
                df['has_emission'] = default_val
            # Apply explicit overrides for matching keys
            ov_map = ov.set_index('key')['has_emission'].to_dict()
            df['has_emission'] = [bool(int(ov_map[k])) if k in ov_map and pd.notna(ov_map[k]) else v for k, v in zip(df['key'], df['has_emission'])]
            df = df.drop(columns=['key'])
    except Exception:
        pass
    return df


def compute_vsys_sigma(
    overlay: pd.DataFrame,
    background: Optional[pd.DataFrame],
    v_sys_cli: Optional[float],
    sigma_cli: Optional[float],
) -> Tuple[float, float]:
    if v_sys_cli is not None:
        v_sys = float(v_sys_cli)
    else:
        # Default to mean of overlay velocities; fallback to common value
        v_sys = float(np.nanmean(overlay["v_kms"])) if not overlay.empty else 1454.389
    if sigma_cli is not None:
        sigma = float(sigma_cli)
    else:
        series = None
        if background is not None and not background.empty:
            series = background["v_kms"]
        elif not overlay.empty:
            series = overlay["v_kms"]
        if series is not None:
            sigma = float(np.nanstd(series))
        else:
            sigma = 700.0
    return v_sys, sigma


def plot_phase_space(
    overlay: pd.DataFrame,
    background: Optional[pd.DataFrame],
    center_ra: float,
    center_dec: float,
    distance_mpc: float,
    v_sys: float,
    sigma: float,
    output: str,
    y_mode: str = "normalized",
    bg_y_sigma: Optional[float] = None,
    bg_y_mad: Optional[float] = None,
    bg_y_range: Optional[Tuple[float, float]] = None,
    bg_y_percentiles: Optional[Tuple[float, float]] = None,
    y_limit_range: Optional[Tuple[float, float]] = None,
    color_by: str = "gradient",
    show_arrows: bool = False,
) -> None:
    logger = logging.getLogger("virgo_phase_diagram")

    # Compute x (projected distance) and y based on selected mode
    if background is not None and not background.empty:
        bg_x = projected_distance_mpc(
            background["ra_deg"].to_numpy(),
            background["dec_deg"].to_numpy(),
            center_ra,
            center_dec,
            distance_mpc,
        )
        if y_mode == "velocity":
            bg_y = background["v_kms"].to_numpy()
        else:
            bg_y = (background["v_kms"].to_numpy() - v_sys) / sigma
    else:
        bg_x = np.array([])
        bg_y = np.array([])

    ov_x = projected_distance_mpc(
        overlay["ra_deg"].to_numpy(),
        overlay["dec_deg"].to_numpy(),
        center_ra,
        center_dec,
        distance_mpc,
    )
    if y_mode == "velocity":
        ov_y = overlay["v_kms"].to_numpy()
    else:
        ov_y = (overlay["v_kms"].to_numpy() - v_sys) / sigma

    # Determine color mapping
    if color_by == "gradient":
        vals = overlay["slope_dex_per_Re"].to_numpy()
        vmax = float(np.nanpercentile(np.abs(vals), 95)) if np.isfinite(vals).any() else 0.2
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cmap = "coolwarm"
        cbar_label = "α/Fe gradient (dex/Re)"
    else:
        vals = overlay["delta_v_kms"].to_numpy()
        vmax = float(np.nanpercentile(np.abs(vals), 95)) if np.isfinite(vals).any() else 500.0
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cmap = "cool"
        cbar_label = "Δv (km/s)"

    fig, ax = plt.subplots(figsize=(9, 6))

    # Background gray points with optional outlier removal
    if bg_x.size > 0:
        mask = np.ones_like(bg_y, dtype=bool)
        initial_n = bg_y.size

        # Stats before clipping (for diagnostics)
        try:
            logger.info(
                "Background y stats (pre-clip): n=%d, min=%.3f, p1=%.3f, p50=%.3f, p99=%.3f, max=%.3f",
                int(initial_n), float(np.nanmin(bg_y)), float(np.nanpercentile(bg_y, 1)), float(np.nanmedian(bg_y)), float(np.nanpercentile(bg_y, 99)), float(np.nanmax(bg_y))
            )
        except Exception:
            pass

        # Sigma clipping
        if bg_y_sigma is not None and bg_y_sigma > 0:
            mu = np.nanmean(bg_y)
            sd = np.nanstd(bg_y)
            if sd > 0:
                mask &= np.abs(bg_y - mu) <= bg_y_sigma * sd

        # MAD clipping
        if bg_y_mad is not None and bg_y_mad > 0:
            med = np.nanmedian(bg_y)
            mad = np.nanmedian(np.abs(bg_y - med))
            if mad > 0:
                mask &= np.abs(bg_y - med) <= bg_y_mad * mad

        # Percentile clipping
        if bg_y_percentiles is not None and len(bg_y_percentiles) == 2:
            pmin, pmax = bg_y_percentiles
            pmin = max(0.0, min(100.0, float(pmin)))
            pmax = max(0.0, min(100.0, float(pmax)))
            if pmax < pmin:
                pmin, pmax = pmax, pmin
            lo = np.nanpercentile(bg_y, pmin)
            hi = np.nanpercentile(bg_y, pmax)
            mask &= (bg_y >= lo) & (bg_y <= hi)

        # Explicit y-range
        if bg_y_range is not None and len(bg_y_range) == 2:
            ymin, ymax = bg_y_range
            mask &= (bg_y >= ymin) & (bg_y <= ymax)

        removed = int(initial_n - np.count_nonzero(mask))
        if removed > 0:
            logging.info(f"Background outlier removal: removed {removed}/{initial_n} points ({removed/initial_n:.1%}).")

        # Stats after clipping
        try:
            y_post = bg_y[mask]
            logger.info(
                "Background y stats (post-clip): n=%d, min=%.3f, p1=%.3f, p50=%.3f, p99=%.3f, max=%.3f",
                int(y_post.size), float(np.nanmin(y_post)), float(np.nanpercentile(y_post, 1)), float(np.nanmedian(y_post)), float(np.nanpercentile(y_post, 99)), float(np.nanmax(y_post))
            )
        except Exception:
            pass

        ax.scatter(bg_x[mask], bg_y[mask], s=8, c="0.8", alpha=0.5, edgecolors="none", label="Background catalog")
    else:
        logger.warning("No background catalog provided or parsed; rendering overlay-only diagram.")

    # Overlay points with our styling
    # Determine marker shape by gradient sign and fill by emission
    slopes = overlay["slope_dex_per_Re"].to_numpy()
    has_em = overlay["has_emission"].astype(bool).to_numpy()
    markers = np.where(slopes >= 0, "^", "v")

    # Plot each point to allow per-point marker/facecolor
    for i in range(len(overlay)):
        fc = plt.get_cmap(cmap)(norm(vals[i]))
        mk = markers[i]
        is_em = bool(has_em[i])
        ax.scatter(
            ov_x[i], ov_y[i], s=80, marker=mk,
            facecolors=fc if is_em else "none",
            edgecolors=("black" if is_em else fc),
            linewidths=0.8, zorder=3,
        )

    # Optional arrows (off by default for phase plot)
    if show_arrows:
        abs_slope = np.abs(slopes)
        if np.isfinite(abs_slope).any():
            s95 = np.nanpercentile(abs_slope, 95) if np.isfinite(abs_slope).any() else 1.0
            scale = 0.3 / s95 if s95 > 0 else 0.0  # arrow length in y-units per dex/Re
        else:
            scale = 0.0
        dy = np.sign(slopes) * abs_slope * scale
        dx = np.zeros_like(dy)
        ax.quiver(ov_x, ov_y, dx, dy, angles="xy", scale_units="xy", scale=1, width=0.003, color="0.2", zorder=2)

    # Colorbar for Δv
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    # Axes labels and guides
    ax.axhline(0.0, color="0.6", lw=1.0, ls=":")
    ax.set_xlabel("Projected distance to M87 (Mpc)")
    ax.set_ylabel("Velocity (km/s)" if y_mode == "velocity" else "(v - v_sys) / σ")
    ax.set_title("Virgo Cluster Phase-Space Diagram")

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="black", label="Positive α/Fe gradient", markerfacecolor="white", markersize=8, linestyle="none"),
        Line2D([0], [0], marker="v", color="black", label="Negative α/Fe gradient", markerfacecolor="white", markersize=8, linestyle="none"),
        Line2D([0], [0], marker="o", color="0.5", label="Background", markerfacecolor="0.8", markersize=6, linestyle="none"),
        Line2D([0], [0], marker="s", color="black", label="Emission present (filled)", markerfacecolor="black", markersize=8, linestyle="none"),
        Line2D([0], [0], marker="s", color="black", label="No emission (hollow)", markerfacecolor="white", markersize=8, linestyle="none"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=False)

    # Enforce y limits if provided
    if y_limit_range is not None and len(y_limit_range) == 2:
        ymin, ymax = y_limit_range
        ax.set_ylim(ymin, ymax)

    ax.grid(True, ls=":", lw=0.5, alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    logger.info(f"Saved Virgo phase-space diagram: {output}")


def main():
    parser = argparse.ArgumentParser(description="Virgo Cluster Phase-Space Diagram generator")
    parser.add_argument("--catalog", type=str, default=None, help="Path to background catalog (CSV/TSV/whitespace). Optional.")
    parser.add_argument("--ra-col", type=str, default=None, help="Background catalog RA column (degrees). Optional.")
    parser.add_argument("--dec-col", type=str, default=None, help="Background catalog Dec column (degrees). Optional.")
    parser.add_argument("--vel-col", type=str, default=None, help="Background catalog velocity column (km/s). Optional.")
    parser.add_argument("--center-ra", type=float, default=DEFAULT_CENTER_RA, help="Virgo center RA in degrees (default=M87)")
    parser.add_argument("--center-dec", type=float, default=DEFAULT_CENTER_DEC, help="Virgo center Dec in degrees (default=M87)")
    parser.add_argument("--distance-mpc", type=float, default=DEFAULT_DISTANCE_MPC, help="Virgo distance in Mpc (for projected separation)")
    parser.add_argument("--v-sys", type=float, default=None, help="Systemic velocity in km/s; default inferred from data")
    parser.add_argument("--sigma", type=float, default=None, help="Velocity dispersion in km/s; default inferred from data")
    parser.add_argument("--overlay", type=str, default=OVERLAY_CSV, help="Path to overlay CSV of our galaxies")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--y-mode", type=str, choices=["normalized", "velocity"], default="normalized", help="Y-axis mode: normalized (default) or raw velocity")
    parser.add_argument("--bg-y-sigma", type=float, default=None, help="Sigma-clip threshold for background y (e.g., 3 or 4).")
    parser.add_argument("--bg-y-mad", type=float, default=None, help="MAD-clip threshold for background y (multiple of MAD from median).")
    parser.add_argument("--bg-y-range", type=float, nargs=2, default=None, help="Explicit y-range [min max] to keep background points.")
    parser.add_argument("--bg-y-percentiles", type=float, nargs=2, default=None, help="Keep background y within [pmin pmax] percentiles (e.g., 0.5 99.5). Applied after sigma/MAD and before explicit range.")
    parser.add_argument("--y-limit-range", type=float, nargs=2, default=None, help="Set y-axis limits [min max] for the plot (in chosen y-mode units).")
    parser.add_argument("--color-by", type=str, choices=["gradient", "delta_v"], default="gradient", help="Color overlay by 'gradient' (dex/Re) or 'delta_v' (km/s). Default: gradient.")
    parser.add_argument("--show-arrows", action="store_true", help="Show gradient arrows (off by default for phase plot).")
    parser.add_argument("--log", type=str, default="INFO", help="Log level (INFO, DEBUG, WARNING)")

    args = parser.parse_args()
    setup_logging(args.log)
    logger = logging.getLogger(__name__)

    # Load overlay data
    if not os.path.exists(args.overlay):
        raise FileNotFoundError(f"Overlay CSV not found: {args.overlay}")
    overlay = load_overlay_csv(args.overlay)

    # Load optional background catalog
    background: Optional[pd.DataFrame] = None
    if args.catalog is not None:
        if os.path.exists(args.catalog):
            raw = try_read_catalog(args.catalog)
            background = harmonize_background(raw, args.ra_col, args.dec_col, args.vel_col)
            if background.empty:
                logger.warning("Background catalog parsed but empty after harmonization; skipping background layer.")
        else:
            logger.warning(f"Background catalog path does not exist: {args.catalog}")

    v_sys, sigma = compute_vsys_sigma(overlay, background, args.v_sys, args.sigma)
    logger.info(f"Using v_sys={v_sys:.3f} km/s, sigma={sigma:.3f} km/s, center=({args.center_ra}, {args.center_dec}), D={args.distance_mpc} Mpc")

    plot_phase_space(
        overlay=overlay,
        background=background,
        center_ra=args.center_ra,
        center_dec=args.center_dec,
        distance_mpc=args.distance_mpc,
        v_sys=v_sys,
        sigma=sigma,
        output=args.output,
        y_mode=args.y_mode,
        bg_y_sigma=args.bg_y_sigma,
        bg_y_mad=args.bg_y_mad,
        bg_y_range=tuple(args.bg_y_range) if args.bg_y_range is not None else None,
        bg_y_percentiles=tuple(args.bg_y_percentiles) if args.bg_y_percentiles is not None else None,
        y_limit_range=tuple(args.y_limit_range) if args.y_limit_range is not None else None,
        color_by=args.color_by,
        show_arrows=args.show_arrows,
    )


if __name__ == "__main__":
    main()
