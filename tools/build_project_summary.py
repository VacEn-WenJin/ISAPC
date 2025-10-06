#!/usr/bin/env python3
"""
Build project summary with key plots for each galaxy and an overall overview.

Outputs:
- output/<GALAXY>/summary_<GALAXY>.pdf (one page per galaxy with key plots)
- output/PROJECT_SUMMARY.pdf (all galaxies, one page each)
- output/PROJECT_SUMMARY.md (links to per-galaxy plots and results)

Plot selection heuristics (best-effort, robust to naming):
- P2P: velocity/dispersion maps if found
- VNB: velocity/dispersion maps if found
- RDB: SAURON-style maps (RDB_*_sauron.png), first 3 found
- AIP: alpha/Fe 2D map and radial profile

This script does not recompute anything; it only assembles existing artifacts.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def find_first(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def find_all(patterns: List[str], limit: int = 3) -> List[str]:
    acc: List[str] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        for m in matches:
            if m not in acc:
                acc.append(m)
            if len(acc) >= limit:
                return acc[:limit]
    return acc[:limit]


def imread_safe(path: Optional[str]):
    if not path:
        return None
    try:
        return plt.imread(path)
    except Exception:
        return None


def draw_panel(ax, image_path: Optional[str], title: str):
    ax.axis('off')
    if image_path:
        img = imread_safe(image_path)
    else:
        img = None
    if img is not None:
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
    else:
        ax.text(0.5, 0.5, f"No image\n{title}", ha='center', va='center')
        ax.set_title(title, fontsize=10)


def build_galaxy_page(galaxy: str, out_dir: Path, pdf: PdfPages) -> Tuple[bool, List[str]]:
    """Create a one-page summary for a galaxy, return (has_any, used_images)."""
    g_out = Path('output') / galaxy
    plots = g_out / 'Plots'
    used: List[str] = []

    # Heuristics for plots
    p2p_vel = find_first([
        str(plots / 'P2P' / '*velocity*.png'),
        str(plots / '*P2P*velocity*.png'),
    ])
    p2p_disp = find_first([
        str(plots / 'P2P' / '*dispersion*.png'),
        str(plots / '*P2P*dispersion*.png'),
    ])
    vnb_vel = find_first([
        str(plots / 'VNB' / '*velocity*.png'),
        str(plots / '*VNB*velocity*.png'),
    ])
    vnb_disp = find_first([
        str(plots / 'VNB' / '*dispersion*.png'),
        str(plots / '*VNB*dispersion*.png'),
    ])
    # RDB SAURON maps (pick up to 3)
    rdb_sauron = find_all([
        str(plots / 'RDB' / 'RDB_*_sauron.png'),
        str(plots / '*RDB*_sauron.png'),
    ], limit=3)
    aip_map = find_first([
        str(plots / f'{galaxy}_AIP_alpha_fe_map.png'),
        str(plots / '*AIP*alpha_fe_map.png'),
        str(g_out / f'{galaxy}_AIP_alpha_fe_map.png'),
    ])
    aip_rad = find_first([
        str(plots / f'{galaxy}_AIP_alpha_fe_radial_profile.png'),
        str(plots / '*AIP*radial_profile*.png'),
        str(g_out / f'{galaxy}_AIP_alpha_fe_radial_profile.png'),
    ])

    has_any = any([p2p_vel, p2p_disp, vnb_vel, vnb_disp, rdb_sauron, aip_map, aip_rad])
    if not has_any:
        return False, []

    # Layout: 2 rows x 3 cols (P2P vel/disp, VNB vel/disp, RDB SAURON, AIP map)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.5))
    fig.suptitle(f'{galaxy} — Project Summary', fontsize=14)
    ax = axes.ravel()

    draw_panel(ax[0], p2p_vel, 'P2P Velocity')
    draw_panel(ax[1], p2p_disp, 'P2P Dispersion')
    draw_panel(ax[2], vnb_vel, 'VNB Velocity')
    draw_panel(ax[3], vnb_disp, 'VNB Dispersion')
    # For RDB, if multiple sauron maps, compose a mini-grid into a single panel
    if rdb_sauron:
        try:
            sub = ax[4].inset_axes([0, 0, 1, 1])
            ax[4].axis('off')
            # create a simple tiled preview of up to 3 images
            for i, p in enumerate(rdb_sauron[:3]):
                img = imread_safe(p)
                if img is None:
                    continue
                # compute position
                x0 = (i % 3) / 3
                y0 = 0
                w = 1/3
                h = 1
                subin = ax[4].inset_axes([x0, y0, w, h])
                subin.axis('off')
                subin.imshow(img)
            ax[4].set_title('RDB SAURON maps', fontsize=10)
            used.extend(rdb_sauron[:3])
        except Exception:
            draw_panel(ax[4], rdb_sauron[0] if rdb_sauron else None, 'RDB SAURON map')
    else:
        draw_panel(ax[4], None, 'RDB SAURON maps')

    draw_panel(ax[5], aip_map or aip_rad, 'AIP alpha/Fe (map/profile)')

    # Track used
    for p in [p2p_vel, p2p_disp, vnb_vel, vnb_disp, aip_map, aip_rad]:
        if p:
            used.append(p)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return True, used


def write_markdown_summary(galaxies: List[str], out_path: Path) -> None:
    lines: List[str] = []
    lines.append('# Project Summary\n')
    lines.append('\n')
    lines.append('This summary follows the paper plan: P2P → VNB → RDB (inner-3 equalized) → AIP (alpha/Fe & gradients).\n')
    lines.append('\n')
    for g in galaxies:
        lines.append(f'## {g}\n')
        plots_dir = Path('output') / g / 'Plots'
        cand = [
            plots_dir / f'{g}_AIP_alpha_fe_map.png',
            plots_dir / f'{g}_AIP_alpha_fe_radial_profile.png',
        ]
        for c in cand:
            if c.exists():
                lines.append(f'![]({c.as_posix()})\n')
        lines.append('\n')
    out_path.write_text('\n'.join(lines))


def main():
    out_root = Path('output')
    galaxies = []
    for d in sorted(out_root.iterdir() if out_root.exists() else []):
        if not d.is_dir():
            continue
        if (d / 'Data').exists():
            galaxies.append(d.name)

    if not galaxies:
        print('No galaxies found under output/.')
        return 0

    # Per-galaxy and combined PDF
    combined_pdf_path = out_root / 'PROJECT_SUMMARY.pdf'
    with PdfPages(combined_pdf_path) as combined_pdf:
        for g in galaxies:
            per_gal_pdf = out_root / g / f'summary_{g}.pdf'
            with PdfPages(per_gal_pdf) as pp:
                ok, _ = build_galaxy_page(g, out_root / g, pp)
                if ok:
                    print(f'Wrote {per_gal_pdf}')
                else:
                    print(f'No plots for {g}, skipped per-galaxy page')
            # Also add the same page to the combined PDF
            ok2, _ = build_galaxy_page(g, out_root / g, combined_pdf)
            if ok2:
                print(f'Added {g} to combined summary')

    print(f'Wrote combined summary: {combined_pdf_path}')

    # Markdown lightweight summary
    md_path = out_root / 'PROJECT_SUMMARY.md'
    write_markdown_summary(galaxies, md_path)
    print(f'Wrote markdown summary: {md_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
