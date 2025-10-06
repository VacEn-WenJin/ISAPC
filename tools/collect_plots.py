#!/usr/bin/env python3
"""
Collect all plot artifacts into a single PLOT_OUTPUT folder for easy sharing.

Per-galaxy: copy output/<GALAXY>/Plots/**/*.(png|pdf) to PLOT_OUTPUT/<GALAXY>/...
Project-level: include key combined PDFs if present (e.g., inner3 overlays gallery, AIP gallery, cluster map).

Usage:
  python tools/collect_plots.py
"""
from __future__ import annotations

import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path, exts: tuple[str, ...] = ('.png', '.pdf')) -> int:
    n = 0
    if not src.exists():
        return 0
    for p in src.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            n += 1
    return n


def main() -> int:
    root = Path('.')
    out_root = root / 'output'
    plot_out = root / 'PLOT_OUTPUT'
    plot_out.mkdir(exist_ok=True)

    total = 0
    # Per-galaxy plots
    for gal_dir in sorted(out_root.iterdir() if out_root.exists() else []):
        if not gal_dir.is_dir():
            continue
        plots = gal_dir / 'Plots'
        if plots.exists():
            n = copy_tree(plots, plot_out / gal_dir.name)
            total += n
            if n > 0:
                print(f'✓ {gal_dir.name}: {n} plots')

    # Project-level combined outputs
    project_files = [
        Path('combined_inner3_overlays_gallery.pdf'),
        Path('AIP_alphaFe_2D_gallery_selected.pdf'),
        Path('FINAL_DELIVERABLES/virgo_cluster_map_with_vectors.png'),
        Path('FINAL_DELIVERABLES/virgo_cluster_map_with_vectors_panels.png'),
        Path('output/PROJECT_SUMMARY.pdf'),
        Path('output/PROJECT_SUMMARY.md'),
    ]
    proj_dst = plot_out / '_PROJECT'
    proj_dst.mkdir(exist_ok=True)
    for f in project_files:
        if f.exists():
            shutil.copy2(f, proj_dst / f.name)
            print(f'✓ Project: {f.name}')

    print(f'Total plots copied: {total}')
    print(f'Destination: {plot_out.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
