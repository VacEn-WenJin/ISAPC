#!/usr/bin/env python3
"""
AIP: After ISAPC Pipeline collector

- Create a clean output folder under each galaxy directory
- Copy key plots/results (velocity, dispersion, indices, normalized spectra, radial profiles)
- Skip noisy/temporary plots

Usage:
    python aip_collect_results.py --outputs-dir ./output --base-out ./FINAL_DELIVERABLES
"""
from pathlib import Path
import shutil
import argparse

KEEP_PATTERNS = [
    # P2P core
    "*_P2P_velocity_dispersion.png",
    "*_P2P_velocity.png",
    "*_P2P_gas_kinematics.png",
    "*_P2P_spectrum_*.png",
    "*_P2P_spectrum_norm_*.png",
    # VNB/RDB maps
    "*_VNB_*map*.png",
    "*_RDB_*map*.png",
    # AIP alpha/Fe artifacts
    "*_AIP_alpha_fe_map.png",
    "*_AIP_alpha_fe_radial_profile.png",
    # Indices and stellar pop
    "*_indices_*.png",
    "*_stellar_pop_*.png",
    # Profiles
    "*_radial_profile*.png",
]

EXCLUDE_PATTERNS = [
    "*_debug*.png",
    "*temp*.png",
]

def collect_for_galaxy(galaxy_dir: Path, out_root: Path):
    plots_dir = galaxy_dir / "Plots"
    if not plots_dir.exists():
        return
    target = out_root / galaxy_dir.name
    target.mkdir(parents=True, exist_ok=True)
    # Copy selected patterns
    copied = 0
    for pat in KEEP_PATTERNS:
        for p in plots_dir.rglob(pat):
            # Exclude unwanted
            skip = any(p.match(epat) for epat in EXCLUDE_PATTERNS)
            if skip:
                continue
            rel = p.relative_to(plots_dir)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, dest)
                copied += 1
            except Exception:
                pass
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", "--base-out", dest="out", default="FINAL_DELIVERABLES", help="Output folder root")
    ap.add_argument("--outputs-dir", dest="outputs", default="output", help="Directory containing galaxy outputs (output/*_stack)")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    if outputs_dir.exists():
        it = outputs_dir.iterdir()
    else:
        it = []

    for galaxy_dir in it:
        if galaxy_dir.is_dir() and galaxy_dir.name.endswith("_stack") and (galaxy_dir / "Plots").exists():
            copied = collect_for_galaxy(galaxy_dir, out_root)
            if copied:
                total += copied
    print(f"AIP collected {total} plot files into {out_root}")

if __name__ == "__main__":
    main()
