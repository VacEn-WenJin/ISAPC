#!/usr/bin/env python3
"""
Redraw per-galaxy alpha/Fe gradient plots using the enhanced analysis.

This script calls analyze_single_galaxy for a target list and writes plots into
alpha_gradient_plots/ as produced by alpha_gradient_analysis.create_alpha_gradient_plots.

Usage: run without args to process a default subset, or set the VCCS env var as a
comma-separated list of galaxy names (e.g., VCC1588,VCC1146).
"""

import os
import logging
from typing import List

from alpha_gradient_analysis import analyze_single_galaxy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RedrawGradients")


DEFAULT_GALAXIES: List[str] = [
    "VCC0308", "VCC0667", "VCC0688", "VCC0990", "VCC1049",
    "VCC1146", "VCC1193", "VCC1368", "VCC1410", "VCC1431",
    "VCC1486", "VCC1499", "VCC1549", "VCC1588", "VCC1695",
    "VCC1811", "VCC1890", "VCC1902", "VCC1910", "VCC1949",
]


def main():
    env = os.getenv("VCCS", "").strip()
    galaxies = [g.strip() for g in env.split(",") if g.strip()] if env else DEFAULT_GALAXIES
    logger.info(f"Redrawing gradients for N={len(galaxies)} galaxies")

    ok = 0
    for gal in galaxies:
        try:
            res = analyze_single_galaxy(gal)
            if res and res.get('analysis_success'):
                ok += 1
                logger.info(f"OK: {gal} -> {res.get('plot_path')}")
            else:
                logger.warning(f"No result for {gal}")
        except Exception as e:
            logger.exception(f"Failed on {gal}: {e}")

    logger.info(f"Done. Successful: {ok}/{len(galaxies)}")


if __name__ == "__main__":
    main()
